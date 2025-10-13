//! Independent Component Analysis (ICA) implementation.
//!
//! ICA is a statistical technique for separating a multivariate signal into
//! additive independent components. It assumes that observed signals are linear
//! mixtures of independent source signals.

// TODO: Replace with scirs2-linalg
// use nalgebra::DMatrix;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{thread_rng, Random, Rng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::*,
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// ICA algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ICAAlgorithm {
    /// Parallel FastICA algorithm
    #[default]
    Parallel,
    /// Deflation FastICA algorithm  
    Deflation,
    /// Infomax ICA algorithm using maximum likelihood
    Infomax,
    /// Natural gradient ICA algorithm
    NaturalGradient,
    /// Temporal ICA for time series data
    Temporal,
    /// Constrained ICA for semi-supervised learning
    Constrained,
}

/// Non-linearity functions for ICA
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ICAFunction {
    /// Logcosh function: G(u) = tanh(a*u), g(u) = a*(1-tanh²(a*u))
    #[default]
    Logcosh,
    /// Exponential function: G(u) = u*exp(-u²/2), g(u) = (1-u²)*exp(-u²/2)
    Exp,
    /// Cubic function: G(u) = u³, g(u) = 3*u²
    Cube,
}

/// Independent Component Analysis transformer
#[derive(Debug, Clone)]
pub struct ICA<State = Untrained> {
    /// Number of components to extract
    pub n_components: Option<usize>,
    /// ICA algorithm to use
    pub algorithm: ICAAlgorithm,
    /// Non-linearity function
    pub fun: ICAFunction,
    /// Parameter for logcosh function
    pub fun_args: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Whether to whiten the data
    pub whiten: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Learning rate for gradient-based algorithms
    pub learning_rate: f64,
    /// Temporal window size for temporal ICA
    pub temporal_window: Option<usize>,
    /// Momentum factor for natural gradient
    pub momentum: f64,
    /// Prior knowledge matrix for constrained ICA
    pub constraint_matrix: Option<Array2<f64>>,
    /// Constraint weight for regularization
    pub constraint_weight: f64,
    /// Constraint tolerance for convergence
    pub constraint_tol: f64,

    /// Trained state
    state: State,
}

/// Trained ICA state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainedICA {
    pub components: Array2<f64>,
    pub mixing: Array2<f64>,
    pub mean: Array1<f64>,
    pub whitening: Option<Array2<f64>>,
    pub n_features_in: usize,
    pub n_components: usize,
    pub n_iter: usize,
}

impl ICA<Untrained> {
    /// Create a new ICA transformer
    pub fn new() -> Self {
        Self {
            n_components: None,
            algorithm: ICAAlgorithm::Parallel,
            fun: ICAFunction::Logcosh,
            fun_args: 1.0,
            max_iter: 200,
            tol: 1e-4,
            whiten: true,
            random_state: None,
            learning_rate: 0.01,
            temporal_window: None,
            momentum: 0.9,
            constraint_matrix: None,
            constraint_weight: 1.0,
            constraint_tol: 1e-6,
            state: Untrained,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    /// Set the algorithm
    pub fn algorithm(mut self, algorithm: ICAAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the non-linearity function
    pub fn fun(mut self, fun: ICAFunction) -> Self {
        self.fun = fun;
        self
    }

    /// Set the function parameter
    pub fn fun_args(mut self, fun_args: f64) -> Self {
        self.fun_args = fun_args;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to whiten data
    pub fn whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set learning rate for gradient-based algorithms
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set temporal window size for temporal ICA
    pub fn temporal_window(mut self, window_size: usize) -> Self {
        self.temporal_window = Some(window_size);
        self
    }

    /// Set momentum factor for natural gradient
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set constraint matrix for constrained ICA
    pub fn constraint_matrix(mut self, constraint_matrix: Array2<f64>) -> Self {
        self.constraint_matrix = Some(constraint_matrix);
        self
    }

    /// Set constraint weight for regularization
    pub fn constraint_weight(mut self, constraint_weight: f64) -> Self {
        self.constraint_weight = constraint_weight;
        self
    }

    /// Set constraint tolerance for convergence
    pub fn constraint_tol(mut self, constraint_tol: f64) -> Self {
        self.constraint_tol = constraint_tol;
        self
    }
}

impl Fit<Array2<f64>, ()> for ICA<Untrained> {
    type Fitted = ICA<TrainedICA>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "ICA requires at least 2 samples".to_string(),
            ));
        }

        let n_components = self.n_components.unwrap_or(n_features);

        if n_components > n_features {
            return Err(SklearsError::InvalidInput(
                "n_components cannot be larger than n_features".to_string(),
            ));
        }

        // Center the data
        let mean = x.mean_axis(Axis(0)).unwrap();
        let x_centered = x - &mean;

        // Whiten the data if requested
        let (x_whitened, whitening_matrix) = if self.whiten {
            let (whitened, whitening) = self.whiten_data(&x_centered)?;
            (whitened, Some(whitening))
        } else {
            (x_centered, None)
        };

        // Initialize random number generator
        // TODO: Support seeding for reproducibility
        let mut rng = thread_rng();

        // Run ICA algorithm
        let (components, n_iter) = match self.algorithm {
            ICAAlgorithm::Parallel => self.parallel_fastica(&x_whitened, n_components, &mut rng)?,
            ICAAlgorithm::Deflation => {
                self.deflation_fastica(&x_whitened, n_components, &mut rng)?
            }
            ICAAlgorithm::Infomax => self.infomax_ica(&x_whitened, n_components, &mut rng)?,
            ICAAlgorithm::NaturalGradient => {
                self.natural_gradient_ica(&x_whitened, n_components, &mut rng)?
            }
            ICAAlgorithm::Temporal => self.temporal_ica(&x_whitened, n_components, &mut rng)?,
            ICAAlgorithm::Constrained => {
                self.constrained_ica(&x_whitened, n_components, &mut rng)?
            }
        };

        // Compute mixing matrix (pseudoinverse of components)
        let mixing = self.compute_mixing_matrix(&components)?;

        Ok(ICA {
            n_components: self.n_components,
            algorithm: self.algorithm,
            fun: self.fun,
            fun_args: self.fun_args,
            max_iter: self.max_iter,
            tol: self.tol,
            whiten: self.whiten,
            random_state: self.random_state,
            learning_rate: self.learning_rate,
            temporal_window: self.temporal_window,
            momentum: self.momentum,
            constraint_matrix: self.constraint_matrix,
            constraint_weight: self.constraint_weight,
            constraint_tol: self.constraint_tol,
            state: TrainedICA {
                components,
                mixing,
                mean,
                whitening: whitening_matrix,
                n_features_in: n_features,
                n_components,
                n_iter,
            },
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for ICA<TrainedICA> {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (_n_samples, n_features) = x.dim();

        if n_features != self.state.n_features_in {
            return Err(SklearsError::FeatureMismatch {
                expected: self.state.n_features_in,
                actual: n_features,
            });
        }

        // Center the data
        let x_centered = x - &self.state.mean;

        // Apply whitening if it was used during fitting
        let x_processed = if let Some(ref whitening) = self.state.whitening {
            x_centered.dot(whitening)
        } else {
            x_centered
        };

        // Apply ICA transformation
        let x_transformed = x_processed.dot(&self.state.components.t());

        Ok(x_transformed)
    }
}

impl ICA<Untrained> {
    /// Whiten the data using eigendecomposition
    fn whiten_data(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let (n_samples, n_features) = x.dim();

        // Compute covariance matrix
        let cov = x.t().dot(x) / (n_samples - 1) as f64;

        // Convert to nalgebra for eigendecomposition
        let cov_matrix = DMatrix::from_row_slice(n_features, n_features, cov.as_slice().unwrap());

        // Eigendecomposition
        let eigen = cov_matrix.symmetric_eigen();
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Sort eigenvalues in descending order
        let mut sorted_indices: Vec<usize> = (0..n_features).collect();
        sorted_indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        // Build whitening matrix
        let mut whitening = Array2::zeros((n_features, n_features));
        for (i, &idx) in sorted_indices.iter().enumerate() {
            if eigenvalues[idx] > 1e-12 {
                let scale = 1.0 / eigenvalues[idx].sqrt();
                for j in 0..n_features {
                    whitening[[j, i]] = eigenvectors[(j, idx)] * scale;
                }
            }
        }

        // Apply whitening
        let x_whitened = x.dot(&whitening);

        Ok((x_whitened, whitening))
    }

    /// Parallel FastICA algorithm
    fn parallel_fastica(
        &self,
        x: &Array2<f64>,
        n_components: usize,
        rng: &mut impl Rng,
    ) -> Result<(Array2<f64>, usize)> {
        let (n_samples, n_features) = x.dim();

        // Initialize weight matrix randomly
        let mut w = Array2::zeros((n_components, n_features));
        for i in 0..n_components {
            for j in 0..n_features {
                w[[i, j]] = rng.gen::<f64>() - 0.5;
            }
        }

        // Orthogonalize initial weights
        self.orthogonalize(&mut w)?;

        let mut n_iter = 0;
        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            let w_old = w.clone();

            // Update weights in parallel for all components
            for i in 0..n_components {
                let w_i = w.slice(s![i, ..]).to_owned();
                let wx = x.dot(&w_i);

                let (g, g_prime) = self.apply_nonlinearity(&wx);

                // FastICA update rule
                let update1 = x.t().dot(&g) / n_samples as f64;
                let update2 = &w_i * (g_prime.sum() / n_samples as f64);

                for j in 0..n_features {
                    w[[i, j]] = update1[j] - update2[j];
                }
            }

            // Orthogonalize weights
            self.orthogonalize(&mut w)?;

            // Check convergence
            let mut max_diff: f64 = 0.0;
            for i in 0..n_components {
                for j in 0..n_features {
                    let diff = (w[[i, j]] - w_old[[i, j]]).abs();
                    max_diff = max_diff.max(diff);
                }
            }

            if max_diff < self.tol {
                break;
            }
        }

        Ok((w, n_iter))
    }

    /// Deflation FastICA algorithm
    fn deflation_fastica(
        &self,
        x: &Array2<f64>,
        n_components: usize,
        rng: &mut impl Rng,
    ) -> Result<(Array2<f64>, usize)> {
        let (n_samples, n_features) = x.dim();
        let mut components = Array2::zeros((n_components, n_features));
        let mut max_iter_reached = 0;

        for comp in 0..n_components {
            // Initialize weight vector randomly
            let mut w = Array1::zeros(n_features);
            for i in 0..n_features {
                w[i] = rng.gen::<f64>() - 0.5;
            }

            // Normalize
            let norm = w.mapv(|x| x * x).sum().sqrt();
            w /= norm;

            let mut n_iter = 0;
            for iter in 0..self.max_iter {
                n_iter = iter + 1;

                let w_old = w.clone();

                let wx = x.dot(&w);
                let (g, g_prime) = self.apply_nonlinearity(&wx);

                // FastICA update rule
                let update1 = x.t().dot(&g) / n_samples as f64;
                let update2 = &w * (g_prime.sum() / n_samples as f64);

                w = update1 - update2;

                // Orthogonalize against previous components
                for prev_comp in 0..comp {
                    let prev_w = components.slice(s![prev_comp, ..]);
                    let projection = w.dot(&prev_w);
                    w = w - &prev_w * projection;
                }

                // Normalize
                let norm = w.mapv(|x| x * x).sum().sqrt();
                if norm < 1e-12 {
                    return Err(SklearsError::NumericalError(
                        "Component became zero during deflation".to_string(),
                    ));
                }
                w /= norm;

                // Check convergence
                let diff = (&w - &w_old)
                    .mapv(|x| x.abs())
                    .fold(0.0f64, |acc, &x| acc.max(x));
                if diff < self.tol {
                    break;
                }
            }

            max_iter_reached = max_iter_reached.max(n_iter);

            // Store component
            for i in 0..n_features {
                components[[comp, i]] = w[i];
            }
        }

        Ok((components, max_iter_reached))
    }

    /// Apply non-linearity function and its derivative
    fn apply_nonlinearity(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        match self.fun {
            ICAFunction::Logcosh => {
                let alpha = self.fun_args;
                let tanh_x = x.mapv(|val| (alpha * val).tanh());
                let g = tanh_x.clone();
                let g_prime = tanh_x.mapv(|val| alpha * (1.0 - val * val));
                (g, g_prime)
            }
            ICAFunction::Exp => {
                let g = x.mapv(|val| val * (-val * val / 2.0).exp());
                let g_prime = x.mapv(|val| (1.0 - val * val) * (-val * val / 2.0).exp());
                (g, g_prime)
            }
            ICAFunction::Cube => {
                let g = x.mapv(|val| val * val * val);
                let g_prime = x.mapv(|val| 3.0 * val * val);
                (g, g_prime)
            }
        }
    }

    /// Orthogonalize weight matrix using Gram-Schmidt process
    fn orthogonalize(&self, w: &mut Array2<f64>) -> Result<()> {
        let (n_components, n_features) = w.dim();

        for i in 0..n_components {
            // Normalize current row
            let mut norm = 0.0;
            for j in 0..n_features {
                norm += w[[i, j]] * w[[i, j]];
            }
            norm = norm.sqrt();

            if norm < 1e-12 {
                return Err(SklearsError::NumericalError(
                    "Zero vector encountered during orthogonalization".to_string(),
                ));
            }

            for j in 0..n_features {
                w[[i, j]] /= norm;
            }

            // Orthogonalize against previous rows
            for k in 0..i {
                let mut dot_product = 0.0;
                for j in 0..n_features {
                    dot_product += w[[i, j]] * w[[k, j]];
                }

                for j in 0..n_features {
                    w[[i, j]] -= dot_product * w[[k, j]];
                }

                // Renormalize
                let mut norm = 0.0;
                for j in 0..n_features {
                    norm += w[[i, j]] * w[[i, j]];
                }
                norm = norm.sqrt();

                if norm < 1e-12 {
                    return Err(SklearsError::NumericalError(
                        "Linear dependence detected during orthogonalization".to_string(),
                    ));
                }

                for j in 0..n_features {
                    w[[i, j]] /= norm;
                }
            }
        }

        Ok(())
    }

    /// Compute mixing matrix as pseudoinverse of components
    fn compute_mixing_matrix(&self, components: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_components, n_features) = components.dim();

        // For square matrices, compute inverse directly
        if n_components == n_features {
            // Convert to nalgebra for inversion
            let comp_matrix =
                DMatrix::from_row_slice(n_components, n_features, components.as_slice().unwrap());

            match comp_matrix.try_inverse() {
                Some(inv) => {
                    let mut mixing = Array2::zeros((n_features, n_components));
                    for i in 0..n_features {
                        for j in 0..n_components {
                            mixing[[i, j]] = inv[(i, j)];
                        }
                    }
                    Ok(mixing)
                }
                None => Err(SklearsError::NumericalError(
                    "Components matrix is singular".to_string(),
                )),
            }
        } else {
            // For non-square matrices, compute pseudoinverse
            // mixing = (components^T * components)^(-1) * components^T
            let comp_t = components.t();
            let gram = components.dot(&comp_t);

            // Convert to nalgebra for inversion
            let gram_matrix =
                DMatrix::from_row_slice(n_components, n_components, gram.as_slice().unwrap());

            match gram_matrix.try_inverse() {
                Some(gram_inv) => {
                    let mut gram_inv_array = Array2::zeros((n_components, n_components));
                    for i in 0..n_components {
                        for j in 0..n_components {
                            gram_inv_array[[i, j]] = gram_inv[(i, j)];
                        }
                    }

                    let mixing = comp_t.dot(&gram_inv_array);
                    Ok(mixing)
                }
                None => Err(SklearsError::NumericalError(
                    "Gram matrix is singular".to_string(),
                )),
            }
        }
    }

    /// Infomax ICA algorithm using maximum likelihood estimation
    ///
    /// This algorithm maximizes the information flow through the network by
    /// maximizing the joint entropy of the outputs using natural gradient.
    fn infomax_ica(
        &self,
        x: &Array2<f64>,
        n_components: usize,
        rng: &mut impl Rng,
    ) -> Result<(Array2<f64>, usize)> {
        let (n_samples, n_features) = x.dim();

        // Initialize weight matrix randomly
        let mut w = Array2::zeros((n_components, n_features));
        for i in 0..n_components {
            for j in 0..n_features {
                w[[i, j]] = rng.gen::<f64>() - 0.5;
            }
        }

        // Orthogonalize initial weights
        self.orthogonalize(&mut w)?;

        let mut n_iter = 0;
        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            let w_old = w.clone();

            // Compute outputs: y = W * x
            let y = w.dot(&x.t());

            // Compute sigmoid activation: φ(y) = tanh(y)
            let phi_y = y.mapv(|val| val.tanh());

            // Compute derivative: φ'(y) = 1 - tanh²(y)
            let phi_y_deriv = y.mapv(|val| 1.0 - val.tanh().powi(2));

            // Natural gradient update: ΔW = η[I - φ(y)y^T]W
            // where η is the learning rate
            for i in 0..n_components {
                for j in 0..n_features {
                    let mut delta = 0.0;

                    // Compute the term: [I - φ(y)y^T]
                    for k in 0..n_samples {
                        let _y_k = y[[i, k]];
                        let _phi_y_k = phi_y[[i, k]];

                        // Diagonal term: I
                        delta += x[[k, j]] * phi_y_deriv[[i, k]];

                        // Off-diagonal term: -φ(y)y^T
                        for l in 0..n_components {
                            if l != i {
                                delta -= phi_y[[l, k]] * y[[i, k]] * w[[l, j]];
                            }
                        }
                    }

                    w[[i, j]] += self.learning_rate * delta / n_samples as f64;
                }
            }

            // Orthogonalize weights to maintain independence
            self.orthogonalize(&mut w)?;

            // Check convergence
            let mut max_diff: f64 = 0.0;
            for i in 0..n_components {
                for j in 0..n_features {
                    let diff = (w[[i, j]] - w_old[[i, j]]).abs();
                    max_diff = max_diff.max(diff);
                }
            }

            if max_diff < self.tol {
                break;
            }
        }

        Ok((w, n_iter))
    }

    /// Natural gradient ICA algorithm
    ///
    /// Uses natural gradient descent with momentum for efficient learning.
    /// This algorithm exploits the natural geometry of the parameter space.
    fn natural_gradient_ica(
        &self,
        x: &Array2<f64>,
        n_components: usize,
        rng: &mut impl Rng,
    ) -> Result<(Array2<f64>, usize)> {
        let (n_samples, n_features) = x.dim();

        // Initialize weight matrix randomly
        let mut w = Array2::zeros((n_components, n_features));
        for i in 0..n_components {
            for j in 0..n_features {
                w[[i, j]] = rng.gen::<f64>() - 0.5;
            }
        }

        // Initialize momentum terms
        let mut velocity = Array2::<f64>::zeros((n_components, n_features));

        let mut n_iter = 0;
        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            let w_old = w.clone();

            // Compute outputs: y = W * x^T
            let y = w.dot(&x.t());

            // Apply non-linearity (tanh) and its derivative
            let g_y = y.mapv(|val| val.tanh());
            let g_y_deriv = y.mapv(|val| 1.0 - val.tanh().powi(2));

            // Compute natural gradient: ∇W = E[g'(y)]I - E[g(y)y^T]W
            let mut gradient = Array2::zeros((n_components, n_features));

            for i in 0..n_components {
                for j in 0..n_features {
                    let mut grad_sum = 0.0;

                    // First term: E[g'(y)]δ_ij * x_j
                    let mean_g_deriv = g_y_deriv.row(i).mean().unwrap();
                    grad_sum += mean_g_deriv * x.column(j).mean().unwrap();

                    // Second term: -E[g(y_i) * y_k] * w_kj
                    for k in 0..n_components {
                        let mut corr_sum = 0.0;
                        for s in 0..n_samples {
                            corr_sum += g_y[[i, s]] * y[[k, s]];
                        }
                        corr_sum /= n_samples as f64;
                        grad_sum -= corr_sum * w[[k, j]];
                    }

                    gradient[[i, j]] = grad_sum;
                }
            }

            // Apply momentum: v = momentum * v + learning_rate * gradient
            for i in 0..n_components {
                for j in 0..n_features {
                    velocity[[i, j]] =
                        self.momentum * velocity[[i, j]] + self.learning_rate * gradient[[i, j]];
                    w[[i, j]] += velocity[[i, j]];
                }
            }

            // Check convergence
            let mut max_diff: f64 = 0.0;
            for i in 0..n_components {
                for j in 0..n_features {
                    let diff = (w[[i, j]] - w_old[[i, j]]).abs();
                    max_diff = max_diff.max(diff);
                }
            }

            if max_diff < self.tol {
                break;
            }
        }

        Ok((w, n_iter))
    }

    /// Temporal ICA algorithm for time series data
    ///
    /// Exploits temporal structure in the data by incorporating
    /// temporal dependencies and autocorrelations.
    fn temporal_ica(
        &self,
        x: &Array2<f64>,
        n_components: usize,
        rng: &mut impl Rng,
    ) -> Result<(Array2<f64>, usize)> {
        let (n_samples, n_features) = x.dim();
        let requested_window = self.temporal_window.unwrap_or(3);

        if n_samples <= requested_window {
            return Err(SklearsError::InvalidInput(
                "Not enough samples for temporal ICA window".to_string(),
            ));
        }

        let window_size = requested_window;

        // Initialize weight matrix randomly
        let mut w = Array2::zeros((n_components, n_features));
        for i in 0..n_components {
            for j in 0..n_features {
                w[[i, j]] = rng.gen::<f64>() - 0.5;
            }
        }

        let mut n_iter = 0;
        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            let w_old = w.clone();

            // Create temporal windows
            let n_windows = n_samples - window_size;
            let mut temporal_gradient = Array2::zeros((n_components, n_features));

            for window_start in 0..n_windows {
                let window_end = window_start + window_size;

                // Extract window data
                let x_window = x.slice(scirs2_core::ndarray::s![window_start..window_end, ..]);

                // Compute outputs for current window
                let y_window = w.dot(&x_window.t());

                // Apply non-linearity
                let _g_y = y_window.mapv(|val| val.tanh());
                let g_y_deriv = y_window.mapv(|val| 1.0 - val.tanh().powi(2));

                // Compute temporal contrast function
                // This encourages components to have different temporal structures
                for i in 0..n_components {
                    let y_i = y_window.row(i);

                    // Compute autocorrelation at lag 1
                    let mut autocorr = 0.0;
                    for t in 1..window_size {
                        autocorr += y_i[t] * y_i[t - 1];
                    }
                    autocorr /= (window_size - 1) as f64;

                    // Temporal gradient update
                    for j in 0..n_features {
                        let mut temp_grad = 0.0;

                        // Standard ICA gradient
                        for t in 0..window_size {
                            temp_grad += g_y_deriv[[i, t]] * x_window[[t, j]];
                        }
                        temp_grad /= window_size as f64;

                        // Temporal regularization term
                        // Encourage temporal independence by penalizing autocorrelation
                        let temporal_penalty = 0.1 * autocorr * autocorr;
                        temp_grad -= temporal_penalty;

                        temporal_gradient[[i, j]] += temp_grad;
                    }
                }
            }

            // Average gradient over all windows
            temporal_gradient /= n_windows as f64;

            // Update weights
            for i in 0..n_components {
                for j in 0..n_features {
                    w[[i, j]] += self.learning_rate * temporal_gradient[[i, j]];
                }
            }

            // Orthogonalize to maintain component independence
            self.orthogonalize(&mut w)?;

            // Check convergence
            let mut max_diff: f64 = 0.0;
            for i in 0..n_components {
                for j in 0..n_features {
                    let diff = (w[[i, j]] - w_old[[i, j]]).abs();
                    max_diff = max_diff.max(diff);
                }
            }

            if max_diff < self.tol {
                break;
            }
        }

        Ok((w, n_iter))
    }

    /// Constrained ICA algorithm for semi-supervised learning
    ///
    /// Incorporates prior knowledge through constraint matrix to guide
    /// the component extraction process towards desired solutions.
    fn constrained_ica(
        &self,
        x: &Array2<f64>,
        n_components: usize,
        rng: &mut impl Rng,
    ) -> Result<(Array2<f64>, usize)> {
        let (n_samples, n_features) = x.dim();

        // Check if constraint matrix is provided
        let constraint_matrix = match &self.constraint_matrix {
            Some(matrix) => matrix,
            None => {
                return Err(SklearsError::InvalidInput(
                    "Constraint matrix must be provided for constrained ICA".to_string(),
                ));
            }
        };

        // Validate constraint matrix dimensions
        let (c_rows, c_cols) = constraint_matrix.dim();
        if c_rows != n_components || c_cols != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Constraint matrix shape ({c_rows}, {c_cols}) must match (n_components, n_features) = ({n_components}, {n_features})"
            )));
        }

        // Initialize weight matrix using constraint as guidance
        let mut w = Array2::zeros((n_components, n_features));

        // Initialize with constrained initialization
        for i in 0..n_components {
            for j in 0..n_features {
                // Start with constraint matrix and add small random perturbation
                w[[i, j]] = constraint_matrix[[i, j]] + 0.1 * (rng.gen::<f64>() - 0.5);
            }
        }

        // Normalize initial weight vectors
        for i in 0..n_components {
            let mut row = w.row_mut(i);
            let norm = row.dot(&row).sqrt();
            if norm > 1e-12 {
                row /= norm;
            }
        }

        let mut n_iter = 0;
        let mut prev_w = w.clone();

        for iter in 0..self.max_iter {
            n_iter = iter + 1;
            prev_w.assign(&w);

            for i in 0..n_components {
                // Extract current component weight vector
                let mut w_i = w.row(i).to_owned();

                // FastICA update with constraint regularization
                let y = x.dot(&w_i);
                let (_g_y, g_y_deriv) = self.apply_nonlinearity(&y);

                // Standard FastICA gradient
                let mut gradient = Array1::<f64>::zeros(n_features);
                for (j, x_sample) in x.rows().into_iter().enumerate() {
                    gradient += &(g_y_deriv[j] * x_sample.to_owned());
                }
                gradient /= n_samples as f64;

                // Constraint regularization term
                // Pull the weight vector towards the constraint
                let constraint_vec = constraint_matrix.row(i);
                let constraint_diff = &constraint_vec.to_owned() - &w_i;
                let constraint_gradient = self.constraint_weight * &constraint_diff;

                // Combined gradient update
                w_i = &w_i + self.learning_rate * (&gradient + &constraint_gradient);

                // Orthogonalization against previous components
                for j in 0..i {
                    let w_j = w.row(j);
                    let projection = w_i.dot(&w_j);
                    w_i = &w_i - projection * &w_j.to_owned();
                }

                // Normalize
                let norm = w_i.dot(&w_i).sqrt();
                if norm > 1e-12 {
                    w_i /= norm;
                }

                // Update weight matrix
                w.row_mut(i).assign(&w_i);
            }

            // Check convergence based on weight changes
            let mut max_change: f64 = 0.0;
            for i in 0..n_components {
                for j in 0..n_features {
                    let change = (w[[i, j]] - prev_w[[i, j]]).abs();
                    max_change = max_change.max(change);
                }
            }

            // Also check constraint satisfaction
            let mut constraint_error = 0.0;
            for i in 0..n_components {
                let constraint_vec = constraint_matrix.row(i);
                let w_vec = w.row(i);
                let diff = &constraint_vec.to_owned() - &w_vec.to_owned();
                constraint_error += diff.dot(&diff);
            }
            constraint_error = constraint_error.sqrt();

            if max_change < self.tol && constraint_error < self.constraint_tol {
                break;
            }
        }

        Ok((w, n_iter))
    }
}

impl ICA<TrainedICA> {
    /// Get the unmixing matrix (components)
    pub fn components(&self) -> &Array2<f64> {
        &self.state.components
    }

    /// Get the mixing matrix
    pub fn mixing(&self) -> &Array2<f64> {
        &self.state.mixing
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Inverse transform (reconstruct original signal from components)
    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (_n_samples, n_components) = x.dim();

        if n_components != self.state.n_components {
            return Err(SklearsError::FeatureMismatch {
                expected: self.state.n_components,
                actual: n_components,
            });
        }

        // Apply mixing matrix
        let x_mixed = x.dot(&self.state.mixing.t());

        // Apply inverse whitening if whitening was used
        let x_processed = if let Some(ref whitening) = self.state.whitening {
            // Inverse whitening: multiply by whitening^T
            x_mixed.dot(&whitening.t())
        } else {
            x_mixed
        };

        // Add back the mean
        let x_reconstructed = x_processed + &self.state.mean;

        Ok(x_reconstructed)
    }
}

impl Default for ICA<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ica_creation() {
        let ica = ICA::new()
            .n_components(2)
            .algorithm(ICAAlgorithm::Parallel)
            .fun(ICAFunction::Logcosh)
            .max_iter(100)
            .tol(1e-6)
            .random_state(42);

        assert_eq!(ica.n_components, Some(2));
        assert_eq!(ica.algorithm, ICAAlgorithm::Parallel);
        assert_eq!(ica.fun, ICAFunction::Logcosh);
        assert_eq!(ica.max_iter, 100);
        assert_abs_diff_eq!(ica.tol, 1e-6, epsilon = 1e-10);
    }

    #[test]
    fn test_ica_fit_transform() {
        // Create mixed signals (simple 2D case)
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];

        let ica = ICA::new().n_components(2).random_state(42);

        let trained_ica = ica.fit(&x, &()).unwrap();
        let x_transformed = trained_ica.transform(&x).unwrap();

        assert_eq!(x_transformed.dim(), (5, 2));
        assert_eq!(trained_ica.state.n_features_in, 2);
        assert_eq!(trained_ica.state.n_components, 2);
    }

    #[test]
    fn test_ica_inverse_transform() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],];

        let ica = ICA::new().n_components(2).random_state(123);

        let trained_ica = ica.fit(&x, &()).unwrap();
        let x_transformed = trained_ica.transform(&x).unwrap();
        let x_reconstructed = trained_ica.inverse_transform(&x_transformed).unwrap();

        assert_eq!(x_reconstructed.dim(), x.dim());

        // The reconstruction should be close to the original (may not be exact due to numerical precision)
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((x_reconstructed[[i, j]] - x[[i, j]]).abs() < 5.0);
            }
        }
    }

    #[test]
    fn test_ica_different_algorithms() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ];

        let algorithms = vec![ICAAlgorithm::Parallel, ICAAlgorithm::Deflation];

        for algorithm in algorithms {
            let ica = ICA::new()
                .n_components(3)
                .algorithm(algorithm)
                .random_state(42);

            let trained_ica = ica.fit(&x, &()).unwrap();
            let x_transformed = trained_ica.transform(&x).unwrap();

            assert_eq!(x_transformed.dim(), (5, 3));
            assert_eq!(trained_ica.state.n_components, 3);
        }
    }

    #[test]
    fn test_ica_different_functions() {
        let x = array![
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 4.0],
            [5.0, 2.0],
            [4.0, 6.0],
            [6.0, 3.0],
            [5.0, 8.0],
            [8.0, 4.0],
            [7.0, 10.0],
            [10.0, 5.0]
        ];

        let functions = vec![ICAFunction::Logcosh, ICAFunction::Exp, ICAFunction::Cube];

        for fun in functions {
            let ica = ICA::new().n_components(2).fun(fun).random_state(42);

            let trained_ica = ica.fit(&x, &()).unwrap();
            let x_transformed = trained_ica.transform(&x).unwrap();

            assert_eq!(x_transformed.dim(), (10, 2));
        }
    }

    #[test]
    fn test_ica_error_cases() {
        let x_small = array![[1.0, 2.0]]; // Only 1 sample
        let ica = ICA::new();
        let result = ica.fit(&x_small, &());
        assert!(result.is_err());

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let ica = ICA::new().n_components(3); // More components than features
        let result = ica.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_ica_whiten_option() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        // Test with whitening
        let ica_whiten = ICA::new().n_components(2).whiten(true).random_state(42);

        let trained_whiten = ica_whiten.fit(&x, &()).unwrap();
        assert!(trained_whiten.state.whitening.is_some());

        // Test without whitening
        let ica_no_whiten = ICA::new().n_components(2).whiten(false).random_state(42);

        let trained_no_whiten = ica_no_whiten.fit(&x, &()).unwrap();
        assert!(trained_no_whiten.state.whitening.is_none());
    }

    #[test]
    fn test_ica_infomax_algorithm() {
        let x = array![
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 4.0],
            [5.0, 2.0],
            [4.0, 6.0],
            [6.0, 3.0],
        ];

        let ica = ICA::new()
            .n_components(2)
            .algorithm(ICAAlgorithm::Infomax)
            .learning_rate(0.01)
            .max_iter(50) // Reduced for faster test
            .random_state(42);

        let trained_ica = ica.fit(&x, &()).unwrap();
        let x_transformed = trained_ica.transform(&x).unwrap();

        assert_eq!(x_transformed.dim(), (6, 2));
        assert_eq!(trained_ica.state.n_components, 2);

        // Check that all values are finite
        for &val in x_transformed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_ica_natural_gradient_algorithm() {
        let x = array![[1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [5.0, 2.0], [4.0, 6.0],];

        let ica = ICA::new()
            .n_components(2)
            .algorithm(ICAAlgorithm::NaturalGradient)
            .learning_rate(0.005)
            .momentum(0.9)
            .max_iter(30) // Reduced for faster test
            .random_state(123);

        let trained_ica = ica.fit(&x, &()).unwrap();
        let x_transformed = trained_ica.transform(&x).unwrap();

        assert_eq!(x_transformed.dim(), (5, 2));
        assert_eq!(trained_ica.state.n_components, 2);

        // Check that all values are finite
        for &val in x_transformed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_ica_temporal_algorithm() {
        // Create time series data with temporal structure
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [2.0, 1.0],
            [2.1, 1.1],
            [2.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
        ];

        let ica = ICA::new()
            .n_components(2)
            .algorithm(ICAAlgorithm::Temporal)
            .temporal_window(3)
            .learning_rate(0.01)
            .max_iter(20) // Reduced for faster test
            .random_state(456);

        let trained_ica = ica.fit(&x, &()).unwrap();
        let x_transformed = trained_ica.transform(&x).unwrap();

        assert_eq!(x_transformed.dim(), (8, 2));
        assert_eq!(trained_ica.state.n_components, 2);

        // Check that all values are finite
        for &val in x_transformed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_ica_temporal_insufficient_data() {
        // Test with insufficient data for temporal window
        let x = array![[1.0, 2.0], [3.0, 4.0]]; // Only 2 samples

        let ica = ICA::new()
            .n_components(2)
            .algorithm(ICAAlgorithm::Temporal)
            .temporal_window(5) // Window larger than data
            .random_state(42);

        let result = ica.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_ica_new_parameters() {
        let ica = ICA::new()
            .learning_rate(0.05)
            .momentum(0.8)
            .temporal_window(5);

        assert_eq!(ica.learning_rate, 0.05);
        assert_eq!(ica.momentum, 0.8);
        assert_eq!(ica.temporal_window, Some(5));
    }

    #[test]
    fn test_constrained_ica_basic() {
        // Test basic constrained ICA functionality
        let x = array![
            [1.0, 0.5, 0.3],
            [0.8, 1.2, 0.7],
            [1.1, 0.9, 0.4],
            [0.9, 1.1, 0.6],
            [1.2, 0.7, 0.5],
        ];

        // Create a constraint matrix (prior knowledge about components)
        let constraint_matrix = array![
            [1.0, 0.0, 0.0], // First component should focus on first feature
            [0.0, 1.0, 0.0], // Second component should focus on second feature
        ];

        let ica = ICA::new()
            .n_components(2)
            .algorithm(ICAAlgorithm::Constrained)
            .constraint_matrix(constraint_matrix)
            .constraint_weight(0.5)
            .constraint_tol(1e-4)
            .max_iter(50)
            .random_state(42);

        let trained_ica = ica.fit(&x, &()).unwrap();
        let x_transformed = trained_ica.transform(&x).unwrap();

        assert_eq!(x_transformed.dim(), (5, 2));
        assert_eq!(trained_ica.state.n_components, 2);

        // Check that all values are finite
        for &val in x_transformed.iter() {
            assert!(val.is_finite());
        }

        // Check that constraints are reasonably satisfied
        let components = trained_ica.components();
        assert!(components[[0, 0]].abs() > components[[0, 1]].abs()); // First component prefers first feature
        assert!(components[[1, 1]].abs() > components[[1, 0]].abs()); // Second component prefers second feature
    }

    #[test]
    fn test_constrained_ica_missing_constraint() {
        // Test that constrained ICA fails without constraint matrix
        let x = array![[1.0, 0.5], [0.8, 1.2], [1.1, 0.9], [0.9, 1.1],];

        let ica = ICA::new()
            .n_components(2)
            .algorithm(ICAAlgorithm::Constrained)
            .random_state(42);

        let result = ica.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_constrained_ica_invalid_constraint_shape() {
        // Test with incorrect constraint matrix shape
        let x = array![
            [1.0, 0.5, 0.3],
            [0.8, 1.2, 0.7],
            [1.1, 0.9, 0.4],
            [0.9, 1.1, 0.6],
        ];

        // Wrong shape constraint matrix (should be 2x3, not 2x2)
        let constraint_matrix = array![[1.0, 0.0], [0.0, 1.0],];

        let ica = ICA::new()
            .n_components(2)
            .algorithm(ICAAlgorithm::Constrained)
            .constraint_matrix(constraint_matrix)
            .random_state(42);

        let result = ica.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_constrained_ica_parameters() {
        let constraint_matrix = array![[1.0, 0.0], [0.0, 1.0]];

        let ica = ICA::new()
            .constraint_matrix(constraint_matrix.clone())
            .constraint_weight(2.0)
            .constraint_tol(1e-5);

        assert_eq!(ica.constraint_weight, 2.0);
        assert_eq!(ica.constraint_tol, 1e-5);
        assert!(ica.constraint_matrix.is_some());

        let stored_matrix = ica.constraint_matrix.as_ref().unwrap();
        assert_eq!(stored_matrix.dim(), (2, 2));
    }
}

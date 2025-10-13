//! Nyström approximation for scalable Gaussian Process regression
//!
//! The Nyström method provides a low-rank approximation to the kernel matrix,
//! reducing computational complexity from O(n³) to O(nm²) where m << n.

// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Fit, Predict, Untrained},
};

use crate::kernels::Kernel;
use crate::utils::{robust_cholesky, triangular_solve};

///
/// let X = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
/// let y = array![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
///
/// let kernel = RBF::new(1.0);
/// let nystrom_gpr = NystromGaussianProcessRegressor::new()
///     .kernel(Box::new(kernel))
///     .n_components(2);
/// let fitted = nystrom_gpr.fit(&X.view(), &y.view()).unwrap();
/// // Predictions would be: fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NystromGaussianProcessRegressor<S = Untrained> {
    state: S,
    kernel: Option<Box<dyn Kernel>>,
    n_components: usize,
    landmark_selection: LandmarkSelection,
    regularization: f64,
    normalize_y: bool,
    copy_x_train: bool,
    random_state: Option<u64>,
}

/// Methods for selecting landmark points
#[derive(Debug, Clone)]
pub enum LandmarkSelection {
    /// Random selection of landmarks
    Random,
    /// Uniform sampling from the data range
    Uniform,
    /// K-means clustering to find representative points
    KMeans,
    /// Farthest point sampling for better coverage
    FarthestPoint,
}

/// Trained state for Nyström Gaussian Process Regressor
#[derive(Debug, Clone)]
pub struct NystromGprTrained {
    /// X_train
    pub X_train: Option<Array2<f64>>, // Training inputs (optional)
    /// y_train
    pub y_train: Array1<f64>, // Training outputs
    /// landmarks
    pub landmarks: Array2<f64>, // Selected landmark points
    /// alpha
    pub alpha: Array1<f64>, // Solution coefficients
    /// K_mm
    pub K_mm: Array2<f64>, // Kernel matrix between landmarks
    /// K_nm
    pub K_nm: Array2<f64>, // Kernel matrix between data and landmarks
    /// L_mm
    pub L_mm: Array2<f64>, // Cholesky decomposition of K_mm
    /// eigenvalues
    pub eigenvalues: Array1<f64>, // Eigenvalues of the approximation
    /// eigenvectors
    pub eigenvectors: Array2<f64>, // Eigenvectors of the approximation
    /// kernel
    pub kernel: Box<dyn Kernel>, // Kernel function
    /// y_mean
    pub y_mean: f64, // Mean of training targets
    /// y_std
    pub y_std: f64, // Standard deviation of training targets
    /// sigma_n
    pub sigma_n: f64, // Noise level
    /// log_marginal_likelihood_value
    pub log_marginal_likelihood_value: f64, // Log marginal likelihood
}

impl NystromGaussianProcessRegressor<Untrained> {
    /// Create a new NystromGaussianProcessRegressor instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: None,
            n_components: 100,
            landmark_selection: LandmarkSelection::KMeans,
            regularization: 1e-10,
            normalize_y: true,
            copy_x_train: false,
            random_state: None,
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the number of Nyström components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the landmark selection method
    pub fn landmark_selection(mut self, landmark_selection: LandmarkSelection) -> Self {
        self.landmark_selection = landmark_selection;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set whether to normalize targets
    pub fn normalize_y(mut self, normalize_y: bool) -> Self {
        self.normalize_y = normalize_y;
        self
    }

    /// Set whether to copy X during training
    pub fn copy_x_train(mut self, copy_x_train: bool) -> Self {
        self.copy_x_train = copy_x_train;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, f64>> for NystromGaussianProcessRegressor<Untrained> {
    type Fitted = NystromGaussianProcessRegressor<NystromGprTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let kernel = self
            .kernel
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?;
        let n_samples = X.nrows();
        let n_components = self.n_components.min(n_samples);

        // Normalize targets if requested
        let (y_normalized, y_mean, y_std) = if self.normalize_y {
            let mean = y.mean().unwrap_or(0.0);
            let std =
                ((y.mapv(|x| (x - mean).powi(2)).sum() / (y.len() - 1) as f64).sqrt()).max(1e-12);
            let y_norm = y.mapv(|x| (x - mean) / std);
            (y_norm, mean, std)
        } else {
            (y.to_owned(), 0.0, 1.0)
        };

        // Select landmark points
        let landmarks =
            select_landmarks(X, n_components, &self.landmark_selection, self.random_state)?;

        // Compute kernel matrices
        let K_mm = kernel.compute_kernel_matrix(&landmarks, None)?;
        let X_owned = X.to_owned();
        let K_nm = kernel.compute_kernel_matrix(&X_owned, Some(&landmarks))?;

        // Add regularization to K_mm
        let mut K_mm_reg = K_mm.clone();
        for i in 0..K_mm_reg.nrows() {
            K_mm_reg[[i, i]] += self.regularization;
        }

        // Cholesky decomposition of K_mm
        let L_mm = robust_cholesky(&K_mm_reg)?;

        // Nyström approximation: K ≈ K_nm * K_mm^{-1} * K_mn
        // Solve K_mm * W = K_nm^T for W
        let mut W = Array2::<f64>::zeros((n_components, n_samples));
        for i in 0..n_samples {
            let k_nm_i = K_nm.row(i).to_owned();
            let w_i = triangular_solve(&L_mm, &k_nm_i)?;
            W.column_mut(i).assign(&w_i);
        }

        // Compute eigendecomposition of the Nyström approximation
        // G = K_nm^T * K_mm^{-1} * K_nm / n
        let G = W.t().dot(&W) / n_samples as f64;
        let (eigenvalues, eigenvectors) = eigendecomposition(&G)?;

        // Solve for alpha using the Nyström approximation
        // (K + σ²I) α = y ≈ (G + σ²I) α = y
        let sigma_n_sq = self.regularization;
        let mut G_reg = G.clone();
        for i in 0..G_reg.nrows() {
            G_reg[[i, i]] += sigma_n_sq;
        }

        let L_G = robust_cholesky(&G_reg)?;
        let alpha = triangular_solve(&L_G, &y_normalized)?;

        // Compute log marginal likelihood (approximation)
        let log_marginal_likelihood_value = {
            let log_det_G = 2.0 * L_G.diag().mapv(|x| x.ln()).sum();
            let quadratic = alpha.dot(&y_normalized);
            -0.5 * (quadratic + log_det_G + n_samples as f64 * (2.0 * std::f64::consts::PI).ln())
        };

        let X_train = if self.copy_x_train {
            Some(X.to_owned())
        } else {
            None
        };

        Ok(NystromGaussianProcessRegressor {
            state: NystromGprTrained {
                X_train,
                y_train: y.to_owned(),
                landmarks,
                alpha,
                K_mm: K_mm_reg,
                K_nm,
                L_mm,
                eigenvalues,
                eigenvectors,
                kernel,
                y_mean,
                y_std,
                sigma_n: self.regularization.sqrt(),
                log_marginal_likelihood_value,
            },
            kernel: None,
            n_components: self.n_components,
            landmark_selection: self.landmark_selection,
            regularization: self.regularization,
            normalize_y: self.normalize_y,
            copy_x_train: self.copy_x_train,
            random_state: self.random_state,
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<f64>>
    for NystromGaussianProcessRegressor<NystromGprTrained>
{
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let (mean, _) = self.predict_with_std(X)?;
        Ok(mean)
    }
}

impl NystromGaussianProcessRegressor<NystromGprTrained> {
    /// Predict with uncertainty estimates
    #[allow(non_snake_case)]
    pub fn predict_with_std(&self, X: &ArrayView2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        // Compute kernel between test points and landmarks
        let X_owned = X.to_owned();
        let K_test_landmarks = self
            .state
            .kernel
            .compute_kernel_matrix(&X_owned, Some(&self.state.landmarks))?;

        // Project test points into the Nyström subspace
        // Use the precomputed Cholesky factor
        let mut W_test = Array2::<f64>::zeros((self.state.landmarks.nrows(), X.nrows()));
        for i in 0..X.nrows() {
            let k_test_i = K_test_landmarks.row(i).to_owned();
            let w_test_i = triangular_solve(&self.state.L_mm, &k_test_i)?;
            W_test.column_mut(i).assign(&w_test_i);
        }

        // Predict mean using Nyström approximation
        // f(x*) = W_test^T * α
        let mean_normalized = W_test.t().dot(&self.state.alpha);

        // Denormalize predictions
        let mean = if self.normalize_y {
            mean_normalized.mapv(|x| x * self.state.y_std + self.state.y_mean)
        } else {
            mean_normalized
        };

        // Predict variance (simplified)
        // Var[f(x*)] = k(x*, x*) - W_test^T * W_test + σ²
        let k_test_diag = X
            .axis_iter(Axis(0))
            .map(|x| self.state.kernel.kernel(&x, &x))
            .collect::<Array1<f64>>();

        let var_reduction = W_test
            .t()
            .axis_iter(Axis(0))
            .map(|w| w.dot(&w))
            .collect::<Array1<f64>>();

        let variance = k_test_diag - var_reduction + self.state.sigma_n.powi(2);
        let std = variance.mapv(|x| x.sqrt().max(1e-10));

        Ok((mean, std))
    }

    /// Get the log marginal likelihood
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.state.log_marginal_likelihood_value
    }

    /// Get the landmark points
    pub fn landmarks(&self) -> &Array2<f64> {
        &self.state.landmarks
    }

    /// Get the eigenvalues of the Nyström approximation
    pub fn eigenvalues(&self) -> &Array1<f64> {
        &self.state.eigenvalues
    }

    /// Get the eigenvectors of the Nyström approximation
    pub fn eigenvectors(&self) -> &Array2<f64> {
        &self.state.eigenvectors
    }

    /// Get the approximation rank (number of positive eigenvalues)
    pub fn effective_rank(&self) -> usize {
        self.state
            .eigenvalues
            .iter()
            .filter(|&&x| x > 1e-12)
            .count()
    }
}

impl Default for NystromGaussianProcessRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Select landmark points based on the specified method
fn select_landmarks(
    X: &ArrayView2<f64>,
    n_components: usize,
    method: &LandmarkSelection,
    random_state: Option<u64>,
) -> SklResult<Array2<f64>> {
    match method {
        LandmarkSelection::Random => {
            crate::utils::random_inducing_points(X, n_components, random_state)
        }
        LandmarkSelection::Uniform => {
            crate::utils::uniform_inducing_points(X, n_components, random_state)
        }
        LandmarkSelection::KMeans => {
            crate::utils::kmeans_inducing_points(X, n_components, random_state)
        }
        LandmarkSelection::FarthestPoint => farthest_point_sampling(X, n_components, random_state),
    }
}

/// Farthest point sampling for landmark selection
fn farthest_point_sampling(
    X: &ArrayView2<f64>,
    n_components: usize,
    random_state: Option<u64>,
) -> SklResult<Array2<f64>> {
    let n_samples = X.nrows();
    let n_features = X.ncols();

    if n_components >= n_samples {
        return Ok(X.to_owned());
    }

    let mut rng = random_state.unwrap_or(42);

    let mut selected_indices = Vec::new();
    let mut landmarks = Array2::<f64>::zeros((n_components, n_features));

    // Start with a random point
    let start_idx = (rng as usize) % n_samples;
    selected_indices.push(start_idx);
    landmarks.row_mut(0).assign(&X.row(start_idx));

    // Greedily select points that are farthest from already selected points
    for i in 1..n_components {
        let mut max_min_dist = -1.0;
        let mut best_idx = 0;

        for j in 0..n_samples {
            if selected_indices.contains(&j) {
                continue;
            }

            // Find minimum distance to already selected points
            let mut min_dist = f64::INFINITY;
            for &selected_idx in &selected_indices {
                let dist = (&X.row(j) - &X.row(selected_idx))
                    .mapv(|x| x.powi(2))
                    .sum()
                    .sqrt();
                min_dist = min_dist.min(dist);
            }

            if min_dist > max_min_dist {
                max_min_dist = min_dist;
                best_idx = j;
            }
        }

        selected_indices.push(best_idx);
        landmarks.row_mut(i).assign(&X.row(best_idx));

        // Update RNG for next iteration
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
    }

    Ok(landmarks)
}

/// Simple eigendecomposition for symmetric matrices
fn eigendecomposition(A: &Array2<f64>) -> SklResult<(Array1<f64>, Array2<f64>)> {
    // Simplified eigendecomposition - in practice, would use LAPACK
    // For now, return identity as a placeholder
    let n = A.nrows();
    let eigenvalues = Array1::<f64>::ones(n);
    let eigenvectors = Array2::<f64>::eye(n);
    Ok((eigenvalues, eigenvectors))
}

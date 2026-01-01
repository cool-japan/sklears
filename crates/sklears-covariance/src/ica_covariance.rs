//! Independent Component Analysis for Covariance Estimation
//!
//! This module implements ICA-based covariance estimation methods. ICA finds statistically
//! independent components in the data, which can be useful for understanding the underlying
//! structure and estimating covariance matrices that capture non-Gaussian dependencies.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use scirs2_linalg::compat::ArrayLinalgExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

/// ICA-based covariance estimator
///
/// Uses Independent Component Analysis to decompose the covariance structure into
/// statistically independent components. This is particularly useful for data
/// with non-Gaussian structure and mixed signals.
#[derive(Debug, Clone)]
pub struct ICACovariance<S = Untrained> {
    state: S,
    /// Number of independent components
    n_components: Option<usize>,
    /// ICA algorithm to use
    algorithm: ICAAlgorithm,
    /// Contrast function for FastICA
    fun: ContrastFunction,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Whether to whiten the data
    whiten: bool,
    /// Whitening method
    whiten_method: WhiteningMethod,
    /// Random state for reproducible results
    random_state: Option<u64>,
    /// Alpha parameter for contrast functions
    alpha: f64,
}

/// ICA algorithms
#[derive(Debug, Clone)]
pub enum ICAAlgorithm {
    FastICA,
    ExtendedInfomax,
    JADE,
    NaturalGradient,
    MutualInfo,
}

/// Contrast functions for FastICA
#[derive(Debug, Clone)]
pub enum ContrastFunction {
    /// Logcosh function (default)
    Logcosh,
    /// Exponential function
    Exp,
    /// Cubic function
    Cube,
    /// Tanh function
    Tanh,
}

/// Whitening methods
#[derive(Debug, Clone)]
pub enum WhiteningMethod {
    /// PCA-based whitening
    PCA,
    /// ZCA (Zero-phase Component Analysis) whitening
    ZCA,
    /// Cholesky whitening
    Cholesky,
}

/// Trained ICA Covariance state
#[derive(Debug, Clone)]
pub struct ICACovarianceTrained {
    /// Estimated covariance matrix
    covariance: Array2<f64>,
    /// Precision matrix (inverse covariance)
    precision: Option<Array2<f64>>,
    /// Unmixing matrix (W)
    unmixing_matrix: Array2<f64>,
    /// Mixing matrix (A = W^-1)
    mixing_matrix: Array2<f64>,
    /// Independent components
    components: Array2<f64>,
    /// Whitening matrix
    whitening_matrix: Array2<f64>,
    /// Mean of the training data
    mean: Array1<f64>,
    /// Number of components
    n_components: usize,
    /// Algorithm used
    algorithm: ICAAlgorithm,
    /// Number of iterations performed
    n_iter: usize,
    /// Final convergence value
    convergence: f64,
    /// Kurtosis of components (measure of non-Gaussianity)
    kurtosis: Array1<f64>,
    /// Negentropy of components
    negentropy: Array1<f64>,
}

impl Default for ICACovariance {
    fn default() -> Self {
        Self::new()
    }
}

impl ICACovariance {
    /// Creates a new ICA covariance estimator
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: None,
            algorithm: ICAAlgorithm::FastICA,
            fun: ContrastFunction::Logcosh,
            max_iter: 200,
            tol: 1e-4,
            whiten: true,
            whiten_method: WhiteningMethod::PCA,
            random_state: None,
            alpha: 1.0,
        }
    }

    /// Sets the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    /// Sets the ICA algorithm
    pub fn algorithm(mut self, algorithm: ICAAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Sets the contrast function for FastICA
    pub fn contrast_function(mut self, fun: ContrastFunction) -> Self {
        self.fun = fun;
        self
    }

    /// Sets the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets whether to whiten the data
    pub fn whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    /// Sets the whitening method
    pub fn whiten_method(mut self, method: WhiteningMethod) -> Self {
        self.whiten_method = method;
        self
    }

    /// Sets the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Sets the alpha parameter for contrast functions
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ICAConfig {
    pub algorithm: ICAAlgorithm,
    pub n_components: Option<usize>,
    pub whitening_method: WhiteningMethod,
    pub max_iter: usize,
    pub tol: f64,
    pub random_state: Option<u64>,
}

impl Estimator for ICACovariance {
    type Config = ICAConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static CONFIG: std::sync::OnceLock<ICAConfig> = std::sync::OnceLock::new();
        CONFIG.get_or_init(|| ICAConfig {
            algorithm: ICAAlgorithm::FastICA,
            n_components: None,
            whitening_method: WhiteningMethod::PCA,
            max_iter: 200,
            tol: 1e-4,
            random_state: None,
        })
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for ICACovariance {
    type Fitted = ICACovariance<ICACovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "ICA requires at least 2 samples".to_string(),
            ));
        }

        let n_components = self.n_components.unwrap_or(n_features);

        if n_components > n_features {
            return Err(SklearsError::InvalidInput(
                "Number of components cannot exceed number of features".to_string(),
            ));
        }

        // Center the data
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut x_centered = x.to_owned();
        for mut row in x_centered.axis_iter_mut(Axis(0)) {
            row -= &mean;
        }

        // Whiten the data if requested
        let (x_whitened, whitening_matrix) = if self.whiten {
            self.whiten_data(&x_centered, n_components)?
        } else {
            (x_centered, Array2::eye(n_features))
        };

        // Apply ICA algorithm
        let (unmixing_matrix, n_iter, convergence) = match &self.algorithm {
            ICAAlgorithm::FastICA => self.fast_ica(&x_whitened, n_components)?,
            ICAAlgorithm::ExtendedInfomax => self.extended_infomax(&x_whitened, n_components)?,
            ICAAlgorithm::JADE => self.jade(&x_whitened, n_components)?,
            ICAAlgorithm::NaturalGradient => self.natural_gradient(&x_whitened, n_components)?,
            ICAAlgorithm::MutualInfo => self.mutual_info(&x_whitened, n_components)?,
        };

        // Compute mixing matrix
        let mixing_matrix = unmixing_matrix.inv().map_err(|_| {
            SklearsError::NumericalError("Failed to compute mixing matrix".to_string())
        })?;

        // Extract independent components
        let components = unmixing_matrix.dot(&x_whitened.t());

        // Compute covariance from independent components
        // For ICA, we assume independence, so covariance should be diagonal in ICA space
        let ica_cov = components.dot(&components.t()) / (n_samples - 1) as f64;

        // Transform back to original space: C = A * D * A^T
        // where D is the diagonal covariance in ICA space
        let full_mixing = if self.whiten {
            whitening_matrix
                .inv()
                .map_err(|_| {
                    SklearsError::NumericalError("Failed to invert whitening matrix".to_string())
                })?
                .dot(&mixing_matrix)
        } else {
            mixing_matrix.clone()
        };

        let covariance = full_mixing.dot(&ica_cov).dot(&full_mixing.t());

        // Compute precision matrix
        let precision = covariance.inv().ok();

        // Compute component statistics
        let kurtosis = self.compute_kurtosis(&components)?;
        let negentropy = self.compute_negentropy(&components)?;

        let trained_state = ICACovarianceTrained {
            covariance,
            precision,
            unmixing_matrix,
            mixing_matrix,
            components: components.t().to_owned(),
            whitening_matrix,
            mean,
            n_components,
            algorithm: self.algorithm.clone(),
            n_iter,
            convergence,
            kurtosis,
            negentropy,
        };

        Ok(ICACovariance {
            state: trained_state,
            n_components: self.n_components,
            algorithm: self.algorithm,
            fun: self.fun,
            max_iter: self.max_iter,
            tol: self.tol,
            whiten: self.whiten,
            whiten_method: self.whiten_method,
            random_state: self.random_state,
            alpha: self.alpha,
        })
    }
}

impl ICACovariance {
    /// Whiten the data using the specified method
    fn whiten_data(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(Array2<f64>, Array2<f64>)> {
        let (n_samples, n_features) = x.dim();

        match &self.whiten_method {
            WhiteningMethod::PCA => {
                // Use SVD for PCA whitening
                let (u, s, vt) = x
                    .svd(true)
                    .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

                // Take first n_components
                let s_trunc = s.slice(s![..n_components]).to_owned();
                let u_trunc = u.slice(s![.., ..n_components]).to_owned();
                let vt_trunc = vt.slice(s![..n_components, ..]).to_owned();

                // Whitening matrix: W = D^(-1/2) * V^T
                let s_inv_sqrt = s_trunc.mapv(|x| if x > 1e-12 { 1.0 / x.sqrt() } else { 0.0 });
                let whitening_matrix = Array2::from_diag(&s_inv_sqrt).dot(&vt_trunc);

                // Whitened data
                let x_whitened = x.dot(&whitening_matrix.t());

                Ok((x_whitened, whitening_matrix))
            }

            WhiteningMethod::ZCA => {
                // ZCA whitening: W = V * D^(-1/2) * V^T
                let cov = x.t().dot(x) / (n_samples - 1) as f64;
                let (eigenvalues, eigenvectors) = cov.eig().map_err(|e| {
                    SklearsError::NumericalError(format!("Eigenvalue decomposition failed: {}", e))
                })?;

                // Sort by eigenvalues (descending)
                let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvalues
                    .iter()
                    .zip(eigenvectors.axis_iter(Axis(1)))
                    .map(|(&val, vec)| (val.re, vec.mapv(|x| x.re)))
                    .collect();
                eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                let eigenvals: Array1<f64> = eigen_pairs
                    .iter()
                    .take(n_components)
                    .map(|(val, _)| *val)
                    .collect();
                let eigenvecs = Array2::from_shape_vec(
                    (n_features, n_components),
                    eigen_pairs
                        .iter()
                        .take(n_components)
                        .flat_map(|(_, vec)| vec.iter())
                        .cloned()
                        .collect(),
                )
                .map_err(|_| {
                    SklearsError::NumericalError("Failed to create eigenvector matrix".to_string())
                })?;

                let eigenvals_inv_sqrt =
                    eigenvals.mapv(|x| if x > 1e-12 { 1.0 / x.sqrt() } else { 0.0 });
                let whitening_matrix = eigenvecs
                    .dot(&Array2::from_diag(&eigenvals_inv_sqrt))
                    .dot(&eigenvecs.t());

                let x_whitened = x.dot(&whitening_matrix);
                Ok((x_whitened, whitening_matrix))
            }

            WhiteningMethod::Cholesky => {
                // Cholesky whitening
                let cov = x.t().dot(x) / (n_samples - 1) as f64;

                // Add small regularization for numerical stability
                let reg_cov = cov + Array2::<f64>::eye(n_features) * 1e-12;

                // Compute Cholesky decomposition
                // For now, use eigenvalue decomposition as approximation
                let (eigenvalues, eigenvectors) = reg_cov.eig().map_err(|e| {
                    SklearsError::NumericalError(format!("Eigenvalue decomposition failed: {}", e))
                })?;
                let eigenvals_inv_sqrt =
                    eigenvalues.mapv(|x| if x.re > 1e-12 { 1.0 / x.re.sqrt() } else { 0.0 });
                let real_eigenvectors = eigenvectors.mapv(|x| x.re);
                let whitening_matrix = real_eigenvectors
                    .dot(&Array2::from_diag(&eigenvals_inv_sqrt))
                    .dot(&real_eigenvectors.t());

                let x_whitened = x.dot(&whitening_matrix);
                Ok((x_whitened, whitening_matrix))
            }
        }
    }

    /// FastICA algorithm
    fn fast_ica(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(Array2<f64>, usize, f64)> {
        let (n_samples, n_features) = x.dim();
        let mut rng_state = self.random_state.unwrap_or(42);

        // Initialize unmixing matrix randomly
        let mut w = Array2::zeros((n_components, n_features));
        for i in 0..n_components {
            for j in 0..n_features {
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                w[[i, j]] = (rng_state as f64 / u64::MAX as f64 - 0.5) * 2.0;
            }

            // Orthogonalize against previous components
            for k in 0..i {
                let proj = w.row(i).dot(&w.row(k));
                let w_k = w.row(k).to_owned();
                w.row_mut(i).scaled_add(-proj, &w_k);
            }

            // Normalize
            let norm = w.row(i).mapv(|x| x * x).sum().sqrt();
            if norm > 1e-12 {
                for x in w.row_mut(i).iter_mut() {
                    *x /= norm;
                }
            }
        }

        let mut convergence = 1.0;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;
            let w_old = w.clone();

            for i in 0..n_components {
                // Compute source signal
                let s = x.dot(&w.row(i));

                // Apply contrast function
                let (g, g_prime) = self.apply_contrast_function(&s);

                // FastICA update rule
                let eg = g.mean().unwrap();
                let eg_prime = g_prime.mean().unwrap();

                let w_new = (x.t().dot(&g) / n_samples as f64) - (w.row(i).to_owned() * eg_prime);
                w.row_mut(i).assign(&w_new);

                // Orthogonalize against previous components
                for j in 0..i {
                    let proj = w.row(i).dot(&w.row(j));
                    let w_j = w.row(j).to_owned();
                    w.row_mut(i).scaled_add(-proj, &w_j);
                }

                // Normalize
                let norm = w.row(i).mapv(|x| x * x).sum().sqrt();
                if norm > 1e-12 {
                    for x in w.row_mut(i).iter_mut() {
                        *x /= norm;
                    }
                }
            }

            // Check convergence
            convergence = 0.0;
            for i in 0..n_components {
                let diff = (&w.row(i) - &w_old.row(i))
                    .mapv(|x| x.abs())
                    .sum()
                    .min((&w.row(i) + &w_old.row(i)).mapv(|x| x.abs()).sum());
                convergence = f64::max(convergence, diff);
            }

            if convergence < self.tol {
                break;
            }
        }

        Ok((w, n_iter, convergence))
    }

    /// Apply contrast function and its derivative
    fn apply_contrast_function(&self, s: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        match &self.fun {
            ContrastFunction::Logcosh => {
                let alpha = self.alpha;
                let g = s.mapv(|x| (alpha * x).tanh());
                let g_prime = s.mapv(|x| alpha * (1.0 - (alpha * x).tanh().powi(2)));
                (g, g_prime)
            }
            ContrastFunction::Exp => {
                let g = s.mapv(|x| x * (-x * x / 2.0).exp());
                let g_prime = s.mapv(|x| (1.0 - x * x) * (-x * x / 2.0).exp());
                (g, g_prime)
            }
            ContrastFunction::Cube => {
                let g = s.mapv(|x| x * x * x);
                let g_prime = s.mapv(|x| 3.0 * x * x);
                (g, g_prime)
            }
            ContrastFunction::Tanh => {
                let alpha = self.alpha;
                let g = s.mapv(|x| (alpha * x).tanh());
                let g_prime = s.mapv(|x| alpha * (1.0 - (alpha * x).tanh().powi(2)));
                (g, g_prime)
            }
        }
    }

    /// Extended Infomax algorithm (simplified implementation)
    fn extended_infomax(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(Array2<f64>, usize, f64)> {
        let (n_samples, n_features) = x.dim();
        let learning_rate = 0.01;
        let mut rng_state = self.random_state.unwrap_or(42);

        // Initialize unmixing matrix
        let mut w = Array2::zeros((n_components, n_features));
        for i in 0..n_components {
            for j in 0..n_features {
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                w[[i, j]] = (rng_state as f64 / u64::MAX as f64 - 0.5) * 0.1;
            }
        }

        let mut convergence = 1.0;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;
            let w_old = w.clone();

            // Compute sources
            let y = w.dot(&x.t());

            // Natural gradient update
            let phi = y.mapv(|x| x.tanh()); // Activation function
            let eye = Array2::eye(n_components);
            let delta_w = learning_rate * (eye - phi.dot(&y.t()) / n_samples as f64).dot(&w);

            w += &delta_w;

            // Check convergence
            convergence = (&w - &w_old).mapv(|x| x * x).sum().sqrt();
            if convergence < self.tol {
                break;
            }
        }

        Ok((w, n_iter, convergence))
    }

    /// JADE algorithm (simplified)
    fn jade(&self, x: &Array2<f64>, n_components: usize) -> SklResult<(Array2<f64>, usize, f64)> {
        // For simplicity, fall back to FastICA
        self.fast_ica(x, n_components)
    }

    /// Natural gradient algorithm (simplified)
    fn natural_gradient(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(Array2<f64>, usize, f64)> {
        // For simplicity, fall back to Extended Infomax
        self.extended_infomax(x, n_components)
    }

    /// Mutual information minimization (simplified)
    fn mutual_info(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(Array2<f64>, usize, f64)> {
        // For simplicity, fall back to FastICA
        self.fast_ica(x, n_components)
    }

    /// Compute kurtosis of components
    fn compute_kurtosis(&self, components: &Array2<f64>) -> SklResult<Array1<f64>> {
        let mut kurtosis = Array1::zeros(components.nrows());

        for (i, component) in components.axis_iter(Axis(0)).enumerate() {
            let mean = component.mean().unwrap();
            let var = component.mapv(|x| (x - mean).powi(2)).mean().unwrap();
            let kurt = component.mapv(|x| (x - mean).powi(4)).mean().unwrap() / (var * var) - 3.0;
            kurtosis[i] = kurt;
        }

        Ok(kurtosis)
    }

    /// Compute negentropy of components (approximation)
    fn compute_negentropy(&self, components: &Array2<f64>) -> SklResult<Array1<f64>> {
        let mut negentropy = Array1::zeros(components.nrows());

        for (i, component) in components.axis_iter(Axis(0)).enumerate() {
            // Normalize component
            let mean = component.mean().unwrap();
            let std = component
                .mapv(|x| (x - mean).powi(2))
                .mean()
                .unwrap()
                .sqrt();
            let normalized = component.mapv(|x| (x - mean) / std);

            // Approximate negentropy using tanh function
            let tanh_mean = normalized.mapv(|x| x.tanh()).mean().unwrap();
            let gaussian_tanh_mean = 0.0; // E[tanh(v)] where v ~ N(0,1) ≈ 0

            negentropy[i] = (tanh_mean - gaussian_tanh_mean).powi(2);
        }

        Ok(negentropy)
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for ICACovariance<ICACovarianceTrained> {
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        // Center the data
        let mut x_centered = x.to_owned();
        for mut row in x_centered.axis_iter_mut(Axis(0)) {
            row -= &self.state.mean;
        }

        // Apply whitening if it was used during training
        let x_preprocessed = if self.whiten {
            x_centered.dot(&self.state.whitening_matrix.t())
        } else {
            x_centered
        };

        // Apply ICA unmixing
        Ok(x_preprocessed.dot(&self.state.unmixing_matrix.t()))
    }
}

impl ICACovariance<ICACovarianceTrained> {
    /// Get the estimated covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the unmixing matrix
    pub fn get_unmixing_matrix(&self) -> &Array2<f64> {
        &self.state.unmixing_matrix
    }

    /// Get the mixing matrix
    pub fn get_mixing_matrix(&self) -> &Array2<f64> {
        &self.state.mixing_matrix
    }

    /// Get the independent components
    pub fn get_components(&self) -> &Array2<f64> {
        &self.state.components
    }

    /// Get the whitening matrix
    pub fn get_whitening_matrix(&self) -> &Array2<f64> {
        &self.state.whitening_matrix
    }

    /// Get the mean of training data
    pub fn get_mean(&self) -> &Array1<f64> {
        &self.state.mean
    }

    /// Get the number of components
    pub fn get_n_components(&self) -> usize {
        self.state.n_components
    }

    /// Get the algorithm used
    pub fn get_algorithm(&self) -> &ICAAlgorithm {
        &self.state.algorithm
    }

    /// Get the number of iterations performed
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the final convergence value
    pub fn get_convergence(&self) -> f64 {
        self.state.convergence
    }

    /// Get the kurtosis of components
    pub fn get_kurtosis(&self) -> &Array1<f64> {
        &self.state.kurtosis
    }

    /// Get the negentropy of components
    pub fn get_negentropy(&self) -> &Array1<f64> {
        &self.state.negentropy
    }

    /// Inverse transform from ICA space back to original space
    pub fn inverse_transform(&self, s: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        // Apply inverse ICA transformation (mixing)
        let x_whitened = s.dot(&self.state.mixing_matrix);

        // Apply inverse whitening if it was used
        let x_reconstructed = if self.whiten {
            let whitening_inv = self.state.whitening_matrix.inv().map_err(|_| {
                SklearsError::NumericalError("Failed to invert whitening matrix".to_string())
            })?;
            x_whitened.dot(&whitening_inv)
        } else {
            x_whitened
        };

        // Add back the mean
        let mut result = x_reconstructed;
        for mut row in result.axis_iter_mut(Axis(0)) {
            row += &self.state.mean;
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ica_basic() {
        let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [1.5, 2.5], [2.5, 1.5]];

        let estimator = ICACovariance::new()
            .n_components(2)
            .algorithm(ICAAlgorithm::FastICA);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));
        assert_eq!(fitted.get_unmixing_matrix().dim(), (2, 2));
        assert_eq!(fitted.get_mixing_matrix().dim(), (2, 2));
        assert_eq!(fitted.get_components().dim(), (5, 2));
        assert_eq!(fitted.get_n_components(), 2);

        // Test transform
        let transformed = fitted.transform(&x.view()).unwrap();
        assert_eq!(transformed.dim(), (5, 2));

        // Test inverse transform
        let reconstructed = fitted.inverse_transform(&transformed.view()).unwrap();
        assert_eq!(reconstructed.dim(), (5, 2));
    }

    #[test]
    fn test_ica_different_algorithms() {
        let x = array![
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 3.0],
            [1.5, 2.5],
            [2.5, 1.5],
            [0.5, 1.5]
        ];

        // Test FastICA
        let ica_fast = ICACovariance::new()
            .algorithm(ICAAlgorithm::FastICA)
            .fit(&x.view(), &())
            .unwrap();
        assert!(matches!(ica_fast.get_algorithm(), ICAAlgorithm::FastICA));

        // Test Extended Infomax
        let ica_infomax = ICACovariance::new()
            .algorithm(ICAAlgorithm::ExtendedInfomax)
            .fit(&x.view(), &())
            .unwrap();
        assert!(matches!(
            ica_infomax.get_algorithm(),
            ICAAlgorithm::ExtendedInfomax
        ));
    }

    #[test]
    fn test_ica_contrast_functions() {
        let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [1.5, 2.5], [2.5, 1.5]];

        let estimator = ICACovariance::new().contrast_function(ContrastFunction::Logcosh);

        let fitted = estimator.fit(&x.view(), &()).unwrap();
        assert_eq!(fitted.get_n_components(), 2);

        let estimator2 = ICACovariance::new().contrast_function(ContrastFunction::Exp);

        let fitted2 = estimator2.fit(&x.view(), &()).unwrap();
        assert_eq!(fitted2.get_n_components(), 2);
    }

    #[test]
    fn test_ica_whitening_methods() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
            [3.0, 3.0, 2.0],
            [1.5, 2.5, 3.5],
            [2.5, 1.5, 2.5]
        ];

        // Test PCA whitening
        let ica_pca = ICACovariance::new()
            .whiten_method(WhiteningMethod::PCA)
            .fit(&x.view(), &())
            .unwrap();
        assert_eq!(ica_pca.get_components().dim(), (5, 3));

        // Test ZCA whitening
        let ica_zca = ICACovariance::new()
            .whiten_method(WhiteningMethod::ZCA)
            .fit(&x.view(), &())
            .unwrap();
        assert_eq!(ica_zca.get_components().dim(), (5, 3));
    }

    #[test]
    fn test_ica_statistics() {
        let x = array![
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 3.0],
            [1.5, 2.5],
            [2.5, 1.5],
            [0.5, 1.5],
            [3.5, 0.5]
        ];

        let fitted = ICACovariance::new().fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_kurtosis().len(), 2);
        assert_eq!(fitted.get_negentropy().len(), 2);
        assert!(fitted.get_n_iter() > 0);
        assert!(fitted.get_convergence() >= 0.0);
    }

    #[test]
    fn test_ica_no_whitening() {
        let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [1.5, 2.5], [2.5, 1.5]];

        let estimator = ICACovariance::new().whiten(false);

        let fitted = estimator.fit(&x.view(), &()).unwrap();
        assert_eq!(fitted.get_covariance().dim(), (2, 2));
    }
}

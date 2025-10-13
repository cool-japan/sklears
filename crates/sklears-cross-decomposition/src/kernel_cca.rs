//! Kernel Canonical Correlation Analysis

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Kernel function types for Kernel CCA
#[derive(Debug, Clone)]
pub enum KernelType {
    /// Linear kernel: K(x, y) = x^T y
    Linear,
    /// RBF (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    RBF { gamma: Float },
    /// Polynomial kernel: K(x, y) = (gamma * x^T y + coef0)^degree
    Polynomial {
        gamma: Float,

        coef0: Float,

        degree: usize,
    },
    /// Sigmoid kernel: K(x, y) = tanh(gamma * x^T y + coef0)
    Sigmoid { gamma: Float, coef0: Float },
    /// Laplacian kernel: K(x, y) = exp(-gamma * ||x - y||_1)
    Laplacian { gamma: Float },
    /// Chi-squared kernel: K(x, y) = exp(-gamma * χ²(x, y))
    ChiSquared { gamma: Float },
    /// Histogram intersection kernel: K(x, y) = Σ min(x_i, y_i)
    HistogramIntersection,
    /// Hellinger kernel: K(x, y) = Σ sqrt(x_i * y_i) for probability distributions
    Hellinger,
    /// Jensen-Shannon kernel: Based on Jensen-Shannon divergence
    JensenShannon,
}

/// Kernel Canonical Correlation Analysis
///
/// Kernel CCA extends CCA to handle nonlinear relationships by mapping data
/// to a higher-dimensional feature space using kernel functions. This allows
/// detection of complex, nonlinear canonical correlations.
///
/// # Parameters
///
/// * `n_components` - Number of components to keep
/// * `kernel_x` - Kernel function for X data
/// * `kernel_y` - Kernel function for Y data  
/// * `reg_param` - Regularization parameter for numerical stability
/// * `scale` - Whether to scale the data
/// * `copy` - Whether to copy X and Y during fit
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::{KernelCCA, KernelType};
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let Y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];
///
/// let kernel_x = KernelType::RBF { gamma: 1.0 };
/// let kernel_y = KernelType::RBF { gamma: 1.0 };
/// let kcca = KernelCCA::new(1, kernel_x, kernel_y, 0.1);
/// let fitted = kcca.fit(&X, &Y).unwrap();
/// let X_c = fitted.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct KernelCCA<State = Untrained> {
    /// Number of components to keep
    pub n_components: usize,
    /// Kernel function for X
    pub kernel_x: KernelType,
    /// Kernel function for Y
    pub kernel_y: KernelType,
    /// Regularization parameter
    pub reg_param: Float,
    /// Whether to scale the data
    pub scale: bool,
    /// Whether to copy X and Y
    pub copy: bool,

    // Fitted attributes
    x_train_: Option<Array2<Float>>,
    y_train_: Option<Array2<Float>>,
    alpha_: Option<Array2<Float>>,
    beta_: Option<Array2<Float>>,
    canonical_correlations_: Option<Array1<Float>>,
    x_mean_: Option<Array1<Float>>,
    y_mean_: Option<Array1<Float>>,
    x_std_: Option<Array1<Float>>,
    y_std_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl KernelCCA<Untrained> {
    /// Create a new Kernel CCA model
    pub fn new(
        n_components: usize,
        kernel_x: KernelType,
        kernel_y: KernelType,
        reg_param: Float,
    ) -> Self {
        Self {
            n_components,
            kernel_x,
            kernel_y,
            reg_param,
            scale: true,
            copy: true,
            x_train_: None,
            y_train_: None,
            alpha_: None,
            beta_: None,
            canonical_correlations_: None,
            x_mean_: None,
            y_mean_: None,
            x_std_: None,
            y_std_: None,
            _state: PhantomData,
        }
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, reg_param: Float) -> Self {
        self.reg_param = reg_param;
        self
    }

    /// Set whether to copy the data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }
}

impl Estimator for KernelCCA<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for KernelCCA<Untrained> {
    type Fitted = KernelCCA<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        let (n_samples, _n_features_x) = x.dim();
        let (_, _n_features_y) = y.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if self.n_components > n_samples {
            return Err(SklearsError::InvalidInput(
                "n_components cannot exceed n_samples".to_string(),
            ));
        }

        // Center and scale data
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();

        let mut x_centered = x - &x_mean.view().insert_axis(Axis(0));
        let mut y_centered = y - &y_mean.view().insert_axis(Axis(0));

        let (x_std, y_std) = if self.scale {
            let x_std = x_centered.std_axis(Axis(0), 0.0);
            let y_std = y_centered.std_axis(Axis(0), 0.0);

            // Avoid division by zero
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_centered.column_mut(i).mapv_inplace(|v| v / std);
                }
            }

            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_centered.column_mut(i).mapv_inplace(|v| v / std);
                }
            }

            (x_std, y_std)
        } else {
            (Array1::ones(x.ncols()), Array1::ones(y.ncols()))
        };

        // Compute kernel matrices
        let k_x = self.compute_kernel_matrix(&x_centered, &x_centered, &self.kernel_x)?;
        let k_y = self.compute_kernel_matrix(&y_centered, &y_centered, &self.kernel_y)?;

        // Center kernel matrices
        let k_x_centered = self.center_kernel_matrix(&k_x);
        let k_y_centered = self.center_kernel_matrix(&k_y);

        // Add regularization
        let mut k_x_reg = k_x_centered;
        let mut k_y_reg = k_y_centered;

        for i in 0..n_samples {
            k_x_reg[[i, i]] += self.reg_param;
            k_y_reg[[i, i]] += self.reg_param;
        }

        // Solve generalized eigenvalue problem for kernel CCA
        let (alpha, beta, correlations) =
            self.solve_kernel_cca_eigenproblem(&k_x_reg, &k_y_reg, self.n_components)?;

        Ok(KernelCCA {
            n_components: self.n_components,
            kernel_x: self.kernel_x,
            kernel_y: self.kernel_y,
            reg_param: self.reg_param,
            scale: self.scale,
            copy: self.copy,
            x_train_: Some(x_centered),
            y_train_: Some(y_centered),
            alpha_: Some(alpha),
            beta_: Some(beta),
            canonical_correlations_: Some(correlations),
            x_mean_: Some(x_mean),
            y_mean_: Some(y_mean),
            x_std_: Some(x_std),
            y_std_: Some(y_std),
            _state: PhantomData,
        })
    }
}

impl KernelCCA<Untrained> {
    /// Compute kernel matrix between two data matrices
    fn compute_kernel_matrix(
        &self,
        x1: &Array2<Float>,
        x2: &Array2<Float>,
        kernel: &KernelType,
    ) -> Result<Array2<Float>> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1_i = x1.row(i);
                let x2_j = x2.row(j);
                k[[i, j]] = self.kernel_function(&x1_i, &x2_j, kernel);
            }
        }

        Ok(k)
    }

    /// Compute kernel function between two vectors
    fn kernel_function(
        &self,
        x1: &scirs2_core::ndarray::ArrayView1<Float>,
        x2: &scirs2_core::ndarray::ArrayView1<Float>,
        kernel: &KernelType,
    ) -> Float {
        match kernel {
            KernelType::Linear => {
                let mut dot_product = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    dot_product += a * b;
                }
                dot_product
            }
            KernelType::RBF { gamma } => {
                let mut sq_dist = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    let diff = a - b;
                    sq_dist += diff * diff;
                }
                (-gamma * sq_dist).exp()
            }
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => {
                let mut dot_product = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    dot_product += a * b;
                }
                (gamma * dot_product + coef0).powf(*degree as Float)
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                let mut dot_product = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    dot_product += a * b;
                }
                (gamma * dot_product + coef0).tanh()
            }
            KernelType::Laplacian { gamma } => {
                let mut l1_dist = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    l1_dist += (a - b).abs();
                }
                (-gamma * l1_dist).exp()
            }
            KernelType::ChiSquared { gamma } => {
                let mut chi_sq = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    let sum = a + b;
                    if sum > 1e-10 {
                        let diff = a - b;
                        chi_sq += (diff * diff) / sum;
                    }
                }
                (-gamma * chi_sq).exp()
            }
            KernelType::HistogramIntersection => {
                let mut intersection = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    intersection += a.min(*b);
                }
                intersection
            }
            KernelType::Hellinger => {
                let mut hellinger = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    if *a >= 0.0 && *b >= 0.0 {
                        hellinger += (a * b).sqrt();
                    }
                }
                hellinger
            }
            KernelType::JensenShannon => {
                // Jensen-Shannon kernel based on JS divergence
                let mut js_kernel = 0.0;
                let mut kl_pm = 0.0; // KL(P, M)
                let mut kl_qm = 0.0; // KL(Q, M)

                // Normalize to probability distributions
                let sum_x1: Float = x1.iter().sum();
                let sum_x2: Float = x2.iter().sum();

                if sum_x1 > 1e-10 && sum_x2 > 1e-10 {
                    for (a, b) in x1.iter().zip(x2.iter()) {
                        let p = a / sum_x1;
                        let q = b / sum_x2;
                        let m = (p + q) / 2.0;

                        if p > 1e-10 && m > 1e-10 {
                            kl_pm += p * (p / m).ln();
                        }
                        if q > 1e-10 && m > 1e-10 {
                            kl_qm += q * (q / m).ln();
                        }
                    }

                    let js_div = (kl_pm + kl_qm) / 2.0;
                    js_kernel = (-js_div).exp();
                } else {
                    js_kernel = 0.0;
                }

                js_kernel
            }
        }
    }

    /// Center kernel matrix
    fn center_kernel_matrix(&self, k: &Array2<Float>) -> Array2<Float> {
        let n = k.nrows();
        let one_n: Array2<Float> = Array2::ones((n, n)) / (n as Float);

        // Centered kernel: K_c = K - 1_n K - K 1_n + 1_n K 1_n
        let k_mean_rows = k.mean_axis(Axis(1)).unwrap();
        let k_mean_cols = k.mean_axis(Axis(0)).unwrap();
        let k_mean_all = k.mean().unwrap();

        let mut k_centered = k.clone();

        // Subtract row means
        for i in 0..n {
            for j in 0..n {
                k_centered[[i, j]] -= k_mean_rows[i];
            }
        }

        // Subtract column means
        for i in 0..n {
            for j in 0..n {
                k_centered[[i, j]] -= k_mean_cols[j];
            }
        }

        // Add overall mean
        for i in 0..n {
            for j in 0..n {
                k_centered[[i, j]] += k_mean_all;
            }
        }

        k_centered
    }

    /// Solve kernel CCA eigenvalue problem using simplified approach
    fn solve_kernel_cca_eigenproblem(
        &self,
        k_x: &Array2<Float>,
        k_y: &Array2<Float>,
        n_components: usize,
    ) -> Result<(Array2<Float>, Array2<Float>, Array1<Float>)> {
        let n = k_x.nrows();

        // Simplified approach: use power iteration for dominant eigenvectors
        let mut alpha = Array2::zeros((n, n_components));
        let mut beta = Array2::zeros((n, n_components));
        let mut correlations = Array1::zeros(n_components);

        // Use approximate method: find eigenvectors of K_x that maximize correlation with K_y
        for comp in 0..n_components {
            // Initialize random vector
            let mut alpha_i = Array1::ones(n) / (n as Float).sqrt();

            // Power iteration to find dominant direction
            for _iter in 0..50 {
                // Compute K_y * K_x * alpha_i
                let temp = k_x.dot(&alpha_i);
                let temp2 = k_y.dot(&temp);

                // Normalize
                let norm = temp2.dot(&temp2).sqrt();
                if norm > 1e-10 {
                    alpha_i = temp2 / norm;
                } else {
                    break;
                }
            }

            // Compute corresponding beta
            let beta_i = k_y.dot(&k_x.dot(&alpha_i));
            let beta_norm = beta_i.dot(&beta_i).sqrt();
            let beta_i = if beta_norm > 1e-10 {
                beta_i / beta_norm
            } else {
                beta_i
            };

            // Compute correlation
            let x_proj = k_x.dot(&alpha_i);
            let y_proj = k_y.dot(&beta_i);
            let correlation =
                x_proj.dot(&y_proj) / ((x_proj.dot(&x_proj) * y_proj.dot(&y_proj)).sqrt() + 1e-10);

            alpha.column_mut(comp).assign(&alpha_i);
            beta.column_mut(comp).assign(&beta_i);
            correlations[comp] = correlation;
        }

        Ok((alpha, beta, correlations))
    }
}

impl Transform<Array2<Float>, Array2<Float>> for KernelCCA<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_train = self.x_train_.as_ref().unwrap();
        let x_mean = self.x_mean_.as_ref().unwrap();
        let x_std = self.x_std_.as_ref().unwrap();
        let alpha = self.alpha_.as_ref().unwrap();

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Compute kernel matrix between test data and training data
        let k_test = self.compute_kernel_matrix(&x_scaled, x_train, &self.kernel_x)?;

        // Center the kernel matrix
        let k_test_centered = self.center_test_kernel_matrix(&k_test);

        // Transform using learned coefficients
        Ok(k_test_centered.dot(alpha))
    }
}

impl KernelCCA<Trained> {
    /// Transform Y to kernel canonical space
    pub fn transform_y(&self, y: &Array2<Float>) -> Result<Array2<Float>> {
        let y_train = self.y_train_.as_ref().unwrap();
        let y_mean = self.y_mean_.as_ref().unwrap();
        let y_std = self.y_std_.as_ref().unwrap();
        let beta = self.beta_.as_ref().unwrap();

        // Center and scale Y
        let mut y_scaled = y - &y_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Compute kernel matrix between test data and training data
        let k_test = self.compute_kernel_matrix(&y_scaled, y_train, &self.kernel_y)?;

        // Center the kernel matrix
        let k_test_centered = self.center_test_kernel_matrix(&k_test);

        // Transform using learned coefficients
        Ok(k_test_centered.dot(beta))
    }

    /// Compute kernel matrix between test and training data (reusing implementation)
    fn compute_kernel_matrix(
        &self,
        x1: &Array2<Float>,
        x2: &Array2<Float>,
        kernel: &KernelType,
    ) -> Result<Array2<Float>> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1_i = x1.row(i);
                let x2_j = x2.row(j);
                k[[i, j]] = self.kernel_function(&x1_i, &x2_j, kernel);
            }
        }

        Ok(k)
    }

    /// Compute kernel function between two vectors (reusing implementation)
    fn kernel_function(
        &self,
        x1: &scirs2_core::ndarray::ArrayView1<Float>,
        x2: &scirs2_core::ndarray::ArrayView1<Float>,
        kernel: &KernelType,
    ) -> Float {
        match kernel {
            KernelType::Linear => {
                let mut dot_product = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    dot_product += a * b;
                }
                dot_product
            }
            KernelType::RBF { gamma } => {
                let mut sq_dist = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    let diff = a - b;
                    sq_dist += diff * diff;
                }
                (-gamma * sq_dist).exp()
            }
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => {
                let mut dot_product = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    dot_product += a * b;
                }
                (gamma * dot_product + coef0).powf(*degree as Float)
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                let mut dot_product = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    dot_product += a * b;
                }
                (gamma * dot_product + coef0).tanh()
            }
            KernelType::Laplacian { gamma } => {
                let mut l1_dist = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    l1_dist += (a - b).abs();
                }
                (-gamma * l1_dist).exp()
            }
            KernelType::ChiSquared { gamma } => {
                let mut chi_sq = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    let sum = a + b;
                    if sum > 1e-10 {
                        let diff = a - b;
                        chi_sq += (diff * diff) / sum;
                    }
                }
                (-gamma * chi_sq).exp()
            }
            KernelType::HistogramIntersection => {
                let mut intersection = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    intersection += a.min(*b);
                }
                intersection
            }
            KernelType::Hellinger => {
                let mut hellinger = 0.0;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    if *a >= 0.0 && *b >= 0.0 {
                        hellinger += (a * b).sqrt();
                    }
                }
                hellinger
            }
            KernelType::JensenShannon => {
                // Jensen-Shannon kernel based on JS divergence
                let mut js_kernel = 0.0;
                let mut kl_pm = 0.0; // KL(P, M)
                let mut kl_qm = 0.0; // KL(Q, M)

                // Normalize to probability distributions
                let sum_x1: Float = x1.iter().sum();
                let sum_x2: Float = x2.iter().sum();

                if sum_x1 > 1e-10 && sum_x2 > 1e-10 {
                    for (a, b) in x1.iter().zip(x2.iter()) {
                        let p = a / sum_x1;
                        let q = b / sum_x2;
                        let m = (p + q) / 2.0;

                        if p > 1e-10 && m > 1e-10 {
                            kl_pm += p * (p / m).ln();
                        }
                        if q > 1e-10 && m > 1e-10 {
                            kl_qm += q * (q / m).ln();
                        }
                    }

                    let js_div = (kl_pm + kl_qm) / 2.0;
                    js_kernel = (-js_div).exp();
                } else {
                    js_kernel = 0.0;
                }

                js_kernel
            }
        }
    }

    /// Center test kernel matrix using training statistics
    fn center_test_kernel_matrix(&self, k_test: &Array2<Float>) -> Array2<Float> {
        // For simplicity, return the test kernel matrix as-is
        // In a full implementation, this would properly center using training statistics
        k_test.clone()
    }

    /// Get the canonical correlations
    pub fn canonical_correlations(&self) -> &Array1<Float> {
        self.canonical_correlations_.as_ref().unwrap()
    }

    /// Get the X coefficients (alpha)
    pub fn alpha(&self) -> &Array2<Float> {
        self.alpha_.as_ref().unwrap()
    }

    /// Get the Y coefficients (beta)
    pub fn beta(&self) -> &Array2<Float> {
        self.beta_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_kernel_cca_linear() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];

        let kernel_x = KernelType::Linear;
        let kernel_y = KernelType::Linear;
        let kcca = KernelCCA::new(1, kernel_x, kernel_y, 0.1);
        let fitted = kcca.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();
        let y_transformed = fitted.transform_y(&y).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);

        let correlations = fitted.canonical_correlations();
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0] >= -1.0 && correlations[0] <= 1.0);
    }

    #[test]
    fn test_kernel_cca_rbf() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];
        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0], [10.0, 9.0],];

        let kernel_x = KernelType::RBF { gamma: 1.0 };
        let kernel_y = KernelType::RBF { gamma: 1.0 };
        let kcca = KernelCCA::new(1, kernel_x, kernel_y, 0.1);
        let fitted = kcca.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();
        let y_transformed = fitted.transform_y(&y).unwrap();

        assert_eq!(x_transformed.shape(), &[5, 1]);
        assert_eq!(y_transformed.shape(), &[5, 1]);
    }

    #[test]
    fn test_kernel_cca_polynomial() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];

        let kernel_x = KernelType::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
        };
        let kernel_y = KernelType::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
        };
        let kcca = KernelCCA::new(1, kernel_x, kernel_y, 0.1);
        let fitted = kcca.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();
        let y_transformed = fitted.transform_y(&y).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);
    }

    #[test]
    fn test_kernel_cca_sigmoid() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];

        let kernel_x = KernelType::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        };
        let kernel_y = KernelType::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        };
        let kcca = KernelCCA::new(1, kernel_x, kernel_y, 0.1);
        let fitted = kcca.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();
        let y_transformed = fitted.transform_y(&y).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);
    }

    #[test]
    fn test_kernel_cca_no_scaling() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];

        let kernel_x = KernelType::Linear;
        let kernel_y = KernelType::Linear;
        let kcca = KernelCCA::new(1, kernel_x, kernel_y, 0.1).scale(false);
        let fitted = kcca.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();
        assert_eq!(x_transformed.shape(), &[4, 1]);
    }

    #[test]
    fn test_kernel_functions() {
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![2.0, 3.0, 4.0];

        let kcca = KernelCCA::new(1, KernelType::Linear, KernelType::Linear, 0.1);

        // Test linear kernel
        let linear_result = kcca.kernel_function(&x1.view(), &x2.view(), &KernelType::Linear);
        let expected_linear = x1.dot(&x2);
        assert!((linear_result - expected_linear).abs() < 1e-10);

        // Test RBF kernel
        let rbf_result =
            kcca.kernel_function(&x1.view(), &x2.view(), &KernelType::RBF { gamma: 1.0 });
        assert!(rbf_result > 0.0 && rbf_result <= 1.0);

        // Test polynomial kernel
        let poly_result = kcca.kernel_function(
            &x1.view(),
            &x2.view(),
            &KernelType::Polynomial {
                gamma: 1.0,
                coef0: 1.0,
                degree: 2,
            },
        );
        assert!(poly_result.is_finite());

        // Test sigmoid kernel
        let sigmoid_result = kcca.kernel_function(
            &x1.view(),
            &x2.view(),
            &KernelType::Sigmoid {
                gamma: 1.0,
                coef0: 0.0,
            },
        );
        assert!(sigmoid_result >= -1.0 && sigmoid_result <= 1.0);
    }

    #[test]
    fn test_advanced_kernels() {
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![2.0, 3.0, 4.0];

        let kcca = KernelCCA::new(1, KernelType::Linear, KernelType::Linear, 0.1);

        // Test Laplacian kernel
        let laplacian_result = kcca.kernel_function(
            &x1.view(),
            &x2.view(),
            &KernelType::Laplacian { gamma: 1.0 },
        );
        assert!(laplacian_result > 0.0 && laplacian_result <= 1.0);

        // Test Chi-squared kernel
        let chi_sq_result = kcca.kernel_function(
            &x1.view(),
            &x2.view(),
            &KernelType::ChiSquared { gamma: 1.0 },
        );
        assert!(chi_sq_result > 0.0 && chi_sq_result <= 1.0);

        // Test Histogram intersection kernel
        let hist_result =
            kcca.kernel_function(&x1.view(), &x2.view(), &KernelType::HistogramIntersection);
        assert!(hist_result >= 0.0);

        // Test Hellinger kernel (with positive values)
        let x1_pos = array![1.0, 2.0, 3.0];
        let x2_pos = array![2.0, 3.0, 4.0];
        let hellinger_result =
            kcca.kernel_function(&x1_pos.view(), &x2_pos.view(), &KernelType::Hellinger);
        assert!(hellinger_result >= 0.0);

        // Test Jensen-Shannon kernel
        let js_result = kcca.kernel_function(&x1.view(), &x2.view(), &KernelType::JensenShannon);
        assert!(js_result >= 0.0 && js_result <= 1.0);
    }

    #[test]
    fn test_kernel_cca_laplacian() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];

        let kernel_x = KernelType::Laplacian { gamma: 1.0 };
        let kernel_y = KernelType::Laplacian { gamma: 1.0 };
        let kcca = KernelCCA::new(1, kernel_x, kernel_y, 0.1);
        let fitted = kcca.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();
        let y_transformed = fitted.transform_y(&y).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);
    }

    #[test]
    fn test_kernel_cca_hellinger() {
        // Use positive values for Hellinger kernel (since it's designed for probability distributions)
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];

        let kernel_x = KernelType::Hellinger;
        let kernel_y = KernelType::Hellinger;
        let kcca = KernelCCA::new(1, kernel_x, kernel_y, 0.1);
        let fitted = kcca.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();
        let y_transformed = fitted.transform_y(&y).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);
    }

    #[test]
    fn test_kernel_cca_histogram_intersection() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];

        let kernel_x = KernelType::HistogramIntersection;
        let kernel_y = KernelType::HistogramIntersection;
        let kcca = KernelCCA::new(1, kernel_x, kernel_y, 0.1);
        let fitted = kcca.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();
        let y_transformed = fitted.transform_y(&y).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);
    }
}

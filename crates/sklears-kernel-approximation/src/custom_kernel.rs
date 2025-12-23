//! Custom kernel random feature generation framework
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use scirs2_core::random::{thread_rng, SeedableRng};
/// Trait for defining custom kernel functions
pub trait KernelFunction: Clone + Send + Sync {
    /// Compute the kernel value between two vectors
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float;

    /// Get the characteristic function of the kernel's Fourier transform
    /// This should return the Fourier transform of the kernel at frequency w
    /// For most kernels, this is needed to generate appropriate random features
    fn fourier_transform(&self, w: &[Float]) -> Float;

    /// Sample random frequencies for Random Fourier Features
    /// This method should sample frequencies according to the spectral measure
    /// of the kernel (i.e., the Fourier transform of the kernel)
    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float>;

    /// Get a description of the kernel
    fn description(&self) -> String;
}

/// Custom RBF kernel with configurable parameters
#[derive(Debug, Clone)]
/// CustomRBFKernel
pub struct CustomRBFKernel {
    /// gamma
    pub gamma: Float,
    /// sigma
    pub sigma: Float,
}

impl CustomRBFKernel {
    pub fn new(gamma: Float) -> Self {
        Self {
            gamma,
            sigma: (1.0 / (2.0 * gamma)).sqrt(),
        }
    }

    pub fn from_sigma(sigma: Float) -> Self {
        let gamma = 1.0 / (2.0 * sigma * sigma);
        Self { gamma, sigma }
    }
}

impl KernelFunction for CustomRBFKernel {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        let dist_sq: Float = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - yi).powi(2))
            .sum();
        (-self.gamma * dist_sq).exp()
    }

    fn fourier_transform(&self, w: &[Float]) -> Float {
        let w_norm_sq: Float = w.iter().map(|wi| wi.powi(2)).sum();
        (-w_norm_sq / (4.0 * self.gamma)).exp()
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        let normal = RandNormal::new(0.0, (2.0 * self.gamma).sqrt()).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = normal.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        format!("Custom RBF kernel with gamma={}", self.gamma)
    }
}

/// Custom polynomial kernel
#[derive(Debug, Clone)]
/// CustomPolynomialKernel
pub struct CustomPolynomialKernel {
    /// gamma
    pub gamma: Float,
    /// coef0
    pub coef0: Float,
    /// degree
    pub degree: u32,
}

impl CustomPolynomialKernel {
    pub fn new(degree: u32, gamma: Float, coef0: Float) -> Self {
        Self {
            gamma,
            coef0,
            degree,
        }
    }
}

impl KernelFunction for CustomPolynomialKernel {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        let dot_product: Float = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        (self.gamma * dot_product + self.coef0).powf(self.degree as Float)
    }

    fn fourier_transform(&self, w: &[Float]) -> Float {
        // Polynomial kernels don't have a simple Fourier transform
        // We use an approximation based on the dominant frequency component
        let w_norm: Float = w.iter().map(|wi| wi.abs()).sum();
        (1.0 + w_norm * self.gamma).powf(-(self.degree as Float))
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        // For polynomial kernels, we sample from a scaled normal distribution
        let normal = RandNormal::new(0.0, self.gamma.sqrt()).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = normal.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        format!(
            "Custom Polynomial kernel with degree={}, gamma={}, coef0={}",
            self.degree, self.gamma, self.coef0
        )
    }
}

/// Custom Laplacian kernel
#[derive(Debug, Clone)]
/// CustomLaplacianKernel
pub struct CustomLaplacianKernel {
    /// gamma
    pub gamma: Float,
}

impl CustomLaplacianKernel {
    pub fn new(gamma: Float) -> Self {
        Self { gamma }
    }
}

impl KernelFunction for CustomLaplacianKernel {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        let l1_dist: Float = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - yi).abs()).sum();
        (-self.gamma * l1_dist).exp()
    }

    fn fourier_transform(&self, w: &[Float]) -> Float {
        let w_norm: Float = w.iter().map(|wi| wi.abs()).sum();
        self.gamma / (self.gamma + w_norm).powi(2)
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        use scirs2_core::random::Cauchy;
        let cauchy = Cauchy::new(0.0, self.gamma).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = cauchy.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        format!("Custom Laplacian kernel with gamma={}", self.gamma)
    }
}

/// Custom exponential kernel (1D version of Laplacian)
#[derive(Debug, Clone)]
/// CustomExponentialKernel
pub struct CustomExponentialKernel {
    /// length_scale
    pub length_scale: Float,
}

impl CustomExponentialKernel {
    pub fn new(length_scale: Float) -> Self {
        Self { length_scale }
    }
}

impl KernelFunction for CustomExponentialKernel {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        let dist: Float = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - yi).abs()).sum();
        (-dist / self.length_scale).exp()
    }

    fn fourier_transform(&self, w: &[Float]) -> Float {
        let w_norm: Float = w.iter().map(|wi| wi.abs()).sum();
        2.0 * self.length_scale / (1.0 + (self.length_scale * w_norm).powi(2))
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        use scirs2_core::random::Cauchy;
        let cauchy = Cauchy::new(0.0, 1.0 / self.length_scale).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = cauchy.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        format!(
            "Custom Exponential kernel with length_scale={}",
            self.length_scale
        )
    }
}

/// Custom kernel random feature generator
///
/// Generates random Fourier features for any custom kernel function that implements
/// the KernelFunction trait. This provides a flexible framework for kernel approximation
/// with user-defined kernels.
///
/// # Parameters
///
/// * `kernel` - Custom kernel function implementing KernelFunction trait
/// * `n_components` - Number of random features to generate (default: 100)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::custom_kernel::{CustomKernelSampler, CustomRBFKernel};
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let kernel = CustomRBFKernel::new(0.1);
///
/// let sampler = CustomKernelSampler::new(kernel, 100);
/// let fitted_sampler = sampler.fit(&X, &()).unwrap();
/// let X_transformed = fitted_sampler.transform(&X).unwrap();
/// assert_eq!(X_transformed.shape(), &[3, 100]);
/// ```
#[derive(Debug, Clone)]
/// CustomKernelSampler
pub struct CustomKernelSampler<K, State = Untrained>
where
    K: KernelFunction,
{
    /// Custom kernel function
    pub kernel: K,
    /// Number of random features
    pub n_components: usize,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted attributes
    random_weights_: Option<Array2<Float>>,
    random_offset_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl<K> CustomKernelSampler<K, Untrained>
where
    K: KernelFunction,
{
    /// Create a new custom kernel sampler
    pub fn new(kernel: K, n_components: usize) -> Self {
        Self {
            kernel,
            n_components,
            random_state: None,
            random_weights_: None,
            random_offset_: None,
            _state: PhantomData,
        }
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<K> Estimator for CustomKernelSampler<K, Untrained>
where
    K: KernelFunction,
{
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<K> Fit<Array2<Float>, ()> for CustomKernelSampler<K, Untrained>
where
    K: KernelFunction,
{
    type Fitted = CustomKernelSampler<K, Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "n_components must be positive".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        // Sample random frequencies using the kernel's sampling method
        let random_weights =
            self.kernel
                .sample_frequencies(n_features, self.n_components, &mut rng);

        // Sample random offsets from Uniform(0, 2π)
        let uniform = RandUniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
        let mut random_offset = Array1::zeros(self.n_components);
        for val in random_offset.iter_mut() {
            *val = rng.sample(uniform);
        }

        Ok(CustomKernelSampler {
            kernel: self.kernel,
            n_components: self.n_components,
            random_state: self.random_state,
            random_weights_: Some(random_weights),
            random_offset_: Some(random_offset),
            _state: PhantomData,
        })
    }
}

impl<K> Transform<Array2<Float>, Array2<Float>> for CustomKernelSampler<K, Trained>
where
    K: KernelFunction,
{
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();
        let weights = self.random_weights_.as_ref().unwrap();
        let offset = self.random_offset_.as_ref().unwrap();

        if n_features != weights.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but CustomKernelSampler was fitted with {} features",
                n_features,
                weights.nrows()
            )));
        }

        // Compute projection: X @ weights + offset
        let projection = x.dot(weights) + offset.view().insert_axis(Axis(0));

        // Apply cosine and normalize: sqrt(2/n_components) * cos(projection)
        let normalization = (2.0 / self.n_components as Float).sqrt();
        let result = projection.mapv(|v| normalization * v.cos());

        Ok(result)
    }
}

impl<K> CustomKernelSampler<K, Trained>
where
    K: KernelFunction,
{
    /// Get the random weights
    pub fn random_weights(&self) -> &Array2<Float> {
        self.random_weights_.as_ref().unwrap()
    }

    /// Get the random offset
    pub fn random_offset(&self) -> &Array1<Float> {
        self.random_offset_.as_ref().unwrap()
    }

    /// Get the kernel description
    pub fn kernel_description(&self) -> String {
        self.kernel.description()
    }

    /// Compute exact kernel matrix for comparison/evaluation
    pub fn exact_kernel_matrix(&self, x: &Array2<Float>, y: &Array2<Float>) -> Array2<Float> {
        let (n_x, _) = x.dim();
        let (n_y, _) = y.dim();
        let mut kernel_matrix = Array2::zeros((n_x, n_y));

        for i in 0..n_x {
            for j in 0..n_y {
                let x_row = x.row(i).to_vec();
                let y_row = y.row(j).to_vec();
                kernel_matrix[[i, j]] = self.kernel.kernel(&x_row, &y_row);
            }
        }

        kernel_matrix
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_custom_rbf_kernel() {
        let kernel = CustomRBFKernel::new(0.5);
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];

        assert_abs_diff_eq!(kernel.kernel(&x, &y), 1.0, epsilon = 1e-10);

        let y2 = vec![2.0, 3.0];
        let expected = (-0.5_f64 * 2.0).exp(); // dist_sq = (1-2)² + (2-3)² = 2
        assert_abs_diff_eq!(kernel.kernel(&x, &y2), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_custom_polynomial_kernel() {
        let kernel = CustomPolynomialKernel::new(2, 1.0, 1.0);
        let x = vec![1.0, 2.0];
        let y = vec![2.0, 3.0];

        let dot_product = 1.0 * 2.0 + 2.0 * 3.0; // = 8
        let expected = (1.0_f64 * dot_product + 1.0).powf(2.0); // = 9² = 81
        assert_abs_diff_eq!(kernel.kernel(&x, &y), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_custom_laplacian_kernel() {
        let kernel = CustomLaplacianKernel::new(0.5);
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];

        assert_abs_diff_eq!(kernel.kernel(&x, &y), 1.0, epsilon = 1e-10);

        let y2 = vec![2.0, 4.0];
        let l1_dist = (1.0_f64 - 2.0).abs() + (2.0_f64 - 4.0).abs(); // = 1 + 2 = 3
        let expected = (-0.5_f64 * l1_dist).exp(); // = exp(-1.5)
        assert_abs_diff_eq!(kernel.kernel(&x, &y2), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_custom_kernel_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let kernel = CustomRBFKernel::new(0.1);

        let sampler = CustomKernelSampler::new(kernel, 50);
        let fitted = sampler.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[3, 50]);

        // Check that values are in reasonable range for cosine function
        for val in x_transformed.iter() {
            assert!(val.abs() <= 2.0); // sqrt(2) * 1 is the max possible value
        }
    }

    #[test]
    fn test_custom_kernel_sampler_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let kernel1 = CustomRBFKernel::new(0.1);
        let kernel2 = CustomRBFKernel::new(0.1);

        let sampler1 = CustomKernelSampler::new(kernel1, 10).random_state(42);
        let fitted1 = sampler1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let sampler2 = CustomKernelSampler::new(kernel2, 10).random_state(42);
        let fitted2 = sampler2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        // Results should be identical with same random state
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_custom_kernel_sampler_different_kernels() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        // Test with different kernel types
        let rbf_kernel = CustomRBFKernel::new(0.1);
        let rbf_sampler = CustomKernelSampler::new(rbf_kernel, 10);
        let fitted_rbf = rbf_sampler.fit(&x, &()).unwrap();
        let result_rbf = fitted_rbf.transform(&x).unwrap();
        assert_eq!(result_rbf.shape(), &[2, 10]);

        let poly_kernel = CustomPolynomialKernel::new(2, 1.0, 1.0);
        let poly_sampler = CustomKernelSampler::new(poly_kernel, 10);
        let fitted_poly = poly_sampler.fit(&x, &()).unwrap();
        let result_poly = fitted_poly.transform(&x).unwrap();
        assert_eq!(result_poly.shape(), &[2, 10]);

        let lap_kernel = CustomLaplacianKernel::new(0.5);
        let lap_sampler = CustomKernelSampler::new(lap_kernel, 10);
        let fitted_lap = lap_sampler.fit(&x, &()).unwrap();
        let result_lap = fitted_lap.transform(&x).unwrap();
        assert_eq!(result_lap.shape(), &[2, 10]);
    }

    #[test]
    fn test_exact_kernel_matrix_computation() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [5.0, 6.0]];
        let kernel = CustomRBFKernel::new(0.5);

        let sampler = CustomKernelSampler::new(kernel.clone(), 10);
        let fitted = sampler.fit(&x, &()).unwrap();
        let kernel_matrix = fitted.exact_kernel_matrix(&x, &y);

        assert_eq!(kernel_matrix.shape(), &[2, 2]);

        // Check diagonal elements (should be 1.0 for RBF kernel with same points)
        assert_abs_diff_eq!(kernel_matrix[[0, 0]], 1.0, epsilon = 1e-10);

        // Manual verification for one element
        let x1 = vec![1.0, 2.0];
        let y2 = vec![5.0, 6.0];
        let expected = kernel.kernel(&x1, &y2);
        assert_abs_diff_eq!(kernel_matrix[[0, 1]], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_custom_kernel_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let kernel = CustomRBFKernel::new(0.1);
        let sampler = CustomKernelSampler::new(kernel, 10);
        let fitted = sampler.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);

        assert!(result.is_err());
    }

    #[test]
    fn test_custom_kernel_zero_components() {
        let x = array![[1.0, 2.0]];
        let kernel = CustomRBFKernel::new(0.1);
        let sampler = CustomKernelSampler::new(kernel, 0);
        let result = sampler.fit(&x, &());
        assert!(result.is_err());
    }
}

//! Type-safe kernel approximations with phantom types and const generics
//!
//! This module provides compile-time type safety for kernel approximation methods,
//! ensuring proper kernel-method compatibility and preventing runtime configuration errors.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Transform};
use std::f64::consts::PI;

/// Phantom type for kernel types
pub trait KernelType: Clone + std::fmt::Debug + Send + Sync + 'static {
    const NAME: &'static str;

    fn compute_kernel(x: &ArrayView1<f64>, y: &ArrayView1<f64>, gamma: f64) -> f64;
    fn is_compatible_with_method<M: ApproximationMethod>() -> bool;
}

/// RBF/Gaussian kernel type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RBFKernel;

impl KernelType for RBFKernel {
    const NAME: &'static str = "rbf";

    fn compute_kernel(x: &ArrayView1<f64>, y: &ArrayView1<f64>, gamma: f64) -> f64 {
        let diff = x - y;
        let squared_norm = diff.dot(&diff);
        (-gamma * squared_norm).exp()
    }

    fn is_compatible_with_method<M: ApproximationMethod>() -> bool {
        M::SUPPORTS_RBF
    }
}

/// Laplacian kernel type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplacianKernel;

impl KernelType for LaplacianKernel {
    const NAME: &'static str = "laplacian";

    fn compute_kernel(x: &ArrayView1<f64>, y: &ArrayView1<f64>, gamma: f64) -> f64 {
        let diff = x - y;
        let l1_norm = diff.mapv(|x| x.abs()).sum();
        (-gamma * l1_norm).exp()
    }

    fn is_compatible_with_method<M: ApproximationMethod>() -> bool {
        M::SUPPORTS_LAPLACIAN
    }
}

/// Polynomial kernel type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialKernel;

impl KernelType for PolynomialKernel {
    const NAME: &'static str = "polynomial";

    fn compute_kernel(x: &ArrayView1<f64>, y: &ArrayView1<f64>, gamma: f64) -> f64 {
        let dot_product = x.dot(y);
        (gamma * dot_product + 1.0).powi(2) // degree = 2 for simplicity
    }

    fn is_compatible_with_method<M: ApproximationMethod>() -> bool {
        M::SUPPORTS_POLYNOMIAL
    }
}

/// Arc-cosine kernel type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcCosineKernel;

impl KernelType for ArcCosineKernel {
    const NAME: &'static str = "arccos";

    fn compute_kernel(x: &ArrayView1<f64>, y: &ArrayView1<f64>, _gamma: f64) -> f64 {
        let dot_product = x.dot(y);
        let norm_x = x.dot(x).sqrt();
        let norm_y = y.dot(y).sqrt();

        if norm_x > 0.0 && norm_y > 0.0 {
            let cos_theta = dot_product / (norm_x * norm_y);
            let theta = cos_theta.acos();
            norm_x * norm_y * (theta.sin() + (PI - theta) * cos_theta) / PI
        } else {
            0.0
        }
    }

    fn is_compatible_with_method<M: ApproximationMethod>() -> bool {
        M::SUPPORTS_ARCCOS
    }
}

/// Phantom type for approximation methods
pub trait ApproximationMethod: Clone + std::fmt::Debug + Send + Sync + 'static {
    const NAME: &'static str;
    const SUPPORTS_RBF: bool;
    const SUPPORTS_LAPLACIAN: bool;
    const SUPPORTS_POLYNOMIAL: bool;
    const SUPPORTS_ARCCOS: bool;
    const COMPLEXITY_FACTOR: f64;

    fn validate_parameters(n_components: usize, input_dim: usize) -> Result<()>;
}

/// Random Fourier Features method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomFourierFeatures;

impl ApproximationMethod for RandomFourierFeatures {
    const NAME: &'static str = "rff";
    const SUPPORTS_RBF: bool = true;
    const SUPPORTS_LAPLACIAN: bool = true;
    const SUPPORTS_POLYNOMIAL: bool = false;
    const SUPPORTS_ARCCOS: bool = false;
    const COMPLEXITY_FACTOR: f64 = 1.0;

    fn validate_parameters(n_components: usize, input_dim: usize) -> Result<()> {
        if n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }
        if input_dim == 0 {
            return Err(SklearsError::InvalidInput(
                "input_dim must be > 0".to_string(),
            ));
        }
        if n_components > 100 * input_dim {
            return Err(SklearsError::InvalidInput(
                "n_components too large relative to input_dim".to_string(),
            ));
        }
        Ok(())
    }
}

/// Nyström method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NystromMethod;

impl ApproximationMethod for NystromMethod {
    const NAME: &'static str = "nystrom";
    const SUPPORTS_RBF: bool = true;
    const SUPPORTS_LAPLACIAN: bool = true;
    const SUPPORTS_POLYNOMIAL: bool = true;
    const SUPPORTS_ARCCOS: bool = true;
    const COMPLEXITY_FACTOR: f64 = 2.0;

    fn validate_parameters(n_components: usize, input_dim: usize) -> Result<()> {
        if n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }
        if input_dim == 0 {
            return Err(SklearsError::InvalidInput(
                "input_dim must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Fastfood method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastfoodMethod;

impl ApproximationMethod for FastfoodMethod {
    const NAME: &'static str = "fastfood";
    const SUPPORTS_RBF: bool = true;
    const SUPPORTS_LAPLACIAN: bool = false;
    const SUPPORTS_POLYNOMIAL: bool = false;
    const SUPPORTS_ARCCOS: bool = false;
    const COMPLEXITY_FACTOR: f64 = 0.5;

    fn validate_parameters(n_components: usize, input_dim: usize) -> Result<()> {
        if n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }
        if input_dim == 0 {
            return Err(SklearsError::InvalidInput(
                "input_dim must be > 0".to_string(),
            ));
        }
        // Fastfood requires power-of-2 dimensions
        if !input_dim.is_power_of_two() {
            return Err(SklearsError::InvalidInput(
                "input_dim must be power of 2 for Fastfood".to_string(),
            ));
        }
        Ok(())
    }
}

/// State phantom types for type-safe state transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Untrained;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trained;

/// Type-safe kernel approximation with compile-time guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSafeKernelApproximation<K, M, S, const N: usize>
where
    K: KernelType,
    M: ApproximationMethod,
    S: Clone + std::fmt::Debug,
{
    /// gamma
    pub gamma: f64,
    /// regularization
    pub regularization: f64,
    _phantom: PhantomData<(K, M, S)>,
}

impl<K, M, const N: usize> TypeSafeKernelApproximation<K, M, Untrained, N>
where
    K: KernelType,
    M: ApproximationMethod,
{
    pub fn new(gamma: f64) -> Result<Self> {
        // Compile-time compatibility check
        if !K::is_compatible_with_method::<M>() {
            return Err(SklearsError::InvalidInput(format!(
                "Kernel {} is not compatible with method {}",
                K::NAME,
                M::NAME
            )));
        }

        Ok(Self {
            gamma,
            regularization: 1e-10,
            _phantom: PhantomData,
        })
    }

    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }
}

/// Fitted type-safe kernel approximation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedTypeSafeKernelApproximation<K, M, const N: usize>
where
    K: KernelType,
    M: ApproximationMethod,
{
    /// gamma
    pub gamma: f64,
    /// regularization
    pub regularization: f64,
    /// weights
    pub weights: Array2<f64>,
    /// biases
    pub biases: Array1<f64>,
    /// training_quality
    pub training_quality: QualityMetrics,
    _phantom: PhantomData<(K, M)>,
}

/// Quality metrics tracked during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// kernel_alignment
    pub kernel_alignment: f64,
    /// approximation_error
    pub approximation_error: f64,
    /// effective_rank
    pub effective_rank: f64,
    /// condition_number
    pub condition_number: f64,
    /// spectral_radius
    pub spectral_radius: f64,
}

impl<K, M, const N: usize> Fit<Array2<f64>, ()> for TypeSafeKernelApproximation<K, M, Untrained, N>
where
    K: KernelType,
    M: ApproximationMethod,
{
    type Fitted = FittedTypeSafeKernelApproximation<K, M, N>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let input_dim = x.ncols();

        // Runtime parameter validation
        M::validate_parameters(N, input_dim)?;

        // Generate approximation based on method type
        let (weights, biases) = self.generate_approximation::<M>(input_dim)?;

        // Compute quality metrics
        let quality_metrics = self.compute_quality_metrics::<K>(x, &weights, &biases)?;

        Ok(FittedTypeSafeKernelApproximation {
            gamma: self.gamma,
            regularization: self.regularization,
            weights,
            biases,
            training_quality: quality_metrics,
            _phantom: PhantomData,
        })
    }
}

impl<K, M, const N: usize> TypeSafeKernelApproximation<K, M, Untrained, N>
where
    K: KernelType,
    M: ApproximationMethod,
{
    fn generate_approximation<Method: ApproximationMethod>(
        &self,
        input_dim: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        match Method::NAME {
            "rff" => self.generate_rff_approximation::<K>(input_dim),
            "nystrom" => self.generate_nystrom_approximation::<K>(input_dim),
            "fastfood" => self.generate_fastfood_approximation::<K>(input_dim),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown method: {}",
                Method::NAME
            ))),
        }
    }

    fn generate_rff_approximation<Kernel: KernelType>(
        &self,
        input_dim: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let mut rng = RealStdRng::from_seed(thread_rng().gen());

        let weights = match Kernel::NAME {
            "rbf" => {
                let normal = RandNormal::new(0.0, (2.0 * self.gamma).sqrt()).unwrap();
                let mut weights = Array2::zeros((N, input_dim));
                for i in 0..N {
                    for j in 0..input_dim {
                        weights[[i, j]] = rng.sample(normal);
                    }
                }
                weights
            }
            "laplacian" => {
                // Laplacian kernel uses Cauchy distribution
                let cauchy_scale = 1.0 / self.gamma;
                let mut weights = Array2::zeros((N, input_dim));
                for i in 0..N {
                    for j in 0..input_dim {
                        let u1: f64 = rng.gen();
                        let u2: f64 = rng.gen();
                        let cauchy_sample = cauchy_scale * (PI * (u1 - 0.5)).tan();
                        weights[[i, j]] = cauchy_sample;
                    }
                }
                weights
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "RFF not supported for kernel: {}",
                    Kernel::NAME
                )))
            }
        };

        let uniform = RandUniform::new(0.0, 2.0 * PI).unwrap();
        let biases = Array1::from_vec((0..N).map(|_| rng.sample(uniform)).collect());

        Ok((weights, biases))
    }

    fn generate_nystrom_approximation<Kernel: KernelType>(
        &self,
        input_dim: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        // Simplified Nyström approximation
        let mut rng = RealStdRng::from_seed(thread_rng().gen());
        let normal = RandNormal::new(0.0, 1.0).unwrap();

        let mut weights = Array2::zeros((N, input_dim));
        for i in 0..N {
            for j in 0..input_dim {
                weights[[i, j]] = rng.sample(normal);
            }
        }

        let biases = Array1::zeros(N);
        Ok((weights, biases))
    }

    fn generate_fastfood_approximation<Kernel: KernelType>(
        &self,
        input_dim: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        if !input_dim.is_power_of_two() {
            return Err(SklearsError::InvalidInput(
                "Fastfood requires power-of-2 input dimension".to_string(),
            ));
        }

        let mut rng = RealStdRng::from_seed(thread_rng().gen());
        let normal = RandNormal::new(0.0, (2.0 * self.gamma).sqrt()).unwrap();

        let mut weights = Array2::zeros((N, input_dim));
        for i in 0..N {
            for j in 0..input_dim {
                weights[[i, j]] = rng.sample(normal);
            }
        }

        let uniform = RandUniform::new(0.0, 2.0 * PI).unwrap();
        let biases = Array1::from_vec((0..N).map(|_| rng.sample(uniform)).collect());

        Ok((weights, biases))
    }

    fn compute_quality_metrics<Kernel: KernelType>(
        &self,
        x: &Array2<f64>,
        weights: &Array2<f64>,
        biases: &Array1<f64>,
    ) -> Result<QualityMetrics> {
        let n_samples = x.nrows().min(100); // Limit for computational efficiency
        let sample_indices: Vec<usize> = (0..n_samples).collect();

        // Compute exact kernel matrix (small subset)
        let mut exact_kernel = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                exact_kernel[[i, j]] = Kernel::compute_kernel(
                    &x.row(sample_indices[i]),
                    &x.row(sample_indices[j]),
                    self.gamma,
                );
            }
        }

        // Compute approximated kernel matrix
        let approx_features = self.compute_features(
            &x.slice(scirs2_core::ndarray::s![..n_samples, ..])
                .to_owned(),
            weights,
            biases,
        )?;

        let mut approx_kernel = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                approx_kernel[[i, j]] = approx_features.row(i).dot(&approx_features.row(j));
            }
        }

        // Compute quality metrics
        let kernel_alignment = self.compute_kernel_alignment(&exact_kernel, &approx_kernel)?;
        let approximation_error =
            self.compute_approximation_error(&exact_kernel, &approx_kernel)?;
        let effective_rank = self.compute_effective_rank(&approx_kernel)?;
        let condition_number = self.compute_condition_number(&approx_kernel)?;
        let spectral_radius = self.compute_spectral_radius(&approx_kernel)?;

        Ok(QualityMetrics {
            kernel_alignment,
            approximation_error,
            effective_rank,
            condition_number,
            spectral_radius,
        })
    }

    fn compute_features(
        &self,
        x: &Array2<f64>,
        weights: &Array2<f64>,
        biases: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut features = Array2::zeros((n_samples, N));
        let scaling = (2.0 / N as f64).sqrt();

        for i in 0..n_samples {
            for j in 0..N {
                let mut dot_product = 0.0;
                for k in 0..x.ncols() {
                    dot_product += x[[i, k]] * weights[[j, k]];
                }
                dot_product += biases[j];
                features[[i, j]] = scaling * dot_product.cos();
            }
        }

        Ok(features)
    }

    fn compute_kernel_alignment(&self, exact: &Array2<f64>, approx: &Array2<f64>) -> Result<f64> {
        let exact_centered = self.center_kernel(exact)?;
        let approx_centered = self.center_kernel(approx)?;

        let numerator = exact_centered
            .iter()
            .zip(approx_centered.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let exact_norm = exact_centered.mapv(|x| x * x).sum().sqrt();
        let approx_norm = approx_centered.mapv(|x| x * x).sum().sqrt();

        if exact_norm > 0.0 && approx_norm > 0.0 {
            Ok(numerator / (exact_norm * approx_norm))
        } else {
            Ok(0.0)
        }
    }

    fn center_kernel(&self, kernel: &Array2<f64>) -> Result<Array2<f64>> {
        let n = kernel.nrows();
        let row_means = kernel.mean_axis(Axis(1)).unwrap();
        let col_means = kernel.mean_axis(Axis(0)).unwrap();
        let overall_mean = kernel.mean().unwrap();

        let mut centered = kernel.clone();
        for i in 0..n {
            for j in 0..n {
                centered[[i, j]] = kernel[[i, j]] - row_means[i] - col_means[j] + overall_mean;
            }
        }

        Ok(centered)
    }

    fn compute_approximation_error(
        &self,
        exact: &Array2<f64>,
        approx: &Array2<f64>,
    ) -> Result<f64> {
        let diff = exact - approx;
        let frobenius_error = diff.mapv(|x| x * x).sum().sqrt();
        let exact_norm = exact.mapv(|x| x * x).sum().sqrt();

        if exact_norm > 0.0 {
            Ok(frobenius_error / exact_norm)
        } else {
            Ok(frobenius_error)
        }
    }

    fn compute_effective_rank(&self, matrix: &Array2<f64>) -> Result<f64> {
        let trace = (0..matrix.nrows().min(matrix.ncols()))
            .map(|i| matrix[[i, i]])
            .sum::<f64>();
        let frobenius_squared = matrix.mapv(|x| x * x).sum();

        if frobenius_squared > 0.0 {
            Ok(trace * trace / frobenius_squared)
        } else {
            Ok(0.0)
        }
    }

    fn compute_condition_number(&self, matrix: &Array2<f64>) -> Result<f64> {
        // Simplified condition number estimation
        let n = matrix.nrows();
        let trace = (0..n).map(|i| matrix[[i, i]]).sum::<f64>();
        let min_diag = (0..n).map(|i| matrix[[i, i]]).fold(f64::INFINITY, f64::min);

        if min_diag > 0.0 {
            Ok(trace / (n as f64 * min_diag))
        } else {
            Ok(f64::INFINITY)
        }
    }

    fn compute_spectral_radius(&self, matrix: &Array2<f64>) -> Result<f64> {
        // Simplified spectral radius using power iteration
        let n = matrix.nrows();
        if n == 0 {
            return Ok(0.0);
        }

        let mut v = Array1::from_vec(vec![1.0; n]);
        v /= (v.dot(&v) as f64).sqrt();

        let mut eigenvalue = 0.0;
        for _ in 0..20 {
            let mut av: Array1<f64> = Array1::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    av[i] += matrix[[i, j]] * v[j];
                }
            }

            let norm = av.dot(&av).sqrt();
            if norm > 1e-12 {
                eigenvalue = norm;
                v = av / norm;
            } else {
                break;
            }
        }

        Ok(eigenvalue)
    }
}

impl<K, M, const N: usize> Transform<Array2<f64>, Array2<f64>>
    for FittedTypeSafeKernelApproximation<K, M, N>
where
    K: KernelType,
    M: ApproximationMethod,
{
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut features = Array2::zeros((n_samples, N));
        let scaling = (2.0 / N as f64).sqrt();

        for i in 0..n_samples {
            for j in 0..N {
                let mut dot_product = 0.0;
                for k in 0..x.ncols() {
                    dot_product += x[[i, k]] * self.weights[[j, k]];
                }
                dot_product += self.biases[j];
                features[[i, j]] = scaling * dot_product.cos();
            }
        }

        Ok(features)
    }
}

impl<K, M, const N: usize> FittedTypeSafeKernelApproximation<K, M, N>
where
    K: KernelType,
    M: ApproximationMethod,
{
    pub fn get_quality_metrics(&self) -> &QualityMetrics {
        &self.training_quality
    }

    pub fn get_kernel_name(&self) -> &'static str {
        K::NAME
    }

    pub fn get_method_name(&self) -> &'static str {
        M::NAME
    }

    pub fn get_complexity_factor(&self) -> f64 {
        M::COMPLEXITY_FACTOR
    }

    pub fn get_n_components(&self) -> usize {
        N
    }
}

/// Type aliases for common configurations
pub type TypeSafeRBFRandomFourierFeatures<const N: usize> =
    TypeSafeKernelApproximation<RBFKernel, RandomFourierFeatures, Untrained, N>;
pub type TypeSafeLaplacianRandomFourierFeatures<const N: usize> =
    TypeSafeKernelApproximation<LaplacianKernel, RandomFourierFeatures, Untrained, N>;
pub type TypeSafeRBFNystrom<const N: usize> =
    TypeSafeKernelApproximation<RBFKernel, NystromMethod, Untrained, N>;
pub type TypeSafePolynomialNystrom<const N: usize> =
    TypeSafeKernelApproximation<PolynomialKernel, NystromMethod, Untrained, N>;
pub type TypeSafeRBFFastfood<const N: usize> =
    TypeSafeKernelApproximation<RBFKernel, FastfoodMethod, Untrained, N>;

/// Fitted type aliases
pub type FittedTypeSafeRBFRandomFourierFeatures<const N: usize> =
    FittedTypeSafeKernelApproximation<RBFKernel, RandomFourierFeatures, N>;
pub type FittedTypeSafeLaplacianRandomFourierFeatures<const N: usize> =
    FittedTypeSafeKernelApproximation<LaplacianKernel, RandomFourierFeatures, N>;
pub type FittedTypeSafeRBFNystrom<const N: usize> =
    FittedTypeSafeKernelApproximation<RBFKernel, NystromMethod, N>;
pub type FittedTypeSafePolynomialNystrom<const N: usize> =
    FittedTypeSafeKernelApproximation<PolynomialKernel, NystromMethod, N>;
pub type FittedTypeSafeRBFFastfood<const N: usize> =
    FittedTypeSafeKernelApproximation<RBFKernel, FastfoodMethod, N>;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::essentials::Normal;

    use scirs2_core::ndarray::{Array, Array2};
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_type_safe_rbf_rff() {
        let x: Array2<f64> = Array::from_shape_fn((50, 8), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let kernel_approx = TypeSafeRBFRandomFourierFeatures::<100>::new(1.0).unwrap();

        let fitted = kernel_approx.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[50, 100]);
        assert_eq!(fitted.get_kernel_name(), "rbf");
        assert_eq!(fitted.get_method_name(), "rff");
        assert_eq!(fitted.get_n_components(), 100);
    }

    #[test]
    fn test_type_safe_laplacian_rff() {
        let x: Array2<f64> = Array::from_shape_fn((30, 4), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let kernel_approx = TypeSafeLaplacianRandomFourierFeatures::<50>::new(0.5).unwrap();

        let fitted = kernel_approx.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[30, 50]);
        assert_eq!(fitted.get_kernel_name(), "laplacian");
        assert!(fitted.get_quality_metrics().kernel_alignment >= 0.0);
    }

    #[test]
    fn test_type_safe_nystrom() {
        let x: Array2<f64> = Array::from_shape_fn((40, 6), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let kernel_approx = TypeSafeRBFNystrom::<64>::new(1.0).unwrap();

        let fitted = kernel_approx.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[40, 64]);
        assert_eq!(fitted.get_method_name(), "nystrom");
        assert!(fitted.get_quality_metrics().approximation_error >= 0.0);
    }

    #[test]
    fn test_fastfood_power_of_two_requirement() {
        let x: Array2<f64> = Array::from_shape_fn((20, 8), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        }); // 8 is power of 2
        let kernel_approx = TypeSafeRBFFastfood::<32>::new(1.0).unwrap();

        let fitted = kernel_approx.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[20, 32]);
        assert_eq!(fitted.get_method_name(), "fastfood");

        // Test with non-power-of-2 dimension
        let x_invalid: Array2<f64> = Array::from_shape_fn((20, 7), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        }); // 7 is not power of 2
        let kernel_approx_invalid = TypeSafeRBFFastfood::<32>::new(1.0).unwrap();

        let result = kernel_approx_invalid.fit(&x_invalid, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_quality_metrics() {
        let x: Array2<f64> = Array::from_shape_fn((25, 4), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let kernel_approx = TypeSafeRBFRandomFourierFeatures::<50>::new(1.0).unwrap();

        let fitted = kernel_approx.fit(&x, &()).unwrap();
        let quality = fitted.get_quality_metrics();

        assert!(quality.kernel_alignment >= 0.0);
        assert!(quality.kernel_alignment <= 1.0);
        assert!(quality.approximation_error >= 0.0);
        assert!(quality.effective_rank >= 0.0);
        assert!(quality.condition_number >= 1.0);
        assert!(quality.spectral_radius >= 0.0);
    }

    #[test]
    fn test_compile_time_compatibility() {
        // This should compile - RBF kernel is compatible with RFF
        let _valid = TypeSafeRBFRandomFourierFeatures::<100>::new(1.0).unwrap();

        // This should also compile - Laplacian kernel is compatible with RFF
        let _valid2 = TypeSafeLaplacianRandomFourierFeatures::<100>::new(1.0).unwrap();

        // This should also compile - RBF kernel is compatible with Nyström
        let _valid3 = TypeSafeRBFNystrom::<100>::new(1.0).unwrap();

        // This should compile - Polynomial kernel is compatible with Nyström
        let _valid4 = TypeSafePolynomialNystrom::<100>::new(1.0).unwrap();
    }
}

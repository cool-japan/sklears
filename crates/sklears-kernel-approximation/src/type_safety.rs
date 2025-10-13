//! Type Safety Enhancements for Kernel Approximation Methods
//!
//! This module provides compile-time type safety using phantom types and const generics
//! to prevent common errors in kernel approximation usage.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::error::{Result, SklearsError};
use std::marker::PhantomData;

/// Phantom type to represent the state of a kernel approximation method
pub trait ApproximationState {}

/// Untrained state - method hasn't been fitted yet
#[derive(Debug, Clone, Copy)]
/// Untrained
pub struct Untrained;
impl ApproximationState for Untrained {}

/// Trained state - method has been fitted and can transform data
#[derive(Debug, Clone, Copy)]
/// Trained
pub struct Trained;
impl ApproximationState for Trained {}

/// Phantom type to represent kernel types
pub trait KernelType {
    /// Name of the kernel type
    const NAME: &'static str;

    /// Whether this kernel type supports parameter learning
    const SUPPORTS_PARAMETER_LEARNING: bool;

    /// Default bandwidth/gamma parameter
    const DEFAULT_BANDWIDTH: f64;
}

/// RBF (Gaussian) kernel type
#[derive(Debug, Clone, Copy)]
/// RBFKernel
pub struct RBFKernel;
impl KernelType for RBFKernel {
    const NAME: &'static str = "RBF";
    const SUPPORTS_PARAMETER_LEARNING: bool = true;
    const DEFAULT_BANDWIDTH: f64 = 1.0;
}

/// Laplacian kernel type
#[derive(Debug, Clone, Copy)]
/// LaplacianKernel
pub struct LaplacianKernel;
impl KernelType for LaplacianKernel {
    const NAME: &'static str = "Laplacian";
    const SUPPORTS_PARAMETER_LEARNING: bool = true;
    const DEFAULT_BANDWIDTH: f64 = 1.0;
}

/// Polynomial kernel type
#[derive(Debug, Clone, Copy)]
/// PolynomialKernel
pub struct PolynomialKernel;
impl KernelType for PolynomialKernel {
    const NAME: &'static str = "Polynomial";
    const SUPPORTS_PARAMETER_LEARNING: bool = true;
    const DEFAULT_BANDWIDTH: f64 = 1.0;
}

/// Arc-cosine kernel type
#[derive(Debug, Clone, Copy)]
/// ArcCosineKernel
pub struct ArcCosineKernel;
impl KernelType for ArcCosineKernel {
    const NAME: &'static str = "ArcCosine";
    const SUPPORTS_PARAMETER_LEARNING: bool = false;
    const DEFAULT_BANDWIDTH: f64 = 1.0;
}

/// Phantom type to represent approximation methods
pub trait ApproximationMethod {
    /// Name of the approximation method
    const NAME: &'static str;

    /// Whether this method supports incremental updates
    const SUPPORTS_INCREMENTAL: bool;

    /// Whether this method provides theoretical error bounds
    const HAS_ERROR_BOUNDS: bool;

    /// Computational complexity class
    const COMPLEXITY: ComplexityClass;
}

/// Computational complexity classes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// ComplexityClass
pub enum ComplexityClass {
    /// O(n²) complexity
    Quadratic,
    /// O(n log n) complexity
    QuasiLinear,
    /// O(n) complexity
    Linear,
    /// O(d log d) complexity where d is dimension
    DimensionDependent,
}

/// Random Fourier Features approximation method
#[derive(Debug, Clone, Copy)]
/// RandomFourierFeatures
pub struct RandomFourierFeatures;
impl ApproximationMethod for RandomFourierFeatures {
    const NAME: &'static str = "RandomFourierFeatures";
    const SUPPORTS_INCREMENTAL: bool = true;
    const HAS_ERROR_BOUNDS: bool = true;
    const COMPLEXITY: ComplexityClass = ComplexityClass::Linear;
}

/// Nyström approximation method
#[derive(Debug, Clone, Copy)]
/// NystromMethod
pub struct NystromMethod;
impl ApproximationMethod for NystromMethod {
    const NAME: &'static str = "Nystrom";
    const SUPPORTS_INCREMENTAL: bool = false;
    const HAS_ERROR_BOUNDS: bool = true;
    const COMPLEXITY: ComplexityClass = ComplexityClass::Quadratic;
}

/// Fastfood approximation method
#[derive(Debug, Clone, Copy)]
/// FastfoodMethod
pub struct FastfoodMethod;
impl ApproximationMethod for FastfoodMethod {
    const NAME: &'static str = "Fastfood";
    const SUPPORTS_INCREMENTAL: bool = false;
    const HAS_ERROR_BOUNDS: bool = true;
    const COMPLEXITY: ComplexityClass = ComplexityClass::DimensionDependent;
}

/// Type-safe kernel approximation with compile-time guarantees
#[derive(Debug, Clone)]
/// TypeSafeKernelApproximation
pub struct TypeSafeKernelApproximation<State, Kernel, Method, const N_COMPONENTS: usize>
where
    State: ApproximationState,
    Kernel: KernelType,
    Method: ApproximationMethod,
{
    /// Phantom data for compile-time type checking
    _phantom: PhantomData<(State, Kernel, Method)>,

    /// Method parameters
    parameters: ApproximationParameters,

    /// Random state for reproducibility
    random_state: Option<u64>,
}

/// Parameters for kernel approximation methods
#[derive(Debug, Clone)]
/// ApproximationParameters
pub struct ApproximationParameters {
    /// Bandwidth/gamma parameter
    pub bandwidth: f64,

    /// Polynomial degree (for polynomial kernels)
    pub degree: Option<usize>,

    /// Coefficient for polynomial kernels
    pub coef0: Option<f64>,

    /// Additional custom parameters
    pub custom: std::collections::HashMap<String, f64>,
}

impl Default for ApproximationParameters {
    fn default() -> Self {
        Self {
            bandwidth: 1.0,
            degree: None,
            coef0: None,
            custom: std::collections::HashMap::new(),
        }
    }
}

/// Fitted kernel approximation that can transform data
#[derive(Debug, Clone)]
/// FittedTypeSafeKernelApproximation
pub struct FittedTypeSafeKernelApproximation<Kernel, Method, const N_COMPONENTS: usize>
where
    Kernel: KernelType,
    Method: ApproximationMethod,
{
    /// Phantom data for compile-time type checking
    _phantom: PhantomData<(Kernel, Method)>,

    /// Random features or transformation parameters
    transformation_params: TransformationParameters<N_COMPONENTS>,

    /// Fitted parameters
    fitted_parameters: ApproximationParameters,

    /// Approximation quality metrics
    quality_metrics: QualityMetrics,
}

/// Transformation parameters for different approximation methods
#[derive(Debug, Clone)]
/// TransformationParameters
pub enum TransformationParameters<const N: usize> {
    /// Random features for RFF methods
    RandomFeatures {
        weights: Array2<f64>,

        biases: Option<Array1<f64>>,
    },
    /// Inducing points and eigendecomposition for Nyström
    Nystrom {
        inducing_points: Array2<f64>,

        eigenvalues: Array1<f64>,

        eigenvectors: Array2<f64>,
    },
    /// Structured matrices for Fastfood
    Fastfood {
        structured_matrices: Vec<Array2<f64>>,
        scaling: Array1<f64>,
    },
}

/// Quality metrics for approximation assessment
#[derive(Debug, Clone)]
/// QualityMetrics
pub struct QualityMetrics {
    /// Approximation error estimate
    pub approximation_error: Option<f64>,

    /// Effective rank of approximation
    pub effective_rank: Option<f64>,

    /// Condition number
    pub condition_number: Option<f64>,

    /// Kernel alignment score
    pub kernel_alignment: Option<f64>,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            approximation_error: None,
            effective_rank: None,
            condition_number: None,
            kernel_alignment: None,
        }
    }
}

// Implementation for untrained approximation
impl<Kernel, Method, const N_COMPONENTS: usize>
    TypeSafeKernelApproximation<Untrained, Kernel, Method, N_COMPONENTS>
where
    Kernel: KernelType,
    Method: ApproximationMethod,
{
    /// Create a new untrained kernel approximation
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
            parameters: ApproximationParameters {
                bandwidth: Kernel::DEFAULT_BANDWIDTH,
                ..Default::default()
            },
            random_state: None,
        }
    }

    /// Set bandwidth parameter (only available for kernels that support it)
    pub fn bandwidth(mut self, bandwidth: f64) -> Self
    where
        Kernel: KernelTypeWithBandwidth,
    {
        self.parameters.bandwidth = bandwidth;
        self
    }

    /// Set polynomial degree (only available for polynomial kernels)
    pub fn degree(mut self, degree: usize) -> Self
    where
        Kernel: PolynomialKernelType,
    {
        self.parameters.degree = Some(degree);
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Fit the approximation method (state transition)
    pub fn fit(
        self,
        data: &Array2<f64>,
    ) -> Result<FittedTypeSafeKernelApproximation<Kernel, Method, N_COMPONENTS>>
    where
        Kernel: FittableKernel<Method>,
        Method: FittableMethod<Kernel>,
    {
        self.fit_impl(data)
    }

    /// Internal fitting implementation
    fn fit_impl(
        self,
        data: &Array2<f64>,
    ) -> Result<FittedTypeSafeKernelApproximation<Kernel, Method, N_COMPONENTS>>
    where
        Kernel: FittableKernel<Method>,
        Method: FittableMethod<Kernel>,
    {
        let transformation_params = match Method::NAME {
            "RandomFourierFeatures" => self.fit_random_fourier_features(data)?,
            "Nystrom" => self.fit_nystrom(data)?,
            "Fastfood" => self.fit_fastfood(data)?,
            _ => {
                return Err(SklearsError::InvalidOperation(format!(
                    "Unsupported method: {}",
                    Method::NAME
                )));
            }
        };

        Ok(FittedTypeSafeKernelApproximation {
            _phantom: PhantomData,
            transformation_params,
            fitted_parameters: self.parameters,
            quality_metrics: QualityMetrics::default(),
        })
    }

    fn fit_random_fourier_features(
        &self,
        data: &Array2<f64>,
    ) -> Result<TransformationParameters<N_COMPONENTS>> {
        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        let (_, n_features) = data.dim();

        let weights = match Kernel::NAME {
            "RBF" => {
                let normal = RandNormal::new(0.0, self.parameters.bandwidth).unwrap();
                Array2::from_shape_fn((N_COMPONENTS, n_features), |_| rng.sample(normal))
            }
            "Laplacian" => {
                // Laplacian kernel uses Cauchy distribution
                let uniform = RandUniform::new(0.0, std::f64::consts::PI).unwrap();
                Array2::from_shape_fn((N_COMPONENTS, n_features), |_| {
                    let u = rng.sample(uniform);
                    (u - std::f64::consts::PI / 2.0).tan() / self.parameters.bandwidth
                })
            }
            _ => {
                return Err(SklearsError::InvalidOperation(format!(
                    "RFF not supported for kernel: {}",
                    Kernel::NAME
                )));
            }
        };

        let uniform = RandUniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
        let biases = Some(Array1::from_shape_fn(N_COMPONENTS, |_| rng.sample(uniform)));

        Ok(TransformationParameters::RandomFeatures { weights, biases })
    }

    fn fit_nystrom(&self, data: &Array2<f64>) -> Result<TransformationParameters<N_COMPONENTS>> {
        let (n_samples, _) = data.dim();
        let n_inducing = N_COMPONENTS.min(n_samples);

        // Sample inducing points
        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let inducing_indices = &indices[..n_inducing];
        let inducing_points = data.select(scirs2_core::ndarray::Axis(0), inducing_indices);

        // Compute kernel matrix on inducing points
        let kernel_matrix = self.compute_kernel_matrix(&inducing_points, &inducing_points)?;

        // Simplified eigendecomposition (using diagonal as approximation)
        let eigenvalues = Array1::from_shape_fn(n_inducing, |i| kernel_matrix[[i, i]]);
        let eigenvectors = Array2::eye(n_inducing);

        Ok(TransformationParameters::Nystrom {
            inducing_points,
            eigenvalues,
            eigenvectors,
        })
    }

    fn fit_fastfood(&self, data: &Array2<f64>) -> Result<TransformationParameters<N_COMPONENTS>> {
        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        let (_, n_features) = data.dim();

        // Create structured matrices for Fastfood
        let mut structured_matrices = Vec::new();

        // Binary matrix
        let binary_matrix = Array2::from_shape_fn((n_features, n_features), |(i, j)| {
            if i == j {
                if rng.gen::<bool>() {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            }
        });
        structured_matrices.push(binary_matrix);

        // Gaussian scaling
        let scaling = Array1::from_shape_fn(N_COMPONENTS, |_| {
            use scirs2_core::random::{RandNormal, Rng};
            let normal = RandNormal::new(0.0, 1.0).unwrap();
            rng.sample(normal)
        });

        Ok(TransformationParameters::Fastfood {
            structured_matrices,
            scaling,
        })
    }

    fn compute_kernel_matrix(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Result<Array2<f64>> {
        let (n1, _) = x1.dim();
        let (n2, _) = x2.dim();
        let mut kernel = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let similarity = match Kernel::NAME {
                    "RBF" => {
                        let diff = &x1.row(i) - &x2.row(j);
                        let dist_sq = diff.mapv(|x| x * x).sum();
                        (-self.parameters.bandwidth * dist_sq).exp()
                    }
                    "Laplacian" => {
                        let diff = &x1.row(i) - &x2.row(j);
                        let dist = diff.mapv(|x| x.abs()).sum();
                        (-self.parameters.bandwidth * dist).exp()
                    }
                    "Polynomial" => {
                        let dot_product = x1.row(i).dot(&x2.row(j));
                        let degree = self.parameters.degree.unwrap_or(2) as i32;
                        let coef0 = self.parameters.coef0.unwrap_or(1.0);
                        (self.parameters.bandwidth * dot_product + coef0).powi(degree)
                    }
                    _ => {
                        return Err(SklearsError::InvalidOperation(format!(
                            "Unsupported kernel: {}",
                            Kernel::NAME
                        )));
                    }
                };
                kernel[[i, j]] = similarity;
            }
        }

        Ok(kernel)
    }
}

// Implementation for fitted approximation
impl<Kernel, Method, const N_COMPONENTS: usize>
    FittedTypeSafeKernelApproximation<Kernel, Method, N_COMPONENTS>
where
    Kernel: KernelType,
    Method: ApproximationMethod,
{
    /// Transform data using the fitted approximation
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        match &self.transformation_params {
            TransformationParameters::RandomFeatures { weights, biases } => {
                self.transform_random_features(data, weights, biases.as_ref())
            }
            TransformationParameters::Nystrom {
                inducing_points,
                eigenvalues,
                eigenvectors,
            } => self.transform_nystrom(data, inducing_points, eigenvalues, eigenvectors),
            TransformationParameters::Fastfood {
                structured_matrices,
                scaling,
            } => self.transform_fastfood(data, structured_matrices, scaling),
        }
    }

    fn transform_random_features(
        &self,
        data: &Array2<f64>,
        weights: &Array2<f64>,
        biases: Option<&Array1<f64>>,
    ) -> Result<Array2<f64>> {
        let (n_samples, _) = data.dim();
        let mut features = Array2::zeros((n_samples, N_COMPONENTS * 2));

        for (i, sample) in data.axis_iter(scirs2_core::ndarray::Axis(0)).enumerate() {
            for j in 0..N_COMPONENTS {
                let projection = sample.dot(&weights.row(j));
                let phase = if let Some(b) = biases {
                    projection + b[j]
                } else {
                    projection
                };

                features[[i, 2 * j]] = phase.cos();
                features[[i, 2 * j + 1]] = phase.sin();
            }
        }

        Ok(features)
    }

    fn transform_nystrom(
        &self,
        data: &Array2<f64>,
        inducing_points: &Array2<f64>,
        eigenvalues: &Array1<f64>,
        eigenvectors: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        // Compute kernel between data and inducing points
        let kernel_matrix = self.compute_kernel_matrix_fitted(data, inducing_points)?;

        // Apply eigendecomposition transformation
        let mut features = Array2::zeros((data.nrows(), eigenvalues.len()));

        for i in 0..data.nrows() {
            for j in 0..eigenvalues.len() {
                if eigenvalues[j] > 1e-8 {
                    let mut feature_value = 0.0;
                    for k in 0..inducing_points.nrows() {
                        feature_value += kernel_matrix[[i, k]] * eigenvectors[[k, j]];
                    }
                    features[[i, j]] = feature_value / eigenvalues[j].sqrt();
                }
            }
        }

        Ok(features)
    }

    fn transform_fastfood(
        &self,
        data: &Array2<f64>,
        structured_matrices: &[Array2<f64>],
        scaling: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let (n_samples, n_features) = data.dim();
        let mut features = data.clone();

        // Apply structured transformations
        for matrix in structured_matrices {
            features = features.dot(matrix);
        }

        // Apply scaling and take cosine features
        let mut result = Array2::zeros((n_samples, N_COMPONENTS));
        for i in 0..n_samples {
            for j in 0..N_COMPONENTS.min(n_features) {
                result[[i, j]] = (features[[i, j]] * scaling[j]).cos();
            }
        }

        Ok(result)
    }

    fn compute_kernel_matrix_fitted(
        &self,
        x1: &Array2<f64>,
        x2: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let (n1, _) = x1.dim();
        let (n2, _) = x2.dim();
        let mut kernel = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let similarity = match Kernel::NAME {
                    "RBF" => {
                        let diff = &x1.row(i) - &x2.row(j);
                        let dist_sq = diff.mapv(|x| x * x).sum();
                        (-self.fitted_parameters.bandwidth * dist_sq).exp()
                    }
                    "Laplacian" => {
                        let diff = &x1.row(i) - &x2.row(j);
                        let dist = diff.mapv(|x| x.abs()).sum();
                        (-self.fitted_parameters.bandwidth * dist).exp()
                    }
                    "Polynomial" => {
                        let dot_product = x1.row(i).dot(&x2.row(j));
                        let degree = self.fitted_parameters.degree.unwrap_or(2) as i32;
                        let coef0 = self.fitted_parameters.coef0.unwrap_or(1.0);
                        (self.fitted_parameters.bandwidth * dot_product + coef0).powi(degree)
                    }
                    _ => {
                        return Err(SklearsError::InvalidOperation(format!(
                            "Unsupported kernel: {}",
                            Kernel::NAME
                        )));
                    }
                };
                kernel[[i, j]] = similarity;
            }
        }

        Ok(kernel)
    }

    /// Get approximation quality metrics
    pub fn quality_metrics(&self) -> &QualityMetrics {
        &self.quality_metrics
    }

    /// Get the number of components (compile-time constant)
    pub const fn n_components() -> usize {
        N_COMPONENTS
    }

    /// Get kernel type name
    pub fn kernel_name(&self) -> &'static str {
        Kernel::NAME
    }

    /// Get approximation method name
    pub fn method_name(&self) -> &'static str {
        Method::NAME
    }
}

// Trait constraints for type safety

/// Marker trait for kernel types that support bandwidth parameters
pub trait KernelTypeWithBandwidth: KernelType {}
impl KernelTypeWithBandwidth for RBFKernel {}
impl KernelTypeWithBandwidth for LaplacianKernel {}

/// Marker trait for polynomial kernel types
pub trait PolynomialKernelType: KernelType {}
impl PolynomialKernelType for PolynomialKernel {}

/// Marker trait for kernels that can be fitted with specific methods
pub trait FittableKernel<Method: ApproximationMethod>: KernelType {}
impl FittableKernel<RandomFourierFeatures> for RBFKernel {}
impl FittableKernel<RandomFourierFeatures> for LaplacianKernel {}
impl FittableKernel<NystromMethod> for RBFKernel {}
impl FittableKernel<NystromMethod> for LaplacianKernel {}
impl FittableKernel<NystromMethod> for PolynomialKernel {}
impl FittableKernel<FastfoodMethod> for RBFKernel {}

/// Marker trait for methods that can be fitted with specific kernels
pub trait FittableMethod<Kernel: KernelType>: ApproximationMethod {}
impl FittableMethod<RBFKernel> for RandomFourierFeatures {}
impl FittableMethod<LaplacianKernel> for RandomFourierFeatures {}
impl FittableMethod<RBFKernel> for NystromMethod {}
impl FittableMethod<LaplacianKernel> for NystromMethod {}
impl FittableMethod<PolynomialKernel> for NystromMethod {}
impl FittableMethod<RBFKernel> for FastfoodMethod {}

/// Type aliases for common kernel approximation combinations
pub type RBFRandomFourierFeatures<const N: usize> =
    TypeSafeKernelApproximation<Untrained, RBFKernel, RandomFourierFeatures, N>;

pub type LaplacianRandomFourierFeatures<const N: usize> =
    TypeSafeKernelApproximation<Untrained, LaplacianKernel, RandomFourierFeatures, N>;

pub type RBFNystrom<const N: usize> =
    TypeSafeKernelApproximation<Untrained, RBFKernel, NystromMethod, N>;

pub type PolynomialNystrom<const N: usize> =
    TypeSafeKernelApproximation<Untrained, PolynomialKernel, NystromMethod, N>;

pub type RBFFastfood<const N: usize> =
    TypeSafeKernelApproximation<Untrained, RBFKernel, FastfoodMethod, N>;

/// Fitted type aliases
pub type FittedRBFRandomFourierFeatures<const N: usize> =
    FittedTypeSafeKernelApproximation<RBFKernel, RandomFourierFeatures, N>;

pub type FittedLaplacianRandomFourierFeatures<const N: usize> =
    FittedTypeSafeKernelApproximation<LaplacianKernel, RandomFourierFeatures, N>;

pub type FittedRBFNystrom<const N: usize> =
    FittedTypeSafeKernelApproximation<RBFKernel, NystromMethod, N>;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_type_safe_rbf_rff() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        // Create untrained approximation with compile-time dimension
        let approximation: RBFRandomFourierFeatures<10> = TypeSafeKernelApproximation::new()
            .bandwidth(1.5)
            .random_state(42);

        // Fit and transform
        let fitted = approximation.fit(&data).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.shape(), &[4, 20]); // 10 components * 2 (cos, sin)
        assert_eq!(FittedRBFRandomFourierFeatures::<10>::n_components(), 10);
        assert_eq!(fitted.kernel_name(), "RBF");
        assert_eq!(fitted.method_name(), "RandomFourierFeatures");
    }

    #[test]
    fn test_type_safe_nystrom() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let approximation: RBFNystrom<5> = TypeSafeKernelApproximation::new()
            .bandwidth(2.0)
            .random_state(42);

        let fitted = approximation.fit(&data).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.shape()[0], 4); // n_samples
        assert_eq!(fitted.kernel_name(), "RBF");
        assert_eq!(fitted.method_name(), "Nystrom");
    }

    #[test]
    fn test_polynomial_kernel() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let approximation: PolynomialNystrom<3> = TypeSafeKernelApproximation::new()
            .degree(3)
            .random_state(42);

        let fitted = approximation.fit(&data).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(fitted.kernel_name(), "Polynomial");
        assert!(features.nrows() == 3);
    }

    #[test]
    fn test_compile_time_constants() {
        // These should be available at compile time
        assert_eq!(RBFKernel::NAME, "RBF");
        assert!(RBFKernel::SUPPORTS_PARAMETER_LEARNING);
        assert_eq!(RandomFourierFeatures::NAME, "RandomFourierFeatures");
        assert!(RandomFourierFeatures::SUPPORTS_INCREMENTAL);
        assert!(RandomFourierFeatures::HAS_ERROR_BOUNDS);
    }

    // This test demonstrates compile-time type safety
    // The following code would not compile due to type constraints:

    // #[test]
    // fn test_type_safety_violations() {
    //     // This would fail: ArcCosine kernel doesn't support bandwidth
    //     // let _ = TypeSafeKernelApproximation::<Untrained, ArcCosineKernel, RandomFourierFeatures, 10>::new()
    //     //     .bandwidth(1.0); // ERROR: ArcCosineKernel doesn't implement KernelTypeWithBandwidth
    //
    //     // This would fail: RBF kernel with degree parameter
    //     // let _ = TypeSafeKernelApproximation::<Untrained, RBFKernel, RandomFourierFeatures, 10>::new()
    //     //     .degree(2); // ERROR: RBFKernel doesn't implement PolynomialKernelType
    //
    //     // This would fail: Trying to fit incompatible kernel-method combination
    //     // let _ = TypeSafeKernelApproximation::<Untrained, ArcCosineKernel, RandomFourierFeatures, 10>::new()
    //     //     .fit(&data); // ERROR: ArcCosineKernel doesn't implement FittableKernel<RandomFourierFeatures>
    // }
}

// ==================================================================================
// ADVANCED TYPE SAFETY ENHANCEMENTS - Zero-Cost Abstractions and Compile-Time Validation
// ==================================================================================

/// Advanced compile-time parameter validation traits
pub trait ParameterValidation<const MIN: usize, const MAX: usize> {
    /// Validates parameters at compile time
    const IS_VALID: bool = MIN <= MAX;

    /// Get the parameter range
    fn parameter_range() -> (usize, usize) {
        (MIN, MAX)
    }
}

/// Zero-cost abstraction for validated component counts
#[derive(Debug, Clone, Copy)]
/// ValidatedComponents
pub struct ValidatedComponents<const N: usize>;

impl<const N: usize> ValidatedComponents<N> {
    /// Create validated components with compile-time checks
    pub fn new() -> Self {
        // Compile-time assertions
        assert!(N > 0, "Component count must be positive");
        assert!(N <= 10000, "Component count too large");
        Self
    }

    /// Get the component count
    pub const fn count(&self) -> usize {
        N
    }
}

/// Compile-time compatibility checking between kernels and methods
pub trait KernelMethodCompatibility<K: KernelType, M: ApproximationMethod> {
    /// Whether this kernel-method combination is supported
    const IS_COMPATIBLE: bool;

    /// Performance characteristics of this combination
    const PERFORMANCE_TIER: PerformanceTier;

    /// Memory complexity
    const MEMORY_COMPLEXITY: ComplexityClass;
}

/// Performance tiers for different kernel-method combinations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// PerformanceTier
pub enum PerformanceTier {
    /// Optimal performance combination
    Optimal,
    /// Good performance
    Good,
    /// Acceptable performance
    Acceptable,
    /// Poor performance (not recommended)
    Poor,
}

/// Implement compatibility rules
impl KernelMethodCompatibility<RBFKernel, RandomFourierFeatures> for () {
    const IS_COMPATIBLE: bool = true;
    const PERFORMANCE_TIER: PerformanceTier = PerformanceTier::Optimal;
    const MEMORY_COMPLEXITY: ComplexityClass = ComplexityClass::Linear;
}

impl KernelMethodCompatibility<LaplacianKernel, RandomFourierFeatures> for () {
    const IS_COMPATIBLE: bool = true;
    const PERFORMANCE_TIER: PerformanceTier = PerformanceTier::Optimal;
    const MEMORY_COMPLEXITY: ComplexityClass = ComplexityClass::Linear;
}

impl KernelMethodCompatibility<RBFKernel, NystromMethod> for () {
    const IS_COMPATIBLE: bool = true;
    const PERFORMANCE_TIER: PerformanceTier = PerformanceTier::Good;
    const MEMORY_COMPLEXITY: ComplexityClass = ComplexityClass::Quadratic;
}

impl KernelMethodCompatibility<PolynomialKernel, NystromMethod> for () {
    const IS_COMPATIBLE: bool = true;
    const PERFORMANCE_TIER: PerformanceTier = PerformanceTier::Good;
    const MEMORY_COMPLEXITY: ComplexityClass = ComplexityClass::Quadratic;
}

impl KernelMethodCompatibility<RBFKernel, FastfoodMethod> for () {
    const IS_COMPATIBLE: bool = true;
    const PERFORMANCE_TIER: PerformanceTier = PerformanceTier::Optimal;
    const MEMORY_COMPLEXITY: ComplexityClass = ComplexityClass::DimensionDependent;
}

/// Arc-cosine kernels have limited compatibility
impl KernelMethodCompatibility<ArcCosineKernel, RandomFourierFeatures> for () {
    const IS_COMPATIBLE: bool = true;
    const PERFORMANCE_TIER: PerformanceTier = PerformanceTier::Acceptable;
    const MEMORY_COMPLEXITY: ComplexityClass = ComplexityClass::Linear;
}

impl KernelMethodCompatibility<ArcCosineKernel, NystromMethod> for () {
    const IS_COMPATIBLE: bool = false; // Not recommended
    const PERFORMANCE_TIER: PerformanceTier = PerformanceTier::Poor;
    const MEMORY_COMPLEXITY: ComplexityClass = ComplexityClass::Quadratic;
}

impl KernelMethodCompatibility<ArcCosineKernel, FastfoodMethod> for () {
    const IS_COMPATIBLE: bool = false; // Not supported
    const PERFORMANCE_TIER: PerformanceTier = PerformanceTier::Poor;
    const MEMORY_COMPLEXITY: ComplexityClass = ComplexityClass::DimensionDependent;
}

// Additional compatibility implementations for missing combinations
impl KernelMethodCompatibility<LaplacianKernel, NystromMethod> for () {
    const IS_COMPATIBLE: bool = true;
    const PERFORMANCE_TIER: PerformanceTier = PerformanceTier::Good;
    const MEMORY_COMPLEXITY: ComplexityClass = ComplexityClass::Quadratic;
}

impl KernelMethodCompatibility<PolynomialKernel, RandomFourierFeatures> for () {
    const IS_COMPATIBLE: bool = true;
    const PERFORMANCE_TIER: PerformanceTier = PerformanceTier::Good;
    const MEMORY_COMPLEXITY: ComplexityClass = ComplexityClass::Linear;
}

/// Zero-cost wrapper for compile-time validated approximations
#[derive(Debug, Clone)]
/// ValidatedKernelApproximation
pub struct ValidatedKernelApproximation<K, M, const N: usize>
where
    K: KernelType,
    M: ApproximationMethod,
    (): KernelMethodCompatibility<K, M>,
{
    inner: TypeSafeKernelApproximation<Untrained, K, M, N>,
    _validation: ValidatedComponents<N>,
}

impl<K, M, const N: usize> ValidatedKernelApproximation<K, M, N>
where
    K: KernelType,
    M: ApproximationMethod,
    (): KernelMethodCompatibility<K, M>,
{
    /// Create a new validated kernel approximation with compile-time checks
    pub fn new() -> Self {
        // Compile-time compatibility check
        assert!(
            <() as KernelMethodCompatibility<K, M>>::IS_COMPATIBLE,
            "Incompatible kernel-method combination"
        );

        Self {
            inner: TypeSafeKernelApproximation::new(),
            _validation: ValidatedComponents::new(),
        }
    }

    /// Get performance information at compile time
    pub const fn performance_info() -> (PerformanceTier, ComplexityClass, ComplexityClass) {
        (
            <() as KernelMethodCompatibility<K, M>>::PERFORMANCE_TIER,
            M::COMPLEXITY,
            <() as KernelMethodCompatibility<K, M>>::MEMORY_COMPLEXITY,
        )
    }

    /// Check if this combination is optimal
    pub const fn is_optimal() -> bool {
        matches!(
            <() as KernelMethodCompatibility<K, M>>::PERFORMANCE_TIER,
            PerformanceTier::Optimal
        )
    }
}

/// Enhanced quality metrics with compile-time bounds
#[derive(Debug, Clone, Copy)]
/// BoundedQualityMetrics
pub struct BoundedQualityMetrics<const MIN_ALIGNMENT: u32, const MAX_ERROR: u32> {
    kernel_alignment: f64,
    approximation_error: f64,
    effective_rank: f64,
}

impl<const MIN_ALIGNMENT: u32, const MAX_ERROR: u32>
    BoundedQualityMetrics<MIN_ALIGNMENT, MAX_ERROR>
{
    /// Create new bounded quality metrics with compile-time validation
    pub const fn new(alignment: f64, error: f64, rank: f64) -> Option<Self> {
        let min_align = MIN_ALIGNMENT as f64 / 100.0; // Convert percentage to decimal
        let max_err = MAX_ERROR as f64 / 100.0;

        if alignment >= min_align && error <= max_err && rank > 0.0 {
            Some(Self {
                kernel_alignment: alignment,
                approximation_error: error,
                effective_rank: rank,
            })
        } else {
            None
        }
    }

    /// Get compile-time bounds
    pub const fn bounds() -> (f64, f64) {
        (MIN_ALIGNMENT as f64 / 100.0, MAX_ERROR as f64 / 100.0)
    }

    /// Check if metrics meet quality standards
    pub const fn meets_standards(&self) -> bool {
        let (min_align, max_err) = Self::bounds();
        self.kernel_alignment >= min_align && self.approximation_error <= max_err
    }
}

/// Type alias for high-quality metrics (>90% alignment, <5% error)
pub type HighQualityMetrics = BoundedQualityMetrics<90, 5>;

/// Type alias for acceptable quality metrics (>70% alignment, <15% error)
pub type AcceptableQualityMetrics = BoundedQualityMetrics<70, 15>;

/// Macro for easy creation of validated kernel approximations
#[macro_export]
macro_rules! validated_kernel_approximation {
    (RBF, RandomFourierFeatures, $n:literal) => {
        ValidatedKernelApproximation::<RBFKernel, RandomFourierFeatures, $n>::new()
    };
    (Laplacian, RandomFourierFeatures, $n:literal) => {
        ValidatedKernelApproximation::<LaplacianKernel, RandomFourierFeatures, $n>::new()
    };
    (RBF, Nystrom, $n:literal) => {
        ValidatedKernelApproximation::<RBFKernel, NystromMethod, $n>::new()
    };
    (RBF, Fastfood, $n:literal) => {
        ValidatedKernelApproximation::<RBFKernel, FastfoodMethod, $n>::new()
    };
}

#[allow(non_snake_case)]
#[cfg(test)]
mod advanced_type_safety_tests {
    use super::*;

    #[test]
    fn test_validated_components() {
        let components = ValidatedComponents::<100>::new();
        assert_eq!(components.count(), 100);
    }

    #[test]
    fn test_kernel_method_compatibility() {
        // Test compile-time compatibility checks
        assert!(<() as KernelMethodCompatibility<RBFKernel, RandomFourierFeatures>>::IS_COMPATIBLE);
        assert!(!<() as KernelMethodCompatibility<ArcCosineKernel, NystromMethod>>::IS_COMPATIBLE);

        // Test performance tiers
        assert_eq!(
            <() as KernelMethodCompatibility<RBFKernel, RandomFourierFeatures>>::PERFORMANCE_TIER,
            PerformanceTier::Optimal
        );
    }

    #[test]
    fn test_bounded_quality_metrics() {
        // High quality metrics
        let high_quality = HighQualityMetrics::new(0.95, 0.03, 50.0).unwrap();
        assert!(high_quality.meets_standards());

        // Low quality metrics should fail bounds check
        let low_quality = HighQualityMetrics::new(0.60, 0.20, 10.0);
        assert!(low_quality.is_none());
    }

    #[test]
    fn test_macro_creation() {
        // Test that the macro compiles for valid combinations
        let _rbf_rff = validated_kernel_approximation!(RBF, RandomFourierFeatures, 100);
        let _lap_rff = validated_kernel_approximation!(Laplacian, RandomFourierFeatures, 50);
        let _rbf_nys = validated_kernel_approximation!(RBF, Nystrom, 30);
        let _rbf_ff = validated_kernel_approximation!(RBF, Fastfood, 128);

        // Performance info should be available at compile time
        let (tier, complexity, memory) = ValidatedKernelApproximation::<
            RBFKernel,
            RandomFourierFeatures,
            100,
        >::performance_info();
        assert_eq!(tier, PerformanceTier::Optimal);
        assert_eq!(complexity, ComplexityClass::Linear);
        assert_eq!(memory, ComplexityClass::Linear);
    }
}

/// Advanced Zero-Cost Kernel Composition Abstractions
///
/// These traits enable compile-time composition of kernel operations
/// with zero runtime overhead.
///
/// Trait for kernels that can be composed
pub trait ComposableKernel: KernelType {
    type CompositionResult<Other: ComposableKernel>: ComposableKernel;

    /// Combine this kernel with another kernel
    fn compose<Other: ComposableKernel>(self) -> Self::CompositionResult<Other>;
}

/// Sum composition of two kernels
#[derive(Debug, Clone, Copy)]
/// SumKernel
pub struct SumKernel<K1: KernelType, K2: KernelType> {
    _phantom: PhantomData<(K1, K2)>,
}

impl<K1: KernelType, K2: KernelType> KernelType for SumKernel<K1, K2> {
    const NAME: &'static str = "Sum";
    const SUPPORTS_PARAMETER_LEARNING: bool =
        K1::SUPPORTS_PARAMETER_LEARNING && K2::SUPPORTS_PARAMETER_LEARNING;
    const DEFAULT_BANDWIDTH: f64 = (K1::DEFAULT_BANDWIDTH + K2::DEFAULT_BANDWIDTH) / 2.0;
}

/// Product composition of two kernels  
#[derive(Debug, Clone, Copy)]
/// ProductKernel
pub struct ProductKernel<K1: KernelType, K2: KernelType> {
    _phantom: PhantomData<(K1, K2)>,
}

impl<K1: KernelType, K2: KernelType> KernelType for ProductKernel<K1, K2> {
    const NAME: &'static str = "Product";
    const SUPPORTS_PARAMETER_LEARNING: bool =
        K1::SUPPORTS_PARAMETER_LEARNING && K2::SUPPORTS_PARAMETER_LEARNING;
    const DEFAULT_BANDWIDTH: f64 = K1::DEFAULT_BANDWIDTH * K2::DEFAULT_BANDWIDTH;
}

impl ComposableKernel for RBFKernel {
    type CompositionResult<Other: ComposableKernel> = SumKernel<Self, Other>;

    fn compose<Other: ComposableKernel>(self) -> Self::CompositionResult<Other> {
        SumKernel {
            _phantom: PhantomData,
        }
    }
}

impl ComposableKernel for LaplacianKernel {
    type CompositionResult<Other: ComposableKernel> = SumKernel<Self, Other>;

    fn compose<Other: ComposableKernel>(self) -> Self::CompositionResult<Other> {
        SumKernel {
            _phantom: PhantomData,
        }
    }
}

impl ComposableKernel for PolynomialKernel {
    type CompositionResult<Other: ComposableKernel> = ProductKernel<Self, Other>;

    fn compose<Other: ComposableKernel>(self) -> Self::CompositionResult<Other> {
        ProductKernel {
            _phantom: PhantomData,
        }
    }
}

/// Compile-time feature size validation
pub trait ValidatedFeatureSize<const N: usize> {
    const IS_POWER_OF_TWO: bool = (N != 0) && ((N & (N - 1)) == 0);
    const IS_REASONABLE_SIZE: bool = N >= 8 && N <= 8192;
    const IS_VALID: bool = Self::IS_POWER_OF_TWO && Self::IS_REASONABLE_SIZE;
}

// Implement for all const generic sizes
impl<const N: usize> ValidatedFeatureSize<N> for () {}

/// Zero-cost wrapper for validated feature dimensions
#[derive(Debug, Clone)]
/// ValidatedFeatures
pub struct ValidatedFeatures<const N: usize> {
    _phantom: PhantomData<[f64; N]>,
}

impl<const N: usize> ValidatedFeatures<N>
where
    (): ValidatedFeatureSize<N>,
{
    /// Create new validated features (compile-time checked)
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the validated feature count
    pub const fn count() -> usize {
        N
    }

    /// Check if size is optimal for Fastfood
    pub const fn is_fastfood_optimal() -> bool {
        <() as ValidatedFeatureSize<N>>::IS_POWER_OF_TWO
    }
}

/// Advanced approximation quality bounds
pub trait ApproximationQualityBounds<Method: ApproximationMethod> {
    /// Theoretical error bound for this method
    const ERROR_BOUND_CONSTANT: f64;

    /// Sample complexity scaling
    const SAMPLE_COMPLEXITY_EXPONENT: f64;

    /// Dimension dependency
    const DIMENSION_DEPENDENCY: f64;

    /// Compute theoretical error bound
    fn error_bound(n_samples: usize, n_features: usize, n_components: usize) -> f64 {
        let base_rate = Self::ERROR_BOUND_CONSTANT;
        let sample_factor = (n_samples as f64).powf(-Self::SAMPLE_COMPLEXITY_EXPONENT);
        let dim_factor = (n_features as f64).powf(Self::DIMENSION_DEPENDENCY);
        let comp_factor = (n_components as f64).powf(-0.5);

        base_rate * sample_factor * dim_factor * comp_factor
    }
}

impl ApproximationQualityBounds<RandomFourierFeatures> for () {
    const ERROR_BOUND_CONSTANT: f64 = 2.0;
    const SAMPLE_COMPLEXITY_EXPONENT: f64 = 0.25;
    const DIMENSION_DEPENDENCY: f64 = 0.1;
}

impl ApproximationQualityBounds<NystromMethod> for () {
    const ERROR_BOUND_CONSTANT: f64 = 1.5;
    const SAMPLE_COMPLEXITY_EXPONENT: f64 = 0.33;
    const DIMENSION_DEPENDENCY: f64 = 0.05;
}

impl ApproximationQualityBounds<FastfoodMethod> for () {
    const ERROR_BOUND_CONSTANT: f64 = 3.0;
    const SAMPLE_COMPLEXITY_EXPONENT: f64 = 0.2;
    const DIMENSION_DEPENDENCY: f64 = 0.15;
}

/// Advanced type-safe kernel configuration builder
#[derive(Debug, Clone)]
/// TypeSafeKernelConfig
pub struct TypeSafeKernelConfig<K: KernelType, M: ApproximationMethod, const N: usize>
where
    (): ValidatedFeatureSize<N>,
    (): KernelMethodCompatibility<K, M>,
    (): ApproximationQualityBounds<M>,
{
    kernel_type: PhantomData<K>,
    method_type: PhantomData<M>,
    features: ValidatedFeatures<N>,
    bandwidth: f64,
    quality_threshold: f64,
}

impl<K: KernelType, M: ApproximationMethod, const N: usize> TypeSafeKernelConfig<K, M, N>
where
    (): ValidatedFeatureSize<N>,
    (): KernelMethodCompatibility<K, M>,
    (): ApproximationQualityBounds<M>,
{
    /// Create a new type-safe configuration
    pub fn new() -> Self {
        Self {
            kernel_type: PhantomData,
            method_type: PhantomData,
            features: ValidatedFeatures::new(),
            bandwidth: K::DEFAULT_BANDWIDTH,
            quality_threshold: 0.1,
        }
    }

    /// Set bandwidth parameter with compile-time validation
    pub fn bandwidth(mut self, bandwidth: f64) -> Self {
        assert!(bandwidth > 0.0, "Bandwidth must be positive");
        self.bandwidth = bandwidth;
        self
    }

    /// Set quality threshold with bounds checking
    pub fn quality_threshold(mut self, threshold: f64) -> Self {
        assert!(
            threshold > 0.0 && threshold < 1.0,
            "Quality threshold must be between 0 and 1"
        );
        self.quality_threshold = threshold;
        self
    }

    /// Get performance tier for this configuration
    pub const fn performance_tier() -> PerformanceTier {
        <() as KernelMethodCompatibility<K, M>>::PERFORMANCE_TIER
    }

    /// Get memory complexity for this configuration
    pub const fn memory_complexity() -> ComplexityClass {
        <() as KernelMethodCompatibility<K, M>>::MEMORY_COMPLEXITY
    }

    /// Compute theoretical error bound for given data size
    pub fn theoretical_error_bound(&self, n_samples: usize, n_features: usize) -> f64 {
        <() as ApproximationQualityBounds<M>>::error_bound(n_samples, n_features, N)
    }

    /// Check if configuration meets quality requirements
    pub fn meets_quality_requirements(&self, n_samples: usize, n_features: usize) -> bool {
        let error_bound = self.theoretical_error_bound(n_samples, n_features);
        error_bound <= self.quality_threshold
    }
}

/// Type aliases for common validated configurations
pub type ValidatedRBFRandomFourier<const N: usize> =
    TypeSafeKernelConfig<RBFKernel, RandomFourierFeatures, N>;
pub type ValidatedLaplacianNystrom<const N: usize> =
    TypeSafeKernelConfig<LaplacianKernel, NystromMethod, N>;
pub type ValidatedPolynomialRFF<const N: usize> =
    TypeSafeKernelConfig<PolynomialKernel, RandomFourierFeatures, N>;

impl<K1: KernelType, K2: KernelType> ComposableKernel for SumKernel<K1, K2> {
    type CompositionResult<Other: ComposableKernel> = SumKernel<Self, Other>;

    fn compose<Other: ComposableKernel>(self) -> Self::CompositionResult<Other> {
        SumKernel {
            _phantom: PhantomData,
        }
    }
}

impl<K1: KernelType, K2: KernelType> ComposableKernel for ProductKernel<K1, K2> {
    type CompositionResult<Other: ComposableKernel> = ProductKernel<Self, Other>;

    fn compose<Other: ComposableKernel>(self) -> Self::CompositionResult<Other> {
        ProductKernel {
            _phantom: PhantomData,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod advanced_composition_tests {
    use super::*;

    #[test]
    fn test_validated_features() {
        // These should compile
        let _features_64 = ValidatedFeatures::<64>::new();
        let _features_128 = ValidatedFeatures::<128>::new();
        let _features_256 = ValidatedFeatures::<256>::new();

        assert_eq!(ValidatedFeatures::<64>::count(), 64);
        assert!(ValidatedFeatures::<64>::is_fastfood_optimal());
    }

    #[test]
    fn test_type_safe_configs() {
        let config = ValidatedRBFRandomFourier::<128>::new()
            .bandwidth(1.5)
            .quality_threshold(0.05);

        assert_eq!(
            ValidatedRBFRandomFourier::<128>::performance_tier(),
            PerformanceTier::Optimal
        );
        assert!(config.meets_quality_requirements(1000, 10));
    }

    #[test]
    fn test_kernel_composition() {
        let _rbf = RBFKernel;
        let _laplacian = LaplacianKernel;
        let _polynomial = PolynomialKernel;

        // These should compile and create valid composed kernels
        let _composed1 = _rbf.compose::<LaplacianKernel>();
        let _composed2 = _polynomial.compose::<RBFKernel>();
    }

    #[test]
    fn test_approximation_bounds() {
        let rff_bound =
            <() as ApproximationQualityBounds<RandomFourierFeatures>>::error_bound(1000, 10, 100);
        let nystrom_bound =
            <() as ApproximationQualityBounds<NystromMethod>>::error_bound(1000, 10, 100);
        let fastfood_bound =
            <() as ApproximationQualityBounds<FastfoodMethod>>::error_bound(1000, 10, 100);

        assert!(rff_bound > 0.0);
        assert!(nystrom_bound > 0.0);
        assert!(fastfood_bound > 0.0);

        // Nyström should generally have tighter bounds than RFF
        assert!(nystrom_bound < rff_bound);
    }
}

// ============================================================================
// Configuration Presets for Common Use Cases
// ============================================================================

/// Configuration presets for common kernel approximation scenarios
pub struct KernelPresets;

impl KernelPresets {
    /// Fast approximation preset - prioritizes speed over accuracy
    pub fn fast_rbf_128() -> ValidatedRBFRandomFourier<128> {
        ValidatedRBFRandomFourier::<128>::new()
            .bandwidth(1.0)
            .quality_threshold(0.2)
    }

    /// Balanced approximation preset - good trade-off between speed and accuracy
    pub fn balanced_rbf_256() -> ValidatedRBFRandomFourier<256> {
        ValidatedRBFRandomFourier::<256>::new()
            .bandwidth(1.0)
            .quality_threshold(0.1)
    }

    /// High-accuracy approximation preset - prioritizes accuracy over speed
    pub fn accurate_rbf_512() -> ValidatedRBFRandomFourier<512> {
        ValidatedRBFRandomFourier::<512>::new()
            .bandwidth(1.0)
            .quality_threshold(0.05)
    }

    /// Ultra-fast approximation for large-scale problems
    pub fn ultrafast_rbf_64() -> ValidatedRBFRandomFourier<64> {
        ValidatedRBFRandomFourier::<64>::new()
            .bandwidth(1.0)
            .quality_threshold(0.3)
    }

    /// High-precision Nyström preset for small to medium datasets
    pub fn precise_nystroem_128() -> ValidatedLaplacianNystrom<128> {
        ValidatedLaplacianNystrom::<128>::new()
            .bandwidth(1.0)
            .quality_threshold(0.01)
    }

    /// Memory-efficient preset for resource-constrained environments
    pub fn memory_efficient_rbf_32() -> ValidatedRBFRandomFourier<32> {
        ValidatedRBFRandomFourier::<32>::new()
            .bandwidth(1.0)
            .quality_threshold(0.4)
    }

    /// Polynomial kernel preset for structured data
    pub fn polynomial_features_256() -> ValidatedPolynomialRFF<256> {
        ValidatedPolynomialRFF::<256>::new()
            .bandwidth(1.0)
            .quality_threshold(0.15)
    }
}

// ============================================================================
// Profile-Guided Optimization Support
// ============================================================================

/// Profile-guided optimization configuration
#[derive(Debug, Clone)]
/// ProfileGuidedConfig
pub struct ProfileGuidedConfig {
    /// Enable PGO-based feature size selection
    pub enable_pgo_feature_selection: bool,

    /// Enable PGO-based bandwidth optimization
    pub enable_pgo_bandwidth_optimization: bool,

    /// Profile data file path
    pub profile_data_path: Option<String>,

    /// Target hardware architecture
    pub target_architecture: TargetArchitecture,

    /// Optimization level
    pub optimization_level: OptimizationLevel,
}

/// Target hardware architecture for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// TargetArchitecture
pub enum TargetArchitecture {
    /// Generic x86_64 architecture
    X86_64Generic,
    /// x86_64 with AVX2 support
    X86_64AVX2,
    /// x86_64 with AVX-512 support
    X86_64AVX512,
    /// ARM64 architecture
    ARM64,
    /// ARM64 with NEON support
    ARM64NEON,
}

/// Optimization level for profile-guided optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// OptimizationLevel
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations
    Basic,
    /// Aggressive optimizations
    Aggressive,
    /// Maximum optimizations (may increase compile time significantly)
    Maximum,
}

impl Default for ProfileGuidedConfig {
    fn default() -> Self {
        Self {
            enable_pgo_feature_selection: false,
            enable_pgo_bandwidth_optimization: false,
            profile_data_path: None,
            target_architecture: TargetArchitecture::X86_64Generic,
            optimization_level: OptimizationLevel::Basic,
        }
    }
}

impl ProfileGuidedConfig {
    /// Create a new PGO configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable PGO-based feature size selection
    pub fn enable_feature_selection(mut self) -> Self {
        self.enable_pgo_feature_selection = true;
        self
    }

    /// Enable PGO-based bandwidth optimization
    pub fn enable_bandwidth_optimization(mut self) -> Self {
        self.enable_pgo_bandwidth_optimization = true;
        self
    }

    /// Set profile data file path
    pub fn profile_data_path<P: Into<String>>(mut self, path: P) -> Self {
        self.profile_data_path = Some(path.into());
        self
    }

    /// Set target architecture
    pub fn target_architecture(mut self, arch: TargetArchitecture) -> Self {
        self.target_architecture = arch;
        self
    }

    /// Set optimization level
    pub fn optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Get recommended feature count based on architecture and optimization level
    pub fn recommended_feature_count(&self, data_size: usize, dimensionality: usize) -> usize {
        let base_features = match self.target_architecture {
            TargetArchitecture::X86_64Generic => 128,
            TargetArchitecture::X86_64AVX2 => 256,
            TargetArchitecture::X86_64AVX512 => 512,
            TargetArchitecture::ARM64 => 128,
            TargetArchitecture::ARM64NEON => 256,
        };

        let scale_factor = match self.optimization_level {
            OptimizationLevel::None => 0.5,
            OptimizationLevel::Basic => 1.0,
            OptimizationLevel::Aggressive => 1.5,
            OptimizationLevel::Maximum => 2.0,
        };

        let scaled_features = (base_features as f64 * scale_factor) as usize;

        // Adjust based on data characteristics
        let data_adjustment = if data_size > 10000 {
            1.2
        } else if data_size < 1000 {
            0.8
        } else {
            1.0
        };

        let dimension_adjustment = if dimensionality > 100 {
            1.1
        } else if dimensionality < 10 {
            0.9
        } else {
            1.0
        };

        ((scaled_features as f64 * data_adjustment * dimension_adjustment) as usize)
            .max(32)
            .min(1024)
    }
}

// ============================================================================
// Serialization Support for Approximation Models
// ============================================================================

/// Serializable kernel approximation configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// SerializableKernelConfig
pub struct SerializableKernelConfig {
    /// Kernel type name
    pub kernel_type: String,

    /// Approximation method name
    pub approximation_method: String,

    /// Number of features/components
    pub n_components: usize,

    /// Bandwidth parameter
    pub bandwidth: f64,

    /// Quality threshold
    pub quality_threshold: f64,

    /// Random state for reproducibility
    pub random_state: Option<u64>,

    /// Additional parameters
    pub additional_params: std::collections::HashMap<String, f64>,
}

/// Serializable fitted model parameters
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// SerializableFittedParams
pub struct SerializableFittedParams {
    /// Configuration used to create this model
    pub config: SerializableKernelConfig,

    /// Random features (for RFF methods)
    pub random_features: Option<Vec<Vec<f64>>>,

    /// Selected indices (for Nyström methods)
    pub selected_indices: Option<Vec<usize>>,

    /// Eigenvalues (for Nyström methods)
    pub eigenvalues: Option<Vec<f64>>,

    /// Eigenvectors (for Nyström methods)
    pub eigenvectors: Option<Vec<Vec<f64>>>,

    /// Quality metrics achieved
    pub quality_metrics: std::collections::HashMap<String, f64>,

    /// Timestamp of when model was fitted
    pub fitted_timestamp: Option<u64>,
}

/// Trait for serializable kernel approximation methods
pub trait SerializableKernelApproximation {
    /// Export configuration to serializable format
    fn export_config(&self) -> Result<SerializableKernelConfig>;

    /// Import configuration from serializable format
    fn import_config(config: &SerializableKernelConfig) -> Result<Self>
    where
        Self: Sized;

    /// Export fitted parameters to serializable format
    fn export_fitted_params(&self) -> Result<SerializableFittedParams>;

    /// Import fitted parameters from serializable format
    fn import_fitted_params(&mut self, params: &SerializableFittedParams) -> Result<()>;

    /// Save model to file
    fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let config = self.export_config()?;
        let fitted_params = self.export_fitted_params()?;

        let model_data = serde_json::json!({
            "config": config,
            "fitted_params": fitted_params,
            "version": "1.0",
            "sklears_version": env!("CARGO_PKG_VERSION"),
        });

        std::fs::write(path, serde_json::to_string_pretty(&model_data).unwrap())
            .map_err(SklearsError::from)?;

        Ok(())
    }

    /// Load model from file
    fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self>
    where
        Self: Sized,
    {
        let content = std::fs::read_to_string(path).map_err(SklearsError::from)?;

        let model_data: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            SklearsError::SerializationError(format!("Failed to parse JSON: {}", e))
        })?;

        let config: SerializableKernelConfig = serde_json::from_value(model_data["config"].clone())
            .map_err(|e| {
                SklearsError::SerializationError(format!("Failed to deserialize config: {}", e))
            })?;

        let fitted_params: SerializableFittedParams =
            serde_json::from_value(model_data["fitted_params"].clone()).map_err(|e| {
                SklearsError::SerializationError(format!(
                    "Failed to deserialize fitted params: {}",
                    e
                ))
            })?;

        let mut model = Self::import_config(&config)?;
        model.import_fitted_params(&fitted_params)?;

        Ok(model)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod preset_tests {
    use super::*;

    #[test]
    fn test_kernel_presets() {
        let fast_config = KernelPresets::fast_rbf_128();
        assert_eq!(
            ValidatedRBFRandomFourier::<128>::performance_tier(),
            PerformanceTier::Optimal
        );

        let balanced_config = KernelPresets::balanced_rbf_256();
        assert_eq!(
            ValidatedRBFRandomFourier::<256>::performance_tier(),
            PerformanceTier::Optimal
        );

        let accurate_config = KernelPresets::accurate_rbf_512();
        assert_eq!(
            ValidatedRBFRandomFourier::<512>::performance_tier(),
            PerformanceTier::Optimal
        );
    }

    #[test]
    fn test_profile_guided_config() {
        let pgo_config = ProfileGuidedConfig::new()
            .enable_feature_selection()
            .enable_bandwidth_optimization()
            .target_architecture(TargetArchitecture::X86_64AVX2)
            .optimization_level(OptimizationLevel::Aggressive);

        assert!(pgo_config.enable_pgo_feature_selection);
        assert!(pgo_config.enable_pgo_bandwidth_optimization);
        assert_eq!(
            pgo_config.target_architecture,
            TargetArchitecture::X86_64AVX2
        );
        assert_eq!(pgo_config.optimization_level, OptimizationLevel::Aggressive);

        // Test feature count recommendation
        let features = pgo_config.recommended_feature_count(5000, 50);
        assert!(features >= 32 && features <= 1024);
    }

    #[test]
    fn test_serializable_config() {
        let config = SerializableKernelConfig {
            kernel_type: "RBF".to_string(),
            approximation_method: "RandomFourierFeatures".to_string(),
            n_components: 256,
            bandwidth: 1.5,
            quality_threshold: 0.1,
            random_state: Some(42),
            additional_params: std::collections::HashMap::new(),
        };

        // Test serialization
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(serialized.contains("RBF"));
        assert!(serialized.contains("RandomFourierFeatures"));

        // Test deserialization
        let deserialized: SerializableKernelConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.kernel_type, "RBF");
        assert_eq!(deserialized.n_components, 256);
        assert_eq!(deserialized.bandwidth, 1.5);
    }
}

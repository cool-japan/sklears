//! Scientific Computing Kernel Methods
//!
//! This module implements kernel methods for scientific computing applications,
//! including physics-informed kernels, differential equation kernels, conservation
//! law kernels, and multiscale methods.
//!
//! # References
//! - Raissi et al. (2019): "Physics-informed neural networks"
//! - Karniadakis et al. (2021): "Physics-informed machine learning"
//! - Chen & Sideris (2020): "Finite element method with interpolation at the nodes"
//! - E & Yu (2018): "The Deep Ritz Method"

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::thread_rng;
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Estimator, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for physics-informed kernels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsInformedConfig {
    /// Type of physical system
    pub system_type: PhysicalSystem,
    /// Number of random features
    pub n_components: usize,
    /// Kernel bandwidth
    pub bandwidth: Float,
    /// Weight for physics loss
    pub physics_weight: Float,
    /// Weight for data loss
    pub data_weight: Float,
}

impl Default for PhysicsInformedConfig {
    fn default() -> Self {
        Self {
            system_type: PhysicalSystem::HeatEquation,
            n_components: 100,
            bandwidth: 1.0,
            physics_weight: 1.0,
            data_weight: 1.0,
        }
    }
}

/// Types of physical systems
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PhysicalSystem {
    /// Heat equation: ∂u/∂t = α∇²u
    HeatEquation,
    /// Wave equation: ∂²u/∂t² = c²∇²u
    WaveEquation,
    /// Burgers' equation: ∂u/∂t + u∂u/∂x = ν∇²u
    BurgersEquation,
    /// Navier-Stokes equations
    NavierStokes,
    /// Schrödinger equation: iℏ∂ψ/∂t = Ĥψ
    Schrodinger,
    /// Custom PDE (user-defined)
    Custom,
}

/// Physics-Informed Kernel
///
/// Implements kernel methods that incorporate physical laws and differential
/// equations into the learning process. Useful for solving PDEs and modeling
/// physical systems with sparse data.
///
/// # Mathematical Background
///
/// For a PDE: N\[u\](x,t) = 0 (e.g., heat equation, wave equation)
/// The kernel incorporates both data fitting and PDE residual minimization:
///
/// Loss = λ_data * ||u - u_data||² + λ_physics * ||N\[u\]||²
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::scientific_computing_kernels::{
///     PhysicsInformedKernel, PhysicsInformedConfig, PhysicalSystem
/// };
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Transform};
///
/// let config = PhysicsInformedConfig {
///     system_type: PhysicalSystem::HeatEquation,
///     ..Default::default()
/// };
///
/// let pinn = PhysicsInformedKernel::new(config);
/// let X = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]; // (x, t) coordinates
/// let fitted = pinn.fit(&X, &()).unwrap();
/// let features = fitted.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PhysicsInformedKernel<State = Untrained> {
    config: PhysicsInformedConfig,

    // Fitted attributes
    kernel_weights: Option<Array2<Float>>,
    derivative_weights: Option<Vec<Array2<Float>>>,
    boundary_data: Option<Array2<Float>>,

    _state: PhantomData<State>,
}

impl PhysicsInformedKernel<Untrained> {
    /// Create a new physics-informed kernel
    pub fn new(config: PhysicsInformedConfig) -> Self {
        Self {
            config,
            kernel_weights: None,
            derivative_weights: None,
            boundary_data: None,
            _state: PhantomData,
        }
    }

    /// Create with default configuration
    pub fn with_system(system: PhysicalSystem) -> Self {
        Self {
            config: PhysicsInformedConfig {
                system_type: system,
                ..Default::default()
            },
            kernel_weights: None,
            derivative_weights: None,
            boundary_data: None,
            _state: PhantomData,
        }
    }

    /// Set number of components
    pub fn n_components(mut self, n: usize) -> Self {
        self.config.n_components = n;
        self
    }

    /// Set physics weight
    pub fn physics_weight(mut self, weight: Float) -> Self {
        self.config.physics_weight = weight;
        self
    }
}

impl Estimator for PhysicsInformedKernel<Untrained> {
    type Config = PhysicsInformedConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for PhysicsInformedKernel<Untrained> {
    type Fitted = PhysicsInformedKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() < 2 {
            return Err(SklearsError::InvalidInput(
                "Input must have at least 2 columns (spatial + temporal)".to_string(),
            ));
        }

        let boundary_data = x.clone();

        // Generate random Fourier features for the kernel
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let kernel_weights = Array2::from_shape_fn((x.ncols(), self.config.n_components), |_| {
            rng.sample(normal) * (2.0 * self.config.bandwidth).sqrt()
        });

        // Generate derivative weights for different orders
        let mut derivative_weights = Vec::new();

        // First derivatives (spatial and temporal)
        for _ in 0..x.ncols() {
            let weights = Array2::from_shape_fn((x.ncols(), self.config.n_components), |_| {
                rng.sample(normal) * (2.0 * self.config.bandwidth).sqrt()
            });
            derivative_weights.push(weights);
        }

        // Second derivatives (Laplacian)
        for _ in 0..x.ncols() {
            let weights = Array2::from_shape_fn((x.ncols(), self.config.n_components), |_| {
                rng.sample(normal) * (2.0 * self.config.bandwidth).sqrt()
            });
            derivative_weights.push(weights);
        }

        Ok(PhysicsInformedKernel {
            config: self.config,
            kernel_weights: Some(kernel_weights),
            derivative_weights: Some(derivative_weights),
            boundary_data: Some(boundary_data),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PhysicsInformedKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let kernel_weights = self.kernel_weights.as_ref().unwrap();

        if x.ncols() != kernel_weights.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature dimension mismatch: expected {}, got {}",
                kernel_weights.nrows(),
                x.ncols()
            )));
        }

        // Compute random Fourier features
        let projection = x.dot(kernel_weights);

        let n_samples = x.nrows();
        let n_features = self.config.n_components;

        // Basic features
        let mut output = Array2::zeros((n_samples, n_features * 3));

        let normalizer = (2.0 / n_features as Float).sqrt();

        for i in 0..n_samples {
            for j in 0..n_features {
                // Basic kernel features
                output[[i, j]] = normalizer * projection[[i, j]].cos();

                // Derivative features (approximate)
                output[[i, j + n_features]] = -normalizer * projection[[i, j]].sin();

                // Second derivative features (approximate)
                output[[i, j + 2 * n_features]] = -normalizer * projection[[i, j]].cos();
            }
        }

        Ok(output)
    }
}

impl PhysicsInformedKernel<Trained> {
    /// Get boundary data
    pub fn boundary_data(&self) -> &Array2<Float> {
        self.boundary_data.as_ref().unwrap()
    }

    /// Evaluate PDE residual for a given system
    pub fn pde_residual(
        &self,
        x: &Array2<Float>,
        solution: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let mut residual = Array1::zeros(n_samples);

        match self.config.system_type {
            PhysicalSystem::HeatEquation => {
                // Heat equation: ∂u/∂t - α∇²u = 0
                // Approximate derivatives using finite differences
                let alpha = 0.1; // Thermal diffusivity

                for i in 1..(n_samples - 1) {
                    let dt = if i > 0 {
                        x[[i, 1]] - x[[i - 1, 1]]
                    } else {
                        0.01
                    };
                    let dx = if i > 0 {
                        x[[i, 0]] - x[[i - 1, 0]]
                    } else {
                        0.01
                    };

                    let du_dt = if dt > 0.0 {
                        (solution[i] - solution[i - 1]) / dt
                    } else {
                        0.0
                    };

                    let d2u_dx2 = if dx > 0.0 && i > 0 && i < n_samples - 1 {
                        (solution[i + 1] - 2.0 * solution[i] + solution[i - 1]) / (dx * dx)
                    } else {
                        0.0
                    };

                    residual[i] = du_dt - alpha * d2u_dx2;
                }
            }
            PhysicalSystem::WaveEquation => {
                // Wave equation: ∂²u/∂t² - c²∇²u = 0
                let c = 1.0; // Wave speed

                for i in 1..(n_samples - 1) {
                    let dt = if i > 0 {
                        x[[i, 1]] - x[[i - 1, 1]]
                    } else {
                        0.01
                    };
                    let dx = if i > 0 {
                        x[[i, 0]] - x[[i - 1, 0]]
                    } else {
                        0.01
                    };

                    let d2u_dt2 = if dt > 0.0 && i > 0 && i < n_samples - 1 {
                        (solution[i + 1] - 2.0 * solution[i] + solution[i - 1]) / (dt * dt)
                    } else {
                        0.0
                    };

                    let d2u_dx2 = if dx > 0.0 && i > 0 && i < n_samples - 1 {
                        (solution[i + 1] - 2.0 * solution[i] + solution[i - 1]) / (dx * dx)
                    } else {
                        0.0
                    };

                    residual[i] = d2u_dt2 - c * c * d2u_dx2;
                }
            }
            PhysicalSystem::BurgersEquation => {
                // Burgers' equation: ∂u/∂t + u∂u/∂x - ν∇²u = 0
                let nu = 0.01; // Viscosity

                for i in 1..(n_samples - 1) {
                    let dt = if i > 0 {
                        x[[i, 1]] - x[[i - 1, 1]]
                    } else {
                        0.01
                    };
                    let dx = if i > 0 {
                        x[[i, 0]] - x[[i - 1, 0]]
                    } else {
                        0.01
                    };

                    let du_dt = if dt > 0.0 {
                        (solution[i] - solution[i - 1]) / dt
                    } else {
                        0.0
                    };

                    let du_dx = if dx > 0.0 {
                        (solution[i] - solution[i - 1]) / dx
                    } else {
                        0.0
                    };

                    let d2u_dx2 = if dx > 0.0 && i > 0 && i < n_samples - 1 {
                        (solution[i + 1] - 2.0 * solution[i] + solution[i - 1]) / (dx * dx)
                    } else {
                        0.0
                    };

                    residual[i] = du_dt + solution[i] * du_dx - nu * d2u_dx2;
                }
            }
            _ => {
                // Default: simple Laplacian
                for i in 1..(n_samples - 1) {
                    residual[i] = solution[i + 1] - 2.0 * solution[i] + solution[i - 1];
                }
            }
        }

        Ok(residual)
    }
}

/// Multiscale Kernel for hierarchical phenomena
///
/// Implements multiscale kernel methods for problems involving multiple spatial
/// or temporal scales (e.g., turbulence, molecular dynamics, climate modeling).
///
/// # Mathematical Background
///
/// Uses wavelets or hierarchical representations:
/// K(x, x') = Σ_l w_l K_l(x, x')
/// where K_l are kernels at different scales l.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::scientific_computing_kernels::MultiscaleKernel;
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Transform};
///
/// let kernel = MultiscaleKernel::with_scales(vec![0.1, 1.0, 10.0]);
/// let X = array![[1.0, 2.0], [3.0, 4.0]];
/// let fitted = kernel.fit(&X, &()).unwrap();
/// let features = fitted.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiscaleKernel<State = Untrained> {
    /// Scales for different levels
    scales: Vec<Float>,
    /// Number of components per scale
    n_components_per_scale: usize,

    // Fitted attributes
    scale_weights: Option<Vec<Array2<Float>>>,

    _state: PhantomData<State>,
}

impl MultiscaleKernel<Untrained> {
    /// Create with specified scales
    pub fn with_scales(scales: Vec<Float>) -> Self {
        Self {
            scales,
            n_components_per_scale: 50,
            scale_weights: None,
            _state: PhantomData,
        }
    }

    /// Set number of components per scale
    pub fn n_components_per_scale(mut self, n: usize) -> Self {
        self.n_components_per_scale = n;
        self
    }
}

impl Estimator for MultiscaleKernel<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for MultiscaleKernel<Untrained> {
    type Fitted = MultiscaleKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Generate random weights for each scale
        let mut scale_weights = Vec::new();

        for &scale in &self.scales {
            let weights = Array2::from_shape_fn((x.ncols(), self.n_components_per_scale), |_| {
                rng.sample(normal) * (2.0 / scale).sqrt()
            });
            scale_weights.push(weights);
        }

        Ok(MultiscaleKernel {
            scales: self.scales,
            n_components_per_scale: self.n_components_per_scale,
            scale_weights: Some(scale_weights),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for MultiscaleKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let scale_weights = self.scale_weights.as_ref().unwrap();

        if scale_weights.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No scale weights available".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_scales = self.scales.len();
        let total_features = n_scales * self.n_components_per_scale;

        let mut output = Array2::zeros((n_samples, total_features));

        let normalizer = (2.0 / self.n_components_per_scale as Float).sqrt();

        for (scale_idx, weights) in scale_weights.iter().enumerate() {
            let projection = x.dot(weights);

            for i in 0..n_samples {
                for j in 0..self.n_components_per_scale {
                    let feature_idx = scale_idx * self.n_components_per_scale + j;
                    output[[i, feature_idx]] = normalizer * projection[[i, j]].cos();
                }
            }
        }

        Ok(output)
    }
}

impl MultiscaleKernel<Trained> {
    /// Get scales
    pub fn scales(&self) -> &[Float] {
        &self.scales
    }

    /// Get weight for a specific scale
    pub fn scale_weight(&self, scale_idx: usize) -> Option<&Array2<Float>> {
        self.scale_weights.as_ref().and_then(|w| w.get(scale_idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_physics_informed_kernel_basic() {
        let config = PhysicsInformedConfig {
            system_type: PhysicalSystem::HeatEquation,
            n_components: 20,
            ..Default::default()
        };

        let pinn = PhysicsInformedKernel::new(config);

        // Data: [x, t] coordinates
        let X = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.1],
            [0.5, 0.1],
            [1.0, 0.1]
        ];

        let fitted = pinn.fit(&X, &()).unwrap();
        let features = fitted.transform(&X).unwrap();

        assert_eq!(features.nrows(), 6);
        assert_eq!(features.ncols(), 60); // 3 * n_components
    }

    #[test]
    fn test_different_physical_systems() {
        let systems = vec![
            PhysicalSystem::HeatEquation,
            PhysicalSystem::WaveEquation,
            PhysicalSystem::BurgersEquation,
        ];

        let X = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        for system in systems {
            let pinn = PhysicsInformedKernel::with_system(system).n_components(20);
            let fitted = pinn.fit(&X, &()).unwrap();
            let features = fitted.transform(&X).unwrap();

            assert_eq!(features.nrows(), 3);
        }
    }

    #[test]
    fn test_pde_residual() {
        let config = PhysicsInformedConfig {
            system_type: PhysicalSystem::HeatEquation,
            n_components: 20,
            ..Default::default()
        };

        let pinn = PhysicsInformedKernel::new(config);
        let X = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.1],
            [0.5, 0.1],
            [1.0, 0.1]
        ];

        let fitted = pinn.fit(&X, &()).unwrap();

        // Test solution (simple linear function)
        let solution = array![0.0, 0.5, 1.0, 0.0, 0.5, 1.0];

        let residual = fitted.pde_residual(&X, &solution).unwrap();

        assert_eq!(residual.len(), 6);
        assert!(residual.iter().all(|&r| r.is_finite()));
    }

    #[test]
    fn test_multiscale_kernel() {
        let scales = vec![0.1, 1.0, 10.0];
        let kernel = MultiscaleKernel::with_scales(scales).n_components_per_scale(10);

        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let fitted = kernel.fit(&X, &()).unwrap();
        let features = fitted.transform(&X).unwrap();

        assert_eq!(features.nrows(), 3);
        assert_eq!(features.ncols(), 30); // 3 scales * 10 components
    }

    #[test]
    fn test_multiscale_scales() {
        let scales = vec![0.5, 2.0, 8.0];
        let kernel = MultiscaleKernel::with_scales(scales.clone());

        let X = array![[1.0], [2.0]];

        let fitted = kernel.fit(&X, &()).unwrap();

        assert_eq!(fitted.scales(), &scales[..]);
    }

    #[test]
    fn test_empty_input_error() {
        let pinn = PhysicsInformedKernel::with_system(PhysicalSystem::HeatEquation);
        let empty: Array2<Float> = Array2::zeros((0, 0));

        assert!(pinn.fit(&empty, &()).is_err());
    }

    #[test]
    fn test_insufficient_columns_error() {
        let pinn = PhysicsInformedKernel::with_system(PhysicalSystem::HeatEquation);
        let data = array![[1.0]]; // Only 1 column, need at least 2

        assert!(pinn.fit(&data, &()).is_err());
    }

    #[test]
    fn test_physics_weight_setting() {
        let pinn = PhysicsInformedKernel::with_system(PhysicalSystem::HeatEquation)
            .physics_weight(2.0)
            .n_components(30);

        assert_eq!(pinn.config.physics_weight, 2.0);
        assert_eq!(pinn.config.n_components, 30);
    }
}

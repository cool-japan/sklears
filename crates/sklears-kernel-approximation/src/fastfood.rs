//! Fastfood Transform for efficient random feature approximation
//!
//! This module implements the Fastfood transform, which approximates
//! Gaussian random projections using structured matrices, reducing
//! computational complexity from O(d²) to O(d log d).

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Uniform as RandUniform;
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Estimator, Trained, Untrained},
    types::Float,
};
use std::f64::consts::PI;
use std::marker::PhantomData;

use crate::structured_random_features::FastWalshHadamardTransform;

/// Fastfood Transform for efficient random Fourier features
///
/// The Fastfood transform uses structured matrices to approximate Gaussian
/// random projections efficiently. It combines three structured transforms:
/// 1. Random diagonal scaling (B)
/// 2. Fast Walsh-Hadamard Transform (H)
/// 3. Random permutation (Π)
/// 4. Random diagonal scaling (G)
///
/// The overall transform is: G * H * Π * B * H
/// This reduces complexity from O(d²) to O(d log d) while maintaining
/// approximation quality for RBF kernels.
///
/// # Parameters
///
/// * `n_components` - Number of random features to generate
/// * `gamma` - RBF kernel parameter (default: 1.0)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::fastfood::FastfoodTransform;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]];
///
/// let fastfood = FastfoodTransform::new(8).gamma(0.5);
/// let fitted = fastfood.fit(&X, &()).unwrap();
/// let X_transformed = fitted.transform(&X).unwrap();
/// assert_eq!(X_transformed.shape(), &[2, 8]);
/// ```
#[derive(Debug, Clone)]
/// FastfoodTransform
pub struct FastfoodTransform<State = Untrained> {
    /// Number of random features
    pub n_components: usize,
    /// RBF kernel gamma parameter
    pub gamma: Float,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted parameters
    // The Fastfood transform consists of: G * H * Π * B * H
    scaling_b_: Option<Array1<Float>>,     // First diagonal scaling
    permutation_: Option<Array1<usize>>,   // Random permutation
    scaling_g_: Option<Array1<Float>>,     // Second diagonal scaling
    random_offset_: Option<Array1<Float>>, // Phase offsets
    padded_dim_: Option<usize>,            // Padded dimension (power of 2)
    n_blocks_: Option<usize>,              // Number of Fastfood blocks

    _state: PhantomData<State>,
}

impl FastfoodTransform<Untrained> {
    /// Create a new Fastfood transform
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            gamma: 1.0,
            random_state: None,
            scaling_b_: None,
            permutation_: None,
            scaling_g_: None,
            random_offset_: None,
            padded_dim_: None,
            n_blocks_: None,
            _state: PhantomData,
        }
    }

    /// Set the gamma parameter for RBF kernel
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for FastfoodTransform<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for FastfoodTransform<Untrained> {
    type Fitted = FastfoodTransform<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        // Find the smallest power of 2 that is >= n_features
        let padded_dim = next_power_of_2(n_features);

        // Number of Fastfood blocks needed
        let n_blocks = (self.n_components + padded_dim - 1) / padded_dim;

        // Generate random diagonal scaling matrices B and G
        let scaling_b = self.generate_random_scaling(padded_dim * n_blocks, &mut rng);
        let scaling_g = self.generate_random_scaling(padded_dim * n_blocks, &mut rng);

        // Generate random permutations
        let permutation = self.generate_random_permutation(padded_dim * n_blocks, &mut rng);

        // Generate random phase offsets
        let uniform = RandUniform::new(0.0, 2.0 * PI).unwrap();
        let random_offset = Array1::from_shape_fn(self.n_components, |_| rng.sample(uniform));

        Ok(FastfoodTransform {
            n_components: self.n_components,
            gamma: self.gamma,
            random_state: self.random_state,
            scaling_b_: Some(scaling_b),
            permutation_: Some(permutation),
            scaling_g_: Some(scaling_g),
            random_offset_: Some(random_offset),
            padded_dim_: Some(padded_dim),
            n_blocks_: Some(n_blocks),
            _state: PhantomData,
        })
    }
}

impl FastfoodTransform<Untrained> {
    /// Generate random diagonal scaling with Rademacher distribution
    fn generate_random_scaling(&self, size: usize, rng: &mut RealStdRng) -> Array1<Float> {
        let mut scaling = Array1::zeros(size);
        for i in 0..size {
            scaling[i] = if rng.gen::<bool>() { 1.0 } else { -1.0 };
        }
        scaling
    }

    /// Generate random permutation
    fn generate_random_permutation(&self, size: usize, rng: &mut RealStdRng) -> Array1<usize> {
        let mut permutation: Vec<usize> = (0..size).collect();
        permutation.shuffle(rng);
        Array1::from_vec(permutation)
    }
}

impl Transform<Array2<Float>> for FastfoodTransform<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let scaling_b = self
            .scaling_b_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let permutation = self
            .permutation_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let scaling_g = self
            .scaling_g_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let random_offset =
            self.random_offset_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let padded_dim = *self
            .padded_dim_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let n_blocks = *self
            .n_blocks_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let (n_samples, n_features) = x.dim();
        let mut features = Array2::zeros((n_samples, self.n_components));

        // Process each sample
        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);

            // Apply Fastfood transform: G * H * Π * B * H * x
            let transformed_sample = self.apply_fastfood_transform(
                &sample,
                scaling_b,
                permutation,
                scaling_g,
                padded_dim,
                n_blocks,
                n_features,
            )?;

            // Take first n_components and add phase offsets
            for j in 0..(self.n_components.min(transformed_sample.len())) {
                let phase = transformed_sample[j] * (2.0 * self.gamma).sqrt() + random_offset[j];
                features[[sample_idx, j]] =
                    (2.0 / (self.n_components as Float)).sqrt() * phase.cos();
            }
        }

        Ok(features)
    }
}

impl FastfoodTransform<Trained> {
    /// Apply the full Fastfood transform: G * H * Π * B * H * x
    fn apply_fastfood_transform(
        &self,
        x: &scirs2_core::ndarray::ArrayBase<
            scirs2_core::ndarray::ViewRepr<&Float>,
            scirs2_core::ndarray::Dim<[usize; 1]>,
        >,
        scaling_b: &Array1<Float>,
        permutation: &Array1<usize>,
        scaling_g: &Array1<Float>,
        padded_dim: usize,
        n_blocks: usize,
        n_features: usize,
    ) -> Result<Array1<Float>> {
        let mut result = Array1::zeros(padded_dim * n_blocks);

        // Process each Fastfood block
        for block in 0..n_blocks {
            let block_start = block * padded_dim;
            let _block_end = block_start + padded_dim;

            // Step 1: Pad input to power of 2
            let mut padded_input = Array1::zeros(padded_dim);
            for i in 0..n_features.min(padded_dim) {
                padded_input[i] = x[i];
            }

            // Step 2: First Hadamard transform (H)
            let mut transformed = FastWalshHadamardTransform::transform(padded_input)?;

            // Step 3: Apply first diagonal scaling (B)
            for i in 0..padded_dim {
                transformed[i] *= scaling_b[block_start + i];
            }

            // Step 4: Apply permutation (Π)
            let mut permuted = Array1::zeros(padded_dim);
            for i in 0..padded_dim {
                let perm_idx = permutation[block_start + i] % padded_dim;
                permuted[i] = transformed[perm_idx];
            }

            // Step 5: Second Hadamard transform (H)
            transformed = FastWalshHadamardTransform::transform(permuted)?;

            // Step 6: Apply second diagonal scaling (G)
            for i in 0..padded_dim {
                transformed[i] *= scaling_g[block_start + i];
            }

            // Store result for this block
            for i in 0..padded_dim {
                result[block_start + i] = transformed[i];
            }
        }

        Ok(result)
    }
}

/// Find the next power of 2 greater than or equal to n
fn next_power_of_2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}

/// Fastfood kernel approximation for multiple kernels
///
/// This variant allows approximating different kernels by adjusting
/// the scaling and normalization factors.
#[derive(Debug, Clone)]
/// FastfoodKernel
pub struct FastfoodKernel<State = Untrained> {
    /// Number of random features
    pub n_components: usize,
    /// Kernel parameters
    pub kernel_params: FastfoodKernelParams,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted parameters
    fastfood_transforms_: Option<Vec<FastfoodTransform<Trained>>>,

    _state: PhantomData<State>,
}

/// Kernel parameters for Fastfood approximation
#[derive(Debug, Clone)]
/// FastfoodKernelParams
pub enum FastfoodKernelParams {
    /// RBF kernel with gamma parameter
    Rbf { gamma: Float },
    /// Matern kernel with nu and length_scale parameters
    Matern { nu: Float, length_scale: Float },
    /// Rational quadratic kernel with alpha and length_scale
    RationalQuadratic { alpha: Float, length_scale: Float },
}

impl FastfoodKernel<Untrained> {
    /// Create a new Fastfood kernel approximation
    pub fn new(n_components: usize, kernel_params: FastfoodKernelParams) -> Self {
        Self {
            n_components,
            kernel_params,
            random_state: None,
            fastfood_transforms_: None,
            _state: PhantomData,
        }
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for FastfoodKernel<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for FastfoodKernel<Untrained> {
    type Fitted = FastfoodKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        // For now, implement RBF kernel case
        let gamma = match &self.kernel_params {
            FastfoodKernelParams::Rbf { gamma } => *gamma,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Only RBF kernel is currently supported for FastfoodKernel".to_string(),
                ))
            }
        };

        let fastfood = FastfoodTransform::new(self.n_components).gamma(gamma);
        let fastfood = match self.random_state {
            Some(seed) => fastfood.random_state(seed),
            None => fastfood,
        };

        let fitted_fastfood = fastfood.fit(x, &())?;
        let transforms = vec![fitted_fastfood];

        Ok(FastfoodKernel {
            n_components: self.n_components,
            kernel_params: self.kernel_params,
            random_state: self.random_state,
            fastfood_transforms_: Some(transforms),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>> for FastfoodKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let transforms =
            self.fastfood_transforms_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        // For now, use the first (and only) transform
        transforms[0].transform(x)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(7), 8);
        assert_eq!(next_power_of_2(8), 8);
        assert_eq!(next_power_of_2(15), 16);
        assert_eq!(next_power_of_2(16), 16);
    }

    #[test]
    fn test_fastfood_transform_basic() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];

        let fastfood = FastfoodTransform::new(8).gamma(0.5);
        let fitted = fastfood.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[3, 8]);
    }

    #[test]
    fn test_fastfood_transform_power_of_2() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]];

        let fastfood = FastfoodTransform::new(4).gamma(1.0);
        let fitted = fastfood.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[2, 4]);
    }

    #[test]
    fn test_fastfood_kernel_rbf() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];

        let kernel_params = FastfoodKernelParams::Rbf { gamma: 0.5 };
        let fastfood_kernel = FastfoodKernel::new(6, kernel_params);
        let fitted = fastfood_kernel.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[2, 6]);
    }

    #[test]
    fn test_fastfood_reproducibility() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];

        let fastfood1 = FastfoodTransform::new(8).random_state(42);
        let fitted1 = fastfood1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let fastfood2 = FastfoodTransform::new(8).random_state(42);
        let fitted2 = fastfood2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        assert_eq!(result1.shape(), result2.shape());
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fastfood_different_gamma() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];

        let fastfood_low = FastfoodTransform::new(4).gamma(0.1);
        let fitted_low = fastfood_low.fit(&x, &()).unwrap();
        let result_low = fitted_low.transform(&x).unwrap();

        let fastfood_high = FastfoodTransform::new(4).gamma(10.0);
        let fitted_high = fastfood_high.fit(&x, &()).unwrap();
        let result_high = fitted_high.transform(&x).unwrap();

        assert_eq!(result_low.shape(), result_high.shape());
        // Results should be different with different gamma values
        let diff_sum: Float = result_low
            .iter()
            .zip(result_high.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff_sum > 1e-6);
    }

    #[test]
    fn test_fastfood_large_dimensions() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        ];

        let fastfood = FastfoodTransform::new(16).gamma(0.1);
        let fitted = fastfood.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[2, 16]);
    }

    #[test]
    fn test_fastfood_single_sample() {
        let x = array![[1.0, 2.0, 3.0, 4.0]];

        let fastfood = FastfoodTransform::new(8).gamma(1.0);
        let fitted = fastfood.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[1, 8]);
    }

    #[test]
    fn test_fastfood_edge_cases() {
        // Test with minimal dimensions
        let x = array![[1.0], [2.0]];

        let fastfood = FastfoodTransform::new(2).gamma(1.0);
        let fitted = fastfood.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[2, 2]);

        // Test with many components
        let x2 = array![[1.0, 2.0], [3.0, 4.0]];
        let fastfood2 = FastfoodTransform::new(32).gamma(0.5);
        let fitted2 = fastfood2.fit(&x2, &()).unwrap();
        let transformed2 = fitted2.transform(&x2).unwrap();

        assert_eq!(transformed2.shape(), &[2, 32]);
    }
}

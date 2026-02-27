//! Structured orthogonal random features for efficient kernel approximation
//!
//! This module implements structured random feature methods that use structured
//! random matrices (like Hadamard matrices) to reduce computational complexity
//! while maintaining approximation quality. Also includes quasi-random and
//! low-discrepancy sequence methods for improved feature distribution.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use scirs2_core::StandardNormal;
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Estimator, Trained, Untrained},
    types::Float,
};
use std::f64::consts::PI;
use std::marker::PhantomData;

/// Type of structured matrix to use
#[derive(Debug, Clone)]
/// StructuredMatrix
pub enum StructuredMatrix {
    Hadamard,
    DCT,
    Circulant,
    Toeplitz,
}

/// Type of quasi-random sequence to use for feature generation
#[derive(Debug, Clone)]
/// QuasiRandomSequence
pub enum QuasiRandomSequence {
    VanDerCorput,
    Halton,
    Sobol,
    PseudoRandom,
}

/// Low-discrepancy sequence generators for quasi-random features
pub struct LowDiscrepancySequences;

impl LowDiscrepancySequences {
    /// Generate Van der Corput sequence in base 2
    /// This provides a 1D low-discrepancy sequence
    pub fn van_der_corput(n: usize) -> Vec<Float> {
        (0..n)
            .map(|i| {
                let mut value = 0.0;
                let mut base_inv = 0.5;
                let mut n = i + 1;

                while n > 0 {
                    if n % 2 == 1 {
                        value += base_inv;
                    }
                    base_inv *= 0.5;
                    n /= 2;
                }
                value
            })
            .collect()
    }

    /// Generate Halton sequence for multi-dimensional low-discrepancy
    /// Uses prime bases for each dimension
    pub fn halton(n: usize, dimensions: usize) -> Array2<Float> {
        let primes = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        ];

        if dimensions > primes.len() {
            panic!("Maximum supported dimensions: {}", primes.len());
        }

        let mut sequence = Array2::zeros((n, dimensions));

        for dim in 0..dimensions {
            let base = primes[dim] as Float;
            for i in 0..n {
                let mut value = 0.0;
                let mut base_inv = 1.0 / base;
                let mut n = i + 1;

                while n > 0 {
                    value += (n % primes[dim]) as Float * base_inv;
                    base_inv /= base;
                    n /= primes[dim];
                }
                sequence[[i, dim]] = value;
            }
        }

        sequence
    }

    /// Generate simplified Sobol sequence for higher dimensions
    /// This is a basic implementation focusing on the key properties
    pub fn sobol(n: usize, dimensions: usize) -> Array2<Float> {
        let mut sequence = Array2::zeros((n, dimensions));

        // For the first dimension, use Van der Corput base 2
        let first_dim = Self::van_der_corput(n);
        for i in 0..n {
            sequence[[i, 0]] = first_dim[i];
        }

        // For additional dimensions, use Van der Corput with different bases
        let bases = [
            3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
        ];

        for dim in 1..dimensions {
            let base = if dim - 1 < bases.len() {
                bases[dim - 1]
            } else {
                2 + dim
            };
            for i in 0..n {
                let mut value = 0.0;
                let mut base_inv = 1.0 / (base as Float);
                let mut n = i + 1;

                while n > 0 {
                    value += (n % base) as Float * base_inv;
                    base_inv /= base as Float;
                    n /= base;
                }
                sequence[[i, dim]] = value;
            }
        }

        sequence
    }

    /// Transform uniform low-discrepancy sequence to Gaussian using inverse normal CDF
    /// Box-Muller-like transformation for quasi-random Gaussian variables
    pub fn uniform_to_gaussian(uniform_sequence: &Array2<Float>) -> Array2<Float> {
        let (n, dim) = uniform_sequence.dim();
        let mut gaussian_sequence = Array2::zeros((n, dim));

        for i in 0..n {
            for j in 0..dim {
                let u = uniform_sequence[[i, j]];
                // Prevent extreme values
                let u_clamped = u.max(1e-10).min(1.0 - 1e-10);

                // Approximate inverse normal CDF using Beasley-Springer-Moro algorithm
                let x = Self::inverse_normal_cdf(u_clamped);
                gaussian_sequence[[i, j]] = x;
            }
        }

        gaussian_sequence
    }

    /// Approximate inverse normal CDF using rational approximation
    fn inverse_normal_cdf(u: Float) -> Float {
        if u <= 0.0 || u >= 1.0 {
            return if u <= 0.0 {
                Float::NEG_INFINITY
            } else {
                Float::INFINITY
            };
        }

        let u = if u > 0.5 { 1.0 - u } else { u };
        let sign = if u == 1.0 - u { 1.0 } else { -1.0 };

        let t = (-2.0 * u.ln()).sqrt();

        // Coefficients for rational approximation
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let numerator = c0 + c1 * t + c2 * t * t;
        let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;

        sign * (t - numerator / denominator)
    }
}

/// Structured orthogonal random features
///
/// Uses structured random matrices to approximate RBF kernels more efficiently
/// than standard random Fourier features. The structured approach reduces
/// computational complexity from O(d*D) to O(d*log(D)) for certain operations.
///
/// # Parameters
///
/// * `n_components` - Number of random features to generate
/// * `gamma` - RBF kernel parameter (default: 1.0)
/// * `structured_matrix` - Type of structured matrix to use
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```text
/// use sklears_kernel_approximation::structured_random_features::{
///     StructuredRandomFeatures, StructuredMatrix,
/// };
///
/// let srf = StructuredRandomFeatures::new(100)
///     .with_gamma(0.5)
///     .with_structured_matrix(StructuredMatrix::Hadamard);
/// ```
#[derive(Debug, Clone)]
pub struct StructuredRandomFeatures<State = Untrained> {
    pub n_components: usize,
    pub gamma: Float,
    pub structured_matrix: StructuredMatrix,
    pub random_state: Option<u64>,

    // Fitted parameters
    random_weights_: Option<Array2<Float>>,
    random_offset_: Option<Array1<Float>>,
    structured_transform_: Option<Array2<Float>>,

    _state: PhantomData<State>,
}

impl StructuredRandomFeatures<Untrained> {
    /// Create a new structured random features transformer
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            gamma: 1.0,
            structured_matrix: StructuredMatrix::Hadamard,
            random_state: None,
            random_weights_: None,
            random_offset_: None,
            structured_transform_: None,
            _state: PhantomData,
        }
    }

    /// Set the gamma parameter for RBF kernel
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the structured matrix type
    pub fn structured_matrix(mut self, matrix_type: StructuredMatrix) -> Self {
        self.structured_matrix = matrix_type;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for StructuredRandomFeatures<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for StructuredRandomFeatures<Untrained> {
    type Fitted = StructuredRandomFeatures<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        // Generate structured transform matrix
        let structured_transform = self.generate_structured_matrix(n_features, &mut rng)?;

        // Generate random weights for mixing
        let normal = RandNormal::new(0.0, (2.0 * self.gamma).sqrt()).unwrap();
        let random_weights =
            Array2::from_shape_fn((n_features, self.n_components), |_| rng.sample(normal));

        // Generate random offset for phase
        let uniform = RandUniform::new(0.0, 2.0 * PI).unwrap();
        let random_offset = Array1::from_shape_fn(self.n_components, |_| rng.sample(uniform));

        Ok(StructuredRandomFeatures {
            n_components: self.n_components,
            gamma: self.gamma,
            structured_matrix: self.structured_matrix,
            random_state: self.random_state,
            random_weights_: Some(random_weights),
            random_offset_: Some(random_offset),
            structured_transform_: Some(structured_transform),
            _state: PhantomData,
        })
    }
}

impl StructuredRandomFeatures<Untrained> {
    /// Generate structured matrix based on the specified type
    fn generate_structured_matrix(
        &self,
        n_features: usize,
        rng: &mut RealStdRng,
    ) -> Result<Array2<Float>> {
        match &self.structured_matrix {
            StructuredMatrix::Hadamard => self.generate_hadamard_matrix(n_features, rng),
            StructuredMatrix::DCT => self.generate_dct_matrix(n_features),
            StructuredMatrix::Circulant => self.generate_circulant_matrix(n_features, rng),
            StructuredMatrix::Toeplitz => self.generate_toeplitz_matrix(n_features, rng),
        }
    }

    /// Generate (approximate) Hadamard matrix
    fn generate_hadamard_matrix(
        &self,
        n_features: usize,
        rng: &mut RealStdRng,
    ) -> Result<Array2<Float>> {
        // For simplicity, generate a randomized orthogonal-like matrix
        // True Hadamard matrices exist only for specific sizes
        let mut matrix = Array2::zeros((n_features, n_features));

        // Generate random signs for each entry
        for i in 0..n_features {
            for j in 0..n_features {
                matrix[[i, j]] = if rng.gen::<bool>() { 1.0 } else { -1.0 };
            }
        }

        // Normalize to make approximately orthogonal
        for mut row in matrix.rows_mut() {
            let norm = (row.dot(&row) as Float).sqrt();
            if norm > 1e-10 {
                row /= norm;
            }
        }

        Ok(matrix)
    }

    /// Generate Discrete Cosine Transform matrix
    fn generate_dct_matrix(&self, n_features: usize) -> Result<Array2<Float>> {
        let mut matrix = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                let coeff = if i == 0 {
                    (1.0 / (n_features as Float)).sqrt()
                } else {
                    (2.0 / (n_features as Float)).sqrt()
                };

                matrix[[i, j]] = coeff
                    * ((PI * (i as Float) * (2.0 * (j as Float) + 1.0))
                        / (2.0 * (n_features as Float)))
                        .cos();
            }
        }

        Ok(matrix)
    }

    /// Generate Circulant matrix
    fn generate_circulant_matrix(
        &self,
        n_features: usize,
        rng: &mut RealStdRng,
    ) -> Result<Array2<Float>> {
        let mut matrix = Array2::zeros((n_features, n_features));

        // Generate first row randomly
        let first_row: Vec<Float> = (0..n_features)
            .map(|_| StandardNormal.sample(rng))
            .collect();

        // Fill circulant matrix
        for i in 0..n_features {
            for j in 0..n_features {
                matrix[[i, j]] = first_row[(j + n_features - i) % n_features];
            }
        }

        // Normalize rows
        for mut row in matrix.rows_mut() {
            let norm = (row.dot(&row) as Float).sqrt();
            if norm > 1e-10 {
                row /= norm;
            }
        }

        Ok(matrix)
    }

    /// Generate Toeplitz matrix
    fn generate_toeplitz_matrix(
        &self,
        n_features: usize,
        rng: &mut RealStdRng,
    ) -> Result<Array2<Float>> {
        let mut matrix = Array2::zeros((n_features, n_features));

        // Generate values for first row and first column
        let first_row: Vec<Float> = (0..n_features)
            .map(|_| StandardNormal.sample(rng))
            .collect();
        let first_col: Vec<Float> = (1..n_features)
            .map(|_| StandardNormal.sample(rng))
            .collect();

        // Fill Toeplitz matrix
        for i in 0..n_features {
            for j in 0..n_features {
                if i <= j {
                    matrix[[i, j]] = first_row[j - i];
                } else {
                    matrix[[i, j]] = first_col[i - j - 1];
                }
            }
        }

        // Normalize rows
        for mut row in matrix.rows_mut() {
            let norm = (row.dot(&row) as Float).sqrt();
            if norm > 1e-10 {
                row /= norm;
            }
        }

        Ok(matrix)
    }
}

impl Transform<Array2<Float>> for StructuredRandomFeatures<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let random_weights =
            self.random_weights_
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

        let structured_transform =
            self.structured_transform_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let (n_samples, _) = x.dim();

        // Apply structured transformation first
        let structured_x = x.dot(structured_transform);

        // Apply random projections
        let projected = structured_x.dot(random_weights);

        // Add phase offsets and compute cosine features
        let mut features = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            for j in 0..self.n_components {
                let phase = projected[[i, j]] + random_offset[j];
                features[[i, j]] = (2.0 / (self.n_components as Float)).sqrt() * phase.cos();
            }
        }

        Ok(features)
    }
}

/// Fast Walsh-Hadamard Transform for efficient structured features
///
/// This implements the Fast Walsh-Hadamard Transform (FWHT) which can be used
/// to accelerate computations with Hadamard-structured random features.
pub struct FastWalshHadamardTransform;

impl FastWalshHadamardTransform {
    /// Apply Fast Walsh-Hadamard Transform to input vector
    ///
    /// Time complexity: O(n log n) where n is the length of the input
    /// Input length must be a power of 2
    pub fn transform(mut data: Array1<Float>) -> Result<Array1<Float>> {
        let n = data.len();

        // Check if n is a power of 2
        if n & (n - 1) != 0 {
            return Err(SklearsError::InvalidInput(
                "Input length must be a power of 2 for FWHT".to_string(),
            ));
        }

        // Perform FWHT
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..i + h {
                    let u = data[j];
                    let v = data[j + h];
                    data[j] = u + v;
                    data[j + h] = u - v;
                }
            }
            h *= 2;
        }

        // Normalize
        data /= (n as Float).sqrt();

        Ok(data)
    }

    /// Apply FWHT to each row of a 2D array
    pub fn transform_rows(mut data: Array2<Float>) -> Result<Array2<Float>> {
        let (n_rows, n_cols) = data.dim();

        // Check if n_cols is a power of 2
        if n_cols & (n_cols - 1) != 0 {
            return Err(SklearsError::InvalidInput(
                "Number of columns must be a power of 2 for FWHT".to_string(),
            ));
        }

        for i in 0..n_rows {
            let row = data.row(i).to_owned();
            let transformed_row = Self::transform(row)?;
            data.row_mut(i).assign(&transformed_row);
        }

        Ok(data)
    }
}

/// Structured Random Features using Fast Walsh-Hadamard Transform
///
/// This is an optimized version that uses FWHT for better computational efficiency.
/// Input dimension must be a power of 2 for optimal performance.
#[derive(Debug, Clone)]
/// StructuredRFFHadamard
pub struct StructuredRFFHadamard<State = Untrained> {
    /// Number of random features
    pub n_components: usize,
    /// RBF kernel gamma parameter
    pub gamma: Float,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted parameters
    random_signs_: Option<Array2<Float>>,
    random_offset_: Option<Array1<Float>>,
    gaussian_weights_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl StructuredRFFHadamard<Untrained> {
    /// Create a new structured RFF with Hadamard transforms
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            gamma: 1.0,
            random_state: None,
            random_signs_: None,
            random_offset_: None,
            gaussian_weights_: None,
            _state: PhantomData,
        }
    }

    /// Set gamma parameter
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for StructuredRFFHadamard<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for StructuredRFFHadamard<Untrained> {
    type Fitted = StructuredRFFHadamard<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        // Check if n_features is a power of 2
        if n_features & (n_features - 1) != 0 {
            return Err(SklearsError::InvalidInput(
                "Number of features must be a power of 2 for structured Hadamard RFF".to_string(),
            ));
        }

        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        // Generate random signs for Hadamard transforms
        let mut random_signs = Array2::zeros((self.n_components, n_features));
        for i in 0..self.n_components {
            for j in 0..n_features {
                random_signs[[i, j]] = if rng.gen::<bool>() { 1.0 } else { -1.0 };
            }
        }

        // Generate Gaussian weights
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let gaussian_weights = Array1::from_shape_fn(n_features, |_| rng.sample(normal));

        // Generate random offsets
        let uniform = RandUniform::new(0.0, 2.0 * PI).unwrap();
        let random_offset = Array1::from_shape_fn(self.n_components, |_| rng.sample(uniform));

        Ok(StructuredRFFHadamard {
            n_components: self.n_components,
            gamma: self.gamma,
            random_state: self.random_state,
            random_signs_: Some(random_signs),
            random_offset_: Some(random_offset),
            gaussian_weights_: Some(gaussian_weights),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>> for StructuredRFFHadamard<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let random_signs = self
            .random_signs_
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

        let gaussian_weights =
            self.gaussian_weights_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let (n_samples, n_features) = x.dim();
        let mut features = Array2::zeros((n_samples, self.n_components));

        // For each component, apply structured transformation
        for comp in 0..self.n_components {
            for sample in 0..n_samples {
                // Element-wise multiplication with random signs
                let mut signed_input = Array1::zeros(n_features);
                for feat in 0..n_features {
                    signed_input[feat] = x[[sample, feat]] * random_signs[[comp, feat]];
                }

                // Apply Fast Walsh-Hadamard Transform
                let transformed = FastWalshHadamardTransform::transform(signed_input)?;

                // Scale by Gaussian weights and gamma
                let mut projected = 0.0;
                for feat in 0..n_features {
                    projected += transformed[feat] * gaussian_weights[feat];
                }
                projected *= (2.0 * self.gamma).sqrt();

                // Add offset and compute cosine
                let phase = projected + random_offset[comp];
                features[[sample, comp]] =
                    (2.0 / (self.n_components as Float)).sqrt() * phase.cos();
            }
        }

        Ok(features)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_structured_random_features_basic() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0]
        ];

        let srf = StructuredRandomFeatures::new(8).gamma(0.5);
        let fitted = srf.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[3, 8]);
    }

    #[test]
    fn test_structured_matrices() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]];

        // Test Hadamard
        let hadamard_srf =
            StructuredRandomFeatures::new(4).structured_matrix(StructuredMatrix::Hadamard);
        let hadamard_fitted = hadamard_srf.fit(&x, &()).unwrap();
        let hadamard_result = hadamard_fitted.transform(&x).unwrap();
        assert_eq!(hadamard_result.shape(), &[2, 4]);

        // Test DCT
        let dct_srf = StructuredRandomFeatures::new(4).structured_matrix(StructuredMatrix::DCT);
        let dct_fitted = dct_srf.fit(&x, &()).unwrap();
        let dct_result = dct_fitted.transform(&x).unwrap();
        assert_eq!(dct_result.shape(), &[2, 4]);

        // Test Circulant
        let circulant_srf =
            StructuredRandomFeatures::new(4).structured_matrix(StructuredMatrix::Circulant);
        let circulant_fitted = circulant_srf.fit(&x, &()).unwrap();
        let circulant_result = circulant_fitted.transform(&x).unwrap();
        assert_eq!(circulant_result.shape(), &[2, 4]);
    }

    #[test]
    fn test_fast_walsh_hadamard_transform() {
        let data = array![1.0, 2.0, 3.0, 4.0];
        let result = FastWalshHadamardTransform::transform(data).unwrap();
        assert_eq!(result.len(), 4);

        // Test with 2D array
        let data_2d = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let result_2d = FastWalshHadamardTransform::transform_rows(data_2d).unwrap();
        assert_eq!(result_2d.shape(), &[2, 4]);
    }

    #[test]
    fn test_structured_rff_hadamard() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]];

        let srf_h = StructuredRFFHadamard::new(6).gamma(0.5);
        let fitted = srf_h.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[2, 6]);
    }

    #[test]
    fn test_fwht_invalid_size() {
        let data = array![1.0, 2.0, 3.0]; // Not a power of 2
        let result = FastWalshHadamardTransform::transform(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]];

        let srf1 = StructuredRandomFeatures::new(8).random_state(42);
        let fitted1 = srf1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let srf2 = StructuredRandomFeatures::new(8).random_state(42);
        let fitted2 = srf2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        assert_eq!(result1.shape(), result2.shape());
        for i in 0..result1.len() {
            assert!(
                (result1.as_slice().unwrap()[i] - result2.as_slice().unwrap()[i]).abs() < 1e-10
            );
        }
    }

    #[test]
    fn test_different_gamma_values() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]];

        let srf_low = StructuredRandomFeatures::new(4).gamma(0.1);
        let fitted_low = srf_low.fit(&x, &()).unwrap();
        let result_low = fitted_low.transform(&x).unwrap();

        let srf_high = StructuredRandomFeatures::new(4).gamma(10.0);
        let fitted_high = srf_high.fit(&x, &()).unwrap();
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
}

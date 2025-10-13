//! Quasi-random feature generation for improved kernel approximations
//!
//! This module implements quasi-random sequences (low-discrepancy sequences) for generating
//! more uniformly distributed random features compared to standard pseudo-random sampling.
//! These sequences provide better convergence properties and improved approximation quality.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Trained, Untrained},
    types::Float,
};
use std::f64::consts::PI;
use std::marker::PhantomData;

/// Quasi-random sequence types for feature generation
#[derive(Debug, Clone, Copy)]
/// QuasiRandomSequence
pub enum QuasiRandomSequence {
    /// Sobol sequence - provides good equidistribution properties
    Sobol,
    /// Halton sequence - based on prime numbers, good for low dimensions
    Halton,
    /// van der Corput sequence - fundamental low-discrepancy sequence
    VanDerCorput,
    /// Faure sequence - generalization of van der Corput
    Faure,
}

/// Quasi-random RBF sampler using low-discrepancy sequences
///
/// This sampler generates random Fourier features for RBF kernels using quasi-random
/// sequences instead of pseudo-random numbers, providing better uniform coverage
/// of the frequency space and improved approximation quality.
///
/// # Mathematical Background
///
/// For an RBF kernel k(x,y) = exp(-γ||x-y||²), the random Fourier features are:
/// z(x) = √(2/n_components) * [cos(ω₁ᵀx + b₁), ..., cos(ωₙᵀx + bₙ)]
///
/// Where ω ~ N(0, 2γI) and b ~ Uniform[0, 2π]. Using quasi-random sequences
/// for generating ω and b provides better approximation properties.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::{QuasiRandomRBFSampler, QuasiRandomSequence};
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let sampler = QuasiRandomRBFSampler::new(100)
///     .gamma(1.0)
///     .sequence_type(QuasiRandomSequence::Sobol);
/// let fitted = sampler.fit(&x, &()).unwrap();
/// let features = fitted.transform(&x).unwrap();
/// assert_eq!(features.shape(), &[3, 100]);
/// ```
#[derive(Debug, Clone)]
/// QuasiRandomRBFSampler
pub struct QuasiRandomRBFSampler<State = Untrained> {
    /// RBF kernel parameter
    pub gamma: Float,
    /// Number of Monte Carlo samples
    pub n_components: usize,
    /// Quasi-random sequence type
    pub sequence_type: QuasiRandomSequence,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,

    // Fitted attributes
    random_weights_: Option<Array2<Float>>,
    random_offset_: Option<Array1<Float>>,

    // State marker
    _state: PhantomData<State>,
}

impl QuasiRandomRBFSampler<Untrained> {
    /// Create a new quasi-random RBF sampler
    ///
    /// # Arguments
    /// * `n_components` - Number of Monte Carlo samples (random features)
    pub fn new(n_components: usize) -> Self {
        Self {
            gamma: 1.0,
            n_components,
            sequence_type: QuasiRandomSequence::Sobol,
            random_state: None,
            random_weights_: None,
            random_offset_: None,
            _state: PhantomData,
        }
    }

    /// Set the gamma parameter for the RBF kernel
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the quasi-random sequence type
    pub fn sequence_type(mut self, sequence_type: QuasiRandomSequence) -> Self {
        self.sequence_type = sequence_type;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Fit<Array2<Float>, ()> for QuasiRandomRBFSampler<Untrained> {
    type Fitted = QuasiRandomRBFSampler<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array is empty".to_string(),
            ));
        }

        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        // Generate quasi-random weights
        let random_weights = match self.sequence_type {
            QuasiRandomSequence::Sobol => {
                generate_sobol_gaussian(self.n_components, n_features, self.gamma, &mut rng)
            }
            QuasiRandomSequence::Halton => {
                generate_halton_gaussian(self.n_components, n_features, self.gamma, &mut rng)
            }
            QuasiRandomSequence::VanDerCorput => generate_van_der_corput_gaussian(
                self.n_components,
                n_features,
                self.gamma,
                &mut rng,
            ),
            QuasiRandomSequence::Faure => {
                generate_faure_gaussian(self.n_components, n_features, self.gamma, &mut rng)
            }
        }?;

        // Generate quasi-random offset using the same sequence type
        let random_offset = match self.sequence_type {
            QuasiRandomSequence::Sobol => generate_sobol_uniform(self.n_components, &mut rng),
            QuasiRandomSequence::Halton => generate_halton_uniform(self.n_components, &mut rng),
            QuasiRandomSequence::VanDerCorput => {
                generate_van_der_corput_uniform(self.n_components, &mut rng)
            }
            QuasiRandomSequence::Faure => generate_faure_uniform(self.n_components, &mut rng),
        }
        .mapv(|x| x * 2.0 * PI);

        Ok(QuasiRandomRBFSampler {
            gamma: self.gamma,
            n_components: self.n_components,
            sequence_type: self.sequence_type,
            random_state: self.random_state,
            random_weights_: Some(random_weights),
            random_offset_: Some(random_offset),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>> for QuasiRandomRBFSampler<Trained> {
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

        let (n_samples, n_features) = x.dim();

        if n_features != random_weights.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Input has {} features, expected {}",
                n_features,
                random_weights.ncols()
            )));
        }

        // Compute X @ W.T + b
        let projection = x.dot(&random_weights.t()) + random_offset;

        // Apply cosine transformation and normalize
        let normalization = (2.0 / random_weights.nrows() as Float).sqrt();
        Ok(projection.mapv(|x| x.cos() * normalization))
    }
}

/// Generate Sobol sequence for Gaussian distribution
fn generate_sobol_gaussian<R: Rng>(
    n_components: usize,
    n_features: usize,
    gamma: Float,
    _rng: &mut R,
) -> Result<Array2<Float>> {
    // For simplicity, we'll use a basic Sobol-like approach
    // In practice, you'd use a proper Sobol sequence generator
    let mut weights = Array2::zeros((n_components, n_features));
    let std_dev = (2.0 * gamma).sqrt();

    // Generate quasi-random points in [0,1]^(2*n_features) and convert to Gaussian
    for i in 0..n_components {
        for j in 0..n_features {
            // Simple Sobol-like sequence using Gray code
            let sobol_u1 = sobol_point(2 * i, 2 * j);
            let sobol_u2 = sobol_point(2 * i + 1, 2 * j + 1);

            // Box-Muller transformation to get Gaussian
            let gaussian = box_muller_transform(sobol_u1, sobol_u2).0;
            weights[[i, j]] = gaussian * std_dev;
        }
    }

    Ok(weights)
}

/// Generate Halton sequence for Gaussian distribution
fn generate_halton_gaussian<R: Rng>(
    n_components: usize,
    n_features: usize,
    gamma: Float,
    _rng: &mut R,
) -> Result<Array2<Float>> {
    let mut weights = Array2::zeros((n_components, n_features));
    let std_dev = (2.0 * gamma).sqrt();

    let primes = get_first_primes(2 * n_features);

    for i in 0..n_components {
        for j in 0..n_features {
            // Use consecutive primes for u1 and u2
            let u1 = halton_sequence(i + 1, primes[2 * j]);
            let u2 = halton_sequence(i + 1, primes[2 * j + 1]);

            // Box-Muller transformation
            let gaussian = box_muller_transform(u1, u2).0;
            weights[[i, j]] = gaussian * std_dev;
        }
    }

    Ok(weights)
}

/// Generate van der Corput sequence for Gaussian distribution
fn generate_van_der_corput_gaussian<R: Rng>(
    n_components: usize,
    n_features: usize,
    gamma: Float,
    _rng: &mut R,
) -> Result<Array2<Float>> {
    let mut weights = Array2::zeros((n_components, n_features));
    let std_dev = (2.0 * gamma).sqrt();

    for i in 0..n_components {
        for j in 0..n_features {
            // Use base 2 and base 3 for van der Corput
            let u1 = van_der_corput_sequence(i + 1, 2);
            let u2 = van_der_corput_sequence(i + 1, 3);

            // Box-Muller transformation
            let gaussian = box_muller_transform(u1, u2).0;
            weights[[i, j]] = gaussian * std_dev;
        }
    }

    Ok(weights)
}

/// Generate Faure sequence for Gaussian distribution
fn generate_faure_gaussian<R: Rng>(
    n_components: usize,
    n_features: usize,
    gamma: Float,
    _rng: &mut R,
) -> Result<Array2<Float>> {
    // For this implementation, we'll use a simplified Faure-like sequence
    generate_halton_gaussian(n_components, n_features, gamma, _rng)
}

/// Generate Sobol sequence for uniform distribution [0, 1]
fn generate_sobol_uniform<R: Rng>(n_components: usize, _rng: &mut R) -> Array1<Float> {
    let mut uniform = Array1::zeros(n_components);
    for i in 0..n_components {
        uniform[i] = sobol_point(i, 0);
    }
    uniform
}

/// Generate Halton sequence for uniform distribution [0, 1]
fn generate_halton_uniform<R: Rng>(n_components: usize, _rng: &mut R) -> Array1<Float> {
    let mut uniform = Array1::zeros(n_components);
    for i in 0..n_components {
        uniform[i] = halton_sequence(i + 1, 2);
    }
    uniform
}

/// Generate van der Corput sequence for uniform distribution [0, 1]
fn generate_van_der_corput_uniform<R: Rng>(n_components: usize, _rng: &mut R) -> Array1<Float> {
    let mut uniform = Array1::zeros(n_components);
    for i in 0..n_components {
        uniform[i] = van_der_corput_sequence(i + 1, 2);
    }
    uniform
}

/// Generate Faure sequence for uniform distribution [0, 1]
fn generate_faure_uniform<R: Rng>(n_components: usize, rng: &mut R) -> Array1<Float> {
    // Simplified implementation
    generate_halton_uniform(n_components, rng)
}

/// Simple Sobol point generation (simplified implementation)
fn sobol_point(i: usize, dim: usize) -> Float {
    // This is a simplified Sobol implementation
    // In practice, you'd use a proper Sobol sequence generator library
    let mut n = i;
    let mut result = 0.0;
    let mut weight = 0.5;

    // Use Gray code for better equidistribution
    let gray_code = n ^ (n >> 1);
    n = gray_code;

    while n > 0 {
        if n & 1 == 1 {
            result += weight;
        }
        weight *= 0.5;
        n >>= 1;
    }

    // Add dimension-specific scrambling
    result = (result + dim as Float * 0.123456789) % 1.0;
    result
}

/// Halton sequence generation
fn halton_sequence(i: usize, base: usize) -> Float {
    let mut result = 0.0;
    let mut f = 1.0 / base as Float;
    let mut i = i;

    while i > 0 {
        result += f * (i % base) as Float;
        i /= base;
        f /= base as Float;
    }

    result
}

/// van der Corput sequence generation
fn van_der_corput_sequence(i: usize, base: usize) -> Float {
    halton_sequence(i, base)
}

/// Box-Muller transformation for converting uniform to Gaussian
fn box_muller_transform(u1: Float, u2: Float) -> (Float, Float) {
    let u1 = u1.max(1e-10).min(1.0 - 1e-10); // Avoid log(0)
    let u2 = u2.max(1e-10).min(1.0 - 1e-10);

    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;

    (r * theta.cos(), r * theta.sin())
}

/// Get first n prime numbers
fn get_first_primes(n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }

    let mut primes = vec![2];
    let mut candidate = 3;

    while primes.len() < n {
        let mut is_prime = true;
        let sqrt_candidate = (candidate as f64).sqrt() as usize;

        for &prime in &primes {
            if prime > sqrt_candidate {
                break;
            }
            if candidate % prime == 0 {
                is_prime = false;
                break;
            }
        }

        if is_prime {
            primes.push(candidate);
        }
        candidate += 2; // Only check odd numbers
    }

    primes
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quasi_random_rbf_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let sampler = QuasiRandomRBFSampler::new(10)
            .gamma(1.0)
            .sequence_type(QuasiRandomSequence::Sobol)
            .random_state(42);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[3, 10]);

        // Check that features are bounded (cosine function)
        for &val in features.iter() {
            assert!(val >= -2.0 && val <= 2.0);
        }
    }

    #[test]
    fn test_different_sequence_types() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let sequences = [
            QuasiRandomSequence::Sobol,
            QuasiRandomSequence::Halton,
            QuasiRandomSequence::VanDerCorput,
            QuasiRandomSequence::Faure,
        ];

        for seq_type in &sequences {
            let sampler = QuasiRandomRBFSampler::new(20)
                .sequence_type(*seq_type)
                .random_state(42);

            let fitted = sampler.fit(&x, &()).unwrap();
            let features = fitted.transform(&x).unwrap();

            assert_eq!(features.shape(), &[2, 20]);
        }
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let sampler1 = QuasiRandomRBFSampler::new(50)
            .gamma(0.5)
            .sequence_type(QuasiRandomSequence::Halton)
            .random_state(123);

        let sampler2 = QuasiRandomRBFSampler::new(50)
            .gamma(0.5)
            .sequence_type(QuasiRandomSequence::Halton)
            .random_state(123);

        let fitted1 = sampler1.fit(&x, &()).unwrap();
        let fitted2 = sampler2.fit(&x, &()).unwrap();

        let features1 = fitted1.transform(&x).unwrap();
        let features2 = fitted2.transform(&x).unwrap();

        for (f1, f2) in features1.iter().zip(features2.iter()) {
            assert_abs_diff_eq!(f1, f2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_halton_sequence() {
        // Test known values of Halton sequence base 2
        assert_abs_diff_eq!(halton_sequence(1, 2), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(halton_sequence(2, 2), 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(halton_sequence(3, 2), 0.75, epsilon = 1e-10);
        assert_abs_diff_eq!(halton_sequence(4, 2), 0.125, epsilon = 1e-10);

        // Test Halton sequence base 3
        assert_abs_diff_eq!(halton_sequence(1, 3), 1.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(halton_sequence(2, 3), 2.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(halton_sequence(3, 3), 1.0 / 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_van_der_corput_sequence() {
        // van der Corput is the same as Halton for one dimension
        for i in 1..10 {
            assert_abs_diff_eq!(
                van_der_corput_sequence(i, 2),
                halton_sequence(i, 2),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_box_muller_transform() {
        let (z1, z2) = box_muller_transform(0.5, 0.5);
        // These should be normally distributed with mean 0
        assert!(z1.abs() < 10.0); // Very loose bounds
        assert!(z2.abs() < 10.0);

        // Test edge case
        let (z1, z2) = box_muller_transform(0.999, 0.001);
        assert!(z1.is_finite());
        assert!(z2.is_finite());
    }

    #[test]
    fn test_get_first_primes() {
        let primes = get_first_primes(10);
        let expected: Vec<usize> = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        assert_eq!(primes, expected);

        let empty = get_first_primes(0);
        let expected_empty: Vec<usize> = vec![];
        assert_eq!(empty, expected_empty);

        let first = get_first_primes(1);
        let expected_first: Vec<usize> = vec![2];
        assert_eq!(first, expected_first);
    }

    #[test]
    fn test_gamma_parameter() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let sampler_low = QuasiRandomRBFSampler::new(100).gamma(0.1).random_state(42);

        let sampler_high = QuasiRandomRBFSampler::new(100).gamma(10.0).random_state(42);

        let fitted_low = sampler_low.fit(&x, &()).unwrap();
        let fitted_high = sampler_high.fit(&x, &()).unwrap();

        let features_low = fitted_low.transform(&x).unwrap();
        let features_high = fitted_high.transform(&x).unwrap();

        // Different gamma values should produce different features
        assert!(features_low != features_high);
    }

    #[test]
    fn test_error_handling() {
        let empty = Array2::<Float>::zeros((0, 0));
        let sampler = QuasiRandomRBFSampler::new(10);

        assert!(sampler.clone().fit(&empty, &()).is_err());

        // Test dimension mismatch
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let fitted = sampler.fit(&x_train, &()).unwrap();
        assert!(fitted.transform(&x_test).is_err());
    }

    #[test]
    fn test_sobol_point_properties() {
        // Test that Sobol points are in [0, 1]
        for i in 0..100 {
            let point = sobol_point(i, 0);
            assert!(point >= 0.0 && point <= 1.0);
        }

        // Test different dimensions give different values
        let point_dim0 = sobol_point(5, 0);
        let point_dim1 = sobol_point(5, 1);
        assert_ne!(point_dim0, point_dim1);
    }
}

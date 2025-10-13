//! Polynomial kernel approximation via Tensor Sketch
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::essentials::Uniform as RandUniform;
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Polynomial kernel approximation via Tensor Sketch
///
/// Implements Tensor Sketch for polynomial kernel approximation:
/// K(x,y) = (gamma * <x,y> + coef0)^degree
///
/// # Parameters
///
/// * `gamma` - Polynomial kernel parameter (default: 1.0)
/// * `degree` - Polynomial degree (default: 2)
/// * `coef0` - Constant term (default: 0.0)
/// * `n_components` - Output dimensionality (default: 100)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::polynomial_count_sketch::PolynomialCountSketch;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0]];
///
/// let poly_sketch = PolynomialCountSketch::new(50);
/// let fitted_sketch = poly_sketch.fit(&X, &()).unwrap();
/// let X_transformed = fitted_sketch.transform(&X).unwrap();
/// assert_eq!(X_transformed.shape(), &[2, 50]);
/// ```
#[derive(Debug, Clone)]
/// PolynomialCountSketch
pub struct PolynomialCountSketch<State = Untrained> {
    /// Polynomial kernel gamma parameter
    pub gamma: Float,
    /// Polynomial degree
    pub degree: u32,
    /// Constant term
    pub coef0: Float,
    /// Output dimensionality
    pub n_components: usize,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted attributes
    index_hash_: Option<Array3<usize>>, // (degree, n_features, 1)
    bit_hash_: Option<Array3<Float>>,   // (degree, n_features, 1)

    _state: PhantomData<State>,
}

impl PolynomialCountSketch<Untrained> {
    /// Create a new Polynomial Count Sketch
    pub fn new(n_components: usize) -> Self {
        Self {
            gamma: 1.0,
            degree: 2,
            coef0: 0.0,
            n_components,
            random_state: None,
            index_hash_: None,
            bit_hash_: None,
            _state: PhantomData,
        }
    }

    /// Set the gamma parameter
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the polynomial degree
    pub fn degree(mut self, degree: u32) -> Self {
        self.degree = degree;
        self
    }

    /// Set the constant term
    pub fn coef0(mut self, coef0: Float) -> Self {
        self.coef0 = coef0;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for PolynomialCountSketch<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for PolynomialCountSketch<Untrained> {
    type Fitted = PolynomialCountSketch<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.degree == 0 {
            return Err(SklearsError::InvalidInput(
                "degree must be positive".to_string(),
            ));
        }

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

        // Generate hash functions for Count Sketch
        let mut index_hash = Array3::zeros((self.degree as usize, n_features, 1));
        let mut bit_hash = Array3::zeros((self.degree as usize, n_features, 1));

        let index_uniform = RandUniform::new(0, self.n_components).unwrap();

        for d in 0..self.degree as usize {
            for j in 0..n_features {
                // Index hash: uniform random in [0, n_components)
                index_hash[[d, j, 0]] = rng.sample(index_uniform);

                // Bit hash: random ±1
                bit_hash[[d, j, 0]] = if rng.gen::<bool>() { 1.0 } else { -1.0 };
            }
        }

        Ok(PolynomialCountSketch {
            gamma: self.gamma,
            degree: self.degree,
            coef0: self.coef0,
            n_components: self.n_components,
            random_state: self.random_state,
            index_hash_: Some(index_hash),
            bit_hash_: Some(bit_hash),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PolynomialCountSketch<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();
        let index_hash = self.index_hash_.as_ref().unwrap();
        let bit_hash = self.bit_hash_.as_ref().unwrap();

        if n_features != index_hash.shape()[1] {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but PolynomialCountSketch was fitted with {} features",
                n_features,
                index_hash.shape()[1]
            )));
        }

        let mut result = Array2::zeros((n_samples, self.n_components));

        // FFT planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.n_components);
        let ifft = planner.plan_fft_inverse(self.n_components);

        for i in 0..n_samples {
            let sample = x.row(i);

            // Add constant term if coef0 != 0
            let extended_sample = if self.coef0 != 0.0 {
                let mut vec = sample.to_vec();
                vec.push(self.coef0.sqrt());
                Array1::from(vec)
            } else {
                sample.to_owned()
            };

            // Compute Count Sketch for each degree
            let mut sketches = Vec::new();

            for d in 0..self.degree as usize {
                let mut sketch = vec![Complex::new(0.0, 0.0); self.n_components];

                for (j, &feature_val) in extended_sample.iter().enumerate() {
                    if j < n_features || self.coef0 != 0.0 {
                        // Include constant term
                        let scaled_val = if j < n_features {
                            self.gamma.sqrt() * feature_val
                        } else {
                            feature_val // Already sqrt(coef0)
                        };

                        let hash_idx = if j < n_features {
                            index_hash[[d, j, 0]]
                        } else {
                            0 // Use index 0 for constant term
                        };

                        let hash_sign = if j < n_features {
                            bit_hash[[d, j, 0]]
                        } else {
                            1.0 // Always positive for constant term
                        };

                        sketch[hash_idx] += Complex::new(hash_sign * scaled_val, 0.0);
                    }
                }

                sketches.push(sketch);
            }

            // Compute FFT of each sketch
            let mut fft_sketches = Vec::new();
            for mut sketch in sketches {
                fft.process(&mut sketch);
                fft_sketches.push(sketch);
            }

            // Compute element-wise product of FFTs (convolution in time domain)
            let mut product = vec![Complex::new(1.0, 0.0); self.n_components];
            for fft_sketch in fft_sketches {
                for (k, val) in fft_sketch.into_iter().enumerate() {
                    product[k] *= val;
                }
            }

            // Compute inverse FFT
            ifft.process(&mut product);

            // Extract real part and normalize
            for (k, val) in product.into_iter().enumerate() {
                result[[i, k]] = val.re / self.n_components as Float;
            }
        }

        Ok(result)
    }
}

impl PolynomialCountSketch<Trained> {
    /// Get the index hash table
    pub fn index_hash(&self) -> &Array3<usize> {
        self.index_hash_.as_ref().unwrap()
    }

    /// Get the bit hash table
    pub fn bit_hash(&self) -> &Array3<Float> {
        self.bit_hash_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_polynomial_count_sketch_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let poly_sketch = PolynomialCountSketch::new(32); // Power of 2 for FFT
        let fitted = poly_sketch.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[2, 32]);
    }

    #[test]
    fn test_polynomial_count_sketch_with_coef0() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let poly_sketch = PolynomialCountSketch::new(16).coef0(1.0).degree(3);
        let fitted = poly_sketch.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[2, 16]);
    }

    #[test]
    fn test_polynomial_count_sketch_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let poly1 = PolynomialCountSketch::new(16).random_state(42);
        let fitted1 = poly1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let poly2 = PolynomialCountSketch::new(16).random_state(42);
        let fitted2 = poly2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        // Results should be identical with same random state
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_polynomial_count_sketch_invalid_degree() {
        let x = array![[1.0, 2.0]];
        let poly_sketch = PolynomialCountSketch::new(16).degree(0);
        let result = poly_sketch.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_polynomial_count_sketch_zero_components() {
        let x = array![[1.0, 2.0]];
        let poly_sketch = PolynomialCountSketch::new(0);
        let result = poly_sketch.fit(&x, &());
        assert!(result.is_err());
    }
}

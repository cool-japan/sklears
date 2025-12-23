//! Random Fourier Features for RBF Kernel Approximation
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use scirs2_core::Cauchy;
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Estimator, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Random Fourier Features for RBF kernel approximation
///
/// Approximates the RBF kernel K(x,y) = exp(-gamma * ||x-y||²) using
/// random Fourier features (Random Kitchen Sinks).
///
/// # Parameters
///
/// * `gamma` - RBF kernel parameter (default: 1.0)
/// * `n_components` - Number of Monte Carlo samples (default: 100)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::RBFSampler;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let rbf = RBFSampler::new(100);
/// let fitted_rbf = rbf.fit(&X, &()).unwrap();
/// let X_transformed = fitted_rbf.transform(&X).unwrap();
/// assert_eq!(X_transformed.shape(), &[3, 100]);
/// ```
#[derive(Debug, Clone)]
/// RBFSampler
pub struct RBFSampler<State = Untrained> {
    /// RBF kernel parameter
    pub gamma: Float,
    /// Number of Monte Carlo samples
    pub n_components: usize,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted attributes
    random_weights_: Option<Array2<Float>>,
    random_offset_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl RBFSampler<Untrained> {
    /// Create a new RBF sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            gamma: 1.0,
            n_components,
            random_state: None,
            random_weights_: None,
            random_offset_: None,
            _state: PhantomData,
        }
    }

    /// Set the gamma parameter
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

impl Estimator for RBFSampler<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for RBFSampler<Untrained> {
    type Fitted = RBFSampler<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.gamma <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "gamma must be positive".to_string(),
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

        // Sample random weights from N(0, 2*gamma)
        let normal = RandNormal::new(0.0, (2.0 * self.gamma).sqrt()).unwrap();
        let mut random_weights = Array2::zeros((n_features, self.n_components));
        for mut col in random_weights.columns_mut() {
            for val in col.iter_mut() {
                *val = rng.sample(normal);
            }
        }

        // Sample random offsets from Uniform(0, 2π)
        let uniform = RandUniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
        let mut random_offset = Array1::zeros(self.n_components);
        for val in random_offset.iter_mut() {
            *val = rng.sample(uniform);
        }

        Ok(RBFSampler {
            gamma: self.gamma,
            n_components: self.n_components,
            random_state: self.random_state,
            random_weights_: Some(random_weights),
            random_offset_: Some(random_offset),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for RBFSampler<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();
        let weights = self.random_weights_.as_ref().unwrap();
        let offset = self.random_offset_.as_ref().unwrap();

        if n_features != weights.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but RBFSampler was fitted with {} features",
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

impl RBFSampler<Trained> {
    /// Get the random weights
    pub fn random_weights(&self) -> &Array2<Float> {
        self.random_weights_.as_ref().unwrap()
    }

    /// Get the random offset
    pub fn random_offset(&self) -> &Array1<Float> {
        self.random_offset_.as_ref().unwrap()
    }
}

/// Laplacian kernel approximation using Random Fourier Features
///
/// Approximates the Laplacian kernel K(x,y) = exp(-gamma * ||x-y||₁) using
/// random Fourier features with Cauchy distribution.
///
/// # Parameters
///
/// * `gamma` - Laplacian kernel parameter (default: 1.0)
/// * `n_components` - Number of Monte Carlo samples (default: 100)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::LaplacianSampler;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let laplacian = LaplacianSampler::new(100);
/// let fitted_laplacian = laplacian.fit(&X, &()).unwrap();
/// let X_transformed = fitted_laplacian.transform(&X).unwrap();
/// assert_eq!(X_transformed.shape(), &[3, 100]);
/// ```
#[derive(Debug, Clone)]
/// LaplacianSampler
pub struct LaplacianSampler<State = Untrained> {
    /// Laplacian kernel parameter
    pub gamma: Float,
    /// Number of Monte Carlo samples
    pub n_components: usize,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted attributes
    random_weights_: Option<Array2<Float>>,
    random_offset_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl LaplacianSampler<Untrained> {
    /// Create a new Laplacian sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            gamma: 1.0,
            n_components,
            random_state: None,
            random_weights_: None,
            random_offset_: None,
            _state: PhantomData,
        }
    }

    /// Set the gamma parameter
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

impl Estimator for LaplacianSampler<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for LaplacianSampler<Untrained> {
    type Fitted = LaplacianSampler<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.gamma <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "gamma must be positive".to_string(),
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

        // Sample random weights from Cauchy distribution (location=0, scale=gamma)
        let cauchy = Cauchy::new(0.0, self.gamma).unwrap();
        let mut random_weights = Array2::zeros((n_features, self.n_components));
        for mut col in random_weights.columns_mut() {
            for val in col.iter_mut() {
                *val = rng.sample(cauchy);
            }
        }

        // Sample random offsets from Uniform(0, 2π)
        let uniform = RandUniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
        let mut random_offset = Array1::zeros(self.n_components);
        for val in random_offset.iter_mut() {
            *val = rng.sample(uniform);
        }

        Ok(LaplacianSampler {
            gamma: self.gamma,
            n_components: self.n_components,
            random_state: self.random_state,
            random_weights_: Some(random_weights),
            random_offset_: Some(random_offset),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for LaplacianSampler<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();
        let weights = self.random_weights_.as_ref().unwrap();
        let offset = self.random_offset_.as_ref().unwrap();

        if n_features != weights.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but LaplacianSampler was fitted with {} features",
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

impl LaplacianSampler<Trained> {
    /// Get the random weights
    pub fn random_weights(&self) -> &Array2<Float> {
        self.random_weights_.as_ref().unwrap()
    }

    /// Get the random offset
    pub fn random_offset(&self) -> &Array1<Float> {
        self.random_offset_.as_ref().unwrap()
    }
}

/// Polynomial kernel approximation using Random Fourier Features
///
/// Approximates the polynomial kernel K(x,y) = (gamma * <x,y> + coef0)^degree using
/// random Fourier features based on the binomial theorem expansion.
///
/// # Parameters
///
/// * `gamma` - Polynomial kernel parameter (default: 1.0)
/// * `coef0` - Independent term in polynomial kernel (default: 1.0)
/// * `degree` - Degree of polynomial kernel (default: 3)
/// * `n_components` - Number of Monte Carlo samples (default: 100)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::rbf_sampler::PolynomialSampler;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let poly = PolynomialSampler::new(100).degree(3).gamma(1.0).coef0(1.0);
/// let fitted_poly = poly.fit(&X, &()).unwrap();
/// let X_transformed = fitted_poly.transform(&X).unwrap();
/// assert_eq!(X_transformed.shape(), &[3, 100]);
/// ```
#[derive(Debug, Clone)]
/// PolynomialSampler
pub struct PolynomialSampler<State = Untrained> {
    /// Polynomial kernel parameter
    pub gamma: Float,
    /// Independent term in polynomial kernel
    pub coef0: Float,
    /// Degree of polynomial kernel
    pub degree: u32,
    /// Number of Monte Carlo samples
    pub n_components: usize,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted attributes
    random_weights_: Option<Array2<Float>>,
    random_offset_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl PolynomialSampler<Untrained> {
    /// Create a new polynomial sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            gamma: 1.0,
            coef0: 1.0,
            degree: 3,
            n_components,
            random_state: None,
            random_weights_: None,
            random_offset_: None,
            _state: PhantomData,
        }
    }

    /// Set the gamma parameter
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the coef0 parameter
    pub fn coef0(mut self, coef0: Float) -> Self {
        self.coef0 = coef0;
        self
    }

    /// Set the degree parameter
    pub fn degree(mut self, degree: u32) -> Self {
        self.degree = degree;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for PolynomialSampler<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for PolynomialSampler<Untrained> {
    type Fitted = PolynomialSampler<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.gamma <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "gamma must be positive".to_string(),
            ));
        }

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

        // For polynomial kernels, we use a different approach:
        // Sample random projections from uniform sphere and scaling factors
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut random_weights = Array2::zeros((n_features, self.n_components));

        for mut col in random_weights.columns_mut() {
            // Sample from standard normal and normalize to get uniform direction on sphere
            for val in col.iter_mut() {
                *val = rng.sample(normal);
            }
            let norm = (col.dot(&col) as Float).sqrt();
            if norm > 1e-12 {
                col /= norm;
            }

            // Scale by gamma
            col *= self.gamma.sqrt();
        }

        // Sample random offsets from Uniform(0, 2π)
        let uniform = RandUniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
        let mut random_offset = Array1::zeros(self.n_components);
        for val in random_offset.iter_mut() {
            *val = rng.sample(uniform);
        }

        Ok(PolynomialSampler {
            gamma: self.gamma,
            coef0: self.coef0,
            degree: self.degree,
            n_components: self.n_components,
            random_state: self.random_state,
            random_weights_: Some(random_weights),
            random_offset_: Some(random_offset),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PolynomialSampler<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();
        let weights = self.random_weights_.as_ref().unwrap();
        let offset = self.random_offset_.as_ref().unwrap();

        if n_features != weights.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but PolynomialSampler was fitted with {} features",
                n_features,
                weights.nrows()
            )));
        }

        // Compute projection: X @ weights + offset
        let projection = x.dot(weights) + offset.view().insert_axis(Axis(0));

        // For polynomial kernels, we apply: (cos(projection) + coef0)^degree
        // This approximates the polynomial kernel through trigonometric expansion
        let normalization = (2.0 / self.n_components as Float).sqrt();
        let result = projection.mapv(|v| {
            let cos_val = v.cos() + self.coef0;
            normalization * cos_val.powf(self.degree as Float)
        });

        Ok(result)
    }
}

impl PolynomialSampler<Trained> {
    /// Get the random weights
    pub fn random_weights(&self) -> &Array2<Float> {
        self.random_weights_.as_ref().unwrap()
    }

    /// Get the random offset
    pub fn random_offset(&self) -> &Array1<Float> {
        self.random_offset_.as_ref().unwrap()
    }

    /// Get the gamma parameter
    pub fn gamma(&self) -> Float {
        self.gamma
    }

    /// Get the coef0 parameter
    pub fn coef0(&self) -> Float {
        self.coef0
    }

    /// Get the degree parameter
    pub fn degree(&self) -> u32 {
        self.degree
    }
}

/// Arc-cosine kernel approximation using Random Fourier Features
///
/// Approximates the arc-cosine kernel which corresponds to infinite-width neural networks.
/// The arc-cosine kernel of degree n is defined as:
/// K_n(x,y) = (1/π) * ||x|| * ||y|| * J_n(θ)
/// where θ is the angle between x and y.
///
/// # Parameters
///
/// * `degree` - Degree of the arc-cosine kernel (0, 1, or 2)
/// * `n_components` - Number of Monte Carlo samples (default: 100)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::rbf_sampler::ArcCosineSampler;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let arc_cosine = ArcCosineSampler::new(100).degree(1);
/// let fitted_arc_cosine = arc_cosine.fit(&X, &()).unwrap();
/// let X_transformed = fitted_arc_cosine.transform(&X).unwrap();
/// assert_eq!(X_transformed.shape(), &[3, 100]);
/// ```
#[derive(Debug, Clone)]
/// ArcCosineSampler
pub struct ArcCosineSampler<State = Untrained> {
    /// Degree of the arc-cosine kernel
    pub degree: u32,
    /// Number of Monte Carlo samples
    pub n_components: usize,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted attributes
    random_weights_: Option<Array2<Float>>,

    _state: PhantomData<State>,
}

impl ArcCosineSampler<Untrained> {
    /// Create a new arc-cosine sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            degree: 1,
            n_components,
            random_state: None,
            random_weights_: None,
            _state: PhantomData,
        }
    }

    /// Set the degree parameter (0, 1, or 2)
    pub fn degree(mut self, degree: u32) -> Self {
        self.degree = degree;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for ArcCosineSampler<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for ArcCosineSampler<Untrained> {
    type Fitted = ArcCosineSampler<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.degree > 2 {
            return Err(SklearsError::InvalidInput(
                "degree must be 0, 1, or 2".to_string(),
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

        // Sample random weights from standard normal distribution
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut random_weights = Array2::zeros((n_features, self.n_components));

        for mut col in random_weights.columns_mut() {
            for val in col.iter_mut() {
                *val = rng.sample(normal);
            }
        }

        Ok(ArcCosineSampler {
            degree: self.degree,
            n_components: self.n_components,
            random_state: self.random_state,
            random_weights_: Some(random_weights),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for ArcCosineSampler<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();
        let weights = self.random_weights_.as_ref().unwrap();

        if n_features != weights.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but ArcCosineSampler was fitted with {} features",
                n_features,
                weights.nrows()
            )));
        }

        // Compute projection: X @ weights
        let projection = x.dot(weights);

        // Apply activation function based on degree
        let normalization = (2.0 / self.n_components as Float).sqrt();
        let result = match self.degree {
            0 => {
                // ReLU: max(0, x)
                projection.mapv(|v| normalization * v.max(0.0))
            }
            1 => {
                // Sigmoid-like: x * I(x > 0)
                projection.mapv(|v| if v > 0.0 { normalization * v } else { 0.0 })
            }
            2 => {
                // Quadratic: x² * I(x > 0)
                projection.mapv(|v| if v > 0.0 { normalization * v * v } else { 0.0 })
            }
            _ => unreachable!("degree validation should prevent this"),
        };

        Ok(result)
    }
}

impl ArcCosineSampler<Trained> {
    /// Get the random weights
    pub fn random_weights(&self) -> &Array2<Float> {
        self.random_weights_.as_ref().unwrap()
    }

    /// Get the degree parameter
    pub fn degree(&self) -> u32 {
        self.degree
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_rbf_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let rbf = RBFSampler::new(50).gamma(0.1);
        let fitted = rbf.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[3, 50]);

        // Check that values are in reasonable range for cosine function
        for val in x_transformed.iter() {
            assert!(val.abs() <= 2.0); // sqrt(2) * 1 is the max possible value
        }
    }

    #[test]
    fn test_rbf_sampler_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let rbf1 = RBFSampler::new(10).random_state(42);
        let fitted1 = rbf1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let rbf2 = RBFSampler::new(10).random_state(42);
        let fitted2 = rbf2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        // Results should be identical with same random state
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rbf_sampler_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0],];

        let x_test = array![
            [1.0, 2.0, 3.0], // Wrong number of features
        ];

        let rbf = RBFSampler::new(10);
        let fitted = rbf.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);

        assert!(result.is_err());
    }

    #[test]
    fn test_rbf_sampler_invalid_gamma() {
        let x = array![[1.0, 2.0]];
        let rbf = RBFSampler::new(10).gamma(-1.0);
        let result = rbf.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_rbf_sampler_zero_components() {
        let x = array![[1.0, 2.0]];
        let rbf = RBFSampler::new(0);
        let result = rbf.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_laplacian_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let laplacian = LaplacianSampler::new(50).gamma(0.1);
        let fitted = laplacian.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[3, 50]);

        // Check that values are in reasonable range for cosine function
        for val in x_transformed.iter() {
            assert!(val.abs() <= 2.0); // sqrt(2) * 1 is the max possible value
        }
    }

    #[test]
    fn test_laplacian_sampler_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let laplacian1 = LaplacianSampler::new(10).random_state(42);
        let fitted1 = laplacian1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let laplacian2 = LaplacianSampler::new(10).random_state(42);
        let fitted2 = laplacian2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        // Results should be identical with same random state
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_laplacian_sampler_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0],];

        let x_test = array![
            [1.0, 2.0, 3.0], // Wrong number of features
        ];

        let laplacian = LaplacianSampler::new(10);
        let fitted = laplacian.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);

        assert!(result.is_err());
    }

    #[test]
    fn test_laplacian_sampler_invalid_gamma() {
        let x = array![[1.0, 2.0]];
        let laplacian = LaplacianSampler::new(10).gamma(-1.0);
        let result = laplacian.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_laplacian_sampler_zero_components() {
        let x = array![[1.0, 2.0]];
        let laplacian = LaplacianSampler::new(0);
        let result = laplacian.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_polynomial_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let poly = PolynomialSampler::new(50).degree(3).gamma(1.0).coef0(1.0);
        let fitted = poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[3, 50]);

        // Check that values are in reasonable range
        for val in x_transformed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_polynomial_sampler_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let poly1 = PolynomialSampler::new(10).degree(2).random_state(42);
        let fitted1 = poly1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let poly2 = PolynomialSampler::new(10).degree(2).random_state(42);
        let fitted2 = poly2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        // Results should be identical with same random state
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_polynomial_sampler_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0],];

        let x_test = array![
            [1.0, 2.0, 3.0], // Wrong number of features
        ];

        let poly = PolynomialSampler::new(10);
        let fitted = poly.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);

        assert!(result.is_err());
    }

    #[test]
    fn test_polynomial_sampler_invalid_gamma() {
        let x = array![[1.0, 2.0]];
        let poly = PolynomialSampler::new(10).gamma(-1.0);
        let result = poly.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_polynomial_sampler_zero_degree() {
        let x = array![[1.0, 2.0]];
        let poly = PolynomialSampler::new(10).degree(0);
        let result = poly.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_polynomial_sampler_zero_components() {
        let x = array![[1.0, 2.0]];
        let poly = PolynomialSampler::new(0);
        let result = poly.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_polynomial_sampler_different_degrees() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        // Test degree 1
        let poly1 = PolynomialSampler::new(10).degree(1);
        let fitted1 = poly1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();
        assert_eq!(result1.shape(), &[2, 10]);

        // Test degree 5
        let poly5 = PolynomialSampler::new(10).degree(5);
        let fitted5 = poly5.fit(&x, &()).unwrap();
        let result5 = fitted5.transform(&x).unwrap();
        assert_eq!(result5.shape(), &[2, 10]);
    }

    #[test]
    fn test_arc_cosine_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let arc_cosine = ArcCosineSampler::new(50).degree(1);
        let fitted = arc_cosine.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[3, 50]);

        // Check that values are non-negative (due to ReLU-like activation)
        for val in x_transformed.iter() {
            assert!(val >= &0.0);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_arc_cosine_sampler_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let arc1 = ArcCosineSampler::new(10).degree(1).random_state(42);
        let fitted1 = arc1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let arc2 = ArcCosineSampler::new(10).degree(1).random_state(42);
        let fitted2 = arc2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        // Results should be identical with same random state
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_arc_cosine_sampler_different_degrees() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        // Test degree 0 (ReLU)
        let arc0 = ArcCosineSampler::new(10).degree(0);
        let fitted0 = arc0.fit(&x, &()).unwrap();
        let result0 = fitted0.transform(&x).unwrap();
        assert_eq!(result0.shape(), &[2, 10]);

        // Test degree 1 (Linear ReLU)
        let arc1 = ArcCosineSampler::new(10).degree(1);
        let fitted1 = arc1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();
        assert_eq!(result1.shape(), &[2, 10]);

        // Test degree 2 (Quadratic ReLU)
        let arc2 = ArcCosineSampler::new(10).degree(2);
        let fitted2 = arc2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();
        assert_eq!(result2.shape(), &[2, 10]);
    }

    #[test]
    fn test_arc_cosine_sampler_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0],];

        let x_test = array![
            [1.0, 2.0, 3.0], // Wrong number of features
        ];

        let arc_cosine = ArcCosineSampler::new(10);
        let fitted = arc_cosine.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);

        assert!(result.is_err());
    }

    #[test]
    fn test_arc_cosine_sampler_invalid_degree() {
        let x = array![[1.0, 2.0]];
        let arc_cosine = ArcCosineSampler::new(10).degree(3);
        let result = arc_cosine.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_arc_cosine_sampler_zero_components() {
        let x = array![[1.0, 2.0]];
        let arc_cosine = ArcCosineSampler::new(0);
        let result = arc_cosine.fit(&x, &());
        assert!(result.is_err());
    }
}

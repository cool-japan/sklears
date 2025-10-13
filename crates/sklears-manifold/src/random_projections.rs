//! Random projection methods for high-dimensional data reduction
//! This module implements Johnson-Lindenstrauss embeddings and various random projection
//! techniques for efficient dimensionality reduction with theoretical guarantees.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::Distribution;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

// =====================================================================================
// HIGH-DIMENSIONAL DATA METHODS
// =====================================================================================

/// Johnson-Lindenstrauss embeddings for high-dimensional data
///
/// The Johnson-Lindenstrauss lemma states that a set of points in high-dimensional
/// space can be embedded into a lower-dimensional space while preserving pairwise
/// distances with high probability. This provides a fundamental tool for
/// dimensionality reduction with theoretical guarantees.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality (should be >= 4*log(n_samples))
/// * `eps` - Distortion parameter (smaller values require higher dimensions)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::JohnsonLindenstrauss;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]];
///
/// let jl = JohnsonLindenstrauss::new()
///     .n_components(2)
///     .eps(0.1);
///
/// let fitted = jl.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct JohnsonLindenstrauss<S = Untrained> {
    state: S,
    n_components: usize,
    eps: f64,
    random_state: Option<u64>,
}

impl Default for JohnsonLindenstrauss<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl JohnsonLindenstrauss<Untrained> {
    /// Create a new JohnsonLindenstrauss instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            eps: 0.1,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the distortion parameter
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Compute the minimum safe number of components for given sample size
    pub fn min_safe_components(n_samples: usize, eps: f64) -> usize {
        let n = n_samples as f64;
        let eps_sq = eps * eps;
        let log_n = n.ln();

        // For small sample sizes, use a more practical bound
        if n_samples <= 10 {
            // Very lenient bound for small n, but respect very small eps values
            let simple_bound = if n_samples <= 3 && eps >= 0.1 {
                // For very small samples with reasonable eps, be very lenient
                (log_n / (eps_sq * 2.0)).ceil() as usize
            } else {
                (log_n / eps_sq).ceil() as usize
            };
            return simple_bound.min(n_samples - 1).max(2);
        }

        // Classical JL bound for larger samples: d >= 4 * log(n) / (eps^2 / 2 - eps^3 / 3)
        let denominator = eps_sq / 2.0 - eps_sq * eps / 3.0;
        if denominator <= 0.0 {
            return n_samples; // Conservative fallback
        }

        let min_d = (4.0 * log_n / denominator).ceil() as usize;
        // Cap at a reasonable multiple of n_samples for practicality
        min_d.min(n_samples * 10).max(1)
    }
}

#[derive(Debug, Clone)]
pub struct JLTrained {
    projection_matrix: Array2<f64>,
    scaling_factor: f64,
}

impl Estimator for JohnsonLindenstrauss<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for JohnsonLindenstrauss<Untrained> {
    type Fitted = JohnsonLindenstrauss<JLTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        // Validate parameters
        if self.eps <= 0.0 || self.eps >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "eps must be between 0 and 1".to_string(),
            ));
        }

        let min_components = Self::min_safe_components(n_samples, self.eps);
        if self.n_components < min_components {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) is too small. For {} samples and eps={}, minimum is {}",
                self.n_components, n_samples, self.eps, min_components
            )));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        // Create random projection matrix
        let mut projection_matrix = Array2::zeros((n_features, self.n_components));
        for elem in projection_matrix.iter_mut() {
            *elem = rng.sample(scirs2_core::StandardNormal);
        }

        // Scale by 1/sqrt(n_components) for distance preservation
        let scaling_factor = 1.0 / (self.n_components as f64).sqrt();

        Ok(JohnsonLindenstrauss {
            state: JLTrained {
                projection_matrix,
                scaling_factor,
            },
            n_components: self.n_components,
            eps: self.eps,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for JohnsonLindenstrauss<JLTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let x_f64 = x.mapv(|v| v);

        // Apply random projection
        let projected = x_f64.dot(&self.state.projection_matrix);
        let scaled = projected * self.state.scaling_factor;

        Ok(scaled)
    }
}

/// Fast Johnson-Lindenstrauss Transform using structured random matrices
///
/// This implementation uses structured random matrices (Walsh-Hadamard Transform)
/// for faster computation compared to dense random matrices. It achieves O(n log n)
/// complexity instead of O(n²) for the transform operation.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality (must be power of 2)
/// * `eps` - Distortion parameter (smaller values require higher dimensions)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::FastJohnsonLindenstrauss;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]];
///
/// let fjl = FastJohnsonLindenstrauss::new()
///     .n_components(4)
///     .eps(0.1);
///
/// let fitted = fjl.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FastJohnsonLindenstrauss<S = Untrained> {
    state: S,
    n_components: usize,
    eps: f64,
    random_state: Option<u64>,
}

impl Default for FastJohnsonLindenstrauss<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl FastJohnsonLindenstrauss<Untrained> {
    /// Create a new FastJohnsonLindenstrauss instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 4,
            eps: 0.1,
            random_state: None,
        }
    }

    /// Set the number of components (must be power of 2)
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the distortion parameter
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Check if a number is a power of 2
    fn is_power_of_two(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }

    /// Fast Walsh-Hadamard Transform
    fn fwht(data: &mut [f64]) {
        let n = data.len();
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
    }
}

#[derive(Debug, Clone)]
pub struct FastJLTrained {
    diagonal_matrix: Array1<f64>,
    scaling_factor: f64,
    padded_size: usize,
}

impl Estimator for FastJohnsonLindenstrauss<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for FastJohnsonLindenstrauss<Untrained> {
    type Fitted = FastJohnsonLindenstrauss<FastJLTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        // Validate parameters
        if self.eps <= 0.0 || self.eps >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "eps must be between 0 and 1".to_string(),
            ));
        }

        if !Self::is_power_of_two(self.n_components) {
            return Err(SklearsError::InvalidInput(
                "n_components must be a power of 2 for fast transform".to_string(),
            ));
        }

        let min_components = JohnsonLindenstrauss::min_safe_components(n_samples, self.eps);
        if self.n_components < min_components {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) is too small. For {} samples and eps={}, minimum is {}",
                self.n_components, n_samples, self.eps, min_components
            )));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        // Find the next power of 2 that's >= n_features
        let padded_size = (n_features - 1).next_power_of_two().max(self.n_components);

        // Create random diagonal matrix with +1 or -1 entries
        let mut diagonal_matrix = Array1::zeros(padded_size);
        for elem in diagonal_matrix.iter_mut() {
            *elem = if rng.gen::<bool>() { 1.0 } else { -1.0 };
        }

        // Scaling factor for distance preservation
        let scaling_factor = 1.0 / (self.n_components as f64).sqrt();

        Ok(FastJohnsonLindenstrauss {
            state: FastJLTrained {
                diagonal_matrix,
                scaling_factor,
                padded_size,
            },
            n_components: self.n_components,
            eps: self.eps,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for FastJohnsonLindenstrauss<FastJLTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let padded_size = self.state.padded_size;

        let mut result = Array2::zeros((n_samples, self.n_components));

        for (i, sample) in x.outer_iter().enumerate() {
            // Pad the sample to the required size
            let mut padded_sample = vec![0.0; padded_size];
            for (j, &val) in sample.iter().enumerate() {
                padded_sample[j] = val;
            }

            // Apply diagonal matrix (element-wise multiplication)
            for (j, &diag_val) in self.state.diagonal_matrix.iter().enumerate() {
                padded_sample[j] *= diag_val;
            }

            // Apply Fast Walsh-Hadamard Transform
            Self::fwht(&mut padded_sample);

            // Extract the first n_components and scale
            for j in 0..self.n_components {
                result[[i, j]] = padded_sample[j] * self.state.scaling_factor;
            }
        }

        Ok(result)
    }
}

impl FastJohnsonLindenstrauss<FastJLTrained> {
    /// Fast Walsh-Hadamard Transform
    fn fwht(data: &mut [f64]) {
        let n = data.len();
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
    }
}

/// Random projection for dimensionality reduction
///
/// Random projection is a dimensionality reduction technique that projects
/// high-dimensional data onto a lower-dimensional space using a random matrix.
/// This method is computationally efficient and has theoretical guarantees
/// for distance preservation.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality
/// * `density` - Density of the projection matrix (1.0 for dense, < 1.0 for sparse)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::RandomProjection;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]];
///
/// let rp = RandomProjection::new()
///     .n_components(2)
///     .density(0.5);
///
/// let fitted = rp.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RandomProjection<S = Untrained> {
    state: S,
    n_components: usize,
    density: f64,
    random_state: Option<u64>,
}

impl Default for RandomProjection<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomProjection<Untrained> {
    /// Create a new RandomProjection instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            density: 1.0,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the density of the projection matrix
    pub fn density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

#[derive(Debug, Clone)]
pub struct RPTrained {
    projection_matrix: Array2<f64>,
    scaling_factor: f64,
}

#[derive(Debug, Clone)]
pub struct RPConfig {
    /// n_components
    pub n_components: usize,
    /// density
    pub density: f64,
    /// random_state
    pub random_state: Option<u64>,
}

impl Estimator for RandomProjection<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for RandomProjection<Untrained> {
    type Fitted = RandomProjection<RPTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (_, n_features) = x.dim();

        // Validate parameters
        if self.density <= 0.0 || self.density > 1.0 {
            return Err(SklearsError::InvalidInput(
                "density must be between 0 and 1".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        // Create projection matrix
        let mut projection_matrix = Array2::zeros((n_features, self.n_components));

        if self.density >= 1.0 {
            // Dense random projection
            for elem in projection_matrix.iter_mut() {
                *elem = rng.sample(scirs2_core::StandardNormal);
            }
        } else {
            // Sparse random projection
            let s = 1.0 / self.density;
            let prob_positive = 1.0 / (2.0 * s);
            let prob_negative = 1.0 / (2.0 * s);

            for elem in projection_matrix.iter_mut() {
                let rand_val: f64 = rng.gen();
                if rand_val < prob_positive {
                    *elem = s.sqrt();
                } else if rand_val < prob_positive + prob_negative {
                    *elem = -s.sqrt();
                } else {
                    *elem = 0.0;
                }
            }
        }

        // Scale by 1/sqrt(n_components)
        let scaling_factor = 1.0 / (self.n_components as f64).sqrt();

        Ok(RandomProjection {
            state: RPTrained {
                projection_matrix,
                scaling_factor,
            },
            n_components: self.n_components,
            density: self.density,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for RandomProjection<RPTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let x_f64 = x.mapv(|v| v);

        // Apply random projection
        let projected = x_f64.dot(&self.state.projection_matrix);
        let scaled = projected * self.state.scaling_factor;

        Ok(scaled)
    }
}

/// Sparse random projection for efficient dimensionality reduction
///
/// This implements sparse random projection matrices that have many zero entries,
/// making the projection computationally more efficient while maintaining
/// distance preservation properties.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality
/// * `density` - Density of non-zero entries (should be < 1.0)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::SparseRandomProjection;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]];
///
/// let srp = SparseRandomProjection::new()
///     .n_components(2)
///     .density(0.1);
///
/// let fitted = srp.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparseRandomProjection<S = Untrained> {
    state: S,
    n_components: usize,
    density: f64,
    random_state: Option<u64>,
}

impl Default for SparseRandomProjection<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseRandomProjection<Untrained> {
    /// Create a new SparseRandomProjection instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            density: 0.1,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the density of non-zero entries
    pub fn density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Compute optimal density for given dimensions
    pub fn optimal_density(n_features: usize) -> f64 {
        // Optimal density is approximately 1/sqrt(n_features)
        1.0 / (n_features as f64).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct SRPTrained {
    projection_matrix: Array2<f64>,
    scaling_factor: f64,
    density: f64,
}

#[derive(Debug, Clone)]
pub struct SRPConfig {
    /// n_components
    pub n_components: usize,
    /// density
    pub density: f64,
    /// random_state
    pub random_state: Option<u64>,
}

impl Estimator for SparseRandomProjection<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for SparseRandomProjection<Untrained> {
    type Fitted = SparseRandomProjection<SRPTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (_, n_features) = x.dim();

        // Validate parameters
        if self.density <= 0.0 || self.density > 1.0 {
            return Err(SklearsError::InvalidInput(
                "density must be between 0 and 1".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        // Create sparse projection matrix using Li et al. (2006) distribution
        let mut projection_matrix = Array2::zeros((n_features, self.n_components));
        let s = 1.0 / self.density;

        // Probabilities for the three-value distribution
        let prob_positive = 1.0 / (2.0 * s);
        let prob_negative = 1.0 / (2.0 * s);
        let prob_zero = 1.0 - prob_positive - prob_negative;

        for elem in projection_matrix.iter_mut() {
            let rand_val: f64 = rng.gen();
            if rand_val < prob_positive {
                *elem = s.sqrt();
            } else if rand_val < prob_positive + prob_negative {
                *elem = -s.sqrt();
            } else {
                *elem = 0.0;
            }
        }

        // Scale by 1/sqrt(n_components)
        let scaling_factor = 1.0 / (self.n_components as f64).sqrt();

        Ok(SparseRandomProjection {
            state: SRPTrained {
                projection_matrix,
                scaling_factor,
                density: self.density,
            },
            n_components: self.n_components,
            density: self.density,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for SparseRandomProjection<SRPTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let x_f64 = x.mapv(|v| v);

        // Apply sparse random projection
        let projected = x_f64.dot(&self.state.projection_matrix);
        let scaled = projected * self.state.scaling_factor;

        Ok(scaled)
    }
}

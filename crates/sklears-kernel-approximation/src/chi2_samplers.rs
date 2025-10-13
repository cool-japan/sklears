//! Chi-squared kernel approximation methods
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::Uniform as RandUniform;
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use scirs2_core::random::{thread_rng, Rng, SeedableRng};
/// Additive Chi-Squared Kernel Approximation
///
/// Approximates the additive chi-squared kernel: K(x,y) = Σᵢ (2xᵢyᵢ)/(xᵢ+yᵢ)
/// Used with histogram data in computer vision. This is a stateless transformer.
///
/// # Parameters
///
/// * `sample_steps` - Number of sampling points (default: 2)
/// * `sample_interval` - Sampling interval (auto-computed if None)
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::AdditiveChi2Sampler;
/// use sklears_core::traits::Transform;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0]];
///
/// let chi2 = AdditiveChi2Sampler::new(2);
/// let X_transformed = chi2.transform(&X).unwrap();
/// assert_eq!(X_transformed.shape(), &[2, 6]); // 2 features * 3 = 6
/// ```
#[derive(Debug, Clone)]
/// AdditiveChi2Sampler
pub struct AdditiveChi2Sampler {
    /// Number of sampling points
    pub sample_steps: usize,
    /// Sampling interval
    pub sample_interval: Float,
}

impl AdditiveChi2Sampler {
    /// Create a new Additive Chi2 sampler
    pub fn new(sample_steps: usize) -> Self {
        let sample_interval = match sample_steps {
            1 => 0.8,
            2 => 0.5,
            3 => 0.4,
            _ => 0.5, // Default fallback
        };

        Self {
            sample_steps,
            sample_interval,
        }
    }

    /// Set the sample interval
    pub fn sample_interval(mut self, interval: Float) -> Self {
        self.sample_interval = interval;
        self
    }
}

impl Transform<Array2<Float>, Array2<Float>> for AdditiveChi2Sampler {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        // Check for non-negative values
        for val in x.iter() {
            if *val < 0.0 {
                return Err(SklearsError::InvalidInput(
                    "Additive chi2 kernel requires non-negative features".to_string(),
                ));
            }
        }

        let n_output_features = n_features * (2 * self.sample_steps - 1);
        let mut result = Array2::zeros((n_samples, n_output_features));

        for i in 0..n_samples {
            let mut feature_idx = 0;

            for j in 0..n_features {
                let x_val = x[[i, j]];

                // First component: sqrt(X * sample_interval)
                result[[i, feature_idx]] = (x_val * self.sample_interval).sqrt();
                feature_idx += 1;

                // Additional components: factor * cos/sin(k * log(X) * sample_interval)
                if x_val > 0.0 {
                    let log_x = x_val.ln();

                    for k in 1..self.sample_steps {
                        let k_float = k as Float;
                        let arg = k_float * log_x * self.sample_interval;
                        let factor = (2.0 * x_val * self.sample_interval
                            / (std::f64::consts::PI * k_float * self.sample_interval).cosh())
                        .sqrt();

                        // Cosine component
                        result[[i, feature_idx]] = factor * arg.cos();
                        feature_idx += 1;

                        // Sine component
                        result[[i, feature_idx]] = factor * arg.sin();
                        feature_idx += 1;
                    }
                } else {
                    // For x_val == 0, set remaining components to 0
                    feature_idx += 2 * (self.sample_steps - 1);
                }
            }
        }

        Ok(result)
    }
}

/// Skewed Chi-Squared Kernel Approximation
///
/// Approximates the skewed chi-squared kernel using Monte Carlo sampling.
/// K(x,y) = ∏ᵢ (2√(xᵢ+c)√(yᵢ+c))/(xᵢ+yᵢ+2c)
///
/// # Parameters
///
/// * `skewedness` - The "c" parameter (default: 1.0)
/// * `n_components` - Number of Monte Carlo samples (default: 100)
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::SkewedChi2Sampler;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0]];
///
/// let skewed_chi2 = SkewedChi2Sampler::new(50);
/// let fitted_chi2 = skewed_chi2.fit(&X, &()).unwrap();
/// let X_transformed = fitted_chi2.transform(&X).unwrap();
/// assert_eq!(X_transformed.shape(), &[2, 50]);
/// ```
#[derive(Debug, Clone)]
/// SkewedChi2Sampler
pub struct SkewedChi2Sampler<State = Untrained> {
    /// Skewedness parameter
    pub skewedness: Float,
    /// Number of Monte Carlo samples
    pub n_components: usize,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted attributes
    random_weights_: Option<Array2<Float>>,
    random_offset_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl SkewedChi2Sampler<Untrained> {
    /// Create a new Skewed Chi2 sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            skewedness: 1.0,
            n_components,
            random_state: None,
            random_weights_: None,
            random_offset_: None,
            _state: PhantomData,
        }
    }

    /// Set the skewedness parameter
    pub fn skewedness(mut self, skewedness: Float) -> Self {
        self.skewedness = skewedness;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for SkewedChi2Sampler<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for SkewedChi2Sampler<Untrained> {
    type Fitted = SkewedChi2Sampler<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.skewedness <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "skewedness must be positive".to_string(),
            ));
        }

        // Check that all values > -skewedness
        for val in x.iter() {
            if *val <= -self.skewedness {
                return Err(SklearsError::InvalidInput(format!(
                    "All values must be > -skewedness ({})",
                    -self.skewedness
                )));
            }
        }

        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        // Sample random weights from inverse CDF of sech distribution
        let uniform = RandUniform::new(0.0, 1.0).unwrap();
        let mut weights = Array2::zeros((n_features, self.n_components));

        for mut col in weights.columns_mut() {
            for weight in col.iter_mut() {
                let u = rng.sample(uniform);
                // Inverse CDF of sech: (1/π) * log(tan(π/2 * u))
                *weight =
                    (1.0 / std::f64::consts::PI) * ((std::f64::consts::PI / 2.0 * u).tan()).ln();
            }
        }

        // Sample random offsets from Uniform(0, 2π)
        let offset_uniform = RandUniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
        let mut random_offset = Array1::zeros(self.n_components);
        for val in random_offset.iter_mut() {
            *val = rng.sample(offset_uniform);
        }

        Ok(SkewedChi2Sampler {
            skewedness: self.skewedness,
            n_components: self.n_components,
            random_state: self.random_state,
            random_weights_: Some(weights),
            random_offset_: Some(random_offset),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for SkewedChi2Sampler<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();
        let weights = self.random_weights_.as_ref().unwrap();
        let offset = self.random_offset_.as_ref().unwrap();

        if n_features != weights.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but SkewedChi2Sampler was fitted with {} features",
                n_features,
                weights.nrows()
            )));
        }

        // Check input validity
        for val in x.iter() {
            if *val <= -self.skewedness {
                return Err(SklearsError::InvalidInput(format!(
                    "All values must be > -skewedness ({})",
                    -self.skewedness
                )));
            }
        }

        // Transform: log(X + skewedness)
        let x_shifted = x.mapv(|v| (v + self.skewedness).ln());

        // Compute projection and apply cosine
        let projection = x_shifted.dot(weights) + &offset.view().insert_axis(Axis(0));
        let normalization = (2.0 / self.n_components as Float).sqrt();
        let result = projection.mapv(|v| normalization * v.cos());

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_additive_chi2_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let chi2 = AdditiveChi2Sampler::new(2);
        let x_transformed = chi2.transform(&x).unwrap();

        // 2 features * (2*2-1) = 6 output features
        assert_eq!(x_transformed.shape(), &[2, 6]);

        // Check non-negativity of first components (sqrt values)
        assert!(x_transformed[[0, 0]] >= 0.0);
        assert!(x_transformed[[0, 3]] >= 0.0);
    }

    #[test]
    fn test_additive_chi2_sampler_negative_input() {
        let x = array![
            [1.0, -2.0], // Negative value
        ];

        let chi2 = AdditiveChi2Sampler::new(2);
        let result = chi2.transform(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_skewed_chi2_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let skewed_chi2 = SkewedChi2Sampler::new(50);
        let fitted = skewed_chi2.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[2, 50]);

        // Check that values are in reasonable range
        for val in x_transformed.iter() {
            assert!(val.abs() <= 2.0);
        }
    }

    #[test]
    fn test_skewed_chi2_sampler_invalid_skewedness() {
        let x = array![[1.0, 2.0]];
        let skewed_chi2 = SkewedChi2Sampler::new(10).skewedness(-1.0);
        let result = skewed_chi2.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_skewed_chi2_sampler_input_validation() {
        let x_train = array![[1.0, 2.0]];
        let x_test = array![[-1.5, 2.0]]; // < -skewedness when skewedness=1.0

        let skewed_chi2 = SkewedChi2Sampler::new(10);
        let fitted = skewed_chi2.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);
        assert!(result.is_err());
    }
}

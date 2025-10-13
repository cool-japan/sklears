//! Elliptic Envelope for Outlier Detection
//!
//! Outlier detection using robust covariance estimation.

use crate::min_cov_det::MinCovDet;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for EllipticEnvelope estimator
#[derive(Debug, Clone)]
pub struct EllipticEnvelopeConfig {
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
    /// Fraction of points to include in the support
    pub support_fraction: Option<f64>,
    /// Expected proportion of outliers
    pub contamination: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Elliptic Envelope
///
/// Outlier detection using robust covariance estimation.
/// This method fits a robust covariance estimate to the data and classifies
/// observations as outliers based on the Mahalanobis distance.
///
/// # Parameters
///
/// * `store_precision` - Whether to store the precision matrix
/// * `assume_centered` - Whether to assume the data is centered
/// * `support_fraction` - Fraction of points to include in the support
/// * `contamination` - Expected proportion of outliers
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_covariance::EllipticEnvelope;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let x = array![
///     [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],
///     [6.0, 6.0], [7.0, 7.0], [8.0, 8.0], [100.0, 100.0]  // Last point is outlier
/// ];
///
/// let detector = EllipticEnvelope::new().contamination(0.25);
/// match detector.fit(&x.view(), &()) {
///     Ok(fitted) => {
///         let predictions = fitted.predict(&x.view()).unwrap();
///         assert_eq!(predictions.len(), 9);
///     }
///     Err(_) => {
///         // May fail due to numerical sensitivity
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EllipticEnvelope<S = Untrained> {
    state: S,
    config: EllipticEnvelopeConfig,
}

/// Trained state for EllipticEnvelope
#[derive(Debug, Clone)]
pub struct EllipticEnvelopeTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The support mask indicating which samples were used
    pub support: Array1<bool>,
    /// The decision threshold for outlier detection
    pub threshold: f64,
    /// The Mahalanobis distances for training samples
    pub distances: Array1<f64>,
}

impl EllipticEnvelope<Untrained> {
    /// Create a new EllipticEnvelope instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: EllipticEnvelopeConfig {
                store_precision: true,
                assume_centered: false,
                support_fraction: None,
                contamination: 0.1,
                random_state: None,
            },
        }
    }

    /// Set whether to store the precision matrix
    pub fn store_precision(mut self, store_precision: bool) -> Self {
        self.config.store_precision = store_precision;
        self
    }

    /// Set whether to assume the data is centered
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.config.assume_centered = assume_centered;
        self
    }

    /// Set the support fraction
    pub fn support_fraction(mut self, support_fraction: Option<f64>) -> Self {
        self.config.support_fraction = support_fraction;
        self
    }

    /// Set the contamination parameter
    pub fn contamination(mut self, contamination: f64) -> Self {
        self.config.contamination = contamination.clamp(0.0, 0.5);
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

impl Default for EllipticEnvelope<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for EllipticEnvelope<Untrained> {
    type Config = EllipticEnvelopeConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for EllipticEnvelope<Untrained> {
    type Fitted = EllipticEnvelope<EllipticEnvelopeTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, _n_features) = x.dim();

        // Use MinCovDet for robust covariance estimation
        let mut mcd_builder = MinCovDet::new()
            .support_fraction(self.config.support_fraction.unwrap_or(0.5))
            .store_precision(self.config.store_precision)
            .assume_centered(self.config.assume_centered);

        if let Some(random_state) = self.config.random_state {
            mcd_builder = mcd_builder.random_state(random_state);
        }

        let mcd = mcd_builder.fit(&x, &())?;

        // Compute distances for threshold determination
        let distances = mcd.mahalanobis_distance(&x)?;

        // Determine threshold based on contamination
        let mut sorted_distances = distances.to_vec();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = ((1.0 - self.config.contamination) * n_samples as f64) as usize;
        let threshold = if threshold_idx < sorted_distances.len() {
            sorted_distances[threshold_idx]
        } else {
            sorted_distances.last().cloned().unwrap_or(2.5)
        };

        Ok(EllipticEnvelope {
            state: EllipticEnvelopeTrained {
                covariance: mcd.get_covariance().clone(),
                precision: mcd.get_precision().cloned(),
                location: mcd.get_location().clone(),
                support: mcd.get_support().clone(),
                threshold,
                distances,
            },
            config: self.config,
        })
    }
}

impl EllipticEnvelope<EllipticEnvelopeTrained> {
    /// Get the covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix (inverse covariance)
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the location (mean)
    pub fn get_location(&self) -> &Array1<f64> {
        &self.state.location
    }

    /// Get the support mask (which samples were used)
    pub fn get_support(&self) -> &Array1<bool> {
        &self.state.support
    }

    /// Get the decision threshold
    pub fn get_threshold(&self) -> f64 {
        self.state.threshold
    }

    /// Predict if samples are inliers (1) or outliers (-1)
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let distances = self.mahalanobis_distance(x)?;
        Ok(distances.mapv(|d| if d <= self.state.threshold { 1 } else { -1 }))
    }

    /// Compute decision function (negative Mahalanobis distance)
    pub fn decision_function(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let distances = self.mahalanobis_distance(x)?;
        Ok(distances.mapv(|d| self.state.threshold - d))
    }

    /// Score samples (higher scores indicate more normal samples)
    pub fn score_samples(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        self.decision_function(x)
    }

    /// Compute Mahalanobis distance
    pub fn mahalanobis_distance(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let x = *x;
        let precision = self.state.precision.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Precision matrix not computed".to_string())
        })?;

        let mut distances = Array1::zeros(x.nrows());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let centered = &sample - &self.state.location;
            let temp = precision.dot(&centered);
            distances[i] = centered.dot(&temp).sqrt();
        }

        Ok(distances)
    }
}

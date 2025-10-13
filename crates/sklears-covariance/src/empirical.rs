//! Empirical Covariance Estimator

use crate::utils::{matrix_inverse, validate_covariance_matrix, CovarianceProperties};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Empirical Covariance Estimator
///
/// Computes the maximum likelihood covariance estimator.
/// This is the simple covariance matrix computed from the sample data.
///
/// # Parameters
///
/// * `store_precision` - Whether to store the precision matrix
/// * `assume_centered` - Whether to assume the data is centered
///
/// # Examples
///
/// ```
/// use sklears_covariance::EmpiricalCovariance;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 1.0], [5.0, 4.0]];
///
/// let estimator = EmpiricalCovariance::new();
/// let fitted = estimator.fit(&x.view(), &()).unwrap();
/// let covariance = fitted.get_covariance();
/// ```
#[derive(Debug, Clone)]
pub struct EmpiricalCovariance<S = Untrained> {
    state: S,
    store_precision: bool,
    assume_centered: bool,
}

/// Trained state for EmpiricalCovariance
#[derive(Debug, Clone)]
pub struct EmpiricalCovarianceTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
}

impl EmpiricalCovariance<Untrained> {
    /// Create a new EmpiricalCovariance instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            store_precision: true,
            assume_centered: false,
        }
    }

    /// Set whether to store the precision matrix
    pub fn store_precision(mut self, store_precision: bool) -> Self {
        self.store_precision = store_precision;
        self
    }

    /// Set whether to assume the data is centered
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.assume_centered = assume_centered;
        self
    }
}

impl Default for EmpiricalCovariance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for EmpiricalCovariance<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for EmpiricalCovariance<Untrained> {
    type Fitted = EmpiricalCovariance<EmpiricalCovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Compute mean if not assumed centered
        let mean = if self.assume_centered {
            Array1::zeros(n_features)
        } else {
            x.mean_axis(Axis(0)).unwrap()
        };

        // Center the data
        let mut x_centered = x.to_owned();
        if !self.assume_centered {
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                row -= &mean;
            }
        }

        // Compute covariance matrix
        let covariance = x_centered.t().dot(&x_centered) / (n_samples - 1) as f64;

        // Compute precision matrix if requested
        let precision = if self.store_precision {
            Some(matrix_inverse(&covariance)?)
        } else {
            None
        };

        Ok(EmpiricalCovariance {
            state: EmpiricalCovarianceTrained {
                covariance,
                precision,
                location: mean,
            },
            store_precision: self.store_precision,
            assume_centered: self.assume_centered,
        })
    }
}

impl EmpiricalCovariance<EmpiricalCovarianceTrained> {
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

    /// Get statistical properties of the covariance matrix
    ///
    /// This provides detailed information about the covariance matrix including
    /// symmetry, positive definiteness, condition number, determinant, and other properties.
    ///
    /// # Returns
    ///
    /// A `CovarianceProperties` struct containing various statistical properties
    ///
    /// # Examples
    ///
    /// ```
    /// use sklears_covariance::EmpiricalCovariance;
    /// use sklears_core::traits::Fit;
    /// use scirs2_core::ndarray::array;
    ///
    /// let x = array![[1.0, 2.0], [3.0, 1.0], [5.0, 4.0]];
    /// let estimator = EmpiricalCovariance::new();
    /// let fitted = estimator.fit(&x.view(), &()).unwrap();
    /// let properties = fitted.covariance_properties().unwrap();
    ///
    /// println!("Is symmetric: {}", properties.is_symmetric);
    /// println!("Condition number: {}", properties.condition_number);
    /// ```
    pub fn covariance_properties(&self) -> SklResult<CovarianceProperties<f64>> {
        validate_covariance_matrix(&self.state.covariance)
    }

    /// Get the condition number of the covariance matrix
    ///
    /// The condition number provides insight into the numerical stability
    /// of matrix operations. A high condition number indicates potential
    /// numerical issues.
    ///
    /// # Returns
    ///
    /// The condition number as a f64
    pub fn condition_number(&self) -> SklResult<f64> {
        let properties = self.covariance_properties()?;
        Ok(properties.condition_number)
    }

    /// Check if the covariance matrix is well-conditioned
    ///
    /// A well-conditioned matrix has a condition number below a reasonable threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Maximum acceptable condition number (default: 1e12)
    ///
    /// # Returns
    ///
    /// `true` if the matrix is well-conditioned, `false` otherwise
    pub fn is_well_conditioned(&self, threshold: Option<f64>) -> SklResult<bool> {
        let threshold = threshold.unwrap_or(1e12);
        let cond_num = self.condition_number()?;
        Ok(cond_num < threshold)
    }
}

// DataFrame integration implementation
use crate::polars_integration::{
    CovarianceDataFrame, CovarianceResult, DataFrameEstimator, EstimatorInfo, PerformanceMetrics,
};
use std::collections::HashMap;
use std::time::Instant;

impl DataFrameEstimator<f64> for EmpiricalCovariance<Untrained> {
    fn fit_dataframe(&self, df: &CovarianceDataFrame) -> SklResult<CovarianceResult<f64>> {
        let start_time = Instant::now();

        // Validate DataFrame
        df.validate()?;

        // Fit using standard method
        let fitted = self.clone().fit(&df.as_array_view(), &())?;

        let computation_time = start_time.elapsed().as_millis() as f64;

        // Create performance metrics
        let performance_metrics = Some(PerformanceMetrics {
            computation_time_ms: computation_time,
            memory_usage_mb: None, // Could be implemented with memory profiling
            condition_number: fitted.condition_number().ok(),
            log_likelihood: None, // Not applicable for empirical covariance
        });

        // Create estimator info
        let estimator_info = EstimatorInfo {
            name: "EmpiricalCovariance".to_string(),
            parameters: self.parameters(),
            convergence: None, // Empirical covariance doesn't require iteration
            metrics: performance_metrics,
        };

        Ok(CovarianceResult::new(
            fitted.get_covariance().clone(),
            fitted.get_precision().cloned(),
            df.column_names().to_vec(),
            df.metadata.clone(),
            estimator_info,
        ))
    }

    fn name(&self) -> &str {
        "EmpiricalCovariance"
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "store_precision".to_string(),
            self.store_precision.to_string(),
        );
        params.insert(
            "assume_centered".to_string(),
            self.assume_centered.to_string(),
        );
        params
    }
}

impl DataFrameEstimator<f64> for EmpiricalCovariance<EmpiricalCovarianceTrained> {
    fn fit_dataframe(&self, df: &CovarianceDataFrame) -> SklResult<CovarianceResult<f64>> {
        // For already trained estimators, just return the current state with DataFrame context
        let estimator_info = EstimatorInfo {
            name: "EmpiricalCovariance".to_string(),
            parameters: self.parameters(),
            convergence: None,
            metrics: None,
        };

        Ok(CovarianceResult::new(
            self.get_covariance().clone(),
            self.get_precision().cloned(),
            df.column_names().to_vec(),
            df.metadata.clone(),
            estimator_info,
        ))
    }

    fn name(&self) -> &str {
        "EmpiricalCovariance"
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "store_precision".to_string(),
            self.store_precision.to_string(),
        );
        params.insert(
            "assume_centered".to_string(),
            self.assume_centered.to_string(),
        );
        params
    }
}

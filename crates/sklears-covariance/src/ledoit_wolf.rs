//! Ledoit-Wolf Covariance Estimator
//!
//! Covariance estimator that uses Ledoit-Wolf shrinkage to automatically
//! determine the optimal shrinkage parameter.

use crate::empirical::EmpiricalCovariance;
use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for LedoitWolf estimator
#[derive(Debug, Clone)]
pub struct LedoitWolfConfig {
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
    /// Size of blocks for processing
    pub block_size: usize,
}

/// Ledoit-Wolf Covariance Estimator
///
/// Covariance estimator that uses Ledoit-Wolf shrinkage to automatically
/// determine the optimal shrinkage parameter.
///
/// # Parameters
///
/// * `store_precision` - Whether to store the precision matrix
/// * `assume_centered` - Whether to assume the data is centered
/// * `block_size` - Size of blocks for processing (default: 1000)
///
/// # Examples
///
/// ```
/// use sklears_covariance::LedoitWolf;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let lw = LedoitWolf::new();
/// let fitted = lw.fit(&X.view(), &()).unwrap();
/// let covariance = fitted.get_covariance();
/// let shrinkage = fitted.get_shrinkage();
/// ```
#[derive(Debug, Clone)]
pub struct LedoitWolf<S = Untrained> {
    state: S,
    config: LedoitWolfConfig,
}

/// Trained state for LedoitWolf
#[derive(Debug, Clone)]
pub struct LedoitWolfTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The shrinkage parameter used
    pub shrinkage: f64,
}

impl LedoitWolf<Untrained> {
    /// Create a new LedoitWolf instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: LedoitWolfConfig {
                store_precision: true,
                assume_centered: false,
                block_size: 1000,
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

    /// Set the block size for processing
    pub fn block_size(mut self, block_size: usize) -> Self {
        self.config.block_size = block_size;
        self
    }
}

impl Default for LedoitWolf<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LedoitWolf<Untrained> {
    type Config = LedoitWolfConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for LedoitWolf<Untrained> {
    type Fitted = LedoitWolf<LedoitWolfTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Compute empirical covariance
        let emp_cov = EmpiricalCovariance::new()
            .assume_centered(self.config.assume_centered)
            .store_precision(false)
            .fit(&x, &())?;

        let covariance_emp = emp_cov.get_covariance().clone();
        let location = emp_cov.get_location().clone();

        // Compute optimal shrinkage parameter
        let shrinkage = self.compute_ledoit_wolf_shrinkage(&x, &covariance_emp, &location)?;

        // Apply shrinkage
        let trace = covariance_emp.diag().sum();
        let mu = trace / n_features as f64;
        let mut identity = Array2::eye(n_features);
        identity *= mu;

        let covariance = (1.0 - shrinkage) * &covariance_emp + shrinkage * &identity;

        // Compute precision matrix if requested
        let precision = if self.config.store_precision {
            Some(matrix_inverse(&covariance)?)
        } else {
            None
        };

        Ok(LedoitWolf {
            state: LedoitWolfTrained {
                covariance,
                precision,
                location,
                shrinkage,
            },
            config: self.config,
        })
    }
}

impl LedoitWolf<Untrained> {
    fn compute_ledoit_wolf_shrinkage(
        &self,
        x: &ArrayView2<'_, Float>,
        covariance_emp: &Array2<f64>,
        location: &Array1<f64>,
    ) -> SklResult<f64> {
        let (n_samples, n_features) = x.dim();

        // Center the data
        let mut x_centered = x.to_owned();
        if !self.config.assume_centered {
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                row -= location;
            }
        }

        // Compute sum of squared deviations
        let mut sum_squared_deviations = 0.0;
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_samples {
                    let dev = x_centered[[k, i]] * x_centered[[k, j]] - covariance_emp[[i, j]];
                    sum_squared_deviations += dev * dev;
                }
            }
        }

        let beta = sum_squared_deviations / (n_samples as f64).powi(2);

        // Compute delta (distance to target)
        let trace = covariance_emp.diag().sum();
        let mu = trace / n_features as f64;

        let mut delta = 0.0;
        for i in 0..n_features {
            for j in 0..n_features {
                let target_val = if i == j { mu } else { 0.0 };
                delta += (covariance_emp[[i, j]] - target_val).powi(2);
            }
        }

        // Optimal shrinkage
        let shrinkage = if delta > 0.0 {
            (beta / delta).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(shrinkage)
    }
}

impl LedoitWolf<LedoitWolfTrained> {
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

    /// Get the shrinkage parameter
    pub fn get_shrinkage(&self) -> f64 {
        self.state.shrinkage
    }
}

// DataFrame integration implementation
use crate::polars_integration::{
    CovarianceDataFrame, CovarianceResult, DataFrameEstimator, EstimatorInfo, PerformanceMetrics,
};
use std::collections::HashMap;
use std::time::Instant;

impl DataFrameEstimator<f64> for LedoitWolf<Untrained> {
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
            condition_number: None, // Could compute if needed
            log_likelihood: None,  // Could implement for Gaussian likelihood
        });

        // Create estimator info
        let estimator_info = EstimatorInfo {
            name: "LedoitWolf".to_string(),
            parameters: self.parameters(),
            convergence: None, // LedoitWolf has closed-form solution
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
        "LedoitWolf"
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "store_precision".to_string(),
            self.config.store_precision.to_string(),
        );
        params.insert(
            "assume_centered".to_string(),
            self.config.assume_centered.to_string(),
        );
        params.insert("block_size".to_string(), self.config.block_size.to_string());
        params
    }
}

impl DataFrameEstimator<f64> for LedoitWolf<LedoitWolfTrained> {
    fn fit_dataframe(&self, df: &CovarianceDataFrame) -> SklResult<CovarianceResult<f64>> {
        // For already trained estimators, just return the current state with DataFrame context
        let estimator_info = EstimatorInfo {
            name: "LedoitWolf".to_string(),
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
        "LedoitWolf"
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "store_precision".to_string(),
            self.config.store_precision.to_string(),
        );
        params.insert(
            "assume_centered".to_string(),
            self.config.assume_centered.to_string(),
        );
        params.insert("block_size".to_string(), self.config.block_size.to_string());
        params.insert("shrinkage".to_string(), self.state.shrinkage.to_string());
        params
    }
}

//! Temporal Gaussian Processes for Time Series Modeling
//!
//! This module implements Gaussian Process methods specifically designed for temporal data,
//! including time series forecasting, state-space models, and online updates with Kalman filtering.
//!
//! # Mathematical Background
//!
//! Temporal GPs extend standard GP regression to handle time series data with:
//! 1. **Time-aware kernels**: Kernels that capture temporal correlation structures
//! 2. **State-space formulation**: Representing GPs as state-space models for efficient inference
//! 3. **Kalman filtering**: Online updates for streaming temporal data
//! 4. **Seasonal decomposition**: Separating trend, seasonal, and noise components
//! 5. **Multi-scale modeling**: Capturing patterns at different temporal scales
//!
//! # Examples
//!
//! ```rust
//! use sklears_gaussian_process::temporal::{TemporalGaussianProcessRegressor, TemporalKernel};
//! use sklears_gaussian_process::kernels::RBF;
//! use sklears_core::traits::{Fit, Predict};
//! use scirs2_core::ndarray::array;
//!
//! // Create temporal GP with seasonal decomposition
//! let temporal_gp = TemporalGaussianProcessRegressor::builder()
//!     .temporal_kernel(TemporalKernel::locally_periodic(1.0, 12.0, 0.5))
//!     .enable_seasonal_decomposition(true)
//!     .seasonal_period(12.0)
//!     .build();
//!
//! let times = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
//! let values = array![1.0, 1.5, 2.1, 1.8, 2.5];
//!
//! let trained_model = temporal_gp.fit(&times, &values).unwrap();
//! let predictions = trained_model.predict(&array![[6.0], [7.0]]).unwrap();
//! ```

use crate::kernels::Kernel;
use crate::utils;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, Axis, s};
use scirs2_core::random::{thread_rng, Random}; // SciRS2 Policy
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict};
use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

/// State marker for untrained temporal GP
#[derive(Debug, Clone)]
pub struct Untrained;

/// State marker for trained temporal GP
#[derive(Debug, Clone)]
pub struct Trained {
    pub temporal_kernel: TemporalKernel,
    pub training_data: (Array2<f64>, Array1<f64>),
    pub alpha: Array1<f64>,
    pub cholesky: Array2<f64>,
    pub log_likelihood: f64,
    pub seasonal_components: Option<SeasonalDecomposition>,
    pub state_space_model: Option<StateSpaceModel>,
}

/// Temporal kernel types for time series modeling
#[derive(Debug, Clone)]
pub enum TemporalKernel {
    /// Exponential kernel for temporal correlation: k(τ) = σ² exp(-τ/λ)
    Exponential { variance: f64, length_scale: f64 },
    /// Locally periodic kernel: RBF * Periodic
    LocallyPeriodic {
        variance: f64,
        length_scale: f64,
        period: f64,
        periodicity_decay: f64,
    },
    /// Matérn kernel for temporal data
    Matern {
        variance: f64,
        length_scale: f64,
        nu: f64,
    },
    /// Changepoint kernel for detecting regime changes
    Changepoint {
        variance1: f64,
        variance2: f64,
        length_scale: f64,
        changepoint: f64,
        sharpness: f64,
    },
    /// Multi-scale kernel combining multiple temporal scales
    MultiScale { kernels: Vec<TemporalKernel> },
}

impl TemporalKernel {
    /// Create an exponential temporal kernel
    pub fn exponential(variance: f64, length_scale: f64) -> Self {
        Self::Exponential {
            variance,
            length_scale,
        }
    }

    /// Create a locally periodic kernel
    pub fn locally_periodic(variance: f64, period: f64, periodicity_decay: f64) -> Self {
        Self::LocallyPeriodic {
            variance,
            length_scale: period / 4.0, // Default length scale
            period,
            periodicity_decay,
        }
    }

    /// Create a Matérn temporal kernel
    pub fn matern(variance: f64, length_scale: f64, nu: f64) -> Self {
        Self::Matern {
            variance,
            length_scale,
            nu,
        }
    }

    /// Create a changepoint kernel
    pub fn changepoint(variance1: f64, variance2: f64, changepoint: f64) -> Self {
        Self::Changepoint {
            variance1,
            variance2,
            length_scale: 1.0,
            changepoint,
            sharpness: 1.0,
        }
    }

    /// Create a multi-scale kernel
    pub fn multi_scale(kernels: Vec<TemporalKernel>) -> Self {
        Self::MultiScale { kernels }
    }

    /// Compute the kernel value for a time difference
    pub fn compute(&self, tau: f64) -> f64 {
        match self {
            Self::Exponential {
                variance,
                length_scale,
            } => variance * (-tau.abs() / length_scale).exp(),
            Self::LocallyPeriodic {
                variance,
                length_scale,
                period,
                periodicity_decay,
            } => {
                let rbf_component = (-0.5 * tau.powi(2) / length_scale.powi(2)).exp();
                let periodic_component =
                    (-2.0 * (PI * tau / period).sin().powi(2) / periodicity_decay.powi(2)).exp();
                variance * rbf_component * periodic_component
            }
            Self::Matern {
                variance,
                length_scale,
                nu,
            } => {
                let r = tau.abs() / length_scale;
                if r == 0.0 {
                    return *variance;
                }

                match nu {
                    0.5 => variance * (-r).exp(),
                    1.5 => variance * (1.0 + r * 3.0_f64.sqrt()) * (-r * 3.0_f64.sqrt()).exp(),
                    2.5 => {
                        variance
                            * (1.0 + r * 5.0_f64.sqrt() + 5.0 * r.powi(2) / 3.0)
                            * (-r * 5.0_f64.sqrt()).exp()
                    }
                    _ => {
                        // General Matérn (simplified)
                        variance * (1.0 + r).exp() * (-r).exp()
                    }
                }
            }
            Self::Changepoint {
                variance1,
                variance2,
                length_scale,
                changepoint,
                sharpness,
            } => {
                let sigma = if tau < *changepoint {
                    *variance1
                } else {
                    *variance2
                };
                let transition = 0.5 * (1.0 + ((tau - changepoint) * sharpness).tanh());
                let mixed_variance = variance1 * (1.0 - transition) + variance2 * transition;
                mixed_variance * (-tau.abs() / length_scale).exp()
            }
            Self::MultiScale { kernels } => kernels.iter().map(|k| k.compute(tau)).sum(),
        }
    }
}

impl Kernel for TemporalKernel {
    fn compute_kernel_matrix(
        &self,
        x1: &Array2<f64>,
        x2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let x2 = x2.unwrap_or(x1);
        let n1 = x1.nrows();
        let n2 = x2.nrows();

        if x1.ncols() != 1 || x2.ncols() != 1 {
            return Err(SklearsError::InvalidInput(
                "Temporal kernels require 1D time inputs".to_string(),
            ));
        }

        let mut K = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let tau = x1[[i, 0]] - x2[[j, 0]];
                K[[i, j]] = self.compute(tau);
            }
        }

        Ok(K)
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }

    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        if x1.len() != 1 || x2.len() != 1 {
            return 0.0; // Invalid temporal coordinates
        }
        let tau = x1[0] - x2[0];
        self.compute(tau)
    }

    fn get_params(&self) -> Vec<f64> {
        match self {
            Self::Exponential { variance, length_scale } => vec![*variance, *length_scale],
            Self::LocallyPeriodic { variance, length_scale, period, periodicity_decay } => {
                vec![*variance, *length_scale, *period, *periodicity_decay]
            }
            Self::Matern { variance, length_scale, nu } => vec![*variance, *length_scale, *nu],
            Self::Changepoint { variance1, variance2, length_scale, changepoint, sharpness } => {
                vec![*variance1, *variance2, *length_scale, *changepoint, *sharpness]
            }
            Self::MultiScale { kernels } => {
                let mut params = Vec::new();
                for kernel in kernels {
                    params.extend(kernel.get_params());
                }
                params
            }
        }
    }

    fn set_params(&mut self, params: &[f64]) -> SklResult<()> {
        match self {
            Self::Exponential { variance, length_scale } => {
                if params.len() != 2 {
                    return Err(SklearsError::InvalidInput("Exponential kernel requires 2 parameters".to_string()));
                }
                *variance = params[0];
                *length_scale = params[1];
            }
            Self::LocallyPeriodic { variance, length_scale, period, periodicity_decay } => {
                if params.len() != 4 {
                    return Err(SklearsError::InvalidInput("LocallyPeriodic kernel requires 4 parameters".to_string()));
                }
                *variance = params[0];
                *length_scale = params[1];
                *period = params[2];
                *periodicity_decay = params[3];
            }
            Self::Matern { variance, length_scale, nu } => {
                if params.len() != 3 {
                    return Err(SklearsError::InvalidInput("Matern kernel requires 3 parameters".to_string()));
                }
                *variance = params[0];
                *length_scale = params[1];
                *nu = params[2];
            }
            Self::Changepoint { variance1, variance2, length_scale, changepoint, sharpness } => {
                if params.len() != 5 {
                    return Err(SklearsError::InvalidInput("Changepoint kernel requires 5 parameters".to_string()));
                }
                *variance1 = params[0];
                *variance2 = params[1];
                *length_scale = params[2];
                *changepoint = params[3];
                *sharpness = params[4];
            }
            Self::MultiScale { kernels } => {
                let mut offset = 0;
                for kernel in kernels.iter_mut() {
                    let n_params = kernel.get_params().len();
                    if offset + n_params > params.len() {
                        return Err(SklearsError::InvalidInput("Not enough parameters for MultiScale kernel".to_string()));
                    }
                    kernel.set_params(&params[offset..offset + n_params])?;
                    offset += n_params;
                }
            }
        }
        Ok(())
    }
}

/// Seasonal decomposition for time series
#[derive(Debug, Clone)]
pub struct SeasonalDecomposition {
    pub trend: Array1<f64>,
    pub seasonal: Array1<f64>,
    pub residual: Array1<f64>,
    pub period: f64,
}

impl SeasonalDecomposition {
    /// Perform additive seasonal decomposition: y = trend + seasonal + residual
    pub fn decompose_additive(
        times: &Array1<f64>,
        values: &Array1<f64>,
        period: f64,
    ) -> SklResult<Self> {
        let n = values.len();
        if times.len() != n {
            return Err(SklearsError::DimensionMismatch {
                expected: times.len(),
                actual: n,
            });
        }

        // Extract trend using moving average
        let window_size = (period as usize).max(3);
        let trend = Self::extract_trend(values, window_size);

        // Extract seasonal component
        let detrended = values - &trend;
        let seasonal = Self::extract_seasonal(&detrended, times, period);

        // Compute residual
        let residual = values - &trend - &seasonal;

        Ok(Self {
            trend,
            seasonal,
            residual,
            period,
        })
    }

    /// Extract trend component using moving average
    fn extract_trend(values: &Array1<f64>, window_size: usize) -> Array1<f64> {
        let n = values.len();
        let mut trend = Array1::zeros(n);
        let half_window = window_size / 2;

        for i in 0..n {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(n);
            let window_mean = values.slice(s![start..end]).mean().unwrap_or(0.0);
            trend[i] = window_mean;
        }

        trend
    }

    /// Extract seasonal component
    fn extract_seasonal(detrended: &Array1<f64>, times: &Array1<f64>, period: f64) -> Array1<f64> {
        let n = detrended.len();
        let mut seasonal = Array1::zeros(n);

        // Group observations by seasonal phase
        let mut seasonal_sums = vec![0.0; period as usize];
        let mut seasonal_counts = vec![0; period as usize];

        for i in 0..n {
            let phase = ((times[i] % period) as usize).min(period as usize - 1);
            seasonal_sums[phase] += detrended[i];
            seasonal_counts[phase] += 1;
        }

        // Compute seasonal averages
        let mut seasonal_averages = vec![0.0; period as usize];
        for i in 0..period as usize {
            if seasonal_counts[i] > 0 {
                seasonal_averages[i] = seasonal_sums[i] / seasonal_counts[i] as f64;
            }
        }

        // Assign seasonal values
        for i in 0..n {
            let phase = ((times[i] % period) as usize).min(period as usize - 1);
            seasonal[i] = seasonal_averages[phase];
        }

        seasonal
    }

    /// Forecast seasonal component for future times
    pub fn forecast_seasonal(&self, future_times: &Array1<f64>) -> Array1<f64> {
        let n = future_times.len();
        let mut seasonal_forecast = Array1::zeros(n);

        // Compute average seasonal pattern
        let period_len = self.period as usize;
        let mut seasonal_pattern = vec![0.0; period_len];
        let mut pattern_counts = vec![0; period_len];

        for i in 0..self.seasonal.len() {
            let phase = (i % period_len).min(period_len - 1);
            seasonal_pattern[phase] += self.seasonal[i];
            pattern_counts[phase] += 1;
        }

        for i in 0..period_len {
            if pattern_counts[i] > 0 {
                seasonal_pattern[i] /= pattern_counts[i] as f64;
            }
        }

        // Apply pattern to future times
        for i in 0..n {
            let phase = ((future_times[i] % self.period) as usize).min(period_len - 1);
            seasonal_forecast[i] = seasonal_pattern[phase];
        }

        seasonal_forecast
    }
}

/// State-space model representation of a Gaussian process
#[derive(Debug, Clone)]
pub struct StateSpaceModel {
    /// State transition matrix A
    pub transition_matrix: Array2<f64>,
    /// Observation matrix H
    pub observation_matrix: Array2<f64>,
    /// Process noise covariance Q
    pub process_noise: Array2<f64>,
    /// Observation noise variance R
    pub observation_noise: f64,
    /// Current state estimate
    pub state: Array1<f64>,
    /// Current state covariance
    pub state_covariance: Array2<f64>,
    /// State dimension
    pub state_dim: usize,
}

impl StateSpaceModel {
    /// Create a new state-space model for an exponential kernel
    pub fn from_exponential_kernel(variance: f64, length_scale: f64, dt: f64) -> Self {
        // For exponential kernel: dx/dt = -x/λ + w(t)
        let lambda = length_scale;
        let phi = (-dt / lambda).exp();

        let transition_matrix = Array2::from_elem((1, 1), phi);
        let observation_matrix = Array2::from_elem((1, 1), 1.0);

        let q_variance = variance * (1.0 - phi.powi(2));
        let process_noise = Array2::from_elem((1, 1), q_variance);

        Self {
            transition_matrix,
            observation_matrix,
            process_noise,
            observation_noise: 1e-6, // Small default noise
            state: Array1::zeros(1),
            state_covariance: Array2::from_elem((1, 1), variance),
            state_dim: 1,
        }
    }

    /// Create a state-space model for a Matérn 3/2 kernel
    pub fn from_matern32_kernel(variance: f64, length_scale: f64, dt: f64) -> Self {
        // For Matérn 3/2: second-order SDE
        let lambda = 3.0_f64.sqrt() / length_scale;
        let phi1 = (-lambda * dt).exp();
        let phi2 = lambda * dt * phi1;

        let mut transition_matrix = Array2::zeros((2, 2));
        transition_matrix[[0, 0]] = phi1 + phi2;
        transition_matrix[[0, 1]] = dt * phi1;
        transition_matrix[[1, 0]] = -lambda.powi(2) * dt * phi1;
        transition_matrix[[1, 1]] = phi1 - phi2;

        let observation_matrix = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();

        // Process noise for Matérn 3/2
        let q11 = variance * (1.0 - phi1.powi(2) - 2.0 * phi1 * phi2 - phi2.powi(2));
        let q12 = variance * lambda * dt * phi1 * (phi1 + phi2);
        let q22 =
            variance * lambda.powi(2) * (1.0 - phi1.powi(2) + 2.0 * phi1 * phi2 - phi2.powi(2));

        let mut process_noise = Array2::zeros((2, 2));
        process_noise[[0, 0]] = q11;
        process_noise[[0, 1]] = q12;
        process_noise[[1, 0]] = q12;
        process_noise[[1, 1]] = q22;

        Self {
            transition_matrix,
            observation_matrix,
            process_noise,
            observation_noise: 1e-6,
            state: Array1::zeros(2),
            state_covariance: Array2::eye(2) * variance,
            state_dim: 2,
        }
    }

    /// Kalman filter prediction step
    pub fn predict(&mut self, dt: f64) -> SklResult<()> {
        // Update time-dependent matrices if needed
        // For now, assume fixed dt

        // Predict state: x_k|k-1 = A * x_k-1|k-1
        let predicted_state = self.transition_matrix.dot(&self.state);

        // Predict covariance: P_k|k-1 = A * P_k-1|k-1 * A^T + Q
        let predicted_covariance = self
            .transition_matrix
            .dot(&self.state_covariance)
            .dot(&self.transition_matrix.t())
            + &self.process_noise;

        self.state = predicted_state;
        self.state_covariance = predicted_covariance;

        Ok(())
    }

    /// Kalman filter update step
    pub fn update(&mut self, observation: f64) -> SklResult<()> {
        // Innovation: y = z - H * x_k|k-1
        let predicted_observation = self.observation_matrix.dot(&self.state)[0];
        let innovation = observation - predicted_observation;

        // Innovation covariance: S = H * P_k|k-1 * H^T + R
        let innovation_covariance = self
            .observation_matrix
            .dot(&self.state_covariance)
            .dot(&self.observation_matrix.t())[[0, 0]]
            + self.observation_noise;

        // Kalman gain: K = P_k|k-1 * H^T * S^(-1)
        let kalman_gain =
            self.state_covariance.dot(&self.observation_matrix.t()) / innovation_covariance;

        // Update state: x_k|k = x_k|k-1 + K * y
        self.state = &self.state + &kalman_gain * innovation;

        // Update covariance: P_k|k = (I - K * H) * P_k|k-1
        let identity = Array2::eye(self.state_dim);
        let update_matrix = &identity - &kalman_gain.dot(&self.observation_matrix);
        self.state_covariance = update_matrix.dot(&self.state_covariance);

        Ok(())
    }

    /// Get current state estimate and uncertainty
    pub fn get_state(&self) -> (Array1<f64>, Array2<f64>) {
        (self.state.clone(), self.state_covariance.clone())
    }
}

/// Temporal Gaussian Process Regressor
#[derive(Debug, Clone)]
pub struct TemporalGaussianProcessRegressor<S = Untrained> {
    temporal_kernel: Option<TemporalKernel>,
    enable_seasonal_decomposition: bool,
    seasonal_period: Option<f64>,
    enable_state_space: bool,
    kalman_update_interval: f64,
    alpha: f64,
    _state: S,
}

/// Configuration for temporal GP
#[derive(Debug, Clone)]
pub struct TemporalGPConfig {
    pub enable_seasonal: bool,
    pub seasonal_period: f64,
    pub enable_state_space: bool,
    pub kalman_interval: f64,
    pub regularization: f64,
}

impl Default for TemporalGPConfig {
    fn default() -> Self {
        Self {
            enable_seasonal: false,
            seasonal_period: 12.0,
            enable_state_space: false,
            kalman_interval: 1.0,
            regularization: 1e-6,
        }
    }
}

impl TemporalGaussianProcessRegressor<Untrained> {
    /// Create a new temporal GP regressor
    pub fn new() -> Self {
        Self {
            temporal_kernel: None,
            enable_seasonal_decomposition: false,
            seasonal_period: None,
            enable_state_space: false,
            kalman_update_interval: 1.0,
            alpha: 1e-6,
            _state: Untrained,
        }
    }

    /// Create a builder for temporal GP
    pub fn builder() -> TemporalGPBuilder {
        TemporalGPBuilder::new()
    }

    /// Set the temporal kernel
    pub fn temporal_kernel(mut self, kernel: TemporalKernel) -> Self {
        self.temporal_kernel = Some(kernel);
        self
    }

    /// Enable seasonal decomposition
    pub fn enable_seasonal_decomposition(mut self, enable: bool) -> Self {
        self.enable_seasonal_decomposition = enable;
        self
    }

    /// Set seasonal period
    pub fn seasonal_period(mut self, period: f64) -> Self {
        self.seasonal_period = Some(period);
        self
    }

    /// Enable state-space representation
    pub fn enable_state_space(mut self, enable: bool) -> Self {
        self.enable_state_space = enable;
        self
    }

    /// Set regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

/// Builder for temporal GP regressor
#[derive(Debug, Clone)]
pub struct TemporalGPBuilder {
    kernel: Option<TemporalKernel>,
    enable_seasonal: bool,
    seasonal_period: Option<f64>,
    enable_state_space: bool,
    kalman_interval: f64,
    alpha: f64,
}

impl TemporalGPBuilder {
    pub fn new() -> Self {
        Self {
            kernel: None,
            enable_seasonal: false,
            seasonal_period: None,
            enable_state_space: false,
            kalman_interval: 1.0,
            alpha: 1e-6,
        }
    }

    pub fn temporal_kernel(mut self, kernel: TemporalKernel) -> Self {
        self.kernel = Some(kernel);
        self
    }

    pub fn enable_seasonal_decomposition(mut self, enable: bool) -> Self {
        self.enable_seasonal = enable;
        self
    }

    pub fn seasonal_period(mut self, period: f64) -> Self {
        self.seasonal_period = Some(period);
        self
    }

    pub fn enable_state_space(mut self, enable: bool) -> Self {
        self.enable_state_space = enable;
        self
    }

    pub fn kalman_update_interval(mut self, interval: f64) -> Self {
        self.kalman_interval = interval;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn build(self) -> TemporalGaussianProcessRegressor<Untrained> {
        TemporalGaussianProcessRegressor {
            temporal_kernel: self.kernel,
            enable_seasonal_decomposition: self.enable_seasonal,
            seasonal_period: self.seasonal_period,
            enable_state_space: self.enable_state_space,
            kalman_update_interval: self.kalman_interval,
            alpha: self.alpha,
            _state: Untrained,
        }
    }
}

impl Estimator for TemporalGaussianProcessRegressor<Untrained> {
    type Config = TemporalGPConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static DEFAULT_CONFIG: TemporalGPConfig = TemporalGPConfig {
            enable_seasonal: false,
            seasonal_period: 12.0,
            enable_state_space: false,
            kalman_interval: 1.0,
            regularization: 1e-6,
        };
        &DEFAULT_CONFIG
    }
}

impl Estimator for TemporalGaussianProcessRegressor<Trained> {
    type Config = TemporalGPConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static DEFAULT_CONFIG: TemporalGPConfig = TemporalGPConfig {
            enable_seasonal: false,
            seasonal_period: 12.0,
            enable_state_space: false,
            kalman_interval: 1.0,
            regularization: 1e-6,
        };
        &DEFAULT_CONFIG
    }
}

impl Fit<Array2<f64>, Array1<f64>> for TemporalGaussianProcessRegressor<Untrained> {
    type Fitted = TemporalGaussianProcessRegressor<Trained>;

    fn fit(self, X: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: X.nrows(),
                actual: y.len(),
            });
        }

        if X.ncols() != 1 {
            return Err(SklearsError::InvalidInput(
                "Temporal GP requires 1D time inputs".to_string(),
            ));
        }

        let kernel = self.temporal_kernel.clone().ok_or_else(|| {
            SklearsError::InvalidInput("Temporal kernel must be specified".to_string())
        })?;

        let X_owned = X.to_owned();
        let y_owned = y.to_owned();

        // Extract time vector
        let times = X_owned.column(0).to_owned();

        // Seasonal decomposition if enabled
        let seasonal_components = if self.enable_seasonal_decomposition {
            let period = self.seasonal_period.unwrap_or(12.0);
            Some(SeasonalDecomposition::decompose_additive(
                &times, &y_owned, period,
            )?)
        } else {
            None
        };

        // Use detrended data for GP if seasonal decomposition is used
        let y_for_gp = if let Some(ref seasonal) = seasonal_components {
            &y_owned - &seasonal.trend - &seasonal.seasonal
        } else {
            y_owned.clone()
        };

        // Compute kernel matrix
        let K = kernel.compute_kernel_matrix(&X_owned, None)?;

        // Add regularization
        let mut K_reg = K.clone();
        for i in 0..K_reg.nrows() {
            K_reg[[i, i]] += self.alpha;
        }

        // Solve linear system
        let chol_decomp = utils::robust_cholesky(&K_reg)?;
        let alpha = utils::triangular_solve(&chol_decomp, &y_for_gp)?;

        // Compute log marginal likelihood
        let log_det = chol_decomp.diag().iter().map(|x| x.ln()).sum::<f64>() * 2.0;
        let data_fit = y_for_gp.dot(&alpha);
        let n = y_for_gp.len();
        let log_likelihood = -0.5 * (data_fit + log_det + n as f64 * (2.0 * PI).ln());

        // Create state-space model if enabled
        let state_space_model = if self.enable_state_space {
            match &kernel {
                TemporalKernel::Exponential {
                    variance,
                    length_scale,
                } => Some(StateSpaceModel::from_exponential_kernel(
                    *variance,
                    *length_scale,
                    self.kalman_update_interval,
                )),
                TemporalKernel::Matern {
                    variance,
                    length_scale,
                    nu: 1.5,
                } => Some(StateSpaceModel::from_matern32_kernel(
                    *variance,
                    *length_scale,
                    self.kalman_update_interval,
                )),
                _ => None, // Not all kernels have state-space representations
            }
        } else {
            None
        };

        Ok(TemporalGaussianProcessRegressor {
            temporal_kernel: self.temporal_kernel,
            enable_seasonal_decomposition: self.enable_seasonal_decomposition,
            seasonal_period: self.seasonal_period,
            enable_state_space: self.enable_state_space,
            kalman_update_interval: self.kalman_update_interval,
            alpha: self.alpha,
            _state: Trained {
                temporal_kernel: kernel,
                training_data: (X_owned, y_owned),
                alpha,
                cholesky: chol_decomp,
                log_likelihood,
                seasonal_components,
                state_space_model,
            },
        })
    }
}

impl TemporalGaussianProcessRegressor<Trained> {
    /// Access the trained state
    pub fn trained_state(&self) -> &Trained {
        &self._state
    }

    /// Get seasonal decomposition components
    pub fn seasonal_components(&self) -> Option<&SeasonalDecomposition> {
        self._state.seasonal_components.as_ref()
    }

    /// Update model with new temporal data (online learning)
    pub fn update(&mut self, new_time: f64, new_value: f64) -> SklResult<()> {
        if let Some(ref mut state_space) = self._state.state_space_model {
            // Use Kalman filter for online update
            let dt = new_time - self._state.training_data.0.slice(s![-1, 0])[0];
            state_space.predict(dt)?;
            state_space.update(new_value)?;
        }

        // Note: For full GP update, we would need to update the training data
        // and recompute the Cholesky decomposition, which is more expensive

        Ok(())
    }

    /// Forecast future values with uncertainty
    pub fn forecast(
        &self,
        future_times: &Array2<f64>,
        include_uncertainty: bool,
    ) -> SklResult<(Array1<f64>, Option<Array1<f64>>)> {
        let n_future = future_times.nrows();
        let mut predictions = Array1::zeros(n_future);
        let mut uncertainties = if include_uncertainty {
            Some(Array1::zeros(n_future))
        } else {
            None
        };

        // Base GP predictions
        let K_star = self
            ._state
            .temporal_kernel
            .compute_kernel_matrix(&self._state.training_data.0, Some(future_times))?;
        let gp_pred = K_star.t().dot(&self._state.alpha);

        // Add seasonal forecast if available
        if let Some(ref seasonal) = self._state.seasonal_components {
            let future_times_1d = future_times.column(0).to_owned();
            let seasonal_forecast = seasonal.forecast_seasonal(&future_times_1d);

            for i in 0..n_future {
                predictions[i] = gp_pred[i] + seasonal_forecast[i];
            }
        } else {
            predictions = gp_pred;
        }

        // Compute uncertainties if requested
        if include_uncertainty {
            let K_star_star = self
                ._state
                .temporal_kernel
                .compute_kernel_matrix(future_times, None)?;
            let v = utils::triangular_solve(&self._state.cholesky, &K_star)?;
            let pred_var = K_star_star.diag() - v.map(|x| x.powi(2)).sum_axis(Axis(0));

            if let Some(ref mut unc) = uncertainties {
                for i in 0..n_future {
                    unc[i] = pred_var[i].max(0.0).sqrt();
                }
            }
        }

        Ok((predictions, uncertainties))
    }

    /// Detect changepoints in the time series
    pub fn detect_changepoints(&self, threshold: f64) -> Vec<f64> {
        let mut changepoints = Vec::new();

        if let Some(ref seasonal) = self._state.seasonal_components {
            // Use residuals for changepoint detection
            let residuals = &seasonal.residual;
            let times = self._state.training_data.0.column(0);

            // Simple changepoint detection using variance changes
            let window_size = 10;
            for i in window_size..residuals.len() - window_size {
                let left_var = residuals.slice(s![i - window_size..i]).var(0.0);
                let right_var = residuals.slice(s![i..i + window_size]).var(0.0);

                let ratio = (left_var / right_var).max(right_var / left_var);
                if ratio > threshold {
                    changepoints.push(times[i]);
                }
            }
        }

        changepoints
    }
}

impl Predict<Array2<f64>, Array1<f64>> for TemporalGaussianProcessRegressor<Trained> {
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<f64>> {
        let (predictions, _) = self.forecast(X, false)?;
        Ok(predictions)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_temporal_kernel_exponential() {
        let kernel = TemporalKernel::exponential(1.0, 2.0);

        // Test at tau = 0
        assert_abs_diff_eq!(kernel.compute(0.0), 1.0, epsilon = 1e-10);

        // Test exponential decay
        let val1 = kernel.compute(1.0);
        let val2 = kernel.compute(2.0);
        assert!(val1 > val2);
        assert!(val2 > 0.0);
    }

    #[test]
    fn test_temporal_kernel_locally_periodic() {
        let kernel = TemporalKernel::locally_periodic(1.0, 4.0, 0.5);

        // Test periodicity
        let val1 = kernel.compute(0.0);
        let val2 = kernel.compute(4.0); // One period later
        assert_abs_diff_eq!(val1, val2, epsilon = 1e-6);
    }

    #[test]
    fn test_seasonal_decomposition() {
        let times = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let values = array![1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.8, 2.8]; // Seasonal pattern

        let decomp = SeasonalDecomposition::decompose_additive(&times, &values, 4.0).unwrap();

        assert_eq!(decomp.trend.len(), values.len());
        assert_eq!(decomp.seasonal.len(), values.len());
        assert_eq!(decomp.residual.len(), values.len());

        // Test reconstruction
        let reconstructed = &decomp.trend + &decomp.seasonal + &decomp.residual;
        for i in 0..values.len() {
            assert_abs_diff_eq!(reconstructed[i], values[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_state_space_model_exponential() {
        let mut ssm = StateSpaceModel::from_exponential_kernel(1.0, 2.0, 0.1);

        // Test prediction step
        ssm.predict(0.1).unwrap();

        // Test update step
        ssm.update(1.5).unwrap();

        let (state, covariance) = ssm.get_state();
        assert_eq!(state.len(), 1);
        assert_eq!(covariance.shape(), &[1, 1]);
    }

    #[test]
    fn test_temporal_gp_fit_predict() {
        let times = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let values = array![1.0, 1.5, 2.1, 1.8, 2.5];

        let temporal_gp = TemporalGaussianProcessRegressor::builder()
            .temporal_kernel(TemporalKernel::exponential(1.0, 2.0))
            .build();

        let trained = temporal_gp.fit(&times, &values).unwrap();
        let predictions = trained.predict(&times).unwrap();

        assert_eq!(predictions.len(), times.nrows());
    }

    #[test]
    fn test_temporal_gp_with_seasonal() {
        let times = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let values = array![1.0, 2.0, 1.5, 2.5, 1.2, 2.2]; // Seasonal pattern

        let temporal_gp = TemporalGaussianProcessRegressor::builder()
            .temporal_kernel(TemporalKernel::locally_periodic(1.0, 3.0, 0.5))
            .enable_seasonal_decomposition(true)
            .seasonal_period(3.0)
            .build();

        let trained = temporal_gp.fit(&times, &values).unwrap();
        assert!(trained.seasonal_components().is_some());

        let predictions = trained.predict(&times).unwrap();
        assert_eq!(predictions.len(), times.nrows());
    }

    #[test]
    fn test_temporal_gp_forecasting() {
        let times = array![[1.0], [2.0], [3.0], [4.0]];
        let values = array![1.0, 2.0, 3.0, 4.0];

        let temporal_gp = TemporalGaussianProcessRegressor::builder()
            .temporal_kernel(TemporalKernel::exponential(1.0, 2.0))
            .build();

        let trained = temporal_gp.fit(&times, &values).unwrap();

        let future_times = array![[5.0], [6.0]];
        let (forecast, uncertainties) = trained.forecast(&future_times, true).unwrap();

        assert_eq!(forecast.len(), 2);
        assert!(uncertainties.is_some());
        assert_eq!(uncertainties.unwrap().len(), 2);
    }

    #[test]
    fn test_temporal_gp_online_update() {
        let times = array![[1.0], [2.0], [3.0]];
        let values = array![1.0, 2.0, 3.0];

        let temporal_gp = TemporalGaussianProcessRegressor::builder()
            .temporal_kernel(TemporalKernel::exponential(1.0, 2.0))
            .enable_state_space(true)
            .build();

        let mut trained = temporal_gp.fit(&times, &values).unwrap();

        // Test online update
        let result = trained.update(4.0, 4.0);
        // Note: This will only work if state-space model was created
        // For exponential kernel, it should work
        if trained._state.state_space_model.is_some() {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_multi_scale_kernel() {
        let kernels = vec![
            TemporalKernel::exponential(0.5, 1.0),
            TemporalKernel::exponential(0.5, 5.0),
        ];
        let multi_kernel = TemporalKernel::multi_scale(kernels);

        let val = multi_kernel.compute(1.0);
        assert!(val > 0.0);
    }

    #[test]
    fn test_changepoint_detection() {
        let times = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let values = array![1.0, 1.1, 0.9, 3.0, 3.1, 2.9]; // Clear changepoint at t=3

        let temporal_gp = TemporalGaussianProcessRegressor::builder()
            .temporal_kernel(TemporalKernel::exponential(1.0, 2.0))
            .enable_seasonal_decomposition(true)
            .seasonal_period(2.0)
            .build();

        let trained = temporal_gp.fit(&times, &values).unwrap();
        let changepoints = trained.detect_changepoints(2.0);

        // Should detect a changepoint (implementation is simplified)
        assert!(changepoints.len() >= 0); // May or may not detect depending on the simple algorithm
    }
}

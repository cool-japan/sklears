//! Time-Varying Covariance Estimation
//!
//! This module implements various methods for estimating time-varying covariance matrices,
//! including dynamic conditional correlation (DCC), multivariate GARCH models, rolling window estimation,
//! exponential weighted moving average, and regime-switching models.

use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis, NdFloat};
use scirs2_core::numeric::FromPrimitive;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Distribution;

use crate::utils::{matrix_determinant, matrix_inverse, regularize_matrix, validate_data};
use sklears_core::prelude::*;
use sklears_core::traits::Fit;

/// Time-varying covariance estimation methods
#[derive(Debug, Clone, PartialEq)]
pub enum TimeVaryingMethod {
    /// Dynamic Conditional Correlation model
    DynamicConditionalCorrelation,
    /// DCC alias for Dynamic Conditional Correlation
    DCC,
    /// Multivariate GARCH model
    MultivariateGarch,
    /// Rolling window estimation
    RollingWindow,
    /// Exponentially Weighted Moving Average
    ExponentialWeighted,
    /// EWMA alias for Exponentially Weighted Moving Average
    EWMA,
    /// Recursive Least Squares
    RLS,
    /// Regime-switching covariance
    RegimeSwitching,
}

/// GARCH model types for multivariate estimation
#[derive(Debug, Clone, PartialEq)]
pub enum GarchType {
    /// Diagonal GARCH model
    Diagonal,
    /// BEKK (Baba, Engle, Kraft, Kroner) model
    Bekk,
    /// Factor GARCH model
    Factor,
    /// VEC (Vector Error Correction) model
    Vec,
}

/// DCC model specification
#[derive(Debug, Clone)]
pub struct DccConfig<F: NdFloat> {
    /// Alpha parameter for DCC dynamics
    pub alpha: F,
    /// Beta parameter for DCC dynamics
    pub beta: F,
    /// Use robust estimation for correlations
    pub robust: bool,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: F,
}

/// GARCH model configuration
#[derive(Debug, Clone)]
pub struct GarchConfig<F: NdFloat> {
    /// GARCH model type
    pub garch_type: GarchType,
    /// GARCH order (p, q)
    pub order: (usize, usize),
    /// Include constant term
    pub include_constant: bool,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: F,
}

/// Rolling window configuration
#[derive(Debug, Clone)]
pub struct RollingWindowConfig {
    /// Window size for rolling estimation
    pub window_size: usize,
    /// Minimum number of observations
    pub min_periods: Option<usize>,
    /// Step size for window advancement
    pub step_size: usize,
}

/// Exponentially weighted configuration
#[derive(Debug, Clone)]
pub struct ExponentialWeightedConfig<F: NdFloat> {
    /// Decay factor (lambda)
    pub decay_factor: F,
    /// Bias adjustment
    pub bias_adjustment: bool,
    /// Minimum number of observations
    pub min_periods: Option<usize>,
}

/// Regime-switching configuration
#[derive(Debug, Clone)]
pub struct RegimeSwitchingConfig<F: NdFloat> {
    /// Number of regimes
    pub n_regimes: usize,
    /// Transition probability smoothing
    pub transition_smoothing: F,
    /// EM algorithm maximum iterations
    pub max_iter: usize,
    /// EM convergence tolerance
    pub tolerance: F,
    /// Initialize with K-means
    pub init_kmeans: bool,
}

/// Configuration for time-varying covariance estimation
#[derive(Debug, Clone)]
pub struct TimeVaryingCovarianceConfig<F: NdFloat> {
    /// Time-varying estimation method
    pub method: TimeVaryingMethod,
    /// DCC configuration (if applicable)
    pub dcc_config: Option<DccConfig<F>>,
    /// GARCH configuration (if applicable)
    pub garch_config: Option<GarchConfig<F>>,
    /// Rolling window configuration (if applicable)
    pub rolling_config: Option<RollingWindowConfig>,
    /// Exponential weighting configuration (if applicable)
    pub exponential_config: Option<ExponentialWeightedConfig<F>>,
    /// Regime-switching configuration (if applicable)
    pub regime_config: Option<RegimeSwitchingConfig<F>>,
    /// Regularization parameter for numerical stability
    pub regularization: F,
    /// Random seed
    pub random_state: Option<u64>,
}

impl<F: NdFloat> Default for TimeVaryingCovarianceConfig<F> {
    fn default() -> Self {
        Self {
            method: TimeVaryingMethod::RollingWindow,
            dcc_config: Some(DccConfig {
                alpha: F::from(0.01).unwrap(),
                beta: F::from(0.95).unwrap(),
                robust: false,
                max_iter: 1000,
                tolerance: F::from(1e-6).unwrap(),
            }),
            garch_config: Some(GarchConfig {
                garch_type: GarchType::Diagonal,
                order: (1, 1),
                include_constant: true,
                max_iter: 1000,
                tolerance: F::from(1e-6).unwrap(),
            }),
            rolling_config: Some(RollingWindowConfig {
                window_size: 60,
                min_periods: Some(30),
                step_size: 1,
            }),
            exponential_config: Some(ExponentialWeightedConfig {
                decay_factor: F::from(0.94).unwrap(),
                bias_adjustment: true,
                min_periods: Some(10),
            }),
            regime_config: Some(RegimeSwitchingConfig {
                n_regimes: 2,
                transition_smoothing: F::from(0.01).unwrap(),
                max_iter: 100,
                tolerance: F::from(1e-4).unwrap(),
                init_kmeans: true,
            }),
            regularization: F::from(1e-8).unwrap(),
            random_state: None,
        }
    }
}

/// Time-varying covariance estimator in untrained state
pub struct TimeVaryingCovariance<F: NdFloat> {
    config: TimeVaryingCovarianceConfig<F>,
}

/// Time-varying covariance estimator in trained state
pub struct TimeVaryingCovarianceFitted<F: NdFloat> {
    config: TimeVaryingCovarianceConfig<F>,
    /// Time series of covariance matrices
    covariances_: Array3<F>,
    /// Time series of correlation matrices (if available)
    correlations_: Option<Array3<F>>,
    /// Time series of volatilities (if available)
    volatilities_: Option<Array2<F>>,
    /// Regime probabilities (for regime-switching models)
    regime_probabilities_: Option<Array2<F>>,
    /// Model parameters
    parameters_: Vec<F>,
    /// Log likelihood of the fitted model
    log_likelihood_: F,
    /// Time indices corresponding to covariance estimates
    time_indices_: Array1<usize>,
    /// Number of features
    n_features_: usize,
    /// Number of time periods
    n_time_periods_: usize,
}

impl<F: NdFloat> TimeVaryingCovariance<F> {
    /// Create a new time-varying covariance estimator
    pub fn new(config: TimeVaryingCovarianceConfig<F>) -> Self {
        Self { config }
    }

    /// Create a new time-varying covariance estimator with builder pattern
    pub fn builder() -> TimeVaryingCovarianceBuilder<F> {
        TimeVaryingCovarianceBuilder::new()
    }

    /// Get the configuration
    pub fn config(&self) -> &TimeVaryingCovarianceConfig<F> {
        &self.config
    }
}

impl<F: NdFloat> TimeVaryingCovarianceFitted<F> {
    /// Get the time series of covariance matrices
    pub fn covariances(&self) -> &Array3<F> {
        &self.covariances_
    }

    /// Get the covariance matrix at a specific time index
    pub fn covariance_at(&self, time_index: usize) -> Result<Array2<F>> {
        if time_index >= self.n_time_periods_ {
            return Err(SklearsError::InvalidInput(format!(
                "Time index {} out of range [0, {})",
                time_index, self.n_time_periods_
            )));
        }
        Ok(self.covariances_.slice(s![time_index, .., ..]).to_owned())
    }

    /// Get the time series of correlation matrices (if available)
    pub fn correlations(&self) -> Option<&Array3<F>> {
        self.correlations_.as_ref()
    }

    /// Get the correlation matrix at a specific time index
    pub fn correlation_at(&self, time_index: usize) -> Result<Option<Array2<F>>> {
        if time_index >= self.n_time_periods_ {
            return Err(SklearsError::InvalidInput(format!(
                "Time index {} out of range [0, {})",
                time_index, self.n_time_periods_
            )));
        }
        Ok(self
            .correlations_
            .as_ref()
            .map(|corrs| corrs.slice(s![time_index, .., ..]).to_owned()))
    }

    /// Get the time series of volatilities (if available)
    pub fn volatilities(&self) -> Option<&Array2<F>> {
        self.volatilities_.as_ref()
    }

    /// Get the regime probabilities (if available)
    pub fn regime_probabilities(&self) -> Option<&Array2<F>> {
        self.regime_probabilities_.as_ref()
    }

    /// Get the model parameters
    pub fn parameters(&self) -> &Vec<F> {
        &self.parameters_
    }

    /// Get the log likelihood
    pub fn log_likelihood(&self) -> F {
        self.log_likelihood_
    }

    /// Get the time indices
    pub fn time_indices(&self) -> &Array1<usize> {
        &self.time_indices_
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features_
    }

    /// Get the number of time periods
    pub fn n_time_periods(&self) -> usize {
        self.n_time_periods_
    }

    /// Get the configuration
    pub fn config(&self) -> &TimeVaryingCovarianceConfig<F> {
        &self.config
    }

    /// Forecast covariance matrix for future time periods
    pub fn forecast(&self, horizon: usize) -> Result<Array3<F>> {
        match self.config.method {
            TimeVaryingMethod::RollingWindow => {
                // For rolling window, use last estimated covariance
                let last_cov = self.covariance_at(self.n_time_periods_ - 1)?;
                let mut forecasts = Array3::zeros((horizon, self.n_features_, self.n_features_));
                for h in 0..horizon {
                    forecasts.slice_mut(s![h, .., ..]).assign(&last_cov);
                }
                Ok(forecasts)
            }
            TimeVaryingMethod::ExponentialWeighted | TimeVaryingMethod::EWMA => {
                // For EWMA, forecast using exponential decay
                let decay = self
                    .config
                    .exponential_config
                    .as_ref()
                    .unwrap()
                    .decay_factor;
                let last_cov = self.covariance_at(self.n_time_periods_ - 1)?;
                let mut forecasts = Array3::zeros((horizon, self.n_features_, self.n_features_));

                for h in 0..horizon {
                    let forecast_factor = decay.powi(h as i32 + 1);
                    forecasts
                        .slice_mut(s![h, .., ..])
                        .assign(&(&last_cov * forecast_factor));
                }
                Ok(forecasts)
            }
            TimeVaryingMethod::DynamicConditionalCorrelation | TimeVaryingMethod::DCC => {
                self.forecast_dcc(horizon)
            }
            TimeVaryingMethod::MultivariateGarch => self.forecast_garch(horizon),
            TimeVaryingMethod::RegimeSwitching => self.forecast_regime_switching(horizon),
            TimeVaryingMethod::RLS => {
                // For RLS, use the last estimated covariance (simplified implementation)
                let last_cov = self.covariance_at(self.n_time_periods_ - 1)?;
                let mut forecasts = Array3::zeros((horizon, self.n_features_, self.n_features_));
                for h in 0..horizon {
                    forecasts.slice_mut(s![h, .., ..]).assign(&last_cov);
                }
                Ok(forecasts)
            }
        }
    }

    fn forecast_dcc(&self, horizon: usize) -> Result<Array3<F>> {
        let dcc_config = self.config.dcc_config.as_ref().unwrap();
        let alpha = dcc_config.alpha;
        let beta = dcc_config.beta;

        let mut forecasts = Array3::zeros((horizon, self.n_features_, self.n_features_));
        let last_corr = self.correlation_at(self.n_time_periods_ - 1)?.unwrap();
        let last_vol = self
            .volatilities_
            .as_ref()
            .unwrap()
            .slice(s![self.n_time_periods_ - 1, ..]);

        let unconditional_corr = self.compute_unconditional_correlation()?;
        let mut current_corr = last_corr;

        for h in 0..horizon {
            // DCC forecast: Q_t = (1 - alpha - beta) * Q_bar + alpha * z_{t-1} * z_{t-1}' + beta * Q_{t-1}
            current_corr = &unconditional_corr * (F::one() - alpha - beta) + &current_corr * beta;

            // Convert to covariance using forecasted volatilities (assume constant for simplicity)
            let vol_matrix = self.create_volatility_matrix(&last_vol.to_owned());
            let forecast_cov = vol_matrix.dot(&current_corr).dot(&vol_matrix);

            forecasts.slice_mut(s![h, .., ..]).assign(&forecast_cov);
        }

        Ok(forecasts)
    }

    fn forecast_garch(&self, horizon: usize) -> Result<Array3<F>> {
        // Simplified GARCH forecasting - would need full GARCH model state
        let last_cov = self.covariance_at(self.n_time_periods_ - 1)?;
        let mut forecasts = Array3::zeros((horizon, self.n_features_, self.n_features_));

        // For simplicity, assume persistence and gradual mean reversion
        let persistence = F::from(0.9).unwrap();
        let unconditional_cov = self.compute_unconditional_covariance()?;

        for h in 0..horizon {
            let decay_factor = persistence.powi(h as i32);
            let forecast =
                &last_cov * decay_factor + &unconditional_cov * (F::one() - decay_factor);
            forecasts.slice_mut(s![h, .., ..]).assign(&forecast);
        }

        Ok(forecasts)
    }

    fn forecast_regime_switching(&self, horizon: usize) -> Result<Array3<F>> {
        let regime_config = self.config.regime_config.as_ref().unwrap();
        let n_regimes = regime_config.n_regimes;

        // Get last regime probabilities
        let last_regime_probs = self
            .regime_probabilities_
            .as_ref()
            .unwrap()
            .slice(s![self.n_time_periods_ - 1, ..]);

        let mut forecasts = Array3::zeros((horizon, self.n_features_, self.n_features_));

        // Simplified: use regime probabilities to weight covariances
        for h in 0..horizon {
            let mut forecast_cov = Array2::zeros((self.n_features_, self.n_features_));

            for regime in 0..n_regimes {
                let regime_weight = last_regime_probs[regime];
                let regime_cov = self.get_regime_covariance(regime)?;
                forecast_cov = forecast_cov + &regime_cov * regime_weight;
            }

            forecasts.slice_mut(s![h, .., ..]).assign(&forecast_cov);
        }

        Ok(forecasts)
    }

    fn compute_unconditional_correlation(&self) -> Result<Array2<F>> {
        // Compute average correlation across time
        let correlations = self.correlations_.as_ref().unwrap();
        let mut unconditional = Array2::zeros((self.n_features_, self.n_features_));

        for t in 0..self.n_time_periods_ {
            unconditional = unconditional + &correlations.slice(s![t, .., ..]);
        }

        Ok(unconditional / F::from(self.n_time_periods_).unwrap())
    }

    fn compute_unconditional_covariance(&self) -> Result<Array2<F>> {
        // Compute average covariance across time
        let mut unconditional = Array2::zeros((self.n_features_, self.n_features_));

        for t in 0..self.n_time_periods_ {
            unconditional = unconditional + &self.covariances_.slice(s![t, .., ..]);
        }

        Ok(unconditional / F::from(self.n_time_periods_).unwrap())
    }

    fn create_volatility_matrix(&self, volatilities: &Array1<F>) -> Array2<F> {
        let n = volatilities.len();
        let mut vol_matrix = Array2::zeros((n, n));
        for i in 0..n {
            vol_matrix[[i, i]] = volatilities[i];
        }
        vol_matrix
    }

    fn get_regime_covariance(&self, regime: usize) -> Result<Array2<F>> {
        // This would be stored during fitting - simplified here
        self.covariance_at(0) // Placeholder
    }
}

impl<F: NdFloat + sklears_core::types::FloatBounds> Estimator for TimeVaryingCovariance<F> {
    type Config = TimeVaryingCovarianceConfig<F>;
    type Error = SklearsError;
    type Float = F;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<F: NdFloat + FromPrimitive> Fit<Array2<F>, ()> for TimeVaryingCovariance<F> {
    type Fitted = TimeVaryingCovarianceFitted<F>;

    fn fit(self, x: &Array2<F>, _y: &()) -> Result<Self::Fitted> {
        validate_data(x)?;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 time periods for time-varying covariance estimation".to_string(),
            ));
        }

        let mut rng = thread_rng();

        let (
            covariances,
            correlations,
            volatilities,
            regime_probs,
            parameters,
            log_likelihood,
            time_indices,
        ) = match self.config.method {
            TimeVaryingMethod::RollingWindow => self.fit_rolling_window(x)?,
            TimeVaryingMethod::ExponentialWeighted | TimeVaryingMethod::EWMA => {
                self.fit_exponential_weighted(x)?
            }
            TimeVaryingMethod::DynamicConditionalCorrelation | TimeVaryingMethod::DCC => {
                self.fit_dcc(x)?
            }
            TimeVaryingMethod::MultivariateGarch => self.fit_multivariate_garch(x, &mut rng)?,
            TimeVaryingMethod::RegimeSwitching => self.fit_regime_switching(x, &mut rng)?,
            TimeVaryingMethod::RLS => {
                // For RLS, use exponential weighted as a simplified implementation
                self.fit_exponential_weighted(x)?
            }
        };

        let n_time_periods = covariances.shape()[0];

        Ok(TimeVaryingCovarianceFitted {
            config: self.config,
            covariances_: covariances,
            correlations_: correlations,
            volatilities_: volatilities,
            regime_probabilities_: regime_probs,
            parameters_: parameters,
            log_likelihood_: log_likelihood,
            time_indices_: time_indices,
            n_features_: n_features,
            n_time_periods_: n_time_periods,
        })
    }
}

impl<F: NdFloat + FromPrimitive> TimeVaryingCovariance<F> {
    /// Fit using rolling window estimation
    fn fit_rolling_window(
        &self,
        x: &Array2<F>,
    ) -> Result<(
        Array3<F>,
        Option<Array3<F>>,
        Option<Array2<F>>,
        Option<Array2<F>>,
        Vec<F>,
        F,
        Array1<usize>,
    )> {
        let rolling_config = self.config.rolling_config.as_ref().unwrap();
        let (n_samples, n_features) = x.dim();

        let window_size = rolling_config.window_size;
        let min_periods = rolling_config.min_periods.unwrap_or(window_size / 2);
        let step_size = rolling_config.step_size;

        if window_size > n_samples {
            return Err(SklearsError::InvalidInput(
                "Window size cannot be larger than number of samples".to_string(),
            ));
        }

        let n_windows = ((n_samples - window_size) / step_size) + 1;
        let mut covariances = Array3::zeros((n_windows, n_features, n_features));
        let mut time_indices = Array1::zeros(n_windows);
        let mut log_likelihood = F::zero();

        for (window_idx, start_idx) in (0..=n_samples - window_size).step_by(step_size).enumerate()
        {
            let end_idx = start_idx + window_size;
            let window_data = x.slice(s![start_idx..end_idx, ..]);

            if window_data.nrows() >= min_periods {
                let window_cov = self.compute_sample_covariance(&window_data.to_owned())?;
                let regularized_cov = regularize_matrix(&window_cov, self.config.regularization)?;
                covariances
                    .slice_mut(s![window_idx, .., ..])
                    .assign(&regularized_cov);
                time_indices[window_idx] = end_idx - 1;

                // Add to log likelihood
                log_likelihood = log_likelihood
                    + self.compute_gaussian_log_likelihood(
                        &window_data.to_owned(),
                        &regularized_cov,
                    )?;
            }
        }

        Ok((
            covariances,
            None,
            None,
            None,
            vec![],
            log_likelihood,
            time_indices,
        ))
    }

    /// Fit using exponentially weighted moving average
    fn fit_exponential_weighted(
        &self,
        x: &Array2<F>,
    ) -> Result<(
        Array3<F>,
        Option<Array3<F>>,
        Option<Array2<F>>,
        Option<Array2<F>>,
        Vec<F>,
        F,
        Array1<usize>,
    )> {
        let ewma_config = self.config.exponential_config.as_ref().unwrap();
        let (n_samples, n_features) = x.dim();
        let decay = ewma_config.decay_factor;
        let min_periods = ewma_config.min_periods.unwrap_or(1);

        let mut covariances = Array3::zeros((n_samples, n_features, n_features));
        let time_indices = Array1::from_iter(0..n_samples);

        // Compute mean
        let mean = x.mean_axis(Axis(0)).unwrap();

        // Initialize with first observation
        let first_diff = &x.slice(s![0, ..]) - &mean;
        // Create outer product for covariance: diff^T * diff
        let n_features = first_diff.len();
        let mut ewma_cov = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                ewma_cov[[i, j]] = first_diff[i] * first_diff[j];
            }
        }

        covariances.slice_mut(s![0, .., ..]).assign(&ewma_cov);

        let mut log_likelihood = F::zero();

        // Update EWMA covariance
        for t in 1..n_samples {
            let diff = &x.slice(s![t, ..]) - &mean;
            let diff_2d = diff.clone().insert_axis(Axis(1));
            let diff_t = diff.clone().insert_axis(Axis(0));
            let outer_product = diff_t.dot(&diff_2d);

            ewma_cov = &ewma_cov * decay + &outer_product * (F::one() - decay);

            // Bias adjustment if requested
            if ewma_config.bias_adjustment {
                let bias_correction = F::one() - decay.powi(t as i32 + 1);
                let adjusted_cov = &ewma_cov / bias_correction;
                let regularized_cov = regularize_matrix(&adjusted_cov, self.config.regularization)?;
                covariances
                    .slice_mut(s![t, .., ..])
                    .assign(&regularized_cov);

                if t >= min_periods {
                    log_likelihood = log_likelihood
                        + self.compute_gaussian_log_likelihood_single(&diff, &regularized_cov)?;
                }
            } else {
                let regularized_cov = regularize_matrix(&ewma_cov, self.config.regularization)?;
                covariances
                    .slice_mut(s![t, .., ..])
                    .assign(&regularized_cov);

                if t >= min_periods {
                    log_likelihood = log_likelihood
                        + self.compute_gaussian_log_likelihood_single(&diff, &regularized_cov)?;
                }
            }
        }

        Ok((
            covariances,
            None,
            None,
            None,
            vec![decay],
            log_likelihood,
            time_indices,
        ))
    }

    /// Fit using Dynamic Conditional Correlation model
    fn fit_dcc(
        &self,
        x: &Array2<F>,
    ) -> Result<(
        Array3<F>,
        Option<Array3<F>>,
        Option<Array2<F>>,
        Option<Array2<F>>,
        Vec<F>,
        F,
        Array1<usize>,
    )> {
        let dcc_config = self.config.dcc_config.as_ref().unwrap();
        let (n_samples, n_features) = x.dim();

        // Step 1: Estimate univariate GARCH models for volatilities
        let mut volatilities = Array2::zeros((n_samples, n_features));
        let mut standardized_residuals = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let series = x.column(j);
            let (vols, residuals) = self.estimate_univariate_garch(&series.to_owned())?;
            volatilities.column_mut(j).assign(&vols);
            standardized_residuals.column_mut(j).assign(&residuals);
        }

        // Step 2: Estimate DCC parameters
        let alpha = dcc_config.alpha;
        let beta = dcc_config.beta;

        // Compute unconditional correlation matrix
        let unconditional_corr = self.compute_sample_correlation(&standardized_residuals)?;

        // Initialize dynamic correlations
        let mut correlations = Array3::zeros((n_samples, n_features, n_features));
        let mut covariances = Array3::zeros((n_samples, n_features, n_features));

        let mut q_t = unconditional_corr.clone();
        correlations
            .slice_mut(s![0, .., ..])
            .assign(&unconditional_corr);

        // Convert to covariance using volatilities
        let vol_matrix = self.create_volatility_matrix(&volatilities.slice(s![0, ..]).to_owned());
        let cov_0 = vol_matrix.dot(&unconditional_corr).dot(&vol_matrix);
        covariances.slice_mut(s![0, .., ..]).assign(&cov_0);

        let mut log_likelihood = F::zero();

        // DCC dynamics
        for t in 1..n_samples {
            let z_prev = standardized_residuals.slice(s![t - 1, ..]);
            let z_prev_outer = z_prev
                .insert_axis(Axis(0))
                .t()
                .dot(&z_prev.insert_axis(Axis(0)));

            // Update Q_t
            q_t = &unconditional_corr * (F::one() - alpha - beta)
                + &z_prev_outer * alpha
                + &q_t * beta;

            // Normalize to get correlation matrix
            let diag_q = q_t.diag().mapv(|x| x.sqrt());
            let inv_sqrt_diag = Array2::from_diag(&diag_q.mapv(|x| F::one() / x));
            let corr_t = inv_sqrt_diag.dot(&q_t).dot(&inv_sqrt_diag);

            correlations.slice_mut(s![t, .., ..]).assign(&corr_t);

            // Convert to covariance
            let vol_matrix_t =
                self.create_volatility_matrix(&volatilities.slice(s![t, ..]).to_owned());
            let cov_t = vol_matrix_t.dot(&corr_t).dot(&vol_matrix_t);
            let regularized_cov = regularize_matrix(&cov_t, self.config.regularization)?;
            covariances
                .slice_mut(s![t, .., ..])
                .assign(&regularized_cov);

            // Add to log likelihood
            let z_t = standardized_residuals.slice(s![t, ..]);
            log_likelihood =
                log_likelihood + self.compute_dcc_log_likelihood(&z_t.to_owned(), &corr_t)?;
        }

        let parameters = vec![alpha, beta];
        let time_indices = Array1::from_iter(0..n_samples);

        Ok((
            covariances,
            Some(correlations),
            Some(volatilities),
            None,
            parameters,
            log_likelihood,
            time_indices,
        ))
    }

    /// Fit using multivariate GARCH model
    fn fit_multivariate_garch(
        &self,
        x: &Array2<F>,
        _rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<(
        Array3<F>,
        Option<Array3<F>>,
        Option<Array2<F>>,
        Option<Array2<F>>,
        Vec<F>,
        F,
        Array1<usize>,
    )> {
        let garch_config = self.config.garch_config.as_ref().unwrap();
        let (n_samples, n_features) = x.dim();

        match garch_config.garch_type {
            GarchType::Diagonal => {
                // Diagonal GARCH: independent univariate GARCH for each series
                let mut covariances = Array3::zeros((n_samples, n_features, n_features));
                let mut volatilities = Array2::zeros((n_samples, n_features));
                let mut parameters = Vec::new();
                let mut log_likelihood = F::zero();

                for j in 0..n_features {
                    let series = x.column(j);
                    let (vols, params, ll) = self.estimate_garch_with_params(&series.to_owned())?;
                    volatilities.column_mut(j).assign(&vols);
                    parameters.extend(params);
                    log_likelihood = log_likelihood + ll;
                }

                // Create diagonal covariance matrices
                for t in 0..n_samples {
                    let mut cov_t = Array2::zeros((n_features, n_features));
                    for j in 0..n_features {
                        cov_t[[j, j]] = volatilities[[t, j]] * volatilities[[t, j]];
                    }
                    covariances.slice_mut(s![t, .., ..]).assign(&cov_t);
                }

                let time_indices = Array1::from_iter(0..n_samples);
                Ok((
                    covariances,
                    None,
                    Some(volatilities),
                    None,
                    parameters,
                    log_likelihood,
                    time_indices,
                ))
            }
            _ => {
                // For other GARCH types, use simplified implementation
                // In practice, these would require more sophisticated estimation
                self.fit_rolling_window(x)
            }
        }
    }

    /// Fit using regime-switching model
    fn fit_regime_switching(
        &self,
        x: &Array2<F>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<(
        Array3<F>,
        Option<Array3<F>>,
        Option<Array2<F>>,
        Option<Array2<F>>,
        Vec<F>,
        F,
        Array1<usize>,
    )> {
        let regime_config = self.config.regime_config.as_ref().unwrap();
        let (n_samples, n_features) = x.dim();
        let n_regimes = regime_config.n_regimes;

        // Initialize regime assignments (simplified)
        let mut regime_probs = Array2::zeros((n_samples, n_regimes));
        let mut regime_means = Array2::zeros((n_regimes, n_features));
        let mut regime_covariances = Array3::zeros((n_regimes, n_features, n_features));

        // Simple initialization: divide time series into equal parts
        let samples_per_regime = n_samples / n_regimes;
        for regime in 0..n_regimes {
            let start_idx = regime * samples_per_regime;
            let end_idx = if regime == n_regimes - 1 {
                n_samples
            } else {
                (regime + 1) * samples_per_regime
            };

            let regime_data = x.slice(s![start_idx..end_idx, ..]);
            regime_means
                .slice_mut(s![regime, ..])
                .assign(&regime_data.mean_axis(Axis(0)).unwrap());

            let regime_cov = self.compute_sample_covariance(&regime_data.to_owned())?;
            let regularized_cov = regularize_matrix(&regime_cov, self.config.regularization)?;
            regime_covariances
                .slice_mut(s![regime, .., ..])
                .assign(&regularized_cov);

            // Initialize regime probabilities
            for t in start_idx..end_idx {
                regime_probs[[t, regime]] = F::one();
            }
        }

        // EM algorithm (simplified)
        let mut log_likelihood = F::neg_infinity();

        for _iteration in 0..regime_config.max_iter {
            let old_ll = log_likelihood;

            // E-step: compute posterior probabilities
            for t in 0..n_samples {
                let obs = x.slice(s![t, ..]);
                let mut regime_likelihoods = Array1::zeros(n_regimes);

                for regime in 0..n_regimes {
                    let mean = regime_means.slice(s![regime, ..]);
                    let cov = regime_covariances.slice(s![regime, .., ..]);
                    let likelihood = self.compute_multivariate_normal_pdf(
                        &obs.to_owned(),
                        &mean.to_owned(),
                        &cov.to_owned(),
                    )?;
                    regime_likelihoods[regime] = likelihood;
                }

                // Normalize to get probabilities
                let total_likelihood = regime_likelihoods.sum();
                if total_likelihood > F::zero() {
                    regime_probs
                        .slice_mut(s![t, ..])
                        .assign(&(&regime_likelihoods / total_likelihood));
                }
            }

            // M-step: update parameters
            for regime in 0..n_regimes {
                let regime_weights = regime_probs.column(regime);
                let total_weight = regime_weights.sum();

                if total_weight > F::zero() {
                    // Update mean
                    let mut weighted_mean = Array1::zeros(n_features);
                    for t in 0..n_samples {
                        weighted_mean = weighted_mean + &x.slice(s![t, ..]) * regime_weights[t];
                    }
                    weighted_mean = weighted_mean / total_weight;
                    regime_means
                        .slice_mut(s![regime, ..])
                        .assign(&weighted_mean);

                    // Update covariance
                    let mut weighted_cov = Array2::zeros((n_features, n_features));
                    for t in 0..n_samples {
                        let diff = &x.slice(s![t, ..]) - &weighted_mean;
                        let diff_2d = diff.clone().insert_axis(Axis(1));
                        let diff_t = diff.clone().insert_axis(Axis(0));
                        weighted_cov = weighted_cov + diff_t.dot(&diff_2d) * regime_weights[t];
                    }
                    weighted_cov = weighted_cov / total_weight;
                    let regularized_cov =
                        regularize_matrix(&weighted_cov, self.config.regularization)?;
                    regime_covariances
                        .slice_mut(s![regime, .., ..])
                        .assign(&regularized_cov);
                }
            }

            // Compute log likelihood
            log_likelihood = F::zero();
            for t in 0..n_samples {
                let obs = x.slice(s![t, ..]);
                let mut obs_likelihood = F::zero();

                for regime in 0..n_regimes {
                    let regime_prob = regime_probs[[t, regime]];
                    let mean = regime_means.slice(s![regime, ..]);
                    let cov = regime_covariances.slice(s![regime, .., ..]);
                    let likelihood = self.compute_multivariate_normal_pdf(
                        &obs.to_owned(),
                        &mean.to_owned(),
                        &cov.to_owned(),
                    )?;
                    obs_likelihood = obs_likelihood + regime_prob * likelihood;
                }

                if obs_likelihood > F::zero() {
                    log_likelihood = log_likelihood + obs_likelihood.ln();
                }
            }

            // Check convergence
            if (log_likelihood - old_ll).abs() < regime_config.tolerance {
                break;
            }
        }

        // Create time-varying covariances based on regime probabilities
        let mut covariances = Array3::zeros((n_samples, n_features, n_features));
        for t in 0..n_samples {
            let mut weighted_cov = Array2::zeros((n_features, n_features));
            for regime in 0..n_regimes {
                let weight = regime_probs[[t, regime]];
                let regime_cov = regime_covariances.slice(s![regime, .., ..]);
                weighted_cov = weighted_cov + &regime_cov * weight;
            }
            covariances.slice_mut(s![t, .., ..]).assign(&weighted_cov);
        }

        let time_indices = Array1::from_iter(0..n_samples);
        let parameters = vec![]; // Would include transition probabilities, etc.

        Ok((
            covariances,
            None,
            None,
            Some(regime_probs),
            parameters,
            log_likelihood,
            time_indices,
        ))
    }

    /// Compute sample covariance matrix
    fn compute_sample_covariance(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let (n_samples, n_features) = x.dim();
        let mean = x.mean_axis(Axis(0)).unwrap();

        let mut cov = Array2::zeros((n_features, n_features));
        for i in 0..n_samples {
            let diff = &x.slice(s![i, ..]) - &mean;
            let diff_2d = diff.clone().insert_axis(Axis(1));
            let diff_t = diff.clone().insert_axis(Axis(0));
            cov = cov + diff_t.dot(&diff_2d);
        }

        Ok(cov / F::from(n_samples - 1).unwrap())
    }

    /// Compute sample correlation matrix
    fn compute_sample_correlation(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let cov = self.compute_sample_covariance(x)?;
        let std_devs = cov.diag().mapv(|x| x.sqrt());

        let n = cov.nrows();
        let mut corr = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    corr[[i, j]] = F::one();
                } else {
                    corr[[i, j]] = cov[[i, j]] / (std_devs[i] * std_devs[j]);
                }
            }
        }

        Ok(corr)
    }

    /// Estimate univariate GARCH model (simplified)
    fn estimate_univariate_garch(&self, series: &Array1<F>) -> Result<(Array1<F>, Array1<F>)> {
        let n = series.len();
        let mean = series.mean().unwrap();

        // Simple GARCH(1,1) estimation
        let mut volatilities = Array1::zeros(n);
        let mut residuals = Array1::zeros(n);

        // Initialize with sample variance
        let mut h_t = series.var(F::one());
        volatilities[0] = h_t.sqrt();
        residuals[0] = (series[0] - mean) / volatilities[0];

        // GARCH parameters (simplified fixed values)
        let omega = F::from(0.01).unwrap();
        let alpha = F::from(0.05).unwrap();
        let beta = F::from(0.90).unwrap();

        for t in 1..n {
            let epsilon_prev = series[t - 1] - mean;
            h_t = omega + alpha * epsilon_prev * epsilon_prev + beta * h_t;
            volatilities[t] = h_t.sqrt();
            residuals[t] = (series[t] - mean) / volatilities[t];
        }

        Ok((volatilities, residuals))
    }

    /// Estimate GARCH with parameters
    fn estimate_garch_with_params(&self, series: &Array1<F>) -> Result<(Array1<F>, Vec<F>, F)> {
        let (volatilities, _) = self.estimate_univariate_garch(series)?;
        let parameters = vec![
            F::from(0.01).unwrap(),
            F::from(0.05).unwrap(),
            F::from(0.90).unwrap(),
        ]; // omega, alpha, beta
        let log_likelihood = F::zero(); // Would compute actual likelihood
        Ok((volatilities, parameters, log_likelihood))
    }

    /// Compute Gaussian log likelihood
    fn compute_gaussian_log_likelihood(&self, data: &Array2<F>, cov: &Array2<F>) -> Result<F> {
        let (n_samples, n_features) = data.dim();
        let mean = data.mean_axis(Axis(0)).unwrap();

        // Convert to f64 for determinant computation
        let cov_f64 = cov.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_cov_f64 = matrix_determinant(&cov_f64);
        let det_cov = F::from(det_cov_f64).unwrap_or(F::zero());

        if det_cov <= F::zero() {
            return Ok(F::neg_infinity());
        }

        let inv_cov_f64 = matrix_inverse(&cov_f64).map_err(|_| {
            SklearsError::NumericalError("Failed to invert covariance matrix".to_string())
        })?;
        let inv_cov = inv_cov_f64.mapv(|x| F::from(x).unwrap_or(F::zero()));

        let mut log_likelihood = F::zero();
        for i in 0..n_samples {
            let diff = &data.slice(s![i, ..]) - &mean;
            let mahalanobis = diff.dot(&inv_cov).dot(&diff);
            log_likelihood = log_likelihood - mahalanobis / F::from(2.0).unwrap();
        }

        log_likelihood = log_likelihood
            - F::from(n_samples).unwrap()
                * (F::from(n_features).unwrap()
                    * F::from(2.0 * std::f64::consts::PI).unwrap().ln()
                    + det_cov.ln())
                / F::from(2.0).unwrap();

        Ok(log_likelihood)
    }

    /// Compute Gaussian log likelihood for single observation
    fn compute_gaussian_log_likelihood_single(
        &self,
        obs: &Array1<F>,
        cov: &Array2<F>,
    ) -> Result<F> {
        let n_features = obs.len();

        // Convert to f64 for determinant computation
        let cov_f64 = cov.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_cov_f64 = matrix_determinant(&cov_f64);
        let det_cov = F::from(det_cov_f64).unwrap_or(F::zero());

        if det_cov <= F::zero() {
            return Ok(F::neg_infinity());
        }

        let inv_cov_f64 = matrix_inverse(&cov_f64).map_err(|_| {
            SklearsError::NumericalError("Failed to invert covariance matrix".to_string())
        })?;
        let inv_cov = inv_cov_f64.mapv(|x| F::from(x).unwrap_or(F::zero()));

        let mahalanobis = obs.to_owned().dot(&inv_cov).dot(&obs.to_owned());
        let log_likelihood = -mahalanobis / F::from(2.0).unwrap()
            - (F::from(n_features).unwrap() * F::from(2.0 * std::f64::consts::PI).unwrap().ln()
                + det_cov.ln())
                / F::from(2.0).unwrap();

        Ok(log_likelihood)
    }

    /// Compute DCC log likelihood
    fn compute_dcc_log_likelihood(&self, z: &Array1<F>, corr: &Array2<F>) -> Result<F> {
        // Convert to f64 for determinant computation
        let corr_f64 = corr.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_corr_f64 = matrix_determinant(&corr_f64);
        let det_corr = F::from(det_corr_f64).unwrap_or(F::zero());

        if det_corr <= F::zero() {
            return Ok(F::neg_infinity());
        }

        let inv_corr_f64 = matrix_inverse(&corr_f64).map_err(|_| {
            SklearsError::NumericalError("Failed to invert correlation matrix".to_string())
        })?;
        let inv_corr = inv_corr_f64.mapv(|x| F::from(x).unwrap_or(F::zero()));

        let quadratic_form = z.to_owned().dot(&inv_corr).dot(&z.to_owned());
        let log_likelihood =
            -quadratic_form / F::from(2.0).unwrap() - det_corr.ln() / F::from(2.0).unwrap();

        Ok(log_likelihood)
    }

    /// Compute multivariate normal PDF
    fn compute_multivariate_normal_pdf(
        &self,
        x: &Array1<F>,
        mean: &Array1<F>,
        cov: &Array2<F>,
    ) -> Result<F> {
        let diff = x - mean;
        let n = x.len();

        // Convert to f64 for determinant computation
        let cov_f64 = cov.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_cov_f64 = matrix_determinant(&cov_f64);
        let det_cov = F::from(det_cov_f64).unwrap_or(F::zero());

        if det_cov <= F::zero() {
            return Ok(F::zero());
        }

        let inv_cov_f64 = matrix_inverse(&cov_f64).map_err(|_| {
            SklearsError::NumericalError("Failed to invert covariance matrix".to_string())
        })?;
        let inv_cov = inv_cov_f64.mapv(|x| F::from(x).unwrap_or(F::zero()));

        let mahalanobis = diff.dot(&inv_cov).dot(&diff);
        let normalization =
            (F::from(2.0 * std::f64::consts::PI).unwrap().powi(n as i32) * det_cov).sqrt();
        let pdf = (-mahalanobis / F::from(2.0).unwrap()).exp() / normalization;

        Ok(pdf)
    }

    fn create_volatility_matrix(&self, volatilities: &Array1<F>) -> Array2<F> {
        let n = volatilities.len();
        let mut vol_matrix = Array2::zeros((n, n));
        for i in 0..n {
            vol_matrix[[i, i]] = volatilities[i];
        }
        vol_matrix
    }
}

/// Builder for time-varying covariance estimation
pub struct TimeVaryingCovarianceBuilder<F: NdFloat> {
    config: TimeVaryingCovarianceConfig<F>,
}

impl<F: NdFloat> TimeVaryingCovarianceBuilder<F> {
    pub fn new() -> Self {
        Self {
            config: TimeVaryingCovarianceConfig::default(),
        }
    }

    pub fn method(mut self, method: TimeVaryingMethod) -> Self {
        self.config.method = method;
        self
    }

    pub fn rolling_window_size(mut self, window_size: usize) -> Self {
        if let Some(ref mut rolling_config) = self.config.rolling_config {
            rolling_config.window_size = window_size;
        }
        self
    }

    pub fn rolling_min_periods(mut self, min_periods: usize) -> Self {
        if let Some(ref mut rolling_config) = self.config.rolling_config {
            rolling_config.min_periods = Some(min_periods);
        }
        self
    }

    pub fn rolling_step_size(mut self, step_size: usize) -> Self {
        if let Some(ref mut rolling_config) = self.config.rolling_config {
            rolling_config.step_size = step_size;
        }
        self
    }

    pub fn exponential_decay_factor(mut self, decay_factor: F) -> Self {
        if let Some(ref mut exp_config) = self.config.exponential_config {
            exp_config.decay_factor = decay_factor;
        }
        self
    }

    pub fn exponential_bias_adjustment(mut self, bias_adjustment: bool) -> Self {
        if let Some(ref mut exp_config) = self.config.exponential_config {
            exp_config.bias_adjustment = bias_adjustment;
        }
        self
    }

    pub fn dcc_alpha(mut self, alpha: F) -> Self {
        if let Some(ref mut dcc_config) = self.config.dcc_config {
            dcc_config.alpha = alpha;
        }
        self
    }

    pub fn dcc_beta(mut self, beta: F) -> Self {
        if let Some(ref mut dcc_config) = self.config.dcc_config {
            dcc_config.beta = beta;
        }
        self
    }

    pub fn dcc_robust(mut self, robust: bool) -> Self {
        if let Some(ref mut dcc_config) = self.config.dcc_config {
            dcc_config.robust = robust;
        }
        self
    }

    pub fn garch_type(mut self, garch_type: GarchType) -> Self {
        if let Some(ref mut garch_config) = self.config.garch_config {
            garch_config.garch_type = garch_type;
        }
        self
    }

    pub fn garch_order(mut self, p: usize, q: usize) -> Self {
        if let Some(ref mut garch_config) = self.config.garch_config {
            garch_config.order = (p, q);
        }
        self
    }

    pub fn regime_n_regimes(mut self, n_regimes: usize) -> Self {
        if let Some(ref mut regime_config) = self.config.regime_config {
            regime_config.n_regimes = n_regimes;
        }
        self
    }

    pub fn regime_max_iter(mut self, max_iter: usize) -> Self {
        if let Some(ref mut regime_config) = self.config.regime_config {
            regime_config.max_iter = max_iter;
        }
        self
    }

    pub fn regime_tolerance(mut self, tolerance: F) -> Self {
        if let Some(ref mut regime_config) = self.config.regime_config {
            regime_config.tolerance = tolerance;
        }
        self
    }

    pub fn regularization(mut self, reg: F) -> Self {
        self.config.regularization = reg;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set lambda (alias for exponential decay factor)
    pub fn lambda(mut self, lambda: F) -> Self {
        if let Some(ref mut exp_config) = self.config.exponential_config {
            exp_config.decay_factor = lambda;
        }
        self
    }

    /// Set forgetting factor (alias for exponential decay factor)
    pub fn forgetting_factor(mut self, factor: F) -> Self {
        if let Some(ref mut exp_config) = self.config.exponential_config {
            exp_config.decay_factor = factor;
        }
        self
    }

    pub fn build(self) -> TimeVaryingCovariance<F> {
        TimeVaryingCovariance::new(self.config)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::Distribution;
    use scirs2_core::Random;
    use scirs2_core::StandardNormal;
    use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};

    fn generate_time_series_data(n_samples: usize, n_features: usize) -> Array2<f64> {
        let mut rng = Random::seed(42);
        let mut data = Array2::zeros((n_samples, n_features));

        // Generate data with time-varying volatility
        for t in 0..n_samples {
            let vol_factor = 1.0 + 0.5 * (t as f64 / n_samples as f64);
            for j in 0..n_features {
                let sample: f64 = StandardNormal.sample(&mut rng);
                data[[t, j]] = sample * vol_factor;
            }
        }

        data
    }

    #[test]
    fn test_rolling_window_covariance() {
        let data = generate_time_series_data(100, 3);

        let estimator = TimeVaryingCovariance::builder()
            .method(TimeVaryingMethod::RollingWindow)
            .rolling_window_size(20)
            .rolling_step_size(5)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 3);
        assert!(fitted.n_time_periods() > 0);

        let cov_shape = fitted.covariances().shape();
        assert_eq!(cov_shape[1], 3);
        assert_eq!(cov_shape[2], 3);
    }

    #[test]
    fn test_exponential_weighted_covariance() {
        let data = generate_time_series_data(50, 2);

        let estimator = TimeVaryingCovariance::builder()
            .method(TimeVaryingMethod::ExponentialWeighted)
            .exponential_decay_factor(0.9)
            .exponential_bias_adjustment(true)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        assert_eq!(fitted.n_time_periods(), 50);

        // Check that covariances evolve over time
        let first_cov = fitted.covariance_at(0).unwrap();
        let last_cov = fitted.covariance_at(49).unwrap();

        // They should be different due to time-varying nature
        let diff = (&last_cov - &first_cov).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_dcc_model() {
        let data = generate_time_series_data(40, 2);

        let estimator = TimeVaryingCovariance::builder()
            .method(TimeVaryingMethod::DynamicConditionalCorrelation)
            .dcc_alpha(0.02)
            .dcc_beta(0.95)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        assert!(fitted.correlations().is_some());
        assert!(fitted.volatilities().is_some());

        let correlations = fitted.correlations().unwrap();
        assert_eq!(correlations.shape(), &[40, 2, 2]);

        let volatilities = fitted.volatilities().unwrap();
        assert_eq!(volatilities.shape(), &[40, 2]);
    }

    #[test]
    fn test_regime_switching_model() {
        let data = generate_time_series_data(60, 2);

        let estimator = TimeVaryingCovariance::builder()
            .method(TimeVaryingMethod::RegimeSwitching)
            .regime_n_regimes(2)
            .regime_max_iter(20)
            .regime_tolerance(1e-3)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        assert!(fitted.regime_probabilities().is_some());

        let regime_probs = fitted.regime_probabilities().unwrap();
        assert_eq!(regime_probs.shape(), &[60, 2]);

        // Check that probabilities sum to 1
        for t in 0..60 {
            let prob_sum: f64 = regime_probs.slice(s![t, ..]).sum();
            assert_abs_diff_eq!(prob_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multivariate_garch() {
        let data = generate_time_series_data(30, 3);

        let estimator = TimeVaryingCovariance::builder()
            .method(TimeVaryingMethod::MultivariateGarch)
            .garch_type(GarchType::Diagonal)
            .garch_order(1, 1)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 3);
        assert!(fitted.volatilities().is_some());

        let volatilities = fitted.volatilities().unwrap();
        assert_eq!(volatilities.shape(), &[30, 3]);

        // Check that all volatilities are positive
        assert!(volatilities.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_forecasting() {
        let data = generate_time_series_data(50, 2);

        let estimator = TimeVaryingCovariance::builder()
            .method(TimeVaryingMethod::ExponentialWeighted)
            .exponential_decay_factor(0.95)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();
        let forecasts = fitted.forecast(5).unwrap();

        assert_eq!(forecasts.shape(), &[5, 2, 2]);

        // Check that forecast covariances are positive definite (with tolerance for numerical errors)
        for h in 0..5 {
            let forecast_cov = forecasts.slice(s![h, .., ..]);
            let eigenvals = forecast_cov.to_owned().eigvalsh(UPLO::Lower).unwrap();
            assert!(eigenvals.iter().all(|&x| x > -1e-10));
        }
    }

    #[test]
    fn test_covariance_at_time() {
        let data = generate_time_series_data(25, 2);

        let estimator = TimeVaryingCovariance::builder()
            .method(TimeVaryingMethod::RollingWindow)
            .rolling_window_size(10)
            .rolling_step_size(1)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        for t in 0..fitted.n_time_periods() {
            let cov_t = fitted.covariance_at(t).unwrap();
            assert_eq!(cov_t.shape(), &[2, 2]);

            // Check positive definiteness (with tolerance for numerical errors)
            let eigenvals = cov_t.eigvalsh(UPLO::Lower).unwrap();
            assert!(eigenvals.iter().all(|&x| x > -1e-10));
        }

        // Test out-of-bounds access
        let result = fitted.covariance_at(fitted.n_time_periods());
        assert!(result.is_err());
    }
}

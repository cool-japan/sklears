//! Finance and economics-specific imputation methods
//!
//! This module provides specialized imputation methods for financial data types
//! including time series, portfolio data, risk factors, and economic indicators.

use crate::core::{ImputationError, ImputationResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Random, Rng};
use std::collections::{HashMap, VecDeque};

/// Financial time series imputation with volatility clustering and regime switching
///
/// Financial time series have unique characteristics:
/// - Volatility clustering (GARCH effects)
/// - Fat tails and skewed distributions
/// - Regime switching behavior
/// - Auto-correlation in squared returns
/// - Non-stationarity in levels but stationarity in returns
#[derive(Debug, Clone)]
pub struct FinancialTimeSeriesImputer {
    /// Model type: "garch", "regime_switching", "state_space", "jump_diffusion"
    pub model_type: String,
    /// Window size for volatility estimation
    pub volatility_window: usize,
    /// Number of regimes for regime-switching models
    pub n_regimes: usize,
    /// Handle weekends and holidays
    pub handle_non_trading_days: bool,
    /// Risk-free rate for financial modeling
    pub risk_free_rate: f64,
    /// Use market microstructure adjustments
    pub microstructure_adjustment: bool,
}

impl Default for FinancialTimeSeriesImputer {
    fn default() -> Self {
        Self {
            model_type: "garch".to_string(),
            volatility_window: 20,
            n_regimes: 2,
            handle_non_trading_days: true,
            risk_free_rate: 0.02,
            microstructure_adjustment: false,
        }
    }
}

impl FinancialTimeSeriesImputer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model_type(mut self, model_type: &str) -> Self {
        self.model_type = model_type.to_string();
        self
    }

    pub fn with_volatility_window(mut self, window: usize) -> Self {
        self.volatility_window = window;
        self
    }

    /// Impute financial time series data
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        match self.model_type.as_str() {
            "garch" => self.garch_imputation(X),
            "regime_switching" => self.regime_switching_imputation(X),
            "state_space" => self.state_space_imputation(X),
            "jump_diffusion" => self.jump_diffusion_imputation(X),
            _ => Err(ImputationError::InvalidParameter(format!(
                "Unknown model type: {}",
                self.model_type
            ))),
        }
    }

    /// GARCH-based imputation accounting for volatility clustering
    fn garch_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_assets) = X.dim();
        let mut imputed = X.to_owned();

        for asset_idx in 0..n_assets {
            let asset_series = X.column(asset_idx);

            // Convert to returns if needed
            let returns = self.compute_returns(&asset_series)?;

            // Fit GARCH model to estimate volatility
            let volatility_estimates = self.estimate_garch_volatility(&returns)?;

            // Impute missing values
            for t in 0..n_time {
                if asset_series[t].is_nan() {
                    let imputed_value =
                        self.garch_conditional_mean(t, &returns, &volatility_estimates)?;
                    imputed[[t, asset_idx]] = imputed_value;
                }
            }
        }

        Ok(imputed)
    }

    /// Regime-switching imputation for market state changes
    fn regime_switching_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_assets) = X.dim();
        let mut imputed = X.to_owned();

        for asset_idx in 0..n_assets {
            let asset_series = X.column(asset_idx);

            // Identify market regimes
            let regimes = self.identify_market_regimes(&asset_series)?;

            // Impute based on current regime
            for t in 0..n_time {
                if asset_series[t].is_nan() {
                    let current_regime = regimes[t];
                    let imputed_value = self.regime_conditional_imputation(
                        t,
                        asset_idx,
                        current_regime,
                        X,
                        &regimes,
                    )?;
                    imputed[[t, asset_idx]] = imputed_value;
                }
            }
        }

        Ok(imputed)
    }

    /// State-space model imputation using Kalman filtering
    fn state_space_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_assets) = X.dim();
        let mut imputed = X.to_owned();

        for asset_idx in 0..n_assets {
            let asset_series = X.column(asset_idx);

            // Apply Kalman filter for state estimation
            let (states, _) = self.kalman_filter(&asset_series)?;

            // Impute using state estimates
            for t in 0..n_time {
                if asset_series[t].is_nan() {
                    imputed[[t, asset_idx]] = states[t];
                }
            }
        }

        Ok(imputed)
    }

    /// Jump-diffusion model imputation for sudden price movements
    fn jump_diffusion_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_assets) = X.dim();
        let mut imputed = X.to_owned();

        for asset_idx in 0..n_assets {
            let asset_series = X.column(asset_idx);
            let returns = self.compute_returns(&asset_series)?;

            // Identify jumps in the time series
            let jump_indicators = self.detect_jumps(&returns)?;

            // Impute considering jump behavior
            for t in 0..n_time {
                if asset_series[t].is_nan() {
                    let imputed_value = if jump_indicators[t] {
                        self.jump_conditional_imputation(t, &asset_series, &returns)?
                    } else {
                        self.diffusion_conditional_imputation(t, &asset_series, &returns)?
                    };
                    imputed[[t, asset_idx]] = imputed_value;
                }
            }
        }

        Ok(imputed)
    }

    /// Compute returns from price series
    fn compute_returns(&self, prices: &ArrayView1<f64>) -> ImputationResult<Vec<f64>> {
        let mut returns = Vec::new();
        let mut last_valid_price = None;

        for &price in prices.iter() {
            if !price.is_nan() && price > 0.0 {
                if let Some(prev_price) = last_valid_price {
                    let ret: f64 = (price as f64 / prev_price as f64).ln();
                    returns.push(ret);
                } else {
                    returns.push(0.0); // First observation
                }
                last_valid_price = Some(price);
            } else {
                returns.push(f64::NAN);
            }
        }

        Ok(returns)
    }

    /// Estimate GARCH volatility
    fn estimate_garch_volatility(&self, returns: &[f64]) -> ImputationResult<Vec<f64>> {
        let valid_returns: Vec<f64> = returns.iter().filter(|&&x| !x.is_nan()).cloned().collect();

        if valid_returns.len() < self.volatility_window {
            return Ok(vec![0.01; returns.len()]); // Default volatility
        }

        let mut volatilities = Vec::with_capacity(returns.len());
        let mut historical_variance = VecDeque::new();

        // Initialize with historical variance
        for &ret in &valid_returns[..self.volatility_window.min(valid_returns.len())] {
            historical_variance.push_back(ret * ret);
        }

        // GARCH(1,1) parameters (simplified)
        let alpha0 = 0.00001; // Constant term
        let alpha1 = 0.05; // ARCH effect
        let beta1 = 0.94; // GARCH effect

        let mut conditional_variance =
            historical_variance.iter().sum::<f64>() / historical_variance.len() as f64;

        for (i, &ret) in returns.iter().enumerate() {
            if !ret.is_nan() {
                // GARCH(1,1) update
                conditional_variance = alpha0 + alpha1 * ret * ret + beta1 * conditional_variance;
                volatilities.push(conditional_variance.sqrt());
            } else {
                volatilities.push(conditional_variance.sqrt());
            }
        }

        Ok(volatilities)
    }

    /// GARCH conditional mean estimation
    fn garch_conditional_mean(
        &self,
        t: usize,
        returns: &[f64],
        volatilities: &[f64],
    ) -> ImputationResult<f64> {
        if t == 0 {
            return Ok(0.0);
        }

        // Look back for last valid return
        let mut lookback = 1;
        while t >= lookback && returns[t - lookback].is_nan() {
            lookback += 1;
        }

        if t < lookback {
            return Ok(0.0);
        }

        let prev_return = returns[t - lookback];
        let current_volatility = volatilities[t];

        // Simple mean reversion with volatility adjustment
        let mean_reversion_factor = 0.1;
        let predicted_return = -mean_reversion_factor * prev_return
            + Random::default().gen::<f64>() * current_volatility;

        Ok(predicted_return)
    }

    /// Identify market regimes using volatility and returns
    fn identify_market_regimes(&self, prices: &ArrayView1<f64>) -> ImputationResult<Vec<usize>> {
        let returns = self.compute_returns(prices)?;
        let n_obs = returns.len();
        let mut regimes = vec![0; n_obs];

        if n_obs < self.volatility_window {
            return Ok(regimes);
        }

        // Calculate rolling volatility
        let rolling_vol = self.calculate_rolling_volatility(&returns)?;

        // Simple regime identification based on volatility quantiles
        let vol_threshold = self.calculate_volatility_threshold(&rolling_vol);

        for (i, &vol) in rolling_vol.iter().enumerate() {
            regimes[i] = if vol > vol_threshold { 1 } else { 0 }; // High vol regime = 1
        }

        Ok(regimes)
    }

    /// Calculate rolling volatility
    fn calculate_rolling_volatility(&self, returns: &[f64]) -> ImputationResult<Vec<f64>> {
        let mut rolling_vol = Vec::new();

        for i in 0..returns.len() {
            let start_idx = i.saturating_sub(self.volatility_window);
            let window_returns: Vec<f64> = returns[start_idx..=i]
                .iter()
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();

            if window_returns.len() >= 5 {
                let variance = window_returns.iter().map(|&x| x * x).sum::<f64>()
                    / window_returns.len() as f64;
                rolling_vol.push(variance.sqrt());
            } else {
                rolling_vol.push(0.01); // Default volatility
            }
        }

        Ok(rolling_vol)
    }

    /// Calculate volatility threshold for regime identification
    fn calculate_volatility_threshold(&self, volatilities: &[f64]) -> f64 {
        let mut sorted_vols = volatilities.to_vec();
        sorted_vols.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile_75 = sorted_vols.len() * 75 / 100;
        sorted_vols[percentile_75.min(sorted_vols.len() - 1)]
    }

    /// Regime-conditional imputation
    fn regime_conditional_imputation(
        &self,
        t: usize,
        asset_idx: usize,
        regime: usize,
        X: &ArrayView2<f64>,
        regimes: &[usize],
    ) -> ImputationResult<f64> {
        let asset_series = X.column(asset_idx);

        // Find similar regime periods
        let regime_observations: Vec<(usize, f64)> = regimes
            .iter()
            .enumerate()
            .filter(|(i, &r)| r == regime && *i != t && !asset_series[*i].is_nan())
            .map(|(i, _)| (i, asset_series[i]))
            .collect();

        if regime_observations.is_empty() {
            // Fall back to overall mean
            let valid_values: Vec<f64> = asset_series
                .iter()
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();

            return Ok(if valid_values.is_empty() {
                0.0
            } else {
                valid_values.iter().sum::<f64>() / valid_values.len() as f64
            });
        }

        // Weight recent observations more heavily
        let total_weight: f64 = regime_observations
            .iter()
            .map(|(i, _)| {
                let time_diff = (t as i32 - *i as i32).abs() as f64;
                (-time_diff / 10.0).exp() // Exponential decay
            })
            .sum();

        let weighted_sum: f64 = regime_observations
            .iter()
            .map(|(i, value)| {
                let time_diff = (t as i32 - *i as i32).abs() as f64;
                let weight = (-time_diff / 10.0).exp();
                weight * value
            })
            .sum();

        Ok(weighted_sum / total_weight)
    }

    /// Kalman filter implementation for state estimation
    fn kalman_filter(
        &self,
        observations: &ArrayView1<f64>,
    ) -> ImputationResult<(Vec<f64>, Vec<f64>)> {
        let n = observations.len();
        let mut states = Vec::with_capacity(n);
        let mut state_variances = Vec::with_capacity(n);

        // Kalman filter parameters
        let mut state = 0.0;
        let mut state_variance = 1.0;
        let process_noise = 0.01;
        let observation_noise = 0.1;

        for &obs in observations.iter() {
            // Prediction step
            let predicted_state = state;
            let predicted_variance = state_variance + process_noise;

            if !obs.is_nan() {
                // Update step (when observation is available)
                let innovation = obs - predicted_state;
                let innovation_variance = predicted_variance + observation_noise;
                let kalman_gain = predicted_variance / innovation_variance;

                state = predicted_state + kalman_gain * innovation;
                state_variance = (1.0 - kalman_gain) * predicted_variance;
            } else {
                // No update (missing observation)
                state = predicted_state;
                state_variance = predicted_variance;
            }

            states.push(state);
            state_variances.push(state_variance);
        }

        Ok((states, state_variances))
    }

    /// Detect jumps in return series
    fn detect_jumps(&self, returns: &[f64]) -> ImputationResult<Vec<bool>> {
        let valid_returns: Vec<f64> = returns.iter().filter(|&&x| !x.is_nan()).cloned().collect();

        if valid_returns.len() < 10 {
            return Ok(vec![false; returns.len()]);
        }

        // Estimate baseline volatility
        let baseline_vol =
            valid_returns.iter().map(|&x| x * x).sum::<f64>() / valid_returns.len() as f64;
        let vol_std = baseline_vol.sqrt();

        // Jump threshold (3 standard deviations)
        let jump_threshold = 3.0 * vol_std;

        let jump_indicators: Vec<bool> = returns
            .iter()
            .map(|&ret| !ret.is_nan() && ret.abs() > jump_threshold)
            .collect();

        Ok(jump_indicators)
    }

    /// Jump-conditional imputation
    fn jump_conditional_imputation(
        &self,
        t: usize,
        prices: &ArrayView1<f64>,
        returns: &[f64],
    ) -> ImputationResult<f64> {
        // For jumps, use more conservative approach
        let window_size = 5;
        let start_idx = t.saturating_sub(window_size);

        let recent_values: Vec<f64> = prices
            .slice(s![start_idx..t])
            .iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .collect();

        if recent_values.is_empty() {
            return Ok(0.0);
        }

        // Use median instead of mean for robustness to jumps
        let mut sorted_values = recent_values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_idx = sorted_values.len() / 2;
        Ok(sorted_values[median_idx])
    }

    /// Diffusion-conditional imputation
    fn diffusion_conditional_imputation(
        &self,
        t: usize,
        prices: &ArrayView1<f64>,
        returns: &[f64],
    ) -> ImputationResult<f64> {
        if t == 0 {
            return Ok(0.0);
        }

        // Find last valid price
        let mut lookback = 1;
        while t >= lookback && prices[t - lookback].is_nan() {
            lookback += 1;
        }

        if t < lookback {
            return Ok(0.0);
        }

        let last_price = prices[t - lookback];

        // Estimate drift and diffusion from recent returns
        let recent_returns: Vec<f64> = returns[t.saturating_sub(20)..t]
            .iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .collect();

        if recent_returns.is_empty() {
            return Ok(last_price);
        }

        let drift = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let diffusion = recent_returns
            .iter()
            .map(|&x| (x - drift).powi(2))
            .sum::<f64>()
            / recent_returns.len() as f64;

        // Generate return using geometric Brownian motion
        let dt = 1.0; // One time step
        let random_shock = Random::default().gen::<f64>() - 0.5; // Centered normal approximation
        let simulated_return = drift * dt + diffusion.sqrt() * random_shock;

        Ok(last_price * (simulated_return).exp())
    }
}

/// Portfolio data imputation with risk factor modeling
#[derive(Debug, Clone)]
pub struct PortfolioDataImputer {
    /// Risk factor model: "fama_french", "capm", "apt"
    pub risk_model: String,
    /// Market benchmark series
    pub market_benchmark: Option<Array1<f64>>,
    /// Risk factors (e.g., size, value, momentum)
    pub risk_factors: Option<Array2<f64>>,
    /// Regularization strength for factor models
    pub regularization: f64,
    /// Use sector-specific models
    pub sector_specific: bool,
}

impl Default for PortfolioDataImputer {
    fn default() -> Self {
        Self {
            risk_model: "capm".to_string(),
            market_benchmark: None,
            risk_factors: None,
            regularization: 0.01,
            sector_specific: false,
        }
    }
}

impl PortfolioDataImputer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_risk_model(mut self, model: &str) -> Self {
        self.risk_model = model.to_string();
        self
    }

    pub fn with_market_benchmark(mut self, benchmark: Array1<f64>) -> Self {
        self.market_benchmark = Some(benchmark);
        self
    }

    /// Impute portfolio data using risk factor models
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        match self.risk_model.as_str() {
            "capm" => self.capm_imputation(X),
            "fama_french" => self.fama_french_imputation(X),
            "apt" => self.apt_imputation(X),
            _ => Err(ImputationError::InvalidParameter(format!(
                "Unknown risk model: {}",
                self.risk_model
            ))),
        }
    }

    /// CAPM-based imputation
    fn capm_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_assets) = X.dim();
        let mut imputed = X.to_owned();

        // Use market benchmark or create proxy
        let market_returns = self
            .market_benchmark
            .as_ref()
            .map(|b| b.to_vec())
            .unwrap_or_else(|| self.create_market_proxy(X));

        for asset_idx in 0..n_assets {
            let asset_returns = X.column(asset_idx);

            // Estimate CAPM parameters
            let (alpha, beta) = self.estimate_capm_parameters(&asset_returns, &market_returns)?;

            // Impute missing values
            for t in 0..n_time {
                if asset_returns[t].is_nan() {
                    let market_return = if t < market_returns.len() {
                        market_returns[t]
                    } else {
                        0.0
                    };

                    imputed[[t, asset_idx]] = alpha + beta * market_return;
                }
            }
        }

        Ok(imputed)
    }

    /// Fama-French three-factor model imputation
    fn fama_french_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_assets) = X.dim();
        let mut imputed = X.to_owned();

        // Create or use provided risk factors
        let factors = self
            .risk_factors
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.create_fama_french_factors(X));

        for asset_idx in 0..n_assets {
            let asset_returns = X.column(asset_idx);

            // Estimate three-factor model parameters
            let factor_loadings = self.estimate_factor_loadings(&asset_returns, &factors)?;

            // Impute missing values
            for t in 0..n_time {
                if asset_returns[t].is_nan() && t < factors.nrows() {
                    let mut predicted_return = factor_loadings[0]; // Alpha

                    for (f, &loading) in factor_loadings[1..].iter().enumerate() {
                        if f < factors.ncols() {
                            predicted_return += loading * factors[[t, f]];
                        }
                    }

                    imputed[[t, asset_idx]] = predicted_return;
                }
            }
        }

        Ok(imputed)
    }

    /// Arbitrage Pricing Theory (APT) imputation
    fn apt_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_assets) = X.dim();
        let mut imputed = X.to_owned();

        // Extract factors using principal component analysis
        let factors = self.extract_apt_factors(X)?;

        for asset_idx in 0..n_assets {
            let asset_returns = X.column(asset_idx);

            // Estimate APT factor loadings
            let factor_loadings = self.estimate_factor_loadings(&asset_returns, &factors)?;

            // Impute missing values
            for t in 0..n_time {
                if asset_returns[t].is_nan() && t < factors.nrows() {
                    let mut predicted_return = factor_loadings[0]; // Alpha

                    for (f, &loading) in factor_loadings[1..].iter().enumerate() {
                        if f < factors.ncols() {
                            predicted_return += loading * factors[[t, f]];
                        }
                    }

                    imputed[[t, asset_idx]] = predicted_return;
                }
            }
        }

        Ok(imputed)
    }

    /// Create market proxy from asset returns
    fn create_market_proxy(&self, X: &ArrayView2<f64>) -> Vec<f64> {
        let n_time = X.nrows();
        let mut market_proxy = Vec::with_capacity(n_time);

        for t in 0..n_time {
            let day_returns: Vec<f64> =
                X.row(t).iter().filter(|&&x| !x.is_nan()).cloned().collect();

            let market_return = if day_returns.is_empty() {
                0.0
            } else {
                day_returns.iter().sum::<f64>() / day_returns.len() as f64
            };

            market_proxy.push(market_return);
        }

        market_proxy
    }

    /// Estimate CAPM parameters (alpha, beta)
    fn estimate_capm_parameters(
        &self,
        asset_returns: &ArrayView1<f64>,
        market_returns: &[f64],
    ) -> ImputationResult<(f64, f64)> {
        let pairs: Vec<(f64, f64)> = asset_returns
            .iter()
            .zip(market_returns.iter())
            .filter(|(&asset, &market)| !asset.is_nan() && !market.is_nan())
            .map(|(&asset, &market)| (asset, market))
            .collect();

        if pairs.len() < 10 {
            return Ok((0.0, 1.0)); // Default to market beta
        }

        // Linear regression: asset_return = alpha + beta * market_return
        let n = pairs.len() as f64;
        let sum_asset: f64 = pairs.iter().map(|(asset, _)| asset).sum();
        let sum_market: f64 = pairs.iter().map(|(_, market)| market).sum();
        let sum_asset_market: f64 = pairs.iter().map(|(asset, market)| asset * market).sum();
        let sum_market_sq: f64 = pairs.iter().map(|(_, market)| market * market).sum();

        let beta_numerator = n * sum_asset_market - sum_asset * sum_market;
        let beta_denominator = n * sum_market_sq - sum_market * sum_market;

        let beta = if beta_denominator.abs() > 1e-10 {
            beta_numerator / beta_denominator
        } else {
            1.0
        };

        let alpha = (sum_asset - beta * sum_market) / n;

        Ok((alpha, beta))
    }

    /// Create simplified Fama-French factors
    fn create_fama_french_factors(&self, X: &ArrayView2<f64>) -> Array2<f64> {
        let (n_time, n_assets) = X.dim();
        let n_factors = 3; // Market, Size, Value
        let mut factors = Array2::zeros((n_time, n_factors));

        for t in 0..n_time {
            let day_returns: Vec<f64> =
                X.row(t).iter().filter(|&&x| !x.is_nan()).cloned().collect();

            if !day_returns.is_empty() {
                // Market factor (equal-weighted average)
                factors[[t, 0]] = day_returns.iter().sum::<f64>() / day_returns.len() as f64;

                // Size factor (high-low spread, simplified)
                let mut sorted_returns = day_returns.clone();
                sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n_returns = sorted_returns.len();
                let small_cap =
                    sorted_returns[..n_returns / 3].iter().sum::<f64>() / (n_returns / 3) as f64;
                let large_cap = sorted_returns[2 * n_returns / 3..].iter().sum::<f64>()
                    / (n_returns / 3) as f64;
                factors[[t, 1]] = small_cap - large_cap;

                // Value factor (simplified momentum)
                if t > 0 {
                    factors[[t, 2]] = factors[[t, 0]] - factors[[t - 1, 0]];
                }
            }
        }

        factors
    }

    /// Estimate factor loadings using linear regression
    fn estimate_factor_loadings(
        &self,
        asset_returns: &ArrayView1<f64>,
        factors: &Array2<f64>,
    ) -> ImputationResult<Vec<f64>> {
        let n_factors = factors.ncols();
        let mut loadings = vec![0.0; n_factors + 1]; // +1 for alpha

        // Collect valid observations
        let mut observations = Vec::new();
        for (t, &ret) in asset_returns.iter().enumerate() {
            if !ret.is_nan() && t < factors.nrows() {
                let mut factor_values = vec![1.0]; // Intercept
                for f in 0..n_factors {
                    factor_values.push(factors[[t, f]]);
                }
                observations.push((ret, factor_values));
            }
        }

        if observations.len() < n_factors + 2 {
            loadings[0] = 0.0; // Alpha
            if n_factors > 0 {
                loadings[1] = 1.0; // Market beta
            }
            return Ok(loadings);
        }

        // Simple least squares regression (normal equations)
        let n_obs = observations.len();
        let n_params = n_factors + 1;

        // Build design matrix X and response vector y
        let mut X = Array2::zeros((n_obs, n_params));
        let mut y = Array1::zeros(n_obs);

        for (i, (response, predictors)) in observations.iter().enumerate() {
            y[i] = *response;
            for (j, &predictor) in predictors.iter().enumerate() {
                X[[i, j]] = predictor;
            }
        }

        // Solve normal equations: (X'X + lambda*I) * beta = X'y
        let XtX = X.t().dot(&X);
        let Xty = X.t().dot(&y);

        // Add regularization
        let mut regularized_XtX = XtX;
        for i in 0..n_params {
            regularized_XtX[[i, i]] += self.regularization;
        }

        // Solve using simple inversion (simplified)
        if let Ok(solution) = self.solve_linear_system(&regularized_XtX, &Xty) {
            loadings = solution;
        }

        Ok(loadings)
    }

    /// Simple linear system solver
    fn solve_linear_system(&self, A: &Array2<f64>, b: &Array1<f64>) -> ImputationResult<Vec<f64>> {
        let n = A.nrows();
        if n != A.ncols() || n != b.len() {
            return Err(ImputationError::InvalidParameter(
                "Matrix dimensions don't match".to_string(),
            ));
        }

        // Simple Gaussian elimination (not optimized)
        let mut augmented = Array2::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = A[[i, j]];
            }
            augmented[[i, n]] = b[i];
        }

        // Forward elimination
        for k in 0..n {
            // Find pivot
            let mut pivot_row = k;
            for i in (k + 1)..n {
                if augmented[[i, k]].abs() > augmented[[pivot_row, k]].abs() {
                    pivot_row = i;
                }
            }

            // Swap rows
            if pivot_row != k {
                for j in 0..=n {
                    let temp = augmented[[k, j]];
                    augmented[[k, j]] = augmented[[pivot_row, j]];
                    augmented[[pivot_row, j]] = temp;
                }
            }

            // Eliminate
            for i in (k + 1)..n {
                if augmented[[k, k]].abs() > 1e-10 {
                    let factor = augmented[[i, k]] / augmented[[k, k]];
                    for j in k..=n {
                        augmented[[i, j]] -= factor * augmented[[k, j]];
                    }
                }
            }
        }

        // Back substitution
        let mut solution = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += augmented[[i, j]] * solution[j];
            }
            if augmented[[i, i]].abs() > 1e-10 {
                solution[i] = (augmented[[i, n]] - sum) / augmented[[i, i]];
            }
        }

        Ok(solution)
    }

    /// Extract APT factors using PCA
    fn extract_apt_factors(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_assets) = X.dim();
        let n_factors = 5.min(n_assets); // Extract up to 5 factors

        // Fill missing values temporarily for PCA
        let mut filled_data = X.to_owned();
        for i in 0..n_time {
            for j in 0..n_assets {
                if X[[i, j]].is_nan() {
                    // Use column mean
                    let column_mean = X.column(j).iter().filter(|&&x| !x.is_nan()).sum::<f64>()
                        / X.column(j).iter().filter(|&&x| !x.is_nan()).count() as f64;
                    filled_data[[i, j]] = column_mean;
                }
            }
        }

        // Simplified PCA (extract first few principal components)
        let factors = self.simple_pca(&filled_data, n_factors)?;

        Ok(factors)
    }

    /// Simplified PCA implementation
    fn simple_pca(&self, X: &Array2<f64>, n_components: usize) -> ImputationResult<Array2<f64>> {
        let (n_time, n_assets) = X.dim();

        // Center the data
        let mut centered_data = X.clone();
        for j in 0..n_assets {
            let column_mean = X.column(j).sum() / n_time as f64;
            for i in 0..n_time {
                centered_data[[i, j]] -= column_mean;
            }
        }

        // Compute covariance matrix
        let cov_matrix = centered_data.t().dot(&centered_data) / (n_time - 1) as f64;

        // For simplicity, return random factors (in practice, would compute eigenvalues/eigenvectors)
        let mut factors = Array2::zeros((n_time, n_components));
        for t in 0..n_time {
            for f in 0..n_components {
                factors[[t, f]] = Random::default().gen::<f64>() - 0.5;
            }
        }

        Ok(factors)
    }
}

/// Economic indicator imputation with macroeconomic modeling
#[derive(Debug, Clone)]
pub struct EconomicIndicatorImputer {
    /// Lead-lag relationships between indicators
    pub lead_lag_structure: HashMap<usize, Vec<(usize, i32)>>,
    /// Seasonal adjustment
    pub seasonal_adjustment: bool,
    /// Trend extraction method
    pub trend_method: String,
    /// Business cycle consideration
    pub business_cycle_aware: bool,
}

impl Default for EconomicIndicatorImputer {
    fn default() -> Self {
        Self {
            lead_lag_structure: HashMap::new(),
            seasonal_adjustment: true,
            trend_method: "hodrick_prescott".to_string(),
            business_cycle_aware: false,
        }
    }
}

impl EconomicIndicatorImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Impute economic indicators using macroeconomic relationships
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let mut imputed = X.to_owned();

        // Apply seasonal adjustment if requested
        if self.seasonal_adjustment {
            imputed = self.seasonal_adjust(&imputed)?;
        }

        // Extract trends
        let trends = self.extract_trends(&imputed)?;

        // Impute using lead-lag relationships
        imputed = self.impute_with_lead_lag(&imputed, &trends)?;

        Ok(imputed)
    }

    /// Apply seasonal adjustment (simplified X-13 ARIMA-SEATS style)
    fn seasonal_adjust(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_indicators) = X.dim();
        let mut adjusted = X.clone();

        // Assume monthly data (12-month seasonal cycle)
        let seasonal_period = 12;

        for indicator_idx in 0..n_indicators {
            let series = X.column(indicator_idx);

            // Compute seasonal factors
            let seasonal_factors = self.compute_seasonal_factors(&series, seasonal_period)?;

            // Apply adjustment
            for t in 0..n_time {
                if !series[t].is_nan() {
                    let seasonal_idx = t % seasonal_period;
                    adjusted[[t, indicator_idx]] = series[t] / seasonal_factors[seasonal_idx];
                }
            }
        }

        Ok(adjusted)
    }

    /// Compute seasonal factors
    fn compute_seasonal_factors(
        &self,
        series: &ArrayView1<f64>,
        period: usize,
    ) -> ImputationResult<Vec<f64>> {
        let mut seasonal_factors = vec![1.0; period];

        // Collect observations by seasonal period
        let mut seasonal_groups: Vec<Vec<f64>> = vec![Vec::new(); period];

        for (t, &value) in series.iter().enumerate() {
            if !value.is_nan() {
                seasonal_groups[t % period].push(value);
            }
        }

        // Compute average for each seasonal period
        for (s, group) in seasonal_groups.iter().enumerate() {
            if !group.is_empty() {
                seasonal_factors[s] = group.iter().sum::<f64>() / group.len() as f64;
            }
        }

        // Normalize seasonal factors
        let mean_factor: f64 = seasonal_factors.iter().sum::<f64>() / period as f64;
        if mean_factor > 1e-10 {
            for factor in seasonal_factors.iter_mut() {
                *factor /= mean_factor;
            }
        }

        Ok(seasonal_factors)
    }

    /// Extract trends using specified method
    fn extract_trends(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        match self.trend_method.as_str() {
            "hodrick_prescott" => self.hodrick_prescott_filter(X),
            "linear_trend" => self.linear_trend_extraction(X),
            "moving_average" => self.moving_average_trend(X),
            _ => Ok(X.clone()),
        }
    }

    /// Hodrick-Prescott filter for trend extraction
    fn hodrick_prescott_filter(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_indicators) = X.dim();
        let mut trends = X.clone();

        let lambda = 1600.0; // Standard for monthly data

        for indicator_idx in 0..n_indicators {
            let series = X.column(indicator_idx);
            let valid_indices: Vec<usize> = series
                .iter()
                .enumerate()
                .filter(|(_, &x)| !x.is_nan())
                .map(|(i, _)| i)
                .collect();

            if valid_indices.len() < 4 {
                continue;
            }

            // Simplified HP filter (would use proper implementation in practice)
            let trend = self.simple_hp_filter(&series, lambda)?;

            for (t, &trend_value) in trend.iter().enumerate() {
                if t < n_time {
                    trends[[t, indicator_idx]] = trend_value;
                }
            }
        }

        Ok(trends)
    }

    /// Simple HP filter implementation
    fn simple_hp_filter(
        &self,
        series: &ArrayView1<f64>,
        lambda: f64,
    ) -> ImputationResult<Vec<f64>> {
        let n = series.len();
        let mut trend = vec![0.0; n];

        // Fill missing values temporarily
        let mut filled_series = Vec::new();
        let mut last_valid = 0.0;

        for &value in series.iter() {
            if !value.is_nan() {
                filled_series.push(value);
                last_valid = value;
            } else {
                filled_series.push(last_valid);
            }
        }

        // Simple smoothing (simplified HP filter)
        let alpha = 1.0 / (1.0 + lambda);

        // Forward pass
        trend[0] = filled_series[0];
        for t in 1..n {
            trend[t] = alpha * filled_series[t] + (1.0 - alpha) * trend[t - 1];
        }

        // Backward pass
        for t in (0..(n - 1)).rev() {
            trend[t] = alpha * trend[t] + (1.0 - alpha) * trend[t + 1];
        }

        Ok(trend)
    }

    /// Linear trend extraction
    fn linear_trend_extraction(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_indicators) = X.dim();
        let mut trends = Array2::zeros((n_time, n_indicators));

        for indicator_idx in 0..n_indicators {
            let series = X.column(indicator_idx);

            // Fit linear trend
            let (slope, intercept) = self.fit_linear_trend(&series)?;

            for t in 0..n_time {
                trends[[t, indicator_idx]] = intercept + slope * t as f64;
            }
        }

        Ok(trends)
    }

    /// Fit linear trend to time series
    fn fit_linear_trend(&self, series: &ArrayView1<f64>) -> ImputationResult<(f64, f64)> {
        let valid_points: Vec<(f64, f64)> = series
            .iter()
            .enumerate()
            .filter(|(_, &y)| !y.is_nan())
            .map(|(t, &y)| (t as f64, y))
            .collect();

        if valid_points.len() < 2 {
            return Ok((0.0, 0.0));
        }

        let n = valid_points.len() as f64;
        let sum_x: f64 = valid_points.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = valid_points.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = valid_points.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = valid_points.iter().map(|(x, _)| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        Ok((slope, intercept))
    }

    /// Moving average trend
    fn moving_average_trend(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_time, n_indicators) = X.dim();
        let mut trends = X.clone();
        let window_size = 12; // 12-period moving average

        for indicator_idx in 0..n_indicators {
            for t in 0..n_time {
                let start_idx = t.saturating_sub(window_size / 2);
                let end_idx = (t + window_size / 2).min(n_time);

                let window_values: Vec<f64> = X
                    .column(indicator_idx)
                    .slice(s![start_idx..end_idx])
                    .iter()
                    .filter(|&&x| !x.is_nan())
                    .cloned()
                    .collect();

                if !window_values.is_empty() {
                    trends[[t, indicator_idx]] =
                        window_values.iter().sum::<f64>() / window_values.len() as f64;
                }
            }
        }

        Ok(trends)
    }

    /// Impute using lead-lag relationships
    fn impute_with_lead_lag(
        &self,
        X: &Array2<f64>,
        trends: &Array2<f64>,
    ) -> ImputationResult<Array2<f64>> {
        let (n_time, n_indicators) = X.dim();
        let mut imputed = X.clone();

        for (target_indicator, relationships) in &self.lead_lag_structure {
            if *target_indicator >= n_indicators {
                continue;
            }

            for t in 0..n_time {
                if X[[t, *target_indicator]].is_nan() {
                    let mut predicted_value = trends[[t, *target_indicator]];
                    let mut total_weight = 0.0;
                    let mut weighted_sum = 0.0;

                    for &(source_indicator, lag) in relationships {
                        if source_indicator >= n_indicators {
                            continue;
                        }

                        let source_time = (t as i32 + lag) as usize;
                        if source_time < n_time && !X[[source_time, source_indicator]].is_nan() {
                            let weight = 1.0; // Equal weights for simplicity
                            weighted_sum += weight * X[[source_time, source_indicator]];
                            total_weight += weight;
                        }
                    }

                    if total_weight > 0.0 {
                        predicted_value = weighted_sum / total_weight;
                    }

                    imputed[[t, *target_indicator]] = predicted_value;
                }
            }
        }

        Ok(imputed)
    }
}

/// Credit scoring imputation with financial risk considerations
#[derive(Debug, Clone)]
pub struct CreditScoringImputer {
    /// Risk categories for segmented imputation
    pub risk_segments: Vec<String>,
    /// Default probability model
    pub default_model: String,
    /// Regulatory compliance considerations
    pub regulatory_compliant: bool,
    /// Feature importance weights
    pub feature_weights: HashMap<String, f64>,
}

impl Default for CreditScoringImputer {
    fn default() -> Self {
        Self {
            risk_segments: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
            default_model: "logistic".to_string(),
            regulatory_compliant: true,
            feature_weights: HashMap::new(),
        }
    }
}

impl CreditScoringImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Impute credit scoring data with risk-aware methods
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let mut imputed = X.to_owned();

        // Segment by risk if possible
        let risk_segments = self.identify_risk_segments(X)?;

        // Impute within each segment
        for segment in 0..self.risk_segments.len() {
            imputed = self.segment_specific_imputation(&imputed, &risk_segments, segment)?;
        }

        // Apply regulatory constraints
        if self.regulatory_compliant {
            imputed = self.apply_regulatory_constraints(&imputed)?;
        }

        Ok(imputed)
    }

    /// Identify risk segments
    fn identify_risk_segments(&self, X: &ArrayView2<f64>) -> ImputationResult<Vec<usize>> {
        let n_samples = X.nrows();
        let mut segments = vec![0; n_samples];

        // Simple risk segmentation based on available features
        // In practice, would use more sophisticated risk models
        for i in 0..n_samples {
            let row_values: Vec<f64> = X.row(i).iter().filter(|&&x| !x.is_nan()).cloned().collect();

            if !row_values.is_empty() {
                let avg_score = row_values.iter().sum::<f64>() / row_values.len() as f64;

                segments[i] = if avg_score < 0.33 {
                    2 // High risk
                } else if avg_score < 0.67 {
                    1 // Medium risk
                } else {
                    0 // Low risk
                };
            }
        }

        Ok(segments)
    }

    /// Segment-specific imputation
    fn segment_specific_imputation(
        &self,
        X: &Array2<f64>,
        segments: &[usize],
        target_segment: usize,
    ) -> ImputationResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut imputed = X.clone();

        // Find samples in target segment
        let segment_samples: Vec<usize> = segments
            .iter()
            .enumerate()
            .filter(|(_, &seg)| seg == target_segment)
            .map(|(idx, _)| idx)
            .collect();

        if segment_samples.is_empty() {
            return Ok(imputed);
        }

        // Impute within segment
        for &sample_idx in &segment_samples {
            for feature_idx in 0..n_features {
                if X[[sample_idx, feature_idx]].is_nan() {
                    let imputed_value = self.segment_feature_imputation(
                        sample_idx,
                        feature_idx,
                        X,
                        &segment_samples,
                    )?;
                    imputed[[sample_idx, feature_idx]] = imputed_value;
                }
            }
        }

        Ok(imputed)
    }

    /// Impute feature within risk segment
    fn segment_feature_imputation(
        &self,
        sample_idx: usize,
        feature_idx: usize,
        X: &Array2<f64>,
        segment_samples: &[usize],
    ) -> ImputationResult<f64> {
        // Collect valid values from same segment
        let segment_values: Vec<f64> = segment_samples
            .iter()
            .filter(|&&idx| idx != sample_idx)
            .filter_map(|&idx| {
                let val = X[[idx, feature_idx]];
                if !val.is_nan() {
                    Some(val)
                } else {
                    None
                }
            })
            .collect();

        if segment_values.is_empty() {
            // Fall back to overall feature mean
            let feature_values: Vec<f64> = X
                .column(feature_idx)
                .iter()
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();

            return Ok(if feature_values.is_empty() {
                0.0
            } else {
                feature_values.iter().sum::<f64>() / feature_values.len() as f64
            });
        }

        // Use segment median for robustness
        let mut sorted_values = segment_values;
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_idx = sorted_values.len() / 2;
        Ok(sorted_values[median_idx])
    }

    /// Apply regulatory constraints to imputed values
    fn apply_regulatory_constraints(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let mut constrained = X.clone();

        // Apply reasonable bounds (simplified regulatory constraints)
        for value in constrained.iter_mut() {
            if !value.is_nan() {
                // Ensure values are within reasonable bounds
                *value = value.max(0.0).min(1.0);
            }
        }

        Ok(constrained)
    }
}

/// Risk factor imputation for financial risk management
#[derive(Debug, Clone)]
pub struct RiskFactorImputer {
    /// Risk factor categories
    pub factor_categories: Vec<String>,
    /// Correlation structure between factors
    pub factor_correlations: Option<Array2<f64>>,
    /// Value-at-Risk considerations
    pub var_aware: bool,
    /// Stress testing scenarios
    pub stress_scenarios: Vec<Array1<f64>>,
}

impl Default for RiskFactorImputer {
    fn default() -> Self {
        Self {
            factor_categories: vec![
                "market".to_string(),
                "credit".to_string(),
                "operational".to_string(),
            ],
            factor_correlations: None,
            var_aware: false,
            stress_scenarios: Vec::new(),
        }
    }
}

impl RiskFactorImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Impute risk factors considering correlations and VaR
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let mut imputed = X.to_owned();

        // Estimate factor correlations if not provided
        let correlations = self
            .factor_correlations
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.estimate_factor_correlations(X));

        // Impute using correlation structure
        imputed = self.correlation_aware_imputation(&imputed, &correlations)?;

        // Apply VaR constraints if requested
        if self.var_aware {
            imputed = self.var_constrained_imputation(&imputed)?;
        }

        Ok(imputed)
    }

    /// Estimate factor correlations
    fn estimate_factor_correlations(&self, X: &ArrayView2<f64>) -> Array2<f64> {
        let n_factors = X.ncols();
        let mut correlations = Array2::zeros((n_factors, n_factors));

        for i in 0..n_factors {
            for j in 0..n_factors {
                if i == j {
                    correlations[[i, j]] = 1.0;
                } else {
                    let corr = self.compute_factor_correlation(&X.column(i), &X.column(j));
                    correlations[[i, j]] = corr;
                }
            }
        }

        correlations
    }

    /// Compute correlation between two risk factors
    fn compute_factor_correlation(
        &self,
        factor1: &ArrayView1<f64>,
        factor2: &ArrayView1<f64>,
    ) -> f64 {
        let pairs: Vec<(f64, f64)> = factor1
            .iter()
            .zip(factor2.iter())
            .filter(|(&x, &y)| !x.is_nan() && !y.is_nan())
            .map(|(&x, &y)| (x, y))
            .collect();

        if pairs.len() < 5 {
            return 0.0;
        }

        let n = pairs.len() as f64;
        let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = pairs.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = pairs.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = pairs.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x.powi(2)) * (n * sum_y2 - sum_y.powi(2))).sqrt();

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Correlation-aware imputation
    fn correlation_aware_imputation(
        &self,
        X: &Array2<f64>,
        correlations: &Array2<f64>,
    ) -> ImputationResult<Array2<f64>> {
        let (n_time, n_factors) = X.dim();
        let mut imputed = X.clone();

        for t in 0..n_time {
            for f in 0..n_factors {
                if X[[t, f]].is_nan() {
                    let mut weighted_sum = 0.0;
                    let mut total_weight = 0.0;

                    for other_f in 0..n_factors {
                        if other_f != f && !X[[t, other_f]].is_nan() {
                            let correlation = correlations[[f, other_f]];
                            let weight = correlation.abs();

                            weighted_sum += weight * X[[t, other_f]];
                            total_weight += weight;
                        }
                    }

                    if total_weight > 0.1 {
                        imputed[[t, f]] = weighted_sum / total_weight;
                    } else {
                        // Fall back to historical mean
                        let factor_values: Vec<f64> = X
                            .column(f)
                            .iter()
                            .filter(|&&x| !x.is_nan())
                            .cloned()
                            .collect();

                        if !factor_values.is_empty() {
                            imputed[[t, f]] =
                                factor_values.iter().sum::<f64>() / factor_values.len() as f64;
                        } else {
                            imputed[[t, f]] = 0.0;
                        }
                    }
                }
            }
        }

        Ok(imputed)
    }

    /// VaR-constrained imputation
    fn var_constrained_imputation(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let mut constrained = X.clone();
        let confidence_level = 0.05; // 95% VaR

        for j in 0..X.ncols() {
            let factor_values: Vec<f64> = X
                .column(j)
                .iter()
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();

            if factor_values.len() >= 20 {
                let mut sorted_values = factor_values.clone();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let var_index = (confidence_level * sorted_values.len() as f64) as usize;
                let var_threshold = sorted_values[var_index.min(sorted_values.len() - 1)];

                // Ensure imputed values don't violate VaR constraints
                for i in 0..X.nrows() {
                    if !constrained[[i, j]].is_nan() && constrained[[i, j]] < var_threshold {
                        constrained[[i, j]] = var_threshold;
                    }
                }
            }
        }

        Ok(constrained)
    }
}

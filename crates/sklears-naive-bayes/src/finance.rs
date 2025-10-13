//! Finance and Economics Naive Bayes Implementations
//!
//! This module provides specialized Naive Bayes implementations for finance and economics applications,
//! including financial time series classification, risk assessment, portfolio classification, credit scoring,
//! and fraud detection.

use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2};

// Type aliases for compatibility with DMatrix/DVector usage
type DMatrix<T> = Array2<T>;
type DVector<T> = Array1<T>;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FinanceError {
    #[error("Invalid time series data: {0}")]
    InvalidTimeSeries(String),
    #[error("Risk assessment error: {0}")]
    RiskAssessment(String),
    #[error("Portfolio classification error: {0}")]
    PortfolioClassification(String),
    #[error("Credit scoring error: {0}")]
    CreditScoring(String),
    #[error("Fraud detection error: {0}")]
    FraudDetection(String),
    #[error("Mathematical computation error: {0}")]
    MathError(String),
}

/// Financial Time Series Classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialTimeSeriesNB<T: Float> {
    /// Class priors
    priors: HashMap<usize, T>,
    /// Feature statistics for each class
    feature_stats: HashMap<usize, FeatureStatistics<T>>,
    /// Technical indicators weights
    technical_indicators: TechnicalIndicators<T>,
    /// Time series parameters
    lookback_window: usize,
    /// Volatility models
    volatility_models: VolatilityModels<T>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics<T: Float> {
    /// Price movement statistics
    price_stats: PriceStatistics<T>,
    /// Volume statistics
    volume_stats: VolumeStatistics<T>,
    /// Volatility statistics
    volatility_stats: VolatilityStatistics<T>,
    /// Technical indicator statistics
    technical_stats: TechnicalStatistics<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceStatistics<T: Float> {
    /// Mean price change
    mean_price_change: T,
    /// Variance of price change
    variance_price_change: T,
    /// Mean relative price change
    mean_relative_change: T,
    /// Variance of relative price change
    variance_relative_change: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeStatistics<T: Float> {
    /// Mean volume
    mean_volume: T,
    /// Variance of volume
    variance_volume: T,
    /// Mean volume-price correlation
    mean_volume_price_corr: T,
    /// Variance of volume-price correlation
    variance_volume_price_corr: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityStatistics<T: Float> {
    /// Mean volatility
    mean_volatility: T,
    /// Variance of volatility
    variance_volatility: T,
    /// Mean GARCH parameters
    mean_garch_alpha: T,
    /// Mean GARCH beta
    mean_garch_beta: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalStatistics<T: Float> {
    /// RSI statistics
    rsi_stats: (T, T), // (mean, variance)
    /// MACD statistics
    macd_stats: (T, T),
    /// Bollinger Bands statistics
    bb_stats: (T, T),
    /// Stochastic oscillator statistics
    stoch_stats: (T, T),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicators<T: Float> {
    /// RSI period
    rsi_period: usize,
    /// MACD parameters
    macd_fast: usize,
    macd_slow: usize,
    macd_signal: usize,
    /// Bollinger Bands parameters
    bb_period: usize,
    bb_std_dev: T,
    /// Stochastic oscillator parameters
    stoch_k_period: usize,
    stoch_d_period: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityModels<T: Float> {
    /// GARCH model parameters
    garch_params: GarchParams<T>,
    /// Exponential weighted moving average parameters
    ewma_lambda: T,
    /// Historical volatility window
    hist_vol_window: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarchParams<T: Float> {
    /// GARCH omega parameter
    omega: T,
    /// GARCH alpha parameter
    alpha: T,
    /// GARCH beta parameter
    beta: T,
}

impl<T: Float + Default + Display + Debug + for<'a> std::iter::Sum<&'a T> + std::iter::Sum>
    FinancialTimeSeriesNB<T>
{
    /// Create a new financial time series classifier
    pub fn new(lookback_window: usize) -> Self {
        Self {
            priors: HashMap::new(),
            feature_stats: HashMap::new(),
            technical_indicators: TechnicalIndicators::default(),
            lookback_window,
            volatility_models: VolatilityModels::default(),
            _phantom: PhantomData,
        }
    }

    /// Fit the model to financial time series data
    pub fn fit(
        &mut self,
        prices: &DMatrix<T>,
        volumes: &DMatrix<T>,
        labels: &DVector<usize>,
    ) -> Result<(), FinanceError> {
        if prices.nrows() != volumes.nrows() || prices.nrows() != labels.len() {
            return Err(FinanceError::InvalidTimeSeries(
                "Dimension mismatch".to_string(),
            ));
        }

        // Calculate class priors
        let mut class_counts = HashMap::new();
        for &label in labels.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let total_samples = T::from(labels.len()).unwrap();
        for (&class, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            self.priors.insert(class, prior);
        }

        // Calculate feature statistics for each class
        for (&class, _) in class_counts.iter() {
            let class_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .collect();

            let feature_stats = self.calculate_feature_stats(prices, volumes, &class_indices)?;
            self.feature_stats.insert(class, feature_stats);
        }

        Ok(())
    }

    /// Calculate feature statistics for a given class
    fn calculate_feature_stats(
        &self,
        prices: &DMatrix<T>,
        volumes: &DMatrix<T>,
        indices: &[usize],
    ) -> Result<FeatureStatistics<T>, FinanceError> {
        let mut price_changes = Vec::new();
        let mut relative_changes = Vec::new();
        let mut volume_values = Vec::new();
        let mut volatilities = Vec::new();

        for &idx in indices {
            if idx >= self.lookback_window {
                // Calculate price changes
                let current_price = prices[(idx, 0)];
                let prev_price = prices[(idx - 1, 0)];
                let price_change = current_price - prev_price;
                let relative_change = price_change / prev_price;

                price_changes.push(price_change);
                relative_changes.push(relative_change);

                // Volume data
                volume_values.push(volumes[(idx, 0)]);

                // Calculate volatility over lookback window
                let volatility = self.calculate_volatility(prices, idx)?;
                volatilities.push(volatility);
            }
        }

        // Calculate statistics
        let price_stats = PriceStatistics {
            mean_price_change: self.calculate_mean(&price_changes),
            variance_price_change: self.calculate_variance(&price_changes),
            mean_relative_change: self.calculate_mean(&relative_changes),
            variance_relative_change: self.calculate_variance(&relative_changes),
        };

        let volume_stats = VolumeStatistics {
            mean_volume: self.calculate_mean(&volume_values),
            variance_volume: self.calculate_variance(&volume_values),
            mean_volume_price_corr: self
                .calculate_volume_price_correlation(prices, volumes, indices)?,
            variance_volume_price_corr: T::from(0.01).unwrap(), // Placeholder
        };

        let volatility_stats = VolatilityStatistics {
            mean_volatility: self.calculate_mean(&volatilities),
            variance_volatility: self.calculate_variance(&volatilities),
            mean_garch_alpha: T::from(0.1).unwrap(), // Placeholder
            mean_garch_beta: T::from(0.8).unwrap(),  // Placeholder
        };

        let technical_stats = TechnicalStatistics {
            rsi_stats: (T::from(50.0).unwrap(), T::from(100.0).unwrap()),
            macd_stats: (T::from(0.0).unwrap(), T::from(1.0).unwrap()),
            bb_stats: (T::from(0.5).unwrap(), T::from(0.1).unwrap()),
            stoch_stats: (T::from(50.0).unwrap(), T::from(100.0).unwrap()),
        };

        Ok(FeatureStatistics {
            price_stats,
            volume_stats,
            volatility_stats,
            technical_stats,
        })
    }

    /// Calculate volatility over a lookback window
    fn calculate_volatility(
        &self,
        prices: &DMatrix<T>,
        current_idx: usize,
    ) -> Result<T, FinanceError> {
        if current_idx < self.lookback_window {
            return Err(FinanceError::InvalidTimeSeries(
                "Insufficient data for volatility calculation".to_string(),
            ));
        }

        let mut returns = Vec::new();
        for i in (current_idx - self.lookback_window + 1)..=current_idx {
            if i > 0 {
                let return_val = (prices[(i, 0)] / prices[(i - 1, 0)]).ln();
                returns.push(return_val);
            }
        }

        let volatility = self.calculate_std_dev(&returns);
        Ok(volatility * T::from(252.0).unwrap().sqrt()) // Annualized volatility
    }

    /// Calculate volume-price correlation
    fn calculate_volume_price_correlation(
        &self,
        prices: &DMatrix<T>,
        volumes: &DMatrix<T>,
        indices: &[usize],
    ) -> Result<T, FinanceError> {
        let mut price_changes = Vec::new();
        let mut volume_changes = Vec::new();

        for &idx in indices {
            if idx > 0 {
                let price_change = prices[(idx, 0)] - prices[(idx - 1, 0)];
                let volume_change = volumes[(idx, 0)] - volumes[(idx - 1, 0)];
                price_changes.push(price_change);
                volume_changes.push(volume_change);
            }
        }

        let correlation = self.calculate_correlation(&price_changes, &volume_changes);
        Ok(correlation)
    }

    /// Predict class probabilities for new financial time series data
    pub fn predict_proba(
        &self,
        prices: &DMatrix<T>,
        volumes: &DMatrix<T>,
    ) -> Result<HashMap<usize, T>, FinanceError> {
        let mut class_probs = HashMap::new();

        // Extract features from the time series
        let features = self.extract_features(prices, volumes)?;

        for (&class, &prior) in self.priors.iter() {
            let class_stats = self
                .feature_stats
                .get(&class)
                .ok_or_else(|| FinanceError::InvalidTimeSeries("Class not found".to_string()))?;

            let likelihood = self.calculate_likelihood(&features, class_stats)?;
            let posterior = prior * likelihood;
            class_probs.insert(class, posterior);
        }

        // Normalize probabilities
        let total_prob: T = class_probs.values().cloned().sum();
        for prob in class_probs.values_mut() {
            *prob = *prob / total_prob;
        }

        Ok(class_probs)
    }

    /// Extract features from financial time series
    fn extract_features(
        &self,
        prices: &DMatrix<T>,
        volumes: &DMatrix<T>,
    ) -> Result<FinancialFeatures<T>, FinanceError> {
        let current_idx = prices.nrows() - 1;

        // Price features
        let price_change = if current_idx > 0 {
            prices[(current_idx, 0)] - prices[(current_idx - 1, 0)]
        } else {
            T::zero()
        };

        let relative_change = if current_idx > 0 && prices[(current_idx - 1, 0)] != T::zero() {
            price_change / prices[(current_idx - 1, 0)]
        } else {
            T::zero()
        };

        // Volume features
        let volume = volumes[(current_idx, 0)];

        // Volatility features
        let volatility = self.calculate_volatility(prices, current_idx)?;

        // Technical indicators (simplified)
        let rsi = self.calculate_rsi(prices, current_idx)?;
        let macd = self.calculate_macd(prices, current_idx)?;

        Ok(FinancialFeatures {
            price_change,
            relative_change,
            volume,
            volatility,
            rsi,
            macd,
        })
    }

    /// Calculate RSI
    fn calculate_rsi(&self, prices: &DMatrix<T>, current_idx: usize) -> Result<T, FinanceError> {
        let period = self.technical_indicators.rsi_period;
        if current_idx < period {
            return Ok(T::from(50.0).unwrap()); // Neutral RSI
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for i in (current_idx - period + 1)..=current_idx {
            if i > 0 {
                let change = prices[(i, 0)] - prices[(i - 1, 0)];
                if change > T::zero() {
                    gains.push(change);
                    losses.push(T::zero());
                } else {
                    gains.push(T::zero());
                    losses.push(-change);
                }
            }
        }

        let avg_gain = self.calculate_mean(&gains);
        let avg_loss = self.calculate_mean(&losses);

        if avg_loss == T::zero() {
            return Ok(T::from(100.0).unwrap());
        }

        let rs = avg_gain / avg_loss;
        let rsi = T::from(100.0).unwrap() - (T::from(100.0).unwrap() / (T::one() + rs));
        Ok(rsi)
    }

    /// Calculate MACD
    fn calculate_macd(&self, prices: &DMatrix<T>, current_idx: usize) -> Result<T, FinanceError> {
        let fast_period = self.technical_indicators.macd_fast;
        let slow_period = self.technical_indicators.macd_slow;

        if current_idx < slow_period {
            return Ok(T::zero());
        }

        let fast_ema = self.calculate_ema(prices, current_idx, fast_period)?;
        let slow_ema = self.calculate_ema(prices, current_idx, slow_period)?;

        Ok(fast_ema - slow_ema)
    }

    /// Calculate Exponential Moving Average
    fn calculate_ema(
        &self,
        prices: &DMatrix<T>,
        current_idx: usize,
        period: usize,
    ) -> Result<T, FinanceError> {
        if current_idx < period {
            return Err(FinanceError::InvalidTimeSeries(
                "Insufficient data for EMA".to_string(),
            ));
        }

        let alpha = T::from(2.0).unwrap() / T::from(period + 1).unwrap();
        let mut ema = prices[(current_idx - period + 1, 0)];

        for i in (current_idx - period + 2)..=current_idx {
            ema = alpha * prices[(i, 0)] + (T::one() - alpha) * ema;
        }

        Ok(ema)
    }

    /// Calculate likelihood of features given class
    fn calculate_likelihood(
        &self,
        features: &FinancialFeatures<T>,
        class_stats: &FeatureStatistics<T>,
    ) -> Result<T, FinanceError> {
        let mut likelihood = T::one();

        // Price change likelihood
        likelihood = likelihood
            * self.gaussian_pdf(
                features.price_change,
                class_stats.price_stats.mean_price_change,
                class_stats.price_stats.variance_price_change,
            );

        // Relative change likelihood
        likelihood = likelihood
            * self.gaussian_pdf(
                features.relative_change,
                class_stats.price_stats.mean_relative_change,
                class_stats.price_stats.variance_relative_change,
            );

        // Volume likelihood
        likelihood = likelihood
            * self.gaussian_pdf(
                features.volume,
                class_stats.volume_stats.mean_volume,
                class_stats.volume_stats.variance_volume,
            );

        // Volatility likelihood
        likelihood = likelihood
            * self.gaussian_pdf(
                features.volatility,
                class_stats.volatility_stats.mean_volatility,
                class_stats.volatility_stats.variance_volatility,
            );

        Ok(likelihood)
    }

    /// Gaussian probability density function
    fn gaussian_pdf(&self, x: T, mean: T, variance: T) -> T {
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let coefficient = T::one() / (two_pi * variance).sqrt();
        let exponent = -((x - mean).powi(2)) / (T::from(2.0).unwrap() * variance);
        coefficient * exponent.exp()
    }

    /// Calculate mean of a vector
    fn calculate_mean(&self, values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let sum: T = values.iter().sum();
        sum / T::from(values.len()).unwrap()
    }

    /// Calculate variance of a vector
    fn calculate_variance(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::from(0.01).unwrap(); // Small default variance
        }
        let mean = self.calculate_mean(values);
        let sum_squared_diff: T = values.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_squared_diff / T::from(values.len() - 1).unwrap()
    }

    /// Calculate standard deviation
    fn calculate_std_dev(&self, values: &[T]) -> T {
        self.calculate_variance(values).sqrt()
    }

    /// Calculate correlation between two vectors
    fn calculate_correlation(&self, x: &[T], y: &[T]) -> T {
        if x.len() != y.len() || x.len() < 2 {
            return T::zero();
        }

        let mean_x = self.calculate_mean(x);
        let mean_y = self.calculate_mean(y);

        let mut numerator = T::zero();
        let mut sum_sq_x = T::zero();
        let mut sum_sq_y = T::zero();

        for i in 0..x.len() {
            let diff_x = x[i] - mean_x;
            let diff_y = y[i] - mean_y;
            numerator = numerator + diff_x * diff_y;
            sum_sq_x = sum_sq_x + diff_x * diff_x;
            sum_sq_y = sum_sq_y + diff_y * diff_y;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == T::zero() {
            T::zero()
        } else {
            numerator / denominator
        }
    }
}

#[derive(Debug, Clone)]
pub struct FinancialFeatures<T: Float> {
    pub price_change: T,
    pub relative_change: T,
    pub volume: T,
    pub volatility: T,
    pub rsi: T,
    pub macd: T,
}

impl<T: Float> Default for TechnicalIndicators<T> {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            bb_period: 20,
            bb_std_dev: T::from(2.0).unwrap(),
            stoch_k_period: 14,
            stoch_d_period: 3,
        }
    }
}

impl<T: Float> Default for VolatilityModels<T> {
    fn default() -> Self {
        Self {
            garch_params: GarchParams::default(),
            ewma_lambda: T::from(0.94).unwrap(),
            hist_vol_window: 30,
        }
    }
}

impl<T: Float> Default for GarchParams<T> {
    fn default() -> Self {
        Self {
            omega: T::from(0.01).unwrap(),
            alpha: T::from(0.1).unwrap(),
            beta: T::from(0.8).unwrap(),
        }
    }
}

/// Risk Assessment Naive Bayes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentNB<T: Float> {
    /// Class priors (risk levels)
    priors: HashMap<RiskLevel, T>,
    /// Feature statistics for each risk level
    feature_stats: HashMap<RiskLevel, RiskFeatureStats<T>>,
    /// Risk assessment parameters
    params: RiskAssessmentParams<T>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low

    Low,
    /// Medium

    Medium,
    /// High

    High,
    /// VeryHigh

    VeryHigh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFeatureStats<T: Float> {
    /// Volatility statistics
    volatility_stats: (T, T), // (mean, variance)
    /// Value at Risk statistics
    var_stats: (T, T),
    /// Sharpe ratio statistics
    sharpe_stats: (T, T),
    /// Maximum drawdown statistics
    max_drawdown_stats: (T, T),
    /// Beta statistics
    beta_stats: (T, T),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentParams<T: Float> {
    /// Confidence level for VaR calculation
    var_confidence: T,
    /// Risk-free rate for Sharpe ratio
    risk_free_rate: T,
    /// Benchmark correlation threshold
    benchmark_threshold: T,
}

impl<T: Float + Default + Display + Debug + for<'a> std::iter::Sum<&'a T> + std::iter::Sum>
    RiskAssessmentNB<T>
{
    /// Create a new risk assessment classifier
    pub fn new() -> Self {
        Self {
            priors: HashMap::new(),
            feature_stats: HashMap::new(),
            params: RiskAssessmentParams::default(),
            _phantom: PhantomData,
        }
    }

    /// Fit the risk assessment model
    pub fn fit(
        &mut self,
        returns: &DMatrix<T>,
        risk_labels: &[RiskLevel],
    ) -> Result<(), FinanceError> {
        if returns.nrows() != risk_labels.len() {
            return Err(FinanceError::RiskAssessment(
                "Dimension mismatch".to_string(),
            ));
        }

        // Calculate class priors
        let mut class_counts = HashMap::new();
        for &risk_level in risk_labels {
            *class_counts.entry(risk_level).or_insert(0) += 1;
        }

        let total_samples = T::from(risk_labels.len()).unwrap();
        for (&risk_level, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            self.priors.insert(risk_level, prior);
        }

        // Calculate feature statistics for each risk level
        for (&risk_level, _) in class_counts.iter() {
            let class_indices: Vec<usize> = risk_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == risk_level)
                .map(|(i, _)| i)
                .collect();

            let feature_stats = self.calculate_risk_features(returns, &class_indices)?;
            self.feature_stats.insert(risk_level, feature_stats);
        }

        Ok(())
    }

    /// Calculate risk features for a given class
    fn calculate_risk_features(
        &self,
        returns: &DMatrix<T>,
        indices: &[usize],
    ) -> Result<RiskFeatureStats<T>, FinanceError> {
        let mut volatilities = Vec::new();
        let mut vars = Vec::new();
        let mut sharpe_ratios = Vec::new();
        let mut max_drawdowns = Vec::new();
        let mut betas = Vec::new();

        for &idx in indices {
            let return_series = returns.row(idx);
            let return_vec: Vec<T> = return_series.iter().cloned().collect();

            // Calculate volatility
            let volatility = self.calculate_volatility_from_returns(&return_vec);
            volatilities.push(volatility);

            // Calculate VaR
            let var = self.calculate_var(&return_vec)?;
            vars.push(var);

            // Calculate Sharpe ratio
            let sharpe = self.calculate_sharpe_ratio(&return_vec);
            sharpe_ratios.push(sharpe);

            // Calculate maximum drawdown
            let max_dd = self.calculate_max_drawdown(&return_vec);
            max_drawdowns.push(max_dd);

            // Calculate beta (simplified)
            let beta = self.calculate_beta(&return_vec);
            betas.push(beta);
        }

        Ok(RiskFeatureStats {
            volatility_stats: (
                self.calculate_mean(&volatilities),
                self.calculate_variance(&volatilities),
            ),
            var_stats: (self.calculate_mean(&vars), self.calculate_variance(&vars)),
            sharpe_stats: (
                self.calculate_mean(&sharpe_ratios),
                self.calculate_variance(&sharpe_ratios),
            ),
            max_drawdown_stats: (
                self.calculate_mean(&max_drawdowns),
                self.calculate_variance(&max_drawdowns),
            ),
            beta_stats: (self.calculate_mean(&betas), self.calculate_variance(&betas)),
        })
    }

    /// Calculate volatility from returns
    fn calculate_volatility_from_returns(&self, returns: &[T]) -> T {
        let variance = self.calculate_variance(returns);
        variance.sqrt() * T::from(252.0).unwrap().sqrt() // Annualized
    }

    /// Calculate Value at Risk
    fn calculate_var(&self, returns: &[T]) -> Result<T, FinanceError> {
        if returns.is_empty() {
            return Err(FinanceError::RiskAssessment("No returns data".to_string()));
        }

        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile_index = ((T::one() - self.params.var_confidence)
            * T::from(sorted_returns.len()).unwrap())
        .floor()
        .to_usize()
        .unwrap();
        let percentile_index = percentile_index.min(sorted_returns.len() - 1);

        Ok(-sorted_returns[percentile_index]) // VaR is typically positive
    }

    /// Calculate Sharpe ratio
    fn calculate_sharpe_ratio(&self, returns: &[T]) -> T {
        let mean_return = self.calculate_mean(returns);
        let volatility = self.calculate_volatility_from_returns(returns);

        if volatility == T::zero() {
            T::zero()
        } else {
            (mean_return - self.params.risk_free_rate) / volatility
        }
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, returns: &[T]) -> T {
        if returns.is_empty() {
            return T::zero();
        }

        let mut cumulative_returns = Vec::new();
        let mut cumulative = T::one();

        for &ret in returns {
            cumulative = cumulative * (T::one() + ret);
            cumulative_returns.push(cumulative);
        }

        let mut max_drawdown = T::zero();
        let mut peak = cumulative_returns[0];

        for &value in cumulative_returns.iter().skip(1) {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    /// Calculate beta (simplified market beta)
    fn calculate_beta(&self, returns: &[T]) -> T {
        // Simplified beta calculation (placeholder)
        T::one()
    }

    /// Predict risk level
    pub fn predict_risk(&self, returns: &[T]) -> Result<RiskLevel, FinanceError> {
        let probabilities = self.predict_risk_proba(returns)?;

        probabilities
            .into_iter()
            .max_by(|(_, a), (_, b)| {
                // Handle NaN values by treating them as smaller than any finite number
                match (a.is_nan(), b.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                }
            })
            .map(|(risk_level, _)| risk_level)
            .ok_or_else(|| FinanceError::RiskAssessment("No predictions available".to_string()))
    }

    /// Predict risk level probabilities
    pub fn predict_risk_proba(&self, returns: &[T]) -> Result<HashMap<RiskLevel, T>, FinanceError> {
        let mut class_probs = HashMap::new();

        // Calculate risk features
        let volatility = self.calculate_volatility_from_returns(returns);
        let var = self.calculate_var(returns)?;
        let sharpe = self.calculate_sharpe_ratio(returns);
        let max_dd = self.calculate_max_drawdown(returns);
        let beta = self.calculate_beta(returns);

        for (&risk_level, &prior) in self.priors.iter() {
            let feature_stats = self
                .feature_stats
                .get(&risk_level)
                .ok_or_else(|| FinanceError::RiskAssessment("Risk level not found".to_string()))?;

            let mut likelihood = T::one();

            // Calculate likelihood for each feature
            likelihood = likelihood
                * self.gaussian_pdf(
                    volatility,
                    feature_stats.volatility_stats.0,
                    feature_stats.volatility_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(var, feature_stats.var_stats.0, feature_stats.var_stats.1);
            likelihood = likelihood
                * self.gaussian_pdf(
                    sharpe,
                    feature_stats.sharpe_stats.0,
                    feature_stats.sharpe_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    max_dd,
                    feature_stats.max_drawdown_stats.0,
                    feature_stats.max_drawdown_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(beta, feature_stats.beta_stats.0, feature_stats.beta_stats.1);

            let posterior = prior * likelihood;
            class_probs.insert(risk_level, posterior);
        }

        // Normalize probabilities
        let total_prob: T = class_probs.values().cloned().sum();
        if total_prob > T::zero() {
            for prob in class_probs.values_mut() {
                *prob = *prob / total_prob;
            }
        }

        Ok(class_probs)
    }

    /// Gaussian probability density function
    fn gaussian_pdf(&self, x: T, mean: T, variance: T) -> T {
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let coefficient = T::one() / (two_pi * variance).sqrt();
        let exponent = -((x - mean).powi(2)) / (T::from(2.0).unwrap() * variance);
        coefficient * exponent.exp()
    }

    /// Calculate mean of a vector
    fn calculate_mean(&self, values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let sum: T = values.iter().sum();
        sum / T::from(values.len()).unwrap()
    }

    /// Calculate variance of a vector
    fn calculate_variance(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::from(0.01).unwrap(); // Small default variance
        }
        let mean = self.calculate_mean(values);
        let sum_squared_diff: T = values.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_squared_diff / T::from(values.len() - 1).unwrap()
    }
}

impl<T: Float> Default for RiskAssessmentParams<T> {
    fn default() -> Self {
        Self {
            var_confidence: T::from(0.95).unwrap(),
            risk_free_rate: T::from(0.02).unwrap(),
            benchmark_threshold: T::from(0.7).unwrap(),
        }
    }
}

/// Portfolio Classification Naive Bayes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioClassificationNB<T: Float> {
    /// Class priors (portfolio categories)
    priors: HashMap<PortfolioCategory, T>,
    /// Feature statistics for each portfolio category
    feature_stats: HashMap<PortfolioCategory, PortfolioFeatureStats<T>>,
    /// Portfolio classification parameters
    params: PortfolioClassificationParams<T>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortfolioCategory {
    /// Conservative

    Conservative,
    /// Moderate

    Moderate,
    /// Aggressive

    Aggressive,
    /// Growth

    Growth,
    /// Income

    Income,
    /// Balanced

    Balanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioFeatureStats<T: Float> {
    /// Asset allocation statistics
    stock_allocation_stats: (T, T), // (mean, variance)
    bond_allocation_stats: (T, T),
    cash_allocation_stats: (T, T),
    /// Risk-return statistics
    expected_return_stats: (T, T),
    volatility_stats: (T, T),
    /// Diversification statistics
    sector_diversification_stats: (T, T),
    geographic_diversification_stats: (T, T),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioClassificationParams<T: Float> {
    /// Minimum diversification threshold
    min_diversification: T,
    /// Maximum concentration threshold
    max_concentration: T,
    /// Risk tolerance parameters
    risk_tolerance: T,
}

impl<T: Float + Default + Display + Debug + for<'a> std::iter::Sum<&'a T> + std::iter::Sum>
    PortfolioClassificationNB<T>
{
    /// Create a new portfolio classifier
    pub fn new() -> Self {
        Self {
            priors: HashMap::new(),
            feature_stats: HashMap::new(),
            params: PortfolioClassificationParams::default(),
            _phantom: PhantomData,
        }
    }

    /// Fit the portfolio classification model
    pub fn fit(
        &mut self,
        portfolios: &[PortfolioData<T>],
        categories: &[PortfolioCategory],
    ) -> Result<(), FinanceError> {
        if portfolios.len() != categories.len() {
            return Err(FinanceError::PortfolioClassification(
                "Dimension mismatch".to_string(),
            ));
        }

        // Calculate class priors
        let mut class_counts = HashMap::new();
        for &category in categories {
            *class_counts.entry(category).or_insert(0) += 1;
        }

        let total_samples = T::from(categories.len()).unwrap();
        for (&category, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            self.priors.insert(category, prior);
        }

        // Calculate feature statistics for each category
        for (&category, _) in class_counts.iter() {
            let class_portfolios: Vec<&PortfolioData<T>> = categories
                .iter()
                .enumerate()
                .filter(|(_, &cat)| cat == category)
                .map(|(i, _)| &portfolios[i])
                .collect();

            let feature_stats = self.calculate_portfolio_features(&class_portfolios)?;
            self.feature_stats.insert(category, feature_stats);
        }

        Ok(())
    }

    /// Calculate portfolio features for a given category
    fn calculate_portfolio_features(
        &self,
        portfolios: &[&PortfolioData<T>],
    ) -> Result<PortfolioFeatureStats<T>, FinanceError> {
        let mut stock_allocations = Vec::new();
        let mut bond_allocations = Vec::new();
        let mut cash_allocations = Vec::new();
        let mut expected_returns = Vec::new();
        let mut volatilities = Vec::new();
        let mut sector_diversifications = Vec::new();
        let mut geographic_diversifications = Vec::new();

        for portfolio in portfolios {
            stock_allocations.push(portfolio.stock_allocation);
            bond_allocations.push(portfolio.bond_allocation);
            cash_allocations.push(portfolio.cash_allocation);
            expected_returns.push(portfolio.expected_return);
            volatilities.push(portfolio.volatility);
            sector_diversifications.push(portfolio.sector_diversification);
            geographic_diversifications.push(portfolio.geographic_diversification);
        }

        Ok(PortfolioFeatureStats {
            stock_allocation_stats: (
                self.calculate_mean(&stock_allocations),
                self.calculate_variance(&stock_allocations),
            ),
            bond_allocation_stats: (
                self.calculate_mean(&bond_allocations),
                self.calculate_variance(&bond_allocations),
            ),
            cash_allocation_stats: (
                self.calculate_mean(&cash_allocations),
                self.calculate_variance(&cash_allocations),
            ),
            expected_return_stats: (
                self.calculate_mean(&expected_returns),
                self.calculate_variance(&expected_returns),
            ),
            volatility_stats: (
                self.calculate_mean(&volatilities),
                self.calculate_variance(&volatilities),
            ),
            sector_diversification_stats: (
                self.calculate_mean(&sector_diversifications),
                self.calculate_variance(&sector_diversifications),
            ),
            geographic_diversification_stats: (
                self.calculate_mean(&geographic_diversifications),
                self.calculate_variance(&geographic_diversifications),
            ),
        })
    }

    /// Predict portfolio category
    pub fn predict_category(
        &self,
        portfolio: &PortfolioData<T>,
    ) -> Result<PortfolioCategory, FinanceError> {
        let probabilities = self.predict_category_proba(portfolio)?;

        probabilities
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(category, _)| category)
            .ok_or_else(|| {
                FinanceError::PortfolioClassification("No predictions available".to_string())
            })
    }

    /// Predict portfolio category probabilities
    pub fn predict_category_proba(
        &self,
        portfolio: &PortfolioData<T>,
    ) -> Result<HashMap<PortfolioCategory, T>, FinanceError> {
        let mut class_probs = HashMap::new();

        for (&category, &prior) in self.priors.iter() {
            let feature_stats = self.feature_stats.get(&category).ok_or_else(|| {
                FinanceError::PortfolioClassification("Category not found".to_string())
            })?;

            let mut likelihood = T::one();

            // Calculate likelihood for each feature
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.stock_allocation,
                    feature_stats.stock_allocation_stats.0,
                    feature_stats.stock_allocation_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.bond_allocation,
                    feature_stats.bond_allocation_stats.0,
                    feature_stats.bond_allocation_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.cash_allocation,
                    feature_stats.cash_allocation_stats.0,
                    feature_stats.cash_allocation_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.expected_return,
                    feature_stats.expected_return_stats.0,
                    feature_stats.expected_return_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.volatility,
                    feature_stats.volatility_stats.0,
                    feature_stats.volatility_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.sector_diversification,
                    feature_stats.sector_diversification_stats.0,
                    feature_stats.sector_diversification_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.geographic_diversification,
                    feature_stats.geographic_diversification_stats.0,
                    feature_stats.geographic_diversification_stats.1,
                );

            let posterior = prior * likelihood;
            class_probs.insert(category, posterior);
        }

        // Normalize probabilities
        let total_prob: T = class_probs.values().cloned().sum();
        if total_prob > T::zero() {
            for prob in class_probs.values_mut() {
                *prob = *prob / total_prob;
            }
        }

        Ok(class_probs)
    }

    /// Gaussian probability density function
    fn gaussian_pdf(&self, x: T, mean: T, variance: T) -> T {
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let coefficient = T::one() / (two_pi * variance).sqrt();
        let exponent = -((x - mean).powi(2)) / (T::from(2.0).unwrap() * variance);
        coefficient * exponent.exp()
    }

    /// Calculate mean of a vector
    fn calculate_mean(&self, values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let sum: T = values.iter().sum();
        sum / T::from(values.len()).unwrap()
    }

    /// Calculate variance of a vector
    fn calculate_variance(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::from(0.01).unwrap(); // Small default variance
        }
        let mean = self.calculate_mean(values);
        let sum_squared_diff: T = values.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_squared_diff / T::from(values.len() - 1).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioData<T: Float> {
    pub stock_allocation: T,
    pub bond_allocation: T,
    pub cash_allocation: T,
    pub expected_return: T,
    pub volatility: T,
    pub sector_diversification: T,
    pub geographic_diversification: T,
}

impl<T: Float> Default for PortfolioClassificationParams<T> {
    fn default() -> Self {
        Self {
            min_diversification: T::from(0.1).unwrap(),
            max_concentration: T::from(0.3).unwrap(),
            risk_tolerance: T::from(0.15).unwrap(),
        }
    }
}

/// Credit Scoring Naive Bayes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditScoringNB<T: Float> {
    /// Class priors (credit risk categories)
    priors: HashMap<CreditRisk, T>,
    /// Feature statistics for each credit risk category
    feature_stats: HashMap<CreditRisk, CreditFeatureStats<T>>,
    /// Credit scoring parameters
    params: CreditScoringParams<T>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CreditRisk {
    /// Low

    Low,
    /// Medium

    Medium,
    /// High

    High,
    /// Default

    Default,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditFeatureStats<T: Float> {
    /// Financial statistics
    credit_score_stats: (T, T), // (mean, variance)
    income_stats: (T, T),
    debt_to_income_stats: (T, T),
    payment_history_stats: (T, T),
    credit_utilization_stats: (T, T),
    /// Behavioral statistics
    account_age_stats: (T, T),
    num_accounts_stats: (T, T),
    recent_inquiries_stats: (T, T),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditScoringParams<T: Float> {
    /// Minimum credit score threshold
    min_credit_score: T,
    /// Maximum debt-to-income ratio
    max_debt_to_income: T,
    /// Payment history weight
    payment_history_weight: T,
}

impl<T: Float + Default + Display + Debug + for<'a> std::iter::Sum<&'a T> + std::iter::Sum>
    CreditScoringNB<T>
{
    /// Create a new credit scoring classifier
    pub fn new() -> Self {
        Self {
            priors: HashMap::new(),
            feature_stats: HashMap::new(),
            params: CreditScoringParams::default(),
            _phantom: PhantomData,
        }
    }

    /// Fit the credit scoring model
    pub fn fit(
        &mut self,
        credit_data: &[CreditData<T>],
        risk_labels: &[CreditRisk],
    ) -> Result<(), FinanceError> {
        if credit_data.len() != risk_labels.len() {
            return Err(FinanceError::CreditScoring(
                "Dimension mismatch".to_string(),
            ));
        }

        // Calculate class priors
        let mut class_counts = HashMap::new();
        for &risk_level in risk_labels {
            *class_counts.entry(risk_level).or_insert(0) += 1;
        }

        let total_samples = T::from(risk_labels.len()).unwrap();
        for (&risk_level, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            self.priors.insert(risk_level, prior);
        }

        // Calculate feature statistics for each risk level
        for (&risk_level, _) in class_counts.iter() {
            let class_data: Vec<&CreditData<T>> = risk_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == risk_level)
                .map(|(i, _)| &credit_data[i])
                .collect();

            let feature_stats = self.calculate_credit_features(&class_data)?;
            self.feature_stats.insert(risk_level, feature_stats);
        }

        Ok(())
    }

    /// Calculate credit features for a given risk level
    fn calculate_credit_features(
        &self,
        credit_data: &[&CreditData<T>],
    ) -> Result<CreditFeatureStats<T>, FinanceError> {
        let mut credit_scores = Vec::new();
        let mut incomes = Vec::new();
        let mut debt_to_incomes = Vec::new();
        let mut payment_histories = Vec::new();
        let mut credit_utilizations = Vec::new();
        let mut account_ages = Vec::new();
        let mut num_accounts = Vec::new();
        let mut recent_inquiries = Vec::new();

        for data in credit_data {
            credit_scores.push(data.credit_score);
            incomes.push(data.income);
            debt_to_incomes.push(data.debt_to_income);
            payment_histories.push(data.payment_history);
            credit_utilizations.push(data.credit_utilization);
            account_ages.push(data.account_age);
            num_accounts.push(data.num_accounts);
            recent_inquiries.push(data.recent_inquiries);
        }

        Ok(CreditFeatureStats {
            credit_score_stats: (
                self.calculate_mean(&credit_scores),
                self.calculate_variance(&credit_scores),
            ),
            income_stats: (
                self.calculate_mean(&incomes),
                self.calculate_variance(&incomes),
            ),
            debt_to_income_stats: (
                self.calculate_mean(&debt_to_incomes),
                self.calculate_variance(&debt_to_incomes),
            ),
            payment_history_stats: (
                self.calculate_mean(&payment_histories),
                self.calculate_variance(&payment_histories),
            ),
            credit_utilization_stats: (
                self.calculate_mean(&credit_utilizations),
                self.calculate_variance(&credit_utilizations),
            ),
            account_age_stats: (
                self.calculate_mean(&account_ages),
                self.calculate_variance(&account_ages),
            ),
            num_accounts_stats: (
                self.calculate_mean(&num_accounts),
                self.calculate_variance(&num_accounts),
            ),
            recent_inquiries_stats: (
                self.calculate_mean(&recent_inquiries),
                self.calculate_variance(&recent_inquiries),
            ),
        })
    }

    /// Predict credit risk
    pub fn predict_risk(&self, credit_data: &CreditData<T>) -> Result<CreditRisk, FinanceError> {
        let probabilities = self.predict_risk_proba(credit_data)?;

        probabilities
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(risk_level, _)| risk_level)
            .ok_or_else(|| FinanceError::CreditScoring("No predictions available".to_string()))
    }

    /// Predict credit risk probabilities
    pub fn predict_risk_proba(
        &self,
        credit_data: &CreditData<T>,
    ) -> Result<HashMap<CreditRisk, T>, FinanceError> {
        let mut class_probs = HashMap::new();

        for (&risk_level, &prior) in self.priors.iter() {
            let feature_stats = self
                .feature_stats
                .get(&risk_level)
                .ok_or_else(|| FinanceError::CreditScoring("Risk level not found".to_string()))?;

            let mut likelihood = T::one();

            // Calculate likelihood for each feature
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.credit_score,
                    feature_stats.credit_score_stats.0,
                    feature_stats.credit_score_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.income,
                    feature_stats.income_stats.0,
                    feature_stats.income_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.debt_to_income,
                    feature_stats.debt_to_income_stats.0,
                    feature_stats.debt_to_income_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.payment_history,
                    feature_stats.payment_history_stats.0,
                    feature_stats.payment_history_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.credit_utilization,
                    feature_stats.credit_utilization_stats.0,
                    feature_stats.credit_utilization_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.account_age,
                    feature_stats.account_age_stats.0,
                    feature_stats.account_age_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.num_accounts,
                    feature_stats.num_accounts_stats.0,
                    feature_stats.num_accounts_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.recent_inquiries,
                    feature_stats.recent_inquiries_stats.0,
                    feature_stats.recent_inquiries_stats.1,
                );

            let posterior = prior * likelihood;
            class_probs.insert(risk_level, posterior);
        }

        // Normalize probabilities
        let total_prob: T = class_probs.values().cloned().sum();
        if total_prob > T::zero() {
            for prob in class_probs.values_mut() {
                *prob = *prob / total_prob;
            }
        }

        Ok(class_probs)
    }

    /// Gaussian probability density function
    fn gaussian_pdf(&self, x: T, mean: T, variance: T) -> T {
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let coefficient = T::one() / (two_pi * variance).sqrt();
        let exponent = -((x - mean).powi(2)) / (T::from(2.0).unwrap() * variance);
        coefficient * exponent.exp()
    }

    /// Calculate mean of a vector
    fn calculate_mean(&self, values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let sum: T = values.iter().sum();
        sum / T::from(values.len()).unwrap()
    }

    /// Calculate variance of a vector
    fn calculate_variance(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::from(0.01).unwrap(); // Small default variance
        }
        let mean = self.calculate_mean(values);
        let sum_squared_diff: T = values.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_squared_diff / T::from(values.len() - 1).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditData<T: Float> {
    pub credit_score: T,
    pub income: T,
    pub debt_to_income: T,
    pub payment_history: T,
    pub credit_utilization: T,
    pub account_age: T,
    pub num_accounts: T,
    pub recent_inquiries: T,
}

impl<T: Float> Default for CreditScoringParams<T> {
    fn default() -> Self {
        Self {
            min_credit_score: T::from(600.0).unwrap(),
            max_debt_to_income: T::from(0.4).unwrap(),
            payment_history_weight: T::from(0.35).unwrap(),
        }
    }
}

/// Fraud Detection Naive Bayes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudDetectionNB<T: Float> {
    /// Class priors (fraud vs legitimate)
    priors: HashMap<FraudLabel, T>,
    /// Feature statistics for each class
    feature_stats: HashMap<FraudLabel, FraudFeatureStats<T>>,
    /// Fraud detection parameters
    params: FraudDetectionParams<T>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FraudLabel {
    /// Legitimate

    Legitimate,
    /// Fraud

    Fraud,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudFeatureStats<T: Float> {
    /// Transaction statistics
    amount_stats: (T, T), // (mean, variance)

    frequency_stats: (T, T),

    time_stats: (T, T),
    /// Behavioral statistics
    merchant_category_stats: (T, T),
    location_stats: (T, T),
    device_stats: (T, T),
    /// Risk statistics
    velocity_stats: (T, T),
    pattern_deviation_stats: (T, T),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudDetectionParams<T: Float> {
    /// Fraud detection threshold
    fraud_threshold: T,
    /// Velocity check window (hours)
    velocity_window: T,
    /// Maximum transaction amount
    max_transaction_amount: T,
}

impl<T: Float + Default + Display + Debug + for<'a> std::iter::Sum<&'a T> + std::iter::Sum>
    FraudDetectionNB<T>
{
    /// Create a new fraud detection classifier
    pub fn new() -> Self {
        Self {
            priors: HashMap::new(),
            feature_stats: HashMap::new(),
            params: FraudDetectionParams::default(),
            _phantom: PhantomData,
        }
    }

    /// Fit the fraud detection model
    pub fn fit(
        &mut self,
        transaction_data: &[TransactionData<T>],
        fraud_labels: &[FraudLabel],
    ) -> Result<(), FinanceError> {
        if transaction_data.len() != fraud_labels.len() {
            return Err(FinanceError::FraudDetection(
                "Dimension mismatch".to_string(),
            ));
        }

        // Calculate class priors
        let mut class_counts = HashMap::new();
        for &fraud_label in fraud_labels {
            *class_counts.entry(fraud_label).or_insert(0) += 1;
        }

        let total_samples = T::from(fraud_labels.len()).unwrap();
        for (&fraud_label, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            self.priors.insert(fraud_label, prior);
        }

        // Calculate feature statistics for each class
        for (&fraud_label, _) in class_counts.iter() {
            let class_data: Vec<&TransactionData<T>> = fraud_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == fraud_label)
                .map(|(i, _)| &transaction_data[i])
                .collect();

            let feature_stats = self.calculate_fraud_features(&class_data)?;
            self.feature_stats.insert(fraud_label, feature_stats);
        }

        Ok(())
    }

    /// Calculate fraud features for a given class
    fn calculate_fraud_features(
        &self,
        transaction_data: &[&TransactionData<T>],
    ) -> Result<FraudFeatureStats<T>, FinanceError> {
        let mut amounts = Vec::new();
        let mut frequencies = Vec::new();
        let mut times = Vec::new();
        let mut merchant_categories = Vec::new();
        let mut locations = Vec::new();
        let mut devices = Vec::new();
        let mut velocities = Vec::new();
        let mut pattern_deviations = Vec::new();

        for data in transaction_data {
            amounts.push(data.amount);
            frequencies.push(data.frequency);
            times.push(data.time);
            merchant_categories.push(data.merchant_category);
            locations.push(data.location);
            devices.push(data.device);
            velocities.push(data.velocity);
            pattern_deviations.push(data.pattern_deviation);
        }

        Ok(FraudFeatureStats {
            amount_stats: (
                self.calculate_mean(&amounts),
                self.calculate_variance(&amounts),
            ),
            frequency_stats: (
                self.calculate_mean(&frequencies),
                self.calculate_variance(&frequencies),
            ),
            time_stats: (self.calculate_mean(&times), self.calculate_variance(&times)),
            merchant_category_stats: (
                self.calculate_mean(&merchant_categories),
                self.calculate_variance(&merchant_categories),
            ),
            location_stats: (
                self.calculate_mean(&locations),
                self.calculate_variance(&locations),
            ),
            device_stats: (
                self.calculate_mean(&devices),
                self.calculate_variance(&devices),
            ),
            velocity_stats: (
                self.calculate_mean(&velocities),
                self.calculate_variance(&velocities),
            ),
            pattern_deviation_stats: (
                self.calculate_mean(&pattern_deviations),
                self.calculate_variance(&pattern_deviations),
            ),
        })
    }

    /// Predict fraud probability
    pub fn predict_fraud(
        &self,
        transaction_data: &TransactionData<T>,
    ) -> Result<FraudLabel, FinanceError> {
        let probabilities = self.predict_fraud_proba(transaction_data)?;

        probabilities
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(fraud_label, _)| fraud_label)
            .ok_or_else(|| FinanceError::FraudDetection("No predictions available".to_string()))
    }

    /// Predict fraud probabilities
    pub fn predict_fraud_proba(
        &self,
        transaction_data: &TransactionData<T>,
    ) -> Result<HashMap<FraudLabel, T>, FinanceError> {
        let mut class_probs = HashMap::new();

        for (&fraud_label, &prior) in self.priors.iter() {
            let feature_stats = self
                .feature_stats
                .get(&fraud_label)
                .ok_or_else(|| FinanceError::FraudDetection("Fraud label not found".to_string()))?;

            let mut likelihood = T::one();

            // Calculate likelihood for each feature
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.amount,
                    feature_stats.amount_stats.0,
                    feature_stats.amount_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.frequency,
                    feature_stats.frequency_stats.0,
                    feature_stats.frequency_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.time,
                    feature_stats.time_stats.0,
                    feature_stats.time_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.merchant_category,
                    feature_stats.merchant_category_stats.0,
                    feature_stats.merchant_category_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.location,
                    feature_stats.location_stats.0,
                    feature_stats.location_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.device,
                    feature_stats.device_stats.0,
                    feature_stats.device_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.velocity,
                    feature_stats.velocity_stats.0,
                    feature_stats.velocity_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.pattern_deviation,
                    feature_stats.pattern_deviation_stats.0,
                    feature_stats.pattern_deviation_stats.1,
                );

            let posterior = prior * likelihood;
            class_probs.insert(fraud_label, posterior);
        }

        // Normalize probabilities
        let total_prob: T = class_probs.values().cloned().sum();
        if total_prob > T::zero() {
            for prob in class_probs.values_mut() {
                *prob = *prob / total_prob;
            }
        }

        Ok(class_probs)
    }

    /// Gaussian probability density function
    fn gaussian_pdf(&self, x: T, mean: T, variance: T) -> T {
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let coefficient = T::one() / (two_pi * variance).sqrt();
        let exponent = -((x - mean).powi(2)) / (T::from(2.0).unwrap() * variance);
        coefficient * exponent.exp()
    }

    /// Calculate mean of a vector
    fn calculate_mean(&self, values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let sum: T = values.iter().sum();
        sum / T::from(values.len()).unwrap()
    }

    /// Calculate variance of a vector
    fn calculate_variance(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::from(0.01).unwrap(); // Small default variance
        }
        let mean = self.calculate_mean(values);
        let sum_squared_diff: T = values.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_squared_diff / T::from(values.len() - 1).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionData<T: Float> {
    pub amount: T,
    pub frequency: T,
    pub time: T,
    pub merchant_category: T,
    pub location: T,
    pub device: T,
    pub velocity: T,
    pub pattern_deviation: T,
}

impl<T: Float> Default for FraudDetectionParams<T> {
    fn default() -> Self {
        Self {
            fraud_threshold: T::from(0.5).unwrap(),
            velocity_window: T::from(24.0).unwrap(),
            max_transaction_amount: T::from(10000.0).unwrap(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::{Array1, Array2};

    // Type aliases for compatibility with DMatrix/DVector usage
    type DMatrix<T> = Array2<T>;
    type DVector<T> = Array1<T>;

    #[test]
    fn test_financial_time_series_nb() {
        let mut classifier = FinancialTimeSeriesNB::<f64>::new(10);

        // Create sample data
        let prices =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| 100.0 + i as f64).collect()).unwrap();
        let volumes =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| 1000.0 + i as f64 * 10.0).collect())
                .unwrap();
        let labels = Array1::from_vec(vec![
            0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
        ]);

        // Fit the model
        assert!(classifier.fit(&prices, &volumes, &labels).is_ok());

        // Test prediction
        let test_prices =
            Array2::from_shape_vec((15, 1), (0..15).map(|i| 120.0 + i as f64).collect()).unwrap();
        let test_volumes =
            Array2::from_shape_vec((15, 1), (0..15).map(|i| 1200.0 + i as f64 * 10.0).collect())
                .unwrap();

        let probabilities = classifier.predict_proba(&test_prices, &test_volumes);
        assert!(probabilities.is_ok());
    }

    #[test]
    fn test_risk_assessment_nb() {
        let mut classifier = RiskAssessmentNB::<f64>::new();

        // Create sample return data
        let returns =
            Array2::from_shape_vec((10, 5), (0..50).map(|i| (i as f64 - 25.0) * 0.01).collect())
                .unwrap();
        let risk_labels = vec![
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::VeryHigh,
        ];

        // Fit the model
        assert!(classifier.fit(&returns, &risk_labels).is_ok());

        // Test prediction
        let test_returns = vec![0.02, -0.01, 0.03, -0.02, 0.01];
        let risk_prediction = classifier.predict_risk(&test_returns);
        assert!(risk_prediction.is_ok());
    }

    #[test]
    fn test_portfolio_classification_nb() {
        let mut classifier = PortfolioClassificationNB::<f64>::new();

        // Create sample portfolio data
        let portfolios = vec![
            PortfolioData {
                stock_allocation: 0.8,
                bond_allocation: 0.2,
                cash_allocation: 0.0,
                expected_return: 0.08,
                volatility: 0.15,
                sector_diversification: 0.7,
                geographic_diversification: 0.6,
            },
            PortfolioData {
                stock_allocation: 0.4,
                bond_allocation: 0.5,
                cash_allocation: 0.1,
                expected_return: 0.05,
                volatility: 0.08,
                sector_diversification: 0.8,
                geographic_diversification: 0.7,
            },
        ];

        let categories = vec![
            PortfolioCategory::Aggressive,
            PortfolioCategory::Conservative,
        ];

        // Fit the model
        assert!(classifier.fit(&portfolios, &categories).is_ok());

        // Test prediction
        let test_portfolio = PortfolioData {
            stock_allocation: 0.6,
            bond_allocation: 0.3,
            cash_allocation: 0.1,
            expected_return: 0.06,
            volatility: 0.12,
            sector_diversification: 0.75,
            geographic_diversification: 0.65,
        };

        let category_prediction = classifier.predict_category(&test_portfolio);
        assert!(category_prediction.is_ok());
    }

    #[test]
    fn test_credit_scoring_nb() {
        let mut classifier = CreditScoringNB::<f64>::new();

        // Create sample credit data
        let credit_data = vec![
            CreditData {
                credit_score: 750.0,
                income: 80000.0,
                debt_to_income: 0.2,
                payment_history: 0.95,
                credit_utilization: 0.15,
                account_age: 120.0,
                num_accounts: 5.0,
                recent_inquiries: 1.0,
            },
            CreditData {
                credit_score: 550.0,
                income: 40000.0,
                debt_to_income: 0.6,
                payment_history: 0.7,
                credit_utilization: 0.8,
                account_age: 24.0,
                num_accounts: 12.0,
                recent_inquiries: 5.0,
            },
        ];

        let risk_labels = vec![CreditRisk::Low, CreditRisk::High];

        // Fit the model
        assert!(classifier.fit(&credit_data, &risk_labels).is_ok());

        // Test prediction
        let test_credit = CreditData {
            credit_score: 650.0,
            income: 60000.0,
            debt_to_income: 0.3,
            payment_history: 0.85,
            credit_utilization: 0.25,
            account_age: 72.0,
            num_accounts: 8.0,
            recent_inquiries: 2.0,
        };

        let risk_prediction = classifier.predict_risk(&test_credit);
        assert!(risk_prediction.is_ok());
    }

    #[test]
    fn test_fraud_detection_nb() {
        let mut classifier = FraudDetectionNB::<f64>::new();

        // Create sample transaction data
        let transaction_data = vec![
            TransactionData {
                amount: 50.0,
                frequency: 5.0,
                time: 14.0,
                merchant_category: 1.0,
                location: 1.0,
                device: 1.0,
                velocity: 2.0,
                pattern_deviation: 0.1,
            },
            TransactionData {
                amount: 5000.0,
                frequency: 1.0,
                time: 3.0,
                merchant_category: 5.0,
                location: 10.0,
                device: 3.0,
                velocity: 10.0,
                pattern_deviation: 0.9,
            },
        ];

        let fraud_labels = vec![FraudLabel::Legitimate, FraudLabel::Fraud];

        // Fit the model
        assert!(classifier.fit(&transaction_data, &fraud_labels).is_ok());

        // Test prediction
        let test_transaction = TransactionData {
            amount: 100.0,
            frequency: 3.0,
            time: 12.0,
            merchant_category: 2.0,
            location: 2.0,
            device: 1.0,
            velocity: 3.0,
            pattern_deviation: 0.2,
        };

        let fraud_prediction = classifier.predict_fraud(&test_transaction);
        assert!(fraud_prediction.is_ok());
    }
}

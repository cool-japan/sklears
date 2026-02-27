//! Financial Time Series Classification

use super::{DMatrix, DVector, FinanceError};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

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

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

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
}

//! Technical indicator computations for financial feature selection
//!
//! This module implements various technical indicators commonly used in
//! quantitative finance for feature engineering and predictive modeling.

use scirs2_core::ndarray::{s, Array1, ArrayView1};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;
type Float = f64;

use super::utilities::compute_pearson_correlation;

/// Compute RSI (Relative Strength Index) predictive power
///
/// RSI is a momentum oscillator that measures the speed and magnitude of price changes.
/// This function computes how well RSI correlates with the target variable.
pub(crate) fn compute_rsi_predictive_power(
    feature: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if feature.len() < period + 1 {
        return Ok(0.0);
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..feature.len() {
        let change = feature[i] - feature[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    if gains.len() < period {
        return Ok(0.0);
    }

    let mut rsi_values = Vec::new();
    for i in period..gains.len() {
        let avg_gain = gains[i - period + 1..=i].iter().sum::<Float>() / period as Float;
        let avg_loss = losses[i - period + 1..=i].iter().sum::<Float>() / period as Float;

        let rs = if avg_loss > 0.0 {
            avg_gain / avg_loss
        } else {
            100.0
        };
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        rsi_values.push(rsi);
    }

    if rsi_values.len() < target.len() - period {
        return Ok(0.0);
    }

    let rsi_array = Array1::from_vec(rsi_values);
    let target_slice = target.slice(s![period..period + rsi_array.len()]);
    let correlation = compute_pearson_correlation(&rsi_array.view(), &target_slice);
    Ok(correlation.abs())
}

/// Compute MACD (Moving Average Convergence Divergence) predictive power
///
/// MACD is a trend-following momentum indicator showing the relationship between
/// two exponential moving averages.
pub(crate) fn compute_macd_predictive_power(
    feature: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if feature.len() < period * 2 {
        return Ok(0.0);
    }

    let fast_period = period;
    let slow_period = period * 2;

    let mut fast_ema = feature[0];
    let mut slow_ema = feature[0];
    let mut macd_values = Vec::new();

    let fast_multiplier = 2.0 / (fast_period + 1) as Float;
    let slow_multiplier = 2.0 / (slow_period + 1) as Float;

    for i in 1..feature.len() {
        fast_ema = (feature[i] - fast_ema) * fast_multiplier + fast_ema;
        slow_ema = (feature[i] - slow_ema) * slow_multiplier + slow_ema;

        if i >= slow_period {
            macd_values.push(fast_ema - slow_ema);
        }
    }

    if macd_values.len() < target.len() - slow_period {
        return Ok(0.0);
    }

    let macd_array = Array1::from_vec(macd_values);
    let target_slice = target.slice(s![slow_period..slow_period + macd_array.len()]);
    let correlation = compute_pearson_correlation(&macd_array.view(), &target_slice);
    Ok(correlation.abs())
}

/// Compute Bollinger Bands predictive power
///
/// Bollinger Bands are volatility bands placed above and below a moving average.
/// This function measures how well band signals correlate with the target.
pub(crate) fn compute_bollinger_bands_predictive_power(
    feature: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if feature.len() < period {
        return Ok(0.0);
    }

    let mut bb_signals = Vec::new();

    for i in period..feature.len() {
        let window = feature.slice(s![i - period + 1..=i]);
        let mean = window.sum() / period as Float;
        let std = ((window.mapv(|x| (x - mean).powi(2)).sum()) / period as Float).sqrt();

        let upper_band = mean + 2.0 * std;
        let lower_band = mean - 2.0 * std;

        let signal = if feature[i] <= lower_band {
            1.0 // Buy signal
        } else if feature[i] >= upper_band {
            -1.0 // Sell signal
        } else {
            0.0 // No signal
        };
        bb_signals.push(signal);
    }

    if bb_signals.len() < target.len() - period {
        return Ok(0.0);
    }

    let bb_array = Array1::from_vec(bb_signals);
    let target_slice = target.slice(s![period..period + bb_array.len()]);
    let correlation = compute_pearson_correlation(&bb_array.view(), &target_slice);
    Ok(correlation.abs())
}

/// Compute ATR (Average True Range) predictive power
///
/// ATR measures market volatility by decomposing the entire range of an asset
/// price for a given period.
pub(crate) fn compute_atr_predictive_power(
    feature: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if feature.len() < period + 1 {
        return Ok(0.0);
    }

    let mut true_ranges = Vec::new();
    for i in 1..feature.len() {
        let tr = (feature[i] - feature[i - 1]).abs();
        true_ranges.push(tr);
    }

    if true_ranges.len() < period {
        return Ok(0.0);
    }

    let mut atr_values = Vec::new();
    for i in period..true_ranges.len() {
        let atr = true_ranges[i - period + 1..=i].iter().sum::<Float>() / period as Float;
        atr_values.push(atr);
    }

    if atr_values.len() < target.len() - period {
        return Ok(0.0);
    }

    let atr_array = Array1::from_vec(atr_values);
    let target_slice = target.slice(s![period..period + atr_array.len()]);
    let correlation = compute_pearson_correlation(&atr_array.view(), &target_slice);
    Ok(correlation.abs())
}

/// Compute OBV (On-Balance Volume) predictive power
///
/// OBV is a momentum indicator that uses volume flow to predict changes in price.
pub(crate) fn compute_obv_predictive_power(
    feature: &ArrayView1<Float>,
    target: &Array1<Float>,
    _period: usize,
) -> Result<Float> {
    if feature.len() < 2 {
        return Ok(0.0);
    }

    let mut obv_values = Vec::new();
    let mut obv = 0.0;

    for i in 1..feature.len() {
        let price_change = feature[i] - feature[i - 1];
        obv += if price_change > 0.0 {
            feature[i]
        } else if price_change < 0.0 {
            -feature[i]
        } else {
            0.0
        };
        obv_values.push(obv);
    }

    if obv_values.is_empty() || target.len() < 2 {
        return Ok(0.0);
    }

    let obv_array = Array1::from_vec(obv_values);
    let target_slice = target.slice(s![1..1 + obv_array.len().min(target.len() - 1)]);
    let correlation = compute_pearson_correlation(&obv_array.view(), &target_slice);
    Ok(correlation.abs())
}

/// Compute VWAP (Volume Weighted Average Price) predictive power
///
/// VWAP gives the average price a security has traded at throughout the day,
/// based on both volume and price.
pub(crate) fn compute_vwap_predictive_power(
    feature: &ArrayView1<Float>,
    target: &Array1<Float>,
    period: usize,
) -> Result<Float> {
    if feature.len() < period {
        return Ok(0.0);
    }

    let mut vwap_values = Vec::new();

    for i in period..feature.len() {
        let window = feature.slice(s![i - period + 1..=i]);
        let vwap = window.sum() / period as Float;
        vwap_values.push(vwap);
    }

    if vwap_values.is_empty() {
        return Ok(0.0);
    }

    let vwap_array = Array1::from_vec(vwap_values);
    let target_slice = target.slice(s![period..period + vwap_array.len()]);
    let correlation = compute_pearson_correlation(&vwap_array.view(), &target_slice);
    Ok(correlation.abs())
}

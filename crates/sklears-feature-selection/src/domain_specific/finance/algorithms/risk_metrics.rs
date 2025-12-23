//! Risk metrics and measures for financial feature selection
//!
//! This module implements various risk metrics including VaR, CVaR, drawdown,
//! tail risk, and extreme value measures.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::error::{Result as SklResult, SklearsError};

type Result<T> = SklResult<T>;
type Float = f64;

/// Compute Value at Risk (VaR) based feature scores
pub(crate) fn compute_var_based_scores(
    x: &Array2<Float>,
    _y: &Array1<Float>,
    confidence: Float,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = x.column(i);
        scores[i] = compute_var(&feature, confidence);
    }

    Ok(scores)
}

/// Compute Value at Risk for a single feature
pub(crate) fn compute_var(feature: &ArrayView1<Float>, confidence_level: Float) -> Float {
    let mut sorted_values: Vec<Float> = feature.iter().cloned().collect();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = ((1.0 - confidence_level) * sorted_values.len() as Float) as usize;
    sorted_values.get(index).cloned().unwrap_or(0.0)
}

/// Compute Conditional Value at Risk (CVaR/Expected Shortfall)
pub(crate) fn compute_cvar(feature: &ArrayView1<Float>, confidence_level: Float) -> Float {
    let var = compute_var(feature, confidence_level);
    let tail_losses: Vec<Float> = feature.iter().filter(|&&x| x <= var).cloned().collect();

    if tail_losses.is_empty() {
        return 0.0;
    }

    tail_losses.iter().sum::<Float>() / tail_losses.len() as Float
}

/// Compute portfolio Conditional Value at Risk
pub(crate) fn compute_portfolio_cvar(x: &Array2<Float>, confidence_level: Float) -> Result<Float> {
    let portfolio_returns = x.mean_axis(scirs2_core::ndarray::Axis(1)).unwrap();
    Ok(compute_cvar(&portfolio_returns.view(), confidence_level))
}

/// Compute drawdown-based feature scores
pub(crate) fn compute_drawdown_based_scores(
    x: &Array2<Float>,
    _y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = x.column(i);
        let prices = feature.to_owned();
        scores[i] = compute_max_drawdown_single(&prices)?;
    }

    Ok(scores)
}

/// Compute maximum drawdown for a single price series
pub(crate) fn compute_max_drawdown_single(prices: &Array1<Float>) -> Result<Float> {
    if prices.is_empty() {
        return Ok(0.0);
    }

    let mut max_price = prices[0];
    let mut max_drawdown = 0.0;

    for &price in prices.iter() {
        if price > max_price {
            max_price = price;
        }

        let drawdown = (max_price - price) / max_price;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    Ok(max_drawdown)
}

/// Compute maximum drawdown across all features
pub(crate) fn compute_max_drawdown(x: &Array2<Float>) -> Result<Float> {
    let n_features = x.ncols();
    let mut max_dd = 0.0;

    for i in 0..n_features {
        let feature = x.column(i).to_owned();
        let dd = compute_max_drawdown_single(&feature)?;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    Ok(max_dd)
}

/// Compute feature maximum drawdown
pub(crate) fn compute_feature_max_drawdown(feature: &ArrayView1<Float>) -> Float {
    if feature.is_empty() {
        return 0.0;
    }

    let mut max_value = feature[0];
    let mut max_drawdown = 0.0;

    for &value in feature.iter() {
        if value > max_value {
            max_value = value;
        }
        let drawdown = (max_value - value) / max_value.max(1e-10);
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    max_drawdown
}

/// Compute tail risk measure
pub(crate) fn compute_tail_risk(feature: &ArrayView1<Float>) -> Float {
    let mean = feature.mean().unwrap_or(0.0);
    let std = feature.std(0.0);

    if std < 1e-10 {
        return 0.0;
    }

    let skewness = feature
        .iter()
        .map(|&x| ((x - mean) / std).powi(3))
        .sum::<Float>()
        / feature.len() as Float;

    let kurtosis = feature
        .iter()
        .map(|&x| ((x - mean) / std).powi(4))
        .sum::<Float>()
        / feature.len() as Float;

    (skewness.abs() + (kurtosis - 3.0).abs()) / 2.0
}

/// Compute extreme value index (Hill estimator)
pub(crate) fn compute_extreme_value_index(feature: &ArrayView1<Float>) -> Float {
    let mut sorted: Vec<Float> = feature.iter().cloned().collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let k = (sorted.len() as Float * 0.1) as usize;
    if k < 2 {
        return 0.0;
    }

    let threshold = sorted[k];
    let mut sum = 0.0;

    for i in 0..k {
        if sorted[i] > 0.0 && threshold > 0.0 {
            sum += (sorted[i] / threshold).ln();
        }
    }

    sum / k as Float
}

/// Compute downside deviation (semi-deviation)
pub(crate) fn compute_downside_deviation(
    feature: &ArrayView1<Float>,
    target_return: Float,
) -> Float {
    let downside_returns: Vec<Float> = feature
        .iter()
        .filter(|&&x| x < target_return)
        .map(|&x| (x - target_return).powi(2))
        .collect();

    if downside_returns.is_empty() {
        return 0.0;
    }

    (downside_returns.iter().sum::<Float>() / downside_returns.len() as Float).sqrt()
}

/// Compute Omega ratio
pub(crate) fn compute_omega_ratio(feature: &ArrayView1<Float>, threshold: Float) -> Float {
    let gains: Float = feature.iter().filter(|&&x| x > threshold).sum();
    let losses: Float = feature
        .iter()
        .filter(|&&x| x < threshold)
        .map(|&x| threshold - x)
        .sum();

    if losses < 1e-10 {
        return 10.0;
    }

    gains / losses
}

/// Compute portfolio Value at Risk
pub(crate) fn compute_portfolio_var(x: &Array2<Float>, confidence: Float) -> Result<Float> {
    if x.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Empty feature matrix".to_string(),
        ));
    }

    let portfolio_returns = x.mean_axis(scirs2_core::ndarray::Axis(1)).unwrap();
    Ok(compute_var(&portfolio_returns.view(), confidence))
}

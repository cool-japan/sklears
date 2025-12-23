//! Performance metrics for financial feature evaluation
//!
//! This module implements performance and quality metrics such as Sharpe ratio,
//! Information ratio, volatility, and related measures.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;
type Float = f64;

/// Compute feature returns (period-over-period changes)
pub(crate) fn compute_feature_returns(feature: &ArrayView1<Float>) -> Vec<Float> {
    if feature.len() < 2 {
        return Vec::new();
    }

    feature
        .windows(2)
        .into_iter()
        .map(|w| {
            let prev = w[0];
            let curr = w[1];
            if prev.abs() > 1e-10 {
                (curr - prev) / prev
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute volatility (standard deviation of returns)
pub(crate) fn compute_volatility(returns: &[Float]) -> Float {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<Float>() / returns.len() as Float;
    let variance =
        returns.iter().map(|&r| (r - mean).powi(2)).sum::<Float>() / returns.len() as Float;

    variance.sqrt()
}

/// Compute tracking error relative to benchmark
pub(crate) fn compute_tracking_error(returns: &[Float], benchmark: Float) -> Float {
    if returns.is_empty() {
        return 0.0;
    }

    let deviations: Vec<Float> = returns.iter().map(|&r| r - benchmark).collect();
    compute_volatility(&deviations)
}

/// Compute Sharpe ratios for all features
pub(crate) fn compute_feature_sharpe_ratios(
    x: &Array2<Float>,
    _y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut sharpe_ratios = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = x.column(i);
        let returns = compute_feature_returns(&feature);

        if returns.is_empty() {
            sharpe_ratios[i] = 0.0;
            continue;
        }

        let mean_return = returns.iter().sum::<Float>() / returns.len() as Float;
        let volatility = compute_volatility(&returns);

        sharpe_ratios[i] = if volatility > 1e-10 {
            mean_return / volatility
        } else {
            0.0
        };
    }

    Ok(sharpe_ratios)
}

/// Compute Information Ratios for all features
pub(crate) fn compute_information_ratios(
    x: &Array2<Float>,
    y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut information_ratios = Array1::zeros(n_features);

    let target_returns = compute_feature_returns(&y.view());
    let benchmark_return = if !target_returns.is_empty() {
        target_returns.iter().sum::<Float>() / target_returns.len() as Float
    } else {
        0.0
    };

    for i in 0..n_features {
        let feature = x.column(i);
        let returns = compute_feature_returns(&feature);

        if returns.is_empty() {
            information_ratios[i] = 0.0;
            continue;
        }

        let excess_returns: Vec<Float> = returns.iter().map(|&r| r - benchmark_return).collect();
        let mean_excess = excess_returns.iter().sum::<Float>() / excess_returns.len() as Float;
        let tracking_error = compute_volatility(&excess_returns);

        information_ratios[i] = if tracking_error > 1e-10 {
            mean_excess / tracking_error
        } else {
            0.0
        };
    }

    Ok(information_ratios)
}

//! Portfolio optimization functions for financial feature selection
//!
//! This module implements mean-variance optimization, risk parity,
//! and other portfolio construction methods.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;
type Float = f64;

/// Compute expected returns for each feature
pub(crate) fn compute_expected_returns(x: &Array2<Float>) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut returns = Array1::zeros(n_features);

    for i in 0..n_features {
        returns[i] = x.column(i).mean().unwrap_or(0.0);
    }

    Ok(returns)
}

/// Mean-variance optimization (maximum Sharpe ratio)
pub(crate) fn mean_variance_optimization(
    expected_returns: &Array1<Float>,
    _cov_matrix: &Array2<Float>,
) -> Result<Array1<Float>> {
    let n_assets = expected_returns.len();
    if n_assets == 0 {
        return Ok(Array1::zeros(0));
    }

    // Simple equal-weighted portfolio as baseline
    let weights = Array1::from_elem(n_assets, 1.0 / n_assets as Float);

    // TODO: Implement proper quadratic programming solver
    // For now, return equal weights
    Ok(weights)
}

/// Minimum variance portfolio
pub(crate) fn minimum_variance_portfolio(cov_matrix: &Array2<Float>) -> Result<Array1<Float>> {
    let n_assets = cov_matrix.nrows();
    if n_assets == 0 {
        return Ok(Array1::zeros(0));
    }

    // Simple inverse volatility weighting as approximation
    let mut weights = Array1::zeros(n_assets);
    for i in 0..n_assets {
        let variance = cov_matrix[[i, i]];
        weights[i] = if variance > 1e-10 {
            1.0 / variance.sqrt()
        } else {
            0.0
        };
    }

    // Normalize weights
    let sum = weights.sum();
    if sum > 1e-10 {
        weights /= sum;
    }

    Ok(weights)
}

/// Risk parity portfolio
pub(crate) fn risk_parity_portfolio(cov_matrix: &Array2<Float>) -> Result<Array1<Float>> {
    let n_assets = cov_matrix.nrows();
    if n_assets == 0 {
        return Ok(Array1::zeros(0));
    }

    // Inverse volatility weighting (approximation of risk parity)
    let mut weights = Array1::zeros(n_assets);
    for i in 0..n_assets {
        let volatility = cov_matrix[[i, i]].sqrt();
        weights[i] = if volatility > 1e-10 {
            1.0 / volatility
        } else {
            0.0
        };
    }

    // Normalize weights to sum to 1
    let sum = weights.sum();
    if sum > 1e-10 {
        weights /= sum;
    }

    Ok(weights)
}

/// Maximum Sharpe ratio portfolio
pub(crate) fn maximum_sharpe_ratio_portfolio(
    expected_returns: &Array1<Float>,
    cov_matrix: &Array2<Float>,
    risk_free_rate: Float,
) -> Result<Array1<Float>> {
    let n_assets = expected_returns.len();
    if n_assets == 0 {
        return Ok(Array1::zeros(0));
    }

    let excess_returns = expected_returns.mapv(|r| r - risk_free_rate);

    // Use mean-variance optimization as approximation
    mean_variance_optimization(&excess_returns, cov_matrix)
}

/// Compute portfolio return
pub(crate) fn compute_portfolio_return(
    weights: &Array1<Float>,
    expected_returns: &Array1<Float>,
) -> Float {
    if weights.len() != expected_returns.len() {
        return 0.0;
    }

    weights
        .iter()
        .zip(expected_returns.iter())
        .map(|(w, r)| w * r)
        .sum()
}

/// Compute portfolio variance
pub(crate) fn compute_portfolio_variance(
    weights: &Array1<Float>,
    cov_matrix: &Array2<Float>,
) -> Float {
    if weights.len() != cov_matrix.nrows() || weights.len() != cov_matrix.ncols() {
        return 0.0;
    }

    let mut variance = 0.0;
    for i in 0..weights.len() {
        for j in 0..weights.len() {
            variance += weights[i] * weights[j] * cov_matrix[[i, j]];
        }
    }

    variance
}

/// Compute diversification ratio
pub(crate) fn compute_diversification_ratio(
    weights: &Array1<Float>,
    cov_matrix: &Array2<Float>,
) -> Float {
    let n = weights.len();
    if n == 0 {
        return 0.0;
    }

    let mut weighted_volatility = 0.0;
    for i in 0..n {
        weighted_volatility += weights[i] * cov_matrix[[i, i]].sqrt();
    }

    let portfolio_volatility = compute_portfolio_variance(weights, cov_matrix).sqrt();

    if portfolio_volatility < 1e-10 {
        return 0.0;
    }

    weighted_volatility / portfolio_volatility
}

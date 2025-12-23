//! Factor models for financial feature selection
//!
//! This module implements various factor models including Fama-French factors,
//! factor loadings, and model R-squared computations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;
type Float = f64;

use super::utilities::compute_pearson_correlation;

/// Compute market factor (CAPM beta)
pub(crate) fn compute_market_factor(
    x: &Array2<Float>,
    _y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let market_returns = x.mean_axis(scirs2_core::ndarray::Axis(1)).unwrap();
    Ok(market_returns)
}

/// Compute SMB (Small Minus Big) factor
pub(crate) fn compute_smb_factor(x: &Array2<Float>) -> Result<Array1<Float>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let small_cap_end = n_features / 2;
    let smb = Array1::from_shape_fn(n_samples, |i| {
        let small_avg = x
            .slice(scirs2_core::ndarray::s![i, ..small_cap_end])
            .mean()
            .unwrap_or(0.0);
        let big_avg = x
            .slice(scirs2_core::ndarray::s![i, small_cap_end..])
            .mean()
            .unwrap_or(0.0);
        small_avg - big_avg
    });

    Ok(smb)
}

/// Compute HML (High Minus Low) factor
pub(crate) fn compute_hml_factor(x: &Array2<Float>) -> Result<Array1<Float>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let third = n_features / 3;
    let hml = Array1::from_shape_fn(n_samples, |i| {
        let high_avg = x
            .slice(scirs2_core::ndarray::s![i, ..third])
            .mean()
            .unwrap_or(0.0);
        let low_avg = x
            .slice(scirs2_core::ndarray::s![i, 2 * third..])
            .mean()
            .unwrap_or(0.0);
        high_avg - low_avg
    });

    Ok(hml)
}

/// Compute RMW (Robust Minus Weak) factor
pub(crate) fn compute_rmw_factor(x: &Array2<Float>) -> Result<Array1<Float>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let mid = n_features / 2;
    let rmw = Array1::from_shape_fn(n_samples, |i| {
        let robust_avg = x
            .slice(scirs2_core::ndarray::s![i, ..mid])
            .mean()
            .unwrap_or(0.0);
        let weak_avg = x
            .slice(scirs2_core::ndarray::s![i, mid..])
            .mean()
            .unwrap_or(0.0);
        robust_avg - weak_avg
    });

    Ok(rmw)
}

/// Compute CMA (Conservative Minus Aggressive) factor
pub(crate) fn compute_cma_factor(x: &Array2<Float>) -> Result<Array1<Float>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let third = n_features / 3;
    let cma = Array1::from_shape_fn(n_samples, |i| {
        let conservative_avg = x
            .slice(scirs2_core::ndarray::s![i, ..third])
            .mean()
            .unwrap_or(0.0);
        let aggressive_avg = x
            .slice(scirs2_core::ndarray::s![i, 2 * third..])
            .mean()
            .unwrap_or(0.0);
        conservative_avg - aggressive_avg
    });

    Ok(cma)
}

/// Compute factor loading (regression coefficient)
pub(crate) fn compute_factor_loading(feature: &ArrayView1<Float>, factor: &Array1<Float>) -> Float {
    if feature.len() != factor.len() || feature.is_empty() {
        return 0.0;
    }

    let correlation = compute_pearson_correlation(feature, &factor.view());
    let factor_std = factor.std(0.0);
    let feature_std = feature.std(0.0);

    if factor_std < 1e-10 {
        return 0.0;
    }

    correlation * (feature_std / factor_std)
}

/// Compute R-squared for factor model
pub(crate) fn compute_factor_model_r_squared(
    feature: &ArrayView1<Float>,
    factors: &[Array1<Float>],
) -> Float {
    if factors.is_empty() || feature.is_empty() {
        return 0.0;
    }

    let mean = feature.mean().unwrap_or(0.0);
    let ss_tot: Float = feature.iter().map(|&x| (x - mean).powi(2)).sum();

    if ss_tot < 1e-10 {
        return 0.0;
    }

    // Compute predicted values using factor loadings
    let mut predictions = vec![mean; feature.len()];
    for factor in factors {
        if factor.len() != feature.len() {
            continue;
        }

        let loading = compute_factor_loading(feature, factor);
        let factor_mean = factor.mean().unwrap_or(0.0);

        for i in 0..feature.len() {
            predictions[i] += loading * (factor[i] - factor_mean);
        }
    }

    let ss_res: Float = feature
        .iter()
        .enumerate()
        .map(|(i, &x)| (x - predictions[i]).powi(2))
        .sum();

    1.0 - (ss_res / ss_tot)
}

/// Compute macroeconomic model R-squared
pub(crate) fn compute_macro_model_r_squared(
    feature: &ArrayView1<Float>,
    macro_factors: &[Array1<Float>],
) -> Float {
    compute_factor_model_r_squared(feature, macro_factors)
}

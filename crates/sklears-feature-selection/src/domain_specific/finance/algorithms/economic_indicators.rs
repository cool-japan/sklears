//! Economic indicator simulations and scoring
//!
//! This module provides functions to simulate and score economic indicators
//! such as GDP growth, inflation, interest rates, and unemployment.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;
type Float = f64;

/// Compute economic indicator scores
pub(crate) fn compute_economic_indicator_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    // Simulate economic indicators
    let gdp_growth = simulate_gdp_growth(x)?;
    let inflation = simulate_inflation(x)?;
    let interest_rate = simulate_interest_rate(x)?;
    let unemployment = simulate_unemployment(x)?;

    // Score each feature by correlation with economic indicators and target
    for i in 0..n_features {
        let feature = x.column(i);

        let corr_gdp = super::utilities::compute_pearson_correlation(&feature, &gdp_growth.view());
        let corr_inflation =
            super::utilities::compute_pearson_correlation(&feature, &inflation.view());
        let corr_ir =
            super::utilities::compute_pearson_correlation(&feature, &interest_rate.view());
        let corr_unemp =
            super::utilities::compute_pearson_correlation(&feature, &unemployment.view());
        let corr_target = super::utilities::compute_pearson_correlation(&feature, &y.view());

        // Weighted average of correlations
        scores[i] = (corr_gdp.abs() * 0.3
            + corr_inflation.abs() * 0.2
            + corr_ir.abs() * 0.2
            + corr_unemp.abs() * 0.1
            + corr_target.abs() * 0.2)
            .abs();
    }

    Ok(scores)
}

/// Simulate GDP growth from features
pub(crate) fn simulate_gdp_growth(x: &Array2<Float>) -> Result<Array1<Float>> {
    let n_samples = x.nrows();

    // Simple simulation: GDP growth as average of positive returns
    let gdp = Array1::from_shape_fn(n_samples, |i| {
        let row = x.row(i);
        let positive_changes: Float = row.iter().filter(|&&v| v > 0.0).sum();
        let count = row.iter().filter(|&&v| v > 0.0).count();

        if count > 0 {
            positive_changes / count as Float
        } else {
            0.0
        }
    });

    Ok(gdp)
}

/// Simulate inflation from features
pub(crate) fn simulate_inflation(x: &Array2<Float>) -> Result<Array1<Float>> {
    let n_samples = x.nrows();

    // Inflation as moving average of price changes
    let inflation = Array1::from_shape_fn(n_samples, |i| {
        if i == 0 {
            return 0.0;
        }

        let curr_row = x.row(i);
        let prev_row = x.row(i - 1);

        let price_changes: Float = curr_row
            .iter()
            .zip(prev_row.iter())
            .map(|(c, p)| if p.abs() > 1e-10 { (c - p) / p } else { 0.0 })
            .sum();

        price_changes / x.ncols() as Float
    });

    Ok(inflation)
}

/// Simulate interest rate from features
pub(crate) fn simulate_interest_rate(x: &Array2<Float>) -> Result<Array1<Float>> {
    let n_samples = x.nrows();

    // Interest rate as function of volatility (higher volatility → higher rates)
    let interest_rate = Array1::from_shape_fn(n_samples, |i| {
        let row = x.row(i);
        let mean = row.mean().unwrap_or(0.0);
        let volatility = row
            .mapv(|v| (v - mean).powi(2))
            .mean()
            .unwrap_or(0.0)
            .sqrt();

        // Base rate + volatility premium
        0.02 + volatility * 0.1
    });

    Ok(interest_rate)
}

/// Simulate unemployment from features
pub(crate) fn simulate_unemployment(x: &Array2<Float>) -> Result<Array1<Float>> {
    let n_samples = x.nrows();

    // Unemployment inversely related to economic activity
    let unemployment = Array1::from_shape_fn(n_samples, |i| {
        let row = x.row(i);
        let activity = row.mean().unwrap_or(0.0);

        // Inverse relationship with activity level
        let base_unemployment = 0.05;
        if activity > 0.0 {
            base_unemployment / (1.0 + activity)
        } else {
            base_unemployment * 2.0
        }
    });

    Ok(unemployment)
}

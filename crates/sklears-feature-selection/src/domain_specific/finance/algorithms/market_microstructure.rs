//! Market microstructure features for financial feature selection
//!
//! This module implements order flow, bid-ask spread, market impact,
//! and transaction cost analysis.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;
type Float = f64;

/// Compute order flow scores
pub(crate) fn compute_order_flow_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = x.column(i);

        // Compute order flow imbalance
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;

        for j in 1..feature.len() {
            if feature[j] > feature[j - 1] {
                buy_volume += feature[j];
            } else {
                sell_volume += feature[j];
            }
        }

        let total_volume = buy_volume + sell_volume;
        let order_flow_imbalance = if total_volume > 1e-10 {
            (buy_volume - sell_volume) / total_volume
        } else {
            0.0
        };

        // Score based on correlation with target
        let mut correlation = 0.0;
        if y.len() == feature.len() {
            let y_changes: Vec<Float> = (1..y.len()).map(|j| y[j] - y[j - 1]).collect();
            if !y_changes.is_empty() {
                let y_mean = y_changes.iter().sum::<Float>() / y_changes.len() as Float;
                correlation = if y_mean > 0.0 {
                    order_flow_imbalance.abs()
                } else {
                    0.0
                };
            }
        }

        scores[i] = correlation;
    }

    Ok(scores)
}

/// Compute bid-ask spread scores
pub(crate) fn compute_bid_ask_scores(
    x: &Array2<Float>,
    _y: &Array1<Float>,
    spread_pct: Float,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = x.column(i);

        // Estimate effective spread
        let mut spreads = Vec::new();
        for j in 1..feature.len() {
            let mid_price = (feature[j] + feature[j - 1]) / 2.0;
            let spread = spread_pct * mid_price;
            spreads.push(spread);
        }

        if spreads.is_empty() {
            scores[i] = 0.0;
            continue;
        }

        // Average spread as liquidity measure (lower is better)
        let avg_spread = spreads.iter().sum::<Float>() / spreads.len() as Float;
        let feature_mean = feature.mean().unwrap_or(1.0);

        // Normalize by price level
        scores[i] = if feature_mean > 1e-10 {
            1.0 / (1.0 + avg_spread / feature_mean)
        } else {
            0.0
        };
    }

    Ok(scores)
}

/// Compute market impact scores
pub(crate) fn compute_market_impact_scores(
    x: &Array2<Float>,
    _y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = x.column(i);

        // Estimate market impact from price changes and volatility
        let mut impacts = Vec::new();
        for j in 2..feature.len() {
            let price_change = (feature[j] - feature[j - 1]).abs();
            let prev_change = (feature[j - 1] - feature[j - 2]).abs();

            // Impact is asymmetry in price changes
            if prev_change > 1e-10 {
                impacts.push(price_change / prev_change);
            }
        }

        if impacts.is_empty() {
            scores[i] = 0.0;
            continue;
        }

        let avg_impact = impacts.iter().sum::<Float>() / impacts.len() as Float;
        scores[i] = 1.0 / (1.0 + avg_impact); // Lower impact is better
    }

    Ok(scores)
}

/// Compute transaction cost scores
pub(crate) fn compute_transaction_cost_scores(
    x: &Array2<Float>,
    _y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    let transaction_cost_rate = 0.001; // 10 basis points

    for i in 0..n_features {
        let feature = x.column(i);

        // Estimate turnover
        let mut total_change = 0.0;
        for j in 1..feature.len() {
            total_change += (feature[j] - feature[j - 1]).abs();
        }

        let avg_price = feature.mean().unwrap_or(1.0);
        if avg_price < 1e-10 {
            scores[i] = 0.0;
            continue;
        }

        let turnover = total_change / (feature.len() as Float * avg_price);
        let total_cost = turnover * transaction_cost_rate;

        // Score inversely with costs (lower cost is better)
        scores[i] = 1.0 / (1.0 + total_cost);
    }

    Ok(scores)
}

/// Compute cross-asset correlation scores
pub(crate) fn compute_cross_asset_scores(
    x: &Array2<Float>,
    _y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let mut avg_correlation = 0.0;
        let mut count = 0;

        for j in 0..n_features {
            if i != j {
                let corr =
                    super::utilities::compute_pearson_correlation(&x.column(i), &x.column(j));
                avg_correlation += corr.abs();
                count += 1;
            }
        }

        scores[i] = if count > 0 {
            avg_correlation / count as Float
        } else {
            0.0
        };
    }

    Ok(scores)
}

/// Compute spillover scores
pub(crate) fn compute_spillover_scores(
    x: &Array2<Float>,
    _y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = x.column(i);

        // Compute lagged correlations with other features
        let mut spillover_effect = 0.0;
        let mut count = 0;

        for j in 0..n_features {
            if i != j {
                let other = x.column(j);

                // Compute lagged correlation (1-period lag)
                if feature.len() > 1 && other.len() > 1 {
                    let feature_lagged =
                        feature.slice(scirs2_core::ndarray::s![..feature.len() - 1]);
                    let other_current = other.slice(scirs2_core::ndarray::s![1..]);

                    let corr = super::utilities::compute_pearson_correlation(
                        &feature_lagged,
                        &other_current,
                    );
                    spillover_effect += corr.abs();
                    count += 1;
                }
            }
        }

        scores[i] = if count > 0 {
            spillover_effect / count as Float
        } else {
            0.0
        };
    }

    Ok(scores)
}

//! Regime detection and transition analysis for financial features
//!
//! This module implements market regime detection using various methods
//! including Hidden Markov Models and threshold-based approaches.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;
type Float = f64;

/// Detect market regimes using various methods
pub(crate) fn detect_market_regimes(x: &Array2<Float>, method: &str) -> Result<Array2<Float>> {
    match method {
        "volatility" => detect_regimes_by_volatility(x),
        "momentum" => detect_regimes_by_momentum(x),
        "hmm" => detect_market_regimes_hmm(x, 2),
        _ => detect_regimes_by_volatility(x),
    }
}

/// Detect regimes based on volatility levels
fn detect_regimes_by_volatility(x: &Array2<Float>) -> Result<Array2<Float>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut regimes = Array2::zeros((n_samples, 2)); // 2 regimes: high/low volatility

    for i in 0..n_features {
        let feature = x.column(i);

        // Compute rolling volatility
        let window_size = 20.min(n_samples);
        for j in window_size..n_samples {
            let window = feature.slice(scirs2_core::ndarray::s![j - window_size..j]);
            let mean = window.mean().unwrap_or(0.0);
            let volatility = window
                .mapv(|x| (x - mean).powi(2))
                .mean()
                .unwrap_or(0.0)
                .sqrt();

            // Classify as high (1) or low (0) volatility
            let threshold = window.std(0.0);
            if volatility > threshold {
                regimes[[j, 1]] += 1.0;
            } else {
                regimes[[j, 0]] += 1.0;
            }
        }
    }

    // Normalize
    for i in 0..n_samples {
        let sum = regimes[[i, 0]] + regimes[[i, 1]];
        if sum > 1e-10 {
            regimes[[i, 0]] /= sum;
            regimes[[i, 1]] /= sum;
        }
    }

    Ok(regimes)
}

/// Detect regimes based on momentum
fn detect_regimes_by_momentum(x: &Array2<Float>) -> Result<Array2<Float>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut regimes = Array2::zeros((n_samples, 2)); // 2 regimes: positive/negative momentum

    for i in 0..n_features {
        let feature = x.column(i);

        for j in 1..n_samples {
            let momentum = feature[j] - feature[j - 1];

            if momentum > 0.0 {
                regimes[[j, 1]] += 1.0; // Positive momentum
            } else {
                regimes[[j, 0]] += 1.0; // Negative momentum
            }
        }
    }

    // Normalize
    for i in 0..n_samples {
        let sum = regimes[[i, 0]] + regimes[[i, 1]];
        if sum > 1e-10 {
            regimes[[i, 0]] /= sum;
            regimes[[i, 1]] /= sum;
        }
    }

    Ok(regimes)
}

/// Detect market regimes using Hidden Markov Model
pub(crate) fn detect_market_regimes_hmm(
    x: &Array2<Float>,
    n_regimes: usize,
) -> Result<Array2<Float>> {
    let n_samples = x.nrows();
    let mut regime_probs = Array2::zeros((n_samples, n_regimes));

    // Simplified HMM using k-means-like clustering
    // Compute feature statistics for each sample
    let mut sample_stats = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let row = x.row(i);
        let mean = row.mean().unwrap_or(0.0);
        let std = row.std(0.0);
        sample_stats.push((mean, std));
    }

    // Assign regimes based on statistics
    // Find min/max for normalization
    let mean_min = sample_stats
        .iter()
        .map(|(m, _)| *m)
        .fold(Float::INFINITY, Float::min);
    let mean_max = sample_stats
        .iter()
        .map(|(m, _)| *m)
        .fold(Float::NEG_INFINITY, Float::max);

    for i in 0..n_samples {
        let (mean, _std) = sample_stats[i];

        // Normalize mean to [0, 1]
        let normalized = if mean_max - mean_min > 1e-10 {
            (mean - mean_min) / (mean_max - mean_min)
        } else {
            0.5
        };

        // Assign regime probabilities
        let regime_idx = (normalized * n_regimes as Float).floor() as usize;
        let regime_idx = regime_idx.min(n_regimes - 1);

        regime_probs[[i, regime_idx]] = 1.0;
    }

    Ok(regime_probs)
}

/// Compute regime transition probabilities
pub(crate) fn compute_regime_transitions(regime_probs: &Array2<Float>) -> Result<Array2<Float>> {
    let n_regimes = regime_probs.ncols();
    let n_samples = regime_probs.nrows();

    let mut transition_matrix = Array2::zeros((n_regimes, n_regimes));

    // Count transitions
    for t in 1..n_samples {
        // Find most likely regime at t-1 and t
        let prev_regime = regime_probs
            .row(t - 1)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let curr_regime = regime_probs
            .row(t)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        transition_matrix[[prev_regime, curr_regime]] += 1.0;
    }

    // Normalize rows to get probabilities
    for i in 0..n_regimes {
        let row_sum = transition_matrix.row(i).sum();
        if row_sum > 1e-10 {
            for j in 0..n_regimes {
                transition_matrix[[i, j]] /= row_sum;
            }
        }
    }

    Ok(transition_matrix)
}

/// Compute regime-based feature scores
pub(crate) fn compute_regime_based_scores(
    x: &Array2<Float>,
    _y: &Array1<Float>,
    method: &str,
) -> Result<Array1<Float>> {
    let regimes = detect_market_regimes(x, method)?;
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    // Score features by how well they predict regime transitions
    for i in 0..n_features {
        let feature = x.column(i);

        // Compute correlation with most likely regime
        let most_likely_regime = Array1::from_shape_fn(regimes.nrows(), |j| {
            regimes
                .row(j)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as Float)
                .unwrap_or(0.0)
        });

        let corr =
            super::utilities::compute_pearson_correlation(&feature, &most_likely_regime.view());
        scores[i] = corr.abs();
    }

    Ok(scores)
}

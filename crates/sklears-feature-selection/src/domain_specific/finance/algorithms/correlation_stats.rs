//! Correlation and covariance statistics for financial feature selection
//!
//! This module provides functions for computing correlation matrices,
//! covariance matrices, and correlation-based feature scores.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;
type Float = f64;

use super::utilities::compute_pearson_correlation;

/// Compute correlation matrix for features
pub(crate) fn compute_correlation_matrix(x: &Array2<Float>) -> Result<Array2<Float>> {
    let n_features = x.ncols();
    let mut corr_matrix = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        corr_matrix[[i, i]] = 1.0;
        for j in (i + 1)..n_features {
            let corr = compute_pearson_correlation(&x.column(i), &x.column(j));
            corr_matrix[[i, j]] = corr;
            corr_matrix[[j, i]] = corr;
        }
    }

    Ok(corr_matrix)
}

/// Compute covariance matrix for features
pub(crate) fn compute_covariance_matrix(x: &Array2<Float>) -> Result<Array2<Float>> {
    let n_features = x.ncols();
    let n_samples = x.nrows();

    if n_samples == 0 {
        return Ok(Array2::zeros((n_features, n_features)));
    }

    let mut cov_matrix = Array2::zeros((n_features, n_features));

    // Compute means
    let means: Vec<Float> = (0..n_features)
        .map(|i| x.column(i).mean().unwrap_or(0.0))
        .collect();

    // Compute covariances
    for i in 0..n_features {
        for j in i..n_features {
            let mut cov = 0.0;
            for k in 0..n_samples {
                cov += (x[[k, i]] - means[i]) * (x[[k, j]] - means[j]);
            }
            cov /= (n_samples - 1).max(1) as Float;

            cov_matrix[[i, j]] = cov;
            cov_matrix[[j, i]] = cov;
        }
    }

    Ok(cov_matrix)
}

/// Compute correlation-based feature scores
pub(crate) fn compute_correlation_based_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = x.column(i);
        scores[i] = compute_pearson_correlation(&feature, &y.view()).abs();
    }

    Ok(scores)
}

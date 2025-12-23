//! Utility functions for financial feature selection
//!
//! This module provides helper functions for correlation computation,
//! feature selection, and other utility operations.

use scirs2_core::ndarray::{Array1, ArrayView1};

type Float = f64;

/// Compute Pearson correlation between two arrays
pub(crate) fn compute_pearson_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as Float;
    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..x.len() {
        let diff_x = x[i] - mean_x;
        let diff_y = y[i] - mean_y;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Select top K features based on scores
pub(crate) fn select_top_k_features(scores: &Array1<Float>, k: usize) -> Vec<usize> {
    let mut indexed_scores: Vec<(usize, Float)> =
        scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();

    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    indexed_scores.iter().take(k).map(|(i, _)| *i).collect()
}

/// Select features above a threshold
pub(crate) fn select_features_by_threshold(scores: &Array1<Float>, threshold: Float) -> Vec<usize> {
    scores
        .iter()
        .enumerate()
        .filter(|(_, &score)| score >= threshold)
        .map(|(i, _)| i)
        .collect()
}

//! Pairwise distance and similarity metrics
//!
//! This module provides functions to compute pairwise distances and similarities
//! between samples in a dataset.

#![allow(non_snake_case)] // Allow X, Y naming convention for ML matrices

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;

/// Supported distance metrics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Chebyshev distance (L∞ norm)
    Chebyshev,
    /// Minkowski distance with parameter p
    Minkowski(f64),
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Hamming distance
    Hamming,
}

/// Supported kernel functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelFunction {
    /// Linear kernel: K(x, y) = x^T y
    Linear,
    /// Polynomial kernel: K(x, y) = (gamma * x^T y + coef0)^degree
    Polynomial { degree: f64, gamma: f64, coef0: f64 },
    /// RBF (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    RBF { gamma: f64 },
    /// Sigmoid kernel: K(x, y) = tanh(gamma * x^T y + coef0)
    Sigmoid { gamma: f64, coef0: f64 },
    /// Cosine similarity: K(x, y) = x^T y / (||x|| * ||y||)
    Cosine,
    /// Laplacian kernel: K(x, y) = exp(-gamma * ||x - y||_1)
    Laplacian { gamma: f64 },
    /// Chi-squared kernel: K(x, y) = exp(-gamma * chi2(x, y))
    ChiSquared { gamma: f64 },
    /// Intersection kernel: K(x, y) = sum(min(x_i, y_i))
    Intersection,
    /// Hellinger kernel: K(x, y) = sum(sqrt(x_i * y_i))
    Hellinger,
    /// Jensen-Shannon kernel: K(x, y) = 1 - JS_divergence(x, y)
    JensenShannon,
    /// Additive chi-squared kernel
    AdditiveChiSquared,
}

/// Compute euclidean distances between samples
///
/// # Arguments
///
/// * `X` - Input array of shape (n_samples_X, n_features)
/// * `Y` - Optional input array of shape (n_samples_Y, n_features). If None, compute pairwise distances within X
///
/// # Returns
///
/// Distance matrix of shape (n_samples_X, n_samples_Y) or (n_samples_X, n_samples_X) if Y is None
///
/// # Examples
///
/// ```rust
/// use sklears_metrics::pairwise::euclidean_distances;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0., 0.], [1., 1.], [2., 2.]];
/// let distances = euclidean_distances(&X.view(), None).unwrap();
/// ```
pub fn euclidean_distances(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
) -> MetricsResult<Array2<f64>> {
    let (n_samples_x, n_features_x) = X.dim();

    if n_samples_x == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let (Y_ref, n_samples_y) = match Y {
        Some(y) => {
            let (n_samples_y, n_features_y) = y.dim();
            if n_features_x != n_features_y {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![n_samples_x, n_features_x],
                    actual: vec![n_samples_y, n_features_y],
                });
            }
            (y.view(), n_samples_y)
        }
        None => (X.view(), n_samples_x),
    };

    let mut distances = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let diff = &X.row(i) - &Y_ref.row(j);
            distances[[i, j]] = diff.dot(&diff).sqrt();
        }
    }

    Ok(distances)
}

/// Compute euclidean distances ignoring NaN values
///
/// # Arguments
///
/// * `X` - Input array of shape (n_samples_X, n_features)
/// * `Y` - Optional input array of shape (n_samples_Y, n_features). If None, compute pairwise distances within X
///
/// # Returns
///
/// Distance matrix of shape (n_samples_X, n_samples_Y) or (n_samples_X, n_samples_X) if Y is None
///
/// # Notes
///
/// When computing distances, NaN values are ignored. If all values are NaN for a pair,
/// the distance is set to NaN.
pub fn nan_euclidean_distances(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
) -> MetricsResult<Array2<f64>> {
    let (n_samples_x, n_features_x) = X.dim();

    if n_samples_x == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let (Y_ref, n_samples_y) = match Y {
        Some(y) => {
            let (n_samples_y, n_features_y) = y.dim();
            if n_features_x != n_features_y {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![n_samples_x, n_features_x],
                    actual: vec![n_samples_y, n_features_y],
                });
            }
            (y.view(), n_samples_y)
        }
        None => (X.view(), n_samples_x),
    };

    let mut distances = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let mut sum_sq_diff = 0.0;
            let mut valid_count = 0;

            for k in 0..n_features_x {
                let x_val = X[[i, k]];
                let y_val = Y_ref[[j, k]];

                if !x_val.is_nan() && !y_val.is_nan() {
                    let diff = x_val - y_val;
                    sum_sq_diff += diff * diff;
                    valid_count += 1;
                }
            }

            distances[[i, j]] = if valid_count > 0 {
                // Scale by the ratio of total features to valid features
                let scale = n_features_x as f64 / valid_count as f64;
                (sum_sq_diff * scale).sqrt()
            } else {
                f64::NAN
            };
        }
    }

    Ok(distances)
}

/// Compute pairwise distances with various metrics
///
/// # Arguments
///
/// * `X` - Input array of shape (n_samples_X, n_features)
/// * `Y` - Optional input array of shape (n_samples_Y, n_features). If None, compute pairwise distances within X
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// Distance matrix of shape (n_samples_X, n_samples_Y) or (n_samples_X, n_samples_X) if Y is None
///
/// # Examples
///
/// ```rust
/// use sklears_metrics::pairwise::{pairwise_distances, DistanceMetric};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0., 0.], [1., 1.], [2., 2.]];
/// let distances = pairwise_distances(&X.view(), None, DistanceMetric::Manhattan).unwrap();
/// ```
pub fn pairwise_distances(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
    metric: DistanceMetric,
) -> MetricsResult<Array2<f64>> {
    match metric {
        DistanceMetric::Euclidean => euclidean_distances(X, Y),
        DistanceMetric::Manhattan => manhattan_distances(X, Y),
        DistanceMetric::Chebyshev => chebyshev_distances(X, Y),
        DistanceMetric::Minkowski(p) => minkowski_distances(X, Y, p),
        DistanceMetric::Cosine => cosine_distances(X, Y),
        DistanceMetric::Hamming => hamming_distances(X, Y),
    }
}

/// Compute Manhattan (L1) distances
fn manhattan_distances(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
) -> MetricsResult<Array2<f64>> {
    let (n_samples_x, n_features_x) = X.dim();

    if n_samples_x == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let (Y_ref, n_samples_y) = match Y {
        Some(y) => {
            let (n_samples_y, n_features_y) = y.dim();
            if n_features_x != n_features_y {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![n_samples_x, n_features_x],
                    actual: vec![n_samples_y, n_features_y],
                });
            }
            (y.view(), n_samples_y)
        }
        None => (X.view(), n_samples_x),
    };

    let mut distances = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let diff = &X.row(i) - &Y_ref.row(j);
            distances[[i, j]] = diff.mapv(f64::abs).sum();
        }
    }

    Ok(distances)
}

/// Compute Chebyshev (L∞) distances
fn chebyshev_distances(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
) -> MetricsResult<Array2<f64>> {
    let (n_samples_x, n_features_x) = X.dim();

    if n_samples_x == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let (Y_ref, n_samples_y) = match Y {
        Some(y) => {
            let (n_samples_y, n_features_y) = y.dim();
            if n_features_x != n_features_y {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![n_samples_x, n_features_x],
                    actual: vec![n_samples_y, n_features_y],
                });
            }
            (y.view(), n_samples_y)
        }
        None => (X.view(), n_samples_x),
    };

    let mut distances = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let diff = &X.row(i) - &Y_ref.row(j);
            distances[[i, j]] = diff
                .mapv(f64::abs)
                .fold(0.0f64, |max_val, &val| max_val.max(val));
        }
    }

    Ok(distances)
}

/// Compute Minkowski distances
fn minkowski_distances(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
    p: f64,
) -> MetricsResult<Array2<f64>> {
    if p <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "Minkowski parameter p must be positive".to_string(),
        ));
    }

    let (n_samples_x, n_features_x) = X.dim();

    if n_samples_x == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let (Y_ref, n_samples_y) = match Y {
        Some(y) => {
            let (n_samples_y, n_features_y) = y.dim();
            if n_features_x != n_features_y {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![n_samples_x, n_features_x],
                    actual: vec![n_samples_y, n_features_y],
                });
            }
            (y.view(), n_samples_y)
        }
        None => (X.view(), n_samples_x),
    };

    let mut distances = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let diff = &X.row(i) - &Y_ref.row(j);
            distances[[i, j]] = diff.mapv(|x| x.abs().powf(p)).sum().powf(1.0 / p);
        }
    }

    Ok(distances)
}

/// Compute cosine distances (1 - cosine similarity)
fn cosine_distances(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
) -> MetricsResult<Array2<f64>> {
    let (n_samples_x, n_features_x) = X.dim();

    if n_samples_x == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let (Y_ref, n_samples_y) = match Y {
        Some(y) => {
            let (n_samples_y, n_features_y) = y.dim();
            if n_features_x != n_features_y {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![n_samples_x, n_features_x],
                    actual: vec![n_samples_y, n_features_y],
                });
            }
            (y.view(), n_samples_y)
        }
        None => (X.view(), n_samples_x),
    };

    let mut distances = Array2::zeros((n_samples_x, n_samples_y));

    // Precompute norms
    let x_norms: Vec<f64> = (0..n_samples_x)
        .map(|i| X.row(i).dot(&X.row(i)).sqrt())
        .collect();

    let y_norms: Vec<f64> = (0..n_samples_y)
        .map(|j| Y_ref.row(j).dot(&Y_ref.row(j)).sqrt())
        .collect();

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let dot_product = X.row(i).dot(&Y_ref.row(j));
            let norm_product = x_norms[i] * y_norms[j];

            distances[[i, j]] = if norm_product > 0.0 {
                1.0 - (dot_product / norm_product)
            } else {
                1.0
            };
        }
    }

    Ok(distances)
}

/// Compute Hamming distances
fn hamming_distances(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
) -> MetricsResult<Array2<f64>> {
    let (n_samples_x, n_features_x) = X.dim();

    if n_samples_x == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let (Y_ref, n_samples_y) = match Y {
        Some(y) => {
            let (n_samples_y, n_features_y) = y.dim();
            if n_features_x != n_features_y {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![n_samples_x, n_features_x],
                    actual: vec![n_samples_y, n_features_y],
                });
            }
            (y.view(), n_samples_y)
        }
        None => (X.view(), n_samples_x),
    };

    let mut distances = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let mut hamming_dist = 0.0;
            for k in 0..n_features_x {
                if (X[[i, k]] - Y_ref[[j, k]]).abs() > f64::EPSILON {
                    hamming_dist += 1.0;
                }
            }
            distances[[i, j]] = hamming_dist / n_features_x as f64;
        }
    }

    Ok(distances)
}

/// Find minimum distances and their indices
///
/// # Arguments
///
/// * `X` - Input array of shape (n_samples_X, n_features)
/// * `Y` - Optional input array of shape (n_samples_Y, n_features). If None, compute within X
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// Array of indices of shape (n_samples_X,) containing the index of the nearest neighbor in Y
/// for each sample in X
///
/// # Examples
///
/// ```rust
/// use sklears_metrics::pairwise::{pairwise_distances_argmin, DistanceMetric};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0., 0.], [1., 1.]];
/// let Y = array![[0.5, 0.5], [2., 2.]];
/// let indices = pairwise_distances_argmin(&X.view(), Some(&Y.view()), DistanceMetric::Euclidean).unwrap();
/// ```
pub fn pairwise_distances_argmin(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
    metric: DistanceMetric,
) -> MetricsResult<Array1<usize>> {
    let distances = pairwise_distances(X, Y, metric)?;
    let n_samples_x = X.nrows();
    let mut argmin = Array1::zeros(n_samples_x);

    for i in 0..n_samples_x {
        let row = distances.row(i);
        let mut min_idx = 0;
        let mut min_val = row[0];

        for j in 1..row.len() {
            if row[j] < min_val {
                min_val = row[j];
                min_idx = j;
            }
        }

        argmin[i] = min_idx;
    }

    Ok(argmin)
}

/// Find both minimum distances and their indices
///
/// # Arguments
///
/// * `X` - Input array of shape (n_samples_X, n_features)
/// * `Y` - Optional input array of shape (n_samples_Y, n_features). If None, compute within X
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// Tuple of (indices, distances) where:
/// - indices: Array of shape (n_samples_X,) with indices of nearest neighbors
/// - distances: Array of shape (n_samples_X,) with distances to nearest neighbors
pub fn pairwise_distances_argmin_min(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
    metric: DistanceMetric,
) -> MetricsResult<(Array1<usize>, Array1<f64>)> {
    let distances = pairwise_distances(X, Y, metric)?;
    let n_samples_x = X.nrows();
    let mut argmin = Array1::zeros(n_samples_x);
    let mut min_distances = Array1::zeros(n_samples_x);

    for i in 0..n_samples_x {
        let row = distances.row(i);
        let mut min_idx = 0;
        let mut min_val = row[0];

        for j in 1..row.len() {
            if row[j] < min_val {
                min_val = row[j];
                min_idx = j;
            }
        }

        argmin[i] = min_idx;
        min_distances[i] = min_val;
    }

    Ok((argmin, min_distances))
}

/// Compute kernel matrix using various kernel functions
///
/// # Arguments
///
/// * `X` - Input array of shape (n_samples_X, n_features)
/// * `Y` - Optional input array of shape (n_samples_Y, n_features). If None, compute within X
/// * `kernel` - Kernel function to use
///
/// # Returns
///
/// Kernel matrix of shape (n_samples_X, n_samples_Y) or (n_samples_X, n_samples_X) if Y is None
///
/// # Examples
///
/// ```rust
/// use sklears_metrics::pairwise::{pairwise_kernels, KernelFunction};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0., 0.], [1., 1.], [2., 2.]];
/// let kernel_matrix = pairwise_kernels(&X.view(), None, KernelFunction::RBF { gamma: 0.5 }).unwrap();
/// ```
pub fn pairwise_kernels<'a>(
    X: &ArrayView2<'a, f64>,
    Y: Option<&ArrayView2<'a, f64>>,
    kernel: KernelFunction,
) -> MetricsResult<Array2<f64>> {
    match kernel {
        KernelFunction::Linear => linear_kernel(X, Y),
        KernelFunction::Polynomial {
            degree,
            gamma,
            coef0,
        } => polynomial_kernel(X, Y, degree, gamma, coef0),
        KernelFunction::RBF { gamma } => rbf_kernel(X, Y, gamma),
        KernelFunction::Sigmoid { gamma, coef0 } => sigmoid_kernel(X, Y, gamma, coef0),
        KernelFunction::Cosine => cosine_kernel(X, Y),
        KernelFunction::Laplacian { gamma } => laplacian_kernel(X, Y, gamma),
        KernelFunction::ChiSquared { gamma } => chi_squared_kernel(X, Y, gamma),
        KernelFunction::Intersection => intersection_kernel(X, Y),
        KernelFunction::Hellinger => hellinger_kernel(X, Y),
        KernelFunction::JensenShannon => jensen_shannon_kernel(X, Y),
        KernelFunction::AdditiveChiSquared => additive_chi_squared_kernel(X, Y),
    }
}

/// Compute the 1-Wasserstein distance (Earth Mover's Distance) between two 1D distributions
///
/// This computes the optimal transport cost between two probability distributions
/// using the closed-form solution for the 1-Wasserstein distance.
///
/// # Arguments
/// * `u_values` - Values of the first distribution
/// * `u_weights` - Weights of the first distribution (must sum to 1)
/// * `v_values` - Values of the second distribution  
/// * `v_weights` - Weights of the second distribution (must sum to 1)
///
/// # Returns
/// The 1-Wasserstein distance between the distributions
///
/// # Examples
///
/// ```rust
/// use sklears_metrics::pairwise::wasserstein_distance;
/// use scirs2_core::ndarray::array;
///
/// let u_values = array![0.0, 1.0, 2.0];
/// let u_weights = array![0.5, 0.3, 0.2];
/// let v_values = array![1.0, 2.0, 3.0];
/// let v_weights = array![0.2, 0.5, 0.3];
/// let distance = wasserstein_distance(&u_values, &u_weights, &v_values, &v_weights).unwrap();
/// ```
pub fn wasserstein_distance(
    u_values: &Array1<f64>,
    u_weights: &Array1<f64>,
    v_values: &Array1<f64>,
    v_weights: &Array1<f64>,
) -> MetricsResult<f64> {
    if u_values.len() != u_weights.len() || v_values.len() != v_weights.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![u_values.len()],
            actual: vec![u_weights.len(), v_values.len(), v_weights.len()],
        });
    }

    if u_values.is_empty() || v_values.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check that weights sum to approximately 1
    let u_sum = u_weights.sum();
    let v_sum = v_weights.sum();
    if (u_sum - 1.0).abs() > 1e-10 || (v_sum - 1.0).abs() > 1e-10 {
        return Err(MetricsError::InvalidParameter(
            "Weights must sum to 1.0".to_string(),
        ));
    }

    // Check that all weights are non-negative
    if u_weights.iter().any(|&w| w < 0.0) || v_weights.iter().any(|&w| w < 0.0) {
        return Err(MetricsError::InvalidParameter(
            "Weights must be non-negative".to_string(),
        ));
    }

    // Sort values with their weights
    let mut u_pairs: Vec<(f64, f64)> = u_values
        .iter()
        .zip(u_weights.iter())
        .map(|(&v, &w)| (v, w))
        .collect();
    let mut v_pairs: Vec<(f64, f64)> = v_values
        .iter()
        .zip(v_weights.iter())
        .map(|(&v, &w)| (v, w))
        .collect();

    u_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    v_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Compute cumulative distribution functions
    let mut u_cdf = vec![0.0; u_pairs.len() + 1];
    let mut v_cdf = vec![0.0; v_pairs.len() + 1];

    for i in 0..u_pairs.len() {
        u_cdf[i + 1] = u_cdf[i] + u_pairs[i].1;
    }
    for i in 0..v_pairs.len() {
        v_cdf[i + 1] = v_cdf[i] + v_pairs[i].1;
    }

    // Merge the sorted values to compute Wasserstein distance
    let mut all_values = Vec::new();
    for &(value, _) in &u_pairs {
        all_values.push(value);
    }
    for &(value, _) in &v_pairs {
        all_values.push(value);
    }
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_values.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    let mut distance = 0.0;
    let mut u_idx = 0;
    let mut v_idx = 0;

    for i in 1..all_values.len() {
        let current_value = all_values[i];
        let prev_value = all_values[i - 1];

        // Find cumulative probabilities at current value
        while u_idx < u_pairs.len() && u_pairs[u_idx].0 <= current_value {
            u_idx += 1;
        }
        while v_idx < v_pairs.len() && v_pairs[v_idx].0 <= current_value {
            v_idx += 1;
        }

        let u_cum = u_cdf[u_idx];
        let v_cum = v_cdf[v_idx];

        distance += (u_cum - v_cum).abs() * (current_value - prev_value);
    }

    Ok(distance)
}

/// Compute the Mahalanobis distance between samples
///
/// The Mahalanobis distance accounts for correlations in the data and is scale-invariant.
/// It requires the covariance matrix or its inverse.
///
/// # Arguments
/// * `X` - Input array of shape (n_samples_X, n_features)
/// * `Y` - Optional input array of shape (n_samples_Y, n_features). If None, compute pairwise distances within X
/// * `VI` - Inverse of the covariance matrix (precision matrix) of shape (n_features, n_features)
///
/// # Returns
/// Distance matrix of shape (n_samples_X, n_samples_Y) or (n_samples_X, n_samples_X) if Y is None
///
/// # Examples
///
/// ```rust
/// use sklears_metrics::pairwise::mahalanobis_distances;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0., 0.], [1., 1.]];
/// let VI = array![[2., 0.], [0., 2.]]; // Inverse covariance matrix
/// let distances = mahalanobis_distances(&X.view(), None, &VI.view()).unwrap();
/// ```
pub fn mahalanobis_distances(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
    VI: &ArrayView2<f64>,
) -> MetricsResult<Array2<f64>> {
    let (n_samples_x, n_features_x) = X.dim();
    let (vi_rows, vi_cols) = VI.dim();

    if n_samples_x == 0 {
        return Err(MetricsError::EmptyInput);
    }

    if vi_rows != vi_cols || vi_rows != n_features_x {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![n_features_x, n_features_x],
            actual: vec![vi_rows, vi_cols],
        });
    }

    let (Y_ref, n_samples_y) = match Y {
        Some(y) => {
            let (n_samples_y, n_features_y) = y.dim();
            if n_features_x != n_features_y {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![n_samples_x, n_features_x],
                    actual: vec![n_samples_y, n_features_y],
                });
            }
            (y.view(), n_samples_y)
        }
        None => (X.view(), n_samples_x),
    };

    let mut distances = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let diff = &X.row(i) - &Y_ref.row(j);

            // Compute (x - y)^T * VI * (x - y)
            let mut mahal_squared: f64 = 0.0;
            for k in 0..n_features_x {
                let mut temp = 0.0;
                for l in 0..n_features_x {
                    temp += VI[[k, l]] * diff[l];
                }
                mahal_squared += diff[k] * temp;
            }

            distances[[i, j]] = mahal_squared.sqrt();
        }
    }

    Ok(distances)
}

/// Compute cosine similarity matrix between samples
///
/// # Arguments
/// * `X` - Input array of shape (n_samples_X, n_features)
/// * `Y` - Optional input array of shape (n_samples_Y, n_features). If None, compute pairwise similarities within X
///
/// # Returns
/// Similarity matrix of shape (n_samples_X, n_samples_Y) or (n_samples_X, n_samples_X) if Y is None
pub fn cosine_similarity(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
) -> MetricsResult<Array2<f64>> {
    let (n_samples_x, n_features_x) = X.dim();

    if n_samples_x == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let (Y_ref, n_samples_y) = match Y {
        Some(y) => {
            let (n_samples_y, n_features_y) = y.dim();
            if n_features_x != n_features_y {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![n_samples_x, n_features_x],
                    actual: vec![n_samples_y, n_features_y],
                });
            }
            (y.view(), n_samples_y)
        }
        None => (X.view(), n_samples_x),
    };

    let mut similarities = Array2::zeros((n_samples_x, n_samples_y));

    // Compute norms for all rows
    let mut x_norms = Array1::zeros(n_samples_x);
    let mut y_norms = Array1::zeros(n_samples_y);

    for i in 0..n_samples_x {
        x_norms[i] = X.row(i).dot(&X.row(i)).sqrt();
    }

    for j in 0..n_samples_y {
        y_norms[j] = Y_ref.row(j).dot(&Y_ref.row(j)).sqrt();
    }

    // Compute cosine similarities
    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            if x_norms[i] == 0.0 || y_norms[j] == 0.0 {
                similarities[[i, j]] = 0.0;
            } else {
                let dot_product = X.row(i).dot(&Y_ref.row(j));
                similarities[[i, j]] = dot_product / (x_norms[i] * y_norms[j]);
            }
        }
    }

    Ok(similarities)
}

/// Linear kernel: K(x, y) = x^T y
fn linear_kernel(X: &ArrayView2<f64>, Y: Option<&ArrayView2<f64>>) -> MetricsResult<Array2<f64>> {
    let (n_samples_x, n_features_x) = X.dim();

    if n_samples_x == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let (Y_ref, n_samples_y) = match Y {
        Some(y) => {
            let (n_samples_y, n_features_y) = y.dim();
            if n_features_x != n_features_y {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![n_samples_x, n_features_x],
                    actual: vec![n_samples_y, n_features_y],
                });
            }
            (y.view(), n_samples_y)
        }
        None => (X.view(), n_samples_x),
    };

    let mut kernel_matrix = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            kernel_matrix[[i, j]] = X.row(i).dot(&Y_ref.row(j));
        }
    }

    Ok(kernel_matrix)
}

/// Polynomial kernel: K(x, y) = (gamma * x^T y + coef0)^degree
fn polynomial_kernel(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
    degree: f64,
    gamma: f64,
    coef0: f64,
) -> MetricsResult<Array2<f64>> {
    let linear_kernel_matrix = linear_kernel(X, Y)?;
    Ok(linear_kernel_matrix.mapv(|val| (gamma * val + coef0).powf(degree)))
}

/// RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
fn rbf_kernel(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
    gamma: f64,
) -> MetricsResult<Array2<f64>> {
    if gamma <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "RBF kernel gamma must be positive".to_string(),
        ));
    }

    let euclidean_dists = euclidean_distances(X, Y)?;
    Ok(euclidean_dists.mapv(|d| (-gamma * d * d).exp()))
}

/// Sigmoid kernel: K(x, y) = tanh(gamma * x^T y + coef0)
fn sigmoid_kernel(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
    gamma: f64,
    coef0: f64,
) -> MetricsResult<Array2<f64>> {
    let linear_kernel_matrix = linear_kernel(X, Y)?;
    Ok(linear_kernel_matrix.mapv(|val| (gamma * val + coef0).tanh()))
}

/// Cosine kernel: K(x, y) = x^T y / (||x|| * ||y||)
fn cosine_kernel(X: &ArrayView2<f64>, Y: Option<&ArrayView2<f64>>) -> MetricsResult<Array2<f64>> {
    let cosine_dists = cosine_distances(X, Y)?;
    Ok(cosine_dists.mapv(|d| 1.0 - d))
}

/// Laplacian kernel: K(x, y) = exp(-gamma * ||x - y||_1)
fn laplacian_kernel(
    X: &ArrayView2<f64>,
    Y: Option<&ArrayView2<f64>>,
    gamma: f64,
) -> MetricsResult<Array2<f64>> {
    if gamma <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "Laplacian kernel gamma must be positive".to_string(),
        ));
    }

    let manhattan_dists = pairwise_distances(X, Y, DistanceMetric::Manhattan)?;
    Ok(manhattan_dists.mapv(|d| (-gamma * d).exp()))
}

/// Chi-squared kernel: K(x, y) = exp(-gamma * chi2(x, y))
fn chi_squared_kernel<'a>(
    X: &ArrayView2<'a, f64>,
    Y: Option<&ArrayView2<'a, f64>>,
    gamma: f64,
) -> MetricsResult<Array2<f64>> {
    if gamma <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "Chi-squared kernel gamma must be positive".to_string(),
        ));
    }

    let Y_ref = Y.unwrap_or(X);
    let n_samples_x = X.nrows();
    let n_samples_y = Y_ref.nrows();

    if X.ncols() != Y_ref.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![X.nrows(), X.ncols()],
            actual: vec![Y_ref.nrows(), Y_ref.ncols()],
        });
    }

    let mut kernel_matrix = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let x_row = X.row(i);
            let y_row = Y_ref.row(j);

            let mut chi2_dist = 0.0;
            for k in 0..x_row.len() {
                let sum = x_row[k] + y_row[k];
                if sum > 0.0 {
                    let diff = x_row[k] - y_row[k];
                    chi2_dist += (diff * diff) / sum;
                }
            }

            kernel_matrix[[i, j]] = (-gamma * chi2_dist).exp();
        }
    }

    Ok(kernel_matrix)
}

/// Intersection kernel: K(x, y) = sum(min(x_i, y_i))
fn intersection_kernel<'a>(
    X: &ArrayView2<'a, f64>,
    Y: Option<&ArrayView2<'a, f64>>,
) -> MetricsResult<Array2<f64>> {
    let Y_ref = Y.unwrap_or(X);
    let n_samples_x = X.nrows();
    let n_samples_y = Y_ref.nrows();

    if X.ncols() != Y_ref.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![X.nrows(), X.ncols()],
            actual: vec![Y_ref.nrows(), Y_ref.ncols()],
        });
    }

    let mut kernel_matrix = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let x_row = X.row(i);
            let y_row = Y_ref.row(j);

            let intersection: f64 = x_row
                .iter()
                .zip(y_row.iter())
                .map(|(&x, &y)| x.min(y))
                .sum();

            kernel_matrix[[i, j]] = intersection;
        }
    }

    Ok(kernel_matrix)
}

/// Hellinger kernel: K(x, y) = sum(sqrt(x_i * y_i))
/// Note: Assumes inputs are probability distributions (non-negative and normalized)
fn hellinger_kernel<'a>(
    X: &ArrayView2<'a, f64>,
    Y: Option<&ArrayView2<'a, f64>>,
) -> MetricsResult<Array2<f64>> {
    let Y_ref = Y.unwrap_or(X);
    let n_samples_x = X.nrows();
    let n_samples_y = Y_ref.nrows();

    if X.ncols() != Y_ref.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![X.nrows(), X.ncols()],
            actual: vec![Y_ref.nrows(), Y_ref.ncols()],
        });
    }

    let mut kernel_matrix = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let x_row = X.row(i);
            let y_row = Y_ref.row(j);

            let hellinger: f64 = x_row
                .iter()
                .zip(y_row.iter())
                .map(|(&x, &y)| {
                    if x >= 0.0 && y >= 0.0 {
                        (x * y).sqrt()
                    } else {
                        0.0 // Handle negative values gracefully
                    }
                })
                .sum();

            kernel_matrix[[i, j]] = hellinger;
        }
    }

    Ok(kernel_matrix)
}

/// Jensen-Shannon kernel: K(x, y) = 1 - JS_divergence(x, y)
/// Jensen-Shannon divergence is a symmetric measure based on KL divergence
fn jensen_shannon_kernel<'a>(
    X: &ArrayView2<'a, f64>,
    Y: Option<&ArrayView2<'a, f64>>,
) -> MetricsResult<Array2<f64>> {
    let Y_ref = Y.unwrap_or(X);
    let n_samples_x = X.nrows();
    let n_samples_y = Y_ref.nrows();

    if X.ncols() != Y_ref.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![X.nrows(), X.ncols()],
            actual: vec![Y_ref.nrows(), Y_ref.ncols()],
        });
    }

    let mut kernel_matrix = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let x_row = X.row(i);
            let y_row = Y_ref.row(j);

            // Normalize to probability distributions
            let x_sum: f64 = x_row.iter().sum();
            let y_sum: f64 = y_row.iter().sum();

            if x_sum <= 0.0 || y_sum <= 0.0 {
                kernel_matrix[[i, j]] = 0.0;
                continue;
            }

            let x_normalized: Vec<f64> = x_row.iter().map(|&val| val / x_sum).collect();
            let y_normalized: Vec<f64> = y_row.iter().map(|&val| val / y_sum).collect();

            // Compute Jensen-Shannon divergence
            let mut js_div = 0.0;
            for k in 0..x_normalized.len() {
                let x_prob = x_normalized[k];
                let y_prob = y_normalized[k];
                let m = (x_prob + y_prob) / 2.0;

                if x_prob > 0.0 && m > 0.0 {
                    js_div += 0.5 * x_prob * (x_prob / m).ln();
                }
                if y_prob > 0.0 && m > 0.0 {
                    js_div += 0.5 * y_prob * (y_prob / m).ln();
                }
            }

            // Convert to kernel (similarity): K = 1 - JS_div
            kernel_matrix[[i, j]] = 1.0 - js_div;
        }
    }

    Ok(kernel_matrix)
}

/// Additive chi-squared kernel: K(x, y) = sum(2 * x_i * y_i / (x_i + y_i))
fn additive_chi_squared_kernel<'a>(
    X: &ArrayView2<'a, f64>,
    Y: Option<&ArrayView2<'a, f64>>,
) -> MetricsResult<Array2<f64>> {
    let Y_ref = Y.unwrap_or(X);
    let n_samples_x = X.nrows();
    let n_samples_y = Y_ref.nrows();

    if X.ncols() != Y_ref.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![X.nrows(), X.ncols()],
            actual: vec![Y_ref.nrows(), Y_ref.ncols()],
        });
    }

    let mut kernel_matrix = Array2::zeros((n_samples_x, n_samples_y));

    for i in 0..n_samples_x {
        for j in 0..n_samples_y {
            let x_row = X.row(i);
            let y_row = Y_ref.row(j);

            let additive_chi2: f64 = x_row
                .iter()
                .zip(y_row.iter())
                .map(|(&x, &y)| {
                    let sum = x + y;
                    if sum > 0.0 {
                        2.0 * x * y / sum
                    } else {
                        0.0
                    }
                })
                .sum();

            kernel_matrix[[i, j]] = additive_chi2;
        }
    }

    Ok(kernel_matrix)
}

/// Kernel-based similarity for string sequences using subsequence kernels
///
/// Computes similarity between sequences based on common subsequences.
/// This is a simplified implementation - for production use, consider more
/// sophisticated string kernels like spectrum kernels or gap-weighted kernels.
///
/// # Arguments
/// * `seq1` - First sequence
/// * `seq2` - Second sequence
/// * `k` - Length of subsequences to consider
///
/// # Returns
/// Similarity score based on common k-mer subsequences
pub fn string_kernel_similarity(seq1: &[u8], seq2: &[u8], k: usize) -> MetricsResult<f64> {
    if k == 0 {
        return Err(MetricsError::InvalidParameter(
            "Subsequence length k must be greater than 0".to_string(),
        ));
    }

    if seq1.len() < k || seq2.len() < k {
        return Ok(0.0);
    }

    // Extract k-mers from both sequences
    let mut kmers1: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut kmers2: HashMap<Vec<u8>, usize> = HashMap::new();

    for i in 0..=(seq1.len() - k) {
        let kmer = seq1[i..i + k].to_vec();
        *kmers1.entry(kmer).or_insert(0) += 1;
    }

    for i in 0..=(seq2.len() - k) {
        let kmer = seq2[i..i + k].to_vec();
        *kmers2.entry(kmer).or_insert(0) += 1;
    }

    // Compute inner product of k-mer count vectors
    let mut similarity = 0.0;
    for (kmer, count1) in kmers1.iter() {
        if let Some(count2) = kmers2.get(kmer) {
            similarity += (*count1 as f64) * (*count2 as f64);
        }
    }

    // Normalize by geometric mean of vector lengths
    let norm1: f64 = kmers1.values().map(|&c| (c * c) as f64).sum::<f64>().sqrt();
    let norm2: f64 = kmers2.values().map(|&c| (c * c) as f64).sum::<f64>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        Ok(0.0)
    } else {
        Ok(similarity / (norm1 * norm2))
    }
}

/// Matrix of string kernel similarities
///
/// Computes pairwise string kernel similarities between all sequence pairs.
///
/// # Arguments
/// * `sequences` - Vector of byte sequences
/// * `k` - Length of subsequences to consider
///
/// # Returns
/// Symmetric similarity matrix
pub fn string_kernel_matrix(sequences: &[&[u8]], k: usize) -> MetricsResult<Array2<f64>> {
    let n = sequences.len();

    if n == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let mut similarity_matrix = Array2::zeros((n, n));

    for i in 0..n {
        for j in i..n {
            if i == j {
                similarity_matrix[[i, j]] = 1.0;
            } else {
                let sim = string_kernel_similarity(sequences[i], sequences[j], k)?;
                similarity_matrix[[i, j]] = sim;
                similarity_matrix[[j, i]] = sim; // Symmetric
            }
        }
    }

    Ok(similarity_matrix)
}

/// Normalized Compression Distance (NCD)
///
/// A universal metric based on compression algorithms that approximates
/// the normalized information distance. It measures the similarity between
/// two byte sequences by comparing their compressed sizes.
///
/// Formula: NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
///
/// Where:
/// - C(x) is the compressed size of x
/// - C(y) is the compressed size of y  
/// - C(xy) is the compressed size of the concatenation of x and y
///
/// # Arguments
///
/// * `x` - First data sequence
/// * `y` - Second data sequence
///
/// # Returns
///
/// NCD value between 0 and 1 (lower values indicate higher similarity)
///
/// # Examples
///
/// ```
/// use sklears_metrics::pairwise::normalized_compression_distance;
///
/// let x = b"hello world hello";
/// let y = b"hello world world";
/// let ncd = normalized_compression_distance(x, y).unwrap();
/// println!("NCD: {:.3}", ncd);
/// ```
pub fn normalized_compression_distance(x: &[u8], y: &[u8]) -> MetricsResult<f64> {
    use flate2::{write::GzEncoder, Compression};
    use std::io::Write;

    if x.is_empty() && y.is_empty() {
        return Ok(0.0);
    }

    if x.is_empty() || y.is_empty() {
        return Ok(1.0);
    }

    // Helper function to compress data and return compressed size
    let compress = |data: &[u8]| -> Result<usize, std::io::Error> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        Ok(compressed.len())
    };

    // Get compressed sizes
    let c_x = compress(x)
        .map_err(|e| MetricsError::InvalidInput(format!("Compression failed for x: {}", e)))?;
    let c_y = compress(y)
        .map_err(|e| MetricsError::InvalidInput(format!("Compression failed for y: {}", e)))?;

    // Concatenate x and y and compress
    let mut xy = Vec::with_capacity(x.len() + y.len());
    xy.extend_from_slice(x);
    xy.extend_from_slice(y);
    let c_xy = compress(&xy)
        .map_err(|e| MetricsError::InvalidInput(format!("Compression failed for xy: {}", e)))?;

    let min_c = c_x.min(c_y) as f64;
    let max_c = c_x.max(c_y) as f64;

    if max_c == 0.0 {
        return Ok(0.0);
    }

    let ncd = (c_xy as f64 - min_c) / max_c;

    // Clamp to [0, 1] range (theoretical bounds, though may be exceeded due to compression artifacts)
    Ok(ncd.clamp(0.0, 1.0))
}

/// Normalized Compression Distance Matrix
///
/// Computes the NCD between all pairs of byte sequences.
///
/// # Arguments
///
/// * `sequences` - Vector of byte sequences
///
/// # Returns
///
/// Symmetric distance matrix where entry (i,j) is NCD(sequences\[i\], sequences\[j\])
pub fn normalized_compression_distance_matrix(sequences: &[&[u8]]) -> MetricsResult<Array2<f64>> {
    let n = sequences.len();

    if n == 0 {
        return Err(MetricsError::EmptyInput);
    }

    let mut distance_matrix = Array2::zeros((n, n));

    for i in 0..n {
        for j in i..n {
            if i == j {
                distance_matrix[[i, j]] = 0.0;
            } else {
                let ncd = normalized_compression_distance(sequences[i], sequences[j])?;
                distance_matrix[[i, j]] = ncd;
                distance_matrix[[j, i]] = ncd; // Symmetric
            }
        }
    }

    Ok(distance_matrix)
}

/// Approximate Kolmogorov Complexity
///
/// Estimates the Kolmogorov complexity of a byte sequence using compression.
/// This is an approximation since true Kolmogorov complexity is uncomputable.
///
/// # Arguments
///
/// * `data` - Input byte sequence
///
/// # Returns
///
/// Approximate Kolmogorov complexity (compressed size in bytes)
pub fn approximate_kolmogorov_complexity(data: &[u8]) -> MetricsResult<usize> {
    use flate2::{write::GzEncoder, Compression};
    use std::io::Write;

    if data.is_empty() {
        return Ok(0);
    }

    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(data)
        .map_err(|e| MetricsError::InvalidInput(format!("Compression failed: {}", e)))?;

    let compressed = encoder
        .finish()
        .map_err(|e| MetricsError::InvalidInput(format!("Compression failed: {}", e)))?;

    Ok(compressed.len())
}

/// Information Distance
///
/// Computes the information distance between two byte sequences.
/// Formula: max(K(x|y), K(y|x)) where K(x|y) is conditional Kolmogorov complexity.
/// Approximated using compression.
///
/// # Arguments
///
/// * `x` - First data sequence
/// * `y` - Second data sequence
///
/// # Returns
///
/// Information distance (in bytes)
pub fn information_distance(x: &[u8], y: &[u8]) -> MetricsResult<usize> {
    if x.is_empty() && y.is_empty() {
        return Ok(0);
    }

    // K(x|y) ≈ C(xy) - C(y)
    let k_x = approximate_kolmogorov_complexity(x)?;
    let k_y = approximate_kolmogorov_complexity(y)?;

    // Concatenate sequences
    let mut xy = Vec::with_capacity(x.len() + y.len());
    xy.extend_from_slice(x);
    xy.extend_from_slice(y);
    let k_xy = approximate_kolmogorov_complexity(&xy)?;

    let mut yx = Vec::with_capacity(x.len() + y.len());
    yx.extend_from_slice(y);
    yx.extend_from_slice(x);
    let k_yx = approximate_kolmogorov_complexity(&yx)?;

    // K(x|y) ≈ C(xy) - C(y), K(y|x) ≈ C(yx) - C(x)
    let k_x_given_y = k_xy.saturating_sub(k_y);
    let k_y_given_x = k_yx.saturating_sub(k_x);

    Ok(k_x_given_y.max(k_y_given_x))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_euclidean_distances() {
        let X = array![[0., 0.], [3., 4.], [1., 1.]];
        let Y = array![[1., 0.], [0., 1.]];

        let distances = euclidean_distances(&X.view(), Some(&Y.view())).unwrap();

        assert_eq!(distances.dim(), (3, 2));
        assert!((distances[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((distances[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((distances[[1, 0]] - 4.47213595499958).abs() < 1e-10);
        assert!((distances[[1, 1]] - 4.242640687119285).abs() < 1e-10);
        assert!((distances[[2, 0]] - 1.0).abs() < 1e-10);
        assert!((distances[[2, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distances_self() {
        let X = array![[0., 0.], [1., 1.], [2., 2.]];

        let distances = euclidean_distances(&X.view(), None).unwrap();

        assert_eq!(distances.dim(), (3, 3));
        // Diagonal should be zero
        for i in 0..3 {
            assert!((distances[[i, i]]).abs() < 1e-10);
        }
        // Should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((distances[[i, j]] - distances[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_nan_euclidean_distances() {
        let X = array![[1., 2., f64::NAN], [4., f64::NAN, 6.]];
        let Y = array![[1., 2., 3.]];

        let distances = nan_euclidean_distances(&X.view(), Some(&Y.view())).unwrap();

        assert_eq!(distances.dim(), (2, 1));
        // First distance: only first two features are valid
        // sqrt((0^2 + 0^2) * 3/2) = 0
        assert!((distances[[0, 0]]).abs() < 1e-10);
        // Second distance: only first and third features are valid
        // sqrt((3^2 + 3^2) * 3/2) = sqrt(18 * 1.5) = sqrt(27)
        assert!((distances[[1, 0]] - 27.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distances() {
        let X = array![[0., 0.], [1., 1.], [2., 2.]];
        let Y = array![[1., 0.], [0., 1.]];

        let distances =
            pairwise_distances(&X.view(), Some(&Y.view()), DistanceMetric::Manhattan).unwrap();

        assert_eq!(distances.dim(), (3, 2));
        assert!((distances[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((distances[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((distances[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((distances[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((distances[[2, 0]] - 3.0).abs() < 1e-10);
        assert!((distances[[2, 1]] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_distances() {
        let X = array![[0., 0.], [1., 1.], [3., 4.]];
        let Y = array![[1., 0.], [0., 1.]];

        let distances =
            pairwise_distances(&X.view(), Some(&Y.view()), DistanceMetric::Chebyshev).unwrap();

        assert_eq!(distances.dim(), (3, 2));
        assert!((distances[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((distances[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((distances[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((distances[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((distances[[2, 0]] - 4.0).abs() < 1e-10);
        assert!((distances[[2, 1]] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distances() {
        let X = array![[1., 0.], [0., 1.], [1., 1.]];

        let distances = pairwise_distances(&X.view(), None, DistanceMetric::Cosine).unwrap();

        assert_eq!(distances.dim(), (3, 3));
        // Diagonal should be zero (perfect similarity with self)
        for i in 0..3 {
            assert!((distances[[i, i]]).abs() < 1e-10);
        }
        // Orthogonal vectors should have distance 1
        assert!((distances[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((distances[[1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_distances_argmin() {
        let X = array![[0., 0.], [5., 5.]];
        let Y = array![[1., 1.], [4., 4.], [2., 2.]];

        let argmin =
            pairwise_distances_argmin(&X.view(), Some(&Y.view()), DistanceMetric::Euclidean)
                .unwrap();

        assert_eq!(argmin.dim(), 2);
        assert_eq!(argmin[0], 0); // [0, 0] is closest to [1, 1]
        assert_eq!(argmin[1], 1); // [5, 5] is closest to [4, 4]
    }

    #[test]
    fn test_pairwise_distances_argmin_min() {
        let X = array![[0., 0.], [5., 5.]];
        let Y = array![[1., 1.], [4., 4.], [2., 2.]];

        let (argmin, min_dists) =
            pairwise_distances_argmin_min(&X.view(), Some(&Y.view()), DistanceMetric::Euclidean)
                .unwrap();

        assert_eq!(argmin.dim(), 2);
        assert_eq!(min_dists.dim(), 2);
        assert_eq!(argmin[0], 0);
        assert_eq!(argmin[1], 1);
        assert!((min_dists[0] - 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((min_dists[1] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_linear_kernel() {
        let X = array![[1., 2.], [3., 4.]];
        let Y = array![[5., 6.], [7., 8.]];

        let kernel_matrix =
            pairwise_kernels(&X.view(), Some(&Y.view()), KernelFunction::Linear).unwrap();

        assert_eq!(kernel_matrix.dim(), (2, 2));
        assert!((kernel_matrix[[0, 0]] - 17.0).abs() < 1e-10); // 1*5 + 2*6
        assert!((kernel_matrix[[0, 1]] - 23.0).abs() < 1e-10); // 1*7 + 2*8
        assert!((kernel_matrix[[1, 0]] - 39.0).abs() < 1e-10); // 3*5 + 4*6
        assert!((kernel_matrix[[1, 1]] - 53.0).abs() < 1e-10); // 3*7 + 4*8
    }

    #[test]
    fn test_rbf_kernel() {
        let X = array![[0., 0.], [1., 1.]];

        let kernel_matrix =
            pairwise_kernels(&X.view(), None, KernelFunction::RBF { gamma: 0.5 }).unwrap();

        assert_eq!(kernel_matrix.dim(), (2, 2));
        // Diagonal should be 1 (identical points)
        assert!((kernel_matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((kernel_matrix[[1, 1]] - 1.0).abs() < 1e-10);
        // Off-diagonal should be exp(-0.5 * 2) = exp(-1)
        assert!((kernel_matrix[[0, 1]] - (-1.0_f64).exp()).abs() < 1e-10);
        assert!((kernel_matrix[[1, 0]] - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_kernel() {
        let X = array![[1., 0.], [0., 1.]];

        let kernel_matrix = pairwise_kernels(
            &X.view(),
            None,
            KernelFunction::Polynomial {
                degree: 2.0,
                gamma: 1.0,
                coef0: 1.0,
            },
        )
        .unwrap();

        assert_eq!(kernel_matrix.dim(), (2, 2));
        // Diagonal: (1*1 + 1)^2 = 4
        assert!((kernel_matrix[[0, 0]] - 4.0).abs() < 1e-10);
        assert!((kernel_matrix[[1, 1]] - 4.0).abs() < 1e-10);
        // Off-diagonal: (0 + 1)^2 = 1
        assert!((kernel_matrix[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((kernel_matrix[[1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape_mismatch() {
        let X = array![[1., 2., 3.]];
        let Y = array![[1., 2.]];

        let result = euclidean_distances(&X.view(), Some(&Y.view()));
        assert!(result.is_err());
        match result {
            Err(MetricsError::ShapeMismatch { .. }) => {}
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_empty_input() {
        let X = array![[]].into_shape((0, 2)).unwrap();

        let result = euclidean_distances(&X.view(), None);
        assert!(result.is_err());
        match result {
            Err(MetricsError::EmptyInput) => {}
            _ => panic!("Expected EmptyInput error"),
        }
    }
}

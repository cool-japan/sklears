//! SIMD utility functions for execution engine
//!
//! This module provides fallback implementations for SIMD operations
//! used in the execution engine for high-performance metrics calculations.

/// Compute mean of a vector of f32 values
#[must_use]
pub fn mean_vec(vec: &[f32]) -> f32 {
    if vec.is_empty() {
        0.0
    } else {
        vec.iter().sum::<f32>() / vec.len() as f32
    }
}

/// Compute variance of a vector of f32 values
#[must_use]
pub fn variance_vec(vec: &[f32]) -> f32 {
    if vec.len() <= 1 {
        return 0.0;
    }
    let mean = mean_vec(vec);
    vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vec.len() as f32
}

/// Compute sum of a vector of f32 values
#[must_use]
pub fn sum_vec(vec: &[f32]) -> f32 {
    vec.iter().sum()
}

/// Add two vectors element-wise
pub fn add_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

/// Multiply two vectors element-wise
pub fn multiply_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    for i in 0..a.len() {
        result[i] = a[i] * b[i];
    }
}

/// Divide two vectors element-wise
pub fn divide_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    for i in 0..a.len() {
        result[i] = a[i] / b[i];
    }
}

/// Compute dot product of two vectors
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Find minimum and maximum values in a vector
#[must_use]
pub fn min_max_vec(vec: &[f32]) -> (f32, f32) {
    if vec.is_empty() {
        return (0.0, 0.0);
    }

    let mut min_val = vec[0];
    let mut max_val = vec[0];

    for &val in vec.iter().skip(1) {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    (min_val, max_val)
}

/// In-place division of target vector by divisor vector
pub fn divide_vec_inplace(target: &mut [f32], divisor: &[f32]) {
    assert_eq!(target.len(), divisor.len());

    for i in 0..target.len() {
        target[i] /= divisor[i];
    }
}

/// In-place multiplication of target vector by multiplier vector
pub fn multiply_vec_inplace(target: &mut [f32], multiplier: &[f32]) {
    assert_eq!(target.len(), multiplier.len());

    for i in 0..target.len() {
        target[i] *= multiplier[i];
    }
}

/// Scale a vector by a scalar value
pub fn scale_vec(vec: &[f32], scale: f32, result: &mut [f32]) {
    assert_eq!(vec.len(), result.len());

    for i in 0..vec.len() {
        result[i] = vec[i] * scale;
    }
}

/// Normalize a vector to unit length
pub fn normalize_vec(vec: &mut [f32]) {
    let norm = dot_product(vec, vec).sqrt();
    if norm > 0.0 {
        for val in vec {
            *val /= norm;
        }
    }
}

/// Compute L2 norm (Euclidean norm) of a vector
#[must_use]
pub fn l2_norm(vec: &[f32]) -> f32 {
    dot_product(vec, vec).sqrt()
}

/// Compute L1 norm (Manhattan norm) of a vector
#[must_use]
pub fn l1_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|&x| x.abs()).sum()
}

/// Element-wise absolute value
pub fn abs_vec(vec: &[f32], result: &mut [f32]) {
    assert_eq!(vec.len(), result.len());

    for i in 0..vec.len() {
        result[i] = vec[i].abs();
    }
}

/// Element-wise square root
pub fn sqrt_vec(vec: &[f32], result: &mut [f32]) {
    assert_eq!(vec.len(), result.len());

    for i in 0..vec.len() {
        result[i] = vec[i].sqrt();
    }
}

/// Element-wise exponential
pub fn exp_vec(vec: &[f32], result: &mut [f32]) {
    assert_eq!(vec.len(), result.len());

    for i in 0..vec.len() {
        result[i] = vec[i].exp();
    }
}

/// Element-wise natural logarithm
pub fn log_vec(vec: &[f32], result: &mut [f32]) {
    assert_eq!(vec.len(), result.len());

    for i in 0..vec.len() {
        result[i] = vec[i].ln();
    }
}

/// Compute Euclidean distance between two vectors
#[must_use]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute cosine similarity between two vectors
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

//! Parallel Processing for High-Performance Metric Computation
//!
//! This module provides parallel implementations of common metrics using Rayon for
//! multi-threaded computation. These implementations automatically distribute work
//! across available CPU cores for significant performance improvements on large datasets.
//!
//! ## Key Features
//!
//! - **Rayon Integration**: Leverages Rayon's work-stealing scheduler for optimal load balancing
//! - **Configurable Thresholds**: Automatically falls back to serial computation for small arrays
//! - **Memory Efficiency**: Chunked processing to optimize cache usage and memory bandwidth
//! - **Multiple Metrics**: Parallel implementations for MAE, MSE, R², cosine similarity, and accuracy

#[cfg(feature = "parallel")]
use super::OptimizedConfig;
#[cfg(feature = "parallel")]
use crate::{MetricsError, MetricsResult};
#[cfg(feature = "parallel")]
use scirs2_core::ndarray::{Array1, Axis};
#[cfg(feature = "parallel")]
use scirs2_core::numeric::{Float as FloatTrait, FromPrimitive};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Parallel mean absolute error using rayon
#[cfg(feature = "parallel")]
pub fn parallel_mean_absolute_error<
    F: FloatTrait
        + FromPrimitive
        + Send
        + Sync
        + std::iter::Sum<F>
        + for<'a> std::iter::Sum<&'a F>
        + 'static,
>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    config: &OptimizedConfig,
) -> MetricsResult<F> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let len = y_true.len();

    if len < config.parallel_threshold {
        // Use serial computation for small arrays - only for f64 arrays
        if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
            unsafe {
                let y_true_f64 = std::mem::transmute(y_true);
                let y_pred_f64 = std::mem::transmute(y_pred);
                let result: f64 = crate::regression::mean_absolute_error(y_true_f64, y_pred_f64)?;
                return Ok(std::mem::transmute_copy(&result));
            }
        } else {
            return Err(MetricsError::InvalidInput(
                "Optimized serial fallback only supports f64 arrays".to_string(),
            ));
        }
    }

    let sum = y_true
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(y_pred.axis_iter(Axis(0)))
        .with_min_len(config.chunk_size)
        .map(|(t, p)| {
            let t_val = t.iter().next().unwrap();
            let p_val = p.iter().next().unwrap();
            (*t_val - *p_val).abs()
        })
        .sum::<F>();

    Ok(sum / F::from(len).unwrap())
}

/// Parallel mean squared error using rayon
#[cfg(feature = "parallel")]
pub fn parallel_mean_squared_error<
    F: FloatTrait
        + FromPrimitive
        + Send
        + Sync
        + std::iter::Sum<F>
        + for<'a> std::iter::Sum<&'a F>
        + 'static,
>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    config: &OptimizedConfig,
) -> MetricsResult<F> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let len = y_true.len();

    if len < config.parallel_threshold {
        // Use serial computation for small arrays - only for f64 arrays
        if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
            unsafe {
                let y_true_f64 = std::mem::transmute(y_true);
                let y_pred_f64 = std::mem::transmute(y_pred);
                let result: f64 = crate::regression::mean_squared_error(y_true_f64, y_pred_f64)?;
                return Ok(std::mem::transmute_copy(&result));
            }
        } else {
            return Err(MetricsError::InvalidInput(
                "Optimized serial fallback only supports f64 arrays".to_string(),
            ));
        }
    }

    let sum = y_true
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(y_pred.axis_iter(Axis(0)))
        .with_min_len(config.chunk_size)
        .map(|(t, p)| {
            let t_val = t.iter().next().unwrap();
            let p_val = p.iter().next().unwrap();
            let diff = *t_val - *p_val;
            diff * diff
        })
        .sum::<F>();

    Ok(sum / F::from(len).unwrap())
}

/// Parallel R² score using rayon
#[cfg(feature = "parallel")]
pub fn parallel_r2_score<
    F: FloatTrait
        + FromPrimitive
        + Send
        + Sync
        + std::iter::Sum<F>
        + for<'a> std::iter::Sum<&'a F>
        + 'static,
>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    config: &OptimizedConfig,
) -> MetricsResult<F> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let len = y_true.len();

    if len < config.parallel_threshold {
        // Use serial computation for small arrays - only for f64 arrays
        if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
            unsafe {
                let y_true_f64 = std::mem::transmute(y_true);
                let y_pred_f64 = std::mem::transmute(y_pred);
                let result: f64 = crate::regression::r2_score(y_true_f64, y_pred_f64)?;
                return Ok(std::mem::transmute_copy(&result));
            }
        } else {
            return Err(MetricsError::InvalidInput(
                "Optimized serial fallback only supports f64 arrays".to_string(),
            ));
        }
    }

    // Calculate sums in parallel
    let (sum_squared_residuals, sum_true, sum_true_squared) = y_true
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(y_pred.axis_iter(Axis(0)))
        .with_min_len(config.chunk_size)
        .map(|(t, p)| {
            let t_val = t.iter().next().unwrap();
            let p_val = p.iter().next().unwrap();
            let diff = *t_val - *p_val;
            let ssr = diff * diff;
            let st = *t_val;
            let st_sq = st * st;
            (ssr, st, st_sq)
        })
        .reduce(
            || (F::zero(), F::zero(), F::zero()),
            |(ssr1, st1, st_sq1), (ssr2, st2, st_sq2)| (ssr1 + ssr2, st1 + st2, st_sq1 + st_sq2),
        );

    // Calculate R² score
    let mean_true = sum_true / F::from(len).unwrap();
    let total_sum_squares = sum_true_squared - F::from(len).unwrap() * mean_true * mean_true;

    if total_sum_squares == F::zero() {
        return Ok(F::zero()); // Perfect prediction when all true values are the same
    }

    Ok(F::one() - (sum_squared_residuals / total_sum_squares))
}

/// Parallel cosine similarity computation
#[cfg(feature = "parallel")]
pub fn parallel_cosine_similarity<
    F: FloatTrait + FromPrimitive + Send + Sync + std::iter::Sum<F> + for<'a> std::iter::Sum<&'a F>,
>(
    a: &Array1<F>,
    b: &Array1<F>,
    config: &OptimizedConfig,
) -> MetricsResult<F> {
    if a.len() != b.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }

    if a.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let len = a.len();

    if len < config.parallel_threshold {
        // Direct cosine similarity calculation for small arrays
        let dot_product = a
            .iter()
            .zip(b.iter())
            .map(|(a_val, b_val)| *a_val * *b_val)
            .sum::<F>();
        let norm_a = a.iter().map(|val| *val * *val).sum::<F>().sqrt();
        let norm_b = b.iter().map(|val| *val * *val).sum::<F>().sqrt();

        if norm_a == F::zero() || norm_b == F::zero() {
            return Err(MetricsError::DivisionByZero);
        }

        return Ok(dot_product / (norm_a * norm_b));
    }

    // Calculate dot product and norms in parallel
    let (dot_product, norm_a_sq, norm_b_sq) = a
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(b.axis_iter(Axis(0)))
        .with_min_len(config.chunk_size)
        .map(|(a_val, b_val)| {
            let a_v = a_val.iter().next().unwrap();
            let b_v = b_val.iter().next().unwrap();
            let dot = *a_v * *b_v;
            let norm_a = *a_v * *a_v;
            let norm_b = *b_v * *b_v;
            (dot, norm_a, norm_b)
        })
        .reduce(
            || (F::zero(), F::zero(), F::zero()),
            |(dot1, na1, nb1), (dot2, na2, nb2)| (dot1 + dot2, na1 + na2, nb1 + nb2),
        );

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a == F::zero() || norm_b == F::zero() {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(dot_product / (norm_a * norm_b))
}

/// Parallel accuracy computation for classification
#[cfg(feature = "parallel")]
pub fn parallel_accuracy<T: PartialEq + Copy + Send + Sync>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
    config: &OptimizedConfig,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let len = y_true.len();

    if len < config.parallel_threshold {
        return crate::classification::accuracy_score(y_true, y_pred);
    }

    let correct_count = y_true
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(y_pred.axis_iter(Axis(0)))
        .with_min_len(config.chunk_size)
        .map(|(t, p)| {
            let t_val = t.iter().next().unwrap();
            let p_val = p.iter().next().unwrap();
            if t_val == p_val {
                1usize
            } else {
                0usize
            }
        })
        .sum::<usize>();

    Ok(correct_count as f64 / len as f64)
}

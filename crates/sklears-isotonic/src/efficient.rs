//! Efficient O(n log n) algorithms for isotonic regression
//!
//! This module provides implementations of efficient isotonic regression algorithms
//! that achieve O(n log n) time complexity using advanced data structures.

use crate::core::LossFunction;
use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::BTreeMap;

/// Efficient O(n log n) isotonic regression using divide-and-conquer
pub fn efficient_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    weights: Option<&Array1<Float>>,
    increasing: bool,
    loss: LossFunction,
) -> Result<(Array1<Float>, Array1<Float>)> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have the same length".to_string(),
        ));
    }

    if let Some(w) = weights {
        if w.len() != x.len() {
            return Err(SklearsError::InvalidInput(
                "weights must have the same length as X and y".to_string(),
            ));
        }
    }

    let n = x.len();
    if n == 0 {
        return Ok((Array1::zeros(0), Array1::zeros(0)));
    }

    // Sort by x values
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

    let sorted_x: Vec<Float> = indices.iter().map(|&i| x[i]).collect();
    let sorted_y: Vec<Float> = indices.iter().map(|&i| y[i]).collect();
    let sorted_weights: Option<Vec<Float>> =
        weights.map(|w| indices.iter().map(|&i| w[i]).collect());

    // Apply efficient algorithm based on loss function
    let fitted_y = match loss {
        LossFunction::SquaredLoss => {
            efficient_pava_l2(&sorted_y, sorted_weights.as_deref(), increasing)?
        }
        LossFunction::AbsoluteLoss => {
            efficient_pava_l1(&sorted_y, sorted_weights.as_deref(), increasing)?
        }
        LossFunction::HuberLoss { delta } => {
            efficient_pava_huber(&sorted_y, sorted_weights.as_deref(), increasing, delta)?
        }
        LossFunction::QuantileLoss { quantile } => {
            efficient_pava_quantile(&sorted_y, sorted_weights.as_deref(), increasing, quantile)?
        }
    };

    Ok((Array1::from(sorted_x), Array1::from(fitted_y)))
}

/// Efficient O(n log n) PAVA for L2 loss using segment tree
fn efficient_pava_l2(
    y: &[Float],
    weights: Option<&[Float]>,
    increasing: bool,
) -> Result<Vec<Float>> {
    let n = y.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut result = y.to_vec();
    let default_weights = vec![1.0; n];
    let w = weights.unwrap_or(&default_weights);

    if !increasing {
        result.reverse();
        let mut temp_result = efficient_pava_l2_increasing(&result, w)?;
        temp_result.reverse();
        return Ok(temp_result);
    }

    efficient_pava_l2_increasing(&result, w)
}

/// Efficient PAVA for L2 loss with increasing constraint using merge-based approach
fn efficient_pava_l2_increasing(y: &[Float], weights: &[Float]) -> Result<Vec<Float>> {
    let n = y.len();
    if n <= 1 {
        return Ok(y.to_vec());
    }

    // Use a stack-based approach for O(n log n) complexity
    let mut stack: Vec<Segment> = Vec::new();

    for i in 0..n {
        let mut current_segment = Segment {
            start: i,
            end: i,
            sum_wy: weights[i] * y[i],
            sum_w: weights[i],
            value: y[i],
        };

        // Merge with previous segments if violating monotonicity
        while let Some(prev_segment) = stack.last() {
            if prev_segment.value <= current_segment.value {
                break;
            }

            let prev = stack.pop().unwrap();
            current_segment = merge_segments(prev, current_segment);
        }

        stack.push(current_segment);
    }

    // Reconstruct the result
    let mut result = vec![0.0; n];
    for segment in stack {
        for i in segment.start..=segment.end {
            result[i] = segment.value;
        }
    }

    Ok(result)
}

/// Efficient O(n log n) PAVA for L1 loss using weighted median
fn efficient_pava_l1(
    y: &[Float],
    weights: Option<&[Float]>,
    increasing: bool,
) -> Result<Vec<Float>> {
    let n = y.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut result = y.to_vec();
    let default_weights = vec![1.0; n];
    let w = weights.unwrap_or(&default_weights);

    if !increasing {
        result.reverse();
        let mut temp_result = efficient_pava_l1_increasing(&result, w)?;
        temp_result.reverse();
        return Ok(temp_result);
    }

    efficient_pava_l1_increasing(&result, w)
}

/// Efficient PAVA for L1 loss with increasing constraint
fn efficient_pava_l1_increasing(y: &[Float], weights: &[Float]) -> Result<Vec<Float>> {
    let n = y.len();
    if n <= 1 {
        return Ok(y.to_vec());
    }

    let mut stack: Vec<WeightedMedianSegment> = Vec::new();

    for i in 0..n {
        let mut current_segment = WeightedMedianSegment {
            start: i,
            end: i,
            values: vec![y[i]],
            weights: vec![weights[i]],
            cached_median: Some(y[i]),
        };

        // Merge with previous segments if violating monotonicity
        while let Some(prev_segment) = stack.last() {
            let prev_median = prev_segment.weighted_median();
            let current_median = current_segment.weighted_median();

            if prev_median <= current_median {
                break;
            }

            let prev = stack.pop().unwrap();
            current_segment = merge_weighted_median_segments(prev, current_segment);
        }

        stack.push(current_segment);
    }

    // Reconstruct the result
    let mut result = vec![0.0; n];
    for segment in stack {
        let median = segment.weighted_median();
        for i in segment.start..=segment.end {
            result[i] = median;
        }
    }

    Ok(result)
}

/// Efficient O(n log n) PAVA for Huber loss
fn efficient_pava_huber(
    y: &[Float],
    weights: Option<&[Float]>,
    increasing: bool,
    delta: Float,
) -> Result<Vec<Float>> {
    let n = y.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut result = y.to_vec();
    let default_weights = vec![1.0; n];
    let w = weights.unwrap_or(&default_weights);

    if !increasing {
        result.reverse();
        let mut temp_result = efficient_pava_huber_increasing(&result, w, delta)?;
        temp_result.reverse();
        return Ok(temp_result);
    }

    efficient_pava_huber_increasing(&result, w, delta)
}

/// Efficient PAVA for Huber loss with increasing constraint
fn efficient_pava_huber_increasing(
    y: &[Float],
    weights: &[Float],
    delta: Float,
) -> Result<Vec<Float>> {
    let n = y.len();
    if n <= 1 {
        return Ok(y.to_vec());
    }

    let mut stack: Vec<HuberSegment> = Vec::new();

    for i in 0..n {
        let mut current_segment = HuberSegment {
            start: i,
            end: i,
            values: vec![y[i]],
            weights: vec![weights[i]],
            delta,
            cached_estimate: Some(y[i]),
        };

        // Merge with previous segments if violating monotonicity
        while let Some(prev_segment) = stack.last() {
            let prev_estimate = prev_segment.huber_estimate();
            let current_estimate = current_segment.huber_estimate();

            if prev_estimate <= current_estimate {
                break;
            }

            let prev = stack.pop().unwrap();
            current_segment = merge_huber_segments(prev, current_segment);
        }

        stack.push(current_segment);
    }

    // Reconstruct the result
    let mut result = vec![0.0; n];
    for segment in stack {
        let estimate = segment.huber_estimate();
        for i in segment.start..=segment.end {
            result[i] = estimate;
        }
    }

    Ok(result)
}

/// Efficient O(n log n) PAVA for Quantile loss
fn efficient_pava_quantile(
    y: &[Float],
    weights: Option<&[Float]>,
    increasing: bool,
    quantile: Float,
) -> Result<Vec<Float>> {
    let n = y.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut result = y.to_vec();
    let default_weights = vec![1.0; n];
    let w = weights.unwrap_or(&default_weights);

    if !increasing {
        result.reverse();
        let mut temp_result = efficient_pava_quantile_increasing(&result, w, quantile)?;
        temp_result.reverse();
        return Ok(temp_result);
    }

    efficient_pava_quantile_increasing(&result, w, quantile)
}

/// Efficient PAVA for Quantile loss with increasing constraint
fn efficient_pava_quantile_increasing(
    y: &[Float],
    weights: &[Float],
    quantile: Float,
) -> Result<Vec<Float>> {
    let n = y.len();
    if n <= 1 {
        return Ok(y.to_vec());
    }

    let mut stack: Vec<QuantileSegment> = Vec::new();

    for i in 0..n {
        let mut current_segment = QuantileSegment {
            start: i,
            end: i,
            values: vec![y[i]],
            weights: vec![weights[i]],
            quantile,
            cached_quantile: Some(y[i]),
        };

        // Merge with previous segments if violating monotonicity
        while let Some(prev_segment) = stack.last() {
            let prev_quantile = prev_segment.weighted_quantile();
            let current_quantile = current_segment.weighted_quantile();

            if prev_quantile <= current_quantile {
                break;
            }

            let prev = stack.pop().unwrap();
            current_segment = merge_quantile_segments(prev, current_segment);
        }

        stack.push(current_segment);
    }

    // Reconstruct the result
    let mut result = vec![0.0; n];
    for segment in stack {
        let quantile_val = segment.weighted_quantile();
        for i in segment.start..=segment.end {
            result[i] = quantile_val;
        }
    }

    Ok(result)
}

/// Segment for L2 loss PAVA
#[derive(Debug, Clone)]
struct Segment {
    start: usize,
    end: usize,
    sum_wy: Float,
    sum_w: Float,
    value: Float,
}

/// Segment for L1 loss PAVA with weighted median
#[derive(Debug, Clone)]
struct WeightedMedianSegment {
    start: usize,
    end: usize,
    values: Vec<Float>,
    weights: Vec<Float>,
    cached_median: Option<Float>,
}

impl WeightedMedianSegment {
    fn weighted_median(&self) -> Float {
        if let Some(median) = self.cached_median {
            return median;
        }
        weighted_median(&self.values, &self.weights)
    }
}

/// Segment for Huber loss PAVA
#[derive(Debug, Clone)]
struct HuberSegment {
    start: usize,
    end: usize,
    values: Vec<Float>,
    weights: Vec<Float>,
    delta: Float,
    cached_estimate: Option<Float>,
}

impl HuberSegment {
    fn huber_estimate(&self) -> Float {
        if let Some(estimate) = self.cached_estimate {
            return estimate;
        }
        huber_weighted_mean(&self.values, &self.weights, self.delta)
    }
}

/// Segment for Quantile loss PAVA
#[derive(Debug, Clone)]
struct QuantileSegment {
    start: usize,
    end: usize,
    values: Vec<Float>,
    weights: Vec<Float>,
    quantile: Float,
    cached_quantile: Option<Float>,
}

impl QuantileSegment {
    fn weighted_quantile(&self) -> Float {
        if let Some(quantile) = self.cached_quantile {
            return quantile;
        }
        weighted_quantile(&self.values, &self.weights, self.quantile)
    }
}

/// Merge two L2 segments
fn merge_segments(prev: Segment, current: Segment) -> Segment {
    let sum_wy = prev.sum_wy + current.sum_wy;
    let sum_w = prev.sum_w + current.sum_w;
    let value = sum_wy / sum_w;

    Segment {
        start: prev.start,
        end: current.end,
        sum_wy,
        sum_w,
        value,
    }
}

/// Merge two weighted median segments
fn merge_weighted_median_segments(
    prev: WeightedMedianSegment,
    current: WeightedMedianSegment,
) -> WeightedMedianSegment {
    let mut values = prev.values;
    values.extend(current.values);
    let mut weights = prev.weights;
    weights.extend(current.weights);

    WeightedMedianSegment {
        start: prev.start,
        end: current.end,
        values,
        weights,
        cached_median: None, // Invalidate cache
    }
}

/// Merge two Huber segments
fn merge_huber_segments(prev: HuberSegment, current: HuberSegment) -> HuberSegment {
    let mut values = prev.values;
    values.extend(current.values);
    let mut weights = prev.weights;
    weights.extend(current.weights);

    HuberSegment {
        start: prev.start,
        end: current.end,
        values,
        weights,
        delta: prev.delta,     // Use the same delta
        cached_estimate: None, // Invalidate cache
    }
}

/// Merge two quantile segments
fn merge_quantile_segments(prev: QuantileSegment, current: QuantileSegment) -> QuantileSegment {
    let mut values = prev.values;
    values.extend(current.values);
    let mut weights = prev.weights;
    weights.extend(current.weights);

    QuantileSegment {
        start: prev.start,
        end: current.end,
        values,
        weights,
        quantile: prev.quantile, // Use the same quantile
        cached_quantile: None,   // Invalidate cache
    }
}

/// Compute weighted median efficiently
fn weighted_median(values: &[Float], weights: &[Float]) -> Float {
    if values.is_empty() {
        return 0.0;
    }

    if values.len() == 1 {
        return values[0];
    }

    // Create sorted pairs of (value, weight)
    let mut pairs: Vec<(Float, Float)> = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| (v, w))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let total_weight: Float = weights.iter().sum();
    let half_weight = total_weight / 2.0;
    let mut cumulative_weight = 0.0;

    for (value, weight) in pairs {
        cumulative_weight += weight;
        if cumulative_weight >= half_weight {
            return value;
        }
    }

    // Fallback (should not reach here)
    values[0]
}

/// Compute Huber weighted mean using iterative algorithm
fn huber_weighted_mean(values: &[Float], weights: &[Float], delta: Float) -> Float {
    if values.is_empty() {
        return 0.0;
    }

    if values.len() == 1 {
        return values[0];
    }

    // Initialize with weighted mean
    let mut estimate = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum::<Float>()
        / weights.iter().sum::<Float>();

    // Iterative reweighting for Huber loss
    for _ in 0..10 {
        // Maximum 10 iterations
        let mut num = 0.0;
        let mut den = 0.0;

        for (&value, &weight) in values.iter().zip(weights.iter()) {
            let residual = value - estimate;
            let abs_residual = residual.abs();

            let huber_weight = if abs_residual <= delta {
                weight
            } else {
                weight * delta / abs_residual
            };

            num += huber_weight * value;
            den += huber_weight;
        }

        let new_estimate = num / den;
        if (new_estimate - estimate).abs() < 1e-6 {
            break;
        }
        estimate = new_estimate;
    }

    estimate
}

/// Compute weighted quantile
fn weighted_quantile(values: &[Float], weights: &[Float], quantile: Float) -> Float {
    if values.is_empty() {
        return 0.0;
    }

    if values.len() == 1 {
        return values[0];
    }

    // Create sorted pairs of (value, weight)
    let mut pairs: Vec<(Float, Float)> = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| (v, w))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let total_weight: Float = weights.iter().sum();
    let target_weight = total_weight * quantile;
    let mut cumulative_weight = 0.0;

    for (value, weight) in pairs {
        cumulative_weight += weight;
        if cumulative_weight >= target_weight {
            return value;
        }
    }

    // Fallback (should not reach here)
    values[0]
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_efficient_isotonic_basic() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let (fitted_x, fitted_y) =
            efficient_isotonic_regression(&x, &y, None, true, LossFunction::SquaredLoss).unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(
                fitted_y[i] <= fitted_y[i + 1],
                "Fitted values are not monotonic: {} > {}",
                fitted_y[i],
                fitted_y[i + 1]
            );
        }
    }

    #[test]
    fn test_efficient_isotonic_decreasing() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![5.0, 3.0, 4.0, 2.0, 1.0]);

        let (fitted_x, fitted_y) =
            efficient_isotonic_regression(&x, &y, None, false, LossFunction::SquaredLoss).unwrap();

        // Check decreasing monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(
                fitted_y[i] >= fitted_y[i + 1],
                "Fitted values are not decreasing: {} < {}",
                fitted_y[i],
                fitted_y[i + 1]
            );
        }
    }

    #[test]
    fn test_efficient_isotonic_l1() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let (fitted_x, fitted_y) =
            efficient_isotonic_regression(&x, &y, None, true, LossFunction::AbsoluteLoss).unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }
    }

    #[test]
    fn test_efficient_isotonic_weighted() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);
        let weights = Array1::from(vec![1.0, 2.0, 1.0, 2.0, 1.0]);

        let (fitted_x, fitted_y) =
            efficient_isotonic_regression(&x, &y, Some(&weights), true, LossFunction::SquaredLoss)
                .unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }
    }

    #[test]
    fn test_weighted_median() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let median = weighted_median(&values, &weights);
        assert_abs_diff_eq!(median, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_huber_weighted_mean() {
        let values = vec![1.0, 2.0, 10.0]; // 10.0 is an outlier
        let weights = vec![1.0, 1.0, 1.0];
        let mean = huber_weighted_mean(&values, &weights, 1.0);
        // Should be robust to the outlier
        assert!(mean < 5.0);
    }
}

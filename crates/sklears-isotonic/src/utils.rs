//! Utility functions for isotonic regression

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::cmp::Ordering;

/// Safe float comparison for sorting that handles NaN values
///
/// NaN values are sorted to the end. For non-NaN values, uses standard float comparison.
/// This function should be used instead of `.partial_cmp().unwrap()` to comply with
/// the No Unwrap Policy.
///
/// # Arguments
///
/// * `a` - First float value
/// * `b` - Second float value
///
/// # Returns
///
/// `Ordering` - The comparison result with NaN values sorted to the end
///
/// # Examples
///
/// ```
/// use sklears_isotonic::utils::safe_float_cmp;
/// use std::cmp::Ordering;
///
/// assert_eq!(safe_float_cmp(&1.0, &2.0), Ordering::Less);
/// assert_eq!(safe_float_cmp(&2.0, &1.0), Ordering::Greater);
/// assert_eq!(safe_float_cmp(&f64::NAN, &1.0), Ordering::Greater); // NaN sorts to end
/// ```
#[inline]
pub fn safe_float_cmp(a: &Float, b: &Float) -> Ordering {
    match a.partial_cmp(b) {
        Some(ord) => ord,
        None => {
            // Handle NaN cases: NaN is greater than everything (sorts to end)
            match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => Ordering::Equal, // Should not happen, but handle safely
            }
        }
    }
}

/// Safe indexed float comparison for sorting vectors of indices by float values
///
/// This is used when sorting indices based on corresponding float values,
/// handling NaN values properly.
///
/// # Arguments
///
/// * `values` - Array of float values to compare
/// * `i` - First index
/// * `j` - Second index
///
/// # Returns
///
/// `Ordering` - The comparison result based on values\[i\] vs values\[j\]
#[inline]
pub fn safe_indexed_float_cmp(values: &Array1<Float>, i: usize, j: usize) -> Ordering {
    safe_float_cmp(&values[i], &values[j])
}

/// Calculate Spearman correlation coefficient between two arrays
pub fn spearman_correlation(x: &Array1<Float>, y: &Array1<Float>) -> Result<Float> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    if x.len() < 2 {
        return Ok(0.0);
    }

    // Create ranks for x and y
    let x_ranks = create_ranks(x);
    let y_ranks = create_ranks(y);

    // Calculate Pearson correlation of ranks
    pearson_correlation(&x_ranks, &y_ranks)
}

/// Calculate Pearson correlation coefficient between two arrays
pub fn pearson_correlation(x: &Array1<Float>, y: &Array1<Float>) -> Result<Float> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    let n = x.len() as Float;
    if n < 2.0 {
        return Ok(0.0);
    }

    let mean_x = x.mean().ok_or_else(|| {
        SklearsError::NumericalError("Failed to compute mean for x array".to_string())
    })?;
    let mean_y = y.mean().ok_or_else(|| {
        SklearsError::NumericalError("Failed to compute mean for y array".to_string())
    })?;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;

        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    if sum_sq_x == 0.0 || sum_sq_y == 0.0 {
        return Ok(0.0);
    }

    Ok(numerator / (sum_sq_x * sum_sq_y).sqrt())
}

/// Create ranks for values (average rank for ties)
fn create_ranks(values: &Array1<Float>) -> Array1<Float> {
    let n = values.len();
    let mut indexed_values: Vec<(Float, usize)> =
        values.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    // Sort by value using safe comparison
    indexed_values.sort_by(|a, b| safe_float_cmp(&a.0, &b.0));

    let mut ranks = Array1::zeros(n);
    let mut i = 0;

    while i < n {
        let current_value = indexed_values[i].0;
        let mut j = i;

        // Find all values equal to current_value
        while j < n && indexed_values[j].0 == current_value {
            j += 1;
        }

        // Assign average rank to all tied values
        let avg_rank = ((i + 1) + j) as Float / 2.0;
        for k in i..j {
            ranks[indexed_values[k].1] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Estimate mutual information between two variables using discretization
pub fn mutual_information(x: &Array1<Float>, y: &Array1<Float>, bins: usize) -> Result<Float> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    if x.len() < bins || bins < 2 {
        return Ok(0.0);
    }

    // Simple discretization and mutual information calculation
    let x_discrete = discretize_variable(x, bins)?;
    let y_discrete = discretize_variable(y, bins)?;

    // Calculate joint and marginal histograms
    let mut joint_hist = Array2::zeros((bins, bins));
    let mut x_hist = Array1::zeros(bins);
    let mut y_hist = Array1::zeros(bins);

    for i in 0..x.len() {
        let x_bin = x_discrete[i].min(bins - 1);
        let y_bin = y_discrete[i].min(bins - 1);

        joint_hist[(x_bin, y_bin)] += 1.0;
        x_hist[x_bin] += 1.0;
        y_hist[y_bin] += 1.0;
    }

    // Normalize to probabilities
    let n = x.len() as Float;
    joint_hist /= n;
    x_hist /= n;
    y_hist /= n;

    // Calculate mutual information
    let mut mi = 0.0;
    for i in 0..bins {
        for j in 0..bins {
            let p_xy = joint_hist[(i, j)];
            let p_x = x_hist[i];
            let p_y = y_hist[j];

            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }

    Ok(mi)
}

/// Discretize a continuous variable into bins
fn discretize_variable(values: &Array1<Float>, bins: usize) -> Result<Array1<usize>> {
    if values.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let min_val = values.iter().fold(Float::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

    if (max_val - min_val).abs() < Float::EPSILON {
        return Ok(Array1::zeros(values.len()));
    }

    let bin_width = (max_val - min_val) / bins as Float;
    let mut result = Array1::zeros(values.len());

    for i in 0..values.len() {
        let bin = ((values[i] - min_val) / bin_width).floor() as usize;
        result[i] = bin.min(bins - 1);
    }

    Ok(result)
}

/// Find the index of the maximum value in an array
pub fn argmax(values: &Array1<Float>) -> Result<usize> {
    if values.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Cannot find argmax of empty array".to_string(),
        ));
    }

    let mut max_idx = 0;
    let mut max_val = values[0];

    for (i, &val) in values.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    Ok(max_idx)
}

/// Calculate mean squared error between predicted and true values
pub fn mean_squared_error(y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float> {
    if y_true.len() != y_pred.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    if y_true.is_empty() {
        return Ok(0.0);
    }

    let diff = y_true - y_pred;
    let squared_diff = &diff * &diff;
    Ok(squared_diff.mean().unwrap())
}

/// Extract subset of data based on indices
pub fn extract_subset_data(data: &Array2<Float>, indices: &[usize]) -> Result<Array2<Float>> {
    if data.is_empty() || indices.is_empty() {
        return Ok(Array2::zeros((
            0,
            if data.is_empty() { 0 } else { data.ncols() },
        )));
    }

    // Validate indices
    for &idx in indices {
        if idx >= data.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Index {} is out of bounds for array with {} rows",
                idx,
                data.nrows()
            )));
        }
    }

    let mut result = Array2::zeros((indices.len(), data.ncols()));
    for (new_i, &old_i) in indices.iter().enumerate() {
        result.row_mut(new_i).assign(&data.row(old_i));
    }

    Ok(result)
}

/// Extract subset of labels based on indices
pub fn extract_subset_labels(labels: &Array1<Float>, indices: &[usize]) -> Result<Array1<Float>> {
    if indices.is_empty() {
        return Ok(Array1::zeros(0));
    }

    // Validate indices
    for &idx in indices {
        if idx >= labels.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Index {} is out of bounds for array with {} elements",
                idx,
                labels.len()
            )));
        }
    }

    let mut result = Array1::zeros(indices.len());
    for (new_i, &old_i) in indices.iter().enumerate() {
        result[new_i] = labels[old_i];
    }

    Ok(result)
}

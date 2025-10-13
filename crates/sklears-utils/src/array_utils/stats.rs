//! Statistical functions for arrays

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, ToPrimitive};

/// Statistics summary for an array
#[derive(Debug, Clone)]
pub struct ArrayStatistics<T> {
    pub mean: T,
    pub std: T,
    pub min: T,
    pub max: T,
    pub median: T,
    pub count: usize,
}

/// Calculate array sum
pub fn array_sum<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + Clone,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let sum = array.iter().fold(T::zero(), |acc, &x| acc + x);
    Ok(sum)
}

/// Calculate array mean
pub fn array_mean<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + Clone + ToPrimitive,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let sum = array.iter().fold(T::zero(), |acc, &x| acc + x);
    let count = T::from(array.len()).unwrap();
    Ok(sum / count)
}

/// Specialized f64 mean with better numerical stability
pub fn array_mean_f64(array: &ArrayView1<f64>) -> UtilsResult<f64> {
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    // Use Kahan summation for better numerical stability
    let mut sum = 0.0;
    let mut c = 0.0; // Compensation for lost low-order bits

    for &value in array {
        let y = value - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    Ok(sum / array.len() as f64)
}

/// Calculate array variance
pub fn array_var<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + Clone + ToPrimitive,
{
    if array.len() <= 1 {
        return Err(UtilsError::InsufficientData {
            min: 2,
            actual: array.len(),
        });
    }

    let mean = array_mean(array)?;
    let variance = array
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .fold(T::zero(), |acc, x| acc + x);

    let n_minus_1 = T::from(array.len() - 1).unwrap();
    Ok(variance / n_minus_1)
}

/// Specialized f64 variance with better numerical stability
pub fn array_variance_f64(array: &ArrayView1<f64>) -> UtilsResult<f64> {
    if array.len() <= 1 {
        return Err(UtilsError::InsufficientData {
            min: 2,
            actual: array.len(),
        });
    }

    let mean = array_mean_f64(array)?;

    // Two-pass algorithm for better numerical stability
    let mut sum_sq_diff = 0.0;
    let mut correction = 0.0; // Compensation for lost low-order bits

    for &value in array {
        let diff = value - mean;
        let sq_diff = diff * diff;
        let y = sq_diff - correction;
        let t = sum_sq_diff + y;
        correction = (t - sum_sq_diff) - y;
        sum_sq_diff = t;
    }

    Ok(sum_sq_diff / (array.len() - 1) as f64)
}

/// Calculate array standard deviation
pub fn array_std<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + Clone + ToPrimitive,
{
    let variance = array_var(array)?;
    Ok(variance.sqrt())
}

/// Find minimum value
pub fn array_min<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: PartialOrd + Clone,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut min_val = &array[0];
    for val in array.iter().skip(1) {
        if val < min_val {
            min_val = val;
        }
    }
    Ok(min_val.clone())
}

/// Find maximum value
pub fn array_max<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: PartialOrd + Clone,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut max_val = &array[0];
    for val in array.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }
    Ok(max_val.clone())
}

/// Find both min and max values efficiently
pub fn array_min_max<T>(array: &Array1<T>) -> UtilsResult<(T, T)>
where
    T: PartialOrd + Clone,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut min_val = &array[0];
    let mut max_val = &array[0];

    for val in array.iter().skip(1) {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    Ok((min_val.clone(), max_val.clone()))
}

/// Calculate median
pub fn array_median<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: PartialOrd + Clone + Float,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut sorted: Vec<T> = array.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = sorted.len();
    if len % 2 == 1 {
        Ok(sorted[len / 2])
    } else {
        let mid = len / 2;
        let sum = sorted[mid - 1] + sorted[mid];
        Ok(sum / T::from(2.0).unwrap())
    }
}

/// Calculate percentile
pub fn array_percentile<T>(array: &Array1<T>, percentile: f64) -> UtilsResult<T>
where
    T: PartialOrd + Clone + Float,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    if !(0.0..=100.0).contains(&percentile) {
        return Err(UtilsError::InvalidParameter(
            "Percentile must be between 0 and 100".to_string(),
        ));
    }

    let mut sorted: Vec<T> = array.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = sorted.len() as f64;
    let index = (percentile / 100.0) * (len - 1.0);
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;

    if lower_index == upper_index {
        Ok(sorted[lower_index])
    } else {
        let weight = T::from(index - lower_index as f64).unwrap();
        let lower_val = sorted[lower_index];
        let upper_val = sorted[upper_index];
        Ok(lower_val + weight * (upper_val - lower_val))
    }
}

/// Calculate multiple quantiles efficiently
pub fn array_quantiles<T>(array: &Array1<T>, quantiles: &[f64]) -> UtilsResult<Vec<T>>
where
    T: PartialOrd + Clone + Float,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut sorted: Vec<T> = array.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut results = Vec::with_capacity(quantiles.len());
    let len = sorted.len() as f64;

    for &q in quantiles {
        if !(0.0..=1.0).contains(&q) {
            return Err(UtilsError::InvalidParameter(
                "Quantiles must be between 0 and 1".to_string(),
            ));
        }

        let index = q * (len - 1.0);
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        let result = if lower_index == upper_index {
            sorted[lower_index]
        } else {
            let weight = T::from(index - lower_index as f64).unwrap();
            let lower_val = sorted[lower_index];
            let upper_val = sorted[upper_index];
            lower_val + weight * (upper_val - lower_val)
        };

        results.push(result);
    }

    Ok(results)
}

/// Cumulative sum
pub fn array_cumsum<T>(array: &Array1<T>) -> UtilsResult<Array1<T>>
where
    T: Float + Clone,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut result = Vec::with_capacity(array.len());
    let mut cumulative = T::zero();

    for &value in array {
        cumulative = cumulative + value;
        result.push(cumulative);
    }

    Ok(Array1::from_vec(result))
}

/// Standardize array to zero mean and unit variance
pub fn array_standardize<T>(array: &Array1<T>) -> UtilsResult<Array1<T>>
where
    T: Float + Clone + ToPrimitive,
{
    if array.len() <= 1 {
        return Err(UtilsError::InsufficientData {
            min: 2,
            actual: array.len(),
        });
    }

    let mean = array_mean(array)?;
    let std = array_std(array)?;

    if std.abs() < T::from(1e-10).unwrap() {
        return Err(UtilsError::InvalidParameter(
            "Standard deviation too small for standardization".to_string(),
        ));
    }

    let standardized = array.mapv(|x| (x - mean) / std);
    Ok(standardized)
}

/// Normalize to [0, 1] range
pub fn array_min_max_normalize<T>(array: &Array1<T>) -> UtilsResult<Array1<T>>
where
    T: PartialOrd + Float + Clone,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let (min_val, max_val) = array_min_max(array)?;
    let range = max_val - min_val;

    if range.abs() < T::from(1e-10).unwrap() {
        // All values are the same, return zeros
        return Ok(Array1::zeros(array.len()));
    }

    let normalized = array.mapv(|x| (x - min_val) / range);
    Ok(normalized)
}

/// Comprehensive array statistics
pub fn array_describe<T>(array: &Array1<T>) -> UtilsResult<ArrayStatistics<T>>
where
    T: Float + Clone + ToPrimitive + PartialOrd,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    Ok(ArrayStatistics {
        mean: array_mean(array)?,
        std: array_std(array)?,
        min: array_min(array)?,
        max: array_max(array)?,
        median: array_median(array)?,
        count: array.len(),
    })
}

//! In-place array operations for memory efficiency

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, ToPrimitive};

/// Standardize array in-place
pub fn array_standardize_inplace<T>(array: &mut Array1<T>) -> UtilsResult<()>
where
    T: Float + Clone + ToPrimitive,
{
    if array.len() <= 1 {
        return Err(UtilsError::InsufficientData {
            min: 2,
            actual: array.len(),
        });
    }

    // Calculate mean
    let sum = array.iter().fold(T::zero(), |acc, &x| acc + x);
    let mean = sum / T::from(array.len()).unwrap();

    // Calculate standard deviation
    let variance = array
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .fold(T::zero(), |acc, x| acc + x)
        / T::from(array.len() - 1).unwrap();

    let std = variance.sqrt();

    if std.abs() < T::from(1e-10).unwrap() {
        return Err(UtilsError::InvalidParameter(
            "Standard deviation too small for standardization".to_string(),
        ));
    }

    // Apply standardization in-place
    for x in array.iter_mut() {
        *x = (*x - mean) / std;
    }

    Ok(())
}

/// Min-max normalize array in-place to [0, 1] range
pub fn array_min_max_normalize_inplace<T>(array: &mut Array1<T>) -> UtilsResult<()>
where
    T: PartialOrd + Float + Clone,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut min_val = array[0];
    let mut max_val = array[0];

    // Find min and max
    for &value in array.iter() {
        if value < min_val {
            min_val = value;
        }
        if value > max_val {
            max_val = value;
        }
    }

    let range = max_val - min_val;

    if range.abs() < T::from(1e-10).unwrap() {
        // All values are the same, set to zero
        for x in array.iter_mut() {
            *x = T::zero();
        }
    } else {
        // Apply normalization in-place
        for x in array.iter_mut() {
            *x = (*x - min_val) / range;
        }
    }

    Ok(())
}

/// Apply function to array elements in-place
pub fn array_apply_inplace<T, F>(array: &mut Array1<T>, func: F) -> UtilsResult<()>
where
    T: Clone,
    F: Fn(T) -> T,
{
    for x in array.iter_mut() {
        *x = func(x.clone());
    }
    Ok(())
}

/// Scale array by constant in-place
pub fn array_scale_inplace<T>(array: &mut Array1<T>, scalar: T) -> UtilsResult<()>
where
    T: Float + Clone,
{
    for x in array.iter_mut() {
        *x = *x * scalar;
    }
    Ok(())
}

/// Add constant to all array elements in-place
pub fn array_add_constant_inplace<T>(array: &mut Array1<T>, constant: T) -> UtilsResult<()>
where
    T: Float + Clone,
{
    for x in array.iter_mut() {
        *x = *x + constant;
    }
    Ok(())
}

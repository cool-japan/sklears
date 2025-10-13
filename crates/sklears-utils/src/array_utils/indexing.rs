//! Advanced indexing operations

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Fancy indexing for 1D arrays with multiple indices
pub fn fancy_indexing_1d<T: Clone>(array: &Array1<T>, indices: &[usize]) -> UtilsResult<Array1<T>> {
    let mut result = Vec::with_capacity(indices.len());

    for &idx in indices {
        if idx >= array.len() {
            return Err(UtilsError::InvalidParameter(format!(
                "Index {} out of bounds for array of length {}",
                idx,
                array.len()
            )));
        }
        result.push(array[idx].clone());
    }

    Ok(Array1::from_vec(result))
}

/// Fancy indexing for 2D arrays
pub fn fancy_indexing_2d<T: Clone>(
    array: &Array2<T>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> UtilsResult<Array2<T>> {
    if row_indices.len() != col_indices.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![row_indices.len()],
            actual: vec![col_indices.len()],
        });
    }

    let mut result = Vec::new();

    for (&row_idx, &col_idx) in row_indices.iter().zip(col_indices.iter()) {
        if row_idx >= array.nrows() {
            return Err(UtilsError::InvalidParameter(format!(
                "Row index {} out of bounds for array with {} rows",
                row_idx,
                array.nrows()
            )));
        }
        if col_idx >= array.ncols() {
            return Err(UtilsError::InvalidParameter(format!(
                "Column index {} out of bounds for array with {} columns",
                col_idx,
                array.ncols()
            )));
        }
        result.push(array[[row_idx, col_idx]].clone());
    }

    // Return as single column matrix
    let result_len = result.len();
    let result_array = Array1::from_vec(result)
        .into_shape_with_order((result_len, 1))
        .map_err(|_| UtilsError::ShapeMismatch {
            expected: vec![result_len, 1],
            actual: vec![result_len],
        })?;

    Ok(result_array)
}

/// Boolean indexing for 1D arrays
pub fn boolean_indexing_1d<T: Clone>(
    array: &Array1<T>,
    mask: &Array1<bool>,
) -> UtilsResult<Array1<T>> {
    if array.len() != mask.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![array.len()],
            actual: vec![mask.len()],
        });
    }

    let mut result = Vec::new();
    for (value, &include) in array.iter().zip(mask.iter()) {
        if include {
            result.push(value.clone());
        }
    }

    Ok(Array1::from_vec(result))
}

/// Boolean indexing for 2D arrays (row-wise)
pub fn boolean_indexing_2d<T: Clone>(
    array: &Array2<T>,
    mask: &Array1<bool>,
) -> UtilsResult<Array2<T>> {
    if array.nrows() != mask.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![array.nrows()],
            actual: vec![mask.len()],
        });
    }

    let mut result_rows = Vec::new();
    let ncols = array.ncols();

    for (row_idx, &include) in mask.iter().enumerate() {
        if include {
            let mut row = Vec::with_capacity(ncols);
            for col_idx in 0..ncols {
                row.push(array[[row_idx, col_idx]].clone());
            }
            result_rows.extend(row);
        }
    }

    let n_selected_rows = mask.iter().filter(|&&x| x).count();

    if n_selected_rows == 0 {
        return Array2::from_shape_vec((0, ncols), vec![]).map_err(|_| UtilsError::ShapeMismatch {
            expected: vec![0, ncols],
            actual: vec![0],
        });
    }

    let result_len = result_rows.len();
    let result_array = Array1::from_vec(result_rows)
        .into_shape_with_order((n_selected_rows, ncols))
        .map_err(|_| UtilsError::ShapeMismatch {
            expected: vec![n_selected_rows, ncols],
            actual: vec![result_len],
        })?;

    Ok(result_array)
}

/// Create boolean mask from condition function
pub fn create_mask<T, F>(array: &Array1<T>, condition: F) -> Array1<bool>
where
    T: Clone,
    F: Fn(&T) -> bool,
{
    array.mapv(|ref x| condition(x))
}

/// Apply where condition (like numpy.where)
pub fn where_condition<T, F>(
    condition: &Array1<bool>,
    true_values: &Array1<T>,
    false_values: &Array1<T>,
) -> UtilsResult<Array1<T>>
where
    T: Clone,
    F: Clone,
{
    if condition.len() != true_values.len() || condition.len() != false_values.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![condition.len()],
            actual: vec![true_values.len(), false_values.len()],
        });
    }

    let mut result = Vec::with_capacity(condition.len());
    for ((cond, true_val), false_val) in condition
        .iter()
        .zip(true_values.iter())
        .zip(false_values.iter())
    {
        if *cond {
            result.push(true_val.clone());
        } else {
            result.push(false_val.clone());
        }
    }

    Ok(Array1::from_vec(result))
}

/// Slice with step (like Python's array[start:end:step])
pub fn slice_with_step<T: Clone>(
    array: &Array1<T>,
    start: Option<usize>,
    end: Option<usize>,
    step: usize,
) -> UtilsResult<Array1<T>> {
    if step == 0 {
        return Err(UtilsError::InvalidParameter(
            "Step cannot be zero".to_string(),
        ));
    }

    let len = array.len();
    let start_idx = start.unwrap_or(0).min(len);
    let end_idx = end.unwrap_or(len).min(len);

    if start_idx >= end_idx {
        return Ok(Array1::from_vec(vec![]));
    }

    let mut result = Vec::new();
    let mut idx = start_idx;

    while idx < end_idx {
        result.push(array[idx].clone());
        idx += step;
    }

    Ok(Array1::from_vec(result))
}

/// Find indices of maximum values
pub fn argmax<T>(array: &Array1<T>) -> UtilsResult<usize>
where
    T: PartialOrd + Clone,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut max_idx = 0;
    let mut max_val = &array[0];

    for (idx, val) in array.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }

    Ok(max_idx)
}

/// Find indices of minimum values
pub fn argmin<T>(array: &Array1<T>) -> UtilsResult<usize>
where
    T: PartialOrd + Clone,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut min_idx = 0;
    let mut min_val = &array[0];

    for (idx, val) in array.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_idx = idx;
        }
    }

    Ok(min_idx)
}

/// Sort indices (argsort)
pub fn argsort<T>(array: &Array1<T>) -> Vec<usize>
where
    T: PartialOrd + Clone,
{
    let mut indices: Vec<usize> = (0..array.len()).collect();
    indices.sort_by(|&a, &b| array[a].partial_cmp(&array[b]).unwrap());
    indices
}

/// Take elements at indices
pub fn take_1d<T: Clone>(array: &Array1<T>, indices: &[usize]) -> UtilsResult<Array1<T>> {
    fancy_indexing_1d(array, indices)
}

/// Put values at indices
pub fn put_1d<T: Clone>(array: &mut Array1<T>, indices: &[usize], values: &[T]) -> UtilsResult<()> {
    if indices.len() != values.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![indices.len()],
            actual: vec![values.len()],
        });
    }

    for (&idx, value) in indices.iter().zip(values.iter()) {
        if idx >= array.len() {
            return Err(UtilsError::InvalidParameter(format!(
                "Index {} out of bounds for array of length {}",
                idx,
                array.len()
            )));
        }
        array[idx] = value.clone();
    }

    Ok(())
}

/// Filter array with condition function
pub fn filter_array<T: Clone>(array: &Array1<T>, predicate: impl Fn(&T) -> bool) -> Array1<T> {
    let filtered: Vec<T> = array.iter().filter(|&x| predicate(x)).cloned().collect();
    Array1::from_vec(filtered)
}

/// Compress array with boolean mask
pub fn compress_1d<T: Clone>(array: &Array1<T>, mask: &Array1<bool>) -> UtilsResult<Array1<T>> {
    boolean_indexing_1d(array, mask)
}

//! Core array utilities and validation functions

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::Zero;
use std::collections::HashMap;

/// Check that a 2D array is not empty
pub fn check_array_2d<T>(array: &Array2<T>) -> UtilsResult<()> {
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }
    Ok(())
}

/// Check that a 1D array is not empty
pub fn check_array_1d<T>(array: &Array1<T>) -> UtilsResult<()> {
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }
    Ok(())
}

/// Safe indexing for 1D arrays
pub fn safe_indexing<T: Clone>(array: &Array1<T>, indices: &[usize]) -> UtilsResult<Array1<T>> {
    let mut result = Vec::with_capacity(indices.len());

    for &idx in indices {
        if idx >= array.len() {
            return Err(UtilsError::InvalidParameter(format!(
                "Index {idx} out of bounds for array of length {}",
                array.len()
            )));
        }
        result.push(array[idx].clone());
    }

    Ok(Array1::from_vec(result))
}

/// Safe indexing for 2D arrays
pub fn safe_indexing_2d<T: Clone>(array: &Array2<T>, indices: &[usize]) -> UtilsResult<Array2<T>> {
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let ncols = array.ncols();
    let mut result = Vec::with_capacity(indices.len() * ncols);

    for &idx in indices {
        if idx >= array.nrows() {
            return Err(UtilsError::InvalidParameter(format!(
                "Row index {idx} out of bounds for array with {} rows",
                array.nrows()
            )));
        }
        for col in 0..ncols {
            result.push(array[[idx, col]].clone());
        }
    }

    let result_len = result.len();
    let result_array = Array1::from_vec(result)
        .into_shape_with_order((indices.len(), ncols))
        .map_err(|_| UtilsError::ShapeMismatch {
            expected: vec![indices.len(), ncols],
            actual: vec![result_len],
        })?;

    Ok(result_array)
}

/// Convert 2D array to 1D if possible, otherwise return error
pub fn column_or_1d<T: Clone>(array: &Array2<T>) -> UtilsResult<Array1<T>> {
    if array.ncols() == 1 {
        Ok(array.column(0).to_owned())
    } else if array.nrows() == 1 {
        Ok(array.row(0).to_owned())
    } else {
        Err(UtilsError::ShapeMismatch {
            expected: vec![1],
            actual: vec![array.nrows(), array.ncols()],
        })
    }
}

/// Normalize array to unit norm (L2)
pub fn normalize_array(array: &mut Array1<f64>) -> UtilsResult<()> {
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let norm = array.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        array.par_mapv_inplace(|x| x / norm);
    }
    Ok(())
}

/// Get unique labels in sorted order
pub fn unique_labels<T: Clone + Ord>(labels: &Array1<T>) -> Vec<T> {
    let mut unique: Vec<T> = labels.iter().cloned().collect();
    unique.sort();
    unique.dedup();
    unique
}

/// Count occurrences of each label
pub fn label_counts<T: Clone + Eq + std::hash::Hash>(labels: &Array1<T>) -> HashMap<T, usize> {
    let mut counts = HashMap::new();
    for label in labels.iter() {
        *counts.entry(label.clone()).or_insert(0) += 1;
    }
    counts
}

/// Split array into chunks
pub fn array_split<T: Clone>(array: &Array1<T>, n_splits: usize) -> UtilsResult<Vec<Array1<T>>> {
    if n_splits == 0 {
        return Err(UtilsError::InvalidParameter(
            "Number of splits must be positive".to_string(),
        ));
    }

    if array.is_empty() {
        return Ok(vec![Array1::from_vec(vec![]); n_splits]);
    }

    let chunk_size = array.len() / n_splits;
    let remainder = array.len() % n_splits;

    let mut splits = Vec::with_capacity(n_splits);
    let mut start = 0;

    for i in 0..n_splits {
        let current_chunk_size = if i < remainder {
            chunk_size + 1
        } else {
            chunk_size
        };

        let end = start + current_chunk_size;
        let chunk = array.slice(s![start..end]).to_owned();
        splits.push(chunk);
        start = end;
    }

    Ok(splits)
}

/// Concatenate arrays
pub fn array_concatenate<T: Clone>(arrays: &[Array1<T>]) -> UtilsResult<Array1<T>> {
    if arrays.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut result = Vec::new();
    for array in arrays {
        result.extend_from_slice(array.as_slice().unwrap());
    }

    Ok(Array1::from_vec(result))
}

/// Resize array to new size, padding with zeros if needed
pub fn array_resize<T: Clone + Zero>(array: &Array1<T>, new_size: usize) -> Array1<T> {
    let mut result = vec![T::zero(); new_size];
    let copy_size = array.len().min(new_size);
    result[..copy_size].clone_from_slice(&array.as_slice().unwrap()[..copy_size]);
    Array1::from_vec(result)
}

/// Count unique elements
pub fn array_unique_counts<T: Clone + Ord + std::hash::Hash>(
    array: &Array1<T>,
) -> HashMap<T, usize> {
    let mut counts = HashMap::new();
    for item in array.iter() {
        *counts.entry(item.clone()).or_insert(0) += 1;
    }
    counts
}

/// Reverse array
pub fn array_reverse<T: Clone>(array: &Array1<T>) -> Array1<T> {
    let mut reversed: Vec<T> = array.iter().cloned().collect();
    reversed.reverse();
    Array1::from_vec(reversed)
}

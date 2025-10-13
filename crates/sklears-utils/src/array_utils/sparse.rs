//! Sparse array operations

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::numeric::{Float, Zero};

/// Type alias for sparse matrix addition result
type SparseAddResult<T> = UtilsResult<(Vec<T>, Vec<(usize, usize)>)>;

/// Sparse-dense dot product with bounds checking
pub fn safe_sparse_dot<T>(
    sparse_indices: &[usize],
    sparse_values: &[T],
    dense: &ArrayView1<T>,
) -> UtilsResult<T>
where
    T: Float + Clone,
{
    if sparse_indices.len() != sparse_values.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![sparse_indices.len()],
            actual: vec![sparse_values.len()],
        });
    }

    let mut result = T::zero();
    for (&idx, &val) in sparse_indices.iter().zip(sparse_values.iter()) {
        if idx >= dense.len() {
            return Err(UtilsError::InvalidParameter(format!(
                "Sparse index {} out of bounds for dense array of length {}",
                idx,
                dense.len()
            )));
        }
        result = result + val * dense[idx];
    }

    Ok(result)
}

/// Specialized f64 sparse-dense dot product
pub fn safe_sparse_dot_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> UtilsResult<f64> {
    if a.len() != b.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }

    let mut result = 0.0;
    for (ai, bi) in a.iter().zip(b.iter()) {
        result += ai * bi;
    }
    Ok(result)
}

/// Specialized f32 sparse-dense dot product
pub fn safe_sparse_dot_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> UtilsResult<f32> {
    if a.len() != b.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }

    let mut result = 0.0;
    for (ai, bi) in a.iter().zip(b.iter()) {
        result += ai * bi;
    }
    Ok(result)
}

/// Sparse matrix transpose (CSR format)
pub fn sparse_transpose<T: Clone + Zero + PartialEq>(
    values: &[T],
    row_indices: &[usize],
    col_indices: &[usize],
    shape: (usize, usize),
) -> UtilsResult<(Vec<T>, Vec<usize>, Vec<usize>)> {
    let (_nrows, ncols) = shape;

    // Count non-zeros per column for the transposed matrix
    let mut col_counts = vec![0; ncols];
    for &col_idx in col_indices {
        if col_idx >= ncols {
            return Err(UtilsError::InvalidParameter(format!(
                "Column index {} exceeds matrix width {}",
                col_idx, ncols
            )));
        }
        col_counts[col_idx] += 1;
    }

    // Build transpose in COO format first
    let mut transpose_values = Vec::with_capacity(values.len());
    let mut transpose_rows = Vec::with_capacity(values.len());
    let mut transpose_cols = Vec::with_capacity(values.len());

    for (idx, (value, &row_idx)) in values.iter().zip(row_indices.iter()).enumerate() {
        let col_idx = col_indices[idx];
        transpose_values.push(value.clone());
        transpose_rows.push(col_idx); // Original column becomes row
        transpose_cols.push(row_idx); // Original row becomes column
    }

    Ok((transpose_values, transpose_rows, transpose_cols))
}

/// Sparse matrix addition
pub fn sparse_add<T>(
    a_values: &[T],
    a_indices: &[(usize, usize)],
    b_values: &[T],
    b_indices: &[(usize, usize)],
) -> SparseAddResult<T>
where
    T: Float + Clone,
{
    let mut result_map = std::collections::HashMap::new();

    // Add values from matrix A
    for (idx, &value) in a_indices.iter().zip(a_values.iter()) {
        *result_map.entry(idx).or_insert(T::zero()) =
            *result_map.get(&idx).unwrap_or(&T::zero()) + value;
    }

    // Add values from matrix B
    for (idx, &value) in b_indices.iter().zip(b_values.iter()) {
        *result_map.entry(idx).or_insert(T::zero()) =
            *result_map.get(&idx).unwrap_or(&T::zero()) + value;
    }

    // Convert back to arrays, filtering out zeros
    let mut result_values = Vec::new();
    let mut result_indices = Vec::new();

    for (idx, value) in result_map {
        if value.abs() > T::from(1e-12).unwrap() {
            // Filter near-zero values
            result_values.push(value);
            result_indices.push(*idx);
        }
    }

    Ok((result_values, result_indices))
}

/// Create sparse diagonal matrix
pub fn sparse_diag<T: Clone + Zero>(
    diagonal: &[T],
) -> UtilsResult<(Vec<T>, Vec<usize>, Vec<usize>)> {
    let _n = diagonal.len();
    let mut values = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for (i, value) in diagonal.iter().enumerate() {
        if !value.is_zero() {
            values.push(value.clone());
            row_indices.push(i);
            col_indices.push(i);
        }
    }

    Ok((values, row_indices, col_indices))
}

/// Convert sparse to dense if density exceeds threshold
pub fn densify_threshold<T>(
    values: &[T],
    indices: &[(usize, usize)],
    shape: (usize, usize),
    threshold: f64,
) -> UtilsResult<Option<Array2<T>>>
where
    T: Clone + Zero,
{
    let (nrows, ncols) = shape;
    let total_elements = nrows * ncols;
    let nnz = values.len();

    let density = nnz as f64 / total_elements as f64;

    if density > threshold {
        let mut dense = Array2::zeros(shape);

        for (idx, value) in indices.iter().zip(values.iter()) {
            let (row, col) = idx;
            if *row >= nrows || *col >= ncols {
                return Err(UtilsError::InvalidParameter(format!(
                    "Index ({}, {}) out of bounds for shape ({}, {})",
                    row, col, nrows, ncols
                )));
            }
            dense[[*row, *col]] = value.clone();
        }

        Ok(Some(dense))
    } else {
        Ok(None)
    }
}

//! Shape manipulation operations for arrays

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::Zero;

/// Reshape 1D array to 2D with specified dimensions
pub fn reshape_1d_to_2d<T: Clone>(
    array: &Array1<T>,
    rows: usize,
    cols: usize,
) -> UtilsResult<Array2<T>> {
    if rows * cols != array.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![rows * cols],
            actual: vec![array.len()],
        });
    }

    let reshaped = array
        .clone()
        .into_shape_with_order((rows, cols))
        .map_err(|_| UtilsError::ShapeMismatch {
            expected: vec![rows, cols],
            actual: vec![array.len()],
        })?;

    Ok(reshaped)
}

/// Flatten 2D array to 1D
pub fn flatten_2d<T: Clone>(array: &Array2<T>) -> Array1<T> {
    let mut flattened = Vec::with_capacity(array.len());
    for row in array.rows() {
        for item in row {
            flattened.push(item.clone());
        }
    }
    Array1::from_vec(flattened)
}

/// Check if two shapes are broadcastable
pub fn is_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
    let max_len = shape1.len().max(shape2.len());

    for i in 0..max_len {
        let dim1 = if i < shape1.len() {
            shape1[shape1.len() - 1 - i]
        } else {
            1
        };
        let dim2 = if i < shape2.len() {
            shape2[shape2.len() - 1 - i]
        } else {
            1
        };

        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }

    true
}

/// Compute broadcasted shape
pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> UtilsResult<Vec<usize>> {
    if !is_broadcastable(shape1, shape2) {
        return Err(UtilsError::ShapeMismatch {
            expected: shape1.to_vec(),
            actual: shape2.to_vec(),
        });
    }

    let max_len = shape1.len().max(shape2.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let dim1 = if i < shape1.len() {
            shape1[shape1.len() - 1 - i]
        } else {
            1
        };
        let dim2 = if i < shape2.len() {
            shape2[shape2.len() - 1 - i]
        } else {
            1
        };

        result.push(dim1.max(dim2));
    }

    result.reverse();
    Ok(result)
}

/// Transpose 2D array
pub fn transpose<T: Clone>(array: &Array2<T>) -> Array2<T> {
    array.t().to_owned()
}

/// Stack 1D arrays along specified axis to form 2D array
pub fn stack_1d<T: Clone + Zero>(arrays: &[&Array1<T>], axis: usize) -> UtilsResult<Array2<T>> {
    if arrays.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    if axis > 1 {
        return Err(UtilsError::InvalidParameter(
            "Axis must be 0 or 1 for stacking 1D arrays into 2D".to_string(),
        ));
    }

    let first_len = arrays[0].len();
    for array in arrays.iter() {
        if array.len() != first_len {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![first_len],
                actual: vec![array.len()],
            });
        }
    }

    let mut result = if axis == 0 {
        // Stack along rows (each array becomes a row)
        Array2::zeros((arrays.len(), first_len))
    } else {
        // Stack along columns (each array becomes a column)
        Array2::zeros((first_len, arrays.len()))
    };

    for (i, array) in arrays.iter().enumerate() {
        if axis == 0 {
            for (j, value) in array.iter().enumerate() {
                result[[i, j]] = value.clone();
            }
        } else {
            for (j, value) in array.iter().enumerate() {
                result[[j, i]] = value.clone();
            }
        }
    }

    Ok(result)
}

/// Concatenate 2D arrays along specified axis
pub fn concatenate_2d<T: Clone + Zero>(
    arrays: &[&Array2<T>],
    axis: usize,
) -> UtilsResult<Array2<T>> {
    if arrays.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    if axis > 1 {
        return Err(UtilsError::InvalidParameter(
            "Axis must be 0 or 1 for 2D arrays".to_string(),
        ));
    }

    let first_shape = arrays[0].raw_dim();

    if axis == 0 {
        // Concatenate along rows - all arrays must have same number of columns
        let ncols = first_shape[1];
        for array in arrays.iter() {
            if array.ncols() != ncols {
                return Err(UtilsError::ShapeMismatch {
                    expected: vec![array.nrows(), ncols],
                    actual: vec![array.nrows(), array.ncols()],
                });
            }
        }

        let total_rows: usize = arrays.iter().map(|arr| arr.nrows()).sum();
        let mut result = Array2::zeros((total_rows, ncols));

        let mut row_offset = 0;
        for array in arrays {
            let nrows = array.nrows();
            for i in 0..nrows {
                for j in 0..ncols {
                    result[[row_offset + i, j]] = array[[i, j]].clone();
                }
            }
            row_offset += nrows;
        }

        Ok(result)
    } else {
        // Concatenate along columns - all arrays must have same number of rows
        let nrows = first_shape[0];
        for array in arrays.iter() {
            if array.nrows() != nrows {
                return Err(UtilsError::ShapeMismatch {
                    expected: vec![nrows, array.ncols()],
                    actual: vec![array.nrows(), array.ncols()],
                });
            }
        }

        let total_cols: usize = arrays.iter().map(|arr| arr.ncols()).sum();
        let mut result = Array2::zeros((nrows, total_cols));

        let mut col_offset = 0;
        for array in arrays {
            let ncols = array.ncols();
            for i in 0..nrows {
                for j in 0..ncols {
                    result[[i, col_offset + j]] = array[[i, j]].clone();
                }
            }
            col_offset += ncols;
        }

        Ok(result)
    }
}

/// Split 2D array along specified axis
pub fn split_2d<T: Clone>(
    array: &Array2<T>,
    indices_or_sections: &[usize],
    axis: usize,
) -> UtilsResult<Vec<Array2<T>>> {
    if axis > 1 {
        return Err(UtilsError::InvalidParameter(
            "Axis must be 0 or 1 for 2D arrays".to_string(),
        ));
    }

    let mut splits = Vec::new();
    let mut start = 0;

    if axis == 0 {
        // Split along rows
        for &split_point in indices_or_sections {
            if split_point > array.nrows() {
                return Err(UtilsError::InvalidParameter(
                    "Split index exceeds array dimensions".to_string(),
                ));
            }

            let section = array.slice(s![start..split_point, ..]).to_owned();
            splits.push(section);
            start = split_point;
        }

        // Add remaining section
        if start < array.nrows() {
            let section = array.slice(s![start.., ..]).to_owned();
            splits.push(section);
        }
    } else {
        // Split along columns
        for &split_point in indices_or_sections {
            if split_point > array.ncols() {
                return Err(UtilsError::InvalidParameter(
                    "Split index exceeds array dimensions".to_string(),
                ));
            }

            let section = array.slice(s![.., start..split_point]).to_owned();
            splits.push(section);
            start = split_point;
        }

        // Add remaining section
        if start < array.ncols() {
            let section = array.slice(s![.., start..]).to_owned();
            splits.push(section);
        }
    }

    Ok(splits)
}

/// Tile 2D array with specified repetitions
pub fn tile_2d<T: Clone + Zero>(array: &Array2<T>, reps: (usize, usize)) -> UtilsResult<Array2<T>> {
    if reps.0 == 0 || reps.1 == 0 {
        return Err(UtilsError::InvalidParameter(
            "Repetitions must be positive".to_string(),
        ));
    }

    let (nrows, ncols) = array.dim();
    let new_shape = (nrows * reps.0, ncols * reps.1);
    let mut result = Array2::zeros(new_shape);

    for rep_row in 0..reps.0 {
        for rep_col in 0..reps.1 {
            let row_offset = rep_row * nrows;
            let col_offset = rep_col * ncols;

            for i in 0..nrows {
                for j in 0..ncols {
                    result[[row_offset + i, col_offset + j]] = array[[i, j]].clone();
                }
            }
        }
    }

    Ok(result)
}

/// Pad 2D array with specified padding
pub fn pad_2d<T: Clone + Zero>(
    array: &Array2<T>,
    padding: ((usize, usize), (usize, usize)),
    constant_value: Option<T>,
) -> UtilsResult<Array2<T>> {
    let (nrows, ncols) = array.dim();
    let (row_padding, col_padding) = padding;

    let new_nrows = nrows + row_padding.0 + row_padding.1;
    let new_ncols = ncols + col_padding.0 + col_padding.1;

    let fill_value = constant_value.unwrap_or_else(T::zero);
    let mut result = Array2::from_elem((new_nrows, new_ncols), fill_value);

    // Copy original array to the center
    for i in 0..nrows {
        for j in 0..ncols {
            result[[i + row_padding.0, j + col_padding.0]] = array[[i, j]].clone();
        }
    }

    Ok(result)
}

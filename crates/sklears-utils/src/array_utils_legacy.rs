use crate::{UtilsError, UtilsResult};
use scirs2_core::numeric::{Float, Zero};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, s};
use std::cmp;
use std::collections::HashMap;

/// SIMD-accelerated operations for high-performance array utility computations
/// Uses SciRS2-Core's unified SIMD abstraction for better performance and compatibility
mod simd_array_utils {
    use super::*;

    /// SIMD-accelerated sum calculation for f64 arrays using scalar fallback
    #[inline]
    pub fn simd_sum_f64(data: &[f64]) -> f64 {
        data.iter().sum()
    }

    /// SIMD-accelerated sum calculation for f32 arrays using scalar fallback
    #[inline]
    pub fn simd_sum_f32(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    /// SIMD-accelerated dot product for f64 arrays using scalar fallback
    #[inline]
    pub fn simd_dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// SIMD-accelerated dot product for f32 arrays using scalar fallback
    #[inline]
    pub fn simd_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// SIMD-accelerated normalization for f64 arrays using scalar fallback
    #[inline]
    pub fn simd_normalize_f64(data: &mut [f64], sum: f64) {
        if sum != 0.0 {
            data.iter_mut().for_each(|x| *x /= sum);
        }
    }

    /// SIMD-accelerated normalization for f32 arrays using scalar fallback
    #[inline]
    pub fn simd_normalize_f32(data: &mut [f32], sum: f32) {
        if sum != 0.0 {
            data.iter_mut().for_each(|x| *x /= sum);
        }
    }

    /// SIMD-accelerated element-wise addition for f64 arrays using scalar fallback
    #[inline]
    pub fn simd_add_arrays_f64(a: &[f64], b: &[f64], result: &mut [f64]) {
        let len = a.len().min(b.len()).min(result.len());
        for i in 0..len {
            result[i] = a[i] + b[i];
        }
    }

    /// SIMD-accelerated element-wise addition for f32 arrays using scalar fallback
    #[inline]
    pub fn simd_add_arrays_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len().min(b.len()).min(result.len());
        for i in 0..len {
            result[i] = a[i] + b[i];
        }
    }

    /// SIMD-accelerated element-wise multiplication for f64 arrays using scalar fallback
    #[inline]
    pub fn simd_multiply_arrays_f64(a: &[f64], b: &[f64], result: &mut [f64]) {
        let len = a.len().min(b.len()).min(result.len());
        for i in 0..len {
            result[i] = a[i] * b[i];
        }
    }

    /// SIMD-accelerated element-wise multiplication for f32 arrays using scalar fallback
    #[inline]
    pub fn simd_multiply_arrays_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len().min(b.len()).min(result.len());
        for i in 0..len {
            result[i] = a[i] * b[i];
        }
    }

    /// SIMD-accelerated scalar multiplication for f64 arrays using scalar fallback
    #[inline]
    pub fn simd_scale_f64(data: &mut [f64], scalar: f64) {
        data.iter_mut().for_each(|x| *x *= scalar);
    }

    /// SIMD-accelerated scalar multiplication for f32 arrays using scalar fallback
    #[inline]
    pub fn simd_scale_f32(data: &mut [f32], scalar: f32) {
        data.iter_mut().for_each(|x| *x *= scalar);
    }
}

pub fn check_array_2d<T>(array: &Array2<T>) -> UtilsResult<()> {
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }
    Ok(())
}

pub fn check_array_1d<T>(array: &Array1<T>) -> UtilsResult<()> {
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }
    Ok(())
}

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
        let row = array.row(idx);
        result.extend(row.iter().cloned());
    }

    Array2::from_shape_vec((indices.len(), ncols), result)
        .map_err(|e| UtilsError::InvalidParameter(format!("Shape error: {e}")))
}

pub fn array_split<T: Clone>(
    array: &Array1<T>,
    indices: &[usize],
) -> UtilsResult<Vec<Array1<T>>> {
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut result = Vec::new();
    let mut start = 0;

    for &split_idx in indices {
        if split_idx > array.len() {
            return Err(UtilsError::InvalidParameter(format!(
                "Split index {split_idx} out of bounds for array of length {}",
                array.len()
            )));
        }

        if split_idx > start {
            let segment = array.slice(s![start..split_idx]).to_owned();
            result.push(segment);
        }
        start = split_idx;
    }

    if start < array.len() {
        let segment = array.slice(s![start..]).to_owned();
        result.push(segment);
    }

    Ok(result)
}

pub fn array_concatenate<T: Clone>(arrays: &[Array1<T>]) -> UtilsResult<Array1<T>> {
    if arrays.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let total_len: usize = arrays.iter().map(|arr| arr.len()).sum();
    let mut result = Vec::with_capacity(total_len);

    for array in arrays {
        result.extend(array.iter().cloned());
    }

    Ok(Array1::from_vec(result))
}

pub fn array_resize<T: Clone + Zero>(array: &Array1<T>, new_size: usize) -> Array1<T> {
    let mut result = vec![T::zero(); new_size];
    let copy_len = array.len().min(new_size);

    result[..copy_len].clone_from_slice(&array.as_slice().unwrap()[..copy_len]);

    Array1::from_vec(result)
}

pub fn array_unique_counts<T: Clone + Ord + std::hash::Hash>(array: &Array1<T>) -> HashMap<T, usize> {
    let mut counts = HashMap::new();
    for item in array.iter() {
        *counts.entry(item.clone()).or_insert(0) += 1;
    }
    counts
}

pub fn array_argsort<T: Clone + PartialOrd>(array: &Array1<T>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..array.len()).collect();
    indices.sort_by(|&a, &b| array[a].partial_cmp(&array[b]).unwrap_or(cmp::Ordering::Equal));
    indices
}

pub fn array_shuffle<T>(array: &mut Array1<T>, rng: &mut impl rand::Rng) {
    let slice = array.as_slice_mut().unwrap();
    for i in (1..slice.len()).rev() {
        let j = rng.gen_range(0..i + 1);
        slice.swap(i, j);
    }
}

pub fn array_reverse<T: Clone>(array: &Array1<T>) -> Array1<T> {
    let mut result = array.to_owned();
    result.as_slice_mut().unwrap().reverse();
    result
}

// Statistical functions with fixed type parameter issues
pub fn array_sum<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + Zero,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    Ok(array.iter().fold(T::zero(), |acc, &x| acc + x))
}

pub fn array_mean<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + Zero,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let sum = array.iter().fold(T::zero(), |acc, &x| acc + x);
    let len = T::from(array.len()).ok_or_else(|| {
        UtilsError::InvalidParameter("Array length cannot be converted to float".to_string())
    })?;

    Ok(sum / len)
}

pub fn array_var<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + Zero,
{
    if array.len() <= 1 {
        return Err(UtilsError::InvalidParameter("Need at least 2 elements for variance".to_string()));
    }

    let mean = array_mean(array)?;
    let sum_sq_diff = array.iter().fold(T::zero(), |acc, &x| {
        let diff = x - mean;
        acc + diff * diff
    });

    let len_minus_one = T::from(array.len() - 1).ok_or_else(|| {
        UtilsError::InvalidParameter("Array length cannot be converted to float".to_string())
    })?;

    Ok(sum_sq_diff / len_minus_one)
}

pub fn array_std<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + Zero,
{
    let var = array_var(array)?;
    Ok(var.sqrt())
}

pub fn array_min<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + PartialOrd + Copy,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut min_val = array[0];
    for &val in array.iter().skip(1) {
        if val < min_val {
            min_val = val;
        }
    }

    Ok(min_val)
}

pub fn array_max<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + PartialOrd + Copy,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut max_val = array[0];
    for &val in array.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }

    Ok(max_val)
}

pub fn array_median<T>(array: &Array1<T>) -> UtilsResult<T>
where
    T: Float + PartialOrd + Copy,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let mut sorted = array.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal));

    let len = sorted.len();
    if len % 2 == 1 {
        Ok(sorted[len / 2])
    } else {
        let mid1 = sorted[len / 2 - 1];
        let mid2 = sorted[len / 2];
        let two = T::from(2.0).unwrap();
        Ok((mid1 + mid2) / two)
    }
}

pub fn array_percentile<T>(array: &Array1<T>, percentile: f64) -> UtilsResult<T>
where
    T: Float + PartialOrd + Copy,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    if !(0.0..=100.0).contains(&percentile) {
        return Err(UtilsError::InvalidParameter("Percentile must be between 0 and 100".to_string()));
    }

    let mut sorted = array.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal));

    let len = sorted.len();
    let index = (percentile / 100.0) * (len - 1) as f64;
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;

    if lower_index == upper_index {
        Ok(sorted[lower_index])
    } else {
        let weight = T::from(index - lower_index as f64).unwrap();
        let one_minus_weight = T::from(1.0).unwrap() - weight;
        Ok(sorted[lower_index] * one_minus_weight + sorted[upper_index] * weight)
    }
}

/// Statistical description of array
#[derive(Debug, Clone)]
pub struct ArrayStatistics<T> {
    pub count: usize,
    pub mean: T,
    pub std: T,
    pub min: T,
    pub percentile_25: T,
    pub percentile_50: T,
    pub percentile_75: T,
    pub max: T,
}

pub fn array_standardize<T>(array: &Array1<T>) -> UtilsResult<Array1<T>>
where
    T: Float + Zero,
{
    if array.len() <= 1 {
        return Err(UtilsError::InvalidParameter("Need at least 2 elements for standardization".to_string()));
    }

    let mean = array_mean(array)?;
    let std = array_std(array)?;

    if std == T::zero() {
        return Err(UtilsError::InvalidParameter("Cannot standardize array with zero standard deviation".to_string()));
    }

    let standardized = array.mapv(|x| (x - mean) / std);
    Ok(standardized)
}

pub fn array_describe<T>(array: &Array1<T>) -> UtilsResult<ArrayStatistics<T>>
where
    T: Float + PartialOrd + Copy,
{
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let count = array.len();
    let mean = array_mean(array)?;
    let std = if count > 1 {
        array_std(array)?
    } else {
        T::zero()
    };
    let min = array_min(array)?;
    let max = array_max(array)?;
    let percentile_25 = array_percentile(array, 25.0)?;
    let percentile_50 = array_percentile(array, 50.0)?;
    let percentile_75 = array_percentile(array, 75.0)?;

    Ok(ArrayStatistics {
        count,
        mean,
        std,
        min,
        percentile_25,
        percentile_50,
        percentile_75,
        max,
    })
}

pub fn array_standardize_inplace<T>(array: &mut Array1<T>) -> UtilsResult<()>
where
    T: Float + Zero,
{
    if array.len() <= 1 {
        return Err(UtilsError::InvalidParameter("Need at least 2 elements for standardization".to_string()));
    }

    let mean = array_mean(array)?;
    let std = array_std(array)?;

    if std == T::zero() {
        return Err(UtilsError::InvalidParameter("Cannot standardize array with zero standard deviation".to_string()));
    }

    array.mapv_inplace(|x| (x - mean) / std);
    Ok(())
}

// Helper functions for SIMD operations (currently using scalar implementations)
pub fn fast_dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
    simd_array_utils::simd_dot_product_f64(a, b)
}

pub fn fast_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    simd_array_utils::simd_dot_product_f32(a, b)
}

pub fn fast_sum_f64(data: &[f64]) -> f64 {
    simd_array_utils::simd_sum_f64(data)
}

pub fn fast_sum_f32(data: &[f32]) -> f32 {
    simd_array_utils::simd_sum_f32(data)
}
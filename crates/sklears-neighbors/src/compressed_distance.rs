//! Compressed distance matrices for memory-efficient neighbor storage
//!
//! This module provides compressed representations of distance matrices
//! using various compression techniques to reduce memory usage while
//! maintaining reasonable accuracy for neighbor search operations.

use crate::{NeighborsError, NeighborsResult};
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Type alias for compressed data with sparse indices
type CompressedData = NeighborsResult<(Vec<u8>, Vec<(usize, usize)>)>;

/// Compression method for distance matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMethod {
    /// 16-bit float compression (half precision)
    Float16,
    /// 8-bit quantization with min-max scaling
    Quantized8Bit,
    /// 4-bit quantization with min-max scaling
    Quantized4Bit,
    /// Sparse representation (only store distances below threshold)
    Sparse,
    /// Delta compression (store differences from previous row)
    Delta,
    /// Hybrid compression combining multiple methods
    Hybrid,
}

/// Statistics for compression performance analysis
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: Float,
    pub max_error: Float,
    pub mean_error: Float,
    pub compression_time_ms: f64,
    pub decompression_time_ms: f64,
}

impl CompressionStats {
    pub fn new(original_size: usize, compressed_size: usize) -> Self {
        let ratio = if compressed_size > 0 {
            original_size as Float / compressed_size as Float
        } else {
            0.0
        };

        Self {
            original_size,
            compressed_size,
            compression_ratio: ratio,
            max_error: 0.0,
            mean_error: 0.0,
            compression_time_ms: 0.0,
            decompression_time_ms: 0.0,
        }
    }
}

/// 16-bit float representation for half precision
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Float16 {
    bits: u16,
}

unsafe impl Pod for Float16 {}
unsafe impl Zeroable for Float16 {}

impl Float16 {
    fn from_f32(value: Float) -> Self {
        // IEEE 754 half precision conversion
        let bits = (value as f32).to_bits();
        let sign = (bits >> 31) & 0x01;
        let exp = (bits >> 23) & 0xff;
        let frac = bits & 0x7fffff;

        let half_bits = if exp == 0 {
            // Zero or denormal
            ((sign << 15) | (frac >> 13)) as u16
        } else if exp == 0xff {
            // Infinity or NaN
            ((sign << 15) | 0x7c00 | (frac >> 13)) as u16
        } else {
            // Normal number
            let exp_half = exp as i32 - 127 + 15;
            if exp_half <= 0 {
                // Underflow to zero
                (sign << 15) as u16
            } else if exp_half >= 31 {
                // Overflow to infinity
                ((sign << 15) | 0x7c00) as u16
            } else {
                ((sign << 15) | ((exp_half as u32) << 10) | (frac >> 13)) as u16
            }
        };

        Self { bits: half_bits }
    }

    fn to_f32(self) -> Float {
        let sign = (self.bits >> 15) & 0x01;
        let exp = (self.bits >> 10) & 0x1f;
        let frac = self.bits & 0x3ff;

        let full_bits = if exp == 0 {
            // Zero or denormal
            if frac == 0 {
                (sign as u32) << 31
            } else {
                // Convert denormal to normal
                let exp_full = 127 - 15 + 1;
                let frac_full = (frac as u32) << 13;
                ((sign as u32) << 31) | ((exp_full as u32) << 23) | frac_full
            }
        } else if exp == 31 {
            // Infinity or NaN
            ((sign as u32) << 31) | 0x7f800000 | ((frac as u32) << 13)
        } else {
            // Normal number
            let exp_full = exp as u32 + 127 - 15;
            let frac_full = (frac as u32) << 13;
            ((sign as u32) << 31) | (exp_full << 23) | frac_full
        };

        f32::from_bits(full_bits) as Float
    }
}

/// Compressed distance matrix using various compression methods
pub struct CompressedDistanceMatrix {
    method: CompressionMethod,
    shape: (usize, usize),
    data: Vec<u8>,
    min_value: Float,
    max_value: Float,
    sparse_threshold: Float,
    sparse_indices: Vec<(usize, usize)>,
    stats: CompressionStats,
}

impl CompressedDistanceMatrix {
    /// Create a new compressed distance matrix from a full distance matrix
    ///
    /// # Arguments
    /// * `distances` - Full distance matrix to compress
    /// * `method` - Compression method to use
    /// * `sparse_threshold` - Threshold for sparse compression (only used for Sparse method)
    ///
    /// # Returns
    /// * `NeighborsResult<Self>` - Compressed distance matrix
    pub fn new(
        distances: &Array2<Float>,
        method: CompressionMethod,
        sparse_threshold: Option<Float>,
    ) -> NeighborsResult<Self> {
        let start_time = std::time::Instant::now();
        let shape = distances.dim();
        let original_size = shape.0 * shape.1 * std::mem::size_of::<Float>();

        // Calculate min/max values for quantization
        let min_value = distances.fold(Float::INFINITY, |acc, &x| acc.min(x));
        let max_value = distances.fold(Float::NEG_INFINITY, |acc, &x| acc.max(x));

        let (data, sparse_indices) = match method {
            CompressionMethod::Float16 => Self::compress_float16(distances)?,
            CompressionMethod::Quantized8Bit => {
                Self::compress_quantized_8bit(distances, min_value, max_value)?
            }
            CompressionMethod::Quantized4Bit => {
                Self::compress_quantized_4bit(distances, min_value, max_value)?
            }
            CompressionMethod::Sparse => {
                let threshold = sparse_threshold.unwrap_or(1.0);
                Self::compress_sparse(distances, threshold)?
            }
            CompressionMethod::Delta => Self::compress_delta(distances)?,
            CompressionMethod::Hybrid => Self::compress_hybrid(
                distances,
                min_value,
                max_value,
                sparse_threshold.unwrap_or(1.0),
            )?,
        };

        let compression_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let compressed_size =
            data.len() + sparse_indices.len() * std::mem::size_of::<(usize, usize)>();
        let mut stats = CompressionStats::new(original_size, compressed_size);
        stats.compression_time_ms = compression_time;

        Ok(Self {
            method,
            shape,
            data,
            min_value,
            max_value,
            sparse_threshold: sparse_threshold.unwrap_or(1.0),
            sparse_indices,
            stats,
        })
    }

    /// Decompress to get a full distance matrix
    pub fn decompress(&self) -> NeighborsResult<Array2<Float>> {
        let start_time = std::time::Instant::now();

        let result = match self.method {
            CompressionMethod::Float16 => Self::decompress_float16(&self.data, self.shape),
            CompressionMethod::Quantized8Bit => Self::decompress_quantized_8bit(
                &self.data,
                self.shape,
                self.min_value,
                self.max_value,
            ),
            CompressionMethod::Quantized4Bit => Self::decompress_quantized_4bit(
                &self.data,
                self.shape,
                self.min_value,
                self.max_value,
            ),
            CompressionMethod::Sparse => Self::decompress_sparse(
                &self.data,
                &self.sparse_indices,
                self.shape,
                self.sparse_threshold,
            ),
            CompressionMethod::Delta => Self::decompress_delta(&self.data, self.shape),
            CompressionMethod::Hybrid => Self::decompress_hybrid(
                &self.data,
                &self.sparse_indices,
                self.shape,
                self.min_value,
                self.max_value,
                self.sparse_threshold,
            ),
        };

        // Update decompression time (note: this is approximate since we can't modify self)
        let _decompression_time = start_time.elapsed().as_secs_f64() * 1000.0;

        result
    }

    /// Get a specific row from the compressed matrix without full decompression
    pub fn get_row(&self, row_idx: usize) -> NeighborsResult<Array1<Float>> {
        if row_idx >= self.shape.0 {
            return Err(NeighborsError::InvalidInput(format!(
                "Row index {} out of bounds",
                row_idx
            )));
        }

        match self.method {
            CompressionMethod::Float16 => Self::get_row_float16(&self.data, self.shape, row_idx),
            CompressionMethod::Quantized8Bit => Self::get_row_quantized_8bit(
                &self.data,
                self.shape,
                row_idx,
                self.min_value,
                self.max_value,
            ),
            CompressionMethod::Quantized4Bit => Self::get_row_quantized_4bit(
                &self.data,
                self.shape,
                row_idx,
                self.min_value,
                self.max_value,
            ),
            CompressionMethod::Sparse => Self::get_row_sparse(
                &self.data,
                &self.sparse_indices,
                self.shape,
                row_idx,
                self.sparse_threshold,
            ),
            CompressionMethod::Delta => {
                // For delta compression, we need to decompress from the beginning
                let full_matrix = self.decompress()?;
                Ok(full_matrix.row(row_idx).to_owned())
            }
            CompressionMethod::Hybrid => Self::get_row_hybrid(
                &self.data,
                &self.sparse_indices,
                self.shape,
                row_idx,
                self.min_value,
                self.max_value,
                self.sparse_threshold,
            ),
        }
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Get the shape of the original matrix
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the compression method used
    pub fn method(&self) -> CompressionMethod {
        self.method
    }

    // Compression implementations

    fn compress_float16(distances: &Array2<Float>) -> CompressedData {
        let mut data = Vec::with_capacity(distances.len() * 2);

        for &value in distances.iter() {
            let half = Float16::from_f32(value);
            data.extend_from_slice(bytes_of(&half));
        }

        Ok((data, Vec::new()))
    }

    fn compress_quantized_8bit(
        distances: &Array2<Float>,
        min_val: Float,
        max_val: Float,
    ) -> CompressedData {
        let range = max_val - min_val;
        let scale = if range > 0.0 { 255.0 / range } else { 0.0 };

        let data: Vec<u8> = distances
            .iter()
            .map(|&val| ((val - min_val) * scale).round().clamp(0.0, 255.0) as u8)
            .collect();

        Ok((data, Vec::new()))
    }

    fn compress_quantized_4bit(
        distances: &Array2<Float>,
        min_val: Float,
        max_val: Float,
    ) -> CompressedData {
        let range = max_val - min_val;
        let scale = if range > 0.0 { 15.0 / range } else { 0.0 };

        let quantized: Vec<u8> = distances
            .iter()
            .map(|&val| ((val - min_val) * scale).round().clamp(0.0, 15.0) as u8)
            .collect();

        // Pack two 4-bit values into each byte
        let mut data = Vec::with_capacity((quantized.len() + 1) / 2);
        for chunk in quantized.chunks(2) {
            let byte = if chunk.len() == 2 {
                (chunk[0] << 4) | chunk[1]
            } else {
                chunk[0] << 4
            };
            data.push(byte);
        }

        Ok((data, Vec::new()))
    }

    fn compress_sparse(distances: &Array2<Float>, threshold: Float) -> CompressedData {
        let mut data = Vec::new();
        let mut indices = Vec::new();

        for (i, row) in distances.axis_iter(Axis(0)).enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val <= threshold {
                    indices.push((i, j));
                    data.extend_from_slice(bytes_of(&val));
                }
            }
        }

        Ok((data, indices))
    }

    fn compress_delta(distances: &Array2<Float>) -> CompressedData {
        let mut data = Vec::new();
        let mut prev_row: Option<ArrayView1<Float>> = None;

        for row in distances.axis_iter(Axis(0)) {
            if let Some(prev) = prev_row {
                // Store differences from previous row
                for (&curr, &prev_val) in row.iter().zip(prev.iter()) {
                    let delta = curr - prev_val;
                    data.extend_from_slice(bytes_of(&delta));
                }
            } else {
                // Store first row as-is
                for &val in row.iter() {
                    data.extend_from_slice(bytes_of(&val));
                }
            }
            prev_row = Some(row);
        }

        Ok((data, Vec::new()))
    }

    fn compress_hybrid(
        distances: &Array2<Float>,
        min_val: Float,
        max_val: Float,
        threshold: Float,
    ) -> CompressedData {
        // Use sparse for small values, quantized for larger values
        let mut data = Vec::new();
        let mut indices = Vec::new();

        // First pass: identify sparse values
        for (i, row) in distances.axis_iter(Axis(0)).enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val <= threshold {
                    indices.push((i, j));
                }
            }
        }

        // Second pass: compress based on classification
        let range = max_val - min_val;
        let scale = if range > 0.0 { 255.0 / range } else { 0.0 };

        for row in distances.axis_iter(Axis(0)) {
            for &val in row.iter() {
                if val <= threshold {
                    // Store as full precision for sparse values
                    data.extend_from_slice(bytes_of(&val));
                } else {
                    // Store as quantized for dense values
                    let quantized = ((val - min_val) * scale).round().clamp(0.0, 255.0) as u8;
                    data.push(quantized);
                }
            }
        }

        Ok((data, indices))
    }

    // Decompression implementations

    fn decompress_float16(data: &[u8], shape: (usize, usize)) -> NeighborsResult<Array2<Float>> {
        let expected_len = shape.0 * shape.1 * 2;
        if data.len() != expected_len {
            return Err(NeighborsError::InvalidInput(format!(
                "Data length mismatch: expected {}, got {}",
                expected_len,
                data.len()
            )));
        }

        let half_slice: &[Float16] = cast_slice(data);
        let values: Vec<Float> = half_slice.iter().map(|h| h.to_f32()).collect();

        Array2::from_shape_vec(shape, values)
            .map_err(|e| NeighborsError::InvalidInput(format!("Shape error: {}", e)))
    }

    fn decompress_quantized_8bit(
        data: &[u8],
        shape: (usize, usize),
        min_val: Float,
        max_val: Float,
    ) -> NeighborsResult<Array2<Float>> {
        let expected_len = shape.0 * shape.1;
        if data.len() != expected_len {
            return Err(NeighborsError::InvalidInput(format!(
                "Data length mismatch: expected {}, got {}",
                expected_len,
                data.len()
            )));
        }

        let range = max_val - min_val;
        let scale = range / 255.0;

        let values: Vec<Float> = data
            .iter()
            .map(|&quantized| min_val + quantized as Float * scale)
            .collect();

        Array2::from_shape_vec(shape, values)
            .map_err(|e| NeighborsError::InvalidInput(format!("Shape error: {}", e)))
    }

    fn decompress_quantized_4bit(
        data: &[u8],
        shape: (usize, usize),
        min_val: Float,
        max_val: Float,
    ) -> NeighborsResult<Array2<Float>> {
        let total_elements = shape.0 * shape.1;
        let expected_len = (total_elements + 1) / 2;

        if data.len() != expected_len {
            return Err(NeighborsError::InvalidInput(format!(
                "Data length mismatch: expected {}, got {}",
                expected_len,
                data.len()
            )));
        }

        let range = max_val - min_val;
        let scale = range / 15.0;

        let mut values = Vec::with_capacity(total_elements);

        for &byte in data.iter() {
            let high_nibble = (byte >> 4) & 0x0f;
            let low_nibble = byte & 0x0f;

            values.push(min_val + high_nibble as Float * scale);
            if values.len() < total_elements {
                values.push(min_val + low_nibble as Float * scale);
            }
        }

        Array2::from_shape_vec(shape, values)
            .map_err(|e| NeighborsError::InvalidInput(format!("Shape error: {}", e)))
    }

    fn decompress_sparse(
        data: &[u8],
        indices: &[(usize, usize)],
        shape: (usize, usize),
        threshold: Float,
    ) -> NeighborsResult<Array2<Float>> {
        let mut result = Array2::from_elem(shape, threshold + 1.0); // Default to above threshold

        let values: &[Float] = cast_slice(data);

        if values.len() != indices.len() {
            return Err(NeighborsError::InvalidInput(
                "Sparse data and indices length mismatch".to_string(),
            ));
        }

        for (&(i, j), &val) in indices.iter().zip(values.iter()) {
            if i < shape.0 && j < shape.1 {
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }

    fn decompress_delta(data: &[u8], shape: (usize, usize)) -> NeighborsResult<Array2<Float>> {
        let values: &[Float] = cast_slice(data);
        let expected_len = shape.0 * shape.1;

        if values.len() != expected_len {
            return Err(NeighborsError::InvalidInput(format!(
                "Data length mismatch: expected {}, got {}",
                expected_len,
                values.len()
            )));
        }

        let mut result = Array2::zeros(shape);
        let mut value_idx = 0;

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                if i == 0 {
                    // First row stored as-is
                    result[[i, j]] = values[value_idx];
                } else {
                    // Subsequent rows stored as deltas
                    result[[i, j]] = result[[i - 1, j]] + values[value_idx];
                }
                value_idx += 1;
            }
        }

        Ok(result)
    }

    fn decompress_hybrid(
        data: &[u8],
        indices: &[(usize, usize)],
        shape: (usize, usize),
        min_val: Float,
        max_val: Float,
        _threshold: Float,
    ) -> NeighborsResult<Array2<Float>> {
        let mut result = Array2::zeros(shape);
        let sparse_count = indices.len();
        let _dense_count = shape.0 * shape.1 - sparse_count;

        // Build sparse indices set for fast lookup
        let sparse_set: HashMap<(usize, usize), usize> = indices
            .iter()
            .enumerate()
            .map(|(idx, &pos)| (pos, idx))
            .collect();

        let sparse_data: &[Float] =
            cast_slice(&data[..sparse_count * std::mem::size_of::<Float>()]);
        let dense_data = &data[sparse_count * std::mem::size_of::<Float>()..];

        let range = max_val - min_val;
        let scale = range / 255.0;
        let mut dense_idx = 0;

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                if let Some(&sparse_idx) = sparse_set.get(&(i, j)) {
                    // Sparse value (full precision)
                    result[[i, j]] = sparse_data[sparse_idx];
                } else {
                    // Dense value (quantized)
                    if dense_idx < dense_data.len() {
                        result[[i, j]] = min_val + dense_data[dense_idx] as Float * scale;
                        dense_idx += 1;
                    }
                }
            }
        }

        Ok(result)
    }

    // Row extraction implementations

    fn get_row_float16(
        data: &[u8],
        shape: (usize, usize),
        row_idx: usize,
    ) -> NeighborsResult<Array1<Float>> {
        let row_size = shape.1 * 2; // 2 bytes per Float16
        let start_idx = row_idx * row_size;
        let end_idx = start_idx + row_size;

        if end_idx > data.len() {
            return Err(NeighborsError::InvalidInput(
                "Row data out of bounds".to_string(),
            ));
        }

        let row_data = &data[start_idx..end_idx];
        let half_slice: &[Float16] = cast_slice(row_data);
        let values: Vec<Float> = half_slice.iter().map(|h| h.to_f32()).collect();

        Ok(Array1::from_vec(values))
    }

    fn get_row_quantized_8bit(
        data: &[u8],
        shape: (usize, usize),
        row_idx: usize,
        min_val: Float,
        max_val: Float,
    ) -> NeighborsResult<Array1<Float>> {
        let start_idx = row_idx * shape.1;
        let end_idx = start_idx + shape.1;

        if end_idx > data.len() {
            return Err(NeighborsError::InvalidInput(
                "Row data out of bounds".to_string(),
            ));
        }

        let range = max_val - min_val;
        let scale = range / 255.0;

        let values: Vec<Float> = data[start_idx..end_idx]
            .iter()
            .map(|&quantized| min_val + quantized as Float * scale)
            .collect();

        Ok(Array1::from_vec(values))
    }

    fn get_row_quantized_4bit(
        data: &[u8],
        shape: (usize, usize),
        row_idx: usize,
        min_val: Float,
        max_val: Float,
    ) -> NeighborsResult<Array1<Float>> {
        let elements_per_row = shape.1;
        let bytes_per_row = (elements_per_row + 1) / 2;
        let start_byte = row_idx * bytes_per_row;
        let end_byte = start_byte + bytes_per_row;

        if end_byte > data.len() {
            return Err(NeighborsError::InvalidInput(
                "Row data out of bounds".to_string(),
            ));
        }

        let range = max_val - min_val;
        let scale = range / 15.0;

        let mut values = Vec::with_capacity(elements_per_row);
        let row_data = &data[start_byte..end_byte];

        for &byte in row_data.iter() {
            let high_nibble = (byte >> 4) & 0x0f;
            values.push(min_val + high_nibble as Float * scale);

            if values.len() < elements_per_row {
                let low_nibble = byte & 0x0f;
                values.push(min_val + low_nibble as Float * scale);
            }
        }

        values.truncate(elements_per_row);
        Ok(Array1::from_vec(values))
    }

    fn get_row_sparse(
        data: &[u8],
        indices: &[(usize, usize)],
        shape: (usize, usize),
        row_idx: usize,
        threshold: Float,
    ) -> NeighborsResult<Array1<Float>> {
        let mut result = Array1::from_elem(shape.1, threshold + 1.0);
        let values: &[Float] = cast_slice(data);

        for (&(i, j), &val) in indices.iter().zip(values.iter()) {
            if i == row_idx && j < shape.1 {
                result[j] = val;
            }
        }

        Ok(result)
    }

    fn get_row_hybrid(
        data: &[u8],
        indices: &[(usize, usize)],
        shape: (usize, usize),
        row_idx: usize,
        min_val: Float,
        max_val: Float,
        _threshold: Float,
    ) -> NeighborsResult<Array1<Float>> {
        let mut result = Array1::zeros(shape.1);
        let sparse_count = indices.len();

        // Build sparse indices for this row
        let _row_sparse_indices: Vec<usize> = indices
            .iter()
            .enumerate()
            .filter_map(|(idx, &(i, j))| if i == row_idx { Some((idx, j)) } else { None })
            .map(|(sparse_idx, col)| (col, sparse_idx))
            .collect::<HashMap<usize, usize>>()
            .into_iter()
            .map(|(_col, sparse_idx)| sparse_idx)
            .collect();

        let sparse_data: &[Float] =
            cast_slice(&data[..sparse_count * std::mem::size_of::<Float>()]);
        let dense_data = &data[sparse_count * std::mem::size_of::<Float>()..];

        let range = max_val - min_val;
        let scale = range / 255.0;
        let mut dense_offset = 0;

        // Calculate dense offset for this row
        for i in 0..row_idx {
            for j in 0..shape.1 {
                if !indices.contains(&(i, j)) {
                    dense_offset += 1;
                }
            }
        }

        let mut current_dense_idx = dense_offset;

        for j in 0..shape.1 {
            if let Some(sparse_idx) = indices
                .iter()
                .position(|&(i, col)| i == row_idx && col == j)
            {
                // Sparse value
                result[j] = sparse_data[sparse_idx];
            } else {
                // Dense value
                if current_dense_idx < dense_data.len() {
                    result[j] = min_val + dense_data[current_dense_idx] as Float * scale;
                    current_dense_idx += 1;
                }
            }
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_float16_conversion() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 10.5, 1000.0];
        for val in values {
            let half = Float16::from_f32(val);
            let recovered = half.to_f32();
            // Allow for some precision loss in half precision
            assert_abs_diff_eq!(val, recovered, epsilon = 0.01);
        }
    }

    #[test]
    fn test_compressed_distance_matrix_float16() {
        let distances = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]);

        let compressed =
            CompressedDistanceMatrix::new(&distances, CompressionMethod::Float16, None).unwrap();

        let decompressed = compressed.decompress().unwrap();

        // Check approximate equality due to precision loss
        for ((i, j), &expected) in distances.indexed_iter() {
            assert_abs_diff_eq!(expected, decompressed[[i, j]], epsilon = 0.01);
        }

        // Check compression ratio
        let stats = compressed.stats();
        assert!(stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_compressed_distance_matrix_quantized() {
        let distances = arr2(&[
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.5, 1.0, 0.0],
        ]);

        let compressed =
            CompressedDistanceMatrix::new(&distances, CompressionMethod::Quantized8Bit, None)
                .unwrap();

        let decompressed = compressed.decompress().unwrap();

        // Check approximate equality due to quantization
        for ((i, j), &expected) in distances.indexed_iter() {
            let error = (expected - decompressed[[i, j]]).abs();
            assert!(error < 0.02); // Allow small quantization error
        }
    }

    #[test]
    fn test_compressed_distance_matrix_sparse() {
        let distances = arr2(&[[0.0, 0.5, 10.0], [0.5, 0.0, 15.0], [10.0, 15.0, 0.0]]);

        let compressed =
            CompressedDistanceMatrix::new(&distances, CompressionMethod::Sparse, Some(1.0))
                .unwrap();

        let decompressed = compressed.decompress().unwrap();

        // Small values should be preserved exactly
        assert_eq!(decompressed[[0, 0]], 0.0);
        assert_eq!(decompressed[[0, 1]], 0.5);
        assert_eq!(decompressed[[1, 0]], 0.5);
        assert_eq!(decompressed[[1, 1]], 0.0);

        // Large values should be set to default (above threshold)
        assert!(decompressed[[0, 2]] > 1.0);
        assert!(decompressed[[2, 0]] > 1.0);
    }

    #[test]
    fn test_compressed_distance_matrix_row_access() {
        let distances = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]);

        let compressed =
            CompressedDistanceMatrix::new(&distances, CompressionMethod::Float16, None).unwrap();

        let row = compressed.get_row(1).unwrap();

        assert_abs_diff_eq!(row[0], 1.0, epsilon = 0.01);
        assert_abs_diff_eq!(row[1], 0.0, epsilon = 0.01);
        assert_abs_diff_eq!(row[2], 1.5, epsilon = 0.01);
    }

    #[test]
    fn test_compression_stats() {
        let distances = arr2(&[
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.5, 1.0, 0.0],
        ]);

        let compressed =
            CompressedDistanceMatrix::new(&distances, CompressionMethod::Quantized8Bit, None)
                .unwrap();

        let stats = compressed.stats();
        assert!(stats.compression_ratio > 1.0);
        assert!(stats.original_size > stats.compressed_size);
        assert!(stats.compression_time_ms >= 0.0);
    }
}

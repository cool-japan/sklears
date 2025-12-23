//! Model compression techniques
//!
//! Implements various compression algorithms for reducing model size.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Run-length encoding (good for sparse data)
    RunLength,
    /// Dictionary-based compression
    Dictionary,
    /// Huffman coding
    Huffman,
}

/// Compressed data container
#[derive(Debug, Clone)]
pub struct CompressedData {
    /// Compressed bytes
    pub data: Vec<u8>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Compression algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Original size in bytes
    pub original_size: usize,
}

impl CompressedData {
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        self.original_size as f64 / self.data.len() as f64
    }

    /// Get compressed size
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }
}

/// Compress a 1D array
pub fn compress_array1(
    array: &Array1<f64>,
    algorithm: CompressionAlgorithm,
) -> SklResult<CompressedData> {
    let original_size = array.len() * std::mem::size_of::<f64>();

    match algorithm {
        CompressionAlgorithm::None => Ok(CompressedData {
            data: array_to_bytes(array),
            shape: vec![array.len()],
            algorithm,
            original_size,
        }),
        CompressionAlgorithm::RunLength => compress_run_length_1d(array),
        CompressionAlgorithm::Dictionary => {
            // Placeholder for dictionary compression
            Err(SklearsError::InvalidInput(
                "Dictionary compression not yet implemented".to_string(),
            ))
        }
        CompressionAlgorithm::Huffman => {
            // Placeholder for Huffman coding
            Err(SklearsError::InvalidInput(
                "Huffman compression not yet implemented".to_string(),
            ))
        }
    }
}

/// Compress a 2D array
pub fn compress_array2(
    array: &Array2<f64>,
    algorithm: CompressionAlgorithm,
) -> SklResult<CompressedData> {
    let (nrows, ncols) = array.dim();
    let original_size = nrows * ncols * std::mem::size_of::<f64>();

    match algorithm {
        CompressionAlgorithm::None => Ok(CompressedData {
            data: array2_to_bytes(array),
            shape: vec![nrows, ncols],
            algorithm,
            original_size,
        }),
        CompressionAlgorithm::RunLength => compress_run_length_2d(array),
        CompressionAlgorithm::Dictionary => Err(SklearsError::InvalidInput(
            "Dictionary compression not yet implemented".to_string(),
        )),
        CompressionAlgorithm::Huffman => Err(SklearsError::InvalidInput(
            "Huffman compression not yet implemented".to_string(),
        )),
    }
}

/// Decompress a 1D array
pub fn decompress_array1(compressed: &CompressedData) -> SklResult<Array1<f64>> {
    match compressed.algorithm {
        CompressionAlgorithm::None => bytes_to_array1(&compressed.data, compressed.shape[0]),
        CompressionAlgorithm::RunLength => decompress_run_length_1d(compressed),
        _ => Err(SklearsError::InvalidInput(format!(
            "Decompression not implemented for {:?}",
            compressed.algorithm
        ))),
    }
}

/// Decompress a 2D array
pub fn decompress_array2(compressed: &CompressedData) -> SklResult<Array2<f64>> {
    match compressed.algorithm {
        CompressionAlgorithm::None => {
            bytes_to_array2(&compressed.data, compressed.shape[0], compressed.shape[1])
        }
        CompressionAlgorithm::RunLength => decompress_run_length_2d(compressed),
        _ => Err(SklearsError::InvalidInput(format!(
            "Decompression not implemented for {:?}",
            compressed.algorithm
        ))),
    }
}

// Helper functions
fn array_to_bytes(array: &Array1<f64>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(array.len() * 8);
    for &val in array.iter() {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn array2_to_bytes(array: &Array2<f64>) -> Vec<u8> {
    let (nrows, ncols) = array.dim();
    let mut bytes = Vec::with_capacity(nrows * ncols * 8);
    for i in 0..nrows {
        for j in 0..ncols {
            bytes.extend_from_slice(&array[[i, j]].to_le_bytes());
        }
    }
    bytes
}

fn bytes_to_array1(bytes: &[u8], len: usize) -> SklResult<Array1<f64>> {
    if bytes.len() != len * 8 {
        return Err(SklearsError::InvalidInput(
            "Invalid compressed data length".to_string(),
        ));
    }

    let mut array = Array1::zeros(len);
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        array[i] = f64::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(array)
}

fn bytes_to_array2(bytes: &[u8], nrows: usize, ncols: usize) -> SklResult<Array2<f64>> {
    if bytes.len() != nrows * ncols * 8 {
        return Err(SklearsError::InvalidInput(
            "Invalid compressed data length".to_string(),
        ));
    }

    let mut array = Array2::zeros((nrows, ncols));
    for (idx, chunk) in bytes.chunks_exact(8).enumerate() {
        let i = idx / ncols;
        let j = idx % ncols;
        array[[i, j]] = f64::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(array)
}

// Run-length encoding implementation
fn compress_run_length_1d(array: &Array1<f64>) -> SklResult<CompressedData> {
    let original_size = array.len() * std::mem::size_of::<f64>();
    let mut compressed = Vec::new();

    if array.is_empty() {
        return Ok(CompressedData {
            data: compressed,
            shape: vec![0],
            algorithm: CompressionAlgorithm::RunLength,
            original_size,
        });
    }

    let mut current_val = array[0];
    let mut count: u32 = 1;

    for i in 1..array.len() {
        if (array[i] - current_val).abs() < 1e-15 {
            count += 1;
        } else {
            // Write value and count
            compressed.extend_from_slice(&current_val.to_le_bytes());
            compressed.extend_from_slice(&count.to_le_bytes());
            current_val = array[i];
            count = 1;
        }
    }

    // Write last run
    compressed.extend_from_slice(&current_val.to_le_bytes());
    compressed.extend_from_slice(&count.to_le_bytes());

    Ok(CompressedData {
        data: compressed,
        shape: vec![array.len()],
        algorithm: CompressionAlgorithm::RunLength,
        original_size,
    })
}

fn compress_run_length_2d(array: &Array2<f64>) -> SklResult<CompressedData> {
    let (nrows, ncols) = array.dim();
    let original_size = nrows * ncols * std::mem::size_of::<f64>();
    let compressed = Vec::new();

    if nrows == 0 || ncols == 0 {
        return Ok(CompressedData {
            data: compressed,
            shape: vec![0, 0],
            algorithm: CompressionAlgorithm::RunLength,
            original_size,
        });
    }

    // Flatten and compress
    let mut values = Vec::with_capacity(nrows * ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            values.push(array[[i, j]]);
        }
    }

    let flat_array = Array1::from_vec(values);
    let flat_compressed = compress_run_length_1d(&flat_array)?;

    Ok(CompressedData {
        data: flat_compressed.data,
        shape: vec![nrows, ncols],
        algorithm: CompressionAlgorithm::RunLength,
        original_size,
    })
}

fn decompress_run_length_1d(compressed: &CompressedData) -> SklResult<Array1<f64>> {
    let len = compressed.shape[0];
    let mut array = Vec::with_capacity(len);

    let mut offset = 0;
    while offset < compressed.data.len() {
        if offset + 12 > compressed.data.len() {
            return Err(SklearsError::InvalidInput(
                "Corrupted compressed data".to_string(),
            ));
        }

        let val = f64::from_le_bytes(compressed.data[offset..offset + 8].try_into().unwrap());
        let count =
            u32::from_le_bytes(compressed.data[offset + 8..offset + 12].try_into().unwrap());

        for _ in 0..count {
            array.push(val);
        }

        offset += 12;
    }

    if array.len() != len {
        return Err(SklearsError::InvalidInput(
            "Decompressed size mismatch".to_string(),
        ));
    }

    Ok(Array1::from_vec(array))
}

fn decompress_run_length_2d(compressed: &CompressedData) -> SklResult<Array2<f64>> {
    let nrows = compressed.shape[0];
    let ncols = compressed.shape[1];

    let flat_compressed = CompressedData {
        data: compressed.data.clone(),
        shape: vec![nrows * ncols],
        algorithm: compressed.algorithm,
        original_size: compressed.original_size,
    };

    let flat_array = decompress_run_length_1d(&flat_compressed)?;

    let mut array = Array2::zeros((nrows, ncols));
    for (idx, &val) in flat_array.iter().enumerate() {
        let i = idx / ncols;
        let j = idx % ncols;
        array[[i, j]] = val;
    }

    Ok(array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_compress_decompress_none() {
        let arr = array![1.0, 2.0, 3.0, 4.0];
        let compressed = compress_array1(&arr, CompressionAlgorithm::None).unwrap();
        let decompressed = decompress_array1(&compressed).unwrap();

        assert_eq!(arr, decompressed);
    }

    #[test]
    fn test_compress_decompress_run_length() {
        let arr = array![1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let compressed = compress_array1(&arr, CompressionAlgorithm::RunLength).unwrap();
        let decompressed = decompress_array1(&compressed).unwrap();

        assert_eq!(arr, decompressed);
        // Should be compressed since we have runs
        assert!(compressed.compressed_size() < compressed.original_size);
    }

    #[test]
    fn test_compress_array2() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let compressed = compress_array2(&arr, CompressionAlgorithm::None).unwrap();
        let decompressed = decompress_array2(&compressed).unwrap();

        assert_eq!(arr, decompressed);
    }

    #[test]
    fn test_compression_ratio() {
        let arr = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let compressed = compress_array1(&arr, CompressionAlgorithm::RunLength).unwrap();

        assert!(compressed.compression_ratio() > 1.0);
    }
}

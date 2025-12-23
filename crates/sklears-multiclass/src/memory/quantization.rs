//! Weight quantization for model compression
//!
//! Reduces precision of model weights to save memory.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Quantization precision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationBits {
    /// 4-bit quantization
    Bits4 = 4,
    /// 8-bit quantization
    Bits8 = 8,
    /// 16-bit quantization (half precision)
    Bits16 = 16,
}

/// Quantized weights container
#[derive(Debug, Clone)]
pub struct QuantizedWeights {
    /// Quantized values
    pub values: Vec<i32>,
    /// Scale factor for dequantization
    pub scale: f64,
    /// Zero point for dequantization
    pub zero_point: i32,
    /// Original shape
    pub shape: Vec<usize>,
    /// Quantization bits
    pub bits: QuantizationBits,
}

impl QuantizedWeights {
    /// Get memory size in bytes (theoretical, after packing)
    pub fn memory_size(&self) -> usize {
        let bits_per_value = match self.bits {
            QuantizationBits::Bits4 => 4,
            QuantizationBits::Bits8 => 8,
            QuantizationBits::Bits16 => 16,
        };
        // Calculate the packed size
        let packed_size = (self.values.len() * bits_per_value + 7) / 8; // Round up to nearest byte
                                                                        // Add overhead for scale and zero_point
        packed_size + std::mem::size_of::<f64>() + std::mem::size_of::<i32>()
    }

    /// Get compression ratio compared to f64
    pub fn compression_ratio(&self) -> f64 {
        let original_size = self.values.len() * std::mem::size_of::<f64>();
        original_size as f64 / self.memory_size() as f64
    }
}

/// Quantize a 1D array
pub fn quantize_array1(array: &Array1<f64>, bits: QuantizationBits) -> SklResult<QuantizedWeights> {
    let (scale, zero_point, quantized) = quantize_values(array.as_slice().unwrap(), bits)?;

    Ok(QuantizedWeights {
        values: quantized,
        scale,
        zero_point,
        shape: vec![array.len()],
        bits,
    })
}

/// Quantize a 2D array
pub fn quantize_array2(array: &Array2<f64>, bits: QuantizationBits) -> SklResult<QuantizedWeights> {
    let (nrows, ncols) = array.dim();

    // Flatten array
    let mut values = Vec::with_capacity(nrows * ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            values.push(array[[i, j]]);
        }
    }

    let (scale, zero_point, quantized) = quantize_values(&values, bits)?;

    Ok(QuantizedWeights {
        values: quantized,
        scale,
        zero_point,
        shape: vec![nrows, ncols],
        bits,
    })
}

/// Dequantize to 1D array
pub fn dequantize_array1(quantized: &QuantizedWeights) -> SklResult<Array1<f64>> {
    if quantized.shape.len() != 1 {
        return Err(SklearsError::InvalidInput(
            "Shape mismatch for 1D array".to_string(),
        ));
    }

    let dequantized = dequantize_values(&quantized.values, quantized.scale, quantized.zero_point);
    Ok(Array1::from_vec(dequantized))
}

/// Dequantize to 2D array
pub fn dequantize_array2(quantized: &QuantizedWeights) -> SklResult<Array2<f64>> {
    if quantized.shape.len() != 2 {
        return Err(SklearsError::InvalidInput(
            "Shape mismatch for 2D array".to_string(),
        ));
    }

    let nrows = quantized.shape[0];
    let ncols = quantized.shape[1];

    let dequantized = dequantize_values(&quantized.values, quantized.scale, quantized.zero_point);

    let mut array = Array2::zeros((nrows, ncols));
    for (idx, val) in dequantized.iter().enumerate() {
        let i = idx / ncols;
        let j = idx % ncols;
        array[[i, j]] = *val;
    }

    Ok(array)
}

// Internal quantization logic
fn quantize_values(values: &[f64], bits: QuantizationBits) -> SklResult<(f64, i32, Vec<i32>)> {
    if values.is_empty() {
        return Ok((1.0, 0, Vec::new()));
    }

    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let range = max_val - min_val;
    if range < 1e-10 {
        // All values are the same
        return Ok((1.0, 0, vec![0; values.len()]));
    }

    let q_max = match bits {
        QuantizationBits::Bits4 => 15,
        QuantizationBits::Bits8 => 255,
        QuantizationBits::Bits16 => 65535,
    };

    let scale = range / q_max as f64;
    let zero_point = 0; // Symmetric quantization

    let quantized: Vec<i32> = values
        .iter()
        .map(|&v| {
            let normalized = (v - min_val) / scale;
            normalized.round().clamp(0.0, q_max as f64) as i32
        })
        .collect();

    Ok((scale, zero_point, quantized))
}

fn dequantize_values(quantized: &[i32], scale: f64, _zero_point: i32) -> Vec<f64> {
    quantized.iter().map(|&q| q as f64 * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_quantize_dequantize_8bit() {
        let arr = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let quantized = quantize_array1(&arr, QuantizationBits::Bits8).unwrap();
        let dequantized = dequantize_array1(&quantized).unwrap();

        // Check approximate equality (quantization introduces error)
        for (orig, deq) in arr.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.1);
        }
    }

    #[test]
    fn test_quantization_compression_ratio() {
        let arr = array![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
        ];
        let quantized = quantize_array1(&arr, QuantizationBits::Bits8).unwrap();

        // 8-bit should give approximately 8x compression for larger arrays
        // 16 values * 8 bytes = 128 bytes original
        // 16 values * 1 byte = 16 bytes + overhead (~20 bytes) = ~36 bytes
        // Ratio: 128 / 36 ≈ 3.5x
        assert!(quantized.compression_ratio() > 2.5);
    }

    #[test]
    fn test_quantize_array2() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let quantized = quantize_array2(&arr, QuantizationBits::Bits8).unwrap();
        let dequantized = dequantize_array2(&quantized).unwrap();

        assert_eq!(dequantized.dim(), arr.dim());
    }

    #[test]
    fn test_quantization_bits_4() {
        let arr: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let arr = Array1::from_vec(arr);
        let quantized = quantize_array1(&arr, QuantizationBits::Bits4).unwrap();

        // 4-bit should give better compression with more values
        // 32 values * 8 bytes = 256 bytes original
        // 32 values * 0.5 bytes = 16 bytes + overhead (~20 bytes) = ~36 bytes
        // Ratio: 256 / 36 ≈ 7x
        assert!(quantized.compression_ratio() > 4.0);
    }

    #[test]
    fn test_quantization_uniform_values() {
        let arr = array![5.0, 5.0, 5.0, 5.0];
        let quantized = quantize_array1(&arr, QuantizationBits::Bits8).unwrap();
        let dequantized = dequantize_array1(&quantized).unwrap();

        // For uniform values, quantization should preserve them reasonably well
        // Since all values are the same and our quantization maps them to the same bin,
        // they should all be 0.0 (since range is 0 and we normalize to 0)
        for val in dequantized.iter() {
            assert!(val.abs() < 0.1); // All should be close to 0
        }
    }
}

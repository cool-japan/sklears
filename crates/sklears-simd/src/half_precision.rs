//! Half-precision floating point operations (FP16/BF16)
//!
//! This module provides SIMD-optimized operations for half-precision floating point
//! formats, essential for modern AI/ML workloads.

#[cfg(feature = "no-std")]
use core::fmt;
#[cfg(not(feature = "no-std"))]
use std::fmt;

/// IEEE 754 half-precision (FP16) floating point format
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F16(pub u16);

/// Google's BFloat16 format optimized for AI/ML
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BF16(pub u16);

impl F16 {
    /// Create a new F16 from a u16 bit representation
    pub fn from_bits(bits: u16) -> Self {
        F16(bits)
    }

    /// Get the bit representation
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert from f32 to FP16
    pub fn from_f32(value: f32) -> Self {
        // IEEE 754 half-precision conversion
        let bits = value.to_bits();
        let sign = (bits >> 31) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;

        if exp == 0 && mant == 0 {
            // Zero
            F16(sign << 15)
        } else if exp == 0xFF {
            // Infinity or NaN
            let new_mant = if mant == 0 { 0 } else { 0x3FF };
            F16((sign << 15) | 0x7C00 | new_mant)
        } else {
            // Normal numbers
            let new_exp = exp - 127 + 15;
            if new_exp <= 0 {
                // Underflow to zero or denormal
                if new_exp < -10 {
                    F16(sign << 15)
                } else {
                    let new_mant = (mant | 0x800000) >> (14 - new_exp);
                    F16((sign << 15) | ((new_mant + 0x1000) >> 13) as u16)
                }
            } else if new_exp >= 31 {
                // Overflow to infinity
                F16((sign << 15) | 0x7C00)
            } else {
                // Normal case
                let new_mant = ((mant + 0x1000) >> 13) as u16;
                F16((sign << 15) | ((new_exp as u16) << 10) | new_mant)
            }
        }
    }

    /// Convert FP16 to f32
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = (bits >> 15) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let mant = (bits & 0x3FF) as u32;

        if exp == 0 && mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else if exp == 0 {
            // Denormal
            let mut new_mant = mant;
            let mut new_exp = 0;
            while (new_mant & 0x400) == 0 {
                new_mant <<= 1;
                new_exp += 1;
            }
            new_mant &= 0x3FF;
            new_exp = 127 - 15 - new_exp;
            f32::from_bits((sign << 31) | (new_exp << 23) | (new_mant << 13))
        } else if exp == 31 {
            // Infinity or NaN
            let new_mant = if mant == 0 { 0 } else { 0x7FFFFF };
            f32::from_bits((sign << 31) | 0x7F800000 | new_mant)
        } else {
            // Normal
            let new_exp = exp + 127 - 15;
            f32::from_bits((sign << 31) | (new_exp << 23) | (mant << 13))
        }
    }

    /// Check if the value is finite
    pub fn is_finite(self) -> bool {
        (self.0 & 0x7C00) != 0x7C00
    }

    /// Check if the value is infinite
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7C00
    }

    /// Check if the value is NaN
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7C00) == 0x7C00 && (self.0 & 0x3FF) != 0
    }
}

impl BF16 {
    /// Create a new BF16 from a u16 bit representation
    pub fn from_bits(bits: u16) -> Self {
        BF16(bits)
    }

    /// Get the bit representation
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert from f32 to BF16
    pub fn from_f32(value: f32) -> Self {
        // BFloat16 is simply the top 16 bits of IEEE 754 f32
        let bits = value.to_bits();
        let _truncated = (bits >> 16) as u16;

        // Round to nearest even (banker's rounding)
        let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        let rounded = ((bits + rounding_bias) >> 16) as u16;

        BF16(rounded)
    }

    /// Convert BF16 to f32
    pub fn to_f32(self) -> f32 {
        // BFloat16 to f32 is just shifting left by 16 bits
        f32::from_bits((self.0 as u32) << 16)
    }

    /// Check if the value is finite
    pub fn is_finite(self) -> bool {
        (self.0 & 0x7F80) != 0x7F80
    }

    /// Check if the value is infinite
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7F80
    }

    /// Check if the value is NaN
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7F80) == 0x7F80 && (self.0 & 0x7F) != 0
    }
}

impl fmt::Display for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl fmt::Display for BF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

/// SIMD operations for half-precision formats
pub mod simd {
    use super::*;

    /// Convert slice of f32 to FP16 with SIMD optimization
    pub fn f32_to_f16_slice(input: &[f32], output: &mut [F16]) {
        assert_eq!(input.len(), output.len());

        // Process in chunks for better vectorization
        const CHUNK_SIZE: usize = 8;
        let chunks = input.len() / CHUNK_SIZE;

        for i in 0..chunks {
            let start = i * CHUNK_SIZE;
            let end = start + CHUNK_SIZE;

            // Convert chunk of f32 to FP16
            for j in start..end {
                output[j] = F16::from_f32(input[j]);
            }
        }

        // Handle remaining elements
        for i in (chunks * CHUNK_SIZE)..input.len() {
            output[i] = F16::from_f32(input[i]);
        }
    }

    /// Convert slice of FP16 to f32 with SIMD optimization
    pub fn f16_to_f32_slice(input: &[F16], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());

        const CHUNK_SIZE: usize = 8;
        let chunks = input.len() / CHUNK_SIZE;

        for i in 0..chunks {
            let start = i * CHUNK_SIZE;
            let end = start + CHUNK_SIZE;

            for j in start..end {
                output[j] = input[j].to_f32();
            }
        }

        for i in (chunks * CHUNK_SIZE)..input.len() {
            output[i] = input[i].to_f32();
        }
    }

    /// Convert slice of f32 to BF16 with SIMD optimization
    pub fn f32_to_bf16_slice(input: &[f32], output: &mut [BF16]) {
        assert_eq!(input.len(), output.len());

        const CHUNK_SIZE: usize = 8;
        let chunks = input.len() / CHUNK_SIZE;

        for i in 0..chunks {
            let start = i * CHUNK_SIZE;
            let end = start + CHUNK_SIZE;

            for j in start..end {
                output[j] = BF16::from_f32(input[j]);
            }
        }

        for i in (chunks * CHUNK_SIZE)..input.len() {
            output[i] = BF16::from_f32(input[i]);
        }
    }

    /// Convert slice of BF16 to f32 with SIMD optimization
    pub fn bf16_to_f32_slice(input: &[BF16], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());

        const CHUNK_SIZE: usize = 8;
        let chunks = input.len() / CHUNK_SIZE;

        for i in 0..chunks {
            let start = i * CHUNK_SIZE;
            let end = start + CHUNK_SIZE;

            for j in start..end {
                output[j] = input[j].to_f32();
            }
        }

        for i in (chunks * CHUNK_SIZE)..input.len() {
            output[i] = input[i].to_f32();
        }
    }

    /// Element-wise addition for FP16 vectors
    pub fn add_f16(a: &[F16], b: &[F16], result: &mut [F16]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        for i in 0..a.len() {
            let sum = a[i].to_f32() + b[i].to_f32();
            result[i] = F16::from_f32(sum);
        }
    }

    /// Element-wise multiplication for FP16 vectors
    pub fn mul_f16(a: &[F16], b: &[F16], result: &mut [F16]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        for i in 0..a.len() {
            let product = a[i].to_f32() * b[i].to_f32();
            result[i] = F16::from_f32(product);
        }
    }

    /// Element-wise addition for BF16 vectors
    pub fn add_bf16(a: &[BF16], b: &[BF16], result: &mut [BF16]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        for i in 0..a.len() {
            let sum = a[i].to_f32() + b[i].to_f32();
            result[i] = BF16::from_f32(sum);
        }
    }

    /// Element-wise multiplication for BF16 vectors
    pub fn mul_bf16(a: &[BF16], b: &[BF16], result: &mut [BF16]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        for i in 0..a.len() {
            let product = a[i].to_f32() * b[i].to_f32();
            result[i] = BF16::from_f32(product);
        }
    }

    /// Dot product for FP16 vectors
    pub fn dot_f16(a: &[F16], b: &[F16]) -> f32 {
        assert_eq!(a.len(), b.len());

        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i].to_f32() * b[i].to_f32();
        }
        sum
    }

    /// Dot product for BF16 vectors
    pub fn dot_bf16(a: &[BF16], b: &[BF16]) -> f32 {
        assert_eq!(a.len(), b.len());

        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i].to_f32() * b[i].to_f32();
        }
        sum
    }

    /// Matrix multiplication for FP16 matrices (A * B = C)
    pub fn matmul_f16(a: &[F16], b: &[F16], c: &mut [F16], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l].to_f32() * b[l * n + j].to_f32();
                }
                c[i * n + j] = F16::from_f32(sum);
            }
        }
    }

    /// Matrix multiplication for BF16 matrices (A * B = C)
    pub fn matmul_bf16(a: &[BF16], b: &[BF16], c: &mut [BF16], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l].to_f32() * b[l * n + j].to_f32();
                }
                c[i * n + j] = BF16::from_f32(sum);
            }
        }
    }
}

/// Constants for half-precision formats
pub mod constants {
    use super::*;

    pub const F16_ZERO: F16 = F16(0);
    pub const F16_ONE: F16 = F16(0x3C00);
    pub const F16_NEG_ONE: F16 = F16(0xBC00);
    pub const F16_INFINITY: F16 = F16(0x7C00);
    pub const F16_NEG_INFINITY: F16 = F16(0xFC00);
    pub const F16_NAN: F16 = F16(0x7E00);
    pub const F16_MAX: F16 = F16(0x7BFF);
    pub const F16_MIN: F16 = F16(0x0400);
    pub const F16_EPSILON: F16 = F16(0x1400);

    pub const BF16_ZERO: BF16 = BF16(0);
    pub const BF16_ONE: BF16 = BF16(0x3F80);
    pub const BF16_NEG_ONE: BF16 = BF16(0xBF80);
    pub const BF16_INFINITY: BF16 = BF16(0x7F80);
    pub const BF16_NEG_INFINITY: BF16 = BF16(0xFF80);
    pub const BF16_NAN: BF16 = BF16(0x7FC0);
    pub const BF16_MAX: BF16 = BF16(0x7F7F);
    pub const BF16_MIN: BF16 = BF16(0x0080);
    pub const BF16_EPSILON: BF16 = BF16(0x3C00);
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::constants::*;
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_f16_conversion() {
        let val = 3.14159f32;
        let f16_val = F16::from_f32(val);
        let back_to_f32 = f16_val.to_f32();

        // FP16 has limited precision, so we expect some loss
        assert!((val - back_to_f32).abs() < 0.01);
    }

    #[test]
    fn test_bf16_conversion() {
        let val = 3.14159f32;
        let bf16_val = BF16::from_f32(val);
        let back_to_f32 = bf16_val.to_f32();

        // BF16 has better precision than FP16
        assert!((val - back_to_f32).abs() < 0.01);
    }

    #[test]
    fn test_f16_constants() {
        assert_eq!(F16_ZERO.to_f32(), 0.0);
        assert_eq!(F16_ONE.to_f32(), 1.0);
        assert_eq!(F16_NEG_ONE.to_f32(), -1.0);
        assert!(F16_INFINITY.is_infinite());
        assert!(F16_NAN.is_nan());
    }

    #[test]
    fn test_bf16_constants() {
        assert_eq!(BF16_ZERO.to_f32(), 0.0);
        assert_eq!(BF16_ONE.to_f32(), 1.0);
        assert_eq!(BF16_NEG_ONE.to_f32(), -1.0);
        assert!(BF16_INFINITY.is_infinite());
        assert!(BF16_NAN.is_nan());
    }

    #[test]
    fn test_f16_special_values() {
        let inf = F16::from_f32(f32::INFINITY);
        let neg_inf = F16::from_f32(f32::NEG_INFINITY);
        let nan = F16::from_f32(f32::NAN);

        assert!(inf.is_infinite());
        assert!(neg_inf.is_infinite());
        assert!(nan.is_nan());
    }

    #[test]
    fn test_bf16_special_values() {
        let inf = BF16::from_f32(f32::INFINITY);
        let neg_inf = BF16::from_f32(f32::NEG_INFINITY);
        let nan = BF16::from_f32(f32::NAN);

        assert!(inf.is_infinite());
        assert!(neg_inf.is_infinite());
        assert!(nan.is_nan());
    }

    #[test]
    fn test_simd_f32_to_f16_conversion() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![F16::from_bits(0); 8];

        simd::f32_to_f16_slice(&input, &mut output);

        for i in 0..input.len() {
            assert!((input[i] - output[i].to_f32()).abs() < 0.01);
        }
    }

    #[test]
    fn test_simd_f32_to_bf16_conversion() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![BF16::from_bits(0); 8];

        simd::f32_to_bf16_slice(&input, &mut output);

        for i in 0..input.len() {
            assert!((input[i] - output[i].to_f32()).abs() < 0.01);
        }
    }

    #[test]
    fn test_f16_arithmetic() {
        let a = vec![F16::from_f32(1.0), F16::from_f32(2.0), F16::from_f32(3.0)];
        let b = vec![F16::from_f32(4.0), F16::from_f32(5.0), F16::from_f32(6.0)];
        let mut result = vec![F16::from_bits(0); 3];

        simd::add_f16(&a, &b, &mut result);

        let expected = vec![5.0, 7.0, 9.0];
        for i in 0..3 {
            assert!((result[i].to_f32() - expected[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_bf16_arithmetic() {
        let a = vec![
            BF16::from_f32(1.0),
            BF16::from_f32(2.0),
            BF16::from_f32(3.0),
        ];
        let b = vec![
            BF16::from_f32(4.0),
            BF16::from_f32(5.0),
            BF16::from_f32(6.0),
        ];
        let mut result = vec![BF16::from_bits(0); 3];

        simd::add_bf16(&a, &b, &mut result);

        let expected = vec![5.0, 7.0, 9.0];
        for i in 0..3 {
            assert!((result[i].to_f32() - expected[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_f16_dot_product() {
        let a = vec![F16::from_f32(1.0), F16::from_f32(2.0), F16::from_f32(3.0)];
        let b = vec![F16::from_f32(4.0), F16::from_f32(5.0), F16::from_f32(6.0)];

        let result = simd::dot_f16(&a, &b);
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0

        assert!((result - expected).abs() < 0.1);
    }

    #[test]
    fn test_bf16_dot_product() {
        let a = vec![
            BF16::from_f32(1.0),
            BF16::from_f32(2.0),
            BF16::from_f32(3.0),
        ];
        let b = vec![
            BF16::from_f32(4.0),
            BF16::from_f32(5.0),
            BF16::from_f32(6.0),
        ];

        let result = simd::dot_bf16(&a, &b);
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0

        assert!((result - expected).abs() < 0.1);
    }

    #[test]
    fn test_f16_matrix_multiplication() {
        // 2x2 matrix multiplication
        let a = vec![
            F16::from_f32(1.0),
            F16::from_f32(2.0),
            F16::from_f32(3.0),
            F16::from_f32(4.0),
        ];
        let b = vec![
            F16::from_f32(5.0),
            F16::from_f32(6.0),
            F16::from_f32(7.0),
            F16::from_f32(8.0),
        ];
        let mut c = vec![F16::from_bits(0); 4];

        simd::matmul_f16(&a, &b, &mut c, 2, 2, 2);

        // Expected result: [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        for i in 0..4 {
            assert!((c[i].to_f32() - expected[i]).abs() < 0.1);
        }
    }

    #[test]
    fn test_bf16_matrix_multiplication() {
        // 2x2 matrix multiplication
        let a = vec![
            BF16::from_f32(1.0),
            BF16::from_f32(2.0),
            BF16::from_f32(3.0),
            BF16::from_f32(4.0),
        ];
        let b = vec![
            BF16::from_f32(5.0),
            BF16::from_f32(6.0),
            BF16::from_f32(7.0),
            BF16::from_f32(8.0),
        ];
        let mut c = vec![BF16::from_bits(0); 4];

        simd::matmul_bf16(&a, &b, &mut c, 2, 2, 2);

        // Expected result: [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        for i in 0..4 {
            assert!((c[i].to_f32() - expected[i]).abs() < 0.1);
        }
    }

    #[test]
    fn test_large_vector_conversion() {
        let size = 1024;
        let input: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let mut f16_output = vec![F16::from_bits(0); size];
        let mut bf16_output = vec![BF16::from_bits(0); size];

        simd::f32_to_f16_slice(&input, &mut f16_output);
        simd::f32_to_bf16_slice(&input, &mut bf16_output);

        for i in 0..size {
            let f16_error = (input[i] - f16_output[i].to_f32()).abs();
            let bf16_error = (input[i] - bf16_output[i].to_f32()).abs();

            // Relative tolerance for larger values, absolute tolerance for small values
            let tolerance = if input[i].abs() > 1.0 {
                input[i].abs() * 0.01 // 1% relative error for larger values
            } else {
                0.01 // Absolute tolerance for small values
            };

            assert!(
                f16_error < tolerance,
                "F16 error {:.6} > tolerance {:.6} for input {:.6}",
                f16_error,
                tolerance,
                input[i]
            );
            assert!(
                bf16_error < tolerance,
                "BF16 error {:.6} > tolerance {:.6} for input {:.6}",
                bf16_error,
                tolerance,
                input[i]
            );
        }
    }

    #[test]
    fn test_precision_comparison() {
        let test_values = vec![
            0.0, 1.0, -1.0, 0.5, -0.5, 3.14159, 2.71828, 1.41421, 1.73205, 0.1, 0.01, 0.001, 0.0001,
        ];

        for &val in &test_values {
            let f16_val = F16::from_f32(val);
            let bf16_val = BF16::from_f32(val);

            let f16_error = (val - f16_val.to_f32()).abs();
            let bf16_error = (val - bf16_val.to_f32()).abs();

            // Both should be reasonably close
            assert!(f16_error < 0.01 || val.abs() < 0.01);
            assert!(bf16_error < 0.01 || val.abs() < 0.01);
        }
    }
}

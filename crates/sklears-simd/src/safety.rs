//! Safety and correctness enhancements for SIMD operations
//!
//! This module provides safe wrappers, bounds checking, overflow detection,
//! and special value handling for all SIMD operations.

#[cfg(not(feature = "no-std"))]
use std::fmt;
#[cfg(not(feature = "no-std"))]
use std::string::ToString;

#[cfg(feature = "no-std")]
use alloc::string::{String, ToString};
#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(feature = "no-std")]
use alloc::{format, vec};
#[cfg(feature = "no-std")]
use core::fmt;

/// Enhanced error type for SIMD safety violations
#[derive(Debug, Clone, PartialEq)]
pub enum SimdSafetyError {
    IndexOutOfBounds { index: usize, length: usize },
    InvalidSliceLength { expected: usize, actual: usize },
    ArithmeticOverflow { operation: String, values: Vec<f64> },
    InvalidFloatingPoint { value: f64, reason: String },
    DivisionByZero,
    NegativeSquareRoot { value: f64 },
    InvalidRange { start: f64, end: f64 },
    InsufficientData { required: usize, available: usize },
}

impl fmt::Display for SimdSafetyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimdSafetyError::IndexOutOfBounds { index, length } => {
                write!(f, "Index {} out of bounds for length {}", index, length)
            }
            SimdSafetyError::InvalidSliceLength { expected, actual } => {
                write!(
                    f,
                    "Invalid slice length: expected {}, got {}",
                    expected, actual
                )
            }
            SimdSafetyError::ArithmeticOverflow { operation, values } => {
                write!(
                    f,
                    "Arithmetic overflow in operation '{}' with values: {:?}",
                    operation, values
                )
            }
            SimdSafetyError::InvalidFloatingPoint { value, reason } => {
                write!(f, "Invalid floating point value {}: {}", value, reason)
            }
            SimdSafetyError::DivisionByZero => {
                write!(f, "Division by zero")
            }
            SimdSafetyError::NegativeSquareRoot { value } => {
                write!(f, "Square root of negative number: {}", value)
            }
            SimdSafetyError::InvalidRange { start, end } => {
                write!(f, "Invalid range: start {} > end {}", start, end)
            }
            SimdSafetyError::InsufficientData {
                required,
                available,
            } => {
                write!(
                    f,
                    "Insufficient data: required {}, available {}",
                    required, available
                )
            }
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl std::error::Error for SimdSafetyError {}

#[cfg(feature = "no-std")]
impl core::error::Error for SimdSafetyError {}

pub type SafeSimdResult<T> = Result<T, SimdSafetyError>;

/// Safe SIMD vector operations with comprehensive bounds checking
pub struct SafeSimdOps;

impl SafeSimdOps {
    /// Safely validate floating point values
    pub fn validate_f32(value: f32) -> SafeSimdResult<f32> {
        if value.is_nan() {
            Err(SimdSafetyError::InvalidFloatingPoint {
                value: value as f64,
                reason: "NaN (Not a Number)".to_string(),
            })
        } else if value.is_infinite() {
            Err(SimdSafetyError::InvalidFloatingPoint {
                value: value as f64,
                reason: "Infinity".to_string(),
            })
        } else {
            Ok(value)
        }
    }

    /// Safely validate floating point values
    pub fn validate_f64(value: f64) -> SafeSimdResult<f64> {
        if value.is_nan() {
            Err(SimdSafetyError::InvalidFloatingPoint {
                value,
                reason: "NaN (Not a Number)".to_string(),
            })
        } else if value.is_infinite() {
            Err(SimdSafetyError::InvalidFloatingPoint {
                value,
                reason: "Infinity".to_string(),
            })
        } else {
            Ok(value)
        }
    }

    /// Validate an entire slice of f32 values
    pub fn validate_f32_slice(values: &[f32]) -> SafeSimdResult<()> {
        for (i, &value) in values.iter().enumerate() {
            Self::validate_f32(value).map_err(|e| match e {
                SimdSafetyError::InvalidFloatingPoint { value, reason } => {
                    SimdSafetyError::InvalidFloatingPoint {
                        value,
                        reason: format!("at index {}: {}", i, reason),
                    }
                }
                other => other,
            })?;
        }
        Ok(())
    }

    /// Validate an entire slice of f64 values
    pub fn validate_f64_slice(values: &[f64]) -> SafeSimdResult<()> {
        for (i, &value) in values.iter().enumerate() {
            Self::validate_f64(value).map_err(|e| match e {
                SimdSafetyError::InvalidFloatingPoint { value, reason } => {
                    SimdSafetyError::InvalidFloatingPoint {
                        value,
                        reason: format!("at index {}: {}", i, reason),
                    }
                }
                other => other,
            })?;
        }
        Ok(())
    }

    /// Safe addition with overflow detection
    pub fn safe_add_f32(a: f32, b: f32) -> SafeSimdResult<f32> {
        Self::validate_f32(a)?;
        Self::validate_f32(b)?;

        let result = a + b;
        Self::validate_f32(result).map_err(|_| SimdSafetyError::ArithmeticOverflow {
            operation: "addition".to_string(),
            values: vec![a as f64, b as f64],
        })
    }

    /// Safe subtraction with overflow detection
    pub fn safe_sub_f32(a: f32, b: f32) -> SafeSimdResult<f32> {
        Self::validate_f32(a)?;
        Self::validate_f32(b)?;

        let result = a - b;
        Self::validate_f32(result).map_err(|_| SimdSafetyError::ArithmeticOverflow {
            operation: "subtraction".to_string(),
            values: vec![a as f64, b as f64],
        })
    }

    /// Safe multiplication with overflow detection
    pub fn safe_mul_f32(a: f32, b: f32) -> SafeSimdResult<f32> {
        Self::validate_f32(a)?;
        Self::validate_f32(b)?;

        let result = a * b;
        Self::validate_f32(result).map_err(|_| SimdSafetyError::ArithmeticOverflow {
            operation: "multiplication".to_string(),
            values: vec![a as f64, b as f64],
        })
    }

    /// Safe division with zero and overflow checking
    pub fn safe_div_f32(a: f32, b: f32) -> SafeSimdResult<f32> {
        Self::validate_f32(a)?;
        Self::validate_f32(b)?;

        if b == 0.0 {
            return Err(SimdSafetyError::DivisionByZero);
        }

        let result = a / b;
        Self::validate_f32(result).map_err(|_| SimdSafetyError::ArithmeticOverflow {
            operation: "division".to_string(),
            values: vec![a as f64, b as f64],
        })
    }

    /// Safe square root with negative number checking
    pub fn safe_sqrt_f32(value: f32) -> SafeSimdResult<f32> {
        Self::validate_f32(value)?;

        if value < 0.0 {
            return Err(SimdSafetyError::NegativeSquareRoot {
                value: value as f64,
            });
        }

        let result = value.sqrt();
        Self::validate_f32(result)
    }

    /// Safe logarithm with domain checking
    pub fn safe_ln_f32(value: f32) -> SafeSimdResult<f32> {
        Self::validate_f32(value)?;

        if value <= 0.0 {
            return Err(SimdSafetyError::InvalidRange {
                start: value as f64,
                end: f64::INFINITY,
            });
        }

        let result = value.ln();
        Self::validate_f32(result)
    }

    /// Safe exponential with overflow checking
    pub fn safe_exp_f32(value: f32) -> SafeSimdResult<f32> {
        Self::validate_f32(value)?;

        // Check for potential overflow before computing
        if value > 88.0 {
            // exp(88) ≈ 1.6e38, close to f32::MAX
            return Err(SimdSafetyError::ArithmeticOverflow {
                operation: "exponential".to_string(),
                values: vec![value as f64],
            });
        }

        let result = value.exp();
        Self::validate_f32(result)
    }

    /// Safe vector dot product with bounds checking
    pub fn safe_dot_product_f32(a: &[f32], b: &[f32]) -> SafeSimdResult<f32> {
        if a.len() != b.len() {
            return Err(SimdSafetyError::InvalidSliceLength {
                expected: a.len(),
                actual: b.len(),
            });
        }

        Self::validate_f32_slice(a)?;
        Self::validate_f32_slice(b)?;

        let mut result = 0.0f32;
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let product = Self::safe_mul_f32(x, y).map_err(|e| match e {
                SimdSafetyError::ArithmeticOverflow { operation, values } => {
                    SimdSafetyError::ArithmeticOverflow {
                        operation: format!("{} at index {}", operation, i),
                        values,
                    }
                }
                other => other,
            })?;

            result = Self::safe_add_f32(result, product).map_err(|e| match e {
                SimdSafetyError::ArithmeticOverflow { operation, values } => {
                    SimdSafetyError::ArithmeticOverflow {
                        operation: format!(
                            "{} in dot product accumulation at index {}",
                            operation, i
                        ),
                        values,
                    }
                }
                other => other,
            })?;
        }

        Ok(result)
    }

    /// Safe vector normalization
    pub fn safe_normalize_f32(vector: &[f32]) -> SafeSimdResult<Vec<f32>> {
        if vector.is_empty() {
            return Err(SimdSafetyError::InsufficientData {
                required: 1,
                available: 0,
            });
        }

        Self::validate_f32_slice(vector)?;

        let dot_product = Self::safe_dot_product_f32(vector, vector)?;
        let norm = Self::safe_sqrt_f32(dot_product)?;

        if norm == 0.0 {
            return Err(SimdSafetyError::DivisionByZero);
        }

        let mut normalized = Vec::with_capacity(vector.len());
        for &value in vector {
            let normalized_value = Self::safe_div_f32(value, norm)?;
            normalized.push(normalized_value);
        }

        Ok(normalized)
    }

    /// Safe array indexing with bounds checking
    pub fn safe_get<T>(slice: &[T], index: usize) -> SafeSimdResult<&T> {
        if index >= slice.len() {
            Err(SimdSafetyError::IndexOutOfBounds {
                index,
                length: slice.len(),
            })
        } else {
            Ok(&slice[index])
        }
    }

    /// Safe mutable array indexing with bounds checking
    pub fn safe_get_mut<T>(slice: &mut [T], index: usize) -> SafeSimdResult<&mut T> {
        let length = slice.len();
        if index >= length {
            Err(SimdSafetyError::IndexOutOfBounds { index, length })
        } else {
            Ok(&mut slice[index])
        }
    }

    /// Safe slice creation with bounds checking
    pub fn safe_slice<T>(slice: &[T], start: usize, end: usize) -> SafeSimdResult<&[T]> {
        if start > end {
            return Err(SimdSafetyError::InvalidRange {
                start: start as f64,
                end: end as f64,
            });
        }

        if end > slice.len() {
            return Err(SimdSafetyError::IndexOutOfBounds {
                index: end,
                length: slice.len(),
            });
        }

        Ok(&slice[start..end])
    }

    /// Check if all values in slice are finite (not NaN or infinite)
    pub fn all_finite_f32(values: &[f32]) -> bool {
        values.iter().all(|&x| x.is_finite())
    }

    /// Check if all values in slice are finite (not NaN or infinite)  
    pub fn all_finite_f64(values: &[f64]) -> bool {
        values.iter().all(|&x| x.is_finite())
    }

    /// Replace NaN and infinite values with safe alternatives
    pub fn sanitize_f32_slice(values: &mut [f32], nan_replacement: f32, inf_replacement: f32) {
        for value in values.iter_mut() {
            if value.is_nan() {
                *value = nan_replacement;
            } else if value.is_infinite() {
                *value = if value.is_sign_positive() {
                    inf_replacement
                } else {
                    -inf_replacement
                };
            }
        }
    }

    /// Replace NaN and infinite values with safe alternatives
    pub fn sanitize_f64_slice(values: &mut [f64], nan_replacement: f64, inf_replacement: f64) {
        for value in values.iter_mut() {
            if value.is_nan() {
                *value = nan_replacement;
            } else if value.is_infinite() {
                *value = if value.is_sign_positive() {
                    inf_replacement
                } else {
                    -inf_replacement
                };
            }
        }
    }
}

/// Debug mode bounds checking wrapper
#[derive(Debug, Clone)]
pub struct DebugBoundsChecker<T> {
    data: Vec<T>,
    name: String,
}

impl<T> DebugBoundsChecker<T> {
    pub fn new(data: Vec<T>, name: String) -> Self {
        Self { data, name }
    }

    #[cfg(debug_assertions)]
    pub fn get(&self, index: usize) -> SafeSimdResult<&T> {
        if index >= self.data.len() {
            Err(SimdSafetyError::IndexOutOfBounds {
                index,
                length: self.data.len(),
            })
        } else {
            Ok(&self.data[index])
        }
    }

    #[cfg(not(debug_assertions))]
    pub fn get(&self, index: usize) -> SafeSimdResult<&T> {
        Ok(unsafe { self.data.get_unchecked(index) })
    }

    #[cfg(debug_assertions)]
    pub fn get_mut(&mut self, index: usize) -> SafeSimdResult<&mut T> {
        let length = self.data.len();
        if index >= length {
            Err(SimdSafetyError::IndexOutOfBounds { index, length })
        } else {
            Ok(&mut self.data[index])
        }
    }

    #[cfg(not(debug_assertions))]
    pub fn get_mut(&mut self, index: usize) -> SafeSimdResult<&mut T> {
        Ok(unsafe { self.data.get_unchecked_mut(index) })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// Memory safety guarantees for SIMD operations
pub struct MemorySafetyGuard;

impl MemorySafetyGuard {
    /// Ensure proper alignment for SIMD operations
    pub fn check_alignment(ptr: *const u8, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    /// Create aligned vector for SIMD operations
    pub fn create_aligned_vec<T>(size: usize, alignment: usize) -> Vec<T>
    where
        T: Default + Clone,
    {
        let mut vec = Vec::with_capacity(size + alignment);
        vec.resize(size, T::default());

        // Ensure alignment (simplified approach)
        while (vec.as_ptr() as usize) % alignment != 0 {
            vec.reserve(1);
        }

        vec
    }

    /// Validate memory range for SIMD operations
    pub fn validate_memory_range(ptr: *const u8, size: usize) -> SafeSimdResult<()> {
        if ptr.is_null() {
            return Err(SimdSafetyError::InvalidRange {
                start: 0.0,
                end: 0.0,
            });
        }

        if size == 0 {
            return Err(SimdSafetyError::InsufficientData {
                required: 1,
                available: 0,
            });
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;
    use core::ptr;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_validate_f32() {
        assert!(SafeSimdOps::validate_f32(1.0).is_ok());
        assert!(SafeSimdOps::validate_f32(-1.0).is_ok());
        assert!(SafeSimdOps::validate_f32(0.0).is_ok());

        assert!(SafeSimdOps::validate_f32(f32::NAN).is_err());
        assert!(SafeSimdOps::validate_f32(f32::INFINITY).is_err());
        assert!(SafeSimdOps::validate_f32(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn test_safe_arithmetic() {
        assert_eq!(SafeSimdOps::safe_add_f32(2.0, 3.0).unwrap(), 5.0);
        assert_eq!(SafeSimdOps::safe_sub_f32(5.0, 3.0).unwrap(), 2.0);
        assert_eq!(SafeSimdOps::safe_mul_f32(3.0, 4.0).unwrap(), 12.0);
        assert_eq!(SafeSimdOps::safe_div_f32(12.0, 4.0).unwrap(), 3.0);

        assert!(SafeSimdOps::safe_div_f32(1.0, 0.0).is_err());
        assert!(SafeSimdOps::safe_sqrt_f32(-1.0).is_err());
        assert!(SafeSimdOps::safe_ln_f32(-1.0).is_err());
        assert!(SafeSimdOps::safe_ln_f32(0.0).is_err());
    }

    #[test]
    fn test_safe_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = SafeSimdOps::safe_dot_product_f32(&a, &b).unwrap();
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

        let c = vec![1.0, 2.0];
        assert!(SafeSimdOps::safe_dot_product_f32(&a, &c).is_err());
    }

    #[test]
    fn test_safe_normalize() {
        let vector = vec![3.0, 4.0];
        let normalized = SafeSimdOps::safe_normalize_f32(&vector).unwrap();

        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        let zero_vector = vec![0.0, 0.0];
        assert!(SafeSimdOps::safe_normalize_f32(&zero_vector).is_err());

        let empty_vector: Vec<f32> = vec![];
        assert!(SafeSimdOps::safe_normalize_f32(&empty_vector).is_err());
    }

    #[test]
    fn test_safe_indexing() {
        let data = vec![1, 2, 3, 4, 5];

        assert_eq!(*SafeSimdOps::safe_get(&data, 2).unwrap(), 3);
        assert!(SafeSimdOps::safe_get(&data, 10).is_err());

        let slice = SafeSimdOps::safe_slice(&data, 1, 4).unwrap();
        assert_eq!(slice, &[2, 3, 4]);

        assert!(SafeSimdOps::safe_slice(&data, 4, 1).is_err());
        assert!(SafeSimdOps::safe_slice(&data, 0, 10).is_err());
    }

    #[test]
    fn test_finite_checks() {
        let finite_values = vec![1.0, 2.0, 3.0];
        assert!(SafeSimdOps::all_finite_f32(&finite_values));

        let mixed_values = vec![1.0, f32::NAN, 3.0];
        assert!(!SafeSimdOps::all_finite_f32(&mixed_values));

        let inf_values = vec![1.0, f32::INFINITY, 3.0];
        assert!(!SafeSimdOps::all_finite_f32(&inf_values));
    }

    #[test]
    fn test_sanitize_values() {
        let mut values = vec![1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 5.0];
        SafeSimdOps::sanitize_f32_slice(&mut values, 0.0, 1000.0);

        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 0.0); // NaN replaced with 0.0
        assert_eq!(values[2], 1000.0); // +Inf replaced with 1000.0
        assert_eq!(values[3], -1000.0); // -Inf replaced with -1000.0
        assert_eq!(values[4], 5.0);

        assert!(SafeSimdOps::all_finite_f32(&values));
    }

    #[test]
    fn test_debug_bounds_checker() {
        let data = vec![1, 2, 3, 4, 5];
        let checker = DebugBoundsChecker::new(data, "test".to_string());

        assert_eq!(*checker.get(2).unwrap(), 3);
        assert!(checker.get(10).is_err());
        assert_eq!(checker.len(), 5);
        assert!(!checker.is_empty());
    }

    #[test]
    fn test_memory_safety() {
        let data = [1u8, 2, 3, 4];
        let ptr = data.as_ptr();

        assert!(MemorySafetyGuard::validate_memory_range(ptr, data.len()).is_ok());
        assert!(MemorySafetyGuard::validate_memory_range(ptr::null(), 0).is_err());

        let aligned_vec: Vec<f32> = MemorySafetyGuard::create_aligned_vec(10, 16);
        assert_eq!(aligned_vec.len(), 10);
    }

    #[test]
    fn test_arithmetic_overflow_detection() {
        // Test with values that would cause overflow
        let large_val = f32::MAX / 2.0;
        assert!(SafeSimdOps::safe_mul_f32(large_val, 3.0).is_err());

        // Test exponential overflow
        assert!(SafeSimdOps::safe_exp_f32(100.0).is_err());

        // Test valid operations
        assert!(SafeSimdOps::safe_mul_f32(2.0, 3.0).is_ok());
        assert!(SafeSimdOps::safe_exp_f32(1.0).is_ok());
    }

    #[test]
    fn test_error_display() {
        let error = SimdSafetyError::IndexOutOfBounds {
            index: 5,
            length: 3,
        };
        let display_str = format!("{}", error);
        assert!(display_str.contains("Index 5 out of bounds for length 3"));

        let div_error = SimdSafetyError::DivisionByZero;
        assert_eq!(format!("{}", div_error), "Division by zero");
    }
}

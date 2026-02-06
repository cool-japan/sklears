/// Core trait definitions for numeric types in machine learning operations
///
/// This module defines the fundamental trait bounds and capabilities required
/// for numeric types used throughout the sklears library.
use bytemuck::{Pod, Zeroable};
use scirs2_core::numeric::{Float as NumFloat, FromPrimitive, NumCast, One, ToPrimitive, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Core numeric trait bounds for machine learning operations with SIMD support
#[cfg(feature = "serde")]
pub trait Numeric:
    Copy
    + Clone
    + Send
    + Sync
    + std::fmt::Debug
    + std::fmt::Display
    + PartialEq
    + PartialOrd
    + Zero
    + One
    + NumCast
    + FromPrimitive
    + 'static
    + Pod
    + Zeroable
    + Serialize
    + for<'de> Deserialize<'de>
{
    /// Check if this type supports SIMD operations
    const SIMD_SUPPORTED: bool = false;

    /// Get the size in bytes for memory layout calculations
    const SIZE_BYTES: usize = std::mem::size_of::<Self>();

    /// Check if the value is approximately equal to zero with tolerance
    fn is_near_zero(self, tolerance: Self) -> bool
    where
        Self: PartialOrd;
}

/// Core numeric trait bounds for machine learning operations with SIMD support (no serde)
#[cfg(not(feature = "serde"))]
pub trait Numeric:
    Copy
    + Clone
    + Send
    + Sync
    + std::fmt::Debug
    + std::fmt::Display
    + PartialEq
    + PartialOrd
    + Zero
    + One
    + NumCast
    + FromPrimitive
    + 'static
    + Pod
    + Zeroable
{
    /// Check if this type supports SIMD operations
    const SIMD_SUPPORTED: bool = false;

    /// Get the size in bytes for memory layout calculations
    const SIZE_BYTES: usize = std::mem::size_of::<Self>();

    /// Check if the value is approximately equal to zero with tolerance
    fn is_near_zero(self, tolerance: Self) -> bool
    where
        Self: PartialOrd;
}

/// Floating point trait bounds for machine learning operations with enhanced constraints
#[cfg(feature = "serde")]
pub trait FloatBounds:
    Numeric
    + NumFloat
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::RemAssign
    + std::ops::Neg<Output = Self>
    + approx::AbsDiffEq<Epsilon = Self>
    + approx::RelativeEq<Epsilon = Self>
    + approx::UlpsEq<Epsilon = Self>
    + Serialize
    + for<'de> Deserialize<'de>
{
    /// Machine epsilon for this floating point type
    const EPSILON: Self;

    /// Minimum positive value
    const MIN_POSITIVE: Self;

    /// Maximum finite value
    const MAX_VALUE: Self;

    /// Value representing positive infinity
    const INFINITY: Self;

    /// Value representing negative infinity  
    const NEG_INFINITY: Self;

    /// Not-a-number value
    const NAN: Self;

    /// Check if value is safe for numerical computation (finite and not NaN)
    fn is_safe_for_computation(self) -> bool {
        self.is_finite() && !self.is_nan()
    }

    /// Clamp value to safe range for computation
    fn clamp_safe(self) -> Self {
        if self.is_nan() {
            Self::zero()
        } else if self == Self::INFINITY {
            Self::MAX_VALUE
        } else if self == Self::NEG_INFINITY {
            -Self::MAX_VALUE
        } else {
            self
        }
    }
}

/// Floating point trait bounds for machine learning operations (no serde)
#[cfg(not(feature = "serde"))]
pub trait FloatBounds:
    Numeric
    + NumFloat
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::RemAssign
    + std::ops::Neg<Output = Self>
    + approx::AbsDiffEq<Epsilon = Self>
    + approx::RelativeEq<Epsilon = Self>
    + approx::UlpsEq<Epsilon = Self>
{
    /// Machine epsilon for this floating point type
    const EPSILON: Self;

    /// Minimum positive value
    const MIN_POSITIVE: Self;

    /// Maximum finite value
    const MAX_VALUE: Self;

    /// Value representing positive infinity
    const INFINITY: Self;

    /// Value representing negative infinity  
    const NEG_INFINITY: Self;

    /// Not-a-number value
    const NAN: Self;

    /// Check if value is safe for numerical computation (finite and not NaN)
    fn is_safe_for_computation(self) -> bool {
        self.is_finite() && !self.is_nan()
    }

    /// Clamp value to safe range for computation
    fn clamp_safe(self) -> Self {
        if self.is_nan() {
            Self::zero()
        } else if self == Self::INFINITY {
            Self::MAX_VALUE
        } else if self == Self::NEG_INFINITY {
            -Self::MAX_VALUE
        } else {
            self
        }
    }
}

/// Integer trait bounds for machine learning operations
pub trait IntBounds:
    Numeric
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::RemAssign
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitXor<Output = Self>
    + std::ops::Shl<usize, Output = Self>
    + std::ops::Shr<usize, Output = Self>
    + std::ops::Not<Output = Self>
    + std::hash::Hash
    + Ord
    + Eq
{
    /// Maximum value for this integer type
    const MAX_VALUE: Self;

    /// Minimum value for this integer type
    const MIN_VALUE: Self;

    /// Number of bits in this integer type
    const BITS: u32;

    /// Check if the value is a power of two
    fn is_power_of_two(self) -> bool;

    /// Count the number of leading zeros
    fn leading_zeros(self) -> u32;

    /// Count the number of trailing zeros
    fn trailing_zeros(self) -> u32;
}

// Implementations for common numeric types

impl Numeric for f32 {
    const SIMD_SUPPORTED: bool = true;
    const SIZE_BYTES: usize = 4;

    fn is_near_zero(self, tolerance: Self) -> bool {
        self.abs() < tolerance
    }
}

impl Numeric for f64 {
    const SIMD_SUPPORTED: bool = true;
    const SIZE_BYTES: usize = 8;

    fn is_near_zero(self, tolerance: Self) -> bool {
        self.abs() < tolerance
    }
}

impl Numeric for i32 {
    const SIMD_SUPPORTED: bool = true;
    const SIZE_BYTES: usize = 4;

    fn is_near_zero(self, tolerance: Self) -> bool {
        self.abs() <= tolerance
    }
}

impl Numeric for i64 {
    const SIMD_SUPPORTED: bool = true;
    const SIZE_BYTES: usize = 8;

    fn is_near_zero(self, tolerance: Self) -> bool {
        self.abs() <= tolerance
    }
}

impl Numeric for usize {
    const SIMD_SUPPORTED: bool = true;
    const SIZE_BYTES: usize = std::mem::size_of::<usize>();

    fn is_near_zero(self, tolerance: Self) -> bool {
        self <= tolerance
    }
}

impl FloatBounds for f32 {
    const EPSILON: Self = f32::EPSILON;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const MAX_VALUE: Self = f32::MAX;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;
}

impl FloatBounds for f64 {
    const EPSILON: Self = f64::EPSILON;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const MAX_VALUE: Self = f64::MAX;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const NAN: Self = f64::NAN;
}

impl IntBounds for i32 {
    const MAX_VALUE: Self = i32::MAX;
    const MIN_VALUE: Self = i32::MIN;
    const BITS: u32 = 32;

    fn is_power_of_two(self) -> bool {
        self > 0 && (self & (self - 1)) == 0
    }

    fn leading_zeros(self) -> u32 {
        self.leading_zeros()
    }

    fn trailing_zeros(self) -> u32 {
        self.trailing_zeros()
    }
}

impl IntBounds for i64 {
    const MAX_VALUE: Self = i64::MAX;
    const MIN_VALUE: Self = i64::MIN;
    const BITS: u32 = 64;

    fn is_power_of_two(self) -> bool {
        self > 0 && (self & (self - 1)) == 0
    }

    fn leading_zeros(self) -> u32 {
        self.leading_zeros()
    }

    fn trailing_zeros(self) -> u32 {
        self.trailing_zeros()
    }
}

/// Trait for types that can be used as array indices
pub trait IndexType: IntBounds + ToPrimitive {
    /// Convert to usize for array indexing
    fn to_usize(self) -> Option<usize>;

    /// Convert from usize
    fn from_usize(value: usize) -> Option<Self>;
}

impl IndexType for i32 {
    fn to_usize(self) -> Option<usize> {
        if self >= 0 {
            Some(self as usize)
        } else {
            None
        }
    }

    fn from_usize(value: usize) -> Option<Self> {
        if value <= i32::MAX as usize {
            Some(value as i32)
        } else {
            None
        }
    }
}

impl IndexType for i64 {
    fn to_usize(self) -> Option<usize> {
        if self >= 0 && self <= usize::MAX as i64 {
            Some(self as usize)
        } else {
            None
        }
    }

    fn from_usize(value: usize) -> Option<Self> {
        Some(value as i64)
    }
}

/// Trait for numeric types that support aggregation operations
pub trait Aggregatable: Numeric {
    /// Sum of an iterator of values
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self;

    /// Product of an iterator of values
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self;

    /// Mean of an iterator of values
    fn mean<I: Iterator<Item = Self>>(iter: I) -> Option<Self>;
}

impl<T> Aggregatable for T
where
    T: Numeric + std::iter::Sum + std::iter::Product + std::ops::Div<Output = T>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.sum()
    }

    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.product()
    }

    fn mean<I: Iterator<Item = Self>>(iter: I) -> Option<Self> {
        let values: Vec<Self> = iter.collect();
        if values.is_empty() {
            None
        } else {
            let sum: Self = values.iter().copied().sum();
            let count = Self::from_usize(values.len())?;
            Some(sum / count)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_trait_f64() {
        assert_eq!(f64::SIZE_BYTES, 8);
        assert!(1.0_f64.is_near_zero(2.0));
        assert!(!1.0_f64.is_near_zero(0.5));
    }

    #[test]
    fn test_float_bounds_constants() {
        assert!(f64::INFINITY.is_infinite());
        assert!(f64::NEG_INFINITY.is_infinite());
        assert!(f64::NAN.is_nan());
    }

    #[test]
    fn test_int_bounds_i32() {
        assert_eq!(i32::BITS, 32);
        assert!(8_i32.is_power_of_two());
        assert!(!6_i32.is_power_of_two());
        assert_eq!(0x00FF_0000_i32.leading_zeros(), 8);
        assert_eq!(0xFF00_i32.trailing_zeros(), 8);
    }

    #[test]
    fn test_index_type_conversion() {
        assert_eq!(42_i32.to_usize(), Some(42));
        assert_eq!((-1_i32).to_usize(), None);
        assert_eq!(<i32 as IndexType>::from_usize(42), Some(42));
    }

    #[test]
    fn test_aggregatable() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = f64::sum(values.iter().copied());
        let product = f64::product(values.iter().copied());
        let mean = f64::mean(values.iter().copied());

        assert_eq!(sum, 15.0);
        assert_eq!(product, 120.0);
        assert_eq!(mean, Some(3.0));
    }
}

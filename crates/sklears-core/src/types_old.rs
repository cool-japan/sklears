use crate::error::Result;
use bytemuck::{Pod, Zeroable};
/// Common type aliases for sklears with enhanced functionality
use scirs2_core::ndarray::{
    Array1 as NdArray1, Array2 as NdArray2 as NdArrayView1 as NdArrayView2,
    ArrayViewMut1 as NdArrayViewMut1 as NdArrayViewMut2,
};
use scirs2_core::numeric::{Float as NumFloat, NumCast, One};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Core numeric trait bounds for machine learning operations with SIMD support
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
    + bytemuck::Pod
    + bytemuck::Zeroable
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
    + approx::AbsDiffEq<Epsilon = Self>
    + approx::RelativeEq<Epsilon = Self>
    + Default
    + std::iter::Sum
    + std::iter::Product
    + serde::Serialize
    + serde::Deserialize<'static>
{
    /// Machine epsilon for this floating point type
    const EPSILON: Self;

    /// Maximum safe value for calculations
    const MAX_SAFE: Self;

    /// Minimum safe positive value
    const MIN_POSITIVE: Self;

    /// Create from f64 with potential precision loss
    fn from_f64_lossy(value: f64) -> Self;

    /// Convert to f64 with potential precision loss
    fn to_f64_lossy(self) -> f64;

    /// Check if value is safe for numerical computations
    fn is_safe_for_computation(self) -> bool;

    /// Clamp value to safe range for computations
    fn clamp_safe(self) -> Self;
}

#[cfg(not(feature = "serde"))]
pub trait FloatBounds:
    Numeric
    + NumFloat
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + approx::AbsDiffEq<Epsilon = Self>
    + approx::RelativeEq<Epsilon = Self>
    + Default
    + std::iter::Sum
    + std::iter::Product
{
    /// Machine epsilon for this floating point type
    const EPSILON: Self;

    /// Maximum safe value for calculations
    const MAX_SAFE: Self;

    /// Minimum safe positive value
    const MIN_POSITIVE: Self;

    /// Create from f64 with potential precision loss
    fn from_f64_lossy(value: f64) -> Self;

    /// Convert to f64 with potential precision loss
    fn to_f64_lossy(self) -> f64;

    /// Check if value is safe for numerical computations
    fn is_safe_for_computation(self) -> bool;

    /// Clamp value to safe range for computations
    fn clamp_safe(self) -> Self;
}

/// Integer trait bounds for labels and indices
pub trait IntBounds:
    Numeric
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::hash::Hash
    + Eq
    + Ord
{
}

// Implement blanket implementations for common types
impl<T> Numeric for T
where
    T: Copy
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
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn is_near_zero(self, tolerance: Self) -> bool {
        if let (Some(val), Some(tol)) = (NumCast::from(self), NumCast::from(tolerance)) {
            let v: f64 = val;
            let t: f64 = tol;
            v.abs() < t
        } else {
            self == Self::zero()
        }
    }
}

// Concrete implementations for f32
impl FloatBounds for f32 {
    const EPSILON: Self = f32::EPSILON;
    const MAX_SAFE: Self = 1e30;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;

    fn from_f64_lossy(value: f64) -> Self {
        value as f32
    }

    fn to_f64_lossy(self) -> f64 {
        self as f64
    }

    fn is_safe_for_computation(self) -> bool {
        self.is_finite() && self.abs() <= Self::MAX_SAFE && self.abs() >= Self::MIN_POSITIVE
    }

    fn clamp_safe(self) -> Self {
        if !self.is_finite() {
            0.0
        } else if self.abs() > Self::MAX_SAFE {
            if self > 0.0 {
                Self::MAX_SAFE
            } else {
                -Self::MAX_SAFE
            }
        } else if self.abs() < Self::MIN_POSITIVE && self != 0.0 {
            if self > 0.0 {
                Self::MIN_POSITIVE
            } else {
                -Self::MIN_POSITIVE
            }
        } else {
            self
        }
    }
}

// Concrete implementations for f64
impl FloatBounds for f64 {
    const EPSILON: Self = f64::EPSILON;
    const MAX_SAFE: Self = 1e300;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;

    fn from_f64_lossy(value: f64) -> Self {
        value
    }

    fn to_f64_lossy(self) -> f64 {
        self
    }

    fn is_safe_for_computation(self) -> bool {
        self.is_finite() && self.abs() <= Self::MAX_SAFE && self.abs() >= Self::MIN_POSITIVE
    }

    fn clamp_safe(self) -> Self {
        if !self.is_finite() {
            0.0
        } else if self.abs() > Self::MAX_SAFE {
            if self > 0.0 {
                Self::MAX_SAFE
            } else {
                -Self::MAX_SAFE
            }
        } else if self.abs() < Self::MIN_POSITIVE && self != 0.0 {
            if self > 0.0 {
                Self::MIN_POSITIVE
            } else {
                -Self::MIN_POSITIVE
            }
        } else {
            self
        }
    }
}

impl<T> IntBounds for T where
    T: Numeric
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::hash::Hash
        + Eq
        + Ord
{
}

/// 1-dimensional array type
pub type Array1<T> = NdArray1<T>;

/// 2-dimensional array type  
pub type Array2<T> = NdArray2<T>;

/// 1-dimensional array view type
pub type ArrayView1<'a, T> = NdArrayView1<'a, T>;

/// 2-dimensional array view type
pub type ArrayView2<'a, T> = NdArrayView2<'a, T>;

/// 1-dimensional mutable array view type
pub type ArrayViewMut1<'a, T> = NdArrayViewMut1<'a, T>;

/// 2-dimensional mutable array view type
pub type ArrayViewMut2<'a, T> = NdArrayViewMut2<'a, T>;

/// Default floating point type
pub type Float = f64;

/// Default integer type for indices and labels
pub type Int = i32;

/// Type alias for feature matrices (samples x features)
/// Generic parameter T should implement FloatBounds for optimal functionality
pub type Features<T = Float> = Array2<T>;

/// Type alias for target vectors
/// Generic parameter T should implement Numeric for optimal functionality  
pub type Target<T = Float> = Array1<T>;

/// Type alias for sample weights
/// Generic parameter T should implement FloatBounds for optimal functionality
pub type SampleWeight<T = Float> = Array1<T>;

/// Type alias for predictions
/// Generic parameter T should implement Numeric for optimal functionality
pub type Predictions<T = Float> = Array1<T>;

/// Type alias for probability predictions (samples x classes)
/// Generic parameter T should implement FloatBounds for optimal functionality
pub type Probabilities<T = Float> = Array2<T>;

/// Type alias for cluster labels
/// Generic parameter T should implement IntBounds for optimal functionality
pub type Labels<T = Int> = Array1<T>;

/// Type alias for distances
/// Generic parameter T should implement FloatBounds for optimal functionality
pub type Distances<T = Float> = Array2<T>;

/// Type alias for similarity matrices
/// Generic parameter T should implement FloatBounds for optimal functionality
pub type Similarities<T = Float> = Array2<T>;

/// Zero-copy array types using Cow (Copy-on-Write) for efficient memory management
/// Features matrix that can be either owned or borrowed
pub type CowFeatures<'a, T = Float> = Cow<'a, Array2<T>>;

/// Target vector that can be either owned or borrowed
pub type CowTarget<'a, T = Float> = Cow<'a, Array1<T>>;

/// Predictions that can be either owned or borrowed
pub type CowPredictions<'a, T = Float> = Cow<'a, Array1<T>>;

/// Probabilities that can be either owned or borrowed
pub type CowProbabilities<'a, T = Float> = Cow<'a, Array2<T>>;

/// Sample weights that can be either owned or borrowed
pub type CowSampleWeight<'a, T = Float> = Cow<'a, Array1<T>>;

/// Labels that can be either owned or borrowed
pub type CowLabels<'a, T = Int> = Cow<'a, Array1<T>>;

/// Newtype wrappers for domain-specific values with compile-time safety
/// Wrapper for probability values ensuring they are in [0, 1] range
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability<T: FloatBounds>(T);

impl<T: FloatBounds> Probability<T> {
    /// Create a new probability value, panicking if not in [0, 1] range
    pub fn new(value: T) -> Self {
        assert!(
            value >= T::zero() && value <= T::one(),
            "Probability must be in [0, 1] range"
        );
        Self(value)
    }

    /// Create a new probability value without validation (use with caution)
    pub const fn new_unchecked(value: T) -> Self {
        Self(value)
    }

    /// Get the inner value
    pub const fn value(self) -> T {
        self.0
    }

    /// Common probability constants for f32
    pub const fn zero_f32() -> Probability<f32> {
        Probability(0.0)
    }

    pub const fn one_f32() -> Probability<f32> {
        Probability(1.0)
    }

    pub const fn half_f32() -> Probability<f32> {
        Probability(0.5)
    }

    /// Common probability constants for f64
    pub const fn zero_f64() -> Probability<f64> {
        Probability(0.0)
    }

    pub const fn one_f64() -> Probability<f64> {
        Probability(1.0)
    }

    pub const fn half_f64() -> Probability<f64> {
        Probability(0.5)
    }
}

/// Wrapper for feature count ensuring non-zero values
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeatureCount(usize);

impl FeatureCount {
    /// Create a new feature count, panicking if zero
    pub const fn new(count: usize) -> Self {
        assert!(count > 0, "Feature count must be positive");
        Self(count)
    }

    /// Create a new feature count without validation (use with caution)
    pub const fn new_unchecked(count: usize) -> Self {
        Self(count)
    }

    /// Get the inner value
    pub const fn value(self) -> usize {
        self.0
    }

    /// Common feature counts
    pub const fn one() -> Self {
        Self(1)
    }

    pub const fn two() -> Self {
        Self(2)
    }

    pub const fn three() -> Self {
        Self(3)
    }

    /// Check if this count matches a compile-time constant
    pub const fn matches(self, other: usize) -> bool {
        self.0 == other
    }

    /// Add two feature counts at compile time
    pub const fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Multiply by a constant at compile time
    pub const fn multiply(self, factor: usize) -> Self {
        Self(self.0 * factor)
    }
}

/// Wrapper for sample count ensuring non-zero values
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleCount(usize);

impl SampleCount {
    /// Create a new sample count, panicking if zero
    pub const fn new(count: usize) -> Self {
        assert!(count > 0, "Sample count must be positive");
        Self(count)
    }

    /// Create a new sample count without validation (use with caution)
    pub const fn new_unchecked(count: usize) -> Self {
        Self(count)
    }

    /// Get the inner value
    pub const fn value(self) -> usize {
        self.0
    }

    /// Common sample counts
    pub const fn one() -> Self {
        Self(1)
    }

    pub const fn ten() -> Self {
        Self(10)
    }

    pub const fn hundred() -> Self {
        Self(100)
    }

    pub const fn thousand() -> Self {
        Self(1000)
    }

    /// Check if this count matches a compile-time constant
    pub const fn matches(self, other: usize) -> bool {
        self.0 == other
    }

    /// Add two sample counts at compile time
    pub const fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Check if count is power of two (useful for batching)
    pub const fn is_power_of_two(self) -> bool {
        self.0 != 0 && (self.0 & (self.0 - 1)) == 0
    }

    /// Get the next power of two
    pub const fn next_power_of_two(self) -> Self {
        if self.0 <= 1 {
            Self(1)
        } else {
            Self(self.0.next_power_of_two())
        }
    }
}

/// Const generic array types for compile-time shape validation
/// Fixed-size feature array with compile-time dimension checking
#[derive(Debug, Clone, PartialEq)]
pub struct FixedFeatures<T: FloatBounds, const N: usize> {
    data: [T; N],
}

impl<T: FloatBounds, const N: usize> FixedFeatures<T, N> {
    /// Create new fixed features from array
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }

    /// Get number of features at compile time
    pub const fn feature_count() -> usize {
        N
    }

    /// Get the underlying data
    pub fn data(&self) -> &[T; N] {
        &self.data
    }

    /// Get mutable access to underlying data
    pub fn data_mut(&mut self) -> &mut [T; N] {
        &mut self.data
    }

    /// Convert to slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

/// Fixed-size sample array with compile-time dimension checking
#[derive(Debug, Clone, PartialEq)]
pub struct FixedSamples<T: Numeric, const M: usize, const N: usize> {
    data: [[T; N]; M],
}

impl<T: Numeric, const M: usize, const N: usize> FixedSamples<T, M, N> {
    /// Create new fixed samples from 2D array
    pub const fn new(data: [[T; N]; M]) -> Self {
        Self { data }
    }

    /// Get number of samples at compile time
    pub const fn sample_count() -> usize {
        M
    }

    /// Get number of features at compile time
    pub const fn feature_count() -> usize {
        N
    }

    /// Get shape as tuple
    pub const fn shape() -> (usize, usize) {
        (M, N)
    }

    /// Get reference to underlying data
    pub fn data(&self) -> &[[T; N]; M] {
        &self.data
    }

    /// Get sample at index
    pub fn sample(&self, idx: usize) -> Option<&[T; N]> {
        self.data.get(idx)
    }
}

/// Compile-time validated matrix dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape<const ROWS: usize, const COLS: usize>;

impl<const ROWS: usize, const COLS: usize> MatrixShape<ROWS, COLS> {
    /// Create new matrix shape validator
    pub const fn new() -> Self {
        Self
    }

    /// Get number of rows
    pub const fn rows() -> usize {
        ROWS
    }

    /// Get number of columns  
    pub const fn cols() -> usize {
        COLS
    }

    /// Check if shapes are compatible for matrix multiplication
    pub const fn can_multiply<const OTHER_COLS: usize>(
        &self,
        _other: MatrixShape<COLS, OTHER_COLS>,
    ) -> bool {
        true // Compile-time guaranteed by type system
    }

    /// Get resulting shape after matrix multiplication
    pub const fn multiply_shape<const OTHER_COLS: usize>(
        &self,
        _other: MatrixShape<COLS, OTHER_COLS>,
    ) -> MatrixShape<ROWS, OTHER_COLS> {
        MatrixShape::<ROWS, OTHER_COLS>
    }
}

/// Type-level dimension constraints for compile-time validation
pub mod dimension_constraints {
    /// Marker trait for positive dimensions
    pub trait PositiveDimension {
        const VALUE: usize;
    }

    /// Implement for non-zero usize values
    macro_rules! impl_positive_dimension {
        ($($n:expr),+) => {
            $(
                impl PositiveDimension for [(); $n] {
                    const VALUE: usize = $n;
                }
            )+
        };
    }

    impl_positive_dimension!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024);

    /// Trait for validating matrix dimensions at compile time
    pub trait ValidMatrixDimensions<const M: usize, const N: usize> {
        const IS_VALID: bool = M > 0 && N > 0;
    }

    impl<const M: usize, const N: usize> ValidMatrixDimensions<M, N> for () {}
}

/// Advanced type aliases with const generic support
/// Features matrix with compile-time dimension validation
pub type FixedFeaturesMatrix<T, const M: usize, const N: usize> = FixedSamples<T, M, N>;

/// Target vector with fixed size
pub type FixedTarget<T, const N: usize> = FixedFeatures<T, N>;

/// Prediction vector with fixed size
pub type FixedPredictions<T, const N: usize> = FixedFeatures<T, N>;

/// Type-safe wrappers for ML-specific values
/// Learning rate with compile-time bounds checking
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct LearningRate<T: FloatBounds>(T);

impl<T: FloatBounds> LearningRate<T> {
    /// Create learning rate with runtime validation
    pub fn new(value: T) -> Result<Self> {
        if value <= T::zero() {
            return Err(crate::error::SklearsError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if !value.is_finite() {
            return Err(crate::error::SklearsError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be finite".to_string(),
            });
        }
        Ok(Self(value))
    }

    /// Create learning rate without validation (use with caution)
    pub const fn new_unchecked(value: T) -> Self {
        Self(value)
    }

    /// Get the inner value
    pub const fn value(self) -> T {
        self.0
    }

    /// Common learning rate presets for f32
    pub const fn small_f32() -> LearningRate<f32> {
        LearningRate(0.001)
    }

    pub const fn medium_f32() -> LearningRate<f32> {
        LearningRate(0.01)
    }

    pub const fn large_f32() -> LearningRate<f32> {
        LearningRate(0.1)
    }

    /// Common learning rate presets for f64
    pub const fn small_f64() -> LearningRate<f64> {
        LearningRate(0.001)
    }

    pub const fn medium_f64() -> LearningRate<f64> {
        LearningRate(0.01)
    }

    pub const fn large_f64() -> LearningRate<f64> {
        LearningRate(0.1)
    }
}

/// Const implementations for specific float types
impl LearningRate<f32> {
    /// Scale learning rate by a constant factor at compile time for f32
    pub const fn scale(self, factor: f32) -> Self {
        Self(self.0 * factor)
    }

    /// Compare with compile-time threshold for f32
    pub const fn is_small(self) -> bool {
        self.0 < 0.01
    }

    /// Decay learning rate by a factor
    pub const fn decay(self, factor: f32) -> Self {
        Self(self.0 * factor)
    }
}

impl LearningRate<f64> {
    /// Scale learning rate by a constant factor at compile time for f64
    pub const fn scale(self, factor: f64) -> Self {
        Self(self.0 * factor)
    }

    /// Compare with compile-time threshold for f64
    pub const fn is_small(self) -> bool {
        self.0 < 0.01
    }

    /// Decay learning rate by a factor
    pub const fn decay(self, factor: f64) -> Self {
        Self(self.0 * factor)
    }
}

/// Regularization strength with type safety
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RegularizationStrength<T: FloatBounds>(T);

impl<T: FloatBounds> RegularizationStrength<T> {
    /// Create regularization strength with validation
    pub fn new(value: T) -> Result<Self> {
        if value < T::zero() {
            return Err(crate::error::SklearsError::InvalidParameter {
                name: "regularization_strength".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        if !value.is_finite() {
            return Err(crate::error::SklearsError::InvalidParameter {
                name: "regularization_strength".to_string(),
                reason: "must be finite".to_string(),
            });
        }
        Ok(Self(value))
    }

    /// Create without validation
    pub const fn new_unchecked(value: T) -> Self {
        Self(value)
    }

    /// Get the inner value
    pub const fn value(self) -> T {
        self.0
    }

    /// No regularization for f32
    pub const fn none_f32() -> RegularizationStrength<f32> {
        RegularizationStrength(0.0)
    }

    /// Weak regularization for f32
    pub const fn weak_f32() -> RegularizationStrength<f32> {
        RegularizationStrength(0.01)
    }

    /// Strong regularization for f32
    pub const fn strong_f32() -> RegularizationStrength<f32> {
        RegularizationStrength(1.0)
    }

    /// No regularization for f64
    pub const fn none_f64() -> RegularizationStrength<f64> {
        RegularizationStrength(0.0)
    }

    /// Weak regularization for f64
    pub const fn weak_f64() -> RegularizationStrength<f64> {
        RegularizationStrength(0.01)
    }

    /// Strong regularization for f64
    pub const fn strong_f64() -> RegularizationStrength<f64> {
        RegularizationStrength(1.0)
    }
}

/// Const implementations for specific RegularizationStrength types
impl RegularizationStrength<f32> {
    /// Scale regularization strength by a constant factor at compile time
    pub const fn scale(self, factor: f32) -> Self {
        Self(self.0 * factor)
    }

    /// Check if regularization is effectively disabled
    pub const fn is_none(self) -> bool {
        self.0 <= 1e-8
    }

    /// Check if regularization is strong
    pub const fn is_strong(self) -> bool {
        self.0 >= 0.5
    }
}

impl RegularizationStrength<f64> {
    /// Scale regularization strength by a constant factor at compile time
    pub const fn scale(self, factor: f64) -> Self {
        Self(self.0 * factor)
    }

    /// Check if regularization is effectively disabled
    pub const fn is_none(self) -> bool {
        self.0 <= 1e-15
    }

    /// Check if regularization is strong
    pub const fn is_strong(self) -> bool {
        self.0 >= 0.5
    }
}

/// Compile-time mathematical operations and constants
pub mod compile_time_math {
    /// Compile-time mathematical constants for f32
    pub mod f32_constants {
        pub const PI: f32 = std::f32::consts::PI;
        pub const E: f32 = std::f32::consts::E;
        pub const SQRT_2: f32 = std::f32::consts::SQRT_2;
        pub const LN_2: f32 = std::f32::consts::LN_2;
        pub const LN_10: f32 = std::f32::consts::LN_10;

        /// Common ML-specific constants
        pub const GOLDEN_RATIO: f32 = 1.618034;
        pub const EULER_MASCHERONI: f32 = 0.5772157;

        /// Default tolerances for numerical computations
        pub const DEFAULT_TOLERANCE: f32 = 1e-6;
        pub const STRICT_TOLERANCE: f32 = 1e-9;
        pub const LOOSE_TOLERANCE: f32 = 1e-3;
    }

    /// Compile-time mathematical constants for f64
    pub mod f64_constants {
        pub const PI: f64 = std::f64::consts::PI;
        pub const E: f64 = std::f64::consts::E;
        pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
        pub const LN_2: f64 = std::f64::consts::LN_2;
        pub const LN_10: f64 = std::f64::consts::LN_10;

        /// Common ML-specific constants
        pub const GOLDEN_RATIO: f64 = 1.618033988749894;
        pub const EULER_MASCHERONI: f64 = 0.5772156649015329;

        /// Default tolerances for numerical computations
        pub const DEFAULT_TOLERANCE: f64 = 1e-12;
        pub const STRICT_TOLERANCE: f64 = 1e-15;
        pub const LOOSE_TOLERANCE: f64 = 1e-6;
    }

    /// Compile-time power computation (limited to small integer exponents)
    pub const fn pow_u32_f32(base: f32, exp: u32) -> f32 {
        match exp {
            0 => 1.0,
            1 => base,
            2 => base * base,
            3 => base * base * base,
            4 => {
                let base2 = base * base;
                base2 * base2
            }
            5 => {
                let base2 = base * base;
                base2 * base2 * base
            }
            _ => {
                // For larger exponents, we would need more complex const implementations
                // For now, this covers common use cases
                let mut result = 1.0;
                let mut i = 0;
                while i < exp {
                    result *= base;
                    i += 1;
                }
                result
            }
        }
    }

    /// Compile-time power computation for f64
    pub const fn pow_u32_f64(base: f64, exp: u32) -> f64 {
        match exp {
            0 => 1.0,
            1 => base,
            2 => base * base,
            3 => base * base * base,
            4 => {
                let base2 = base * base;
                base2 * base2
            }
            5 => {
                let base2 = base * base;
                base2 * base2 * base
            }
            _ => {
                let mut result = 1.0;
                let mut i = 0;
                while i < exp {
                    result *= base;
                    i += 1;
                }
                result
            }
        }
    }

    /// Compile-time factorial computation (limited to prevent overflow)
    pub const fn factorial(n: u32) -> u64 {
        match n {
            0 | 1 => 1,
            2 => 2,
            3 => 6,
            4 => 24,
            5 => 120,
            6 => 720,
            7 => 5040,
            8 => 40320,
            9 => 362880,
            10 => 3628800,
            11 => 39916800,
            12 => 479001600,
            _ => {
                // For larger values, compute iteratively
                let mut result = 1u64;
                let mut i = 1u32;
                while i <= n && i <= 20 {
                    // Limit to prevent overflow
                    result *= i as u64;
                    i += 1;
                }
                result
            }
        }
    }

    /// Compile-time binomial coefficient computation
    pub const fn binomial_coefficient(n: u32, k: u32) -> u64 {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = if k > n - k { n - k } else { k }; // Take advantage of symmetry

        let mut result = 1u64;
        let mut i = 0u32;
        while i < k {
            result = result * (n - i) as u64 / (i + 1) as u64;
            i += 1;
        }
        result
    }

    /// Compile-time GCD computation
    pub const fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    /// Compile-time LCM computation
    pub const fn lcm(a: u64, b: u64) -> u64 {
        if a == 0 || b == 0 {
            0
        } else {
            (a / gcd(a, b)) * b
        }
    }
}

/// Zero-copy operations and utilities for efficient memory management
pub mod zero_copy {
    use super::*;
    use std::borrow::Cow;

    /// Trait for types that support zero-copy operations
    pub trait ZeroCopy<T: Clone> {
        /// Create a zero-copy view if possible, otherwise clone
        fn as_cow(&self) -> Cow<'_, T>;

        /// Convert to owned data, cloning if necessary
        fn into_owned(self) -> T
        where
            Self: Sized;

        /// Check if this is a borrowed reference
        fn is_borrowed(&self) -> bool;

        /// Check if this is owned data
        fn is_owned(&self) -> bool {
            !self.is_borrowed()
        }
    }

    /// Wrapper for efficient array operations with zero-copy semantics
    #[derive(Debug, Clone)]
    pub struct ZeroCopyArray<'a, T: Clone> {
        data: Cow<'a, T>,
    }

    impl<'a, T: Clone> ZeroCopyArray<'a, T> {
        /// Create from borrowed data (zero-copy)
        pub fn borrowed(data: &'a T) -> Self {
            Self {
                data: Cow::Borrowed(data),
            }
        }

        /// Create from owned data
        pub fn owned(data: T) -> Self {
            Self {
                data: Cow::Owned(data),
            }
        }

        /// Get a reference to the data
        pub fn as_ref(&self) -> &T {
            &self.data
        }

        /// Convert to owned data, cloning if necessary
        pub fn into_owned(self) -> T {
            self.data.into_owned()
        }

        /// Check if data is borrowed
        pub fn is_borrowed(&self) -> bool {
            matches!(self.data, Cow::Borrowed(_))
        }

        /// Clone data if it's shared, returning an owned version
        pub fn clone_if_shared(&mut self) -> &mut T {
            self.data.to_mut()
        }

        /// Apply a function to the data, potentially creating a new owned instance
        pub fn map<F, U>(self, f: F) -> ZeroCopyArray<'static, U>
        where
            F: FnOnce(T) -> U,
            U: Clone,
        {
            ZeroCopyArray::owned(f(self.into_owned()))
        }

        /// Try to apply an in-place operation without cloning
        pub fn try_modify_inplace<F>(&mut self, f: F) -> bool
        where
            F: FnOnce(&mut T),
        {
            if self.is_borrowed() {
                false // Cannot modify borrowed data in-place
            } else {
                f(self.data.to_mut());
                true
            }
        }
    }

    /// Zero-copy feature matrix wrapper
    pub type ZeroCopyFeatures<'a, T = Float> = ZeroCopyArray<'a, Array2<T>>;

    /// Zero-copy target vector wrapper
    pub type ZeroCopyTarget<'a, T = Float> = ZeroCopyArray<'a, Array1<T>>;

    /// Utilities for working with array views
    pub mod array_views {
        use super::*;

        /// Create a view from slice data
        pub fn slice_to_array_view1<T>(data: &[T]) -> ArrayView1<'_, T> {
            ArrayView1::from(data)
        }

        /// Create a 2D view from slice data with explicit shape
        pub fn slice_to_array_view2<T>(
            data: &[T],
            shape: (usize, usize),
        ) -> Option<ArrayView2<'_, T>> {
            if data.len() != shape.0 * shape.1 {
                return None;
            }
            Some(ArrayView2::from_shape(shape, data).expect("valid array shape"))
        }

        /// Create a mutable view from slice data
        pub fn slice_to_array_view_mut1<T>(data: &mut [T]) -> ArrayViewMut1<'_, T> {
            ArrayViewMut1::from(data)
        }

        /// Create a 2D mutable view from slice data with explicit shape
        pub fn slice_to_array_view_mut2<T>(
            data: &mut [T],
            shape: (usize, usize),
        ) -> Option<ArrayViewMut2<'_, T>> {
            if data.len() != shape.0 * shape.1 {
                return None;
            }
            Some(ArrayViewMut2::from_shape(shape, data).expect("valid array shape"))
        }

        /// Convert array to slice view (zero-copy when possible)
        pub fn array_as_slice<T>(array: &Array1<T>) -> &[T] {
            array.as_slice().unwrap_or_else(|| {
                // Fallback for non-contiguous arrays
                unsafe { std::slice::from_raw_parts(array.as_ptr(), array.len()) }
            })
        }

        /// Convert 2D array to slice view (zero-copy when possible)
        pub fn array2_as_slice<T>(array: &Array2<T>) -> Option<&[T]> {
            if array.is_standard_layout() {
                Some(array.as_slice().unwrap_or(&[]))
            } else {
                None // Cannot create contiguous slice view
            }
        }
    }

    /// Memory-efficient dataset operations
    pub mod dataset_ops {
        use super::*;

        /// Split data without copying using views
        pub fn split_features_view<'a, T>(
            features: &'a Array2<T>,
            indices: &[usize],
        ) -> Result<Vec<ArrayView1<'a, T>>> {
            let mut views = Vec::with_capacity(indices.len());
            for &idx in indices {
                if idx >= features.nrows() {
                    return Err(crate::error::SklearsError::InvalidInput(format!(
                        "Index {} out of bounds for {} rows",
                        idx,
                        features.nrows()
                    )));
                }
                views.push(features.row(idx));
            }
            Ok(views)
        }

        /// Create batched views of data without copying
        pub fn batch_views<T>(data: &Array2<T>, batch_size: usize) -> Vec<ArrayView2<'_, T>> {
            let n_rows = data.nrows();
            let mut batches = Vec::new();

            for start in (0..n_rows).step_by(batch_size) {
                let end = std::cmp::min(start + batch_size, n_rows);
                let batch_view = data.slice(ndarray::s![start..end, ..]);
                batches.push(batch_view);
            }

            batches
        }

        /// Transpose view without copying data
        pub fn transpose_view<T>(data: &Array2<T>) -> ArrayView2<'_, T> {
            data.t()
        }
    }

    /// Cow-based dataset for optimal memory usage
    #[derive(Debug, Clone)]
    pub struct CowDataset<'a, X: Clone = Array2<Float>, Y: Clone = Array1<Float>> {
        /// Feature data (can be borrowed or owned)
        pub data: Cow<'a, X>,
        /// Target data (can be borrowed or owned)
        pub target: Cow<'a, Y>,
        /// Feature names
        pub feature_names: Vec<String>,
        /// Target names
        pub target_names: Option<Vec<String>>,
        /// Dataset description
        pub description: String,
    }

    impl<'a, X: Clone, Y: Clone> CowDataset<'a, X, Y> {
        /// Create dataset from borrowed data (zero-copy)
        pub fn borrowed(data: &'a X, target: &'a Y) -> Self {
            Self {
                data: Cow::Borrowed(data),
                target: Cow::Borrowed(target),
                feature_names: Vec::new(),
                target_names: None,
                description: String::new(),
            }
        }

        /// Create dataset from owned data
        pub fn owned(data: X, target: Y) -> Self {
            Self {
                data: Cow::Owned(data),
                target: Cow::Owned(target),
                feature_names: Vec::new(),
                target_names: None,
                description: String::new(),
            }
        }

        /// Convert to owned dataset, cloning if necessary
        pub fn into_owned(self) -> crate::dataset::Dataset<X, Y> {
            crate::dataset::Dataset {
                data: self.data.into_owned(),
                target: self.target.into_owned(),
                feature_names: self.feature_names,
                target_names: self.target_names,
                description: self.description,
            }
        }

        /// Check if data is borrowed
        pub fn is_data_borrowed(&self) -> bool {
            matches!(self.data, Cow::Borrowed(_))
        }

        /// Check if target is borrowed
        pub fn is_target_borrowed(&self) -> bool {
            matches!(self.target, Cow::Borrowed(_))
        }

        /// Get immutable reference to data
        pub fn data(&self) -> &X {
            &self.data
        }

        /// Get immutable reference to target
        pub fn target(&self) -> &Y {
            &self.target
        }

        /// Set feature names
        pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
            self.feature_names = names;
            self
        }

        /// Set target names
        pub fn with_target_names(mut self, names: Vec<String>) -> Self {
            self.target_names = Some(names);
            self
        }

        /// Set description
        pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
            self.description = description.into();
            self
        }
    }

    /// Memory pool for efficient array allocation
    #[derive(Debug)]
    pub struct ArrayPool<T> {
        pools: Vec<Vec<Vec<T>>>,
        max_size: usize,
    }

    impl<T: Default + Clone> ArrayPool<T> {
        /// Create a new array pool
        pub fn new(max_size: usize) -> Self {
            Self {
                pools: Vec::new(),
                max_size,
            }
        }

        /// Get a vector from the pool or allocate a new one
        pub fn get(&mut self, size: usize) -> Vec<T> {
            if size > self.max_size {
                // For very large arrays, don't pool them
                return vec![T::default(); size];
            }

            // Find appropriate pool bucket (power of 2 sizing)
            let bucket = size.next_power_of_two().trailing_zeros() as usize;

            // Ensure we have enough buckets
            while self.pools.len() <= bucket {
                self.pools.push(Vec::new());
            }

            // Try to reuse from pool
            if let Some(mut vec) = self.pools[bucket].pop() {
                vec.clear();
                vec.resize(size, T::default());
                vec
            } else {
                vec![T::default(); size]
            }
        }

        /// Return a vector to the pool for reuse
        pub fn return_vec(&mut self, mut vec: Vec<T>) {
            let capacity = vec.capacity();
            if capacity <= self.max_size && !vec.is_empty() {
                let bucket = capacity.next_power_of_two().trailing_zeros() as usize;
                if bucket < self.pools.len() {
                    vec.clear();
                    self.pools[bucket].push(vec);
                }
            }
            // If vec is too large or capacity doesn't fit, just drop it
        }

        /// Clear all pools
        pub fn clear(&mut self) {
            for pool in &mut self.pools {
                pool.clear();
            }
        }
    }
}

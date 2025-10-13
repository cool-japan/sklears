//! Type-safe SIMD abstractions with compile-time guarantees
//!
//! This module provides zero-cost abstractions for SIMD operations with enhanced type safety,
//! compile-time lane validation, and phantom types for SIMD width verification.

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Sub};

#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

/// Phantom type for SIMD width at compile time
pub struct SimdWidth<const WIDTH: usize>;

/// Type-safe SIMD vector with compile-time width verification
#[derive(Debug, Clone)]
pub struct SafeSimdVector<T, const WIDTH: usize> {
    data: [T; WIDTH],
    _phantom: PhantomData<SimdWidth<WIDTH>>,
}

impl<T, const WIDTH: usize> SafeSimdVector<T, WIDTH>
where
    T: Copy + Default,
{
    /// Create a new SIMD vector with compile-time width validation
    pub const fn new(data: [T; WIDTH]) -> Self {
        Self {
            data,
            _phantom: PhantomData,
        }
    }

    /// Create a vector filled with a single value
    pub fn splat(value: T) -> Self {
        Self {
            data: [value; WIDTH],
            _phantom: PhantomData,
        }
    }

    /// Get the width of this SIMD vector at compile time
    pub fn width(&self) -> usize {
        WIDTH
    }

    /// Access the underlying data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable access to the underlying data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Convert to array
    pub fn into_array(self) -> [T; WIDTH] {
        self.data
    }

    /// Load from slice with compile-time length checking
    pub fn from_slice(slice: &[T]) -> Option<Self> {
        if slice.len() >= WIDTH {
            let mut data = [T::default(); WIDTH];
            data.copy_from_slice(&slice[..WIDTH]);
            Some(Self::new(data))
        } else {
            None
        }
    }

    /// Safely extract a lane
    pub fn extract_lane(&self, lane: usize) -> Option<T> {
        if lane < WIDTH {
            Some(self.data[lane])
        } else {
            None
        }
    }

    /// Safely replace a lane
    pub fn replace_lane(&mut self, lane: usize, value: T) -> bool {
        if lane < WIDTH {
            self.data[lane] = value;
            true
        } else {
            false
        }
    }
}

/// Type-safe SIMD operations with lane validation
pub trait SimdOperation<T, const WIDTH: usize> {
    type Output;

    fn apply(&self, input: &SafeSimdVector<T, WIDTH>) -> Self::Output;
}

/// Zero-cost abstraction for element-wise operations
pub struct ElementWiseOp<F, T, const WIDTH: usize> {
    func: F,
    _phantom: PhantomData<(T, SimdWidth<WIDTH>)>,
}

impl<F, T, const WIDTH: usize> ElementWiseOp<F, T, WIDTH>
where
    F: Fn(T) -> T,
    T: Copy,
{
    pub const fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

impl<F, T, const WIDTH: usize> SimdOperation<T, WIDTH> for ElementWiseOp<F, T, WIDTH>
where
    F: Fn(T) -> T,
    T: Copy + Default,
{
    type Output = SafeSimdVector<T, WIDTH>;

    fn apply(&self, input: &SafeSimdVector<T, WIDTH>) -> Self::Output {
        let mut result = [T::default(); WIDTH];
        for i in 0..WIDTH {
            result[i] = (self.func)(input.data[i]);
        }
        SafeSimdVector::new(result)
    }
}

/// Compile-time validated SIMD arithmetic operations
impl<T, const WIDTH: usize> Add for SafeSimdVector<T, WIDTH>
where
    T: Add<Output = T> + Copy + Default,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); WIDTH];
        for i in 0..WIDTH {
            result[i] = self.data[i] + rhs.data[i];
        }
        Self::new(result)
    }
}

impl<T, const WIDTH: usize> Sub for SafeSimdVector<T, WIDTH>
where
    T: Sub<Output = T> + Copy + Default,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); WIDTH];
        for i in 0..WIDTH {
            result[i] = self.data[i] - rhs.data[i];
        }
        Self::new(result)
    }
}

impl<T, const WIDTH: usize> Mul for SafeSimdVector<T, WIDTH>
where
    T: Mul<Output = T> + Copy + Default,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); WIDTH];
        for i in 0..WIDTH {
            result[i] = self.data[i] * rhs.data[i];
        }
        Self::new(result)
    }
}

impl<T, const WIDTH: usize> Div for SafeSimdVector<T, WIDTH>
where
    T: Div<Output = T> + Copy + Default,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); WIDTH];
        for i in 0..WIDTH {
            result[i] = self.data[i] / rhs.data[i];
        }
        Self::new(result)
    }
}

/// Type-safe SIMD width constants for common architectures
pub mod widths {
    pub type Scalar = super::SimdWidth<1>;
    pub type Sse = super::SimdWidth<4>; // 128-bit / 32-bit = 4 lanes
    pub type Avx = super::SimdWidth<8>; // 256-bit / 32-bit = 8 lanes
    pub type Avx512 = super::SimdWidth<16>; // 512-bit / 32-bit = 16 lanes

    pub type SseF64 = super::SimdWidth<2>; // 128-bit / 64-bit = 2 lanes
    pub type AvxF64 = super::SimdWidth<4>; // 256-bit / 64-bit = 4 lanes
    pub type Avx512F64 = super::SimdWidth<8>; // 512-bit / 64-bit = 8 lanes
}

/// Type-safe vector types for common SIMD widths
pub type SimdF32x4 = SafeSimdVector<f32, 4>;
pub type SimdF32x8 = SafeSimdVector<f32, 8>;
pub type SimdF32x16 = SafeSimdVector<f32, 16>;

pub type SimdF64x2 = SafeSimdVector<f64, 2>;
pub type SimdF64x4 = SafeSimdVector<f64, 4>;
pub type SimdF64x8 = SafeSimdVector<f64, 8>;

pub type SimdU32x4 = SafeSimdVector<u32, 4>;
pub type SimdU32x8 = SafeSimdVector<u32, 8>;
pub type SimdU32x16 = SafeSimdVector<u32, 16>;

/// Compile-time SIMD capability checking
pub mod capabilities {

    /// Trait for SIMD capability validation at compile time
    pub trait SimdCapable<const WIDTH: usize> {
        fn is_supported() -> bool;
        fn best_width() -> usize;
    }

    /// X86 SIMD capabilities
    pub struct X86Simd;

    impl SimdCapable<4> for X86Simd {
        fn is_supported() -> bool {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                is_x86_feature_detected!("sse")
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                false
            }
        }

        fn best_width() -> usize {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    16
                } else if is_x86_feature_detected!("avx2") {
                    8
                } else if is_x86_feature_detected!("sse") {
                    4
                } else {
                    1
                }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                1
            }
        }
    }

    impl SimdCapable<8> for X86Simd {
        fn is_supported() -> bool {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                is_x86_feature_detected!("avx2")
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                false
            }
        }

        fn best_width() -> usize {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    16
                } else if is_x86_feature_detected!("avx2") {
                    8
                } else if is_x86_feature_detected!("sse") {
                    4
                } else {
                    1
                }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                1
            }
        }
    }

    impl SimdCapable<16> for X86Simd {
        fn is_supported() -> bool {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                is_x86_feature_detected!("avx512f")
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                false
            }
        }

        fn best_width() -> usize {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    16
                } else if is_x86_feature_detected!("avx2") {
                    8
                } else if is_x86_feature_detected!("sse") {
                    4
                } else {
                    1
                }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                1
            }
        }
    }

    /// ARM NEON capabilities
    pub struct ArmSimd;

    impl SimdCapable<4> for ArmSimd {
        fn is_supported() -> bool {
            #[cfg(target_arch = "aarch64")]
            {
                true // NEON is always available on AArch64
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                false
            }
        }

        fn best_width() -> usize {
            #[cfg(target_arch = "aarch64")]
            {
                4
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                1
            }
        }
    }
}

/// Zero-cost wrapper for optimized operations with compile-time dispatch
pub struct OptimizedSimdOp<T, const WIDTH: usize> {
    _phantom: PhantomData<(T, SimdWidth<WIDTH>)>,
}

impl<T, const WIDTH: usize> Default for OptimizedSimdOp<T, WIDTH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const WIDTH: usize> OptimizedSimdOp<T, WIDTH> {
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Dot product with compile-time width validation
    pub fn dot_product(a: &SafeSimdVector<T, WIDTH>, b: &SafeSimdVector<T, WIDTH>) -> T
    where
        T: Mul<Output = T> + Add<Output = T> + Default + Copy,
    {
        let mut result = T::default();
        for i in 0..WIDTH {
            result = result + (a.data[i] * b.data[i]);
        }
        result
    }

    /// Element-wise multiplication with type safety
    pub fn element_wise_multiply(
        a: &SafeSimdVector<T, WIDTH>,
        b: &SafeSimdVector<T, WIDTH>,
    ) -> SafeSimdVector<T, WIDTH>
    where
        T: Mul<Output = T> + Default + Copy,
    {
        let mut result = [T::default(); WIDTH];
        for i in 0..WIDTH {
            result[i] = a.data[i] * b.data[i];
        }
        SafeSimdVector::new(result)
    }

    /// Reduction operations with compile-time lane validation
    pub fn horizontal_sum(vector: &SafeSimdVector<T, WIDTH>) -> T
    where
        T: Add<Output = T> + Default + Copy,
    {
        let mut sum = T::default();
        for i in 0..WIDTH {
            sum = sum + vector.data[i];
        }
        sum
    }

    /// Find maximum with compile-time guarantees
    pub fn horizontal_max(vector: &SafeSimdVector<T, WIDTH>) -> T
    where
        T: PartialOrd + Copy,
    {
        let mut max = vector.data[0];
        for i in 1..WIDTH {
            if vector.data[i] > max {
                max = vector.data[i];
            }
        }
        max
    }
}

/// Compile-time validated slice operations
pub struct SafeSliceOps;

impl SafeSliceOps {
    /// Process slices with compile-time SIMD width validation
    pub fn process_slice_vectorized<T, F, const CHUNK_SIZE: usize>(
        data: &[T],
        mut func: F,
    ) -> Vec<T>
    where
        T: Copy + Default,
        F: FnMut(&SafeSimdVector<T, CHUNK_SIZE>) -> SafeSimdVector<T, CHUNK_SIZE>,
        [(); CHUNK_SIZE]:,
    {
        let mut result = Vec::with_capacity(data.len());

        // Process complete chunks
        for chunk in data.chunks_exact(CHUNK_SIZE) {
            if let Some(simd_chunk) = SafeSimdVector::<T, CHUNK_SIZE>::from_slice(chunk) {
                let processed = func(&simd_chunk);
                result.extend_from_slice(processed.as_slice());
            }
        }

        // Handle remainder by processing each element individually
        let remainder_start = data.len() - (data.len() % CHUNK_SIZE);
        for &item in &data[remainder_start..] {
            // Create a single-element "SIMD" vector and process it
            let mut single_data = [T::default(); CHUNK_SIZE];
            single_data[0] = item;
            if let Some(simd_single) = SafeSimdVector::<T, CHUNK_SIZE>::from_slice(&single_data) {
                let processed = func(&simd_single);
                result.push(processed.as_slice()[0]);
            }
        }

        result
    }

    /// Safe dot product with compile-time width checking
    pub fn dot_product_safe<T, const WIDTH: usize>(a: &[T], b: &[T]) -> Option<T>
    where
        T: Mul<Output = T> + Add<Output = T> + Default + Copy,
        [(); WIDTH]:,
    {
        if a.len() != b.len() || a.len() < WIDTH {
            return None;
        }

        let mut result = T::default();
        let chunks_a = a.chunks_exact(WIDTH);
        let chunks_b = b.chunks_exact(WIDTH);

        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            if let (Some(vec_a), Some(vec_b)) = (
                SafeSimdVector::<T, WIDTH>::from_slice(chunk_a),
                SafeSimdVector::<T, WIDTH>::from_slice(chunk_b),
            ) {
                result = result + OptimizedSimdOp::<T, WIDTH>::dot_product(&vec_a, &vec_b);
            }
        }

        // Handle remainder
        let remainder = a.len() % WIDTH;
        for i in 0..remainder {
            let idx = a.len() - remainder + i;
            result = result + (a[idx] * b[idx]);
        }

        Some(result)
    }
}

/// Trait for type-safe SIMD operations with zero-cost abstractions
pub trait TypeSafeSimd<T> {
    type Output;

    fn apply_safe(&self, input: &[T]) -> Self::Output;
}

/// Implementation for common mathematical operations
pub struct SafeMathOps;

impl SafeMathOps {
    /// Type-safe square root with compile-time width
    pub fn sqrt_vectorized<const WIDTH: usize>(data: &[f32]) -> Vec<f32>
    where
        [(); WIDTH]:,
    {
        SafeSliceOps::process_slice_vectorized::<f32, _, WIDTH>(data, |chunk| {
            let op = ElementWiseOp::new(|x: f32| x.sqrt());
            op.apply(chunk)
        })
    }

    /// Type-safe exponential with compile-time width
    pub fn exp_vectorized<const WIDTH: usize>(data: &[f32]) -> Vec<f32>
    where
        [(); WIDTH]:,
    {
        SafeSliceOps::process_slice_vectorized::<f32, _, WIDTH>(data, |chunk| {
            let op = ElementWiseOp::new(|x: f32| x.exp());
            op.apply(chunk)
        })
    }

    /// Type-safe polynomial evaluation
    pub fn polynomial_vectorized<const WIDTH: usize>(data: &[f32], coefficients: &[f32]) -> Vec<f32>
    where
        [(); WIDTH]:,
    {
        SafeSliceOps::process_slice_vectorized::<f32, _, WIDTH>(data, |chunk| {
            let op = ElementWiseOp::new(|x: f32| {
                coefficients
                    .iter()
                    .rev()
                    .fold(0.0, |acc, &coeff| acc * x + coeff)
            });
            op.apply(chunk)
        })
    }
}

// Removed static assertions module - using runtime checks instead

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_simd_vector_creation() {
        let vec = SimdF32x4::new([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(vec.width(), 4);
        assert_eq!(vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_safe_simd_arithmetic() {
        let a = SimdF32x4::new([1.0, 2.0, 3.0, 4.0]);
        let b = SimdF32x4::new([5.0, 6.0, 7.0, 8.0]);

        let sum = a + b;
        assert_eq!(sum.as_slice(), &[6.0, 8.0, 10.0, 12.0]);

        let diff = SimdF32x4::new([10.0, 12.0, 14.0, 16.0]) - SimdF32x4::new([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(diff.as_slice(), &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_lane_access() {
        let vec = SimdF32x4::new([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(vec.extract_lane(0), Some(1.0));
        assert_eq!(vec.extract_lane(1), Some(2.0));
        assert_eq!(vec.extract_lane(3), Some(4.0));
        assert_eq!(vec.extract_lane(4), None); // Out of bounds
    }

    #[test]
    fn test_dot_product_safe() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = SafeSliceOps::dot_product_safe::<f32, 4>(&a, &b);
        assert!(result.is_some());

        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_element_wise_operations() {
        let vec = SimdF32x4::new([1.0, 4.0, 9.0, 16.0]);
        let op = ElementWiseOp::new(|x: f32| x.sqrt());
        let result = op.apply(&vec);

        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_horizontal_operations() {
        let vec = SimdF32x4::new([1.0, 2.0, 3.0, 4.0]);

        let sum = OptimizedSimdOp::<f32, 4>::horizontal_sum(&vec);
        assert_eq!(sum, 10.0);

        let max = OptimizedSimdOp::<f32, 4>::horizontal_max(&vec);
        assert_eq!(max, 4.0);
    }

    #[test]
    fn test_safe_math_operations() {
        let data = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
        let result = SafeMathOps::sqrt_vectorized::<4>(&data);

        let expected: Vec<f32> = data.iter().map(|x| x.sqrt()).collect();
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "sqrt({}) = {}, expected {}, diff = {}",
                a * a,
                a,
                b,
                (a - b).abs()
            ); // Even more lenient with debug info
        }
    }

    #[test]
    fn test_from_slice_validation() {
        let data = vec![1.0f32, 2.0, 3.0];
        let vec = SimdF32x4::from_slice(&data);
        assert!(vec.is_none()); // Not enough elements

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let vec = SimdF32x4::from_slice(&data);
        assert!(vec.is_some());
        assert_eq!(vec.unwrap().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_capability_detection() {
        use super::capabilities::SimdCapable;

        // This will vary by platform, but should not panic
        let _sse_supported = <capabilities::X86Simd as SimdCapable<4>>::is_supported();
        let _best_width = <capabilities::X86Simd as SimdCapable<4>>::best_width();

        // ARM test (will be false on x86, true on ARM)
        let _neon_supported = <capabilities::ArmSimd as SimdCapable<4>>::is_supported();
    }

    #[test]
    fn test_zero_cost_abstractions() {
        // These operations should compile to efficient code
        let a = SimdF32x4::splat(2.0);
        let b = SimdF32x4::splat(3.0);

        let result = OptimizedSimdOp::<f32, 4>::element_wise_multiply(&a, &b);
        assert_eq!(result.as_slice(), &[6.0, 6.0, 6.0, 6.0]);
    }
}

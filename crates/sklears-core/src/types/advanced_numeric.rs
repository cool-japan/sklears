/// Advanced numeric traits and implementations for high-performance machine learning
///
/// This module provides specialized numeric traits and implementations that extend
/// the basic numeric capabilities with advanced features like SIMD operations,
/// GPU compatibility, complex number support, and memory-efficient operations.
use crate::types::FloatBounds;
use std::ops::{Add, Div, Mul, Sub};

/// Trait for complex number operations in machine learning contexts
pub trait ComplexOps<T>
where
    T: FloatBounds,
{
    /// Create a complex number from real and imaginary parts
    fn from_parts(real: T, imag: T) -> Self;

    /// Get the real part
    fn real(&self) -> T;

    /// Get the imaginary part
    fn imag(&self) -> T;

    /// Calculate the complex conjugate
    fn conj(&self) -> Self;

    /// Calculate the magnitude (absolute value)
    fn magnitude(&self) -> T;

    /// Calculate the phase (argument)
    fn phase(&self) -> T;

    /// Calculate the squared magnitude (norm squared)
    fn norm_sqr(&self) -> T;
}

/// Basic complex number implementation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<T> {
    pub real: T,
    pub imag: T,
}

impl<T> Complex<T>
where
    T: FloatBounds + Copy,
{
    /// Create a new complex number
    pub fn new(real: T, imag: T) -> Self {
        Self { real, imag }
    }

    /// Create a complex number from a real value
    pub fn from_real(real: T) -> Self {
        Self {
            real,
            imag: T::zero(),
        }
    }

    /// Create a complex number from an imaginary value
    pub fn from_imag(imag: T) -> Self {
        Self {
            real: T::zero(),
            imag,
        }
    }
}

impl<T> ComplexOps<T> for Complex<T>
where
    T: FloatBounds + Copy,
{
    fn from_parts(real: T, imag: T) -> Self {
        Self { real, imag }
    }

    fn real(&self) -> T {
        self.real
    }

    fn imag(&self) -> T {
        self.imag
    }

    fn conj(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }

    fn magnitude(&self) -> T {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    fn phase(&self) -> T {
        self.imag.atan2(self.real)
    }

    fn norm_sqr(&self) -> T {
        self.real * self.real + self.imag * self.imag
    }
}

// Arithmetic operations for Complex numbers
impl<T> Add for Complex<T>
where
    T: FloatBounds,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl<T> Sub for Complex<T>
where
    T: FloatBounds,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
}

impl<T> Mul for Complex<T>
where
    T: FloatBounds,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

impl<T> Div for Complex<T>
where
    T: FloatBounds,
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let norm_sqr = other.norm_sqr();
        let conj = other.conj();
        let num = self * conj;
        Self {
            real: num.real / norm_sqr,
            imag: num.imag / norm_sqr,
        }
    }
}

/// Trait for numeric conversions between different precision levels
pub trait NumericConversion<T> {
    /// Convert to higher precision if possible
    fn to_higher_precision(&self) -> Option<T>;

    /// Convert from higher precision with potential loss
    fn from_higher_precision(value: T) -> Self;

    /// Check if conversion would cause precision loss
    fn would_lose_precision(&self, target_type: std::marker::PhantomData<T>) -> bool;
}

/// Implementation for f32 to f64 conversion
impl NumericConversion<f64> for f32 {
    fn to_higher_precision(&self) -> Option<f64> {
        Some(*self as f64)
    }

    fn from_higher_precision(value: f64) -> Self {
        value as f32
    }

    fn would_lose_precision(&self, _target: std::marker::PhantomData<f64>) -> bool {
        false // f32 to f64 never loses precision
    }
}

/// Implementation for f64 to f32 conversion
impl NumericConversion<f32> for f64 {
    fn to_higher_precision(&self) -> Option<f32> {
        None // f64 is already higher precision
    }

    fn from_higher_precision(value: f32) -> Self {
        value as f64
    }

    fn would_lose_precision(&self, _target: std::marker::PhantomData<f32>) -> bool {
        self.abs() > f32::MAX as f64 || (self.abs() > 0.0 && self.abs() < f32::MIN_POSITIVE as f64)
    }
}

/// Trait for SIMD-optimized numeric operations
pub trait SimdOps<T> {
    /// The SIMD width for this type
    const SIMD_WIDTH: usize;

    /// Check if SIMD operations are available at runtime
    fn simd_available() -> bool;

    /// Perform SIMD addition
    fn simd_add(a: &[T], b: &[T], result: &mut [T]);

    /// Perform SIMD multiplication
    fn simd_mul(a: &[T], b: &[T], result: &mut [T]);

    /// Perform SIMD dot product
    fn simd_dot(a: &[T], b: &[T]) -> T;

    /// Perform SIMD reduction sum
    fn simd_sum(values: &[T]) -> T;
}

/// Basic SIMD operations implementation for f32
pub struct SimdF32;

impl SimdOps<f32> for SimdF32 {
    const SIMD_WIDTH: usize = 8; // AVX256 width

    fn simd_available() -> bool {
        // In a real implementation, this would check CPU features
        cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64")
    }

    fn simd_add(a: &[f32], b: &[f32], result: &mut [f32]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx") {
                unsafe { simd_add_f32_avx(a, b, result) };
                return;
            }
        }

        // Fallback scalar implementation
        for ((a_val, b_val), r) in a.iter().zip(b).zip(result) {
            *r = a_val + b_val;
        }
    }

    fn simd_mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx") {
                unsafe { simd_mul_f32_avx(a, b, result) };
                return;
            }
        }

        // Fallback scalar implementation
        for ((a_val, b_val), r) in a.iter().zip(b).zip(result) {
            *r = a_val * b_val;
        }
    }

    fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx") {
                return unsafe { simd_dot_f32_avx(a, b) };
            }
        }

        // Fallback scalar implementation
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn simd_sum(values: &[f32]) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx") {
                return unsafe { simd_sum_f32_avx(values) };
            }
        }

        // Fallback scalar implementation
        values.iter().sum()
    }
}

/// Basic SIMD operations implementation for f64
pub struct SimdF64;

impl SimdOps<f64> for SimdF64 {
    const SIMD_WIDTH: usize = 4; // AVX256 width for f64

    fn simd_available() -> bool {
        cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64")
    }

    fn simd_add(a: &[f64], b: &[f64], result: &mut [f64]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx") {
                unsafe { simd_add_f64_avx(a, b, result) };
                return;
            }
        }

        // Fallback scalar implementation
        for ((a_val, b_val), r) in a.iter().zip(b).zip(result) {
            *r = a_val + b_val;
        }
    }

    fn simd_mul(a: &[f64], b: &[f64], result: &mut [f64]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx") {
                unsafe { simd_mul_f64_avx(a, b, result) };
                return;
            }
        }

        // Fallback scalar implementation
        for ((a_val, b_val), r) in a.iter().zip(b).zip(result) {
            *r = a_val * b_val;
        }
    }

    fn simd_dot(a: &[f64], b: &[f64]) -> f64 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx") {
                return unsafe { simd_dot_f64_avx(a, b) };
            }
        }

        // Fallback scalar implementation
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn simd_sum(values: &[f64]) -> f64 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx") {
                return unsafe { simd_sum_f64_avx(values) };
            }
        }

        // Fallback scalar implementation
        values.iter().sum()
    }
}

/// Trait for memory-efficient numeric operations
pub trait MemoryEfficientOps<T> {
    /// Perform in-place operations to minimize memory allocations
    fn inplace_scale(data: &mut [T], factor: T);

    /// Perform in-place addition
    fn inplace_add(data: &mut [T], other: &[T]);

    /// Perform in-place element-wise multiplication
    fn inplace_mul(data: &mut [T], other: &[T]);

    /// Calculate statistics without additional memory allocation
    fn streaming_mean_var(data: &[T]) -> (T, T);

    /// Find min/max in a single pass
    fn min_max(data: &[T]) -> (T, T);
}

/// Memory-efficient operations for floating point types
pub struct MemoryEfficientFloat;

impl<T> MemoryEfficientOps<T> for MemoryEfficientFloat
where
    T: FloatBounds + Copy + PartialOrd,
{
    fn inplace_scale(data: &mut [T], factor: T) {
        for item in data.iter_mut() {
            *item *= factor;
        }
    }

    fn inplace_add(data: &mut [T], other: &[T]) {
        for (item, other_item) in data.iter_mut().zip(other) {
            *item += *other_item;
        }
    }

    fn inplace_mul(data: &mut [T], other: &[T]) {
        for (item, other_item) in data.iter_mut().zip(other) {
            *item *= *other_item;
        }
    }

    fn streaming_mean_var(data: &[T]) -> (T, T) {
        if data.is_empty() {
            return (T::zero(), T::zero());
        }

        // Welford's online algorithm for numerical stability
        let mut mean = T::zero();
        let mut m2 = T::zero();
        let mut count = T::zero();

        for &value in data {
            count += T::one();
            let delta = value - mean;
            mean += delta / count;
            let delta2 = value - mean;
            m2 += delta * delta2;
        }

        let variance = if count > T::one() {
            m2 / (count - T::one())
        } else {
            T::zero()
        };

        (mean, variance)
    }

    fn min_max(data: &[T]) -> (T, T) {
        if data.is_empty() {
            return (T::zero(), T::zero());
        }

        let mut min_val = data[0];
        let mut max_val = data[0];

        for &value in data.iter().skip(1) {
            if value < min_val {
                min_val = value;
            }
            if value > max_val {
                max_val = value;
            }
        }

        (min_val, max_val)
    }
}

/// Trait for GPU-compatible numeric operations
///
/// This trait defines operations that can be efficiently executed on GPU hardware.
/// When SciRS2's GPU support becomes available, these operations will be backed
/// by actual GPU kernels.
pub trait GpuOps<T> {
    /// Check if GPU operations are available
    fn gpu_available() -> bool;

    /// Get preferred GPU device for operations
    fn preferred_device() -> Option<usize>;

    /// Perform element-wise operations on GPU
    fn gpu_elementwise_op<F>(data: &[T], op: F) -> Vec<T>
    where
        F: Fn(T) -> T + Send + Sync;

    /// Perform matrix operations on GPU
    fn gpu_matrix_mul(a: &[T], b: &[T], m: usize, n: usize, k: usize) -> Vec<T>;

    /// Perform reduction operations on GPU
    fn gpu_reduce_sum(data: &[T]) -> T;
}

/// GPU operations implementation (fallback to CPU for now)
pub struct GpuFloat;

impl<T> GpuOps<T> for GpuFloat
where
    T: FloatBounds + Send + Sync,
{
    fn gpu_available() -> bool {
        // TODO: Check for actual GPU when scirs2-core::gpu is available
        false
    }

    fn preferred_device() -> Option<usize> {
        // TODO: Return actual GPU device ID when available
        None
    }

    fn gpu_elementwise_op<F>(data: &[T], op: F) -> Vec<T>
    where
        F: Fn(T) -> T + Send + Sync,
    {
        // Fallback to parallel CPU implementation
        use rayon::prelude::*;
        data.par_iter().map(|&x| op(x)).collect()
    }

    fn gpu_matrix_mul(a: &[T], b: &[T], m: usize, n: usize, k: usize) -> Vec<T> {
        // Fallback to simple CPU matrix multiplication
        let mut result = vec![T::zero(); m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        result
    }

    fn gpu_reduce_sum(data: &[T]) -> T {
        // Fallback to parallel CPU reduction
        use rayon::prelude::*;
        data.par_iter().copied().reduce(|| T::zero(), |a, b| a + b)
    }
}

// SIMD implementation functions for f32
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn simd_add_f32_avx(a: &[f32], b: &[f32], result: &mut [f32]) {
    use std::arch::x86_64::*;
    const LANES: usize = 8;
    let len = a.len().min(b.len()).min(result.len());
    let mut i = 0;

    while i + LANES <= len {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let sum = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), sum);
        i += LANES;
    }

    for j in i..len {
        result[j] = a[j] + b[j];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn simd_mul_f32_avx(a: &[f32], b: &[f32], result: &mut [f32]) {
    use std::arch::x86_64::*;
    const LANES: usize = 8;
    let len = a.len().min(b.len()).min(result.len());
    let mut i = 0;

    while i + LANES <= len {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let prod = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), prod);
        i += LANES;
    }

    for j in i..len {
        result[j] = a[j] * b[j];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn simd_dot_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    const LANES: usize = 8;
    let len = a.len().min(b.len());
    let mut dot_vec = _mm256_setzero_ps();
    let mut i = 0;

    while i + LANES <= len {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let prod = _mm256_mul_ps(a_vec, b_vec);
        dot_vec = _mm256_add_ps(dot_vec, prod);
        i += LANES;
    }

    let mut sum_array = [0.0f32; 8];
    _mm256_storeu_ps(sum_array.as_mut_ptr(), dot_vec);
    let mut dot = sum_array.iter().sum::<f32>();

    for j in i..len {
        dot += a[j] * b[j];
    }
    dot
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn simd_sum_f32_avx(values: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    const LANES: usize = 8;
    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0;

    while i + LANES <= values.len() {
        let vec = _mm256_loadu_ps(values.as_ptr().add(i));
        sum_vec = _mm256_add_ps(sum_vec, vec);
        i += LANES;
    }

    let mut sum_array = [0.0f32; 8];
    _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
    let mut sum = sum_array.iter().sum::<f32>();

    for j in i..values.len() {
        sum += values[j];
    }
    sum
}

// SIMD implementation functions for f64
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn simd_add_f64_avx(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;
    const LANES: usize = 4;
    let len = a.len().min(b.len()).min(result.len());
    let mut i = 0;

    while i + LANES <= len {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let sum = _mm256_add_pd(a_vec, b_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), sum);
        i += LANES;
    }

    for j in i..len {
        result[j] = a[j] + b[j];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn simd_mul_f64_avx(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;
    const LANES: usize = 4;
    let len = a.len().min(b.len()).min(result.len());
    let mut i = 0;

    while i + LANES <= len {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod = _mm256_mul_pd(a_vec, b_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), prod);
        i += LANES;
    }

    for j in i..len {
        result[j] = a[j] * b[j];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn simd_dot_f64_avx(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    const LANES: usize = 4;
    let len = a.len().min(b.len());
    let mut dot_vec = _mm256_setzero_pd();
    let mut i = 0;

    while i + LANES <= len {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod = _mm256_mul_pd(a_vec, b_vec);
        dot_vec = _mm256_add_pd(dot_vec, prod);
        i += LANES;
    }

    let mut sum_array = [0.0; 4];
    _mm256_storeu_pd(sum_array.as_mut_ptr(), dot_vec);
    let mut dot = sum_array.iter().sum::<f64>();

    for j in i..len {
        dot += a[j] * b[j];
    }
    dot
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn simd_sum_f64_avx(values: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    const LANES: usize = 4;
    let mut sum_vec = _mm256_setzero_pd();
    let mut i = 0;

    while i + LANES <= values.len() {
        let vec = _mm256_loadu_pd(values.as_ptr().add(i));
        sum_vec = _mm256_add_pd(sum_vec, vec);
        i += LANES;
    }

    let mut sum_array = [0.0; 4];
    _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
    let mut sum = sum_array.iter().sum::<f64>();

    for j in i..values.len() {
        sum += values[j];
    }
    sum
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let c1 = Complex::new(3.0, 4.0);
        let c2 = Complex::new(1.0, 2.0);

        assert_eq!(c1.magnitude(), 5.0);
        assert_eq!(c1.norm_sqr(), 25.0);

        let sum = c1 + c2;
        assert_eq!(sum.real(), 4.0);
        assert_eq!(sum.imag(), 6.0);

        let conj = c1.conj();
        assert_eq!(conj.real(), 3.0);
        assert_eq!(conj.imag(), -4.0);
    }

    #[test]
    fn test_numeric_conversion() {
        let f32_val = std::f32::consts::PI;
        let f64_val: Option<f64> = f32_val.to_higher_precision();
        assert!(f64_val.is_some());
        assert!((f64_val.expect("expected valid value") - std::f64::consts::PI).abs() < 1e-6);

        let converted_back = f64::from_higher_precision(f32_val);
        assert!((converted_back - std::f64::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn test_simd_operations() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let mut result = vec![0.0_f32; 4];

        SimdF32::simd_add(&a, &b, &mut result);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);

        let dot_product = SimdF32::simd_dot(&a, &b);
        assert_eq!(dot_product, 70.0); // 1*5 + 2*6 + 3*7 + 4*8 = 70
    }

    #[test]
    fn test_memory_efficient_ops() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        MemoryEfficientFloat::inplace_scale(&mut data, 2.0);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let other = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        MemoryEfficientFloat::inplace_add(&mut data, &other);
        assert_eq!(data, vec![3.0, 5.0, 7.0, 9.0, 11.0]);

        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, var) = MemoryEfficientFloat::streaming_mean_var(&test_data);
        assert!((mean - 3.0_f64).abs() < f64::EPSILON);
        assert!((var - 2.5_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gpu_fallback_ops() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let doubled = GpuFloat::gpu_elementwise_op(&data, |x: f64| x * 2.0);
        assert_eq!(doubled, vec![2.0, 4.0, 6.0, 8.0]);

        let sum = GpuFloat::gpu_reduce_sum(&data);
        assert_eq!(sum, 10.0);

        // Test simple 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let result = GpuFloat::gpu_matrix_mul(&a, &b, 2, 2, 2);
        // [1,2] * [5,6] = [19,22]
        // [3,4]   [7,8]   [43,50]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }
}

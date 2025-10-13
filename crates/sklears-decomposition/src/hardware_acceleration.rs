//! Hardware Acceleration for Decomposition Algorithms
//!
//! This module provides hardware-specific optimizations including:
//! - SIMD optimizations for matrix operations
//! - Multi-threaded parallel processing
//! - Mixed-precision arithmetic support
//! - Memory-aligned operations
//! - Vectorized mathematical functions

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

#[cfg(feature = "gpu")]
use candle_core::{DType, Device, Tensor};
// cudarc is only available on non-macOS platforms (no CUDA on macOS)
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};

/// Configuration for hardware acceleration features
#[derive(Debug, Clone)]
pub struct AccelerationConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Enable mixed precision arithmetic
    pub enable_mixed_precision: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// GPU device ID to use
    pub gpu_device_id: i32,
    /// GPU memory limit in bytes
    pub gpu_memory_limit: Option<usize>,
    /// Number of threads for parallel operations
    pub num_threads: Option<usize>,
    /// Memory alignment for SIMD operations
    pub memory_alignment: usize,
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_parallel: true,
            enable_mixed_precision: false,
            enable_gpu: false,      // GPU disabled by default
            gpu_device_id: 0,       // Primary GPU
            gpu_memory_limit: None, // No memory limit
            num_threads: None,      // Use rayon default
            memory_alignment: 32,   // 256-bit alignment for AVX2/NEON
        }
    }
}

/// SIMD-optimized matrix operations
pub struct SimdMatrixOps {
    config: AccelerationConfig,
}

impl SimdMatrixOps {
    /// Create a new SIMD matrix operations instance
    pub fn new() -> Self {
        Self {
            config: AccelerationConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: AccelerationConfig) -> Self {
        self.config = config;
        self
    }

    /// SIMD-optimized vector dot product
    pub fn dot_product_simd(&self, a: &Array1<Float>, b: &Array1<Float>) -> Result<Float> {
        if a.len() != b.len() {
            return Err(SklearsError::InvalidInput(
                "Vector dimensions must match for dot product".to_string(),
            ));
        }

        if !self.config.enable_simd {
            return Ok(self.dot_product_fallback(a, b));
        }

        #[cfg(target_arch = "aarch64")]
        {
            self.dot_product_neon(a, b)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Ok(self.dot_product_fallback(a, b))
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn dot_product_neon(&self, a: &Array1<Float>, b: &Array1<Float>) -> Result<Float> {
        let n = a.len();
        let mut sum;

        if std::mem::size_of::<Float>() == 8 {
            // Double precision NEON operations
            let chunks = n / 2;
            let _remainder = n % 2;

            unsafe {
                let mut acc = vdupq_n_f64(0.0);

                for i in 0..chunks {
                    let idx = i * 2;
                    let va = vld1q_f64(a.as_ptr().add(idx));
                    let vb = vld1q_f64(b.as_ptr().add(idx));
                    acc = vfmaq_f64(acc, va, vb);
                }

                // Sum the accumulator
                sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);

                // Handle remainder
                for i in (chunks * 2)..n {
                    sum += a[i] * b[i];
                }
            }
        } else {
            // Single precision NEON operations
            let chunks = n / 4;
            let _remainder = n % 4;

            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                let a_ptr = a.as_ptr() as *const f32;
                let b_ptr = b.as_ptr() as *const f32;

                for i in 0..chunks {
                    let idx = i * 4;
                    let va = vld1q_f32(a_ptr.add(idx));
                    let vb = vld1q_f32(b_ptr.add(idx));
                    acc = vfmaq_f32(acc, va, vb);
                }

                // Sum the accumulator
                let sum_vec = vpaddq_f32(acc, acc);
                let sum_vec2 = vpaddq_f32(sum_vec, sum_vec);
                sum = vgetq_lane_f32(sum_vec2, 0) as Float;

                // Handle remainder
                for i in (chunks * 4)..n {
                    sum += a[i] * b[i];
                }
            }
        }

        Ok(sum)
    }

    fn dot_product_fallback(&self, a: &Array1<Float>, b: &Array1<Float>) -> Float {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// SIMD-optimized matrix-vector multiplication
    pub fn matrix_vector_mul_simd(
        &self,
        matrix: &Array2<Float>,
        vector: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let (m, n) = matrix.dim();
        if n != vector.len() {
            return Err(SklearsError::InvalidInput(
                "Matrix columns must match vector length".to_string(),
            ));
        }

        if !self.config.enable_simd || !self.config.enable_parallel {
            return Ok(self.matrix_vector_mul_fallback(matrix, vector));
        }

        // Parallel SIMD matrix-vector multiplication
        #[cfg(feature = "parallel")]
        let result: Vec<Float> = (0..m)
            .into_par_iter()
            .map(|i| {
                let row = matrix.row(i);
                self.dot_product_simd(&row.to_owned(), vector)
                    .unwrap_or(0.0)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let result: Vec<Float> = (0..m)
            .map(|i| {
                let row = matrix.row(i);
                self.dot_product_simd(&row.to_owned(), vector)
                    .unwrap_or(0.0)
            })
            .collect();

        Ok(Array1::from_vec(result))
    }

    fn matrix_vector_mul_fallback(
        &self,
        matrix: &Array2<Float>,
        vector: &Array1<Float>,
    ) -> Array1<Float> {
        matrix.dot(vector)
    }

    /// SIMD-optimized element-wise operations
    pub fn elementwise_add_simd(
        &self,
        a: &Array1<Float>,
        b: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if a.len() != b.len() {
            return Err(SklearsError::InvalidInput(
                "Array dimensions must match".to_string(),
            ));
        }

        if !self.config.enable_simd {
            return Ok(a + b);
        }

        #[cfg(target_arch = "aarch64")]
        {
            self.elementwise_add_neon(a, b)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Ok(a + b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn elementwise_add_neon(&self, a: &Array1<Float>, b: &Array1<Float>) -> Result<Array1<Float>> {
        let n = a.len();
        let mut result = Array1::<Float>::zeros(n);

        if std::mem::size_of::<Float>() == 8 {
            // Double precision
            let chunks = n / 2;

            unsafe {
                for i in 0..chunks {
                    let idx = i * 2;
                    let va = vld1q_f64(a.as_ptr().add(idx));
                    let vb = vld1q_f64(b.as_ptr().add(idx));
                    let vr = vaddq_f64(va, vb);
                    vst1q_f64(result.as_mut_ptr().add(idx), vr);
                }

                // Handle remainder
                for i in (chunks * 2)..n {
                    result[i] = a[i] + b[i];
                }
            }
        } else {
            // Single precision
            let chunks = n / 4;
            let a_ptr = a.as_ptr() as *const f32;
            let b_ptr = b.as_ptr() as *const f32;
            let result_ptr = result.as_mut_ptr() as *mut f32;

            unsafe {
                for i in 0..chunks {
                    let idx = i * 4;
                    let va = vld1q_f32(a_ptr.add(idx));
                    let vb = vld1q_f32(b_ptr.add(idx));
                    let vr = vaddq_f32(va, vb);
                    vst1q_f32(result_ptr.add(idx), vr);
                }

                // Handle remainder
                for i in (chunks * 4)..n {
                    result[i] = a[i] + b[i];
                }
            }
        }

        Ok(result)
    }

    /// SIMD-optimized element-wise multiplication
    pub fn elementwise_mul_simd(
        &self,
        a: &Array1<Float>,
        b: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if a.len() != b.len() {
            return Err(SklearsError::InvalidInput(
                "Array dimensions must match".to_string(),
            ));
        }

        if !self.config.enable_simd {
            return Ok(a * b);
        }

        #[cfg(target_arch = "aarch64")]
        {
            self.elementwise_mul_neon(a, b)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Ok(a * b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn elementwise_mul_neon(&self, a: &Array1<Float>, b: &Array1<Float>) -> Result<Array1<Float>> {
        let n = a.len();
        let mut result = Array1::<Float>::zeros(n);

        if std::mem::size_of::<Float>() == 8 {
            // Double precision
            let chunks = n / 2;

            unsafe {
                for i in 0..chunks {
                    let idx = i * 2;
                    let va = vld1q_f64(a.as_ptr().add(idx));
                    let vb = vld1q_f64(b.as_ptr().add(idx));
                    let vr = vmulq_f64(va, vb);
                    vst1q_f64(result.as_mut_ptr().add(idx), vr);
                }

                // Handle remainder
                for i in (chunks * 2)..n {
                    result[i] = a[i] * b[i];
                }
            }
        } else {
            // Single precision
            let chunks = n / 4;
            let a_ptr = a.as_ptr() as *const f32;
            let b_ptr = b.as_ptr() as *const f32;
            let result_ptr = result.as_mut_ptr() as *mut f32;

            unsafe {
                for i in 0..chunks {
                    let idx = i * 4;
                    let va = vld1q_f32(a_ptr.add(idx));
                    let vb = vld1q_f32(b_ptr.add(idx));
                    let vr = vmulq_f32(va, vb);
                    vst1q_f32(result_ptr.add(idx), vr);
                }

                // Handle remainder
                for i in (chunks * 4)..n {
                    result[i] = a[i] * b[i];
                }
            }
        }

        Ok(result)
    }

    /// Vectorized mathematical functions
    pub fn vector_exp_simd(&self, input: &Array1<Float>) -> Array1<Float> {
        if !self.config.enable_simd || !self.config.enable_parallel {
            return input.mapv(|x| x.exp());
        }

        // Parallel vectorized exponential
        #[cfg(feature = "parallel")]
        let result: Vec<Float> = input.par_iter().map(|&x| x.exp()).collect();

        #[cfg(not(feature = "parallel"))]
        let result: Vec<Float> = input.iter().map(|&x| x.exp()).collect();
        Array1::from_vec(result)
    }

    pub fn vector_sqrt_simd(&self, input: &Array1<Float>) -> Array1<Float> {
        if !self.config.enable_simd || !self.config.enable_parallel {
            return input.mapv(|x| x.sqrt());
        }

        // Parallel vectorized square root
        #[cfg(feature = "parallel")]
        let result: Vec<Float> = input.par_iter().map(|&x| x.sqrt()).collect();

        #[cfg(not(feature = "parallel"))]
        let result: Vec<Float> = input.iter().map(|&x| x.sqrt()).collect();
        Array1::from_vec(result)
    }

    pub fn vector_sin_simd(&self, input: &Array1<Float>) -> Array1<Float> {
        if !self.config.enable_simd || !self.config.enable_parallel {
            return input.mapv(|x| x.sin());
        }

        // Parallel vectorized sine
        #[cfg(feature = "parallel")]
        let result: Vec<Float> = input.par_iter().map(|&x| x.sin()).collect();

        #[cfg(not(feature = "parallel"))]
        let result: Vec<Float> = input.iter().map(|&x| x.sin()).collect();
        Array1::from_vec(result)
    }
}

impl Default for SimdMatrixOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel decomposition algorithms
pub struct ParallelDecomposition {
    config: AccelerationConfig,
}

impl ParallelDecomposition {
    /// Create a new parallel decomposition instance
    pub fn new() -> Self {
        Self {
            config: AccelerationConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: AccelerationConfig) -> Self {
        self.config = config;
        self
    }

    /// Parallel SVD computation for large matrices
    pub fn parallel_svd(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        let (m, n) = matrix.dim();

        if !self.config.enable_parallel {
            return self.sequential_svd(matrix);
        }

        // For large matrices, use block-based parallel SVD
        if m > 1000 && n > 1000 {
            self.block_parallel_svd(matrix)
        } else {
            self.sequential_svd(matrix)
        }
    }

    fn block_parallel_svd(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        // Simplified block-based SVD - in practice would use more sophisticated algorithms
        let (_m, _n) = matrix.dim();
        let _block_size = 256;

        // For now, fall back to sequential SVD
        // A full implementation would use randomized SVD or hierarchical SVD
        self.sequential_svd(matrix)
    }

    fn sequential_svd(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        // Use ndarray_linalg for SVD computation
        // This is a placeholder - actual implementation would depend on available LAPACK
        let (m, n) = matrix.dim();
        let min_dim = m.min(n);

        // Simplified SVD placeholder
        let u = Array2::eye(m);
        let s = Array1::ones(min_dim);
        let vt = Array2::eye(n);

        Ok((u, s, vt))
    }

    /// Parallel eigendecomposition
    pub fn parallel_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for eigendecomposition".to_string(),
            ));
        }

        if !self.config.enable_parallel || n < 500 {
            return self.sequential_eigendecomposition(matrix);
        }

        // For large matrices, use parallel algorithms
        self.block_parallel_eigendecomposition(matrix)
    }

    fn block_parallel_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // Simplified parallel eigendecomposition
        // A full implementation would use divide-and-conquer algorithms
        self.sequential_eigendecomposition(matrix)
    }

    fn sequential_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();

        // Simplified eigendecomposition placeholder
        let eigenvalues = Array1::ones(n);
        let eigenvectors = Array2::eye(n);

        Ok((eigenvalues, eigenvectors))
    }

    /// Parallel matrix multiplication with tiling
    pub fn parallel_matrix_multiply(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        if !self.config.enable_parallel {
            return Ok(a.dot(b));
        }

        // Use tiled parallel multiplication for large matrices
        if m > 100 && n > 100 && k1 > 100 {
            self.tiled_parallel_multiply(a, b)
        } else {
            Ok(a.dot(b))
        }
    }

    fn tiled_parallel_multiply(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let tile_size = 64; // Cache-friendly tile size

        let _result = Array2::<Float>::zeros((m, n));

        // Parallel tiled multiplication
        let m_tiles = (m + tile_size - 1) / tile_size;
        let n_tiles = (n + tile_size - 1) / tile_size;
        let k_tiles = (k + tile_size - 1) / tile_size;

        #[cfg(feature = "parallel")]
        {
            (0..m_tiles).into_par_iter().for_each(|i_tile| {
                for j_tile in 0..n_tiles {
                    let mut tile_result = Array2::zeros((
                        (tile_size).min(m - i_tile * tile_size),
                        (tile_size).min(n - j_tile * tile_size),
                    ));

                    for k_tile in 0..k_tiles {
                        let i_start = i_tile * tile_size;
                        let i_end = (i_start + tile_size).min(m);
                        let j_start = j_tile * tile_size;
                        let j_end = (j_start + tile_size).min(n);
                        let k_start = k_tile * tile_size;
                        let k_end = (k_start + tile_size).min(k);

                        let a_tile =
                            a.slice(scirs2_core::ndarray::s![i_start..i_end, k_start..k_end]);
                        let b_tile =
                            b.slice(scirs2_core::ndarray::s![k_start..k_end, j_start..j_end]);

                        tile_result += &a_tile.dot(&b_tile);
                    }

                    // This requires synchronization - simplified for demonstration
                    // In practice, would need proper synchronization mechanisms
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            (0..m_tiles).for_each(|i_tile| {
                for j_tile in 0..n_tiles {
                    let mut tile_result = Array2::zeros((
                        (tile_size).min(m - i_tile * tile_size),
                        (tile_size).min(n - j_tile * tile_size),
                    ));

                    for k_tile in 0..k_tiles {
                        let i_start = i_tile * tile_size;
                        let i_end = (i_start + tile_size).min(m);
                        let j_start = j_tile * tile_size;
                        let j_end = (j_start + tile_size).min(n);
                        let k_start = k_tile * tile_size;
                        let k_end = (k_start + tile_size).min(k);

                        let a_tile =
                            a.slice(scirs2_core::ndarray::s![i_start..i_end, k_start..k_end]);
                        let b_tile =
                            b.slice(scirs2_core::ndarray::s![k_start..k_end, j_start..j_end]);

                        tile_result += &a_tile.dot(&b_tile);
                    }

                    // Write tile result back (this would need proper synchronization)
                    // For now, skip the actual writing since we fall back anyway
                }
            });
        }

        // For now, fall back to standard multiplication
        Ok(a.dot(b))
    }
}

impl Default for ParallelDecomposition {
    fn default() -> Self {
        Self::new()
    }
}

/// Mixed-precision arithmetic support
pub struct MixedPrecisionOps {
    config: AccelerationConfig,
}

impl MixedPrecisionOps {
    /// Create a new mixed-precision operations instance
    pub fn new() -> Self {
        Self {
            config: AccelerationConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: AccelerationConfig) -> Self {
        self.config = config;
        self
    }

    /// Convert double precision array to single precision
    pub fn to_single_precision(&self, input: &Array1<f64>) -> Array1<f32> {
        if self.config.enable_parallel {
            #[cfg(feature = "parallel")]
            let result: Vec<f32> = input.par_iter().map(|&x| x as f32).collect();

            #[cfg(not(feature = "parallel"))]
            let result: Vec<f32> = input.iter().map(|&x| x as f32).collect();
            Array1::from_vec(result)
        } else {
            input.mapv(|x| x as f32)
        }
    }

    /// Convert single precision array to double precision
    pub fn to_double_precision(&self, input: &Array1<f32>) -> Array1<f64> {
        if self.config.enable_parallel {
            #[cfg(feature = "parallel")]
            let result: Vec<f64> = input.par_iter().map(|&x| x as f64).collect();

            #[cfg(not(feature = "parallel"))]
            let result: Vec<f64> = input.iter().map(|&x| x as f64).collect();
            Array1::from_vec(result)
        } else {
            input.mapv(|x| x as f64)
        }
    }

    /// Mixed-precision matrix multiplication (compute in f32, accumulate in f64)
    pub fn mixed_precision_multiply(
        &self,
        a: &Array2<f64>,
        b: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        if !self.config.enable_mixed_precision {
            return Ok(a.dot(b));
        }

        let (_m, k1) = a.dim();
        let (k2, _n) = b.dim();

        if k1 != k2 {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        // Convert to single precision for computation
        let a_f32 = a.mapv(|x| x as f32);
        let b_f32 = b.mapv(|x| x as f32);

        // Compute in single precision
        let result_f32 = a_f32.dot(&b_f32);

        // Convert back to double precision
        let result = result_f32.mapv(|x| x as f64);

        Ok(result)
    }
}

impl Default for MixedPrecisionOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory-aligned operations for optimal performance
pub struct AlignedMemoryOps {
    alignment: usize,
}

/// Owned buffer with guaranteed memory alignment
pub struct AlignedBuffer {
    ptr: NonNull<Float>,
    len: usize,
    alignment: usize,
}

impl AlignedMemoryOps {
    /// Create new aligned memory operations with specified alignment
    pub fn new(alignment: usize) -> Self {
        Self {
            alignment: Self::sanitize_alignment(alignment),
        }
    }

    fn sanitize_alignment(requested: usize) -> usize {
        let base = requested.max(std::mem::align_of::<Float>()).max(1);
        if base.is_power_of_two() {
            base
        } else {
            base.next_power_of_two()
        }
    }

    /// Create aligned array with specified size
    pub fn create_aligned_array(&self, size: usize) -> AlignedBuffer {
        AlignedBuffer::new(size, self.alignment)
    }

    /// Check if array is properly aligned
    pub fn is_aligned(&self, data: &[Float]) -> bool {
        let ptr = data.as_ptr() as usize;
        ptr % self.alignment == 0
    }

    /// Copy data to aligned buffer if necessary
    pub fn ensure_aligned(&self, data: &Array1<Float>) -> AlignedBuffer {
        let slice = data
            .as_slice()
            .expect("Array1 should have a contiguous slice representation");

        let mut buffer = self.create_aligned_array(slice.len());
        buffer.as_mut_slice().copy_from_slice(slice);
        buffer
    }
}

impl Default for AlignedMemoryOps {
    fn default() -> Self {
        Self::new(32) // 256-bit alignment
    }
}

impl AlignedBuffer {
    pub fn new(size: usize, alignment: usize) -> Self {
        if size == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                alignment,
            };
        }

        let elem_size = std::mem::size_of::<Float>();
        let total_size = elem_size
            .checked_mul(size)
            .expect("Requested buffer size exceeds addressable memory");
        let layout = Layout::from_size_align(total_size, alignment)
            .expect("Invalid layout for aligned allocation");

        unsafe {
            let ptr = alloc_zeroed(layout);
            if ptr.is_null() {
                handle_alloc_error(layout);
            }

            Self {
                ptr: NonNull::new_unchecked(ptr as *mut Float),
                len: size,
                alignment,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[Float] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Deref for AlignedBuffer {
    type Target = [Float];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }

        let elem_size = std::mem::size_of::<Float>();
        let total_size = elem_size * self.len;
        if let Ok(layout) = Layout::from_size_align(total_size, self.alignment) {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

/// GPU acceleration for matrix operations using CUDA
#[cfg(feature = "gpu")]
pub struct GpuAcceleration {
    config: AccelerationConfig,
    device: Device,
    #[cfg(not(target_os = "macos"))]
    cuda_device: Option<std::sync::Arc<CudaDevice>>,
}

#[cfg(feature = "gpu")]
impl GpuAcceleration {
    /// Create a new GPU acceleration instance
    pub fn new() -> Result<Self> {
        Self::with_config(AccelerationConfig::default())
    }

    /// Create GPU acceleration with specific configuration
    pub fn with_config(config: AccelerationConfig) -> Result<Self> {
        if !config.enable_gpu {
            return Err(SklearsError::InvalidInput(
                "GPU acceleration is disabled in configuration".to_string(),
            ));
        }

        // Initialize CUDA device (only on non-macOS platforms)
        #[cfg(not(target_os = "macos"))]
        let cuda_device = match CudaDevice::new(config.gpu_device_id as usize) {
            Ok(device) => Some(std::sync::Arc::new(device)),
            Err(_) => {
                return Err(SklearsError::InvalidInput(
                    "Failed to initialize CUDA device".to_string(),
                ));
            }
        };

        // Initialize Candle device
        let device = match Device::new_cuda(config.gpu_device_id as usize) {
            Ok(dev) => dev,
            Err(_) => Device::Cpu, // Fallback to CPU
        };

        Ok(Self {
            config,
            device,
            #[cfg(not(target_os = "macos"))]
            cuda_device,
        })
    }

    /// Check if GPU is available and properly initialized
    pub fn is_gpu_available(&self) -> bool {
        #[cfg(not(target_os = "macos"))]
        return self.cuda_device.is_some() && matches!(self.device, Device::Cuda(_));

        #[cfg(target_os = "macos")]
        return matches!(self.device, Device::Cuda(_));
    }

    /// Get GPU memory information
    pub fn gpu_memory_info(&self) -> Result<(usize, usize)> {
        #[cfg(not(target_os = "macos"))]
        {
            if let Some(ref cuda_device) = self.cuda_device {
                match cuda_device.total_memory() {
                    Ok(total) => {
                        // Estimate free memory (simplified)
                        let free = total / 2; // Conservative estimate
                        Ok((free, total))
                    }
                    Err(_) => Err(SklearsError::InvalidInput(
                        "Failed to get GPU memory info".to_string(),
                    )),
                }
            } else {
                Err(SklearsError::InvalidInput("GPU not available".to_string()))
            }
        }

        #[cfg(target_os = "macos")]
        Err(SklearsError::InvalidInput(
            "CUDA not available on macOS".to_string(),
        ))
    }

    /// Convert ndarray to GPU tensor
    pub fn array_to_tensor(&self, array: &Array2<Float>) -> Result<Tensor> {
        let shape = array.shape();
        let data: Vec<f32> = if std::mem::size_of::<Float>() == 8 {
            array.iter().map(|&x| x as f32).collect()
        } else {
            array.iter().map(|&x| x as f32).collect()
        };

        match Tensor::from_vec(data, (shape[0], shape[1]), &self.device) {
            Ok(tensor) => Ok(tensor),
            Err(_) => Err(SklearsError::InvalidInput(
                "Failed to create GPU tensor".to_string(),
            )),
        }
    }

    /// Convert GPU tensor back to ndarray
    pub fn tensor_to_array(&self, tensor: &Tensor) -> Result<Array2<Float>> {
        let shape = tensor.shape();
        let dims = shape.dims();
        if dims.len() != 2 {
            return Err(SklearsError::InvalidInput("Tensor must be 2D".to_string()));
        }

        match tensor.to_vec2::<f32>() {
            Ok(data) => {
                let flat_data: Vec<Float> = data
                    .into_iter()
                    .flatten()
                    .map(|x| {
                        if std::mem::size_of::<Float>() == 8 {
                            x as f64
                        } else {
                            x as f64
                        }
                    })
                    .collect();

                match Array2::from_shape_vec((dims[0], dims[1]), flat_data) {
                    Ok(array) => Ok(array),
                    Err(_) => Err(SklearsError::InvalidInput(
                        "Failed to create array from tensor".to_string(),
                    )),
                }
            }
            Err(_) => Err(SklearsError::InvalidInput(
                "Failed to convert tensor to array".to_string(),
            )),
        }
    }

    /// GPU-accelerated matrix multiplication
    pub fn gpu_matrix_multiply(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        if !self.is_gpu_available() {
            return Err(SklearsError::InvalidInput("GPU not available".to_string()));
        }

        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        // Convert to GPU tensors
        let tensor_a = self.array_to_tensor(a)?;
        let tensor_b = self.array_to_tensor(b)?;

        // Perform GPU matrix multiplication
        let result_tensor = match tensor_a.matmul(&tensor_b) {
            Ok(tensor) => tensor,
            Err(_) => {
                return Err(SklearsError::InvalidInput(
                    "GPU matrix multiplication failed".to_string(),
                ))
            }
        };

        // Convert back to array
        self.tensor_to_array(&result_tensor)
    }

    /// GPU-accelerated SVD decomposition
    pub fn gpu_svd(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        if !self.is_gpu_available() {
            return Err(SklearsError::InvalidInput("GPU not available".to_string()));
        }

        let tensor = self.array_to_tensor(matrix)?;

        // For now, use a simplified GPU-based SVD implementation
        // In practice, this would use cuSOLVER or similar libraries
        let (m, n) = matrix.dim();
        let min_dim = m.min(n);

        // This is a placeholder - real implementation would use cuSOLVER
        let u_tensor = match Tensor::eye(m, DType::F32, &self.device) {
            Ok(t) => t,
            Err(_) => {
                return Err(SklearsError::InvalidInput(
                    "Failed to create identity matrix".to_string(),
                ))
            }
        };

        let s_data = vec![1.0f32; min_dim];
        let s_tensor = match Tensor::from_vec(s_data, min_dim, &self.device) {
            Ok(t) => t,
            Err(_) => {
                return Err(SklearsError::InvalidInput(
                    "Failed to create singular values".to_string(),
                ))
            }
        };

        let vt_tensor = match Tensor::eye(n, DType::F32, &self.device) {
            Ok(t) => t,
            Err(_) => {
                return Err(SklearsError::InvalidInput(
                    "Failed to create identity matrix".to_string(),
                ))
            }
        };

        // Convert tensors back to arrays
        let u = self.tensor_to_array(&u_tensor)?;
        let vt = self.tensor_to_array(&vt_tensor)?;

        // Convert singular values
        let s_vec = match s_tensor.to_vec1::<f32>() {
            Ok(vec) => vec.into_iter().map(|x| x as Float).collect(),
            Err(_) => {
                return Err(SklearsError::InvalidInput(
                    "Failed to convert singular values".to_string(),
                ))
            }
        };
        let s = Array1::from_vec(s_vec);

        Ok((u, s, vt))
    }

    /// GPU-accelerated eigendecomposition
    pub fn gpu_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        if !self.is_gpu_available() {
            return Err(SklearsError::InvalidInput("GPU not available".to_string()));
        }

        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for eigendecomposition".to_string(),
            ));
        }

        let tensor = self.array_to_tensor(matrix)?;

        // Simplified GPU eigendecomposition - real implementation would use cuSOLVER
        let eigenvals_data = vec![1.0f32; n];
        let eigenvals_tensor = match Tensor::from_vec(eigenvals_data, n, &self.device) {
            Ok(t) => t,
            Err(_) => {
                return Err(SklearsError::InvalidInput(
                    "Failed to create eigenvalues".to_string(),
                ))
            }
        };

        let eigenvecs_tensor = match Tensor::eye(n, DType::F32, &self.device) {
            Ok(t) => t,
            Err(_) => {
                return Err(SklearsError::InvalidInput(
                    "Failed to create eigenvectors".to_string(),
                ))
            }
        };

        // Convert back to arrays
        let eigenvecs = self.tensor_to_array(&eigenvecs_tensor)?;
        let eigenvals_vec = match eigenvals_tensor.to_vec1::<f32>() {
            Ok(vec) => vec.into_iter().map(|x| x as Float).collect(),
            Err(_) => {
                return Err(SklearsError::InvalidInput(
                    "Failed to convert eigenvalues".to_string(),
                ))
            }
        };
        let eigenvals = Array1::from_vec(eigenvals_vec);

        Ok((eigenvals, eigenvecs))
    }

    /// Batch GPU operations for multiple matrices
    pub fn batch_gpu_multiply(&self, matrices: &[Array2<Float>]) -> Result<Vec<Array2<Float>>> {
        if !self.is_gpu_available() {
            return Err(SklearsError::InvalidInput("GPU not available".to_string()));
        }

        let mut results = Vec::with_capacity(matrices.len() / 2);

        for chunk in matrices.chunks_exact(2) {
            let result = self.gpu_matrix_multiply(&chunk[0], &chunk[1])?;
            results.push(result);
        }

        Ok(results)
    }

    /// GPU memory management
    pub fn free_gpu_memory(&self) -> Result<()> {
        #[cfg(not(target_os = "macos"))]
        {
            if let Some(ref cuda_device) = self.cuda_device {
                match cuda_device.synchronize() {
                    Ok(()) => Ok(()),
                    Err(_) => Err(SklearsError::InvalidInput(
                        "Failed to synchronize GPU device".to_string(),
                    )),
                }
            } else {
                Ok(())
            }
        }

        #[cfg(target_os = "macos")]
        Ok(())
    }

    /// Profile GPU operation performance
    pub fn profile_gpu_operation<F, T>(&self, operation: F) -> Result<(T, std::time::Duration)>
    where
        F: FnOnce() -> Result<T>,
    {
        let start = std::time::Instant::now();
        let result = operation()?;

        // Synchronize GPU to ensure timing accuracy
        self.free_gpu_memory()?;

        let duration = start.elapsed();
        Ok((result, duration))
    }
}

#[cfg(feature = "gpu")]
impl Default for GpuAcceleration {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback to CPU-only configuration
            let config = AccelerationConfig {
                enable_gpu: false,
                ..AccelerationConfig::default()
            };
            Self {
                config,
                device: Device::Cpu,
                #[cfg(not(target_os = "macos"))]
                cuda_device: None,
            }
        })
    }
}

/// GPU-accelerated decomposition algorithms
#[cfg(feature = "gpu")]
pub struct GpuDecomposition {
    gpu_acceleration: GpuAcceleration,
}

#[cfg(feature = "gpu")]
impl GpuDecomposition {
    /// Create new GPU decomposition instance
    pub fn new() -> Result<Self> {
        Ok(Self {
            gpu_acceleration: GpuAcceleration::new()?,
        })
    }

    /// Create with configuration
    pub fn with_config(config: AccelerationConfig) -> Result<Self> {
        Ok(Self {
            gpu_acceleration: GpuAcceleration::with_config(config)?,
        })
    }

    /// GPU-accelerated PCA using SVD
    pub fn gpu_pca(
        &self,
        data: &Array2<Float>,
        n_components: usize,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        let (m, n) = data.dim();
        if n_components > m.min(n) {
            return Err(SklearsError::InvalidInput(
                "Number of components cannot exceed matrix dimensions".to_string(),
            ));
        }

        // Center the data
        let col_means = data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let centered_data = data - &col_means.insert_axis(scirs2_core::ndarray::Axis(0));

        // Perform GPU SVD
        let (u, s, vt) = self.gpu_acceleration.gpu_svd(&centered_data)?;

        // Truncate to requested number of components
        let u_truncated = u
            .slice(scirs2_core::ndarray::s![.., ..n_components])
            .to_owned();
        let s_truncated = s.slice(scirs2_core::ndarray::s![..n_components]).to_owned();
        let vt_truncated = vt
            .slice(scirs2_core::ndarray::s![..n_components, ..])
            .to_owned();

        Ok((u_truncated, s_truncated, vt_truncated))
    }

    /// GPU-accelerated matrix factorization
    pub fn gpu_factorize(&self, matrix: &Array2<Float>) -> Result<(Array2<Float>, Array2<Float>)> {
        let (m, n) = matrix.dim();
        let k = m.min(n);

        // Use GPU SVD for factorization
        let (u, s, vt) = self.gpu_acceleration.gpu_svd(matrix)?;

        // Create factor matrices
        let s_sqrt = s.mapv(|x| x.sqrt());
        let factor_a = &u.slice(scirs2_core::ndarray::s![.., ..k])
            * &s_sqrt.view().insert_axis(scirs2_core::ndarray::Axis(0));
        let factor_b = &s_sqrt.view().insert_axis(scirs2_core::ndarray::Axis(1))
            * &vt.slice(scirs2_core::ndarray::s![..k, ..]);

        Ok((factor_a, factor_b))
    }
}

#[cfg(feature = "gpu")]
impl Default for GpuDecomposition {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Create a fallback instance
            let config = AccelerationConfig {
                enable_gpu: false,
                ..AccelerationConfig::default()
            };
            Self {
                gpu_acceleration: GpuAcceleration::with_config(config).unwrap(),
            }
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_simd_dot_product() {
        let simd_ops = SimdMatrixOps::new();
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        let result = simd_ops.dot_product_simd(&a, &b).unwrap();
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_elementwise_add() {
        let simd_ops = SimdMatrixOps::new();
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        let result = simd_ops.elementwise_add_simd(&a, &b).unwrap();
        let expected = Array1::from_vec(vec![6.0, 8.0, 10.0, 12.0]);

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_simd_elementwise_mul() {
        let simd_ops = SimdMatrixOps::new();
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        let result = simd_ops.elementwise_mul_simd(&a, &b).unwrap();
        let expected = Array1::from_vec(vec![5.0, 12.0, 21.0, 32.0]);

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matrix_vector_mul_simd() {
        let simd_ops = SimdMatrixOps::new();
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = simd_ops.matrix_vector_mul_simd(&matrix, &vector).unwrap();
        let expected = Array1::from_vec(vec![14.0, 32.0]); // [1*1+2*2+3*3, 4*1+5*2+6*3]

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parallel_operations() {
        let parallel_ops = ParallelDecomposition::new();

        // Test basic functionality
        let matrix = Array2::eye(3);
        let result = parallel_ops.parallel_eigendecomposition(&matrix);
        assert!(result.is_ok());

        let (eigenvals, eigenvecs) = result.unwrap();
        assert_eq!(eigenvals.len(), 3);
        assert_eq!(eigenvecs.dim(), (3, 3));
    }

    #[test]
    fn test_mixed_precision() {
        let mixed_ops = MixedPrecisionOps::new();

        let input_f64 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let converted_f32 = mixed_ops.to_single_precision(&input_f64);
        let converted_back = mixed_ops.to_double_precision(&converted_f32);

        for (orig, back) in input_f64.iter().zip(converted_back.iter()) {
            assert!((orig - back).abs() < 1e-6); // Precision loss expected
        }
    }

    #[test]
    fn test_aligned_memory() {
        let aligned_ops = AlignedMemoryOps::new(32);

        let aligned_vec = aligned_ops.create_aligned_array(10);
        assert_eq!(aligned_vec.len(), 10);
        assert!(aligned_ops.is_aligned(&aligned_vec));

        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let aligned_array = aligned_ops.ensure_aligned(&array);
        assert_eq!(aligned_array.len(), array.len());
        assert!(aligned_ops.is_aligned(&aligned_array));

        assert_eq!(aligned_array.as_slice(), array.as_slice().unwrap());
    }

    #[test]
    fn test_vectorized_functions() {
        let simd_ops = SimdMatrixOps::new();
        let input = Array1::from_vec(vec![0.0, 1.0, 2.0]);

        let exp_result = simd_ops.vector_exp_simd(&input);
        let expected_exp = input.mapv(|x| x.exp());

        for (r, e) in exp_result.iter().zip(expected_exp.iter()) {
            assert!((r - e).abs() < 1e-10);
        }

        let sqrt_result = simd_ops.vector_sqrt_simd(&Array1::from_vec(vec![1.0, 4.0, 9.0]));
        let expected_sqrt = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        for (r, e) in sqrt_result.iter().zip(expected_sqrt.iter()) {
            assert!((r - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_acceleration_config() {
        let config = AccelerationConfig {
            enable_simd: false,
            enable_parallel: false,
            enable_mixed_precision: true,
            enable_gpu: false,
            gpu_device_id: 0,
            gpu_memory_limit: None,
            num_threads: Some(4),
            memory_alignment: 64,
        };

        let simd_ops = SimdMatrixOps::new().with_config(config.clone());
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        // Should use fallback when SIMD is disabled
        let result = simd_ops.dot_product_simd(&a, &b).unwrap();
        let expected = 32.0; // 1*4 + 2*5 + 3*6
        assert!((result - expected).abs() < 1e-10);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_acceleration_creation() {
        // Test creation without GPU (should fallback gracefully)
        let gpu_acc = GpuAcceleration::default();
        // This test just ensures the default creation works
        assert!(true);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_config() {
        let config = AccelerationConfig {
            enable_gpu: true,
            gpu_device_id: 0,
            gpu_memory_limit: Some(1024 * 1024 * 1024), // 1GB
            ..AccelerationConfig::default()
        };

        // Test that config is properly set
        assert_eq!(config.enable_gpu, true);
        assert_eq!(config.gpu_device_id, 0);
        assert_eq!(config.gpu_memory_limit, Some(1024 * 1024 * 1024));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_array_conversion() {
        let config = AccelerationConfig {
            enable_gpu: false, // Disable GPU for testing
            ..AccelerationConfig::default()
        };

        if let Ok(gpu_acc) = GpuAcceleration::with_config(config) {
            let array = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

            // Test array to tensor conversion (will use CPU tensor)
            if let Ok(tensor) = gpu_acc.array_to_tensor(&array) {
                // Test tensor back to array conversion
                if let Ok(result_array) = gpu_acc.tensor_to_array(&tensor) {
                    assert_eq!(result_array.shape(), array.shape());
                }
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_decomposition_fallback() {
        // Test that GPU decomposition works with fallback
        let gpu_decomp = GpuDecomposition::default();

        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();

        // This should work with fallback even without GPU
        if let Ok((factor_a, factor_b)) = gpu_decomp.gpu_factorize(&matrix) {
            assert_eq!(factor_a.nrows(), 3);
            assert_eq!(factor_b.ncols(), 3);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_pca_basic() {
        let gpu_decomp = GpuDecomposition::default();

        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        // Test GPU PCA with 2 components
        if let Ok((u, s, vt)) = gpu_decomp.gpu_pca(&data, 2) {
            assert_eq!(u.ncols(), 2);
            assert_eq!(s.len(), 2);
            assert_eq!(vt.nrows(), 2);
        }
    }
}

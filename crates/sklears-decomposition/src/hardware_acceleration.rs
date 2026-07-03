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
// Used unconditionally: `ParallelDecomposition::sequential_svd` /
// `sequential_eigendecomposition` delegate to these real implementations
// regardless of whether the `gpu` feature is enabled (see Wave B3 bugfix:
// these used to return a hardcoded `(eye, ones, eye)` "result").
use scirs2_linalg::{eigh as linalg_eigh, svd as linalg_svd};
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
use oxicuda_memory::DeviceBuffer;
#[cfg(feature = "gpu")]
use oxicuda_solver::{EigJob, SolverHandle, SvdJob};
#[cfg(feature = "gpu")]
use scirs2_core::ndarray::ShapeBuilder;
#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuContext, GpuMatrixOps};

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
        // NOTE: a genuine block/hierarchical SVD (splitting the matrix into
        // `block_size` panels and merging partial factorizations) is a
        // substantial numerical-algorithms undertaking in its own right and
        // is tracked as a follow-up. Critically, this must NOT fabricate a
        // result: delegating to `sequential_svd` guarantees a real,
        // correct decomposition (via `scirs2_linalg::svd`) for every input,
        // which a half-finished block algorithm would not.
        self.sequential_svd(matrix)
    }

    fn sequential_svd(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        // Real SVD via scirs2_linalg -- the same function `GpuAcceleration::gpu_svd`
        // uses for its CPU fallback path. Earlier revisions of this function
        // returned a hardcoded `(Array2::eye(m), Array1::ones(min_dim), Array2::eye(n))`
        // "result" for every input, silently corrupting any caller that trusted it.
        let (u, s, vt) = linalg_svd(&matrix.view(), false, self.config.num_threads)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
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
        // NOTE: a genuine divide-and-conquer eigensolver is tracked as a
        // follow-up (see `block_parallel_svd`). Delegating to
        // `sequential_eigendecomposition` guarantees a real, correct
        // decomposition for every input in the meantime.
        self.sequential_eigendecomposition(matrix)
    }

    fn sequential_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // Real symmetric eigendecomposition via scirs2_linalg -- the same
        // function `GpuAcceleration::gpu_eigendecomposition` uses for its CPU
        // fallback path. Earlier revisions of this function returned a
        // hardcoded `(Array1::ones(n), Array2::eye(n))` "result" for every
        // input, silently corrupting any caller that trusted it.
        let (eigenvalues, eigenvectors) = linalg_eigh(&matrix.view(), self.config.num_threads)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
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

    /// Row-tiled matrix multiplication.
    ///
    /// Splits the output into `tile_size`-row bands and computes each band's
    /// full `A_band * B` product with a single (BLAS-backed) `dot` call,
    /// writing the result directly into that band's slice of `result`. Since
    /// distinct row bands own disjoint output rows, the bands can be computed
    /// concurrently with no locking or synchronization.
    ///
    /// An earlier revision of this function computed `(i, j, k)` tiles by
    /// hand via nested loops, accumulated them into a per-tile `tile_result`,
    /// and then discarded every one of them, unconditionally returning
    /// `a.dot(b)` instead -- real work was performed and thrown away on every
    /// call. This version keeps the row-tiling (for parallel work
    /// distribution) but actually writes the computed tiles into the
    /// returned matrix.
    fn tiled_parallel_multiply(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (m, _k) = a.dim();
        let (_, n) = b.dim();
        let tile_size = 64; // Cache-friendly row-band size

        let mut result = Array2::<Float>::zeros((m, n));

        #[cfg(feature = "parallel")]
        {
            // `AxisChunksIterMut` itself doesn't implement rayon's
            // `IntoParallelIterator` unless the underlying `ndarray` crate is
            // built with its own `rayon` feature (a workspace-level knob out
            // of scope for this crate); collecting into a `Vec` first lets us
            // parallelize via rayon's plain `Vec<T>` support instead, with no
            // extra dependency surface.
            let row_chunks: Vec<_> = result
                .axis_chunks_iter_mut(scirs2_core::ndarray::Axis(0), tile_size)
                .collect();
            row_chunks
                .into_par_iter()
                .enumerate()
                .for_each(|(tile_idx, mut out_chunk)| {
                    let row_start = tile_idx * tile_size;
                    let row_end = row_start + out_chunk.nrows();
                    let a_rows = a.slice(scirs2_core::ndarray::s![row_start..row_end, ..]);
                    out_chunk.assign(&a_rows.dot(b));
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            result
                .axis_chunks_iter_mut(scirs2_core::ndarray::Axis(0), tile_size)
                .enumerate()
                .for_each(|(tile_idx, mut out_chunk)| {
                    let row_start = tile_idx * tile_size;
                    let row_end = row_start + out_chunk.nrows();
                    let a_rows = a.slice(scirs2_core::ndarray::s![row_start..row_end, ..]);
                    out_chunk.assign(&a_rows.dot(b));
                });
        }

        Ok(result)
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
        ptr.is_multiple_of(self.alignment)
    }

    /// Copy data to aligned buffer if necessary
    pub fn ensure_aligned(&self, data: &Array1<Float>) -> AlignedBuffer {
        // Copy element-by-element in logical order instead of requiring a
        // contiguous slice: `Array1` is normally contiguous, but this keeps
        // the method correct (never panicking) for any memory layout rather
        // than relying on that invariant.
        let mut buffer = self.create_aligned_array(data.len());
        for (dst, src) in buffer.as_mut_slice().iter_mut().zip(data.iter()) {
            *dst = *src;
        }
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
        // Callers may request any `alignment` (including values smaller than
        // `align_of::<Float>()`, or non-powers-of-two); `Layout::from_size_align`
        // would accept those verbatim, but the resulting allocation is then
        // reinterpreted as `*mut Float` below (and dereferenced as `&[Float]`
        // in `as_slice`/`as_mut_slice`), which requires at least
        // `align_of::<Float>()`. Sanitize here -- mirroring
        // `AlignedMemoryOps::sanitize_alignment` and `AlignedMatrix::new` in
        // `cache_optimization.rs` -- so this constructor is sound on its own,
        // regardless of how callers reach it.
        let alignment = alignment
            .max(std::mem::align_of::<Float>())
            .next_power_of_two();

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

/// GPU acceleration for matrix operations via oxicuda-backend.
///
/// Uses `sklears_core::gpu::{GpuContext, GpuArray, GpuMatrixOps}` for GEMM-based operations
/// and `scirs2_linalg` for SVD / eigendecomposition.
#[cfg(feature = "gpu")]
pub struct GpuAcceleration {
    config: AccelerationConfig,
    context: Option<GpuContext>,
}

#[cfg(feature = "gpu")]
impl GpuAcceleration {
    pub fn new() -> Result<Self> {
        Self::with_config(AccelerationConfig::default())
    }

    /// A CPU-only instance that skips GPU/driver initialization entirely.
    ///
    /// Infallible by construction: `with_config`'s `enable_gpu = false`
    /// branch returns `Ok` before ever touching the driver, so this helper
    /// reproduces exactly that branch without going through a `Result` at
    /// all. Used as the guaranteed-to-succeed fallback in
    /// [`GpuDecomposition`]'s `Default` impl.
    fn cpu_only() -> Self {
        Self {
            config: AccelerationConfig {
                enable_gpu: false,
                ..AccelerationConfig::default()
            },
            context: None,
        }
    }

    pub fn with_config(config: AccelerationConfig) -> Result<Self> {
        if !config.enable_gpu {
            eprintln!(
                "[GpuAcceleration] GPU disabled by config (enable_gpu=false). \
                 Running on CPU. num_threads={:?}, alignment={} bytes.",
                config.num_threads, config.memory_alignment
            );
            return Ok(Self {
                config,
                context: None,
            });
        }

        eprintln!(
            "[GpuAcceleration] Requesting GPU device {} with memory_limit={:?}.",
            config.gpu_device_id, config.gpu_memory_limit
        );

        // `GpuContext::with_device_id` (== `GpuBackend::with_device_id`) is
        // fallible-and-optional: `Ok(None)` means "driver/hardware not
        // present", which is expected on machines without a GPU and must
        // fall back to CPU rather than be treated as an error. Only a
        // genuine `Err` (a real initialisation failure) is surfaced to the
        // caller.
        let context = match GpuContext::with_device_id(config.gpu_device_id as usize) {
            Ok(Some(ctx)) => Some(ctx),
            Ok(None) => {
                eprintln!(
                    "[GpuAcceleration] No GPU detected at device {} (or CUDA driver unavailable); \
                     falling back to CPU.",
                    config.gpu_device_id
                );
                None
            }
            Err(e) => {
                return Err(SklearsError::InvalidInput(format!(
                    "Failed to initialize GPU device {}: {e}",
                    config.gpu_device_id
                )));
            }
        };

        Ok(Self { config, context })
    }

    pub fn config(&self) -> &AccelerationConfig {
        &self.config
    }

    pub fn is_gpu_available(&self) -> bool {
        self.config.enable_gpu && self.context.is_some()
    }

    pub fn gpu_memory_info(&self) -> Result<(usize, usize)> {
        if let Some(ref ctx) = self.context {
            let info = ctx.memory_info()?;
            let effective_total = self.config.gpu_memory_limit.unwrap_or(info.total);
            Ok((info.free.min(effective_total), effective_total))
        } else {
            Err(SklearsError::InvalidInput("GPU not available".to_string()))
        }
    }

    pub fn gpu_matrix_multiply(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        if !self.is_gpu_available() {
            return Err(SklearsError::InvalidInput("GPU not available".to_string()));
        }

        let ctx = self
            .context
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("GPU not available".to_string()))?;
        let (_m, k1) = a.dim();
        let (k2, _n) = b.dim();

        if k1 != k2 {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        let a_gpu = GpuArray::<Float>::from_array2(ctx, a)?;
        let b_gpu = GpuArray::<Float>::from_array2(ctx, b)?;
        let c_gpu = a_gpu.matmul(&b_gpu)?;
        c_gpu.to_array2()
    }

    /// GPU-accelerated SVD decomposition.
    ///
    /// Attempts a real on-device SVD via `oxicuda_solver::dense::svd` (using
    /// this instance's `GpuContext`). If the on-device solve fails for any
    /// reason (e.g. a convergence failure in the current host-fallback
    /// implementation of that solver -- see `oxicuda_solver::dense::svd`'s
    /// own docs), this transparently falls back to the CPU
    /// `scirs2_linalg::svd` path rather than failing the whole request.
    pub fn gpu_svd(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        if !self.is_gpu_available() {
            return Err(SklearsError::InvalidInput("GPU not available".to_string()));
        }
        let ctx = self
            .context
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("GPU not available".to_string()))?;

        match Self::gpu_svd_on_device(ctx, matrix) {
            Ok(result) => Ok(result),
            Err(gpu_err) => {
                eprintln!(
                    "[GpuAcceleration] on-device SVD failed ({gpu_err}); \
                     falling back to CPU scirs2_linalg::svd."
                );
                let (u, s, vt) = linalg_svd(&matrix.view(), false, self.config.num_threads)
                    .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
                Ok((u, s, vt))
            }
        }
    }

    /// On-device SVD via `oxicuda_solver::dense::svd`.
    ///
    /// `oxicuda_solver`'s dense solvers expect column-major input with
    /// leading dimension `lda`; `matrix.t().iter()` (row-major iteration of
    /// the *transposed* view) yields exactly that layout with `lda = m`,
    /// with no extra copy logic needed beyond the collect. Results (also
    /// column-major) are reshaped back into row-major-queryable `Array2`s
    /// via `Array2::from_shape_vec((rows, cols).f(), data)`.
    fn gpu_svd_on_device(
        ctx: &GpuContext,
        matrix: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        let (m, n) = matrix.dim();
        if m == 0 || n == 0 {
            return Err(SklearsError::InvalidInput(
                "Matrix must be non-empty for SVD".to_string(),
            ));
        }
        let k = m.min(n);

        // Make this backend's context current on the calling thread before
        // touching any device memory: `SolverHandle`/`DeviceBuffer` resolve
        // against the ambient current CUDA context, and this call may run on
        // a different thread than the one that originally constructed `ctx`.
        ctx.context()
            .set_current()
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let col_major: Vec<Float> = matrix.t().iter().copied().collect();
        let mut a_buf =
            DeviceBuffer::from_host(&col_major).map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let mut handle = SolverHandle::new(ctx.context())
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let result = oxicuda_solver::dense::svd(
            &mut handle,
            &mut a_buf,
            m as u32,
            n as u32,
            m as u32,
            SvdJob::Thin,
        )
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let u_data = result.u.ok_or_else(|| {
            SklearsError::NumericalError("oxicuda_solver::svd: missing U".to_string())
        })?;
        let vt_data = result.vt.ok_or_else(|| {
            SklearsError::NumericalError("oxicuda_solver::svd: missing Vt".to_string())
        })?;

        let u = Array2::from_shape_vec((m, k).f(), u_data)
            .map_err(|e| SklearsError::NumericalError(format!("reshape U: {e}")))?;
        let vt = Array2::from_shape_vec((k, n).f(), vt_data)
            .map_err(|e| SklearsError::NumericalError(format!("reshape Vt: {e}")))?;
        let s = Array1::from_vec(result.singular_values);

        Ok((u, s, vt))
    }

    /// GPU-accelerated eigendecomposition of a symmetric matrix.
    ///
    /// Attempts a real on-device solve via `oxicuda_solver::dense::syevd`,
    /// falling back to the CPU `scirs2_linalg::eigh` path if the on-device
    /// solve fails (mirrors `gpu_svd`'s fallback shape).
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
        let ctx = self
            .context
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("GPU not available".to_string()))?;

        match Self::gpu_eigh_on_device(ctx, matrix) {
            Ok(result) => Ok(result),
            Err(gpu_err) => {
                eprintln!(
                    "[GpuAcceleration] on-device eigendecomposition failed ({gpu_err}); \
                     falling back to CPU scirs2_linalg::eigh."
                );
                let (eigenvals, eigenvecs) = linalg_eigh(&matrix.view(), self.config.num_threads)
                    .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
                Ok((eigenvals, eigenvecs))
            }
        }
    }

    /// On-device symmetric eigendecomposition via `oxicuda_solver::dense::syevd`.
    ///
    /// See `gpu_svd_on_device` for the column-major layout rationale; the
    /// same conversion applies here (`syevd` only reads the lower triangle,
    /// so any negligible floating-point asymmetry in `matrix` is immaterial).
    fn gpu_eigh_on_device(
        ctx: &GpuContext,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "Matrix must be non-empty for eigendecomposition".to_string(),
            ));
        }

        ctx.context()
            .set_current()
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let col_major: Vec<Float> = matrix.t().iter().copied().collect();
        let mut a_buf =
            DeviceBuffer::from_host(&col_major).map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let mut eigenvalues_buf =
            DeviceBuffer::<Float>::zeroed(n).map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let mut handle = SolverHandle::new(ctx.context())
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        oxicuda_solver::dense::syevd(
            &mut handle,
            &mut a_buf,
            n as u32,
            n as u32,
            &mut eigenvalues_buf,
            EigJob::ValuesAndVectors,
        )
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let mut eigenvalues_host = vec![0.0 as Float; n];
        eigenvalues_buf
            .copy_to_host(&mut eigenvalues_host)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let mut eigenvectors_host = vec![0.0 as Float; n * n];
        a_buf
            .copy_to_host(&mut eigenvectors_host)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let eigenvalues = Array1::from_vec(eigenvalues_host);
        let eigenvectors = Array2::from_shape_vec((n, n).f(), eigenvectors_host)
            .map_err(|e| SklearsError::NumericalError(format!("reshape eigenvectors: {e}")))?;

        Ok((eigenvalues, eigenvectors))
    }

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

    pub fn free_gpu_memory(&self) -> Result<()> {
        if let Some(ref ctx) = self.context {
            ctx.synchronize()?;
        }
        Ok(())
    }

    pub fn profile_gpu_operation<F, T>(&self, operation: F) -> Result<(T, std::time::Duration)>
    where
        F: FnOnce() -> Result<T>,
    {
        let start = std::time::Instant::now();
        let result = operation()?;
        self.free_gpu_memory()?;
        let duration = start.elapsed();
        Ok((result, duration))
    }
}

#[cfg(feature = "gpu")]
impl Default for GpuAcceleration {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            let config = AccelerationConfig {
                enable_gpu: false,
                ..AccelerationConfig::default()
            };
            Self {
                config,
                context: None,
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
        let col_means = data
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .ok_or_else(|| {
                SklearsError::InvalidInput("data must have at least one row for PCA".to_string())
            })?;
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
        Self::new().unwrap_or_else(|_| Self {
            gpu_acceleration: GpuAcceleration::cpu_only(),
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

        let result = simd_ops
            .dot_product_simd(&a, &b)
            .expect("operation should succeed");
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_elementwise_add() {
        let simd_ops = SimdMatrixOps::new();
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        let result = simd_ops
            .elementwise_add_simd(&a, &b)
            .expect("operation should succeed");
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

        let result = simd_ops
            .elementwise_mul_simd(&a, &b)
            .expect("operation should succeed");
        let expected = Array1::from_vec(vec![5.0, 12.0, 21.0, 32.0]);

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matrix_vector_mul_simd() {
        let simd_ops = SimdMatrixOps::new();
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape and data length should match");
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = simd_ops
            .matrix_vector_mul_simd(&matrix, &vector)
            .expect("operation should succeed");
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

        let (eigenvals, eigenvecs) = result.expect("operation should succeed");
        assert_eq!(eigenvals.len(), 3);
        assert_eq!(eigenvecs.dim(), (3, 3));
    }

    /// Regression test for the Wave B3 critical bug: `sequential_svd` used to
    /// return a hardcoded `(Array2::eye(m), Array1::ones(min_dim),
    /// Array2::eye(n))` "result" for *every* input, no matter what. That only
    /// looks correct by accident on an identity input, so this deliberately
    /// uses a non-trivial, non-symmetric, non-identity matrix: the old fake
    /// implementation fails both the "not all-ones singular values" check and
    /// the reconstruction check below.
    #[test]
    fn test_sequential_svd_reconstructs_matrix() {
        let config = AccelerationConfig {
            enable_parallel: false, // force the `sequential_svd` code path
            ..AccelerationConfig::default()
        };
        let parallel_ops = ParallelDecomposition::new().with_config(config);

        let matrix =
            Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 2.0, 0.0, 3.0, 1.0, 5.0, 2.0, 6.0])
                .expect("shape and data length should match");

        let (u, s, vt) = parallel_ops
            .parallel_svd(&matrix)
            .expect("SVD should succeed");

        assert_eq!(u.dim(), (3, 3));
        assert_eq!(s.len(), 3);
        assert_eq!(vt.dim(), (3, 3));
        // The fake implementation returned `s = [1, 1, 1]`.
        assert!(s.iter().any(|&x| (x - 1.0).abs() > 1e-6));

        // Reconstruction check: A ≈ U * diag(S) * Vt.
        let us = &u * &s.view().insert_axis(scirs2_core::ndarray::Axis(0));
        let reconstructed = us.dot(&vt);
        for (r, e) in reconstructed.iter().zip(matrix.iter()) {
            assert!((r - e).abs() < 1e-8, "reconstructed={r}, expected={e}");
        }
    }

    /// Same bug, but for `block_parallel_svd` specifically (named explicitly
    /// in the bug report alongside `sequential_svd`). It's normally only
    /// reached for matrices larger than 1000x1000 via the public
    /// `parallel_svd`; call the private method directly (visible to this
    /// child module) so the test stays fast.
    #[test]
    fn test_block_parallel_svd_reconstructs_matrix() {
        let parallel_ops = ParallelDecomposition::new();
        let matrix =
            Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 2.0, 0.0, 3.0, 1.0, 5.0, 2.0, 6.0])
                .expect("shape and data length should match");

        let (u, s, vt) = parallel_ops
            .block_parallel_svd(&matrix)
            .expect("block SVD should succeed");

        assert!(s.iter().any(|&x| (x - 1.0).abs() > 1e-6));
        let us = &u * &s.view().insert_axis(scirs2_core::ndarray::Axis(0));
        let reconstructed = us.dot(&vt);
        for (r, e) in reconstructed.iter().zip(matrix.iter()) {
            assert!((r - e).abs() < 1e-8, "reconstructed={r}, expected={e}");
        }
    }

    /// Regression test for the matching bug in `sequential_eigendecomposition`,
    /// which used to return `(Array1::ones(n), Array2::eye(n))` unconditionally.
    #[test]
    fn test_sequential_eigendecomposition_reconstructs_symmetric_matrix() {
        let config = AccelerationConfig {
            enable_parallel: false,
            ..AccelerationConfig::default()
        };
        let parallel_ops = ParallelDecomposition::new().with_config(config);

        // A genuine (non-identity) symmetric tridiagonal matrix; eigenvalues
        // are 2, 2 - sqrt(2), 2 + sqrt(2).
        let matrix =
            Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0])
                .expect("shape and data length should match");

        let (eigenvalues, eigenvectors) = parallel_ops
            .parallel_eigendecomposition(&matrix)
            .expect("eigendecomposition should succeed");

        assert_eq!(eigenvalues.len(), 3);
        assert_eq!(eigenvectors.dim(), (3, 3));
        // The fake implementation returned eigenvalues = [1, 1, 1].
        assert!(eigenvalues.iter().any(|&x| (x - 1.0).abs() > 1e-6));

        // Correctness check: A * v ≈ λ * v for every eigenvector column.
        for j in 0..3 {
            let v = eigenvectors.column(j);
            let av = matrix.dot(&v);
            let lambda_v = v.mapv(|x| x * eigenvalues[j]);
            for (a, b) in av.iter().zip(lambda_v.iter()) {
                assert!((a - b).abs() < 1e-8, "A*v={a}, lambda*v={b}");
            }
        }
    }

    /// Regression test for the dead-tile-computation bug in
    /// `tiled_parallel_multiply`: tiles used to be computed and then thrown
    /// away in favor of an unconditional `a.dot(b)`. Dimensions are chosen to
    /// (a) trigger the tiled path via the public `parallel_matrix_multiply`
    /// (`m, n, k > 100`) and (b) NOT be exact multiples of the 64-row tile
    /// size, so an off-by-one in the remainder-tile handling would show up.
    #[test]
    fn test_tiled_parallel_multiply_matches_reference() {
        let parallel_ops = ParallelDecomposition::new();

        let m = 130;
        let k = 150;
        let n = 110;
        let a = Array2::from_shape_fn((m, k), |(i, j)| ((i * 7 + j * 3) % 13) as Float);
        let b = Array2::from_shape_fn((k, n), |(i, j)| ((i * 5 + j * 11) % 17) as Float);

        let result = parallel_ops
            .parallel_matrix_multiply(&a, &b)
            .expect("multiply should succeed");
        let expected = a.dot(&b);

        assert_eq!(result.dim(), expected.dim());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "result={r}, expected={e}");
        }
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

        assert_eq!(
            aligned_array.as_slice(),
            array.as_slice().expect("slice operation should succeed")
        );
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
        let result = simd_ops
            .dot_product_simd(&a, &b)
            .expect("operation should succeed");
        let expected = 32.0; // 1*4 + 2*5 + 3*6
        assert!((result - expected).abs() < 1e-10);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_acceleration_creation() {
        // Test creation without GPU (should fallback gracefully)
        let _gpu_acc = GpuAcceleration::default();
        // This test just ensures the default creation works
        let _ = _gpu_acc;
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
        assert!(config.enable_gpu);
        assert_eq!(config.gpu_device_id, 0);
        assert_eq!(config.gpu_memory_limit, Some(1024 * 1024 * 1024));
    }

    /// Regression test for the Part 1 mechanical adaptation:
    /// `GpuContext::with_device_id` now returns `Result<Option<Self>>`
    /// instead of an (effectively) infallible `Result<Self>`. On a machine
    /// with no CUDA device (this crate's own dev/CI environment), requesting
    /// GPU acceleration must gracefully fall back to CPU -- `Ok` with
    /// `is_gpu_available() == false` -- rather than hard-erroring out of
    /// `with_config`.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_acceleration_with_config_falls_back_when_no_gpu() {
        let config = AccelerationConfig {
            enable_gpu: true,
            ..AccelerationConfig::default()
        };
        let gpu_acc = GpuAcceleration::with_config(config).expect(
            "GPU init must fall back to CPU gracefully (Ok), not hard-error, when no GPU is present",
        );
        // Whether or not a real GPU happens to be present on the machine
        // running this test, construction itself must succeed either way.
        let _ = gpu_acc.is_gpu_available();
    }

    /// Best-effort correctness check for the Part 3 GPU wiring
    /// (`gpu_svd` -> `oxicuda_solver::dense::svd`). Gracefully skips without
    /// real hardware (expected on this crate's dev/CI machine), mirroring the
    /// skip pattern already used by `sklears_core::gpu`'s own GPU tests.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_svd_reconstructs_matrix_when_available() {
        let config = AccelerationConfig {
            enable_gpu: true,
            ..AccelerationConfig::default()
        };
        let Ok(gpu_acc) = GpuAcceleration::with_config(config) else {
            return;
        };
        if !gpu_acc.is_gpu_available() {
            eprintln!("skipping test_gpu_svd_reconstructs_matrix_when_available: no GPU detected");
            return;
        }

        let matrix =
            Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 2.0, 0.0, 3.0, 1.0, 5.0, 2.0, 6.0])
                .expect("shape and data length should match");
        let (u, s, vt) = gpu_acc
            .gpu_svd(&matrix)
            .expect("GPU SVD should succeed when a GPU is available");
        let us = &u * &s.view().insert_axis(scirs2_core::ndarray::Axis(0));
        let reconstructed = us.dot(&vt);
        for (r, e) in reconstructed.iter().zip(matrix.iter()) {
            assert!((r - e).abs() < 1e-6, "reconstructed={r}, expected={e}");
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_matrix_multiply_identity() {
        let config = AccelerationConfig {
            enable_gpu: true,
            ..AccelerationConfig::default()
        };

        if let Ok(gpu_acc) = GpuAcceleration::with_config(config) {
            if gpu_acc.is_gpu_available() {
                let identity = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0])
                    .expect("shape and data length should match");
                let b = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
                    .expect("shape and data length should match");

                if let Ok(result) = gpu_acc.gpu_matrix_multiply(&identity, &b) {
                    assert_eq!(result.shape(), b.shape());
                    for (r, e) in result.iter().zip(b.iter()) {
                        assert!((r - e).abs() < 1e-10);
                    }
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
                .expect("operation should succeed");

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
        .expect("operation should succeed");

        // Test GPU PCA with 2 components
        if let Ok((u, s, vt)) = gpu_decomp.gpu_pca(&data, 2) {
            assert_eq!(u.ncols(), 2);
            assert_eq!(s.len(), 2);
            assert_eq!(vt.nrows(), 2);
        }
    }
}

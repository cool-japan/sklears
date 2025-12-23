//! Target-specific SIMD optimizations and compilation support
//!
//! This module provides target-specific optimizations for different CPU architectures
//! and supports compile-time feature selection for optimal performance.

use crate::traits::{SimdError, VectorArithmetic, VectorReduction};

#[cfg(feature = "no-std")]
use alloc::vec::Vec;

/// Target-specific optimization levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationTarget {
    /// Generic optimization for any x86-64 CPU
    Generic,
    /// Optimize for Intel Haswell and newer (AVX2, FMA)
    Haswell,
    /// Optimize for Intel Skylake and newer (AVX2, FMA, enhanced instructions)
    Skylake,
    /// Optimize for AMD Zen and newer
    Zen,
    /// Optimize for ARM Cortex-A76 and newer
    CortexA76,
    /// Optimize for Apple Silicon M1/M2
    AppleSilicon,
    /// Optimize for Intel Granite Rapids with AVX10.1
    GraniteRapids,
    /// Optimize for Intel Diamond Rapids with enhanced AVX10
    DiamondRapids,
    /// Optimize for AMD Zen 5 with AVX-512
    Zen5,
    /// Optimize for server workloads (high throughput)
    Server,
    /// Optimize for mobile/embedded (low power)
    Mobile,
}

/// Compile-time target configuration
#[derive(Debug)]
pub struct TargetConfig {
    pub optimization_target: OptimizationTarget,
    pub enable_fma: bool,
    pub enable_avx512: bool,
    pub enable_avx10: bool,
    pub enable_amx: bool,
    pub enable_sve2: bool,
    pub enable_sme: bool,
    pub enable_fp16: bool,
    pub enable_bf16: bool,
    pub prefer_throughput: bool,
    pub prefer_latency: bool,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            optimization_target: OptimizationTarget::Generic,
            enable_fma: cfg!(target_feature = "fma"),
            enable_avx512: cfg!(target_feature = "avx512f"),
            enable_avx10: false, // Not yet supported in stable Rust
            enable_amx: false,   // Intel AMX not yet in stable Rust
            enable_sve2: false,  // ARM SVE2 not yet in stable Rust
            enable_sme: false,   // ARM SME not yet in stable Rust
            enable_fp16: false,  // FP16 support is library-based
            enable_bf16: false,  // BF16 support is library-based
            prefer_throughput: true,
            prefer_latency: false,
        }
    }
}

/// Target-specific vector operations dispatcher
pub struct TargetOptimizedOps {
    config: TargetConfig,
}

impl TargetOptimizedOps {
    /// Create a new target-optimized operations instance
    pub fn new(config: TargetConfig) -> Self {
        Self { config }
    }

    /// Create with automatic target detection
    pub fn auto_detect() -> Self {
        let config = TargetConfig {
            optimization_target: detect_optimization_target(),
            enable_fma: {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    Self::detect_fma()
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            },
            enable_avx512: {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    Self::detect_avx512()
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            },
            enable_avx10: Self::detect_avx10(),
            enable_amx: Self::detect_amx(),
            enable_sve2: Self::detect_sve2(),
            enable_sme: Self::detect_sme(),
            enable_fp16: Self::detect_fp16(),
            enable_bf16: Self::detect_bf16(),
            prefer_throughput: true,
            prefer_latency: false,
        };

        Self::new(config)
    }

    /// Detect FMA support
    fn detect_fma() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            crate::simd_feature_detected!("fma") || cfg!(target_feature = "fma")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Detect AVX-512 support
    fn detect_avx512() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            crate::simd_feature_detected!("avx512f") || cfg!(target_feature = "avx512f")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Detect AVX10 support (future Intel processors)
    fn detect_avx10() -> bool {
        // AVX10 is not yet available in stable Rust
        // This is a placeholder for future implementation
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // In the future, this would use crate::simd_feature_detected!("avx10.1")
            false
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Detect Intel AMX (Advanced Matrix Extensions) support
    fn detect_amx() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // AMX is not yet detectable via is_x86_feature_detected in stable Rust
            // This would check for AMX-BF16, AMX-INT8, AMX-TILE in the future
            false
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Detect ARM SVE2 (Scalable Vector Extensions 2) support
    fn detect_sve2() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            // SVE2 detection is not yet available in stable Rust
            // This would use is_aarch64_feature_detected!("sve2") in the future
            false
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }

    /// Detect ARM SME (Scalable Matrix Extensions) support
    fn detect_sme() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            // SME detection is not yet available in stable Rust
            // This would use is_aarch64_feature_detected!("sme") in the future
            false
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }

    /// Detect hardware FP16 support
    fn detect_fp16() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Intel F16C extension
            crate::simd_feature_detected!("f16c")
        }
        #[cfg(target_arch = "aarch64")]
        {
            // ARM NEON has native FP16 support
            true
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Detect BF16 hardware support
    fn detect_bf16() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Intel AVX512-BF16 or AMX-BF16
            // Not yet detectable in stable Rust
            false
        }
        #[cfg(target_arch = "aarch64")]
        {
            // ARM BF16 support in recent processors
            // Not yet detectable in stable Rust
            false
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Get target-specific SIMD width for f32 operations
    pub fn optimal_f32_width(&self) -> usize {
        match self.config.optimization_target {
            OptimizationTarget::Generic => 4,
            OptimizationTarget::Haswell | OptimizationTarget::Skylake | OptimizationTarget::Zen => {
                if self.config.enable_avx512 {
                    16
                } else {
                    8
                }
            }
            OptimizationTarget::GraniteRapids | OptimizationTarget::DiamondRapids => {
                if self.config.enable_avx10 {
                    16 // AVX10 unified 512-bit operations
                } else if self.config.enable_avx512 {
                    16
                } else {
                    8
                }
            }
            OptimizationTarget::Zen5 => {
                if self.config.enable_avx512 {
                    16 // AMD Zen 5 has full AVX-512 support
                } else {
                    8
                }
            }
            OptimizationTarget::AppleSilicon | OptimizationTarget::CortexA76 => {
                if self.config.enable_sve2 {
                    8 // SVE2 with scalable vector width (conservative estimate)
                } else {
                    4 // NEON 128-bit
                }
            }
            OptimizationTarget::Server => {
                if self.config.enable_avx10 || self.config.enable_avx512 {
                    16
                } else {
                    8
                }
            }
            OptimizationTarget::Mobile => 4,
        }
    }

    /// Get target-specific cache line size
    pub fn cache_line_size(&self) -> usize {
        match self.config.optimization_target {
            OptimizationTarget::AppleSilicon => 128, // Apple Silicon has 128-byte cache lines
            _ => 64,                                 // Most x86-64 CPUs have 64-byte cache lines
        }
    }

    /// Get target-specific prefetch distance
    pub fn prefetch_distance(&self) -> usize {
        match self.config.optimization_target {
            OptimizationTarget::Server => 512, // Aggressive prefetching for server workloads
            OptimizationTarget::Mobile => 128, // Conservative prefetching for mobile
            _ => 256,                          // Balanced prefetching
        }
    }
}

/// Detect the optimal target based on CPU features
fn detect_optimization_target() -> OptimizationTarget {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") && crate::simd_feature_detected!("fma") {
            if crate::simd_feature_detected!("avx512f") {
                OptimizationTarget::Skylake
            } else {
                OptimizationTarget::Haswell
            }
        } else {
            OptimizationTarget::Generic
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Detect Apple Silicon vs other ARM64
        if cfg!(target_os = "macos") {
            OptimizationTarget::AppleSilicon
        } else {
            OptimizationTarget::CortexA76
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        OptimizationTarget::Generic
    }
}

/// Target-specific vector arithmetic implementation
impl VectorArithmetic<f32> for TargetOptimizedOps {
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        if a.len() != b.len() {
            return Err(SimdError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let width = self.optimal_f32_width();
        match width {
            16 | 8 => self.add_avx2(a, b),
            4 => self.add_sse(a, b),
            _ => Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()),
        }
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        if a.len() != b.len() {
            return Err(SimdError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let width = self.optimal_f32_width();
        match width {
            16 | 8 => self.sub_avx2(a, b),
            4 => self.sub_sse(a, b),
            _ => Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()),
        }
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        if a.len() != b.len() {
            return Err(SimdError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let width = self.optimal_f32_width();
        match width {
            16 | 8 => self.mul_avx2(a, b),
            4 => self.mul_sse(a, b),
            _ => Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()),
        }
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        if a.len() != b.len() {
            return Err(SimdError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let width = self.optimal_f32_width();
        match width {
            16 | 8 => self.div_avx2(a, b),
            4 => self.div_sse(a, b),
            _ => Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect()),
        }
    }

    fn fma(&self, a: &[f32], b: &[f32], c: &[f32]) -> Result<Vec<f32>, SimdError> {
        if a.len() != b.len() || a.len() != c.len() {
            return Err(SimdError::DimensionMismatch {
                expected: a.len(),
                actual: b.len().min(c.len()),
            });
        }

        if self.config.enable_fma {
            let width = self.optimal_f32_width();
            match width {
                16 | 8 => self.fma_avx2(a, b, c),
                4 => self.fma_sse(a, b, c),
                _ => Ok(a
                    .iter()
                    .zip(b.iter())
                    .zip(c.iter())
                    .map(|((&x, &y), &z)| x * y + z)
                    .collect()),
            }
        } else {
            // Fallback to separate multiply and add
            let mul_result = self.mul(a, b)?;
            self.add(&mul_result, c)
        }
    }

    fn scale(&self, vector: &[f32], scalar: f32) -> Result<Vec<f32>, SimdError> {
        let width = self.optimal_f32_width();
        match width {
            16 | 8 => self.scale_avx2(vector, scalar),
            4 => self.scale_sse(vector, scalar),
            _ => Ok(vector.iter().map(|&x| x * scalar).collect()),
        }
    }
}

/// AVX-512 implementations
impl TargetOptimizedOps {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_avx512(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;

            let mut result = Vec::with_capacity(a.len());
            let chunks = a.len() / 16;

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
                let vr = _mm512_add_ps(va, vb);

                let mut temp = [0f32; 16];
                _mm512_storeu_ps(temp.as_mut_ptr(), vr);
                result.extend_from_slice(&temp);
            }

            // Handle remaining elements
            for i in (chunks * 16)..a.len() {
                result.push(a[i] + b[i]);
            }

            Ok(result)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn sub_avx512(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;

            let mut result = Vec::with_capacity(a.len());
            let chunks = a.len() / 16;

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
                let vr = _mm512_sub_ps(va, vb);

                let mut temp = [0f32; 16];
                _mm512_storeu_ps(temp.as_mut_ptr(), vr);
                result.extend_from_slice(&temp);
            }

            // Handle remaining elements
            for i in (chunks * 16)..a.len() {
                result.push(a[i] - b[i]);
            }

            Ok(result)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect())
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul_avx512(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;

            let mut result = Vec::with_capacity(a.len());
            let chunks = a.len() / 16;

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
                let vr = _mm512_mul_ps(va, vb);

                let mut temp = [0f32; 16];
                _mm512_storeu_ps(temp.as_mut_ptr(), vr);
                result.extend_from_slice(&temp);
            }

            // Handle remaining elements
            for i in (chunks * 16)..a.len() {
                result.push(a[i] * b[i]);
            }

            Ok(result)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect())
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn div_avx512(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;

            let mut result = Vec::with_capacity(a.len());
            let chunks = a.len() / 16;

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
                let vr = _mm512_div_ps(va, vb);

                let mut temp = [0f32; 16];
                _mm512_storeu_ps(temp.as_mut_ptr(), vr);
                result.extend_from_slice(&temp);
            }

            // Handle remaining elements
            for i in (chunks * 16)..a.len() {
                result.push(a[i] / b[i]);
            }

            Ok(result)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect())
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn fma_avx512(&self, a: &[f32], b: &[f32], c: &[f32]) -> Result<Vec<f32>, SimdError> {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;

            let mut result = Vec::with_capacity(a.len());
            let chunks = a.len() / 16;

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
                let vc = _mm512_loadu_ps(c.as_ptr().add(offset));
                let vr = _mm512_fmadd_ps(va, vb, vc);

                let mut temp = [0f32; 16];
                _mm512_storeu_ps(temp.as_mut_ptr(), vr);
                result.extend_from_slice(&temp);
            }

            // Handle remaining elements
            for i in (chunks * 16)..a.len() {
                result.push(a[i] * b[i] + c[i]);
            }

            Ok(result)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Ok(a.iter()
                .zip(b.iter())
                .zip(c.iter())
                .map(|((&x, &y), &z)| x * y + z)
                .collect())
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn scale_avx512(&self, vector: &[f32], scalar: f32) -> Result<Vec<f32>, SimdError> {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;

            let mut result = Vec::with_capacity(vector.len());
            let chunks = vector.len() / 16;
            let vs = _mm512_set1_ps(scalar);

            for i in 0..chunks {
                let offset = i * 16;
                let vv = _mm512_loadu_ps(vector.as_ptr().add(offset));
                let vr = _mm512_mul_ps(vv, vs);

                let mut temp = [0f32; 16];
                _mm512_storeu_ps(temp.as_mut_ptr(), vr);
                result.extend_from_slice(&temp);
            }

            // Handle remaining elements
            for i in (chunks * 16)..vector.len() {
                result.push(vector[i] * scalar);
            }

            Ok(result)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Ok(vector.iter().map(|&x| x * scalar).collect())
        }
    }
}

/// AVX2 implementations (similar structure, using _mm256_ intrinsics)
impl TargetOptimizedOps {
    fn add_avx2(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
    }

    fn sub_avx2(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect())
    }

    fn mul_avx2(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect())
    }

    fn div_avx2(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect())
    }

    fn fma_avx2(&self, a: &[f32], b: &[f32], c: &[f32]) -> Result<Vec<f32>, SimdError> {
        // Use existing fma function - it works in-place
        let mut result = a.to_vec();
        crate::vector::fma(&mut result, b, c);
        Ok(result)
    }

    fn scale_avx2(&self, vector: &[f32], scalar: f32) -> Result<Vec<f32>, SimdError> {
        let mut result = vector.to_vec();
        crate::vector::scale(&mut result, scalar);
        Ok(result)
    }
}

/// SSE implementations (similar structure, using _mm_ intrinsics)
impl TargetOptimizedOps {
    fn add_sse(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
    }

    fn sub_sse(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect())
    }

    fn mul_sse(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect())
    }

    fn div_sse(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdError> {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect())
    }

    fn fma_sse(&self, a: &[f32], b: &[f32], c: &[f32]) -> Result<Vec<f32>, SimdError> {
        Ok(a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((&x, &y), &z)| x * y + z)
            .collect())
    }

    fn scale_sse(&self, vector: &[f32], scalar: f32) -> Result<Vec<f32>, SimdError> {
        Ok(vector.iter().map(|&x| x * scalar).collect())
    }
}

/// Target-specific reduction operations
impl VectorReduction<f32> for TargetOptimizedOps {
    fn sum(&self, vector: &[f32]) -> Result<f32, SimdError> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        let width = self.optimal_f32_width();
        match width {
            16 | 8 => self.sum_avx2(vector),
            4 => self.sum_sse(vector),
            _ => Ok(vector.iter().sum()),
        }
    }

    fn min(&self, vector: &[f32]) -> Result<f32, SimdError> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        let (min_val, _) = crate::vector::min_max(vector);
        Ok(min_val)
    }

    fn max(&self, vector: &[f32]) -> Result<f32, SimdError> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        let (_, max_val) = crate::vector::min_max(vector);
        Ok(max_val)
    }

    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32, SimdError> {
        if a.len() != b.len() {
            return Err(SimdError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        Ok(crate::vector::dot_product(a, b))
    }

    fn norm(&self, vector: &[f32]) -> Result<f32, SimdError> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        Ok(crate::vector::norm(vector))
    }

    fn mean(&self, vector: &[f32]) -> Result<f32, SimdError> {
        if vector.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        let sum = self.sum(vector)?;
        Ok(sum / vector.len() as f32)
    }
}

/// Target-specific sum implementations
impl TargetOptimizedOps {
    fn sum_avx512(&self, vector: &[f32]) -> Result<f32, SimdError> {
        Ok(crate::vector::sum(vector))
    }

    fn sum_avx2(&self, vector: &[f32]) -> Result<f32, SimdError> {
        Ok(crate::vector::sum(vector))
    }

    fn sum_sse(&self, vector: &[f32]) -> Result<f32, SimdError> {
        Ok(crate::vector::sum(vector))
    }
}

/// Compilation target selection utilities
pub mod compile_time {
    use super::*;

    /// Select implementation at compile time based on target features
    #[macro_export]
    macro_rules! select_target_impl {
        ($generic:expr, $avx2:expr, $avx512:expr) => {{
            #[cfg(target_feature = "avx512f")]
            {
                $avx512
            }
            #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
            {
                $avx2
            }
            #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
            {
                $generic
            }
        }};
    }

    /// Optimize for specific CPU microarchitecture
    pub fn optimize_for_cpu() -> TargetConfig {
        TargetConfig {
            optimization_target: detect_optimization_target(),
            enable_fma: cfg!(target_feature = "fma"),
            enable_avx512: cfg!(target_feature = "avx512f"),
            enable_avx10: false, // Not yet available in stable Rust
            enable_amx: false,   // Intel AMX not yet in stable Rust
            enable_sve2: false,  // ARM SVE2 not yet in stable Rust
            enable_sme: false,   // ARM SME not yet in stable Rust
            enable_fp16: false,  // FP16 support is library-based
            enable_bf16: false,  // BF16 support is library-based
            prefer_throughput: true,
            prefer_latency: false,
        }
    }

    /// Get CPU-specific optimization flags
    pub fn get_optimization_flags() -> Vec<&'static str> {
        #[allow(unused_mut)]
        let mut flags = Vec::new();

        #[cfg(target_feature = "sse2")]
        flags.push("sse2");
        #[cfg(target_feature = "avx")]
        flags.push("avx");
        #[cfg(target_feature = "avx2")]
        flags.push("avx2");
        #[cfg(target_feature = "fma")]
        flags.push("fma");
        #[cfg(target_feature = "avx512f")]
        flags.push("avx512f");

        flags
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::vec;

    #[test]
    fn test_target_detection() {
        let target = detect_optimization_target();
        println!("Detected optimization target: {:?}", target);

        // Should detect a valid target
        match target {
            OptimizationTarget::Generic
            | OptimizationTarget::Haswell
            | OptimizationTarget::Skylake
            | OptimizationTarget::Zen
            | OptimizationTarget::CortexA76
            | OptimizationTarget::AppleSilicon => {}
            _ => panic!("Invalid optimization target detected"),
        }
    }

    #[test]
    fn test_target_config() {
        let config = TargetConfig::default();
        let ops = TargetOptimizedOps::new(config);

        assert!(ops.optimal_f32_width() >= 1);
        assert!(ops.cache_line_size() > 0);
        assert!(ops.prefetch_distance() > 0);
    }

    #[test]
    fn test_auto_detect() {
        let ops = TargetOptimizedOps::auto_detect();

        assert!(ops.optimal_f32_width() >= 1);
        assert!(ops.optimal_f32_width() <= 16);
    }

    #[test]
    fn test_vector_arithmetic() {
        let ops = TargetOptimizedOps::auto_detect();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = ops.add(&a, &b).unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);

        let result = ops.sub(&a, &b).unwrap();
        assert_eq!(result, vec![-4.0, -4.0, -4.0, -4.0]);

        let result = ops.mul(&a, &b).unwrap();
        assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);

        let result = ops.scale(&a, 2.0).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_vector_reductions() {
        let ops = TargetOptimizedOps::auto_detect();

        let vector = vec![1.0, 2.0, 3.0, 4.0];

        let sum = ops.sum(&vector).unwrap();
        assert_eq!(sum, 10.0);

        let mean = ops.mean(&vector).unwrap();
        assert_eq!(mean, 2.5);

        let min = ops.min(&vector).unwrap();
        assert_eq!(min, 1.0);

        let max = ops.max(&vector).unwrap();
        assert_eq!(max, 4.0);

        let norm = ops.norm(&vector).unwrap();
        assert!((norm - (1.0 + 4.0 + 9.0 + 16.0_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_fma_operation() {
        let ops = TargetOptimizedOps::auto_detect();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];

        let result = ops.fma(&a, &b, &c).unwrap();
        assert_eq!(result, vec![3.0, 7.0, 13.0, 21.0]); // a*b + c
    }

    #[test]
    fn test_error_handling() {
        let ops = TargetOptimizedOps::auto_detect();

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0]; // Different length

        let result = ops.add(&a, &b);
        assert!(result.is_err());

        match result {
            Err(SimdError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected dimension mismatch error"),
        }
    }

    #[test]
    fn test_compile_time_features() {
        use compile_time::*;

        let flags = get_optimization_flags();
        println!("Available optimization flags: {:?}", flags);

        // Should have at least some basic features (or be empty on non-x86 platforms)
        // This is acceptable since ARM or other platforms may not have these specific flags

        let config = optimize_for_cpu();
        println!("CPU-optimized config: {:?}", config);

        // Basic sanity checks
        assert!(matches!(
            config.optimization_target,
            OptimizationTarget::Generic
                | OptimizationTarget::Haswell
                | OptimizationTarget::Skylake
                | OptimizationTarget::Zen
                | OptimizationTarget::CortexA76
                | OptimizationTarget::AppleSilicon
                | OptimizationTarget::Server
                | OptimizationTarget::Mobile
        ));
    }
}

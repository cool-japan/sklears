//! SIMD Type Definitions and Common Structures
//!
//! This module provides the core type definitions, enumerations, and data structures
//! used throughout the SIMD vector operations framework. It includes architecture
//! detection, performance metrics, error types, and configuration structures.
//!
//! # Key Types
//!
//! - **SimdInstructionSet**: Enumeration of supported SIMD instruction sets
//! - **VectorOperationResult**: Result types with performance information
//! - **PerformanceMetrics**: Comprehensive performance tracking structures
//! - **Error Types**: Specialized error handling for vector operations
//! - **Configuration**: Runtime configuration and optimization settings

#[cfg(feature = "no-std")]
use alloc::{string::String, vec::Vec};
#[cfg(not(feature = "no-std"))]
use std::{string::String, vec::Vec};

/// Enumeration of supported SIMD instruction sets
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdInstructionSet {
    /// No SIMD support - scalar operations only
    None,
    /// Scalar operations (fallback)
    Scalar,
    /// SSE2 (Streaming SIMD Extensions 2) - 128-bit vectors
    Sse2,
    /// AVX2 (Advanced Vector Extensions 2) - 256-bit vectors
    Avx2,
    /// AVX-512 - 512-bit vectors
    Avx512,
    /// ARM NEON - 128-bit vectors
    Neon,
    /// x86 FMA (Fused Multiply-Add)
    Fma,
    /// RISC-V Vector Extensions
    RiscvVector,
}

impl SimdInstructionSet {
    /// Get the vector width (number of f32 elements) for this instruction set
    pub fn vector_width(self) -> usize {
        match self {
            SimdInstructionSet::None | SimdInstructionSet::Scalar => 1,
            SimdInstructionSet::Sse2 | SimdInstructionSet::Neon => 4,
            SimdInstructionSet::Avx2 | SimdInstructionSet::Fma => 8,
            SimdInstructionSet::Avx512 => 16,
            SimdInstructionSet::RiscvVector => 4, // Varies, but 4 is common
        }
    }

    /// Get theoretical performance multiplier vs scalar operations
    pub fn performance_multiplier(self) -> f32 {
        match self {
            SimdInstructionSet::None | SimdInstructionSet::Scalar => 1.0,
            SimdInstructionSet::Sse2 => 3.5,
            SimdInstructionSet::Avx2 => 7.0,
            SimdInstructionSet::Avx512 => 14.0,
            SimdInstructionSet::Neon => 3.5,
            SimdInstructionSet::Fma => 8.0,
            SimdInstructionSet::RiscvVector => 3.5,
        }
    }

    /// Get human-readable name for this instruction set
    pub fn name(self) -> &'static str {
        match self {
            SimdInstructionSet::None => "None",
            SimdInstructionSet::Scalar => "Scalar",
            SimdInstructionSet::Sse2 => "SSE2",
            SimdInstructionSet::Avx2 => "AVX2",
            SimdInstructionSet::Avx512 => "AVX-512",
            SimdInstructionSet::Neon => "NEON",
            SimdInstructionSet::Fma => "FMA",
            SimdInstructionSet::RiscvVector => "RISC-V Vector",
        }
    }

    /// Check if this instruction set supports fast transcendental functions
    pub fn supports_fast_transcendentals(self) -> bool {
        matches!(self, SimdInstructionSet::Avx2 | SimdInstructionSet::Avx512 | SimdInstructionSet::Fma)
    }

    /// Get required memory alignment for this instruction set
    pub fn required_alignment(self) -> usize {
        match self {
            SimdInstructionSet::None | SimdInstructionSet::Scalar => 4,
            SimdInstructionSet::Sse2 | SimdInstructionSet::Neon => 16,
            SimdInstructionSet::Avx2 | SimdInstructionSet::Fma => 32,
            SimdInstructionSet::Avx512 => 64,
            SimdInstructionSet::RiscvVector => 16,
        }
    }
}

impl Default for SimdInstructionSet {
    fn default() -> Self {
        detect_best_simd()
    }
}

/// Detect the best available SIMD instruction set
pub fn detect_best_simd() -> SimdInstructionSet {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdInstructionSet::Avx512;
        } else if is_x86_feature_detected!("fma") {
            return SimdInstructionSet::Fma;
        } else if is_x86_feature_detected!("avx2") {
            return SimdInstructionSet::Avx2;
        } else if is_x86_feature_detected!("sse2") {
            return SimdInstructionSet::Sse2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return SimdInstructionSet::Neon;
    }

    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    {
        // RISC-V vector extension detection would go here
        // For now, fall back to scalar
        return SimdInstructionSet::Scalar;
    }

    SimdInstructionSet::Scalar
}

/// Architecture-specific SIMD level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// Level 0: Scalar operations only
    L0 = 0,
    /// Level 1: Basic SIMD (SSE2, NEON)
    L1 = 1,
    /// Level 2: Advanced SIMD (AVX2)
    L2 = 2,
    /// Level 3: High-performance SIMD (AVX-512)
    L3 = 3,
}

impl From<SimdInstructionSet> for SimdLevel {
    fn from(simd: SimdInstructionSet) -> Self {
        match simd {
            SimdInstructionSet::None | SimdInstructionSet::Scalar => SimdLevel::L0,
            SimdInstructionSet::Sse2 | SimdInstructionSet::Neon | SimdInstructionSet::RiscvVector => SimdLevel::L1,
            SimdInstructionSet::Avx2 | SimdInstructionSet::Fma => SimdLevel::L2,
            SimdInstructionSet::Avx512 => SimdLevel::L3,
        }
    }
}

/// Get optimal SIMD level for current architecture
pub fn get_optimal_simd_level() -> SimdLevel {
    detect_best_simd().into()
}

/// Vector operation performance metrics
#[derive(Debug, Clone)]
pub struct VectorPerformanceMetrics {
    /// Total execution time in nanoseconds
    pub execution_time_ns: u64,
    /// Number of operations performed
    pub operations_count: usize,
    /// Throughput in operations per second
    pub throughput_ops_per_sec: f64,
    /// Memory bandwidth utilization in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// SIMD instruction set used
    pub simd_used: SimdInstructionSet,
    /// Vector size processed
    pub vector_size: usize,
    /// Number of SIMD lanes utilized
    pub simd_lanes_used: usize,
    /// Whether fallback to scalar was needed
    pub fallback_used: bool,
}

impl Default for VectorPerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time_ns: 0,
            operations_count: 0,
            throughput_ops_per_sec: 0.0,
            memory_bandwidth_gbps: 0.0,
            cache_hit_rate: 0.0,
            simd_used: SimdInstructionSet::Scalar,
            vector_size: 0,
            simd_lanes_used: 1,
            fallback_used: true,
        }
    }
}

impl VectorPerformanceMetrics {
    /// Create new performance metrics
    pub fn new(
        execution_time_ns: u64,
        operations_count: usize,
        simd_used: SimdInstructionSet,
        vector_size: usize,
    ) -> Self {
        let throughput_ops_per_sec = if execution_time_ns > 0 {
            (operations_count as f64 * 1_000_000_000.0) / execution_time_ns as f64
        } else {
            0.0
        };

        let simd_lanes_used = simd_used.vector_width();
        let fallback_used = simd_used == SimdInstructionSet::Scalar;

        Self {
            execution_time_ns,
            operations_count,
            throughput_ops_per_sec,
            memory_bandwidth_gbps: 0.0, // Would need additional measurement
            cache_hit_rate: 0.0,         // Would need hardware counters
            simd_used,
            vector_size,
            simd_lanes_used,
            fallback_used,
        }
    }

    /// Calculate performance efficiency vs theoretical peak
    pub fn efficiency_percent(&self) -> f64 {
        let theoretical_peak = self.simd_used.performance_multiplier() as f64;
        let actual_speedup = self.throughput_ops_per_sec / (self.throughput_ops_per_sec / theoretical_peak);
        (actual_speedup / theoretical_peak) * 100.0
    }
}

/// Vector operation result with optional performance tracking
#[derive(Debug, Clone)]
pub enum VectorOperationResult<T> {
    /// Successful operation with result
    Success {
        /// The operation result
        result: T,
        /// Performance metrics (if enabled)
        metrics: Option<VectorPerformanceMetrics>,
    },
    /// Operation failed with error
    Error {
        /// Error information
        error: VectorError,
        /// Partial performance data (if available)
        partial_metrics: Option<VectorPerformanceMetrics>,
    },
}

impl<T> VectorOperationResult<T> {
    /// Create successful result
    pub fn success(result: T) -> Self {
        Self::Success {
            result,
            metrics: None,
        }
    }

    /// Create successful result with metrics
    pub fn success_with_metrics(result: T, metrics: VectorPerformanceMetrics) -> Self {
        Self::Success {
            result,
            metrics: Some(metrics),
        }
    }

    /// Create error result
    pub fn error(error: VectorError) -> Self {
        Self::Error {
            error,
            partial_metrics: None,
        }
    }

    /// Extract result or return error
    pub fn into_result(self) -> Result<T, VectorError> {
        match self {
            VectorOperationResult::Success { result, .. } => Ok(result),
            VectorOperationResult::Error { error, .. } => Err(error),
        }
    }

    /// Get performance metrics if available
    pub fn metrics(&self) -> Option<&VectorPerformanceMetrics> {
        match self {
            VectorOperationResult::Success { metrics, .. } => metrics.as_ref(),
            VectorOperationResult::Error { partial_metrics, .. } => partial_metrics.as_ref(),
        }
    }

    /// Check if operation was successful
    pub fn is_success(&self) -> bool {
        matches!(self, VectorOperationResult::Success { .. })
    }

    /// Check if operation failed
    pub fn is_error(&self) -> bool {
        matches!(self, VectorOperationResult::Error { .. })
    }
}

/// Vector operation error types
#[derive(Debug, Clone, PartialEq)]
pub enum VectorError {
    /// Vector dimensions don't match
    DimensionMismatch {
        expected: usize,
        actual: usize,
        operation: String,
    },
    /// Empty vector provided where non-empty expected
    EmptyVector {
        operation: String,
    },
    /// Invalid vector size for operation
    InvalidSize {
        size: usize,
        min_required: usize,
        max_allowed: Option<usize>,
        operation: String,
    },
    /// Memory alignment error
    AlignmentError {
        address: usize,
        required_alignment: usize,
        simd_set: SimdInstructionSet,
    },
    /// SIMD instruction set not supported
    UnsupportedSimd {
        requested: SimdInstructionSet,
        available: SimdInstructionSet,
    },
    /// Numerical computation error
    NumericalError {
        message: String,
        value: Option<f32>,
    },
    /// Operation not supported on current architecture
    UnsupportedOperation {
        operation: String,
        architecture: String,
    },
    /// Resource exhaustion (memory, compute, etc.)
    ResourceExhausted {
        resource: String,
        requested: usize,
        available: usize,
    },
    /// Generic computation error
    ComputationError {
        message: String,
        cause: Option<String>,
    },
}

impl VectorError {
    /// Get error category
    pub fn category(&self) -> VectorErrorCategory {
        match self {
            VectorError::DimensionMismatch { .. } | VectorError::InvalidSize { .. } => {
                VectorErrorCategory::Input
            }
            VectorError::EmptyVector { .. } => VectorErrorCategory::Input,
            VectorError::AlignmentError { .. } => VectorErrorCategory::Memory,
            VectorError::UnsupportedSimd { .. } | VectorError::UnsupportedOperation { .. } => {
                VectorErrorCategory::Platform
            }
            VectorError::NumericalError { .. } | VectorError::ComputationError { .. } => {
                VectorErrorCategory::Computation
            }
            VectorError::ResourceExhausted { .. } => VectorErrorCategory::Resource,
        }
    }

    /// Get error severity
    pub fn severity(&self) -> VectorErrorSeverity {
        match self {
            VectorError::DimensionMismatch { .. } | VectorError::EmptyVector { .. } => {
                VectorErrorSeverity::High
            }
            VectorError::InvalidSize { .. } => VectorErrorSeverity::High,
            VectorError::AlignmentError { .. } => VectorErrorSeverity::Medium,
            VectorError::UnsupportedSimd { .. } => VectorErrorSeverity::Low,
            VectorError::UnsupportedOperation { .. } => VectorErrorSeverity::High,
            VectorError::NumericalError { .. } => VectorErrorSeverity::Medium,
            VectorError::ResourceExhausted { .. } => VectorErrorSeverity::High,
            VectorError::ComputationError { .. } => VectorErrorSeverity::Medium,
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            VectorError::UnsupportedSimd { .. } => true,  // Can fallback to scalar
            VectorError::AlignmentError { .. } => true,    // Can use unaligned operations
            VectorError::ResourceExhausted { .. } => true, // Might recover with smaller batches
            _ => false,
        }
    }
}

/// Vector error categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorErrorCategory {
    Input,
    Memory,
    Platform,
    Computation,
    Resource,
}

/// Vector error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VectorErrorSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Vector operation configuration
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Preferred SIMD instruction set (None = auto-detect)
    pub preferred_simd: Option<SimdInstructionSet>,
    /// Force specific SIMD level (overrides auto-detection)
    pub force_simd_level: Option<SimdLevel>,
    /// Enable performance metrics collection
    pub enable_metrics: bool,
    /// Enable detailed profiling
    pub enable_profiling: bool,
    /// Memory alignment preference
    pub alignment_bytes: usize,
    /// Minimum vector size for SIMD optimization
    pub simd_threshold: usize,
    /// Enable fallback to scalar operations
    pub allow_scalar_fallback: bool,
    /// Enable unsafe optimizations
    pub enable_unsafe_optimizations: bool,
    /// Maximum memory usage for vector operations (bytes)
    pub max_memory_usage: Option<usize>,
    /// Prefetch distance for memory operations
    pub prefetch_distance: usize,
}

impl Default for VectorConfig {
    fn default() -> Self {
        let best_simd = detect_best_simd();
        Self {
            preferred_simd: None,
            force_simd_level: None,
            enable_metrics: false,
            enable_profiling: false,
            alignment_bytes: best_simd.required_alignment(),
            simd_threshold: best_simd.vector_width(),
            allow_scalar_fallback: true,
            enable_unsafe_optimizations: false,
            max_memory_usage: None,
            prefetch_distance: 64, // Common cache line size
        }
    }
}

impl VectorConfig {
    /// Create performance-optimized configuration
    pub fn performance_optimized() -> Self {
        Self {
            enable_unsafe_optimizations: true,
            simd_threshold: 2, // Use SIMD even for small vectors
            prefetch_distance: 128,
            ..Default::default()
        }
    }

    /// Create safety-focused configuration
    pub fn safety_focused() -> Self {
        Self {
            enable_unsafe_optimizations: false,
            allow_scalar_fallback: true,
            simd_threshold: 16, // Higher threshold for safety
            enable_profiling: true, // For debugging
            ..Default::default()
        }
    }

    /// Create memory-constrained configuration
    pub fn memory_constrained(max_memory_mb: usize) -> Self {
        Self {
            max_memory_usage: Some(max_memory_mb * 1024 * 1024),
            simd_threshold: 8,
            prefetch_distance: 32,
            ..Default::default()
        }
    }

    /// Get effective SIMD instruction set for this configuration
    pub fn effective_simd(&self) -> SimdInstructionSet {
        if let Some(preferred) = self.preferred_simd {
            preferred
        } else if let Some(level) = self.force_simd_level {
            match level {
                SimdLevel::L0 => SimdInstructionSet::Scalar,
                SimdLevel::L1 => {
                    #[cfg(target_arch = "aarch64")]
                    return SimdInstructionSet::Neon;
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    return SimdInstructionSet::Sse2;
                    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
                    return SimdInstructionSet::Scalar;
                }
                SimdLevel::L2 => SimdInstructionSet::Avx2,
                SimdLevel::L3 => SimdInstructionSet::Avx512,
            }
        } else {
            detect_best_simd()
        }
    }

    /// Check if vector size meets SIMD threshold
    pub fn should_use_simd(&self, vector_size: usize) -> bool {
        vector_size >= self.simd_threshold && self.effective_simd() != SimdInstructionSet::Scalar
    }
}

/// Architecture capability detection
pub struct ArchitectureInfo {
    /// Available SIMD instruction sets
    pub available_simd: Vec<SimdInstructionSet>,
    /// Best available SIMD set
    pub best_simd: SimdInstructionSet,
    /// CPU vendor (if available)
    pub cpu_vendor: Option<String>,
    /// CPU model (if available)
    pub cpu_model: Option<String>,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Cache line size
    pub cache_line_size: usize,
    /// L1 cache size (if detectable)
    pub l1_cache_size: Option<usize>,
    /// L2 cache size (if detectable)
    pub l2_cache_size: Option<usize>,
    /// L3 cache size (if detectable)
    pub l3_cache_size: Option<usize>,
}

impl ArchitectureInfo {
    /// Detect current architecture capabilities
    pub fn detect() -> Self {
        let mut available_simd = vec![SimdInstructionSet::Scalar];
        let mut best_simd = SimdInstructionSet::Scalar;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse2") {
                available_simd.push(SimdInstructionSet::Sse2);
                best_simd = SimdInstructionSet::Sse2;
            }
            if is_x86_feature_detected!("fma") {
                available_simd.push(SimdInstructionSet::Fma);
                best_simd = SimdInstructionSet::Fma;
            }
            if is_x86_feature_detected!("avx2") {
                available_simd.push(SimdInstructionSet::Avx2);
                best_simd = SimdInstructionSet::Avx2;
            }
            if is_x86_feature_detected!("avx512f") {
                available_simd.push(SimdInstructionSet::Avx512);
                best_simd = SimdInstructionSet::Avx512;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            available_simd.push(SimdInstructionSet::Neon);
            best_simd = SimdInstructionSet::Neon;
        }

        Self {
            available_simd,
            best_simd,
            cpu_vendor: None, // Would need platform-specific detection
            cpu_model: None,  // Would need platform-specific detection
            cpu_cores: num_cpus::get_physical(),
            cache_line_size: 64, // Common default
            l1_cache_size: None,
            l2_cache_size: None,
            l3_cache_size: None,
        }
    }

    /// Check if specific SIMD instruction set is available
    pub fn supports_simd(&self, simd: SimdInstructionSet) -> bool {
        self.available_simd.contains(&simd)
    }

    /// Get recommended vector configuration for this architecture
    pub fn recommended_config(&self) -> VectorConfig {
        VectorConfig {
            preferred_simd: Some(self.best_simd),
            alignment_bytes: self.best_simd.required_alignment(),
            simd_threshold: self.best_simd.vector_width(),
            prefetch_distance: self.cache_line_size,
            ..Default::default()
        }
    }
}

/// SIMD capability testing utilities
pub fn supports_sse2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("sse2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

pub fn supports_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

pub fn supports_avx512() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

pub fn supports_fma() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

pub fn supports_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true // NEON is mandatory on AArch64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_instruction_set_properties() {
        assert_eq!(SimdInstructionSet::Sse2.vector_width(), 4);
        assert_eq!(SimdInstructionSet::Avx2.vector_width(), 8);
        assert_eq!(SimdInstructionSet::Avx512.vector_width(), 16);

        assert_eq!(SimdInstructionSet::Sse2.required_alignment(), 16);
        assert_eq!(SimdInstructionSet::Avx2.required_alignment(), 32);
        assert_eq!(SimdInstructionSet::Avx512.required_alignment(), 64);

        assert!(SimdInstructionSet::Avx2.performance_multiplier() > 1.0);
    }

    #[test]
    fn test_simd_level_conversion() {
        assert_eq!(SimdLevel::from(SimdInstructionSet::Scalar), SimdLevel::L0);
        assert_eq!(SimdLevel::from(SimdInstructionSet::Sse2), SimdLevel::L1);
        assert_eq!(SimdLevel::from(SimdInstructionSet::Avx2), SimdLevel::L2);
        assert_eq!(SimdLevel::from(SimdInstructionSet::Avx512), SimdLevel::L3);
    }

    #[test]
    fn test_vector_config_defaults() {
        let config = VectorConfig::default();
        assert!(config.allow_scalar_fallback);
        assert!(!config.enable_unsafe_optimizations);
        assert!(!config.enable_metrics);
    }

    #[test]
    fn test_vector_config_specialized() {
        let perf_config = VectorConfig::performance_optimized();
        assert!(perf_config.enable_unsafe_optimizations);
        assert_eq!(perf_config.simd_threshold, 2);

        let safe_config = VectorConfig::safety_focused();
        assert!(!safe_config.enable_unsafe_optimizations);
        assert!(safe_config.allow_scalar_fallback);
    }

    #[test]
    fn test_vector_error_properties() {
        let error = VectorError::DimensionMismatch {
            expected: 4,
            actual: 3,
            operation: "test".to_string(),
        };

        assert_eq!(error.category(), VectorErrorCategory::Input);
        assert_eq!(error.severity(), VectorErrorSeverity::High);
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_architecture_detection() {
        let arch = ArchitectureInfo::detect();
        assert!(!arch.available_simd.is_empty());
        assert!(arch.available_simd.contains(&SimdInstructionSet::Scalar));
        assert!(arch.cpu_cores > 0);
        assert!(arch.cache_line_size > 0);
    }

    #[test]
    fn test_simd_capability_functions() {
        // These functions should not panic
        let _ = supports_sse2();
        let _ = supports_avx2();
        let _ = supports_avx512();
        let _ = supports_fma();
        let _ = supports_neon();
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = VectorPerformanceMetrics::new(
            1000000, // 1ms
            1000,    // 1000 operations
            SimdInstructionSet::Avx2,
            1000,    // vector size
        );

        assert_eq!(metrics.operations_count, 1000);
        assert_eq!(metrics.simd_used, SimdInstructionSet::Avx2);
        assert!(!metrics.fallback_used);
        assert!(metrics.throughput_ops_per_sec > 0.0);
    }

    #[test]
    fn test_vector_operation_result() {
        let success_result = VectorOperationResult::success(42);
        assert!(success_result.is_success());
        assert!(!success_result.is_error());
        assert_eq!(success_result.into_result().unwrap(), 42);

        let error_result: VectorOperationResult<i32> = VectorOperationResult::error(
            VectorError::EmptyVector {
                operation: "test".to_string(),
            }
        );
        assert!(!error_result.is_success());
        assert!(error_result.is_error());
    }
}
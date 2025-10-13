//! Performance Optimizations for Dummy Estimators
//!
//! This module provides high-performance implementations of dummy estimator operations
//! using SIMD vectorization, parallel computation, and cache-friendly algorithms.
//!
//! The module is organized into specialized submodules for better maintainability
//! and compliance with the 2000-line refactoring policy.
//!
//! Features:
//! - SIMD-optimized statistical operations
//! - Parallel prediction for large datasets
//! - Cache-friendly data layouts
//! - Memory-efficient algorithms
//! - Vectorized sampling operations
//! - Profile-guided optimizations
//! - Benchmarking utilities

pub mod benchmarks;
pub mod cache_friendly;
pub mod constant_time;
pub mod memory_efficient;
pub mod parallel;
pub mod profile_guided;
pub mod simd_dummy;
pub mod simd_sampling;
pub mod simd_stats;
pub mod unsafe_optimizations;
pub mod vectorized;

// Re-export main functionality
pub use benchmarks::*;
pub use cache_friendly::*;
pub use constant_time::*;
pub use memory_efficient::*;
pub use parallel::*;
pub use profile_guided::*;
pub use simd_dummy::*;
pub use simd_sampling::*;
pub use simd_stats::*;
pub use unsafe_optimizations::*;
pub use vectorized::*;

//! Test modules for sklears-manifold
//!
//! This module organizes tests into logical groups to maintain the refactoring
//! policy of keeping individual files under 2000 lines.

/// Basic algorithm tests for core manifold learning methods
pub mod basic_algorithms;

/// Advanced algorithm tests for specialized methods
pub mod advanced_algorithms;

/// Quality metrics and neighborhood preservation tests
pub mod quality_metrics;

/// Property-based tests using the proptest framework
pub mod property_tests;

/// Tests for high-dimensional data methods
pub mod high_dimensional;

// Re-export all test modules for convenient access
pub use advanced_algorithms::*;
pub use basic_algorithms::*;
pub use high_dimensional::*;
pub use property_tests::*;
pub use quality_metrics::*;

//! Array utilities module
//!
//! This module provides comprehensive array manipulation utilities split into
//! logical submodules for better organization and maintainability.

pub mod core;
pub mod indexing;
pub mod inplace;
pub mod memory;
pub mod shape_ops;
pub mod simd_ops;
pub mod sparse;
pub mod stats;

// Re-export commonly used functions
pub use core::*;
pub use indexing::*;
pub use inplace::*;
pub use memory::*;
pub use shape_ops::*;
pub use simd_ops::*;
pub use sparse::*;
pub use stats::*;

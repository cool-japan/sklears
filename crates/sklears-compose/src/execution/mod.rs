//! Execution engine module
//!
//! This module provides a composable execution engine framework for flexible
//! pipeline runtime configurations with pluggable execution strategies.

pub mod config;
pub mod engine;
pub mod metrics;
pub mod resources;
pub mod scheduling;
pub mod simd_utils;
pub mod strategies;
pub mod tasks;

// Re-export commonly used items
pub use config::*;
pub use metrics::*;
pub use resources::*;
pub use scheduling::*;
pub use simd_utils::*;
pub use strategies::*;
pub use tasks::*;

// Re-export specific items from engine to avoid conflicts
pub use engine::{ComposableExecutionEngine, ExecutionContext, ExecutionPhase, ResourceUsage};

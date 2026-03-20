//! Pipeline visualization module
//!
//! This module provides comprehensive visualization capabilities for machine learning
//! pipelines, including graph rendering, interactive visualizations, and export functionality.

pub mod core;
// TODO: Add these modules when fully refactored:
// pub mod graph;
// pub mod metrics;
// pub mod rendering;
// pub mod interactive;

// Re-export core visualization components
pub use core::*;

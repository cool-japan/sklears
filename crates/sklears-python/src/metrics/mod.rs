//! Python bindings for evaluation metrics
//!
//! This module provides Python bindings for sklears metrics,
//! offering scikit-learn compatible evaluation functions.
//! Metrics are organized by type (regression, classification)
//! for better code organization.

// Common functionality shared across metrics
mod common;
pub use common::*;

// Metric implementations by type
pub mod classification;
pub mod regression;

// Re-export all metric functions for PyO3
pub use classification::*;
pub use regression::*;

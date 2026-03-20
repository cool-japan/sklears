//! Utility functions and structures for multiclass classification
//!
//! This module provides utilities including:
//! - Evaluation metrics and confusion matrix analysis
//! - Prediction caching for improved performance
//! - Future utilities: compressed model storage, batch prediction optimization, streaming/incremental learning support

pub mod batch_optimization;
pub mod caching;
pub mod evaluation;

pub use batch_optimization::*;
pub use caching::*;
pub use evaluation::*;

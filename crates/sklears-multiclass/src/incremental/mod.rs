//! Incremental Learning Framework for Multiclass Classification
//!
//! This module provides online and incremental learning capabilities for multiclass classifiers.
//! It includes drift detection, memory management, adaptive learning strategies, and warm start.

pub mod drift_detection;
pub mod memory_management;
pub mod online_learning;
pub mod warm_start;

pub use drift_detection::*;
pub use memory_management::*;
pub use online_learning::*;
pub use warm_start::*;

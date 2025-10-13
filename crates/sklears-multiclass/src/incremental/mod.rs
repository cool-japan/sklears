//! Incremental Learning Framework for Multiclass Classification
//!
//! This module provides online and incremental learning capabilities for multiclass classifiers.
//! It includes drift detection, memory management, and adaptive learning strategies.

pub mod drift_detection;
pub mod memory_management;
pub mod online_learning;

pub use drift_detection::*;
pub use memory_management::*;
pub use online_learning::*;

//! Ensemble methods for multiclass classification
//!
//! This module provides ensemble learning strategies including:
//! Bagging, Dynamic Ensemble Selection, Gradient Boosting, Rotation Forest and other ensemble methods.

pub mod bagging;
pub mod dynamic_ensemble;
pub mod gradient_boosting;
pub mod rotation_forest;

pub use bagging::*;
pub use dynamic_ensemble::*;
pub use gradient_boosting::*;
pub use rotation_forest::*;

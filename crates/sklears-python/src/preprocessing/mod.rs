//! Python bindings for preprocessing transformers
//!
//! This module provides Python bindings for sklears preprocessing,
//! offering scikit-learn compatible data transformation utilities.
//! Each transformer type is implemented in its own submodule for
//! better code organization.

// Common functionality shared across preprocessing transformers
mod common;
pub use common::*;

// Individual transformer implementations
mod label_encoder;
mod minmax_scaler;
mod standard_scaler;

// Re-export all transformer classes for PyO3
pub use label_encoder::PyLabelEncoder;
pub use minmax_scaler::PyMinMaxScaler;
pub use standard_scaler::PyStandardScaler;

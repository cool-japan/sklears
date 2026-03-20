//! Temporal feature engineering utilities
//!
//! This module provides comprehensive temporal feature engineering capabilities including:
//! - Date component extraction (year, month, day, hour, minute, second)
//! - Cyclical feature encoding (sin/cos transformations for periodic features)
//! - Lag and rolling window features
//! - Holiday and business day indicators
//! - Time-based aggregations and trends
//! - Seasonal decomposition and analysis
//! - Trend detection and quantification
//! - Change point detection
//! - Fourier-based frequency domain features
//!
//! All modules have been refactored for better maintainability and comply with
//! the 2000-line refactoring policy.

pub mod changepoint;
pub mod datetime_utils;
pub mod fourier;
pub mod interpolation;
pub mod lag_features;
pub mod seasonal;
pub mod stationarity;
pub mod temporal_features;
pub mod trend;

// Re-export main types and functions
pub use changepoint::*;
pub use datetime_utils::*;
pub use fourier::*;
pub use interpolation::*;
pub use lag_features::*;
pub use seasonal::*;
pub use stationarity::*;
pub use temporal_features::*;
pub use trend::*;

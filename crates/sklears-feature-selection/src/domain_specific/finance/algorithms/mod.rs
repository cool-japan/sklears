//! Finance algorithm modules for feature selection
//!
//! This module contains domain-specific financial algorithms organized by category.

// Algorithm modules
pub mod correlation_stats;
pub mod economic_indicators;
pub mod factor_models;
pub mod market_microstructure;
pub mod performance_metrics;
pub mod portfolio_optimization;
pub mod regime_detection;
pub mod risk_metrics;
pub mod technical_indicators;
pub mod utilities;

// Re-export key functions for use within the finance module
pub use correlation_stats::*;
pub use economic_indicators::*;
pub use factor_models::*;
pub use market_microstructure::*;
pub use performance_metrics::*;
pub use portfolio_optimization::*;
pub use regime_detection::*;
pub use risk_metrics::*;
pub use technical_indicators::*;
pub use utilities::*;

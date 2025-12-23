//! Finance algorithm modules for feature selection
//!
//! This module contains domain-specific financial algorithms organized by category.

// Algorithm modules
pub(crate) mod correlation_stats;
pub(crate) mod economic_indicators;
pub(crate) mod factor_models;
pub(crate) mod market_microstructure;
pub(crate) mod performance_metrics;
pub(crate) mod portfolio_optimization;
pub(crate) mod regime_detection;
pub(crate) mod risk_metrics;
pub(crate) mod technical_indicators;
pub(crate) mod utilities;

// Re-export key functions for use within the finance module
pub(crate) use correlation_stats::*;
pub(crate) use economic_indicators::*;
pub(crate) use factor_models::*;
pub(crate) use market_microstructure::*;
pub(crate) use performance_metrics::*;
pub(crate) use portfolio_optimization::*;
pub(crate) use regime_detection::*;
pub(crate) use risk_metrics::*;
pub(crate) use technical_indicators::*;
pub(crate) use utilities::*;

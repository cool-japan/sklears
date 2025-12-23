//! Finance domain-specific feature selection
//!
//! This module provides specialized feature selection methods for financial data,
//! including technical indicators, risk metrics, portfolio optimization, and more.
//!
//! # Module Organization
//!
//! - `types`: Core types, enums, and structs for finance feature selection
//! - `traits`: Trait implementations for FinanceFeatureSelector
//! - `algorithms`: Domain-specific financial algorithms organized by category
//!   - `technical_indicators`: RSI, MACD, Bollinger Bands, ATR, OBV, VWAP
//!   - `risk_metrics`: VaR, CVaR, drawdown, tail risk, extreme value measures
//!   - `performance_metrics`: Sharpe ratio, Information ratio, volatility
//!   - `factor_models`: Fama-French factors, factor loadings, R-squared
//!   - `portfolio_optimization`: Mean-variance, risk parity, Sharpe optimization
//!   - `market_microstructure`: Order flow, bid-ask, market impact, transaction costs
//!   - `regime_detection`: HMM, regime transitions, volatility/momentum regimes
//!   - `economic_indicators`: GDP, inflation, interest rates, unemployment
//!   - `correlation_stats`: Correlation and covariance matrices
//!   - `utilities`: Helper functions for feature selection
//!
//! # Refactoring
//!
//! This module was refactored from a monolithic 3020-line file into a well-organized
//! structure with 13 modules, each under 260 lines, following the 2000-line policy.
//!
//! Original: `finance.rs` (3020 lines)
//! Refactored: 13 modules (max 259 lines each)
//!
//! 🔧 Refactored with [SplitRS](https://github.com/cool-japan/splitrs)

// Core modules
pub mod algorithms;
pub mod traits;
pub mod types;

// Re-export public types and functions
pub use traits::*;
pub use types::*;

// Re-export algorithm functions for convenience (keeping them private to crate)

//! Robust cross-decomposition methods
//!
//! This module provides robust versions of cross-decomposition algorithms that are resistant
//! to outliers and contaminated data, including Robust CCA, M-estimator based PLS,
//! and Huber-type robust decomposition methods.

pub mod common;
pub mod robust_cca;
pub mod robust_pls;

// Re-export common types
pub use common::{MEstimatorType, Trained, Untrained};

// Re-export from modules
pub use robust_cca::RobustCCA;
pub use robust_pls::RobustPLS;

// Note: Additional analysis tools like BreakdownPointAnalysis and InfluenceDiagnostics
// can be added as separate modules when their implementations are ready.

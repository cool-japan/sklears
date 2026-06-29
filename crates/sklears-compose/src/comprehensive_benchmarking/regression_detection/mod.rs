//! Regression Detection System
//!
//! This module provides comprehensive regression detection capabilities including
//! performance regression detection, alerting systems, severity assessment, and remediation guidance.

pub mod adaptivethresholds_traits;
pub mod alertsuppression_traits;
pub mod baselinecomparisons_traits;
pub mod businessimpactassessment_traits;
pub mod effectsizeanalysis_traits;
pub mod patternrecognition_traits;
pub mod regressionalertsystem_traits;
pub mod regressioncache_traits;
pub mod regressiondetector_traits;
pub mod regressiondetectorconfig_traits;
pub mod regressionmetadata_traits;
pub mod severityassessment_traits;
pub mod significancetesting_traits;
pub mod smartsuppression_traits;
pub mod systemmetrics_traits;
pub mod thresholdmanagement_traits;
pub mod types;
pub mod types_19;
pub mod types_20;
pub mod userimpactassessment_traits;

// Re-export all types
pub use types::*;
pub use types_19::*;
pub use types_20::*;

#[cfg(test)]
mod tests;

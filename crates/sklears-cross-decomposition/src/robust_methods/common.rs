//! Common types and utilities for robust methods

/// Types of M-estimators for robust estimation
#[derive(Debug, Clone, Default)]
pub enum MEstimatorType {
    /// Huber M-estimator (bounded influence)
    #[default]
    Huber,
    /// Bisquare (Tukey) M-estimator (redescending)
    Bisquare,
    /// Hampel M-estimator (three-part redescending)
    Hampel,
    /// Andrews sine M-estimator
    Andrews,
}

/// Marker type for untrained state
#[derive(Debug, Clone)]
pub struct Untrained;

/// Marker type for trained state
#[derive(Debug, Clone)]
pub struct Trained;

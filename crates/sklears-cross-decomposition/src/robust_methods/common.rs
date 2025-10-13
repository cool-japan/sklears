//! Common types and utilities for robust methods

use std::marker::PhantomData;

/// Types of M-estimators for robust estimation
#[derive(Debug, Clone)]
pub enum MEstimatorType {
    /// Huber M-estimator (bounded influence)
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

impl Default for MEstimatorType {
    fn default() -> Self {
        MEstimatorType::Huber
    }
}

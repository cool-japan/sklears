//! # ResourceManagementError - Trait Implementations
//!
//! This module contains trait implementations for `ResourceManagementError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl fmt::Display for ResourceManagementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceManagementError::AllocationFailed(msg) => {
                write!(f, "Allocation failed: {}", msg)
            }
            ResourceManagementError::InsufficientResources => {
                write!(f, "Insufficient resources available")
            }
            ResourceManagementError::NoSuitableNodes => {
                write!(f, "No suitable nodes found for allocation")
            }
            ResourceManagementError::InvalidRequest(msg) => {
                write!(f, "Invalid resource request: {}", msg)
            }
            ResourceManagementError::OptimizationFailed(msg) => {
                write!(f, "Optimization failed: {}", msg)
            }
            ResourceManagementError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            ResourceManagementError::CostCalculationError(msg) => {
                write!(f, "Cost calculation error: {}", msg)
            }
            ResourceManagementError::CapacityPlanningError(msg) => {
                write!(f, "Capacity planning error: {}", msg)
            }
            ResourceManagementError::LoadBalancingError(msg) => {
                write!(f, "Load balancing error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ResourceManagementError {}


//! # `ComplianceError` - Trait Implementations
//!
//! This module contains trait implementations for `ComplianceError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types_6::ComplianceError;

impl std::fmt::Display for ComplianceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplianceError::AssessmentError(msg) => {
                write!(f, "Assessment error: {}", msg)
            }
            ComplianceError::FrameworkError(msg) => write!(f, "Framework error: {}", msg),
            ComplianceError::RegulatoryError(msg) => {
                write!(f, "Regulatory error: {}", msg)
            }
            ComplianceError::AuditError(msg) => write!(f, "Audit error: {}", msg),
            ComplianceError::PolicyError(msg) => write!(f, "Policy error: {}", msg),
            ComplianceError::ControlsError(msg) => write!(f, "Controls error: {}", msg),
            ComplianceError::CertificationError(msg) => {
                write!(f, "Certification error: {}", msg)
            }
            ComplianceError::DocumentationError(msg) => {
                write!(f, "Documentation error: {}", msg)
            }
            ComplianceError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            ComplianceError::DataError(msg) => write!(f, "Data error: {}", msg),
        }
    }
}

impl std::error::Error for ComplianceError {}

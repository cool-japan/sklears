//! # MessageType - Trait Implementations
//!
//! This module contains trait implementations for `MessageType`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MessageType;

impl std::fmt::Display for MessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageType::JobSubmission => write!(f, "job_submission"),
            MessageType::JobResult => write!(f, "job_result"),
            MessageType::Heartbeat => write!(f, "heartbeat"),
            MessageType::ResourceUpdate => write!(f, "resource_update"),
            MessageType::ConsensusRequest => write!(f, "consensus_request"),
            MessageType::DataPartition => write!(f, "data_partition"),
            MessageType::Custom(s) => write!(f, "{s}"),
        }
    }
}

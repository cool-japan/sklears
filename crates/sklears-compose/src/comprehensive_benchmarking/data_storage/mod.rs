//! Data storage engine module
//!
//! This module provides comprehensive data storage capabilities for benchmark results
//! including backend management, indexing, retention, compression, caching, and backup.

// Re-export config_types so child modules can use `super::config_types::*`
pub use crate::comprehensive_benchmarking::config_types;

pub mod backup;
pub mod cache;
pub mod compression;
pub mod errors;
pub mod indexing;
pub mod integrity;
pub mod query;
pub mod retention;
pub mod storage_backend;

// Re-export main types
pub use backup::*;
pub use cache::*;
pub use compression::*;
pub use errors::*;
pub use indexing::*;
pub use integrity::*;
pub use query::*;
pub use retention::*;
pub use storage_backend::*;

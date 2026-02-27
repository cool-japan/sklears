//! Data storage engine module
//!
//! This module provides comprehensive data storage capabilities for benchmark results
//! including backend management, indexing, retention, compression, caching, and backup.

pub mod storage_backend;
pub mod indexing;
pub mod retention;
pub mod compression;
pub mod cache;
pub mod backup;
pub mod query;
pub mod integrity;
pub mod errors;

// Re-export main types
pub use storage_backend::*;
pub use indexing::*;
pub use retention::*;
pub use compression::*;
pub use cache::*;
pub use backup::*;
pub use query::*;
pub use integrity::*;
pub use errors::*;

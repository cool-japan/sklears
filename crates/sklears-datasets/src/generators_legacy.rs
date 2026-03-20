//! Legacy generator functions
//!
//! This module provides backward compatibility by re-exporting functions from the new
//! modular generator system. It serves as a bridge between the old monolithic generator
//! API and the new organized structure.

// Re-export functions from the new modular generators
pub use crate::generators::basic::*;
pub use crate::generators::experimental::*;
pub use crate::generators::multimodal::*;
pub use crate::generators::privacy::*;
pub use crate::generators::spatial::*;

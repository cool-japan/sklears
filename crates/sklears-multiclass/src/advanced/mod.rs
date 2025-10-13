//! Advanced multiclass classification strategies
//!
//! This module provides advanced and specialized multiclass classification approaches including:
//! Hierarchical classification, Taxonomy-aware classification, Adaptive decomposition,
//! Multi-level decomposition, and Stacking methods.

pub mod adaptive;
pub mod hierarchical;
pub mod multi_level;
pub mod stacking;
pub mod taxonomy_aware;

pub use hierarchical::*;
pub use stacking::*;

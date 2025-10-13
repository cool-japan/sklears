//! Multi-View Discriminant Analysis
//!
//! This module implements discriminant analysis for multi-modal data,
//! where multiple views or representations of the same instances are available.
//! It also supports heterogeneous feature integration for different data types.

pub mod trained;
pub mod types;
pub mod untrained;
pub mod views;

// Re-export main types for convenience
pub use trained::MultiViewDiscriminantAnalysisTrained;
pub use types::{
    FeatureType, FusionStrategy, HeterogeneousDistance, MultiViewDiscriminantAnalysisConfig,
    PreprocessingMethod,
};
pub use untrained::MultiViewDiscriminantAnalysisUntrained;
pub use views::{FeatureGroup, ViewInfo};

// Re-export with generic state parameter for backward compatibility
pub use untrained::MultiViewDiscriminantAnalysisUntrained as MultiViewDiscriminantAnalysis;

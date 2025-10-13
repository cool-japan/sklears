//! Advanced feature engineering utilities
//!
//! This module provides comprehensive advanced feature engineering functionality
//! through a modular architecture supporting multiple engineering paradigms.
//!
//! ## Architecture
//!
//! The advanced feature engineering system is organized into focused modules:
//! - **Topological Features**: Persistent homology, Mapper, simplicial complexes
//! - **Parallel Processing**: High-performance parallel feature extraction
//! - **Probabilistic Methods**: Count-Min Sketch, Johnson-Lindenstrauss embeddings
//! - **Mixed Type Features**: Heterogeneous data type processing
//! - **Sampling Methods**: Reservoir, importance, and stratified sampling
//! - **Fast Transforms**: High-speed transformation algorithms
//! - **Engineering Types**: Common type definitions and enumerations
//! - **Engineering Utils**: Shared utilities and helper functions
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use sklears_feature_extraction::engineering::*;
//! use scirs2_core::ndarray::Array2;
//!
//! // Topological data analysis
//! let topo_extractor = PersistentHomologyExtractor::new()
//!     .max_dimension(1)
//!     .resolution(50);
//!
//! // Parallel feature extraction
//! let parallel_extractor = ParallelFeatureExtractor::new()
//!     .n_jobs(4)
//!     .chunk_size(1000);
//!
//! // Mixed type processing
//! let mixed_extractor = MixedTypeFeatureExtractor::new()
//!     .missing_value_strategy(MissingValueStrategy::Mean)
//!     .categorical_encoding(CategoricalEncoding::OneHot);
//!
//! let X = Array2::from_elem((100, 10), 1.0);
//! let features = topo_extractor.extract_features(&X.view())?;
//! ```

mod topological_features;
mod parallel_processing;
mod probabilistic_methods;
mod mixed_type_features;
mod sampling_methods;
mod fast_transforms;
mod engineering_types;
mod engineering_utils;

pub use topological_features::*;
pub use parallel_processing::*;
pub use probabilistic_methods::*;
pub use mixed_type_features::*;
pub use sampling_methods::*;
pub use fast_transforms::*;
pub use engineering_types::*;
pub use engineering_utils::*;

// Re-export basic feature engineering methods
pub use crate::basic_features::*;

// Re-export RBF and kernel approximation methods
pub use crate::rbf_methods::*;

// Re-export temporal feature extraction methods
pub use crate::temporal_features::*;

// Re-export statistical feature extraction methods
pub use crate::statistical_features::*;

// =============================================================================
// Re-exports from time_series_features for backward compatibility
// =============================================================================

// Time series wavelet transform components have been moved to time_series_features.rs
pub use crate::time_series_features::{
    TimeSeriesWaveletExtractor, TimeSeriesWaveletType, WaveletFeatureType,
};

// Time-frequency feature extractor has been moved to time_series_features.rs
pub use crate::time_series_features::TimeFrequencyExtractor;

// =============================================================================
// Re-exports from topological_features for backward compatibility
// =============================================================================

// Topological feature extractors have been moved to topological_features.rs
// TODO: Re-enable after removing original definitions
// pub use crate::topological_features::{
//     PersistentHomologyExtractor, MapperExtractor, SimplicialComplexExtractor,
// };

#[allow(non_snake_case)]
#[cfg(test)]
mod tests;
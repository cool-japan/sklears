#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! Feature extraction from raw data (text, images)
//!
//! This module provides tools for extracting features from raw data such as
//! text documents and images.

// #![warn(missing_docs)]

use scirs2_core::ndarray::{s, Array1};
use scirs2_core::numeric::Float as NumTraitsFloat;
use sklears_core::{error::Result as SklResult, prelude::SklearsError, types::Float};

// Module declarations
pub mod audio;
pub mod basic_features;
pub mod biological;
pub mod dict_learning;
pub mod engineering;
pub mod feature_traits; // Trait-based feature extraction framework
pub mod graph;
pub mod image;
pub mod image_advanced;
pub mod information_theory;
pub mod manifold;
pub mod neural_simple;
pub use neural_simple as neural;
pub mod pipelines; // Feature extraction pipelines
pub mod signal_processing;
pub mod simd_image; // SIMD modules now use SciRS2-Core (stable Rust compatible)
pub mod simd_ops;
pub mod text;

// Extracted feature engineering modules
pub mod correlation_info_theory;
pub mod custom_transformers;
pub mod interaction;
pub mod polynomial;
pub mod polynomial_spline;
pub mod rbf_methods;
pub mod sketching_sampling;
pub mod statistical_features;
pub mod streaming_projection;
pub mod temporal_features;
pub mod time_series_features;
pub mod topological_features;
pub mod wavelet_features;

// Re-exports for convenience
pub use audio::*;
pub use biological::*;
pub use dict_learning::*;
pub use engineering::*;
pub use graph::*;
pub use image::*;
pub use image_advanced::*;
pub use information_theory::*;
pub use manifold::*;
pub use neural::*;
pub use signal_processing::*;
pub use simd_ops::*;
pub use text::*;

// Re-exports for extracted feature engineering modules
pub use correlation_info_theory::*;
pub use interaction::RadialBasisFunctions;
pub use streaming_projection::*;
pub use topological_features::*;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests;

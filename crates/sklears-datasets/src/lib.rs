#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
//! Dataset loading utilities and synthetic data generators

// Core modules that work
pub mod generators;
pub mod parallel_rng;
pub mod simd_gen;
pub mod traits;
pub mod validation;
pub mod versioning;
pub mod viz;

// Re-exports: Core traits
pub use parallel_rng::ParallelRng;
pub use simd_gen::SimdCapabilities;
pub use traits::{Dataset, DatasetGenerator, InMemoryDataset};
pub use versioning::{DatasetVersion, ProvenanceInfo};
pub use viz::PlotConfig;

// Re-exports: Generator functions from generators module
pub use generators::basic::{
    make_blobs, make_circles, make_classification, make_moons, make_regression,
};
pub use generators::performance::{parallel_generate, DatasetStream, LazyDatasetGenerator};
pub use generators::type_safe::{
    make_typed_blobs, make_typed_classification, make_typed_regression,
};

// Simple dataset
use scirs2_core::ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct SimpleDataset {
    pub features: Array2<f64>,
    pub targets: Option<Array1<f64>>,
}

impl SimpleDataset {
    pub fn new(features: Array2<f64>, targets: Option<Array1<f64>>) -> Self {
        Self { features, targets }
    }
    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }
}

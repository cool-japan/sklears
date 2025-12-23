#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! Dataset loading utilities and synthetic data generators
//!
//! This module provides functions to load built-in datasets and generate
//! synthetic data for testing and experimentation, compatible with
//! scikit-learn's datasets module.
//!
//! # Architecture
//!
//! The crate is organized into several focused modules:
//!
//! - **Core Traits**: Extensible trait-based framework for datasets
//! - **Generators**: Synthetic data generation (classification, regression, clustering, etc.)
//! - **Loaders**: Built-in dataset loaders (Iris, MNIST, etc.)
//! - **Validation**: Statistical validation and quality metrics
//! - **Format Support**: CSV, JSON, Parquet, HDF5, and cloud storage
//! - **Performance**: SIMD-optimized, parallel, and streaming generation
//! - **Memory Management**: Zero-copy views, memory pools, and memory-mapped storage
//! - **Versioning**: Dataset versioning, provenance tracking, and checksums
//! - **Visualization**: Dataset visualization utilities
//!
//! # SciRS2 Policy Compliance
//!
//! This crate adheres to the SciRS2 Policy:
//! - Uses `scirs2_core::ndarray` instead of direct ndarray
//! - Uses `scirs2_core::random` instead of rand/rand_distr
//! - Follows SciRS2 ecosystem patterns

// ============================================================================
// Core trait system
// ============================================================================
pub mod traits;

// ============================================================================
// Dataset generators
// ============================================================================
pub mod generators;

// ============================================================================
// Validation and quality metrics
// ============================================================================
pub mod validation;

// ============================================================================
// Format support and I/O
// ============================================================================
pub mod format;
pub mod loaders;

// ============================================================================
// Memory management
// ============================================================================
pub mod memory;
pub mod memory_pool;
pub mod zero_copy;

// ============================================================================
// Configuration and templates
// ============================================================================
pub mod config;
pub mod config_templates;

// ============================================================================
// Advanced features
// ============================================================================
pub mod composable;
pub mod plugins;
pub mod streaming;

// ============================================================================
// Performance optimizations
// ============================================================================
pub mod parallel_rng;
pub mod simd_gen;

// ============================================================================
// Versioning and provenance
// ============================================================================
pub mod versioning;

// ============================================================================
// Visualization
// ============================================================================
pub mod viz;

// ============================================================================
// Specialized modules
// ============================================================================
pub mod basic;
pub mod benchmarks;
pub mod classification_regression;
pub mod distributions;
pub mod domain_specific;
pub mod graphs;
pub mod manifold;
pub mod manifolds_spatial;
pub mod matrix;
pub mod missing_data;
pub mod specialized;
pub mod timeseries;

// ============================================================================
// Re-exports: Core types and traits
// ============================================================================
pub use traits::{
    Dataset, DatasetGenerator, DatasetLoader, DatasetTraitError, DatasetTraitResult,
    DatasetTransformer, DatasetValidator, GenerationStrategy, GeneratorConfig, GeneratorRegistry,
    InMemoryDataset, MutableDataset, StreamingDataset,
};

// ============================================================================
// Re-exports: Versioning and provenance
// ============================================================================
pub use versioning::{
    calculate_checksum, verify_checksum, DatasetVersion, ProvenanceInfo, TransformationStep,
    VersionRegistry, VersioningError, VersioningResult,
};

// ============================================================================
// Re-exports: Visualization
// ============================================================================
pub use viz::{
    plot_2d_classification, plot_2d_regression, plot_feature_distributions, PlotConfig,
    VisualizationError, VisualizationResult,
};

// ============================================================================
// Re-exports: SIMD-optimized generators
// ============================================================================
pub use simd_gen::{
    add_vectors_simd, generate_normal_matrix_simd, make_classification_simd, make_regression_simd,
    scale_vector_simd, SimdCapabilities,
};

// ============================================================================
// Re-exports: Parallel RNG generators
// ============================================================================
pub use parallel_rng::{
    make_blobs_parallel, make_classification_parallel, make_regression_parallel, ParallelRng,
};

// ============================================================================
// Re-exports: Generator functions (from generators module)
// ============================================================================
pub use generators::{
    // Basic generators
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
    make_regression,
    // Type-safe generators
    make_typed_blobs,
    make_typed_classification,
    make_typed_regression,
    // Performance generators
    DatasetStream,
    LazyDatasetGenerator,
    parallel_generate,
    // Privacy and federated learning
    make_federated_partitions,
    make_privacy_preserving_dataset,
    // Multi-modal and multi-agent
    make_audio_visual_dataset,
    make_communication_cost_datasets,
    make_cross_modal_retrieval_dataset,
    make_multi_agent_environment,
    make_multimodal_alignment_dataset,
    make_sensor_fusion_dataset,
    make_vision_language_dataset,
    // Spatial and geostatistical
    make_geographic_information_dataset,
    make_geostatistical_data,
    make_spatial_clustering_dataset,
    make_spatial_point_process,
    // Experimental design
    make_ab_testing_simulation,
    make_factorial_design,
};

// ============================================================================
// Re-exports: Validation
// ============================================================================
pub use validation::{
    // Types
    ValidationConfig,
    ValidationMetrics,
    ValidationReport,
    ValidationResult,
    // Basic validation
    validate_dataset_basic,
    validate_feature_names,
    validate_no_missing_values,
    validate_shape,
    // Advanced validation
    calculate_dataset_quality_metrics,
    detect_anomalies,
    detect_data_drift,
    // Distribution testing
    chi_square_test,
    kolmogorov_smirnov_test,
    validate_exponential_distribution,
    validate_normal_distribution,
    validate_uniform_distribution,
    // Summary statistics
    generate_statistical_summary,
};

// ============================================================================
// Re-exports: Configuration
// ============================================================================
pub use config::{ConfigError, ConfigResult, GenerationConfig};

// ============================================================================
// Re-exports: Memory management
// ============================================================================
pub use memory::{MemoryError, MemoryResult, MmapDataset, MmapDatasetBuilder};
pub use memory_pool::{
    ArenaAllocator, MemoryBlock as MemoryPoolBlock, MemoryPool, MemoryPoolConfig,
    MemoryPoolError, MemoryPoolResult, PooledArray1, PooledArray2,
};
pub use zero_copy::{
    BatchView, DatasetView, DatasetViewMut, ZeroCopyError, ZeroCopyResult, ZeroCopySlice,
};

// ============================================================================
// Re-exports: Streaming
// ============================================================================
pub use streaming::{
    BufferedStream, ChunkIterator, StreamingConfig, StreamingDatasetGenerator, StreamingError,
    StreamingResult,
};

// ============================================================================
// Re-exports: Plugins
// ============================================================================
pub use plugins::{
    GeneratorPlugin, PluginError, PluginManager, PluginMetadata, PluginRegistry, PluginResult,
};

// ============================================================================
// Re-exports: Composable strategies
// ============================================================================
pub use composable::{
    ComposableError, ComposableResult, GenerationPipeline, StrategyConfig, StrategyValue,
};

// ============================================================================
// Simple dataset structure for basic functionality
// ============================================================================
use scirs2_core::ndarray::{Array1, Array2};

/// Simple dataset structure for basic functionality
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

// ============================================================================
// Tests
// ============================================================================
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_dataset() {
        let features = Array2::zeros((10, 3));
        let targets = Some(Array1::zeros(10));
        let dataset = SimpleDataset::new(features, targets);

        assert_eq!(dataset.n_samples(), 10);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.targets.is_some());
    }

    #[test]
    fn test_module_structure() {
        // Verify that key modules are accessible
        let _ = ValidationConfig::default();
        let _ = PlotConfig::default();
        let _ = SimdCapabilities::detect();
    }
}

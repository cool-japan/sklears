//! Dataset loading utilities and synthetic data generators
//!
//! This module provides functions to load built-in datasets and generate
//! synthetic data for testing and experimentation, compatible with
//! scikit-learn's datasets module.

pub mod benchmarks;
pub mod config;
pub mod config_templates;
pub mod format;
pub mod generators;
pub mod generators_legacy;
pub mod loaders;
pub mod memory;
pub mod memory_pool;
pub mod plugins;
pub mod samples;
pub mod traits;
pub mod validation;
pub mod visualization;
pub mod zero_copy;

// TODO: Refactored generator modules (to be implemented)
// pub mod basic;
// pub mod manifold;
// pub mod matrix;
// pub mod specialized;
// pub mod tests;

// Re-export only the functions that are actually implemented
pub use crate::generators_legacy::{
    // Basic generators that exist
    make_blobs,
    make_circles,
    make_classification,
    make_gaussian_quantiles,
    // TODO: Many functions are not yet implemented - they will be added as modules are completed
    make_moons,
    make_regression,
};

// TODO: The following functions need to be implemented in generator modules:
// make_hastie_10_2, make_friedman1, make_friedman2, make_friedman3,
// make_swiss_roll, make_s_curve, make_biclusters, make_checkerboard,
// make_low_rank_matrix, make_sparse_coded_signal, make_sparse_spd_matrix, make_spd_matrix,
// make_sparse_uncorrelated, make_polynomial_regression, make_nonstationary_timeseries,
// make_erdos_renyi_graph, make_barabasi_albert_graph, make_watts_strogatz_graph,
// make_stochastic_block_graph, make_random_tree, make_gaussian_mixture,
// make_distribution_mixture, make_multivariate_mixture, make_heavy_tailed_distribution,
// make_custom_manifold, make_n_sphere, make_n_torus, make_spatial_point_pattern,
// make_missing_completely_at_random, make_missing_at_random, make_missing_not_at_random,
// make_outliers, make_imbalanced_classification, make_anomalies,
// make_synthetic_image_classification, make_gene_expression_dataset, make_dna_sequence_dataset,
// make_document_clustering_dataset, make_privacy_preserving_dataset,
// make_multi_agent_environment, make_ab_testing_simulation, make_geostatistical_data,
// ABTestConfig, ManifoldGenerator, MultiAgentConfig

// Re-export new modular generators
pub use crate::generators::{
    classification_builder,
    clustering_builder,
    distributed_blobs,
    distributed_classification,
    distributed_regression,
    // SIMD-optimized generators
    get_simd_info,
    lazy_classification,
    lazy_regression,
    make_communication_cost_datasets,
    make_cross_modal_retrieval_dataset,
    make_geographic_information_dataset,
    make_multimodal_alignment_dataset,
    make_sensor_fusion_dataset,
    make_simd_classification,
    make_simd_regression,
    make_spatial_clustering_dataset,
    make_typed_blobs,
    make_typed_classification,
    make_typed_regression,
    parallel_blobs,
    parallel_classification,
    parallel_regression,
    regression_builder,
    stream_blobs,
    stream_classification,
    stream_regression,
    Classification,
    Clustering,
    CommunicationCostConfig,
    DatasetBuilder,
    DatasetConfig,
    // Performance and streaming generators
    DatasetStream,
    DatasetTargets,
    // Distributed generation
    DistributedConfig,
    DistributedGenerationResult,
    DistributedGenerator,
    GeographicConfig,
    LazyDatasetGenerator,
    LoadBalancingStrategy,
    NodeInfo,
    NodeResult,
    NodeStatus,
    ParallelGenerationResult,
    Regression,
    // SIMD configuration
    SimdConfig,
    SimdError,
    SimdResult,
    Spatial,
    StreamConfig,
    TimeSeries,
    // Type-safe dataset abstractions
    TypeSafeDataset,
    ValidateDataset,
};

pub use crate::loaders::{
    load_boston, load_breast_cancer, load_california_housing, load_cifar10, load_diabetes,
    load_digits, load_fashion_mnist, load_iris, load_linnerud, load_mnist, load_newsgroups,
    load_olivetti_faces, load_reuters, load_wine, Dataset,
};

pub use crate::validation::{
    // Dataset quality metrics and advanced analysis
    calculate_dataset_quality_metrics,
    chi_square_goodness_of_fit_test,
    detect_anomalies,
    detect_data_drift,
    generate_statistical_summary,
    kolmogorov_smirnov_test,
    validate_basic_statistics,
    validate_correlation_structure,
    validate_dataset,
    validate_distribution_properties,
    validate_exponential_distribution,
    validate_normal_distribution,
    validate_normality,
    validate_outliers,
    validate_uniform_distribution,
    AnomalyDetectionResult,
    DataDriftReport,
    DatasetQualityMetrics,
    DistributionType,
    FeatureStatistics,
    StatisticalSummary,
    SummaryConfig,
    TargetStatistics,
    ValidationConfig,
    ValidationReport,
    ValidationResult,
};

pub use crate::benchmarks::{
    benchmark_generator, benchmark_memory_scalability, benchmark_parallel_generator,
    benchmark_streaming_performance, run_comprehensive_benchmarks, BenchmarkConfig,
    BenchmarkMetrics, BenchmarkReport,
};

#[cfg(feature = "serde")]
pub use crate::config::{
    generate_example_config, ClassificationConfig, ClusteringConfig, ConfigError, ConfigLoader,
    ConfigMetadata, ConfigResult, CustomDatasetConfig, DatasetSpec, ExportConfig, ExportFormat,
    GenerationConfig, GlobalSettings, ManifoldConfig, ManifoldType, RegressionConfig,
    StatisticalTest, TimeSeriesConfig, ValidationSettings,
};

#[cfg(not(feature = "serde"))]
pub use crate::config::{generate_example_config, ConfigError, ConfigLoader};

pub use crate::format::{
    export_classification_csv, export_classification_jsonl, export_classification_tsv,
    export_regression_csv, export_regression_tsv, import_classification_csv, import_regression_csv,
    CsvConfig, FormatError, FormatResult, SerializableDataset,
};

#[cfg(feature = "serde")]
pub use crate::format::{export_classification_json, export_regression_json};

#[cfg(feature = "parquet")]
pub use crate::format::{
    export_classification_parquet, export_regression_parquet, import_classification_parquet,
    import_regression_parquet,
};

#[cfg(feature = "hdf5")]
pub use crate::format::{
    export_classification_hdf5, export_regression_hdf5, import_classification_hdf5,
    import_regression_hdf5,
};

#[cfg(feature = "cloud-storage")]
pub use crate::format::{upload_classification_to_cloud, CloudStorageProvider};

#[cfg(feature = "cloud-s3")]
pub use crate::format::{
    download_classification_from_s3, download_regression_from_s3, upload_classification_to_s3,
    upload_regression_to_s3,
};

#[cfg(feature = "cloud-gcs")]
pub use crate::format::{
    download_classification_from_gcs, download_regression_from_gcs, upload_classification_to_gcs,
    upload_regression_to_gcs,
};

#[cfg(feature = "cloud-storage")]
pub use crate::format::upload_regression_to_cloud;

#[cfg(feature = "visualization")]
pub use crate::visualization::{
    plot_2d_classification, plot_2d_regression, plot_correlation_matrix, plot_dataset_comparison,
    plot_feature_distributions, plot_quality_metrics, PlotConfig, VisualizationError,
    VisualizationResult,
};

pub use crate::memory::{
    ArenaAllocation, ArenaUsageStats, DatasetArena, MemoryError, MemoryResult, MmapBatchIterator,
    MmapDataset, MmapDatasetMut,
};

pub use crate::traits::{
    create_default_registry, ClassificationGenerator, ConfigValue, Dataset, DatasetGenerator,
    DatasetLoader, DatasetTraitError, DatasetTraitResult, DatasetTransformer, DatasetValidator,
    GenerationStrategy, GeneratorConfig, GeneratorRegistry, InMemoryDataset, MutableDataset,
    RegressionGenerator, StreamingDataset,
};

pub use crate::zero_copy::{
    BatchIterator, DatasetSampleIterator, DatasetView, DatasetViewMut, FilteredDatasetView,
    SelectedFeaturesView, StridedDatasetView, WindowView, ZeroCopyError, ZeroCopyResult,
};

pub use crate::plugins::{
    CustomLinearGenerator, LoggingHook, ParameterConstraint, ParameterInfo, ParameterType,
    PluginError, PluginGenerator, PluginHook, PluginManager, PluginMetadata, PluginRegistry,
    PluginResult,
};

pub use crate::memory_pool::{
    allocate_array1_global, allocate_array2_global, allocate_global, create_shared_pool,
    global_pool, MemoryPool, MemoryPoolConfig, MemoryPoolError, MemoryPoolResult, MemoryPoolStats,
    PooledArray1, PooledArray2, PooledBlock, SharedMemoryPool,
};

#[cfg(feature = "serde")]
pub use crate::config_templates::{
    create_classification_template, create_regression_template, ConfigTemplateError,
    ConfigTemplateResult, DatasetSizeConstraints, DatasetTemplate, ExperimentTemplate,
    ExportTemplate, GeneratorTemplateConfig, ParameterConstraints, ParameterTemplate,
    ParameterTypeTemplate, StatisticalConstraints, TemplateBuilder, TemplateLibrary,
    TemplateMetadata, ValidationRules,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

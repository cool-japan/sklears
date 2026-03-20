#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::iter_kv_map)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::unused_enumerate_index)]
#![allow(clippy::manual_map)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_unwrap_or_default)]
//! Neighbor-based algorithms for machine learning
//!
//! This crate provides k-nearest neighbors (k-NN) and related algorithms for
//! classification, regression, and outlier detection.
//!
//! # Examples
//!
//! ```rust
//! use sklears_neighbors::KNeighborsClassifier;
//! use sklears_core::traits::{Fit, Predict};
//! use scirs2_core::ndarray::{array, Array2};
//!
//! // Create training data
//! let X = Array2::from_shape_vec((4, 2), vec![
//!     1.0, 2.0,
//!     2.0, 3.0,
//!     3.0, 1.0,
//!     4.0, 2.0,
//! ]).unwrap();
//! let y = array![0, 0, 1, 1];
//!
//! // Train classifier
//! let classifier = KNeighborsClassifier::new(3);
//! let fitted = classifier.fit(&X, &y).unwrap();
//!
//! // Make predictions
//! let X_test = Array2::from_shape_vec((2, 2), vec![
//!     1.5, 2.5,
//!     3.5, 1.5,
//! ]).unwrap();
//! let predictions = fitted.predict(&X_test).unwrap();
//! ```

pub mod abod;
pub mod adaptive_distance;
pub mod advanced_outliers;
pub mod approximate_distance;
pub mod batch_processing;
pub mod bayesian_neighbors;
pub mod bioinformatics;
pub mod compressed_distance;
pub mod computer_vision;
pub mod cross_validation;
pub mod density_estimation;
pub mod distance;
pub mod distributed_neighbors;
pub mod federated_neighbors;
pub mod gpu_distance;
pub mod graph_methods;
pub mod incremental_index;
pub mod interpretability;
pub mod knn;
pub mod local_outlier_factor;
pub mod lsh;
pub mod manifold_learning;
pub mod mapreduce_neighbors;
pub mod memory_constrained;
pub mod memory_mapped;
pub mod metric_learning;
pub mod multi_view_learning;
pub mod nearest_centroid;
pub mod nearest_neighbors;
pub mod nlp;
pub mod online_learning;
pub mod parallel_tree;
pub mod performance;
pub mod radius_neighbors;
pub mod simd_distance;
pub mod sparse_neighbors;
pub mod spatial;
pub mod specialized_distances;
pub mod streaming;
pub mod time_series_neighbors;
pub mod transformers;
pub mod tree;
pub mod validation;

#[allow(non_snake_case)]
#[cfg(test)]
mod property_tests;

#[allow(non_snake_case)]
#[cfg(test)]
pub mod comprehensive_tests;

pub use abod::AngleBasedOutlierDetection;
pub use adaptive_distance::{
    AdaptiveDensityDistance, CombinationMethod, ContextDependentDistance, EnsembleDistance,
    OnlineAdaptiveDistance,
};
pub use advanced_outliers::{
    ConnectivityBasedOutlierFactor, IsolationForest, LocalCorrelationIntegral,
};
pub use approximate_distance::ApproximateDistance;
pub use batch_processing::{
    BatchConfiguration, BatchNeighborSearch, BatchProcessable, BatchProcessor, BatchResult,
    BatchStatistics, MemoryMonitor,
};
pub use bayesian_neighbors::{
    BayesianKNeighborsClassifier, BayesianKNeighborsRegressor, BayesianPrediction,
    BayesianRegressionPrediction, CredibleNeighborSet, UncertaintyMethod,
};
pub use bioinformatics::{
    BioSearchConfig, GeneExpressionNeighbors, GeneExpressionResult, GeneMetadata, KmerIndex,
    ProteinMetadata, ProteinSearchResult, ProteinStructure, ProteinStructureSearch, ScoringScheme,
    SequenceAligner, SequenceAlignment, SequenceMetadata, SequenceSearchResult,
    SequenceSimilaritySearch, SequenceType,
};
pub use compressed_distance::{CompressedDistanceMatrix, CompressionMethod, CompressionStats};
pub use computer_vision::{
    ColorHistogramExtractor, DescriptorMatch, FeatureDescriptorMatcher, FeatureExtractor,
    FeatureType, HistogramOfGradientsExtractor, ImageMetadata, ImageSearchConfig,
    ImageSearchResult, ImageSimilaritySearch, Keypoint, LocalBinaryPatternExtractor,
    PatchBasedMatching, VisualWordRecognizer,
};
pub use cross_validation::{CVFoldResult, CVResults, CVStrategy, NeighborCrossValidator};
pub use density_estimation::{
    BandwidthMethod, DensityBasedClustering, KNeighborsDensityEstimator, KernelType,
    LocalDensityEstimator, VariableBandwidthKDE,
};
pub use distance::Distance;
pub use distributed_neighbors::{
    DataPartitioner, DistributedConfiguration, DistributedNeighborSearch,
    DistributedNeighborSearchResult, DistributedWorker, LoadBalanceStrategy, PartitionInfo,
    PartitionStats,
};
pub use federated_neighbors::{
    FederatedConfig, FederatedNeighborCoordinator, FederatedParticipant, NoiseStrategy,
    PrivacyLevel, PrivacyPreservingProtocol,
};
pub use gpu_distance::{
    GpuBackend, GpuComputationStats, GpuConfig, GpuDeviceInfo, GpuDistanceCalculator,
    GpuDistanceResult, GpuKNeighborsSearch, GpuMemoryEstimator, GpuMemoryStrategy,
};
pub use graph_methods::{
    EpsilonGraph, GabrielGraph, GraphEdge, GraphNeighborSearch, GraphStatistics,
    KNearestNeighborGraph, MutualKNearestNeighbors, NeighborhoodGraph, RelativeNeighborhoodGraph,
};
pub use incremental_index::{
    IncrementalIndexBuilder, IncrementalIndexType, IncrementalNeighborIndex,
    IndexPerformanceMetrics, UpdateStrategy,
};
pub use interpretability::{
    InfluenceAnalysis, LocalImportanceExplanation, NeighborExplainer, NeighborExplanation,
    Prototype,
};
pub use knn::{KNeighborsClassifier, KNeighborsRegressor};
pub use local_outlier_factor::LocalOutlierFactor;
pub use lsh::{HashFamily, LshIndex, LshKNeighborsClassifier};
pub use manifold_learning::{Isomap, LaplacianEigenmaps, LocallyLinearEmbedding, TSNENeighbors};
pub use mapreduce_neighbors::{
    DistributedMapReduce, MapReduceConfig, MapReduceNeighborSearch, PartitionStrategy,
    ReduceStrategy,
};
pub use memory_constrained::{
    CacheObliviousNeighbors, ExternalMemoryKNN, MemoryBoundedApproximateNeighbors,
};
pub use memory_mapped::{MmapNeighborIndex, MmapNeighborIndexBuilder};
pub use metric_learning::{
    EnhancedLMNN, InformationTheoreticMetricLearning, LargeMarginNearestNeighbor,
    NeighborhoodComponentsAnalysis, OnlineMetricLearning,
};
pub use multi_view_learning::{
    ConsensusAnalysis, FusionStrategy, MultiViewKNeighborsClassifier, MultiViewKNeighborsRegressor,
    RegressionFusionStrategy, ViewConfig,
};
pub use nearest_centroid::{CentroidType, ClassConfig, NearestCentroid};
pub use nearest_neighbors::{kneighbors_graph, radius_neighbors_graph, NearestNeighbors};
pub use nlp::{
    DocumentFeatureExtractor, DocumentMetadata, DocumentSearchResult, DocumentSimilaritySearch,
    NlpSearchConfig, SentenceSimilaritySearch, TextFeatureType, TextPreprocessor, TfIdfExtractor,
    WordEmbeddingSearch,
};
pub use online_learning::{
    AdaptiveKNeighborsClassifier, DriftDetectionMethod, DriftDetector, StreamingOutlierDetector,
};
pub use parallel_tree::{ParallelBuildStrategy, ParallelTreeBuilder, ParallelTreeIndex, WorkUnit};
pub use performance::{
    BenchmarkConfig, BenchmarkResult, NeighborBenchmark, PerformanceMetrics, QuickProfiler,
};
pub use radius_neighbors::{
    AdaptiveRadiusNeighborsClassifier, AdaptiveRadiusNeighborsRegressor, RadiusNeighborsClassifier,
    RadiusNeighborsRegressor, RadiusStrategy,
};
pub use simd_distance::{
    batch_euclidean_distances, pairwise_distances_simd, SimdCapability, SimdDistanceCalculator,
};
pub use sparse_neighbors::{SparseIndexType, SparseNeighborBuilder, SparseNeighborMatrix};
pub use spatial::{
    OctPoint, OctTree, QuadPoint, QuadTree, RTree, Rectangle, SpatialHash, SpatialHashStats,
};
pub use specialized_distances::{
    CategoricalDistance, GraphDistance, ProbabilisticDistance, SetDistance, SimpleGraph,
    StringDistance,
};
pub use streaming::{
    IncrementalKNeighborsClassifier, IncrementalKNeighborsRegressor, MemoryStrategy,
};
pub use time_series_neighbors::{
    DtwDistance, DtwStepPattern, Shapelet, ShapeletDiscovery, StreamingTimeSeriesNeighbors,
    SubsequenceSearch, TemporalNeighborSearch,
};
pub use transformers::{KNeighborsTransformer, RadiusNeighborsTransformer};
pub use type_safe_distance::{
    ChebyshevMetric, ComputeDistance, CosineMetric, EuclideanMetric, ManhattanMetric,
    MetricDistance, MinkowskiMetric, NonMetricDistance, NormalizedDistance, TypeSafeDistance,
    TypeSafeKnnConfig,
};
pub use validation::{
    BootstrapResult, BootstrapValidator, ClassificationMetric, CrossValidationResult, GridSearchCV,
    GridSearchResult, KFoldValidator, RegressionMetric,
};

use sklears_core::types::Float;

/// Common error type for neighbors algorithms
#[derive(thiserror::Error, Debug)]
pub enum NeighborsError {
    #[error("Invalid number of neighbors: {0}")]
    InvalidNeighbors(usize),
    #[error("Invalid radius: {0}")]
    InvalidRadius(Float),
    #[error("Empty input data")]
    EmptyInput,
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("No neighbors found")]
    NoNeighbors,
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl From<NeighborsError> for sklears_core::error::SklearsError {
    fn from(err: NeighborsError) -> Self {
        sklears_core::error::SklearsError::InvalidInput(err.to_string())
    }
}

impl From<sklears_core::error::SklearsError> for NeighborsError {
    fn from(err: sklears_core::error::SklearsError) -> Self {
        NeighborsError::InvalidInput(err.to_string())
    }
}

impl From<scirs2_core::ndarray::ShapeError> for NeighborsError {
    fn from(err: scirs2_core::ndarray::ShapeError) -> Self {
        NeighborsError::InvalidInput(format!("Shape error: {}", err))
    }
}

/// Type alias for neighbors results
pub type NeighborsResult<T> = std::result::Result<T, NeighborsError>;
pub mod type_safe_distance;

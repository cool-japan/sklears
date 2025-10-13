#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_doc_comments)]
#![allow(unused_parens)]
#![allow(unused_comparisons)]
//! Clustering algorithms for sklears
//!
//! This crate provides implementations of clustering algorithms including:
//! - K-Means clustering with various initialization methods
//! - X-Means for automatic cluster number selection
//! - G-Means for Gaussian cluster detection with automatic number selection
//! - Mini-batch K-Means for large datasets
//! - Fuzzy C-Means clustering with membership degrees
//! - DBSCAN (Density-Based Spatial Clustering)
//! - Incremental DBSCAN for streaming data and large datasets
//! - HDBSCAN (Hierarchical Density-Based Spatial Clustering)
//! - OPTICS (Ordering Points To Identify Clustering Structure)
//! - BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
//! - Hierarchical clustering
//! - Mean Shift with adaptive bandwidth estimation
//! - Density Peaks clustering for automatic cluster center detection
//! - KDE Clustering using kernel density estimation for density-based clustering
//! - Spectral Clustering
//! - Gaussian Mixture Models with model selection criteria (AIC, BIC, ICL)
//! - Dirichlet Process Mixture Models for infinite mixture modeling
//! - Local Outlier Factor (LOF) for density-based outlier detection
//! - CURE (Clustering Using REpresentatives) for large datasets with irregular shapes
//! - ROCK (RObust Clustering using linKs) for categorical data clustering
//! - Streaming clustering algorithms (Online K-Means, CluStream, Sliding Window K-Means)
//! - Graph clustering algorithms (Modularity-based, Louvain, Label Propagation, Spectral)
//! - Evolutionary and bio-inspired clustering algorithms (PSO, GA, ACO, ABC, Differential Evolution)
//! - Comprehensive validation metrics for clustering evaluation including stability analysis
//!
//! These implementations leverage scirs2's cluster module for efficient computation.

pub mod birch;
pub mod cure;
pub mod dbscan;
pub mod density_peaks;
pub mod dirichlet_process;
#[cfg(feature = "parallel")]
pub mod distributed;
pub mod evolutionary;
pub mod fuzzy_cmeans;
pub mod gmm;
#[cfg(feature = "gpu")]
pub mod gpu_distances;
pub mod graph_clustering;
pub mod hdbscan;
pub mod hierarchical;
pub mod incremental_dbscan;
pub mod kde_clustering;
pub mod kmeans;
pub mod locality_sensitive_hashing;
pub mod lof;
pub mod mean_shift;
pub mod memory_mapped;
pub mod multi_view;
pub mod optics;
pub mod out_of_core;
pub mod rock;
pub mod semi_supervised;
pub mod simd_distances;
pub mod sparse_matrix;
pub mod spectral;
pub mod streaming;
pub mod text_clustering;
pub mod time_series;
pub mod validation;

#[cfg(feature = "parallel")]
pub mod parallel;
#[cfg(feature = "parallel")]
pub mod parallel_hierarchical;
pub mod performance;

pub use birch::{BIRCHConfig, ClusteringFeature, BIRCH};
pub use cure::{CUREConfig, CUREDistanceMetric, CUREFitted, CURE};
pub use dbscan::{DBSCANConfig, DBSCAN, NOISE};
pub use density_peaks::{
    DensityPeaks, DensityPeaksConfig, DistanceMetric as DensityPeaksDistanceMetric,
};
pub use dirichlet_process::{DirichletProcessConfig, DirichletProcessMixture, PredictProbaDP};
#[cfg(feature = "parallel")]
pub use distributed::{
    DBSCANWorker, DataPartition, DistributedConfig, DistributedDBSCAN, WorkerMessage,
};
pub use evolutionary::{PSOClustering, PSOClusteringBuilder, PSOClusteringFitted};
pub use fuzzy_cmeans::{FuzzyCMeans, FuzzyCMeansConfig, PredictMembership};
pub use gmm::{
    BayesianGaussianMixture, CovarianceType, GaussianMixture, GaussianMixtureConfig,
    ModelSelectionCriterion, ModelSelectionResult, PredictProba, WeightInit,
};
#[cfg(feature = "gpu")]
pub use gpu_distances::{GpuConfig, GpuDistanceComputer, GpuDistanceMetric};
pub use graph_clustering::{
    Graph, GraphClusteringResult, LabelPropagationClustering,
    LabelPropagationConfig as GraphLabelPropagationConfig, LouvainClustering, LouvainConfig,
    LouvainResult, ModularityClustering, ModularityClusteringConfig, SpectralGraphClustering,
    SpectralGraphConfig,
};
pub use hdbscan::{ClusterStat, HDBSCANConfig, HDBSCAN};
pub use hierarchical::{
    AgglomerativeClustering, AgglomerativeClusteringConfig, Constraint, ConstraintSet, Dendrogram,
    DendrogramExport, DendrogramLinkExport, DendrogramNode, DendrogramNodeExport, MemoryStrategy,
};
pub use incremental_dbscan::{
    DistanceMetric as IncrementalDistanceMetric, IncrementalDBSCAN, IncrementalDBSCANConfig,
};
pub use kde_clustering::{BandwidthMethod, KDEClustering, KDEClusteringConfig, KernelType};
pub use kmeans::{
    GMeans, GMeansConfig, InformationCriterion, KMeans, KMeansConfig, KMeansInit, MiniBatchKMeans,
    MiniBatchKMeansConfig, XMeans, XMeansConfig,
};
pub use locality_sensitive_hashing::{
    LSHConfig, LSHFamily, LSHIndex, LSHIndexStats, MemoryUsage, TableStats,
};
pub use lof::{DistanceMetric as LOFDistanceMetric, LOFConfig, LOF};
pub use mean_shift::{MeanShift, MeanShiftConfig};
pub use memory_mapped::{MemoryMappedConfig, MemoryMappedDistanceMatrix, MemoryStats};
pub use multi_view::{
    ConsensusClustering, ConsensusClusteringConfig, ConsensusClusteringFitted, ConsensusMethod,
    MultiViewData, MultiViewKMeans, MultiViewKMeansConfig, MultiViewKMeansFitted, ViewWeighting,
    WeightLearning,
};
pub use optics::{
    Algorithm, ClusterMethod, DistanceMetric as OpticsDistanceMetric, Optics, OpticsConfig,
    OpticsOrdering,
};
pub use out_of_core::{ClusterSummary, OutOfCoreConfig, OutOfCoreDataLoader, OutOfCoreKMeans};
pub use rock::{ROCKConfig, ROCKFitted, ROCKSimilarity, ROCK};
pub use semi_supervised::{
    ConstrainedKMeans, ConstrainedKMeansConfig, ConstrainedKMeansFitted, ConstraintHandling,
    ConstraintType, LabelPropagation, LabelPropagationConfig, LabelPropagationFitted,
};
pub use simd_distances::{
    simd_distance, simd_distance_batch, simd_k_nearest_neighbors, DistanceMetric,
    OptimizedDistanceComputer, SimdDistanceMetric,
};
pub use sparse_matrix::{
    GraphStats, SparseDistanceMatrix, SparseEntry, SparseMatrixConfig, SparseMatrixStats,
    SparseNeighborhoodGraph,
};
pub use spectral::{
    Affinity, EigenSolver, NormalizationMethod, SpectralClustering, SpectralClusteringConfig,
};
pub use streaming::{CluStream, MicroCluster, OnlineKMeans, SlidingWindowKMeans, StreamingConfig};
pub use text_clustering::{
    DocumentClustering, DocumentClusteringConfig, DocumentClusteringResult, SphericalInit,
    SphericalKMeans, SphericalKMeansConfig, SphericalKMeansFitted,
};
pub use time_series::{
    CentroidAveraging, ChangeDetectionTest, DTWKMeans, DTWKMeansConfig, DTWKMeansFitted,
    RegimeChangeConfig, RegimeChangeDetector, RegimeChangeResult, ShapeClustering,
    ShapeClusteringConfig, ShapeClusteringFitted, ShapeDistanceMetric,
    TemporalSegmentationClustering, TemporalSegmentationConfig, TemporalSegmentationResult,
};
pub use validation::{
    ClusteringValidator,
    GapStatisticResult,
    SilhouetteResult,
    ValidationMetric,
    // TODO: Fix these imports
    // AccuracyMetrics, CrossValidationStabilityResult,
    // ExternalValidationMetrics, FoldResult, NoiseStabilityResult,
    // ParameterAgreement, ParameterResult, ParameterSensitivityResult, PerturbationStabilityResult,
    // StabilityMetrics, StabilityResult, SubsampleStabilityResult, ValidationMetrics,
};

#[cfg(feature = "parallel")]
pub use parallel::{SimpleParallelKMeans, SimpleParallelKMeansFitted};
#[cfg(feature = "parallel")]
pub use parallel_hierarchical::{
    ClusterMerge, DistanceChunk, ParallelClusteringState, ParallelHierarchicalClustering,
    ParallelHierarchicalConfig,
};

// Re-export parallel DBSCAN when parallel feature is enabled
#[cfg(feature = "parallel")]
pub use sklears_core::parallel::ParallelFit;

// Re-export commonly used types from scirs2
pub use scirs2_cluster::density::DistanceMetric as DensityDistanceMetric;
pub use scirs2_cluster::hierarchy::{LinkageMethod, Metric};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::birch::{ClusteringFeature, BIRCH};
    pub use crate::cure::{CUREDistanceMetric, CURE};
    pub use crate::dbscan::{DBSCAN, NOISE};
    pub use crate::density_peaks::{DensityPeaks, DistanceMetric as DensityPeaksDistanceMetric};
    pub use crate::dirichlet_process::{DirichletProcessMixture, PredictProbaDP};
    #[cfg(feature = "parallel")]
    pub use crate::distributed::{DBSCANWorker, DataPartition, DistributedDBSCAN};
    pub use crate::evolutionary::PSOClustering;
    pub use crate::fuzzy_cmeans::{FuzzyCMeans, PredictMembership};
    pub use crate::gmm::{
        BayesianGaussianMixture, CovarianceType, GaussianMixture, ModelSelectionCriterion,
        ModelSelectionResult, PredictProba,
    };
    #[cfg(feature = "gpu")]
    pub use crate::gpu_distances::{GpuDistanceComputer, GpuDistanceMetric};
    pub use crate::graph_clustering::{
        Graph, GraphClusteringResult, LabelPropagationClustering, LouvainClustering, LouvainResult,
        ModularityClustering, SpectralGraphClustering,
    };
    pub use crate::hdbscan::{ClusterStat, HDBSCAN};
    pub use crate::hierarchical::{
        AgglomerativeClustering, Constraint, ConstraintSet, Dendrogram, DendrogramExport,
        DendrogramNode, MemoryStrategy,
    };
    pub use crate::incremental_dbscan::{
        DistanceMetric as IncrementalDistanceMetric, IncrementalDBSCAN,
    };
    pub use crate::kde_clustering::{BandwidthMethod, KDEClustering, KernelType};
    pub use crate::kmeans::{
        GMeans, InformationCriterion, KMeans, KMeansInit, MiniBatchKMeans, XMeans,
    };
    pub use crate::locality_sensitive_hashing::{LSHFamily, LSHIndex};
    pub use crate::lof::{DistanceMetric as LOFDistanceMetric, LOF};
    pub use crate::mean_shift::MeanShift;
    pub use crate::memory_mapped::{MemoryMappedDistanceMatrix, MemoryStats};
    pub use crate::multi_view::{
        ConsensusClustering, ConsensusMethod, MultiViewData, MultiViewKMeans, ViewWeighting,
        WeightLearning,
    };
    pub use crate::optics::{ClusterMethod, DistanceMetric as OpticsDistanceMetric, Optics};
    pub use crate::out_of_core::{ClusterSummary, OutOfCoreDataLoader, OutOfCoreKMeans};
    pub use crate::rock::{ROCKSimilarity, ROCK};
    pub use crate::semi_supervised::{
        ConstrainedKMeans, ConstraintHandling, ConstraintType, LabelPropagation,
    };
    pub use crate::sparse_matrix::{SparseDistanceMatrix, SparseNeighborhoodGraph};
    pub use crate::spectral::{Affinity, NormalizationMethod, SpectralClustering};
    pub use crate::streaming::{CluStream, MicroCluster, OnlineKMeans, SlidingWindowKMeans};
    pub use crate::text_clustering::{DocumentClustering, SphericalInit, SphericalKMeans};
    pub use crate::time_series::{
        CentroidAveraging, ChangeDetectionTest, DTWKMeans, RegimeChangeDetector, ShapeClustering,
        ShapeDistanceMetric, TemporalSegmentationClustering,
    };
    pub use crate::validation::{
        ClusteringValidator,
        ValidationMetric,
        // TODO: Fix these imports
        // AccuracyMetrics, ConsensusStabilityResult, BootstrapStabilityResult,
        // CrossValidationStabilityResult, NoiseStabilityResult, ParameterSensitivityResult,
        // PerturbationStabilityResult, StabilityResult, SubsampleStabilityResult,
    };

    #[cfg(feature = "parallel")]
    pub use crate::parallel::{SimpleParallelKMeans, SimpleParallelKMeansFitted};
    #[cfg(feature = "parallel")]
    pub use crate::parallel_hierarchical::{
        ParallelClusteringState, ParallelHierarchicalClustering,
    };
}

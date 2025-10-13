//! Types and configuration for Multi-View Discriminant Analysis

use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;

/// Configuration for Multi-View Discriminant Analysis
#[derive(Debug, Clone)]
pub struct MultiViewDiscriminantAnalysisConfig {
    /// Fusion strategy for combining multiple views
    pub fusion_strategy: FusionStrategy,
    /// Regularization parameter for view weighting
    pub view_regularization: Float,
    /// Number of components for each view
    pub n_components_per_view: Option<usize>,
    /// Total number of components for final representation
    pub n_components: Option<usize>,
    /// Whether to standardize each view independently
    pub standardize_views: bool,
    /// Tolerance for convergence
    pub tol: Float,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Enable heterogeneous feature integration
    pub enable_heterogeneous: bool,
    /// Global distance metric for heterogeneous features
    pub heterogeneous_distance: HeterogeneousDistance,
    /// Automatic feature type detection
    pub auto_detect_types: bool,
    /// Minimum samples for categorical detection
    pub categorical_threshold: usize,
    /// Sparsity threshold for sparse feature detection
    pub sparsity_threshold: Float,
}

impl Default for MultiViewDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            fusion_strategy: FusionStrategy::ConcatenationFusion,
            view_regularization: 0.01,
            n_components_per_view: None,
            n_components: None,
            standardize_views: true,
            tol: 1e-6,
            max_iter: 1000,
            random_state: None,
            enable_heterogeneous: false,
            heterogeneous_distance: HeterogeneousDistance::Euclidean,
            auto_detect_types: true,
            categorical_threshold: 10,
            sparsity_threshold: 0.1,
        }
    }
}

/// Fusion strategies for multi-view learning
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Early fusion: concatenate all views before discriminant analysis
    ConcatenationFusion,
    /// Late fusion: train separate discriminants and combine predictions
    LateFusion,
    /// Intermediate fusion: canonical correlation analysis then discriminant analysis
    CanonicalCorrelationFusion,
    /// Shared-specific factorization: decompose into shared and view-specific components
    SharedSpecificFusion { shared_ratio: Float },
    /// Multi-view kernel learning: learn optimal kernel combination
    KernelFusion {
        kernel_weights: Option<Array1<Float>>,
    },
}

/// Feature types for heterogeneous data
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureType {
    /// Continuous numerical features
    Continuous,
    /// Categorical features (ordinal or nominal)
    Categorical {
        categories: Vec<String>,

        is_ordinal: bool,
    },
    /// Binary features
    Binary,
    /// Text features (requires text processing)
    Text {
        max_features: Option<usize>,
        min_df: Float,
        max_df: Float,
    },
    /// Count/frequency features
    Count,
    /// Sparse features (mostly zeros)
    Sparse { sparsity_threshold: Float },
}

/// Heterogeneous feature preprocessing methods
#[derive(Debug, Clone)]
pub enum PreprocessingMethod {
    /// Standard scaling (mean=0, std=1)
    StandardScaling,
    /// Min-Max scaling (range [0,1])
    MinMaxScaling,
    /// Robust scaling (median and IQR)
    RobustScaling,
    /// One-hot encoding for categorical
    OneHotEncoding,
    /// Ordinal encoding for categorical
    OrdinalEncoding,
    /// TF-IDF for text features
    TfIdf,
    /// Count vectorization for text
    CountVectorization,
    /// Binary encoding for high-cardinality categorical
    BinaryEncoding,
    /// Target encoding for categorical
    TargetEncoding,
}

/// Heterogeneous distance metrics
#[derive(Debug, Clone)]
pub enum HeterogeneousDistance {
    /// Euclidean distance for continuous features
    Euclidean,
    /// Manhattan distance for continuous features
    Manhattan,
    /// Hamming distance for categorical features
    Hamming,
    /// Jaccard distance for binary/sparse features
    Jaccard,
    /// Cosine distance for text/count features
    Cosine,
    /// Gower distance for mixed types
    Gower,
    /// Custom weighted distance
    WeightedMixed {
        continuous_weight: Float,
        categorical_weight: Float,
        text_weight: Float,
    },
}

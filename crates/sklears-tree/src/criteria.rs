//! Split criteria and constraints for decision trees
//!
//! This module contains enums and structs that define how decision trees
//! make splitting decisions and handle constraints.

use scirs2_core::ndarray::Array2;
use sklears_core::types::Float;

// Import types from config module

/// Split criterion for decision trees
#[derive(Debug, Clone, Copy)]
pub enum SplitCriterion {
    /// Gini impurity (for classification)
    Gini,
    /// Information gain / entropy (for classification)
    Entropy,
    /// Mean squared error (for regression)
    MSE,
    /// Mean absolute error (for regression)
    MAE,
    /// Twoing criterion for binary splits (classification)
    Twoing,
    /// Log-loss criterion for probability-based splitting (classification)
    LogLoss,
    /// Chi-squared automatic interaction detection (CHAID)
    CHAID { significance_level: f64 },
    /// Conditional inference trees with statistical testing
    ConditionalInference {
        significance_level: f64,
        /// Type of statistical test to use
        test_type: ConditionalTestType,
    },
}

/// Statistical test types for conditional inference trees
#[derive(Debug, Clone, Copy)]
pub enum ConditionalTestType {
    QuadraticForm,
    MaxType,
    MonteCarlo {
        n_permutations: usize,
    },
    /// Asymptotic chi-squared test
    AsymptoticChiSquared,
}

/// Monotonic constraint for a feature
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonotonicConstraint {
    /// No constraint on the relationship
    None,
    /// Feature must have increasing relationship with target (positive monotonicity)
    Increasing,
    /// Feature must have decreasing relationship with target (negative monotonicity)
    Decreasing,
}

/// Interaction constraint between features
#[derive(Debug, Clone)]
pub enum InteractionConstraint {
    /// No constraints on feature interactions
    None,
    /// Allow interactions only within specified groups
    Groups(Vec<Vec<usize>>),
    /// Forbid specific feature pairs from interacting
    Forbidden(Vec<(usize, usize)>),
    /// Allow only specific feature pairs to interact
    Allowed(Vec<(usize, usize)>),
}

/// Feature grouping strategy for handling correlated features
#[derive(Debug, Clone)]
pub enum FeatureGrouping {
    /// No feature grouping (default)
    None,
    /// Automatic grouping based on correlation threshold
    AutoCorrelation {
        /// Correlation threshold above which features are grouped together
        threshold: Float,
        /// Method to select representative feature from each group
        selection_method: GroupSelectionMethod,
    },
    /// Manual feature groups specified by user
    Manual {
        /// List of feature groups, each group contains feature indices
        groups: Vec<Vec<usize>>,
        /// Method to select representative feature from each group
        selection_method: GroupSelectionMethod,
    },
    /// Hierarchical clustering-based grouping
    Hierarchical {
        /// Number of clusters to create
        n_clusters: usize,
        /// Linkage method for hierarchical clustering
        linkage: LinkageMethod,
        /// Method to select representative feature from each group
        selection_method: GroupSelectionMethod,
    },
}

/// Method for selecting representative feature from a group
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GroupSelectionMethod {
    /// Select feature with highest variance within the group
    MaxVariance,
    /// Select feature with highest correlation to target
    MaxTargetCorrelation,
    /// Select first feature in the group (index order)
    First,
    /// Select random feature from the group
    Random,
    /// Use all features from the group but with reduced weight
    WeightedAll,
}

/// Linkage method for hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkageMethod {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage
    Average,
    /// Ward linkage (minimize within-cluster variance)
    Ward,
}

/// Information about feature groups discovered or specified
#[derive(Debug, Clone)]
pub struct FeatureGroupInfo {
    /// Groups of correlated features
    pub groups: Vec<Vec<usize>>,
    /// Representative feature index for each group
    pub representatives: Vec<usize>,
    /// Correlation matrix used for grouping (if applicable)
    pub correlation_matrix: Option<Array2<Float>>,
    /// Within-group correlations for each group
    pub group_correlations: Vec<Float>,
}

/// Pruning strategy for decision trees
#[derive(Debug, Clone, Copy)]
pub enum PruningStrategy {
    /// No pruning
    None,
    /// Cost-complexity pruning (post-pruning)
    CostComplexity { alpha: f64 },
    /// Reduced error pruning
    ReducedError,
}

/// Missing value handling strategy
#[derive(Debug, Clone, Copy)]
pub enum MissingValueStrategy {
    /// Skip samples with missing values
    Skip,
    /// Use majority class/mean for splits
    Majority,
    /// Use surrogate splits
    Surrogate,
}

/// Feature type specification for multiway splits
#[derive(Debug, Clone)]
pub enum FeatureType {
    /// Continuous numerical feature (binary splits)
    Continuous,
    /// Categorical feature with specified categories (multiway splits)
    Categorical(Vec<String>),
}

/// Information about a multiway split
#[derive(Debug, Clone)]
pub struct MultiwaySplit {
    /// Feature index
    pub feature_idx: usize,
    /// Category assignments for each branch
    pub category_branches: Vec<Vec<String>>,
    /// Impurity decrease achieved by this split
    pub impurity_decrease: f64,
}

/// Tree growing strategy
#[derive(Debug, Clone, Copy)]
pub enum TreeGrowingStrategy {
    /// Depth-first growing (traditional CART)
    DepthFirst,
    /// Best-first growing (expand node with highest impurity decrease)
    BestFirst { max_leaves: Option<usize> },
}

/// Split type for decision trees
#[derive(Debug, Clone, Copy)]
pub enum SplitType {
    /// Traditional axis-aligned splits (threshold on single feature)
    AxisAligned,
    /// Linear hyperplane splits (linear combination of features)
    Oblique {
        /// Number of random hyperplanes to evaluate per split
        n_hyperplanes: usize,
        /// Use ridge regression to find optimal hyperplane
        use_ridge: bool,
    },
}

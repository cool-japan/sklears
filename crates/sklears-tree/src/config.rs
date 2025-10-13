//! Configuration types and enums for decision trees
//!
//! This module contains all the configuration enums, structs, and parameters
//! used by decision tree classifiers and regressors.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::Float;
use smartcore::linalg::basic::matrix::DenseMatrix;

// Import types from criteria module
use crate::criteria::SplitCriterion;

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

/// Strategy for selecting max features
#[derive(Debug, Clone)]
pub enum MaxFeatures {
    /// Use all features
    All,
    /// Use sqrt(n_features)
    Sqrt,
    /// Use log2(n_features)
    Log2,
    /// Use a specific number of features
    Number(usize),
    /// Use a fraction of features
    Fraction(f64),
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

/// Hyperplane split information for oblique trees
#[derive(Debug, Clone)]
pub struct HyperplaneSplit {
    /// Feature coefficients for the hyperplane (w^T x >= threshold)
    pub coefficients: Array1<f64>,
    /// Threshold for the hyperplane split
    pub threshold: f64,
    /// Bias term for the hyperplane
    pub bias: f64,
    /// Impurity decrease achieved by this split
    pub impurity_decrease: f64,
}

impl HyperplaneSplit {
    /// Evaluate the hyperplane split for a sample
    pub fn evaluate(&self, sample: &Array1<f64>) -> bool {
        let dot_product = self.coefficients.dot(sample) + self.bias;
        dot_product >= self.threshold
    }

    /// Create a random hyperplane with normalized coefficients
    pub fn random(n_features: usize, rng: &mut scirs2_core::CoreRandom) -> Self {
        let mut coefficients = Array1::zeros(n_features);
        for i in 0..n_features {
            coefficients[i] = rng.gen_range(-1.0..1.0);
        }

        // Normalize coefficients
        let dot_product: f64 = coefficients.dot(&coefficients);
        let norm = dot_product.sqrt();
        if norm > 1e-10_f64 {
            coefficients /= norm;
        }

        Self {
            coefficients,
            threshold: rng.gen_range(-1.0..1.0),
            bias: rng.gen_range(-0.1..0.1),
            impurity_decrease: 0.0,
        }
    }

    /// Find optimal hyperplane using ridge regression
    #[cfg(feature = "oblique")]
    pub fn from_ridge_regression(x: &Array2<f64>, y: &Array1<f64>, alpha: f64) -> Result<Self> {
        use scirs2_core::ndarray::s;
        use sklears_core::error::SklearsError;

        let n_features = x.ncols();
        if x.nrows() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for ridge regression".to_string(),
            ));
        }

        // Add bias column to X
        let mut x_bias = Array2::ones((x.nrows(), n_features + 1));
        x_bias.slice_mut(s![.., ..n_features]).assign(x);

        // Ridge regression: w = (X^T X + α I)^(-1) X^T y
        let xtx = x_bias.t().dot(&x_bias);
        let ridge_matrix = xtx + Array2::<f64>::eye(n_features + 1) * alpha;
        let xty = x_bias.t().dot(y);

        // Simple matrix inverse using Gauss-Jordan elimination
        match gauss_jordan_inverse(&ridge_matrix) {
            Ok(inv_matrix) => {
                let coefficients_full = inv_matrix.dot(&xty);

                let coefficients = coefficients_full.slice(s![..n_features]).to_owned();
                let bias = coefficients_full[n_features];

                Ok(Self {
                    coefficients,
                    threshold: 0.0, // Will be set during split evaluation
                    bias,
                    impurity_decrease: 0.0,
                })
            }
            Err(_) => {
                // Fallback to random hyperplane if matrix is singular
                let mut rng = scirs2_core::random::thread_rng();
                Ok(Self::random(n_features, &mut rng))
            }
        }
    }
}

/// Configuration for Decision Trees
#[derive(Debug, Clone)]
pub struct DecisionTreeConfig {
    /// Split criterion
    pub criterion: SplitCriterion,
    /// Maximum depth of the tree
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Maximum number of features to consider for splits
    pub max_features: MaxFeatures,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Minimum weighted fraction of samples required to be at a leaf
    pub min_weight_fraction_leaf: f64,
    /// Minimum impurity decrease required for a split
    pub min_impurity_decrease: f64,
    /// Pruning strategy to apply
    pub pruning: PruningStrategy,
    /// Strategy for handling missing values
    pub missing_values: MissingValueStrategy,
    /// Feature types for each feature (enables multiway splits for categorical features)
    pub feature_types: Option<Vec<FeatureType>>,
    /// Tree growing strategy
    pub growing_strategy: TreeGrowingStrategy,
    /// Split type (axis-aligned or oblique)
    pub split_type: SplitType,
    /// Monotonic constraints for each feature
    pub monotonic_constraints: Option<Vec<MonotonicConstraint>>,
    /// Interaction constraints between features
    pub interaction_constraints: InteractionConstraint,
    /// Feature grouping strategy for handling correlated features
    pub feature_grouping: FeatureGrouping,
}

impl Default for DecisionTreeConfig {
    fn default() -> Self {
        Self {
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::All,
            random_state: None,
            min_weight_fraction_leaf: 0.0,
            min_impurity_decrease: 0.0,
            pruning: PruningStrategy::None,
            missing_values: MissingValueStrategy::Skip,
            feature_types: None,
            growing_strategy: TreeGrowingStrategy::DepthFirst,
            split_type: SplitType::AxisAligned,
            monotonic_constraints: None,
            interaction_constraints: InteractionConstraint::None,
            feature_grouping: FeatureGrouping::None,
        }
    }
}

/// Helper function to convert ndarray to DenseMatrix
pub fn ndarray_to_dense_matrix(arr: &Array2<f64>) -> DenseMatrix<f64> {
    let _rows = arr.nrows();
    let _cols = arr.ncols();
    let mut data = Vec::new();
    for row in arr.outer_iter() {
        data.push(row.to_vec());
    }
    DenseMatrix::from_2d_vec(&data)
}

/// Simple Gauss-Jordan elimination for matrix inversion
#[cfg(feature = "oblique")]
fn gauss_jordan_inverse(matrix: &Array2<f64>) -> std::result::Result<Array2<f64>, &'static str> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err("Matrix must be square");
    }

    // Create augmented matrix [A | I]
    let mut augmented = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = matrix[[i, j]];
            if i == j {
                augmented[[i, j + n]] = 1.0;
            }
        }
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..n {
            if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..2 * n {
                let temp = augmented[[i, j]];
                augmented[[i, j]] = augmented[[max_row, j]];
                augmented[[max_row, j]] = temp;
            }
        }

        // Check for singular matrix
        if augmented[[i, i]].abs() < 1e-10 {
            return Err("Matrix is singular");
        }

        // Make diagonal element 1
        let pivot = augmented[[i, i]];
        for j in 0..2 * n {
            augmented[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = augmented[[k, i]];
                for j in 0..2 * n {
                    augmented[[k, j]] -= factor * augmented[[i, j]];
                }
            }
        }
    }

    // Extract inverse matrix
    let mut inverse = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inverse[[i, j]] = augmented[[i, j + n]];
        }
    }

    Ok(inverse)
}

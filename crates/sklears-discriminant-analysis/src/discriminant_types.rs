//! Common Types and Enumerations for Discriminant Analysis
//!
//! This module provides shared type definitions, enumerations, and configurations
//! used across all discriminant analysis algorithms (LDA, QDA, MDA).

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::SklearsError,
    types::Float,
};
use std::collections::HashMap;

/// Solver types for Linear Discriminant Analysis
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LdaSolver {
    /// Singular Value Decomposition (most stable)
    Svd,
    /// Linear System solver using LSQR
    Lsqr,
    /// Eigenvalue decomposition
    Eigen,
    /// Least squares solver
    LeastSquares,
}

impl Default for LdaSolver {
    fn default() -> Self {
        LdaSolver::Svd
    }
}

impl std::fmt::Display for LdaSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LdaSolver::Svd => write!(f, "svd"),
            LdaSolver::Lsqr => write!(f, "lsqr"),
            LdaSolver::Eigen => write!(f, "eigen"),
            LdaSolver::LeastSquares => write!(f, "lsqr"),
        }
    }
}

impl std::str::FromStr for LdaSolver {
    type Err = SklearsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "svd" => Ok(LdaSolver::Svd),
            "lsqr" => Ok(LdaSolver::Lsqr),
            "eigen" => Ok(LdaSolver::Eigen),
            "least_squares" => Ok(LdaSolver::LeastSquares),
            _ => Err(SklearsError::InvalidInput(format!("Unknown LDA solver: {}", s))),
        }
    }
}

/// Covariance estimation types for discriminant analysis
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CovarianceEstimationType {
    /// Full covariance matrix (most flexible)
    Full,
    /// Diagonal covariance matrix (faster, less parameters)
    Diagonal,
    /// Spherical covariance (isotropic, σ²I)
    Spherical,
    /// Tied covariance (same across classes/components)
    Tied,
    /// Shrinkage covariance (Ledoit-Wolf)
    Shrinkage,
    /// Robust covariance estimation
    Robust,
}

impl Default for CovarianceEstimationType {
    fn default() -> Self {
        CovarianceEstimationType::Full
    }
}

impl std::fmt::Display for CovarianceEstimationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CovarianceEstimationType::Full => write!(f, "full"),
            CovarianceEstimationType::Diagonal => write!(f, "diagonal"),
            CovarianceEstimationType::Spherical => write!(f, "spherical"),
            CovarianceEstimationType::Tied => write!(f, "tied"),
            CovarianceEstimationType::Shrinkage => write!(f, "shrinkage"),
            CovarianceEstimationType::Robust => write!(f, "robust"),
        }
    }
}

/// Robust estimation methods
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RobustEstimationMethod {
    /// Minimum Covariance Determinant
    Mcd,
    /// Minimum Volume Ellipsoid
    Mve,
    /// One-Class SVM
    OneClassSvm,
    /// Isolation Forest
    IsolationForest,
    /// Outlier detection via Local Outlier Factor
    Lof,
}

impl Default for RobustEstimationMethod {
    fn default() -> Self {
        RobustEstimationMethod::Mcd
    }
}

impl std::fmt::Display for RobustEstimationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RobustEstimationMethod::Mcd => write!(f, "mcd"),
            RobustEstimationMethod::Mve => write!(f, "mve"),
            RobustEstimationMethod::OneClassSvm => write!(f, "one_class_svm"),
            RobustEstimationMethod::IsolationForest => write!(f, "isolation_forest"),
            RobustEstimationMethod::Lof => write!(f, "lof"),
        }
    }
}

/// Initialization strategies for mixture components
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InitializationStrategy {
    /// Random initialization
    Random,
    /// K-means++ initialization (most robust)
    KMeansPlusPlus,
    /// Uniform partitioning
    Uniform,
    /// Manual initialization with user-provided parameters
    Manual,
    /// Agglomerative clustering-based initialization
    Agglomerative,
    /// Principal Component Analysis-based initialization
    Pca,
}

impl Default for InitializationStrategy {
    fn default() -> Self {
        InitializationStrategy::KMeansPlusPlus
    }
}

/// Convergence criteria for iterative algorithms
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConvergenceCriterion {
    /// Convergence based on log-likelihood change
    LogLikelihood,
    /// Convergence based on parameter change
    Parameters,
    /// Convergence based on both criteria
    Both,
    /// Convergence based on objective function
    Objective,
}

impl Default for ConvergenceCriterion {
    fn default() -> Self {
        ConvergenceCriterion::LogLikelihood
    }
}

/// Shrinkage estimation methods
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShrinkageMethod {
    /// Ledoit-Wolf shrinkage
    LedoitWolf,
    /// Oracle Approximating Shrinkage
    Oas,
    /// Empirical Bayes
    EmpiricalBayes,
    /// Cross-validation based
    CrossValidation,
    /// Manual shrinkage parameter
    Manual(Float),
}

impl Default for ShrinkageMethod {
    fn default() -> Self {
        ShrinkageMethod::LedoitWolf
    }
}

/// Regularization types
#[derive(Debug, Clone, PartialEq)]
pub enum RegularizationType {
    /// L1 regularization (Lasso)
    L1(Float),
    /// L2 regularization (Ridge)
    L2(Float),
    /// Elastic Net (combination of L1 and L2)
    ElasticNet { alpha: Float, l1_ratio: Float },
    /// Diagonal regularization
    Diagonal(Float),
    /// None (no regularization)
    None,
}

impl Default for RegularizationType {
    fn default() -> Self {
        RegularizationType::None
    }
}

/// Prediction output types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PredictionType {
    /// Class predictions
    Classes,
    /// Class probabilities
    Probabilities,
    /// Log probabilities
    LogProbabilities,
    /// Decision function values
    DecisionFunction,
    /// Transformed features (for LDA)
    Transform,
}

/// Cross-validation strategy
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CrossValidationStrategy {
    /// K-fold cross-validation
    KFold(usize),
    /// Stratified K-fold
    StratifiedKFold(usize),
    /// Leave-one-out
    LeaveOneOut,
    /// Leave-p-out
    LeavePOut(usize),
    /// Time series split
    TimeSeriesSplit(usize),
}

impl Default for CrossValidationStrategy {
    fn default() -> Self {
        CrossValidationStrategy::StratifiedKFold(5)
    }
}

/// Feature selection methods for dimensionality reduction
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FeatureSelectionMethod {
    /// Select k best features
    SelectKBest(usize),
    /// Select percentile of features
    SelectPercentile(Float),
    /// Recursive feature elimination
    Rfe,
    /// Univariate feature selection
    Univariate,
    /// Principal component analysis
    Pca,
    /// Independent component analysis
    Ica,
}

/// Distance metrics for nearest neighbor computations
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Mahalanobis distance
    Mahalanobis,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
    /// Minkowski distance with parameter p
    Minkowski(Float),
    /// Custom distance function
    Custom,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        DistanceMetric::Euclidean
    }
}

/// Common configuration for all discriminant analysis methods
#[derive(Debug, Clone)]
pub struct DiscriminantConfig {
    /// Whether to include bias/intercept term
    pub fit_intercept: bool,
    /// Random seed for reproducible results
    pub random_state: Option<u64>,
    /// Verbose output level
    pub verbose: bool,
    /// Number of CPU cores to use (-1 for all)
    pub n_jobs: Option<i32>,
    /// Whether to copy input data
    pub copy: bool,
    /// Tolerance for numerical computations
    pub tolerance: Float,
    /// Maximum number of iterations
    pub max_iterations: usize,
}

impl Default for DiscriminantConfig {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            random_state: None,
            verbose: false,
            n_jobs: None,
            copy: true,
            tolerance: 1e-6,
            max_iterations: 1000,
        }
    }
}

/// Result structure for discriminant analysis predictions
#[derive(Debug, Clone)]
pub struct DiscriminantPredictionResult {
    /// Predicted class labels
    pub predictions: Array1<i32>,
    /// Prediction probabilities (if available)
    pub probabilities: Option<Array2<Float>>,
    /// Log probabilities (if available)
    pub log_probabilities: Option<Array2<Float>>,
    /// Decision function values
    pub decision_function: Option<Array2<Float>>,
    /// Confidence scores
    pub confidence: Option<Array1<Float>>,
}

/// Result structure for discriminant analysis transformation
#[derive(Debug, Clone)]
pub struct DiscriminantTransformResult {
    /// Transformed features
    pub transformed_features: Array2<Float>,
    /// Explained variance ratio (for LDA)
    pub explained_variance_ratio: Option<Array1<Float>>,
    /// Singular values (for LDA)
    pub singular_values: Option<Array1<Float>>,
    /// Number of components used
    pub n_components: usize,
}

/// Performance metrics for discriminant analysis
#[derive(Debug, Clone)]
pub struct DiscriminantMetrics {
    /// Classification accuracy
    pub accuracy: Float,
    /// Precision per class
    pub precision: Array1<Float>,
    /// Recall per class
    pub recall: Array1<Float>,
    /// F1-score per class
    pub f1_score: Array1<Float>,
    /// Confusion matrix
    pub confusion_matrix: Array2<i32>,
    /// Log-likelihood
    pub log_likelihood: Option<Float>,
    /// AIC (Akaike Information Criterion)
    pub aic: Option<Float>,
    /// BIC (Bayesian Information Criterion)
    pub bic: Option<Float>,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Cross-validation scores
    pub scores: Array1<Float>,
    /// Mean score
    pub mean_score: Float,
    /// Standard deviation of scores
    pub std_score: Float,
    /// Best parameters (if applicable)
    pub best_params: Option<HashMap<String, String>>,
}

/// Feature importance scores
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Feature importance scores
    pub scores: Array1<Float>,
    /// Feature names (if available)
    pub feature_names: Option<Vec<String>>,
    /// Ranking of features by importance
    pub ranking: Array1<usize>,
}

/// Model validation results
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Training score
    pub train_score: Float,
    /// Validation score
    pub validation_score: Float,
    /// Test score (if available)
    pub test_score: Option<Float>,
    /// Overfitting indicator
    pub is_overfitting: bool,
    /// Underfitting indicator
    pub is_underfitting: bool,
}

/// Hyperparameter search space
#[derive(Debug, Clone)]
pub struct HyperparameterSpace {
    /// Parameter name
    pub name: String,
    /// Parameter values to search
    pub values: Vec<String>,
    /// Search type (grid, random, bayesian)
    pub search_type: String,
}

/// Model selection results
#[derive(Debug, Clone)]
pub struct ModelSelectionResult {
    /// Best hyperparameters
    pub best_params: HashMap<String, String>,
    /// Best score achieved
    pub best_score: Float,
    /// All parameter combinations tried
    pub param_grid: Vec<HashMap<String, String>>,
    /// Corresponding scores
    pub scores: Array1<Float>,
}

/// Error types specific to discriminant analysis
#[derive(Debug, Clone)]
pub enum DiscriminantError {
    /// Singular covariance matrix
    SingularCovariance(String),
    /// Insufficient samples for estimation
    InsufficientSamples { required: usize, provided: usize },
    /// Invalid number of components
    InvalidComponents { n_components: usize, max_components: usize },
    /// Convergence failure
    ConvergenceFailure { iterations: usize, tolerance: Float },
    /// Numerical instability
    NumericalInstability(String),
    /// Invalid solver for the given problem
    InvalidSolver { solver: String, reason: String },
    /// Dimensionality mismatch
    DimensionalityMismatch { expected: usize, actual: usize },
    /// Invalid regularization parameter
    InvalidRegularization(Float),
}

impl std::fmt::Display for DiscriminantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiscriminantError::SingularCovariance(msg) => {
                write!(f, "Singular covariance matrix: {}", msg)
            },
            DiscriminantError::InsufficientSamples { required, provided } => {
                write!(f, "Insufficient samples: required {}, provided {}", required, provided)
            },
            DiscriminantError::InvalidComponents { n_components, max_components } => {
                write!(f, "Invalid number of components: {} (max: {})", n_components, max_components)
            },
            DiscriminantError::ConvergenceFailure { iterations, tolerance } => {
                write!(f, "Failed to converge after {} iterations (tolerance: {})", iterations, tolerance)
            },
            DiscriminantError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            },
            DiscriminantError::InvalidSolver { solver, reason } => {
                write!(f, "Invalid solver '{}': {}", solver, reason)
            },
            DiscriminantError::DimensionalityMismatch { expected, actual } => {
                write!(f, "Dimensionality mismatch: expected {}, actual {}", expected, actual)
            },
            DiscriminantError::InvalidRegularization(value) => {
                write!(f, "Invalid regularization parameter: {}", value)
            },
        }
    }
}

impl std::error::Error for DiscriminantError {}

/// Constants used throughout discriminant analysis
pub mod constants {
    use super::Float;

    /// Minimum eigenvalue threshold
    pub const MIN_EIGENVALUE: Float = 1e-12;

    /// Maximum condition number for matrix inversion
    pub const MAX_CONDITION_NUMBER: Float = 1e12;

    /// Default regularization parameter
    pub const DEFAULT_REG_PARAM: Float = 1e-6;

    /// Default convergence tolerance
    pub const DEFAULT_TOLERANCE: Float = 1e-6;

    /// Default maximum iterations
    pub const DEFAULT_MAX_ITER: usize = 1000;

    /// Minimum weight for mixture components
    pub const MIN_COMPONENT_WEIGHT: Float = 1e-4;

    /// Maximum number of mixture components per class
    pub const MAX_COMPONENTS_PER_CLASS: usize = 100;

    /// Default contamination rate for robust estimation
    pub const DEFAULT_CONTAMINATION: Float = 0.1;

    /// Numerical precision for floating point comparisons
    pub const NUMERICAL_PRECISION: Float = 1e-12;
}

/// Utility functions for type conversions and validations
pub mod utils {
    use super::*;
    use sklears_core::error::Result as SklResult;

    /// Convert string to LDA solver
    pub fn string_to_lda_solver(s: &str) -> SklResult<LdaSolver> {
        s.parse()
    }

    /// Convert covariance type string to enum
    pub fn string_to_covariance_type(s: &str) -> SklResult<CovarianceEstimationType> {
        match s.to_lowercase().as_str() {
            "full" => Ok(CovarianceEstimationType::Full),
            "diagonal" | "diag" => Ok(CovarianceEstimationType::Diagonal),
            "spherical" | "sphere" => Ok(CovarianceEstimationType::Spherical),
            "tied" => Ok(CovarianceEstimationType::Tied),
            "shrinkage" => Ok(CovarianceEstimationType::Shrinkage),
            "robust" => Ok(CovarianceEstimationType::Robust),
            _ => Err(SklearsError::InvalidInput(format!("Unknown covariance type: {}", s))),
        }
    }

    /// Validate number of components
    pub fn validate_n_components(n_components: usize, n_classes: usize, n_features: usize) -> SklResult<()> {
        if n_components == 0 {
            return Err(SklearsError::InvalidInput("Number of components must be positive".to_string()));
        }

        let max_components = (n_classes - 1).min(n_features);
        if n_components > max_components {
            return Err(SklearsError::InvalidInput(format!(
                "Number of components ({}) exceeds maximum possible ({})",
                n_components, max_components
            )));
        }

        Ok(())
    }

    /// Validate regularization parameter
    pub fn validate_regularization(reg_param: Float) -> SklResult<()> {
        if reg_param < 0.0 {
            return Err(SklearsError::InvalidInput("Regularization parameter must be non-negative".to_string()));
        }
        Ok(())
    }

    /// Validate tolerance parameter
    pub fn validate_tolerance(tolerance: Float) -> SklResult<()> {
        if tolerance <= 0.0 {
            return Err(SklearsError::InvalidInput("Tolerance must be positive".to_string()));
        }
        Ok(())
    }

    /// Check if covariance matrix is valid
    pub fn validate_covariance_matrix(cov: &Array2<Float>) -> SklResult<()> {
        if cov.nrows() != cov.ncols() {
            return Err(SklearsError::InvalidInput("Covariance matrix must be square".to_string()));
        }

        // Check if matrix is positive definite (simplified check)
        for i in 0..cov.nrows() {
            if cov[[i, i]] <= 0.0 {
                return Err(SklearsError::InvalidInput("Covariance matrix must be positive definite".to_string()));
            }
        }

        Ok(())
    }

    /// Check for numerical stability
    pub fn check_numerical_stability(values: &[Float]) -> SklResult<()> {
        for &value in values {
            if !value.is_finite() {
                return Err(SklearsError::NumericalError("Non-finite value encountered".to_string()));
            }
        }
        Ok(())
    }

    /// Compute condition number approximation
    pub fn estimate_condition_number(matrix: &Array2<Float>) -> Float {
        let diag = matrix.diag();
        let max_val = diag.iter().fold(0.0, |a, &b| a.max(b.abs()));
        let min_val = diag.iter().fold(Float::INFINITY, |a, &b| a.min(b.abs()));

        if min_val > constants::NUMERICAL_PRECISION {
            max_val / min_val
        } else {
            Float::INFINITY
        }
    }
}

/// Trait for discriminant analysis algorithms
pub trait DiscriminantAnalysis {
    /// Get the number of classes
    fn n_classes(&self) -> usize;

    /// Get the number of features
    fn n_features(&self) -> usize;

    /// Get class labels
    fn classes(&self) -> &Array1<i32>;

    /// Check if the model is trained
    fn is_fitted(&self) -> bool;

    /// Get model complexity (number of parameters)
    fn model_complexity(&self) -> usize;
}

/// Trait for models that support cross-validation
pub trait CrossValidate {
    /// Perform cross-validation
    fn cross_validate(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<i32>,
        cv: CrossValidationStrategy,
    ) -> SklResult<CrossValidationResult>;
}

/// Trait for models that support feature importance
pub trait FeatureImportanceProvider {
    /// Get feature importance scores
    fn feature_importance(&self) -> SklResult<FeatureImportance>;
}

/// Trait for hyperparameter tuning
pub trait HyperparameterTuning {
    /// Perform grid search
    fn grid_search(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<i32>,
        param_grid: &[HyperparameterSpace],
        cv: CrossValidationStrategy,
    ) -> SklResult<ModelSelectionResult>;
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_lda_solver_conversion() {
        assert_eq!("svd".parse::<LdaSolver>().unwrap(), LdaSolver::Svd);
        assert_eq!("lsqr".parse::<LdaSolver>().unwrap(), LdaSolver::Lsqr);
        assert_eq!("eigen".parse::<LdaSolver>().unwrap(), LdaSolver::Eigen);

        assert!("invalid".parse::<LdaSolver>().is_err());
    }

    #[test]
    fn test_covariance_type_conversion() {
        assert_eq!(
            utils::string_to_covariance_type("full").unwrap(),
            CovarianceEstimationType::Full
        );
        assert_eq!(
            utils::string_to_covariance_type("diagonal").unwrap(),
            CovarianceEstimationType::Diagonal
        );

        assert!(utils::string_to_covariance_type("invalid").is_err());
    }

    #[test]
    fn test_validation_functions() {
        // Test component validation
        assert!(utils::validate_n_components(2, 3, 5).is_ok());
        assert!(utils::validate_n_components(0, 3, 5).is_err());
        assert!(utils::validate_n_components(10, 3, 5).is_err());

        // Test regularization validation
        assert!(utils::validate_regularization(0.1).is_ok());
        assert!(utils::validate_regularization(0.0).is_ok());
        assert!(utils::validate_regularization(-0.1).is_err());

        // Test tolerance validation
        assert!(utils::validate_tolerance(1e-6).is_ok());
        assert!(utils::validate_tolerance(0.0).is_err());
        assert!(utils::validate_tolerance(-1e-6).is_err());
    }

    #[test]
    fn test_covariance_matrix_validation() {
        let valid_cov = array![[1.0, 0.5], [0.5, 1.0]];
        assert!(utils::validate_covariance_matrix(&valid_cov).is_ok());

        let invalid_cov = array![[0.0, 0.5], [0.5, 1.0]];
        assert!(utils::validate_covariance_matrix(&invalid_cov).is_err());

        let non_square = array![[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]];
        assert!(utils::validate_covariance_matrix(&non_square).is_err());
    }

    #[test]
    fn test_numerical_stability_check() {
        let stable_values = vec![1.0, 2.0, 3.0, -1.0];
        assert!(utils::check_numerical_stability(&stable_values).is_ok());

        let unstable_values = vec![1.0, Float::NAN, 3.0];
        assert!(utils::check_numerical_stability(&unstable_values).is_err());

        let infinite_values = vec![1.0, Float::INFINITY, 3.0];
        assert!(utils::check_numerical_stability(&infinite_values).is_err());
    }

    #[test]
    fn test_condition_number_estimation() {
        let well_conditioned = array![[2.0, 0.0], [0.0, 2.0]];
        let cond_num = utils::estimate_condition_number(&well_conditioned);
        assert!((cond_num - 1.0).abs() < 1e-10);

        let ill_conditioned = array![[1.0, 0.0], [0.0, 1e-10]];
        let cond_num_ill = utils::estimate_condition_number(&ill_conditioned);
        assert!(cond_num_ill > 1e8);
    }

    #[test]
    fn test_discriminant_error_display() {
        let error = DiscriminantError::InsufficientSamples { required: 10, provided: 5 };
        let error_str = format!("{}", error);
        assert!(error_str.contains("Insufficient samples"));
        assert!(error_str.contains("10"));
        assert!(error_str.contains("5"));
    }

    #[test]
    fn test_default_implementations() {
        let config = DiscriminantConfig::default();
        assert_eq!(config.fit_intercept, true);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.max_iterations, 1000);

        let solver = LdaSolver::default();
        assert_eq!(solver, LdaSolver::Svd);

        let cov_type = CovarianceEstimationType::default();
        assert_eq!(cov_type, CovarianceEstimationType::Full);
    }
}
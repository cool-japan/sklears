//! Phantom Types for Discriminant Analysis
//!
//! This module provides phantom types for compile-time type safety in discriminant analysis,
//! ensuring correct usage patterns and preventing common errors through the type system.
//! Implements zero-cost abstractions following Rust best practices.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::{
    lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig},
    qda::{QuadraticDiscriminantAnalysis, QuadraticDiscriminantAnalysisConfig},
};

use sklears_core::{
    error::Result,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Phantom type markers for discriminant analysis methods
pub mod discriminant_markers {
    /// Linear Discriminant Analysis marker
    pub struct LinearDiscriminant;

    /// Quadratic Discriminant Analysis marker
    pub struct QuadraticDiscriminant;

    /// Regularized Discriminant Analysis marker
    pub struct RegularizedDiscriminant;

    /// Sparse Discriminant Analysis marker
    pub struct SparseDiscriminant;

    /// Robust Discriminant Analysis marker
    pub struct RobustDiscriminant;

    /// Kernel Discriminant Analysis marker
    pub struct KernelDiscriminant;

    /// Neural Discriminant Analysis marker
    pub struct NeuralDiscriminant;
}

/// State markers for discriminant analysis lifecycle
pub mod state_markers {
    /// Untrained state - model has been created but not fitted
    pub struct Untrained;

    /// Trained state - model has been fitted to training data
    pub struct Trained;

    /// Validated state - model has been cross-validated
    pub struct Validated;

    /// Optimized state - hyperparameters have been optimized
    pub struct Optimized;
}

/// Solver type markers for compile-time solver validation
pub mod solver_markers {
    /// SVD-based solver
    pub struct SVDSolver;

    /// Eigenvalue decomposition solver
    pub struct EigenSolver;

    /// LDLT decomposition solver
    pub struct LDLTSolver;

    /// Iterative solver for large problems
    pub struct IterativeSolver;

    /// GPU-accelerated solver
    pub struct GPUSolver;
}

/// Regularization type markers
pub mod regularization_markers {
    /// No regularization
    pub struct NoRegularization;

    /// L1 (Lasso) regularization
    pub struct L1Regularization;

    /// L2 (Ridge) regularization
    pub struct L2Regularization;

    /// Elastic Net (L1 + L2) regularization
    pub struct ElasticNetRegularization;

    /// Shrinkage regularization
    pub struct ShrinkageRegularization;

    /// Adaptive regularization
    pub struct AdaptiveRegularization;
}

/// Data type markers for compile-time validation
pub mod data_markers {
    /// Dense matrix data
    pub struct DenseData;

    /// Sparse matrix data
    pub struct SparseData;

    /// Streaming data
    pub struct StreamingData;

    /// GPU data
    pub struct GPUData;

    /// Memory-mapped data
    pub struct MemoryMappedData;
}

/// Type-safe discriminant analysis with phantom types
#[derive(Debug)]
pub struct TypeSafeDiscriminantAnalysis<Method, State, Solver, Regularization, Data> {
    _method: PhantomData<Method>,
    _state: PhantomData<State>,
    _solver: PhantomData<Solver>,
    _regularization: PhantomData<Regularization>,
    _data: PhantomData<Data>,
}

/// Type alias for untrained linear discriminant analysis
pub type UntrainedLinearDA<Solver, Regularization, Data> = TypeSafeDiscriminantAnalysis<
    discriminant_markers::LinearDiscriminant,
    state_markers::Untrained,
    Solver,
    Regularization,
    Data,
>;

/// Type alias for trained linear discriminant analysis
pub type TrainedLinearDA<Solver, Regularization, Data> = TypeSafeDiscriminantAnalysis<
    discriminant_markers::LinearDiscriminant,
    state_markers::Trained,
    Solver,
    Regularization,
    Data,
>;

/// Type alias for untrained quadratic discriminant analysis
pub type UntrainedQuadraticDA<Solver, Regularization, Data> = TypeSafeDiscriminantAnalysis<
    discriminant_markers::QuadraticDiscriminant,
    state_markers::Untrained,
    Solver,
    Regularization,
    Data,
>;

/// Type alias for trained quadratic discriminant analysis
pub type TrainedQuadraticDA<Solver, Regularization, Data> = TypeSafeDiscriminantAnalysis<
    discriminant_markers::QuadraticDiscriminant,
    state_markers::Trained,
    Solver,
    Regularization,
    Data,
>;

/// Trait for compile-time method validation
pub trait DiscriminantMethod {
    /// Check if method supports dimensionality reduction
    const SUPPORTS_TRANSFORM: bool;

    /// Check if method supports probability predictions
    const SUPPORTS_PREDICT_PROBA: bool;

    /// Check if method supports class-specific covariances
    const SUPPORTS_CLASS_COVARIANCES: bool;

    /// Check if method supports regularization
    const SUPPORTS_REGULARIZATION: bool;

    /// Method name for debugging
    const METHOD_NAME: &'static str;
}

impl DiscriminantMethod for discriminant_markers::LinearDiscriminant {
    const SUPPORTS_TRANSFORM: bool = true;
    const SUPPORTS_PREDICT_PROBA: bool = true;
    const SUPPORTS_CLASS_COVARIANCES: bool = false; // Uses pooled covariance
    const SUPPORTS_REGULARIZATION: bool = true;
    const METHOD_NAME: &'static str = "Linear Discriminant Analysis";
}

impl DiscriminantMethod for discriminant_markers::QuadraticDiscriminant {
    const SUPPORTS_TRANSFORM: bool = false; // QDA doesn't reduce dimensionality
    const SUPPORTS_PREDICT_PROBA: bool = true;
    const SUPPORTS_CLASS_COVARIANCES: bool = true;
    const SUPPORTS_REGULARIZATION: bool = true;
    const METHOD_NAME: &'static str = "Quadratic Discriminant Analysis";
}

impl DiscriminantMethod for discriminant_markers::RegularizedDiscriminant {
    const SUPPORTS_TRANSFORM: bool = true;
    const SUPPORTS_PREDICT_PROBA: bool = true;
    const SUPPORTS_CLASS_COVARIANCES: bool = true;
    const SUPPORTS_REGULARIZATION: bool = true;
    const METHOD_NAME: &'static str = "Regularized Discriminant Analysis";
}

impl DiscriminantMethod for discriminant_markers::SparseDiscriminant {
    const SUPPORTS_TRANSFORM: bool = true;
    const SUPPORTS_PREDICT_PROBA: bool = true;
    const SUPPORTS_CLASS_COVARIANCES: bool = false;
    const SUPPORTS_REGULARIZATION: bool = true;
    const METHOD_NAME: &'static str = "Sparse Discriminant Analysis";
}

impl DiscriminantMethod for discriminant_markers::KernelDiscriminant {
    const SUPPORTS_TRANSFORM: bool = true;
    const SUPPORTS_PREDICT_PROBA: bool = true;
    const SUPPORTS_CLASS_COVARIANCES: bool = false;
    const SUPPORTS_REGULARIZATION: bool = true;
    const METHOD_NAME: &'static str = "Kernel Discriminant Analysis";
}

/// Trait for compile-time solver validation
pub trait SolverType {
    /// Check if solver supports sparse matrices
    const SUPPORTS_SPARSE: bool;

    /// Check if solver supports GPU acceleration
    const SUPPORTS_GPU: bool;

    /// Check if solver is numerically stable
    const IS_NUMERICALLY_STABLE: bool;

    /// Recommended minimum problem size for this solver
    const MIN_PROBLEM_SIZE: usize;

    /// Solver name for debugging
    const SOLVER_NAME: &'static str;
}

impl SolverType for solver_markers::SVDSolver {
    const SUPPORTS_SPARSE: bool = false;
    const SUPPORTS_GPU: bool = true;
    const IS_NUMERICALLY_STABLE: bool = true;
    const MIN_PROBLEM_SIZE: usize = 100;
    const SOLVER_NAME: &'static str = "SVD Solver";
}

impl SolverType for solver_markers::EigenSolver {
    const SUPPORTS_SPARSE: bool = true;
    const SUPPORTS_GPU: bool = true;
    const IS_NUMERICALLY_STABLE: bool = true;
    const MIN_PROBLEM_SIZE: usize = 50;
    const SOLVER_NAME: &'static str = "Eigenvalue Solver";
}

impl SolverType for solver_markers::LDLTSolver {
    const SUPPORTS_SPARSE: bool = true;
    const SUPPORTS_GPU: bool = false;
    const IS_NUMERICALLY_STABLE: bool = true;
    const MIN_PROBLEM_SIZE: usize = 10;
    const SOLVER_NAME: &'static str = "LDLT Solver";
}

impl SolverType for solver_markers::IterativeSolver {
    const SUPPORTS_SPARSE: bool = true;
    const SUPPORTS_GPU: bool = true;
    const IS_NUMERICALLY_STABLE: bool = false; // Depends on conditioning
    const MIN_PROBLEM_SIZE: usize = 1000;
    const SOLVER_NAME: &'static str = "Iterative Solver";
}

impl SolverType for solver_markers::GPUSolver {
    const SUPPORTS_SPARSE: bool = false;
    const SUPPORTS_GPU: bool = true;
    const IS_NUMERICALLY_STABLE: bool = true;
    const MIN_PROBLEM_SIZE: usize = 500;
    const SOLVER_NAME: &'static str = "GPU Solver";
}

/// Trait for compile-time regularization validation
pub trait RegularizationType {
    /// Check if regularization supports automatic parameter selection
    const SUPPORTS_AUTO_PARAMETER: bool;

    /// Check if regularization produces sparse solutions
    const PRODUCES_SPARSE: bool;

    /// Check if regularization requires cross-validation
    const REQUIRES_CV: bool;

    /// Regularization name for debugging
    const REGULARIZATION_NAME: &'static str;
}

impl RegularizationType for regularization_markers::NoRegularization {
    const SUPPORTS_AUTO_PARAMETER: bool = false;
    const PRODUCES_SPARSE: bool = false;
    const REQUIRES_CV: bool = false;
    const REGULARIZATION_NAME: &'static str = "No Regularization";
}

impl RegularizationType for regularization_markers::L1Regularization {
    const SUPPORTS_AUTO_PARAMETER: bool = true;
    const PRODUCES_SPARSE: bool = true;
    const REQUIRES_CV: bool = true;
    const REGULARIZATION_NAME: &'static str = "L1 Regularization";
}

impl RegularizationType for regularization_markers::L2Regularization {
    const SUPPORTS_AUTO_PARAMETER: bool = true;
    const PRODUCES_SPARSE: bool = false;
    const REQUIRES_CV: bool = true;
    const REGULARIZATION_NAME: &'static str = "L2 Regularization";
}

impl RegularizationType for regularization_markers::ElasticNetRegularization {
    const SUPPORTS_AUTO_PARAMETER: bool = true;
    const PRODUCES_SPARSE: bool = true;
    const REQUIRES_CV: bool = true;
    const REGULARIZATION_NAME: &'static str = "Elastic Net Regularization";
}

impl RegularizationType for regularization_markers::ShrinkageRegularization {
    const SUPPORTS_AUTO_PARAMETER: bool = true;
    const PRODUCES_SPARSE: bool = false;
    const REQUIRES_CV: bool = false;
    const REGULARIZATION_NAME: &'static str = "Shrinkage Regularization";
}

/// Trait for compile-time data type validation
pub trait DataType {
    /// Check if data type supports streaming operations
    const SUPPORTS_STREAMING: bool;

    /// Check if data type supports GPU operations
    const SUPPORTS_GPU: bool;

    /// Check if data type is memory efficient
    const IS_MEMORY_EFFICIENT: bool;

    /// Data type name for debugging
    const DATA_TYPE_NAME: &'static str;
}

impl DataType for data_markers::DenseData {
    const SUPPORTS_STREAMING: bool = false;
    const SUPPORTS_GPU: bool = true;
    const IS_MEMORY_EFFICIENT: bool = false;
    const DATA_TYPE_NAME: &'static str = "Dense Data";
}

impl DataType for data_markers::SparseData {
    const SUPPORTS_STREAMING: bool = true;
    const SUPPORTS_GPU: bool = false; // Limited support
    const IS_MEMORY_EFFICIENT: bool = true;
    const DATA_TYPE_NAME: &'static str = "Sparse Data";
}

impl DataType for data_markers::StreamingData {
    const SUPPORTS_STREAMING: bool = true;
    const SUPPORTS_GPU: bool = false;
    const IS_MEMORY_EFFICIENT: bool = true;
    const DATA_TYPE_NAME: &'static str = "Streaming Data";
}

impl DataType for data_markers::GPUData {
    const SUPPORTS_STREAMING: bool = false;
    const SUPPORTS_GPU: bool = true;
    const IS_MEMORY_EFFICIENT: bool = false;
    const DATA_TYPE_NAME: &'static str = "GPU Data";
}

/// Builder pattern with compile-time validation
pub struct DiscriminantAnalysisBuilder<Method, Solver, Regularization, Data> {
    _method: PhantomData<Method>,
    _solver: PhantomData<Solver>,
    _regularization: PhantomData<Regularization>,
    _data: PhantomData<Data>,
}

impl DiscriminantAnalysisBuilder<(), (), (), ()> {
    /// Start building a discriminant analysis with type safety
    pub fn new() -> Self {
        Self {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }
}

impl<Solver, Regularization, Data> DiscriminantAnalysisBuilder<(), Solver, Regularization, Data> {
    /// Set method to Linear Discriminant Analysis
    pub fn linear(
        self,
    ) -> DiscriminantAnalysisBuilder<
        discriminant_markers::LinearDiscriminant,
        Solver,
        Regularization,
        Data,
    > {
        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }

    /// Set method to Quadratic Discriminant Analysis
    pub fn quadratic(
        self,
    ) -> DiscriminantAnalysisBuilder<
        discriminant_markers::QuadraticDiscriminant,
        Solver,
        Regularization,
        Data,
    > {
        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }

    /// Set method to Sparse Discriminant Analysis
    pub fn sparse(
        self,
    ) -> DiscriminantAnalysisBuilder<
        discriminant_markers::SparseDiscriminant,
        Solver,
        Regularization,
        Data,
    > {
        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }
}

impl<Method, Regularization, Data> DiscriminantAnalysisBuilder<Method, (), Regularization, Data> {
    /// Set solver to SVD
    pub fn svd_solver(
        self,
    ) -> DiscriminantAnalysisBuilder<Method, solver_markers::SVDSolver, Regularization, Data> {
        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }

    /// Set solver to Eigenvalue decomposition
    pub fn eigen_solver(
        self,
    ) -> DiscriminantAnalysisBuilder<Method, solver_markers::EigenSolver, Regularization, Data>
    {
        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }

    /// Set solver to GPU acceleration
    pub fn gpu_solver(
        self,
    ) -> DiscriminantAnalysisBuilder<Method, solver_markers::GPUSolver, Regularization, Data>
    where
        Method: DiscriminantMethod,
    {
        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }
}

impl<Method, Solver, Data> DiscriminantAnalysisBuilder<Method, Solver, (), Data> {
    /// Set no regularization
    pub fn no_regularization(
        self,
    ) -> DiscriminantAnalysisBuilder<Method, Solver, regularization_markers::NoRegularization, Data>
    {
        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }

    /// Set L1 regularization (only for methods that support regularization)
    pub fn l1_regularization(
        self,
    ) -> DiscriminantAnalysisBuilder<Method, Solver, regularization_markers::L1Regularization, Data>
    where
        Method: DiscriminantMethod,
    {
        // Compile-time assertion that method supports regularization
        const _: () = assert!(
            Method::SUPPORTS_REGULARIZATION,
            "Method does not support regularization"
        );

        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }

    /// Set L2 regularization
    pub fn l2_regularization(
        self,
    ) -> DiscriminantAnalysisBuilder<Method, Solver, regularization_markers::L2Regularization, Data>
    where
        Method: DiscriminantMethod,
    {
        const _: () = assert!(
            Method::SUPPORTS_REGULARIZATION,
            "Method does not support regularization"
        );

        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }

    /// Set Elastic Net regularization
    pub fn elastic_net(
        self,
    ) -> DiscriminantAnalysisBuilder<
        Method,
        Solver,
        regularization_markers::ElasticNetRegularization,
        Data,
    >
    where
        Method: DiscriminantMethod,
    {
        const _: () = assert!(
            Method::SUPPORTS_REGULARIZATION,
            "Method does not support regularization"
        );

        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }
}

impl<Method, Solver, Regularization>
    DiscriminantAnalysisBuilder<Method, Solver, Regularization, ()>
{
    /// Set data type to dense
    pub fn dense_data(
        self,
    ) -> DiscriminantAnalysisBuilder<Method, Solver, Regularization, data_markers::DenseData> {
        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }

    /// Set data type to sparse (only for compatible solvers)
    pub fn sparse_data(
        self,
    ) -> DiscriminantAnalysisBuilder<Method, Solver, Regularization, data_markers::SparseData>
    where
        Solver: SolverType,
    {
        const _: () = assert!(
            Solver::SUPPORTS_SPARSE,
            "Solver does not support sparse data"
        );

        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }

    /// Set data type to GPU data (only for GPU-compatible solvers)
    pub fn gpu_data(
        self,
    ) -> DiscriminantAnalysisBuilder<Method, Solver, Regularization, data_markers::GPUData>
    where
        Solver: SolverType,
    {
        const _: () = assert!(Solver::SUPPORTS_GPU, "Solver does not support GPU data");

        DiscriminantAnalysisBuilder {
            _method: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }
}

impl<Method, Solver, Regularization, Data>
    DiscriminantAnalysisBuilder<Method, Solver, Regularization, Data>
where
    Method: DiscriminantMethod,
    Solver: SolverType,
    Regularization: RegularizationType,
    Data: DataType,
{
    /// Build the discriminant analysis (compile-time validation complete)
    pub fn build(
        self,
    ) -> TypeSafeDiscriminantAnalysis<Method, state_markers::Untrained, Solver, Regularization, Data>
    {
        TypeSafeDiscriminantAnalysis {
            _method: PhantomData,
            _state: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        }
    }

    /// Get method information at compile time
    pub const fn method_name() -> &'static str {
        Method::METHOD_NAME
    }

    /// Get solver information at compile time
    pub const fn solver_name() -> &'static str {
        Solver::SOLVER_NAME
    }

    /// Check compatibility at compile time
    pub const fn is_compatible() -> bool {
        // Add compatibility checks between method, solver, regularization, and data type
        true // Simplified - real implementation would have complex compatibility matrix
    }
}

/// Implementation for untrained discriminant analysis
impl<Method, Solver, Regularization, Data>
    TypeSafeDiscriminantAnalysis<Method, state_markers::Untrained, Solver, Regularization, Data>
where
    Method: DiscriminantMethod,
    Solver: SolverType,
    Regularization: RegularizationType,
    Data: DataType,
{
    /// Fit the model (state transition from Untrained to Trained)
    pub fn fit(
        self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
    ) -> Result<
        TypeSafeDiscriminantAnalysis<Method, state_markers::Trained, Solver, Regularization, Data>,
    > {
        // Compile-time validation
        const _: () = assert!(
            Solver::IS_NUMERICALLY_STABLE || Regularization::SUPPORTS_AUTO_PARAMETER,
            "Numerically unstable solver requires automatic regularization"
        );

        // Perform actual fitting (delegated to appropriate implementation)
        self.fit_impl(X, y)?;

        Ok(TypeSafeDiscriminantAnalysis {
            _method: PhantomData,
            _state: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        })
    }

    /// Internal fitting implementation
    fn fit_impl(&self, X: &ArrayView2<Float>, y: &ArrayView1<usize>) -> Result<()> {
        // This would delegate to the appropriate concrete implementation
        // based on the phantom types - for now, just validate dimensions
        if X.nrows() != y.len() {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }
        Ok(())
    }
}

/// Implementation for trained discriminant analysis
impl<Method, Solver, Regularization, Data>
    TypeSafeDiscriminantAnalysis<Method, state_markers::Trained, Solver, Regularization, Data>
where
    Method: DiscriminantMethod,
    Solver: SolverType,
    Regularization: RegularizationType,
    Data: DataType,
{
    /// Predict class labels
    pub fn predict(&self, X: &ArrayView2<Float>) -> Result<Array1<usize>> {
        self.predict_impl(X)
    }

    /// Predict class probabilities (only for methods that support it)
    pub fn predict_proba(&self, X: &ArrayView2<Float>) -> Result<Array2<Float>>
    where
        Method: DiscriminantMethod,
    {
        const _: () = assert!(
            Method::SUPPORTS_PREDICT_PROBA,
            "Method does not support probability predictions"
        );
        self.predict_proba_impl(X)
    }

    /// Transform data to discriminant space (only for methods that support it)
    pub fn transform(&self, X: &ArrayView2<Float>) -> Result<Array2<Float>>
    where
        Method: DiscriminantMethod,
    {
        const _: () = assert!(
            Method::SUPPORTS_TRANSFORM,
            "Method does not support dimensionality reduction"
        );
        self.transform_impl(X)
    }

    /// Cross-validate the model (state transition to Validated)
    pub fn cross_validate(
        self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        cv_folds: usize,
    ) -> Result<
        TypeSafeDiscriminantAnalysis<
            Method,
            state_markers::Validated,
            Solver,
            Regularization,
            Data,
        >,
    > {
        self.cross_validate_impl(X, y, cv_folds)?;

        Ok(TypeSafeDiscriminantAnalysis {
            _method: PhantomData,
            _state: PhantomData,
            _solver: PhantomData,
            _regularization: PhantomData,
            _data: PhantomData,
        })
    }

    // Internal implementation methods
    fn predict_impl(&self, _X: &ArrayView2<Float>) -> Result<Array1<usize>> {
        // Placeholder - would delegate to concrete implementation
        Ok(Array1::zeros(0))
    }

    fn predict_proba_impl(&self, _X: &ArrayView2<Float>) -> Result<Array2<Float>> {
        // Placeholder - would delegate to concrete implementation
        Ok(Array2::zeros((0, 0)))
    }

    fn transform_impl(&self, _X: &ArrayView2<Float>) -> Result<Array2<Float>> {
        // Placeholder - would delegate to concrete implementation
        Ok(Array2::zeros((0, 0)))
    }

    fn cross_validate_impl(
        &self,
        _X: &ArrayView2<Float>,
        _y: &ArrayView1<usize>,
        _cv_folds: usize,
    ) -> Result<()> {
        // Placeholder - would implement cross-validation
        Ok(())
    }
}

/// Convenience type aliases for common configurations
pub type StandardLDA = UntrainedLinearDA<
    solver_markers::EigenSolver,
    regularization_markers::NoRegularization,
    data_markers::DenseData,
>;

pub type RegularizedLDA = UntrainedLinearDA<
    solver_markers::EigenSolver,
    regularization_markers::L2Regularization,
    data_markers::DenseData,
>;

pub type SparseLDA = UntrainedLinearDA<
    solver_markers::EigenSolver,
    regularization_markers::L1Regularization,
    data_markers::DenseData,
>;

pub type StandardQDA = UntrainedQuadraticDA<
    solver_markers::LDLTSolver,
    regularization_markers::NoRegularization,
    data_markers::DenseData,
>;

pub type RegularizedQDA = UntrainedQuadraticDA<
    solver_markers::LDLTSolver,
    regularization_markers::L2Regularization,
    data_markers::DenseData,
>;

pub type GPULDA = UntrainedLinearDA<
    solver_markers::GPUSolver,
    regularization_markers::NoRegularization,
    data_markers::GPUData,
>;

/// Factory functions for common configurations
impl StandardLDA {
    /// Create standard LDA configuration
    pub fn new() -> Self {
        DiscriminantAnalysisBuilder::new()
            .linear()
            .eigen_solver()
            .no_regularization()
            .dense_data()
            .build()
    }
}

impl RegularizedLDA {
    /// Create regularized LDA configuration
    pub fn new() -> Self {
        DiscriminantAnalysisBuilder::new()
            .linear()
            .eigen_solver()
            .l2_regularization()
            .dense_data()
            .build()
    }
}

impl SparseLDA {
    /// Create sparse LDA configuration
    pub fn new() -> Self {
        DiscriminantAnalysisBuilder::new()
            .linear()
            .eigen_solver()
            .l1_regularization()
            .dense_data()
            .build()
    }
}

impl StandardQDA {
    /// Create standard QDA configuration
    pub fn new() -> Self {
        DiscriminantAnalysisBuilder::new()
            .quadratic()
            .eigen_solver() // Updated from LDLT to avoid solver compatibility issues
            .no_regularization()
            .dense_data()
            .build()
    }
}

impl GPULDA {
    /// Create GPU-accelerated LDA configuration
    pub fn new() -> Self {
        DiscriminantAnalysisBuilder::new()
            .linear()
            .gpu_solver()
            .no_regularization()
            .gpu_data()
            .build()
    }
}

/// Compile-time configuration validation
pub struct ConfigurationValidator;

impl ConfigurationValidator {
    /// Validate method-solver compatibility at compile time
    pub const fn validate_method_solver<Method: DiscriminantMethod, Solver: SolverType>() -> bool {
        // Example validation rules
        match (Method::METHOD_NAME, Solver::SOLVER_NAME) {
            ("Linear Discriminant Analysis", "Iterative Solver") => true,
            ("Quadratic Discriminant Analysis", "SVD Solver") => false, // QDA doesn't use SVD typically
            _ => true, // Default to allowing combination
        }
    }

    /// Validate regularization-method compatibility at compile time
    pub const fn validate_regularization<Method: DiscriminantMethod, Reg: RegularizationType>(
    ) -> bool {
        Method::SUPPORTS_REGULARIZATION || Reg::REGULARIZATION_NAME == "No Regularization"
    }

    /// Validate data-solver compatibility at compile time
    pub const fn validate_data_solver<Solver: SolverType, Data: DataType>() -> bool {
        match (Data::DATA_TYPE_NAME, Solver::SOLVER_NAME) {
            ("Sparse Data", solver_name) => {
                // Only certain solvers support sparse data
                matches!(
                    solver_name,
                    "Eigenvalue Solver" | "Iterative Solver" | "LDLT Solver"
                )
            }
            ("GPU Data", solver_name) => {
                // Only GPU-compatible solvers support GPU data
                matches!(
                    solver_name,
                    "SVD Solver" | "Eigenvalue Solver" | "GPU Solver"
                )
            }
            _ => true,
        }
    }
}

/// Trait for zero-cost type erasure when needed
pub trait TypeErasedDiscriminant {
    /// Erase types for runtime polymorphism when needed
    fn erase_types(self) -> Box<dyn DiscriminantPredictor>;
}

/// Trait for runtime discriminant prediction
pub trait DiscriminantPredictor: Send + Sync {
    fn predict(&self, X: &ArrayView2<Float>) -> Result<Array1<usize>>;
    fn predict_proba(&self, X: &ArrayView2<Float>) -> Result<Array2<Float>>;
    fn method_name(&self) -> &'static str;
}

/// Implementation of type erasure for trained models
impl<Method, Solver, Regularization, Data> TypeErasedDiscriminant
    for TypeSafeDiscriminantAnalysis<Method, state_markers::Trained, Solver, Regularization, Data>
where
    Method: DiscriminantMethod + Send + Sync + 'static,
    Solver: SolverType + Send + Sync + 'static,
    Regularization: RegularizationType + Send + Sync + 'static,
    Data: DataType + Send + Sync + 'static,
{
    fn erase_types(self) -> Box<dyn DiscriminantPredictor> {
        Box::new(TypeErasedWrapper { inner: self })
    }
}

struct TypeErasedWrapper<Method, Solver, Regularization, Data> {
    inner:
        TypeSafeDiscriminantAnalysis<Method, state_markers::Trained, Solver, Regularization, Data>,
}

impl<Method, Solver, Regularization, Data> DiscriminantPredictor
    for TypeErasedWrapper<Method, Solver, Regularization, Data>
where
    Method: DiscriminantMethod + Send + Sync,
    Solver: SolverType + Send + Sync,
    Regularization: RegularizationType + Send + Sync,
    Data: DataType + Send + Sync,
{
    fn predict(&self, X: &ArrayView2<Float>) -> Result<Array1<usize>> {
        self.inner.predict(X)
    }

    fn predict_proba(&self, X: &ArrayView2<Float>) -> Result<Array2<Float>> {
        if Method::SUPPORTS_PREDICT_PROBA {
            self.inner.predict_proba_impl(X)
        } else {
            Err(sklears_core::prelude::SklearsError::InvalidOperation(
                format!(
                    "{} does not support probability predictions",
                    Method::METHOD_NAME
                ),
            ))
        }
    }

    fn method_name(&self) -> &'static str {
        Method::METHOD_NAME
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_time_validation() {
        // This should compile successfully
        let _lda = DiscriminantAnalysisBuilder::new()
            .linear()
            .eigen_solver()
            .l2_regularization()
            .dense_data()
            .build();

        // This should compile successfully
        let _qda = DiscriminantAnalysisBuilder::new()
            .quadratic()
            .eigen_solver()
            .no_regularization()
            .dense_data()
            .build();
    }

    #[test]
    fn test_method_capabilities() {
        use discriminant_markers::*;

        // Test compile-time constants
        assert!(LinearDiscriminant::SUPPORTS_TRANSFORM);
        assert!(!QuadraticDiscriminant::SUPPORTS_TRANSFORM);
        assert!(LinearDiscriminant::SUPPORTS_PREDICT_PROBA);
        assert!(QuadraticDiscriminant::SUPPORTS_PREDICT_PROBA);
    }

    #[test]
    fn test_solver_capabilities() {
        use solver_markers::*;

        // Test compile-time constants
        assert!(EigenSolver::SUPPORTS_SPARSE);
        assert!(!SVDSolver::SUPPORTS_SPARSE);
        assert!(GPUSolver::SUPPORTS_GPU);
        assert!(!LDLTSolver::SUPPORTS_GPU);
    }

    #[test]
    fn test_standard_configurations() {
        let _std_lda = StandardLDA::new();
        let _reg_lda = RegularizedLDA::new();
        let _sparse_lda = SparseLDA::new();
        let _std_qda = StandardQDA::new();
        let _gpu_lda = GPULDA::new();

        // All configurations should compile successfully
    }
}

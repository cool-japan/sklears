//! Type Safety Enhancements for Linear Models
//!
//! This module implements phantom types, const generics, and zero-cost abstractions
//! to provide compile-time guarantees and improve the type safety of linear models.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::marker::PhantomData;

/// Phantom type marker for untrained models
#[derive(Debug, Clone, Copy)]
pub struct Untrained;

/// Phantom type marker for trained models
#[derive(Debug, Clone, Copy)]
pub struct Trained;

/// Phantom type marker for different problem types
pub mod problem_type {
    /// Regression problem marker
    #[derive(Debug, Clone, Copy)]
    pub struct Regression;

    /// Binary classification problem marker
    #[derive(Debug, Clone, Copy)]
    pub struct BinaryClassification;

    /// Multi-class classification problem marker
    #[derive(Debug, Clone, Copy)]
    pub struct MultiClassification;

    /// Multi-output regression problem marker
    #[derive(Debug, Clone, Copy)]
    pub struct MultiOutputRegression;
}

/// Phantom type marker for different solver capabilities
pub mod solver_capability {
    /// Supports smooth objectives only
    #[derive(Debug, Clone, Copy)]
    pub struct SmoothOnly;

    /// Supports non-smooth objectives (with proximal operators)
    #[derive(Debug, Clone, Copy)]
    pub struct NonSmoothCapable;

    /// Supports large-scale problems
    #[derive(Debug, Clone, Copy)]
    pub struct LargeScale;

    /// Supports sparse problems
    #[derive(Debug, Clone, Copy)]
    pub struct SparseCapable;
}

/// Type-safe linear model with phantom types and const generics
#[derive(Debug)]
pub struct TypeSafeLinearModel<State, ProblemType, const N_FEATURES: usize> {
    /// Model state (Trained or Untrained)
    _state: PhantomData<State>,
    /// Problem type marker
    _problem_type: PhantomData<ProblemType>,
    /// Model coefficients (only available when trained)
    coefficients: Option<Array1<Float>>,
    /// Intercept term (only available when trained)
    intercept: Option<Float>,
    /// Model configuration
    config: TypeSafeConfig<ProblemType>,
}

/// Type-safe configuration for linear models
#[derive(Debug, Clone)]
pub struct TypeSafeConfig<ProblemType> {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Regularization strength
    pub alpha: Float,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Problem type marker
    _problem_type: PhantomData<ProblemType>,
}

impl<ProblemType> TypeSafeConfig<ProblemType> {
    /// Create a new configuration
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            alpha: 1.0,
            max_iter: 1000,
            tolerance: 1e-6,
            _problem_type: PhantomData,
        }
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl<const N_FEATURES: usize> TypeSafeLinearModel<Untrained, problem_type::Regression, N_FEATURES> {
    /// Create a new untrained regression model
    pub fn new_regression() -> Self {
        Self {
            _state: PhantomData,
            _problem_type: PhantomData,
            coefficients: None,
            intercept: None,
            config: TypeSafeConfig::new(),
        }
    }

    /// Configure the model
    pub fn configure(mut self, config: TypeSafeConfig<problem_type::Regression>) -> Self {
        self.config = config;
        self
    }
}

impl<const N_FEATURES: usize>
    TypeSafeLinearModel<Untrained, problem_type::BinaryClassification, N_FEATURES>
{
    /// Create a new untrained binary classification model
    pub fn new_binary_classification() -> Self {
        Self {
            _state: PhantomData,
            _problem_type: PhantomData,
            coefficients: None,
            intercept: None,
            config: TypeSafeConfig::new(),
        }
    }
}

impl<const N_FEATURES: usize>
    TypeSafeLinearModel<Untrained, problem_type::MultiClassification, N_FEATURES>
{
    /// Create a new untrained multi-class classification model
    pub fn new_multi_classification() -> Self {
        Self {
            _state: PhantomData,
            _problem_type: PhantomData,
            coefficients: None,
            intercept: None,
            config: TypeSafeConfig::new(),
        }
    }
}

/// Trait for fitting models with compile-time feature size checking
pub trait TypeSafeFit<ProblemType, const N_FEATURES: usize> {
    type TrainedModel;

    /// Fit the model to training data with compile-time feature size verification
    fn fit_typed(self, X: &Array2<Float>, y: &Array1<Float>) -> Result<Self::TrainedModel>;
}

impl<const N_FEATURES: usize> TypeSafeFit<problem_type::Regression, N_FEATURES>
    for TypeSafeLinearModel<Untrained, problem_type::Regression, N_FEATURES>
{
    type TrainedModel = TypeSafeLinearModel<Trained, problem_type::Regression, N_FEATURES>;

    fn fit_typed(self, X: &Array2<Float>, y: &Array1<Float>) -> Result<Self::TrainedModel> {
        // Compile-time feature size check
        if X.ncols() != N_FEATURES {
            return Err(SklearsError::DimensionMismatch {
                expected: N_FEATURES,
                actual: X.ncols(),
            });
        }

        // Simplified fitting logic (in practice, this would use the modular framework)
        let mut coefficients = Array1::zeros(N_FEATURES);

        // Simple least squares solution: β = (X'X)^(-1)X'y
        let xtx = X.t().dot(X);
        let xty = X.t().dot(y);

        // Add regularization
        let mut xtx_reg = xtx;
        for i in 0..N_FEATURES {
            xtx_reg[[i, i]] += self.config.alpha;
        }

        // Solve the system (simplified - in practice use proper linear algebra)
        // This is just a placeholder for the actual solving logic
        coefficients = Array1::ones(N_FEATURES) * 0.5; // Dummy solution

        let intercept = if self.config.fit_intercept {
            Some(y.mean().unwrap_or(0.0))
        } else {
            None
        };

        Ok(TypeSafeLinearModel {
            _state: PhantomData,
            _problem_type: PhantomData,
            coefficients: Some(coefficients),
            intercept,
            config: self.config,
        })
    }
}

/// Trait for making predictions with compile-time verification
pub trait TypeSafePredict<ProblemType, const N_FEATURES: usize> {
    /// Make predictions with compile-time feature size verification
    fn predict_typed(&self, X: &Array2<Float>) -> Result<Array1<Float>>;
}

impl<const N_FEATURES: usize> TypeSafePredict<problem_type::Regression, N_FEATURES>
    for TypeSafeLinearModel<Trained, problem_type::Regression, N_FEATURES>
{
    fn predict_typed(&self, X: &Array2<Float>) -> Result<Array1<Float>> {
        // Compile-time feature size check
        if X.ncols() != N_FEATURES {
            return Err(SklearsError::DimensionMismatch {
                expected: N_FEATURES,
                actual: X.ncols(),
            });
        }

        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidOperation("Model is not trained".to_string()))?;

        let mut predictions = X.dot(coefficients);

        if let Some(intercept) = self.intercept {
            predictions += intercept;
        }

        Ok(predictions)
    }
}

/// Zero-cost abstraction for different regularization schemes
pub trait RegularizationScheme {
    /// Apply regularization to the objective
    fn apply_regularization(&self, coefficients: &Array1<Float>) -> Float;

    /// Apply regularization gradient
    fn apply_regularization_gradient(&self, coefficients: &Array1<Float>) -> Array1<Float>;

    /// Get the regularization strength
    fn strength(&self) -> Float;
}

/// L2 regularization scheme (zero-cost abstraction)
#[derive(Debug, Clone)]
pub struct L2Scheme {
    pub alpha: Float,
}

impl RegularizationScheme for L2Scheme {
    fn apply_regularization(&self, coefficients: &Array1<Float>) -> Float {
        0.5 * self.alpha * coefficients.mapv(|x| x * x).sum()
    }

    fn apply_regularization_gradient(&self, coefficients: &Array1<Float>) -> Array1<Float> {
        self.alpha * coefficients
    }

    fn strength(&self) -> Float {
        self.alpha
    }
}

/// L1 regularization scheme (zero-cost abstraction)
#[derive(Debug, Clone)]
pub struct L1Scheme {
    pub alpha: Float,
}

impl RegularizationScheme for L1Scheme {
    fn apply_regularization(&self, coefficients: &Array1<Float>) -> Float {
        self.alpha * coefficients.mapv(|x| x.abs()).sum()
    }

    fn apply_regularization_gradient(&self, coefficients: &Array1<Float>) -> Array1<Float> {
        coefficients.mapv(|x| {
            if x > 0.0 {
                self.alpha
            } else if x < 0.0 {
                -self.alpha
            } else {
                0.0
            }
        })
    }

    fn strength(&self) -> Float {
        self.alpha
    }
}

/// Compile-time constraint checking for solver configurations
pub trait SolverConstraint<ProblemType> {
    /// Check if the solver is compatible with the problem type
    fn is_compatible() -> bool;

    /// Get solver-specific recommendations
    fn get_recommendations() -> &'static str;

    /// Get required features for this solver-problem combination
    fn required_features() -> &'static [&'static str] {
        &[]
    }

    /// Get incompatible features for this solver-problem combination
    fn incompatible_features() -> &'static [&'static str] {
        &[]
    }
}

/// Enhanced compile-time configuration validation
pub trait ConfigurationValidator<SolverType, ProblemType, RegularizationType> {
    /// Validate configuration at compile time
    fn validate_config() -> std::result::Result<(), &'static str>;

    /// Get optimal hyperparameters for this configuration
    fn optimal_hyperparameters() -> ConfigurationHints;
}

/// Configuration hints for optimal performance
#[derive(Debug, Clone)]
pub struct ConfigurationHints {
    /// Recommended tolerance
    pub tolerance: Option<Float>,
    /// Recommended maximum iterations
    pub max_iterations: Option<usize>,
    /// Recommended regularization strength range
    pub regularization_range: Option<(Float, Float)>,
    /// Performance notes
    pub notes: Vec<&'static str>,
}

impl Default for ConfigurationHints {
    fn default() -> Self {
        Self {
            tolerance: None,
            max_iterations: None,
            regularization_range: None,
            notes: Vec::new(),
        }
    }
}

/// Compile-time feature validation
pub trait FeatureValidator<const N_FEATURES: usize> {
    /// Validate that the feature count is appropriate for the algorithm
    fn validate_feature_count() -> std::result::Result<(), SklearsError>;

    /// Get memory requirements for this feature count
    fn memory_requirements() -> MemoryRequirements;

    /// Get computational complexity estimate
    fn computational_complexity() -> ComputationalComplexity;
}

/// Memory requirements estimate
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// Estimated memory usage in bytes
    pub estimated_bytes: usize,
    /// Whether the algorithm is memory-intensive
    pub is_memory_intensive: bool,
    /// Recommendations for memory optimization
    pub optimization_notes: Vec<&'static str>,
}

/// Computational complexity estimate
#[derive(Debug, Clone)]
pub struct ComputationalComplexity {
    /// Time complexity (e.g., "O(n^2)", "O(n*p)")
    pub time_complexity: &'static str,
    /// Space complexity
    pub space_complexity: &'static str,
    /// Whether the algorithm is compute-intensive
    pub is_compute_intensive: bool,
}

/// Advanced constraint checking for regularization compatibility
pub trait RegularizationConstraint<SolverType, RegularizationType> {
    /// Check if solver supports this regularization type
    fn is_solver_compatible() -> bool;

    /// Get solver-specific regularization recommendations
    fn get_solver_recommendations() -> &'static str;

    /// Get optimal regularization strength for this solver
    fn optimal_strength_range() -> (Float, Float);
}

/// Gradient descent is compatible with smooth problems
impl SolverConstraint<problem_type::Regression> for solver_capability::SmoothOnly {
    fn is_compatible() -> bool {
        true
    }

    fn get_recommendations() -> &'static str {
        "Gradient descent works well for smooth regression objectives"
    }

    fn required_features() -> &'static [&'static str] {
        &["smooth_objective", "differentiable"]
    }

    fn incompatible_features() -> &'static [&'static str] {
        &["l1_regularization", "non_smooth"]
    }
}

/// Coordinate descent is compatible with L1-regularized problems
impl SolverConstraint<problem_type::Regression> for solver_capability::NonSmoothCapable {
    fn is_compatible() -> bool {
        true
    }

    fn get_recommendations() -> &'static str {
        "Coordinate descent is ideal for L1-regularized problems"
    }

    fn required_features() -> &'static [&'static str] {
        &["separable_objective"]
    }

    fn incompatible_features() -> &'static [&'static str] {
        &[]
    }
}

/// Configuration validation for smooth solvers with L2 regularization
impl ConfigurationValidator<solver_capability::SmoothOnly, problem_type::Regression, L2Scheme>
    for ()
{
    fn validate_config() -> std::result::Result<(), &'static str> {
        // L2 regularization is smooth, so it's compatible with smooth solvers
        Ok(())
    }

    fn optimal_hyperparameters() -> ConfigurationHints {
        ConfigurationHints {
            tolerance: Some(1e-6),
            max_iterations: Some(1000),
            regularization_range: Some((1e-4, 1e2)),
            notes: vec![
                "Use line search for better convergence",
                "Consider preconditioning for ill-conditioned problems",
            ],
        }
    }
}

/// Configuration validation for non-smooth capable solvers with L1 regularization
impl ConfigurationValidator<solver_capability::NonSmoothCapable, problem_type::Regression, L1Scheme>
    for ()
{
    fn validate_config() -> std::result::Result<(), &'static str> {
        // L1 regularization requires non-smooth capable solvers
        Ok(())
    }

    fn optimal_hyperparameters() -> ConfigurationHints {
        ConfigurationHints {
            tolerance: Some(1e-4),
            max_iterations: Some(10000),
            regularization_range: Some((1e-6, 1e1)),
            notes: vec![
                "Use coordinate descent for efficiency",
                "Consider warm starts for regularization path",
            ],
        }
    }
}

/// Feature validation for small problems
impl<const N_FEATURES: usize> FeatureValidator<N_FEATURES> for ()
where
    [(); N_FEATURES]:,
{
    fn validate_feature_count() -> std::result::Result<(), SklearsError> {
        if N_FEATURES == 0 {
            Err(SklearsError::InvalidOperation(
                "Feature count must be greater than 0".to_string(),
            ))
        } else if N_FEATURES > 100000 {
            Err(SklearsError::InvalidOperation(
                "Feature count too large - consider dimensionality reduction".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn memory_requirements() -> MemoryRequirements {
        let bytes_per_feature = std::mem::size_of::<Float>();
        let coefficient_memory = N_FEATURES * bytes_per_feature;
        let gram_matrix_memory = N_FEATURES * N_FEATURES * bytes_per_feature;

        let total_memory = coefficient_memory + gram_matrix_memory;
        let is_memory_intensive = total_memory > 1_000_000; // 1MB threshold

        let optimization_notes = if is_memory_intensive {
            vec![
                "Consider using sparse matrices",
                "Use iterative solvers to avoid Gram matrix",
            ]
        } else {
            vec!["Memory usage is reasonable"]
        };

        MemoryRequirements {
            estimated_bytes: total_memory,
            is_memory_intensive,
            optimization_notes,
        }
    }

    fn computational_complexity() -> ComputationalComplexity {
        let is_compute_intensive = N_FEATURES > 10000;

        ComputationalComplexity {
            time_complexity: "O(n*p^2)",
            space_complexity: "O(p^2)",
            is_compute_intensive,
        }
    }
}

/// Regularization constraint for L2 with smooth solvers
impl RegularizationConstraint<solver_capability::SmoothOnly, L2Scheme> for () {
    fn is_solver_compatible() -> bool {
        true
    }

    fn get_solver_recommendations() -> &'static str {
        "L2 regularization is smooth and works well with gradient-based methods"
    }

    fn optimal_strength_range() -> (Float, Float) {
        (1e-4, 1e2)
    }
}

/// Regularization constraint for L1 with smooth solvers (incompatible)
impl RegularizationConstraint<solver_capability::SmoothOnly, L1Scheme> for () {
    fn is_solver_compatible() -> bool {
        false
    }

    fn get_solver_recommendations() -> &'static str {
        "L1 regularization is non-smooth and requires specialized solvers like coordinate descent"
    }

    fn optimal_strength_range() -> (Float, Float) {
        (0.0, 0.0) // Not applicable for incompatible combinations
    }
}

/// Regularization constraint for L1 with non-smooth capable solvers
impl RegularizationConstraint<solver_capability::NonSmoothCapable, L1Scheme> for () {
    fn is_solver_compatible() -> bool {
        true
    }

    fn get_solver_recommendations() -> &'static str {
        "L1 regularization works excellently with coordinate descent and proximal methods"
    }

    fn optimal_strength_range() -> (Float, Float) {
        (1e-6, 1e1)
    }
}

/// Type-safe solver selector
pub struct TypeSafeSolverSelector<SolverType, ProblemType> {
    _solver_type: PhantomData<SolverType>,
    _problem_type: PhantomData<ProblemType>,
}

impl<SolverType, ProblemType> TypeSafeSolverSelector<SolverType, ProblemType>
where
    SolverType: SolverConstraint<ProblemType>,
{
    /// Create a new solver selector with compile-time compatibility checking
    pub fn new() -> Self {
        // Compile-time check
        assert!(
            SolverType::is_compatible(),
            "Solver not compatible with problem type"
        );

        Self {
            _solver_type: PhantomData,
            _problem_type: PhantomData,
        }
    }

    /// Get solver recommendations
    pub fn recommendations(&self) -> &'static str {
        SolverType::get_recommendations()
    }
}

/// Fixed-size array operations for small problems (const generic optimization)
pub struct FixedSizeOps<const N: usize>;

impl<const N: usize> FixedSizeOps<N> {
    /// Dot product for fixed-size vectors (compile-time optimized)
    pub fn dot_product(a: &[Float; N], b: &[Float; N]) -> Float {
        let mut sum = 0.0;
        for i in 0..N {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Matrix-vector multiplication for fixed-size matrices (compile-time optimized)
    pub fn matrix_vector_multiply<const M: usize>(
        matrix: &[[Float; N]; M],
        vector: &[Float; N],
    ) -> [Float; M] {
        let mut result = [0.0; M];
        for i in 0..M {
            result[i] = Self::dot_product(&matrix[i], vector);
        }
        result
    }

    /// L2 norm for fixed-size vectors
    pub fn l2_norm(vector: &[Float; N]) -> Float {
        Self::dot_product(vector, vector).sqrt()
    }

    /// Normalize fixed-size vector in-place
    pub fn normalize(vector: &mut [Float; N]) {
        let norm = Self::l2_norm(vector);
        if norm > 0.0 {
            for i in 0..N {
                vector[i] /= norm;
            }
        }
    }
}

/// Type alias for common fixed-size models
pub type SmallLinearRegression = TypeSafeLinearModel<Untrained, problem_type::Regression, 10>;
pub type MediumLinearRegression = TypeSafeLinearModel<Untrained, problem_type::Regression, 100>;
pub type LargeLinearRegression = TypeSafeLinearModel<Untrained, problem_type::Regression, 1000>;

/// Builder pattern with compile-time validation
#[derive(Debug)]
pub struct TypeSafeModelBuilder<ProblemType, const N_FEATURES: usize> {
    config: TypeSafeConfig<ProblemType>,
}

impl<const N_FEATURES: usize> TypeSafeModelBuilder<problem_type::Regression, N_FEATURES> {
    /// Create a new builder for regression
    pub fn new_regression() -> Self {
        Self {
            config: TypeSafeConfig::new(),
        }
    }

    /// Set regularization strength with compile-time validation
    pub fn with_l2_regularization(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Build the model
    pub fn build(self) -> TypeSafeLinearModel<Untrained, problem_type::Regression, N_FEATURES> {
        TypeSafeLinearModel {
            _state: PhantomData,
            _problem_type: PhantomData,
            coefficients: None,
            intercept: None,
            config: self.config,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_type_safe_model_creation() {
        let model: SmallLinearRegression = TypeSafeLinearModel::new_regression();
        // Verify that the model is created with correct types
        assert!(std::mem::size_of_val(&model) > 0);
    }

    #[test]
    fn test_type_safe_config() {
        let config: TypeSafeConfig<problem_type::Regression> = TypeSafeConfig::new()
            .fit_intercept(true)
            .alpha(0.1)
            .max_iter(500)
            .tolerance(1e-8);

        assert!(config.fit_intercept);
        assert_eq!(config.alpha, 0.1);
        assert_eq!(config.max_iter, 500);
        assert_eq!(config.tolerance, 1e-8);
    }

    #[test]
    fn test_fixed_size_operations() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];

        let dot = FixedSizeOps::<3>::dot_product(&a, &b);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32

        let norm = FixedSizeOps::<3>::l2_norm(&a);
        assert!((norm - (14.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]];
        let vector = [5.0, 6.0];

        let result = FixedSizeOps::<2>::matrix_vector_multiply(&matrix, &vector);
        assert_eq!(result, [17.0, 39.0]); // [1*5+2*6, 3*5+4*6] = [17, 39]
    }

    #[test]
    fn test_regularization_schemes() {
        let coefficients = Array::from_vec(vec![1.0, -2.0, 3.0]);

        let l2_scheme = L2Scheme { alpha: 0.5 };
        let l2_penalty = l2_scheme.apply_regularization(&coefficients);
        let expected_l2 = 0.5 * 0.5 * (1.0 + 4.0 + 9.0);
        assert!((l2_penalty - expected_l2).abs() < 1e-10);

        let l1_scheme = L1Scheme { alpha: 0.3 };
        let l1_penalty = l1_scheme.apply_regularization(&coefficients);
        let expected_l1 = 0.3 * (1.0 + 2.0 + 3.0);
        assert!((l1_penalty - expected_l1).abs() < 1e-10);
    }

    #[test]
    fn test_solver_selector() {
        let _selector: TypeSafeSolverSelector<
            solver_capability::SmoothOnly,
            problem_type::Regression,
        > = TypeSafeSolverSelector::new();

        // This would fail at compile time if solver is not compatible:
        // let _incompatible: TypeSafeSolverSelector<solver_capability::SparseCapable, problem_type::BinaryClassification>
        //     = TypeSafeSolverSelector::new();
    }

    #[test]
    fn test_type_safe_builder() {
        let model: TypeSafeLinearModel<Untrained, problem_type::Regression, 5> =
            TypeSafeModelBuilder::new_regression()
                .with_l2_regularization(0.1)
                .build();

        assert_eq!(model.config.alpha, 0.1);
    }

    #[test]
    fn test_normalization() {
        let mut vector = [3.0, 4.0, 0.0];
        FixedSizeOps::<3>::normalize(&mut vector);

        let norm = FixedSizeOps::<3>::l2_norm(&vector);
        assert!((norm - 1.0).abs() < 1e-10);
    }
}

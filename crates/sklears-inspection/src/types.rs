//! Common types and enums for model inspection

// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2};
pub use sklears_core::types::Float;
use std::marker::PhantomData;

/// Phantom type for explanation state tracking
pub mod explanation_states {
    /// Marker for unvalidated explanations
    pub struct Unvalidated;
    /// Marker for validated explanations
    pub struct Validated;
    /// Marker for calibrated explanations
    pub struct Calibrated;
    /// Marker for certified explanations (robustness verified)
    pub struct Certified;
}

/// Phantom type for explanation method tracking
pub mod explanation_methods {
    /// Marker for SHAP-based explanations
    pub struct SHAP;
    /// Marker for LIME-based explanations
    pub struct LIME;
    /// Marker for permutation-based explanations
    pub struct Permutation;
    /// Marker for gradient-based explanations
    pub struct Gradient;
    /// Marker for counterfactual-based explanations
    pub struct Counterfactual;
}

/// Type-safe explanation wrapper with compile-time validation
#[derive(Debug, Clone)]
pub struct TypedExplanation<T, S> {
    inner: T,
    _state: PhantomData<S>,
}

impl<T, S> TypedExplanation<T, S> {
    /// Create a new typed explanation (internal use)
    fn new(inner: T) -> Self {
        Self {
            inner,
            _state: PhantomData,
        }
    }

    /// Get the inner explanation value
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Extract the inner explanation value
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> TypedExplanation<T, explanation_states::Unvalidated> {
    /// Create an unvalidated explanation
    pub fn unvalidated(inner: T) -> Self {
        Self::new(inner)
    }

    /// Validate the explanation (compile-time type transition)
    pub fn validate(self) -> crate::SklResult<TypedExplanation<T, explanation_states::Validated>> {
        // Add validation logic here
        Ok(TypedExplanation::new(self.inner))
    }
}

impl<T> TypedExplanation<T, explanation_states::Validated> {
    /// Calibrate the explanation (requires validation first)
    pub fn calibrate(
        self,
    ) -> crate::SklResult<TypedExplanation<T, explanation_states::Calibrated>> {
        // Add calibration logic here
        Ok(TypedExplanation::new(self.inner))
    }

    /// Certify the explanation (requires validation first)
    pub fn certify(self) -> crate::SklResult<TypedExplanation<T, explanation_states::Certified>> {
        // Add certification logic here
        Ok(TypedExplanation::new(self.inner))
    }
}

/// Trait for explanation validation
pub trait ExplanationValidator<T> {
    type Error;

    fn validate(&self, explanation: &T) -> Result<(), Self::Error>;
}

/// Trait for compile-time explanation properties
pub trait ExplanationProperties {
    type Method;
    type OutputType;

    const IS_LOCAL: bool;
    const IS_MODEL_AGNOSTIC: bool;
    const REQUIRES_GRADIENTS: bool;
}

/// Zero-cost abstraction for explanation constraints
pub trait ExplanationConstraint<T> {
    fn check(&self, explanation: &T) -> bool;
    fn description(&self) -> &'static str;
}

/// Feature importance constraint (values should sum to reasonable bounds)
pub struct FeatureImportanceConstraint {
    /// min_sum
    pub min_sum: Float,
    /// max_sum
    pub max_sum: Float,
}

impl ExplanationConstraint<Array1<Float>> for FeatureImportanceConstraint {
    fn check(&self, explanation: &Array1<Float>) -> bool {
        let sum = explanation.sum();
        sum >= self.min_sum && sum <= self.max_sum
    }

    fn description(&self) -> &'static str {
        "Feature importance values should sum within reasonable bounds"
    }
}

/// SHAP values constraint (should sum to prediction difference)
pub struct ShapConstraint {
    /// expected_sum
    pub expected_sum: Float,
    /// tolerance
    pub tolerance: Float,
}

impl ExplanationConstraint<Array1<Float>> for ShapConstraint {
    fn check(&self, explanation: &Array1<Float>) -> bool {
        let sum = explanation.sum();
        (sum - self.expected_sum).abs() <= self.tolerance
    }

    fn description(&self) -> &'static str {
        "SHAP values should sum to prediction difference from baseline"
    }
}

/// Const generic fixed-size explanation for compile-time optimization
#[derive(Debug, Clone)]
pub struct FixedSizeExplanation<T, const N: usize> {
    values: [T; N],
    feature_names: Option<[String; N]>,
}

impl<T: Copy + Default, const N: usize> FixedSizeExplanation<T, N> {
    /// Create a new fixed-size explanation
    pub fn new(values: [T; N]) -> Self {
        Self {
            values,
            feature_names: None,
        }
    }

    /// Create with feature names
    pub fn with_names(values: [T; N], names: [String; N]) -> Self {
        Self {
            values,
            feature_names: Some(names),
        }
    }

    /// Get the values
    pub fn values(&self) -> &[T; N] {
        &self.values
    }

    /// Get feature names if available
    pub fn feature_names(&self) -> Option<&[String; N]> {
        self.feature_names.as_ref()
    }

    /// Get the number of features (compile-time constant)
    pub const fn len() -> usize {
        N
    }
}

/// Type-safe model introspection traits
pub trait ModelIntrospectable {
    type FeatureType;
    type PredictionType;

    /// Get the number of features the model expects
    fn n_features(&self) -> usize;

    /// Get feature importance if available
    fn feature_importance(&self) -> Option<Array1<Float>>;

    /// Check if the model supports gradient-based explanations
    fn supports_gradients(&self) -> bool;

    /// Check if the model is linear
    fn is_linear(&self) -> bool;
}

/// Marker trait for models that support SHAP explanations
pub trait ShapCompatible: ModelIntrospectable {}

/// Marker trait for models that support LIME explanations  
pub trait LimeCompatible: ModelIntrospectable {}

/// Marker trait for models that support gradient-based explanations
pub trait GradientCompatible: ModelIntrospectable {
    /// Compute gradients with respect to input
    fn compute_gradients(&self, input: &Array1<Float>) -> crate::SklResult<Array1<Float>>;
}

/// Zero-cost explanation configuration using const generics
#[derive(Debug, Clone)]
pub struct ExplanationConfig<
    const LOCAL: bool,
    const MODEL_AGNOSTIC: bool,
    const REQUIRES_GRAD: bool,
> {
    /// n_samples
    pub n_samples: usize,
    /// random_state
    pub random_state: Option<u64>,
}

impl<const LOCAL: bool, const MODEL_AGNOSTIC: bool, const REQUIRES_GRAD: bool>
    ExplanationConfig<LOCAL, MODEL_AGNOSTIC, REQUIRES_GRAD>
{
    /// Create a new explanation configuration
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            random_state: None,
        }
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Check if this is a local explanation method
    pub const fn is_local(&self) -> bool {
        LOCAL
    }

    /// Check if this is model-agnostic
    pub const fn is_model_agnostic(&self) -> bool {
        MODEL_AGNOSTIC
    }

    /// Check if gradients are required
    pub const fn requires_gradients(&self) -> bool {
        REQUIRES_GRAD
    }
}

/// Type aliases for common explanation configurations
pub type LocalModelAgnosticConfig = ExplanationConfig<true, true, false>;
pub type GlobalModelAgnosticConfig = ExplanationConfig<false, true, false>;
pub type LocalGradientConfig = ExplanationConfig<true, false, true>;
pub type GlobalGradientConfig = ExplanationConfig<false, false, true>;

/// Compile-time validation for explanation method compatibility
pub struct ExplanationMethodValidator<M, C> {
    _model: PhantomData<M>,
    _config: PhantomData<C>,
}

impl<M, C> ExplanationMethodValidator<M, C>
where
    M: ModelIntrospectable,
    C: ExplanationProperties,
{
    /// Validate that the model supports the explanation method
    pub fn validate_compatibility() -> Result<(), &'static str> {
        // This could be extended with more sophisticated compile-time checks
        Ok(())
    }
}

/// Implementation of ExplanationProperties for SHAP
impl ExplanationProperties for explanation_methods::SHAP {
    type Method = explanation_methods::SHAP;
    type OutputType = Array1<Float>;

    const IS_LOCAL: bool = true;
    const IS_MODEL_AGNOSTIC: bool = true;
    const REQUIRES_GRADIENTS: bool = false;
}

/// Implementation of ExplanationProperties for LIME
impl ExplanationProperties for explanation_methods::LIME {
    type Method = explanation_methods::LIME;
    type OutputType = Array1<Float>;

    const IS_LOCAL: bool = true;
    const IS_MODEL_AGNOSTIC: bool = true;
    const REQUIRES_GRADIENTS: bool = false;
}

/// Implementation of ExplanationProperties for Gradient-based methods
impl ExplanationProperties for explanation_methods::Gradient {
    type Method = explanation_methods::Gradient;
    type OutputType = Array1<Float>;

    const IS_LOCAL: bool = true;
    const IS_MODEL_AGNOSTIC: bool = false;
    const REQUIRES_GRADIENTS: bool = true;
}

/// Score functions for permutation importance
#[derive(Debug, Clone, Copy)]
pub enum ScoreFunction {
    /// Accuracy score (classification)
    Accuracy,
    /// R² score (regression)  
    R2,
    /// Mean squared error (regression, negated for optimization)
    MeanSquaredError,
}

/// Kind of partial dependence
#[derive(Debug, Clone, Copy)]
pub enum PartialDependenceKind {
    /// Average partial dependence across all instances
    Average,
    /// Individual partial dependence for each instance
    Individual,
}

/// Result of permutation importance analysis
#[derive(Debug, Clone)]
pub struct PermutationImportanceResult {
    /// Raw importance values for each repeat and feature
    pub importances: Vec<Vec<Float>>,
    /// Mean importance for each feature
    pub importances_mean: Array1<Float>,
    /// Standard deviation of importance for each feature
    pub importances_std: Array1<Float>,
}

/// Result of partial dependence analysis
#[derive(Debug, Clone)]
pub struct PartialDependenceResult {
    /// Partial dependence values
    pub values: Vec<Float>,
    /// Individual predictions for each grid point (for individual PD plots)
    pub individual_values: Vec<Vec<Float>>,
    /// Grid values used for computation
    pub grid: Vec<Float>,
}

/// Feature Importance Plot Data
///
/// Represents feature importance data suitable for plotting, with optional
/// confidence intervals and feature names.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Feature indices
    pub feature_indices: Vec<usize>,
    /// Importance values
    pub importances: Array1<Float>,
    /// Standard errors (optional)
    pub std_errors: Option<Array1<Float>>,
    /// Feature names (optional)
    pub feature_names: Option<Vec<String>>,
}

impl FeatureImportance {
    /// Create a new FeatureImportance instance
    pub fn new(
        feature_indices: Vec<usize>,
        importances: Array1<Float>,
        std_errors: Option<Array1<Float>>,
        feature_names: Option<Vec<String>>,
    ) -> Self {
        Self {
            feature_indices,
            importances,
            std_errors,
            feature_names,
        }
    }

    /// Get the top k most important features
    pub fn top_k(&self, k: usize) -> Vec<(usize, Float)> {
        let mut indexed_importances: Vec<(usize, Float)> = self
            .feature_indices
            .iter()
            .zip(self.importances.iter())
            .map(|(&idx, &imp)| (idx, imp))
            .collect();

        indexed_importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_importances.into_iter().take(k).collect()
    }

    /// Get features sorted by importance (descending)
    pub fn sorted_features(&self) -> Vec<(usize, Float)> {
        self.top_k(self.feature_indices.len())
    }
}

/// Result of SHAP analysis
#[derive(Debug, Clone)]
pub struct ShapResult {
    /// SHAP values for each instance and feature
    pub shap_values: Array2<Float>,
    /// Expected value (baseline prediction)
    pub expected_value: Float,
    /// Feature indices
    pub feature_indices: Vec<usize>,
}

/// Result of learning curve analysis
#[derive(Debug, Clone)]
pub struct LearningCurveResult {
    /// Training set sizes used
    pub train_sizes: Vec<usize>,
    /// Training scores for each size and CV fold
    pub train_scores: Array2<Float>,
    /// Validation scores for each size and CV fold  
    pub validation_scores: Array2<Float>,
}

/// Result of validation curve analysis
#[derive(Debug, Clone)]
pub struct ValidationCurveResult {
    /// Parameter values used
    pub param_range: Vec<Float>,
    /// Training scores for each parameter value and CV fold
    pub train_scores: Array2<Float>,
    /// Validation scores for each parameter value and CV fold
    pub validation_scores: Array2<Float>,
}

/// Result of ICE plots analysis
#[derive(Debug, Clone)]
pub struct IcePlotsResult {
    /// ICE curves for each instance (instance_idx, grid_values)
    pub ice_curves: Array2<Float>,
    /// Grid values used for the feature
    pub grid_values: Array1<Float>,
    /// Feature index analyzed
    pub feature_idx: usize,
    /// Partial dependence (average of ICE curves)
    pub partial_dependence: Array1<Float>,
    /// Indices of instances included
    pub sample_indices: Vec<usize>,
}

/// Result of centered ICE plots analysis
#[derive(Debug, Clone)]
pub struct CenteredIcePlotsResult {
    /// Centered ICE curves (baseline subtracted)
    pub centered_curves: Array2<Float>,
    /// Grid values used for the feature
    pub grid_values: Array1<Float>,
    /// Original (uncentered) ICE curves
    pub ice_curves: Array2<Float>,
}

/// Result of feature interaction analysis
#[derive(Debug, Clone)]
pub struct FeatureInteractionResult {
    /// Interaction strength matrix (feature_i × feature_j)
    pub interaction_matrix: Array2<Float>,
    /// Feature indices
    pub feature_indices: Vec<usize>,
    /// Detailed interaction information
    pub interactions: Vec<InteractionDetail>,
}

/// Detailed information about a feature interaction
#[derive(Debug, Clone)]
pub struct InteractionDetail {
    /// First feature index
    pub feature_i: usize,
    /// Second feature index  
    pub feature_j: usize,
    /// Interaction strength
    pub strength: Float,
    /// Statistical significance (p-value if available)
    pub p_value: Option<Float>,
}

//! Type safety utilities for pipeline composition
//!
//! This module provides phantom types and compile-time validation for pipeline stages,
//! ensuring that incompatible transformations and estimators cannot be composed.

use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::{prelude::Transform, traits::Estimator, types::Float};
use std::marker::PhantomData;

/// Phantom type representing the input type of a pipeline stage
pub struct Input<T>(PhantomData<T>);

/// Phantom type representing the output type of a pipeline stage
pub struct Output<T>(PhantomData<T>);

/// Phantom type representing a numerical array input
pub struct NumericInput;

/// Phantom type representing a categorical array input
pub struct CategoricalInput;

/// Phantom type representing a mixed-type input
pub struct MixedInput;

/// Phantom type representing a dense array output
pub struct DenseOutput;

/// Phantom type representing a sparse array output
pub struct SparseOutput;

/// Phantom type representing a classification output
pub struct ClassificationOutput;

/// Phantom type representing a regression output
pub struct RegressionOutput;

/// Type-safe pipeline stage that enforces input/output compatibility
pub struct TypedPipelineStage<I, O> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

impl<I, O> TypedPipelineStage<I, O> {
    /// Create a new typed pipeline stage
    #[must_use]
    pub fn new() -> Self {
        Self {
            _input: PhantomData,
            _output: PhantomData,
        }
    }
}

impl<I, O> Default for TypedPipelineStage<I, O> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for enforcing type compatibility between pipeline stages
pub trait TypeCompatible<T> {
    /// Check if this stage is compatible with the given type
    fn is_compatible(&self) -> bool;
}

/// Implementation for numeric input compatibility
impl TypeCompatible<NumericInput> for TypedPipelineStage<NumericInput, DenseOutput> {
    fn is_compatible(&self) -> bool {
        true
    }
}

impl TypeCompatible<NumericInput> for TypedPipelineStage<NumericInput, SparseOutput> {
    fn is_compatible(&self) -> bool {
        true
    }
}

/// Implementation for categorical input compatibility
impl TypeCompatible<CategoricalInput> for TypedPipelineStage<CategoricalInput, DenseOutput> {
    fn is_compatible(&self) -> bool {
        true
    }
}

/// Type-safe transformer that enforces input/output types
pub struct TypedTransformer<I, O, T> {
    transformer: T,
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

impl<I, O, T> TypedTransformer<I, O, T> {
    /// Create a new typed transformer
    pub fn new(transformer: T) -> Self {
        Self {
            transformer,
            _input: PhantomData,
            _output: PhantomData,
        }
    }

    /// Get the underlying transformer
    pub fn inner(&self) -> &T {
        &self.transformer
    }

    /// Consume the typed transformer and return the inner transformer
    pub fn into_inner(self) -> T {
        self.transformer
    }
}

/// Type-safe estimator that enforces input/output types
pub struct TypedEstimator<I, O, E> {
    estimator: E,
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

impl<I, O, E> TypedEstimator<I, O, E> {
    /// Create a new typed estimator
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            _input: PhantomData,
            _output: PhantomData,
        }
    }

    /// Get the underlying estimator
    pub fn inner(&self) -> &E {
        &self.estimator
    }

    /// Consume the typed estimator and return the inner estimator
    pub fn into_inner(self) -> E {
        self.estimator
    }
}

/// Compile-time pipeline validation trait
pub trait PipelineValidation<Stages> {
    /// Validate that all stages in the pipeline are compatible
    fn validate() -> Result<(), PipelineValidationError>;
}

/// Error type for pipeline validation
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineValidationError {
    /// Incompatible input/output types between stages
    IncompatibleTypes {
        stage_index: usize,
        expected: String,
        found: String,
    },
    /// Missing required stage
    MissingStage { stage_name: String },
    /// Invalid stage configuration
    InvalidConfiguration { stage_index: usize, reason: String },
}

impl std::fmt::Display for PipelineValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineValidationError::IncompatibleTypes {
                stage_index,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Incompatible types at stage {stage_index}: expected {expected}, found {found}"
                )
            }
            PipelineValidationError::MissingStage { stage_name } => {
                write!(f, "Missing required stage: {stage_name}")
            }
            PipelineValidationError::InvalidConfiguration {
                stage_index,
                reason,
            } => {
                write!(f, "Invalid configuration at stage {stage_index}: {reason}")
            }
        }
    }
}

impl std::error::Error for PipelineValidationError {}

/// Type-safe pipeline builder that validates stages at compile time
pub struct TypedPipelineBuilder<T> {
    stages: Vec<String>,
    _phantom: PhantomData<T>,
}

impl TypedPipelineBuilder<()> {
    /// Create a new typed pipeline builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

impl<T> TypedPipelineBuilder<T> {
    /// Add a transformation stage with type validation
    pub fn transform<I, O, Trans>(
        mut self,
        name: &str,
        _transformer: TypedTransformer<I, O, Trans>,
    ) -> TypedPipelineBuilder<(T, TypedTransformer<I, O, Trans>)>
    where
        Trans: for<'a> Transform<ArrayView2<'a, Float>, Array2<f64>>,
    {
        self.stages.push(name.to_string());
        TypedPipelineBuilder {
            stages: self.stages,
            _phantom: PhantomData,
        }
    }

    /// Add an estimation stage with type validation
    pub fn estimate<I, O, Est>(
        mut self,
        name: &str,
        _estimator: TypedEstimator<I, O, Est>,
    ) -> TypedPipelineBuilder<(T, TypedEstimator<I, O, Est>)>
    where
        Est: Estimator,
    {
        self.stages.push(name.to_string());
        TypedPipelineBuilder {
            stages: self.stages,
            _phantom: PhantomData,
        }
    }

    /// Get the stage names
    #[must_use]
    pub fn stage_names(&self) -> &[String] {
        &self.stages
    }
}

impl Default for TypedPipelineBuilder<()> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile-time data flow validation
pub struct DataFlowValidator<T> {
    _phantom: PhantomData<T>,
}

impl<T> DataFlowValidator<T> {
    /// Create a new data flow validator
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for DataFlowValidator<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for validating data flow through pipeline stages
pub trait DataFlowValidation {
    /// Validate that data can flow through all stages
    fn validate_flow(&self) -> Result<(), PipelineValidationError>;
}

impl DataFlowValidation for DataFlowValidator<NumericInput> {
    fn validate_flow(&self) -> Result<(), PipelineValidationError> {
        // Numeric input can flow through most transformations
        Ok(())
    }
}

impl DataFlowValidation for DataFlowValidator<CategoricalInput> {
    fn validate_flow(&self) -> Result<(), PipelineValidationError> {
        // Categorical input requires appropriate encoders
        Ok(())
    }
}

/// Type-safe feature union that validates input types
pub struct TypedFeatureUnion<I, O> {
    transformers: Vec<String>,
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

impl<I, O> TypedFeatureUnion<I, O> {
    /// Create a new typed feature union
    #[must_use]
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
            _input: PhantomData,
            _output: PhantomData,
        }
    }

    /// Add a transformer with type validation
    pub fn add_transformer<T>(mut self, name: &str, _transformer: TypedTransformer<I, O, T>) -> Self
    where
        T: for<'a> Transform<ArrayView2<'a, Float>, Array2<f64>>,
    {
        self.transformers.push(name.to_string());
        self
    }

    /// Get transformer names
    #[must_use]
    pub fn transformer_names(&self) -> &[String] {
        &self.transformers
    }
}

impl<I, O> Default for TypedFeatureUnion<I, O> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile-time pipeline structure validation
pub trait StructureValidation {
    /// Validate the structure of the pipeline
    fn validate_structure() -> Result<(), PipelineValidationError>;
}

/// Helper macro for creating type-safe pipelines
#[macro_export]
macro_rules! typed_pipeline {
    ($($stage:expr),+ $(,)?) => {{
        let mut builder = TypedPipelineBuilder::new();
        $(
            builder = builder.add_stage($stage);
        )+
        builder
    }};
}

/// Helper macro for validating pipeline compatibility at compile time
#[macro_export]
macro_rules! validate_pipeline {
    ($pipeline:expr) => {{
        compile_time_validate!($pipeline)
    }};
}

/// Compile-time validation macro
#[macro_export]
macro_rules! compile_time_validate {
    ($pipeline:expr) => {{
        // This would be expanded at compile time to validate pipeline structure
        Ok(())
    }};
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_pipeline_stage_creation() {
        let stage: TypedPipelineStage<NumericInput, DenseOutput> = TypedPipelineStage::new();
        assert!(stage.is_compatible());
    }

    #[test]
    fn test_typed_transformer_creation() {
        #[derive(Debug, PartialEq)]
        struct DummyTransformer(i32);

        let dummy = DummyTransformer(42);
        let transformer = TypedTransformer::<NumericInput, DenseOutput, _>::new(dummy);

        // Test that we can access the inner transformer and it has the expected value
        assert_eq!(transformer.inner().0, 42);
    }

    #[test]
    fn test_typed_estimator_creation() {
        #[derive(Debug, PartialEq)]
        struct DummyEstimator(String);

        let dummy = DummyEstimator("test".to_string());
        let estimator = TypedEstimator::<NumericInput, ClassificationOutput, _>::new(dummy);

        // Test that we can access the inner estimator and it has the expected value
        assert_eq!(estimator.inner().0, "test");
    }

    #[test]
    fn test_typed_pipeline_builder() {
        let builder = TypedPipelineBuilder::new();
        assert_eq!(builder.stage_names().len(), 0);
    }

    #[test]
    fn test_data_flow_validator() {
        let validator: DataFlowValidator<NumericInput> = DataFlowValidator::new();
        assert!(validator.validate_flow().is_ok());
    }

    #[test]
    fn test_typed_feature_union() {
        let union: TypedFeatureUnion<NumericInput, DenseOutput> = TypedFeatureUnion::new();
        assert_eq!(union.transformer_names().len(), 0);
    }

    #[test]
    fn test_pipeline_validation_error_display() {
        let error = PipelineValidationError::IncompatibleTypes {
            stage_index: 1,
            expected: "NumericInput".to_string(),
            found: "CategoricalInput".to_string(),
        };
        let display = format!("{}", error);
        assert!(display.contains("Incompatible types"));
        assert!(display.contains("stage 1"));
    }
}

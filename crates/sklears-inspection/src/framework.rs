//! Trait-based explanation framework for composable strategies
//!
//! This module provides a comprehensive framework for building composable explanation
//! strategies using traits, allowing for modular and extensible explanation pipelines.

use crate::{Float, ParallelConfig, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::fmt::Debug;

/// Core trait for all explanation methods
pub trait Explainer {
    /// Input type for the explanation method
    type Input;
    /// Output type containing the explanation
    type Output;
    /// Configuration type for the method
    type Config;

    /// Generate explanation for given input
    fn explain(&self, input: &Self::Input, config: &Self::Config) -> SklResult<Self::Output>;

    /// Get the name/identifier of this explanation method
    fn name(&self) -> &'static str;

    /// Get method-specific metadata
    fn metadata(&self) -> ExplanationMetadata {
        ExplanationMetadata::default()
    }
}

/// Metadata about an explanation method
#[derive(Debug, Clone)]
pub struct ExplanationMetadata {
    /// Whether this method provides local explanations
    pub is_local: bool,
    /// Whether this method is model-agnostic
    pub is_model_agnostic: bool,
    /// Whether this method requires gradients
    pub requires_gradients: bool,
    /// Whether this method supports parallel computation
    pub supports_parallel: bool,
    /// Computational complexity (Big O notation as string)
    pub complexity: String,
    /// Description of the method
    pub description: String,
}

impl Default for ExplanationMetadata {
    fn default() -> Self {
        Self {
            is_local: false,
            is_model_agnostic: true,
            requires_gradients: false,
            supports_parallel: true,
            complexity: "O(n)".to_string(),
            description: "Generic explanation method".to_string(),
        }
    }
}

/// Trait for feature-based explanations
pub trait FeatureExplainer: Explainer {
    /// Get feature importance values
    fn feature_importance(
        &self,
        input: &Self::Input,
        config: &Self::Config,
    ) -> SklResult<Array1<Float>>;

    /// Get top-k most important features
    fn top_features(
        &self,
        input: &Self::Input,
        config: &Self::Config,
        k: usize,
    ) -> SklResult<Vec<(usize, Float)>> {
        let importance = self.feature_importance(input, config)?;
        let mut indexed_importance: Vec<(usize, Float)> = importance
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        indexed_importance
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(indexed_importance.into_iter().take(k).collect())
    }
}

/// Trait for local explanation methods
pub trait LocalExplainer: Explainer {
    /// Generate explanation for a single instance
    fn explain_instance(
        &self,
        instance: &ArrayView1<Float>,
        config: &Self::Config,
    ) -> SklResult<Self::Output>;
}

/// Trait for global explanation methods
pub trait GlobalExplainer: Explainer {
    /// Generate explanation for the entire dataset/model
    fn explain_global(
        &self,
        data: &ArrayView2<Float>,
        config: &Self::Config,
    ) -> SklResult<Self::Output>;
}

/// Trait for counterfactual explanation methods
pub trait CounterfactualExplainer: Explainer {
    /// Generate counterfactual explanations
    fn generate_counterfactual(
        &self,
        instance: &ArrayView1<Float>,
        desired_outcome: Float,
        config: &Self::Config,
    ) -> SklResult<Self::Output>;

    /// Generate multiple diverse counterfactuals
    fn generate_diverse_counterfactuals(
        &self,
        instance: &ArrayView1<Float>,
        desired_outcome: Float,
        n_counterfactuals: usize,
        config: &Self::Config,
    ) -> SklResult<Vec<Self::Output>>;
}

/// Trait for uncertainty-aware explanations
pub trait UncertaintyAwareExplainer: Explainer {
    /// Generate explanation with uncertainty quantification
    fn explain_with_uncertainty(
        &self,
        input: &Self::Input,
        config: &Self::Config,
    ) -> SklResult<UncertainExplanation<Self::Output>>;
}

/// Explanation with uncertainty information
#[derive(Debug, Clone)]
pub struct UncertainExplanation<T> {
    /// explanation
    pub explanation: T,
    /// confidence
    pub confidence: Float,
    /// confidence_intervals
    pub confidence_intervals: Option<(Float, Float)>,
    /// epistemic_uncertainty
    pub epistemic_uncertainty: Option<Float>,
    /// aleatoric_uncertainty
    pub aleatoric_uncertainty: Option<Float>,
}

/// Trait for composable explanation strategies
pub trait ExplanationStrategy {
    type Input;
    type Output;
    type Config;

    /// Execute the strategy
    fn execute(&self, input: &Self::Input, config: &Self::Config) -> SklResult<Self::Output>;

    /// Combine with another strategy
    fn combine<S>(self, other: S) -> CombinedStrategy<Self, S>
    where
        Self: Sized,
        S: ExplanationStrategy<Input = Self::Input>,
    {
        CombinedStrategy::new(self, other)
    }

    /// Chain with another strategy
    fn chain<S>(self, other: S) -> ChainedStrategy<Self, S>
    where
        Self: Sized,
    {
        ChainedStrategy::new(self, other)
    }
}

/// Combined explanation strategy that runs multiple strategies
#[derive(Debug)]
pub struct CombinedStrategy<A, B> {
    strategy_a: A,
    strategy_b: B,
}

impl<A, B> CombinedStrategy<A, B> {
    pub fn new(strategy_a: A, strategy_b: B) -> Self {
        Self {
            strategy_a,
            strategy_b,
        }
    }
}

impl<A, B> ExplanationStrategy for CombinedStrategy<A, B>
where
    A: ExplanationStrategy,
    B: ExplanationStrategy<Input = A::Input>,
{
    type Input = A::Input;
    type Output = CombinedOutput<A::Output, B::Output>;
    type Config = CombinedConfig<A::Config, B::Config>;

    fn execute(&self, input: &Self::Input, config: &Self::Config) -> SklResult<Self::Output> {
        let output_a = self.strategy_a.execute(input, &config.config_a)?;
        let output_b = self.strategy_b.execute(input, &config.config_b)?;

        Ok(CombinedOutput { output_a, output_b })
    }
}

/// Output from combined strategies
#[derive(Debug, Clone)]
pub struct CombinedOutput<A, B> {
    /// output_a
    pub output_a: A,
    /// output_b
    pub output_b: B,
}

/// Configuration for combined strategies
#[derive(Debug, Clone)]
pub struct CombinedConfig<A, B> {
    /// config_a
    pub config_a: A,
    /// config_b
    pub config_b: B,
}

/// Chained explanation strategy that passes output from one to another
#[derive(Debug)]
pub struct ChainedStrategy<A, B> {
    first: A,
    second: B,
}

impl<A, B> ChainedStrategy<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A, B> ExplanationStrategy for ChainedStrategy<A, B>
where
    A: ExplanationStrategy,
    B: ExplanationStrategy<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;
    type Config = ChainedConfig<A::Config, B::Config>;

    fn execute(&self, input: &Self::Input, config: &Self::Config) -> SklResult<Self::Output> {
        let intermediate = self.first.execute(input, &config.first_config)?;
        self.second.execute(&intermediate, &config.second_config)
    }
}

/// Configuration for chained strategies
#[derive(Debug, Clone)]
pub struct ChainedConfig<A, B> {
    /// first_config
    pub first_config: A,
    /// second_config
    pub second_config: B,
}

/// Trait for explanation post-processing
pub trait ExplanationPostProcessor {
    type Input;
    type Output;

    /// Process explanation output
    fn process(&self, explanation: Self::Input) -> SklResult<Self::Output>;
}

/// Trait for explanation validation
pub trait ExplanationValidator {
    type Input;

    /// Validate explanation
    fn validate(&self, explanation: &Self::Input) -> SklResult<ValidationResult>;
}

/// Result of explanation validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// is_valid
    pub is_valid: bool,
    /// violations
    pub violations: Vec<ValidationViolation>,
    /// confidence_score
    pub confidence_score: Float,
}

/// Validation violation
#[derive(Debug, Clone)]
pub struct ValidationViolation {
    /// rule
    pub rule: String,
    /// severity
    pub severity: ViolationSeverity,
    /// description
    pub description: String,
}

/// Severity of validation violation
#[derive(Debug, Clone)]
pub enum ViolationSeverity {
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Explanation pipeline builder
pub struct ExplanationPipeline<T> {
    explainers:
        Vec<Box<dyn Explainer<Input = T, Output = Array1<Float>, Config = ()> + Send + Sync>>,
    post_processors: Vec<
        Box<
            dyn ExplanationPostProcessor<Input = Array1<Float>, Output = Array1<Float>>
                + Send
                + Sync,
        >,
    >,
    validators: Vec<Box<dyn ExplanationValidator<Input = Array1<Float>> + Send + Sync>>,
    parallel_config: ParallelConfig,
}

impl<T> ExplanationPipeline<T>
where
    T: Send + Sync,
{
    /// Create a new explanation pipeline
    pub fn new() -> Self {
        Self {
            explainers: Vec::new(),
            post_processors: Vec::new(),
            validators: Vec::new(),
            parallel_config: ParallelConfig::default(),
        }
    }

    /// Add an explainer to the pipeline
    pub fn add_explainer<E>(mut self, explainer: E) -> Self
    where
        E: Explainer<Input = T, Output = Array1<Float>, Config = ()> + Send + Sync + 'static,
    {
        self.explainers.push(Box::new(explainer));
        self
    }

    /// Add a post-processor to the pipeline
    pub fn add_post_processor<P>(mut self, processor: P) -> Self
    where
        P: ExplanationPostProcessor<Input = Array1<Float>, Output = Array1<Float>>
            + Send
            + Sync
            + 'static,
    {
        self.post_processors.push(Box::new(processor));
        self
    }

    /// Add a validator to the pipeline
    pub fn add_validator<V>(mut self, validator: V) -> Self
    where
        V: ExplanationValidator<Input = Array1<Float>> + Send + Sync + 'static,
    {
        self.validators.push(Box::new(validator));
        self
    }

    /// Set parallel configuration
    pub fn with_parallel_config(mut self, config: ParallelConfig) -> Self {
        self.parallel_config = config;
        self
    }

    /// Execute the pipeline
    pub fn execute(&self, input: &T) -> SklResult<PipelineResult> {
        let mut explanations = Vec::new();
        let mut validation_results = Vec::new();

        // Run all explainers
        for explainer in &self.explainers {
            let mut explanation = explainer.explain(input, &())?;

            // Apply post-processors
            for processor in &self.post_processors {
                explanation = processor.process(explanation)?;
            }

            // Run validators
            for validator in &self.validators {
                let validation = validator.validate(&explanation)?;
                validation_results.push(validation);
            }

            explanations.push(explanation);
        }

        Ok(PipelineResult {
            explanations,
            validation_results,
        })
    }
}

/// Result of pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// explanations
    pub explanations: Vec<Array1<Float>>,
    /// validation_results
    pub validation_results: Vec<ValidationResult>,
}

/// Trait for feature attribution methods
pub trait FeatureAttributor {
    /// Compute feature attributions
    fn attribute(
        &self,
        input: &ArrayView1<Float>,
        target: Option<usize>,
    ) -> SklResult<Array1<Float>>;

    /// Compute attributions for multiple targets (multi-class)
    fn attribute_multiple(
        &self,
        input: &ArrayView1<Float>,
        targets: &[usize],
    ) -> SklResult<Array2<Float>> {
        let mut attributions = Vec::new();
        for &target in targets {
            let attr = self.attribute(input, Some(target))?;
            attributions.push(attr);
        }

        let n_targets = targets.len();
        let n_features = attributions[0].len();
        let mut result = Array2::zeros((n_targets, n_features));

        for (i, attribution) in attributions.into_iter().enumerate() {
            result.row_mut(i).assign(&attribution);
        }

        Ok(result)
    }
}

/// Trait for gradient-based attribution methods
pub trait GradientAttributor: FeatureAttributor {
    /// Compute gradients
    fn compute_gradients(
        &self,
        input: &ArrayView1<Float>,
        target: Option<usize>,
    ) -> SklResult<Array1<Float>>;

    /// Compute integrated gradients
    fn integrated_gradients(
        &self,
        input: &ArrayView1<Float>,
        baseline: &ArrayView1<Float>,
        steps: usize,
        target: Option<usize>,
    ) -> SklResult<Array1<Float>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    // Mock explainer for testing
    struct MockExplainer;

    impl Explainer for MockExplainer {
        type Input = Array1<Float>;
        type Output = Array1<Float>;
        type Config = ();

        fn explain(&self, input: &Self::Input, _config: &Self::Config) -> SklResult<Self::Output> {
            Ok(input.clone())
        }

        fn name(&self) -> &'static str {
            "MockExplainer"
        }
    }

    impl FeatureExplainer for MockExplainer {
        fn feature_importance(
            &self,
            input: &Self::Input,
            _config: &Self::Config,
        ) -> SklResult<Array1<Float>> {
            Ok(input.clone())
        }
    }

    impl LocalExplainer for MockExplainer {
        fn explain_instance(
            &self,
            instance: &ArrayView1<Float>,
            _config: &Self::Config,
        ) -> SklResult<Self::Output> {
            Ok(instance.to_owned())
        }
    }

    // Mock strategy for testing
    struct MockStrategy;

    impl ExplanationStrategy for MockStrategy {
        type Input = Array1<Float>;
        type Output = Array1<Float>;
        type Config = ();

        fn execute(&self, input: &Self::Input, _config: &Self::Config) -> SklResult<Self::Output> {
            Ok(input.clone())
        }
    }

    #[test]
    fn test_explanation_metadata_default() {
        let metadata = ExplanationMetadata::default();
        assert!(!metadata.is_local);
        assert!(metadata.is_model_agnostic);
        assert!(!metadata.requires_gradients);
        assert!(metadata.supports_parallel);
    }

    #[test]
    fn test_mock_explainer() {
        let explainer = MockExplainer;
        let input = array![1.0, 2.0, 3.0];

        let result = explainer.explain(&input, &());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), input);
        assert_eq!(explainer.name(), "MockExplainer");
    }

    #[test]
    fn test_feature_explainer_top_features() {
        let explainer = MockExplainer;
        let input = array![3.0, 1.0, 4.0, 2.0];

        let top_features = explainer.top_features(&input, &(), 2);
        assert!(top_features.is_ok());

        let features = top_features.unwrap();
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].0, 2); // Index of max value (4.0)
        assert_eq!(features[1].0, 0); // Index of second max (3.0)
    }

    #[test]
    fn test_local_explainer() {
        let explainer = MockExplainer;
        let instance = array![1.0, 2.0];

        let result = explainer.explain_instance(&instance.view(), &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_combined_strategy() {
        let strategy_a = MockStrategy;
        let strategy_b = MockStrategy;
        let combined = strategy_a.combine(strategy_b);

        let input = array![1.0, 2.0];
        let config = CombinedConfig {
            config_a: (),
            config_b: (),
        };

        let result = combined.execute(&input, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_chained_strategy() {
        let first = MockStrategy;
        let second = MockStrategy;
        let chained = first.chain(second);

        let input = array![1.0, 2.0];
        let config = ChainedConfig {
            first_config: (),
            second_config: (),
        };

        let result = chained.execute(&input, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_uncertain_explanation_creation() {
        let explanation = array![1.0, 2.0, 3.0];
        let uncertain = UncertainExplanation {
            explanation,
            confidence: 0.8,
            confidence_intervals: Some((0.7, 0.9)),
            epistemic_uncertainty: Some(0.1),
            aleatoric_uncertainty: Some(0.05),
        };

        assert_eq!(uncertain.confidence, 0.8);
        assert!(uncertain.confidence_intervals.is_some());
    }

    #[test]
    fn test_validation_result_creation() {
        let violation = ValidationViolation {
            rule: "Sum to one".to_string(),
            severity: ViolationSeverity::Warning,
            description: "Values should sum to 1.0".to_string(),
        };

        let result = ValidationResult {
            is_valid: false,
            violations: vec![violation],
            confidence_score: 0.7,
        };

        assert!(!result.is_valid);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.confidence_score, 0.7);
    }

    #[test]
    fn test_explanation_pipeline_creation() {
        let pipeline: ExplanationPipeline<ArrayView1<Float>> = ExplanationPipeline::new();
        assert_eq!(pipeline.explainers.len(), 0);
        assert_eq!(pipeline.post_processors.len(), 0);
        assert_eq!(pipeline.validators.len(), 0);
    }
}

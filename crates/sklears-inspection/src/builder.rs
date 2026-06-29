//! Builder pattern and fluent API for complex explanations
//!
//! This module provides builder patterns for constructing complex explanation pipelines
//! with a fluent API that makes it easy to chain operations and configure explanations.

#[cfg(feature = "parallel")]
use crate::ParallelConfig;
use crate::{Float, SklResult};

#[cfg(not(feature = "parallel"))]
#[derive(Debug, Clone, Default)]
struct ParallelConfig;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::Array1;
use std::marker::PhantomData;

/// Fluent builder for explanation configurations
#[derive(Debug, Clone)]
pub struct ExplanationBuilder<T> {
    target_type: PhantomData<T>,
    n_samples: Option<usize>,
    n_features: Option<usize>,
    random_state: Option<u64>,
    parallel_config: ParallelConfig,
    validation_enabled: bool,
    preprocessing_enabled: bool,
    postprocessing_enabled: bool,
}

impl<T> Default for ExplanationBuilder<T> {
    fn default() -> Self {
        Self {
            target_type: PhantomData,
            n_samples: None,
            n_features: None,
            random_state: None,
            parallel_config: ParallelConfig::default(),
            validation_enabled: true,
            preprocessing_enabled: false,
            postprocessing_enabled: false,
        }
    }
}

impl<T> ExplanationBuilder<T> {
    /// Create a new explanation builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of samples to use
    pub fn with_n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = Some(n_samples);
        self
    }

    /// Set the number of features
    pub fn with_n_features(mut self, n_features: usize) -> Self {
        self.n_features = Some(n_features);
        self
    }

    /// Set the random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Configure parallel computation
    pub fn with_parallel_config(mut self, config: ParallelConfig) -> Self {
        self.parallel_config = config;
        self
    }

    /// Enable/disable validation
    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.validation_enabled = enabled;
        self
    }

    /// Enable preprocessing
    pub fn with_preprocessing(mut self) -> Self {
        self.preprocessing_enabled = true;
        self
    }

    /// Enable postprocessing
    pub fn with_postprocessing(mut self) -> Self {
        self.postprocessing_enabled = true;
        self
    }

    /// Set the number of threads for parallel computation
    pub fn with_threads(mut self, n_threads: usize) -> Self {
        self.parallel_config = self.parallel_config.with_threads(n_threads);
        self
    }

    /// Force sequential computation
    pub fn sequential(mut self) -> Self {
        self.parallel_config = self.parallel_config.sequential();
        self
    }

    /// Build a SHAP configuration
    pub fn build_shap_config(self) -> ShapConfig {
        ShapConfig {
            n_samples: self.n_samples.unwrap_or(1000),
            random_state: self.random_state,
            parallel_config: self.parallel_config,
            validation_enabled: self.validation_enabled,
        }
    }

    /// Build a LIME configuration
    pub fn build_lime_config(self) -> LimeConfig {
        LimeConfig {
            n_samples: self.n_samples.unwrap_or(5000),
            random_state: self.random_state,
            parallel_config: self.parallel_config,
            kernel_width: 0.75,
            feature_selection: FeatureSelection::Auto,
        }
    }

    /// Build a permutation importance configuration
    pub fn build_permutation_config(self) -> PermutationConfig {
        PermutationConfig {
            n_repeats: self.n_samples.unwrap_or(10),
            random_state: self.random_state,
            parallel_config: self.parallel_config,
            score_function: ScoreFunction::Accuracy,
        }
    }

    /// Build a counterfactual configuration
    pub fn build_counterfactual_config(self) -> CounterfactualConfig {
        CounterfactualConfig {
            max_iterations: self.n_samples.unwrap_or(1000),
            random_state: self.random_state,
            distance_threshold: 0.1,
            optimization_method: OptimizationMethod::GradientDescent,
        }
    }
}

/// Configuration for SHAP explanations
#[derive(Debug, Clone)]
pub struct ShapConfig {
    /// n_samples
    pub n_samples: usize,
    /// random_state
    pub random_state: Option<u64>,
    /// parallel_config
    pub parallel_config: ParallelConfig,
    /// validation_enabled
    pub validation_enabled: bool,
}

/// Configuration for LIME explanations
#[derive(Debug, Clone)]
pub struct LimeConfig {
    /// n_samples
    pub n_samples: usize,
    /// random_state
    pub random_state: Option<u64>,
    /// parallel_config
    pub parallel_config: ParallelConfig,
    /// kernel_width
    pub kernel_width: Float,
    /// feature_selection
    pub feature_selection: FeatureSelection,
}

/// Feature selection method for LIME
#[derive(Debug, Clone)]
pub enum FeatureSelection {
    /// Auto
    Auto,
    /// Lasso
    Lasso,
    /// Forward
    Forward,

    None,
}

/// Configuration for permutation importance
#[derive(Debug, Clone)]
pub struct PermutationConfig {
    /// n_repeats
    pub n_repeats: usize,
    /// random_state
    pub random_state: Option<u64>,
    /// parallel_config
    pub parallel_config: ParallelConfig,
    /// score_function
    pub score_function: ScoreFunction,
}

/// Score function for permutation importance
#[derive(Debug, Clone)]
pub enum ScoreFunction {
    /// Accuracy
    Accuracy,
    /// R2
    R2,
    /// MeanSquaredError
    MeanSquaredError,
    /// MeanAbsoluteError
    MeanAbsoluteError,
}

/// Configuration for counterfactual explanations
#[derive(Debug, Clone)]
pub struct CounterfactualConfig {
    /// max_iterations
    pub max_iterations: usize,
    /// random_state
    pub random_state: Option<u64>,
    /// distance_threshold
    pub distance_threshold: Float,
    /// optimization_method
    pub optimization_method: OptimizationMethod,
}

/// Optimization method for counterfactuals
#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    /// GradientDescent
    GradientDescent,
    /// SimulatedAnnealing
    SimulatedAnnealing,
    /// GeneticAlgorithm
    GeneticAlgorithm,
}

/// Fluent builder for complex explanation pipelines
#[derive(Debug)]
pub struct PipelineBuilder<Input> {
    steps: Vec<PipelineStep>,

    parallel_config: ParallelConfig,
    _input_type: PhantomData<Input>,
}

impl<Input> Default for PipelineBuilder<Input> {
    fn default() -> Self {
        Self {
            steps: Vec::new(),
            parallel_config: ParallelConfig::default(),
            _input_type: PhantomData,
        }
    }
}

impl<Input> PipelineBuilder<Input>
where
    Input: Send + Sync + 'static,
{
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a SHAP explanation step
    pub fn add_shap(mut self, config: ShapConfig) -> Self {
        self.steps.push(PipelineStep::Shap(config));
        self
    }

    /// Add a LIME explanation step
    pub fn add_lime(mut self, config: LimeConfig) -> Self {
        self.steps.push(PipelineStep::Lime(config));
        self
    }

    /// Add a permutation importance step
    pub fn add_permutation(mut self, config: PermutationConfig) -> Self {
        self.steps.push(PipelineStep::Permutation(config));
        self
    }

    /// Add a counterfactual explanation step
    pub fn add_counterfactual(mut self, config: CounterfactualConfig) -> Self {
        self.steps.push(PipelineStep::Counterfactual(config));
        self
    }

    /// Add a custom explanation step
    pub fn add_custom(mut self, name: String) -> Self {
        self.steps.push(PipelineStep::Custom { name });
        self
    }

    /// Add a validation step
    pub fn add_validation(mut self) -> Self {
        self.steps.push(PipelineStep::Validation);
        self
    }

    /// Add a normalization step
    pub fn add_normalization(mut self) -> Self {
        self.steps.push(PipelineStep::Normalization);
        self
    }

    /// Configure parallel execution
    pub fn with_parallel_config(mut self, config: ParallelConfig) -> Self {
        self.parallel_config = config;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> ExplanationPipelineExecutor<Input> {
        ExplanationPipelineExecutor {
            steps: self.steps,
            parallel_config: self.parallel_config,
            _input_type: PhantomData,
        }
    }
}

/// A step in the explanation pipeline
#[derive(Debug, Clone)]
pub enum PipelineStep {
    /// Shap
    Shap(ShapConfig),
    /// Lime
    Lime(LimeConfig),
    /// Permutation
    Permutation(PermutationConfig),
    /// Counterfactual
    Counterfactual(CounterfactualConfig),
    /// Validation
    Validation,
    /// Normalization
    Normalization,
    /// Custom
    Custom { name: String },
}

/// Executor for the explanation pipeline
pub struct ExplanationPipelineExecutor<Input> {
    steps: Vec<PipelineStep>,
    #[allow(dead_code)] // stored for future parallel execution use
    parallel_config: ParallelConfig,
    _input_type: PhantomData<Input>,
}

impl<Input> ExplanationPipelineExecutor<Input>
where
    Input: Send + Sync,
{
    /// Execute the pipeline
    pub fn execute(&self, _input: &Input) -> SklResult<PipelineExecutionResult> {
        let mut results: Vec<Array1<Float>> = Vec::new();
        let mut metadata: Vec<StepMetadata> = Vec::new();

        for (i, step) in self.steps.iter().enumerate() {
            let step_name = format!("Step_{}", i);
            let start_time = std::time::Instant::now();

            // The pipeline executor only stores step *configurations* and an opaque,
            // unconstrained `Input` type. It captures neither a model's prediction
            // function nor the feature matrix / background data that SHAP, LIME,
            // permutation importance, counterfactual search, and arbitrary custom
            // explainers all require. There is therefore no honest way to *compute*
            // these explanations here. Rather than fabricate a plausible-but-fake
            // result (e.g. `Array1::zeros(10)`), we surface an explicit error that
            // tells the caller exactly what infrastructure is missing.
            let result: Result<Array1<Float>, crate::SklearsError> = match step {
                PipelineStep::Shap(_config) => Err(crate::SklearsError::NotImplemented(
                    "SHAP pipeline step requires a model prediction function and background \
                     dataset, which ExplanationPipelineExecutor does not capture. Use the \
                     `shapley` module APIs directly with a fitted model."
                        .to_string(),
                )),
                PipelineStep::Lime(_config) => Err(crate::SklearsError::NotImplemented(
                    "LIME pipeline step requires a model prediction function and a target \
                     instance, which ExplanationPipelineExecutor does not capture. Use the \
                     `local_explanations` module APIs directly with a fitted model."
                        .to_string(),
                )),
                PipelineStep::Permutation(_config) => Err(crate::SklearsError::NotImplemented(
                    "Permutation importance pipeline step requires a model, feature matrix, \
                     and labels, which ExplanationPipelineExecutor does not capture. Use the \
                     `permutation` module APIs directly with a fitted model."
                        .to_string(),
                )),
                PipelineStep::Counterfactual(_config) => Err(crate::SklearsError::NotImplemented(
                    "Counterfactual pipeline step requires a model prediction function and \
                         a query instance, which ExplanationPipelineExecutor does not capture. \
                         Use the `counterfactual` module APIs directly with a fitted model."
                        .to_string(),
                )),
                PipelineStep::Validation => {
                    // Validation step doesn't produce explanations
                    continue;
                }
                PipelineStep::Normalization => {
                    // Normalization step modifies existing results
                    if let Some(last_result) = results.last_mut() {
                        let sum = last_result.sum();
                        if sum != 0.0 {
                            *last_result = last_result.mapv(|x| x / sum);
                        }
                    }
                    continue;
                }
                PipelineStep::Custom { name } => Err(crate::SklearsError::NotImplemented(format!(
                    "Custom pipeline step '{name}' has no registered executor. \
                     ExplanationPipelineExecutor cannot run user-defined explainers without a \
                     bound implementation."
                ))),
            };

            let execution_time = start_time.elapsed();

            match result {
                Ok(explanation) => {
                    results.push(explanation);
                    metadata.push(StepMetadata {
                        step_name,
                        execution_time,
                        success: true,
                        error_message: None,
                    });
                }
                Err(e) => {
                    metadata.push(StepMetadata {
                        step_name,
                        execution_time,
                        success: false,
                        error_message: Some(e.to_string()),
                    });
                    return Err(e);
                }
            }
        }

        Ok(PipelineExecutionResult {
            explanations: results,
            metadata,
        })
    }
}

/// Result of pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineExecutionResult {
    /// explanations
    pub explanations: Vec<Array1<Float>>,
    /// metadata
    pub metadata: Vec<StepMetadata>,
}

/// Metadata for a pipeline step
#[derive(Debug, Clone)]
pub struct StepMetadata {
    /// step_name
    pub step_name: String,
    /// execution_time
    pub execution_time: std::time::Duration,
    /// success
    pub success: bool,
    /// error_message
    pub error_message: Option<String>,
}

/// Builder for explanation comparison studies
#[derive(Debug, Default)]
pub struct ComparisonStudyBuilder {
    methods: Vec<String>,
    datasets: Vec<String>,
    metrics: Vec<String>,
    parallel_config: ParallelConfig,
}

impl ComparisonStudyBuilder {
    /// Create a new comparison study builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an explanation method to compare
    pub fn add_method<S: Into<String>>(mut self, method: S) -> Self {
        self.methods.push(method.into());
        self
    }

    /// Add a dataset to test on
    pub fn add_dataset<S: Into<String>>(mut self, dataset: S) -> Self {
        self.datasets.push(dataset.into());
        self
    }

    /// Add an evaluation metric
    pub fn add_metric<S: Into<String>>(mut self, metric: S) -> Self {
        self.metrics.push(metric.into());
        self
    }

    /// Configure parallel execution
    pub fn with_parallel_config(mut self, config: ParallelConfig) -> Self {
        self.parallel_config = config;
        self
    }

    /// Build the comparison study
    pub fn build(self) -> ComparisonStudy {
        ComparisonStudy {
            methods: self.methods,
            datasets: self.datasets,
            metrics: self.metrics,
            parallel_config: self.parallel_config,
        }
    }
}

/// A comparison study configuration
#[derive(Debug, Clone)]
pub struct ComparisonStudy {
    /// methods
    pub methods: Vec<String>,
    /// datasets
    pub datasets: Vec<String>,
    /// metrics
    pub metrics: Vec<String>,
    /// parallel_config
    pub parallel_config: ParallelConfig,
}

impl ComparisonStudy {
    /// Execute the comparison study.
    ///
    /// A comparison study scores each (method, dataset, metric) triple. The builder,
    /// however, only records the *names* of methods, datasets, and metrics as strings;
    /// it binds neither runnable explainer implementations, loaded datasets, nor metric
    /// evaluators. Computing a real score would require all three. Returning a random
    /// or otherwise fabricated score would silently masquerade as a genuine measurement,
    /// so this method instead reports the missing infrastructure honestly.
    pub fn execute(&self) -> SklResult<ComparisonResults> {
        Err(crate::SklearsError::NotImplemented(format!(
            "ComparisonStudy::execute cannot score {} method(s) x {} dataset(s) x {} metric(s): \
             the study only holds names, not bound explainer implementations, loaded datasets, \
             or metric evaluators. Run each explainer/metric directly against real data to \
             obtain genuine comparison scores.",
            self.methods.len(),
            self.datasets.len(),
            self.metrics.len()
        )))
    }
}

/// Results of a comparison study
#[derive(Debug, Clone)]
pub struct ComparisonResults {
    /// results
    pub results: Vec<ComparisonResult>,
}

/// Individual comparison result
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// method
    pub method: String,
    /// dataset
    pub dataset: String,
    /// metric
    pub metric: String,
    /// score
    pub score: Float,
}

#[cfg(test)]
mod tests {
    use super::*;
    use sklears_core::prelude::ArrayView1;

    #[test]
    fn test_explanation_builder_creation() {
        let builder: ExplanationBuilder<ArrayView1<Float>> = ExplanationBuilder::new();
        assert!(builder.n_samples.is_none());
        assert!(builder.random_state.is_none());
        assert!(builder.validation_enabled);
    }

    #[test]
    fn test_explanation_builder_fluent_api() {
        let builder: ExplanationBuilder<ArrayView1<Float>> = ExplanationBuilder::new()
            .with_n_samples(1000)
            .with_random_state(42)
            .with_threads(4)
            .with_validation(false);

        assert_eq!(builder.n_samples, Some(1000));
        assert_eq!(builder.random_state, Some(42));
        assert!(!builder.validation_enabled);
    }

    #[test]
    fn test_shap_config_building() {
        let config = ExplanationBuilder::<ArrayView1<Float>>::new()
            .with_n_samples(2000)
            .with_random_state(123)
            .build_shap_config();

        assert_eq!(config.n_samples, 2000);
        assert_eq!(config.random_state, Some(123));
        assert!(config.validation_enabled);
    }

    #[test]
    fn test_lime_config_building() {
        let config = ExplanationBuilder::<ArrayView1<Float>>::new()
            .with_n_samples(5000)
            .build_lime_config();

        assert_eq!(config.n_samples, 5000);
        assert_eq!(config.kernel_width, 0.75);
        assert!(matches!(config.feature_selection, FeatureSelection::Auto));
    }

    #[test]
    fn test_pipeline_builder_creation() {
        let builder: PipelineBuilder<ArrayView1<Float>> = PipelineBuilder::new();
        assert_eq!(builder.steps.len(), 0);
    }

    #[test]
    fn test_pipeline_builder_fluent_api() {
        let shap_config = ExplanationBuilder::<ArrayView1<Float>>::new().build_shap_config();
        let lime_config = ExplanationBuilder::<ArrayView1<Float>>::new().build_lime_config();

        let pipeline = PipelineBuilder::<ArrayView1<Float>>::new()
            .add_shap(shap_config)
            .add_lime(lime_config)
            .add_validation()
            .add_normalization()
            .build();

        assert_eq!(pipeline.steps.len(), 4);
    }

    #[test]
    fn test_comparison_study_builder() {
        let study = ComparisonStudyBuilder::new()
            .add_method("SHAP")
            .add_method("LIME")
            .add_dataset("iris")
            .add_dataset("wine")
            .add_metric("fidelity")
            .add_metric("stability")
            .build();

        assert_eq!(study.methods.len(), 2);
        assert_eq!(study.datasets.len(), 2);
        assert_eq!(study.metrics.len(), 2);
    }

    #[test]
    fn test_comparison_study_execution_reports_missing_infrastructure() {
        let study = ComparisonStudyBuilder::new()
            .add_method("SHAP")
            .add_dataset("iris")
            .add_metric("fidelity")
            .build();

        // The study only holds names, so it must NOT fabricate a score. It must report
        // honestly that it cannot compute a real comparison.
        let results = study.execute();
        assert!(matches!(
            results,
            Err(crate::SklearsError::NotImplemented(_))
        ));
    }

    #[test]
    fn test_pipeline_steps_do_not_fabricate_results() {
        // SHAP/LIME/permutation/counterfactual/custom steps have no model or data
        // bound, so executing them must yield an honest error rather than a fake
        // all-zeros explanation.
        let shap_config = ExplanationBuilder::<ArrayView1<Float>>::new().build_shap_config();
        let pipeline = PipelineBuilder::<ArrayView1<Float>>::new()
            .add_shap(shap_config)
            .build();

        let input = scirs2_core::ndarray::array![1.0, 2.0, 3.0];
        let outcome = pipeline.execute(&input.view());
        assert!(matches!(
            outcome,
            Err(crate::SklearsError::NotImplemented(_))
        ));
    }

    #[test]
    fn test_pipeline_normalization_only_is_empty() {
        // A pipeline consisting solely of structural steps (validation/normalization)
        // produces no explanations and must succeed without fabricating any.
        let pipeline = PipelineBuilder::<ArrayView1<Float>>::new()
            .add_validation()
            .add_normalization()
            .build();

        let input = scirs2_core::ndarray::array![1.0, 2.0, 3.0];
        let result = pipeline
            .execute(&input.view())
            .expect("structural-only pipeline should succeed");
        assert!(result.explanations.is_empty());
    }

    #[test]
    fn test_score_function_variants() {
        assert!(matches!(ScoreFunction::Accuracy, ScoreFunction::Accuracy));
        assert!(matches!(ScoreFunction::R2, ScoreFunction::R2));
        assert!(matches!(
            ScoreFunction::MeanSquaredError,
            ScoreFunction::MeanSquaredError
        ));
        assert!(matches!(
            ScoreFunction::MeanAbsoluteError,
            ScoreFunction::MeanAbsoluteError
        ));
    }

    #[test]
    fn test_optimization_method_variants() {
        assert!(matches!(
            OptimizationMethod::GradientDescent,
            OptimizationMethod::GradientDescent
        ));
        assert!(matches!(
            OptimizationMethod::SimulatedAnnealing,
            OptimizationMethod::SimulatedAnnealing
        ));
        assert!(matches!(
            OptimizationMethod::GeneticAlgorithm,
            OptimizationMethod::GeneticAlgorithm
        ));
    }

    #[test]
    fn test_feature_selection_variants() {
        assert!(matches!(FeatureSelection::Auto, FeatureSelection::Auto));
        assert!(matches!(FeatureSelection::Lasso, FeatureSelection::Lasso));
        assert!(matches!(
            FeatureSelection::Forward,
            FeatureSelection::Forward
        ));
        assert!(matches!(FeatureSelection::None, FeatureSelection::None));
    }
}

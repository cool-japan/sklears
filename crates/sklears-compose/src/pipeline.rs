//! Pipeline components
//!
//! Sequential pipeline execution with transformers and final estimator.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError, Transform},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Pipeline step trait for type erasure
pub trait PipelineStep: Send + Sync + std::fmt::Debug {
    /// Transform the input data
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>>;

    /// Fit the step to data (if applicable)
    fn fit(
        &mut self,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<()>;

    /// Clone the step
    fn clone_step(&self) -> Box<dyn PipelineStep>;
}

/// Pipeline predictor trait for final estimators
pub trait PipelinePredictor: Send + Sync + std::fmt::Debug {
    /// Predict on the transformed data
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>>;

    /// Fit the predictor to data
    fn fit(&mut self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<()>;

    /// Clone the predictor
    fn clone_predictor(&self) -> Box<dyn PipelinePredictor>;
}

/// Pipeline
///
/// Sequential application of transformers and a final estimator.
///
/// # Type Parameters
///
/// * `S` - State type (Untrained or `PipelineTrained`)
///
/// # Examples
///
/// ```ignore
/// use sklears_compose::{Pipeline, MockTransformer, MockPredictor};
/// use scirs2_core::ndarray::array;
///
/// let pipeline = Pipeline::builder()
///     .step("scaler", Box::new(MockTransformer::new()))
///     .step("selector", Box::new(MockTransformer::new()))
///     .estimator(Box::new(MockPredictor::new()))
///     .build();
/// ```
pub struct Pipeline<S = Untrained> {
    state: S,
    steps: Vec<(String, Box<dyn PipelineStep>)>,
    final_estimator: Option<Box<dyn PipelinePredictor>>,
    memory: Option<String>,
    verbose: bool,
}

/// Trained state for Pipeline
pub struct PipelineTrained {
    fitted_steps: Vec<(String, Box<dyn PipelineStep>)>,
    fitted_estimator: Option<Box<dyn PipelinePredictor>>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl Pipeline<Untrained> {
    /// Create a new Pipeline instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            steps: Vec::new(),
            final_estimator: None,
            memory: None,
            verbose: false,
        }
    }

    /// Create a pipeline builder
    #[must_use]
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Add a pipeline step
    pub fn add_step(&mut self, name: String, step: Box<dyn PipelineStep>) {
        self.steps.push((name, step));
    }

    /// Set the final estimator
    pub fn set_estimator(&mut self, estimator: Box<dyn PipelinePredictor>) {
        self.final_estimator = Some(estimator);
    }

    /// Set memory caching
    #[must_use]
    pub fn memory(mut self, memory: Option<String>) -> Self {
        self.memory = memory;
        self
    }

    /// Set verbosity
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl Default for Pipeline<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for Pipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for Pipeline<Untrained> {
    type Fitted = Pipeline<PipelineTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let mut current_x = x.to_owned();
        let mut fitted_steps = Vec::new();

        // Fit and transform through all steps
        for (name, mut step) in self.steps {
            step.fit(&current_x.view(), y.as_ref().copied())?;
            let transformed = step.transform(&current_x.view())?;
            current_x = transformed;
            fitted_steps.push((name, step));
        }

        // Fit the final estimator if present
        let mut fitted_estimator = None;
        if let Some(mut estimator) = self.final_estimator {
            if let Some(y_values) = y.as_ref() {
                estimator.fit(&current_x.view(), y_values)?;
                fitted_estimator = Some(estimator);
            }
        }

        Ok(Pipeline {
            state: PipelineTrained {
                fitted_steps,
                fitted_estimator,
                n_features_in: x.ncols(),
                feature_names_in: None,
            },
            steps: Vec::new(), // Move semantics
            final_estimator: None,
            memory: self.memory,
            verbose: self.verbose,
        })
    }
}

impl Pipeline<PipelineTrained> {
    /// Transform data through the fitted pipeline
    pub fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let mut current_x = x.to_owned();

        for (_, step) in &self.state.fitted_steps {
            current_x = step.transform(&current_x.view())?;
        }

        Ok(current_x)
    }

    /// Predict using the fitted pipeline
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let transformed = self.transform(x)?;

        if let Some(estimator) = &self.state.fitted_estimator {
            estimator.predict(&transformed.view())
        } else {
            Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            })
        }
    }

    /// Get the fitted steps
    #[must_use]
    pub fn steps(&self) -> &[(String, Box<dyn PipelineStep>)] {
        &self.state.fitted_steps
    }

    /// Get the fitted estimator
    #[must_use]
    pub fn estimator(&self) -> Option<&Box<dyn PipelinePredictor>> {
        self.state.fitted_estimator.as_ref()
    }
}

/// Pipeline builder for fluent construction
pub struct PipelineBuilder {
    steps: Vec<(String, Box<dyn PipelineStep>)>,
    final_estimator: Option<Box<dyn PipelinePredictor>>,
    memory: Option<String>,
    verbose: bool,
}

impl PipelineBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            final_estimator: None,
            memory: None,
            verbose: false,
        }
    }

    /// Add a pipeline step
    #[must_use]
    pub fn step(mut self, name: &str, step: Box<dyn PipelineStep>) -> Self {
        self.steps.push((name.to_string(), step));
        self
    }

    /// Set the final estimator
    #[must_use]
    pub fn estimator(mut self, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.final_estimator = Some(estimator);
        self
    }

    /// Set memory caching
    #[must_use]
    pub fn memory(mut self, memory: Option<String>) -> Self {
        self.memory = memory;
        self
    }

    /// Set verbosity
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the pipeline
    #[must_use]
    pub fn build(self) -> Pipeline<Untrained> {
        Pipeline {
            state: Untrained,
            steps: self.steps,
            final_estimator: self.final_estimator,
            memory: self.memory,
            verbose: self.verbose,
        }
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

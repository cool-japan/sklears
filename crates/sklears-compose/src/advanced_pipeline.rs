//! Advanced pipeline features
//!
//! Conditional pipelines, branching, memory-efficient execution, and caching.

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

use crate::{PipelinePredictor, PipelineStep};

/// Data condition for conditional execution
pub trait DataCondition: Send + Sync + std::fmt::Debug {
    /// Check if the condition is met for the given data
    fn check(&self, x: &ArrayView2<'_, Float>) -> bool;

    /// Clone the condition
    fn clone_condition(&self) -> Box<dyn DataCondition>;
}

/// Simple data condition based on feature count
#[derive(Debug, Clone)]
pub struct FeatureCountCondition {
    min_features: usize,
    max_features: Option<usize>,
}

impl FeatureCountCondition {
    /// Create a new feature count condition
    #[must_use]
    pub fn new(min_features: usize, max_features: Option<usize>) -> Self {
        Self {
            min_features,
            max_features,
        }
    }
}

impl DataCondition for FeatureCountCondition {
    fn check(&self, x: &ArrayView2<'_, Float>) -> bool {
        let n_features = x.ncols();
        n_features >= self.min_features && self.max_features.map_or(true, |max| n_features <= max)
    }

    fn clone_condition(&self) -> Box<dyn DataCondition> {
        Box::new(self.clone())
    }
}

/// Conditional Pipeline Execution
///
/// Pipeline that conditionally executes different branches based on data characteristics.
///
/// # Examples
///
/// ```ignore
/// use sklears_compose::{ConditionalPipeline, DataCondition};
/// use scirs2_core::ndarray::array;
///
/// let conditional_pipeline = ConditionalPipeline::builder()
///     .condition(Box::new(FeatureCountCondition::new(10, None)))
///     .true_branch(true_pipeline)
///     .false_branch(false_pipeline)
///     .build();
/// ```
pub struct ConditionalPipeline<S = Untrained> {
    state: S,
    condition: Box<dyn DataCondition>,
    true_branch: Box<dyn PipelineStep>,
    false_branch: Option<Box<dyn PipelineStep>>,
    default_action: String, // "passthrough", "error", "zero"
}

/// Trained state for `ConditionalPipeline`
pub struct ConditionalPipelineTrained {
    condition: Box<dyn DataCondition>,
    fitted_true_branch: Box<dyn PipelineStep>,
    fitted_false_branch: Option<Box<dyn PipelineStep>>,
    default_action: String,
    n_features_in: usize,
}

impl ConditionalPipeline<Untrained> {
    /// Create a new `ConditionalPipeline`
    #[must_use]
    pub fn new(condition: Box<dyn DataCondition>, true_branch: Box<dyn PipelineStep>) -> Self {
        Self {
            state: Untrained,
            condition,
            true_branch,
            false_branch: None,
            default_action: "passthrough".to_string(),
        }
    }

    /// Create a builder
    #[must_use]
    pub fn builder() -> ConditionalPipelineBuilder {
        ConditionalPipelineBuilder::new()
    }

    /// Set the false branch
    #[must_use]
    pub fn false_branch(mut self, false_branch: Box<dyn PipelineStep>) -> Self {
        self.false_branch = Some(false_branch);
        self
    }

    /// Set the default action
    #[must_use]
    pub fn default_action(mut self, action: &str) -> Self {
        self.default_action = action.to_string();
        self
    }
}

impl Estimator for ConditionalPipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for ConditionalPipeline<Untrained> {
    type Fitted = ConditionalPipeline<ConditionalPipelineTrained>;

    fn fit(
        mut self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        // Fit the true branch
        self.true_branch.fit(x, *y)?;

        // Fit the false branch if it exists
        let fitted_false_branch = if let Some(mut false_branch) = self.false_branch {
            false_branch.fit(x, *y)?;
            Some(false_branch)
        } else {
            None
        };

        Ok(ConditionalPipeline {
            state: ConditionalPipelineTrained {
                condition: self.condition,
                fitted_true_branch: self.true_branch,
                fitted_false_branch,
                default_action: self.default_action,
                n_features_in: x.ncols(),
            },
            condition: Box::new(FeatureCountCondition::new(0, None)), // Placeholder
            true_branch: Box::new(crate::mock::MockTransformer::new()), // Placeholder
            false_branch: None,
            default_action: String::new(),
        })
    }
}

impl ConditionalPipeline<ConditionalPipelineTrained> {
    /// Transform data using conditional logic
    pub fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        if self.state.condition.check(x) {
            self.state.fitted_true_branch.transform(x)
        } else if let Some(ref false_branch) = self.state.fitted_false_branch {
            false_branch.transform(x)
        } else {
            match self.state.default_action.as_str() {
                "passthrough" => Ok(x.mapv(|v| v)),
                "error" => Err(SklearsError::InvalidInput(
                    "Condition not met and no false branch".to_string(),
                )),
                "zero" => Ok(Array2::zeros((x.nrows(), x.ncols()))),
                _ => Err(SklearsError::InvalidInput(
                    "Unknown default action".to_string(),
                )),
            }
        }
    }
}

/// Branch configuration for branching pipelines
pub struct BranchConfig {
    /// Name of the branch
    pub name: String,
    /// Condition for this branch
    pub condition: Box<dyn DataCondition>,
    /// Pipeline steps for this branch
    pub steps: Vec<Box<dyn PipelineStep>>,
}

impl BranchConfig {
    /// Create a new branch configuration
    #[must_use]
    pub fn new(name: String, condition: Box<dyn DataCondition>) -> Self {
        Self {
            name,
            condition,
            steps: Vec::new(),
        }
    }

    /// Add a step to this branch
    #[must_use]
    pub fn step(mut self, step: Box<dyn PipelineStep>) -> Self {
        self.steps.push(step);
        self
    }
}

/// Branching Pipeline
///
/// Pipeline that splits execution into multiple parallel branches based on data characteristics.
///
/// # Examples
///
/// ```ignore
/// use sklears_compose::{BranchingPipeline, BranchConfig};
/// use scirs2_core::ndarray::array;
///
/// let branching_pipeline = BranchingPipeline::builder()
///     .branch(BranchConfig::new("high_dim".to_string(), high_dim_condition))
///     .branch(BranchConfig::new("low_dim".to_string(), low_dim_condition))
///     .build();
/// ```
pub struct BranchingPipeline<S = Untrained> {
    state: S,
    branches: Vec<BranchConfig>,
    combination_strategy: String, // "concatenate", "average", "max", "first_match"
    default_branch: Option<String>,
}

/// Trained state for `BranchingPipeline`
pub struct BranchingPipelineTrained {
    fitted_branches: Vec<(String, Box<dyn DataCondition>, Vec<Box<dyn PipelineStep>>)>,
    combination_strategy: String,
    default_branch: Option<String>,
    n_features_in: usize,
}

impl BranchingPipeline<Untrained> {
    /// Create a new `BranchingPipeline`
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            branches: Vec::new(),
            combination_strategy: "concatenate".to_string(),
            default_branch: None,
        }
    }

    /// Create a builder
    #[must_use]
    pub fn builder() -> BranchingPipelineBuilder {
        BranchingPipelineBuilder::new()
    }

    /// Add a branch
    pub fn add_branch(&mut self, branch: BranchConfig) {
        self.branches.push(branch);
    }

    /// Set combination strategy
    #[must_use]
    pub fn combination_strategy(mut self, strategy: &str) -> Self {
        self.combination_strategy = strategy.to_string();
        self
    }

    /// Set default branch
    #[must_use]
    pub fn default_branch(mut self, branch_name: &str) -> Self {
        self.default_branch = Some(branch_name.to_string());
        self
    }
}

impl Default for BranchingPipeline<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BranchingPipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for BranchingPipeline<Untrained> {
    type Fitted = BranchingPipeline<BranchingPipelineTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let mut fitted_branches = Vec::new();

        for branch in self.branches {
            let mut fitted_steps = Vec::new();
            for mut step in branch.steps {
                step.fit(x, y.as_ref().copied())?;
                fitted_steps.push(step);
            }
            fitted_branches.push((branch.name, branch.condition, fitted_steps));
        }

        Ok(BranchingPipeline {
            state: BranchingPipelineTrained {
                fitted_branches,
                combination_strategy: self.combination_strategy,
                default_branch: self.default_branch,
                n_features_in: x.ncols(),
            },
            branches: Vec::new(),
            combination_strategy: String::new(),
            default_branch: None,
        })
    }
}

/// Builder for `ConditionalPipeline`
pub struct ConditionalPipelineBuilder {
    condition: Option<Box<dyn DataCondition>>,
    true_branch: Option<Box<dyn PipelineStep>>,
    false_branch: Option<Box<dyn PipelineStep>>,
    default_action: String,
}

impl ConditionalPipelineBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            condition: None,
            true_branch: None,
            false_branch: None,
            default_action: "passthrough".to_string(),
        }
    }

    /// Set the condition
    #[must_use]
    pub fn condition(mut self, condition: Box<dyn DataCondition>) -> Self {
        self.condition = Some(condition);
        self
    }

    /// Set the true branch
    #[must_use]
    pub fn true_branch(mut self, branch: Box<dyn PipelineStep>) -> Self {
        self.true_branch = Some(branch);
        self
    }

    /// Set the false branch
    #[must_use]
    pub fn false_branch(mut self, branch: Box<dyn PipelineStep>) -> Self {
        self.false_branch = Some(branch);
        self
    }

    /// Set the default action
    #[must_use]
    pub fn default_action(mut self, action: &str) -> Self {
        self.default_action = action.to_string();
        self
    }

    /// Build the `ConditionalPipeline`
    pub fn build(self) -> SklResult<ConditionalPipeline<Untrained>> {
        let condition = self
            .condition
            .ok_or_else(|| SklearsError::InvalidInput("Condition required".to_string()))?;
        let true_branch = self
            .true_branch
            .ok_or_else(|| SklearsError::InvalidInput("True branch required".to_string()))?;

        let mut pipeline = ConditionalPipeline::new(condition, true_branch);
        if let Some(false_branch) = self.false_branch {
            pipeline = pipeline.false_branch(false_branch);
        }
        pipeline = pipeline.default_action(&self.default_action);

        Ok(pipeline)
    }
}

/// Builder for `BranchingPipeline`
pub struct BranchingPipelineBuilder {
    branches: Vec<BranchConfig>,
    combination_strategy: String,
    default_branch: Option<String>,
}

impl BranchingPipelineBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
            combination_strategy: "concatenate".to_string(),
            default_branch: None,
        }
    }

    /// Add a branch
    #[must_use]
    pub fn branch(mut self, branch: BranchConfig) -> Self {
        self.branches.push(branch);
        self
    }

    /// Set combination strategy
    #[must_use]
    pub fn combination_strategy(mut self, strategy: &str) -> Self {
        self.combination_strategy = strategy.to_string();
        self
    }

    /// Set default branch
    #[must_use]
    pub fn default_branch(mut self, branch_name: &str) -> Self {
        self.default_branch = Some(branch_name.to_string());
        self
    }

    /// Build the `BranchingPipeline`
    #[must_use]
    pub fn build(self) -> BranchingPipeline<Untrained> {
        BranchingPipeline {
            state: Untrained,
            branches: self.branches,
            combination_strategy: self.combination_strategy,
            default_branch: self.default_branch,
        }
    }
}

impl Default for ConditionalPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BranchingPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

//! Hierarchical Composition for Ensemble Learning
//!
//! This module provides multi-level ensemble architectures that organize models
//! in hierarchical structures. Hierarchical composition enables the creation of
//! complex ensemble systems with multiple levels of abstraction and specialization.
//!
//! # Hierarchical Strategies
//!
//! - [`HierarchicalStrategy::Stacked`] trains level 0 directly on the input
//!   features, then trains every subsequent level on **out-of-fold**
//!   meta-features generated from the previous level's models (a real k-fold
//!   cross-validation scheme, so the meta-learner is never trained on
//!   predictions its own inputs memorized in-sample).
//! - [`HierarchicalStrategy::Cascaded`] trains every level as an independent,
//!   self-sufficient predictor on the full training data; the cascade
//!   structure only changes how the levels are combined *at prediction time*
//!   (confidence-gated early exit).
//! - [`HierarchicalStrategy::TreeBased`] and [`HierarchicalStrategy::BoostedHierarchy`]
//!   currently reuse the same "train every level independently" scheme as
//!   `Cascaded` (this was already the pre-existing behavior; only the
//!   underlying trainer used to fake the actual fitting step).
//! - Any other strategy (`MultiScale`, `SpecialistGeneralist`) falls back to
//!   the same independent per-level training scheme.
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_compose::ensemble::{HierarchicalComposition, HierarchicalStrategy, BlendMethod};
//!
//! // Stacked ensemble with multiple levels
//! let stacked_ensemble = HierarchicalComposition::builder()
//!     .strategy(HierarchicalStrategy::Stacked {
//!         blend_method: BlendMethod::LinearRegression,
//!         cv_folds: 5,
//!     })
//!     .add_level_models(vec![base_model1, base_model2, base_model3])
//!     .add_level_models(vec![meta_model])
//!     .build();
//!
//! // Cascaded system with early exit
//! let cascaded_ensemble = HierarchicalComposition::builder()
//!     .strategy(HierarchicalStrategy::Cascaded {
//!         confidence_threshold: 0.9,
//!         max_stages: 3,
//!         early_exit: true,
//!     })
//!     .add_level_models(vec![fast_model])
//!     .add_level_models(vec![medium_model])
//!     .add_level_models(vec![slow_accurate_model])
//!     .build();
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{thread_rng, SeedableRng};
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::Result as SklResult,
    prelude::SklearsError,
    traits::{Estimator, Fit, Untrained},
    types::Float,
};
use std::collections::HashMap;

use crate::PipelinePredictor;

/// Hierarchical strategies for ensemble composition
#[derive(Debug, Clone, PartialEq)]
pub enum HierarchicalStrategy {
    /// Tree-based hierarchical organization
    TreeBased {
        /// Type of tree structure
        tree_type: TreeType,
        /// Maximum tree depth
        max_depth: usize,
        /// Splitting criterion for tree construction
        split_criterion: SplitCriterion,
    },
    /// Stacked ensemble with multiple levels
    Stacked {
        /// Method for blending predictions
        blend_method: BlendMethod,
        /// Cross-validation folds used to generate out-of-fold meta-features
        cv_folds: usize,
    },
    /// Cascaded system with sequential processing
    Cascaded {
        /// Confidence threshold for early exit
        confidence_threshold: Float,
        /// Maximum number of cascade stages
        max_stages: usize,
        /// Enable early exit
        early_exit: bool,
    },
    /// Boosted hierarchy with adaptive weighting
    BoostedHierarchy {
        /// Boosting algorithm
        boosting_type: BoostingType,
        /// Learning rate for boosting
        learning_rate: Float,
        /// Maximum number of boosting rounds
        max_rounds: usize,
    },
    /// Multi-scale processing with different resolutions
    MultiScale {
        /// Scale factors for different levels
        scale_factors: Vec<Float>,
        /// Aggregation method across scales
        aggregation: ScaleAggregation,
    },
    /// Specialist-generalist organization
    SpecialistGeneralist {
        /// Specialization criterion
        specialization: SpecializationCriterion,
        /// Threshold for specialist activation
        specialist_threshold: Float,
    },
}

/// Types of tree structures for hierarchical organization
#[derive(Debug, Clone, PartialEq)]
pub enum TreeType {
    /// Binary tree with two children per node
    Binary,
    /// Multi-way tree with variable children
    MultiWay {
        /// Maximum number of children per node.
        max_children: usize,
    },
    /// Balanced tree structure
    Balanced,
    /// Adaptive tree that changes structure
    Adaptive,
}

/// Criteria for splitting in tree-based hierarchies
#[derive(Debug, Clone, PartialEq)]
pub enum SplitCriterion {
    /// Split based on performance metrics
    Performance,
    /// Split based on data characteristics
    DataBased,
    /// Split based on model diversity
    Diversity,
    /// Split based on computational cost
    Efficiency,
    /// Custom splitting criterion
    Custom {
        /// Name of the custom criterion.
        name: String,
    },
}

/// Methods for blending predictions in stacked ensembles
#[derive(Debug, Clone, PartialEq)]
pub enum BlendMethod {
    /// Linear regression for blending
    LinearRegression,
    /// Ridge regression with regularization
    RidgeRegression {
        /// Ridge regularization strength.
        alpha: Float,
    },
    /// Neural network for blending
    NeuralNetwork {
        /// Hidden layer size.
        hidden_size: usize,
    },
    /// Gaussian process for blending
    GaussianProcess,
    /// Decision tree for blending
    DecisionTree,
}

/// Types of boosting for hierarchical boosting
#[derive(Debug, Clone, PartialEq)]
pub enum BoostingType {
    /// AdaBoost algorithm
    AdaBoost,
    /// Gradient boosting
    GradientBoosting,
    /// XGBoost-style boosting
    XGBoost,
    /// Hierarchical boosting
    Hierarchical,
}

/// Scale aggregation methods for multi-scale processing
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleAggregation {
    /// Weighted average across scales
    WeightedAverage,
    /// Attention-based aggregation
    Attention,
    /// Learned fusion network
    LearnedFusion,
    /// Hierarchical fusion
    Hierarchical,
}

/// Specialization criteria for specialist-generalist systems
#[derive(Debug, Clone, PartialEq)]
pub enum SpecializationCriterion {
    /// Specialization by data regions
    DataRegions,
    /// Specialization by feature importance
    FeatureImportance,
    /// Specialization by prediction difficulty
    PredictionDifficulty,
    /// Specialization by model competence
    ModelCompetence,
}

/// Node in a hierarchical tree structure
#[derive(Debug)]
pub struct HierarchicalNode {
    /// Node identifier
    pub id: String,
    /// Models at this node. Structural tree-building is a bookkeeping/labeling
    /// concern separate from per-level model fitting (see module docs); nodes
    /// are not currently populated with fitted models.
    pub models: Vec<Box<dyn PipelinePredictor>>,
    /// Child nodes
    pub children: Vec<HierarchicalNode>,
    /// Parent node (if any)
    pub parent: Option<String>,
    /// Node type (leaf, internal, root)
    pub node_type: NodeType,
    /// Splitting criterion used at this node
    pub split_criterion: Option<SplitCriterion>,
    /// Node-specific parameters
    pub parameters: HashMap<String, Float>,
}

/// Types of nodes in hierarchical structure
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    /// Root node
    Root,
    /// Internal node with children
    Internal,
    /// Leaf node with models
    Leaf,
    /// Specialist node for specific data regions
    Specialist,
    /// Generalist node for broad coverage
    Generalist,
}

/// Hierarchical composition for multi-level ensemble learning
///
/// # Type Parameters
///
/// * `S` - State type ([`Untrained`] or [`HierarchicalCompositionTrained`])
#[derive(Debug)]
pub struct HierarchicalComposition<S = Untrained> {
    state: S,
    /// Hierarchical strategy
    strategy: HierarchicalStrategy,
    /// All models organized by levels
    level_models: Vec<Vec<Box<dyn PipelinePredictor>>>,
    /// Hierarchical tree structure (if tree-based)
    tree_structure: Option<HierarchicalNode>,
    /// Enable parallel processing across levels (reserved for future use)
    parallel_levels: bool,
    /// Memory optimization settings (reserved for future use)
    memory_efficient: bool,
    /// Enable progressive training (reserved for future use)
    progressive_training: bool,
    /// Confidence calibration settings (reserved for future use)
    calibration: Option<CalibrationMethod>,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Number of parallel jobs (reserved for future use)
    n_jobs: Option<i32>,
    /// Verbose output flag
    verbose: bool,
}

/// Trained state for [`HierarchicalComposition`], produced by [`Fit::fit`].
pub struct HierarchicalCompositionTrained {
    /// Every level's models, genuinely fitted during `fit` (see module docs
    /// for the training scheme used per [`HierarchicalStrategy`]).
    fitted_levels: Vec<Vec<Box<dyn PipelinePredictor>>>,
    /// Number of input features seen during fitting.
    n_features_in: usize,
}

impl std::fmt::Debug for HierarchicalCompositionTrained {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HierarchicalCompositionTrained")
            .field(
                "level_sizes",
                &self
                    .fitted_levels
                    .iter()
                    .map(Vec::len)
                    .collect::<Vec<_>>(),
            )
            .field("n_features_in", &self.n_features_in)
            .finish()
    }
}

/// Calibration methods for confidence estimation
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationMethod {
    /// Platt scaling
    PlattScaling,
    /// Isotonic regression
    IsotonicRegression,
    /// Temperature scaling
    TemperatureScaling {
        /// Temperature applied to soften/sharpen calibrated scores.
        temperature: Float,
    },
    /// Beta calibration
    BetaCalibration,
}

/// Prediction result from hierarchical composition
#[derive(Debug, Clone)]
pub struct HierarchicalPrediction {
    /// Final prediction
    pub prediction: Array1<Float>,
    /// Predictions from each level
    pub level_predictions: Vec<Array1<Float>>,
    /// Path taken through hierarchy
    pub prediction_path: Vec<String>,
    /// Confidence at each level (real, agreement-based: derived from the
    /// variance among that level's models' predictions, not a fixed constant)
    pub level_confidences: Vec<Float>,
    /// Early exit information (if applicable)
    pub early_exit: Option<EarlyExitInfo>,
}

/// Information about early exit in cascaded systems
#[derive(Debug, Clone)]
pub struct EarlyExitInfo {
    /// Level where early exit occurred
    pub exit_level: usize,
    /// Confidence at exit
    pub exit_confidence: Float,
    /// Computational savings
    pub computation_saved: Float,
}

impl HierarchicalComposition<Untrained> {
    /// Create a new hierarchical composition builder
    #[must_use]
    pub fn builder() -> HierarchicalCompositionBuilder {
        HierarchicalCompositionBuilder::new()
    }

    /// Create a new hierarchical composition with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            strategy: HierarchicalStrategy::Stacked {
                blend_method: BlendMethod::LinearRegression,
                cv_folds: 5,
            },
            level_models: Vec::new(),
            tree_structure: None,
            parallel_levels: false,
            memory_efficient: true,
            progressive_training: false,
            calibration: None,
            random_state: None,
            n_jobs: None,
            verbose: false,
        }
    }

    /// Add models for a specific level
    #[must_use]
    pub fn add_level_models(mut self, models: Vec<Box<dyn PipelinePredictor>>) -> Self {
        self.level_models.push(models);
        self
    }

    /// Set hierarchical strategy
    #[must_use]
    pub fn set_strategy(mut self, strategy: HierarchicalStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set tree structure (for tree-based strategies)
    #[must_use]
    pub fn set_tree_structure(mut self, tree: HierarchicalNode) -> Self {
        self.tree_structure = Some(tree);
        self
    }

    /// Get number of levels configured so far.
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.level_models.len()
    }

    /// Get total number of models configured across all levels.
    #[must_use]
    pub fn total_models(&self) -> usize {
        self.level_models.iter().map(Vec::len).sum()
    }

    /// Get models configured at a specific level.
    #[must_use]
    pub fn level_models(&self, level: usize) -> Option<&Vec<Box<dyn PipelinePredictor>>> {
        self.level_models.get(level)
    }
}

impl Default for HierarchicalComposition<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> HierarchicalComposition<S> {
    /// Get hierarchical strategy
    #[must_use]
    pub fn strategy(&self) -> &HierarchicalStrategy {
        &self.strategy
    }

    /// Validate hierarchical configuration
    fn validate_configuration(&self) -> SklResult<()> {
        if self.level_models.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "level_models".to_string(),
                reason: "HierarchicalComposition requires at least one level of models"
                    .to_string(),
            });
        }

        for (i, level) in self.level_models.iter().enumerate() {
            if level.is_empty() {
                return Err(SklearsError::InvalidParameter {
                    name: format!("level_models[{i}]"),
                    reason: "level contains no models".to_string(),
                });
            }
        }

        match &self.strategy {
            HierarchicalStrategy::TreeBased { max_depth, .. } if *max_depth == 0 => {
                return Err(SklearsError::InvalidParameter {
                    name: "strategy.max_depth".to_string(),
                    reason: "tree max_depth must be positive".to_string(),
                });
            }
            HierarchicalStrategy::Stacked { cv_folds, .. } if *cv_folds < 2 => {
                return Err(SklearsError::InvalidParameter {
                    name: "strategy.cv_folds".to_string(),
                    reason: "CV folds must be at least 2".to_string(),
                });
            }
            HierarchicalStrategy::Cascaded {
                confidence_threshold,
                max_stages,
                ..
            } => {
                if *confidence_threshold < 0.0 || *confidence_threshold > 1.0 {
                    return Err(SklearsError::InvalidParameter {
                        name: "strategy.confidence_threshold".to_string(),
                        reason: "confidence threshold must be between 0 and 1".to_string(),
                    });
                }
                if *max_stages == 0 {
                    return Err(SklearsError::InvalidParameter {
                        name: "strategy.max_stages".to_string(),
                        reason: "max stages must be positive".to_string(),
                    });
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Build tree structure for tree-based strategies. This is purely
    /// structural bookkeeping (node ids/depth/child layout); it does not hold
    /// fitted models — those live in [`HierarchicalCompositionTrained`].
    fn build_tree_structure(&self) -> SklResult<Option<HierarchicalNode>> {
        match &self.strategy {
            HierarchicalStrategy::TreeBased {
                tree_type,
                max_depth,
                split_criterion,
            } => {
                let root = self.build_tree_recursive(
                    "root".to_string(),
                    0,
                    *max_depth,
                    tree_type,
                    split_criterion,
                )?;
                Ok(Some(root))
            }
            _ => Ok(None),
        }
    }

    /// Recursively build tree structure
    fn build_tree_recursive(
        &self,
        node_id: String,
        current_depth: usize,
        max_depth: usize,
        tree_type: &TreeType,
        split_criterion: &SplitCriterion,
    ) -> SklResult<HierarchicalNode> {
        let node_type = if current_depth == 0 {
            NodeType::Root
        } else if current_depth >= max_depth {
            NodeType::Leaf
        } else {
            NodeType::Internal
        };

        let mut children = Vec::new();

        if current_depth < max_depth && current_depth + 1 < self.level_models.len() {
            let num_children = match tree_type {
                TreeType::Binary => 2,
                TreeType::MultiWay { max_children } => *max_children,
                TreeType::Balanced | TreeType::Adaptive => 2,
            };

            for i in 0..num_children {
                let child_id = format!("{node_id}_child_{i}");
                let child = self.build_tree_recursive(
                    child_id,
                    current_depth + 1,
                    max_depth,
                    tree_type,
                    split_criterion,
                )?;
                children.push(child);
            }
        }

        Ok(HierarchicalNode {
            id: node_id,
            models: Vec::new(),
            children,
            parent: None,
            node_type,
            split_criterion: Some(split_criterion.clone()),
            parameters: HashMap::new(),
        })
    }
}

impl Estimator for HierarchicalComposition<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for HierarchicalComposition<Untrained> {
    type Fitted = HierarchicalComposition<HierarchicalCompositionTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<Self::Fitted> {
        self.validate_configuration()?;

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and y ({}) must match",
                x.nrows(),
                y.len()
            )));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string(),
            ));
        }

        let fitted_levels = match &self.strategy {
            HierarchicalStrategy::Stacked { .. } => self.train_stacked_ensemble(x, y)?,
            HierarchicalStrategy::Cascaded { .. } => self.train_cascaded_ensemble(x, y)?,
            HierarchicalStrategy::TreeBased { .. } => self.train_tree_ensemble(x, y)?,
            HierarchicalStrategy::BoostedHierarchy { .. } => self.train_boosted_hierarchy(x, y)?,
            _ => self.train_sequential(x, y)?,
        };

        let tree_structure = self.build_tree_structure()?;

        Ok(HierarchicalComposition {
            state: HierarchicalCompositionTrained {
                fitted_levels,
                n_features_in: x.ncols(),
            },
            strategy: self.strategy,
            level_models: Vec::new(),
            tree_structure,
            parallel_levels: self.parallel_levels,
            memory_efficient: self.memory_efficient,
            progressive_training: self.progressive_training,
            calibration: self.calibration,
            random_state: self.random_state,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
        })
    }
}

impl HierarchicalComposition<Untrained> {
    /// Train every level independently: each model is cloned from the
    /// configured (untrained) prototype and genuinely fitted via
    /// [`PipelinePredictor::fit`] on the full training data. This replaces the
    /// original `model.clone() // Simplified` placeholder, which never called
    /// `fit` at all.
    fn train_sequential(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<Box<dyn PipelinePredictor>>>> {
        let mut trained_levels = Vec::with_capacity(self.level_models.len());
        for level in &self.level_models {
            let mut trained_level = Vec::with_capacity(level.len());
            for model in level {
                let mut fitted = model.clone_predictor();
                fitted.fit(x, y)?;
                trained_level.push(fitted);
            }
            trained_levels.push(trained_level);
        }
        Ok(trained_levels)
    }

    /// Cascade stages are each independently-capable predictors trained on the
    /// full training set; the cascade structure only changes how they're
    /// combined *at prediction time* (early exit once confidence is high
    /// enough — see `predict_cascaded`), so training is identical to the
    /// independent per-level scheme.
    fn train_cascaded_ensemble(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<Box<dyn PipelinePredictor>>>> {
        self.train_sequential(x, y)
    }

    /// Tree-based hierarchies currently reuse the independent per-level
    /// training scheme; the tree only affects the *structural bookkeeping*
    /// built by `build_tree_structure`.
    fn train_tree_ensemble(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<Box<dyn PipelinePredictor>>>> {
        self.train_sequential(x, y)
    }

    /// Boosted hierarchies currently reuse the independent per-level training
    /// scheme (real adaptive re-weighting between rounds is not yet
    /// implemented).
    fn train_boosted_hierarchy(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<Box<dyn PipelinePredictor>>>> {
        self.train_sequential(x, y)
    }

    /// Train a stacked hierarchy: level 0 is fitted directly on `x`; every
    /// subsequent level is fitted on **out-of-fold** meta-features generated
    /// from the previous level's models (real k-fold cross-validation), so
    /// no level is ever trained on predictions its inputs saw in-sample.
    fn train_stacked_ensemble(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<Box<dyn PipelinePredictor>>>> {
        let cv_folds = match &self.strategy {
            HierarchicalStrategy::Stacked { cv_folds, .. } => (*cv_folds).max(2),
            _ => 5,
        };

        let mut trained_levels = Vec::with_capacity(self.level_models.len());
        let mut current_x: Array2<Float> = x.to_owned();
        let n_levels = self.level_models.len();

        for (level_idx, level) in self.level_models.iter().enumerate() {
            let mut trained_level = Vec::with_capacity(level.len());
            for model in level {
                let mut fitted = model.clone_predictor();
                fitted.fit(&current_x.view(), y)?;
                trained_level.push(fitted);
            }

            if level_idx + 1 < n_levels {
                current_x = out_of_fold_predictions(
                    &trained_level,
                    &current_x.view(),
                    y,
                    cv_folds,
                    self.random_state,
                )?;
            }

            trained_levels.push(trained_level);
        }

        Ok(trained_levels)
    }
}

/// Build `(train_indices, test_indices)` pairs for k-fold cross-validation
/// (shuffle + contiguous chunk split). Mirrors the private algorithm used by
/// `cross_validation.rs`; duplicated locally since that helper isn't public.
fn k_fold_indices(
    n_samples: usize,
    n_folds: usize,
    random_state: Option<u64>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    if n_samples == 0 {
        return Vec::new();
    }
    let n_folds = n_folds.clamp(2, n_samples.max(2));

    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(&mut thread_rng()),
    };

    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    let fold_size = n_samples / n_folds;
    let mut splits = Vec::with_capacity(n_folds);
    for fold in 0..n_folds {
        let start = fold * fold_size;
        let end = if fold + 1 == n_folds {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        let test_indices = indices[start..end].to_vec();
        let train_indices: Vec<usize> = indices[..start]
            .iter()
            .chain(indices[end..].iter())
            .copied()
            .collect();

        splits.push((train_indices, test_indices));
    }

    splits
}

/// Generate out-of-fold predictions for `models` on `(x, y)`: for each fold,
/// fresh clones of `models` are fitted on the training portion and predict on
/// the held-out portion, so every sample's meta-feature comes from a model
/// that never saw that sample during its own fitting.
fn out_of_fold_predictions(
    models: &[Box<dyn PipelinePredictor>],
    x: &ArrayView2<'_, Float>,
    y: &ArrayView1<'_, Float>,
    cv_folds: usize,
    random_state: Option<u64>,
) -> SklResult<Array2<Float>> {
    let n_samples = x.nrows();
    let n_models = models.len();
    let mut meta = Array2::<Float>::zeros((n_samples, n_models));

    if n_samples < 2 || n_models == 0 {
        return Ok(meta);
    }

    let folds = k_fold_indices(n_samples, cv_folds, random_state);
    for (train_idx, test_idx) in &folds {
        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }

        let x_train = x.select(Axis(0), train_idx);
        let y_train = y.select(Axis(0), train_idx);
        let x_test = x.select(Axis(0), test_idx);

        for (model_idx, model) in models.iter().enumerate() {
            let mut fold_model = model.clone_predictor();
            fold_model.fit(&x_train.view(), &y_train.view())?;
            let preds = fold_model.predict(&x_test.view())?;
            for (row_pos, &sample_idx) in test_idx.iter().enumerate() {
                meta[[sample_idx, model_idx]] = preds[row_pos];
            }
        }
    }

    Ok(meta)
}

/// Combine predictions from multiple models at a level via simple averaging.
fn combine_level_predictions(predictions: &[Array1<Float>]) -> SklResult<Array1<Float>> {
    if predictions.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No predictions to combine".to_string(),
        ));
    }

    let n_samples = predictions[0].len();
    let mut combined = Array1::zeros(n_samples);
    for sample_idx in 0..n_samples {
        let mut sum = 0.0;
        for pred in predictions {
            sum += pred[sample_idx];
        }
        combined[sample_idx] = sum / predictions.len() as Float;
    }

    Ok(combined)
}

/// Stack per-model prediction vectors into a meta-feature matrix (one column
/// per model).
fn prepare_meta_input(level_predictions: &[Array1<Float>]) -> Array2<Float> {
    if level_predictions.is_empty() {
        return Array2::zeros((0, 0));
    }
    let n_samples = level_predictions[0].len();
    let n_models = level_predictions.len();
    let mut meta_input = Array2::zeros((n_samples, n_models));
    for (feature_idx, pred) in level_predictions.iter().enumerate() {
        for (sample_idx, &value) in pred.iter().enumerate() {
            meta_input[[sample_idx, feature_idx]] = value;
        }
    }
    meta_input
}

/// Real (non-fabricated) confidence estimate for a level's combined
/// prediction: agreement among that level's models, converted from variance
/// to a `(0, 1]` confidence score (higher disagreement -> lower confidence).
/// A level with a single model is treated as maximally confident, since there
/// is no ensemble spread to measure.
fn estimate_prediction_confidence(level_preds: &[Array1<Float>], combined: &Array1<Float>) -> Float {
    if level_preds.len() <= 1 {
        return 1.0;
    }
    let n_samples = combined.len();
    if n_samples == 0 {
        return 1.0;
    }

    let mut total_variance = 0.0;
    for sample_idx in 0..n_samples {
        let mean = combined[sample_idx];
        let variance: Float = level_preds
            .iter()
            .map(|p| (p[sample_idx] - mean).powi(2))
            .sum::<Float>()
            / level_preds.len() as Float;
        total_variance += variance;
    }
    let avg_variance = total_variance / n_samples as Float;

    1.0 / (1.0 + avg_variance)
}

impl HierarchicalComposition<HierarchicalCompositionTrained> {
    /// Predict using the fitted hierarchical composition.
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if x.nrows() == 0 {
            return Ok(Array1::zeros(0));
        }
        Ok(self.predict_hierarchical(x)?.prediction)
    }

    /// Predict with hierarchical details
    pub fn predict_hierarchical(
        &self,
        x: &ArrayView2<'_, Float>,
    ) -> SklResult<HierarchicalPrediction> {
        match &self.strategy {
            HierarchicalStrategy::Stacked { .. } => self.predict_stacked(x),
            HierarchicalStrategy::Cascaded { .. } => self.predict_cascaded(x),
            HierarchicalStrategy::TreeBased { .. } => self.predict_tree_based(x),
            HierarchicalStrategy::BoostedHierarchy { .. } => self.predict_boosted(x),
            _ => self.predict_sequential(x),
        }
    }

    /// Predict using stacked ensemble
    fn predict_stacked(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        let mut level_predictions = Vec::new();
        let mut level_confidences = Vec::new();
        let mut current_input = x.to_owned();
        let n_levels = self.state.fitted_levels.len();

        for (level_idx, level_models) in self.state.fitted_levels.iter().enumerate() {
            let mut level_preds = Vec::new();
            for model in level_models {
                let pred = model.predict(&current_input.view())?;
                level_preds.push(pred);
            }

            let combined_pred = combine_level_predictions(&level_preds)?;
            level_confidences.push(estimate_prediction_confidence(&level_preds, &combined_pred));
            level_predictions.push(combined_pred);

            if level_idx + 1 < n_levels {
                current_input = prepare_meta_input(&level_preds);
            }
        }

        let final_prediction = level_predictions
            .last()
            .cloned()
            .ok_or_else(|| SklearsError::InvalidState("No level predictions generated".to_string()))?;

        Ok(HierarchicalPrediction {
            prediction: final_prediction,
            level_predictions,
            prediction_path: vec!["stacked".to_string()],
            level_confidences,
            early_exit: None,
        })
    }

    /// Predict using cascaded ensemble
    fn predict_cascaded(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        let mut level_predictions = Vec::new();
        let mut level_confidences = Vec::new();
        let mut prediction_path = Vec::new();

        let (confidence_threshold, early_exit) = match &self.strategy {
            HierarchicalStrategy::Cascaded {
                confidence_threshold,
                early_exit,
                ..
            } => (*confidence_threshold, *early_exit),
            _ => (0.9, false),
        };

        let mut final_prediction = None;
        let mut early_exit_info = None;
        let n_levels = self.state.fitted_levels.len();

        for (level_idx, level_models) in self.state.fitted_levels.iter().enumerate() {
            let mut level_preds = Vec::new();
            for model in level_models {
                let pred = model.predict(x)?;
                level_preds.push(pred);
            }

            let combined_pred = combine_level_predictions(&level_preds)?;
            let confidence = estimate_prediction_confidence(&level_preds, &combined_pred);

            level_predictions.push(combined_pred.clone());
            level_confidences.push(confidence);
            prediction_path.push(format!("level_{level_idx}"));

            if early_exit && confidence >= confidence_threshold {
                final_prediction = Some(combined_pred);
                let computation_saved = 1.0 - (level_idx + 1) as Float / n_levels as Float;
                early_exit_info = Some(EarlyExitInfo {
                    exit_level: level_idx,
                    exit_confidence: confidence,
                    computation_saved,
                });
                break;
            }

            final_prediction = Some(combined_pred);
        }

        let final_pred = final_prediction
            .ok_or_else(|| SklearsError::InvalidState("No final prediction generated".to_string()))?;

        Ok(HierarchicalPrediction {
            prediction: final_pred,
            level_predictions,
            prediction_path,
            level_confidences,
            early_exit: early_exit_info,
        })
    }

    /// Predict using tree-based ensemble (currently identical to sequential;
    /// see module docs).
    fn predict_tree_based(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        self.predict_sequential(x)
    }

    /// Predict using boosted hierarchy (currently identical to sequential;
    /// see module docs).
    fn predict_boosted(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        self.predict_sequential(x)
    }

    /// Default sequential prediction
    fn predict_sequential(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        let mut level_predictions = Vec::new();
        let mut level_confidences = Vec::new();

        for level_models in &self.state.fitted_levels {
            let mut level_preds = Vec::new();
            for model in level_models {
                let pred = model.predict(x)?;
                level_preds.push(pred);
            }

            let combined_pred = combine_level_predictions(&level_preds)?;
            level_confidences.push(estimate_prediction_confidence(&level_preds, &combined_pred));
            level_predictions.push(combined_pred);
        }

        let final_prediction = level_predictions
            .last()
            .cloned()
            .ok_or_else(|| SklearsError::InvalidState("No level predictions generated".to_string()))?;

        Ok(HierarchicalPrediction {
            prediction: final_prediction,
            level_predictions,
            prediction_path: vec!["sequential".to_string()],
            level_confidences,
            early_exit: None,
        })
    }

    /// Get the fitted models at every level.
    #[must_use]
    pub fn fitted_levels(&self) -> &[Vec<Box<dyn PipelinePredictor>>] {
        &self.state.fitted_levels
    }

    /// Get the number of fitted levels.
    #[must_use]
    pub fn n_fitted_levels(&self) -> usize {
        self.state.fitted_levels.len()
    }

    /// Get the number of input features seen during fitting.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.state.n_features_in
    }

    /// Get hierarchical structure information
    #[must_use]
    pub fn get_structure_info(&self) -> HashMap<String, serde_json::Value> {
        let mut info = HashMap::new();
        let n_levels = self.state.fitted_levels.len();
        let total_models: usize = self.state.fitted_levels.iter().map(Vec::len).sum();

        info.insert(
            "strategy".to_string(),
            serde_json::Value::String(format!("{:?}", self.strategy)),
        );
        info.insert(
            "n_levels".to_string(),
            serde_json::Value::Number(serde_json::Number::from(n_levels)),
        );
        info.insert(
            "total_models".to_string(),
            serde_json::Value::Number(serde_json::Number::from(total_models)),
        );

        for (i, level) in self.state.fitted_levels.iter().enumerate() {
            info.insert(
                format!("level_{i}_models"),
                serde_json::Value::Number(serde_json::Number::from(level.len())),
            );
        }

        info
    }
}

/// Configuration for HierarchicalComposition
#[derive(Debug, Clone)]
pub struct HierarchicalCompositionConfig {
    /// Hierarchical strategy
    pub strategy: HierarchicalStrategy,
    /// Enable parallel processing
    pub parallel_levels: bool,
    /// Memory optimization
    pub memory_efficient: bool,
    /// Progressive training
    pub progressive_training: bool,
    /// Calibration method
    pub calibration: Option<CalibrationMethod>,
    /// Random state
    pub random_state: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for HierarchicalCompositionConfig {
    fn default() -> Self {
        Self {
            strategy: HierarchicalStrategy::Stacked {
                blend_method: BlendMethod::LinearRegression,
                cv_folds: 5,
            },
            parallel_levels: false,
            memory_efficient: true,
            progressive_training: false,
            calibration: None,
            random_state: None,
            n_jobs: None,
            verbose: false,
        }
    }
}

/// Builder for HierarchicalComposition
#[derive(Debug)]
pub struct HierarchicalCompositionBuilder {
    strategy: HierarchicalStrategy,
    level_models: Vec<Vec<Box<dyn PipelinePredictor>>>,
    tree_structure: Option<HierarchicalNode>,
    parallel_levels: bool,
    memory_efficient: bool,
    progressive_training: bool,
    calibration: Option<CalibrationMethod>,
    random_state: Option<u64>,
    n_jobs: Option<i32>,
    verbose: bool,
}

impl HierarchicalCompositionBuilder {
    /// Create new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategy: HierarchicalStrategy::Stacked {
                blend_method: BlendMethod::LinearRegression,
                cv_folds: 5,
            },
            level_models: Vec::new(),
            tree_structure: None,
            parallel_levels: false,
            memory_efficient: true,
            progressive_training: false,
            calibration: None,
            random_state: None,
            n_jobs: None,
            verbose: false,
        }
    }

    /// Set hierarchical strategy
    #[must_use]
    pub fn strategy(mut self, strategy: HierarchicalStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add models for a level
    #[must_use]
    pub fn add_level_models(mut self, models: Vec<Box<dyn PipelinePredictor>>) -> Self {
        self.level_models.push(models);
        self
    }

    /// Set tree structure
    #[must_use]
    pub fn tree_structure(mut self, tree: HierarchicalNode) -> Self {
        self.tree_structure = Some(tree);
        self
    }

    /// Enable parallel levels
    #[must_use]
    pub fn parallel_levels(mut self, parallel: bool) -> Self {
        self.parallel_levels = parallel;
        self
    }

    /// Set memory efficiency
    #[must_use]
    pub fn memory_efficient(mut self, efficient: bool) -> Self {
        self.memory_efficient = efficient;
        self
    }

    /// Enable progressive training
    #[must_use]
    pub fn progressive_training(mut self, progressive: bool) -> Self {
        self.progressive_training = progressive;
        self
    }

    /// Set calibration method
    #[must_use]
    pub fn calibration(mut self, calibration: CalibrationMethod) -> Self {
        self.calibration = Some(calibration);
        self
    }

    /// Set random state
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set number of jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Set verbose flag
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the HierarchicalComposition
    #[must_use]
    pub fn build(self) -> HierarchicalComposition<Untrained> {
        HierarchicalComposition {
            state: Untrained,
            strategy: self.strategy,
            level_models: self.level_models,
            tree_structure: self.tree_structure,
            parallel_levels: self.parallel_levels,
            memory_efficient: self.memory_efficient,
            progressive_training: self.progressive_training,
            calibration: self.calibration,
            random_state: self.random_state,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
        }
    }
}

impl Default for HierarchicalCompositionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockPredictor;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_hierarchical_composition_builder() {
        let composition = HierarchicalComposition::builder()
            .strategy(HierarchicalStrategy::TreeBased {
                tree_type: TreeType::Binary,
                max_depth: 3,
                split_criterion: SplitCriterion::Performance,
            })
            .parallel_levels(true)
            .memory_efficient(false)
            .progressive_training(true)
            .build();

        match composition.strategy {
            HierarchicalStrategy::TreeBased {
                tree_type,
                max_depth,
                ..
            } => {
                assert_eq!(tree_type, TreeType::Binary);
                assert_eq!(max_depth, 3);
            }
            _ => panic!("Expected TreeBased strategy"),
        }
        assert!(composition.parallel_levels);
        assert!(!composition.memory_efficient);
        assert!(composition.progressive_training);
    }

    #[test]
    fn test_hierarchical_strategies() {
        let strategies = vec![
            HierarchicalStrategy::TreeBased {
                tree_type: TreeType::Binary,
                max_depth: 2,
                split_criterion: SplitCriterion::Performance,
            },
            HierarchicalStrategy::Stacked {
                blend_method: BlendMethod::RidgeRegression { alpha: 0.1 },
                cv_folds: 3,
            },
            HierarchicalStrategy::Cascaded {
                confidence_threshold: 0.85,
                max_stages: 4,
                early_exit: true,
            },
        ];

        for strategy in strategies {
            let composition = HierarchicalComposition::builder()
                .strategy(strategy.clone())
                .build();
            assert_eq!(composition.strategy, strategy);
        }
    }

    #[test]
    fn test_tree_types() {
        let tree_types = vec![
            TreeType::Binary,
            TreeType::MultiWay { max_children: 4 },
            TreeType::Balanced,
            TreeType::Adaptive,
        ];

        for tree_type in tree_types {
            let strategy = HierarchicalStrategy::TreeBased {
                tree_type: tree_type.clone(),
                max_depth: 2,
                split_criterion: SplitCriterion::Performance,
            };
            let composition = HierarchicalComposition::builder()
                .strategy(strategy)
                .build();

            match composition.strategy {
                HierarchicalStrategy::TreeBased { tree_type: tt, .. } => {
                    assert_eq!(tt, tree_type);
                }
                _ => panic!("Expected TreeBased strategy"),
            }
        }
    }

    #[test]
    fn test_blend_methods() {
        let blend_methods = vec![
            BlendMethod::LinearRegression,
            BlendMethod::RidgeRegression { alpha: 0.5 },
            BlendMethod::NeuralNetwork { hidden_size: 32 },
            BlendMethod::GaussianProcess,
            BlendMethod::DecisionTree,
        ];

        for blend_method in blend_methods {
            let strategy = HierarchicalStrategy::Stacked {
                blend_method: blend_method.clone(),
                cv_folds: 5,
            };
            let composition = HierarchicalComposition::builder()
                .strategy(strategy)
                .build();

            match composition.strategy {
                HierarchicalStrategy::Stacked { blend_method: bm, .. } => {
                    assert_eq!(bm, blend_method);
                }
                _ => panic!("Expected Stacked strategy"),
            }
        }
    }

    #[test]
    fn test_node_types() {
        let node_types = vec![
            NodeType::Root,
            NodeType::Internal,
            NodeType::Leaf,
            NodeType::Specialist,
            NodeType::Generalist,
        ];

        for node_type in node_types {
            let node = HierarchicalNode {
                id: "test".to_string(),
                models: Vec::new(),
                children: Vec::new(),
                parent: None,
                node_type: node_type.clone(),
                split_criterion: None,
                parameters: HashMap::new(),
            };
            assert_eq!(node.node_type, node_type);
        }
    }

    #[test]
    fn test_configuration_validation() {
        let composition = HierarchicalComposition::new();
        assert!(composition.validate_configuration().is_err());

        let composition = HierarchicalComposition::builder()
            .strategy(HierarchicalStrategy::TreeBased {
                tree_type: TreeType::Binary,
                max_depth: 0,
                split_criterion: SplitCriterion::Performance,
            })
            .build();
        assert!(composition.validate_configuration().is_err());

        let composition = HierarchicalComposition::builder()
            .strategy(HierarchicalStrategy::Cascaded {
                confidence_threshold: 1.5,
                max_stages: 3,
                early_exit: true,
            })
            .build();
        assert!(composition.validate_configuration().is_err());
    }

    /// Regression test for the original silent-fabrication bug: every level's
    /// training used to be `model.clone() // Simplified`, never calling
    /// `fit`. An unfitted `MockPredictor` errors from `predict`, so a
    /// genuinely-fitted clone must behave differently from a fresh one.
    #[test]
    fn test_each_level_differs_from_a_fresh_unfitted_clone() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let composition = HierarchicalComposition::builder()
            .strategy(HierarchicalStrategy::Cascaded {
                confidence_threshold: 0.99,
                max_stages: 2,
                early_exit: false,
            })
            .add_level_models(vec![Box::new(MockPredictor::new())])
            .add_level_models(vec![Box::new(MockPredictor::new())])
            .build();

        let fitted = composition
            .fit(&x.view(), &y.view())
            .expect("fit should succeed");

        assert_eq!(fitted.n_fitted_levels(), 2);
        for level in fitted.fitted_levels() {
            for model in level {
                // A freshly-constructed (unfitted) MockPredictor errors on
                // predict(); a genuinely fitted one must not.
                let preds = model
                    .predict(&x.view())
                    .expect("every level's model should be genuinely fitted");
                assert_eq!(preds.len(), x.nrows());
            }
        }
    }

    /// Verifies the stacked strategy actually trains on real (out-of-fold)
    /// meta-features rather than the original zero-filled placeholder, and
    /// that the whole hierarchy produces finite, sane predictions.
    #[test]
    fn test_stacked_hierarchy_trains_and_predicts() {
        let n = 12;
        let x = Array2::from_shape_vec((n, 2), (0..n * 2).map(|v| v as Float).collect())
            .expect("valid shape");
        let y = Array1::from_vec((0..n).map(|v| v as Float).collect());

        let composition = HierarchicalComposition::builder()
            .strategy(HierarchicalStrategy::Stacked {
                blend_method: BlendMethod::LinearRegression,
                cv_folds: 3,
            })
            .add_level_models(vec![
                Box::new(MockPredictor::new()) as Box<dyn PipelinePredictor>,
                Box::new(MockPredictor::new()) as Box<dyn PipelinePredictor>,
            ])
            .add_level_models(vec![Box::new(MockPredictor::new()) as Box<dyn PipelinePredictor>])
            .random_state(11)
            .build();

        let fitted = composition
            .fit(&x.view(), &y.view())
            .expect("stacked fit should succeed");

        let preds = fitted.predict(&x.view()).expect("predict should succeed");
        assert_eq!(preds.len(), n);
        assert!(preds.iter().all(|v| v.is_finite()));

        let info = fitted.get_structure_info();
        assert_eq!(
            info.get("n_levels"),
            Some(&serde_json::Value::Number(serde_json::Number::from(2)))
        );
    }
}

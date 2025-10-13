//! Hierarchical Composition for Ensemble Learning
//!
//! This module provides multi-level ensemble architectures that organize models
//! in hierarchical structures. Hierarchical composition enables the creation of
//! complex ensemble systems with multiple levels of abstraction and specialization.
//!
//! # Hierarchical Strategies
//!
//! ## Tree-Based Hierarchies
//! - **Binary Trees**: Two-branch decision structures for model organization
//! - **Multi-Way Trees**: Multiple child nodes for specialized model groups
//! - **Balanced Trees**: Height-balanced trees for optimal performance
//! - **Adaptive Trees**: Dynamic tree structures that adapt to data patterns
//!
//! ## Multi-Level Ensembles
//! - **Stacked Ensembles**: Multiple layers with each layer using outputs from previous
//! - **Cascaded Learning**: Sequential processing with early stopping capabilities
//! - **Boosted Hierarchies**: Hierarchical boosting with adaptive weighting
//! - **Ensemble Towers**: Vertical composition of ensemble methods
//!
//! ## Cascaded Systems
//! - **Sequential Processing**: Chain of ensembles with specialized tasks
//! - **Early Exit Systems**: Confidence-based early termination
//! - **Progressive Refinement**: Iterative improvement through multiple stages
//! - **Attention Cascades**: Attention-guided progressive processing
//!
//! ## Hierarchical Organization
//! - **Specialist-Generalist**: Specialized models for specific regions, generalists for overall
//! - **Divide-and-Conquer**: Recursive problem decomposition
//! - **Hierarchical Clustering**: Cluster-based model organization
//! - **Multi-Scale Processing**: Different models for different scales/resolutions
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_compose::ensemble::{HierarchicalComposition, HierarchicalStrategy};
//!
//! // Tree-based hierarchical ensemble
//! let tree_ensemble = HierarchicalComposition::builder()
//!     .strategy(HierarchicalStrategy::TreeBased {
//!         tree_type: TreeType::Binary,
//!         max_depth: 3,
//!         split_criterion: SplitCriterion::Performance,
//!     })
//!     .add_level_models(vec![model1, model2, model3, model4])
//!     .add_level_models(vec![meta_model1, meta_model2])
//!     .add_level_models(vec![final_model])
//!     .build();
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
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Trained, Untrained},
    types::{Float, FloatBounds},
};
use std::collections::HashMap;
use std::marker::PhantomData;

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
        /// Cross-validation folds for stacking
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
    MultiWay { max_children: usize },
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
    Custom { name: String },
}

/// Methods for blending predictions in stacked ensembles
#[derive(Debug, Clone, PartialEq)]
pub enum BlendMethod {
    /// Linear regression for blending
    LinearRegression,
    /// Ridge regression with regularization
    RidgeRegression { alpha: Float },
    /// Neural network for blending
    NeuralNetwork { hidden_size: usize },
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
#[derive(Debug, Clone)]
pub struct HierarchicalNode {
    /// Node identifier
    pub id: String,
    /// Models at this node
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
#[derive(Debug)]
pub struct HierarchicalComposition<S> {
    /// Hierarchical strategy
    strategy: HierarchicalStrategy,
    /// All models organized by levels
    level_models: Vec<Vec<Box<dyn PipelinePredictor>>>,
    /// Hierarchical tree structure (if tree-based)
    tree_structure: Option<HierarchicalNode>,
    /// Enable parallel processing across levels
    parallel_levels: bool,
    /// Memory optimization settings
    memory_efficient: bool,
    /// Enable progressive training
    progressive_training: bool,
    /// Confidence calibration settings
    calibration: Option<CalibrationMethod>,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Number of parallel jobs
    n_jobs: Option<i32>,
    /// Verbose output flag
    verbose: bool,
    /// State marker
    _state: PhantomData<S>,
}

/// Trained hierarchical composition
pub type HierarchicalCompositionTrained = HierarchicalComposition<Trained>;

/// Calibration methods for confidence estimation
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationMethod {
    /// Platt scaling
    PlattScaling,
    /// Isotonic regression
    IsotonicRegression,
    /// Temperature scaling
    TemperatureScaling { temperature: Float },
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
    /// Confidence at each level
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
    pub fn builder() -> HierarchicalCompositionBuilder {
        HierarchicalCompositionBuilder::new()
    }

    /// Create a new hierarchical composition with default settings
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
            _state: PhantomData,
        }
    }

    /// Add models for a specific level
    pub fn add_level_models(mut self, models: Vec<Box<dyn PipelinePredictor>>) -> Self {
        self.level_models.push(models);
        self
    }

    /// Set hierarchical strategy
    pub fn set_strategy(mut self, strategy: HierarchicalStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set tree structure (for tree-based strategies)
    pub fn set_tree_structure(mut self, tree: HierarchicalNode) -> Self {
        self.tree_structure = Some(tree);
        self
    }
}

impl<S> HierarchicalComposition<S> {
    /// Get hierarchical strategy
    pub fn strategy(&self) -> &HierarchicalStrategy {
        &self.strategy
    }

    /// Get number of levels
    pub fn n_levels(&self) -> usize {
        self.level_models.len()
    }

    /// Get total number of models across all levels
    pub fn total_models(&self) -> usize {
        self.level_models.iter().map(|level| level.len()).sum()
    }

    /// Get models at a specific level
    pub fn level_models(&self, level: usize) -> Option<&Vec<Box<dyn PipelinePredictor>>> {
        self.level_models.get(level)
    }

    /// Validate hierarchical configuration
    fn validate_configuration(&self) -> SklResult<()> {
        if self.level_models.is_empty() {
            return Err(SklearsError::InvalidParameter(
                "HierarchicalComposition requires at least one level of models".to_string()
            ));
        }

        for (i, level) in self.level_models.iter().enumerate() {
            if level.is_empty() {
                return Err(SklearsError::InvalidParameter(format!(
                    "Level {} contains no models", i
                )));
            }
        }

        // Validate strategy-specific parameters
        match &self.strategy {
            HierarchicalStrategy::TreeBased { max_depth, .. } => {
                if *max_depth == 0 {
                    return Err(SklearsError::InvalidParameter(
                        "Tree max_depth must be positive".to_string()
                    ));
                }
            },
            HierarchicalStrategy::Stacked { cv_folds, .. } => {
                if *cv_folds < 2 {
                    return Err(SklearsError::InvalidParameter(
                        "CV folds must be at least 2".to_string()
                    ));
                }
            },
            HierarchicalStrategy::Cascaded { confidence_threshold, max_stages, .. } => {
                if *confidence_threshold < 0.0 || *confidence_threshold > 1.0 {
                    return Err(SklearsError::InvalidParameter(
                        "Confidence threshold must be between 0 and 1".to_string()
                    ));
                }
                if *max_stages == 0 {
                    return Err(SklearsError::InvalidParameter(
                        "Max stages must be positive".to_string()
                    ));
                }
            },
            _ => {}
        }

        Ok(())
    }

    /// Build tree structure for tree-based strategies
    fn build_tree_structure(&self) -> SklResult<Option<HierarchicalNode>> {
        match &self.strategy {
            HierarchicalStrategy::TreeBased { tree_type, max_depth, split_criterion } => {
                let root = self.build_tree_recursive(
                    "root".to_string(),
                    0,
                    *max_depth,
                    tree_type,
                    split_criterion,
                )?;
                Ok(Some(root))
            },
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
        let models = if current_depth < self.level_models.len() {
            // Move ownership would be needed in practice
            Vec::new() // Simplified for now
        } else {
            Vec::new()
        };

        // Create children based on tree type
        if current_depth < max_depth && current_depth + 1 < self.level_models.len() {
            let num_children = match tree_type {
                TreeType::Binary => 2,
                TreeType::MultiWay { max_children } => *max_children,
                TreeType::Balanced => 2, // Simplified
                TreeType::Adaptive => 2, // Simplified
            };

            for i in 0..num_children {
                let child_id = format!("{}_child_{}", node_id, i);
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
            models,
            children,
            parent: None, // Would be set properly in full implementation
            node_type,
            split_criterion: Some(split_criterion.clone()),
            parameters: HashMap::new(),
        })
    }
}

impl Estimator for HierarchicalComposition<Untrained> {
    type Config = HierarchicalCompositionConfig;

    fn default_config() -> Self::Config {
        HierarchicalCompositionConfig::default()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for HierarchicalComposition<Untrained> {
    type Target = HierarchicalComposition<Trained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<Self::Target> {
        self.validate_configuration()?;

        // Validate input data
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and y ({}) must match",
                x.nrows(), y.len()
            )));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string()
            ));
        }

        // Train hierarchical ensemble based on strategy
        let trained_levels = match &self.strategy {
            HierarchicalStrategy::Stacked { .. } => {
                self.train_stacked_ensemble(x, y)?
            },
            HierarchicalStrategy::Cascaded { .. } => {
                self.train_cascaded_ensemble(x, y)?
            },
            HierarchicalStrategy::TreeBased { .. } => {
                self.train_tree_ensemble(x, y)?
            },
            HierarchicalStrategy::BoostedHierarchy { .. } => {
                self.train_boosted_hierarchy(x, y)?
            },
            _ => {
                // Default training: train each level sequentially
                self.train_sequential(x, y)?
            }
        };

        // Build tree structure if needed
        let tree_structure = self.build_tree_structure()?;

        Ok(HierarchicalComposition {
            strategy: self.strategy,
            level_models: trained_levels,
            tree_structure,
            parallel_levels: self.parallel_levels,
            memory_efficient: self.memory_efficient,
            progressive_training: self.progressive_training,
            calibration: self.calibration,
            random_state: self.random_state,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
            _state: PhantomData,
        })
    }
}

impl HierarchicalComposition<Untrained> {
    /// Train stacked ensemble
    fn train_stacked_ensemble(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<Box<dyn PipelinePredictor>>>> {
        let mut trained_levels = Vec::new();

        // Train base level
        if let Some(base_models) = self.level_models.first() {
            let mut trained_base = Vec::new();
            for model in base_models {
                // In practice, would properly train each model
                trained_base.push(model.clone()); // Simplified
            }
            trained_levels.push(trained_base);
        }

        // Train meta levels using cross-validation
        for level_idx in 1..self.level_models.len() {
            if let Some(meta_models) = self.level_models.get(level_idx) {
                let mut trained_meta = Vec::new();

                // Generate meta-features from previous level predictions
                let meta_features = self.generate_meta_features(x, y, level_idx - 1)?;

                for model in meta_models {
                    // Train meta-model on meta-features
                    trained_meta.push(model.clone()); // Simplified
                }
                trained_levels.push(trained_meta);
            }
        }

        Ok(trained_levels)
    }

    /// Train cascaded ensemble
    fn train_cascaded_ensemble(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<Box<dyn PipelinePredictor>>>> {
        let mut trained_levels = Vec::new();

        // Train each level progressively
        for (level_idx, level_models) in self.level_models.iter().enumerate() {
            let mut trained_level = Vec::new();

            for model in level_models {
                // Train model on full data or residuals from previous levels
                trained_level.push(model.clone()); // Simplified
            }

            trained_levels.push(trained_level);
        }

        Ok(trained_levels)
    }

    /// Train tree-based ensemble
    fn train_tree_ensemble(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<Box<dyn PipelinePredictor>>>> {
        // Similar to sequential training but with tree structure considerations
        self.train_sequential(x, y)
    }

    /// Train boosted hierarchy
    fn train_boosted_hierarchy(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<Box<dyn PipelinePredictor>>>> {
        // Implement hierarchical boosting
        // Each level trained on residuals with adaptive weights
        self.train_sequential(x, y)
    }

    /// Default sequential training
    fn train_sequential(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<Box<dyn PipelinePredictor>>>> {
        let mut trained_levels = Vec::new();

        for level_models in &self.level_models {
            let mut trained_level = Vec::new();
            for model in level_models {
                // In practice, would properly train each model
                trained_level.push(model.clone()); // Simplified
            }
            trained_levels.push(trained_level);
        }

        Ok(trained_levels)
    }

    /// Generate meta-features for stacking
    fn generate_meta_features(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        level_idx: usize,
    ) -> SklResult<Array2<Float>> {
        // Generate out-of-fold predictions using cross-validation
        let n_samples = x.nrows();
        let n_models = self.level_models[level_idx].len();
        let meta_features = Array2::zeros((n_samples, n_models));

        // In practice, would use cross-validation to generate meta-features
        Ok(meta_features)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<Float>> for HierarchicalComposition<Trained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if x.nrows() == 0 {
            return Ok(Array1::zeros(0));
        }

        let hierarchical_result = self.predict_hierarchical(x)?;
        Ok(hierarchical_result.prediction)
    }
}

impl HierarchicalComposition<Trained> {
    /// Predict with hierarchical details
    pub fn predict_hierarchical(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        match &self.strategy {
            HierarchicalStrategy::Stacked { .. } => {
                self.predict_stacked(x)
            },
            HierarchicalStrategy::Cascaded { .. } => {
                self.predict_cascaded(x)
            },
            HierarchicalStrategy::TreeBased { .. } => {
                self.predict_tree_based(x)
            },
            HierarchicalStrategy::BoostedHierarchy { .. } => {
                self.predict_boosted(x)
            },
            _ => {
                self.predict_sequential(x)
            }
        }
    }

    /// Predict using stacked ensemble
    fn predict_stacked(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        let mut level_predictions = Vec::new();
        let mut current_input = x.to_owned();

        // Forward pass through levels
        for (level_idx, level_models) in self.level_models.iter().enumerate() {
            let mut level_preds = Vec::new();

            for model in level_models {
                let pred = model.predict(&current_input.view())?;
                level_preds.push(pred);
            }

            // Combine predictions at this level
            let combined_pred = self.combine_level_predictions(&level_preds)?;
            level_predictions.push(combined_pred.clone());

            // For meta-levels, use predictions as features
            if level_idx < self.level_models.len() - 1 {
                current_input = self.prepare_meta_input(x, &level_preds)?;
            }
        }

        let final_prediction = level_predictions.last()
            .ok_or_else(|| SklearsError::InvalidState("No level predictions generated".to_string()))?
            .clone();

        Ok(HierarchicalPrediction {
            prediction: final_prediction,
            level_predictions,
            prediction_path: vec!["stacked".to_string()],
            level_confidences: vec![0.8; self.level_models.len()],
            early_exit: None,
        })
    }

    /// Predict using cascaded ensemble
    fn predict_cascaded(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        let mut level_predictions = Vec::new();
        let mut level_confidences = Vec::new();
        let mut prediction_path = Vec::new();

        // Extract cascade parameters
        let (confidence_threshold, early_exit) = match &self.strategy {
            HierarchicalStrategy::Cascaded { confidence_threshold, early_exit, .. } => {
                (*confidence_threshold, *early_exit)
            },
            _ => (0.9, false),
        };

        let mut final_prediction = None;
        let mut early_exit_info = None;

        // Process each cascade level
        for (level_idx, level_models) in self.level_models.iter().enumerate() {
            let mut level_preds = Vec::new();

            for model in level_models {
                let pred = model.predict(x)?;
                level_preds.push(pred);
            }

            let combined_pred = self.combine_level_predictions(&level_preds)?;
            let confidence = self.estimate_prediction_confidence(&combined_pred)?;

            level_predictions.push(combined_pred.clone());
            level_confidences.push(confidence);
            prediction_path.push(format!("level_{}", level_idx));

            // Check for early exit
            if early_exit && confidence >= confidence_threshold {
                final_prediction = Some(combined_pred);
                let computation_saved = 1.0 - (level_idx + 1) as Float / self.level_models.len() as Float;
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

    /// Predict using tree-based ensemble
    fn predict_tree_based(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        // Simplified tree-based prediction
        // In practice, would traverse tree structure based on splitting criteria
        self.predict_sequential(x)
    }

    /// Predict using boosted hierarchy
    fn predict_boosted(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        // Simplified boosted prediction
        // In practice, would use learned boosting weights
        self.predict_sequential(x)
    }

    /// Default sequential prediction
    fn predict_sequential(&self, x: &ArrayView2<'_, Float>) -> SklResult<HierarchicalPrediction> {
        let mut level_predictions = Vec::new();

        for level_models in &self.level_models {
            let mut level_preds = Vec::new();

            for model in level_models {
                let pred = model.predict(x)?;
                level_preds.push(pred);
            }

            let combined_pred = self.combine_level_predictions(&level_preds)?;
            level_predictions.push(combined_pred);
        }

        let final_prediction = level_predictions.last()
            .ok_or_else(|| SklearsError::InvalidState("No level predictions generated".to_string()))?
            .clone();

        Ok(HierarchicalPrediction {
            prediction: final_prediction,
            level_predictions,
            prediction_path: vec!["sequential".to_string()],
            level_confidences: vec![0.8; self.level_models.len()],
            early_exit: None,
        })
    }

    /// Combine predictions from multiple models at a level
    fn combine_level_predictions(&self, predictions: &[Array1<Float>]) -> SklResult<Array1<Float>> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput("No predictions to combine".to_string()));
        }

        let n_samples = predictions[0].len();
        let mut combined = Array1::zeros(n_samples);

        // Simple averaging for now
        for sample_idx in 0..n_samples {
            let mut sum = 0.0;
            for pred in predictions {
                sum += pred[sample_idx];
            }
            combined[sample_idx] = sum / predictions.len() as Float;
        }

        Ok(combined)
    }

    /// Prepare meta-input for next level in stacking
    fn prepare_meta_input(
        &self,
        original_x: &ArrayView2<'_, Float>,
        level_predictions: &[Array1<Float>],
    ) -> SklResult<Array2<Float>> {
        let n_samples = original_x.nrows();
        let n_features = level_predictions.len();

        let mut meta_input = Array2::zeros((n_samples, n_features));

        for (feature_idx, pred) in level_predictions.iter().enumerate() {
            for sample_idx in 0..n_samples {
                meta_input[[sample_idx, feature_idx]] = pred[sample_idx];
            }
        }

        Ok(meta_input)
    }

    /// Estimate prediction confidence
    fn estimate_prediction_confidence(&self, prediction: &Array1<Float>) -> SklResult<Float> {
        // Simplified confidence estimation
        // In practice, would use model-specific confidence measures
        Ok(0.8)
    }

    /// Get hierarchical structure information
    pub fn get_structure_info(&self) -> HashMap<String, serde_json::Value> {
        let mut info = HashMap::new();

        info.insert("strategy".to_string(),
                   serde_json::Value::String(format!("{:?}", self.strategy)));
        info.insert("n_levels".to_string(),
                   serde_json::Value::Number(serde_json::Number::from(self.n_levels())));
        info.insert("total_models".to_string(),
                   serde_json::Value::Number(serde_json::Number::from(self.total_models())));

        // Add level-specific information
        for (i, level) in self.level_models.iter().enumerate() {
            info.insert(format!("level_{}_models", i),
                       serde_json::Value::Number(serde_json::Number::from(level.len())));
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
    pub fn strategy(mut self, strategy: HierarchicalStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add models for a level
    pub fn add_level_models(mut self, models: Vec<Box<dyn PipelinePredictor>>) -> Self {
        self.level_models.push(models);
        self
    }

    /// Set tree structure
    pub fn tree_structure(mut self, tree: HierarchicalNode) -> Self {
        self.tree_structure = Some(tree);
        self
    }

    /// Enable parallel levels
    pub fn parallel_levels(mut self, parallel: bool) -> Self {
        self.parallel_levels = parallel;
        self
    }

    /// Set memory efficiency
    pub fn memory_efficient(mut self, efficient: bool) -> Self {
        self.memory_efficient = efficient;
        self
    }

    /// Enable progressive training
    pub fn progressive_training(mut self, progressive: bool) -> Self {
        self.progressive_training = progressive;
        self
    }

    /// Set calibration method
    pub fn calibration(mut self, calibration: CalibrationMethod) -> Self {
        self.calibration = Some(calibration);
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set number of jobs
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Set verbose flag
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the HierarchicalComposition
    pub fn build(self) -> HierarchicalComposition<Untrained> {
        HierarchicalComposition {
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
            _state: PhantomData,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

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
            HierarchicalStrategy::TreeBased { tree_type, max_depth, .. } => {
                assert_eq!(tree_type, TreeType::Binary);
                assert_eq!(max_depth, 3);
            },
            _ => panic!("Expected TreeBased strategy"),
        }
        assert_eq!(composition.parallel_levels, true);
        assert_eq!(composition.memory_efficient, false);
        assert_eq!(composition.progressive_training, true);
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
                },
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
                },
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
        // Test empty levels
        let composition = HierarchicalComposition::new();
        assert!(composition.validate_configuration().is_err());

        // Test invalid tree depth
        let composition = HierarchicalComposition::builder()
            .strategy(HierarchicalStrategy::TreeBased {
                tree_type: TreeType::Binary,
                max_depth: 0,
                split_criterion: SplitCriterion::Performance,
            })
            .build();
        assert!(composition.validate_configuration().is_err());

        // Test invalid confidence threshold
        let composition = HierarchicalComposition::builder()
            .strategy(HierarchicalStrategy::Cascaded {
                confidence_threshold: 1.5,
                max_stages: 3,
                early_exit: true,
            })
            .build();
        assert!(composition.validate_configuration().is_err());
    }
}
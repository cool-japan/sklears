//! Type definitions for AdaBoost ensemble methods

use scirs2_core::ndarray::Array1;
use sklears_core::{
    traits::{Trained, Untrained},
    types::{Float, Int},
};
use std::marker::PhantomData;

/// AdaBoost algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaBoostAlgorithm {
    /// SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss function)
    SAMME,
    /// SAMME.R (SAMME with Real-valued predictions)
    SAMMER,
    /// Gentle AdaBoost
    Gentle,
    /// Discrete AdaBoost
    Discrete,
    /// Real AdaBoost
    RealAdaBoost,
    /// AdaBoost.M1 for multiclass
    M1,
    /// AdaBoost.M2 for multiclass
    M2,
}

/// Split criterion for decision trees
#[derive(Debug, Clone, Copy)]
pub enum SplitCriterion {
    Gini,
    Entropy,
}

/// Configuration for AdaBoost
#[derive(Debug, Clone)]
pub struct AdaBoostConfig {
    pub(crate) n_estimators: usize,
    pub(crate) learning_rate: Float,
    pub(crate) algorithm: AdaBoostAlgorithm,
    pub(crate) random_state: Option<u64>,
}

/// Configuration for LogitBoost
#[derive(Debug, Clone)]
#[allow(dead_code)] // planned API fields for LogitBoost configuration
pub struct LogitBoostConfig {
    pub(crate) n_estimators: usize,
    pub(crate) learning_rate: Float,
    pub(crate) random_state: Option<u64>,
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) tolerance: Float,
    pub(crate) max_iter: usize,
}

// ---------------------------------------------------------------------------
// Internal decision-tree node types (used by both classifier and regressor)
// ---------------------------------------------------------------------------

/// Internal node for classification trees.
#[derive(Debug, Clone)]
pub(crate) enum ClassifierNode {
    Leaf(Int),
    Split {
        feature_index: usize,
        threshold: Float,
        left: Box<ClassifierNode>,
        right: Box<ClassifierNode>,
    },
}

/// Internal node for regression trees.
#[derive(Debug, Clone)]
pub(crate) enum RegressorNode {
    Leaf(Float),
    Split {
        feature_index: usize,
        threshold: Float,
        left: Box<RegressorNode>,
        right: Box<RegressorNode>,
    },
}

/// Trained state held by `DecisionTreeClassifier<Trained>`.
#[derive(Debug, Clone)]
pub struct DecisionTreeClassifierState {
    pub(crate) root: ClassifierNode,
    #[allow(dead_code)]
    pub(crate) n_features: usize,
}

/// Trained state held by `DecisionTreeRegressor<Trained>`.
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressorState {
    pub(crate) root: RegressorNode,
    #[allow(dead_code)]
    pub(crate) n_features: usize,
}

/// Decision tree classifier used as base learner in AdaBoost.
#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier<T> {
    pub(crate) criterion: SplitCriterion,
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) random_state: Option<u64>,
    pub(crate) state: PhantomData<T>,
    /// Populated after fitting; `None` in `Untrained` state.
    pub(crate) tree_: Option<DecisionTreeClassifierState>,
}

/// Decision tree regressor used as base learner in LogitBoost / gradient boosting.
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor<T> {
    pub(crate) criterion: SplitCriterion,
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) random_state: Option<u64>,
    pub(crate) state: PhantomData<T>,
    /// Populated after fitting; `None` in `Untrained` state.
    pub(crate) tree_: Option<DecisionTreeRegressorState>,
}

/// AdaBoost Classifier
///
/// AdaBoost is a meta-algorithm that can be used in conjunction with many other
/// types of learning algorithms to improve performance. The key idea is to fit
/// a sequence of weak learners on repeatedly modified versions of the data.
#[derive(Clone)]
pub struct AdaBoostClassifier<State = Untrained> {
    pub(crate) config: AdaBoostConfig,
    pub(crate) state: PhantomData<State>,
    pub(crate) estimators_: Option<Vec<DecisionTreeClassifier<Trained>>>,
    pub(crate) estimator_weights_: Option<Array1<Float>>,
    pub(crate) estimator_errors_: Option<Array1<Float>>,
    pub(crate) classes_: Option<Array1<Float>>,
    pub(crate) n_classes_: Option<usize>,
    pub(crate) n_features_in_: Option<usize>,
}

/// LogitBoost Classifier
#[derive(Debug, Clone)]
pub struct LogitBoostClassifier<State = Untrained> {
    pub(crate) config: LogitBoostConfig,
    pub(crate) state: PhantomData<State>,
    pub(crate) estimators_: Option<Vec<DecisionTreeRegressor<Trained>>>,
    pub(crate) estimator_weights_: Option<Array1<Float>>,
    pub(crate) classes_: Option<Array1<Float>>,
    pub(crate) n_classes_: Option<usize>,
    pub(crate) n_features_in_: Option<usize>,
    pub(crate) intercept_: Option<Float>,
}

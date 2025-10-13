//! Hierarchical Classification Methods
//!
//! This module provides hierarchical classification strategies that leverage natural class hierarchies
//! or tree structures. It includes nested dichotomies, recursive binary partitioning, and full
//! hierarchical classification with support for various traversal strategies.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

// ============================================
// Nested Dichotomies
// ============================================

/// Strategy for building nested dichotomies tree
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DichotomyStrategy {
    /// Random splits at each node
    Random,
    /// Balanced splits (try to keep equal-sized subtrees)
    Balanced,
    /// Class-distance based splits (cluster similar classes)
    Distance,
}

impl Default for DichotomyStrategy {
    fn default() -> Self {
        Self::Random
    }
}

/// Configuration for Nested Dichotomies Classifier
#[derive(Debug, Clone, Default)]
pub struct NestedDichotomiesConfig {
    /// Strategy for building the dichotomy tree
    pub strategy: DichotomyStrategy,
    /// Random seed for reproducible splits
    pub random_seed: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
}

/// A node in the nested dichotomies tree
#[derive(Debug, Clone)]
pub struct DichotomyNode<T> {
    /// Classes in the left subtree
    pub left_classes: Vec<i32>,
    /// Classes in the right subtree  
    pub right_classes: Vec<i32>,
    /// Binary classifier at this node (if internal node)
    pub classifier: Option<T>,
    /// Left child node
    pub left_child: Option<Box<DichotomyNode<T>>>,
    /// Right child node
    pub right_child: Option<Box<DichotomyNode<T>>>,
    /// Whether this is a leaf node
    pub is_leaf: bool,
    /// Final class prediction (if leaf node)
    pub predicted_class: Option<i32>,
}

impl<T> DichotomyNode<T> {
    /// Create a new leaf node
    pub fn new_leaf(class: i32) -> Self {
        Self {
            left_classes: vec![class],
            right_classes: vec![],
            classifier: None,
            left_child: None,
            right_child: None,
            is_leaf: true,
            predicted_class: Some(class),
        }
    }

    /// Create a new internal node
    pub fn new_internal(left_classes: Vec<i32>, right_classes: Vec<i32>) -> Self {
        Self {
            left_classes,
            right_classes,
            classifier: None,
            left_child: None,
            right_child: None,
            is_leaf: false,
            predicted_class: None,
        }
    }
}

/// Trained data for Nested Dichotomies
#[derive(Debug, Clone)]
pub struct NestedDichotomiesTrainedData<T> {
    /// Root of the dichotomy tree
    pub tree_root: DichotomyNode<T>,
    /// All unique classes
    pub classes: Array1<i32>,
    /// Number of features
    pub n_features: usize,
}

/// Builder for NestedDichotomiesClassifier
#[derive(Debug)]
pub struct NestedDichotomiesBuilder<C> {
    base_estimator: C,
    config: NestedDichotomiesConfig,
}

impl<C> NestedDichotomiesBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: NestedDichotomiesConfig::default(),
        }
    }

    /// Set the dichotomy strategy
    pub fn strategy(mut self, strategy: DichotomyStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the random seed
    pub fn random_seed(mut self, seed: Option<u64>) -> Self {
        self.config.random_seed = seed;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Build the classifier
    pub fn build(self) -> NestedDichotomiesClassifier<C, Untrained> {
        NestedDichotomiesClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            trained_data: None,
            state: PhantomData,
        }
    }
}

/// Nested Dichotomies Classifier
///
/// This strategy recursively partitions the class set into two disjoint subsets
/// and trains a binary classifier at each internal node. At prediction time,
/// it follows the path down the tree based on binary classifier decisions.
///
/// This approach can be more efficient than One-vs-One (requires only n-1 classifiers
/// instead of n(n-1)/2) and often more accurate than One-vs-Rest.
#[derive(Debug)]
pub struct NestedDichotomiesClassifier<C, S = Untrained> {
    base_estimator: C,
    config: NestedDichotomiesConfig,
    trained_data: Option<NestedDichotomiesTrainedData<C>>,
    state: PhantomData<S>,
}

impl<C> NestedDichotomiesClassifier<C, Untrained> {
    /// Create a new NestedDichotomiesClassifier
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: NestedDichotomiesConfig::default(),
            trained_data: None,
            state: PhantomData,
        }
    }

    /// Create a builder for NestedDichotomiesClassifier
    pub fn builder(base_estimator: C) -> NestedDichotomiesBuilder<C> {
        NestedDichotomiesBuilder::new(base_estimator)
    }

    /// Set the dichotomy strategy
    pub fn strategy(mut self, strategy: DichotomyStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the random seed
    pub fn random_seed(mut self, seed: Option<u64>) -> Self {
        self.config.random_seed = seed;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }
}

impl<C: Clone> Clone for NestedDichotomiesClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            trained_data: self.trained_data.clone(),
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for NestedDichotomiesBuilder<C> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
        }
    }
}

impl<C> Estimator for NestedDichotomiesClassifier<C, Untrained> {
    type Config = NestedDichotomiesConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Trained Nested Dichotomies wrapper
#[derive(Debug)]
pub struct TrainedNestedDichotomies<T> {
    /// data
    pub data: NestedDichotomiesTrainedData<T>,
    /// config
    pub config: NestedDichotomiesConfig,
}

// Implementation of training (Fit trait) - simplified for now
impl<C> Fit<Array2<Float>, Array1<i32>> for NestedDichotomiesClassifier<C, Untrained>
where
    C: Clone + Send + Sync,
{
    type Fitted = TrainedNestedDichotomies<C>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_features = x.ncols();

        // Get unique classes
        let mut unique_classes: Vec<i32> = y.iter().copied().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();

        if unique_classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for nested dichotomies".to_string(),
            ));
        }

        // For now, create a simple binary tree structure without actual training
        // This is a placeholder implementation that can be expanded later
        let tree_root = DichotomyNode::new_leaf(unique_classes[0]);

        let trained_data = NestedDichotomiesTrainedData {
            tree_root,
            classes: Array1::from(unique_classes),
            n_features,
        };

        Ok(TrainedNestedDichotomies {
            data: trained_data,
            config: self.config,
        })
    }
}

// Implementation of Predict trait for trained classifier
impl<C> Predict<Array2<Float>, Array1<i32>> for TrainedNestedDichotomies<C>
where
    C: Clone,
{
    fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        let n_samples = x.nrows();

        // For now, just return the first class for all samples
        // This is a placeholder implementation
        let predictions = Array1::from_elem(n_samples, self.data.classes[0]);

        Ok(predictions)
    }
}

// ============================================
// Recursive Binary Partitioning
// ============================================

/// Strategy for selecting which class to partition at each step
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitioningStrategy {
    /// Select classes in order (0, 1, 2, ...)
    Sequential,
    /// Select the class with the most samples
    MostFrequent,
    /// Select the class with the least samples
    LeastFrequent,
    /// Random selection
    Random,
}

impl Default for PartitioningStrategy {
    fn default() -> Self {
        Self::Sequential
    }
}

/// Configuration for Recursive Binary Partitioning Classifier
#[derive(Debug, Clone, Default)]
pub struct RecursiveBinaryPartitioningConfig {
    /// Strategy for selecting which class to partition
    pub strategy: PartitioningStrategy,
    /// Random seed for reproducible partitioning
    pub random_seed: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
}

/// A binary partition in the recursive partitioning tree
#[derive(Debug, Clone)]
pub struct BinaryPartition<T> {
    /// The target class for this partition
    pub target_class: i32,
    /// The binary classifier (target class vs all others)
    pub classifier: Option<T>,
    /// Remaining classes after this partition
    pub remaining_classes: Vec<i32>,
    /// Child partition for remaining classes
    pub child_partition: Option<Box<BinaryPartition<T>>>,
}

impl<T> BinaryPartition<T> {
    /// Create a new leaf partition (single class)
    pub fn new_leaf(class: i32) -> Self {
        Self {
            target_class: class,
            classifier: None,
            remaining_classes: vec![],
            child_partition: None,
        }
    }

    /// Create a new internal partition
    pub fn new_internal(target_class: i32, remaining_classes: Vec<i32>) -> Self {
        Self {
            target_class,
            classifier: None,
            remaining_classes,
            child_partition: None,
        }
    }
}

/// Trained data for Recursive Binary Partitioning
#[derive(Debug, Clone)]
pub struct RecursiveBinaryPartitioningTrainedData<T> {
    /// Root of the partitioning tree
    pub partition_root: BinaryPartition<T>,
    /// All unique classes
    pub classes: Array1<i32>,
    /// Number of features
    pub n_features: usize,
}

/// Recursive Binary Partitioning Classifier
///
/// This strategy recursively partitions the multiclass problem by selecting
/// one class at a time and training a binary classifier to separate it from
/// all remaining classes. The process continues recursively on the remaining
/// classes until all classes are separated.
///
/// This approach requires n-1 binary classifiers for n classes and can be
/// more efficient than One-vs-One while often being more accurate than One-vs-Rest.
#[derive(Debug)]
pub struct RecursiveBinaryPartitioningClassifier<C, S = Untrained> {
    base_estimator: C,
    config: RecursiveBinaryPartitioningConfig,
    trained_data: Option<RecursiveBinaryPartitioningTrainedData<C>>,
    state: PhantomData<S>,
}

impl<C> RecursiveBinaryPartitioningClassifier<C, Untrained> {
    /// Create a new RecursiveBinaryPartitioningClassifier
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: RecursiveBinaryPartitioningConfig::default(),
            trained_data: None,
            state: PhantomData,
        }
    }

    /// Create a builder for RecursiveBinaryPartitioningClassifier
    pub fn builder(base_estimator: C) -> RecursiveBinaryPartitioningBuilder<C> {
        RecursiveBinaryPartitioningBuilder::new(base_estimator)
    }

    /// Set the partitioning strategy
    pub fn strategy(mut self, strategy: PartitioningStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the random seed
    pub fn random_seed(mut self, seed: Option<u64>) -> Self {
        self.config.random_seed = seed;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }
}

/// Builder for RecursiveBinaryPartitioningClassifier
#[derive(Debug)]
pub struct RecursiveBinaryPartitioningBuilder<C> {
    base_estimator: C,
    config: RecursiveBinaryPartitioningConfig,
}

impl<C> RecursiveBinaryPartitioningBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: RecursiveBinaryPartitioningConfig::default(),
        }
    }

    /// Set the partitioning strategy
    pub fn strategy(mut self, strategy: PartitioningStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the random seed
    pub fn random_seed(mut self, seed: Option<u64>) -> Self {
        self.config.random_seed = seed;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Build the classifier
    pub fn build(self) -> RecursiveBinaryPartitioningClassifier<C, Untrained> {
        RecursiveBinaryPartitioningClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            trained_data: None,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for RecursiveBinaryPartitioningClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            trained_data: self.trained_data.clone(),
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for RecursiveBinaryPartitioningBuilder<C> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
        }
    }
}

impl<C> Estimator for RecursiveBinaryPartitioningClassifier<C, Untrained> {
    type Config = RecursiveBinaryPartitioningConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Trained Recursive Binary Partitioning wrapper
#[derive(Debug)]
pub struct TrainedRecursiveBinaryPartitioning<T> {
    /// data
    pub data: RecursiveBinaryPartitioningTrainedData<T>,
    /// config
    pub config: RecursiveBinaryPartitioningConfig,
}

// Implementation of training (Fit trait)
impl<C> Fit<Array2<Float>, Array1<i32>> for RecursiveBinaryPartitioningClassifier<C, Untrained>
where
    C: Clone + Send + Sync,
{
    type Fitted = TrainedRecursiveBinaryPartitioning<C>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_features = x.ncols();

        // Get unique classes
        let mut unique_classes: Vec<i32> = y.iter().copied().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();

        if unique_classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for recursive binary partitioning".to_string(),
            ));
        }

        // Create ordered class list based on strategy
        let ordered_classes = self.order_classes(&unique_classes, y)?;

        // Build the partitioning tree
        let partition_root = self.build_partition_tree(&ordered_classes);

        let trained_data = RecursiveBinaryPartitioningTrainedData {
            partition_root,
            classes: Array1::from(unique_classes),
            n_features,
        };

        Ok(TrainedRecursiveBinaryPartitioning {
            data: trained_data,
            config: self.config,
        })
    }
}

impl<C> RecursiveBinaryPartitioningClassifier<C, Untrained>
where
    C: Clone,
{
    /// Order classes according to the partitioning strategy
    fn order_classes(&self, classes: &[i32], y: &Array1<i32>) -> SklResult<Vec<i32>> {
        match self.config.strategy {
            PartitioningStrategy::Sequential => Ok(classes.to_vec()),
            PartitioningStrategy::MostFrequent => {
                let mut class_counts: Vec<(i32, usize)> = classes
                    .iter()
                    .map(|&class| {
                        let count = y.iter().filter(|&&label| label == class).count();
                        (class, count)
                    })
                    .collect();

                // Sort by count descending, then by class ascending for determinism
                class_counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

                Ok(class_counts.into_iter().map(|(class, _)| class).collect())
            }
            PartitioningStrategy::LeastFrequent => {
                let mut class_counts: Vec<(i32, usize)> = classes
                    .iter()
                    .map(|&class| {
                        let count = y.iter().filter(|&&label| label == class).count();
                        (class, count)
                    })
                    .collect();

                // Sort by count ascending, then by class ascending for determinism
                class_counts.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

                Ok(class_counts.into_iter().map(|(class, _)| class).collect())
            }
            PartitioningStrategy::Random => {
                let mut rng = match self.config.random_seed {
                    Some(seed) => Random::seed(seed),
                    None => Random::seed(42),
                };

                let mut shuffled_classes = classes.to_vec();
                for i in (1..shuffled_classes.len()).rev() {
                    let j = rng.random_range(0, i + 1);
                    shuffled_classes.swap(i, j);
                }

                Ok(shuffled_classes)
            }
        }
    }

    /// Build the partitioning tree recursively
    fn build_partition_tree(&self, classes: &[i32]) -> BinaryPartition<C> {
        // Base case: single class
        if classes.len() == 1 {
            return BinaryPartition::new_leaf(classes[0]);
        }

        // Take the first class as target, rest as remaining
        let target_class = classes[0];
        let remaining_classes: Vec<i32> = classes[1..].to_vec();

        let mut partition = BinaryPartition::new_internal(target_class, remaining_classes.clone());

        // For placeholder implementation, we don't actually train classifiers
        // In a full implementation, we would train a binary classifier here

        // Recursively build partition for remaining classes
        if remaining_classes.len() > 1 {
            partition.child_partition =
                Some(Box::new(self.build_partition_tree(&remaining_classes)));
        } else if remaining_classes.len() == 1 {
            partition.child_partition =
                Some(Box::new(BinaryPartition::new_leaf(remaining_classes[0])));
        }

        partition
    }
}

// Implementation of Predict trait for trained classifier
impl<C> Predict<Array2<Float>, Array1<i32>> for TrainedRecursiveBinaryPartitioning<C>
where
    C: Clone,
{
    fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        let n_samples = x.nrows();

        // For now, just return the first class for all samples
        // This is a placeholder implementation
        let predictions = Array1::from_elem(n_samples, self.data.classes[0]);

        Ok(predictions)
    }
}

// ============================================
// Hierarchical Classification
// ============================================

/// Represents a node in a class hierarchy tree
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchyNode {
    /// The class label for this node
    pub class_id: i32,
    /// Parent node class ID (None for root)
    pub parent: Option<i32>,
    /// Children node class IDs
    pub children: Vec<i32>,
    /// Depth in the hierarchy (0 for root)
    pub depth: usize,
    /// Whether this is a leaf node (actual class)
    pub is_leaf: bool,
}

impl HierarchyNode {
    /// Create a new hierarchy node
    pub fn new(class_id: i32, parent: Option<i32>, depth: usize) -> Self {
        Self {
            class_id,
            parent,
            children: Vec::new(),
            depth,
            is_leaf: true,
        }
    }

    /// Add a child to this node
    pub fn add_child(&mut self, child_id: i32) {
        self.children.push(child_id);
        self.is_leaf = false;
    }

    /// Check if this node is a descendant of another node
    pub fn is_descendant_of(&self, ancestor_id: i32, hierarchy: &ClassHierarchy) -> bool {
        let mut current_parent = self.parent;
        while let Some(parent_id) = current_parent {
            if parent_id == ancestor_id {
                return true;
            }
            current_parent = hierarchy.get_node(parent_id).and_then(|node| node.parent);
        }
        false
    }
}

/// Class hierarchy representation for hierarchical classification
#[derive(Debug, Clone)]
pub struct ClassHierarchy {
    /// All nodes in the hierarchy
    nodes: HashMap<i32, HierarchyNode>,
    /// Root nodes (nodes with no parent)
    roots: Vec<i32>,
    /// Leaf nodes (actual classes)
    leaves: Vec<i32>,
    /// Maximum depth in the hierarchy
    max_depth: usize,
}

impl ClassHierarchy {
    /// Create a new empty class hierarchy
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
            leaves: Vec::new(),
            max_depth: 0,
        }
    }

    /// Add a node to the hierarchy
    pub fn add_node(&mut self, node: HierarchyNode) -> SklResult<()> {
        let class_id = node.class_id;
        let parent = node.parent;
        let depth = node.depth;
        let is_leaf = node.is_leaf;

        // Update parent's children if this node has a parent
        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
                parent_node.add_child(class_id);
                // Remove parent from leaves if it was previously a leaf
                if let Some(pos) = self.leaves.iter().position(|&x| x == parent_id) {
                    self.leaves.remove(pos);
                }
            } else {
                return Err(SklearsError::InvalidInput(format!(
                    "Parent node {} not found",
                    parent_id
                )));
            }
        } else {
            self.roots.push(class_id);
        }

        // Track leaf nodes
        if is_leaf {
            self.leaves.push(class_id);
        }

        // Update max depth
        self.max_depth = self.max_depth.max(depth);

        self.nodes.insert(class_id, node);
        Ok(())
    }

    /// Get a node by class ID
    pub fn get_node(&self, class_id: i32) -> Option<&HierarchyNode> {
        self.nodes.get(&class_id)
    }

    /// Get all leaf nodes (actual classes)
    pub fn get_leaves(&self) -> &[i32] {
        &self.leaves
    }

    /// Get all root nodes
    pub fn get_roots(&self) -> &[i32] {
        &self.roots
    }

    /// Get the path from root to a leaf node
    pub fn get_path_to_leaf(&self, leaf_id: i32) -> Option<Vec<i32>> {
        let mut path = Vec::new();
        let mut current_id = leaf_id;

        // Trace back from leaf to root
        loop {
            path.push(current_id);
            if let Some(node) = self.get_node(current_id) {
                if let Some(parent_id) = node.parent {
                    current_id = parent_id;
                } else {
                    break; // Reached root
                }
            } else {
                return None; // Invalid node
            }
        }

        path.reverse(); // Make it root-to-leaf
        Some(path)
    }

    /// Get all nodes at a specific depth
    pub fn get_nodes_at_depth(&self, depth: usize) -> Vec<i32> {
        self.nodes
            .values()
            .filter(|node| node.depth == depth)
            .map(|node| node.class_id)
            .collect()
    }

    /// Validate the hierarchy structure
    pub fn validate(&self) -> SklResult<()> {
        // Check that all parent references are valid
        for node in self.nodes.values() {
            if let Some(parent_id) = node.parent {
                if !self.nodes.contains_key(&parent_id) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Node {} references invalid parent {}",
                        node.class_id, parent_id
                    )));
                }
            }
        }

        // Check that all children references are valid
        for node in self.nodes.values() {
            for &child_id in &node.children {
                if !self.nodes.contains_key(&child_id) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Node {} references invalid child {}",
                        node.class_id, child_id
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get the maximum depth of the hierarchy
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Get the number of nodes in the hierarchy
    pub fn size(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for ClassHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

/// Hierarchical classification strategy
#[derive(Debug, Clone, PartialEq, Default)]
pub enum HierarchicalStrategy {
    /// Top-down approach: classify at each level of the hierarchy
    #[default]
    TopDown,
    /// Bottom-up approach: classify leaf nodes then aggregate up
    BottomUp,
    /// Flat approach: ignore hierarchy, classify all classes equally
    Flat,
    /// Multi-path approach: use multiple classification paths
    MultiPath,
}

/// Configuration for hierarchical classifier
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    pub strategy: HierarchicalStrategy,
    pub use_probability_threshold: bool,
    pub probability_threshold: f64,
    pub early_stopping: bool,
    pub n_jobs: Option<i32>,
    pub random_state: Option<u64>,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            strategy: HierarchicalStrategy::default(),
            use_probability_threshold: true,
            probability_threshold: 0.5,
            early_stopping: false,
            n_jobs: None,
            random_state: None,
        }
    }
}

/// Hierarchical classifier
///
/// This classifier handles problems where classes have a natural hierarchy
/// or tree structure. It can use different strategies like top-down,
/// bottom-up, or flat classification.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```rust
/// use sklears_multiclass::{HierarchicalClassifier, ClassHierarchy, HierarchyNode, HierarchicalStrategy};
/// use scirs2_autograd::ndarray::array;
///
/// // Create a simple hierarchy: root -> [class1, class2]
/// let mut hierarchy = ClassHierarchy::new();
/// hierarchy.add_node(HierarchyNode::new(0, None, 0)).unwrap(); // root
/// hierarchy.add_node(HierarchyNode::new(1, Some(0), 1)).unwrap(); // class1
/// hierarchy.add_node(HierarchyNode::new(2, Some(0), 1)).unwrap(); // class2
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let hierarchical = HierarchicalClassifier::new(base_classifier, hierarchy)
/// //     .strategy(HierarchicalStrategy::TopDown);
/// ```
#[derive(Debug)]
pub struct HierarchicalClassifier<C, S = Untrained> {
    base_estimator: C,
    hierarchy: ClassHierarchy,
    config: HierarchicalConfig,
    state: PhantomData<S>,
}

impl<C> HierarchicalClassifier<C, Untrained> {
    /// Create a new HierarchicalClassifier
    pub fn new(base_estimator: C, hierarchy: ClassHierarchy) -> SklResult<Self> {
        // Validate the hierarchy
        hierarchy.validate()?;

        Ok(Self {
            base_estimator,
            hierarchy,
            config: HierarchicalConfig::default(),
            state: PhantomData,
        })
    }

    /// Create a builder for HierarchicalClassifier
    pub fn builder(
        base_estimator: C,
        hierarchy: ClassHierarchy,
    ) -> SklResult<HierarchicalBuilder<C>> {
        hierarchy.validate()?;
        Ok(HierarchicalBuilder::new(base_estimator, hierarchy))
    }

    /// Set the hierarchical strategy
    pub fn strategy(mut self, strategy: HierarchicalStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set whether to use probability thresholding
    pub fn use_probability_threshold(mut self, use_threshold: bool) -> Self {
        self.config.use_probability_threshold = use_threshold;
        self
    }

    /// Set the probability threshold
    pub fn probability_threshold(mut self, threshold: f64) -> Self {
        self.config.probability_threshold = threshold;
        self
    }

    /// Enable or disable early stopping
    pub fn early_stopping(mut self, early_stopping: bool) -> Self {
        self.config.early_stopping = early_stopping;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Get a reference to the base estimator
    pub fn base_estimator(&self) -> &C {
        &self.base_estimator
    }

    /// Get a reference to the class hierarchy
    pub fn hierarchy(&self) -> &ClassHierarchy {
        &self.hierarchy
    }
}

/// Builder for HierarchicalClassifier
#[derive(Debug)]
pub struct HierarchicalBuilder<C> {
    base_estimator: C,
    hierarchy: ClassHierarchy,
    config: HierarchicalConfig,
}

impl<C> HierarchicalBuilder<C> {
    /// Create a new builder
    fn new(base_estimator: C, hierarchy: ClassHierarchy) -> Self {
        Self {
            base_estimator,
            hierarchy,
            config: HierarchicalConfig::default(),
        }
    }

    /// Set the hierarchical strategy
    pub fn strategy(mut self, strategy: HierarchicalStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set whether to use probability thresholding
    pub fn use_probability_threshold(mut self, use_threshold: bool) -> Self {
        self.config.use_probability_threshold = use_threshold;
        self
    }

    /// Set the probability threshold
    pub fn probability_threshold(mut self, threshold: f64) -> Self {
        self.config.probability_threshold = threshold;
        self
    }

    /// Enable or disable early stopping
    pub fn early_stopping(mut self, early_stopping: bool) -> Self {
        self.config.early_stopping = early_stopping;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Build the HierarchicalClassifier
    pub fn build(self) -> HierarchicalClassifier<C, Untrained> {
        HierarchicalClassifier {
            base_estimator: self.base_estimator,
            hierarchy: self.hierarchy,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for HierarchicalClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            hierarchy: self.hierarchy.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C> Estimator for HierarchicalClassifier<C, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Trained hierarchical classifier state
#[derive(Debug)]
pub struct TrainedHierarchical<C> {
    /// Classifiers for each level/node in the hierarchy
    node_classifiers: HashMap<i32, C>,
    /// The class hierarchy
    hierarchy: ClassHierarchy,
    /// Configuration used for training
    config: HierarchicalConfig,
    /// Number of features seen during training
    n_features: usize,
}

/// Type alias for trained hierarchical classifier
pub type TrainedHierarchicalClassifier<C> = HierarchicalClassifier<TrainedHierarchical<C>, Trained>;

impl<C> TrainedHierarchical<C> {
    /// Get the class hierarchy
    pub fn hierarchy(&self) -> &ClassHierarchy {
        &self.hierarchy
    }

    /// Get the number of leaf classes
    pub fn n_classes(&self) -> usize {
        self.hierarchy.get_leaves().len()
    }

    /// Get the leaf classes
    pub fn classes(&self) -> &[i32] {
        self.hierarchy.get_leaves()
    }

    /// Get the classifier for a specific node
    pub fn get_node_classifier(&self, node_id: i32) -> Option<&C> {
        self.node_classifiers.get(&node_id)
    }

    /// Get the maximum depth of the hierarchy
    pub fn max_depth(&self) -> usize {
        self.hierarchy.max_depth()
    }
}

// ============================================
// Taxonomy-Aware Classification
// ============================================

/// Taxonomy-aware classification that leverages external taxonomies and ontologies
/// for better multiclass classification. This extends hierarchical classification
/// with support for pre-defined taxonomies like scientific classifications,
/// product categories, or domain-specific ontologies.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```rust
/// use sklears_multiclass::{TaxonomyAwareClassifier, Taxonomy, TaxonomyNode};
/// use scirs2_autograd::ndarray::array;
///
/// // Create a taxonomy for animal classification
/// let mut taxonomy = Taxonomy::new();
/// taxonomy.add_concept(TaxonomyNode::new("Animal", None, vec!["mammal", "reptile"])).unwrap();
/// taxonomy.add_concept(TaxonomyNode::new("Mammal", Some("Animal"), vec!["dog", "cat"])).unwrap();
/// taxonomy.add_concept(TaxonomyNode::new("Reptile", Some("Animal"), vec!["snake", "lizard"])).unwrap();
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let classifier = TaxonomyAwareClassifier::new(base_classifier, taxonomy);
/// ```
#[derive(Debug)]
pub struct TaxonomyAwareClassifier<C, S = Untrained> {
    base_estimator: C,
    taxonomy: Taxonomy,
    config: TaxonomyConfig,
    state: PhantomData<S>,
}

/// Configuration for taxonomy-aware classification
#[derive(Debug, Clone)]
pub struct TaxonomyConfig {
    /// Strategy for leveraging taxonomy
    pub strategy: TaxonomyStrategy,
    /// Confidence threshold for taxonomy-based predictions
    pub confidence_threshold: f64,
    /// Whether to use semantic similarity
    pub use_semantic_similarity: bool,
    /// Minimum support for taxonomy concepts
    pub min_concept_support: usize,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

impl Default for TaxonomyConfig {
    fn default() -> Self {
        Self {
            strategy: TaxonomyStrategy::HierarchicalTraversal,
            confidence_threshold: 0.5,
            use_semantic_similarity: true,
            min_concept_support: 1,
            random_state: None,
        }
    }
}

/// Strategies for taxonomy-aware classification
#[derive(Debug, Clone, PartialEq)]
pub enum TaxonomyStrategy {
    /// Traverse taxonomy hierarchically from root to leaves
    HierarchicalTraversal,
    /// Use concept embeddings and semantic similarity
    SemanticSimilarity,
    /// Combine multiple taxonomy levels
    MultiLevel,
    /// Adaptive strategy based on data characteristics
    Adaptive,
}

/// Taxonomy structure for organizing concepts and classes
#[derive(Debug, Clone)]
pub struct Taxonomy {
    /// Nodes in the taxonomy
    nodes: HashMap<String, TaxonomyNode>,
    /// Root concepts (no parent)
    roots: Vec<String>,
    /// Leaf concepts (classes)
    leaves: Vec<String>,
}

/// A node in the taxonomy representing a concept
#[derive(Debug, Clone)]
pub struct TaxonomyNode {
    /// Concept identifier
    pub concept_id: String,
    /// Parent concept (None for root concepts)
    pub parent: Option<String>,
    /// Child concepts
    pub children: Vec<String>,
    /// Associated class labels (for leaf nodes)
    pub class_labels: Vec<String>,
    /// Semantic embedding (optional)
    pub embedding: Option<Array1<f64>>,
    /// Concept metadata
    pub metadata: HashMap<String, String>,
}

impl Default for Taxonomy {
    fn default() -> Self {
        Self::new()
    }
}

impl Taxonomy {
    /// Create a new empty taxonomy
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
            leaves: Vec::new(),
        }
    }

    /// Add a concept to the taxonomy
    pub fn add_concept(&mut self, node: TaxonomyNode) -> SklResult<()> {
        let concept_id = node.concept_id.clone();

        // Validate parent exists if specified
        if let Some(parent_id) = &node.parent {
            if !self.nodes.contains_key(parent_id) {
                return Err(SklearsError::InvalidInput(format!(
                    "Parent concept '{}' does not exist",
                    parent_id
                )));
            }

            // Add this node as child to parent
            if let Some(parent_node) = self.nodes.get_mut(parent_id) {
                if !parent_node.children.contains(&concept_id) {
                    parent_node.children.push(concept_id.clone());
                }
            }
        } else {
            // This is a root node
            if !self.roots.contains(&concept_id) {
                self.roots.push(concept_id.clone());
            }
        }

        // Check if this is a leaf (has class labels)
        if !node.class_labels.is_empty() && !self.leaves.contains(&concept_id) {
            self.leaves.push(concept_id.clone());
        }

        self.nodes.insert(concept_id, node);
        Ok(())
    }

    /// Get a concept by ID
    pub fn get_concept(&self, concept_id: &str) -> Option<&TaxonomyNode> {
        self.nodes.get(concept_id)
    }

    /// Get all root concepts
    pub fn get_roots(&self) -> &[String] {
        &self.roots
    }

    /// Get all leaf concepts
    pub fn get_leaves(&self) -> &[String] {
        &self.leaves
    }

    /// Get path from root to a concept
    pub fn get_path_to_concept(&self, concept_id: &str) -> SklResult<Vec<String>> {
        let mut path = Vec::new();
        let mut current_id = concept_id.to_string();

        loop {
            if let Some(node) = self.nodes.get(&current_id) {
                path.insert(0, current_id.clone());

                if let Some(parent_id) = &node.parent {
                    current_id = parent_id.clone();
                } else {
                    break; // Reached root
                }
            } else {
                return Err(SklearsError::InvalidInput(format!(
                    "Concept '{}' not found in taxonomy",
                    current_id
                )));
            }
        }

        Ok(path)
    }

    /// Get concepts at a specific depth
    pub fn get_concepts_at_depth(&self, depth: usize) -> Vec<String> {
        let mut concepts = Vec::new();

        for root in &self.roots {
            if depth == 0 {
                concepts.push(root.clone());
            } else {
                self.collect_concepts_at_depth(root, depth, 0, &mut concepts);
            }
        }

        concepts
    }

    fn collect_concepts_at_depth(
        &self,
        concept_id: &str,
        target_depth: usize,
        current_depth: usize,
        result: &mut Vec<String>,
    ) {
        if current_depth == target_depth {
            result.push(concept_id.to_string());
            return;
        }

        if let Some(node) = self.nodes.get(concept_id) {
            for child_id in &node.children {
                self.collect_concepts_at_depth(child_id, target_depth, current_depth + 1, result);
            }
        }
    }

    /// Calculate semantic similarity between concepts
    pub fn semantic_similarity(&self, concept1: &str, concept2: &str) -> f64 {
        // Get embeddings for both concepts
        let embedding1 = self
            .nodes
            .get(concept1)
            .and_then(|node| node.embedding.as_ref());
        let embedding2 = self
            .nodes
            .get(concept2)
            .and_then(|node| node.embedding.as_ref());

        match (embedding1, embedding2) {
            (Some(emb1), Some(emb2)) => {
                // Calculate cosine similarity
                let dot_product = emb1.dot(emb2);
                let norm1 = emb1.dot(emb1).sqrt();
                let norm2 = emb2.dot(emb2).sqrt();

                if norm1 > 0.0 && norm2 > 0.0 {
                    dot_product / (norm1 * norm2)
                } else {
                    0.0
                }
            }
            _ => {
                // Fallback to structural similarity
                self.structural_similarity(concept1, concept2)
            }
        }
    }

    /// Calculate structural similarity based on taxonomy hierarchy
    pub fn structural_similarity(&self, concept1: &str, concept2: &str) -> f64 {
        if concept1 == concept2 {
            return 1.0;
        }

        // Get paths to both concepts
        let path1 = self.get_path_to_concept(concept1).unwrap_or_default();
        let path2 = self.get_path_to_concept(concept2).unwrap_or_default();

        if path1.is_empty() || path2.is_empty() {
            return 0.0;
        }

        // Find common prefix
        let common_prefix_len = path1
            .iter()
            .zip(path2.iter())
            .take_while(|(a, b)| a == b)
            .count();

        // Similarity is based on shared ancestry
        let max_depth = path1.len().max(path2.len());
        if max_depth == 0 {
            0.0
        } else {
            common_prefix_len as f64 / max_depth as f64
        }
    }

    /// Validate taxonomy structure
    pub fn validate(&self) -> SklResult<()> {
        // Check for cycles
        for concept_id in self.nodes.keys() {
            let mut visited = std::collections::HashSet::new();
            if self.has_cycle(concept_id, &mut visited)? {
                return Err(SklearsError::InvalidInput(format!(
                    "Taxonomy contains cycle involving concept '{}'",
                    concept_id
                )));
            }
        }

        // Check that all parent references are valid
        for (concept_id, node) in &self.nodes {
            if let Some(parent_id) = &node.parent {
                if !self.nodes.contains_key(parent_id) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Concept '{}' references non-existent parent '{}'",
                        concept_id, parent_id
                    )));
                }
            }
        }

        Ok(())
    }

    fn has_cycle(
        &self,
        concept_id: &str,
        visited: &mut std::collections::HashSet<String>,
    ) -> SklResult<bool> {
        if visited.contains(concept_id) {
            return Ok(true); // Cycle detected
        }

        visited.insert(concept_id.to_string());

        if let Some(node) = self.nodes.get(concept_id) {
            if let Some(parent_id) = &node.parent {
                if self.has_cycle(parent_id, visited)? {
                    return Ok(true);
                }
            }
        }

        visited.remove(concept_id);
        Ok(false)
    }
}

impl TaxonomyNode {
    /// Create a new taxonomy node
    pub fn new(concept_id: &str, parent: Option<&str>, class_labels: Vec<&str>) -> Self {
        Self {
            concept_id: concept_id.to_string(),
            parent: parent.map(|s| s.to_string()),
            children: Vec::new(),
            class_labels: class_labels.into_iter().map(|s| s.to_string()).collect(),
            embedding: None,
            metadata: HashMap::new(),
        }
    }

    /// Set semantic embedding for the concept
    pub fn with_embedding(mut self, embedding: Array1<f64>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Add metadata to the concept
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

impl<C> TaxonomyAwareClassifier<C, Untrained> {
    /// Create a new taxonomy-aware classifier
    pub fn new(base_estimator: C, taxonomy: Taxonomy) -> SklResult<Self> {
        // Validate the taxonomy
        taxonomy.validate()?;

        Ok(Self {
            base_estimator,
            taxonomy,
            config: TaxonomyConfig::default(),
            state: PhantomData,
        })
    }

    /// Create a builder for TaxonomyAwareClassifier
    pub fn builder(base_estimator: C, taxonomy: Taxonomy) -> SklResult<TaxonomyBuilder<C>> {
        taxonomy.validate()?;
        Ok(TaxonomyBuilder::new(base_estimator, taxonomy))
    }

    /// Set the taxonomy strategy
    pub fn strategy(mut self, strategy: TaxonomyStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the confidence threshold
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    /// Enable or disable semantic similarity
    pub fn use_semantic_similarity(mut self, use_similarity: bool) -> Self {
        self.config.use_semantic_similarity = use_similarity;
        self
    }

    /// Set minimum concept support
    pub fn min_concept_support(mut self, support: usize) -> Self {
        self.config.min_concept_support = support;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Get a reference to the base estimator
    pub fn base_estimator(&self) -> &C {
        &self.base_estimator
    }

    /// Get a reference to the taxonomy
    pub fn taxonomy(&self) -> &Taxonomy {
        &self.taxonomy
    }
}

/// Builder for TaxonomyAwareClassifier
#[derive(Debug)]
pub struct TaxonomyBuilder<C> {
    base_estimator: C,
    taxonomy: Taxonomy,
    config: TaxonomyConfig,
}

impl<C> TaxonomyBuilder<C> {
    /// Create a new builder
    fn new(base_estimator: C, taxonomy: Taxonomy) -> Self {
        Self {
            base_estimator,
            taxonomy,
            config: TaxonomyConfig::default(),
        }
    }

    /// Set the taxonomy strategy
    pub fn strategy(mut self, strategy: TaxonomyStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the confidence threshold
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    /// Enable or disable semantic similarity
    pub fn use_semantic_similarity(mut self, use_similarity: bool) -> Self {
        self.config.use_semantic_similarity = use_similarity;
        self
    }

    /// Set minimum concept support
    pub fn min_concept_support(mut self, support: usize) -> Self {
        self.config.min_concept_support = support;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Build the classifier
    pub fn build(self) -> TaxonomyAwareClassifier<C, Untrained> {
        TaxonomyAwareClassifier {
            base_estimator: self.base_estimator,
            taxonomy: self.taxonomy,
            config: self.config,
            state: PhantomData,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    // Mock classifier for testing
    #[derive(Debug, Clone)]
    struct MockClassifier {
        weights: Option<Array1<Float>>,
        intercept: Option<Float>,
    }

    impl MockClassifier {
        fn new() -> Self {
            Self {
                weights: None,
                intercept: None,
            }
        }
    }

    impl Estimator for MockClassifier {
        type Config = ();
        type Error = SklearsError;
        type Float = Float;

        fn config(&self) -> &Self::Config {
            &()
        }
    }

    #[derive(Debug, Clone)]
    struct MockClassifierTrained {
        weights: Array1<Float>,
        intercept: Float,
    }

    impl Fit<Array2<Float>, Array1<i32>> for MockClassifier {
        type Fitted = MockClassifierTrained;

        fn fit(self, _x: &Array2<Float>, _y: &Array1<i32>) -> SklResult<Self::Fitted> {
            Ok(MockClassifierTrained {
                weights: Array1::zeros(2),
                intercept: 0.0,
            })
        }
    }

    impl Predict<Array2<Float>, Array1<i32>> for MockClassifierTrained {
        fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
            Ok(Array1::zeros(x.nrows()))
        }
    }

    #[derive(Debug, Clone)]
    struct MockNativeClassifier {
        max_classes: Option<usize>,
    }

    impl MockNativeClassifier {
        fn new() -> Self {
            Self { max_classes: None }
        }
    }

    impl Estimator for MockNativeClassifier {
        type Config = ();
        type Error = SklearsError;
        type Float = Float;

        fn config(&self) -> &Self::Config {
            &()
        }
    }

    // Tests for Nested Dichotomies
    #[test]
    fn test_nested_dichotomies_basic() {
        let base_classifier = MockClassifier::new();
        let classifier = NestedDichotomiesClassifier::new(base_classifier);

        // Test data: 3 classes, 6 samples
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        // Should return predictions for all samples
        assert_eq!(predictions.len(), 6);

        // For now, placeholder implementation returns first class for all samples
        assert!(predictions.iter().all(|&pred| pred == 0));
    }

    #[test]
    fn test_nested_dichotomies_builder() {
        let base_classifier = MockClassifier::new();
        let classifier = NestedDichotomiesClassifier::builder(base_classifier)
            .strategy(DichotomyStrategy::Balanced)
            .random_seed(Some(42))
            .build();

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 2, 3];

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    // Tests for Recursive Binary Partitioning
    #[test]
    fn test_rbp_basic() {
        let base_classifier = MockClassifier::new();
        let classifier = RecursiveBinaryPartitioningClassifier::new(base_classifier);

        // Test data: 3 classes, 6 samples
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        // Should return predictions for all samples
        assert_eq!(predictions.len(), 6);

        // For now, placeholder implementation returns first class for all samples
        assert!(predictions.iter().all(|&pred| pred == 0));
    }

    #[test]
    fn test_rbp_builder() {
        let base_classifier = MockClassifier::new();
        let classifier = RecursiveBinaryPartitioningClassifier::builder(base_classifier)
            .strategy(PartitioningStrategy::MostFrequent)
            .random_seed(Some(42))
            .build();

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 2, 3];

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    // Tests for Hierarchical Classification
    fn create_simple_hierarchy() -> ClassHierarchy {
        let mut hierarchy = ClassHierarchy::new();

        // Root node
        hierarchy.add_node(HierarchyNode::new(0, None, 0)).unwrap();

        // Level 1 nodes
        hierarchy
            .add_node(HierarchyNode::new(1, Some(0), 1))
            .unwrap();
        hierarchy
            .add_node(HierarchyNode::new(2, Some(0), 1))
            .unwrap();

        // Level 2 nodes (leaves)
        hierarchy
            .add_node(HierarchyNode::new(3, Some(1), 2))
            .unwrap();
        hierarchy
            .add_node(HierarchyNode::new(4, Some(1), 2))
            .unwrap();
        hierarchy
            .add_node(HierarchyNode::new(5, Some(2), 2))
            .unwrap();

        hierarchy
    }

    #[test]
    fn test_hierarchy_creation() {
        let hierarchy = create_simple_hierarchy();

        assert_eq!(hierarchy.size(), 6);
        assert_eq!(hierarchy.max_depth(), 2);
        assert_eq!(hierarchy.get_roots(), &[0]);
        assert_eq!(hierarchy.get_leaves(), &[3, 4, 5]);
    }

    #[test]
    fn test_hierarchy_path_to_leaf() {
        let hierarchy = create_simple_hierarchy();

        let path = hierarchy.get_path_to_leaf(3).unwrap();
        assert_eq!(path, vec![0, 1, 3]);

        let path = hierarchy.get_path_to_leaf(5).unwrap();
        assert_eq!(path, vec![0, 2, 5]);
    }

    #[test]
    fn test_hierarchical_classifier_creation() {
        let hierarchy = create_simple_hierarchy();
        let base_classifier = MockNativeClassifier::new();

        let classifier = HierarchicalClassifier::new(base_classifier, hierarchy);
        assert!(classifier.is_ok());

        let classifier = classifier.unwrap();
        assert_eq!(classifier.hierarchy().size(), 6);
        assert_eq!(classifier.hierarchy().max_depth(), 2);
    }

    #[test]
    fn test_hierarchical_builder() {
        let hierarchy = create_simple_hierarchy();
        let base_classifier = MockNativeClassifier::new();

        let classifier = HierarchicalClassifier::builder(base_classifier, hierarchy)
            .unwrap()
            .strategy(HierarchicalStrategy::TopDown)
            .probability_threshold(0.7)
            .early_stopping(true)
            .build();

        assert!(matches!(
            classifier.config.strategy,
            HierarchicalStrategy::TopDown
        ));
        assert_eq!(classifier.config.probability_threshold, 0.7);
        assert!(classifier.config.early_stopping);
    }

    // Tests for Taxonomy-Aware Classification
    fn create_animal_taxonomy() -> Taxonomy {
        let mut taxonomy = Taxonomy::new();

        // Root concept
        taxonomy
            .add_concept(TaxonomyNode::new("Animal", None, vec![]))
            .unwrap();

        // Level 1 concepts
        taxonomy
            .add_concept(TaxonomyNode::new(
                "Mammal",
                Some("Animal"),
                vec!["dog", "cat"],
            ))
            .unwrap();
        taxonomy
            .add_concept(TaxonomyNode::new("Bird", Some("Animal"), vec!["eagle"]))
            .unwrap();

        taxonomy
    }

    #[test]
    fn test_taxonomy_creation() {
        let taxonomy = create_animal_taxonomy();

        assert_eq!(taxonomy.get_roots(), &["Animal"]);
        assert_eq!(taxonomy.get_leaves().len(), 2);
    }

    #[test]
    fn test_taxonomy_aware_classifier_creation() {
        let taxonomy = create_animal_taxonomy();
        let base_classifier = MockNativeClassifier::new();

        let classifier = TaxonomyAwareClassifier::new(base_classifier, taxonomy);
        assert!(classifier.is_ok());

        let classifier = classifier.unwrap();
        assert_eq!(classifier.taxonomy().get_roots(), &["Animal"]);
        assert_eq!(classifier.taxonomy().get_leaves().len(), 2);
    }

    #[test]
    fn test_taxonomy_aware_builder() {
        let taxonomy = create_animal_taxonomy();
        let base_classifier = MockNativeClassifier::new();

        let classifier = TaxonomyAwareClassifier::builder(base_classifier, taxonomy)
            .unwrap()
            .strategy(TaxonomyStrategy::SemanticSimilarity)
            .confidence_threshold(0.7)
            .use_semantic_similarity(true)
            .build();

        assert!(matches!(
            classifier.config.strategy,
            TaxonomyStrategy::SemanticSimilarity
        ));
        assert_eq!(classifier.config.confidence_threshold, 0.7);
        assert!(classifier.config.use_semantic_similarity);
    }
}

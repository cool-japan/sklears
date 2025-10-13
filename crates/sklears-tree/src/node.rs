//! Tree node data structures and compact representations
//!
//! This module contains various tree node representations including compact,
//! bit-packed, and shared node structures for memory efficiency and optimization.

use crate::config::DecisionTreeConfig;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, error::SklearsError};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Custom split information for tree nodes
#[derive(Debug, Clone)]
pub struct CustomSplit {
    pub feature_idx: usize,
    pub threshold: f64,
    pub impurity_decrease: f64,
    pub left_count: usize,
    pub right_count: usize,
}

/// Surrogate split information for handling missing values
#[derive(Debug, Clone)]
pub struct SurrogateSplit {
    pub feature_idx: usize,
    pub threshold: f64,
    pub agreement: f64,       // How well this surrogate agrees with primary split
    pub left_direction: bool, // True if missing values go left, false if right
}

/// Compact tree node representation for memory efficiency
#[derive(Debug, Clone)]
pub struct CompactTreeNode {
    /// Compact node data packed into a single u64
    /// Bits 0-15: feature index (16 bits, max 65535 features)
    /// Bits 16-47: threshold as f32 bits (32 bits)
    /// Bits 48-63: node flags and metadata (16 bits)
    pub packed_data: u64,
    /// Prediction value for leaf nodes or impurity for split nodes
    pub value: f32,
    /// Left child index (0 means no left child)
    pub left_child: u32,
    /// Right child index (0 means no right child)
    pub right_child: u32,
}

impl CompactTreeNode {
    /// Create a new leaf node
    pub fn new_leaf(prediction: f64) -> Self {
        Self {
            packed_data: 0x8000_0000_0000_0000, // Set leaf flag in bit 63
            value: prediction as f32,
            left_child: 0,
            right_child: 0,
        }
    }

    /// Create a new split node
    pub fn new_split(feature_idx: u16, threshold: f32, impurity: f64) -> Self {
        let mut packed_data = 0u64;

        // Pack feature index (bits 0-15)
        packed_data |= feature_idx as u64;

        // Pack threshold (bits 16-47)
        let threshold_bits = threshold.to_bits() as u64;
        packed_data |= threshold_bits << 16;

        Self {
            packed_data,
            value: impurity as f32,
            left_child: 0,
            right_child: 0,
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        (self.packed_data & 0x8000_0000_0000_0000) != 0
    }

    /// Get feature index for split nodes
    pub fn feature_idx(&self) -> u16 {
        (self.packed_data & 0xFFFF) as u16
    }

    /// Get threshold for split nodes
    pub fn threshold(&self) -> f32 {
        let threshold_bits = ((self.packed_data >> 16) & 0xFFFFFFFF) as u32;
        f32::from_bits(threshold_bits)
    }

    /// Get prediction value for leaf nodes
    pub fn prediction(&self) -> f64 {
        self.value as f64
    }

    /// Get impurity value for split nodes
    pub fn impurity(&self) -> f64 {
        self.value as f64
    }

    /// Set left child index
    pub fn set_left_child(&mut self, child_idx: u32) {
        self.left_child = child_idx;
    }

    /// Set right child index
    pub fn set_right_child(&mut self, child_idx: u32) {
        self.right_child = child_idx;
    }
}

/// Compact tree representation for memory efficiency
#[derive(Debug, Clone)]
pub struct CompactTree {
    /// Array of compact tree nodes
    pub nodes: Vec<CompactTreeNode>,
    /// Feature importance scores
    pub feature_importances: Vec<f32>,
    /// Number of features
    pub n_features: usize,
    /// Tree depth
    pub depth: usize,
}

impl CompactTree {
    /// Create a new compact tree
    pub fn new(n_features: usize) -> Self {
        Self {
            nodes: Vec::new(),
            feature_importances: vec![0.0; n_features],
            n_features,
            depth: 0,
        }
    }

    /// Add a new node and return its index
    pub fn add_node(&mut self, node: CompactTreeNode) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(node);
        idx
    }

    /// Predict a single sample
    pub fn predict_single(&self, sample: &[f64]) -> Result<f64> {
        if self.nodes.is_empty() {
            return Err(SklearsError::InvalidInput("Empty tree".to_string()));
        }

        let mut current_idx = 0;

        loop {
            if current_idx >= self.nodes.len() {
                return Err(SklearsError::InvalidInput("Invalid node index".to_string()));
            }

            let node = &self.nodes[current_idx];

            if node.is_leaf() {
                return Ok(node.prediction());
            }

            let feature_idx = node.feature_idx() as usize;
            if feature_idx >= sample.len() {
                return Err(SklearsError::InvalidInput(
                    "Feature index out of bounds".to_string(),
                ));
            }

            let feature_value = sample[feature_idx];
            let threshold = node.threshold() as f64;

            if feature_value <= threshold {
                current_idx = node.left_child as usize;
            } else {
                current_idx = node.right_child as usize;
            }

            if current_idx == 0 {
                return Err(SklearsError::InvalidInput("Invalid child node".to_string()));
            }
            current_idx -= 1; // Convert to 0-based indexing
        }
    }

    /// Predict multiple samples
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = x.row(i).to_vec();
            predictions[i] = self.predict_single(&sample)?;
        }

        Ok(predictions)
    }

    /// Calculate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let node_size = std::mem::size_of::<CompactTreeNode>();
        let feature_importance_size = std::mem::size_of::<f32>() * self.feature_importances.len();
        let metadata_size = std::mem::size_of::<Self>()
            - std::mem::size_of::<Vec<CompactTreeNode>>()
            - std::mem::size_of::<Vec<f32>>();

        self.nodes.len() * node_size + feature_importance_size + metadata_size
    }

    /// Get the number of nodes in the tree
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of leaves in the tree
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|node| node.is_leaf()).count()
    }
}

/// Bit-packed decision path for ultra-compact storage
#[derive(Debug, Clone)]
pub struct BitPackedPath {
    /// Packed decision bits (left = 0, right = 1)
    pub path_bits: u128,
    /// Number of decisions in the path
    pub path_length: u8,
    /// Final prediction value
    pub prediction: f32,
}

impl Default for BitPackedPath {
    fn default() -> Self {
        Self::new()
    }
}

impl BitPackedPath {
    /// Create a new bit-packed path
    pub fn new() -> Self {
        Self {
            path_bits: 0,
            path_length: 0,
            prediction: 0.0,
        }
    }

    /// Add a decision to the path (false = left, true = right)
    pub fn add_decision(&mut self, go_right: bool) -> Result<()> {
        if self.path_length >= 127 {
            return Err(SklearsError::InvalidInput("Path too long".to_string()));
        }

        if go_right {
            self.path_bits |= 1u128 << self.path_length;
        }
        self.path_length += 1;
        Ok(())
    }

    /// Get decision at position (false = left, true = right)
    pub fn get_decision(&self, position: u8) -> bool {
        if position >= self.path_length {
            return false;
        }

        (self.path_bits & (1u128 << position)) != 0
    }

    /// Set final prediction
    pub fn set_prediction(&mut self, prediction: f64) {
        self.prediction = prediction as f32;
    }

    /// Get final prediction
    pub fn get_prediction(&self) -> f64 {
        self.prediction as f64
    }
}

/// Ultra-compact tree using bit-packed decision paths
#[derive(Debug, Clone)]
pub struct BitPackedTree {
    /// Decision paths for each possible outcome
    pub paths: Vec<BitPackedPath>,
    /// Feature indices used in decisions
    pub feature_indices: Vec<u16>,
    /// Thresholds used in decisions
    pub thresholds: Vec<f32>,
    /// Number of features
    pub n_features: usize,
}

impl BitPackedTree {
    /// Create a new bit-packed tree
    pub fn new(n_features: usize) -> Self {
        Self {
            paths: Vec::new(),
            feature_indices: Vec::new(),
            thresholds: Vec::new(),
            n_features,
        }
    }

    /// Add a decision path
    pub fn add_path(&mut self, path: BitPackedPath) {
        self.paths.push(path);
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let path_size = std::mem::size_of::<BitPackedPath>() * self.paths.len();
        let feature_size = std::mem::size_of::<u16>() * self.feature_indices.len();
        let threshold_size = std::mem::size_of::<f32>() * self.thresholds.len();
        let metadata_size = std::mem::size_of::<Self>()
            - std::mem::size_of::<Vec<BitPackedPath>>()
            - std::mem::size_of::<Vec<u16>>()
            - std::mem::size_of::<Vec<f32>>();

        path_size + feature_size + threshold_size + metadata_size
    }
}

/// Memory-efficient ensemble representation
#[derive(Debug, Clone)]
pub struct CompactEnsemble {
    /// Array of compact trees
    pub trees: Vec<CompactTree>,
    /// Shared feature importance across all trees
    pub global_feature_importances: Vec<f32>,
    /// Number of features
    pub n_features: usize,
    /// Number of trees
    pub n_trees: usize,
}

impl CompactEnsemble {
    /// Create a new compact ensemble
    pub fn new(n_features: usize) -> Self {
        Self {
            trees: Vec::new(),
            global_feature_importances: vec![0.0; n_features],
            n_features,
            n_trees: 0,
        }
    }

    /// Add a tree to the ensemble
    pub fn add_tree(&mut self, tree: CompactTree) {
        self.trees.push(tree);
        self.n_trees += 1;
    }

    /// Calculate total memory usage
    pub fn total_memory_usage(&self) -> usize {
        let trees_memory: usize = self.trees.iter().map(|t| t.memory_usage()).sum();
        let importance_memory = std::mem::size_of::<f32>() * self.global_feature_importances.len();
        let metadata_memory = std::mem::size_of::<Self>()
            - std::mem::size_of::<Vec<CompactTree>>()
            - std::mem::size_of::<Vec<f32>>();

        trees_memory + importance_memory + metadata_memory
    }
}

/// Tree node for building algorithms
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Node ID
    pub id: usize,
    /// Depth of this node
    pub depth: usize,
    /// Samples in this node
    pub sample_indices: Vec<usize>,
    /// Impurity of this node
    pub impurity: f64,
    /// Predicted value/class for this node
    pub prediction: f64,
    /// Potential impurity decrease if this node is split
    pub potential_decrease: f64,
    /// Best split for this node (if any)
    pub best_split: Option<CustomSplit>,
    /// Parent node ID
    pub parent_id: Option<usize>,
    /// Whether this is a leaf node
    pub is_leaf: bool,
}

/// Priority wrapper for nodes in the queue
#[derive(Debug, Clone)]
pub struct NodePriority {
    pub node_id: usize,
    pub priority: f64, // Negative of impurity decrease for max-heap behavior
}

impl PartialEq for NodePriority {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for NodePriority {}

impl PartialOrd for NodePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for max-heap (highest priority first)
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Wrapper for f64 to make it hashable and orderable
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl std::cmp::Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Subtree pattern for identifying reusable subtrees
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct SubtreePattern {
    /// Maximum depth of the pattern
    pub depth: usize,
    /// Minimum number of samples to consider for sharing
    pub min_samples: usize,
    /// Feature and threshold pairs in the pattern
    pub splits: Vec<(usize, OrderedFloat)>,
}

/// Configuration for shared subtree optimization
#[derive(Debug, Clone)]
pub struct SubtreeConfig {
    /// Minimum number of samples in a subtree to consider sharing
    pub min_samples_for_sharing: usize,
    /// Maximum depth of subtrees to share
    pub max_shared_depth: usize,
    /// Minimum frequency of pattern to justify sharing
    pub min_pattern_frequency: usize,
    /// Enable/disable subtree sharing
    pub enabled: bool,
}

impl Default for SubtreeConfig {
    fn default() -> Self {
        Self {
            min_samples_for_sharing: 10,
            max_shared_depth: 3,
            min_pattern_frequency: 2,
            enabled: false,
        }
    }
}

/// A shared tree node that can be referenced by multiple trees
#[derive(Debug, Clone)]
pub struct SharedTreeNode {
    /// Unique identifier for this shared node
    pub id: usize,
    /// Feature index for split (None for leaf nodes)
    pub feature_idx: Option<usize>,
    /// Split threshold (None for leaf nodes)
    pub threshold: Option<f64>,
    /// Prediction value for this node
    pub prediction: f64,
    /// Number of samples that reached this node
    pub n_samples: usize,
    /// Impurity at this node
    pub impurity: f64,
    /// Left child node ID (None for leaf)
    pub left_child: Option<usize>,
    /// Right child node ID (None for leaf)
    pub right_child: Option<usize>,
    /// Hash representing the subtree structure for sharing
    pub subtree_hash: u64,
}

/// Statistics about subtree sharing efficiency
#[derive(Debug, Clone)]
pub struct SubtreeSharingStats {
    /// Total number of shared nodes
    pub total_shared_nodes: usize,
    /// Total number of unique patterns
    pub total_patterns: usize,
    /// Estimated memory saved in bytes
    pub estimated_memory_saved: usize,
    /// Sharing efficiency (shared nodes / patterns)
    pub sharing_efficiency: f64,
}

/// Reference to a shared subtree in a tree ensemble
#[derive(Debug, Clone)]
pub struct SubtreeReference {
    /// ID of the shared subtree
    pub shared_id: usize,
    /// Local node ID in the referencing tree
    pub local_node_id: usize,
    /// Tree ID that contains this reference
    pub tree_id: usize,
}

/// Shared subtree manager for memory optimization
#[derive(Debug, Clone)]
pub struct SharedSubtreeManager {
    /// Shared nodes indexed by ID
    pub shared_nodes: Arc<RwLock<HashMap<usize, SharedTreeNode>>>,
    /// Pattern to shared node ID mapping
    pub pattern_cache: Arc<RwLock<HashMap<SubtreePattern, usize>>>,
    /// Next available node ID
    pub next_node_id: Arc<RwLock<usize>>,
    /// Configuration for subtree sharing
    pub config: SubtreeConfig,
}

impl SharedSubtreeManager {
    pub fn new(config: SubtreeConfig) -> Self {
        Self {
            shared_nodes: Arc::new(RwLock::new(HashMap::new())),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            next_node_id: Arc::new(RwLock::new(0)),
            config,
        }
    }

    /// Extract subtree patterns from a tree for potential sharing
    pub fn extract_patterns(&self, tree_nodes: &[TreeNode]) -> Result<Vec<SubtreePattern>> {
        if !self.config.enabled {
            return Ok(vec![]);
        }

        let mut patterns = Vec::new();

        for node in tree_nodes {
            if node.is_leaf || node.sample_indices.len() < self.config.min_samples_for_sharing {
                continue;
            }

            if let Some(pattern) = self.extract_pattern_from_node(tree_nodes, node.id, 0) {
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Extract a pattern starting from a specific node
    fn extract_pattern_from_node(
        &self,
        tree_nodes: &[TreeNode],
        node_id: usize,
        current_depth: usize,
    ) -> Option<SubtreePattern> {
        if current_depth >= self.config.max_shared_depth {
            return None;
        }

        if node_id >= tree_nodes.len() {
            return None;
        }

        let node = &tree_nodes[node_id];

        if node.is_leaf {
            return Some(SubtreePattern {
                depth: current_depth,
                min_samples: node.sample_indices.len(),
                splits: vec![],
            });
        }

        let mut splits = Vec::new();

        if let Some(ref split) = node.best_split {
            splits.push((split.feature_idx, OrderedFloat(split.threshold)));
        }

        Some(SubtreePattern {
            depth: current_depth,
            min_samples: node.sample_indices.len(),
            splits,
        })
    }

    /// Find or create a shared subtree for a given pattern
    pub fn get_or_create_shared_subtree(
        &self,
        pattern: &SubtreePattern,
        tree_nodes: &[TreeNode],
        root_node_id: usize,
    ) -> Result<usize> {
        // Check if pattern already exists
        {
            let cache = self.pattern_cache.read().unwrap();
            if let Some(&shared_id) = cache.get(pattern) {
                return Ok(shared_id);
            }
        }

        // Create new shared subtree
        let shared_id = {
            let mut next_id = self.next_node_id.write().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let shared_node = self.create_shared_node(tree_nodes, root_node_id, shared_id)?;

        // Store the shared node
        {
            let mut nodes = self.shared_nodes.write().unwrap();
            nodes.insert(shared_id, shared_node);
        }

        // Cache the pattern
        {
            let mut cache = self.pattern_cache.write().unwrap();
            cache.insert(pattern.clone(), shared_id);
        }

        Ok(shared_id)
    }

    /// Create a shared node from a tree node
    fn create_shared_node(
        &self,
        tree_nodes: &[TreeNode],
        node_id: usize,
        shared_id: usize,
    ) -> Result<SharedTreeNode> {
        if node_id >= tree_nodes.len() {
            return Err(SklearsError::InvalidInput("Invalid node ID".to_string()));
        }

        let node = &tree_nodes[node_id];
        let subtree_hash = self.calculate_subtree_hash(tree_nodes, node_id);

        let (feature_idx, threshold) = if let Some(ref split) = node.best_split {
            (Some(split.feature_idx), Some(split.threshold))
        } else {
            (None, None)
        };

        Ok(SharedTreeNode {
            id: shared_id,
            feature_idx,
            threshold,
            prediction: node.prediction,
            n_samples: node.sample_indices.len(),
            impurity: node.impurity,
            left_child: None,  // Would need to be set based on actual children
            right_child: None, // Would need to be set based on actual children
            subtree_hash,
        })
    }

    /// Calculate a hash representing the structure of a subtree
    fn calculate_subtree_hash(&self, tree_nodes: &[TreeNode], node_id: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        self.hash_subtree_recursive(tree_nodes, node_id, &mut hasher, 0);
        hasher.finish()
    }

    /// Recursively hash a subtree structure
    fn hash_subtree_recursive(
        &self,
        tree_nodes: &[TreeNode],
        node_id: usize,
        hasher: &mut dyn std::hash::Hasher,
        depth: usize,
    ) {
        if depth >= self.config.max_shared_depth || node_id >= tree_nodes.len() {
            return;
        }

        let node = &tree_nodes[node_id];

        // Hash node properties
        hasher.write_u8(if node.is_leaf { 1 } else { 0 });
        if let Some(ref split) = node.best_split {
            hasher.write_usize(split.feature_idx);
            hasher.write(&split.threshold.to_le_bytes());
        }
    }

    /// Calculate memory savings from subtree sharing
    pub fn calculate_memory_savings(&self) -> Result<SubtreeSharingStats> {
        let shared_nodes = self.shared_nodes.read().unwrap();
        let pattern_cache = self.pattern_cache.read().unwrap();

        let total_shared_nodes = shared_nodes.len();
        let total_patterns = pattern_cache.len();

        // Estimate memory saved (simplified calculation)
        let estimated_memory_saved = total_shared_nodes * std::mem::size_of::<SharedTreeNode>();

        Ok(SubtreeSharingStats {
            total_shared_nodes,
            total_patterns,
            estimated_memory_saved,
            sharing_efficiency: if total_patterns > 0 {
                total_shared_nodes as f64 / total_patterns as f64
            } else {
                0.0
            },
        })
    }
}

/// Tree-specific data that cannot be shared
#[derive(Debug, Clone)]
pub struct TreeSpecificData {
    /// Tree ID
    pub tree_id: usize,
    /// Sample weights specific to this tree
    pub sample_weights: Vec<f64>,
    /// Tree-specific configuration
    pub config: DecisionTreeConfig,
    /// Non-shared nodes (typically small, tree-specific parts)
    pub local_nodes: Vec<TreeNode>,
}

/// Ensemble of trees with shared subtree optimization
#[derive(Debug, Clone)]
pub struct SharedTreeEnsemble {
    /// Shared subtree manager
    pub subtree_manager: SharedSubtreeManager,
    /// Tree-specific data (non-shared parts)
    pub tree_specific_data: Vec<TreeSpecificData>,
    /// References to shared subtrees
    pub subtree_references: Vec<SubtreeReference>,
}

impl SharedTreeEnsemble {
    pub fn new(subtree_config: SubtreeConfig) -> Self {
        Self {
            subtree_manager: SharedSubtreeManager::new(subtree_config),
            tree_specific_data: Vec::new(),
            subtree_references: Vec::new(),
        }
    }

    /// Add a new tree to the ensemble with shared subtree optimization
    pub fn add_tree(
        &mut self,
        tree_nodes: Vec<TreeNode>,
        config: DecisionTreeConfig,
        tree_id: usize,
    ) -> Result<()> {
        // Extract patterns from the tree
        let patterns = self.subtree_manager.extract_patterns(&tree_nodes)?;

        // Create shared subtrees for frequent patterns
        for pattern in patterns {
            if let Ok(shared_id) = self.subtree_manager.get_or_create_shared_subtree(
                &pattern,
                &tree_nodes,
                0, // Assuming root node
            ) {
                self.subtree_references.push(SubtreeReference {
                    shared_id,
                    local_node_id: 0,
                    tree_id,
                });
            }
        }

        // Store tree-specific data
        let tree_data = TreeSpecificData {
            tree_id,
            sample_weights: vec![1.0; tree_nodes.len()], // Default weights
            config,
            local_nodes: tree_nodes,
        };

        self.tree_specific_data.push(tree_data);

        Ok(())
    }

    /// Get sharing statistics for the ensemble
    pub fn get_sharing_stats(&self) -> Result<SubtreeSharingStats> {
        self.subtree_manager.calculate_memory_savings()
    }
}

//! Shared subtree management for memory optimization
//!
//! This module provides functionality for sharing common subtrees across
//! multiple decision trees in an ensemble to reduce memory usage.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};

use crate::building::TreeNode;
use crate::criteria::DecisionTreeConfig;

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

/// Wrapper for f64 to make it hashable and orderable
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl std::cmp::Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
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

            // Recursively extract patterns from children
            // Note: In a real implementation, you'd need to track child node IDs
            // This is a simplified version for demonstration
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

        // Note: In a real implementation, you'd recursively hash children
        // This is simplified for demonstration
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
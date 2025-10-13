//! Compact tree storage and compression for memory efficiency
//!
//! This module provides memory-efficient representations of decision trees
//! including bit-packing and compression techniques.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};

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

impl Default for BitPackedPath {
    fn default() -> Self {
        Self::new()
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
        // Update global feature importances
        for (i, &importance) in tree.feature_importances.iter().enumerate() {
            if i < self.global_feature_importances.len() {
                self.global_feature_importances[i] += importance;
            }
        }

        self.trees.push(tree);
        self.n_trees += 1;
    }

    /// Predict using the ensemble
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if self.trees.is_empty() {
            return Err(SklearsError::InvalidInput("Empty ensemble".to_string()));
        }

        let n_samples = x.nrows();
        let mut ensemble_predictions = Array1::zeros(n_samples);

        // Average predictions from all trees
        for tree in &self.trees {
            let tree_predictions = tree.predict(x)?;
            ensemble_predictions = ensemble_predictions + tree_predictions;
        }

        ensemble_predictions = ensemble_predictions / (self.n_trees as f64);
        Ok(ensemble_predictions)
    }

    /// Calculate total memory usage
    pub fn memory_usage(&self) -> usize {
        let tree_memory: usize = self.trees.iter().map(|tree| tree.memory_usage()).sum();
        let feature_importance_memory =
            std::mem::size_of::<f32>() * self.global_feature_importances.len();
        let metadata_memory = std::mem::size_of::<Self>()
            - std::mem::size_of::<Vec<CompactTree>>()
            - std::mem::size_of::<Vec<f32>>();

        tree_memory + feature_importance_memory + metadata_memory
    }

    /// Get compression ratio compared to standard representation
    pub fn compression_ratio(&self, original_size: usize) -> f64 {
        let compressed_size = self.memory_usage();
        if compressed_size == 0 {
            return 1.0;
        }
        original_size as f64 / compressed_size as f64
    }
}
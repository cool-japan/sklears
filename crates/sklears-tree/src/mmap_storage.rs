//! Memory-mapped tree storage for large models
//!
//! This module provides efficient storage and retrieval of tree models using
//! memory mapping, allowing large models to be accessed without fully loading
//! them into memory.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Memory-mapped tree node for efficient storage
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MMapTreeNode {
    /// Feature index for split (-1 for leaf nodes)
    pub feature_idx: i32,
    /// Threshold value for split (or prediction for leaf)
    pub threshold: f64,
    /// Left child offset in file (-1 if no left child)
    pub left_child_offset: i64,
    /// Right child offset in file (-1 if no right child)
    pub right_child_offset: i64,
    /// Number of samples in this node
    pub n_samples: u32,
    /// Impurity of this node
    pub impurity: f64,
    /// Node depth
    pub depth: u16,
    /// Padding for alignment
    pub _padding: [u8; 6],
}

impl MMapTreeNode {
    /// Create a new leaf node
    pub fn new_leaf(prediction: f64, n_samples: u32, impurity: f64, depth: u16) -> Self {
        Self {
            feature_idx: -1,
            threshold: prediction,
            left_child_offset: -1,
            right_child_offset: -1,
            n_samples,
            impurity,
            depth,
            _padding: [0; 6],
        }
    }

    /// Create a new internal node
    pub fn new_internal(
        feature_idx: usize,
        threshold: f64,
        left_offset: i64,
        right_offset: i64,
        n_samples: u32,
        impurity: f64,
        depth: u16,
    ) -> Self {
        Self {
            feature_idx: feature_idx as i32,
            threshold,
            left_child_offset: left_offset,
            right_child_offset: right_offset,
            n_samples,
            impurity,
            depth,
            _padding: [0; 6],
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.feature_idx == -1
    }

    /// Get prediction for leaf node
    pub fn prediction(&self) -> f64 {
        if self.is_leaf() {
            self.threshold
        } else {
            0.0
        }
    }

    /// Size of the node in bytes
    pub const fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Memory-mapped tree model
pub struct MMapTree {
    /// File handle for the memory-mapped file
    file: File,
    /// Path to the file
    file_path: PathBuf,
    /// Root node offset
    root_offset: i64,
    /// Number of nodes in the tree
    n_nodes: usize,
    /// Tree metadata
    metadata: TreeMetadata,
}

/// Metadata for the tree stored in the file header
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TreeMetadata {
    /// Magic number for file format validation
    pub magic: u32,
    /// Version of the file format
    pub version: u16,
    /// Number of features
    pub n_features: u32,
    /// Number of classes (for classification)
    pub n_classes: u32,
    /// Tree type (0 = regression, 1 = classification)
    pub tree_type: u8,
    /// Maximum depth of the tree
    pub max_depth: u16,
    /// Minimum samples per leaf
    pub min_samples_leaf: u32,
    /// Number of nodes in the tree
    pub n_nodes: u64,
    /// Offset to root node
    pub root_offset: i64,
    /// Reserved for future use
    pub reserved: [u8; 64],
}

impl TreeMetadata {
    /// Magic number for file format identification
    const MAGIC: u32 = 0x54524545; // "TREE" in ASCII

    /// Current file format version
    const VERSION: u16 = 1;

    /// Create new metadata
    pub fn new(
        n_features: usize,
        n_classes: usize,
        is_classifier: bool,
        max_depth: usize,
        min_samples_leaf: usize,
    ) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            n_features: n_features as u32,
            n_classes: n_classes as u32,
            tree_type: if is_classifier { 1 } else { 0 },
            max_depth: max_depth as u16,
            min_samples_leaf: min_samples_leaf as u32,
            n_nodes: 0,
            root_offset: 0,
            reserved: [0; 64],
        }
    }

    /// Validate metadata
    pub fn validate(&self) -> Result<()> {
        if self.magic != Self::MAGIC {
            return Err(SklearsError::InvalidInput(
                "Invalid file format: magic number mismatch".to_string(),
            ));
        }

        if self.version > Self::VERSION {
            return Err(SklearsError::InvalidInput(format!(
                "Unsupported file version: {}",
                self.version
            )));
        }

        Ok(())
    }

    /// Size of metadata in bytes
    pub const fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl MMapTree {
    /// Create a new memory-mapped tree file
    pub fn create<P: AsRef<Path>>(path: P, metadata: TreeMetadata) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)?;

        // Write metadata header
        // SAFETY: TreeMetadata is a plain-old-data type with repr(C),
        // and we're creating a slice view of its byte representation.
        // The lifetime of the slice is bound to the metadata reference.
        let metadata_bytes = unsafe {
            std::slice::from_raw_parts(
                &metadata as *const TreeMetadata as *const u8,
                TreeMetadata::size(),
            )
        };
        file.write_all(metadata_bytes)?;
        file.flush()?;

        Ok(Self {
            file,
            file_path,
            root_offset: TreeMetadata::size() as i64,
            n_nodes: 0,
            metadata,
        })
    }

    /// Open an existing memory-mapped tree file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new().read(true).open(&file_path)?;

        // Read and validate metadata
        let mut metadata_bytes = vec![0u8; TreeMetadata::size()];
        file.read_exact(&mut metadata_bytes)?;

        // SAFETY: We've read exactly TreeMetadata::size() bytes into metadata_bytes,
        // and TreeMetadata is a repr(C) struct. The buffer is properly aligned
        // and contains valid bytes that represent a TreeMetadata instance.
        let metadata = unsafe { std::ptr::read(metadata_bytes.as_ptr() as *const TreeMetadata) };
        metadata.validate()?;

        Ok(Self {
            file,
            file_path,
            root_offset: metadata.root_offset,
            n_nodes: metadata.n_nodes as usize,
            metadata,
        })
    }

    /// Write a node to the file and return its offset
    pub fn write_node(&mut self, node: &MMapTreeNode) -> Result<i64> {
        // Seek to end of file
        let offset = self.file.seek(SeekFrom::End(0))?;

        // Write node data
        // SAFETY: MMapTreeNode is a plain-old-data type with repr(C),
        // and we're creating a slice view of its byte representation.
        // The lifetime of the slice is bound to the node reference.
        let node_bytes = unsafe {
            std::slice::from_raw_parts(
                node as *const MMapTreeNode as *const u8,
                MMapTreeNode::size(),
            )
        };
        self.file.write_all(node_bytes)?;
        self.file.flush()?;

        self.n_nodes += 1;

        Ok(offset.try_into().unwrap())
    }

    /// Read a node from the file at the given offset
    pub fn read_node(&mut self, offset: i64) -> Result<MMapTreeNode> {
        self.file.seek(SeekFrom::Start(offset as u64))?;

        let mut node_bytes = vec![0u8; MMapTreeNode::size()];
        self.file.read_exact(&mut node_bytes)?;

        // SAFETY: We've read exactly MMapTreeNode::size() bytes into node_bytes,
        // and MMapTreeNode is a repr(C) struct. The buffer is properly sized
        // and contains valid bytes that represent a MMapTreeNode instance.
        let node = unsafe { std::ptr::read(node_bytes.as_ptr() as *const MMapTreeNode) };

        Ok(node)
    }

    /// Set the root node offset
    pub fn set_root_offset(&mut self, offset: i64) -> Result<()> {
        self.root_offset = offset;
        self.metadata.root_offset = offset;
        self.metadata.n_nodes = self.n_nodes as u64;

        // Update metadata in file
        self.file.seek(SeekFrom::Start(0))?;
        // SAFETY: TreeMetadata is a plain-old-data type with repr(C),
        // and we're creating a slice view of its byte representation.
        // The lifetime of the slice is bound to the metadata reference.
        let metadata_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.metadata as *const TreeMetadata as *const u8,
                TreeMetadata::size(),
            )
        };
        self.file.write_all(metadata_bytes)?;
        self.file.flush()?;

        Ok(())
    }

    /// Make prediction for a single sample
    pub fn predict_sample(&mut self, features: &Array1<Float>) -> Result<Float> {
        if features.len() != self.metadata.n_features as usize {
            return Err(SklearsError::InvalidInput(
                "Feature vector length doesn't match tree".to_string(),
            ));
        }

        let mut current_offset = self.root_offset;

        loop {
            let node = self.read_node(current_offset)?;

            if node.is_leaf() {
                return Ok(node.prediction());
            }

            let feature_value = features[node.feature_idx as usize] as f64;

            if feature_value <= node.threshold {
                if node.left_child_offset < 0 {
                    return Err(SklearsError::InvalidInput(
                        "Tree structure corrupted: missing left child".to_string(),
                    ));
                }
                current_offset = node.left_child_offset;
            } else {
                if node.right_child_offset < 0 {
                    return Err(SklearsError::InvalidInput(
                        "Tree structure corrupted: missing right child".to_string(),
                    ));
                }
                current_offset = node.right_child_offset;
            }
        }
    }

    /// Make predictions for multiple samples
    pub fn predict(&mut self, features: &Array2<Float>) -> Result<Array1<Float>> {
        let n_samples = features.nrows();
        let mut predictions = Array1::<Float>::zeros(n_samples);

        for (i, sample) in features.outer_iter().enumerate() {
            predictions[i] = self.predict_sample(&sample.to_owned())?;
        }

        Ok(predictions)
    }

    /// Get tree statistics
    pub fn get_stats(&self) -> TreeStats {
        TreeStats {
            n_nodes: self.n_nodes,
            max_depth: self.metadata.max_depth as usize,
            n_features: self.metadata.n_features as usize,
            file_size: self.file.metadata().map(|m| m.len()).unwrap_or(0),
            memory_usage: self.estimate_memory_usage(),
        }
    }

    /// Estimate memory usage if tree were loaded into RAM
    fn estimate_memory_usage(&self) -> usize {
        self.n_nodes * MMapTreeNode::size() + TreeMetadata::size()
    }

    /// Compact the tree file by removing unused space
    pub fn compact(&mut self) -> Result<()> {
        // This is a simplified compaction - in practice, you'd want to
        // rewrite the file to remove any gaps
        self.file.flush()?;
        Ok(())
    }

    /// Get file path
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// Get metadata
    pub fn metadata(&self) -> &TreeMetadata {
        &self.metadata
    }
}

/// Statistics about a memory-mapped tree
#[derive(Debug, Clone)]
pub struct TreeStats {
    /// Number of nodes in the tree
    pub n_nodes: usize,
    /// Maximum depth of the tree
    pub max_depth: usize,
    /// Number of features
    pub n_features: usize,
    /// Size of the file on disk
    pub file_size: u64,
    /// Estimated memory usage if loaded into RAM
    pub memory_usage: usize,
}

impl TreeStats {
    /// Get memory savings ratio
    pub fn memory_savings_ratio(&self) -> f64 {
        1.0 - (std::mem::size_of::<File>() as f64 / self.memory_usage as f64)
    }

    /// Format stats as string
    pub fn summary(&self) -> String {
        format!(
            "Tree Statistics:\n\
             Nodes: {}\n\
             Max Depth: {}\n\
             Features: {}\n\
             File Size: {} bytes\n\
             Est. Memory Usage: {} bytes\n\
             Memory Savings: {:.2}%",
            self.n_nodes,
            self.max_depth,
            self.n_features,
            self.file_size,
            self.memory_usage,
            self.memory_savings_ratio() * 100.0
        )
    }
}

/// Memory-mapped ensemble for storing multiple trees efficiently
pub struct MMapEnsemble {
    /// Individual tree files
    trees: Vec<MMapTree>,
    /// Ensemble metadata
    metadata: EnsembleMetadata,
    /// Base directory for tree files
    base_dir: PathBuf,
}

/// Metadata for ensembles
#[derive(Debug, Clone)]
pub struct EnsembleMetadata {
    /// Number of trees in the ensemble
    pub n_estimators: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of classes (for classification)
    pub n_classes: usize,
    /// Whether this is a classification ensemble
    pub is_classifier: bool,
    /// Tree file naming pattern
    pub tree_pattern: String,
}

impl MMapEnsemble {
    /// Create a new memory-mapped ensemble
    pub fn create<P: AsRef<Path>>(base_dir: P, metadata: EnsembleMetadata) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&base_dir)?;

        Ok(Self {
            trees: Vec::new(),
            metadata,
            base_dir,
        })
    }

    /// Add a tree to the ensemble
    pub fn add_tree(&mut self, tree_metadata: TreeMetadata) -> Result<&mut MMapTree> {
        let tree_idx = self.trees.len();
        let tree_path = self.base_dir.join(format!("tree_{:04}.mmap", tree_idx));

        let tree = MMapTree::create(tree_path, tree_metadata)?;
        self.trees.push(tree);

        Ok(self.trees.last_mut().unwrap())
    }

    /// Load existing ensemble from directory
    pub fn load<P: AsRef<Path>>(base_dir: P, metadata: EnsembleMetadata) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        let mut trees = Vec::new();

        // Load all tree files in order
        for tree_idx in 0..metadata.n_estimators {
            let tree_path = base_dir.join(format!("tree_{:04}.mmap", tree_idx));

            if tree_path.exists() {
                let tree = MMapTree::open(tree_path)?;
                trees.push(tree);
            } else {
                return Err(SklearsError::InvalidInput(format!(
                    "Missing tree file: tree_{:04}.mmap",
                    tree_idx
                )));
            }
        }

        Ok(Self {
            trees,
            metadata,
            base_dir,
        })
    }

    /// Make predictions using all trees in the ensemble
    pub fn predict(&mut self, features: &Array2<Float>) -> Result<Array1<Float>> {
        if self.trees.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No trees in ensemble".to_string(),
            ));
        }

        let n_samples = features.nrows();
        let mut ensemble_predictions = Array1::<Float>::zeros(n_samples);

        if self.metadata.is_classifier {
            // For classification, use voting
            let mut vote_counts = Array2::<usize>::zeros((n_samples, self.metadata.n_classes));

            for tree in &mut self.trees {
                let tree_predictions = tree.predict(features)?;

                for (sample_idx, &prediction) in tree_predictions.iter().enumerate() {
                    let class_idx = prediction as usize;
                    if class_idx < self.metadata.n_classes {
                        vote_counts[[sample_idx, class_idx]] += 1;
                    }
                }
            }

            // Find majority class for each sample
            for sample_idx in 0..n_samples {
                let mut max_votes = 0;
                let mut predicted_class = 0;

                for class_idx in 0..self.metadata.n_classes {
                    let votes = vote_counts[[sample_idx, class_idx]];
                    if votes > max_votes {
                        max_votes = votes;
                        predicted_class = class_idx;
                    }
                }

                ensemble_predictions[sample_idx] = predicted_class as Float;
            }
        } else {
            // For regression, average predictions
            for tree in &mut self.trees {
                let tree_predictions = tree.predict(features)?;
                ensemble_predictions = ensemble_predictions + tree_predictions;
            }
            ensemble_predictions /= self.trees.len() as Float;
        }

        Ok(ensemble_predictions)
    }

    /// Get ensemble statistics
    pub fn get_stats(&self) -> EnsembleStats {
        let total_nodes: usize = self.trees.iter().map(|t| t.n_nodes).sum();
        let total_file_size: u64 = self
            .trees
            .iter()
            .map(|t| t.file.metadata().map(|m| m.len()).unwrap_or(0))
            .sum();
        let total_memory_usage: usize = self.trees.iter().map(|t| t.estimate_memory_usage()).sum();

        EnsembleStats {
            n_trees: self.trees.len(),
            total_nodes,
            avg_nodes_per_tree: if self.trees.is_empty() {
                0.0
            } else {
                total_nodes as f64 / self.trees.len() as f64
            },
            total_file_size,
            total_memory_usage,
            memory_savings_ratio: 1.0
                - (self.trees.len() * std::mem::size_of::<File>()) as f64
                    / total_memory_usage as f64,
        }
    }
}

/// Statistics for memory-mapped ensembles
#[derive(Debug, Clone)]
pub struct EnsembleStats {
    /// Number of trees
    pub n_trees: usize,
    /// Total number of nodes across all trees
    pub total_nodes: usize,
    /// Average nodes per tree
    pub avg_nodes_per_tree: f64,
    /// Total file size on disk
    pub total_file_size: u64,
    /// Total estimated memory usage if loaded into RAM
    pub total_memory_usage: usize,
    /// Memory savings ratio
    pub memory_savings_ratio: f64,
}

impl EnsembleStats {
    /// Format stats as string
    pub fn summary(&self) -> String {
        format!(
            "Ensemble Statistics:\n\
             Trees: {}\n\
             Total Nodes: {}\n\
             Avg Nodes/Tree: {:.1}\n\
             Total File Size: {} bytes\n\
             Est. Total Memory: {} bytes\n\
             Memory Savings: {:.2}%",
            self.n_trees,
            self.total_nodes,
            self.avg_nodes_per_tree,
            self.total_file_size,
            self.total_memory_usage,
            self.memory_savings_ratio * 100.0
        )
    }
}

/// Utilities for working with memory-mapped trees
pub struct MMapUtils;

impl MMapUtils {
    /// Convert a standard tree to memory-mapped format
    pub fn convert_tree_to_mmap<P: AsRef<Path>>(
        tree_nodes: &[StandardTreeNode],
        path: P,
        metadata: TreeMetadata,
    ) -> Result<MMapTree> {
        let mut mmap_tree = MMapTree::create(path, metadata)?;
        let mut node_offsets = HashMap::new();

        // Write all nodes and record their offsets
        for (i, node) in tree_nodes.iter().enumerate() {
            let mmap_node = MMapTreeNode {
                feature_idx: if node.is_leaf {
                    -1
                } else {
                    node.feature_idx as i32
                },
                threshold: if node.is_leaf {
                    node.prediction
                } else {
                    node.threshold
                },
                left_child_offset: -1,  // Will be updated later
                right_child_offset: -1, // Will be updated later
                n_samples: node.n_samples,
                impurity: node.impurity,
                depth: node.depth,
                _padding: [0; 6],
            };

            let offset = mmap_tree.write_node(&mmap_node)?;
            node_offsets.insert(i, offset);
        }

        // Update child offsets
        for (i, node) in tree_nodes.iter().enumerate() {
            if !node.is_leaf {
                let node_offset = node_offsets[&i];
                let mut mmap_node = mmap_tree.read_node(node_offset)?;

                if let Some(left_idx) = node.left_child {
                    mmap_node.left_child_offset = node_offsets[&left_idx];
                }
                if let Some(right_idx) = node.right_child {
                    mmap_node.right_child_offset = node_offsets[&right_idx];
                }

                // Write updated node back
                mmap_tree.file.seek(SeekFrom::Start(node_offset as u64))?;
                // SAFETY: MMapTreeNode is a plain-old-data type with repr(C),
                // and we're creating a slice view of its byte representation.
                // The lifetime of the slice is bound to the mmap_node reference.
                let node_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &mmap_node as *const MMapTreeNode as *const u8,
                        MMapTreeNode::size(),
                    )
                };
                mmap_tree.file.write_all(node_bytes)?;
            }
        }

        // Set root offset (assuming root is at index 0)
        if let Some(&root_offset) = node_offsets.get(&0) {
            mmap_tree.set_root_offset(root_offset)?;
        }

        Ok(mmap_tree)
    }

    /// Estimate memory savings from using memory-mapped storage
    pub fn estimate_memory_savings(tree_nodes: &[StandardTreeNode]) -> (usize, usize, f64) {
        let in_memory_size = tree_nodes.len() * std::mem::size_of::<StandardTreeNode>();
        let mmap_size = std::mem::size_of::<File>() + TreeMetadata::size();
        let savings_ratio = 1.0 - (mmap_size as f64 / in_memory_size as f64);

        (in_memory_size, mmap_size, savings_ratio)
    }
}

/// Standard tree node for conversion purposes
#[derive(Debug, Clone)]
pub struct StandardTreeNode {
    pub is_leaf: bool,
    pub feature_idx: usize,
    pub threshold: f64,
    pub prediction: f64,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub n_samples: u32,
    pub impurity: f64,
    pub depth: u16,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_tree_creation() {
        let temp_dir = tempdir().unwrap();
        let tree_path = temp_dir.path().join("test_tree.mmap");

        let metadata = TreeMetadata::new(4, 2, true, 10, 1);
        let mut tree = MMapTree::create(&tree_path, metadata).unwrap();

        // Create a simple tree: root -> left leaf, right leaf
        let left_leaf = MMapTreeNode::new_leaf(0.0, 10, 0.1, 1);
        let right_leaf = MMapTreeNode::new_leaf(1.0, 15, 0.2, 1);

        let left_offset = tree.write_node(&left_leaf).unwrap();
        let right_offset = tree.write_node(&right_leaf).unwrap();

        let root = MMapTreeNode::new_internal(0, 0.5, left_offset, right_offset, 25, 0.5, 0);
        let root_offset = tree.write_node(&root).unwrap();
        tree.set_root_offset(root_offset).unwrap();

        // Test prediction
        let features = Array1::from(vec![0.3, 0.0, 0.0, 0.0]);
        let prediction = tree.predict_sample(&features).unwrap();
        assert_eq!(prediction, 0.0); // Should go to left leaf

        let features = Array1::from(vec![0.7, 0.0, 0.0, 0.0]);
        let prediction = tree.predict_sample(&features).unwrap();
        assert_eq!(prediction, 1.0); // Should go to right leaf
    }

    #[test]
    fn test_tree_persistence() {
        let temp_dir = tempdir().unwrap();
        let tree_path = temp_dir.path().join("persist_tree.mmap");

        // Create and save tree
        {
            let metadata = TreeMetadata::new(2, 2, true, 5, 1);
            let mut tree = MMapTree::create(&tree_path, metadata).unwrap();

            let leaf = MMapTreeNode::new_leaf(42.0, 100, 0.0, 0);
            let root_offset = tree.write_node(&leaf).unwrap();
            tree.set_root_offset(root_offset).unwrap();
        }

        // Load and test tree
        {
            let mut tree = MMapTree::open(&tree_path).unwrap();
            let features = Array1::from(vec![1.0, 2.0]);
            let prediction = tree.predict_sample(&features).unwrap();
            assert_eq!(prediction, 42.0);
        }
    }
}

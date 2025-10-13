//! Core tree structures for incremental decision trees
//!
//! This module provides the fundamental data structures and traits for incremental decision tree
//! algorithms. It includes the main IncrementalDecisionTree implementation, streaming tree traits,
//! and basic tree node structures that can be updated incrementally as new data arrives.
//!
//! ## Core Components
//!
//! - **IncrementalDecisionTree**: Main incremental decision tree for streaming data
//! - **StreamingTreeModel**: Trait defining the interface for streaming tree models
//! - **SimpleIncrementalTree**: Basic implementation of streaming tree model
//! - **IncrementalTreeNode**: Tree node that can be updated with new samples
//! - **IncrementalTreeStats**: Performance statistics and monitoring

use super::simd_operations as simd_tree;
use super::streaming_infrastructure::{
    ConceptDriftDetector, IncrementalTreeConfig, StreamingBuffer,
};
use crate::{DecisionTreeConfig, Trained, Untrained};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Incremental Decision Tree for streaming data
pub struct IncrementalDecisionTree<State = Untrained> {
    /// Configuration
    config: IncrementalTreeConfig,
    /// Current tree (if trained)
    tree: Option<Box<dyn StreamingTreeModel>>,
    /// Streaming data buffer
    buffer: StreamingBuffer,
    /// Concept drift detector
    drift_detector: ConceptDriftDetector,
    /// Number of samples processed
    samples_processed: usize,
    /// State marker
    state: PhantomData<State>,
}

/// Trait for streaming tree models that can be updated incrementally
pub trait StreamingTreeModel: Send + Sync {
    /// Predict on new data
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;

    /// Update the model with new data
    fn update(&mut self, x: &Array2<f64>, y: &Array1<f64>, weights: &Array1<f64>) -> Result<()>;

    /// Get current model accuracy on recent data
    fn get_accuracy(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64>;

    /// Reset/rebuild the model
    fn rebuild(&mut self, x: &Array2<f64>, y: &Array1<f64>, weights: &Array1<f64>) -> Result<()>;
}

/// Simple incremental tree implementation
#[derive(Debug, Clone)]
pub struct SimpleIncrementalTree {
    /// Tree nodes
    nodes: Vec<IncrementalTreeNode>,
    /// Configuration
    config: DecisionTreeConfig,
    /// Feature importance scores
    feature_importances: HashMap<usize, f64>,
}

/// Incremental tree node that can be updated
#[derive(Debug, Clone)]
pub struct IncrementalTreeNode {
    /// Node ID
    pub id: usize,
    /// Feature index for split (None for leaf)
    pub feature_idx: Option<usize>,
    /// Split threshold (None for leaf)
    pub threshold: Option<f64>,
    /// Prediction value
    pub prediction: f64,
    /// Number of samples seen
    pub n_samples: usize,
    /// Sum of target values
    pub sum_y: f64,
    /// Sum of squared target values
    pub sum_y_squared: f64,
    /// Child node IDs
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    /// Is this a leaf node?
    pub is_leaf: bool,
    /// Node depth
    pub depth: usize,
}

impl IncrementalTreeNode {
    pub fn new_leaf(id: usize, depth: usize) -> Self {
        Self {
            id,
            feature_idx: None,
            threshold: None,
            prediction: 0.0,
            n_samples: 0,
            sum_y: 0.0,
            sum_y_squared: 0.0,
            left_child: None,
            right_child: None,
            is_leaf: true,
            depth,
        }
    }

    /// Update node statistics with new sample
    pub fn update_stats(&mut self, y: f64, weight: f64) {
        self.n_samples += 1;
        self.sum_y += y * weight;
        self.sum_y_squared += y * y * weight;

        // Update prediction (mean for regression)
        if self.n_samples > 0 {
            self.prediction = self.sum_y / self.n_samples as f64;
        }
    }

    /// Calculate node impurity (MSE for regression)
    pub fn calculate_impurity(&self) -> f64 {
        if self.n_samples <= 1 {
            return 0.0;
        }

        let mean = self.sum_y / self.n_samples as f64;
        let variance = (self.sum_y_squared / self.n_samples as f64) - (mean * mean);
        variance.max(0.0)
    }
}

impl StreamingTreeModel for SimpleIncrementalTree {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut node_id = 0; // Start at root

            loop {
                if node_id >= self.nodes.len() {
                    return Err(SklearsError::PredictError("Invalid node ID".to_string()));
                }

                let node = &self.nodes[node_id];

                if node.is_leaf {
                    predictions[sample_idx] = node.prediction;
                    break;
                }

                if let (Some(feature_idx), Some(threshold)) = (node.feature_idx, node.threshold) {
                    if feature_idx >= x.ncols() {
                        return Err(SklearsError::PredictError(
                            "Invalid feature index".to_string(),
                        ));
                    }

                    let feature_value = x[[sample_idx, feature_idx]];

                    if feature_value <= threshold {
                        node_id = node.left_child.unwrap_or(node_id);
                    } else {
                        node_id = node.right_child.unwrap_or(node_id);
                    }
                } else {
                    predictions[sample_idx] = node.prediction;
                    break;
                }
            }
        }

        Ok(predictions)
    }

    fn update(&mut self, x: &Array2<f64>, y: &Array1<f64>, weights: &Array1<f64>) -> Result<()> {
        // Simple update: add samples to leaf nodes and update statistics
        for sample_idx in 0..x.nrows() {
            let mut node_id = 0;

            // Find the leaf node for this sample
            loop {
                if node_id >= self.nodes.len() {
                    break;
                }

                let node = &self.nodes[node_id];

                if node.is_leaf {
                    // Update leaf node statistics
                    let node = &mut self.nodes[node_id];
                    node.update_stats(y[sample_idx], weights[sample_idx]);
                    break;
                }

                if let (Some(feature_idx), Some(threshold)) = (node.feature_idx, node.threshold) {
                    let feature_value = x[[sample_idx, feature_idx]];

                    if feature_value <= threshold {
                        node_id = node.left_child.unwrap_or(node_id);
                    } else {
                        node_id = node.right_child.unwrap_or(node_id);
                    }
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    fn get_accuracy(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Calculate MSE for regression using SIMD acceleration
        let pred_slice: Vec<f64> = predictions.iter().cloned().collect();
        let target_slice: Vec<f64> = y.iter().cloned().collect();
        let mse = simd_tree::simd_mse_evaluation(&pred_slice, &target_slice);

        // Return R² as accuracy measure (1 - MSE/variance)
        let y_mean = y.mean().unwrap_or(0.0);
        let y_variance = y.iter().map(|&val| (val - y_mean).powi(2)).sum::<f64>() / y.len() as f64;

        if y_variance == 0.0 {
            Ok(1.0)
        } else {
            Ok((1.0 - mse / y_variance).max(0.0))
        }
    }

    fn rebuild(&mut self, x: &Array2<f64>, y: &Array1<f64>, weights: &Array1<f64>) -> Result<()> {
        // Simple rebuild: create a single leaf node with all data
        self.nodes.clear();

        let mut root = IncrementalTreeNode::new_leaf(0, 0);

        for i in 0..x.nrows() {
            root.update_stats(y[i], weights[i]);
        }

        self.nodes.push(root);

        Ok(())
    }
}

impl IncrementalDecisionTree<Untrained> {
    pub fn new(config: IncrementalTreeConfig) -> Self {
        let buffer = StreamingBuffer::new(config.buffer_size);
        let drift_detector = ConceptDriftDetector::new(config.window_size, config.drift_threshold);

        Self {
            config,
            tree: None,
            buffer,
            drift_detector,
            samples_processed: 0,
            state: PhantomData,
        }
    }

    /// Add a new sample to the streaming tree
    pub fn add_sample(
        &mut self,
        x: Vec<f64>,
        y: f64,
        weight: Option<f64>,
        timestamp: Option<u64>,
    ) -> Result<()> {
        let weight = weight.unwrap_or(1.0);
        let timestamp = timestamp.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });

        self.buffer.add_sample(x, y, weight, timestamp);
        self.samples_processed += 1;

        // Check if we should update the tree
        if self.samples_processed % self.config.update_frequency == 0 {
            self.update_tree()?;
        }

        Ok(())
    }

    /// Update the tree with buffered data
    fn update_tree(&mut self) -> Result<()> {
        if self.buffer.len() < self.config.min_samples_before_build {
            return Ok(());
        }

        let (x_data, y_data, weights) = self.buffer.to_arrays()?;

        if let Some(ref mut tree) = self.tree {
            // Update existing tree
            tree.update(&x_data, &y_data, &weights)?;

            // Check for concept drift
            if self.config.enable_drift_detection {
                let accuracy = tree.get_accuracy(&x_data, &y_data)?;

                if self.drift_detector.update(accuracy) {
                    // Drift detected, rebuild tree
                    tree.rebuild(&x_data, &y_data, &weights)?;
                }
            }
        } else {
            // Build initial tree
            let mut new_tree = SimpleIncrementalTree {
                nodes: Vec::new(),
                config: self.config.base_config.clone(),
                feature_importances: HashMap::new(),
            };

            new_tree.rebuild(&x_data, &y_data, &weights)?;
            self.tree = Some(Box::new(new_tree));
        }

        Ok(())
    }

    /// Get current tree performance statistics
    pub fn get_performance_stats(&self) -> Result<IncrementalTreeStats> {
        if let Some(ref tree) = self.tree {
            let (x_data, y_data, _) = self.buffer.to_arrays()?;
            let accuracy = tree.get_accuracy(&x_data, &y_data)?;

            Ok(IncrementalTreeStats {
                samples_processed: self.samples_processed,
                buffer_size: self.buffer.len(),
                current_accuracy: accuracy,
                drift_detections: 0, // Would track this in real implementation
                tree_updates: self.samples_processed / self.config.update_frequency,
            })
        } else {
            Ok(IncrementalTreeStats {
                samples_processed: self.samples_processed,
                buffer_size: self.buffer.len(),
                current_accuracy: 0.0,
                drift_detections: 0,
                tree_updates: 0,
            })
        }
    }
}

impl IncrementalDecisionTree<Trained> {
    /// Predict on new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if let Some(ref tree) = self.tree {
            tree.predict(x)
        } else {
            Err(SklearsError::PredictError("Tree not trained".to_string()))
        }
    }
}

/// Performance statistics for incremental trees
#[derive(Debug, Clone)]
pub struct IncrementalTreeStats {
    /// Total number of samples processed
    pub samples_processed: usize,
    /// Current buffer size
    pub buffer_size: usize,
    /// Current model accuracy
    pub current_accuracy: f64,
    /// Number of concept drift detections
    pub drift_detections: usize,
    /// Number of tree updates performed
    pub tree_updates: usize,
}

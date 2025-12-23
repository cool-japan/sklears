//! Ensemble methods for incremental learning
//!
//! This module provides advanced ensemble methods that can learn incrementally from streaming data,
//! including online gradient boosting and incremental random forest implementations. These algorithms
//! combine multiple weak learners to create robust models capable of handling concept drift and
//! adapting to changing data distributions.
//!
//! ## Core Components
//!
//! - **OnlineGradientBoosting**: Incremental gradient boosting with configurable loss functions
//! - **IncrementalRandomForest**: Streaming random forest with bootstrap sampling and feature subsets
//! - **Adaptive Ensembles**: Dynamic ensemble size adjustment based on performance
//! - **Concept Drift Handling**: Automatic model adaptation when data distribution changes

use super::core_tree_structures::StreamingTreeModel;
use super::hoeffding_tree::{HoeffdingTree, HoeffdingTreeConfig};
use super::simd_operations as simd_tree;
use super::streaming_infrastructure::{ConceptDriftDetector, StreamingBuffer};
use crate::MaxFeatures;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

/// Online Gradient Boosting for streaming data
///
/// Implements an online variant of gradient boosting that can update
/// incrementally as new data arrives.
#[derive(Debug)]
pub struct OnlineGradientBoosting {
    /// Base learners (Hoeffding trees)
    estimators: Vec<HoeffdingTree>,
    /// Learning rate
    learning_rate: f64,
    /// Configuration for base trees
    tree_config: HoeffdingTreeConfig,
    /// Number of estimators to maintain
    n_estimators: usize,
    /// Current number of active estimators
    current_estimators: usize,
    /// Loss function type
    loss_function: OnlineLossFunction,
    /// Feature importance scores
    feature_importances: HashMap<usize, f64>,
}

/// Loss function types for online gradient boosting
#[derive(Debug, Clone, Copy)]
pub enum OnlineLossFunction {
    /// Squared loss for regression
    SquaredLoss,
    /// Logistic loss for binary classification
    LogisticLoss,
    /// Exponential loss (AdaBoost-style)
    ExponentialLoss,
}

/// Configuration for Online Gradient Boosting
#[derive(Debug, Clone)]
pub struct OnlineGradientBoostingConfig {
    /// Number of estimators to maintain
    pub n_estimators: usize,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Configuration for base Hoeffding trees
    pub tree_config: HoeffdingTreeConfig,
    /// Loss function
    pub loss_function: OnlineLossFunction,
    /// Minimum samples before adding new estimator
    pub min_samples_per_estimator: usize,
    /// Maximum memory usage (bytes)
    pub max_memory_usage: Option<usize>,
}

impl Default for OnlineGradientBoostingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            learning_rate: 0.1,
            tree_config: HoeffdingTreeConfig::default(),
            loss_function: OnlineLossFunction::SquaredLoss,
            min_samples_per_estimator: 100,
            max_memory_usage: Some(100_000_000), // 100MB
        }
    }
}

impl OnlineGradientBoosting {
    /// Create a new Online Gradient Boosting model
    pub fn new(config: OnlineGradientBoostingConfig, n_features: usize) -> Self {
        Self {
            estimators: Vec::with_capacity(config.n_estimators),
            learning_rate: config.learning_rate,
            tree_config: config.tree_config,
            n_estimators: config.n_estimators,
            current_estimators: 0,
            loss_function: config.loss_function,
            feature_importances: HashMap::new(),
        }
    }

    /// Update model with new sample
    pub fn update_single(&mut self, x: &[f64], y: f64) -> Result<()> {
        // Compute current prediction
        let current_prediction = self.predict_single(x)?;

        // Compute gradient
        let gradient = self.compute_gradient(y, current_prediction);

        // Update existing estimators
        for estimator in &mut self.estimators {
            estimator.update(x, gradient, None)?;
        }

        // Add new estimator if needed
        if self.current_estimators < self.n_estimators {
            let mut new_estimator = HoeffdingTree::new(self.tree_config.clone(), x.len());
            new_estimator.update(x, gradient, None)?;
            self.estimators.push(new_estimator);
            self.current_estimators += 1;
        }

        Ok(())
    }

    /// Compute gradient based on loss function
    pub fn compute_gradient(&self, y_true: f64, y_pred: f64) -> f64 {
        match self.loss_function {
            OnlineLossFunction::SquaredLoss => 2.0 * (y_pred - y_true),
            OnlineLossFunction::LogisticLoss => {
                let prob = 1.0 / (1.0 + (-y_pred).exp());
                prob - y_true
            }
            OnlineLossFunction::ExponentialLoss => {
                if y_true * y_pred < 0.0 {
                    y_true * (-y_true * y_pred).exp()
                } else {
                    0.0
                }
            }
        }
    }

    /// Predict on a single sample
    pub fn predict_single(&self, x: &[f64]) -> Result<f64> {
        let mut prediction = 0.0;

        for estimator in &self.estimators {
            prediction += self.learning_rate * estimator.predict_single(x)?;
        }

        Ok(prediction)
    }

    /// Get ensemble statistics
    pub fn get_ensemble_stats(&self) -> OnlineGradientBoostingStats {
        let total_nodes: usize = self
            .estimators
            .iter()
            .map(|est| est.get_stats().n_nodes)
            .sum();
        let total_samples: usize = self
            .estimators
            .iter()
            .map(|est| est.get_stats().total_samples)
            .sum();
        let memory_usage: usize = self
            .estimators
            .iter()
            .map(|est| est.get_stats().memory_usage)
            .sum();

        OnlineGradientBoostingStats {
            n_estimators: self.current_estimators,
            total_nodes,
            total_samples,
            memory_usage,
            avg_tree_depth: if self.current_estimators > 0 {
                self.estimators
                    .iter()
                    .map(|est| est.get_stats().max_depth)
                    .sum::<usize>() as f64
                    / self.current_estimators as f64
            } else {
                0.0
            },
        }
    }
}

/// Statistics for Online Gradient Boosting
#[derive(Debug, Clone)]
pub struct OnlineGradientBoostingStats {
    /// Number of estimators in the ensemble
    pub n_estimators: usize,
    /// Total number of nodes across all trees
    pub total_nodes: usize,
    /// Total samples processed
    pub total_samples: usize,
    /// Total memory usage (bytes)
    pub memory_usage: usize,
    /// Average tree depth
    pub avg_tree_depth: f64,
}

impl StreamingTreeModel for OnlineGradientBoosting {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let mut predictions = Array1::zeros(x.nrows());

        for (i, sample) in x.rows().into_iter().enumerate() {
            let sample_vec: Vec<f64> = sample.to_vec();
            predictions[i] = self.predict_single(&sample_vec)?;
        }

        Ok(predictions)
    }

    fn update(&mut self, x: &Array2<f64>, y: &Array1<f64>, _weights: &Array1<f64>) -> Result<()> {
        for (i, sample) in x.rows().into_iter().enumerate() {
            let sample_vec: Vec<f64> = sample.to_vec();
            self.update_single(&sample_vec, y[i])?;
        }

        Ok(())
    }

    fn get_accuracy(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;

        match self.loss_function {
            OnlineLossFunction::LogisticLoss => {
                // Binary classification accuracy
                let correct = predictions
                    .iter()
                    .zip(y.iter())
                    .map(|(&pred, &actual)| {
                        let predicted_class = if pred > 0.0 { 1.0 } else { 0.0 };
                        if (predicted_class - actual).abs() < 1e-6 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>();
                Ok(correct / predictions.len() as f64)
            }
            _ => {
                // Regression R²
                let y_mean = y.mean().unwrap_or(0.0);
                let ss_res = predictions
                    .iter()
                    .zip(y.iter())
                    .map(|(&pred, &actual)| (actual - pred).powi(2))
                    .sum::<f64>();
                let ss_tot = y
                    .iter()
                    .map(|&actual| (actual - y_mean).powi(2))
                    .sum::<f64>();

                if ss_tot == 0.0 {
                    Ok(1.0)
                } else {
                    Ok((1.0 - ss_res / ss_tot).max(0.0))
                }
            }
        }
    }

    fn rebuild(&mut self, x: &Array2<f64>, y: &Array1<f64>, weights: &Array1<f64>) -> Result<()> {
        // Clear existing estimators
        self.estimators.clear();
        self.current_estimators = 0;

        // Re-train on all data
        self.update(x, y, weights)
    }
}

/// Configuration for Incremental Random Forest
#[derive(Debug, Clone)]
pub struct IncrementalRandomForestConfig {
    /// Number of trees in the forest
    pub n_estimators: usize,
    /// Base tree configuration
    pub base_tree_config: HoeffdingTreeConfig,
    /// Fraction of features to consider at each split
    pub max_features: MaxFeatures,
    /// Use bootstrap sampling for each tree
    pub bootstrap: bool,
    /// Enable out-of-bag error estimation
    pub oob_score: bool,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Window size for concept drift detection
    pub window_size: usize,
    /// Concept drift detection threshold
    pub drift_threshold: f64,
    /// Enable adaptive ensemble size
    pub adaptive_ensemble_size: bool,
    /// Maximum number of trees to maintain
    pub max_ensemble_size: Option<usize>,
    /// Minimum accuracy threshold to retain a tree
    pub min_tree_accuracy: f64,
}

impl Default for IncrementalRandomForestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            base_tree_config: HoeffdingTreeConfig::default(),
            max_features: MaxFeatures::Sqrt,
            bootstrap: true,
            oob_score: true,
            random_state: None,
            window_size: 1000,
            drift_threshold: 0.05,
            adaptive_ensemble_size: false,
            max_ensemble_size: Some(100),
            min_tree_accuracy: 0.5,
        }
    }
}

/// Tree statistics for incremental random forest
#[derive(Debug, Clone)]
pub struct RandomForestTreeStats {
    /// Tree ID
    pub tree_id: usize,
    /// Number of samples seen
    pub n_samples: usize,
    /// Current accuracy on validation data
    pub accuracy: f64,
    /// Age of the tree (number of updates)
    pub age: usize,
    /// Feature importance scores
    pub feature_importances: HashMap<usize, f64>,
    /// Out-of-bag accuracy
    pub oob_accuracy: Option<f64>,
    /// Sample indices used for this tree (for bootstrap tracking)
    pub bootstrap_indices: Vec<usize>,
}

impl RandomForestTreeStats {
    pub fn new(tree_id: usize) -> Self {
        Self {
            tree_id,
            n_samples: 0,
            accuracy: 0.0,
            age: 0,
            feature_importances: HashMap::new(),
            oob_accuracy: None,
            bootstrap_indices: Vec::new(),
        }
    }
}

/// Incremental Random Forest for streaming data
#[derive(Debug)]
pub struct IncrementalRandomForest {
    /// Collection of Hoeffding trees
    trees: Vec<HoeffdingTree>,
    /// Tree statistics and metadata
    tree_stats: Vec<RandomForestTreeStats>,
    /// Configuration
    config: IncrementalRandomForestConfig,
    /// Global feature importance scores
    feature_importances: HashMap<usize, f64>,
    /// Out-of-bag samples for evaluation
    oob_buffer: StreamingBuffer,
    /// Concept drift detector
    drift_detector: ConceptDriftDetector,
    /// Random number generator
    rng: Random,
    /// Number of features in the dataset
    n_features: Option<usize>,
    /// Sample counter for bootstrap tracking
    sample_counter: usize,
}

impl IncrementalRandomForest {
    /// Create a new incremental random forest
    pub fn new(config: IncrementalRandomForestConfig) -> Self {
        let mut rng = scirs2_core::random::thread_rng();

        // Set seed if provided
        if let Some(seed) = config.random_state {
            rng = Random::seed_from_u64(seed);
        }

        let drift_detector = ConceptDriftDetector::new(config.window_size, config.drift_threshold);
        let oob_buffer = StreamingBuffer::new(config.window_size);

        Self {
            trees: Vec::new(),
            tree_stats: Vec::new(),
            config,
            feature_importances: HashMap::new(),
            oob_buffer,
            drift_detector,
            rng,
            n_features: None,
            sample_counter: 0,
        }
    }

    /// Initialize the forest with the specified number of trees
    fn initialize_trees(&mut self, n_features: usize) -> Result<()> {
        self.n_features = Some(n_features);

        for i in 0..self.config.n_estimators {
            let mut tree_config = self.config.base_tree_config.clone();

            // Randomize some tree parameters for diversity
            tree_config.confidence = self.rng.random_range(0.9, 0.99);
            tree_config.grace_period = self.rng.gen_range(150..250);

            let tree = HoeffdingTree::new(tree_config, n_features);
            let stats = RandomForestTreeStats::new(i);

            self.trees.push(tree);
            self.tree_stats.push(stats);
        }

        Ok(())
    }

    /// Get feature subset for a tree based on max_features configuration
    fn get_feature_subset(&mut self, n_features: usize) -> Vec<usize> {
        let max_features_count = match self.config.max_features {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => (n_features as f64).sqrt().ceil() as usize,
            MaxFeatures::Log2 => ((n_features as f64).log2().ceil() as usize).max(1),
            MaxFeatures::Number(count) => count.min(n_features),
            MaxFeatures::Fraction(frac) => ((n_features as f64 * frac).ceil() as usize).max(1),
        };

        let mut features: Vec<usize> = (0..n_features).collect();

        // Shuffle and take the first max_features_count
        self.rng.shuffle(&mut features);
        features.truncate(max_features_count);
        features.sort(); // Keep sorted for consistency

        features
    }

    /// Create bootstrap sample indices
    fn create_bootstrap_sample(&mut self, n_samples: usize) -> Vec<usize> {
        if !self.config.bootstrap {
            return (0..n_samples).collect();
        }

        let mut bootstrap_indices = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            bootstrap_indices.push(self.rng.gen_range(0..n_samples));
        }

        bootstrap_indices
    }

    /// Update a single tree with new data
    fn update_tree(
        &mut self,
        tree_idx: usize,
        x: &Array2<f64>,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        bootstrap_indices: &[usize],
        feature_subset: &[usize],
    ) -> Result<()> {
        if tree_idx >= self.trees.len() {
            return Err(SklearsError::InvalidInput(
                "Tree index out of bounds".to_string(),
            ));
        }

        // Create subsampled data
        let n_samples = bootstrap_indices.len();
        let n_features = feature_subset.len();

        let mut x_sub = Array2::zeros((n_samples, n_features));
        let mut y_sub = Array1::zeros(n_samples);
        let mut w_sub = Array1::zeros(n_samples);

        for (new_idx, &orig_idx) in bootstrap_indices.iter().enumerate() {
            y_sub[new_idx] = y[orig_idx];
            w_sub[new_idx] = weights[orig_idx];

            for (new_feat_idx, &orig_feat_idx) in feature_subset.iter().enumerate() {
                x_sub[[new_idx, new_feat_idx]] = x[[orig_idx, orig_feat_idx]];
            }
        }

        // Update tree with individual samples
        for i in 0..n_samples {
            let x_sample: Vec<f64> = x_sub.row(i).to_vec();
            let y_sample = y_sub[i];
            let class_label = if y_sample.fract() == 0.0 {
                Some(y_sample as i32)
            } else {
                None
            };
            self.trees[tree_idx].update(&x_sample, y_sample, class_label)?;
        }

        // Update statistics
        self.tree_stats[tree_idx].n_samples += n_samples;
        self.tree_stats[tree_idx].age += 1;
        self.tree_stats[tree_idx].bootstrap_indices = bootstrap_indices.to_vec();

        Ok(())
    }

    /// Calculate out-of-bag predictions for a sample
    fn calculate_oob_prediction(&self, sample_idx: usize, x_sample: &[f64]) -> Result<f64> {
        let mut predictions = Vec::new();

        for (tree_idx, tree_stats) in self.tree_stats.iter().enumerate() {
            // Check if this sample was out-of-bag for this tree
            if !tree_stats.bootstrap_indices.contains(&sample_idx) {
                // This tree didn't see this sample during training
                let x_array = Array2::from_shape_vec((1, x_sample.len()), x_sample.to_vec())
                    .map_err(|e| {
                        SklearsError::InvalidInput(format!("Array creation error: {}", e))
                    })?;

                let pred = self.trees[tree_idx].predict(&x_array)?;
                predictions.push(pred[0]);
            }
        }

        if predictions.is_empty() {
            return Err(SklearsError::PredictError(
                "No out-of-bag predictions available".to_string(),
            ));
        }

        // Return mean prediction
        Ok(predictions.iter().sum::<f64>() / predictions.len() as f64)
    }

    /// Update feature importance scores
    fn update_feature_importances(&mut self) {
        self.feature_importances.clear();

        for tree in &self.trees {
            for (&feature_idx, &importance) in tree.get_feature_importances() {
                *self.feature_importances.entry(feature_idx).or_insert(0.0) += importance;
            }
        }

        // Normalize by number of trees
        let n_trees = self.trees.len() as f64;
        for importance in self.feature_importances.values_mut() {
            *importance /= n_trees;
        }
    }

    /// Remove poorly performing trees
    fn prune_poor_trees(&mut self) -> Result<()> {
        if !self.config.adaptive_ensemble_size {
            return Ok(());
        }

        let mut trees_to_remove = Vec::new();

        for (idx, stats) in self.tree_stats.iter().enumerate() {
            if stats.accuracy < self.config.min_tree_accuracy && stats.age > 10 {
                trees_to_remove.push(idx);
            }
        }

        // Remove trees in reverse order to maintain indices
        trees_to_remove.reverse();
        for idx in trees_to_remove {
            self.trees.remove(idx);
            self.tree_stats.remove(idx);
        }

        Ok(())
    }

    /// Add new trees if ensemble is below target size
    fn add_new_trees(&mut self) -> Result<()> {
        if !self.config.adaptive_ensemble_size {
            return Ok(());
        }

        let current_size = self.trees.len();
        let target_size = self.config.n_estimators;

        if current_size < target_size {
            let n_features = self.n_features.ok_or_else(|| {
                SklearsError::InvalidInput("Features not initialized".to_string())
            })?;

            for i in current_size..target_size {
                let mut tree_config = self.config.base_tree_config.clone();
                tree_config.confidence = self.rng.random_range(0.9, 0.99);

                let tree = HoeffdingTree::new(tree_config, self.n_features.unwrap_or(1));
                let stats = RandomForestTreeStats::new(i);

                self.trees.push(tree);
                self.tree_stats.push(stats);
            }
        }

        Ok(())
    }

    /// Get the current ensemble size
    pub fn get_ensemble_size(&self) -> usize {
        self.trees.len()
    }

    /// Get feature importance scores
    pub fn get_feature_importances(&self) -> &HashMap<usize, f64> {
        &self.feature_importances
    }

    /// Get tree statistics
    pub fn get_tree_stats(&self) -> &[RandomForestTreeStats] {
        &self.tree_stats
    }
}

impl StreamingTreeModel for IncrementalRandomForest {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if self.trees.is_empty() {
            return Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let n_samples = x.nrows();
        let mut ensemble_predictions = Array1::zeros(n_samples);

        // Collect predictions from all trees
        for tree in &self.trees {
            let tree_predictions = tree.predict(x)?;
            ensemble_predictions += &tree_predictions;
        }

        // Average predictions
        ensemble_predictions /= self.trees.len() as f64;

        Ok(ensemble_predictions)
    }

    fn update(&mut self, x: &Array2<f64>, y: &Array1<f64>, weights: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize trees if this is the first call
        if self.trees.is_empty() {
            self.initialize_trees(n_features)?;
        }

        // Update each tree with different bootstrap samples and feature subsets
        for tree_idx in 0..self.trees.len() {
            let bootstrap_indices = self.create_bootstrap_sample(n_samples);
            let feature_subset = self.get_feature_subset(n_features);

            self.update_tree(tree_idx, x, y, weights, &bootstrap_indices, &feature_subset)?;
        }

        // Update out-of-bag buffer for evaluation
        if self.config.oob_score {
            for i in 0..n_samples {
                let x_sample: Vec<f64> = x.row(i).to_vec();
                self.oob_buffer
                    .add_sample(x_sample, y[i], weights[i], self.sample_counter as u64);
                self.sample_counter += 1;
            }
        }

        // Update feature importances
        self.update_feature_importances();

        // Detect concept drift
        let current_accuracy = self.get_accuracy(x, y)?;
        let drift_detected = self.drift_detector.update(current_accuracy);

        if drift_detected {
            // Handle concept drift by refreshing some trees
            let n_trees_to_refresh = (self.trees.len() / 4).max(1);
            for _ in 0..n_trees_to_refresh {
                let tree_idx = self.rng.gen_range(0..self.trees.len());
                self.trees[tree_idx] = HoeffdingTree::new(
                    self.config.base_tree_config.clone(),
                    self.n_features.unwrap_or(1),
                );
                self.tree_stats[tree_idx] = RandomForestTreeStats::new(tree_idx);
            }
        }

        // Adaptive ensemble management
        self.prune_poor_trees()?;
        self.add_new_trees()?;

        Ok(())
    }

    fn get_accuracy(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Calculate accuracy (for classification) or R² (for regression)
        let mean_y = y.mean().unwrap_or(0.0);
        // Calculate residual sum of squares using SIMD acceleration
        let pred_slice: Vec<f64> = predictions.iter().cloned().collect();
        let target_slice: Vec<f64> = y.iter().cloned().collect();
        let ss_res =
            simd_tree::simd_mse_evaluation(&pred_slice, &target_slice) * predictions.len() as f64;
        // Calculate total sum of squares using SIMD acceleration
        let y_slice: Vec<f64> = y.iter().cloned().collect();
        let mean_vec: Vec<f64> = vec![mean_y; y_slice.len()];
        let ss_tot = simd_tree::simd_mse_evaluation(&y_slice, &mean_vec) * y_slice.len() as f64;

        if ss_tot == 0.0 {
            Ok(1.0)
        } else {
            Ok((1.0 - ss_res / ss_tot).max(0.0))
        }
    }

    fn rebuild(&mut self, x: &Array2<f64>, y: &Array1<f64>, weights: &Array1<f64>) -> Result<()> {
        // Clear existing trees
        self.trees.clear();
        self.tree_stats.clear();
        self.feature_importances.clear();

        // Re-initialize and train
        let n_features = x.ncols();
        self.initialize_trees(n_features)?;
        self.update(x, y, weights)
    }
}

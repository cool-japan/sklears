//! Isolation Forest for Anomaly Detection
//!
//! Implementation of Isolation Forest and Extended Isolation Forest algorithms
//! for unsupervised anomaly detection.
//!
//! # References
//!
//! - Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
//!   In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422).
//! - Hariri, S., Kind, M. C., & Brunner, R. J. (2019). Extended Isolation Forest.
//!   IEEE Transactions on Knowledge and Data Engineering.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{thread_rng, CoreRandom};
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict, Trained, Untrained};
use sklears_core::types::Float;
use std::marker::PhantomData;

/// Node in an Isolation Tree
#[derive(Debug, Clone)]
pub struct IsolationNode {
    /// Feature index for splitting (None for leaf nodes)
    pub feature: Option<usize>,
    /// Split threshold value
    pub threshold: Float,
    /// Left child node (samples < threshold)
    pub left: Option<Box<IsolationNode>>,
    /// Right child node (samples >= threshold)
    pub right: Option<Box<IsolationNode>>,
    /// Size of the node (number of samples)
    pub size: usize,
}

impl IsolationNode {
    /// Create a new leaf node
    pub fn new_leaf(size: usize) -> Self {
        Self {
            feature: None,
            threshold: 0.0,
            left: None,
            right: None,
            size,
        }
    }

    /// Create a new internal node
    pub fn new_internal(
        feature: usize,
        threshold: Float,
        left: Self,
        right: Self,
        size: usize,
    ) -> Self {
        Self {
            feature: Some(feature),
            threshold,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            size,
        }
    }

    /// Compute path length for a sample
    pub fn path_length(&self, sample: &ArrayView1<Float>, current_depth: usize) -> Float {
        if self.left.is_none() || self.right.is_none() {
            // Leaf node - add average path length adjustment
            current_depth as Float + self.avg_path_length_adjustment()
        } else if let Some(feature_idx) = self.feature {
            let value = sample[feature_idx];
            if value < self.threshold {
                if let Some(left) = self.left.as_ref() {
                    left.path_length(sample, current_depth + 1)
                } else {
                    current_depth as Float
                }
            } else {
                if let Some(right) = self.right.as_ref() {
                    right.path_length(sample, current_depth + 1)
                } else {
                    current_depth as Float
                }
            }
        } else {
            current_depth as Float
        }
    }

    /// Average path length adjustment for unsuccessful search in BST
    fn avg_path_length_adjustment(&self) -> Float {
        if self.size <= 1 {
            0.0
        } else {
            2.0 * (((self.size - 1) as Float).ln() + 0.5772156649)
                - 2.0 * (self.size - 1) as Float / self.size as Float
        }
    }
}

/// Extended Isolation Forest node using hyperplanes
#[derive(Debug, Clone)]
pub struct ExtendedIsolationNode {
    /// Normal vector for the hyperplane split
    pub normal: Option<Array1<Float>>,
    /// Intercept for the hyperplane
    pub intercept: Float,
    /// Left child node
    pub left: Option<Box<ExtendedIsolationNode>>,
    /// Right child node
    pub right: Option<Box<ExtendedIsolationNode>>,
    /// Size of the node
    pub size: usize,
}

impl ExtendedIsolationNode {
    /// Create a new leaf node
    pub fn new_leaf(size: usize) -> Self {
        Self {
            normal: None,
            intercept: 0.0,
            left: None,
            right: None,
            size,
        }
    }

    /// Create a new internal node with hyperplane split
    pub fn new_internal(
        normal: Array1<Float>,
        intercept: Float,
        left: Self,
        right: Self,
        size: usize,
    ) -> Self {
        Self {
            normal: Some(normal),
            intercept,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            size,
        }
    }

    /// Compute path length for a sample
    pub fn path_length(&self, sample: &ArrayView1<Float>, current_depth: usize) -> Float {
        if self.left.is_none() || self.right.is_none() {
            current_depth as Float + self.avg_path_length_adjustment()
        } else if let Some(ref normal) = self.normal {
            let projection = sample.dot(normal) - self.intercept;
            if projection < 0.0 {
                if let Some(left) = self.left.as_ref() {
                    left.path_length(sample, current_depth + 1)
                } else {
                    current_depth as Float
                }
            } else {
                if let Some(right) = self.right.as_ref() {
                    right.path_length(sample, current_depth + 1)
                } else {
                    current_depth as Float
                }
            }
        } else {
            current_depth as Float
        }
    }

    /// Average path length adjustment
    fn avg_path_length_adjustment(&self) -> Float {
        if self.size <= 1 {
            0.0
        } else {
            2.0 * (((self.size - 1) as Float).ln() + 0.5772156649)
                - 2.0 * (self.size - 1) as Float / self.size as Float
        }
    }
}

/// Configuration for Isolation Forest
#[derive(Debug, Clone)]
pub struct IsolationForestConfig {
    /// Number of trees in the ensemble
    pub n_estimators: usize,
    /// Maximum tree depth (None for unlimited)
    pub max_depth: Option<usize>,
    /// Number of samples to draw for each tree
    pub max_samples: MaxSamples,
    /// Contamination (expected proportion of outliers)
    pub contamination: Float,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Use Extended Isolation Forest (hyperplane splits)
    pub extended: bool,
    /// Extension level for Extended IF (number of dimensions)
    pub extension_level: Option<usize>,
}

impl Default for IsolationForestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            max_samples: MaxSamples::Auto,
            contamination: 0.1,
            random_state: None,
            extended: false,
            extension_level: None,
        }
    }
}

/// Maximum samples strategy
#[derive(Debug, Clone)]
pub enum MaxSamples {
    /// Automatic (min(256, n_samples))
    Auto,
    /// Fixed number of samples
    Number(usize),
    /// Fraction of total samples
    Fraction(Float),
}

/// Isolation Forest for anomaly detection
pub struct IsolationForest<State = Untrained> {
    config: IsolationForestConfig,
    state: PhantomData<State>,
    trees: Vec<IsolationNode>,
    extended_trees: Vec<ExtendedIsolationNode>,
    n_features: Option<usize>,
    threshold: Option<Float>,
    offset: Option<Float>,
}

impl IsolationForest<Untrained> {
    /// Create a new Isolation Forest
    pub fn new() -> Self {
        Self {
            config: IsolationForestConfig::default(),
            state: PhantomData,
            trees: Vec::new(),
            extended_trees: Vec::new(),
            n_features: None,
            threshold: None,
            offset: None,
        }
    }

    /// Set the number of trees
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the maximum tree depth
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = Some(max_depth);
        self
    }

    /// Set the number of samples per tree
    pub fn max_samples(mut self, max_samples: MaxSamples) -> Self {
        self.config.max_samples = max_samples;
        self
    }

    /// Set the contamination level
    pub fn contamination(mut self, contamination: Float) -> Self {
        self.config.contamination = contamination;
        self
    }

    /// Set the random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Use Extended Isolation Forest
    pub fn extended(mut self, extended: bool) -> Self {
        self.config.extended = extended;
        self
    }

    /// Set extension level for Extended IF
    pub fn extension_level(mut self, level: usize) -> Self {
        self.config.extension_level = Some(level);
        self
    }
}

impl Default for IsolationForest<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for IsolationForest<Untrained> {
    type Config = IsolationForestConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for IsolationForest<Untrained> {
    type Fitted = IsolationForest<Trained>;

    fn fit(mut self, x: &Array2<Float>, _y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        // Determine sample size for each tree
        let max_samples = match self.config.max_samples {
            MaxSamples::Auto => n_samples.min(256),
            MaxSamples::Number(n) => n.min(n_samples),
            MaxSamples::Fraction(f) => ((n_samples as Float * f) as usize).min(n_samples),
        };

        // Determine maximum depth
        let max_depth = self
            .config
            .max_depth
            .unwrap_or_else(|| ((max_samples as Float).log2().ceil() as usize).min(100));

        // TODO: Proper seeded RNG support when scirs2 provides compatible API
        let mut rng = thread_rng();
        let _ = self.config.random_state; // Acknowledge parameter for future use

        // Build trees
        if self.config.extended {
            // Build Extended Isolation Trees
            let extension_level = self.config.extension_level.unwrap_or(n_features);
            for _ in 0..self.config.n_estimators {
                let indices = sample_indices(n_samples, max_samples, &mut rng)?;
                let tree =
                    build_extended_tree(x, &indices, 0, max_depth, extension_level, &mut rng)?;
                self.extended_trees.push(tree);
            }
        } else {
            // Build standard Isolation Trees
            for _ in 0..self.config.n_estimators {
                let indices = sample_indices(n_samples, max_samples, &mut rng)?;
                let tree = build_isolation_tree(x, &indices, 0, max_depth, &mut rng)?;
                self.trees.push(tree);
            }
        }

        // Compute threshold based on contamination
        let avg_path_length = average_path_length(max_samples);
        let offset = -0.5;
        let threshold = 2.0_f64.powf(-self.config.contamination / avg_path_length);

        Ok(IsolationForest::<Trained> {
            config: self.config,
            state: PhantomData,
            trees: self.trees,
            extended_trees: self.extended_trees,
            n_features: Some(n_features),
            threshold: Some(threshold as Float),
            offset: Some(offset as Float),
        })
    }
}

impl IsolationForest<Trained> {
    /// Compute anomaly scores for samples
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = x.row(i);
            let avg_path = if self.config.extended {
                self.average_path_length_extended(&sample)
            } else {
                self.average_path_length_standard(&sample)
            };

            let max_samples = match self.config.max_samples {
                MaxSamples::Auto => 256,
                MaxSamples::Number(n) => n,
                MaxSamples::Fraction(_) => 256, // Approximate
            };

            let c = average_path_length(max_samples);
            scores[i] = 2.0_f64.powf(-avg_path / c) as Float;
        }

        Ok(scores)
    }

    /// Average path length for a sample (standard IF)
    fn average_path_length_standard(&self, sample: &ArrayView1<Float>) -> Float {
        if self.trees.is_empty() {
            return 0.0;
        }

        let sum: Float = self
            .trees
            .iter()
            .map(|tree| tree.path_length(sample, 0))
            .sum();

        sum / self.trees.len() as Float
    }

    /// Average path length for a sample (extended IF)
    fn average_path_length_extended(&self, sample: &ArrayView1<Float>) -> Float {
        if self.extended_trees.is_empty() {
            return 0.0;
        }

        let sum: Float = self
            .extended_trees
            .iter()
            .map(|tree| tree.path_length(sample, 0))
            .sum();

        sum / self.extended_trees.len() as Float
    }

    /// Get the anomaly threshold
    pub fn threshold(&self) -> Float {
        self.threshold.unwrap_or(0.5)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for IsolationForest<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let scores = self.decision_function(x)?;
        let threshold = self.threshold();

        let predictions = scores.mapv(|score| {
            if score >= threshold {
                -1 // Outlier
            } else {
                1 // Inlier
            }
        });

        Ok(predictions)
    }
}

/// Sample random indices without replacement
fn sample_indices(
    n_samples: usize,
    n_to_sample: usize,
    rng: &mut CoreRandom,
) -> Result<Vec<usize>> {
    use scirs2_core::random::essentials::Uniform;

    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Fisher-Yates shuffle for first n_to_sample elements
    for i in 0..n_to_sample.min(n_samples) {
        let uniform = Uniform::new(i, n_samples).map_err(|_| {
            SklearsError::InvalidInput(format!(
                "Failed to create uniform distribution for range [{}, {})",
                i, n_samples
            ))
        })?;
        let j = rng.sample(uniform);
        indices.swap(i, j);
    }

    indices.truncate(n_to_sample);
    Ok(indices)
}

/// Build a standard Isolation Tree
fn build_isolation_tree(
    x: &Array2<Float>,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    rng: &mut CoreRandom,
) -> Result<IsolationNode> {
    use scirs2_core::random::essentials::Uniform;

    let n_samples = indices.len();
    let n_features = x.ncols();

    // Base cases
    if n_samples <= 1 || depth >= max_depth {
        return Ok(IsolationNode::new_leaf(n_samples));
    }

    // Check if all samples are identical
    let first_sample = x.row(indices[0]);
    let all_identical = indices.iter().skip(1).all(|&idx| {
        let sample = x.row(idx);
        sample
            .iter()
            .zip(first_sample.iter())
            .all(|(a, b)| (a - b).abs() < 1e-10)
    });

    if all_identical {
        return Ok(IsolationNode::new_leaf(n_samples));
    }

    // Randomly select a feature
    let feature_dist = Uniform::new(0, n_features).map_err(|_| {
        SklearsError::InvalidInput("Failed to create uniform distribution".to_string())
    })?;
    let feature = rng.sample(feature_dist);

    // Find min and max values for the selected feature
    let mut min_val = Float::INFINITY;
    let mut max_val = Float::NEG_INFINITY;

    for &idx in indices {
        let val = x[[idx, feature]];
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }

    if (max_val - min_val).abs() < 1e-10 {
        return Ok(IsolationNode::new_leaf(n_samples));
    }

    // Randomly select a split point
    let split_dist = Uniform::new(min_val, max_val).map_err(|_| {
        SklearsError::InvalidInput("Failed to create split distribution".to_string())
    })?;
    let threshold = rng.sample(split_dist);

    // Partition samples
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &idx in indices {
        if x[[idx, feature]] < threshold {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    // Handle edge case where all samples go to one side
    if left_indices.is_empty() || right_indices.is_empty() {
        return Ok(IsolationNode::new_leaf(n_samples));
    }

    // Recursively build subtrees
    let left_tree = build_isolation_tree(x, &left_indices, depth + 1, max_depth, rng)?;
    let right_tree = build_isolation_tree(x, &right_indices, depth + 1, max_depth, rng)?;

    Ok(IsolationNode::new_internal(
        feature, threshold, left_tree, right_tree, n_samples,
    ))
}

/// Build an Extended Isolation Tree using hyperplane splits
fn build_extended_tree(
    x: &Array2<Float>,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    extension_level: usize,
    rng: &mut CoreRandom,
) -> Result<ExtendedIsolationNode> {
    use scirs2_core::random::essentials::Uniform;

    let n_samples = indices.len();
    let n_features = x.ncols();

    // Base cases
    if n_samples <= 1 || depth >= max_depth {
        return Ok(ExtendedIsolationNode::new_leaf(n_samples));
    }

    // Sample a random slope (normal vector)
    let n_dims = extension_level.min(n_features);
    let mut normal = Array1::zeros(n_features);

    for i in 0..n_dims {
        let normal_dist = Uniform::new(-1.0, 1.0).map_err(|_| {
            SklearsError::InvalidInput("Failed to create normal distribution".to_string())
        })?;
        normal[i] = rng.sample(normal_dist);
    }

    // Normalize the normal vector
    let dot_product: Float = normal.dot(&normal);
    let norm = dot_product.sqrt();
    if norm > 1e-10 {
        normal /= norm;
    } else {
        return Ok(ExtendedIsolationNode::new_leaf(n_samples));
    }

    // Compute projections
    let mut projections = Vec::with_capacity(n_samples);
    for &idx in indices {
        let sample = x.row(idx);
        let projection = sample.dot(&normal);
        projections.push(projection);
    }

    // Find min and max projections
    let min_proj = projections
        .iter()
        .cloned()
        .fold(Float::INFINITY, Float::min);
    let max_proj = projections
        .iter()
        .cloned()
        .fold(Float::NEG_INFINITY, Float::max);

    if (max_proj - min_proj).abs() < 1e-10 {
        return Ok(ExtendedIsolationNode::new_leaf(n_samples));
    }

    // Random intercept
    let intercept_dist = Uniform::new(min_proj, max_proj).map_err(|_| {
        SklearsError::InvalidInput("Failed to create intercept distribution".to_string())
    })?;
    let intercept = rng.sample(intercept_dist);

    // Partition samples
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for (&idx, &proj) in indices.iter().zip(projections.iter()) {
        if proj < intercept {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    if left_indices.is_empty() || right_indices.is_empty() {
        return Ok(ExtendedIsolationNode::new_leaf(n_samples));
    }

    // Recursively build subtrees
    let left_tree =
        build_extended_tree(x, &left_indices, depth + 1, max_depth, extension_level, rng)?;
    let right_tree = build_extended_tree(
        x,
        &right_indices,
        depth + 1,
        max_depth,
        extension_level,
        rng,
    )?;

    Ok(ExtendedIsolationNode::new_internal(
        normal, intercept, left_tree, right_tree, n_samples,
    ))
}

/// Average path length of unsuccessful search in BST
fn average_path_length(n: usize) -> Float {
    if n <= 1 {
        0.0
    } else {
        2.0 * (((n - 1) as Float).ln() + 0.5772156649) - 2.0 * (n - 1) as Float / n as Float
    }
}

/// Streaming Isolation Forest for online anomaly detection
pub struct StreamingIsolationForest {
    config: IsolationForestConfig,
    trees: Vec<IsolationNode>,
    window_size: usize,
    buffer: Vec<Array1<Float>>,
    update_frequency: usize,
    samples_seen: usize,
}

impl StreamingIsolationForest {
    /// Create a new streaming isolation forest
    pub fn new(config: IsolationForestConfig, window_size: usize, update_frequency: usize) -> Self {
        Self {
            config,
            trees: Vec::new(),
            window_size,
            buffer: Vec::new(),
            update_frequency,
            samples_seen: 0,
        }
    }

    /// Process a new sample and return its anomaly score
    pub fn process_sample(&mut self, sample: Array1<Float>) -> Result<Float> {
        self.samples_seen += 1;

        // Add to buffer
        self.buffer.push(sample.clone());
        if self.buffer.len() > self.window_size {
            self.buffer.remove(0);
        }

        // Rebuild trees periodically
        if self.samples_seen % self.update_frequency == 0 && self.buffer.len() >= 32 {
            self.rebuild_trees()?;
        }

        // Compute anomaly score if trees exist
        if self.trees.is_empty() {
            return Ok(0.5); // Neutral score
        }

        let sample_view = sample.view();
        let avg_path: Float = self
            .trees
            .iter()
            .map(|tree| tree.path_length(&sample_view, 0))
            .sum::<Float>()
            / self.trees.len() as Float;

        let max_samples = self.buffer.len().min(256);
        let c = average_path_length(max_samples);
        let score = 2.0_f64.powf(-avg_path as f64 / c) as Float;

        Ok(score)
    }

    /// Rebuild trees from current buffer
    fn rebuild_trees(&mut self) -> Result<()> {
        let n_samples = self.buffer.len();
        if n_samples < 2 {
            return Ok(());
        }

        // Convert buffer to Array2
        let n_features = self.buffer[0].len();
        let mut x = Array2::zeros((n_samples, n_features));
        for (i, sample) in self.buffer.iter().enumerate() {
            x.row_mut(i).assign(sample);
        }

        let max_samples = n_samples.min(256);
        let max_depth = ((max_samples as Float).log2().ceil() as usize).min(100);

        // TODO: Proper seeded RNG support when scirs2 provides compatible API
        let mut rng = thread_rng();
        let _ = self.config.random_state; // Acknowledge parameter for future use

        // Clear and rebuild trees
        self.trees.clear();
        for _ in 0..self.config.n_estimators {
            let indices = sample_indices(n_samples, max_samples, &mut rng)?;
            let tree = build_isolation_tree(&x, &indices, 0, max_depth, &mut rng)?;
            self.trees.push(tree);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::s;

    #[test]
    fn test_isolation_forest_basic() {
        let mut x = Array2::zeros((100, 2));

        // Normal samples clustered tightly around origin
        for i in 0..90 {
            let angle = (i as Float) * 2.0 * std::f64::consts::PI / 90.0;
            let radius = 0.5 + ((i % 10) as Float) * 0.05;
            x[[i, 0]] = radius * angle.cos();
            x[[i, 1]] = radius * angle.sin();
        }

        // Outliers very far from cluster
        for i in 90..100 {
            x[[i, 0]] = 20.0 + ((i - 90) as Float);
            x[[i, 1]] = 20.0 + ((i - 90) as Float);
        }

        let y = Array1::zeros(100); // Dummy labels for unsupervised

        let model = IsolationForest::new()
            .n_estimators(100)
            .contamination(0.1)
            .random_state(42);

        let fitted = model.fit(&x, &y).unwrap();
        let scores = fitted.decision_function(&x).unwrap();

        // Compare average scores (fairer comparison)
        let outlier_avg: Float = scores.slice(s![90..]).mean().unwrap();
        let inlier_avg: Float = scores.slice(s![..90]).mean().unwrap();

        assert!(
            outlier_avg > inlier_avg,
            "Outliers should have higher average anomaly scores: outlier_avg={}, inlier_avg={}",
            outlier_avg,
            inlier_avg
        );
    }

    #[test]
    fn test_extended_isolation_forest() {
        let mut x = Array2::zeros((50, 2));

        // Normal samples
        for i in 0..45 {
            x[[i, 0]] = (i as Float / 22.5) - 1.0;
            x[[i, 1]] = (i as Float / 22.5) - 1.0;
        }

        // Outliers
        for i in 45..50 {
            x[[i, 0]] = ((i - 45) as Float) * 5.0;
            x[[i, 1]] = ((i - 45) as Float) * 5.0;
        }

        let y = Array1::zeros(50);

        let model = IsolationForest::new()
            .n_estimators(30)
            .extended(true)
            .extension_level(2)
            .random_state(42);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Most normal samples should be classified as inliers (1)
        let inliers: i32 = predictions
            .slice(s![..45])
            .iter()
            .filter(|&&x| x == 1)
            .count() as i32;
        assert!(inliers > 40, "Most normal samples should be inliers");
    }

    #[test]
    fn test_streaming_isolation_forest() {
        let config = IsolationForestConfig {
            n_estimators: 20,
            ..Default::default()
        };

        let mut streaming_if = StreamingIsolationForest::new(config, 100, 50);

        // Process normal samples
        for i in 0..60 {
            let sample =
                Array1::from_vec(vec![(i as Float / 30.0) - 1.0, (i as Float / 30.0) - 1.0]);
            let _score = streaming_if.process_sample(sample).unwrap();
        }

        // Process an outlier
        let outlier = Array1::from_vec(vec![10.0, 10.0]);
        let outlier_score = streaming_if.process_sample(outlier).unwrap();

        // Process a normal sample
        let normal = Array1::from_vec(vec![0.1, 0.1]);
        let normal_score = streaming_if.process_sample(normal).unwrap();

        // Outlier should have higher score (after trees are built)
        if streaming_if.trees.len() > 0 {
            assert!(
                outlier_score > normal_score,
                "Outlier should have higher score than normal sample"
            );
        }
    }

    #[test]
    fn test_average_path_length() {
        let c_256 = average_path_length(256);
        // c(n) = 2*H(n-1) - 2*(n-1)/n where H is harmonic number
        assert_relative_eq!(c_256, 10.24, epsilon = 0.1);

        let c_100 = average_path_length(100);
        assert_relative_eq!(c_100, 8.36, epsilon = 0.1);

        // Verify monotonicity
        assert!(
            c_256 > c_100,
            "Average path length should increase with sample size"
        );

        // Verify c(n) > 0 for all n > 1
        assert!(average_path_length(10) > 0.0);
        assert!(average_path_length(2) > 0.0);
    }

    #[test]
    fn test_isolation_node_path_length() {
        // Create a simple tree manually
        let left_leaf = IsolationNode::new_leaf(5);
        let right_leaf = IsolationNode::new_leaf(5);
        let root = IsolationNode::new_internal(0, 0.5, left_leaf, right_leaf, 10);

        // Test path length computation
        let sample_left = Array1::from_vec(vec![0.0, 0.0]);
        let path_left = root.path_length(&sample_left.view(), 0);
        assert!(path_left > 0.0, "Path length should be positive");

        let sample_right = Array1::from_vec(vec![1.0, 1.0]);
        let path_right = root.path_length(&sample_right.view(), 0);
        assert!(path_right > 0.0, "Path length should be positive");
    }
}

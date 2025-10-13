//! Multi-label learning algorithms
//!
//! This module provides algorithms for multi-label classification where each instance
//! can belong to multiple classes simultaneously.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Binary Relevance
///
/// A multi-label classification strategy that treats each label as a separate
/// binary classification problem. For each label, it trains a binary classifier
/// to predict whether that label is relevant or not, independently of other labels.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::BinaryRelevance;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // This is a simple example showing the structure
/// let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let labels = array![[1, 0], [0, 1], [1, 1]]; // Multi-label: each column is a binary label
/// ```
#[derive(Debug, Clone)]
pub struct BinaryRelevance<S = Untrained> {
    /// state
    pub state: S,
    n_jobs: Option<i32>,
}

impl BinaryRelevance<Untrained> {
    /// Create a new BinaryRelevance instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_jobs: None,
        }
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for BinaryRelevance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BinaryRelevance<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for BinaryRelevance<Untrained> {
    type Fitted = BinaryRelevance<BinaryRelevanceTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_labels = y.ncols();
        if n_labels == 0 {
            return Err(SklearsError::InvalidInput(
                "y must have at least one label".to_string(),
            ));
        }

        let mut binary_classifiers = HashMap::new();
        let mut classes_per_label = Vec::new();

        // Train one binary classifier per label
        for label_idx in 0..n_labels {
            let y_label = y.column(label_idx);

            // Get unique classes for this label (should be binary: 0 and 1)
            let mut label_classes: Vec<i32> = y_label
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            label_classes.sort();

            // Validate that we have binary labels
            if label_classes.len() > 2 {
                return Err(SklearsError::InvalidInput(format!(
                    "Label {} has {} classes, but BinaryRelevance expects binary labels",
                    label_idx,
                    label_classes.len()
                )));
            }

            // Ensure we have at least one positive and one negative example for training
            let has_positive = label_classes.contains(&1);
            let has_negative = label_classes.contains(&0);

            if !has_positive && !has_negative {
                return Err(SklearsError::InvalidInput(format!(
                    "Label {} has no training examples",
                    label_idx
                )));
            }

            // Train binary classifier using logistic regression approach
            let weights = train_binary_classifier(&X, &y_label)?;
            binary_classifiers.insert(label_idx, weights);
            classes_per_label.push(label_classes);
        }

        Ok(BinaryRelevance {
            state: BinaryRelevanceTrained {
                binary_classifiers,
                classes_per_label,
                n_labels,
                n_features,
            },
            n_jobs: self.n_jobs,
        })
    }
}

/// Simple binary classifier training using logistic regression approximation
fn train_binary_classifier(
    X: &Array2<Float>,
    y: &scirs2_core::ndarray::ArrayView1<i32>,
) -> SklResult<(Array1<f64>, f64)> {
    let (n_samples, n_features) = X.dim();

    // Simple approach: use correlation-based weights similar to linear regression
    let mut weights = Array1::<Float>::zeros(n_features);
    let mut bias = 0.0;

    // Compute mean of labels (proportion of positive class)
    let y_mean: f64 = y.iter().map(|&label| label as f64).sum::<f64>() / n_samples as f64;

    // Use logit of the mean as initial bias
    bias = if y_mean > 0.0 && y_mean < 1.0 {
        (y_mean / (1.0 - y_mean)).ln()
    } else if y_mean >= 1.0 {
        2.0 // Large positive value
    } else {
        -2.0 // Large negative value
    };

    // Compute feature-label correlations
    for feature_idx in 0..n_features {
        let mut x_mean = 0.0;
        for sample_idx in 0..n_samples {
            x_mean += X[[sample_idx, feature_idx]];
        }
        x_mean /= n_samples as f64;

        // Compute correlation between feature and label
        let mut numerator: f64 = 0.0;
        let mut x_var: f64 = 0.0;
        let mut y_var: f64 = 0.0;

        for sample_idx in 0..n_samples {
            let x_diff = X[[sample_idx, feature_idx]] - x_mean;
            let y_diff = y[sample_idx] as f64 - y_mean;
            numerator += x_diff * y_diff;
            x_var += x_diff * x_diff;
            y_var += y_diff * y_diff;
        }

        if x_var > 1e-10 && y_var > 1e-10 {
            let correlation = numerator / (x_var.sqrt() * y_var.sqrt());
            weights[feature_idx] = correlation; // Use correlation as weight
        }
    }

    Ok((weights, bias))
}

impl BinaryRelevance<BinaryRelevanceTrained> {
    /// Get the classes for each label
    pub fn classes(&self) -> &[Vec<i32>] {
        &self.state.classes_per_label
    }

    /// Get the number of labels
    pub fn n_labels(&self) -> usize {
        self.state.n_labels
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.state.n_features
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>> for BinaryRelevance<BinaryRelevanceTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        // Get predictions from each binary classifier
        for label_idx in 0..self.state.n_labels {
            if let Some((weights, bias)) = self.state.binary_classifiers.get(&label_idx) {
                for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
                    // Compute logistic regression score
                    let score: f64 = sample
                        .iter()
                        .zip(weights.iter())
                        .map(|(&x, &w)| x * w)
                        .sum::<f64>()
                        + bias;

                    // Apply sigmoid and threshold at 0.5 for binary classification
                    let prob = 1.0 / (1.0 + (-score).exp());
                    let prediction = if prob > 0.5 { 1 } else { 0 };

                    predictions[[sample_idx, label_idx]] = prediction;
                }
            }
        }

        Ok(predictions)
    }
}

/// Predict probabilities for each label
impl BinaryRelevance<BinaryRelevanceTrained> {
    /// Predict class probabilities for each label
    #[allow(non_snake_case)]
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let mut probabilities = Array2::<Float>::zeros((n_samples, self.state.n_labels));

        // Get probabilities from each binary classifier
        for label_idx in 0..self.state.n_labels {
            if let Some((weights, bias)) = self.state.binary_classifiers.get(&label_idx) {
                for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
                    // Compute logistic regression score
                    let score: f64 = sample
                        .iter()
                        .zip(weights.iter())
                        .map(|(&x, &w)| x * w)
                        .sum::<f64>()
                        + bias;

                    // Apply sigmoid to get probability
                    let prob = 1.0 / (1.0 + (-score).exp());
                    probabilities[[sample_idx, label_idx]] = prob;
                }
            }
        }

        Ok(probabilities)
    }
}

/// Trained state for BinaryRelevance
#[derive(Debug, Clone)]
pub struct BinaryRelevanceTrained {
    /// Binary classifiers for each label (weights, bias)
    pub binary_classifiers: HashMap<usize, (Array1<f64>, f64)>,
    /// Classes for each label
    pub classes_per_label: Vec<Vec<i32>>,
    /// Number of labels
    pub n_labels: usize,
    /// Number of features
    pub n_features: usize,
}

/// Label Powerset
///
/// A multi-label classification strategy that transforms the multi-label problem
/// into a multi-class problem by treating each unique combination of labels as
/// a single class. Each label combination becomes a distinct class in the transformed
/// problem space.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::LabelPowerset;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // This is a simple example showing the structure
/// let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let labels = array![[1, 0], [0, 1], [1, 1]]; // Multi-label: each row is a label combination
/// ```
#[derive(Debug, Clone)]
pub struct LabelPowerset<S = Untrained> {
    state: S,
}

impl LabelPowerset<Untrained> {
    /// Create a new LabelPowerset instance
    pub fn new() -> Self {
        Self { state: Untrained }
    }
}

impl Default for LabelPowerset<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LabelPowerset<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for LabelPowerset<Untrained> {
    type Fitted = LabelPowerset<LabelPowersetTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_labels = y.ncols();
        if n_labels == 0 {
            return Err(SklearsError::InvalidInput(
                "y must have at least one label".to_string(),
            ));
        }

        // Validate that labels are binary
        for sample_idx in 0..n_samples {
            for label_idx in 0..n_labels {
                let label_value = y[[sample_idx, label_idx]];
                if label_value != 0 && label_value != 1 {
                    return Err(SklearsError::InvalidInput(format!(
                        "LabelPowerset expects binary labels, but found {} at position ({}, {})",
                        label_value, sample_idx, label_idx
                    )));
                }
            }
        }

        // Transform multi-label to multi-class by creating unique combinations
        let mut class_to_combination: HashMap<usize, Vec<i32>> = HashMap::new();
        let mut combination_to_class: HashMap<Vec<i32>, usize> = HashMap::new();
        let mut transformed_labels = Vec::new();
        let mut next_class_id = 0;

        for sample_idx in 0..n_samples {
            // Extract the label combination for this sample
            let combination: Vec<i32> = (0..n_labels)
                .map(|label_idx| y[[sample_idx, label_idx]])
                .collect();

            // Check if we've seen this combination before
            let class_id = if let Some(&existing_class_id) = combination_to_class.get(&combination)
            {
                existing_class_id
            } else {
                // New combination, assign a new class ID
                let class_id = next_class_id;
                combination_to_class.insert(combination.clone(), class_id);
                class_to_combination.insert(class_id, combination);
                next_class_id += 1;
                class_id
            };

            transformed_labels.push(class_id);
        }

        // Train a single multi-class classifier on the transformed problem
        // We'll use a simple nearest centroid approach
        let mut class_centroids: HashMap<usize, Array1<f64>> = HashMap::new();

        for &class_id in class_to_combination.keys() {
            let mut centroid = Array1::<Float>::zeros(n_features);
            let mut count = 0;

            // Compute centroid for this class
            for (sample_idx, &sample_class) in transformed_labels.iter().enumerate() {
                if sample_class == class_id {
                    for feature_idx in 0..n_features {
                        centroid[feature_idx] += X[[sample_idx, feature_idx]];
                    }
                    count += 1;
                }
            }

            if count > 0 {
                centroid /= count as f64;
            }
            class_centroids.insert(class_id, centroid);
        }

        let unique_classes: Vec<usize> = class_to_combination.keys().cloned().collect();

        Ok(LabelPowerset {
            state: LabelPowersetTrained {
                class_to_combination,
                combination_to_class,
                class_centroids,
                unique_classes,
                n_labels,
                n_features,
            },
        })
    }
}

impl LabelPowerset<LabelPowersetTrained> {
    /// Get the unique label combinations
    pub fn classes(&self) -> &HashMap<usize, Vec<i32>> {
        &self.state.class_to_combination
    }

    /// Get the number of unique label combinations
    pub fn n_classes(&self) -> usize {
        self.state.unique_classes.len()
    }

    /// Get the number of labels
    pub fn n_labels(&self) -> usize {
        self.state.n_labels
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>> for LabelPowerset<LabelPowersetTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        // For each sample, find the nearest class centroid
        for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut best_class_id = 0;

            // Find the closest class centroid
            for (&class_id, centroid) in &self.state.class_centroids {
                let mut distance = 0.0;
                for feature_idx in 0..n_features {
                    let diff = sample[feature_idx] - centroid[feature_idx];
                    distance += diff * diff;
                }
                distance = distance.sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    best_class_id = class_id;
                }
            }

            // Convert the predicted class back to label combination
            if let Some(label_combination) = self.state.class_to_combination.get(&best_class_id) {
                for label_idx in 0..self.state.n_labels {
                    predictions[[sample_idx, label_idx]] = label_combination[label_idx];
                }
            }
        }

        Ok(predictions)
    }
}

/// Get decision scores for each class
impl LabelPowerset<LabelPowersetTrained> {
    /// Predict decision scores (negative distances to centroids)
    #[allow(non_snake_case)]
    pub fn decision_function(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.state.unique_classes.len();
        let mut scores = Array2::<Float>::zeros((n_samples, n_classes));

        // For each sample, compute distances to all class centroids
        for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
            for (class_idx, &class_id) in self.state.unique_classes.iter().enumerate() {
                if let Some(centroid) = self.state.class_centroids.get(&class_id) {
                    let mut distance = 0.0;
                    for feature_idx in 0..n_features {
                        let diff = sample[feature_idx] - centroid[feature_idx];
                        distance += diff * diff;
                    }
                    distance = distance.sqrt();

                    // Use negative distance as score (higher score = closer)
                    scores[[sample_idx, class_idx]] = -distance;
                }
            }
        }

        Ok(scores)
    }
}

/// Trained state for LabelPowerset
#[derive(Debug, Clone)]
pub struct LabelPowersetTrained {
    /// Mapping from class ID to label combination
    pub class_to_combination: HashMap<usize, Vec<i32>>,
    /// Mapping from label combination to class ID
    pub combination_to_class: HashMap<Vec<i32>, usize>,
    /// Centroids for each class (nearest centroid classifier)
    pub class_centroids: HashMap<usize, Array1<f64>>,
    /// List of unique class IDs
    pub unique_classes: Vec<usize>,
    /// Number of labels
    pub n_labels: usize,
    /// Number of features
    pub n_features: usize,
}

/// Pruned Label Powerset
///
/// An extension of the Label Powerset method that prunes rare label combinations
/// to reduce the complexity of the multi-class problem. Label combinations that
/// appear less than `min_frequency` times in the training data are either mapped
/// to the most similar frequent combination or to a default combination.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::PrunedLabelPowerset;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // This is a simple example showing the structure
/// let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let labels = array![[0, 1], [1, 0], [1, 1]];
/// ```
#[derive(Debug, Clone)]
pub struct PrunedLabelPowerset<S = Untrained> {
    state: S,
    min_frequency: usize,
    strategy: PruningStrategy,
}

/// Strategy for handling pruned label combinations
#[derive(Debug, Clone)]
pub enum PruningStrategy {
    /// Map rare combinations to the most similar frequent combination
    SimilarityMapping,
    /// Map rare combinations to a default combination (typically all zeros)
    DefaultMapping(Vec<i32>),
}

impl PrunedLabelPowerset<Untrained> {
    /// Create a new PrunedLabelPowerset instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            min_frequency: 2,
            strategy: PruningStrategy::DefaultMapping(vec![]),
        }
    }

    /// Set the minimum frequency threshold for label combinations
    pub fn min_frequency(mut self, min_frequency: usize) -> Self {
        self.min_frequency = min_frequency;
        self
    }

    /// Set the pruning strategy
    pub fn strategy(mut self, strategy: PruningStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Get the minimum frequency threshold
    pub fn get_min_frequency(&self) -> usize {
        self.min_frequency
    }

    /// Get the pruning strategy
    pub fn get_strategy(&self) -> &PruningStrategy {
        &self.strategy
    }
}

impl Default for PrunedLabelPowerset<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for PrunedLabelPowerset<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for PrunedLabelPowerset<Untrained> {
    type Fitted = PrunedLabelPowerset<PrunedLabelPowersetTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_labels = y.ncols();
        if n_labels == 0 {
            return Err(SklearsError::InvalidInput(
                "y must have at least one label".to_string(),
            ));
        }

        // Validate that labels are binary
        for sample_idx in 0..n_samples {
            for label_idx in 0..n_labels {
                let label_value = y[[sample_idx, label_idx]];
                if label_value != 0 && label_value != 1 {
                    return Err(SklearsError::InvalidInput(format!(
                        "PrunedLabelPowerset expects binary labels, but found {} at position ({}, {})",
                        label_value, sample_idx, label_idx
                    )));
                }
            }
        }

        // Count frequencies of label combinations
        let mut combination_counts: HashMap<Vec<i32>, usize> = HashMap::new();
        for sample_idx in 0..n_samples {
            let combination: Vec<i32> = (0..n_labels)
                .map(|label_idx| y[[sample_idx, label_idx]])
                .collect();
            *combination_counts.entry(combination).or_insert(0) += 1;
        }

        // Determine which combinations to keep (frequent ones)
        let frequent_combinations: Vec<Vec<i32>> = combination_counts
            .iter()
            .filter(|(_, &count)| count >= self.min_frequency)
            .map(|(combination, _)| combination.clone())
            .collect();

        if frequent_combinations.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No label combinations meet the minimum frequency threshold".to_string(),
            ));
        }

        // Handle default strategy setup
        let default_combination = match &self.strategy {
            PruningStrategy::DefaultMapping(ref default) => {
                if default.is_empty() {
                    vec![0; n_labels] // Default to all zeros if not specified
                } else if default.len() != n_labels {
                    return Err(SklearsError::InvalidInput(
                        "Default combination length must match number of labels".to_string(),
                    ));
                } else {
                    default.clone()
                }
            }
            PruningStrategy::SimilarityMapping => vec![], // Not used for similarity mapping
        };

        // For default mapping strategy, ensure the default combination is included
        // in frequent combinations if it's not already there
        let mut final_frequent_combinations = frequent_combinations.clone();
        if let PruningStrategy::DefaultMapping(_) = &self.strategy {
            if !final_frequent_combinations.contains(&default_combination) {
                final_frequent_combinations.push(default_combination.clone());
            }
        }

        // Create mapping for rare combinations
        let mut combination_mapping: HashMap<Vec<i32>, Vec<i32>> = HashMap::new();

        for (combination, &count) in &combination_counts {
            if count >= self.min_frequency {
                // Keep frequent combinations as-is
                combination_mapping.insert(combination.clone(), combination.clone());
            } else {
                // Map rare combinations based on strategy
                let mapped_combination = match &self.strategy {
                    PruningStrategy::SimilarityMapping => {
                        // Find the most similar frequent combination
                        let mut best_similarity = -1.0;
                        let mut best_combination = &final_frequent_combinations[0];

                        for freq_combo in &final_frequent_combinations {
                            // Compute Jaccard similarity
                            let intersection: i32 = combination
                                .iter()
                                .zip(freq_combo.iter())
                                .map(|(&a, &b)| if a == 1 && b == 1 { 1 } else { 0 })
                                .sum();
                            let union: i32 = combination
                                .iter()
                                .zip(freq_combo.iter())
                                .map(|(&a, &b)| if a == 1 || b == 1 { 1 } else { 0 })
                                .sum();

                            let similarity = if union > 0 {
                                intersection as f64 / union as f64
                            } else {
                                1.0 // Both empty sets
                            };

                            if similarity > best_similarity {
                                best_similarity = similarity;
                                best_combination = freq_combo;
                            }
                        }
                        best_combination.clone()
                    }
                    PruningStrategy::DefaultMapping(_) => default_combination.clone(),
                };
                combination_mapping.insert(combination.clone(), mapped_combination);
            }
        }

        // Create the final class mapping using only frequent combinations
        let mut class_to_combination: HashMap<usize, Vec<i32>> = HashMap::new();
        let mut combination_to_class: HashMap<Vec<i32>, usize> = HashMap::new();
        let mut next_class_id = 0;

        for combo in &final_frequent_combinations {
            class_to_combination.insert(next_class_id, combo.clone());
            combination_to_class.insert(combo.clone(), next_class_id);
            next_class_id += 1;
        }

        // Transform labels using the mapping
        let mut transformed_labels = Vec::new();
        for sample_idx in 0..n_samples {
            let original_combination: Vec<i32> = (0..n_labels)
                .map(|label_idx| y[[sample_idx, label_idx]])
                .collect();

            let mapped_combination = combination_mapping
                .get(&original_combination)
                .unwrap()
                .clone();

            let class_id = *combination_to_class.get(&mapped_combination).unwrap();
            transformed_labels.push(class_id);
        }

        // Train a nearest centroid classifier on the pruned problem
        let mut class_centroids: HashMap<usize, Array1<f64>> = HashMap::new();

        for &class_id in class_to_combination.keys() {
            let mut centroid = Array1::<Float>::zeros(n_features);
            let mut count = 0;

            for (sample_idx, &sample_class) in transformed_labels.iter().enumerate() {
                if sample_class == class_id {
                    for feature_idx in 0..n_features {
                        centroid[feature_idx] += X[[sample_idx, feature_idx]];
                    }
                    count += 1;
                }
            }

            if count > 0 {
                centroid /= count as f64;
            }
            class_centroids.insert(class_id, centroid);
        }

        let unique_classes: Vec<usize> = class_to_combination.keys().cloned().collect();

        Ok(PrunedLabelPowerset {
            state: PrunedLabelPowersetTrained {
                class_to_combination,
                combination_to_class,
                combination_mapping,
                class_centroids,
                unique_classes,
                frequent_combinations: final_frequent_combinations,
                n_labels,
                n_features,
                min_frequency: self.min_frequency,
                strategy: self.strategy.clone(),
            },
            min_frequency: self.min_frequency,
            strategy: self.strategy.clone(),
        })
    }
}

impl PrunedLabelPowerset<PrunedLabelPowersetTrained> {
    /// Get the frequent label combinations
    pub fn frequent_combinations(&self) -> &[Vec<i32>] {
        &self.state.frequent_combinations
    }

    /// Get the number of frequent combinations
    pub fn n_frequent_classes(&self) -> usize {
        self.state.unique_classes.len()
    }

    /// Get the combination mapping used for pruning
    pub fn combination_mapping(&self) -> &HashMap<Vec<i32>, Vec<i32>> {
        &self.state.combination_mapping
    }

    /// Get the minimum frequency threshold used
    pub fn min_frequency(&self) -> usize {
        self.state.min_frequency
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.state.n_features
    }

    /// Get the number of labels
    pub fn n_labels(&self) -> usize {
        self.state.n_labels
    }

    /// Get the class centroids
    pub fn class_centroids(&self) -> &HashMap<usize, Array1<f64>> {
        &self.state.class_centroids
    }

    /// Get the class to combination mapping
    pub fn class_to_combination(&self) -> &HashMap<usize, Vec<i32>> {
        &self.state.class_to_combination
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for PrunedLabelPowerset<PrunedLabelPowersetTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        // For each sample, find the nearest class centroid
        for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut best_class_id = 0;

            // Find the closest class centroid among frequent combinations
            for (&class_id, centroid) in &self.state.class_centroids {
                let mut distance = 0.0;
                for feature_idx in 0..n_features {
                    let diff = sample[feature_idx] - centroid[feature_idx];
                    distance += diff * diff;
                }
                distance = distance.sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    best_class_id = class_id;
                }
            }

            // Convert the predicted class back to label combination
            if let Some(label_combination) = self.state.class_to_combination.get(&best_class_id) {
                for label_idx in 0..self.state.n_labels {
                    predictions[[sample_idx, label_idx]] = label_combination[label_idx];
                }
            }
        }

        Ok(predictions)
    }
}

/// Trained state for PrunedLabelPowerset
#[derive(Debug, Clone)]
pub struct PrunedLabelPowersetTrained {
    /// Mapping from class ID to frequent label combination
    pub class_to_combination: HashMap<usize, Vec<i32>>,
    /// Mapping from frequent label combination to class ID
    pub combination_to_class: HashMap<Vec<i32>, usize>,
    /// Mapping from original combinations to frequent combinations
    pub combination_mapping: HashMap<Vec<i32>, Vec<i32>>,
    /// Centroids for each frequent class
    pub class_centroids: HashMap<usize, Array1<f64>>,
    /// List of unique class IDs for frequent combinations
    pub unique_classes: Vec<usize>,
    /// The frequent label combinations that were kept
    pub frequent_combinations: Vec<Vec<i32>>,
    /// Number of labels
    pub n_labels: usize,
    /// Number of features
    pub n_features: usize,
    /// Minimum frequency threshold used
    pub min_frequency: usize,
    /// Pruning strategy used
    pub strategy: PruningStrategy,
}

/// One-vs-Rest Classifier
///
/// A multi-label classification strategy that treats each label as a separate
/// binary classification problem using a one-vs-rest approach. This is essentially
/// the same as BinaryRelevance but with explicit one-vs-rest semantics.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::OneVsRestClassifier;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // This is a simple example showing the structure
/// let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let labels = array![[1, 0], [0, 1], [1, 1]];
/// ```
#[derive(Debug, Clone)]
pub struct OneVsRestClassifier<S = Untrained> {
    state: S,
    n_jobs: Option<i32>,
}

impl OneVsRestClassifier<Untrained> {
    /// Create a new OneVsRestClassifier instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_jobs: None,
        }
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for OneVsRestClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for OneVsRestClassifier<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for OneVsRestClassifier<Untrained> {
    type Fitted = OneVsRestClassifier<OneVsRestClassifierTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        // Delegate to BinaryRelevance implementation
        let br = BinaryRelevance::new().n_jobs(self.n_jobs);
        let fitted_br = br.fit(X, y)?;

        Ok(OneVsRestClassifier {
            state: OneVsRestClassifierTrained {
                binary_relevance: fitted_br,
            },
            n_jobs: self.n_jobs,
        })
    }
}

impl OneVsRestClassifier<OneVsRestClassifierTrained> {
    /// Get the classes for each label
    pub fn classes(&self) -> &[Vec<i32>] {
        self.state.binary_relevance.classes()
    }

    /// Get the number of labels
    pub fn n_labels(&self) -> usize {
        self.state.binary_relevance.n_labels()
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for OneVsRestClassifier<OneVsRestClassifierTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        self.state.binary_relevance.predict(X)
    }
}

impl OneVsRestClassifier<OneVsRestClassifierTrained> {
    /// Predict class probabilities for each label
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        self.state.binary_relevance.predict_proba(X)
    }

    /// Get decision scores for each label
    #[allow(non_snake_case)]
    pub fn decision_function(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.binary_relevance.n_features() {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let mut scores =
            Array2::<Float>::zeros((n_samples, self.state.binary_relevance.n_labels()));

        // Get decision scores from each binary classifier (raw logistic regression scores)
        for label_idx in 0..self.state.binary_relevance.n_labels() {
            if let Some((weights, bias)) = self
                .state
                .binary_relevance
                .state
                .binary_classifiers
                .get(&label_idx)
            {
                for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
                    let score: f64 = sample
                        .iter()
                        .zip(weights.iter())
                        .map(|(&x, &w)| x * w)
                        .sum::<f64>()
                        + bias;

                    scores[[sample_idx, label_idx]] = score;
                }
            }
        }

        Ok(scores)
    }
}

/// Trained state for OneVsRestClassifier
#[derive(Debug, Clone)]
pub struct OneVsRestClassifierTrained {
    /// The underlying BinaryRelevance classifier
    pub binary_relevance: BinaryRelevance<BinaryRelevanceTrained>,
}

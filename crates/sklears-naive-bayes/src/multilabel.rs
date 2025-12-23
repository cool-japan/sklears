//! Multi-label Naive Bayes classifier implementation
//!
//! This module provides multi-label classification capabilities with
//! label dependency modeling and hierarchical label structures.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Fit, Predict},
    traits::{PredictProba, Trained},
};
use std::collections::HashMap;
use std::hash::Hash;

use crate::{GaussianNB, MultinomialNB};

/// Strategy for multi-label classification
#[derive(Debug, Clone)]
pub enum MultiLabelStrategy {
    /// One classifier per label (Binary Relevance)
    BinaryRelevance,
    /// Label powerset - treat each unique label combination as a class
    LabelPowerset,
    /// Classifier chains - model label dependencies
    ClassifierChains,
    /// Multi-label k-NN style approach
    MLkNN,
}

/// Label dependency modeling
#[derive(Debug, Clone)]
pub struct LabelDependencyGraph {
    /// Adjacency matrix for label dependencies
    pub dependencies: Array2<f64>,
    /// Label names
    pub label_names: Vec<String>,
    /// Conditional probability tables
    pub conditional_probs: HashMap<(usize, usize), f64>,
}

impl LabelDependencyGraph {
    pub fn new(n_labels: usize) -> Self {
        Self {
            dependencies: Array2::zeros((n_labels, n_labels)),
            label_names: (0..n_labels).map(|i| format!("label_{}", i)).collect(),
            conditional_probs: HashMap::new(),
        }
    }

    /// Learn label dependencies from training data
    pub fn learn_dependencies(&mut self, y: &Array2<i32>) -> Result<()> {
        let (n_samples, n_labels) = y.dim();

        // Compute pairwise label co-occurrence
        for i in 0..n_labels {
            for j in 0..n_labels {
                if i != j {
                    let mut co_occurrence = 0;
                    let mut total = 0;

                    for sample in 0..n_samples {
                        if y[[sample, i]] == 1 {
                            total += 1;
                            if y[[sample, j]] == 1 {
                                co_occurrence += 1;
                            }
                        }
                    }

                    let dependency = if total > 0 {
                        co_occurrence as f64 / total as f64
                    } else {
                        0.0
                    };

                    self.dependencies[[i, j]] = dependency;
                    self.conditional_probs.insert((i, j), dependency);
                }
            }
        }

        Ok(())
    }

    /// Get most dependent labels for a given label
    pub fn get_dependencies(&self, label_idx: usize, threshold: f64) -> Vec<usize> {
        let mut dependencies = Vec::new();

        for j in 0..self.dependencies.ncols() {
            if label_idx != j && self.dependencies[[label_idx, j]] > threshold {
                dependencies.push(j);
            }
        }

        dependencies.sort_by(|&a, &b| {
            self.dependencies[[label_idx, b]]
                .partial_cmp(&self.dependencies[[label_idx, a]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        dependencies
    }
}

/// Hierarchical label structure
#[derive(Debug, Clone)]
pub struct LabelHierarchy {
    /// Parent-child relationships
    pub parent_child: HashMap<usize, Vec<usize>>,
    /// Child-parent relationships
    pub child_parent: HashMap<usize, usize>,
    /// Root labels (no parents)
    pub roots: Vec<usize>,
    /// Leaf labels (no children)
    pub leaves: Vec<usize>,
}

impl Default for LabelHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelHierarchy {
    pub fn new() -> Self {
        Self {
            parent_child: HashMap::new(),
            child_parent: HashMap::new(),
            roots: Vec::new(),
            leaves: Vec::new(),
        }
    }

    /// Add a parent-child relationship
    pub fn add_relationship(&mut self, parent: usize, child: usize) {
        self.parent_child.entry(parent).or_default().push(child);
        self.child_parent.insert(child, parent);
    }

    /// Get ancestors of a label
    pub fn get_ancestors(&self, label: usize) -> Vec<usize> {
        let mut ancestors = Vec::new();
        let mut current = label;

        while let Some(&parent) = self.child_parent.get(&current) {
            ancestors.push(parent);
            current = parent;
        }

        ancestors
    }

    /// Get descendants of a label
    pub fn get_descendants(&self, label: usize) -> Vec<usize> {
        let mut descendants = Vec::new();
        let mut stack = vec![label];

        while let Some(current) = stack.pop() {
            if let Some(children) = self.parent_child.get(&current) {
                for &child in children {
                    descendants.push(child);
                    stack.push(child);
                }
            }
        }

        descendants
    }

    /// Ensure hierarchical consistency in predictions
    pub fn enforce_consistency(&self, predictions: &mut Array2<f64>) -> Result<()> {
        for sample in 0..predictions.nrows() {
            for (&parent, children) in &self.parent_child {
                let parent_pred = predictions[[sample, parent]];

                // If parent is not predicted, children shouldn't be either
                if parent_pred < 0.5 {
                    for &child in children {
                        if predictions[[sample, child]] > parent_pred {
                            predictions[[sample, child]] = parent_pred;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Label correlation analysis for understanding relationships between labels
#[derive(Debug, Clone)]
pub struct LabelCorrelationAnalysis {
    /// Correlation matrix between labels
    pub correlation_matrix: Array2<f64>,
    /// Mutual information between labels
    pub mutual_information: Array2<f64>,
    /// Jaccard similarity between labels
    pub jaccard_similarity: Array2<f64>,
    /// Label frequencies
    pub label_frequencies: Array1<f64>,
    /// Co-occurrence counts
    pub co_occurrence_counts: Array2<usize>,
    /// Label names
    pub label_names: Vec<String>,
}

impl LabelCorrelationAnalysis {
    pub fn new(n_labels: usize) -> Self {
        Self {
            correlation_matrix: Array2::zeros((n_labels, n_labels)),
            mutual_information: Array2::zeros((n_labels, n_labels)),
            jaccard_similarity: Array2::zeros((n_labels, n_labels)),
            label_frequencies: Array1::zeros(n_labels),
            co_occurrence_counts: Array2::zeros((n_labels, n_labels)),
            label_names: (0..n_labels).map(|i| format!("label_{}", i)).collect(),
        }
    }

    /// Analyze label correlations from training data
    pub fn analyze(&mut self, y: &Array2<i32>) -> Result<()> {
        let (n_samples, n_labels) = y.dim();

        // Compute label frequencies
        for label_idx in 0..n_labels {
            let frequency =
                y.column(label_idx).iter().map(|&x| x as f64).sum::<f64>() / n_samples as f64;
            self.label_frequencies[label_idx] = frequency;
        }

        // Compute co-occurrence counts and correlations
        for i in 0..n_labels {
            for j in 0..n_labels {
                let mut co_occurrence = 0;
                let mut i_count = 0;
                let mut j_count = 0;
                let mut both_count = 0;

                for sample in 0..n_samples {
                    let i_val = y[[sample, i]];
                    let j_val = y[[sample, j]];

                    if i_val == 1 {
                        i_count += 1;
                    }
                    if j_val == 1 {
                        j_count += 1;
                    }
                    if i_val == 1 && j_val == 1 {
                        both_count += 1;
                        co_occurrence += 1;
                    }
                }

                self.co_occurrence_counts[[i, j]] = co_occurrence;

                // Compute Pearson correlation
                if i == j {
                    self.correlation_matrix[[i, j]] = 1.0;
                } else {
                    let mut sum_xy = 0.0;
                    let mut sum_x = 0.0;
                    let mut sum_y = 0.0;
                    let mut sum_x2 = 0.0;
                    let mut sum_y2 = 0.0;

                    for sample in 0..n_samples {
                        let x = y[[sample, i]] as f64;
                        let y_val = y[[sample, j]] as f64;

                        sum_xy += x * y_val;
                        sum_x += x;
                        sum_y += y_val;
                        sum_x2 += x * x;
                        sum_y2 += y_val * y_val;
                    }

                    let n = n_samples as f64;
                    let numerator = n * sum_xy - sum_x * sum_y;
                    let denominator =
                        ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

                    if denominator > 1e-10 {
                        self.correlation_matrix[[i, j]] = numerator / denominator;
                    } else {
                        self.correlation_matrix[[i, j]] = 0.0;
                    }
                }

                // Compute Jaccard similarity
                let union_count = i_count + j_count - both_count;
                if union_count > 0 {
                    self.jaccard_similarity[[i, j]] = both_count as f64 / union_count as f64;
                } else {
                    self.jaccard_similarity[[i, j]] = 0.0;
                }

                // Compute mutual information
                let p_i = self.label_frequencies[i];
                let p_j = self.label_frequencies[j];
                let p_ij = both_count as f64 / n_samples as f64;

                if p_i > 0.0 && p_j > 0.0 && p_ij > 0.0 {
                    self.mutual_information[[i, j]] = p_ij * (p_ij / (p_i * p_j)).ln();
                } else {
                    self.mutual_information[[i, j]] = 0.0;
                }
            }
        }

        Ok(())
    }

    /// Get highly correlated label pairs
    pub fn get_correlated_pairs(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        let mut correlations = Vec::new();
        let n_labels = self.correlation_matrix.nrows();

        for i in 0..n_labels {
            for j in (i + 1)..n_labels {
                let correlation = self.correlation_matrix[[i, j]].abs();
                if correlation > threshold {
                    correlations.push((i, j, correlation));
                }
            }
        }

        // Sort by correlation strength (descending)
        correlations.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        correlations
    }

    /// Get label clusters based on correlation
    pub fn get_label_clusters(&self, threshold: f64) -> Vec<Vec<usize>> {
        let n_labels = self.correlation_matrix.nrows();
        let mut clusters = Vec::new();
        let mut visited = vec![false; n_labels];

        for i in 0..n_labels {
            if visited[i] {
                continue;
            }

            let mut cluster = vec![i];
            visited[i] = true;

            // Find all labels correlated with this one
            for (j, is_visited) in visited.iter_mut().enumerate().take(n_labels) {
                if !*is_visited && self.correlation_matrix[[i, j]].abs() > threshold {
                    cluster.push(j);
                    *is_visited = true;
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    /// Get label imbalance ratios
    pub fn get_imbalance_ratios(&self) -> Array1<f64> {
        let max_freq = self
            .label_frequencies
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b));
        self.label_frequencies.mapv(|freq| {
            if freq > 0.0 {
                max_freq / freq
            } else {
                f64::INFINITY
            }
        })
    }

    /// Generate correlation report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Label Correlation Analysis Report ===\n\n");

        // Label frequencies
        report.push_str("Label Frequencies:\n");
        for (i, &freq) in self.label_frequencies.iter().enumerate() {
            report.push_str(&format!("  {}: {:.4}\n", self.label_names[i], freq));
        }
        report.push('\n');

        // High correlations
        report.push_str("High Correlations (|r| > 0.3):\n");
        let high_corr = self.get_correlated_pairs(0.3);
        if high_corr.is_empty() {
            report.push_str("  None found\n");
        } else {
            for (i, j, corr) in high_corr.iter().take(10) {
                report.push_str(&format!(
                    "  {} <-> {}: {:.4}\n",
                    self.label_names[*i], self.label_names[*j], corr
                ));
            }
        }
        report.push('\n');

        // Label clusters
        report.push_str("Label Clusters (correlation > 0.5):\n");
        let clusters = self.get_label_clusters(0.5);
        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            if cluster.len() > 1 {
                report.push_str(&format!("  Cluster {}: ", cluster_idx + 1));
                for (idx, &label_idx) in cluster.iter().enumerate() {
                    if idx > 0 {
                        report.push_str(", ");
                    }
                    report.push_str(&self.label_names[label_idx]);
                }
                report.push('\n');
            }
        }

        report
    }
}

/// Advanced chain classifier with sophisticated label ordering
pub struct AdvancedChainClassifier {
    /// Base classifier type
    pub base_classifier: String,
    /// Chain ordering strategy
    pub ordering_strategy: ChainOrderingStrategy,
    /// Number of chains for ensemble
    pub n_chains: usize,
    /// Alpha parameter for smoothing
    pub alpha: f64,
    /// Whether to fit class priors
    pub fit_prior: bool,
    /// Individual chain classifiers
    chains: Vec<Vec<Box<dyn MultiLabelClassifier>>>,
    /// Label orderings for each chain
    label_orderings: Vec<Vec<usize>>,
    /// Label correlation analysis
    correlation_analysis: Option<LabelCorrelationAnalysis>,
    /// Number of labels
    n_labels: Option<usize>,
    /// Whether the model is fitted
    fitted: bool,
}

/// Strategy for ordering labels in classifier chains
#[derive(Debug, Clone)]
pub enum ChainOrderingStrategy {
    /// Random ordering
    Random,
    /// Order by label frequency
    Frequency,
    /// Order by label correlation
    Correlation,
    /// Multiple random orderings (ensemble)
    EnsembleRandom,
    /// Optimized ordering based on mutual information
    MutualInformation,
}

impl Default for AdvancedChainClassifier {
    fn default() -> Self {
        Self {
            base_classifier: "gaussian".to_string(),
            ordering_strategy: ChainOrderingStrategy::EnsembleRandom,
            n_chains: 3,
            alpha: 1e-9,
            fit_prior: true,
            chains: Vec::new(),
            label_orderings: Vec::new(),
            correlation_analysis: None,
            n_labels: None,
            fitted: false,
        }
    }
}

impl AdvancedChainClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn base_classifier(mut self, classifier: &str) -> Self {
        self.base_classifier = classifier.to_string();
        self
    }

    pub fn ordering_strategy(mut self, strategy: ChainOrderingStrategy) -> Self {
        self.ordering_strategy = strategy;
        self
    }

    pub fn n_chains(mut self, n_chains: usize) -> Self {
        self.n_chains = n_chains;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Create a base classifier
    fn create_classifier(&self) -> Box<dyn MultiLabelClassifier> {
        match self.base_classifier.as_str() {
            "gaussian" => Box::new(GaussianNBWrapper::new(self.alpha, self.fit_prior)),
            "multinomial" => Box::new(MultinomialNBWrapper::new(self.alpha, self.fit_prior)),
            _ => Box::new(GaussianNBWrapper::new(self.alpha, self.fit_prior)),
        }
    }

    /// Generate label ordering based on strategy
    fn generate_ordering(&self, y: &Array2<i32>, chain_idx: usize) -> Vec<usize> {
        let n_labels = y.ncols();
        let mut ordering: Vec<usize> = (0..n_labels).collect();

        match self.ordering_strategy {
            ChainOrderingStrategy::Random | ChainOrderingStrategy::EnsembleRandom => {
                // Simple pseudo-random shuffle based on chain index
                use std::collections::hash_map::DefaultHasher;
                use std::hash::Hasher;

                let mut hasher = DefaultHasher::new();
                chain_idx.hash(&mut hasher);
                let seed = hasher.finish();

                // Simple shuffling algorithm using the seed
                for i in 0..n_labels {
                    let mut hasher = DefaultHasher::new();
                    (seed, i).hash(&mut hasher);
                    let j = (hasher.finish() as usize) % (i + 1);
                    ordering.swap(i, j);
                }
            }
            ChainOrderingStrategy::Frequency => {
                if let Some(ref analysis) = self.correlation_analysis {
                    // Sort by frequency (ascending - rare labels first)
                    ordering.sort_by(|&a, &b| {
                        analysis.label_frequencies[a]
                            .partial_cmp(&analysis.label_frequencies[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
            ChainOrderingStrategy::Correlation => {
                if let Some(ref analysis) = self.correlation_analysis {
                    // Order by average correlation with other labels
                    let mut avg_correlations: Vec<(usize, f64)> = (0..n_labels)
                        .map(|i| {
                            let avg_corr = analysis
                                .correlation_matrix
                                .row(i)
                                .iter()
                                .enumerate()
                                .filter(|(j, _)| *j != i)
                                .map(|(_, &corr)| corr.abs())
                                .sum::<f64>()
                                / (n_labels - 1) as f64;
                            (i, avg_corr)
                        })
                        .collect();

                    avg_correlations
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    ordering = avg_correlations.into_iter().map(|(idx, _)| idx).collect();
                }
            }
            ChainOrderingStrategy::MutualInformation => {
                if let Some(ref analysis) = self.correlation_analysis {
                    // Order by mutual information
                    let mut mi_scores: Vec<(usize, f64)> = (0..n_labels)
                        .map(|i| {
                            let avg_mi = analysis
                                .mutual_information
                                .row(i)
                                .iter()
                                .enumerate()
                                .filter(|(j, _)| *j != i)
                                .map(|(_, &mi)| mi)
                                .sum::<f64>()
                                / (n_labels - 1) as f64;
                            (i, avg_mi)
                        })
                        .collect();

                    mi_scores
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    ordering = mi_scores.into_iter().map(|(idx, _)| idx).collect();
                }
            }
        }

        ordering
    }

    /// Fit the advanced chain classifier
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array2<i32>) -> Result<()> {
        let (n_samples, n_labels) = y.dim();
        let n_features = x.ncols();
        self.n_labels = Some(n_labels);

        // Perform correlation analysis
        let mut analysis = LabelCorrelationAnalysis::new(n_labels);
        analysis.analyze(y)?;
        self.correlation_analysis = Some(analysis);

        // Clear previous chains
        self.chains.clear();
        self.label_orderings.clear();

        // Create multiple chains
        let n_chains = match self.ordering_strategy {
            ChainOrderingStrategy::EnsembleRandom => self.n_chains,
            _ => 1,
        };

        for chain_idx in 0..n_chains {
            let ordering = self.generate_ordering(y, chain_idx);
            self.label_orderings.push(ordering.clone());

            let mut chain_classifiers = Vec::new();

            for (pos, &label_idx) in ordering.iter().enumerate() {
                let mut classifier = self.create_classifier();

                // Build extended feature matrix including previous labels in chain
                let mut x_extended = Array2::zeros((n_samples, n_features + pos));

                // Copy original features
                for i in 0..n_samples {
                    for j in 0..n_features {
                        x_extended[[i, j]] = x[[i, j]];
                    }
                }

                // Add previous labels in chain as features
                for i in 0..n_samples {
                    for (prev_pos, &prev_label_idx) in ordering.iter().take(pos).enumerate() {
                        x_extended[[i, n_features + prev_pos]] = y[[i, prev_label_idx]] as f64;
                    }
                }

                let y_label = y.column(label_idx).to_owned();
                classifier.fit(&x_extended, &y_label)?;
                chain_classifiers.push(classifier);
            }

            self.chains.push(chain_classifiers);
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict using the advanced chain classifier
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<i32>> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let n_samples = x.nrows();
        let n_labels = self.n_labels.unwrap();
        let n_features = x.ncols();

        if self.chains.len() == 1 {
            // Single chain prediction
            self.predict_single_chain(x, 0)
        } else {
            // Ensemble prediction
            let mut all_predictions = Vec::new();

            for chain_idx in 0..self.chains.len() {
                let chain_pred = self.predict_single_chain(x, chain_idx)?;
                all_predictions.push(chain_pred);
            }

            // Aggregate predictions using majority voting
            let mut final_predictions = Array2::zeros((n_samples, n_labels));

            for sample in 0..n_samples {
                for label in 0..n_labels {
                    let votes: i32 = all_predictions
                        .iter()
                        .map(|pred| pred[[sample, label]])
                        .sum();

                    final_predictions[[sample, label]] = if votes > (self.chains.len() as i32) / 2 {
                        1
                    } else {
                        0
                    };
                }
            }

            Ok(final_predictions)
        }
    }

    /// Predict using a single chain
    fn predict_single_chain(&self, x: &Array2<f64>, chain_idx: usize) -> Result<Array2<i32>> {
        let n_samples = x.nrows();
        let n_labels = self.n_labels.unwrap();
        let n_features = x.ncols();

        let ordering = &self.label_orderings[chain_idx];
        let chain = &self.chains[chain_idx];

        let mut predictions = Array2::zeros((n_samples, n_labels));

        for (pos, &label_idx) in ordering.iter().enumerate() {
            // Build extended feature matrix
            let mut x_extended = Array2::zeros((n_samples, n_features + pos));

            // Copy original features
            for i in 0..n_samples {
                for j in 0..n_features {
                    x_extended[[i, j]] = x[[i, j]];
                }
            }

            // Add previous predictions as features
            for i in 0..n_samples {
                for (prev_pos, &prev_label_idx) in ordering.iter().take(pos).enumerate() {
                    x_extended[[i, n_features + prev_pos]] =
                        predictions[[i, prev_label_idx]] as f64;
                }
            }

            let pred = chain[pos].predict(&x_extended)?;
            for (i, &p) in pred.iter().enumerate() {
                predictions[[i, label_idx]] = p;
            }
        }

        Ok(predictions)
    }

    /// Get correlation analysis results
    pub fn get_correlation_analysis(&self) -> Option<&LabelCorrelationAnalysis> {
        self.correlation_analysis.as_ref()
    }
}

/// Multi-label Naive Bayes classifier
pub struct MultiLabelNB {
    /// Strategy for multi-label classification
    pub strategy: MultiLabelStrategy,
    /// Base classifier type
    pub base_classifier: String,
    /// Alpha parameter for smoothing
    pub alpha: f64,
    /// Whether to fit class priors
    pub fit_prior: bool,
    /// Label dependency threshold
    pub dependency_threshold: f64,
    /// Individual classifiers for each label
    classifiers: Vec<Box<dyn MultiLabelClassifier>>,
    /// Label dependency graph
    label_dependencies: Option<LabelDependencyGraph>,
    /// Label hierarchy
    label_hierarchy: Option<LabelHierarchy>,
    /// Number of labels
    n_labels: Option<usize>,
    /// Whether the model is fitted
    fitted: bool,
}

/// Trait for multi-label classifiers
pub trait MultiLabelClassifier: Send + Sync {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<()>;
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>>;
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
}

/// Wrapper for Gaussian NB
pub struct GaussianNBWrapper {
    classifier: Option<GaussianNB<Trained>>,
    alpha: f64,
    fit_prior: bool,
}

impl GaussianNBWrapper {
    pub fn new(alpha: f64, fit_prior: bool) -> Self {
        Self {
            classifier: None,
            alpha,
            fit_prior,
        }
    }
}

impl MultiLabelClassifier for GaussianNBWrapper {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<()> {
        let classifier = GaussianNB::new().var_smoothing(self.alpha);

        let trained = classifier.fit(x, y)?;
        self.classifier = Some(trained);
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        match &self.classifier {
            Some(clf) => clf.predict(x),
            None => Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            }),
        }
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        match &self.classifier {
            Some(clf) => {
                let proba_matrix = clf.predict_proba(x)?;
                // Return probability of positive class (assumes binary classification)
                Ok(proba_matrix.column(1).to_owned())
            }
            None => Err(SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            }),
        }
    }
}

/// Wrapper for Multinomial NB
pub struct MultinomialNBWrapper {
    classifier: Option<MultinomialNB<Trained>>,
    alpha: f64,
    fit_prior: bool,
}

impl MultinomialNBWrapper {
    pub fn new(alpha: f64, fit_prior: bool) -> Self {
        Self {
            classifier: None,
            alpha,
            fit_prior,
        }
    }
}

impl MultiLabelClassifier for MultinomialNBWrapper {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<()> {
        let classifier = MultinomialNB::new()
            .alpha(self.alpha)
            .fit_prior(self.fit_prior);

        let trained = classifier.fit(x, y)?;
        self.classifier = Some(trained);
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        match &self.classifier {
            Some(clf) => clf.predict(x),
            None => Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            }),
        }
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        match &self.classifier {
            Some(clf) => {
                let proba_matrix = clf.predict_proba(x)?;
                // Return probability of positive class
                Ok(proba_matrix.column(1).to_owned())
            }
            None => Err(SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            }),
        }
    }
}

impl Default for MultiLabelNB {
    fn default() -> Self {
        Self {
            strategy: MultiLabelStrategy::BinaryRelevance,
            base_classifier: "gaussian".to_string(),
            alpha: 1e-9,
            fit_prior: true,
            dependency_threshold: 0.1,
            classifiers: Vec::new(),
            label_dependencies: None,
            label_hierarchy: None,
            n_labels: None,
            fitted: false,
        }
    }
}

impl MultiLabelNB {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn strategy(mut self, strategy: MultiLabelStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn base_classifier(mut self, classifier: &str) -> Self {
        self.base_classifier = classifier.to_string();
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    pub fn dependency_threshold(mut self, threshold: f64) -> Self {
        self.dependency_threshold = threshold;
        self
    }

    pub fn set_label_hierarchy(&mut self, hierarchy: LabelHierarchy) {
        self.label_hierarchy = Some(hierarchy);
    }

    /// Create a base classifier
    fn create_classifier(&self) -> Box<dyn MultiLabelClassifier> {
        match self.base_classifier.as_str() {
            "gaussian" => Box::new(GaussianNBWrapper::new(self.alpha, self.fit_prior)),
            "multinomial" => Box::new(MultinomialNBWrapper::new(self.alpha, self.fit_prior)),
            _ => Box::new(GaussianNBWrapper::new(self.alpha, self.fit_prior)),
        }
    }

    /// Fit the multi-label classifier
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array2<i32>) -> Result<()> {
        let (n_samples, n_labels) = y.dim();
        self.n_labels = Some(n_labels);

        match self.strategy {
            MultiLabelStrategy::BinaryRelevance => {
                self.fit_binary_relevance(x, y)?;
            }
            MultiLabelStrategy::ClassifierChains => {
                self.fit_classifier_chains(x, y)?;
            }
            MultiLabelStrategy::LabelPowerset => {
                self.fit_label_powerset(x, y)?;
            }
            MultiLabelStrategy::MLkNN => {
                return Err(SklearsError::NotImplemented("MLkNN".to_string()));
            }
        }

        // Learn label dependencies if using classifier chains
        if matches!(self.strategy, MultiLabelStrategy::ClassifierChains) {
            let mut dependencies = LabelDependencyGraph::new(n_labels);
            dependencies.learn_dependencies(y)?;
            self.label_dependencies = Some(dependencies);
        }

        self.fitted = true;
        Ok(())
    }

    /// Fit using binary relevance strategy
    fn fit_binary_relevance(&mut self, x: &Array2<f64>, y: &Array2<i32>) -> Result<()> {
        let n_labels = y.ncols();
        self.classifiers.clear();

        for label_idx in 0..n_labels {
            let mut classifier = self.create_classifier();
            let y_label = y.column(label_idx).to_owned();
            classifier.fit(x, &y_label)?;
            self.classifiers.push(classifier);
        }

        Ok(())
    }

    /// Fit using classifier chains strategy
    fn fit_classifier_chains(&mut self, x: &Array2<f64>, y: &Array2<i32>) -> Result<()> {
        let (n_samples, n_labels) = y.dim();
        let n_features = x.ncols();
        self.classifiers.clear();

        for label_idx in 0..n_labels {
            let mut classifier = self.create_classifier();

            // For classifier chains, include previous labels as features
            let mut x_extended = Array2::zeros((n_samples, n_features + label_idx));

            // Copy original features
            for i in 0..n_samples {
                for j in 0..n_features {
                    x_extended[[i, j]] = x[[i, j]];
                }
            }

            // Add previous labels as features
            for i in 0..n_samples {
                for prev_label in 0..label_idx {
                    x_extended[[i, n_features + prev_label]] = y[[i, prev_label]] as f64;
                }
            }

            let y_label = y.column(label_idx).to_owned();
            classifier.fit(&x_extended, &y_label)?;
            self.classifiers.push(classifier);
        }

        Ok(())
    }

    /// Fit using label powerset strategy
    fn fit_label_powerset(&mut self, x: &Array2<f64>, y: &Array2<i32>) -> Result<()> {
        // Convert multi-label matrix to single-label by treating each
        // unique label combination as a separate class
        let n_samples = y.nrows();
        let mut label_combinations: HashMap<Vec<i32>, i32> = HashMap::new();
        let mut y_single = Array1::zeros(n_samples);
        let mut class_id = 0;

        for sample in 0..n_samples {
            let label_combo: Vec<i32> = y.row(sample).to_vec();

            let class_id_for_combo = *label_combinations
                .entry(label_combo.clone())
                .or_insert_with(|| {
                    let id = class_id;
                    class_id += 1;
                    id
                });

            y_single[sample] = class_id_for_combo;
        }

        // Train single classifier on the transformed problem
        let mut classifier = self.create_classifier();
        classifier.fit(x, &y_single)?;
        self.classifiers.push(classifier);

        Ok(())
    }

    /// Predict labels for new samples
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<i32>> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let n_samples = x.nrows();
        let n_labels = self.n_labels.unwrap();
        let mut predictions = Array2::zeros((n_samples, n_labels));

        match self.strategy {
            MultiLabelStrategy::BinaryRelevance => {
                for (label_idx, classifier) in self.classifiers.iter().enumerate() {
                    let pred = classifier.predict(x)?;
                    predictions.column_mut(label_idx).assign(&pred);
                }
            }
            MultiLabelStrategy::ClassifierChains => {
                self.predict_classifier_chains(x, &mut predictions)?;
            }
            MultiLabelStrategy::LabelPowerset => {
                return Err(SklearsError::NotImplemented(
                    "LabelPowerset prediction".to_string(),
                ));
            }
            MultiLabelStrategy::MLkNN => {
                return Err(SklearsError::NotImplemented("MLkNN".to_string()));
            }
        }

        // Apply hierarchical consistency if hierarchy is defined
        if let Some(ref hierarchy) = self.label_hierarchy {
            let mut float_predictions = predictions.mapv(|x| x as f64);
            hierarchy.enforce_consistency(&mut float_predictions)?;
            predictions = float_predictions.mapv(|x| if x > 0.5 { 1 } else { 0 });
        }

        Ok(predictions)
    }

    /// Predict using classifier chains
    fn predict_classifier_chains(
        &self,
        x: &Array2<f64>,
        predictions: &mut Array2<i32>,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let n_labels = self.n_labels.unwrap();

        for label_idx in 0..n_labels {
            // Build extended feature matrix including previous predictions
            let mut x_extended = Array2::zeros((n_samples, n_features + label_idx));

            // Copy original features
            for i in 0..n_samples {
                for j in 0..n_features {
                    x_extended[[i, j]] = x[[i, j]];
                }
            }

            // Add previous predictions as features
            for i in 0..n_samples {
                for prev_label in 0..label_idx {
                    x_extended[[i, n_features + prev_label]] = predictions[[i, prev_label]] as f64;
                }
            }

            let pred = self.classifiers[label_idx].predict(&x_extended)?;
            predictions.column_mut(label_idx).assign(&pred);
        }

        Ok(())
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            });
        }

        let n_samples = x.nrows();
        let n_labels = self.n_labels.unwrap();
        let mut probabilities = Array2::zeros((n_samples, n_labels));

        match self.strategy {
            MultiLabelStrategy::BinaryRelevance => {
                for (label_idx, classifier) in self.classifiers.iter().enumerate() {
                    let proba = classifier.predict_proba(x)?;
                    probabilities.column_mut(label_idx).assign(&proba);
                }
            }
            MultiLabelStrategy::ClassifierChains => {
                // For classifier chains, we need to predict sequentially
                self.predict_proba_classifier_chains(x, &mut probabilities)?;
            }
            MultiLabelStrategy::LabelPowerset => {
                return Err(SklearsError::NotImplemented(
                    "LabelPowerset probability prediction not yet implemented".to_string(),
                ));
            }
            MultiLabelStrategy::MLkNN => {
                return Err(SklearsError::NotImplemented("MLkNN".to_string()));
            }
        }

        Ok(probabilities)
    }

    /// Predict probabilities using classifier chains
    fn predict_proba_classifier_chains(
        &self,
        x: &Array2<f64>,
        probabilities: &mut Array2<f64>,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let n_labels = self.n_labels.unwrap();

        for label_idx in 0..n_labels {
            // Build extended feature matrix
            let mut x_extended = Array2::zeros((n_samples, n_features + label_idx));

            // Copy original features
            for i in 0..n_samples {
                for j in 0..n_features {
                    x_extended[[i, j]] = x[[i, j]];
                }
            }

            // Add previous predictions as features (use thresholded probabilities)
            for i in 0..n_samples {
                for prev_label in 0..label_idx {
                    let pred = if probabilities[[i, prev_label]] > 0.5 {
                        1.0
                    } else {
                        0.0
                    };
                    x_extended[[i, n_features + prev_label]] = pred;
                }
            }

            let proba = self.classifiers[label_idx].predict_proba(&x_extended)?;
            probabilities.column_mut(label_idx).assign(&proba);
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_label_dependency_graph() {
        let y = Array2::from_shape_vec((4, 3), vec![1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]).unwrap();

        let mut graph = LabelDependencyGraph::new(3);
        graph.learn_dependencies(&y).unwrap();

        // Check that dependencies were learned
        assert!(graph.dependencies[[0, 1]] > 0.0);
        assert!(graph.dependencies[[0, 2]] > 0.0);
    }

    #[test]
    fn test_multilabel_nb_binary_relevance() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 1.0]).unwrap();

        let y = Array2::from_shape_vec((4, 2), vec![1, 0, 0, 1, 1, 1, 0, 0]).unwrap();

        let mut model = MultiLabelNB::new()
            .strategy(MultiLabelStrategy::BinaryRelevance)
            .base_classifier("gaussian");

        model.fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();

        assert_eq!(predictions.dim(), (4, 2));
    }

    #[test]
    fn test_multilabel_nb_classifier_chains() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 1.0]).unwrap();

        let y = Array2::from_shape_vec((4, 2), vec![1, 0, 0, 1, 1, 1, 0, 0]).unwrap();

        let mut model = MultiLabelNB::new()
            .strategy(MultiLabelStrategy::ClassifierChains)
            .base_classifier("gaussian");

        model.fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();

        assert_eq!(predictions.dim(), (4, 2));
    }

    #[test]
    fn test_label_correlation_analysis() {
        let y = Array2::from_shape_vec(
            (6, 3),
            vec![
                1, 1, 0, // labels 0 and 1 co-occur
                1, 1, 0, // labels 0 and 1 co-occur
                0, 0, 1, // only label 2
                1, 0, 1, // labels 0 and 2 co-occur
                0, 1, 1, // labels 1 and 2 co-occur
                1, 1, 1, // all labels co-occur
            ],
        )
        .unwrap();

        let mut analysis = LabelCorrelationAnalysis::new(3);
        analysis.analyze(&y).unwrap();

        // Check that frequencies are computed correctly
        assert_abs_diff_eq!(analysis.label_frequencies[0], 4.0 / 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(analysis.label_frequencies[1], 4.0 / 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(analysis.label_frequencies[2], 4.0 / 6.0, epsilon = 1e-10);

        // Check that correlations are computed
        assert!(analysis.correlation_matrix[[0, 1]].abs() > 0.0);
        assert!(analysis.correlation_matrix[[0, 2]].abs() > 0.0);
        assert!(analysis.correlation_matrix[[1, 2]].abs() > 0.0);

        // Test correlated pairs
        let correlated_pairs = analysis.get_correlated_pairs(0.1);
        assert!(!correlated_pairs.is_empty());

        // Test report generation
        let report = analysis.generate_report();
        assert!(report.contains("Label Correlation Analysis Report"));
        assert!(report.contains("Label Frequencies"));
    }

    #[test]
    fn test_advanced_chain_classifier() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0],
        )
        .unwrap();

        let y = Array2::from_shape_vec(
            (6, 3),
            vec![1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        )
        .unwrap();

        let mut model = AdvancedChainClassifier::new()
            .ordering_strategy(ChainOrderingStrategy::Frequency)
            .base_classifier("gaussian");

        model.fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();

        assert_eq!(predictions.dim(), (6, 3));

        // Test that correlation analysis was performed
        assert!(model.get_correlation_analysis().is_some());

        // Test ensemble chains
        let mut ensemble_model = AdvancedChainClassifier::new()
            .ordering_strategy(ChainOrderingStrategy::EnsembleRandom)
            .n_chains(3)
            .base_classifier("gaussian");

        ensemble_model.fit(&x, &y).unwrap();
        let ensemble_predictions = ensemble_model.predict(&x).unwrap();
        assert_eq!(ensemble_predictions.dim(), (6, 3));
    }

    #[test]
    fn test_chain_ordering_strategies() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 1.0]).unwrap();

        let y = Array2::from_shape_vec((4, 3), vec![1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]).unwrap();

        // Test different ordering strategies
        let strategies = vec![
            ChainOrderingStrategy::Random,
            ChainOrderingStrategy::Frequency,
            ChainOrderingStrategy::Correlation,
            ChainOrderingStrategy::MutualInformation,
        ];

        for strategy in strategies {
            let mut model = AdvancedChainClassifier::new()
                .ordering_strategy(strategy)
                .base_classifier("gaussian");

            model.fit(&x, &y).unwrap();
            let predictions = model.predict(&x).unwrap();
            assert_eq!(predictions.dim(), (4, 3));
        }
    }

    #[test]
    fn test_label_hierarchy_consistency() {
        let mut hierarchy = LabelHierarchy::new();
        hierarchy.add_relationship(0, 1); // 0 is parent of 1
        hierarchy.add_relationship(0, 2); // 0 is parent of 2

        let mut predictions = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.0, 0.8, 0.7, // Child predictions higher than parent
                0.9, 0.3, 0.6, // Normal case
            ],
        )
        .unwrap();

        hierarchy.enforce_consistency(&mut predictions).unwrap();

        // First sample: children should be reduced to parent level
        assert!(predictions[[0, 1]] <= predictions[[0, 0]]);
        assert!(predictions[[0, 2]] <= predictions[[0, 0]]);

        // Second sample: should be unchanged
        assert_abs_diff_eq!(predictions[[1, 0]], 0.9, epsilon = 1e-10);
    }

    #[test]
    fn test_multilabel_nb_with_hierarchy() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 1.0]).unwrap();

        let y = Array2::from_shape_vec((4, 3), vec![1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]).unwrap();

        let mut hierarchy = LabelHierarchy::new();
        hierarchy.add_relationship(0, 1); // 0 is parent of 1
        hierarchy.add_relationship(0, 2); // 0 is parent of 2

        let mut model = MultiLabelNB::new()
            .strategy(MultiLabelStrategy::BinaryRelevance)
            .base_classifier("gaussian");

        model.set_label_hierarchy(hierarchy);
        model.fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();

        assert_eq!(predictions.dim(), (4, 3));
    }
}

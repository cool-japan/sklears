//! Tree-Augmented Naive Bayes (TAN) implementation
//!
//! TAN extends Naive Bayes by allowing dependencies between features while
//! maintaining computational efficiency through a tree structure.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TANError {
    #[error("Features and targets have different number of samples")]
    DimensionMismatch,
    #[error("Empty dataset provided")]
    EmptyDataset,
    #[error("Invalid feature index: {0}")]
    InvalidFeatureIndex(usize),
    #[error("Numerical computation error: {0}")]
    NumericalError(String),
}

/// Edge in the dependency tree representing feature dependencies
#[derive(Debug, Clone)]
pub struct TreeEdge {
    pub parent: usize,
    pub child: usize,
    pub weight: f64,
}

/// Tree structure for feature dependencies
#[derive(Debug, Clone)]
pub struct DependencyTree {
    edges: Vec<TreeEdge>,
    adjacency: HashMap<usize, Vec<usize>>,
    parents: HashMap<usize, usize>,
}

impl Default for DependencyTree {
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyTree {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            adjacency: HashMap::new(),
            parents: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, edge: TreeEdge) {
        self.adjacency
            .entry(edge.parent)
            .or_default()
            .push(edge.child);
        self.parents.insert(edge.child, edge.parent);
        self.edges.push(edge);
    }

    pub fn get_parent(&self, node: usize) -> Option<usize> {
        self.parents.get(&node).copied()
    }

    pub fn get_children(&self, node: usize) -> Vec<usize> {
        self.adjacency.get(&node).cloned().unwrap_or_default()
    }
}

/// Configuration for Tree-Augmented Naive Bayes
#[derive(Debug, Clone)]
pub struct TANConfig {
    pub smoothing: f64,
    pub max_dependencies: Option<usize>,
    pub dependency_threshold: f64,
    pub use_conditional_mutual_info: bool,
}

impl Default for TANConfig {
    fn default() -> Self {
        Self {
            smoothing: 1.0,
            max_dependencies: None,
            dependency_threshold: 0.001,
            use_conditional_mutual_info: true,
        }
    }
}

/// Tree-Augmented Naive Bayes classifier
pub struct TreeAugmentedNB {
    config: TANConfig,
    classes: Array1<i32>,
    class_log_prior: Array1<f64>,
    dependency_tree: DependencyTree,
    feature_log_prob: Array3<f64>, // [feature, parent_value, feature_value]
    conditional_log_prob: Array3<f64>, // [child, parent, class]
    n_features: usize,
    n_classes: usize,
    feature_counts: HashMap<usize, usize>,
}

impl TreeAugmentedNB {
    pub fn new(config: TANConfig) -> Self {
        Self {
            config,
            classes: Array1::default((0,)),
            class_log_prior: Array1::default((0,)),
            dependency_tree: DependencyTree::new(),
            feature_log_prob: Array3::default((0, 0, 0)),
            conditional_log_prob: Array3::default((0, 0, 0)),
            n_features: 0,
            n_classes: 0,
            feature_counts: HashMap::new(),
        }
    }

    /// Fit the TAN classifier to training data
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<i32>) -> Result<(), TANError> {
        if X.nrows() != y.len() {
            return Err(TANError::DimensionMismatch);
        }
        if X.nrows() == 0 {
            return Err(TANError::EmptyDataset);
        }

        self.n_features = X.ncols();
        self.classes = self.get_unique_classes(y);
        self.n_classes = self.classes.len();

        // Compute class priors
        self.compute_class_priors(y)?;

        // Discretize continuous features
        let X_discrete = self.discretize_features(X)?;

        // Learn dependency tree structure
        self.learn_tree_structure(&X_discrete, y)?;

        // Estimate conditional probabilities
        self.estimate_conditional_probabilities(&X_discrete, y)?;

        Ok(())
    }

    /// Predict class labels for test samples
    pub fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>, TANError> {
        let log_proba = self.predict_log_proba(X)?;
        let predictions = log_proba
            .axis_iter(scirs2_core::ndarray::Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                self.classes[max_idx]
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    /// Predict class log-probabilities for test samples
    #[allow(non_snake_case)]
    pub fn predict_log_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>, TANError> {
        if X.ncols() != self.n_features {
            return Err(TANError::DimensionMismatch);
        }

        let X_discrete = self.discretize_features(X)?;
        let mut log_proba = Array2::zeros((X.nrows(), self.n_classes));

        for (sample_idx, sample) in X_discrete
            .axis_iter(scirs2_core::ndarray::Axis(0))
            .enumerate()
        {
            for (class_idx, _) in self.classes.iter().enumerate() {
                let mut log_prob = self.class_log_prior[class_idx];

                // Add log probabilities for root features (no parents)
                for feature_idx in 0..self.n_features {
                    if self.dependency_tree.get_parent(feature_idx).is_none() {
                        let feature_value = sample[feature_idx] as usize;
                        if feature_value < self.feature_log_prob.shape()[2] {
                            log_prob += self.feature_log_prob[[feature_idx, 0, feature_value]];
                        }
                    } else {
                        // Add conditional log probabilities for dependent features
                        let parent_idx = self.dependency_tree.get_parent(feature_idx).unwrap();
                        let parent_value = sample[parent_idx] as usize;
                        let feature_value = sample[feature_idx] as usize;

                        if parent_value < self.conditional_log_prob.shape()[1]
                            && feature_value < self.conditional_log_prob.shape()[2]
                        {
                            log_prob +=
                                self.conditional_log_prob[[feature_idx, parent_value, class_idx]];
                        }
                    }
                }

                log_proba[[sample_idx, class_idx]] = log_prob;
            }
        }

        Ok(log_proba)
    }

    fn get_unique_classes(&self, y: &Array1<i32>) -> Array1<i32> {
        let mut unique_classes: Vec<i32> = y
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        unique_classes.sort_unstable();
        Array1::from_vec(unique_classes)
    }

    fn compute_class_priors(&mut self, y: &Array1<i32>) -> Result<(), TANError> {
        let n_samples = y.len() as f64;
        let mut class_counts = Array1::zeros(self.n_classes);

        for &label in y.iter() {
            for (i, &class) in self.classes.iter().enumerate() {
                if label == class {
                    class_counts[i] += 1.0;
                }
            }
        }

        // Apply smoothing
        class_counts += self.config.smoothing;
        let total_count = class_counts.sum();

        self.class_log_prior = class_counts.map(|&count| (count / total_count).ln());
        Ok(())
    }

    fn discretize_features(&self, X: &Array2<f64>) -> Result<Array2<i32>, TANError> {
        // Simple equal-width discretization into 5 bins
        let n_bins = 5;
        let mut X_discrete = Array2::zeros((X.nrows(), X.ncols()));

        for j in 0..X.ncols() {
            let column = X.column(j);
            let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let bin_width = (max_val - min_val) / n_bins as f64;

            for i in 0..X.nrows() {
                let bin = if bin_width > 0.0 {
                    ((X[[i, j]] - min_val) / bin_width).floor() as i32
                } else {
                    0
                };
                X_discrete[[i, j]] = bin.min(n_bins - 1).max(0);
            }
        }

        Ok(X_discrete)
    }

    fn learn_tree_structure(&mut self, X: &Array2<i32>, y: &Array1<i32>) -> Result<(), TANError> {
        // Compute mutual information between all feature pairs conditioned on class
        let mut mutual_info_matrix = Array2::zeros((self.n_features, self.n_features));

        for i in 0..self.n_features {
            for j in (i + 1)..self.n_features {
                let mi = self.compute_conditional_mutual_information(X, y, i, j)?;
                mutual_info_matrix[[i, j]] = mi;
                mutual_info_matrix[[j, i]] = mi;
            }
        }

        // Build maximum spanning tree using Kruskal's algorithm
        let mut edges: Vec<TreeEdge> = Vec::new();
        for i in 0..self.n_features {
            for j in (i + 1)..self.n_features {
                edges.push(TreeEdge {
                    parent: i,
                    child: j,
                    weight: mutual_info_matrix[[i, j]],
                });
            }
        }

        // Sort edges by weight (descending)
        edges.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build tree using union-find
        let mut parent: Vec<usize> = (0..self.n_features).collect();
        let mut rank: Vec<usize> = vec![0; self.n_features];
        let mut selected_edges = Vec::new();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, rank: &mut [usize], x: usize, y: usize) -> bool {
            let root_x = find(parent, x);
            let root_y = find(parent, y);

            if root_x == root_y {
                return false;
            }

            match rank[root_x].cmp(&rank[root_y]) {
                std::cmp::Ordering::Less => parent[root_x] = root_y,
                std::cmp::Ordering::Greater => parent[root_y] = root_x,
                std::cmp::Ordering::Equal => {
                    parent[root_y] = root_x;
                    rank[root_x] += 1;
                }
            }
            true
        }

        for edge in edges {
            if selected_edges.len() >= self.n_features - 1 {
                break;
            }
            if union(&mut parent, &mut rank, edge.parent, edge.child) {
                selected_edges.push(edge);
            }
        }

        // Add edges to dependency tree
        for edge in selected_edges {
            self.dependency_tree.add_edge(edge);
        }

        Ok(())
    }

    fn compute_conditional_mutual_information(
        &self,
        X: &Array2<i32>,
        y: &Array1<i32>,
        feature1: usize,
        feature2: usize,
    ) -> Result<f64, TANError> {
        let mut mi = 0.0;

        // Get unique values for features
        let values1: HashSet<i32> = X.column(feature1).iter().cloned().collect();
        let values2: HashSet<i32> = X.column(feature2).iter().cloned().collect();

        for &class in self.classes.iter() {
            let class_mask: Vec<bool> = y.iter().map(|&label| label == class).collect();
            let class_prob =
                class_mask.iter().filter(|&&mask| mask).count() as f64 / y.len() as f64;

            if class_prob == 0.0 {
                continue;
            }

            for &val1 in &values1 {
                for &val2 in &values2 {
                    let count_joint = X
                        .axis_iter(scirs2_core::ndarray::Axis(0))
                        .zip(class_mask.iter())
                        .filter(|(row, &mask)| {
                            mask && row[feature1] == val1 && row[feature2] == val2
                        })
                        .count() as f64;

                    let count_val1 = X
                        .axis_iter(scirs2_core::ndarray::Axis(0))
                        .zip(class_mask.iter())
                        .filter(|(row, &mask)| mask && row[feature1] == val1)
                        .count() as f64;

                    let count_val2 = X
                        .axis_iter(scirs2_core::ndarray::Axis(0))
                        .zip(class_mask.iter())
                        .filter(|(row, &mask)| mask && row[feature2] == val2)
                        .count() as f64;

                    let class_count = class_mask.iter().filter(|&&mask| mask).count() as f64;

                    if count_joint > 0.0
                        && count_val1 > 0.0
                        && count_val2 > 0.0
                        && class_count > 0.0
                    {
                        let p_joint = count_joint / class_count;
                        let p_val1 = count_val1 / class_count;
                        let p_val2 = count_val2 / class_count;

                        mi += class_prob * p_joint * (p_joint / (p_val1 * p_val2)).ln();
                    }
                }
            }
        }

        Ok(mi.max(0.0))
    }

    fn estimate_conditional_probabilities(
        &mut self,
        X: &Array2<i32>,
        y: &Array1<i32>,
    ) -> Result<(), TANError> {
        // Determine maximum values for each feature
        let max_values: Vec<usize> = (0..self.n_features)
            .map(|j| X.column(j).iter().max().unwrap_or(&0) + 1)
            .map(|x| x as usize)
            .collect();

        let max_val = max_values.iter().max().copied().unwrap_or(5);

        // Initialize probability arrays
        self.feature_log_prob =
            Array3::from_elem((self.n_features, max_val, max_val), f64::NEG_INFINITY);
        self.conditional_log_prob = Array3::from_elem(
            (self.n_features, max_val, self.n_classes),
            f64::NEG_INFINITY,
        );

        // Estimate probabilities for root features (no parents)
        for feature_idx in 0..self.n_features {
            if self.dependency_tree.get_parent(feature_idx).is_none() {
                for (class_idx, &class) in self.classes.iter().enumerate() {
                    let class_mask: Vec<bool> = y.iter().map(|&label| label == class).collect();
                    let class_count = class_mask.iter().filter(|&&mask| mask).count() as f64;

                    for val in 0..max_values[feature_idx] {
                        let count = X
                            .axis_iter(scirs2_core::ndarray::Axis(0))
                            .zip(class_mask.iter())
                            .filter(|(row, &mask)| mask && row[feature_idx] == val as i32)
                            .count() as f64;

                        let prob = (count + self.config.smoothing)
                            / (class_count
                                + self.config.smoothing * max_values[feature_idx] as f64);
                        self.feature_log_prob[[feature_idx, 0, val]] = prob.ln();
                    }
                }
            }
        }

        // Estimate conditional probabilities for dependent features
        for feature_idx in 0..self.n_features {
            if let Some(parent_idx) = self.dependency_tree.get_parent(feature_idx) {
                for (class_idx, &class) in self.classes.iter().enumerate() {
                    let class_mask: Vec<bool> = y.iter().map(|&label| label == class).collect();

                    for parent_val in 0..max_values[parent_idx] {
                        let parent_count = X
                            .axis_iter(scirs2_core::ndarray::Axis(0))
                            .zip(class_mask.iter())
                            .filter(|(row, &mask)| mask && row[parent_idx] == parent_val as i32)
                            .count() as f64;

                        if parent_count > 0.0 {
                            for feature_val in 0..max_values[feature_idx] {
                                let joint_count =
                                    X.axis_iter(scirs2_core::ndarray::Axis(0))
                                        .zip(class_mask.iter())
                                        .filter(|(row, &mask)| {
                                            mask && row[parent_idx] == parent_val as i32
                                                && row[feature_idx] == feature_val as i32
                                        })
                                        .count() as f64;

                                let prob = (joint_count + self.config.smoothing)
                                    / (parent_count
                                        + self.config.smoothing * max_values[feature_idx] as f64);
                                self.conditional_log_prob[[feature_idx, parent_val, class_idx]] =
                                    prob.ln();
                            }
                        }
                    }
                }
            }
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
    #[allow(non_snake_case)]
    fn test_tan_basic_functionality() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let mut tan = TreeAugmentedNB::new(TANConfig::default());
        assert!(tan.fit(&X, &y).is_ok());

        let predictions = tan.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_tan_predict_proba() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let mut tan = TreeAugmentedNB::new(TANConfig::default());
        tan.fit(&X, &y).unwrap();

        let log_proba = tan.predict_log_proba(&X).unwrap();
        assert_eq!(log_proba.shape(), &[4, 2]);

        // Check that log probabilities are reasonable (not all negative infinity)
        for row in log_proba.axis_iter(scirs2_core::ndarray::Axis(0)) {
            let has_finite_values = row.iter().any(|&x| x.is_finite());
            assert!(has_finite_values, "All log probabilities are infinite");
        }
    }

    #[test]
    fn test_dependency_tree() {
        let mut tree = DependencyTree::new();
        tree.add_edge(TreeEdge {
            parent: 0,
            child: 1,
            weight: 0.5,
        });
        tree.add_edge(TreeEdge {
            parent: 1,
            child: 2,
            weight: 0.3,
        });

        assert_eq!(tree.get_parent(1), Some(0));
        assert_eq!(tree.get_parent(2), Some(1));
        assert_eq!(tree.get_parent(0), None);

        assert_eq!(tree.get_children(0), vec![1]);
        assert_eq!(tree.get_children(1), vec![2]);
        assert_eq!(tree.get_children(2), Vec::<usize>::new());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empty_dataset() {
        let X = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let y = Array1::from_vec(vec![]);

        let mut tan = TreeAugmentedNB::new(TANConfig::default());
        assert!(matches!(tan.fit(&X, &y), Err(TANError::EmptyDataset)));
    }
}

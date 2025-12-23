//! Bayesian Network Augmented Naive Bayes (BAN) implementation
//!
//! BAN extends Naive Bayes by learning a Bayesian network structure that can
//! capture more complex dependencies between features than tree-based approaches.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BANError {
    #[error("Features and targets have different number of samples")]
    DimensionMismatch,
    #[error("Empty dataset provided")]
    EmptyDataset,
    #[error("Invalid feature index: {0}")]
    InvalidFeatureIndex(usize),
    #[error("Cyclic dependency detected in network structure")]
    CyclicDependency,
    #[error("Maximum parents exceeded for feature {0}")]
    MaxParentsExceeded(usize),
    #[error("Numerical computation error: {0}")]
    NumericalError(String),
}

/// Edge in the Bayesian network representing conditional dependencies
#[derive(Debug, Clone)]
pub struct NetworkEdge {
    pub parent: usize,
    pub child: usize,
    pub weight: f64,
    pub conditional_mi: f64,
}

/// Bayesian network structure for feature dependencies
#[derive(Debug, Clone)]
pub struct BayesianNetwork {
    edges: Vec<NetworkEdge>,
    adjacency: HashMap<usize, Vec<usize>>,
    parents: HashMap<usize, Vec<usize>>,
    children: HashMap<usize, Vec<usize>>,
    n_features: usize,
}

impl BayesianNetwork {
    pub fn new(n_features: usize) -> Self {
        Self {
            edges: Vec::new(),
            adjacency: HashMap::new(),
            parents: HashMap::new(),
            children: HashMap::new(),
            n_features,
        }
    }

    pub fn add_edge(&mut self, edge: NetworkEdge) -> Result<(), BANError> {
        // Check for cycles before adding edge
        if self.would_create_cycle(edge.parent, edge.child) {
            return Err(BANError::CyclicDependency);
        }

        self.adjacency
            .entry(edge.parent)
            .or_default()
            .push(edge.child);

        self.parents
            .entry(edge.child)
            .or_default()
            .push(edge.parent);

        self.children
            .entry(edge.parent)
            .or_default()
            .push(edge.child);

        self.edges.push(edge);
        Ok(())
    }

    pub fn get_parents(&self, node: usize) -> Vec<usize> {
        self.parents.get(&node).cloned().unwrap_or_default()
    }

    pub fn get_children(&self, node: usize) -> Vec<usize> {
        self.children.get(&node).cloned().unwrap_or_default()
    }

    fn would_create_cycle(&self, from: usize, to: usize) -> bool {
        // BFS to check if there's a path from 'to' to 'from'
        let mut visited = vec![false; self.n_features];
        let mut queue = VecDeque::new();
        queue.push_back(to);
        visited[to] = true;

        while let Some(current) = queue.pop_front() {
            if current == from {
                return true;
            }

            for &child in self.get_children(current).iter() {
                if !visited[child] {
                    visited[child] = true;
                    queue.push_back(child);
                }
            }
        }

        false
    }

    pub fn topological_sort(&self) -> Result<Vec<usize>, BANError> {
        let mut in_degree = vec![0; self.n_features];
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Calculate in-degrees
        for (i, degree) in in_degree.iter_mut().enumerate() {
            *degree = self.get_parents(i).len();
            if *degree == 0 {
                queue.push_back(i);
            }
        }

        while let Some(node) = queue.pop_front() {
            result.push(node);

            for &child in self.get_children(node).iter() {
                in_degree[child] -= 1;
                if in_degree[child] == 0 {
                    queue.push_back(child);
                }
            }
        }

        if result.len() != self.n_features {
            return Err(BANError::CyclicDependency);
        }

        Ok(result)
    }
}

/// Configuration for Bayesian Network Augmented Naive Bayes
#[derive(Debug, Clone)]
pub struct BANConfig {
    pub smoothing: f64,
    pub max_parents: usize,
    pub structure_learning_method: StructureLearningMethod,
    pub score_threshold: f64,
    pub use_k2_algorithm: bool,
    pub max_iterations: usize,
}

#[derive(Debug, Clone)]
pub enum StructureLearningMethod {
    /// GreedyHillClimbing
    GreedyHillClimbing,
    /// K2Algorithm
    K2Algorithm,
    /// ConstraintBased
    ConstraintBased,
    /// HybridApproach
    HybridApproach,
}

impl Default for BANConfig {
    fn default() -> Self {
        Self {
            smoothing: 1.0,
            max_parents: 3,
            structure_learning_method: StructureLearningMethod::GreedyHillClimbing,
            score_threshold: 0.01,
            use_k2_algorithm: false,
            max_iterations: 100,
        }
    }
}

/// Bayesian Network Augmented Naive Bayes classifier
pub struct BayesianNetworkAugmentedNB {
    config: BANConfig,
    classes: Array1<i32>,
    class_log_prior: Array1<f64>,
    network: BayesianNetwork,
    conditional_prob_tables: HashMap<usize, Array3<f64>>, // [feature][parent_config][feature_value]
    n_features: usize,
    n_classes: usize,
    feature_cardinalities: Vec<usize>,
}

impl BayesianNetworkAugmentedNB {
    pub fn new(config: BANConfig) -> Self {
        Self {
            config,
            classes: Array1::default((0,)),
            class_log_prior: Array1::default((0,)),
            network: BayesianNetwork::new(0),
            conditional_prob_tables: HashMap::new(),
            n_features: 0,
            n_classes: 0,
            feature_cardinalities: Vec::new(),
        }
    }

    /// Fit the BAN classifier to training data
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<i32>) -> Result<(), BANError> {
        if X.nrows() != y.len() {
            return Err(BANError::DimensionMismatch);
        }
        if X.nrows() == 0 {
            return Err(BANError::EmptyDataset);
        }

        self.n_features = X.ncols();
        self.classes = self.get_unique_classes(y);
        self.n_classes = self.classes.len();
        self.network = BayesianNetwork::new(self.n_features);

        // Compute class priors
        self.compute_class_priors(y)?;

        // Discretize continuous features
        let X_discrete = self.discretize_features(X)?;

        // Compute feature cardinalities
        self.compute_feature_cardinalities(&X_discrete);

        // Learn network structure
        self.learn_network_structure(&X_discrete, y)?;

        // Estimate conditional probability tables
        self.estimate_conditional_probabilities(&X_discrete, y)?;

        Ok(())
    }

    /// Predict class labels for test samples
    pub fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>, BANError> {
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
    pub fn predict_log_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>, BANError> {
        if X.ncols() != self.n_features {
            return Err(BANError::DimensionMismatch);
        }

        let X_discrete = self.discretize_features(X)?;
        let mut log_proba = Array2::zeros((X.nrows(), self.n_classes));

        // Get topological ordering for feature evaluation
        let topo_order = self.network.topological_sort()?;

        for (sample_idx, sample) in X_discrete
            .axis_iter(scirs2_core::ndarray::Axis(0))
            .enumerate()
        {
            for (class_idx, _) in self.classes.iter().enumerate() {
                let mut log_prob = self.class_log_prior[class_idx];

                // Process features in topological order
                for &feature_idx in topo_order.iter() {
                    let feature_value = sample[feature_idx] as usize;
                    let parents = self.network.get_parents(feature_idx);

                    if parents.is_empty() {
                        // Root feature - use class-conditional probability
                        if let Some(cpt) = self.conditional_prob_tables.get(&feature_idx) {
                            if feature_value < cpt.shape()[2] {
                                log_prob += cpt[[0, class_idx, feature_value]];
                            }
                        }
                    } else {
                        // Feature with parents - use conditional probability
                        let parent_config = self.encode_parent_configuration(&sample, &parents);

                        if let Some(cpt) = self.conditional_prob_tables.get(&feature_idx) {
                            if parent_config < cpt.shape()[0] && feature_value < cpt.shape()[2] {
                                log_prob += cpt[[parent_config, class_idx, feature_value]];
                            }
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

    fn compute_class_priors(&mut self, y: &Array1<i32>) -> Result<(), BANError> {
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

    fn discretize_features(&self, X: &Array2<f64>) -> Result<Array2<i32>, BANError> {
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

    fn compute_feature_cardinalities(&mut self, X: &Array2<i32>) {
        self.feature_cardinalities = (0..self.n_features)
            .map(|j| X.column(j).iter().max().unwrap_or(&0) + 1)
            .map(|x| x as usize)
            .collect();
    }

    fn learn_network_structure(
        &mut self,
        X: &Array2<i32>,
        y: &Array1<i32>,
    ) -> Result<(), BANError> {
        match self.config.structure_learning_method {
            StructureLearningMethod::GreedyHillClimbing => self.greedy_hill_climbing(X, y),
            StructureLearningMethod::K2Algorithm => self.k2_algorithm(X, y),
            _ => {
                // Default to greedy hill climbing
                self.greedy_hill_climbing(X, y)
            }
        }
    }

    fn greedy_hill_climbing(&mut self, X: &Array2<i32>, y: &Array1<i32>) -> Result<(), BANError> {
        let mut current_score = self.compute_network_score(X, y)?;
        let mut improved = true;

        for _ in 0..self.config.max_iterations {
            if !improved {
                break;
            }
            improved = false;

            // Try adding edges
            for parent in 0..self.n_features {
                for child in 0..self.n_features {
                    if parent == child {
                        continue;
                    }

                    let current_parents = self.network.get_parents(child);
                    if current_parents.len() >= self.config.max_parents {
                        continue;
                    }

                    if current_parents.contains(&parent) {
                        continue;
                    }

                    // Try adding this edge
                    let edge = NetworkEdge {
                        parent,
                        child,
                        weight: 0.0,
                        conditional_mi: 0.0,
                    };

                    if self.network.add_edge(edge).is_ok() {
                        let new_score = self.compute_network_score(X, y)?;

                        if new_score > current_score + self.config.score_threshold {
                            current_score = new_score;
                            improved = true;
                        } else {
                            // Remove the edge if it doesn't improve the score
                            self.remove_last_edge();
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn k2_algorithm(&mut self, X: &Array2<i32>, y: &Array1<i32>) -> Result<(), BANError> {
        // Simplified K2 algorithm implementation
        for child in 0..self.n_features {
            let mut current_score = self.compute_feature_score(X, y, child, &[])?;
            let mut current_parents = Vec::new();

            for _ in 0..self.config.max_parents {
                let mut best_parent = None;
                let mut best_score = current_score;

                for parent in 0..self.n_features {
                    if parent == child || current_parents.contains(&parent) {
                        continue;
                    }

                    let mut test_parents = current_parents.clone();
                    test_parents.push(parent);

                    let score = self.compute_feature_score(X, y, child, &test_parents)?;
                    if score > best_score {
                        best_score = score;
                        best_parent = Some(parent);
                    }
                }

                if let Some(parent) = best_parent {
                    current_parents.push(parent);
                    current_score = best_score;

                    let edge = NetworkEdge {
                        parent,
                        child,
                        weight: best_score - current_score,
                        conditional_mi: 0.0,
                    };
                    self.network.add_edge(edge)?;
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    fn compute_network_score(&self, X: &Array2<i32>, y: &Array1<i32>) -> Result<f64, BANError> {
        let mut total_score = 0.0;

        for feature in 0..self.n_features {
            let parents = self.network.get_parents(feature);
            total_score += self.compute_feature_score(X, y, feature, &parents)?;
        }

        Ok(total_score)
    }

    fn compute_feature_score(
        &self,
        X: &Array2<i32>,
        y: &Array1<i32>,
        feature: usize,
        parents: &[usize],
    ) -> Result<f64, BANError> {
        // Compute BDeu score (Bayesian Dirichlet equivalent uniform)
        let alpha = 1.0; // Equivalent sample size
        let feature_cardinality = self.feature_cardinalities[feature] as f64;
        let parent_configs = if parents.is_empty() {
            1
        } else {
            parents
                .iter()
                .map(|&p| self.feature_cardinalities[p])
                .product()
        };

        let mut score = 0.0;

        for class_idx in 0..self.n_classes {
            let class_mask: Vec<bool> = y
                .iter()
                .map(|&label| label == self.classes[class_idx])
                .collect();

            for parent_config in 0..parent_configs {
                let parent_mask = if parents.is_empty() {
                    vec![true; X.nrows()]
                } else {
                    self.create_parent_mask(X, parents, parent_config)
                };

                let combined_mask: Vec<bool> = class_mask
                    .iter()
                    .zip(parent_mask.iter())
                    .map(|(&c, &p)| c && p)
                    .collect();

                let n_ijk = combined_mask.iter().filter(|&&mask| mask).count() as f64;

                for feature_value in 0..self.feature_cardinalities[feature] {
                    let feature_mask: Vec<bool> = X
                        .column(feature)
                        .iter()
                        .map(|&val| val == feature_value as i32)
                        .collect();

                    let count = combined_mask
                        .iter()
                        .zip(feature_mask.iter())
                        .filter(|(&c, &f)| c && f)
                        .count() as f64;

                    let alpha_ijk = alpha / (feature_cardinality * parent_configs as f64);

                    // Log gamma function approximation
                    score += self.log_gamma(count + alpha_ijk) - self.log_gamma(alpha_ijk);
                }

                let alpha_ij = alpha / parent_configs as f64;
                score += self.log_gamma(alpha_ij) - self.log_gamma(n_ijk + alpha_ij);
            }
        }

        Ok(score)
    }

    fn create_parent_mask(&self, X: &Array2<i32>, parents: &[usize], config: usize) -> Vec<bool> {
        let mut masks = Vec::new();
        let mut remaining_config = config;

        for &parent in parents.iter().rev() {
            let cardinality = self.feature_cardinalities[parent];
            let value = remaining_config % cardinality;
            remaining_config /= cardinality;

            let mask: Vec<bool> = X
                .column(parent)
                .iter()
                .map(|&val| val == value as i32)
                .collect();
            masks.push(mask);
        }

        // Combine all parent masks with AND operation
        (0..X.nrows())
            .map(|i| masks.iter().all(|mask| mask[i]))
            .collect()
    }

    fn log_gamma(&self, x: f64) -> f64 {
        // Stirling's approximation for log gamma function
        if x < 1.0 {
            return 0.0;
        }
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    fn remove_last_edge(&mut self) {
        if let Some(edge) = self.network.edges.pop() {
            // Remove from adjacency lists
            if let Some(children) = self.network.adjacency.get_mut(&edge.parent) {
                children.retain(|&x| x != edge.child);
            }
            if let Some(parents) = self.network.parents.get_mut(&edge.child) {
                parents.retain(|&x| x != edge.parent);
            }
            if let Some(children) = self.network.children.get_mut(&edge.parent) {
                children.retain(|&x| x != edge.child);
            }
        }
    }

    fn estimate_conditional_probabilities(
        &mut self,
        X: &Array2<i32>,
        y: &Array1<i32>,
    ) -> Result<(), BANError> {
        for feature in 0..self.n_features {
            let parents = self.network.get_parents(feature);
            let parent_configs = if parents.is_empty() {
                1
            } else {
                parents
                    .iter()
                    .map(|&p| self.feature_cardinalities[p])
                    .product()
            };

            let feature_cardinality = self.feature_cardinalities[feature];
            let mut cpt = Array3::from_elem(
                (parent_configs, self.n_classes, feature_cardinality),
                f64::NEG_INFINITY,
            );

            for class_idx in 0..self.n_classes {
                let class_mask: Vec<bool> = y
                    .iter()
                    .map(|&label| label == self.classes[class_idx])
                    .collect();

                for parent_config in 0..parent_configs {
                    let parent_mask = if parents.is_empty() {
                        vec![true; X.nrows()]
                    } else {
                        self.create_parent_mask(X, &parents, parent_config)
                    };

                    let combined_mask: Vec<bool> = class_mask
                        .iter()
                        .zip(parent_mask.iter())
                        .map(|(&c, &p)| c && p)
                        .collect();

                    let total_count = combined_mask.iter().filter(|&&mask| mask).count() as f64;

                    if total_count > 0.0 {
                        for feature_value in 0..feature_cardinality {
                            let feature_mask: Vec<bool> = X
                                .column(feature)
                                .iter()
                                .map(|&val| val == feature_value as i32)
                                .collect();

                            let count = combined_mask
                                .iter()
                                .zip(feature_mask.iter())
                                .filter(|(&c, &f)| c && f)
                                .count() as f64;

                            let prob = (count + self.config.smoothing)
                                / (total_count
                                    + self.config.smoothing * feature_cardinality as f64);
                            cpt[[parent_config, class_idx, feature_value]] = prob.ln();
                        }
                    }
                }
            }

            self.conditional_prob_tables.insert(feature, cpt);
        }

        Ok(())
    }

    fn encode_parent_configuration(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<i32>,
        parents: &[usize],
    ) -> usize {
        let mut config = 0;
        let mut multiplier = 1;

        for &parent in parents.iter().rev() {
            config += sample[parent] as usize * multiplier;
            multiplier *= self.feature_cardinalities[parent];
        }

        config
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    #[allow(non_snake_case)]
    fn test_ban_basic_functionality() {
        let X = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 1.0, 5.0, 6.0, 2.0, 6.0,
                7.0, 3.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let mut ban = BayesianNetworkAugmentedNB::new(BANConfig::default());
        assert!(ban.fit(&X, &y).is_ok());

        let predictions = ban.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_bayesian_network_cycle_detection() {
        let mut network = BayesianNetwork::new(3);

        let edge1 = NetworkEdge {
            parent: 0,
            child: 1,
            weight: 0.5,
            conditional_mi: 0.0,
        };
        let edge2 = NetworkEdge {
            parent: 1,
            child: 2,
            weight: 0.3,
            conditional_mi: 0.0,
        };
        let edge3 = NetworkEdge {
            parent: 2,
            child: 0,
            weight: 0.1,
            conditional_mi: 0.0,
        };

        assert!(network.add_edge(edge1).is_ok());
        assert!(network.add_edge(edge2).is_ok());
        assert!(matches!(
            network.add_edge(edge3),
            Err(BANError::CyclicDependency)
        ));
    }

    #[test]
    fn test_topological_sort() {
        let mut network = BayesianNetwork::new(4);

        network
            .add_edge(NetworkEdge {
                parent: 0,
                child: 1,
                weight: 0.5,
                conditional_mi: 0.0,
            })
            .unwrap();
        network
            .add_edge(NetworkEdge {
                parent: 0,
                child: 2,
                weight: 0.3,
                conditional_mi: 0.0,
            })
            .unwrap();
        network
            .add_edge(NetworkEdge {
                parent: 1,
                child: 3,
                weight: 0.2,
                conditional_mi: 0.0,
            })
            .unwrap();

        let topo_order = network.topological_sort().unwrap();
        assert_eq!(topo_order.len(), 4);

        // Node 0 should come before nodes 1 and 2
        let pos_0 = topo_order.iter().position(|&x| x == 0).unwrap();
        let pos_1 = topo_order.iter().position(|&x| x == 1).unwrap();
        let pos_2 = topo_order.iter().position(|&x| x == 2).unwrap();
        let pos_3 = topo_order.iter().position(|&x| x == 3).unwrap();

        assert!(pos_0 < pos_1);
        assert!(pos_0 < pos_2);
        assert!(pos_1 < pos_3);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empty_dataset() {
        let X = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let y = Array1::from_vec(vec![]);

        let mut ban = BayesianNetworkAugmentedNB::new(BANConfig::default());
        assert!(matches!(ban.fit(&X, &y), Err(BANError::EmptyDataset)));
    }
}

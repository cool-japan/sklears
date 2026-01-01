//! Adversarial graph learning for robust semi-supervised scenarios
//!
//! This module provides adversarial graph learning algorithms that can handle
//! adversarial perturbations, malicious nodes, and robust graph construction
//! in adversarial environments.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rand_prelude::*;
use scirs2_core::random::Random;
use sklears_core::error::SklearsError;

/// Adversarial graph learning for robust semi-supervised learning
#[derive(Clone)]
pub struct AdversarialGraphLearning {
    /// Number of neighbors for graph construction
    pub k_neighbors: usize,
    /// Robustness parameter for adversarial defense
    pub robustness_lambda: f64,
    /// Maximum perturbation magnitude allowed
    pub max_perturbation: f64,
    /// Number of adversarial iterations
    pub adversarial_steps: usize,
    /// Learning rate for adversarial updates
    pub adversarial_lr: f64,
    /// Defense strategy: "spectral", "robust_pca", "consensus", "adaptive"
    pub defense_strategy: String,
    /// Consensus threshold for agreement-based defense
    pub consensus_threshold: f64,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Adversarial attack configuration
#[derive(Clone, Debug)]
pub struct AdversarialAttack {
    /// Attack type: "node_injection", "edge_manipulation", "feature_perturbation"
    pub attack_type: String,
    /// Attack strength (0.0 to 1.0)
    pub attack_strength: f64,
    /// Number of nodes to attack
    pub target_nodes: usize,
    /// Perturbation strategy: "random", "gradient", "targeted"
    pub perturbation_strategy: String,
}

impl AdversarialGraphLearning {
    /// Create a new adversarial graph learning instance
    pub fn new() -> Self {
        Self {
            k_neighbors: 5,
            robustness_lambda: 0.1,
            max_perturbation: 0.1,
            adversarial_steps: 10,
            adversarial_lr: 0.01,
            defense_strategy: "spectral".to_string(),
            consensus_threshold: 0.7,
            max_iter: 100,
            tolerance: 1e-6,
            random_state: None,
        }
    }

    /// Set the number of neighbors for graph construction
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the robustness parameter
    pub fn robustness_lambda(mut self, lambda: f64) -> Self {
        self.robustness_lambda = lambda;
        self
    }

    /// Set the maximum perturbation magnitude
    pub fn max_perturbation(mut self, max_pert: f64) -> Self {
        self.max_perturbation = max_pert;
        self
    }

    /// Set the number of adversarial steps
    pub fn adversarial_steps(mut self, steps: usize) -> Self {
        self.adversarial_steps = steps;
        self
    }

    /// Set the adversarial learning rate
    pub fn adversarial_lr(mut self, lr: f64) -> Self {
        self.adversarial_lr = lr;
        self
    }

    /// Set the defense strategy
    pub fn defense_strategy(mut self, strategy: String) -> Self {
        self.defense_strategy = strategy;
        self
    }

    /// Set the consensus threshold
    pub fn consensus_threshold(mut self, threshold: f64) -> Self {
        self.consensus_threshold = threshold;
        self
    }

    /// Set the maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Learn a robust graph in the presence of adversarial perturbations
    pub fn fit_robust(
        &self,
        features: ArrayView2<f64>,
        labels: Option<ArrayView1<i32>>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = features.nrows();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        match self.defense_strategy.as_str() {
            "spectral" => self.spectral_defense(features, labels),
            "robust_pca" => self.robust_pca_defense(features, labels),
            "consensus" => self.consensus_defense(features, labels),
            "adaptive" => self.adaptive_defense(features, labels),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown defense strategy: {}",
                self.defense_strategy
            ))),
        }
    }

    /// Spectral defense using eigenvalue decomposition for robustness
    fn spectral_defense(
        &self,
        features: ArrayView2<f64>,
        _labels: Option<ArrayView1<i32>>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = features.nrows();
        let mut adjacency = Array2::zeros((n_samples, n_samples));

        // Build initial graph
        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = self.compute_robust_distance(features.row(i), features.row(j));
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(neighbor, dist) in distances.iter().take(self.k_neighbors) {
                let weight = (-dist / (2.0 * self.max_perturbation.powi(2))).exp();
                adjacency[[i, neighbor]] = weight;
                adjacency[[neighbor, i]] = weight;
            }
        }

        // Apply spectral regularization for robustness
        self.apply_spectral_regularization(&mut adjacency)?;

        Ok(adjacency)
    }

    /// Robust PCA defense using outlier-resistant principal components
    fn robust_pca_defense(
        &self,
        features: ArrayView2<f64>,
        _labels: Option<ArrayView1<i32>>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        // Estimate robust mean and covariance
        let robust_mean = self.compute_robust_mean(features)?;
        let robust_cov = self.compute_robust_covariance(features, &robust_mean)?;

        // Project data using robust PCA
        let robust_features = self.robust_pca_projection(features, &robust_mean, &robust_cov)?;

        // Build graph using robust features
        let mut adjacency = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = self.mahalanobis_distance(
                        robust_features.row(i),
                        robust_features.row(j),
                        &robust_cov,
                    )?;
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(neighbor, dist) in distances.iter().take(self.k_neighbors) {
                let weight = (-dist).exp();
                adjacency[[i, neighbor]] = weight;
                adjacency[[neighbor, i]] = weight;
            }
        }

        Ok(adjacency)
    }

    /// Consensus defense using multiple graph constructions
    fn consensus_defense(
        &self,
        features: ArrayView2<f64>,
        labels: Option<ArrayView1<i32>>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = features.nrows();
        let num_graphs = 5; // Number of consensus graphs

        let mut consensus_adjacency = Array2::zeros((n_samples, n_samples));
        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        // Generate multiple graphs with different perturbations
        for graph_idx in 0..num_graphs {
            let mut perturbed_features = features.to_owned();

            // Add small random perturbations
            for i in 0..n_samples {
                for j in 0..features.ncols() {
                    let noise = rng.random_range(-self.max_perturbation..self.max_perturbation);
                    perturbed_features[[i, j]] += noise;
                }
            }

            // Build graph from perturbed features
            let graph = self.build_knn_graph(perturbed_features.view())?;

            // Add to consensus
            consensus_adjacency = consensus_adjacency + graph;
        }

        // Normalize by number of graphs
        consensus_adjacency /= num_graphs as f64;

        // Apply consensus threshold
        consensus_adjacency.mapv_inplace(|x| {
            if x >= self.consensus_threshold {
                x
            } else {
                0.0
            }
        });

        Ok(consensus_adjacency)
    }

    /// Adaptive defense that combines multiple strategies
    fn adaptive_defense(
        &self,
        features: ArrayView2<f64>,
        labels: Option<ArrayView1<i32>>,
    ) -> Result<Array2<f64>, SklearsError> {
        // Combine spectral and consensus defenses
        let spectral_graph = self.spectral_defense(features, labels)?;
        let consensus_graph = self.consensus_defense(features, labels)?;

        let n_samples = features.nrows();
        let mut adaptive_graph = Array2::zeros((n_samples, n_samples));

        // Adaptive weighting based on local graph properties
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let spectral_weight = spectral_graph[[i, j]];
                    let consensus_weight = consensus_graph[[i, j]];

                    // Adaptive combination based on weight agreement
                    let agreement = (spectral_weight - consensus_weight).abs();
                    let confidence = (-agreement / self.max_perturbation).exp();

                    adaptive_graph[[i, j]] =
                        confidence * spectral_weight + (1.0 - confidence) * consensus_weight;
                }
            }
        }

        Ok(adaptive_graph)
    }

    /// Apply adversarial attack to test robustness
    pub fn apply_attack(
        &self,
        features: ArrayView2<f64>,
        attack: &AdversarialAttack,
    ) -> Result<Array2<f64>, SklearsError> {
        let mut attacked_features = features.to_owned();
        let n_samples = features.nrows();

        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        match attack.attack_type.as_str() {
            "feature_perturbation" => {
                let num_target_nodes = attack.target_nodes.min(n_samples);
                let target_indices: Vec<usize> = (0..n_samples)
                    .choose_multiple(&mut rng, num_target_nodes)
                    .into_iter()
                    .collect();

                for &node_idx in &target_indices {
                    for feature_idx in 0..features.ncols() {
                        let perturbation = match attack.perturbation_strategy.as_str() {
                            "random" => {
                                rng.random_range(-attack.attack_strength..attack.attack_strength)
                            }
                            "gradient" => {
                                self.compute_gradient_perturbation(features, node_idx, feature_idx)?
                                    * attack.attack_strength
                            }
                            "targeted" => {
                                self.compute_targeted_perturbation(features, node_idx, feature_idx)?
                                    * attack.attack_strength
                            }
                            _ => rng.random_range(-attack.attack_strength..attack.attack_strength),
                        };

                        attacked_features[[node_idx, feature_idx]] += perturbation;
                    }
                }
            }
            "node_injection" => {
                // This would require extending the feature matrix
                return Err(SklearsError::InvalidInput(
                    "Node injection not implemented in this context".to_string(),
                ));
            }
            "edge_manipulation" => {
                // This would be applied to the adjacency matrix after construction
                return Err(SklearsError::InvalidInput(
                    "Edge manipulation should be applied to adjacency matrix".to_string(),
                ));
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown attack type: {}",
                    attack.attack_type
                )));
            }
        }

        Ok(attacked_features)
    }

    /// Compute robust distance metric resistant to outliers
    fn compute_robust_distance(&self, feat1: ArrayView1<f64>, feat2: ArrayView1<f64>) -> f64 {
        // Use Huber loss-based distance for robustness
        let delta = self.max_perturbation;

        feat1
            .iter()
            .zip(feat2.iter())
            .map(|(&a, &b)| {
                let diff = (a - b).abs();
                if diff <= delta {
                    0.5 * diff * diff
                } else {
                    delta * (diff - 0.5 * delta)
                }
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Apply spectral regularization to improve robustness
    fn apply_spectral_regularization(
        &self,
        adjacency: &mut Array2<f64>,
    ) -> Result<(), SklearsError> {
        let n = adjacency.nrows();

        // Compute degree matrix
        let mut degree = Array1::zeros(n);
        for i in 0..n {
            degree[i] = adjacency.row(i).sum();
        }

        // Apply regularization to improve spectral properties
        for i in 0..n {
            for j in 0..n {
                if i != j && adjacency[[i, j]] > 0.0 {
                    // Regularize edge weights based on degree difference
                    let degree_penalty =
                        (degree[i] - degree[j]).abs() / (degree[i] + degree[j] + 1e-8);
                    adjacency[[i, j]] *= 1.0 - self.robustness_lambda * degree_penalty;
                }
            }
        }

        Ok(())
    }

    /// Compute robust mean using median
    fn compute_robust_mean(&self, features: ArrayView2<f64>) -> Result<Array1<f64>, SklearsError> {
        let n_features = features.ncols();
        let mut robust_mean = Array1::zeros(n_features);

        for j in 0..n_features {
            let mut column: Vec<f64> = features.column(j).to_vec();
            column.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median_idx = column.len() / 2;
            robust_mean[j] = if column.len() % 2 == 0 {
                (column[median_idx - 1] + column[median_idx]) / 2.0
            } else {
                column[median_idx]
            };
        }

        Ok(robust_mean)
    }

    /// Compute robust covariance using MAD (Median Absolute Deviation)
    fn compute_robust_covariance(
        &self,
        features: ArrayView2<f64>,
        robust_mean: &Array1<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_features = features.ncols();
        let mut robust_cov = Array2::eye(n_features);

        for j in 0..n_features {
            let mut deviations: Vec<f64> = features
                .column(j)
                .iter()
                .map(|&x| (x - robust_mean[j]).abs())
                .collect();

            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = deviations[deviations.len() / 2] * 1.4826; // Scale factor for normal distribution

            robust_cov[[j, j]] = mad * mad;
        }

        Ok(robust_cov)
    }

    /// Project features using robust PCA
    fn robust_pca_projection(
        &self,
        features: ArrayView2<f64>,
        robust_mean: &Array1<f64>,
        robust_cov: &Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        // Simple robust projection (in practice, you'd use proper robust PCA)
        let mut projected = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                projected[[i, j]] = (features[[i, j]] - robust_mean[j]) / robust_cov[[j, j]].sqrt();
            }
        }

        Ok(projected)
    }

    /// Compute Mahalanobis distance
    fn mahalanobis_distance(
        &self,
        feat1: ArrayView1<f64>,
        feat2: ArrayView1<f64>,
        cov: &Array2<f64>,
    ) -> Result<f64, SklearsError> {
        let diff: Array1<f64> = &feat1.to_owned() - &feat2.to_owned();

        // Simplified Mahalanobis distance (assuming diagonal covariance)
        let mut distance = 0.0;
        for (i, &d) in diff.iter().enumerate() {
            distance += d * d / cov[[i, i]];
        }

        Ok(distance.sqrt())
    }

    /// Build k-NN graph from features
    fn build_knn_graph(&self, features: ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n_samples = features.nrows();
        let mut adjacency = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = self.compute_robust_distance(features.row(i), features.row(j));
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(neighbor, dist) in distances.iter().take(self.k_neighbors) {
                let weight = (-dist).exp();
                adjacency[[i, neighbor]] = weight;
                adjacency[[neighbor, i]] = weight;
            }
        }

        Ok(adjacency)
    }

    /// Compute gradient-based perturbation (simplified)
    fn compute_gradient_perturbation(
        &self,
        _features: ArrayView2<f64>,
        _node_idx: usize,
        _feature_idx: usize,
    ) -> Result<f64, SklearsError> {
        // Simplified gradient computation
        // In practice, this would involve computing gradients of the loss function
        Ok(0.1) // Placeholder
    }

    /// Compute targeted perturbation
    fn compute_targeted_perturbation(
        &self,
        _features: ArrayView2<f64>,
        _node_idx: usize,
        _feature_idx: usize,
    ) -> Result<f64, SklearsError> {
        // Simplified targeted perturbation
        // In practice, this would target specific nodes or classes
        Ok(0.05) // Placeholder
    }

    /// Evaluate robustness against attack
    pub fn evaluate_robustness(
        &self,
        original_graph: &Array2<f64>,
        attacked_graph: &Array2<f64>,
    ) -> Result<f64, SklearsError> {
        if original_graph.dim() != attacked_graph.dim() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", original_graph.dim()),
                actual: format!("{:?}", attacked_graph.dim()),
            });
        }

        // Compute Frobenius norm of the difference
        let diff = original_graph - attacked_graph;
        let frobenius_norm = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();

        // Normalize by original graph norm
        let original_norm = original_graph.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if original_norm > 0.0 {
            Ok(frobenius_norm / original_norm)
        } else {
            Ok(0.0)
        }
    }
}

impl Default for AdversarialGraphLearning {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::array;

    #[test]
    fn test_adversarial_graph_learning_spectral() {
        let agl = AdversarialGraphLearning::new()
            .k_neighbors(2)
            .defense_strategy("spectral".to_string())
            .robustness_lambda(0.1);

        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let result = agl.fit_robust(features.view(), None);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (3, 3));

        // Check that diagonal is zero
        for i in 0..3 {
            assert_eq!(graph[[i, i]], 0.0);
        }
    }

    #[test]
    fn test_adversarial_graph_learning_consensus() {
        let agl = AdversarialGraphLearning::new()
            .k_neighbors(2)
            .defense_strategy("consensus".to_string())
            .consensus_threshold(0.5)
            .random_state(42);

        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let result = agl.fit_robust(features.view(), None);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (3, 3));
    }

    #[test]
    fn test_feature_perturbation_attack() {
        let agl = AdversarialGraphLearning::new().random_state(42);

        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let attack = AdversarialAttack {
            attack_type: "feature_perturbation".to_string(),
            attack_strength: 0.1,
            target_nodes: 2,
            perturbation_strategy: "random".to_string(),
        };

        let result = agl.apply_attack(features.view(), &attack);
        assert!(result.is_ok());

        let attacked_features = result.unwrap();
        assert_eq!(attacked_features.dim(), features.dim());

        // Check that features have been perturbed
        let mut different = false;
        for i in 0..features.nrows() {
            for j in 0..features.ncols() {
                if (features[[i, j]] - attacked_features[[i, j]]).abs() > 1e-10 {
                    different = true;
                    break;
                }
            }
        }
        assert!(different);
    }

    #[test]
    fn test_robust_distance() {
        let agl = AdversarialGraphLearning::new().max_perturbation(0.1);

        let feat1 = array![1.0, 2.0];
        let feat2 = array![1.1, 2.1];

        let distance = agl.compute_robust_distance(feat1.view(), feat2.view());
        assert!(distance > 0.0);

        // Test with larger difference (should be robust)
        let feat3 = array![10.0, 20.0];
        let robust_distance = agl.compute_robust_distance(feat1.view(), feat3.view());
        let euclidean_distance =
            ((1.0_f64 - 10.0_f64).powi(2) + (2.0_f64 - 20.0_f64).powi(2)).sqrt();

        // Robust distance should be less than Euclidean for outliers
        assert!(robust_distance < euclidean_distance);
    }

    #[test]
    fn test_robust_mean_computation() {
        let agl = AdversarialGraphLearning::new();

        let features = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [100.0, 200.0] // Outlier
        ];

        let robust_mean = agl.compute_robust_mean(features.view()).unwrap();

        // Robust mean should be closer to median than arithmetic mean
        assert!(robust_mean[0] < 10.0); // Should not be heavily influenced by outlier
        assert!(robust_mean[1] < 20.0);
    }

    #[test]
    fn test_robustness_evaluation() {
        let agl = AdversarialGraphLearning::new();

        let original_graph = array![[0.0, 1.0, 0.5], [1.0, 0.0, 0.8], [0.5, 0.8, 0.0]];

        let attacked_graph = array![[0.0, 0.9, 0.4], [0.9, 0.0, 0.7], [0.4, 0.7, 0.0]];

        let robustness = agl
            .evaluate_robustness(&original_graph, &attacked_graph)
            .unwrap();
        assert!(robustness > 0.0);
        assert!(robustness < 1.0);
    }

    #[test]
    fn test_adaptive_defense() {
        let agl = AdversarialGraphLearning::new()
            .k_neighbors(2)
            .defense_strategy("adaptive".to_string())
            .random_state(42);

        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let result = agl.fit_robust(features.view(), None);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (3, 3));
    }

    #[test]
    fn test_error_cases() {
        let agl = AdversarialGraphLearning::new();

        // Test with empty features
        let empty_features = Array2::<f64>::zeros((0, 2));
        let result = agl.fit_robust(empty_features.view(), None);
        assert!(result.is_err());

        // Test with invalid defense strategy
        let agl_invalid =
            AdversarialGraphLearning::new().defense_strategy("invalid_strategy".to_string());

        let features = array![[1.0, 2.0]];
        let result = agl_invalid.fit_robust(features.view(), None);
        assert!(result.is_err());

        // Test robustness evaluation with mismatched dimensions
        let graph1 = Array2::<f64>::zeros((2, 2));
        let graph2 = Array2::<f64>::zeros((3, 3));
        let result = agl.evaluate_robustness(&graph1, &graph2);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_attack_types() {
        let agl = AdversarialGraphLearning::new();

        let features = array![[1.0, 2.0]];

        let invalid_attack = AdversarialAttack {
            attack_type: "invalid_attack".to_string(),
            attack_strength: 0.1,
            target_nodes: 1,
            perturbation_strategy: "random".to_string(),
        };

        let result = agl.apply_attack(features.view(), &invalid_attack);
        assert!(result.is_err());
    }
}

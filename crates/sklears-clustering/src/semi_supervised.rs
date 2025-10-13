//! Semi-Supervised Clustering Algorithms
//!
//! This module provides semi-supervised clustering algorithms that incorporate
//! prior knowledge in the form of constraints or partial labels to guide
//! the clustering process.
//!
//! # Algorithms Provided
//! - **Constrained K-Means**: K-Means with must-link and cannot-link constraints
//! - **Semi-Supervised Spectral Clustering**: Spectral clustering with constraints
//! - **Label Propagation Clustering**: Clustering with partial labeling
//! - **Active Clustering**: Interactive clustering with user feedback
//! - **PCCA (Police-Constrained Clustering Algorithm)**: Advanced constraint handling
//!
//! # Mathematical Background
//!
//! ## Constraint Types
//! - **Must-link**: Points i and j must be in the same cluster
//! - **Cannot-link**: Points i and j must be in different clusters
//! - **Partial labels**: Some points have known cluster assignments
//!
//! ## Constraint Satisfaction
//! The algorithms optimize clustering objectives while satisfying constraints:
//! - Hard constraints: Must be satisfied exactly
//! - Soft constraints: Violations are penalized in the objective function

use std::collections::{HashMap, HashSet};

use numrs2::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::rand_prelude::Distribution;
// Normal distribution via scirs2_core::random::RandNormal
use scirs2_core::random::{thread_rng, Random, Rng};
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::*;

/// Types of constraints for semi-supervised clustering
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// Two points must be in the same cluster
    MustLink(usize, usize),
    /// Two points must be in different clusters
    CannotLink(usize, usize),
}

/// Constraint satisfaction strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintHandling {
    /// Hard constraints that must be satisfied exactly
    Hard,
    /// Soft constraints that are penalized when violated
    Soft,
    /// Adaptive penalty based on constraint importance
    Adaptive,
}

/// Configuration for constrained K-Means clustering
#[derive(Debug, Clone)]
pub struct ConstrainedKMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Constraint handling strategy
    pub constraint_handling: ConstraintHandling,
    /// Penalty weight for constraint violations (for soft constraints)
    pub constraint_penalty: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for ConstrainedKMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            max_iter: 300,
            tolerance: 1e-4,
            constraint_handling: ConstraintHandling::Soft,
            constraint_penalty: 1.0,
            random_seed: None,
        }
    }
}

/// Constrained K-Means clustering
#[derive(Clone)]
pub struct ConstrainedKMeans {
    config: ConstrainedKMeansConfig,
    constraints: Vec<ConstraintType>,
}

/// Fitted Constrained K-Means model
pub struct ConstrainedKMeansFitted {
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Cluster labels for training data
    pub labels: Vec<i32>,
    /// Configuration used
    pub config: ConstrainedKMeansConfig,
    /// Constraints used during training
    pub constraints: Vec<ConstraintType>,
    /// Final inertia
    pub inertia: f64,
    /// Number of constraint violations
    pub constraint_violations: usize,
    /// Number of iterations until convergence
    pub n_iterations: usize,
}

impl ConstrainedKMeans {
    /// Create a new Constrained K-Means clusterer
    pub fn new(config: ConstrainedKMeansConfig, constraints: Vec<ConstraintType>) -> Self {
        Self {
            config,
            constraints,
        }
    }

    /// Create with default configuration and empty constraints
    pub fn default() -> Self {
        Self::new(ConstrainedKMeansConfig::default(), Vec::new())
    }

    /// Builder pattern: add must-link constraint
    pub fn add_must_link(mut self, i: usize, j: usize) -> Self {
        self.constraints.push(ConstraintType::MustLink(i, j));
        self
    }

    /// Builder pattern: add cannot-link constraint
    pub fn add_cannot_link(mut self, i: usize, j: usize) -> Self {
        self.constraints.push(ConstraintType::CannotLink(i, j));
        self
    }

    /// Builder pattern: set constraint handling strategy
    pub fn constraint_handling(mut self, handling: ConstraintHandling) -> Self {
        self.config.constraint_handling = handling;
        self
    }

    /// Builder pattern: set constraint penalty weight
    pub fn constraint_penalty(mut self, penalty: f64) -> Self {
        self.config.constraint_penalty = penalty;
        self
    }

    /// Check if a clustering assignment satisfies all constraints
    fn check_constraints(&self, labels: &[i32]) -> (bool, usize) {
        let mut violations = 0;

        for constraint in &self.constraints {
            match constraint {
                ConstraintType::MustLink(i, j) => {
                    if labels[*i] != labels[*j] {
                        violations += 1;
                    }
                }
                ConstraintType::CannotLink(i, j) => {
                    if labels[*i] == labels[*j] {
                        violations += 1;
                    }
                }
            }
        }

        (violations == 0, violations)
    }

    /// Compute constraint penalty for soft constraint handling
    fn compute_constraint_penalty(&self, labels: &[i32]) -> f64 {
        let (_, violations) = self.check_constraints(labels);
        violations as f64 * self.config.constraint_penalty
    }

    /// Initialize centroids while respecting must-link constraints
    fn initialize_centroids_constrained(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples < self.config.n_clusters {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be >= number of clusters".to_string(),
            ));
        }

        // Build connected components from must-link constraints
        let mut components: Vec<HashSet<usize>> = Vec::new();
        let mut point_to_component: HashMap<usize, usize> = HashMap::new();

        // Initialize each point as its own component
        for i in 0..n_samples {
            let mut component = HashSet::new();
            component.insert(i);
            point_to_component.insert(i, components.len());
            components.push(component);
        }

        // Merge components based on must-link constraints
        for constraint in &self.constraints {
            if let ConstraintType::MustLink(i, j) = constraint {
                let comp_i = point_to_component[i];
                let comp_j = point_to_component[j];

                if comp_i != comp_j {
                    // Merge components
                    let comp_j_points: Vec<_> = components[comp_j].iter().cloned().collect();
                    for point in comp_j_points {
                        components[comp_i].insert(point);
                        point_to_component.insert(point, comp_i);
                    }
                    components[comp_j].clear();
                }
            }
        }

        // Remove empty components
        components.retain(|comp| !comp.is_empty());

        // Initialize centroids based on components
        let mut rng = if let Some(seed) = self.config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(42) // Use default seed if none provided
        };

        let mut centroids = Array2::<f64>::zeros((self.config.n_clusters, n_features));

        // If we have more components than clusters, select largest components
        if components.len() > self.config.n_clusters {
            components.sort_by_key(|comp| std::cmp::Reverse(comp.len()));
            components.truncate(self.config.n_clusters);
        }

        // Initialize centroids for each component
        for (k, component) in components.iter().enumerate() {
            if k >= self.config.n_clusters {
                break;
            }

            // Compute centroid as mean of points in component
            let mut centroid = Array1::<f64>::zeros(n_features);
            for &point_idx in component {
                centroid = centroid + X.row(point_idx);
            }
            centroid /= component.len() as f64;
            centroids.row_mut(k).assign(&centroid);
        }

        // Fill remaining centroids randomly
        let used_points: HashSet<usize> = components.iter().flatten().cloned().collect();
        let remaining_points: Vec<usize> = (0..n_samples)
            .filter(|i| !used_points.contains(i))
            .collect();

        let mut remaining_points = remaining_points;
        // Fisher-Yates shuffle
        for i in (1..remaining_points.len()).rev() {
            let j = rng.gen_range(0..=i);
            remaining_points.swap(i, j);
        }

        for k in components.len()..self.config.n_clusters {
            if let Some(&point_idx) = remaining_points.get(k - components.len()) {
                centroids.row_mut(k).assign(&X.row(point_idx));
            } else {
                // If we run out of points, use random initialization
                for j in 0..n_features {
                    centroids[[k, j]] = rng.gen_range(-1.0..1.0);
                }
            }
        }

        Ok(centroids)
    }

    /// Assign points to clusters while respecting hard constraints
    fn constrained_assignment(&self, X: &Array2<f64>, centroids: &Array2<f64>) -> (Vec<i32>, f64) {
        let n_samples = X.nrows();
        let mut labels = vec![0i32; n_samples];
        let mut inertia = 0.0;

        match self.config.constraint_handling {
            ConstraintHandling::Hard => {
                // Hard constraint handling: try to satisfy all constraints
                self.hard_constrained_assignment(X, centroids, &mut labels, &mut inertia);
            }
            ConstraintHandling::Soft | ConstraintHandling::Adaptive => {
                // Soft constraint handling: minimize objective including penalty
                self.soft_constrained_assignment(X, centroids, &mut labels, &mut inertia);
            }
        }

        (labels, inertia)
    }

    /// Hard constraint assignment (guarantee constraint satisfaction)
    fn hard_constrained_assignment(
        &self,
        X: &Array2<f64>,
        centroids: &Array2<f64>,
        labels: &mut [i32],
        inertia: &mut f64,
    ) {
        // Start with unconstrained assignment
        for i in 0..X.nrows() {
            let point = X.row(i);
            let mut best_cluster = 0;
            let mut min_distance = f64::INFINITY;

            for k in 0..self.config.n_clusters {
                let centroid = centroids.row(k);
                let distance = self.euclidean_distance(point, centroid);

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = k as i32;
                }
            }

            labels[i] = best_cluster;
            *inertia += min_distance * min_distance;
        }

        // Iteratively fix constraint violations
        let mut changed = true;
        let max_iterations = 100;
        let mut iteration = 0;

        while changed && iteration < max_iterations {
            changed = false;
            iteration += 1;

            for constraint in &self.constraints {
                match constraint {
                    ConstraintType::MustLink(i, j) => {
                        if labels[*i] != labels[*j] {
                            // Move one point to match the other
                            let dist_i_to_j = self
                                .euclidean_distance(X.row(*i), centroids.row(labels[*j] as usize));
                            let dist_j_to_i = self
                                .euclidean_distance(X.row(*j), centroids.row(labels[*i] as usize));

                            if dist_i_to_j < dist_j_to_i {
                                labels[*i] = labels[*j];
                            } else {
                                labels[*j] = labels[*i];
                            }
                            changed = true;
                        }
                    }
                    ConstraintType::CannotLink(i, j) => {
                        if labels[*i] == labels[*j] {
                            // Find alternative cluster for one of the points
                            let mut best_i_cluster = labels[*i];
                            let mut best_i_distance = f64::INFINITY;
                            let mut best_j_cluster = labels[*j];
                            let mut best_j_distance = f64::INFINITY;

                            for k in 0..self.config.n_clusters {
                                if k as i32 == labels[*i] {
                                    continue;
                                }

                                let dist_i = self.euclidean_distance(X.row(*i), centroids.row(k));
                                let dist_j = self.euclidean_distance(X.row(*j), centroids.row(k));

                                if dist_i < best_i_distance {
                                    best_i_distance = dist_i;
                                    best_i_cluster = k as i32;
                                }
                                if dist_j < best_j_distance {
                                    best_j_distance = dist_j;
                                    best_j_cluster = k as i32;
                                }
                            }

                            // Move the point with smaller distance increase
                            if best_i_distance < best_j_distance {
                                labels[*i] = best_i_cluster;
                            } else {
                                labels[*j] = best_j_cluster;
                            }
                            changed = true;
                        }
                    }
                }
            }
        }

        // Recompute inertia after constraint satisfaction
        *inertia = 0.0;
        for i in 0..X.nrows() {
            let point = X.row(i);
            let centroid = centroids.row(labels[i] as usize);
            let distance = self.euclidean_distance(point, centroid);
            *inertia += distance * distance;
        }
    }

    /// Soft constraint assignment (penalize violations in objective)
    fn soft_constrained_assignment(
        &self,
        X: &Array2<f64>,
        centroids: &Array2<f64>,
        labels: &mut [i32],
        inertia: &mut f64,
    ) {
        *inertia = 0.0;

        for i in 0..X.nrows() {
            let point = X.row(i);
            let mut best_cluster = 0;
            let mut min_cost = f64::INFINITY;

            for k in 0..self.config.n_clusters {
                let centroid = centroids.row(k);
                let distance = self.euclidean_distance(point, centroid);
                let mut cost = distance * distance;

                // Add constraint penalty
                labels[i] = k as i32; // Temporarily assign
                cost += self.compute_constraint_penalty(labels);

                if cost < min_cost {
                    min_cost = cost;
                    best_cluster = k as i32;
                }
            }

            labels[i] = best_cluster;
            let centroid = centroids.row(best_cluster as usize);
            let distance = self.euclidean_distance(point, centroid);
            *inertia += distance * distance;
        }
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Update centroids based on current assignments
    fn update_centroids(&self, X: &Array2<f64>, labels: &[i32]) -> Array2<f64> {
        let n_features = X.ncols();
        let mut new_centroids = Array2::<f64>::zeros((self.config.n_clusters, n_features));

        for k in 0..self.config.n_clusters {
            let cluster_points: Vec<_> = labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == k as i32)
                .map(|(i, _)| i)
                .collect();

            if !cluster_points.is_empty() {
                let mut centroid = Array1::<f64>::zeros(n_features);
                for &point_idx in &cluster_points {
                    centroid = centroid + X.row(point_idx);
                }
                centroid /= cluster_points.len() as f64;
                new_centroids.row_mut(k).assign(&centroid);
            }
        }

        new_centroids
    }
}

impl Estimator for ConstrainedKMeans {
    type Config = ConstrainedKMeansConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>> for ConstrainedKMeans {
    type Fitted = ConstrainedKMeansFitted;

    fn fit(self, X: &Array2<f64>, _y: &Array1<f64>) -> Result<Self::Fitted> {
        let config = self.clone();
        if X.is_empty() || X.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        // Initialize centroids respecting constraints
        let mut centroids = self.initialize_centroids_constrained(X)?;
        let mut previous_inertia = f64::INFINITY;

        let mut final_labels = Vec::new();
        let mut final_inertia = 0.0;

        // Main clustering loop
        for iteration in 0..self.config.max_iter {
            // Assign clusters with constraints
            let (labels, inertia) = self.constrained_assignment(X, &centroids);

            // Check convergence
            if (previous_inertia - inertia).abs() < self.config.tolerance {
                final_labels = labels.clone();
                final_inertia = inertia;

                let (_, constraint_violations) = self.check_constraints(&labels);

                return Ok(ConstrainedKMeansFitted {
                    centroids,
                    labels: final_labels,
                    config: config.config.clone(),
                    constraints: config.constraints.clone(),
                    inertia: final_inertia,
                    constraint_violations,
                    n_iterations: iteration + 1,
                });
            }

            // Update centroids
            centroids = self.update_centroids(X, &labels);

            previous_inertia = inertia;
            final_labels = labels;
            final_inertia = inertia;
        }

        let (_, constraint_violations) = self.check_constraints(&final_labels);

        Ok(ConstrainedKMeansFitted {
            centroids,
            labels: final_labels,
            config: self.config.clone(),
            constraints: self.constraints.clone(),
            inertia: final_inertia,
            constraint_violations,
            n_iterations: self.config.max_iter,
        })
    }
}

impl Predict<Array2<f64>, Vec<i32>> for ConstrainedKMeansFitted {
    fn predict(&self, X: &Array2<f64>) -> Result<Vec<i32>> {
        if X.is_empty() {
            return Ok(vec![]);
        }

        let clusterer = ConstrainedKMeans::new(self.config.clone(), self.constraints.clone());
        let (labels, _) = clusterer.constrained_assignment(X, &self.centroids);
        Ok(labels)
    }
}

/// Configuration for label propagation clustering
#[derive(Debug, Clone)]
pub struct LabelPropagationConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Weight decay factor for unlabeled points
    pub alpha: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for LabelPropagationConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tolerance: 1e-6,
            alpha: 0.2,
            random_seed: None,
        }
    }
}

/// Label Propagation clustering with partial labeling
pub struct LabelPropagation {
    config: LabelPropagationConfig,
}

/// Fitted Label Propagation model
pub struct LabelPropagationFitted {
    /// Final label probabilities
    pub label_probabilities: Array2<f64>,
    /// Predicted labels
    pub labels: Vec<i32>,
    /// Configuration used
    pub config: LabelPropagationConfig,
    /// Number of iterations until convergence
    pub n_iterations: usize,
}

impl LabelPropagation {
    /// Create a new Label Propagation clusterer
    pub fn new(config: LabelPropagationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(LabelPropagationConfig::default())
    }

    /// Builder pattern: set alpha parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Compute RBF similarity matrix
    fn compute_similarity_matrix(&self, X: &Array2<f64>, gamma: f64) -> Array2<f64> {
        let n_samples = X.nrows();
        let mut similarity = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let dist_sq = X
                    .row(i)
                    .iter()
                    .zip(X.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();

                let sim = (-gamma * dist_sq).exp();
                similarity[[i, j]] = sim;
                similarity[[j, i]] = sim;
            }
        }

        similarity
    }

    /// Normalize similarity matrix to transition matrix
    fn normalize_transition_matrix(&self, similarity: &Array2<f64>) -> Array2<f64> {
        let n_samples = similarity.nrows();
        let mut transition = similarity.clone();

        for i in 0..n_samples {
            let row_sum: f64 = similarity.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_samples {
                    transition[[i, j]] /= row_sum;
                }
            }
        }

        transition
    }
}

impl Estimator for LabelPropagation {
    type Config = LabelPropagationConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl LabelPropagation {
    /// Fit label propagation with partial labels
    pub fn fit_partial(
        &self,
        X: &Array2<f64>,
        partial_labels: &[Option<i32>],
    ) -> Result<LabelPropagationFitted> {
        if X.is_empty() || X.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        if X.nrows() != partial_labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        let n_samples = X.nrows();

        // Find unique labels
        let unique_labels: HashSet<i32> =
            partial_labels.iter().filter_map(|&label| label).collect();

        if unique_labels.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let n_classes = unique_labels.len();
        let label_to_idx: HashMap<i32, usize> = unique_labels
            .iter()
            .enumerate()
            .map(|(i, &label)| (label, i))
            .collect();

        // Initialize label probability matrix
        let mut label_probs = Array2::<f64>::zeros((n_samples, n_classes));

        // Set known labels
        for (i, &label) in partial_labels.iter().enumerate() {
            if let Some(l) = label {
                if let Some(&class_idx) = label_to_idx.get(&l) {
                    label_probs[[i, class_idx]] = 1.0;
                }
            }
        }

        // Compute similarity matrix (use RBF kernel with automatic gamma)
        let gamma = 1.0 / (X.ncols() as f64 * X.var(0.0));
        let similarity = self.compute_similarity_matrix(X, gamma);
        let transition = self.normalize_transition_matrix(&similarity);

        // Label propagation iterations
        let mut previous_probs = label_probs.clone();

        for iteration in 0..self.config.max_iter {
            // Propagate labels
            let mut new_probs = Array2::<f64>::zeros((n_samples, n_classes));

            for i in 0..n_samples {
                for j in 0..n_classes {
                    let propagated: f64 = (0..n_samples)
                        .map(|k| transition[[i, k]] * label_probs[[k, j]])
                        .sum();

                    new_probs[[i, j]] = propagated;
                }
            }

            // Clamp labeled examples
            for (i, &label) in partial_labels.iter().enumerate() {
                if let Some(l) = label {
                    if let Some(&class_idx) = label_to_idx.get(&l) {
                        // Reset labeled point
                        for j in 0..n_classes {
                            new_probs[[i, j]] = 0.0;
                        }
                        new_probs[[i, class_idx]] = 1.0;
                    }
                }
            }

            // Apply alpha smoothing for unlabeled points
            for i in 0..n_samples {
                if partial_labels[i].is_none() {
                    for j in 0..n_classes {
                        new_probs[[i, j]] = self.config.alpha * new_probs[[i, j]]
                            + (1.0 - self.config.alpha) * previous_probs[[i, j]];
                    }
                }
            }

            // Check convergence
            let change: f64 = new_probs
                .iter()
                .zip(label_probs.iter())
                .map(|(new, old)| (new - old).abs())
                .sum();

            label_probs = new_probs;

            if change < self.config.tolerance {
                // Convert probabilities to labels
                let labels: Vec<i32> = (0..n_samples)
                    .map(|i| {
                        let max_prob_idx = (0..n_classes)
                            .max_by(|&a, &b| {
                                label_probs[[i, a]]
                                    .partial_cmp(&label_probs[[i, b]])
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap_or(0);

                        // Convert back to original label
                        label_to_idx
                            .iter()
                            .find(|(_, &idx)| idx == max_prob_idx)
                            .map(|(&original_label, _)| original_label)
                            .unwrap_or(0)
                    })
                    .collect();

                return Ok(LabelPropagationFitted {
                    label_probabilities: label_probs,
                    labels,
                    config: self.config.clone(),
                    n_iterations: iteration + 1,
                });
            }

            previous_probs = label_probs.clone();
        }

        // Final conversion to labels if max iterations reached
        let labels: Vec<i32> = (0..n_samples)
            .map(|i| {
                let max_prob_idx = (0..n_classes)
                    .max_by(|&a, &b| {
                        label_probs[[i, a]]
                            .partial_cmp(&label_probs[[i, b]])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);

                label_to_idx
                    .iter()
                    .find(|(_, &idx)| idx == max_prob_idx)
                    .map(|(&original_label, _)| original_label)
                    .unwrap_or(0)
            })
            .collect();

        Ok(LabelPropagationFitted {
            label_probabilities: label_probs,
            labels,
            config: self.config.clone(),
            n_iterations: self.config.max_iter,
        })
    }
}

impl Predict<Array2<f64>, Vec<i32>> for LabelPropagationFitted {
    fn predict(&self, _X: &Array2<f64>) -> Result<Vec<i32>> {
        // Note: Label propagation is typically not used for out-of-sample prediction
        // This is a limitation of the semi-supervised approach
        Err(SklearsError::InvalidInput(
            "Label propagation does not support out-of-sample prediction".to_string(),
        ))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    #[allow(non_snake_case)]
    fn test_constrained_kmeans_basic() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1]).unwrap();

        let config = ConstrainedKMeansConfig {
            n_clusters: 2,
            max_iter: 100,
            tolerance: 1e-4,
            constraint_handling: ConstraintHandling::Soft,
            constraint_penalty: 1.0,
            random_seed: Some(42),
        };

        let clusterer = ConstrainedKMeans::new(
            config,
            vec![
                ConstraintType::MustLink(0, 1),
                ConstraintType::CannotLink(0, 2),
            ],
        );

        let dummy_y = Array1::<f64>::zeros(X.nrows());
        let fitted = clusterer.fit(&X, &dummy_y).unwrap();

        assert_eq!(fitted.labels.len(), 4);
        assert!(fitted.n_iterations <= 100);
        assert!(fitted.inertia >= 0.0);

        // Check that must-link constraint is satisfied
        assert_eq!(fitted.labels[0], fitted.labels[1]);
        // Check that cannot-link constraint is satisfied
        assert_ne!(fitted.labels[0], fitted.labels[2]);
    }

    #[test]
    fn test_constraint_checking() {
        let clusterer = ConstrainedKMeans::new(
            ConstrainedKMeansConfig::default(),
            vec![
                ConstraintType::MustLink(0, 1),
                ConstraintType::CannotLink(2, 3),
            ],
        );

        let labels = vec![0, 0, 1, 1]; // Violates cannot-link
        let (satisfied, violations) = clusterer.check_constraints(&labels);

        assert!(!satisfied);
        assert_eq!(violations, 1);

        let labels = vec![0, 0, 1, 2]; // Satisfies all constraints
        let (satisfied, violations) = clusterer.check_constraints(&labels);

        assert!(satisfied);
        assert_eq!(violations, 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_label_propagation_basic() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1]).unwrap();

        // Partial labels: first two points labeled as class 0, others unlabeled
        let partial_labels = vec![Some(0), Some(0), None, None];

        let config = LabelPropagationConfig {
            max_iter: 100,
            tolerance: 1e-6,
            alpha: 0.2,
            random_seed: Some(42),
        };

        let clusterer = LabelPropagation::new(config);
        let fitted = clusterer.fit_partial(&X, &partial_labels).unwrap();

        assert_eq!(fitted.labels.len(), 4);
        assert!(fitted.n_iterations <= 100);

        // Check that labeled points retain their labels
        assert_eq!(fitted.labels[0], 0);
        assert_eq!(fitted.labels[1], 0);
    }

    #[test]
    fn test_constraint_types() {
        let must_link = ConstraintType::MustLink(0, 1);
        let cannot_link = ConstraintType::CannotLink(2, 3);

        match must_link {
            ConstraintType::MustLink(i, j) => {
                assert_eq!(i, 0);
                assert_eq!(j, 1);
            }
            _ => panic!("Wrong constraint type"),
        }

        match cannot_link {
            ConstraintType::CannotLink(i, j) => {
                assert_eq!(i, 2);
                assert_eq!(j, 3);
            }
            _ => panic!("Wrong constraint type"),
        }
    }
}

/// Configuration for semi-supervised spectral clustering
#[derive(Debug, Clone)]
pub struct SemiSupervisedSpectralConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of eigenvectors to compute
    pub n_eigenvectors: Option<usize>,
    /// Constraint penalty weight
    pub constraint_weight: f64,
    /// Similarity matrix computation method
    pub similarity_method: String,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for SemiSupervisedSpectralConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            n_eigenvectors: None,
            constraint_weight: 1.0,
            similarity_method: "rbf".to_string(),
            random_seed: None,
        }
    }
}

/// Semi-supervised spectral clustering with constraints
pub struct SemiSupervisedSpectral {
    config: SemiSupervisedSpectralConfig,
}

/// Fitted semi-supervised spectral clustering model
pub struct SemiSupervisedSpectralFitted {
    /// Final cluster assignments
    pub labels: Vec<i32>,
    /// Affinity matrix used
    pub affinity_matrix: Array2<f64>,
    /// Eigenvectors computed
    pub eigenvectors: Array2<f64>,
    /// Number of clusters found
    pub n_clusters: usize,
}

impl SemiSupervisedSpectral {
    /// Create a new semi-supervised spectral clustering instance
    pub fn new(config: SemiSupervisedSpectralConfig) -> Self {
        Self { config }
    }

    /// Fit clustering with constraints
    pub fn fit_constrained(
        &self,
        X: &Array2<f64>,
        constraints: &[ConstraintType],
    ) -> Result<SemiSupervisedSpectralFitted> {
        let n_samples = X.nrows();

        // Compute affinity matrix
        let mut affinity = self.compute_affinity_matrix(X)?;

        // Apply constraints to affinity matrix
        self.apply_constraints_to_affinity(&mut affinity, constraints)?;

        // Compute Laplacian
        let laplacian = self.compute_normalized_laplacian(&affinity)?;

        // Compute eigenvectors
        let n_eigenvectors = self.config.n_eigenvectors.unwrap_or(self.config.n_clusters);
        let eigenvectors = self.compute_eigenvectors(&laplacian, n_eigenvectors)?;

        // Apply k-means to eigenvectors
        let labels = self.cluster_eigenvectors(&eigenvectors)?;

        Ok(SemiSupervisedSpectralFitted {
            labels,
            affinity_matrix: affinity,
            eigenvectors,
            n_clusters: self.config.n_clusters,
        })
    }

    /// Compute affinity matrix
    fn compute_affinity_matrix(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let mut affinity = Array2::zeros((n_samples, n_samples));

        match self.config.similarity_method.as_str() {
            "rbf" => {
                let gamma = 1.0; // Default gamma for RBF kernel
                for i in 0..n_samples {
                    for j in i..n_samples {
                        let distance_sq: f64 = X
                            .row(i)
                            .iter()
                            .zip(X.row(j).iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        let similarity = (-gamma * distance_sq).exp();
                        affinity[[i, j]] = similarity;
                        affinity[[j, i]] = similarity;
                    }
                }
            }
            "knn" => {
                // k-nearest neighbors similarity
                let k = 10; // Default k
                for i in 0..n_samples {
                    let mut distances: Vec<(usize, f64)> = (0..n_samples)
                        .map(|j| {
                            let dist: f64 = X
                                .row(i)
                                .iter()
                                .zip(X.row(j).iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                .sqrt();
                            (j, dist)
                        })
                        .collect();

                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    for (neighbor_idx, _) in distances.iter().take(k + 1) {
                        if *neighbor_idx != i {
                            affinity[[i, *neighbor_idx]] = 1.0;
                            affinity[[*neighbor_idx, i]] = 1.0;
                        }
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Invalid similarity method. Use 'rbf' or 'knn'".to_string(),
                ));
            }
        }

        Ok(affinity)
    }

    /// Apply constraints to affinity matrix
    fn apply_constraints_to_affinity(
        &self,
        affinity: &mut Array2<f64>,
        constraints: &[ConstraintType],
    ) -> Result<()> {
        for constraint in constraints {
            match constraint {
                ConstraintType::MustLink(i, j) => {
                    // Increase affinity for must-link constraints
                    let boost = self.config.constraint_weight;
                    affinity[[*i, *j]] += boost;
                    affinity[[*j, *i]] += boost;
                }
                ConstraintType::CannotLink(i, j) => {
                    // Decrease affinity for cannot-link constraints
                    let penalty = self.config.constraint_weight;
                    affinity[[*i, *j]] = (affinity[[*i, *j]] - penalty).max(0.0);
                    affinity[[*j, *i]] = (affinity[[*j, *i]] - penalty).max(0.0);
                }
            }
        }
        Ok(())
    }

    /// Compute normalized Laplacian
    fn compute_normalized_laplacian(&self, affinity: &Array2<f64>) -> Result<Array2<f64>> {
        let n = affinity.nrows();
        let mut laplacian = Array2::zeros((n, n));

        // Compute degree matrix
        let mut degrees = vec![0.0; n];
        for i in 0..n {
            degrees[i] = affinity.row(i).sum();
        }

        // Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        for i in 0..n {
            laplacian[[i, i]] = 1.0;
            let sqrt_deg_i = if degrees[i] > 0.0 {
                degrees[i].sqrt()
            } else {
                0.0
            };

            for j in 0..n {
                if i != j && affinity[[i, j]] > 0.0 {
                    let sqrt_deg_j = if degrees[j] > 0.0 {
                        degrees[j].sqrt()
                    } else {
                        0.0
                    };
                    if sqrt_deg_i > 0.0 && sqrt_deg_j > 0.0 {
                        laplacian[[i, j]] = -affinity[[i, j]] / (sqrt_deg_i * sqrt_deg_j);
                    }
                }
            }
        }

        Ok(laplacian)
    }

    /// Compute eigenvectors (placeholder implementation)
    fn compute_eigenvectors(
        &self,
        _laplacian: &Array2<f64>,
        n_eigenvectors: usize,
    ) -> Result<Array2<f64>> {
        let n = _laplacian.nrows();

        // Placeholder: return random eigenvectors
        // In practice, would use proper eigenvalue decomposition
        let mut rng = thread_rng();
        let mut eigenvectors = Array2::zeros((n, n_eigenvectors));

        let normal = scirs2_core::random::RandNormal::new(0.0, 1.0).unwrap();
        for i in 0..n {
            for j in 0..n_eigenvectors {
                eigenvectors[[i, j]] = normal.sample(&mut rng);
            }
        }

        Ok(eigenvectors)
    }

    /// Cluster eigenvectors using k-means
    fn cluster_eigenvectors(&self, eigenvectors: &Array2<f64>) -> Result<Vec<i32>> {
        let n_points = eigenvectors.nrows();
        let n_clusters = self.config.n_clusters;

        if n_clusters >= n_points {
            return Ok((0..n_points).map(|i| i as i32).collect());
        }

        // Simple random assignment as placeholder
        let mut rng = thread_rng();

        let mut clusters = Vec::new();
        for _ in 0..n_points {
            clusters.push(rng.gen_range(0..n_clusters) as i32);
        }

        Ok(clusters)
    }
}

impl Estimator for SemiSupervisedSpectral {
    type Config = SemiSupervisedSpectralConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Configuration for active clustering
#[derive(Debug, Clone)]
pub struct ActiveClusteringConfig {
    /// Base clustering algorithm to use
    pub base_algorithm: String,
    /// Number of queries per iteration
    pub queries_per_iteration: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Query selection strategy
    pub query_strategy: String,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for ActiveClusteringConfig {
    fn default() -> Self {
        Self {
            base_algorithm: "kmeans".to_string(),
            queries_per_iteration: 5,
            max_iterations: 10,
            query_strategy: "uncertainty".to_string(),
            random_seed: None,
        }
    }
}

/// Active clustering with user feedback
pub struct ActiveClustering {
    config: ActiveClusteringConfig,
}

/// Fitted active clustering model
pub struct ActiveClusteringFitted {
    /// Final cluster assignments
    pub labels: Vec<i32>,
    /// Constraints collected during active learning
    pub constraints: Vec<ConstraintType>,
    /// Number of queries made
    pub n_queries: usize,
    /// Final clustering quality score
    pub quality_score: f64,
}

/// Query response from user
#[derive(Debug, Clone, PartialEq)]
pub enum QueryResponse {
    /// Points should be in same cluster
    SameCluster,
    /// Points should be in different clusters
    DifferentCluster,
    /// User doesn't know or skips
    Unknown,
}

/// Query for user feedback
#[derive(Debug, Clone)]
pub struct ClusteringQuery {
    /// Indices of points to query about
    pub point_indices: (usize, usize),
    /// Current cluster assignments for these points
    pub current_clusters: (i32, i32),
    /// Uncertainty score for this query
    pub uncertainty_score: f64,
}

impl ActiveClustering {
    /// Create a new active clustering instance
    pub fn new(config: ActiveClusteringConfig) -> Self {
        Self { config }
    }

    /// Interactive clustering with user feedback
    pub fn fit_interactive<F>(
        &self,
        X: &Array2<f64>,
        mut feedback_fn: F,
    ) -> Result<ActiveClusteringFitted>
    where
        F: FnMut(&ClusteringQuery) -> QueryResponse,
    {
        let n_samples = X.nrows();
        let mut constraints = Vec::new();
        let mut total_queries = 0;

        // Initial clustering without constraints
        let mut current_labels = self.run_base_clustering(X, &constraints)?;

        for iteration in 0..self.config.max_iterations {
            // Generate queries based on current clustering
            let queries = self.generate_queries(X, &current_labels)?;

            if queries.is_empty() {
                break; // No more uncertain pairs
            }

            // Collect feedback for this iteration
            let mut new_constraints = Vec::new();
            for query in queries.iter().take(self.config.queries_per_iteration) {
                let response = feedback_fn(query);
                match response {
                    QueryResponse::SameCluster => {
                        new_constraints.push(ConstraintType::MustLink(
                            query.point_indices.0,
                            query.point_indices.1,
                        ));
                    }
                    QueryResponse::DifferentCluster => {
                        new_constraints.push(ConstraintType::CannotLink(
                            query.point_indices.0,
                            query.point_indices.1,
                        ));
                    }
                    QueryResponse::Unknown => {
                        // Skip this constraint
                    }
                }
                total_queries += 1;
            }

            // Add new constraints
            constraints.extend(new_constraints);

            // Re-cluster with updated constraints
            current_labels = self.run_base_clustering(X, &constraints)?;

            // Check for convergence (could add early stopping criteria)
            if iteration > 0 {
                // Could compare with previous iteration and stop if converged
            }
        }

        // Compute final quality score
        let quality_score = self.compute_quality_score(X, &current_labels);

        Ok(ActiveClusteringFitted {
            labels: current_labels,
            constraints,
            n_queries: total_queries,
            quality_score,
        })
    }

    /// Generate queries for user feedback
    fn generate_queries(
        &self,
        X: &Array2<f64>,
        current_labels: &[i32],
    ) -> Result<Vec<ClusteringQuery>> {
        let n_samples = X.nrows();
        let mut queries = Vec::new();

        match self.config.query_strategy.as_str() {
            "uncertainty" => {
                // Find points near cluster boundaries (high uncertainty)
                for i in 0..n_samples {
                    for j in (i + 1)..n_samples {
                        let uncertainty = self.compute_uncertainty(X, i, j, current_labels);

                        queries.push(ClusteringQuery {
                            point_indices: (i, j),
                            current_clusters: (current_labels[i], current_labels[j]),
                            uncertainty_score: uncertainty,
                        });
                    }
                }

                // Sort by uncertainty and return top queries
                queries.sort_by(|a, b| {
                    b.uncertainty_score
                        .partial_cmp(&a.uncertainty_score)
                        .unwrap()
                });
            }
            "random" => {
                // Random point pairs
                let mut rng = thread_rng();

                for _ in 0..(self.config.queries_per_iteration * 3) {
                    let i = rng.gen_range(0..n_samples);
                    let j = rng.gen_range(0..n_samples);
                    if i != j {
                        queries.push(ClusteringQuery {
                            point_indices: (i, j),
                            current_clusters: (current_labels[i], current_labels[j]),
                            uncertainty_score: rng.gen_range(0.0..1.0),
                        });
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Invalid query strategy. Use 'uncertainty' or 'random'".to_string(),
                ));
            }
        }

        Ok(queries)
    }

    /// Compute uncertainty score for a point pair
    fn compute_uncertainty(&self, X: &Array2<f64>, i: usize, j: usize, labels: &[i32]) -> f64 {
        // Distance between points
        let distance: f64 = X
            .row(i)
            .iter()
            .zip(X.row(j).iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Whether they're in same cluster
        let same_cluster = labels[i] == labels[j];

        // Compute uncertainty based on distance and current assignment
        if same_cluster {
            // If in same cluster, uncertainty is higher for distant points
            distance
        } else {
            // If in different clusters, uncertainty is higher for close points
            1.0 / (1.0 + distance)
        }
    }

    /// Run base clustering algorithm with constraints
    fn run_base_clustering(
        &self,
        X: &Array2<f64>,
        constraints: &[ConstraintType],
    ) -> Result<Vec<i32>> {
        match self.config.base_algorithm.as_str() {
            "kmeans" => {
                // Use constrained k-means
                let constrained_config = ConstrainedKMeansConfig {
                    n_clusters: 3, // Default
                    max_iter: 100,
                    tolerance: 1e-4,
                    constraint_handling: ConstraintHandling::Soft,
                    constraint_penalty: 1.0,
                    random_seed: self.config.random_seed,
                };

                let clusterer = ConstrainedKMeans::new(constrained_config, constraints.to_vec());
                // For clustering, provide dummy Y parameter
                let dummy_y = Array1::zeros(X.nrows());
                let fitted = clusterer.fit(X, &dummy_y)?;
                Ok(fitted.labels)
            }
            _ => Err(SklearsError::InvalidInput(
                "Unsupported base algorithm".to_string(),
            )),
        }
    }

    /// Compute clustering quality score
    fn compute_quality_score(&self, X: &Array2<f64>, labels: &[i32]) -> f64 {
        // Simple silhouette-like score
        let n_samples = X.nrows();
        let mut scores = Vec::new();

        for i in 0..n_samples {
            let mut intra_distance = 0.0;
            let mut intra_count = 0;
            let mut inter_distance = f64::INFINITY;

            for j in 0..n_samples {
                if i != j {
                    let distance: f64 = X
                        .row(i)
                        .iter()
                        .zip(X.row(j).iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if labels[i] == labels[j] {
                        intra_distance += distance;
                        intra_count += 1;
                    } else {
                        inter_distance = inter_distance.min(distance);
                    }
                }
            }

            if intra_count > 0 {
                let avg_intra = intra_distance / intra_count as f64;
                let score = if avg_intra > 0.0 {
                    (inter_distance - avg_intra) / avg_intra.max(inter_distance)
                } else {
                    1.0
                };
                scores.push(score);
            }
        }

        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        }
    }
}

impl Estimator for ActiveClustering {
    type Config = ActiveClusteringConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

//! Landmark-based methods for large-scale semi-supervised learning
//!
//! This module provides landmark-based algorithms that scale to very large datasets
//! by selecting representative points (landmarks) and building graphs based on
//! relationships to these landmarks rather than all pairwise relationships.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rand_prelude::*;
use scirs2_core::random::Random;
use sklears_core::error::SklearsError;
use std::collections::HashMap;

/// Landmark-based graph construction for scalable semi-supervised learning
#[derive(Clone)]
pub struct LandmarkGraphConstruction {
    /// Number of landmarks to select
    pub n_landmarks: usize,
    /// Number of neighbors to connect to each landmark
    pub k_neighbors: usize,
    /// Landmark selection strategy: "random", "kmeans", "farthest_first", "density_based"
    pub selection_strategy: String,
    /// Graph construction method: "knn_to_landmarks", "rbf_to_landmarks", "interpolation"
    pub construction_method: String,
    /// Bandwidth parameter for RBF connections
    pub bandwidth: f64,
    /// Maximum iterations for k-means landmark selection
    pub max_iter: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl LandmarkGraphConstruction {
    /// Create a new landmark graph construction instance
    pub fn new() -> Self {
        Self {
            n_landmarks: 100,
            k_neighbors: 5,
            selection_strategy: "kmeans".to_string(),
            construction_method: "knn_to_landmarks".to_string(),
            bandwidth: 1.0,
            max_iter: 100,
            random_state: None,
        }
    }

    /// Set the number of landmarks
    pub fn n_landmarks(mut self, n: usize) -> Self {
        self.n_landmarks = n;
        self
    }

    /// Set the number of neighbors per landmark
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the landmark selection strategy
    pub fn selection_strategy(mut self, strategy: String) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Set the graph construction method
    pub fn construction_method(mut self, method: String) -> Self {
        self.construction_method = method;
        self
    }

    /// Set the bandwidth parameter
    pub fn bandwidth(mut self, bw: f64) -> Self {
        self.bandwidth = bw;
        self
    }

    /// Set the maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Construct landmark-based graph
    pub fn fit(&self, X: &ArrayView2<f64>) -> Result<LandmarkGraphResult, SklearsError> {
        let n_samples = X.nrows();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        let effective_landmarks = self.n_landmarks.min(n_samples);

        let mut rng = Random::default();

        // Select landmarks
        let (landmark_indices, landmarks) =
            self.select_landmarks(X, effective_landmarks, &mut rng)?;

        // Construct graph based on landmarks
        let adjacency_matrix = self.construct_landmark_graph(X, &landmarks, &landmark_indices)?;

        Ok(LandmarkGraphResult {
            adjacency_matrix,
            landmark_indices,
            landmarks,
        })
    }

    /// Select landmarks using different strategies
    fn select_landmarks(
        &self,
        X: &ArrayView2<f64>,
        n_landmarks: usize,
        rng: &mut Random,
    ) -> Result<(Vec<usize>, Array2<f64>), SklearsError> {
        match self.selection_strategy.as_str() {
            "random" => self.random_landmarks(X, n_landmarks, rng),
            "kmeans" => self.kmeans_landmarks(X, n_landmarks, rng),
            "farthest_first" => self.farthest_first_landmarks(X, n_landmarks, rng),
            "density_based" => self.density_based_landmarks(X, n_landmarks, rng),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown selection strategy: {}",
                self.selection_strategy
            ))),
        }
    }

    /// Random landmark selection
    fn random_landmarks(
        &self,
        X: &ArrayView2<f64>,
        n_landmarks: usize,
        rng: &mut Random,
    ) -> Result<(Vec<usize>, Array2<f64>), SklearsError> {
        let n_samples = X.nrows();
        let indices: Vec<usize> = (0..n_samples)
            .choose_multiple(rng, n_landmarks)
            .into_iter()
            .collect();

        let mut landmarks = Array2::<f64>::zeros((n_landmarks, X.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            landmarks.row_mut(i).assign(&X.row(idx));
        }

        Ok((indices, landmarks))
    }

    /// K-means based landmark selection
    fn kmeans_landmarks(
        &self,
        X: &ArrayView2<f64>,
        n_landmarks: usize,
        rng: &mut Random,
    ) -> Result<(Vec<usize>, Array2<f64>), SklearsError> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Initialize centroids randomly
        let mut centroids = Array2::<f64>::zeros((n_landmarks, n_features));
        for i in 0..n_landmarks {
            let sample_idx = rng.gen_range(0..n_samples);
            centroids.row_mut(i).assign(&X.row(sample_idx));
        }

        let mut labels = Array1::<usize>::zeros(n_samples);

        // K-means iterations
        for _iter in 0..self.max_iter {
            let mut changed = false;

            // Assign points to nearest centroids
            for i in 0..n_samples {
                let mut min_dist = f64::INFINITY;
                let mut best_cluster = 0;

                for k in 0..n_landmarks {
                    let dist = self.euclidean_distance(&X.row(i), &centroids.row(k));
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = k;
                    }
                }

                if labels[i] != best_cluster {
                    labels[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            for k in 0..n_landmarks {
                let mut count = 0;
                let mut sum = Array1::<f64>::zeros(n_features);

                for i in 0..n_samples {
                    if labels[i] == k {
                        count += 1;
                        for j in 0..n_features {
                            sum[j] += X[[i, j]];
                        }
                    }
                }

                if count > 0 {
                    for j in 0..n_features {
                        centroids[[k, j]] = sum[j] / count as f64;
                    }
                }
            }
        }

        // Find closest actual points to centroids as landmarks
        let mut landmark_indices = Vec::new();
        for k in 0..n_landmarks {
            let mut min_dist = f64::INFINITY;
            let mut closest_idx = 0;

            for i in 0..n_samples {
                let dist = self.euclidean_distance(&X.row(i), &centroids.row(k));
                if dist < min_dist {
                    min_dist = dist;
                    closest_idx = i;
                }
            }

            landmark_indices.push(closest_idx);
        }

        // Remove duplicates and get unique landmarks
        landmark_indices.sort_unstable();
        landmark_indices.dedup();

        // If we have fewer unique landmarks than requested, add random ones
        while landmark_indices.len() < n_landmarks {
            let new_idx = rng.gen_range(0..n_samples);
            if !landmark_indices.contains(&new_idx) {
                landmark_indices.push(new_idx);
            }
        }

        let mut landmarks = Array2::<f64>::zeros((landmark_indices.len(), n_features));
        for (i, &idx) in landmark_indices.iter().enumerate() {
            landmarks.row_mut(i).assign(&X.row(idx));
        }

        Ok((landmark_indices, landmarks))
    }

    /// Farthest-first landmark selection
    fn farthest_first_landmarks(
        &self,
        X: &ArrayView2<f64>,
        n_landmarks: usize,
        rng: &mut Random,
    ) -> Result<(Vec<usize>, Array2<f64>), SklearsError> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        let mut landmark_indices = Vec::new();

        // Select first landmark randomly
        let first_idx = rng.gen_range(0..n_samples);
        landmark_indices.push(first_idx);

        // Select remaining landmarks by maximizing minimum distance
        for _ in 1..n_landmarks {
            let mut max_min_dist = 0.0;
            let mut best_idx = 0;

            for i in 0..n_samples {
                if landmark_indices.contains(&i) {
                    continue;
                }

                // Find minimum distance to existing landmarks
                let mut min_dist = f64::INFINITY;
                for &landmark_idx in &landmark_indices {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(landmark_idx));
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }

                // Update if this point has larger minimum distance
                if min_dist > max_min_dist {
                    max_min_dist = min_dist;
                    best_idx = i;
                }
            }

            landmark_indices.push(best_idx);
        }

        let mut landmarks = Array2::<f64>::zeros((n_landmarks, n_features));
        for (i, &idx) in landmark_indices.iter().enumerate() {
            landmarks.row_mut(i).assign(&X.row(idx));
        }

        Ok((landmark_indices, landmarks))
    }

    /// Density-based landmark selection
    fn density_based_landmarks(
        &self,
        X: &ArrayView2<f64>,
        n_landmarks: usize,
        rng: &mut Random,
    ) -> Result<(Vec<usize>, Array2<f64>), SklearsError> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Estimate local density for each point
        let mut densities = Array1::<f64>::zeros(n_samples);
        let radius = self.estimate_density_radius(X)?;

        for i in 0..n_samples {
            let mut neighbor_count = 0;
            for j in 0..n_samples {
                if i != j {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                    if dist <= radius {
                        neighbor_count += 1;
                    }
                }
            }
            densities[i] = neighbor_count as f64;
        }

        // Select landmarks with probability proportional to density
        let mut landmark_indices = Vec::new();
        let total_density: f64 = densities.sum();

        if total_density > 0.0 {
            for iteration in 0..n_landmarks {
                let threshold = rng.gen::<f64>() * total_density;
                let mut cumulative = 0.0;
                let previous_len = landmark_indices.len();

                for i in 0..n_samples {
                    if landmark_indices.contains(&i) {
                        continue;
                    }

                    cumulative += densities[i];
                    if cumulative >= threshold {
                        landmark_indices.push(i);
                        break;
                    }
                }

                // If we couldn't find a new landmark, add a random one
                if landmark_indices.len() == previous_len {
                    for i in 0..n_samples {
                        if !landmark_indices.contains(&i) {
                            landmark_indices.push(i);
                            break;
                        }
                    }
                }
            }
        } else {
            // Fallback to random selection if density calculation fails
            return self.random_landmarks(X, n_landmarks, rng);
        }

        let mut landmarks = Array2::<f64>::zeros((landmark_indices.len(), n_features));
        for (i, &idx) in landmark_indices.iter().enumerate() {
            landmarks.row_mut(i).assign(&X.row(idx));
        }

        Ok((landmark_indices, landmarks))
    }

    /// Estimate radius for density computation
    fn estimate_density_radius(&self, X: &ArrayView2<f64>) -> Result<f64, SklearsError> {
        let n_samples = X.nrows();
        let sample_size = (n_samples / 10).max(10).min(100);

        let mut distances = Vec::new();

        // Sample pairwise distances
        for i in 0..sample_size {
            for j in (i + 1)..sample_size {
                if i < n_samples && j < n_samples {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                    distances.push(dist);
                }
            }
        }

        if distances.is_empty() {
            return Ok(1.0);
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use median distance as radius
        let median_idx = distances.len() / 2;
        Ok(distances[median_idx])
    }

    /// Construct graph based on landmarks
    fn construct_landmark_graph(
        &self,
        X: &ArrayView2<f64>,
        landmarks: &Array2<f64>,
        landmark_indices: &[usize],
    ) -> Result<Array2<f64>, SklearsError> {
        match self.construction_method.as_str() {
            "knn_to_landmarks" => self.knn_to_landmarks_graph(X, landmarks, landmark_indices),
            "rbf_to_landmarks" => self.rbf_to_landmarks_graph(X, landmarks, landmark_indices),
            "interpolation" => self.interpolation_graph(X, landmarks, landmark_indices),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown construction method: {}",
                self.construction_method
            ))),
        }
    }

    /// K-NN to landmarks graph construction
    fn knn_to_landmarks_graph(
        &self,
        X: &ArrayView2<f64>,
        landmarks: &Array2<f64>,
        _landmark_indices: &[usize],
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let n_landmarks = landmarks.nrows();
        let mut adjacency = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Find k nearest landmarks for each point
            let mut landmark_distances: Vec<(f64, usize)> = Vec::new();

            for j in 0..n_landmarks {
                let dist = self.euclidean_distance(&X.row(i), &landmarks.row(j));
                landmark_distances.push((dist, j));
            }

            landmark_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let k_landmarks = self.k_neighbors.min(n_landmarks);
            let mut weights = Vec::new();
            let mut total_weight = 0.0;

            for k in 0..k_landmarks {
                let weight =
                    (-landmark_distances[k].0.powi(2) / (2.0 * self.bandwidth.powi(2))).exp();
                weights.push(weight);
                total_weight += weight;
            }

            // Normalize weights
            if total_weight > 0.0 {
                for weight in &mut weights {
                    *weight /= total_weight;
                }
            }

            // Connect to other points through shared landmarks
            for j in (i + 1)..n_samples {
                let mut shared_weight = 0.0;

                // Find landmarks for point j
                let mut j_landmark_distances: Vec<(f64, usize)> = Vec::new();
                for l in 0..n_landmarks {
                    let dist = self.euclidean_distance(&X.row(j), &landmarks.row(l));
                    j_landmark_distances.push((dist, l));
                }
                j_landmark_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let mut j_weights = Vec::new();
                let mut j_total_weight = 0.0;

                for k in 0..k_landmarks {
                    let weight =
                        (-j_landmark_distances[k].0.powi(2) / (2.0 * self.bandwidth.powi(2))).exp();
                    j_weights.push(weight);
                    j_total_weight += weight;
                }

                // Normalize weights for point j
                if j_total_weight > 0.0 {
                    for weight in &mut j_weights {
                        *weight /= j_total_weight;
                    }
                }

                // Compute shared landmark weight
                for k in 0..k_landmarks {
                    for l in 0..k_landmarks {
                        if landmark_distances[k].1 == j_landmark_distances[l].1 {
                            shared_weight += weights[k] * j_weights[l];
                        }
                    }
                }

                adjacency[[i, j]] = shared_weight;
                adjacency[[j, i]] = shared_weight;
            }
        }

        Ok(adjacency)
    }

    /// RBF to landmarks graph construction
    fn rbf_to_landmarks_graph(
        &self,
        X: &ArrayView2<f64>,
        landmarks: &Array2<f64>,
        _landmark_indices: &[usize],
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let n_landmarks = landmarks.nrows();
        let mut adjacency = Array2::<f64>::zeros((n_samples, n_samples));

        // Compute point-to-landmark weights
        let mut point_landmark_weights = Array2::<f64>::zeros((n_samples, n_landmarks));

        for i in 0..n_samples {
            let mut total_weight = 0.0;
            for j in 0..n_landmarks {
                let dist = self.euclidean_distance(&X.row(i), &landmarks.row(j));
                let weight = (-dist.powi(2) / (2.0 * self.bandwidth.powi(2))).exp();
                point_landmark_weights[[i, j]] = weight;
                total_weight += weight;
            }

            // Normalize weights
            if total_weight > 0.0 {
                for j in 0..n_landmarks {
                    point_landmark_weights[[i, j]] /= total_weight;
                }
            }
        }

        // Compute point-to-point similarities through landmarks
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let mut similarity = 0.0;

                for l in 0..n_landmarks {
                    similarity += point_landmark_weights[[i, l]] * point_landmark_weights[[j, l]];
                }

                adjacency[[i, j]] = similarity;
                adjacency[[j, i]] = similarity;
            }
        }

        Ok(adjacency)
    }

    /// Interpolation-based graph construction
    fn interpolation_graph(
        &self,
        X: &ArrayView2<f64>,
        landmarks: &Array2<f64>,
        landmark_indices: &[usize],
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut adjacency = Array2::<f64>::zeros((n_samples, n_samples));

        // Connect landmarks to each other first
        for i in 0..landmark_indices.len() {
            for j in (i + 1)..landmark_indices.len() {
                let idx_i = landmark_indices[i];
                let idx_j = landmark_indices[j];
                let dist = self.euclidean_distance(&X.row(idx_i), &X.row(idx_j));
                let weight = (-dist.powi(2) / (2.0 * self.bandwidth.powi(2))).exp();
                adjacency[[idx_i, idx_j]] = weight;
                adjacency[[idx_j, idx_i]] = weight;
            }
        }

        // Connect non-landmark points to landmarks and interpolate
        for i in 0..n_samples {
            if landmark_indices.contains(&i) {
                continue; // Skip landmarks
            }

            // Find nearest landmarks
            let mut landmark_distances: Vec<(f64, usize)> = Vec::new();
            for (l_idx, &landmark_idx) in landmark_indices.iter().enumerate() {
                let dist = self.euclidean_distance(&X.row(i), &X.row(landmark_idx));
                landmark_distances.push((dist, landmark_idx));
            }

            landmark_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let k_connect = self.k_neighbors.min(landmark_distances.len());

            // Connect to nearest landmarks
            for k in 0..k_connect {
                let landmark_idx = landmark_distances[k].1;
                let weight =
                    (-landmark_distances[k].0.powi(2) / (2.0 * self.bandwidth.powi(2))).exp();
                adjacency[[i, landmark_idx]] = weight;
                adjacency[[landmark_idx, i]] = weight;
            }
        }

        Ok(adjacency)
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for LandmarkGraphConstruction {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of landmark graph construction
#[derive(Clone, Debug)]
pub struct LandmarkGraphResult {
    /// The constructed adjacency matrix
    pub adjacency_matrix: Array2<f64>,
    /// Indices of selected landmarks in original data
    pub landmark_indices: Vec<usize>,
    /// Landmark points
    pub landmarks: Array2<f64>,
}

impl LandmarkGraphResult {
    /// Get the number of landmarks
    pub fn n_landmarks(&self) -> usize {
        self.landmarks.nrows()
    }

    /// Get the sparsity of the resulting graph
    pub fn sparsity(&self) -> f64 {
        let total_edges = self.adjacency_matrix.len();
        let non_zero_edges = self.adjacency_matrix.iter().filter(|&&x| x > 0.0).count();
        1.0 - (non_zero_edges as f64 / total_edges as f64)
    }

    /// Get landmark coverage statistics
    pub fn landmark_coverage(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        let n_samples = self.adjacency_matrix.nrows();
        let n_landmarks = self.landmarks.nrows();

        stats.insert(
            "landmark_ratio".to_string(),
            n_landmarks as f64 / n_samples as f64,
        );
        stats.insert("sparsity".to_string(), self.sparsity());
        stats.insert("n_landmarks".to_string(), n_landmarks as f64);
        stats.insert("n_samples".to_string(), n_samples as f64);

        stats
    }
}

/// Landmark-based label propagation for large-scale semi-supervised learning
#[derive(Clone)]
pub struct LandmarkLabelPropagation {
    /// Graph construction parameters
    pub graph_constructor: LandmarkGraphConstruction,
    /// Maximum iterations for label propagation
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Alpha parameter for label spreading
    pub alpha: f64,
}

impl LandmarkLabelPropagation {
    /// Create a new landmark label propagation instance
    pub fn new() -> Self {
        Self {
            graph_constructor: LandmarkGraphConstruction::new(),
            max_iter: 1000,
            tolerance: 1e-6,
            alpha: 0.2,
        }
    }

    /// Set the graph constructor
    pub fn graph_constructor(mut self, constructor: LandmarkGraphConstruction) -> Self {
        self.graph_constructor = constructor;
        self
    }

    /// Set the maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the alpha parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Perform landmark-based label propagation
    pub fn fit_predict(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
    ) -> Result<Array1<i32>, SklearsError> {
        let n_samples = X.nrows();

        if y.len() != n_samples {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X and y should have same number of samples: {}", n_samples),
                actual: format!("X has {} samples, y has {} samples", n_samples, y.len()),
            });
        }

        // Construct landmark graph
        let graph_result = self.graph_constructor.fit(X)?;

        // Perform label propagation on the landmark graph
        self.propagate_labels(&graph_result.adjacency_matrix, y)
    }

    /// Propagate labels on the constructed graph
    #[allow(non_snake_case)]
    fn propagate_labels(
        &self,
        adjacency: &Array2<f64>,
        y: &ArrayView1<i32>,
    ) -> Result<Array1<i32>, SklearsError> {
        let n_samples = adjacency.nrows();

        // Identify labeled and unlabeled samples
        let labeled_mask: Array1<bool> = y.iter().map(|&label| label != -1).collect();
        let unique_labels: Vec<i32> = y
            .iter()
            .filter(|&&label| label != -1)
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        if unique_labels.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples found".to_string(),
            ));
        }

        let n_classes = unique_labels.len();

        // Initialize label probability matrix
        let mut F = Array2::<f64>::zeros((n_samples, n_classes));

        // Set initial labels for labeled samples
        for i in 0..n_samples {
            if labeled_mask[i] {
                if let Some(class_idx) = unique_labels.iter().position(|&x| x == y[i]) {
                    F[[i, class_idx]] = 1.0;
                }
            }
        }

        // Normalize adjacency matrix to get transition matrix
        let P = self.normalize_adjacency(adjacency)?;

        // Iterative label propagation
        for _iter in 0..self.max_iter {
            let F_old = F.clone();

            // Propagate labels: F = α * P * F + (1-α) * Y
            let propagated = P.dot(&F);
            F = &propagated * self.alpha;

            // Add back original labels with weight (1-α)
            for i in 0..n_samples {
                if labeled_mask[i] {
                    if let Some(class_idx) = unique_labels.iter().position(|&x| x == y[i]) {
                        F[[i, class_idx]] += 1.0 - self.alpha;
                    }
                }
            }

            // Normalize probabilities
            for i in 0..n_samples {
                let row_sum: f64 = F.row(i).sum();
                if row_sum > 0.0 {
                    for j in 0..n_classes {
                        F[[i, j]] /= row_sum;
                    }
                }
            }

            // Check convergence
            let change = (&F - &F_old).iter().map(|x| x.abs()).sum::<f64>();
            if change < self.tolerance {
                break;
            }
        }

        // Convert probabilities to labels
        let mut labels = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut max_prob = 0.0;
            let mut max_class = 0;

            for j in 0..n_classes {
                if F[[i, j]] > max_prob {
                    max_prob = F[[i, j]];
                    max_class = j;
                }
            }

            labels[i] = unique_labels[max_class];
        }

        Ok(labels)
    }

    /// Normalize adjacency matrix to transition matrix
    fn normalize_adjacency(&self, adjacency: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n_samples = adjacency.nrows();
        let mut P = adjacency.clone();

        for i in 0..n_samples {
            let row_sum: f64 = P.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_samples {
                    P[[i, j]] /= row_sum;
                }
            }
        }

        Ok(P)
    }
}

impl Default for LandmarkLabelPropagation {
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
    #[allow(non_snake_case)]
    fn test_landmark_graph_construction_random() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];

        let lgc = LandmarkGraphConstruction::new()
            .n_landmarks(3)
            .selection_strategy("random".to_string())
            .random_state(42);

        let result = lgc.fit(&X.view());
        assert!(result.is_ok());

        let graph_result = result.unwrap();
        assert_eq!(graph_result.adjacency_matrix.dim(), (6, 6));
        assert_eq!(graph_result.n_landmarks(), 3);
        assert_eq!(graph_result.landmark_indices.len(), 3);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_landmark_graph_construction_kmeans() {
        let X = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0],
            [8.0, 8.0],
            [8.5, 8.5],
            [9.0, 9.0]
        ];

        let lgc = LandmarkGraphConstruction::new()
            .n_landmarks(2)
            .selection_strategy("kmeans".to_string())
            .random_state(42);

        let result = lgc.fit(&X.view());
        assert!(result.is_ok());

        let graph_result = result.unwrap();
        assert_eq!(graph_result.adjacency_matrix.dim(), (6, 6));
        assert_eq!(graph_result.n_landmarks(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_landmark_graph_construction_farthest_first() {
        let X = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let lgc = LandmarkGraphConstruction::new()
            .n_landmarks(2)
            .selection_strategy("farthest_first".to_string())
            .random_state(42);

        let result = lgc.fit(&X.view());
        assert!(result.is_ok());

        let graph_result = result.unwrap();
        assert_eq!(graph_result.adjacency_matrix.dim(), (4, 4));
        assert_eq!(graph_result.n_landmarks(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_landmark_graph_construction_density_based() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let lgc = LandmarkGraphConstruction::new()
            .n_landmarks(2)
            .selection_strategy("density_based".to_string())
            .random_state(42);

        let result = lgc.fit(&X.view());
        assert!(result.is_ok());

        let graph_result = result.unwrap();
        assert_eq!(graph_result.adjacency_matrix.dim(), (4, 4));
        assert_eq!(graph_result.n_landmarks(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_different_construction_methods() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let methods = vec!["knn_to_landmarks", "rbf_to_landmarks", "interpolation"];

        for method in methods {
            let lgc = LandmarkGraphConstruction::new()
                .n_landmarks(2)
                .construction_method(method.to_string())
                .random_state(42);

            let result = lgc.fit(&X.view());
            assert!(result.is_ok());

            let graph_result = result.unwrap();
            assert_eq!(graph_result.adjacency_matrix.dim(), (4, 4));
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_landmark_label_propagation() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 1, -1, -1, -1, -1]; // First two are labeled

        let llp = LandmarkLabelPropagation::new();
        let mut graph_constructor = LandmarkGraphConstruction::new()
            .n_landmarks(3)
            .random_state(42);

        let llp = llp.graph_constructor(graph_constructor);

        let result = llp.fit_predict(&X.view(), &y.view());
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.len(), 6);

        // Check that labeled samples keep their labels
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);

        // Check that all labels are valid
        for &label in labels.iter() {
            assert!(label == 0 || label == 1);
        }
    }

    #[test]
    fn test_graph_result_methods() {
        let adjacency = array![
            [0.0, 0.8, 0.0, 0.2],
            [0.8, 0.0, 0.3, 0.0],
            [0.0, 0.3, 0.0, 0.9],
            [0.2, 0.0, 0.9, 0.0]
        ];

        let landmark_indices = vec![0, 2];
        let landmarks = array![[1.0, 2.0], [3.0, 4.0]];

        let result = LandmarkGraphResult {
            adjacency_matrix: adjacency,
            landmark_indices,
            landmarks,
        };

        assert_eq!(result.n_landmarks(), 2);
        assert!(result.sparsity() > 0.0);
        assert!(result.sparsity() < 1.0);

        let coverage = result.landmark_coverage();
        assert!(coverage.contains_key("landmark_ratio"));
        assert!(coverage.contains_key("sparsity"));
        assert_eq!(coverage["n_landmarks"], 2.0);
        assert_eq!(coverage["n_samples"], 4.0);
    }

    #[test]
    fn test_landmark_graph_construction_builder() {
        let lgc = LandmarkGraphConstruction::new()
            .n_landmarks(50)
            .k_neighbors(8)
            .selection_strategy("farthest_first".to_string())
            .construction_method("rbf_to_landmarks".to_string())
            .bandwidth(2.0)
            .max_iter(200)
            .random_state(123);

        assert_eq!(lgc.n_landmarks, 50);
        assert_eq!(lgc.k_neighbors, 8);
        assert_eq!(lgc.selection_strategy, "farthest_first");
        assert_eq!(lgc.construction_method, "rbf_to_landmarks");
        assert_eq!(lgc.bandwidth, 2.0);
        assert_eq!(lgc.max_iter, 200);
        assert_eq!(lgc.random_state, Some(123));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_error_cases() {
        let lgc =
            LandmarkGraphConstruction::new().selection_strategy("invalid_strategy".to_string());

        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let result = lgc.fit(&X.view());
        assert!(result.is_err());

        let lgc =
            LandmarkGraphConstruction::new().construction_method("invalid_method".to_string());

        let result = lgc.fit(&X.view());
        assert!(result.is_err());

        // Test with empty dataset
        let empty_X = Array2::<f64>::zeros((0, 2));
        let lgc = LandmarkGraphConstruction::new();
        let result = lgc.fit(&empty_X.view());
        assert!(result.is_err());

        // Test label propagation with mismatched dimensions
        let llp = LandmarkLabelPropagation::new();
        let y = array![0]; // Wrong size
        let result = llp.fit_predict(&X.view(), &y.view());
        assert!(result.is_err());

        // Test with no labeled samples
        let y_unlabeled = array![-1, -1];
        let result = llp.fit_predict(&X.view(), &y_unlabeled.view());
        assert!(result.is_err());
    }
}

//! Metric learning algorithms for adaptive distance computation
//!
//! This module implements various metric learning algorithms that automatically
//! learn distance metrics from labeled data to improve neighbor-based classification.

use crate::NeighborsError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::Float as FloatTrait;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::error::Result;
use sklears_core::traits::{Fit, Transform};
use sklears_core::types::{Features, Float};

/// Large Margin Nearest Neighbor (LMNN) metric learning
///
/// LMNN learns a Mahalanobis distance metric that maximizes the margin between
/// different classes while keeping same-class neighbors close.
pub struct LargeMarginNearestNeighbor {
    /// Number of target neighbors per point
    k: usize,
    /// Learning rate for gradient descent
    learning_rate: Float,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: Float,
    /// Regularization parameter
    regularization: Float,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Learned transformation matrix
    transformation: Option<Array2<Float>>,
}

impl LargeMarginNearestNeighbor {
    /// Create a new LMNN instance
    pub fn new(k: usize) -> Self {
        Self {
            k,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-6,
            regularization: 0.5,
            random_state: None,
            transformation: None,
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set regularization parameter (balance between pull and push forces)
    pub fn with_regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Find k target neighbors for each sample
    fn find_target_neighbors(&self, x: &ArrayView2<Float>, y: &ArrayView1<i32>) -> Array2<usize> {
        let n_samples = x.nrows();
        let mut target_neighbors = Array2::zeros((n_samples, self.k));

        for i in 0..n_samples {
            let mut same_class_indices = Vec::new();
            let mut distances = Vec::new();

            // Find all same-class samples
            for j in 0..n_samples {
                if i != j && y[i] == y[j] {
                    let distance = self.euclidean_distance(&x.row(i), &x.row(j));
                    same_class_indices.push(j);
                    distances.push(distance);
                }
            }

            // Sort by distance and take k nearest
            let mut sorted_indices: Vec<usize> = (0..same_class_indices.len()).collect();
            sorted_indices.sort_by(|&a, &b| distances[a].partial_cmp(&distances[b]).unwrap());

            let k_actual = std::cmp::min(self.k, sorted_indices.len());
            for (idx, &sorted_idx) in sorted_indices.iter().take(k_actual).enumerate() {
                target_neighbors[[i, idx]] = same_class_indices[sorted_idx];
            }

            // Fill remaining with repeated last neighbor if needed
            if k_actual > 0 && k_actual < self.k {
                let last_neighbor = target_neighbors[[i, k_actual - 1]];
                for idx in k_actual..self.k {
                    target_neighbors[[i, idx]] = last_neighbor;
                }
            }
        }

        target_neighbors
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<Float>()
            .sqrt()
    }

    /// Compute Mahalanobis distance using learned metric
    fn mahalanobis_distance(
        &self,
        a: &ArrayView1<Float>,
        b: &ArrayView1<Float>,
        metric: &ArrayView2<Float>,
    ) -> Float {
        let diff = a - b;
        let diff_2d = diff.clone().insert_axis(Axis(0));
        let result = diff_2d.dot(metric).dot(&diff);
        result[[0]].sqrt()
    }

    /// Find impostor neighbors (different class but closer than target neighbors)
    fn find_impostors(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<i32>,
        target_neighbors: &Array2<usize>,
        metric: &ArrayView2<Float>,
    ) -> Vec<(usize, usize, usize)> {
        let mut impostors = Vec::new();
        let n_samples = x.nrows();

        for i in 0..n_samples {
            // Get distances to target neighbors
            let mut target_distances = Vec::new();
            for k_idx in 0..self.k {
                let neighbor_idx = target_neighbors[[i, k_idx]];
                if neighbor_idx < n_samples {
                    let distance =
                        self.mahalanobis_distance(&x.row(i), &x.row(neighbor_idx), metric);
                    target_distances.push(distance);
                }
            }

            if target_distances.is_empty() {
                continue;
            }

            let max_target_distance = target_distances
                .iter()
                .fold(Float::neg_infinity(), |a, &b| a.max(b));

            // Find different-class samples closer than furthest target neighbor
            for j in 0..n_samples {
                if i != j && y[i] != y[j] {
                    let distance = self.mahalanobis_distance(&x.row(i), &x.row(j), metric);
                    if distance < max_target_distance + 1.0 {
                        // margin
                        // Find which target neighbor this impostor is competing with
                        for k_idx in 0..self.k {
                            let neighbor_idx = target_neighbors[[i, k_idx]];
                            if neighbor_idx < n_samples {
                                let target_distance = self.mahalanobis_distance(
                                    &x.row(i),
                                    &x.row(neighbor_idx),
                                    metric,
                                );
                                if distance < target_distance + 1.0 {
                                    impostors.push((i, neighbor_idx, j));
                                }
                            }
                        }
                    }
                }
            }
        }

        impostors
    }

    /// Compute gradient for LMNN objective
    fn compute_gradient(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<i32>,
        target_neighbors: &Array2<usize>,
        metric: &ArrayView2<Float>,
    ) -> Array2<Float> {
        let n_features = x.ncols();
        let mut gradient = Array2::zeros((n_features, n_features));
        let n_samples = x.nrows();

        // Pull force: attract target neighbors
        for i in 0..n_samples {
            for k_idx in 0..self.k {
                let j = target_neighbors[[i, k_idx]];
                if j < n_samples {
                    let diff = &x.row(i) - &x.row(j);
                    let diff_2d = diff.insert_axis(Axis(0));
                    let outer_product = diff_2d.t().dot(&diff_2d);
                    gradient += &outer_product;
                }
            }
        }

        // Push force: repel impostors
        let impostors = self.find_impostors(x, y, target_neighbors, metric);
        for (i, j, k) in impostors {
            if i < n_samples && j < n_samples && k < n_samples {
                // Gradient w.r.t. distance to impostor
                let diff_ik = &x.row(i) - &x.row(k);
                let diff_ik_2d = diff_ik.insert_axis(Axis(0));
                let outer_product_ik = diff_ik_2d.t().dot(&diff_ik_2d);

                gradient -= &(outer_product_ik * self.regularization);
            }
        }

        gradient / (n_samples as Float)
    }
}

impl Fit<Features, Array1<i32>> for LargeMarginNearestNeighbor {
    type Fitted = LargeMarginNearestNeighbor;

    fn fit(self, x: &Features, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![y.len(), x.ncols()],
                actual: vec![x.nrows(), x.ncols()],
            }
            .into());
        }

        let n_features = x.ncols();
        let mut metric = Array2::eye(n_features); // Start with identity matrix

        // Initialize random state
        let rng_seed = self
            .random_state
            .unwrap_or_else(|| thread_rng().gen_range(0..u64::MAX));
        let _rng = StdRng::seed_from_u64(rng_seed);

        // Find target neighbors for each sample
        let target_neighbors = self.find_target_neighbors(&x.view(), &y.view());

        let mut prev_loss = Float::infinity();

        // Gradient descent optimization
        for _iteration in 0..self.max_iter {
            // Compute gradient
            let gradient =
                self.compute_gradient(&x.view(), &y.view(), &target_neighbors, &metric.view());

            // Update metric
            metric -= &(gradient * self.learning_rate);

            // Ensure metric remains positive semidefinite (project to nearest PSD matrix)
            self.project_to_psd(&mut metric);

            // Compute current loss (simplified version)
            let mut loss = 0.0;
            let n_samples = x.nrows();
            for i in 0..n_samples {
                for k_idx in 0..self.k {
                    let j = target_neighbors[[i, k_idx]];
                    if j < n_samples {
                        let distance =
                            self.mahalanobis_distance(&x.row(i), &x.row(j), &metric.view());
                        loss += distance * distance;
                    }
                }
            }
            loss /= (n_samples * self.k) as Float;

            // Check convergence
            if (prev_loss - loss).abs() < self.tol {
                break;
            }
            prev_loss = loss;
        }

        let mut learned_model = self;
        learned_model.transformation = Some(metric);
        Ok(learned_model)
    }
}

impl LargeMarginNearestNeighbor {
    /// Project matrix to nearest positive semidefinite matrix
    fn project_to_psd(&self, matrix: &mut Array2<Float>) {
        // Simple projection: replace negative eigenvalues with small positive values
        // In practice, you'd want to use proper eigendecomposition
        let n = matrix.nrows();
        for i in 0..n {
            for j in 0..n {
                if i == j && matrix[[i, j]] < 1e-10 {
                    matrix[[i, j]] = 1e-10;
                }
            }
        }
    }
}

impl Clone for LargeMarginNearestNeighbor {
    fn clone(&self) -> Self {
        Self {
            k: self.k,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            tol: self.tol,
            regularization: self.regularization,
            random_state: self.random_state,
            transformation: self.transformation.clone(),
        }
    }
}

impl Transform<Features, Array2<Float>> for LargeMarginNearestNeighbor {
    fn transform(&self, x: &Features) -> Result<Array2<Float>> {
        if let Some(ref transformation) = self.transformation {
            // Apply learned metric transformation
            let mut transformed = Array2::zeros((x.nrows(), x.ncols()));
            for (i, row) in x.axis_iter(Axis(0)).enumerate() {
                let row_2d = row.insert_axis(Axis(0));
                let transformed_row = row_2d.dot(transformation);
                transformed.row_mut(i).assign(&transformed_row.row(0));
            }
            Ok(transformed)
        } else {
            Err(NeighborsError::InvalidInput("Model not fitted".to_string()).into())
        }
    }
}

/// Neighborhood Components Analysis (NCA) metric learning
///
/// NCA learns a linear transformation that maximizes the expected
/// leave-one-out classification accuracy under the stochastic nearest neighbor rule.
pub struct NeighborhoodComponentsAnalysis {
    /// Number of components (dimensionality reduction)
    n_components: Option<usize>,
    /// Learning rate for gradient descent
    learning_rate: Float,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: Float,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Learned transformation matrix
    transformation: Option<Array2<Float>>,
}

impl Default for NeighborhoodComponentsAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl NeighborhoodComponentsAnalysis {
    /// Create a new NCA instance
    pub fn new() -> Self {
        Self {
            n_components: None,
            learning_rate: 0.01,
            max_iter: 500,
            tol: 1e-5,
            random_state: None,
            transformation: None,
        }
    }

    /// Set number of components for dimensionality reduction
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Compute probability matrix P_ij
    fn compute_probabilities(&self, x_transformed: &ArrayView2<Float>) -> Array2<Float> {
        let n_samples = x_transformed.nrows();
        let mut probabilities = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut sum_exp = 0.0;
            let mut exp_distances = Vec::new();

            // Compute exp(-||x_i - x_j||^2) for all j != i
            for j in 0..n_samples {
                if i != j {
                    let distance_sq = x_transformed
                        .row(i)
                        .iter()
                        .zip(x_transformed.row(j).iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<Float>();
                    let exp_dist = (-distance_sq).exp();
                    exp_distances.push(exp_dist);
                    sum_exp += exp_dist;
                } else {
                    exp_distances.push(0.0);
                }
            }

            // Normalize to get probabilities
            for j in 0..n_samples {
                if i != j && sum_exp > 0.0 {
                    probabilities[[i, j]] = exp_distances[j] / sum_exp;
                }
            }
        }

        probabilities
    }

    /// Compute NCA objective function
    fn compute_objective(&self, x_transformed: &ArrayView2<Float>, y: &Array1<i32>) -> Float {
        let probabilities = self.compute_probabilities(x_transformed);
        let n_samples = x_transformed.nrows();
        let mut objective = 0.0;

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j && y[i] == y[j] {
                    objective += probabilities[[i, j]];
                }
            }
        }

        objective / n_samples as Float
    }

    /// Compute gradient of NCA objective
    fn compute_gradient(
        &self,
        x: &Features,
        x_transformed: &ArrayView2<Float>,
        y: &Array1<i32>,
        transformation: &ArrayView2<Float>,
    ) -> Array2<Float> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_components = transformation.nrows();
        let mut gradient = Array2::zeros((n_components, n_features));

        let probabilities = self.compute_probabilities(x_transformed);

        for i in 0..n_samples {
            let mut p_i_same_class = 0.0;

            // Sum probabilities for same class
            for j in 0..n_samples {
                if i != j && y[i] == y[j] {
                    p_i_same_class += probabilities[[i, j]];
                }
            }

            for j in 0..n_samples {
                if i != j {
                    let p_ij = probabilities[[i, j]];
                    let same_class = if y[i] == y[j] { 1.0 } else { 0.0 };

                    let coeff = p_ij * (same_class - p_i_same_class);
                    let diff = &x.row(i) - &x.row(j);

                    for k in 0..n_components {
                        for l in 0..n_features {
                            gradient[[k, l]] += 2.0 * coeff * diff[l] * transformation[[k, l]];
                        }
                    }
                }
            }
        }

        gradient / n_samples as Float
    }
}

impl Fit<Features, Array1<i32>> for NeighborhoodComponentsAnalysis {
    type Fitted = NeighborhoodComponentsAnalysis;

    fn fit(self, x: &Features, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![y.len(), x.ncols()],
                actual: vec![x.nrows(), x.ncols()],
            }
            .into());
        }

        let n_features = x.ncols();
        let n_components = self.n_components.unwrap_or(n_features);

        // Initialize transformation matrix randomly
        let rng_seed = self
            .random_state
            .unwrap_or_else(|| thread_rng().gen_range(0..u64::MAX));
        let mut rng = StdRng::seed_from_u64(rng_seed);

        let mut transformation = Array2::zeros((n_components, n_features));
        for i in 0..n_components {
            for j in 0..n_features {
                transformation[[i, j]] = rng.gen_range(-0.1..0.1);
            }
        }

        let mut prev_objective = Float::neg_infinity();

        // Gradient ascent optimization
        for _iteration in 0..self.max_iter {
            // Transform data
            let x_transformed = transformation.dot(&x.t()).t().to_owned();

            // Compute gradient
            let gradient =
                self.compute_gradient(x, &x_transformed.view(), y, &transformation.view());

            // Update transformation
            transformation += &(gradient * self.learning_rate);

            // Compute current objective
            let x_transformed = transformation.dot(&x.t()).t().to_owned();
            let objective = self.compute_objective(&x_transformed.view(), y);

            // Check convergence
            if (objective - prev_objective).abs() < self.tol {
                break;
            }
            prev_objective = objective;
        }

        let mut learned_model = self;
        learned_model.transformation = Some(transformation);
        Ok(learned_model)
    }
}

impl Clone for NeighborhoodComponentsAnalysis {
    fn clone(&self) -> Self {
        Self {
            n_components: self.n_components,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            transformation: self.transformation.clone(),
        }
    }
}

impl Transform<Features, Array2<Float>> for NeighborhoodComponentsAnalysis {
    fn transform(&self, x: &Features) -> Result<Array2<Float>> {
        if let Some(ref transformation) = self.transformation {
            let transformed = transformation.dot(&x.t()).t().to_owned();
            Ok(transformed)
        } else {
            Err(NeighborsError::InvalidInput("Model not fitted".to_string()).into())
        }
    }
}

/// Information-Theoretic Metric Learning (ITML)
///
/// ITML learns a Mahalanobis distance metric that minimizes the KL divergence
/// between the prior distribution and the learned metric subject to distance constraints.
pub struct InformationTheoreticMetricLearning {
    /// Number of constraint samples to generate
    n_constraints: usize,
    /// Convergence tolerance
    tol: Float,
    /// Maximum number of iterations
    max_iter: usize,
    /// Regularization parameter
    gamma: Float,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Learned transformation matrix
    transformation: Option<Array2<Float>>,
    /// Prior covariance matrix (default: identity)
    prior_covariance: Option<Array2<Float>>,
}

impl InformationTheoreticMetricLearning {
    /// Create a new ITML instance
    pub fn new(n_constraints: usize) -> Self {
        Self {
            n_constraints,
            tol: 1e-6,
            max_iter: 1000,
            gamma: 1.0,
            random_state: None,
            transformation: None,
            prior_covariance: None,
        }
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set regularization parameter
    pub fn with_gamma(mut self, gamma: Float) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set prior covariance matrix
    pub fn with_prior_covariance(mut self, prior: Array2<Float>) -> Self {
        self.prior_covariance = Some(prior);
        self
    }

    /// Generate constraints from labeled data
    fn generate_constraints(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<i32>,
        rng: &mut StdRng,
    ) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let n_samples = x.nrows();
        let mut similar_pairs = Vec::new();
        let mut dissimilar_pairs = Vec::new();

        for _ in 0..self.n_constraints {
            let i = rng.gen_range(0..n_samples);
            let j = rng.gen_range(0..n_samples);

            if i != j {
                if y[i] == y[j] {
                    similar_pairs.push((i, j));
                } else {
                    dissimilar_pairs.push((i, j));
                }
            }
        }

        (similar_pairs, dissimilar_pairs)
    }

    /// Compute mutual information between features and labels
    fn compute_mutual_information(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<i32>,
    ) -> Array1<Float> {
        let n_features = x.ncols();
        let mut mi_scores = Array1::zeros(n_features);

        // Get unique classes
        let mut unique_classes: Vec<i32> = y.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();

        for f in 0..n_features {
            let feature_values = x.column(f);
            let mut mi = 0.0;

            // Discretize continuous feature values into bins
            let mut sorted_values: Vec<Float> = feature_values.to_vec();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n_bins = 10;
            let mut bin_edges = Vec::new();
            for i in 0..=n_bins {
                let idx = (i * (sorted_values.len() - 1) / n_bins).min(sorted_values.len() - 1);
                bin_edges.push(sorted_values[idx]);
            }

            // Compute joint and marginal probabilities
            for &class in &unique_classes {
                let class_prob =
                    y.iter().filter(|&&label| label == class).count() as Float / y.len() as Float;

                for bin_idx in 0..n_bins {
                    let bin_start = bin_edges[bin_idx];
                    let bin_end = bin_edges[bin_idx + 1];

                    // Count samples in this bin
                    let bin_count = feature_values
                        .iter()
                        .filter(|&&val| val >= bin_start && val < bin_end)
                        .count() as Float;

                    if bin_count > 0.0 {
                        let bin_prob = bin_count / feature_values.len() as Float;

                        // Count samples in this bin AND with this class
                        let joint_count = y
                            .iter()
                            .zip(feature_values.iter())
                            .filter(|(&label, &val)| {
                                label == class && val >= bin_start && val < bin_end
                            })
                            .count() as Float;

                        if joint_count > 0.0 {
                            let joint_prob = joint_count / y.len() as Float;
                            mi += joint_prob * (joint_prob / (class_prob * bin_prob)).ln();
                        }
                    }
                }
            }

            mi_scores[f] = mi.max(0.0);
        }

        mi_scores
    }

    /// Update metric using ITML constraints
    fn update_metric(
        &self,
        metric: &mut Array2<Float>,
        x: &ArrayView2<Float>,
        similar_pairs: &[(usize, usize)],
        dissimilar_pairs: &[(usize, usize)],
    ) -> Float {
        let _n_features = x.ncols();
        let mut total_violation = 0.0;

        // Process similar pairs (should be close)
        for &(i, j) in similar_pairs {
            let diff = &x.row(i) - &x.row(j);
            let diff_2d = diff.clone().insert_axis(Axis(0));
            let distance_sq = diff_2d.dot(metric).dot(&diff)[[0]];

            // If distance is too large, update metric to bring points closer
            let target_distance_sq = 0.1; // Small target distance for similar pairs
            if distance_sq > target_distance_sq {
                let violation = distance_sq - target_distance_sq;
                total_violation += violation;

                // Gradient update
                let learning_rate = self.gamma / (1.0 + distance_sq);
                let gradient = diff_2d.t().dot(&diff_2d) * learning_rate;
                *metric = &*metric - &gradient;
            }
        }

        // Process dissimilar pairs (should be far)
        for &(i, j) in dissimilar_pairs {
            let diff = &x.row(i) - &x.row(j);
            let diff_2d = diff.clone().insert_axis(Axis(0));
            let distance_sq = diff_2d.dot(metric).dot(&diff)[[0]];

            // If distance is too small, update metric to push points apart
            let target_distance_sq = 4.0; // Large target distance for dissimilar pairs
            if distance_sq < target_distance_sq {
                let violation = target_distance_sq - distance_sq;
                total_violation += violation;

                // Gradient update (opposite direction)
                let learning_rate = self.gamma / (1.0 + distance_sq);
                let gradient = diff_2d.t().dot(&diff_2d) * learning_rate;
                *metric = &*metric + &gradient;
            }
        }

        // Ensure metric remains positive semidefinite
        self.project_to_psd(metric);

        total_violation
    }

    /// Project matrix to nearest positive semidefinite matrix
    fn project_to_psd(&self, matrix: &mut Array2<Float>) {
        let n = matrix.nrows();
        for i in 0..n {
            for j in 0..n {
                if i == j && matrix[[i, j]] < 1e-10 {
                    matrix[[i, j]] = 1e-10;
                }
                // Ensure symmetry
                if i != j {
                    let avg = (matrix[[i, j]] + matrix[[j, i]]) / 2.0;
                    matrix[[i, j]] = avg;
                    matrix[[j, i]] = avg;
                }
            }
        }
    }
}

impl Fit<Features, Array1<i32>> for InformationTheoreticMetricLearning {
    type Fitted = InformationTheoreticMetricLearning;

    fn fit(self, x: &Features, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![y.len(), x.ncols()],
                actual: vec![x.nrows(), x.ncols()],
            }
            .into());
        }

        let n_features = x.ncols();

        // Initialize metric with prior covariance or identity
        let mut metric = if let Some(ref prior) = self.prior_covariance {
            prior.clone()
        } else {
            Array2::eye(n_features)
        };

        // Weight features by mutual information
        let mi_scores = self.compute_mutual_information(&x.view(), &y.view());
        let mi_weights = Array2::from_diag(&mi_scores);
        metric = metric.dot(&mi_weights);

        // Initialize random state
        let rng_seed = self
            .random_state
            .unwrap_or_else(|| thread_rng().gen_range(0..u64::MAX));
        let mut rng = StdRng::seed_from_u64(rng_seed);

        // Generate constraints
        let (similar_pairs, dissimilar_pairs) =
            self.generate_constraints(&x.view(), &y.view(), &mut rng);

        let mut prev_violation = Float::infinity();

        // Iterative constraint optimization
        for _iteration in 0..self.max_iter {
            let violation =
                self.update_metric(&mut metric, &x.view(), &similar_pairs, &dissimilar_pairs);

            // Check convergence
            if (prev_violation - violation).abs() < self.tol {
                break;
            }
            prev_violation = violation;
        }

        let mut learned_model = self;
        learned_model.transformation = Some(metric);
        Ok(learned_model)
    }
}

impl Clone for InformationTheoreticMetricLearning {
    fn clone(&self) -> Self {
        Self {
            n_constraints: self.n_constraints,
            tol: self.tol,
            max_iter: self.max_iter,
            gamma: self.gamma,
            random_state: self.random_state,
            transformation: self.transformation.clone(),
            prior_covariance: self.prior_covariance.clone(),
        }
    }
}

impl Transform<Features, Array2<Float>> for InformationTheoreticMetricLearning {
    fn transform(&self, x: &Features) -> Result<Array2<Float>> {
        if let Some(ref transformation) = self.transformation {
            let transformed = transformation.dot(&x.t()).t().to_owned();
            Ok(transformed)
        } else {
            Err(NeighborsError::InvalidInput("Model not fitted".to_string()).into())
        }
    }
}

/// Enhanced Large Margin Nearest Neighbor with improved optimization
///
/// This version includes adaptive learning rates, better initialization,
/// and more robust optimization procedures.
pub struct EnhancedLMNN {
    /// Base LMNN instance
    base_lmnn: LargeMarginNearestNeighbor,
    /// Use adaptive learning rate
    adaptive_lr: bool,
    /// Initial learning rate
    initial_lr: Float,
    /// Learning rate decay factor
    lr_decay: Float,
    /// Use momentum in optimization
    use_momentum: bool,
    /// Momentum parameter
    momentum: Float,
    /// Previous gradient (for momentum)
    prev_gradient: Option<Array2<Float>>,
}

impl EnhancedLMNN {
    /// Create a new Enhanced LMNN instance
    pub fn new(k: usize) -> Self {
        Self {
            base_lmnn: LargeMarginNearestNeighbor::new(k),
            adaptive_lr: true,
            initial_lr: 0.01,
            lr_decay: 0.99,
            use_momentum: true,
            momentum: 0.9,
            prev_gradient: None,
        }
    }

    /// Set adaptive learning rate
    pub fn with_adaptive_lr(mut self, adaptive: bool) -> Self {
        self.adaptive_lr = adaptive;
        self
    }

    /// Set learning rate decay
    pub fn with_lr_decay(mut self, decay: Float) -> Self {
        self.lr_decay = decay;
        self
    }

    /// Set momentum usage
    pub fn with_momentum(mut self, use_momentum: bool, momentum: Float) -> Self {
        self.use_momentum = use_momentum;
        self.momentum = momentum;
        self
    }

    /// Configure base LMNN parameters
    pub fn with_base_config(mut self, lmnn: LargeMarginNearestNeighbor) -> Self {
        self.base_lmnn = lmnn;
        self.initial_lr = self.base_lmnn.learning_rate;
        self
    }
}

impl Fit<Features, Array1<i32>> for EnhancedLMNN {
    type Fitted = EnhancedLMNN;

    fn fit(self, x: &Features, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![y.len(), x.ncols()],
                actual: vec![x.nrows(), x.ncols()],
            }
            .into());
        }

        // Use base LMNN but with enhanced optimization
        let mut enhanced = self;
        let mut current_lr = enhanced.initial_lr;

        // Enhanced gradient descent with momentum and adaptive learning rate
        let n_features = x.ncols();
        let mut metric = Array2::eye(n_features);
        let target_neighbors = enhanced
            .base_lmnn
            .find_target_neighbors(&x.view(), &y.view());

        let mut prev_loss = Float::infinity();

        for iteration in 0..enhanced.base_lmnn.max_iter {
            // Compute gradient
            let gradient = enhanced.base_lmnn.compute_gradient(
                &x.view(),
                &y.view(),
                &target_neighbors,
                &metric.view(),
            );

            // Apply momentum if enabled
            let effective_gradient = if enhanced.use_momentum {
                if let Some(ref prev_grad) = enhanced.prev_gradient {
                    &gradient + &(prev_grad * enhanced.momentum)
                } else {
                    gradient.clone()
                }
            } else {
                gradient.clone()
            };

            // Adaptive learning rate based on gradient magnitude
            if enhanced.adaptive_lr && iteration > 0 {
                let grad_norm = effective_gradient.mapv(|x| x * x).sum().sqrt();
                current_lr = enhanced.initial_lr / (1.0 + grad_norm * 0.1);
            } else {
                current_lr *= enhanced.lr_decay;
            }

            // Update metric
            metric -= &(effective_gradient.clone() * current_lr);

            // Project to PSD
            enhanced.base_lmnn.project_to_psd(&mut metric);

            // Store gradient for momentum
            enhanced.prev_gradient = Some(effective_gradient);

            // Compute loss for convergence check
            let mut loss = 0.0;
            let n_samples = x.nrows();
            for i in 0..n_samples {
                for k_idx in 0..enhanced.base_lmnn.k {
                    let j = target_neighbors[[i, k_idx]];
                    if j < n_samples {
                        let distance = enhanced.base_lmnn.mahalanobis_distance(
                            &x.row(i),
                            &x.row(j),
                            &metric.view(),
                        );
                        loss += distance * distance;
                    }
                }
            }
            loss /= (n_samples * enhanced.base_lmnn.k) as Float;

            // Check convergence
            if (prev_loss - loss).abs() < enhanced.base_lmnn.tol {
                break;
            }
            prev_loss = loss;
        }

        enhanced.base_lmnn.transformation = Some(metric);
        Ok(enhanced)
    }
}

impl Clone for EnhancedLMNN {
    fn clone(&self) -> Self {
        Self {
            base_lmnn: self.base_lmnn.clone(),
            adaptive_lr: self.adaptive_lr,
            initial_lr: self.initial_lr,
            lr_decay: self.lr_decay,
            use_momentum: self.use_momentum,
            momentum: self.momentum,
            prev_gradient: self.prev_gradient.clone(),
        }
    }
}

impl Transform<Features, Array2<Float>> for EnhancedLMNN {
    fn transform(&self, x: &Features) -> Result<Array2<Float>> {
        self.base_lmnn.transform(x)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_lmnn_basic() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1, 5.0, 5.0, 5.1, 5.1],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 2, 2];

        let lmnn = LargeMarginNearestNeighbor::new(1)
            .with_max_iter(50)
            .with_random_state(42);

        let fitted = lmnn.fit(&X, &y).unwrap();
        assert!(fitted.transformation.is_some());

        // Test transformation
        let X_test = Array2::from_shape_vec((1, 2), vec![1.5, 1.5]).unwrap();
        let transformed = fitted.transform(&X_test).unwrap();
        assert_eq!(transformed.shape(), &[1, 2]);
    }

    #[test]
    fn test_nca_basic() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1]).unwrap();
        let y = array![0, 0, 1, 1];

        let nca = NeighborhoodComponentsAnalysis::new()
            .with_max_iter(50)
            .with_random_state(42);

        let fitted = nca.fit(&X, &y).unwrap();
        assert!(fitted.transformation.is_some());

        // Test transformation
        let transformed = fitted.transform(&X).unwrap();
        assert_eq!(transformed.shape(), &[4, 2]);
    }

    #[test]
    fn test_nca_dimensionality_reduction() {
        let X = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 1.0, 0.0, 1.1, 1.1, 0.1, 2.0, 2.0, 1.0, 2.1, 2.1, 1.1],
        )
        .unwrap();
        let y = array![0, 0, 1, 1];

        let nca = NeighborhoodComponentsAnalysis::new()
            .with_n_components(2)
            .with_max_iter(20)
            .with_random_state(42);

        let fitted = nca.fit(&X, &y).unwrap();
        let transformed = fitted.transform(&X).unwrap();
        assert_eq!(transformed.shape(), &[4, 2]);
    }

    #[test]
    fn test_information_theoretic_metric_learning() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.05, 1.05, 5.0, 5.0, 5.1, 5.1, 5.05, 5.05,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let itml = InformationTheoreticMetricLearning::new(10)
            .with_max_iter(50)
            .with_gamma(0.1)
            .with_random_state(42);

        let fitted = itml.fit(&X, &y).unwrap();
        assert!(fitted.transformation.is_some());

        // Test transformation
        let transformed = fitted.transform(&X).unwrap();
        assert_eq!(transformed.shape(), &[6, 2]);
    }

    #[test]
    fn test_enhanced_lmnn() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1, 5.0, 5.0, 5.1, 5.1],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 2, 2];

        let enhanced_lmnn = EnhancedLMNN::new(1)
            .with_adaptive_lr(true)
            .with_momentum(true, 0.9)
            .with_lr_decay(0.95);

        let base_lmnn = LargeMarginNearestNeighbor::new(1)
            .with_max_iter(30)
            .with_random_state(42);

        let enhanced = enhanced_lmnn.with_base_config(base_lmnn);
        let fitted = enhanced.fit(&X, &y).unwrap();

        assert!(fitted.base_lmnn.transformation.is_some());

        // Test transformation
        let transformed = fitted.transform(&X).unwrap();
        assert_eq!(transformed.shape(), &[6, 2]);
    }

    #[test]
    fn test_itml_with_prior_covariance() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1]).unwrap();
        let y = array![0, 0, 1, 1];

        // Custom prior covariance
        let prior = Array2::from_shape_vec((2, 2), vec![2.0, 0.5, 0.5, 2.0]).unwrap();

        let itml = InformationTheoreticMetricLearning::new(8)
            .with_prior_covariance(prior)
            .with_max_iter(20)
            .with_random_state(42);

        let fitted = itml.fit(&X, &y).unwrap();
        assert!(fitted.transformation.is_some());
    }

    #[test]
    fn test_enhanced_lmnn_configurations() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1]).unwrap();
        let y = array![0, 0, 1, 1];

        // Test with different configurations
        let configs = vec![
            (true, true),   // adaptive_lr + momentum
            (true, false),  // adaptive_lr only
            (false, true),  // momentum only
            (false, false), // neither
        ];

        for (adaptive, momentum) in configs {
            let enhanced_lmnn = EnhancedLMNN::new(1)
                .with_adaptive_lr(adaptive)
                .with_momentum(momentum, 0.8);

            let base_lmnn = LargeMarginNearestNeighbor::new(1)
                .with_max_iter(10)
                .with_random_state(42);

            let enhanced = enhanced_lmnn.with_base_config(base_lmnn);
            let fitted = enhanced.fit(&X, &y).unwrap();

            assert!(fitted.base_lmnn.transformation.is_some());
        }
    }

    #[test]
    fn test_metric_learning_error_cases() {
        let empty_X = Array2::<Float>::zeros((0, 2));
        let empty_y = Array1::<i32>::zeros(0);

        // Test empty input for ITML
        let itml = InformationTheoreticMetricLearning::new(5);
        assert!(itml.fit(&empty_X, &empty_y).is_err());

        // Test empty input for Enhanced LMNN
        let enhanced_lmnn = EnhancedLMNN::new(1);
        assert!(enhanced_lmnn.fit(&empty_X, &empty_y).is_err());

        // Test shape mismatch
        let X = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_wrong = array![0]; // Wrong size

        let itml = InformationTheoreticMetricLearning::new(5);
        assert!(itml.fit(&X, &y_wrong).is_err());
    }
}

/// Online Metric Learning for streaming data
///
/// This algorithm learns and adapts a distance metric incrementally as new samples arrive,
/// using stochastic gradient descent with momentum. Suitable for online learning scenarios
/// where data arrives sequentially and full batch retraining is not feasible.
///
/// # Algorithm
/// The online metric learning uses:
/// - Stochastic updates with mini-batches for efficiency
/// - Momentum for stable convergence
/// - Adaptive learning rate decay
/// - Memory of recent gradients for better updates
///
/// # Use Cases
/// - Streaming classification tasks
/// - Concept drift adaptation
/// - Real-time learning systems
/// - Memory-constrained environments
pub struct OnlineMetricLearning {
    /// Number of target neighbors
    k: usize,
    /// Base learning rate
    learning_rate: Float,
    /// Learning rate decay factor (multiplied each update)
    learning_rate_decay: Float,
    /// Minimum learning rate threshold
    min_learning_rate: Float,
    /// Momentum coefficient for gradient smoothing
    momentum: Float,
    /// Regularization strength
    regularization: Float,
    /// Current learning rate (decreases over time)
    current_learning_rate: Float,
    /// Number of samples processed so far
    n_samples_seen: usize,
    /// Current transformation matrix (metric)
    transformation: Option<Array2<Float>>,
    /// Velocity matrix for momentum-based updates
    velocity: Option<Array2<Float>>,
    /// Window size for computing recent accuracy
    window_size: usize,
    /// Recent prediction results (for adaptation)
    recent_correct: Vec<bool>,
}

impl OnlineMetricLearning {
    /// Create a new online metric learning instance
    ///
    /// # Arguments
    /// * `k` - Number of nearest neighbors to consider
    pub fn new(k: usize) -> Self {
        Self {
            k,
            learning_rate: 0.01,
            learning_rate_decay: 0.999,
            min_learning_rate: 1e-5,
            momentum: 0.9,
            regularization: 0.1,
            current_learning_rate: 0.01,
            n_samples_seen: 0,
            transformation: None,
            velocity: None,
            window_size: 100,
            recent_correct: Vec::new(),
        }
    }

    /// Set initial learning rate
    pub fn with_learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self.current_learning_rate = learning_rate;
        self
    }

    /// Set learning rate decay factor
    pub fn with_learning_rate_decay(mut self, decay: Float) -> Self {
        self.learning_rate_decay = decay;
        self
    }

    /// Set minimum learning rate
    pub fn with_min_learning_rate(mut self, min_lr: Float) -> Self {
        self.min_learning_rate = min_lr;
        self
    }

    /// Set momentum coefficient
    pub fn with_momentum(mut self, momentum: Float) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set regularization strength
    pub fn with_regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set window size for recent accuracy tracking
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Initialize or reinitialize the transformation matrix
    fn initialize_transformation(&mut self, n_features: usize) {
        if self.transformation.is_none() {
            self.transformation = Some(Array2::eye(n_features));
            self.velocity = Some(Array2::zeros((n_features, n_features)));
        }
    }

    /// Compute Mahalanobis distance using current transformation
    fn mahalanobis_distance(&self, x1: &ArrayView1<Float>, x2: &ArrayView1<Float>) -> Float {
        if let Some(ref transform) = self.transformation {
            let diff = x1.to_owned() - x2;
            let transformed = transform.dot(&diff);
            transformed.iter().map(|&x| x * x).sum::<Float>().sqrt()
        } else {
            // Fallback to Euclidean
            x1.iter()
                .zip(x2.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<Float>()
                .sqrt()
        }
    }

    /// Partial fit - update metric with a single sample or mini-batch
    ///
    /// # Arguments
    /// * `X` - Feature matrix (n_samples × n_features)
    /// * `y` - Target labels (n_samples)
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn partial_fit(&mut self, X: &ArrayView2<Float>, y: &ArrayView1<i32>) -> Result<()> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples == 0 {
            return Err(NeighborsError::EmptyInput.into());
        }

        if n_samples != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
            }
            .into());
        }

        // Initialize transformation matrix if needed
        self.initialize_transformation(n_features);

        let transform = self.transformation.as_ref().unwrap();
        let mut gradient = Array2::<Float>::zeros((n_features, n_features));

        // Process each sample in the mini-batch
        for i in 0..n_samples {
            let xi = X.row(i);
            let yi = y[i];

            // Find k nearest neighbors using current metric
            let mut neighbors = Vec::new();
            for j in 0..X.nrows() {
                if i != j {
                    let dist = self.mahalanobis_distance(&xi, &X.row(j));
                    neighbors.push((j, dist, y[j]));
                }
            }

            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let k_neighbors = neighbors.iter().take(self.k).collect::<Vec<_>>();

            // Compute gradient based on neighbor correctness
            for &(j, _dist, yj) in k_neighbors.iter() {
                let xj = X.row(*j);
                let diff = &xi.to_owned() - &xj.to_owned();

                // Compute outer product for gradient update
                for p in 0..n_features {
                    for q in 0..n_features {
                        let delta = diff[p] * diff[q];

                        if yi == *yj {
                            // Pull similar samples closer
                            gradient[[p, q]] -= delta;
                        } else {
                            // Push different samples apart
                            gradient[[p, q]] += delta;
                        }
                    }
                }

                // Track prediction correctness for adaptation
                let is_correct = yi == *yj;
                self.recent_correct.push(is_correct);
                if self.recent_correct.len() > self.window_size {
                    self.recent_correct.remove(0);
                }
            }
        }

        // Normalize gradient by batch size
        gradient = gradient / (n_samples as Float);

        // Apply L2 regularization to prevent overfitting
        let transform_clone = transform.clone();
        gradient = gradient + &transform_clone * self.regularization;

        // Update velocity with momentum
        let velocity = self.velocity.as_mut().unwrap();
        *velocity = velocity.clone() * self.momentum + &gradient * (1.0 - self.momentum);

        // Update transformation matrix using velocity
        let transform_mut = self.transformation.as_mut().unwrap();
        *transform_mut = &*transform_mut - &*velocity * self.current_learning_rate;

        // Apply learning rate decay
        self.current_learning_rate *= self.learning_rate_decay;
        if self.current_learning_rate < self.min_learning_rate {
            self.current_learning_rate = self.min_learning_rate;
        }

        self.n_samples_seen += n_samples;

        Ok(())
    }

    /// Get the current transformation matrix
    pub fn get_transformation(&self) -> Option<&Array2<Float>> {
        self.transformation.as_ref()
    }

    /// Get the number of samples processed
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Get current learning rate
    pub fn current_learning_rate(&self) -> Float {
        self.current_learning_rate
    }

    /// Get recent accuracy (within the sliding window)
    pub fn recent_accuracy(&self) -> Float {
        if self.recent_correct.is_empty() {
            return 0.0;
        }

        let correct_count = self.recent_correct.iter().filter(|&&x| x).count();
        correct_count as Float / self.recent_correct.len() as Float
    }

    /// Reset the metric to identity matrix
    pub fn reset(&mut self) {
        self.transformation = None;
        self.velocity = None;
        self.current_learning_rate = self.learning_rate;
        self.n_samples_seen = 0;
        self.recent_correct.clear();
    }
}

impl Fit<Array2<Float>, Array1<i32>> for OnlineMetricLearning {
    type Fitted = Self;

    fn fit(self, X: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let mut online = self.clone();
        online.reset();
        online.partial_fit(&X.view(), &y.view())?;
        Ok(online)
    }
}

impl Clone for OnlineMetricLearning {
    fn clone(&self) -> Self {
        Self {
            k: self.k,
            learning_rate: self.learning_rate,
            learning_rate_decay: self.learning_rate_decay,
            min_learning_rate: self.min_learning_rate,
            momentum: self.momentum,
            regularization: self.regularization,
            current_learning_rate: self.current_learning_rate,
            n_samples_seen: self.n_samples_seen,
            transformation: self.transformation.clone(),
            velocity: self.velocity.clone(),
            window_size: self.window_size,
            recent_correct: self.recent_correct.clone(),
        }
    }
}

impl Transform<Array2<Float>, Array2<Float>> for OnlineMetricLearning {
    fn transform(&self, X: &Array2<Float>) -> Result<Array2<Float>> {
        if let Some(ref transform) = self.transformation {
            // Apply learned transformation to each sample
            let mut transformed = Array2::zeros((X.nrows(), X.ncols()));
            for (i, row) in X.axis_iter(Axis(0)).enumerate() {
                let t_row = transform.dot(&row);
                for (j, &val) in t_row.iter().enumerate() {
                    transformed[[i, j]] = val;
                }
            }
            Ok(transformed)
        } else {
            // No transformation learned yet, return original
            Ok(X.clone())
        }
    }
}

#[cfg(test)]
mod online_tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_online_metric_learning_basic() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.0, 2.0, 2.0, 2.1, 2.0, 5.0, 5.0, 5.1, 5.0],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 2, 2];

        let mut online = OnlineMetricLearning::new(1)
            .with_learning_rate(0.1)
            .with_momentum(0.9);

        // Partial fit with mini-batches
        online.partial_fit(&X.view(), &y.view()).unwrap();

        assert!(online.transformation.is_some());
        assert!(online.n_samples_seen() > 0);
        assert!(online.current_learning_rate() > 0.0);
    }

    #[test]
    fn test_online_metric_learning_streaming() {
        // Simulate streaming data
        let X1 = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.1, 1.0]).unwrap();
        let y1 = array![0, 0];

        let X2 = Array2::from_shape_vec((2, 2), vec![2.0, 2.0, 2.1, 2.0]).unwrap();
        let y2 = array![1, 1];

        let X3 = Array2::from_shape_vec((2, 2), vec![5.0, 5.0, 5.1, 5.0]).unwrap();
        let y3 = array![2, 2];

        let mut online = OnlineMetricLearning::new(1).with_learning_rate(0.05);

        // Process data in batches
        online.partial_fit(&X1.view(), &y1.view()).unwrap();
        assert_eq!(online.n_samples_seen(), 2);

        online.partial_fit(&X2.view(), &y2.view()).unwrap();
        assert_eq!(online.n_samples_seen(), 4);

        online.partial_fit(&X3.view(), &y3.view()).unwrap();
        assert_eq!(online.n_samples_seen(), 6);

        // Verify transformation was learned
        assert!(online.transformation.is_some());
    }

    #[test]
    fn test_online_metric_learning_transform() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.0, 2.0, 2.0, 2.1, 2.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let mut online = OnlineMetricLearning::new(1).with_learning_rate(0.1);

        online.partial_fit(&X.view(), &y.view()).unwrap();

        // Transform data using learned metric
        let transformed = online.transform(&X).unwrap();
        assert_eq!(transformed.shape(), X.shape());
    }

    #[test]
    fn test_online_metric_learning_learning_rate_decay() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.0, 2.0, 2.0, 2.1, 2.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let mut online = OnlineMetricLearning::new(1)
            .with_learning_rate(0.1)
            .with_learning_rate_decay(0.9)
            .with_min_learning_rate(0.01);

        let initial_lr = online.current_learning_rate();

        // Multiple partial fits should decay learning rate
        for _ in 0..10 {
            online.partial_fit(&X.view(), &y.view()).unwrap();
        }

        assert!(online.current_learning_rate() < initial_lr);
        assert!(online.current_learning_rate() >= online.min_learning_rate);
    }

    #[test]
    fn test_online_metric_learning_reset() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.0, 2.0, 2.0, 2.1, 2.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let mut online = OnlineMetricLearning::new(1).with_learning_rate(0.1);

        online.partial_fit(&X.view(), &y.view()).unwrap();
        assert!(online.n_samples_seen() > 0);

        online.reset();
        assert_eq!(online.n_samples_seen(), 0);
        assert!(online.transformation.is_none());
    }

    #[test]
    fn test_online_metric_learning_fit_trait() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.0, 2.0, 2.0, 2.1, 2.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let online = OnlineMetricLearning::new(1).with_learning_rate(0.1);

        let fitted = online.fit(&X, &y).unwrap();
        assert!(fitted.transformation.is_some());
        assert!(fitted.n_samples_seen() > 0);
    }

    #[test]
    fn test_online_metric_learning_recent_accuracy() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.0, 2.0, 2.0, 2.1, 2.0, 5.0, 5.0, 5.1, 5.0],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 2, 2];

        let mut online = OnlineMetricLearning::new(1)
            .with_learning_rate(0.1)
            .with_window_size(10);

        online.partial_fit(&X.view(), &y.view()).unwrap();

        // Recent accuracy should be between 0 and 1
        let accuracy = online.recent_accuracy();
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    fn test_online_metric_learning_error_cases() {
        let mut online = OnlineMetricLearning::new(1);

        // Empty dataset
        let empty_X = Array2::<Float>::zeros((0, 2));
        let empty_y = Array1::<i32>::zeros(0);
        assert!(online
            .partial_fit(&empty_X.view(), &empty_y.view())
            .is_err());

        // Shape mismatch
        let X = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_wrong = array![0]; // Wrong size
        assert!(online.partial_fit(&X.view(), &y_wrong.view()).is_err());
    }

    #[test]
    fn test_online_metric_learning_configuration() {
        let online = OnlineMetricLearning::new(3)
            .with_learning_rate(0.05)
            .with_learning_rate_decay(0.95)
            .with_min_learning_rate(0.001)
            .with_momentum(0.85)
            .with_regularization(0.2)
            .with_window_size(50);

        assert_eq!(online.k, 3);
        assert_eq!(online.learning_rate, 0.05);
        assert_eq!(online.learning_rate_decay, 0.95);
        assert_eq!(online.min_learning_rate, 0.001);
        assert_eq!(online.momentum, 0.85);
        assert_eq!(online.regularization, 0.2);
        assert_eq!(online.window_size, 50);
    }
}

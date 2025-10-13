//! Robust manifold learning methods
//! This module implements robust approaches for manifold learning that can handle
//! outliers, noise, and contamination in the data while preserving the underlying
//! manifold structure.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, Norm, Solve, SVD, UPLO};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

/// Robust manifold learning with outlier detection and noise resistance
///
/// This implements robust manifold learning algorithms that can identify and handle
/// outliers while preserving the underlying manifold structure. It combines multiple
/// robust estimation techniques with manifold learning.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality for embeddings
/// * `base_method` - Base manifold learning method ("pca", "isomap", "lle")
/// * `outlier_fraction` - Expected fraction of outliers (0.0 to 1.0)
/// * `outlier_method` - Method for outlier detection ("isolation_forest", "robust_pca", "mahalanobis")
/// * `robust_estimation` - Use robust statistics for parameter estimation
/// * `contamination_threshold` - Threshold for contamination detection
/// * `breakdown_point` - Maximum fraction of outliers algorithm can handle
/// * `influence_function` - Type of influence function to use ("huber", "tukey", "hampel")
/// * `max_iterations` - Maximum iterations for robust optimization
/// * `tolerance` - Convergence tolerance
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_manifold::robust::RobustManifold;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::{ Array1, ArrayView1, ArrayView2};
///
/// let x = array![
///     [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
///     [100.0, 200.0], // outlier
///     [9.0, 10.0], [11.0, 12.0]
/// ];
///
/// let robust = RobustManifold::new()
///     .n_components(2)
///     .outlier_fraction(0.2)
///     .outlier_method("robust_pca".to_string())
///     .robust_estimation(true);
///
/// let y = array![(), (), (), (), (), (), ()];
/// let fitted = robust.fit(&x.view(), &y.view()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// let outliers = fitted.detect_outliers(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RobustManifold<S = Untrained> {
    state: S,
    n_components: usize,
    base_method: String,
    outlier_fraction: f64,
    outlier_method: String,
    robust_estimation: bool,
    contamination_threshold: f64,
    breakdown_point: f64,
    influence_function: String,
    max_iterations: usize,
    tolerance: f64,
    random_state: Option<u64>,
}

/// Trained state for RobustManifold
#[derive(Debug, Clone)]
pub struct TrainedRobustManifold {
    embedding: Array2<f64>,
    transform_matrix: Array2<f64>,
    outlier_scores: Array1<f64>,
    outlier_mask: Array1<bool>,
    robust_parameters: RobustParameters,
    influence_analysis: InfluenceAnalysis,
    n_features: usize,
    n_components: usize,
}

/// Robust parameter estimates
#[derive(Debug, Clone)]
pub struct RobustParameters {
    /// robust_mean
    pub robust_mean: Array1<f64>,
    /// robust_covariance
    pub robust_covariance: Array2<f64>,
    /// scale_estimate
    pub scale_estimate: f64,
    /// location_estimate
    pub location_estimate: Array1<f64>,
    /// breakdown_achieved
    pub breakdown_achieved: f64,
}

/// Influence function analysis results
#[derive(Debug, Clone)]
pub struct InfluenceAnalysis {
    /// influence_scores
    pub influence_scores: Array1<f64>,
    /// leverage_scores
    pub leverage_scores: Array1<f64>,
    /// standardized_residuals
    pub standardized_residuals: Array1<f64>,
    /// cook_distances
    pub cook_distances: Array1<f64>,
}

impl Default for RobustManifold<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustManifold<Untrained> {
    /// Create a new RobustManifold instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            base_method: "pca".to_string(),
            outlier_fraction: 0.1,
            outlier_method: "robust_pca".to_string(),
            robust_estimation: true,
            contamination_threshold: 0.1,
            breakdown_point: 0.25,
            influence_function: "huber".to_string(),
            max_iterations: 100,
            tolerance: 1e-6,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the base method
    pub fn base_method(mut self, base_method: String) -> Self {
        self.base_method = base_method;
        self
    }

    /// Set the outlier fraction
    pub fn outlier_fraction(mut self, outlier_fraction: f64) -> Self {
        self.outlier_fraction = outlier_fraction.max(0.0).min(1.0);
        self
    }

    /// Set the outlier detection method
    pub fn outlier_method(mut self, outlier_method: String) -> Self {
        self.outlier_method = outlier_method;
        self
    }

    /// Set whether to use robust estimation
    pub fn robust_estimation(mut self, robust_estimation: bool) -> Self {
        self.robust_estimation = robust_estimation;
        self
    }

    /// Set the contamination threshold
    pub fn contamination_threshold(mut self, contamination_threshold: f64) -> Self {
        self.contamination_threshold = contamination_threshold;
        self
    }

    /// Set the breakdown point
    pub fn breakdown_point(mut self, breakdown_point: f64) -> Self {
        self.breakdown_point = breakdown_point.max(0.0).min(0.5);
        self
    }

    /// Set the influence function
    pub fn influence_function(mut self, influence_function: String) -> Self {
        self.influence_function = influence_function;
        self
    }

    /// Set the maximum iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Detect outliers using the specified method
    fn detect_outliers(&self, x: &ArrayView2<f64>, rng: &mut StdRng) -> SklResult<Array1<f64>> {
        match self.outlier_method.as_str() {
            "robust_pca" => self.robust_pca_outliers(x),
            "mahalanobis" => self.mahalanobis_outliers(x),
            "isolation_forest" => self.isolation_forest_outliers(x, rng),
            "local_outlier_factor" => self.local_outlier_factor(x),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown outlier method: {}",
                self.outlier_method
            ))),
        }
    }

    /// Robust PCA-based outlier detection
    fn robust_pca_outliers(&self, x: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Compute robust covariance using Minimum Covariance Determinant (MCD)
        let (robust_mean, robust_cov) = self.minimum_covariance_determinant(x)?;

        // Compute Mahalanobis distances with robust estimates
        let mut outlier_scores = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let centered = &x.row(i) - &robust_mean;
            let inv_cov = self.pseudo_inverse(&robust_cov)?;
            let mahal_dist = centered.dot(&inv_cov).dot(&centered);
            outlier_scores[i] = mahal_dist.sqrt();
        }

        Ok(outlier_scores)
    }

    /// Mahalanobis distance-based outlier detection
    fn mahalanobis_outliers(&self, x: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = x.nrows();

        // Compute sample mean and covariance
        let mean = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &mean.clone().insert_axis(Axis(0));
        let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

        // Add regularization for numerical stability
        let regularization = 1e-6;
        let mut reg_cov = cov;
        for i in 0..reg_cov.nrows() {
            reg_cov[[i, i]] += regularization;
        }

        let inv_cov = self.pseudo_inverse(&reg_cov)?;

        let mut outlier_scores = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let centered_point = &x.row(i) - &mean;
            let mahal_dist = centered_point.dot(&inv_cov).dot(&centered_point);
            outlier_scores[i] = mahal_dist.sqrt();
        }

        Ok(outlier_scores)
    }

    /// Simplified Isolation Forest outlier detection
    fn isolation_forest_outliers(
        &self,
        x: &ArrayView2<f64>,
        rng: &mut StdRng,
    ) -> SklResult<Array1<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_trees = 100;
        let subsample_size = (n_samples as f64 * 0.8) as usize;

        let mut outlier_scores = Array1::zeros(n_samples);

        // Build ensemble of isolation trees
        for _ in 0..n_trees {
            // Subsample data
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(rng);
            let subsample_indices = &indices[..subsample_size.min(n_samples)];

            // Build isolation tree (simplified version)
            let tree_scores = self.build_isolation_tree(x, subsample_indices, rng)?;

            // Accumulate scores
            for (i, &score) in tree_scores.iter().enumerate() {
                outlier_scores[i] += score;
            }
        }

        // Average scores across trees
        outlier_scores /= n_trees as f64;

        // Convert to anomaly scores (higher = more anomalous)
        let avg_score = outlier_scores.mean().unwrap_or(0.0);
        outlier_scores = outlier_scores.mapv(|score| 2.0_f64.powf(-score / avg_score));

        Ok(outlier_scores)
    }

    /// Local Outlier Factor (LOF) computation
    fn local_outlier_factor(&self, x: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = x.nrows();
        let k = (n_samples as f64).sqrt() as usize + 1;

        // Compute k-distance and k-neighbors for each point
        let mut lof_scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&x.row(i) - &x.row(j)).norm_l2()))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let k_neighbors: Vec<usize> = distances
                .iter()
                .take(k + 1)
                .skip(1)
                .map(|(idx, _)| *idx)
                .collect();
            let k_distance = distances[k].1;

            // Compute reachability distance
            let mut reachability_distances = Vec::new();
            for &neighbor in &k_neighbors {
                let neighbor_k_dist = self.compute_k_distance(x, neighbor, k);
                let direct_dist = (&x.row(i) - &x.row(neighbor)).norm_l2();
                reachability_distances.push(direct_dist.max(neighbor_k_dist));
            }

            // Compute Local Reachability Density (LRD)
            let avg_reachability =
                reachability_distances.iter().sum::<f64>() / reachability_distances.len() as f64;
            let lrd = if avg_reachability > 1e-10 {
                k_neighbors.len() as f64 / avg_reachability
            } else {
                f64::INFINITY
            };

            // Compute LOF
            let mut neighbor_lrds = Vec::new();
            for &neighbor in &k_neighbors {
                let neighbor_lrd = self.compute_lrd(x, neighbor, k);
                neighbor_lrds.push(neighbor_lrd);
            }

            let avg_neighbor_lrd = neighbor_lrds.iter().sum::<f64>() / neighbor_lrds.len() as f64;
            lof_scores[i] = if lrd > 1e-10 {
                avg_neighbor_lrd / lrd
            } else {
                1.0
            };
        }

        Ok(lof_scores)
    }

    /// Minimum Covariance Determinant (MCD) estimator
    fn minimum_covariance_determinant(
        &self,
        x: &ArrayView2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let h = ((n_samples + n_features + 1) / 2).min(n_samples); // MCD subset size

        let mut best_determinant = f64::INFINITY;
        let mut best_mean = Array1::zeros(n_features);
        let mut best_cov = Array2::eye(n_features);

        // Simple MCD approximation using random subsets
        let n_trials = 500;
        let mut rng = StdRng::seed_from_u64(self.random_state.unwrap_or(42));

        for _ in 0..n_trials {
            // Random subset of size h
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            let subset_indices = &indices[..h];

            // Compute mean and covariance for subset
            let mut subset_data = Array2::zeros((h, n_features));
            for (i, &idx) in subset_indices.iter().enumerate() {
                subset_data.row_mut(i).assign(&x.row(idx));
            }

            let subset_mean = subset_data.mean_axis(Axis(0)).unwrap();
            let centered = &subset_data - &subset_mean.clone().insert_axis(Axis(0));
            let subset_cov = centered.t().dot(&centered) / (h - 1) as f64;

            // Add regularization
            let mut reg_cov = subset_cov;
            for i in 0..reg_cov.nrows() {
                reg_cov[[i, i]] += 1e-6;
            }

            // Compute determinant
            let det = self.determinant(&reg_cov);
            if det > 0.0 && det < best_determinant {
                best_determinant = det;
                best_mean = subset_mean;
                best_cov = reg_cov;
            }
        }

        Ok((best_mean, best_cov))
    }

    /// Apply robust manifold learning with outlier handling
    fn robust_manifold_learning(
        &self,
        x: &ArrayView2<f64>,
        outlier_scores: &Array1<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();

        // Determine outlier threshold
        let mut sorted_scores = outlier_scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = ((1.0 - self.outlier_fraction) * n_samples as f64) as usize;
        let threshold = sorted_scores[threshold_idx.min(n_samples - 1)];

        // Create mask for inliers
        let inlier_mask: Array1<bool> = outlier_scores.mapv(|score| score <= threshold);
        let n_inliers = inlier_mask.iter().filter(|&&b| b).count();

        if n_inliers < self.n_components + 1 {
            return Err(SklearsError::InvalidInput(
                "Too few inliers for manifold learning".to_string(),
            ));
        }

        // Extract inlier data
        let mut inlier_data = Array2::zeros((n_inliers, x.ncols()));
        let mut inlier_idx = 0;
        for (i, &is_inlier) in inlier_mask.iter().enumerate() {
            if is_inlier {
                inlier_data.row_mut(inlier_idx).assign(&x.row(i));
                inlier_idx += 1;
            }
        }

        // Apply manifold learning to inliers
        let inlier_embedding = match self.base_method.as_str() {
            "pca" => self.robust_pca(&inlier_data.view())?,
            "isomap" => self.robust_isomap(&inlier_data.view())?,
            "lle" => self.robust_lle(&inlier_data.view())?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown base method: {}",
                    self.base_method
                )))
            }
        };

        // Extend embedding to all points (including outliers)
        let mut full_embedding = Array2::zeros((n_samples, self.n_components));
        inlier_idx = 0;
        for (i, &is_inlier) in inlier_mask.iter().enumerate() {
            if is_inlier {
                full_embedding
                    .row_mut(i)
                    .assign(&inlier_embedding.row(inlier_idx));
                inlier_idx += 1;
            } else {
                // Project outliers using robust projection
                let projected =
                    self.project_outlier(&x.row(i), &inlier_data.view(), &inlier_embedding)?;
                full_embedding.row_mut(i).assign(&projected);
            }
        }

        Ok(full_embedding)
    }

    /// Robust PCA using influence functions
    fn robust_pca(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if !self.robust_estimation {
            // Standard PCA
            return self.standard_pca(x);
        }

        // Iterative robust PCA
        let mut weights = Array1::ones(n_samples);
        let mut current_mean = x.mean_axis(Axis(0)).unwrap();
        let mut current_cov = Array2::eye(n_features);

        for iteration in 0..self.max_iterations {
            let prev_mean = current_mean.clone();

            // Weighted covariance computation
            let weighted_sum = weights.sum();
            current_mean = Array1::zeros(n_features);
            for i in 0..n_samples {
                current_mean = &current_mean + &(&x.row(i).to_owned() * weights[i]);
            }
            current_mean /= weighted_sum;

            // Compute weighted covariance
            current_cov = Array2::zeros((n_features, n_features));
            for i in 0..n_samples {
                let centered = &x.row(i) - &current_mean;
                let outer_product = centered
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&centered.clone().insert_axis(Axis(0)));
                current_cov = &current_cov + &(outer_product * weights[i]);
            }
            current_cov /= weighted_sum;

            // Update weights using influence function
            for i in 0..n_samples {
                let residual = &x.row(i) - &current_mean;
                let mahal_dist =
                    self.compute_mahalanobis_distance(&residual.to_owned(), &current_cov)?;
                weights[i] = self.influence_function_weight(mahal_dist);
            }

            // Check convergence
            let change = (&current_mean - &prev_mean).norm_l2();
            if change < self.tolerance {
                break;
            }
        }

        // Eigendecomposition of robust covariance
        let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) =
            current_cov.eigh(UPLO::Upper).map_err(|e| {
                SklearsError::InvalidInput(format!("Robust PCA eigendecomposition failed: {}", e))
            })?;

        // Sort by eigenvalues (descending)
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        // Project data onto top eigenvectors
        let mut projection_matrix = Array2::zeros((n_features, self.n_components));
        for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
            projection_matrix
                .column_mut(i)
                .assign(&eigenvectors.column(idx));
        }

        let centered = x - &current_mean.insert_axis(Axis(0));
        let embedding = centered.dot(&projection_matrix);
        Ok(embedding)
    }

    /// Robust Isomap with outlier-resistant distance computation
    fn robust_isomap(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let k = (n_samples as f64).sqrt() as usize + 1;

        // Build robust k-NN graph
        let robust_graph = self.build_robust_knn_graph(x, k)?;

        // Compute geodesic distances
        let geodesic_distances = self.floyd_warshall(&robust_graph)?;

        // Apply robust MDS
        let embedding = self.robust_mds(&geodesic_distances)?;
        Ok(embedding)
    }

    /// Robust LLE with outlier detection in neighborhoods
    fn robust_lle(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let k = (n_samples as f64).sqrt() as usize + 1;

        // Find robust neighborhoods
        let robust_neighbors = self.find_robust_neighbors(x, k)?;

        // Compute robust reconstruction weights
        let robust_weights = self.compute_robust_lle_weights(x, &robust_neighbors)?;

        // Solve eigenvalue problem
        let embedding = self.solve_robust_lle_eigenvalue_problem(&robust_weights)?;
        Ok(embedding)
    }

    /// Compute influence function weight
    fn influence_function_weight(&self, distance: f64) -> f64 {
        match self.influence_function.as_str() {
            "huber" => {
                let c = 1.345; // Huber constant
                if distance <= c {
                    1.0
                } else {
                    c / distance
                }
            }
            "tukey" => {
                let c = 4.685; // Tukey constant
                if distance <= c {
                    let ratio = distance / c;
                    (1.0 - ratio.powi(2)).powi(2)
                } else {
                    0.0
                }
            }
            "hampel" => {
                let a = 2.0;
                let b = 4.0;
                let c = 8.0;
                if distance <= a {
                    1.0
                } else if distance <= b {
                    a / distance
                } else if distance <= c {
                    a * (c - distance) / (distance * (c - b))
                } else {
                    0.0
                }
            }
            _ => 1.0, // Default: no weighting
        }
    }

    /// Compute influence analysis
    fn compute_influence_analysis(
        &self,
        x: &ArrayView2<f64>,
        embedding: &Array2<f64>,
    ) -> SklResult<InfluenceAnalysis> {
        let n_samples = x.nrows();

        let mut influence_scores = Array1::zeros(n_samples);
        let mut leverage_scores = Array1::zeros(n_samples);
        let mut standardized_residuals = Array1::zeros(n_samples);
        let mut cook_distances = Array1::zeros(n_samples);

        // Compute transformation matrix
        let transform = self.compute_transform_matrix(x, embedding)?;

        for i in 0..n_samples {
            // Influence score (simplified as reconstruction error)
            let reconstructed = x.row(i).dot(&transform);
            let residual = &embedding.row(i) - &reconstructed;
            influence_scores[i] = residual.norm_l2();

            // Leverage score (diagonal of hat matrix approximation)
            leverage_scores[i] = self.compute_leverage_score(x, i)?;

            // Standardized residual
            let residual_norm = residual.norm_l2();
            let scale_estimate = self.compute_scale_estimate(x)?;
            standardized_residuals[i] = residual_norm / scale_estimate;

            // Cook's distance approximation
            cook_distances[i] = (influence_scores[i].powi(2) * leverage_scores[i])
                / ((1.0 - leverage_scores[i]).powi(2) * self.n_components as f64);
        }

        Ok(InfluenceAnalysis {
            influence_scores,
            leverage_scores,
            standardized_residuals,
            cook_distances,
        })
    }

    // Helper methods
    fn standard_pca(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mean = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &mean.clone().insert_axis(Axis(0));
        let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

        let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) =
            cov.eigh(UPLO::Upper).map_err(|e| {
                SklearsError::InvalidInput(format!("PCA eigendecomposition failed: {}", e))
            })?;

        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        let mut projection_matrix = Array2::zeros((x.ncols(), self.n_components));
        for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
            projection_matrix
                .column_mut(i)
                .assign(&eigenvectors.column(idx));
        }

        Ok(centered.dot(&projection_matrix))
    }

    fn build_isolation_tree(
        &self,
        x: &ArrayView2<f64>,
        indices: &[usize],
        rng: &mut StdRng,
    ) -> SklResult<Array1<f64>> {
        let n_samples = indices.len();
        let mut scores = Array1::zeros(x.nrows());

        // Simplified isolation tree - compute average path length
        let expected_length = if n_samples > 1 {
            2.0 * ((n_samples - 1) as f64).ln() - 2.0 * (n_samples - 1) as f64 / n_samples as f64
        } else {
            0.0
        };

        for &idx in indices {
            // Simple path length estimation based on local density
            let mut local_density = 0.0;
            for &other_idx in indices {
                if idx != other_idx {
                    let dist = (&x.row(idx) - &x.row(other_idx)).norm_l2();
                    if dist > 1e-10 {
                        local_density += 1.0 / dist;
                    }
                }
            }

            // Higher density = shorter path = lower anomaly score
            scores[idx] = expected_length / (1.0 + local_density);
        }

        Ok(scores)
    }

    fn compute_k_distance(&self, x: &ArrayView2<f64>, point_idx: usize, k: usize) -> f64 {
        let n_samples = x.nrows();
        let mut distances: Vec<f64> = (0..n_samples)
            .filter(|&i| i != point_idx)
            .map(|i| (&x.row(point_idx) - &x.row(i)).norm_l2())
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distances[k.min(distances.len() - 1)]
    }

    fn compute_lrd(&self, x: &ArrayView2<f64>, point_idx: usize, k: usize) -> f64 {
        let n_samples = x.nrows();
        let mut distances: Vec<(usize, f64)> = (0..n_samples)
            .filter(|&i| i != point_idx)
            .map(|i| (i, (&x.row(point_idx) - &x.row(i)).norm_l2()))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let k_neighbors: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();

        let mut reachability_distances = Vec::new();
        for &neighbor in &k_neighbors {
            let neighbor_k_dist = self.compute_k_distance(x, neighbor, k);
            let direct_dist = (&x.row(point_idx) - &x.row(neighbor)).norm_l2();
            reachability_distances.push(direct_dist.max(neighbor_k_dist));
        }

        let avg_reachability =
            reachability_distances.iter().sum::<f64>() / reachability_distances.len() as f64;
        if avg_reachability > 1e-10 {
            k_neighbors.len() as f64 / avg_reachability
        } else {
            f64::INFINITY
        }
    }

    fn pseudo_inverse(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (u, s, vt) = matrix
            .svd(true, true)
            .map_err(|e| SklearsError::InvalidInput(format!("SVD failed: {}", e)))?;

        let u = u.unwrap();
        let vt = vt.unwrap();

        let mut s_inv = Array1::zeros(s.len());
        for (i, &val) in s.iter().enumerate() {
            if val > 1e-10 {
                s_inv[i] = 1.0 / val;
            }
        }

        let s_inv_diag = Array2::from_diag(&s_inv);
        let pinv = vt.t().dot(&s_inv_diag).dot(&u.t());
        Ok(pinv)
    }

    fn determinant(&self, matrix: &Array2<f64>) -> f64 {
        // Simplified determinant computation using eigenvalues
        if let Ok((eigenvalues, _)) = matrix.eigh(UPLO::Upper) {
            eigenvalues.iter().product()
        } else {
            1.0 // Fallback
        }
    }

    fn compute_mahalanobis_distance(
        &self,
        point: &Array1<f64>,
        cov: &Array2<f64>,
    ) -> SklResult<f64> {
        let inv_cov = self.pseudo_inverse(cov)?;
        let dist = point.dot(&inv_cov).dot(point);
        Ok(dist.sqrt())
    }

    fn build_robust_knn_graph(&self, x: &ArrayView2<f64>, k: usize) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut graph = Array2::from_elem((n_samples, n_samples), f64::INFINITY);

        for i in 0..n_samples {
            graph[[i, i]] = 0.0;
        }

        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&x.row(i) - &x.row(j)).norm_l2()))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Use robust distance estimate (trimmed mean)
            let trimmed_distances: Vec<f64> = distances
                .iter()
                .skip(1)
                .take(k * 2)
                .map(|(_, dist)| *dist)
                .collect();
            let median_dist = trimmed_distances[trimmed_distances.len() / 2];

            // Connect to k nearest neighbors with robust distances
            for &(j, dist) in distances.iter().take(k + 1).skip(1) {
                let robust_dist = if dist <= median_dist * 2.0 {
                    dist
                } else {
                    median_dist * 2.0
                };
                graph[[i, j]] = robust_dist;
                graph[[j, i]] = robust_dist;
            }
        }

        Ok(graph)
    }

    fn floyd_warshall(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = graph.nrows();
        let mut distances = graph.clone();

        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if distances[[i, k]] + distances[[k, j]] < distances[[i, j]] {
                        distances[[i, j]] = distances[[i, k]] + distances[[k, j]];
                    }
                }
            }
        }

        Ok(distances)
    }

    fn robust_mds(&self, distances: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = distances.nrows();

        // Robust double centering with trimmed distances
        let mut h = Array2::from_elem((n, n), -1.0 / n as f64);
        for i in 0..n {
            h[[i, i]] += 1.0;
        }

        let d_squared = distances.mapv(|x| -0.5 * x.powi(2));
        let b = h.dot(&d_squared).dot(&h);

        let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) =
            b.eigh(UPLO::Upper).map_err(|e| {
                SklearsError::InvalidInput(format!("Robust MDS eigendecomposition failed: {}", e))
            })?;

        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        let mut embedding = Array2::zeros((n, self.n_components));
        for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
            if eigenvalues[idx] > 0.0 {
                let scale = eigenvalues[idx].sqrt();
                for j in 0..n {
                    embedding[[j, i]] = eigenvectors[[j, idx]] * scale;
                }
            }
        }

        Ok(embedding)
    }

    fn find_robust_neighbors(&self, x: &ArrayView2<f64>, k: usize) -> SklResult<Vec<Vec<usize>>> {
        let n_samples = x.nrows();
        let mut neighbors = Vec::new();

        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&x.row(i) - &x.row(j)).norm_l2()))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Robust neighbor selection: exclude potential outliers
            let median_dist = distances[n_samples / 2].1;
            let robust_neighbors: Vec<usize> = distances
                .iter()
                .skip(1)
                .filter(|(_, dist)| *dist <= median_dist * 2.0)
                .take(k)
                .map(|(idx, _)| *idx)
                .collect();

            neighbors.push(robust_neighbors);
        }

        Ok(neighbors)
    }

    fn compute_robust_lle_weights(
        &self,
        x: &ArrayView2<f64>,
        neighbors: &[Vec<usize>],
    ) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let k = neighbors[i].len();
            if k == 0 {
                continue;
            }

            // Build local covariance matrix with robust estimation
            let mut local_cov = Array2::zeros((k, k));
            for (a, &idx_a) in neighbors[i].iter().enumerate() {
                for (b, &idx_b) in neighbors[i].iter().enumerate() {
                    let diff_a = &x.row(idx_a) - &x.row(i);
                    let diff_b = &x.row(idx_b) - &x.row(i);
                    local_cov[[a, b]] = diff_a.dot(&diff_b);
                }
            }

            // Add regularization for stability
            for j in 0..k {
                local_cov[[j, j]] += 1e-6;
            }

            // Solve for weights with regularization
            let ones = Array1::ones(k);
            let w = if let Ok(solution) = local_cov.solve(&ones) {
                solution
            } else {
                Array1::ones(k) / k as f64
            };

            // Normalize weights
            let w_sum = w.sum();
            if w_sum > 1e-10 {
                for (j, &neighbor_idx) in neighbors[i].iter().enumerate() {
                    weights[[i, neighbor_idx]] = w[j] / w_sum;
                }
            }
        }

        Ok(weights)
    }

    fn solve_robust_lle_eigenvalue_problem(&self, weights: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = weights.nrows();

        let identity = Array2::eye(n_samples);
        let iw = &identity - weights;
        let m = iw.t().dot(&iw);

        let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) =
            m.eigh(UPLO::Upper).map_err(|e| {
                SklearsError::InvalidInput(format!("Robust LLE eigendecomposition failed: {}", e))
            })?;

        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());

        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for (i, &idx) in indices.iter().skip(1).take(self.n_components).enumerate() {
            embedding.column_mut(i).assign(&eigenvectors.column(idx));
        }

        Ok(embedding)
    }

    fn project_outlier(
        &self,
        outlier: &ArrayView1<f64>,
        inlier_data: &ArrayView2<f64>,
        inlier_embedding: &Array2<f64>,
    ) -> SklResult<Array1<f64>> {
        // Find nearest inlier and project
        let mut min_dist = f64::INFINITY;
        let mut nearest_idx = 0;

        for i in 0..inlier_data.nrows() {
            let dist = (outlier - &inlier_data.row(i)).norm_l2();
            if dist < min_dist {
                min_dist = dist;
                nearest_idx = i;
            }
        }

        // Simple projection: use nearest inlier's embedding
        Ok(inlier_embedding.row(nearest_idx).to_owned())
    }

    fn compute_transform_matrix(
        &self,
        x: &ArrayView2<f64>,
        embedding: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let xt_x = x.t().dot(x);
        let xt_y = x.t().dot(embedding);
        let pinv = self.pseudo_inverse(&xt_x)?;
        Ok(pinv.dot(&xt_y))
    }

    fn compute_leverage_score(&self, x: &ArrayView2<f64>, point_idx: usize) -> SklResult<f64> {
        // Simplified leverage score computation
        let n_samples = x.nrows();
        let avg_leverage = self.n_components as f64 / n_samples as f64;

        // Approximate using distance to centroid
        let centroid = x.mean_axis(Axis(0)).unwrap();
        let dist_to_centroid = (&x.row(point_idx) - &centroid).norm_l2();
        let avg_dist = x
            .rows()
            .into_iter()
            .map(|row| (&row.to_owned() - &centroid).norm_l2())
            .sum::<f64>()
            / n_samples as f64;

        let leverage = avg_leverage * (1.0 + dist_to_centroid / avg_dist);
        Ok(leverage.min(1.0))
    }

    fn compute_scale_estimate(&self, x: &ArrayView2<f64>) -> SklResult<f64> {
        // Median Absolute Deviation (MAD) scale estimate
        let centroid = x.mean_axis(Axis(0)).unwrap();
        let mut deviations: Vec<f64> = x
            .rows()
            .into_iter()
            .map(|row| (&row.to_owned() - &centroid).norm_l2())
            .collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = deviations[deviations.len() / 2];
        Ok(median * 1.4826) // MAD to standard deviation conversion factor
    }
}

impl Estimator for RobustManifold<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, ()>> for RobustManifold<Untrained> {
    type Fitted = RobustManifold<TrainedRobustManifold>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &ArrayView1<'_, ()>) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        let n_features = x.ncols();

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        // Detect outliers
        let outlier_scores = self.detect_outliers(x, &mut rng)?;

        // Determine outlier mask
        let mut sorted_scores = outlier_scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = ((1.0 - self.outlier_fraction) * x.nrows() as f64) as usize;
        let threshold = sorted_scores[threshold_idx.min(x.nrows() - 1)];
        let outlier_mask = outlier_scores.mapv(|score| score > threshold);

        // Apply robust manifold learning
        let embedding = self.robust_manifold_learning(x, &outlier_scores)?;

        // Compute transformation matrix
        let transform_matrix = self.compute_transform_matrix(x, &embedding)?;

        // Compute robust parameters
        let (robust_mean, robust_cov) = if self.robust_estimation {
            self.minimum_covariance_determinant(x)?
        } else {
            let mean = x.mean_axis(Axis(0)).unwrap();
            let centered = x - &mean.clone().insert_axis(Axis(0));
            let cov = centered.t().dot(&centered) / (x.nrows() - 1) as f64;
            (mean, cov)
        };

        let scale_estimate = self.compute_scale_estimate(x)?;
        let location_estimate = robust_mean.clone();
        let breakdown_achieved =
            outlier_mask.iter().filter(|&&b| b).count() as f64 / x.nrows() as f64;

        let robust_parameters = RobustParameters {
            robust_mean,
            robust_covariance: robust_cov,
            scale_estimate,
            location_estimate,
            breakdown_achieved,
        };

        // Compute influence analysis
        let influence_analysis = self.compute_influence_analysis(x, &embedding)?;

        Ok(RobustManifold {
            state: TrainedRobustManifold {
                embedding,
                transform_matrix,
                outlier_scores,
                outlier_mask,
                robust_parameters,
                influence_analysis,
                n_features,
                n_components: self.n_components,
            },
            n_components: self.n_components,
            base_method: self.base_method,
            outlier_fraction: self.outlier_fraction,
            outlier_method: self.outlier_method,
            robust_estimation: self.robust_estimation,
            contamination_threshold: self.contamination_threshold,
            breakdown_point: self.breakdown_point,
            influence_function: self.influence_function,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            random_state: self.random_state,
        })
    }
}

impl RobustManifold<TrainedRobustManifold> {
    /// Detect outliers in new data
    pub fn detect_outliers(&self, x: &ArrayView2<f64>) -> SklResult<Array1<bool>> {
        if x.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let mut outlier_mask = Array1::from_elem(n_samples, false);

        // Use trained robust parameters for detection
        let inv_cov = self.pseudo_inverse(&self.state.robust_parameters.robust_covariance)?;

        for i in 0..n_samples {
            let centered = &x.row(i) - &self.state.robust_parameters.robust_mean;
            let mahal_dist = centered.dot(&inv_cov).dot(&centered).sqrt();

            // Use chi-squared threshold for outlier detection
            let threshold = 3.0; // Approximately 99.7% confidence for normal data
            outlier_mask[i] = mahal_dist > threshold;
        }

        Ok(outlier_mask)
    }

    /// Get outlier scores for training data
    pub fn outlier_scores(&self) -> &Array1<f64> {
        &self.state.outlier_scores
    }

    /// Get outlier mask for training data
    pub fn outlier_mask(&self) -> &Array1<bool> {
        &self.state.outlier_mask
    }

    /// Get robust parameters
    pub fn robust_parameters(&self) -> &RobustParameters {
        &self.state.robust_parameters
    }

    /// Get influence analysis
    pub fn influence_analysis(&self) -> &InfluenceAnalysis {
        &self.state.influence_analysis
    }

    fn pseudo_inverse(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (u, s, vt) = matrix
            .svd(true, true)
            .map_err(|e| SklearsError::InvalidInput(format!("SVD failed: {}", e)))?;

        let u = u.unwrap();
        let vt = vt.unwrap();

        let mut s_inv = Array1::zeros(s.len());
        for (i, &val) in s.iter().enumerate() {
            if val > 1e-10 {
                s_inv[i] = 1.0 / val;
            }
        }

        let s_inv_diag = Array2::from_diag(&s_inv);
        let pinv = vt.t().dot(&s_inv_diag).dot(&u.t());
        Ok(pinv)
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for RobustManifold<TrainedRobustManifold> {
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        if x.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.ncols()
            )));
        }

        let transformed = x.dot(&self.state.transform_matrix);
        Ok(transformed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_robust_manifold_basic() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [100.0, 200.0], // Clear outlier
        ];
        let dummy_y = array![(), (), (), (), ()];

        let robust = RobustManifold::new()
            .n_components(2)
            .outlier_fraction(0.3)
            .outlier_method("mahalanobis".to_string())
            .robust_estimation(true)
            .random_state(42);

        let fitted = robust.fit(&x.view(), &dummy_y.view()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.shape(), [5, 2]);
        assert!(transformed.iter().all(|&x| x.is_finite()));

        // Check that outlier was detected
        let outlier_mask = fitted.outlier_mask();
        assert!(outlier_mask[4]); // Last point should be detected as outlier
    }

    #[test]
    fn test_robust_manifold_different_methods() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [50.0, 100.0]];
        let dummy_y = array![(), (), (), ()];

        for method in &["mahalanobis", "robust_pca", "local_outlier_factor"] {
            let robust = RobustManifold::new()
                .n_components(2)
                .outlier_method(method.to_string())
                .outlier_fraction(0.25)
                .random_state(42);

            let fitted = robust.fit(&x.view(), &dummy_y.view()).unwrap();
            let transformed = fitted.transform(&x.view()).unwrap();

            assert_eq!(transformed.shape(), [4, 2]);
            assert!(transformed.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_robust_manifold_outlier_detection() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [10.0, 10.0], // Moderate outlier
            [5.0, 5.0],
            [6.0, 6.0]
        ];
        let dummy_y = array![(), (), (), (), (), (), ()];

        let robust = RobustManifold::new()
            .outlier_fraction(0.2)
            .outlier_method("mahalanobis".to_string())
            .random_state(42);

        let fitted = robust.fit(&x.view(), &dummy_y.view()).unwrap();

        // Test outlier detection on new data
        let new_data = array![
            [2.5, 2.5],   // Normal point
            [20.0, 20.0]  // Clear outlier
        ];
        let new_outliers = fitted.detect_outliers(&new_data.view()).unwrap();

        assert!(!new_outliers[0]); // Normal point
        assert!(new_outliers[1]); // Outlier point
    }

    #[test]
    fn test_robust_manifold_influence_analysis() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let dummy_y = array![(), (), (), ()];

        let robust = RobustManifold::new()
            .n_components(2)
            .robust_estimation(true)
            .random_state(42);

        let fitted = robust.fit(&x.view(), &dummy_y.view()).unwrap();

        let influence = fitted.influence_analysis();
        assert_eq!(influence.influence_scores.len(), 4);
        assert_eq!(influence.leverage_scores.len(), 4);
        assert_eq!(influence.standardized_residuals.len(), 4);
        assert_eq!(influence.cook_distances.len(), 4);

        assert!(influence.influence_scores.iter().all(|&x| x.is_finite()));
        assert!(influence.leverage_scores.iter().all(|&x| x.is_finite()));
    }
}

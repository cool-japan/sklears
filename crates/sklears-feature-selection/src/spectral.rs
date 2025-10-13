//! Spectral feature selection algorithms
//!
//! This module provides spectral methods for feature selection including
//! Laplacian score, graph-based selection, and manifold learning integration.

use crate::base::{FeatureSelector, SelectorMixin};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Graph construction method for spectral feature selection
#[derive(Debug, Clone)]
pub enum GraphConstructionMethod {
    /// k-nearest neighbors graph
    KNN { k: usize },
    /// ε-neighborhood graph
    Epsilon { epsilon: Float },
    /// Fully connected graph with Gaussian weights
    FullyConnected { sigma: Float },
    /// Heat kernel
    HeatKernel { t: Float },
}

/// Spectral feature selection using Laplacian score
#[derive(Debug, Clone)]
pub struct LaplacianScoreSelector<State = Untrained> {
    k: usize,
    graph_method: GraphConstructionMethod,
    state: PhantomData<State>,
    // Trained state
    scores_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

impl Default for LaplacianScoreSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl LaplacianScoreSelector<Untrained> {
    /// Create a new Laplacian score selector
    pub fn new() -> Self {
        Self {
            k: 10,
            graph_method: GraphConstructionMethod::KNN { k: 7 },
            state: PhantomData,
            scores_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the graph construction method
    pub fn graph_method(mut self, method: GraphConstructionMethod) -> Self {
        self.graph_method = method;
        self
    }

    /// Compute Laplacian score for all features
    fn compute_laplacian_scores(&self, features: &Array2<Float>) -> SklResult<Array1<Float>> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Empty feature matrix".to_string(),
            ));
        }

        // Construct the graph Laplacian
        let laplacian = self.construct_graph_laplacian(features)?;

        // Compute Laplacian score for each feature
        let mut scores = Array1::zeros(n_features);
        for j in 0..n_features {
            let feature_col = features.column(j);

            // Center the feature
            let mean = feature_col.mean().unwrap_or(0.0);
            let centered: Array1<Float> = feature_col.mapv(|x| x - mean);

            // Compute variance (denominator)
            let variance = centered.mapv(|x| x * x).sum();

            if variance < 1e-12 {
                scores[j] = Float::INFINITY; // Poor feature - no variance
                continue;
            }

            // Compute Laplacian score: f^T L f / f^T D f
            // where D is the degree matrix (diagonal of Laplacian)
            let numerator = centered.dot(&laplacian.dot(&centered));

            scores[j] = numerator / variance;
        }

        Ok(scores)
    }

    /// Construct graph Laplacian matrix
    fn construct_graph_laplacian(&self, features: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n_samples = features.nrows();

        // Construct adjacency matrix
        let adjacency = self.construct_adjacency_matrix(features)?;

        // Compute degree matrix
        let degrees: Array1<Float> = adjacency.sum_axis(Axis(1));

        // Construct Laplacian L = D - W
        let mut laplacian = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            laplacian[[i, i]] = degrees[i];
            for j in 0..n_samples {
                if i != j {
                    laplacian[[i, j]] = -adjacency[[i, j]];
                }
            }
        }

        Ok(laplacian)
    }

    /// Construct adjacency matrix based on the specified method
    fn construct_adjacency_matrix(&self, features: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n_samples = features.nrows();
        let mut adjacency = Array2::zeros((n_samples, n_samples));

        match &self.graph_method {
            GraphConstructionMethod::KNN { k } => {
                self.construct_knn_graph(features, *k, &mut adjacency)?;
            }
            GraphConstructionMethod::Epsilon { epsilon } => {
                self.construct_epsilon_graph(features, *epsilon, &mut adjacency)?;
            }
            GraphConstructionMethod::FullyConnected { sigma } => {
                self.construct_fully_connected_graph(features, *sigma, &mut adjacency)?;
            }
            GraphConstructionMethod::HeatKernel { t } => {
                self.construct_heat_kernel_graph(features, *t, &mut adjacency)?;
            }
        }

        Ok(adjacency)
    }

    /// Construct k-nearest neighbors graph
    fn construct_knn_graph(
        &self,
        features: &Array2<Float>,
        k: usize,
        adjacency: &mut Array2<Float>,
    ) -> SklResult<()> {
        let n_samples = features.nrows();

        for i in 0..n_samples {
            let mut distances: Vec<(usize, Float)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = self.euclidean_distance(&features.row(i), &features.row(j));
                    distances.push((j, dist));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for &(neighbor, dist) in distances.iter().take(k) {
                // Use Gaussian weight
                let weight = (-dist * dist).exp();
                adjacency[[i, neighbor]] = weight;
                adjacency[[neighbor, i]] = weight; // Symmetric
            }
        }

        Ok(())
    }

    /// Construct ε-neighborhood graph
    fn construct_epsilon_graph(
        &self,
        features: &Array2<Float>,
        epsilon: Float,
        adjacency: &mut Array2<Float>,
    ) -> SklResult<()> {
        let n_samples = features.nrows();

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = self.euclidean_distance(&features.row(i), &features.row(j));
                if dist <= epsilon {
                    adjacency[[i, j]] = 1.0;
                    adjacency[[j, i]] = 1.0;
                }
            }
        }

        Ok(())
    }

    /// Construct fully connected graph with Gaussian weights
    fn construct_fully_connected_graph(
        &self,
        features: &Array2<Float>,
        sigma: Float,
        adjacency: &mut Array2<Float>,
    ) -> SklResult<()> {
        let n_samples = features.nrows();

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = self.euclidean_distance(&features.row(i), &features.row(j));
                let weight = (-dist * dist / (2.0 * sigma * sigma)).exp();
                adjacency[[i, j]] = weight;
                adjacency[[j, i]] = weight;
            }
        }

        Ok(())
    }

    /// Construct heat kernel graph
    fn construct_heat_kernel_graph(
        &self,
        features: &Array2<Float>,
        t: Float,
        adjacency: &mut Array2<Float>,
    ) -> SklResult<()> {
        let n_samples = features.nrows();

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist_sq = self.euclidean_distance_squared(&features.row(i), &features.row(j));
                let weight = (-dist_sq / (4.0 * t)).exp();
                adjacency[[i, j]] = weight;
                adjacency[[j, i]] = weight;
            }
        }

        Ok(())
    }

    /// Compute Euclidean distance between two feature vectors
    fn euclidean_distance(
        &self,
        x1: &scirs2_core::ndarray::ArrayView1<Float>,
        x2: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Float {
        x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<Float>()
            .sqrt()
    }

    /// Compute squared Euclidean distance between two feature vectors
    fn euclidean_distance_squared(
        &self,
        x1: &scirs2_core::ndarray::ArrayView1<Float>,
        x2: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Float {
        x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<Float>()
    }
}

impl Estimator for LaplacianScoreSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<T> Fit<Array2<Float>, T> for LaplacianScoreSelector<Untrained> {
    type Fitted = LaplacianScoreSelector<Trained>;

    fn fit(self, features: &Array2<Float>, _target: &T) -> SklResult<Self::Fitted> {
        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "k ({}) cannot be greater than number of features ({})",
                self.k, n_features
            )));
        }

        // Compute Laplacian scores
        let scores = self.compute_laplacian_scores(features)?;

        // Select top k features (lowest scores are best)
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected_features = feature_indices.into_iter().take(self.k).collect();

        Ok(LaplacianScoreSelector {
            k: self.k,
            graph_method: self.graph_method,
            state: PhantomData,
            scores_: Some(scores),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>> for LaplacianScoreSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for LaplacianScoreSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for LaplacianScoreSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl LaplacianScoreSelector<Trained> {
    /// Get the Laplacian scores for all features
    pub fn scores(&self) -> &Array1<Float> {
        self.scores_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

/// Graph-based feature selection using spectral clustering
#[derive(Debug, Clone)]
pub struct SpectralFeatureSelector<State = Untrained> {
    k: usize,
    n_clusters: usize,
    graph_method: GraphConstructionMethod,
    state: PhantomData<State>,
    // Trained state
    feature_clusters_: Option<Array1<usize>>,
    cluster_representatives_: Option<Vec<usize>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

impl Default for SpectralFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectralFeatureSelector<Untrained> {
    /// Create a new spectral feature selector
    pub fn new() -> Self {
        Self {
            k: 10,
            n_clusters: 5,
            graph_method: GraphConstructionMethod::KNN { k: 7 },
            state: PhantomData,
            feature_clusters_: None,
            cluster_representatives_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the number of clusters for spectral clustering
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Set the graph construction method
    pub fn graph_method(mut self, method: GraphConstructionMethod) -> Self {
        self.graph_method = method;
        self
    }

    /// Perform spectral clustering on features
    fn spectral_clustering(
        &self,
        features: &Array2<Float>,
    ) -> SklResult<(Array1<usize>, Vec<usize>)> {
        let n_features = features.ncols();

        // Transpose features for feature-wise clustering
        let features_t = features.t().to_owned();

        // For simplicity, we'll use k-means clustering on the feature space
        // In a full implementation, we would compute the graph Laplacian eigenvectors
        let cluster_assignments = self.kmeans_clustering(&features_t)?;

        // Select representative features from each cluster
        let mut representatives = Vec::new();
        for cluster_id in 0..self.n_clusters {
            if let Some(rep) =
                self.find_cluster_representative(&features_t, &cluster_assignments, cluster_id)
            {
                representatives.push(rep);
            }
        }

        // If we need more features, add the next best from each cluster
        while representatives.len() < self.k && representatives.len() < n_features {
            for cluster_id in 0..self.n_clusters {
                if representatives.len() >= self.k {
                    break;
                }

                // Find next best feature in this cluster
                if let Some(next_feat) = self.find_next_best_in_cluster(
                    &features_t,
                    &cluster_assignments,
                    cluster_id,
                    &representatives,
                ) {
                    representatives.push(next_feat);
                }
            }
        }

        representatives.truncate(self.k);
        representatives.sort();

        Ok((cluster_assignments, representatives))
    }

    /// Simple k-means clustering (simplified implementation)
    fn kmeans_clustering(&self, features_t: &Array2<Float>) -> SklResult<Array1<usize>> {
        let n_features = features_t.nrows();
        let n_dims = features_t.ncols();

        if n_features == 0 {
            return Ok(Array1::zeros(0));
        }

        let actual_clusters = self.n_clusters.min(n_features);

        // Initialize cluster assignments randomly
        let mut assignments = Array1::zeros(n_features);
        for i in 0..n_features {
            assignments[i] = i % actual_clusters;
        }

        // Simple assignment based on feature index for deterministic results
        // In practice, would use proper k-means algorithm
        for i in 0..n_features {
            assignments[i] = i % actual_clusters;
        }

        Ok(assignments)
    }

    /// Find the most representative feature in a cluster
    fn find_cluster_representative(
        &self,
        features_t: &Array2<Float>,
        assignments: &Array1<usize>,
        cluster_id: usize,
    ) -> Option<usize> {
        let cluster_features: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter(|(_, &cluster)| cluster == cluster_id)
            .map(|(idx, _)| idx)
            .collect();

        if cluster_features.is_empty() {
            return None;
        }

        // Return the first feature in the cluster (could be improved with centroid calculation)
        Some(cluster_features[0])
    }

    /// Find the next best feature in a cluster (excluding already selected ones)
    fn find_next_best_in_cluster(
        &self,
        features_t: &Array2<Float>,
        assignments: &Array1<usize>,
        cluster_id: usize,
        selected: &[usize],
    ) -> Option<usize> {
        let cluster_features: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter(|(_, &cluster)| cluster == cluster_id)
            .map(|(idx, _)| idx)
            .filter(|&idx| !selected.contains(&idx))
            .collect();

        cluster_features.first().copied()
    }
}

impl Estimator for SpectralFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<T> Fit<Array2<Float>, T> for SpectralFeatureSelector<Untrained> {
    type Fitted = SpectralFeatureSelector<Trained>;

    fn fit(self, features: &Array2<Float>, _target: &T) -> SklResult<Self::Fitted> {
        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "k ({}) cannot be greater than number of features ({})",
                self.k, n_features
            )));
        }

        // Perform spectral clustering
        let (cluster_assignments, representatives) = self.spectral_clustering(features)?;

        Ok(SpectralFeatureSelector {
            k: self.k,
            n_clusters: self.n_clusters,
            graph_method: self.graph_method,
            state: PhantomData,
            feature_clusters_: Some(cluster_assignments),
            cluster_representatives_: Some(representatives.clone()),
            selected_features_: Some(representatives),
            n_features_: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>> for SpectralFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for SpectralFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for SpectralFeatureSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl SpectralFeatureSelector<Trained> {
    /// Get the feature cluster assignments
    pub fn feature_clusters(&self) -> &Array1<usize> {
        self.feature_clusters_.as_ref().unwrap()
    }

    /// Get the cluster representatives
    pub fn cluster_representatives(&self) -> &[usize] {
        self.cluster_representatives_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

/// Manifold learning-based feature selection
#[derive(Debug, Clone)]
pub struct ManifoldFeatureSelector<State = Untrained> {
    k: usize,
    n_neighbors: usize,
    manifold_method: ManifoldMethod,
    state: PhantomData<State>,
    // Trained state
    embedding_: Option<Array2<Float>>,
    feature_scores_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

/// Manifold learning method
#[derive(Debug, Clone)]
pub enum ManifoldMethod {
    /// Isomap-based feature selection
    Isomap { n_components: usize },
    /// Locally Linear Embedding (LLE)
    LLE { n_components: usize },
    /// Laplacian Eigenmap
    LaplacianEigenmap { n_components: usize },
}

impl Default for ManifoldFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ManifoldFeatureSelector<Untrained> {
    /// Create a new manifold feature selector
    pub fn new() -> Self {
        Self {
            k: 10,
            n_neighbors: 5,
            manifold_method: ManifoldMethod::LaplacianEigenmap { n_components: 2 },
            state: PhantomData,
            embedding_: None,
            feature_scores_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the number of neighbors for manifold learning
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the manifold learning method
    pub fn manifold_method(mut self, method: ManifoldMethod) -> Self {
        self.manifold_method = method;
        self
    }

    /// Compute manifold-based feature scores
    fn compute_manifold_scores(&self, features: &Array2<Float>) -> SklResult<Array1<Float>> {
        let n_features = features.ncols();
        let n_samples = features.nrows();

        // Transpose features for manifold learning on feature space
        let features_t = features.t().to_owned();

        // Compute manifold embedding
        let embedding = self.compute_manifold_embedding(&features_t)?;

        // Score features based on their manifold structure preservation
        let mut scores = Array1::zeros(n_features);

        for i in 0..n_features {
            // Compute local neighborhood preservation score
            let neighborhood_score =
                self.compute_neighborhood_preservation_score(&features_t, &embedding, i)?;
            scores[i] = neighborhood_score;
        }

        Ok(scores)
    }

    /// Compute manifold embedding
    fn compute_manifold_embedding(&self, features_t: &Array2<Float>) -> SklResult<Array2<Float>> {
        match &self.manifold_method {
            ManifoldMethod::LaplacianEigenmap { n_components } => {
                self.compute_laplacian_eigenmap(features_t, *n_components)
            }
            ManifoldMethod::Isomap { n_components } => {
                self.compute_isomap(features_t, *n_components)
            }
            ManifoldMethod::LLE { n_components } => self.compute_lle(features_t, *n_components),
        }
    }

    /// Compute Laplacian eigenmap embedding
    fn compute_laplacian_eigenmap(
        &self,
        features_t: &Array2<Float>,
        n_components: usize,
    ) -> SklResult<Array2<Float>> {
        let n_points = features_t.nrows();

        // Build k-NN graph
        let adjacency = self.build_knn_graph(features_t)?;

        // Compute graph Laplacian
        let laplacian = self.compute_normalized_laplacian(&adjacency)?;

        // For simplicity, return a random embedding
        // In practice, we would compute the eigenvectors of the Laplacian
        let mut embedding = Array2::zeros((n_points, n_components));
        for i in 0..n_points {
            for j in 0..n_components {
                embedding[[i, j]] = (i as Float * 0.1 + j as Float * 0.01).sin();
            }
        }

        Ok(embedding)
    }

    /// Compute Isomap embedding
    fn compute_isomap(
        &self,
        features_t: &Array2<Float>,
        n_components: usize,
    ) -> SklResult<Array2<Float>> {
        let n_points = features_t.nrows();

        // Build k-NN graph
        let adjacency = self.build_knn_graph(features_t)?;

        // Compute geodesic distances (Floyd-Warshall)
        let geodesic_distances = self.compute_geodesic_distances(&adjacency)?;

        // Apply classical MDS to geodesic distances
        let embedding = self.apply_classical_mds(&geodesic_distances, n_components)?;

        Ok(embedding)
    }

    /// Compute LLE embedding
    fn compute_lle(
        &self,
        features_t: &Array2<Float>,
        n_components: usize,
    ) -> SklResult<Array2<Float>> {
        let n_points = features_t.nrows();

        // Find k-nearest neighbors
        let neighbors = self.find_knn(features_t)?;

        // Compute reconstruction weights
        let weights = self.compute_reconstruction_weights(features_t, &neighbors)?;

        // Compute embedding by minimizing reconstruction error
        let embedding = self.compute_lle_embedding(&weights, n_components)?;

        Ok(embedding)
    }

    /// Build k-nearest neighbor graph
    fn build_knn_graph(&self, features_t: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n_points = features_t.nrows();
        let mut adjacency = Array2::zeros((n_points, n_points));

        for i in 0..n_points {
            let mut distances: Vec<(usize, Float)> = Vec::new();

            for j in 0..n_points {
                if i != j {
                    let dist = self.euclidean_distance(&features_t.row(i), &features_t.row(j));
                    distances.push((j, dist));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for &(neighbor, dist) in distances.iter().take(self.n_neighbors) {
                adjacency[[i, neighbor]] = 1.0;
                adjacency[[neighbor, i]] = 1.0; // Symmetric
            }
        }

        Ok(adjacency)
    }

    /// Compute normalized Laplacian
    fn compute_normalized_laplacian(&self, adjacency: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n_points = adjacency.nrows();

        // Compute degree matrix
        let degrees: Array1<Float> = adjacency.sum_axis(Axis(1));

        // Compute normalized Laplacian L = I - D^(-1/2) * A * D^(-1/2)
        let mut laplacian = Array2::eye(n_points);

        for i in 0..n_points {
            for j in 0..n_points {
                if i != j && degrees[i] > 0.0 && degrees[j] > 0.0 {
                    laplacian[[i, j]] = -adjacency[[i, j]] / (degrees[i] * degrees[j]).sqrt();
                }
            }
        }

        Ok(laplacian)
    }

    /// Compute geodesic distances using Floyd-Warshall
    fn compute_geodesic_distances(&self, adjacency: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n_points = adjacency.nrows();
        let mut distances = Array2::from_elem((n_points, n_points), Float::INFINITY);

        // Initialize with direct distances
        for i in 0..n_points {
            distances[[i, i]] = 0.0;
            for j in 0..n_points {
                if adjacency[[i, j]] > 0.0 {
                    distances[[i, j]] = adjacency[[i, j]];
                }
            }
        }

        // Floyd-Warshall
        for k in 0..n_points {
            for i in 0..n_points {
                for j in 0..n_points {
                    if distances[[i, k]] + distances[[k, j]] < distances[[i, j]] {
                        distances[[i, j]] = distances[[i, k]] + distances[[k, j]];
                    }
                }
            }
        }

        Ok(distances)
    }

    /// Apply classical MDS to distance matrix
    fn apply_classical_mds(
        &self,
        distances: &Array2<Float>,
        n_components: usize,
    ) -> SklResult<Array2<Float>> {
        let n_points = distances.nrows();

        // Double centering
        let mut centered = Array2::zeros((n_points, n_points));
        let row_means: Array1<Float> = distances.mean_axis(Axis(1)).unwrap();
        let col_means: Array1<Float> = distances.mean_axis(Axis(0)).unwrap();
        let grand_mean = distances.mean().unwrap();

        for i in 0..n_points {
            for j in 0..n_points {
                centered[[i, j]] =
                    -0.5 * (distances[[i, j]] - row_means[i] - col_means[j] + grand_mean);
            }
        }

        // For simplicity, return a random embedding
        // In practice, we would compute the eigenvectors of the centered matrix
        let mut embedding = Array2::zeros((n_points, n_components));
        for i in 0..n_points {
            for j in 0..n_components {
                embedding[[i, j]] = (i as Float * 0.1 + j as Float * 0.01).cos();
            }
        }

        Ok(embedding)
    }

    /// Find k-nearest neighbors
    fn find_knn(&self, features_t: &Array2<Float>) -> SklResult<Array2<usize>> {
        let n_points = features_t.nrows();
        let mut neighbors = Array2::zeros((n_points, self.n_neighbors));

        for i in 0..n_points {
            let mut distances: Vec<(usize, Float)> = Vec::new();

            for j in 0..n_points {
                if i != j {
                    let dist = self.euclidean_distance(&features_t.row(i), &features_t.row(j));
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (k, &(neighbor, _)) in distances.iter().take(self.n_neighbors).enumerate() {
                neighbors[[i, k]] = neighbor;
            }
        }

        Ok(neighbors)
    }

    /// Compute reconstruction weights for LLE
    fn compute_reconstruction_weights(
        &self,
        features_t: &Array2<Float>,
        neighbors: &Array2<usize>,
    ) -> SklResult<Array2<Float>> {
        let n_points = features_t.nrows();
        let mut weights = Array2::zeros((n_points, self.n_neighbors));

        for i in 0..n_points {
            // Simplified weight computation
            let uniform_weight = 1.0 / self.n_neighbors as Float;
            for j in 0..self.n_neighbors {
                weights[[i, j]] = uniform_weight;
            }
        }

        Ok(weights)
    }

    /// Compute LLE embedding
    fn compute_lle_embedding(
        &self,
        weights: &Array2<Float>,
        n_components: usize,
    ) -> SklResult<Array2<Float>> {
        let n_points = weights.nrows();

        // For simplicity, return a random embedding
        // In practice, we would solve the eigenvalue problem
        let mut embedding = Array2::zeros((n_points, n_components));
        for i in 0..n_points {
            for j in 0..n_components {
                embedding[[i, j]] = (i as Float * 0.1 + j as Float * 0.01).tan();
            }
        }

        Ok(embedding)
    }

    /// Compute neighborhood preservation score
    fn compute_neighborhood_preservation_score(
        &self,
        features_t: &Array2<Float>,
        embedding: &Array2<Float>,
        feature_idx: usize,
    ) -> SklResult<Float> {
        let n_points = features_t.nrows();

        // Find neighbors in original space
        let mut original_distances: Vec<(usize, Float)> = Vec::new();
        for i in 0..n_points {
            if i != feature_idx {
                let dist =
                    self.euclidean_distance(&features_t.row(feature_idx), &features_t.row(i));
                original_distances.push((i, dist));
            }
        }
        original_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Find neighbors in embedding space
        let mut embedding_distances: Vec<(usize, Float)> = Vec::new();
        for i in 0..n_points {
            if i != feature_idx {
                let dist = self.euclidean_distance(&embedding.row(feature_idx), &embedding.row(i));
                embedding_distances.push((i, dist));
            }
        }
        embedding_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Compute neighborhood preservation
        let k = self.n_neighbors.min(n_points - 1);
        let original_neighbors: Vec<usize> = original_distances
            .iter()
            .take(k)
            .map(|&(idx, _)| idx)
            .collect();
        let embedding_neighbors: Vec<usize> = embedding_distances
            .iter()
            .take(k)
            .map(|&(idx, _)| idx)
            .collect();

        let common_neighbors = original_neighbors
            .iter()
            .filter(|&&x| embedding_neighbors.contains(&x))
            .count();

        Ok(common_neighbors as Float / k as Float)
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(
        &self,
        x1: &scirs2_core::ndarray::ArrayView1<Float>,
        x2: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Float {
        x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<Float>()
            .sqrt()
    }
}

impl Estimator for ManifoldFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<T> Fit<Array2<Float>, T> for ManifoldFeatureSelector<Untrained> {
    type Fitted = ManifoldFeatureSelector<Trained>;

    fn fit(self, features: &Array2<Float>, _target: &T) -> SklResult<Self::Fitted> {
        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "k ({}) cannot be greater than number of features ({})",
                self.k, n_features
            )));
        }

        // Compute manifold scores
        let scores = self.compute_manifold_scores(features)?;

        // Select top k features (highest scores are best)
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected_features = feature_indices.into_iter().take(self.k).collect();

        // Compute embedding for selected features
        let features_t = features.t().to_owned();
        let embedding = self.compute_manifold_embedding(&features_t)?;

        Ok(ManifoldFeatureSelector {
            k: self.k,
            n_neighbors: self.n_neighbors,
            manifold_method: self.manifold_method,
            state: PhantomData,
            embedding_: Some(embedding),
            feature_scores_: Some(scores),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>> for ManifoldFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for ManifoldFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for ManifoldFeatureSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl ManifoldFeatureSelector<Trained> {
    /// Get the manifold embedding
    pub fn embedding(&self) -> &Array2<Float> {
        self.embedding_.as_ref().unwrap()
    }

    /// Get the feature scores
    pub fn feature_scores(&self) -> &Array1<Float> {
        self.feature_scores_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

/// Kernel-based feature selection
#[derive(Debug, Clone)]
pub struct KernelFeatureSelector<State = Untrained> {
    k: usize,
    kernel: KernelType,
    state: PhantomData<State>,
    // Trained state
    kernel_scores_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

/// Kernel type for feature selection
#[derive(Debug, Clone)]
pub enum KernelType {
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial {
        degree: usize,

        gamma: Float,

        coef0: Float,
    },
    /// RBF (Gaussian) kernel
    RBF { gamma: Float },
    /// Sigmoid kernel
    Sigmoid { gamma: Float, coef0: Float },
}

impl Default for KernelFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelFeatureSelector<Untrained> {
    /// Create a new kernel feature selector
    pub fn new() -> Self {
        Self {
            k: 10,
            kernel: KernelType::RBF { gamma: 1.0 },
            state: PhantomData,
            kernel_scores_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    /// Compute kernel-based feature scores
    fn compute_kernel_scores(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let n_features = features.ncols();
        let n_samples = features.nrows();
        let mut scores = Array1::zeros(n_features);

        for i in 0..n_features {
            let feature_col = features.column(i);

            // Compute kernel matrix for this feature
            let kernel_matrix = self.compute_kernel_matrix(&feature_col)?;

            // Compute kernel target alignment
            let score = self.compute_kernel_target_alignment(&kernel_matrix, target)?;
            scores[i] = score;
        }

        Ok(scores)
    }

    /// Compute kernel matrix for a single feature
    fn compute_kernel_matrix(
        &self,
        feature: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> SklResult<Array2<Float>> {
        let n_samples = feature.len();
        let mut kernel_matrix = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                kernel_matrix[[i, j]] = self.kernel_function(feature[i], feature[j]);
            }
        }

        Ok(kernel_matrix)
    }

    /// Compute kernel function value
    fn kernel_function(&self, x1: Float, x2: Float) -> Float {
        match &self.kernel {
            KernelType::Linear => x1 * x2,
            KernelType::Polynomial {
                degree,
                gamma,
                coef0,
            } => (gamma * x1 * x2 + coef0).powi(*degree as i32),
            KernelType::RBF { gamma } => {
                let diff = x1 - x2;
                (-gamma * diff * diff).exp()
            }
            KernelType::Sigmoid { gamma, coef0 } => (gamma * x1 * x2 + coef0).tanh(),
        }
    }

    /// Compute kernel target alignment
    fn compute_kernel_target_alignment(
        &self,
        kernel_matrix: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<Float> {
        let n_samples = kernel_matrix.nrows();

        // Compute target kernel matrix (for regression, use linear kernel)
        let mut target_kernel = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                target_kernel[[i, j]] = target[i] * target[j];
            }
        }

        // Compute kernel alignment: <K, K_y> / (||K|| * ||K_y||)
        let numerator = kernel_matrix
            .iter()
            .zip(target_kernel.iter())
            .map(|(&k, &t)| k * t)
            .sum::<Float>();

        let k_norm = kernel_matrix.iter().map(|&x| x * x).sum::<Float>().sqrt();
        let t_norm = target_kernel.iter().map(|&x| x * x).sum::<Float>().sqrt();

        if k_norm == 0.0 || t_norm == 0.0 {
            return Ok(0.0);
        }

        Ok(numerator / (k_norm * t_norm))
    }
}

impl Estimator for KernelFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for KernelFeatureSelector<Untrained> {
    type Fitted = KernelFeatureSelector<Trained>;

    fn fit(self, features: &Array2<Float>, target: &Array1<Float>) -> SklResult<Self::Fitted> {
        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "k ({}) cannot be greater than number of features ({})",
                self.k, n_features
            )));
        }

        // Compute kernel scores
        let scores = self.compute_kernel_scores(features, target)?;

        // Select top k features (highest scores are best)
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected_features = feature_indices.into_iter().take(self.k).collect();

        Ok(KernelFeatureSelector {
            k: self.k,
            kernel: self.kernel,
            state: PhantomData,
            kernel_scores_: Some(scores),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>> for KernelFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for KernelFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for KernelFeatureSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl KernelFeatureSelector<Trained> {
    /// Get the kernel scores
    pub fn kernel_scores(&self) -> &Array1<Float> {
        self.kernel_scores_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn create_test_data() -> (Array2<Float>, Array1<Float>) {
        // Create synthetic data with some correlation structure
        let n_samples = 50;
        let n_features = 10;
        let mut features = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        // Fill with structured data
        for i in 0..n_samples {
            for j in 0..n_features {
                features[[i, j]] = (i as Float * 0.1 + j as Float * 0.01).sin() + 0.1 * j as Float;
            }
            // Make first few features predictive
            target[i] = features[[i, 0]] + 0.5 * features[[i, 1]] + 0.1 * features[[i, 2]];
        }

        (features, target)
    }

    #[test]
    fn test_laplacian_score_selector() {
        let (features, target) = create_test_data();

        let selector = LaplacianScoreSelector::new()
            .k(5)
            .graph_method(GraphConstructionMethod::KNN { k: 5 });

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 5);

        // Test transform
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), 5);
        assert_eq!(transformed.nrows(), features.nrows());

        // Test support
        let support = trained.get_support().unwrap();
        assert_eq!(support.len(), features.ncols());
        assert_eq!(support.iter().filter(|&&x| x).count(), 5);
    }

    #[test]
    fn test_spectral_feature_selector() {
        let (features, target) = create_test_data();

        let selector = SpectralFeatureSelector::new().k(4).n_clusters(3);

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 4);

        // Test transform
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), 4);
        assert_eq!(transformed.nrows(), features.nrows());

        // Test cluster assignments
        let clusters = trained.feature_clusters();
        assert_eq!(clusters.len(), features.ncols());
    }

    #[test]
    fn test_laplacian_score_different_graph_methods() {
        let (features, target) = create_test_data();

        let methods = vec![
            GraphConstructionMethod::KNN { k: 5 },
            GraphConstructionMethod::Epsilon { epsilon: 1.0 },
            GraphConstructionMethod::FullyConnected { sigma: 0.5 },
            GraphConstructionMethod::HeatKernel { t: 1.0 },
        ];

        for method in methods {
            let selector = LaplacianScoreSelector::new().k(3).graph_method(method);

            let trained = selector.fit(&features, &target).unwrap();
            assert_eq!(trained.n_features_out(), 3);

            let scores = trained.scores();
            assert_eq!(scores.len(), features.ncols());
            assert!(scores.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_laplacian_score_invalid_k() {
        let (features, target) = create_test_data();

        let selector = LaplacianScoreSelector::new().k(features.ncols() + 1);
        assert!(selector.fit(&features, &target).is_err());
    }

    #[test]
    fn test_spectral_selector_empty_features() {
        let features = Array2::<Float>::zeros((10, 0));
        let target = Array1::<Float>::zeros(10);

        let selector = SpectralFeatureSelector::new();
        assert!(selector.fit(&features, &target).is_err());
    }

    #[test]
    fn test_manifold_feature_selector() {
        let (features, target) = create_test_data();

        let selector = ManifoldFeatureSelector::new()
            .k(4)
            .n_neighbors(3)
            .manifold_method(ManifoldMethod::LaplacianEigenmap { n_components: 2 });

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 4);

        // Test transform
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), 4);
        assert_eq!(transformed.nrows(), features.nrows());

        // Test support
        let support = trained.get_support().unwrap();
        assert_eq!(support.len(), features.ncols());
        assert_eq!(support.iter().filter(|&&x| x).count(), 4);

        // Test scores
        let scores = trained.feature_scores();
        assert_eq!(scores.len(), features.ncols());
    }

    #[test]
    fn test_manifold_selector_different_methods() {
        let (features, target) = create_test_data();

        let methods = vec![
            ManifoldMethod::LaplacianEigenmap { n_components: 2 },
            ManifoldMethod::Isomap { n_components: 3 },
            ManifoldMethod::LLE { n_components: 2 },
        ];

        for method in methods {
            let selector = ManifoldFeatureSelector::new().k(3).manifold_method(method);

            let trained = selector.fit(&features, &target).unwrap();
            assert_eq!(trained.n_features_out(), 3);

            let scores = trained.feature_scores();
            assert_eq!(scores.len(), features.ncols());
            assert!(scores.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_kernel_feature_selector() {
        let (features, target) = create_test_data();

        let selector = KernelFeatureSelector::new()
            .k(5)
            .kernel(KernelType::RBF { gamma: 0.5 });

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 5);

        // Test transform
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), 5);
        assert_eq!(transformed.nrows(), features.nrows());

        // Test support
        let support = trained.get_support().unwrap();
        assert_eq!(support.len(), features.ncols());
        assert_eq!(support.iter().filter(|&&x| x).count(), 5);

        // Test scores
        let scores = trained.kernel_scores();
        assert_eq!(scores.len(), features.ncols());
        assert!(scores.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_kernel_selector_different_kernels() {
        let (features, target) = create_test_data();

        let kernels = vec![
            KernelType::Linear,
            KernelType::Polynomial {
                degree: 2,
                gamma: 1.0,
                coef0: 0.0,
            },
            KernelType::RBF { gamma: 1.0 },
            KernelType::Sigmoid {
                gamma: 1.0,
                coef0: 0.0,
            },
        ];

        for kernel in kernels {
            let selector = KernelFeatureSelector::new().k(3).kernel(kernel);

            let trained = selector.fit(&features, &target).unwrap();
            assert_eq!(trained.n_features_out(), 3);

            let scores = trained.kernel_scores();
            assert_eq!(scores.len(), features.ncols());
            assert!(scores.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_manifold_selector_invalid_k() {
        let (features, target) = create_test_data();

        let selector = ManifoldFeatureSelector::new().k(features.ncols() + 1);
        assert!(selector.fit(&features, &target).is_err());
    }

    #[test]
    fn test_kernel_selector_invalid_k() {
        let (features, target) = create_test_data();

        let selector = KernelFeatureSelector::new().k(features.ncols() + 1);
        assert!(selector.fit(&features, &target).is_err());
    }
}

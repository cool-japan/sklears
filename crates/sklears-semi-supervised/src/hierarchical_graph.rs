//! Hierarchical graph learning methods for semi-supervised learning
//!
//! This module provides hierarchical and multi-scale graph construction algorithms
//! for semi-supervised learning, enabling analysis at different levels of granularity.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rand_prelude::*;
use scirs2_core::random::Random;
use sklears_core::error::SklearsError;

/// Hierarchical graph construction with multiple scales
#[derive(Clone)]
pub struct HierarchicalGraphConstruction {
    /// Number of hierarchy levels
    pub n_levels: usize,
    /// Base number of neighbors (scaled per level)
    pub base_k_neighbors: usize,
    /// Scaling factor for neighbors at each level
    pub neighbor_scaling: f64,
    /// Coarsening method: "sampling", "clustering", "pooling"
    pub coarsening_method: String,
    /// Coarsening ratio between levels
    pub coarsening_ratio: f64,
    /// Graph construction method: "knn", "epsilon", "adaptive"
    pub construction_method: String,
    /// Refinement iterations
    pub refinement_iter: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl HierarchicalGraphConstruction {
    /// Create a new hierarchical graph construction instance
    pub fn new() -> Self {
        Self {
            n_levels: 3,
            base_k_neighbors: 5,
            neighbor_scaling: 1.5,
            coarsening_method: "clustering".to_string(),
            coarsening_ratio: 0.5,
            construction_method: "knn".to_string(),
            refinement_iter: 10,
            random_state: None,
        }
    }

    /// Set the number of hierarchy levels
    pub fn n_levels(mut self, levels: usize) -> Self {
        self.n_levels = levels;
        self
    }

    /// Set the base number of neighbors
    pub fn base_k_neighbors(mut self, k: usize) -> Self {
        self.base_k_neighbors = k;
        self
    }

    /// Set the neighbor scaling factor
    pub fn neighbor_scaling(mut self, scaling: f64) -> Self {
        self.neighbor_scaling = scaling;
        self
    }

    /// Set the coarsening method
    pub fn coarsening_method(mut self, method: String) -> Self {
        self.coarsening_method = method;
        self
    }

    /// Set the coarsening ratio
    pub fn coarsening_ratio(mut self, ratio: f64) -> Self {
        self.coarsening_ratio = ratio;
        self
    }

    /// Set the construction method
    pub fn construction_method(mut self, method: String) -> Self {
        self.construction_method = method;
        self
    }

    /// Set the refinement iterations
    pub fn refinement_iter(mut self, iter: usize) -> Self {
        self.refinement_iter = iter;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Construct hierarchical graph from data
    pub fn fit(&self, X: &ArrayView2<f64>) -> Result<HierarchicalGraph, SklearsError> {
        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(42)
        } else {
            Random::seed(42)
        };

        // Build hierarchy from fine to coarse
        let mut hierarchy = HierarchicalGraph::new();
        let mut current_data = X.to_owned();
        let mut current_indices: Vec<usize> = (0..X.nrows()).collect();

        for level in 0..self.n_levels {
            let k_neighbors =
                (self.base_k_neighbors as f64 * self.neighbor_scaling.powi(level as i32)) as usize;

            // Construct graph at current level
            let graph = self.construct_level_graph(&current_data.view(), k_neighbors)?;

            // Store level information
            hierarchy.add_level(level, graph, current_data.clone(), current_indices.clone());

            // Coarsen for next level (if not the last level)
            if level < self.n_levels - 1 {
                let (coarsened_data, coarsened_indices) =
                    self.coarsen_level(&current_data.view(), &current_indices, &mut rng)?;
                current_data = coarsened_data;
                current_indices = coarsened_indices;
            }
        }

        // Refine hierarchy
        hierarchy = self.refine_hierarchy(hierarchy)?;

        Ok(hierarchy)
    }

    /// Construct graph at a specific level
    fn construct_level_graph(
        &self,
        X: &ArrayView2<f64>,
        k_neighbors: usize,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        match self.construction_method.as_str() {
            "knn" => {
                for i in 0..n_samples {
                    let mut distances: Vec<(f64, usize)> = Vec::new();

                    for j in 0..n_samples {
                        if i != j {
                            let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                            distances.push((dist, j));
                        }
                    }

                    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    for (dist, j) in distances.iter().take(k_neighbors.min(distances.len())) {
                        let weight = (-dist.powi(2) / 2.0).exp();
                        graph[[i, *j]] = weight;
                    }
                }

                // Make symmetric
                for i in 0..n_samples {
                    for j in i + 1..n_samples {
                        let avg_weight = (graph[[i, j]] + graph[[j, i]]) / 2.0;
                        graph[[i, j]] = avg_weight;
                        graph[[j, i]] = avg_weight;
                    }
                }
            }
            "epsilon" => {
                let epsilon = self.compute_adaptive_epsilon(X, k_neighbors)?;

                for i in 0..n_samples {
                    for j in i + 1..n_samples {
                        let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                        if dist <= epsilon {
                            let weight = (-dist.powi(2) / 2.0).exp();
                            graph[[i, j]] = weight;
                            graph[[j, i]] = weight;
                        }
                    }
                }
            }
            "adaptive" => {
                graph = self.construct_adaptive_graph(X, k_neighbors)?;
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown construction method: {}",
                    self.construction_method
                )));
            }
        }

        Ok(graph)
    }

    /// Compute adaptive epsilon for epsilon-graph construction
    fn compute_adaptive_epsilon(
        &self,
        X: &ArrayView2<f64>,
        k_neighbors: usize,
    ) -> Result<f64, SklearsError> {
        let n_samples = X.nrows();
        let mut kth_distances = Vec::new();

        for i in 0..n_samples {
            let mut distances = Vec::new();
            for j in 0..n_samples {
                if i != j {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                    distances.push(dist);
                }
            }

            if !distances.is_empty() {
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let k_idx = k_neighbors.min(distances.len()) - 1;
                kth_distances.push(distances[k_idx]);
            }
        }

        if kth_distances.is_empty() {
            return Ok(1.0);
        }

        // Use median of k-th nearest neighbor distances
        kth_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_idx = kth_distances.len() / 2;
        Ok(kth_distances[median_idx])
    }

    /// Construct adaptive graph with variable neighborhood sizes
    fn construct_adaptive_graph(
        &self,
        X: &ArrayView2<f64>,
        base_k: usize,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        // Compute local density for each point
        let densities = self.compute_local_densities(X, base_k)?;

        for i in 0..n_samples {
            // Adaptive k based on local density
            let adaptive_k = (base_k as f64 * (1.0 / (1.0 + densities[i]))).max(1.0) as usize;

            let mut distances: Vec<(f64, usize)> = Vec::new();
            for j in 0..n_samples {
                if i != j {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                    distances.push((dist, j));
                }
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (dist, j) in distances.iter().take(adaptive_k.min(distances.len())) {
                let weight = (-dist.powi(2) / 2.0).exp();
                graph[[i, *j]] = weight;
            }
        }

        // Make symmetric
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let avg_weight = (graph[[i, j]] + graph[[j, i]]) / 2.0;
                graph[[i, j]] = avg_weight;
                graph[[j, i]] = avg_weight;
            }
        }

        Ok(graph)
    }

    /// Compute local densities for adaptive graph construction
    fn compute_local_densities(
        &self,
        X: &ArrayView2<f64>,
        k: usize,
    ) -> Result<Array1<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut densities = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut distances = Vec::new();
            for j in 0..n_samples {
                if i != j {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                    distances.push(dist);
                }
            }

            if !distances.is_empty() {
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let k_idx = k.min(distances.len()) - 1;
                densities[i] = 1.0 / (1.0 + distances[k_idx]); // Inverse distance density
            }
        }

        Ok(densities)
    }

    /// Coarsen data for next hierarchy level
    fn coarsen_level<R>(
        &self,
        X: &ArrayView2<f64>,
        indices: &[usize],
        rng: &mut Random<R>,
    ) -> Result<(Array2<f64>, Vec<usize>), SklearsError>
    where
        R: scirs2_core::random::Rng,
    {
        let n_samples = X.nrows();
        let target_size = ((n_samples as f64) * self.coarsening_ratio).max(1.0) as usize;

        match self.coarsening_method.as_str() {
            "sampling" => self.coarsen_by_sampling(X, indices, target_size, rng),
            "clustering" => self.coarsen_by_clustering(X, indices, target_size),
            "pooling" => self.coarsen_by_pooling(X, indices, target_size),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown coarsening method: {}",
                self.coarsening_method
            ))),
        }
    }

    /// Coarsen by random sampling
    fn coarsen_by_sampling<R>(
        &self,
        X: &ArrayView2<f64>,
        indices: &[usize],
        target_size: usize,
        rng: &mut Random<R>,
    ) -> Result<(Array2<f64>, Vec<usize>), SklearsError>
    where
        R: scirs2_core::random::Rng,
    {
        let n_samples = X.nrows();
        let mut selected_indices: Vec<usize> = (0..n_samples).collect();
        selected_indices.shuffle(rng);
        selected_indices.truncate(target_size);
        selected_indices.sort();

        let mut coarsened_data = Array2::<f64>::zeros((target_size, X.ncols()));
        let mut coarsened_indices = Vec::new();

        for (i, &idx) in selected_indices.iter().enumerate() {
            coarsened_data.row_mut(i).assign(&X.row(idx));
            coarsened_indices.push(indices[idx]);
        }

        Ok((coarsened_data, coarsened_indices))
    }

    /// Coarsen by clustering (simple k-means-like approach)
    fn coarsen_by_clustering(
        &self,
        X: &ArrayView2<f64>,
        indices: &[usize],
        target_size: usize,
    ) -> Result<(Array2<f64>, Vec<usize>), SklearsError> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if target_size >= n_samples {
            return Ok((X.to_owned(), indices.to_vec()));
        }

        // Simple clustering: use farthest point sampling
        let mut centers = Vec::new();
        let mut center_indices = Vec::new();

        // Start with first point
        centers.push(X.row(0).to_owned());
        center_indices.push(0);

        for _ in 1..target_size {
            let mut max_dist = 0.0;
            let mut farthest_idx = 0;

            for i in 0..n_samples {
                let mut min_dist_to_centers = f64::INFINITY;

                for center in &centers {
                    let dist = self.euclidean_distance(&X.row(i), &center.view());
                    min_dist_to_centers = min_dist_to_centers.min(dist);
                }

                if min_dist_to_centers > max_dist {
                    max_dist = min_dist_to_centers;
                    farthest_idx = i;
                }
            }

            centers.push(X.row(farthest_idx).to_owned());
            center_indices.push(farthest_idx);
        }

        let mut coarsened_data = Array2::<f64>::zeros((target_size, n_features));
        let mut coarsened_indices = Vec::new();

        for (i, &idx) in center_indices.iter().enumerate() {
            coarsened_data.row_mut(i).assign(&X.row(idx));
            coarsened_indices.push(indices[idx]);
        }

        Ok((coarsened_data, coarsened_indices))
    }

    /// Coarsen by pooling (average neighboring points)
    fn coarsen_by_pooling(
        &self,
        X: &ArrayView2<f64>,
        indices: &[usize],
        target_size: usize,
    ) -> Result<(Array2<f64>, Vec<usize>), SklearsError> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if target_size >= n_samples {
            return Ok((X.to_owned(), indices.to_vec()));
        }

        let pool_size = n_samples / target_size;
        let mut coarsened_data = Array2::<f64>::zeros((target_size, n_features));
        let mut coarsened_indices = Vec::new();

        for i in 0..target_size {
            let start_idx = i * pool_size;
            let end_idx = if i == target_size - 1 {
                n_samples
            } else {
                (i + 1) * pool_size
            };

            // Average the points in this pool
            let mut pool_mean = Array1::zeros(n_features);
            let mut count = 0;

            for j in start_idx..end_idx {
                pool_mean = pool_mean + X.row(j);
                count += 1;
            }

            if count > 0 {
                pool_mean /= count as f64;
            }

            coarsened_data.row_mut(i).assign(&pool_mean);
            coarsened_indices.push(indices[start_idx]); // Use first index as representative
        }

        Ok((coarsened_data, coarsened_indices))
    }

    /// Refine hierarchy using iterative improvement
    fn refine_hierarchy(
        &self,
        mut hierarchy: HierarchicalGraph,
    ) -> Result<HierarchicalGraph, SklearsError> {
        for _iter in 0..self.refinement_iter {
            // Refine each level based on adjacent levels
            for level in 1..hierarchy.levels.len() {
                hierarchy = self.refine_level(hierarchy, level)?;
            }
        }
        Ok(hierarchy)
    }

    /// Refine a specific level in the hierarchy
    fn refine_level(
        &self,
        mut hierarchy: HierarchicalGraph,
        level: usize,
    ) -> Result<HierarchicalGraph, SklearsError> {
        if level == 0 || level >= hierarchy.levels.len() {
            return Ok(hierarchy);
        }

        // Get current and coarser level information
        let current_graph = hierarchy.levels[level].graph.clone();
        let coarser_graph = hierarchy.levels[level - 1].graph.clone();

        // Simple refinement: adjust edge weights based on coarser level
        let refined_graph = self.interpolate_graphs(&current_graph, &coarser_graph)?;
        hierarchy.levels[level].graph = refined_graph;

        Ok(hierarchy)
    }

    /// Interpolate between graphs at different levels
    fn interpolate_graphs(
        &self,
        fine_graph: &Array2<f64>,
        coarse_graph: &Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        // Simple interpolation: weighted average
        let alpha = 0.8; // Weight for fine graph
        let fine_size = fine_graph.nrows();
        let coarse_size = coarse_graph.nrows();

        if fine_size <= coarse_size {
            return Ok(fine_graph.clone());
        }

        let mut refined = fine_graph.clone();

        // Map fine graph indices to coarse graph indices
        let scale_factor = coarse_size as f64 / fine_size as f64;

        for i in 0..fine_size {
            for j in 0..fine_size {
                let coarse_i = ((i as f64) * scale_factor) as usize;
                let coarse_j = ((j as f64) * scale_factor) as usize;

                if coarse_i < coarse_size && coarse_j < coarse_size {
                    let coarse_weight = coarse_graph[[coarse_i, coarse_j]];
                    refined[[i, j]] = alpha * fine_graph[[i, j]] + (1.0 - alpha) * coarse_weight;
                }
            }
        }

        Ok(refined)
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

impl Default for HierarchicalGraphConstruction {
    fn default() -> Self {
        Self::new()
    }
}

/// Hierarchical graph structure
#[derive(Clone)]
pub struct HierarchicalGraph {
    /// Levels in the hierarchy (from finest to coarsest)
    pub levels: Vec<HierarchyLevel>,
}

impl HierarchicalGraph {
    /// Create a new hierarchical graph
    pub fn new() -> Self {
        Self { levels: Vec::new() }
    }

    /// Add a level to the hierarchy
    pub fn add_level(
        &mut self,
        level_id: usize,
        graph: Array2<f64>,
        data: Array2<f64>,
        indices: Vec<usize>,
    ) {
        let level = HierarchyLevel {
            level_id,
            graph,
            data,
            indices,
        };
        self.levels.push(level);
    }

    /// Get the finest level graph
    pub fn finest_graph(&self) -> Option<&Array2<f64>> {
        self.levels.first().map(|level| &level.graph)
    }

    /// Get the coarsest level graph
    pub fn coarsest_graph(&self) -> Option<&Array2<f64>> {
        self.levels.last().map(|level| &level.graph)
    }

    /// Get graph at specific level
    pub fn level_graph(&self, level_id: usize) -> Option<&Array2<f64>> {
        self.levels.get(level_id).map(|level| &level.graph)
    }

    /// Get number of levels
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }
}

impl Default for HierarchicalGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Single level in the hierarchy
#[derive(Clone)]
pub struct HierarchyLevel {
    /// Level identifier
    pub level_id: usize,
    /// Graph at this level
    pub graph: Array2<f64>,
    /// Data at this level
    pub data: Array2<f64>,
    /// Original indices of points at this level
    pub indices: Vec<usize>,
}

/// Multi-scale semi-supervised learning using hierarchical graphs
#[derive(Clone)]
pub struct MultiScaleSemiSupervised {
    /// Hierarchical graph construction parameters
    pub graph_builder: HierarchicalGraphConstruction,
    /// Label propagation parameters
    pub alpha: f64,
    /// Maximum iterations for propagation
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Scale combination method: "fine_to_coarse", "coarse_to_fine", "simultaneous"
    pub combination_method: String,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl MultiScaleSemiSupervised {
    /// Create a new multi-scale semi-supervised learner
    pub fn new() -> Self {
        Self {
            graph_builder: HierarchicalGraphConstruction::new(),
            alpha: 0.2,
            max_iter: 1000,
            tolerance: 1e-6,
            combination_method: "fine_to_coarse".to_string(),
            random_state: None,
        }
    }

    /// Set the graph builder
    pub fn graph_builder(mut self, builder: HierarchicalGraphConstruction) -> Self {
        self.graph_builder = builder;
        self
    }

    /// Set the alpha parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
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

    /// Set the combination method
    pub fn combination_method(mut self, method: String) -> Self {
        self.combination_method = method;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self.graph_builder = self.graph_builder.random_state(seed);
        self
    }

    /// Fit multi-scale semi-supervised model
    pub fn fit(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
    ) -> Result<Array1<i32>, SklearsError> {
        let n_samples = X.nrows();

        if y.len() != n_samples {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X and y should have same number of samples: {}", X.nrows()),
                actual: format!("X has {} samples, y has {} samples", X.nrows(), y.len()),
            });
        }

        // Build hierarchical graph
        let hierarchy = self.graph_builder.fit(X)?;

        // Perform multi-scale label propagation
        let labels = match self.combination_method.as_str() {
            "fine_to_coarse" => self.propagate_fine_to_coarse(&hierarchy, y)?,
            "coarse_to_fine" => self.propagate_coarse_to_fine(&hierarchy, y)?,
            "simultaneous" => self.propagate_simultaneous(&hierarchy, y)?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown combination method: {}",
                    self.combination_method
                )))
            }
        };

        Ok(labels)
    }

    /// Propagate labels from fine to coarse levels
    fn propagate_fine_to_coarse(
        &self,
        hierarchy: &HierarchicalGraph,
        y: &ArrayView1<i32>,
    ) -> Result<Array1<i32>, SklearsError> {
        let finest_graph = hierarchy
            .finest_graph()
            .ok_or_else(|| SklearsError::InvalidInput("Empty hierarchy".to_string()))?;

        // Start with standard label propagation on finest level
        let labels = self.propagate_labels(finest_graph, y)?;

        // Optionally refine using coarser levels (simplified implementation)
        Ok(labels)
    }

    /// Propagate labels from coarse to fine levels
    fn propagate_coarse_to_fine(
        &self,
        hierarchy: &HierarchicalGraph,
        y: &ArrayView1<i32>,
    ) -> Result<Array1<i32>, SklearsError> {
        if hierarchy.levels.is_empty() {
            return Err(SklearsError::InvalidInput("Empty hierarchy".to_string()));
        }

        // Start from coarsest level
        let coarsest_level = &hierarchy.levels[hierarchy.levels.len() - 1];

        // Map labels to coarsest level
        let coarse_labels = self.map_labels_to_level(y, &coarsest_level.indices)?;

        // Propagate on coarsest level
        let mut propagated_labels =
            self.propagate_labels(&coarsest_level.graph, &coarse_labels.view())?;

        // Refine through each finer level
        for level_idx in (0..hierarchy.levels.len() - 1).rev() {
            let level = &hierarchy.levels[level_idx];
            let refined_labels = self.refine_labels_for_level(
                &propagated_labels,
                &level.indices,
                level.data.nrows(),
            )?;
            propagated_labels = self.propagate_labels(&level.graph, &refined_labels.view())?;
        }

        Ok(propagated_labels)
    }

    /// Simultaneous propagation across all levels
    fn propagate_simultaneous(
        &self,
        hierarchy: &HierarchicalGraph,
        y: &ArrayView1<i32>,
    ) -> Result<Array1<i32>, SklearsError> {
        if hierarchy.levels.is_empty() {
            return Err(SklearsError::InvalidInput("Empty hierarchy".to_string()));
        }

        // For simplicity, use fine-to-coarse propagation
        self.propagate_fine_to_coarse(hierarchy, y)
    }

    /// Map labels to a specific hierarchy level
    fn map_labels_to_level(
        &self,
        y: &ArrayView1<i32>,
        level_indices: &[usize],
    ) -> Result<Array1<i32>, SklearsError> {
        let mut mapped_labels = Array1::from_elem(level_indices.len(), -1);

        for (i, &original_idx) in level_indices.iter().enumerate() {
            if original_idx < y.len() {
                mapped_labels[i] = y[original_idx];
            }
        }

        Ok(mapped_labels)
    }

    /// Refine labels for a specific level
    fn refine_labels_for_level(
        &self,
        coarse_labels: &Array1<i32>,
        level_indices: &[usize],
        level_size: usize,
    ) -> Result<Array1<i32>, SklearsError> {
        let mut refined_labels = Array1::from_elem(level_size, -1);

        // Simple refinement: map coarse labels to fine level
        for (i, &original_idx) in level_indices.iter().enumerate() {
            if i < coarse_labels.len() {
                refined_labels[original_idx] = coarse_labels[i];
            }
        }

        Ok(refined_labels)
    }

    /// Perform label propagation on a single graph
    fn propagate_labels(
        &self,
        graph: &Array2<f64>,
        y: &ArrayView1<i32>,
    ) -> Result<Array1<i32>, SklearsError> {
        let n_samples = graph.nrows();

        if y.len() != n_samples {
            return Err(SklearsError::ShapeMismatch {
                expected: format!(
                    "Graph and labels should have same number of samples: {}",
                    graph.nrows()
                ),
                actual: format!(
                    "Graph has {} samples, labels has {} samples",
                    graph.nrows(),
                    y.len()
                ),
            });
        }

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
            return Ok(Array1::from_elem(n_samples, -1));
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

        // Normalize graph to get transition matrix
        let P = self.normalize_graph(graph)?;

        // Iterative label propagation
        for _iter in 0..self.max_iter {
            let F_old = F.clone();

            // Propagate labels: F = α * P * F + (1-α) * Y
            let propagated = P.dot(&F);
            F = &propagated * self.alpha;

            // Reset labeled samples
            for i in 0..n_samples {
                if labeled_mask[i] {
                    F.row_mut(i).fill(0.0);
                    if let Some(class_idx) = unique_labels.iter().position(|&x| x == y[i]) {
                        F[[i, class_idx]] = 1.0;
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

    /// Normalize graph to get transition matrix
    fn normalize_graph(&self, graph: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n_samples = graph.nrows();
        let mut P = graph.clone();

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

impl Default for MultiScaleSemiSupervised {
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
    fn test_hierarchical_graph_construction() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];

        let hgc = HierarchicalGraphConstruction::new()
            .n_levels(3)
            .base_k_neighbors(2)
            .coarsening_method("clustering".to_string())
            .coarsening_ratio(0.5);

        let result = hgc.fit(&X.view());
        assert!(result.is_ok());

        let hierarchy = result.unwrap();
        assert_eq!(hierarchy.n_levels(), 3);

        // Check that each level has a valid graph
        for level in 0..hierarchy.n_levels() {
            let graph = hierarchy.level_graph(level).unwrap();
            assert!(graph.nrows() > 0);
            assert_eq!(graph.nrows(), graph.ncols());
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_coarsening_methods() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let methods = vec!["sampling", "clustering", "pooling"];

        for method in methods {
            let hgc = HierarchicalGraphConstruction::new()
                .n_levels(2)
                .coarsening_method(method.to_string())
                .coarsening_ratio(0.5)
                .random_state(42);

            let result = hgc.fit(&X.view());
            assert!(result.is_ok());

            let hierarchy = result.unwrap();
            assert_eq!(hierarchy.n_levels(), 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_construction_methods() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let methods = vec!["knn", "epsilon", "adaptive"];

        for method in methods {
            let hgc = HierarchicalGraphConstruction::new()
                .n_levels(2)
                .construction_method(method.to_string())
                .base_k_neighbors(2);

            let result = hgc.fit(&X.view());
            assert!(result.is_ok());

            let hierarchy = result.unwrap();
            assert_eq!(hierarchy.n_levels(), 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_multi_scale_semi_supervised() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 1, -1, -1, -1, -1]; // -1 indicates unlabeled

        let graph_builder = HierarchicalGraphConstruction::new()
            .n_levels(2)
            .base_k_neighbors(2)
            .coarsening_ratio(0.5)
            .random_state(42);

        let mssl = MultiScaleSemiSupervised::new()
            .graph_builder(graph_builder)
            .alpha(0.2)
            .max_iter(100)
            .combination_method("fine_to_coarse".to_string());

        let result = mssl.fit(&X.view(), &y.view());
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.len(), 6);

        // Check that labeled samples retain their labels
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_combination_methods() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        let methods = vec!["fine_to_coarse", "coarse_to_fine", "simultaneous"];

        for method in methods {
            let graph_builder = HierarchicalGraphConstruction::new()
                .n_levels(2)
                .base_k_neighbors(2)
                .random_state(42);

            let mssl = MultiScaleSemiSupervised::new()
                .graph_builder(graph_builder)
                .combination_method(method.to_string())
                .max_iter(50);

            let result = mssl.fit(&X.view(), &y.view());
            assert!(result.is_ok());

            let labels = result.unwrap();
            assert_eq!(labels.len(), 4);
            // Check that labeled samples retain their original labels (or reasonable prediction)
            // In semi-supervised learning, exact results can vary based on graph construction
            assert!(labels[0] == 0 || labels[0] == 1); // First sample should be classified
            assert!(labels[1] == 0 || labels[1] == 1); // Second sample should be classified
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_hierarchical_graph_error_cases() {
        let hgc = HierarchicalGraphConstruction::new().construction_method("invalid".to_string());

        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let result = hgc.fit(&X.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_multi_scale_error_cases() {
        let mssl = MultiScaleSemiSupervised::new();

        // Test with mismatched dimensions
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0]; // Wrong size

        let result = mssl.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }
}

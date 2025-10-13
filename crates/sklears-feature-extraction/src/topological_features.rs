//! Topological Feature Engineering Components
//!
//! This module contains topological data analysis and feature extraction methods
//! including persistent homology, Mapper algorithm, and simplicial complex analysis.

use crate::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::prelude::{SklearsError, Transform};

/// Persistent Homology Feature Extractor
///
/// Implements topological data analysis through persistent homology computation.
/// This extractor computes topological features that capture the "shape" of data
/// at multiple scales, providing insights into data connectivity and holes.
///
/// Persistent homology tracks the birth and death of topological features
/// (connected components, holes, voids) as the scale parameter increases.
/// This implementation focuses on 0-dimensional (connected components) and
/// 1-dimensional (holes/loops) persistence.
///
/// # Parameters
///
/// * `max_dimension` - Maximum homology dimension to compute (0, 1, or 2)
/// * `distance_metric` - Distance metric for point cloud ('euclidean', 'manhattan')
/// * `max_edge_length` - Maximum edge length in Rips complex construction
/// * `resolution` - Number of filtration steps for persistence computation
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::topological_features::PersistentHomologyExtractor;
/// use scirs2_core::ndarray::Array2;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let extractor = PersistentHomologyExtractor::new()
///     .max_dimension(1)
///     .distance_metric("euclidean")
///     .resolution(50);
///
/// let X = Array2::from_elem((50, 2), 1.0);
/// let features = extractor.extract_features(&X.view())?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PersistentHomologyExtractor {
    max_dimension: usize,
    distance_metric: String,
    max_edge_length: Option<Float>,
    resolution: usize,
    normalize_persistence: bool,
    include_birth_death_features: bool,
    include_persistence_statistics: bool,
    include_betti_curves: bool,
}

impl PersistentHomologyExtractor {
    /// Create a new persistent homology extractor
    pub fn new() -> Self {
        Self {
            max_dimension: 1,
            distance_metric: "euclidean".to_string(),
            max_edge_length: None,
            resolution: 100,
            normalize_persistence: true,
            include_birth_death_features: true,
            include_persistence_statistics: true,
            include_betti_curves: false,
        }
    }

    /// Set the maximum homology dimension
    pub fn max_dimension(mut self, dimension: usize) -> Self {
        self.max_dimension = dimension;
        self
    }

    /// Set the distance metric
    pub fn distance_metric(mut self, metric: &str) -> Self {
        self.distance_metric = metric.to_string();
        self
    }

    /// Set the maximum edge length for Rips complex
    pub fn max_edge_length(mut self, length: Float) -> Self {
        self.max_edge_length = Some(length);
        self
    }

    /// Set the resolution for filtration
    pub fn resolution(mut self, resolution: usize) -> Self {
        self.resolution = resolution;
        self
    }

    /// Enable/disable persistence normalization
    pub fn normalize_persistence(mut self, normalize: bool) -> Self {
        self.normalize_persistence = normalize;
        self
    }

    /// Include birth-death features
    pub fn include_birth_death_features(mut self, include: bool) -> Self {
        self.include_birth_death_features = include;
        self
    }

    /// Include persistence statistics
    pub fn include_persistence_statistics(mut self, include: bool) -> Self {
        self.include_persistence_statistics = include;
        self
    }

    /// Include Betti number curves
    pub fn include_betti_curves(mut self, include: bool) -> Self {
        self.include_betti_curves = include;
        self
    }

    /// Extract topological features from point cloud data
    pub fn extract_features(&self, data: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        // Compute distance matrix
        let distance_matrix = self.compute_distance_matrix(data)?;

        // Build filtered simplicial complex (Rips complex)
        let filtration = self.build_rips_filtration(&distance_matrix)?;

        // Compute persistence diagrams
        let persistence_diagrams = self.compute_persistence(&filtration)?;

        // Extract features from persistence diagrams
        let features = self.extract_topological_features(&persistence_diagrams)?;

        Ok(features)
    }

    fn compute_distance_matrix(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let n_points = data.nrows();
        let mut distance_matrix = Array2::zeros((n_points, n_points));

        for i in 0..n_points {
            for j in i + 1..n_points {
                let dist = match self.distance_metric.as_str() {
                    "euclidean" => {
                        let diff = &data.row(i) - &data.row(j);
                        diff.mapv(|x| x * x).sum().sqrt()
                    }
                    "manhattan" => {
                        let diff = &data.row(i) - &data.row(j);
                        diff.mapv(|x| x.abs()).sum()
                    }
                    _ => {
                        return Err(SklearsError::InvalidInput(
                            "Unsupported distance metric".to_string(),
                        ));
                    }
                };
                distance_matrix[[i, j]] = dist;
                distance_matrix[[j, i]] = dist;
            }
        }

        Ok(distance_matrix)
    }

    fn build_rips_filtration(
        &self,
        distance_matrix: &Array2<Float>,
    ) -> SklResult<Vec<FiltrationStep>> {
        let n_points = distance_matrix.nrows();
        let mut edges: Vec<(usize, usize, Float)> = Vec::new();

        // Collect all edges with their weights (distances)
        for i in 0..n_points {
            for j in i + 1..n_points {
                let weight = distance_matrix[[i, j]];
                if let Some(max_len) = self.max_edge_length {
                    if weight <= max_len {
                        edges.push((i, j, weight));
                    }
                } else {
                    edges.push((i, j, weight));
                }
            }
        }

        // Sort edges by weight
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Create filtration steps
        let mut filtration = Vec::new();

        // Add vertices (0-simplices) at filtration value 0
        for i in 0..n_points {
            filtration.push(FiltrationStep {
                simplex: vec![i],
                filtration_value: 0.0,
                dimension: 0,
            });
        }

        // Add edges (1-simplices) at their corresponding filtration values
        for (i, j, weight) in edges {
            filtration.push(FiltrationStep {
                simplex: vec![i, j],
                filtration_value: weight,
                dimension: 1,
            });
        }

        // If computing 2-dimensional homology, add triangles
        if self.max_dimension >= 2 {
            // This is computationally expensive, so we'll skip for this basic implementation
            // In a full implementation, we would add all 2-simplices (triangles)
        }

        // Sort by filtration value
        filtration.sort_by(|a, b| a.filtration_value.partial_cmp(&b.filtration_value).unwrap());

        Ok(filtration)
    }

    fn compute_persistence(
        &self,
        filtration: &[FiltrationStep],
    ) -> SklResult<Vec<PersistenceDiagram>> {
        // Simplified persistence computation using Union-Find for 0-dimensional persistence
        let mut diagrams = vec![PersistenceDiagram::new(0), PersistenceDiagram::new(1)];

        // Track connected components for 0-dimensional persistence
        let mut union_find =
            UnionFind::new(filtration.iter().filter(|step| step.dimension == 0).count());

        let mut vertex_birth_time = std::collections::HashMap::new();
        let mut next_vertex_id = 0;

        for step in filtration {
            match step.dimension {
                0 => {
                    // Vertex birth
                    vertex_birth_time.insert(step.simplex[0], step.filtration_value);
                    union_find.add_vertex(next_vertex_id);
                    next_vertex_id += 1;
                }
                1 => {
                    // Edge addition - may merge components or create holes
                    let v1 = step.simplex[0];
                    let v2 = step.simplex[1];

                    if let (Some(&birth1), Some(&birth2)) =
                        (vertex_birth_time.get(&v1), vertex_birth_time.get(&v2))
                    {
                        let (merged, older_birth) = union_find.union(v1, v2);

                        if merged {
                            // Component merge - death of younger component
                            let death_time = step.filtration_value;
                            let younger_birth = birth1.max(birth2);

                            if death_time > younger_birth {
                                diagrams[0].points.push(PersistencePoint {
                                    birth: younger_birth,
                                    death: death_time,
                                    dimension: 0,
                                });
                            }
                        }
                    }
                }
                _ => {} // Higher dimensions not implemented in this basic version
            }
        }

        // Add infinite components (those that never die)
        for &birth_time in vertex_birth_time.values() {
            if union_find.is_root_component(birth_time) {
                diagrams[0].points.push(PersistencePoint {
                    birth: birth_time,
                    death: Float::INFINITY,
                    dimension: 0,
                });
            }
        }

        Ok(diagrams)
    }

    fn extract_topological_features(
        &self,
        diagrams: &[PersistenceDiagram],
    ) -> SklResult<Array1<Float>> {
        let mut features = Vec::new();

        for (dim, diagram) in diagrams.iter().enumerate() {
            if dim > self.max_dimension {
                break;
            }

            // Basic persistence statistics
            if self.include_persistence_statistics {
                let persistences: Vec<Float> = diagram
                    .points
                    .iter()
                    .filter(|p| p.death.is_finite())
                    .map(|p| p.death - p.birth)
                    .collect();

                if !persistences.is_empty() {
                    // Count of features
                    features.push(persistences.len() as Float);

                    // Mean persistence
                    let mean_persistence =
                        persistences.iter().sum::<Float>() / persistences.len() as Float;
                    features.push(mean_persistence);

                    // Max persistence
                    let max_persistence = persistences.iter().cloned().fold(0.0, Float::max);
                    features.push(max_persistence);

                    // Standard deviation of persistence
                    let variance = persistences
                        .iter()
                        .map(|p| (p - mean_persistence).powi(2))
                        .sum::<Float>()
                        / persistences.len() as Float;
                    features.push(variance.sqrt());

                    // Sum of persistence
                    features.push(persistences.iter().sum());
                } else {
                    // No finite features
                    features.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
                }
            }

            // Birth-death features
            if self.include_birth_death_features {
                let birth_times: Vec<Float> = diagram
                    .points
                    .iter()
                    .filter(|p| p.birth.is_finite())
                    .map(|p| p.birth)
                    .collect();

                if !birth_times.is_empty() {
                    features.push(birth_times.iter().cloned().fold(0.0, Float::min)); // Min birth
                    features.push(birth_times.iter().cloned().fold(0.0, Float::max)); // Max birth
                    features.push(birth_times.iter().sum::<Float>() / birth_times.len() as Float);
                // Mean birth
                } else {
                    features.extend_from_slice(&[0.0, 0.0, 0.0]);
                }
            }

            // Betti number curves (simplified)
            if self.include_betti_curves {
                let betti_curve = self.compute_betti_curve(diagram);
                features.extend(betti_curve);
            }
        }

        // Ensure we return a valid feature vector
        if features.is_empty() {
            features.push(0.0); // At least one feature
        }

        Ok(Array1::from_vec(features))
    }

    fn compute_betti_curve(&self, diagram: &PersistenceDiagram) -> Vec<Float> {
        let mut curve = vec![0.0; self.resolution];

        if diagram.points.is_empty() {
            return curve;
        }

        // Find filtration range
        let min_birth = diagram
            .points
            .iter()
            .map(|p| p.birth)
            .filter(|&b| b.is_finite())
            .fold(Float::INFINITY, Float::min);

        let max_death = diagram
            .points
            .iter()
            .map(|p| p.death)
            .filter(|&d| d.is_finite())
            .fold(0.0, Float::max);

        if min_birth.is_infinite() || max_death <= min_birth {
            return curve;
        }

        let step_size = (max_death - min_birth) / self.resolution as Float;

        for i in 0..self.resolution {
            let filtration_value = min_birth + i as Float * step_size;
            let betti_number = diagram
                .points
                .iter()
                .filter(|p| p.birth <= filtration_value && filtration_value < p.death)
                .count();
            curve[i] = betti_number as Float;
        }

        curve
    }
}

impl Default for PersistentHomologyExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// A step in the filtration process
#[derive(Debug, Clone)]
struct FiltrationStep {
    simplex: Vec<usize>,
    filtration_value: Float,
    dimension: usize,
}

/// Persistence diagram for a specific dimension
#[derive(Debug, Clone)]
struct PersistenceDiagram {
    dimension: usize,
    points: Vec<PersistencePoint>,
}

impl PersistenceDiagram {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            points: Vec::new(),
        }
    }
}

/// A point in a persistence diagram
#[derive(Debug, Clone)]
struct PersistencePoint {
    birth: Float,
    death: Float,
    dimension: usize,
}

/// Union-Find data structure for connected components
#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    component_births: std::collections::HashMap<usize, Float>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            component_births: std::collections::HashMap::new(),
        }
    }

    fn add_vertex(&mut self, id: usize) {
        if id >= self.parent.len() {
            self.parent.resize(id + 1, id);
            self.rank.resize(id + 1, 0);
        }
        self.parent[id] = id;
        self.rank[id] = 0;
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> (bool, Float) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return (false, 0.0); // Already in same component
        }

        // Merge by rank
        let (smaller, larger) = if self.rank[root_x] < self.rank[root_y] {
            (root_x, root_y)
        } else if self.rank[root_x] > self.rank[root_y] {
            (root_y, root_x)
        } else {
            self.rank[root_y] += 1;
            (root_x, root_y)
        };

        self.parent[smaller] = larger;

        // Return the birth time of the older component
        let birth_x = self.component_births.get(&root_x).copied().unwrap_or(0.0);
        let birth_y = self.component_births.get(&root_y).copied().unwrap_or(0.0);

        (true, birth_x.min(birth_y))
    }

    fn is_root_component(&self, _birth_time: Float) -> bool {
        // Simplified - in practice we'd track which components are still alive
        true
    }
}

/// Mapper-based topological feature extractor
///
/// The Mapper algorithm creates a graph representation of high-dimensional data
/// by applying a filter function, covering the filter range with overlapping
/// intervals, clustering within each interval, and connecting overlapping clusters.
///
/// This implementation provides features extracted from the resulting mapper graph
/// including node statistics, edge statistics, and topological properties.
///
/// # Examples
/// ```
/// # use sklears_feature_extraction::topological_features::MapperExtractor;
/// # use scirs2_core::ndarray::Array2;
/// # use sklears_core::traits::{Transform, Fit};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect())?;
///
/// let extractor = MapperExtractor::new()
///     .n_intervals(5)
///     .overlap_ratio(0.3)
///     .filter_function("first_coordinate".to_string());
///
/// let features = extractor.transform(&data.view())?;
/// assert_eq!(features.len(), 7); // Expected number of features
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MapperExtractor {
    n_intervals: usize,
    overlap_ratio: f64,
    filter_function: String,
    cluster_method: String,
    min_cluster_size: usize,
    include_node_stats: bool,
    include_edge_stats: bool,
    include_component_stats: bool,
}

impl MapperExtractor {
    /// Create a new mapper extractor
    pub fn new() -> Self {
        Self {
            n_intervals: 10,
            overlap_ratio: 0.3,
            filter_function: "first_coordinate".to_string(),
            cluster_method: "single_linkage".to_string(),
            min_cluster_size: 2,
            include_node_stats: true,
            include_edge_stats: true,
            include_component_stats: true,
        }
    }

    /// Set the number of intervals in the cover
    pub fn n_intervals(mut self, n_intervals: usize) -> Self {
        self.n_intervals = n_intervals;
        self
    }

    /// Set the overlap ratio between intervals
    pub fn overlap_ratio(mut self, overlap_ratio: f64) -> Self {
        self.overlap_ratio = overlap_ratio;
        self
    }

    /// Set the filter function type
    pub fn filter_function(mut self, filter_function: String) -> Self {
        self.filter_function = filter_function;
        self
    }

    /// Set the clustering method
    pub fn cluster_method(mut self, cluster_method: String) -> Self {
        self.cluster_method = cluster_method;
        self
    }

    /// Set the minimum cluster size
    pub fn min_cluster_size(mut self, min_cluster_size: usize) -> Self {
        self.min_cluster_size = min_cluster_size;
        self
    }

    /// Set whether to include node statistics
    pub fn include_node_stats(mut self, include_node_stats: bool) -> Self {
        self.include_node_stats = include_node_stats;
        self
    }

    /// Set whether to include edge statistics
    pub fn include_edge_stats(mut self, include_edge_stats: bool) -> Self {
        self.include_edge_stats = include_edge_stats;
        self
    }

    /// Set whether to include component statistics
    pub fn include_component_stats(mut self, include_component_stats: bool) -> Self {
        self.include_component_stats = include_component_stats;
        self
    }

    /// Apply the filter function to the data
    fn apply_filter(&self, data: &Array2<f64>) -> Vec<f64> {
        match self.filter_function.as_str() {
            "first_coordinate" => data.column(0).to_vec(),
            "sum_coordinates" => data.outer_iter().map(|row| row.sum()).collect(),
            "l2_norm" => data.outer_iter().map(|row| row.dot(&row).sqrt()).collect(),
            "density" => {
                // Estimate local density using k-nearest neighbors
                let k = 5.min(data.nrows() - 1);
                let mut densities = Vec::with_capacity(data.nrows());

                for i in 0..data.nrows() {
                    let point = data.row(i);
                    let mut distances = Vec::new();

                    for j in 0..data.nrows() {
                        if i != j {
                            let other = data.row(j);
                            let dist = (&point - &other).mapv(|x| x * x).sum().sqrt();
                            distances.push(dist);
                        }
                    }

                    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let kth_distance = distances.get(k - 1).copied().unwrap_or(1.0);
                    densities.push(1.0 / (kth_distance + 1e-10));
                }

                densities
            }
            _ => {
                // Default to first coordinate
                data.column(0).to_vec()
            }
        }
    }

    /// Create cover intervals
    fn create_cover(&self, filter_values: &[f64]) -> Vec<(f64, f64)> {
        let min_val = filter_values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = filter_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        if min_val == max_val {
            return vec![(min_val - 1.0, max_val + 1.0)];
        }

        let range = max_val - min_val;
        let interval_length = range / self.n_intervals as f64;
        let overlap = interval_length * self.overlap_ratio;

        let mut intervals = Vec::new();
        for i in 0..self.n_intervals {
            let start = min_val + i as f64 * interval_length - overlap;
            let end = min_val + (i + 1) as f64 * interval_length + overlap;
            intervals.push((start, end));
        }

        intervals
    }

    /// Simple clustering using single linkage
    fn cluster_points(&self, data: &Array2<f64>, indices: &[usize]) -> Vec<Vec<usize>> {
        if indices.len() < self.min_cluster_size {
            return vec![indices.to_vec()];
        }

        // For simplicity, use a basic clustering approach
        // In practice, you'd use a more sophisticated clustering algorithm
        match self.cluster_method.as_str() {
            "single_linkage" => self.single_linkage_clustering(data, indices),
            _ => {
                // Default: return all points as one cluster
                vec![indices.to_vec()]
            }
        }
    }

    /// Single linkage clustering implementation
    fn single_linkage_clustering(&self, data: &Array2<f64>, indices: &[usize]) -> Vec<Vec<usize>> {
        if indices.len() <= 1 {
            return vec![indices.to_vec()];
        }

        // Compute pairwise distances
        let mut distances = Vec::new();
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let p1 = data.row(indices[i]);
                let p2 = data.row(indices[j]);
                let dist = (&p1 - &p2).mapv(|x| x * x).sum().sqrt();
                distances.push((dist, i, j));
            }
        }

        // Sort by distance
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Use median distance as threshold
        let threshold = if distances.is_empty() {
            1.0
        } else {
            distances[distances.len() / 2].0
        };

        // Build clusters using union-find
        let mut parent = (0..indices.len()).collect::<Vec<_>>();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
            let root_x = find(parent, x);
            let root_y = find(parent, y);
            if root_x != root_y {
                parent[root_x] = root_y;
            }
        }

        for (dist, i, j) in distances {
            if dist <= threshold {
                union(&mut parent, i, j);
            }
        }

        // Group indices by cluster
        let mut clusters = std::collections::HashMap::new();
        for i in 0..indices.len() {
            let root = find(&mut parent, i);
            clusters
                .entry(root)
                .or_insert_with(Vec::new)
                .push(indices[i]);
        }

        clusters.into_values().collect()
    }

    /// Build mapper graph
    fn build_mapper_graph(&self, data: &Array2<f64>) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
        let filter_values = self.apply_filter(data);
        let intervals = self.create_cover(&filter_values);

        let mut all_clusters = Vec::new();

        // For each interval, find points and cluster them
        for (start, end) in intervals {
            let mut interval_points = Vec::new();
            for (i, &filter_val) in filter_values.iter().enumerate() {
                if filter_val >= start && filter_val <= end {
                    interval_points.push(i);
                }
            }

            if !interval_points.is_empty() {
                let clusters = self.cluster_points(data, &interval_points);
                all_clusters.extend(clusters);
            }
        }

        // Remove empty clusters
        all_clusters.retain(|cluster| !cluster.is_empty());

        // Build edges between overlapping clusters
        let mut edges = Vec::new();
        for i in 0..all_clusters.len() {
            for j in (i + 1)..all_clusters.len() {
                // Check if clusters overlap
                let cluster1: std::collections::HashSet<_> = all_clusters[i].iter().collect();
                let cluster2: std::collections::HashSet<_> = all_clusters[j].iter().collect();

                if cluster1.intersection(&cluster2).next().is_some() {
                    edges.push((i, j));
                }
            }
        }

        (all_clusters, edges)
    }

    /// Extract features from the mapper graph
    fn extract_features(&self, nodes: &[Vec<usize>], edges: &[(usize, usize)]) -> Vec<f64> {
        let mut features = Vec::new();

        if self.include_node_stats {
            // Node statistics
            features.push(nodes.len() as f64); // Number of nodes

            let node_sizes: Vec<_> = nodes.iter().map(|cluster| cluster.len()).collect();
            if !node_sizes.is_empty() {
                let mean_size = node_sizes.iter().sum::<usize>() as f64 / node_sizes.len() as f64;
                let max_size = *node_sizes.iter().max().unwrap_or(&0);
                let min_size = *node_sizes.iter().min().unwrap_or(&0);

                features.push(mean_size);
                features.push(max_size as f64);
                features.push(min_size as f64);
            } else {
                features.extend(vec![0.0; 3]);
            }
        }

        if self.include_edge_stats {
            // Edge statistics
            features.push(edges.len() as f64); // Number of edges

            // Average degree
            let mut degree_sum = 0;
            for i in 0..nodes.len() {
                let degree = edges.iter().filter(|(u, v)| *u == i || *v == i).count();
                degree_sum += degree;
            }

            let avg_degree = if nodes.is_empty() {
                0.0
            } else {
                degree_sum as f64 / nodes.len() as f64
            };
            features.push(avg_degree);
        }

        if self.include_component_stats {
            // Connected components
            let n_components = self.count_connected_components(nodes.len(), edges);
            features.push(n_components as f64);
        }

        features
    }

    /// Count connected components in the graph
    fn count_connected_components(&self, n_nodes: usize, edges: &[(usize, usize)]) -> usize {
        if n_nodes == 0 {
            return 0;
        }

        let mut parent = (0..n_nodes).collect::<Vec<_>>();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
            let root_x = find(parent, x);
            let root_y = find(parent, y);
            if root_x != root_y {
                parent[root_x] = root_y;
            }
        }

        for &(u, v) in edges {
            if u < n_nodes && v < n_nodes {
                union(&mut parent, u, v);
            }
        }

        let mut components = std::collections::HashSet::new();
        for i in 0..n_nodes {
            components.insert(find(&mut parent, i));
        }

        components.len()
    }
}

impl Default for MapperExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform<ArrayView2<'_, Float>, Array1<Float>> for MapperExtractor {
    fn transform(&self, data: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot extract features from empty data".to_string(),
            ));
        }

        // Convert to Array2<f64> for internal processing
        let data_owned = data.mapv(|x| x).to_owned();

        let (nodes, edges) = self.build_mapper_graph(&data_owned);
        let features = self.extract_features(&nodes, &edges);

        // Convert back to Array1<Float>
        Ok(Array1::from_vec(
            features.into_iter().map(|x| x as Float).collect(),
        ))
    }
}

/// Simplicial Complex Feature Extractor
///
/// Extracts topological features from simplicial complexes built from point cloud data.
/// This extractor computes various properties of the simplicial complex including
/// face counts, Euler characteristic, boundary operator properties, and depth measures.
///
/// A simplicial complex is a collection of simplices (points, edges, triangles, etc.)
/// that captures the topological structure of the data. This implementation builds
/// complexes using distance thresholds and extracts informative features.
///
/// # Parameters
///
/// * `max_dimension` - Maximum dimension of simplices to consider
/// * `distance_threshold` - Distance threshold for edge creation
/// * `distance_metric` - Distance metric to use ('euclidean', 'manhattan')
/// * `include_euler_characteristic` - Whether to compute Euler characteristic
/// * `include_face_counts` - Whether to include face counts by dimension
/// * `include_boundary_properties` - Whether to compute boundary operator properties
/// * `include_depth_measures` - Whether to compute simplicial depth measures
/// * `normalize_by_vertices` - Whether to normalize counts by number of vertices
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::topological_features::SimplicialComplexExtractor;
/// use scirs2_core::ndarray::Array2;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let extractor = SimplicialComplexExtractor::new()
///     .max_dimension(2)
///     .distance_threshold(1.5)
///     .include_euler_characteristic(true);
///
/// let X = Array2::from_elem((20, 3), 1.0);
/// let features = extractor.extract_features(&X.view())?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SimplicialComplexExtractor {
    max_dimension: usize,
    distance_threshold: Float,
    distance_metric: String,
    include_euler_characteristic: bool,
    include_face_counts: bool,
    include_boundary_properties: bool,
    include_depth_measures: bool,
    normalize_by_vertices: bool,
}

impl SimplicialComplexExtractor {
    /// Create a new simplicial complex extractor
    pub fn new() -> Self {
        Self {
            max_dimension: 2,
            distance_threshold: 1.0,
            distance_metric: "euclidean".to_string(),
            include_euler_characteristic: true,
            include_face_counts: true,
            include_boundary_properties: false,
            include_depth_measures: false,
            normalize_by_vertices: false,
        }
    }

    /// Set the maximum dimension of simplices to consider
    pub fn max_dimension(mut self, dimension: usize) -> Self {
        self.max_dimension = dimension;
        self
    }

    /// Set the distance threshold for edge creation
    pub fn distance_threshold(mut self, threshold: Float) -> Self {
        self.distance_threshold = threshold;
        self
    }

    /// Set the distance metric
    pub fn distance_metric(mut self, metric: &str) -> Self {
        self.distance_metric = metric.to_string();
        self
    }

    /// Set whether to compute Euler characteristic
    pub fn include_euler_characteristic(mut self, include: bool) -> Self {
        self.include_euler_characteristic = include;
        self
    }

    /// Set whether to include face counts by dimension
    pub fn include_face_counts(mut self, include: bool) -> Self {
        self.include_face_counts = include;
        self
    }

    /// Set whether to compute boundary operator properties
    pub fn include_boundary_properties(mut self, include: bool) -> Self {
        self.include_boundary_properties = include;
        self
    }

    /// Set whether to compute simplicial depth measures
    pub fn include_depth_measures(mut self, include: bool) -> Self {
        self.include_depth_measures = include;
        self
    }

    /// Set whether to normalize counts by number of vertices
    pub fn normalize_by_vertices(mut self, normalize: bool) -> Self {
        self.normalize_by_vertices = normalize;
        self
    }

    /// Extract features from simplicial complex
    pub fn extract_features(&self, data: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        if data.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        // Build simplicial complex
        let complex = self.build_simplicial_complex(data)?;

        // Extract features
        let mut features = Vec::new();

        if self.include_face_counts {
            features.extend(self.compute_face_counts(&complex, data.nrows()));
        }

        if self.include_euler_characteristic {
            features.push(self.compute_euler_characteristic(&complex));
        }

        if self.include_boundary_properties {
            features.extend(self.compute_boundary_properties(&complex));
        }

        if self.include_depth_measures {
            features.extend(self.compute_depth_measures(&complex));
        }

        if features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected for extraction".to_string(),
            ));
        }

        // Ensure all features are finite
        for &feature in &features {
            if !feature.is_finite() {
                return Err(SklearsError::InvalidInput(
                    "Non-finite feature detected".to_string(),
                ));
            }
        }

        Ok(Array1::from_vec(features))
    }

    /// Build simplicial complex from point cloud data
    fn build_simplicial_complex(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Vec<usize>>> {
        let n_points = data.nrows();
        let mut complex = Vec::new();

        // Add vertices (0-simplices)
        for i in 0..n_points {
            complex.push(vec![i]);
        }

        // Add edges (1-simplices)
        let mut edges = Vec::new();
        for i in 0..n_points {
            for j in i + 1..n_points {
                let dist = self.compute_distance(&data.row(i), &data.row(j))?;
                if dist <= self.distance_threshold {
                    edges.push(vec![i, j]);
                }
            }
        }
        complex.extend(edges.clone());

        // Add higher-dimensional simplices based on clique expansion
        if self.max_dimension >= 2 {
            let triangles = self.find_triangles(&edges, n_points);
            complex.extend(triangles);
        }

        if self.max_dimension >= 3 {
            let tetrahedra = self.find_tetrahedra(&complex, n_points);
            complex.extend(tetrahedra);
        }

        Ok(complex)
    }

    /// Compute distance between two points
    fn compute_distance(&self, p1: &ArrayView1<Float>, p2: &ArrayView1<Float>) -> SklResult<Float> {
        match self.distance_metric.as_str() {
            "euclidean" => {
                let dist = p1
                    .iter()
                    .zip(p2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>()
                    .sqrt();
                Ok(dist)
            }
            "manhattan" => {
                let dist = p1
                    .iter()
                    .zip(p2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<Float>();
                Ok(dist)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unsupported distance metric: {}",
                self.distance_metric
            ))),
        }
    }

    /// Find triangles (2-simplices) from edge list
    fn find_triangles(&self, edges: &[Vec<usize>], n_points: usize) -> Vec<Vec<usize>> {
        let mut triangles = Vec::new();
        let mut adjacency = vec![vec![false; n_points]; n_points];

        // Build adjacency matrix
        for edge in edges {
            if edge.len() == 2 {
                adjacency[edge[0]][edge[1]] = true;
                adjacency[edge[1]][edge[0]] = true;
            }
        }

        // Find triangles
        for i in 0..n_points {
            for j in i + 1..n_points {
                if adjacency[i][j] {
                    for k in j + 1..n_points {
                        if adjacency[i][k] && adjacency[j][k] {
                            triangles.push(vec![i, j, k]);
                        }
                    }
                }
            }
        }

        triangles
    }

    /// Find tetrahedra (3-simplices) from existing complex
    fn find_tetrahedra(&self, complex: &[Vec<usize>], n_points: usize) -> Vec<Vec<usize>> {
        let mut tetrahedra = Vec::new();
        let mut adjacency = vec![vec![false; n_points]; n_points];

        // Build adjacency matrix from edges
        for simplex in complex {
            if simplex.len() == 2 {
                adjacency[simplex[0]][simplex[1]] = true;
                adjacency[simplex[1]][simplex[0]] = true;
            }
        }

        // Find tetrahedra
        for i in 0..n_points {
            for j in i + 1..n_points {
                if adjacency[i][j] {
                    for k in j + 1..n_points {
                        if adjacency[i][k] && adjacency[j][k] {
                            for l in k + 1..n_points {
                                if adjacency[i][l] && adjacency[j][l] && adjacency[k][l] {
                                    tetrahedra.push(vec![i, j, k, l]);
                                }
                            }
                        }
                    }
                }
            }
        }

        tetrahedra
    }

    /// Compute face counts by dimension
    fn compute_face_counts(&self, complex: &[Vec<usize>], n_vertices: usize) -> Vec<Float> {
        let mut counts = vec![0.0; self.max_dimension + 1];

        for simplex in complex {
            if simplex.len() <= self.max_dimension + 1 {
                let dimension = simplex.len() - 1;
                counts[dimension] += 1.0;
            }
        }

        // Normalize if requested
        if self.normalize_by_vertices && n_vertices > 0 {
            let n_vertices_float = n_vertices as Float;
            for count in &mut counts {
                *count /= n_vertices_float;
            }
        }

        counts
    }

    /// Compute Euler characteristic
    fn compute_euler_characteristic(&self, complex: &[Vec<usize>]) -> Float {
        let mut euler = 0.0;
        let mut dim_counts = vec![0; self.max_dimension + 1];

        for simplex in complex {
            if simplex.len() <= self.max_dimension + 1 {
                let dimension = simplex.len() - 1;
                dim_counts[dimension] += 1;
            }
        }

        // Euler characteristic: alternating sum of face counts
        for (i, &count) in dim_counts.iter().enumerate() {
            if i % 2 == 0 {
                euler += count as Float;
            } else {
                euler -= count as Float;
            }
        }

        euler
    }

    /// Compute boundary operator properties
    fn compute_boundary_properties(&self, complex: &[Vec<usize>]) -> Vec<Float> {
        let mut properties = Vec::new();

        // Compute boundary matrix dimensions and ranks for each dimension
        for dim in 1..=self.max_dimension {
            let faces_dim = complex.iter().filter(|s| s.len() == dim + 1).count();
            let faces_dim_minus_1 = complex.iter().filter(|s| s.len() == dim).count();

            // Boundary matrix would be faces_dim_minus_1 x faces_dim
            properties.push(faces_dim as Float);
            properties.push(faces_dim_minus_1 as Float);

            // Compute boundary matrix density (non-zero entries ratio)
            if faces_dim > 0 && faces_dim_minus_1 > 0 {
                let expected_nonzeros = faces_dim * (dim + 1); // Each d-simplex has d+1 faces
                let total_entries = faces_dim * faces_dim_minus_1;
                let density = expected_nonzeros as Float / total_entries as Float;
                properties.push(density);
            } else {
                properties.push(0.0);
            }
        }

        properties
    }

    /// Compute simplicial depth measures
    fn compute_depth_measures(&self, complex: &[Vec<usize>]) -> Vec<Float> {
        let mut measures = Vec::new();

        // Maximum simplex dimension
        let max_actual_dim = complex.iter().map(|s| s.len() - 1).max().unwrap_or(0);
        measures.push(max_actual_dim as Float);

        // Simplicial depth (how many simplices contain each vertex)
        let mut vertex_depths = std::collections::HashMap::new();
        for simplex in complex {
            for &vertex in simplex {
                *vertex_depths.entry(vertex).or_insert(0) += 1;
            }
        }

        if !vertex_depths.is_empty() {
            let depths: Vec<usize> = vertex_depths.values().cloned().collect();
            measures.push(depths.iter().sum::<usize>() as Float / depths.len() as Float); // Mean depth
            measures.push(*depths.iter().max().unwrap() as Float); // Max depth
            measures.push(*depths.iter().min().unwrap() as Float); // Min depth
        } else {
            measures.extend(vec![0.0, 0.0, 0.0]);
        }

        // Complexity measure: ratio of actual simplices to theoretical maximum
        let n_vertices = vertex_depths.len();
        if n_vertices > 0 {
            let theoretical_max = (0..=self.max_dimension)
                .map(|d| Self::binomial_coefficient(n_vertices, d + 1))
                .sum::<usize>();
            let complexity = complex.len() as Float / theoretical_max as Float;
            measures.push(complexity);
        } else {
            measures.push(0.0);
        }

        measures
    }

    /// Compute binomial coefficient (n choose k)
    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = k.min(n - k); // Take advantage of symmetry
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}

impl Default for SimplicialComplexExtractor {
    fn default() -> Self {
        Self::new()
    }
}

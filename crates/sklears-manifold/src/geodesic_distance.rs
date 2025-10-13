//! Geodesic distance computation for manifold-aware metrics
//!
//! This module provides various methods to compute geodesic distances on manifolds:
//! - **Graph-based geodesic distances**: Using k-NN graphs and shortest path algorithms
//! - **Approximation methods**: Fast approximations for large datasets
//! - **Manifold-aware metrics**: Distance metrics that respect manifold structure
//! - **Adaptive neighborhood**: Dynamic neighborhood selection for robust geodesics

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use sklears_core::types::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Geodesic distance computation methods
#[derive(Debug, Clone, Copy)]
pub enum GeodesicMethod {
    /// Floyd-Warshall algorithm for all-pairs shortest paths
    FloydWarshall,
    /// Dijkstra's algorithm for single-source shortest paths
    Dijkstra,
    /// Approximate geodesic using landmark points
    Landmark,
    /// Fast Marching Method for continuous surfaces
    FastMarching,
}

/// Neighborhood construction strategies
#[derive(Debug, Clone, Copy)]
pub enum NeighborhoodMethod {
    /// K-nearest neighbors
    KNearest(usize),
    /// Epsilon-ball neighborhoods
    EpsilonBall(Float),
    /// Adaptive neighborhoods based on local density
    Adaptive {
        min_neighbors: usize,
        max_neighbors: usize,
    },
}

/// Edge for graph construction
#[derive(Debug, Clone)]
struct Edge {
    from: usize,
    to: usize,
    weight: Float,
}

/// Priority queue element for Dijkstra's algorithm
#[derive(Debug, Clone)]
struct DijkstraNode {
    node: usize,
    distance: Float,
}

impl Eq for DijkstraNode {}

impl PartialEq for DijkstraNode {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Ord for DijkstraNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DijkstraNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute pairwise Euclidean distances
fn compute_euclidean_distances(data: &ArrayView2<Float>) -> Array2<Float> {
    let n = data.nrows();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            let dist = data
                .row(i)
                .iter()
                .zip(data.row(j).iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<Float>()
                .sqrt();
            distances[(i, j)] = dist;
            distances[(j, i)] = dist;
        }
    }
    distances
}

/// Build neighborhood graph based on the specified method
fn build_neighborhood_graph(
    data: &ArrayView2<Float>,
    method: NeighborhoodMethod,
) -> Result<Vec<Edge>, String> {
    let n = data.nrows();
    let distances = compute_euclidean_distances(data);
    let mut edges = Vec::new();

    match method {
        NeighborhoodMethod::KNearest(k) => {
            if k >= n {
                return Err("k must be less than number of samples".to_string());
            }

            for i in 0..n {
                // Find k-nearest neighbors
                let mut neighbors: Vec<(usize, Float)> = (0..n)
                    .filter(|&j| i != j)
                    .map(|j| (j, distances[(i, j)]))
                    .collect();

                neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                for (j, dist) in neighbors.into_iter().take(k) {
                    edges.push(Edge {
                        from: i,
                        to: j,
                        weight: dist,
                    });
                }
            }
        }

        NeighborhoodMethod::EpsilonBall(epsilon) => {
            for i in 0..n {
                for j in i + 1..n {
                    if distances[(i, j)] <= epsilon {
                        edges.push(Edge {
                            from: i,
                            to: j,
                            weight: distances[(i, j)],
                        });
                        edges.push(Edge {
                            from: j,
                            to: i,
                            weight: distances[(i, j)],
                        });
                    }
                }
            }
        }

        NeighborhoodMethod::Adaptive {
            min_neighbors,
            max_neighbors,
        } => {
            for i in 0..n {
                let mut neighbors: Vec<(usize, Float)> = (0..n)
                    .filter(|&j| i != j)
                    .map(|j| (j, distances[(i, j)]))
                    .collect();

                neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                // Adaptive selection based on local density
                let k_adaptive = estimate_adaptive_k(&neighbors, min_neighbors, max_neighbors);

                for (j, dist) in neighbors.into_iter().take(k_adaptive) {
                    edges.push(Edge {
                        from: i,
                        to: j,
                        weight: dist,
                    });
                }
            }
        }
    }

    Ok(edges)
}

/// Estimate adaptive k based on local density
fn estimate_adaptive_k(neighbors: &[(usize, Float)], min_k: usize, max_k: usize) -> usize {
    if neighbors.len() <= min_k {
        return neighbors.len();
    }

    // Simple heuristic: look for "elbow" in distance curve
    let mut best_k = min_k;
    let mut max_curvature = 0.0;

    for k in min_k..=(max_k.min(neighbors.len() - 1)) {
        if k < 2 {
            continue;
        }

        let d_prev = neighbors[k - 2].1;
        let d_curr = neighbors[k - 1].1;
        let d_next = neighbors[k].1;

        // Compute curvature as second derivative approximation
        let curvature = (d_next - 2.0 * d_curr + d_prev).abs();

        if curvature > max_curvature {
            max_curvature = curvature;
            best_k = k;
        }
    }

    best_k.max(min_k).min(max_k)
}

/// Build adjacency list from edges
fn build_adjacency_list(edges: &[Edge], n_nodes: usize) -> Vec<Vec<(usize, Float)>> {
    let mut adjacency = vec![Vec::new(); n_nodes];

    for edge in edges {
        adjacency[edge.from].push((edge.to, edge.weight));
    }

    adjacency
}

/// Compute geodesic distances using Floyd-Warshall algorithm
pub fn floyd_warshall_geodesic(
    data: &ArrayView2<Float>,
    neighborhood_method: NeighborhoodMethod,
) -> Result<Array2<Float>, String> {
    let n = data.nrows();
    let edges = build_neighborhood_graph(data, neighborhood_method)?;

    // Initialize distance matrix with infinity
    let mut distances = Array2::from_elem((n, n), Float::INFINITY);

    // Set diagonal to zero
    for i in 0..n {
        distances[(i, i)] = 0.0;
    }

    // Set edge weights
    for edge in &edges {
        distances[(edge.from, edge.to)] = edge.weight;
    }

    // Floyd-Warshall algorithm
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let via_k = distances[(i, k)] + distances[(k, j)];
                if via_k < distances[(i, j)] {
                    distances[(i, j)] = via_k;
                }
            }
        }
    }

    // Check for disconnected components
    for i in 0..n {
        for j in 0..n {
            if distances[(i, j)].is_infinite() {
                return Err(format!(
                    "Graph is disconnected: no path from {} to {}",
                    i, j
                ));
            }
        }
    }

    Ok(distances)
}

/// Compute geodesic distances from a single source using Dijkstra's algorithm
pub fn dijkstra_geodesic(
    data: &ArrayView2<Float>,
    source: usize,
    neighborhood_method: NeighborhoodMethod,
) -> Result<Array1<Float>, String> {
    let n = data.nrows();
    if source >= n {
        return Err("Source node index out of bounds".to_string());
    }

    let edges = build_neighborhood_graph(data, neighborhood_method)?;
    let adjacency = build_adjacency_list(&edges, n);

    let mut distances = vec![Float::INFINITY; n];
    let mut visited = vec![false; n];
    let mut heap = BinaryHeap::new();

    distances[source] = 0.0;
    heap.push(DijkstraNode {
        node: source,
        distance: 0.0,
    });

    while let Some(DijkstraNode {
        node: u,
        distance: dist,
    }) = heap.pop()
    {
        if visited[u] {
            continue;
        }
        visited[u] = true;

        for &(v, weight) in &adjacency[u] {
            let new_dist = dist + weight;
            if new_dist < distances[v] {
                distances[v] = new_dist;
                heap.push(DijkstraNode {
                    node: v,
                    distance: new_dist,
                });
            }
        }
    }

    // Check for unreachable nodes
    for (i, &dist) in distances.iter().enumerate() {
        if dist.is_infinite() {
            return Err(format!("Node {} is unreachable from source {}", i, source));
        }
    }

    Ok(Array1::from(distances))
}

/// Compute all-pairs geodesic distances using multiple Dijkstra runs
pub fn dijkstra_all_pairs_geodesic(
    data: &ArrayView2<Float>,
    neighborhood_method: NeighborhoodMethod,
) -> Result<Array2<Float>, String> {
    let n = data.nrows();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        let row_distances = dijkstra_geodesic(data, i, neighborhood_method)?;
        for j in 0..n {
            distances[(i, j)] = row_distances[j];
        }
    }

    Ok(distances)
}

/// Approximate geodesic distances using landmark points
pub fn landmark_geodesic(
    data: &ArrayView2<Float>,
    n_landmarks: usize,
    neighborhood_method: NeighborhoodMethod,
    seed: Option<u64>,
) -> Result<Array2<Float>, String> {
    let n = data.nrows();
    if n_landmarks >= n {
        return Err("Number of landmarks must be less than number of samples".to_string());
    }

    // Select landmark points (random sampling for simplicity)
    use scirs2_core::random::SeedableRng;
    use scirs2_core::SliceRandomExt;
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::seed_from_u64(thread_rng().gen::<u64>())
    };

    let mut landmarks = Vec::new();
    let mut available: Vec<usize> = (0..n).collect();

    use scirs2_core::random::seq::SliceRandom;
    available.shuffle(&mut rng);
    landmarks.extend(available.iter().take(n_landmarks));
    landmarks.sort();

    // Compute distances from all points to landmarks
    let mut landmark_distances = Array2::zeros((n, n_landmarks));

    for (i, &landmark) in landmarks.iter().enumerate() {
        let distances_from_landmark = dijkstra_geodesic(data, landmark, neighborhood_method)?;
        for j in 0..n {
            landmark_distances[(j, i)] = distances_from_landmark[j];
        }
    }

    // Approximate all-pairs distances using landmark triangulation
    let mut approximate_distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in i..n {
            if i == j {
                approximate_distances[(i, j)] = 0.0;
                continue;
            }

            // Find minimum distance through landmarks
            let mut min_dist = Float::INFINITY;
            for k in 0..n_landmarks {
                let dist_via_landmark = landmark_distances[(i, k)] + landmark_distances[(j, k)];
                min_dist = min_dist.min(dist_via_landmark);
            }

            approximate_distances[(i, j)] = min_dist;
            approximate_distances[(j, i)] = min_dist;
        }
    }

    Ok(approximate_distances)
}

/// Fast Marching Method for continuous geodesic computation (simplified version)
pub fn fast_marching_geodesic(
    data: &ArrayView2<Float>,
    source: usize,
    grid_resolution: usize,
) -> Result<Array1<Float>, String> {
    let n = data.nrows();
    if source >= n {
        return Err("Source index out of bounds".to_string());
    }

    // This is a simplified version - a full FMM implementation would require
    // proper grid discretization and level set methods

    // For now, fall back to Dijkstra with dense neighborhoods
    let neighborhood_method = NeighborhoodMethod::KNearest((n as f64).sqrt().ceil() as usize);

    dijkstra_geodesic(data, source, neighborhood_method)
}

/// Geodesic distance computer with configurable parameters
pub struct GeodesicDistanceComputer {
    method: GeodesicMethod,
    neighborhood_method: NeighborhoodMethod,
    n_landmarks: Option<usize>,
    seed: Option<u64>,
}

impl GeodesicDistanceComputer {
    /// Create a new geodesic distance computer
    pub fn new(method: GeodesicMethod) -> Self {
        Self {
            method,
            neighborhood_method: NeighborhoodMethod::KNearest(8),
            n_landmarks: None,
            seed: Some(42),
        }
    }

    /// Set the neighborhood construction method
    pub fn neighborhood_method(mut self, method: NeighborhoodMethod) -> Self {
        self.neighborhood_method = method;
        self
    }

    /// Set number of landmarks for landmark-based approximation
    pub fn n_landmarks(mut self, n_landmarks: usize) -> Self {
        self.n_landmarks = Some(n_landmarks);
        self
    }

    /// Set random seed for reproducibility
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Compute all-pairs geodesic distances
    pub fn compute_all_pairs(&self, data: &ArrayView2<Float>) -> Result<Array2<Float>, String> {
        match self.method {
            GeodesicMethod::FloydWarshall => {
                floyd_warshall_geodesic(data, self.neighborhood_method)
            }
            GeodesicMethod::Dijkstra => dijkstra_all_pairs_geodesic(data, self.neighborhood_method),
            GeodesicMethod::Landmark => {
                let n_landmarks = self
                    .n_landmarks
                    .unwrap_or_else(|| (data.nrows() as f64).sqrt().ceil() as usize);
                landmark_geodesic(data, n_landmarks, self.neighborhood_method, self.seed)
            }
            GeodesicMethod::FastMarching => {
                // For all-pairs, run FMM from each point
                let n = data.nrows();
                let mut distances = Array2::zeros((n, n));

                for i in 0..n {
                    let row_distances = fast_marching_geodesic(data, i, 100)?;
                    for j in 0..n {
                        distances[(i, j)] = row_distances[j];
                    }
                }

                Ok(distances)
            }
        }
    }

    /// Compute geodesic distances from a single source
    pub fn compute_single_source(
        &self,
        data: &ArrayView2<Float>,
        source: usize,
    ) -> Result<Array1<Float>, String> {
        match self.method {
            GeodesicMethod::FloydWarshall => {
                let all_pairs = floyd_warshall_geodesic(data, self.neighborhood_method)?;
                Ok(all_pairs.row(source).to_owned())
            }
            GeodesicMethod::Dijkstra => dijkstra_geodesic(data, source, self.neighborhood_method),
            GeodesicMethod::Landmark => {
                let all_pairs = self.compute_all_pairs(data)?;
                Ok(all_pairs.row(source).to_owned())
            }
            GeodesicMethod::FastMarching => fast_marching_geodesic(data, source, 100),
        }
    }
}

/// Utilities for geodesic distance analysis
pub mod geodesic_utils {
    use super::*;

    /// Check if geodesic distances satisfy metric properties
    pub fn validate_metric_properties(
        distances: &Array2<Float>,
        tolerance: Float,
    ) -> Result<(), String> {
        let n = distances.nrows();

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                if (distances[(i, j)] - distances[(j, i)]).abs() > tolerance {
                    return Err(format!(
                        "Asymmetric distance: d({},{}) = {}, d({},{}) = {}",
                        i,
                        j,
                        distances[(i, j)],
                        j,
                        i,
                        distances[(j, i)]
                    ));
                }
            }
        }

        // Check diagonal is zero
        for i in 0..n {
            if distances[(i, i)].abs() > tolerance {
                return Err(format!(
                    "Non-zero diagonal: d({},{}) = {}",
                    i,
                    i,
                    distances[(i, i)]
                ));
            }
        }

        // Check triangle inequality
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let direct = distances[(i, j)];
                    let via_k = distances[(i, k)] + distances[(k, j)];
                    if direct > via_k + tolerance {
                        return Err(format!("Triangle inequality violated: d({},{}) = {} > d({},{}) + d({},{}) = {}",
                                         i, j, direct, i, k, k, j, via_k));
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute geodesic embedding quality vs Euclidean
    pub fn compare_geodesic_euclidean(
        data: &ArrayView2<Float>,
        geodesic_distances: &Array2<Float>,
    ) -> Float {
        let euclidean_distances = compute_euclidean_distances(data);
        let n = data.nrows();

        let mut correlation_num = 0.0;
        let mut geo_sum = 0.0;
        let mut euc_sum = 0.0;
        let mut geo_sq_sum = 0.0;
        let mut euc_sq_sum = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in i + 1..n {
                let geo_dist = geodesic_distances[(i, j)];
                let euc_dist = euclidean_distances[(i, j)];

                correlation_num += geo_dist * euc_dist;
                geo_sum += geo_dist;
                euc_sum += euc_dist;
                geo_sq_sum += geo_dist * geo_dist;
                euc_sq_sum += euc_dist * euc_dist;
                count += 1;
            }
        }

        let count_f = count as Float;
        let geo_mean = geo_sum / count_f;
        let euc_mean = euc_sum / count_f;

        let numerator = correlation_num - count_f * geo_mean * euc_mean;
        let denominator = ((geo_sq_sum - count_f * geo_mean * geo_mean)
            * (euc_sq_sum - count_f * euc_mean * euc_mean))
            .sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Estimate intrinsic dimensionality using geodesic distances
    pub fn estimate_intrinsic_dimension(
        geodesic_distances: &Array2<Float>,
        k_neighbors: usize,
    ) -> Float {
        let n = geodesic_distances.nrows();
        let mut dimension_estimates = Vec::new();

        for i in 0..n {
            // Get k-nearest neighbors by geodesic distance
            let mut neighbors: Vec<(usize, Float)> = (0..n)
                .filter(|&j| i != j)
                .map(|j| (j, geodesic_distances[(i, j)]))
                .collect();

            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            neighbors.truncate(k_neighbors);

            // Estimate local dimension using MLE
            if neighbors.len() >= 2 {
                let max_dist = neighbors.last().unwrap().1;
                let mut log_ratio_sum = 0.0;

                for (_, dist) in &neighbors {
                    if *dist > 0.0 {
                        log_ratio_sum += (max_dist / dist).ln();
                    }
                }

                let dimension = (neighbors.len() - 1) as Float / log_ratio_sum;
                if dimension.is_finite() && dimension > 0.0 {
                    dimension_estimates.push(dimension);
                }
            }
        }

        if dimension_estimates.is_empty() {
            return Float::NAN;
        }

        // Return median estimate for robustness
        dimension_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = dimension_estimates.len() / 2;
        dimension_estimates[mid]
    }
}

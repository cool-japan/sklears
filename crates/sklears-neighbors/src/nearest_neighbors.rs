//! Unsupervised Nearest Neighbors learner

use crate::{Distance, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Transform};
use sklears_core::types::{Features, Float};

/// Unsupervised learner for implementing neighbor searches
///
/// This estimator acts as a uniform interface to three different nearest
/// neighbors algorithms: BallTree, KDTree, and a brute-force algorithm
/// based on routines in sklearn.metrics.pairwise.
#[derive(Debug, Clone)]
pub struct NearestNeighbors<State = sklears_core::traits::Untrained> {
    /// Number of neighbors to use by default for kneighbors queries
    pub n_neighbors: usize,
    /// Radius of neighborhood to use by default for radius_neighbors queries
    pub radius: Float,
    /// Distance metric to use for the tree
    pub metric: Distance,
    /// Algorithm to use for neighbor computation
    pub algorithm: crate::knn::Algorithm,
    /// Training data (only available after fitting)
    pub(crate) x_train: Option<Array2<Float>>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl NearestNeighbors {
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            radius: 1.0,
            metric: Distance::default(),
            algorithm: crate::knn::Algorithm::Brute,
            x_train: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the default radius for radius_neighbors queries
    pub fn with_radius(mut self, radius: Float) -> Self {
        self.radius = radius;
        self
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: Distance) -> Self {
        self.metric = metric;
        self
    }

    /// Set the algorithm
    pub fn with_algorithm(mut self, algorithm: crate::knn::Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

impl Estimator for NearestNeighbors {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, ()> for NearestNeighbors {
    type Fitted = NearestNeighbors<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, _y: &()) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if self.n_neighbors == 0 {
            return Err(NeighborsError::InvalidNeighbors(self.n_neighbors).into());
        }

        if self.radius <= 0.0 {
            return Err(NeighborsError::InvalidRadius(self.radius).into());
        }

        Ok(NearestNeighbors {
            n_neighbors: self.n_neighbors,
            radius: self.radius,
            metric: self.metric,
            algorithm: self.algorithm,
            x_train: Some(x.clone()),
            _state: std::marker::PhantomData,
        })
    }
}

impl Transform<Features> for NearestNeighbors<sklears_core::traits::Trained> {
    fn transform(&self, x: &Features) -> Result<Array2<Float>> {
        // Transform to k-neighbors graph
        self.kneighbors_graph(x, None, None, true)
    }
}

impl NearestNeighbors<sklears_core::traits::Trained> {
    /// Find the k-neighbors of a point
    ///
    /// Returns indices and distances of neighbors
    pub fn kneighbors(
        &self,
        x: &Features,
        n_neighbors: Option<usize>,
        return_distance: bool,
    ) -> NeighborsResult<(Option<Array2<Float>>, Array2<usize>)> {
        let x_train = self.x_train.as_ref().unwrap();
        let k = n_neighbors.unwrap_or(self.n_neighbors);

        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            });
        }

        let n_queries = x.nrows();
        let mut distances_opt = if return_distance {
            Some(Array2::zeros((n_queries, k)))
        } else {
            None
        };
        let mut indices = Array2::zeros((n_queries, k));

        for (query_idx, sample) in x.axis_iter(Axis(0)).enumerate() {
            let distances = self.metric.to_matrix(&sample, &x_train.view());

            // Create pairs of (distance, index) and sort by distance
            let mut neighbors: Vec<(Float, usize)> = distances
                .iter()
                .enumerate()
                .map(|(idx, &dist)| (dist, idx))
                .collect();

            neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Take k nearest neighbors
            let actual_k = k.min(neighbors.len());
            for i in 0..actual_k {
                let (dist, idx) = neighbors[i];
                if let Some(ref mut dists) = distances_opt {
                    dists[[query_idx, i]] = dist;
                }
                indices[[query_idx, i]] = idx;
            }
        }

        Ok((distances_opt, indices))
    }

    /// Find neighbors within a given radius
    ///
    /// Returns indices and distances of neighbors within radius
    #[allow(clippy::type_complexity)]
    pub fn radius_neighbors(
        &self,
        x: &Features,
        radius: Option<Float>,
        return_distance: bool,
    ) -> NeighborsResult<(Option<Vec<Array1<Float>>>, Vec<Array1<usize>>)> {
        let x_train = self.x_train.as_ref().unwrap();
        let r = radius.unwrap_or(self.radius);

        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            });
        }

        let n_queries = x.nrows();
        let mut distances_opt = if return_distance {
            Some(Vec::with_capacity(n_queries))
        } else {
            None
        };
        let mut indices = Vec::with_capacity(n_queries);

        for sample in x.axis_iter(Axis(0)) {
            let distances = self.metric.to_matrix(&sample, &x_train.view());

            let neighbors: Vec<(Float, usize)> = distances
                .iter()
                .enumerate()
                .filter_map(
                    |(idx, &dist)| {
                        if dist <= r {
                            Some((dist, idx))
                        } else {
                            None
                        }
                    },
                )
                .collect();

            if let Some(ref mut dists) = distances_opt {
                let query_distances: Vec<Float> = neighbors.iter().map(|(d, _)| *d).collect();
                dists.push(Array1::from_vec(query_distances));
            }

            let query_indices: Vec<usize> = neighbors.iter().map(|(_, idx)| *idx).collect();
            indices.push(Array1::from_vec(query_indices));
        }

        Ok((distances_opt, indices))
    }

    /// Compute the (weighted) graph of k-neighbors
    pub fn kneighbors_graph(
        &self,
        x: &Features,
        n_neighbors: Option<usize>,
        mode: Option<&str>,
        include_self: bool,
    ) -> Result<Array2<Float>> {
        let k = n_neighbors.unwrap_or(self.n_neighbors);
        let mode = mode.unwrap_or("connectivity");

        let (distances_opt, indices) = self.kneighbors(x, Some(k), mode == "distance")?;

        let n_queries = x.nrows();
        let n_samples = self.x_train.as_ref().unwrap().nrows();
        let mut graph = Array2::zeros((n_queries, n_samples));

        for i in 0..n_queries {
            for j in 0..k {
                let neighbor_idx = indices[[i, j]];

                // Skip self-connections if not included
                if !include_self && neighbor_idx == i {
                    continue;
                }

                let value = match mode {
                    "distance" => {
                        if let Some(ref distances) = distances_opt {
                            distances[[i, j]]
                        } else {
                            1.0
                        }
                    }
                    "connectivity" => 1.0,
                    _ => 1.0,
                };

                graph[[i, neighbor_idx]] = value;
            }
        }

        Ok(graph)
    }

    /// Compute the (weighted) graph of neighbors within a given radius
    pub fn radius_neighbors_graph(
        &self,
        x: &Features,
        radius: Option<Float>,
        mode: Option<&str>,
        include_self: bool,
    ) -> Result<Array2<Float>> {
        let r = radius.unwrap_or(self.radius);
        let mode = mode.unwrap_or("connectivity");

        let (distances_opt, indices) = self.radius_neighbors(x, Some(r), mode == "distance")?;

        let n_queries = x.nrows();
        let n_samples = self.x_train.as_ref().unwrap().nrows();
        let mut graph = Array2::zeros((n_queries, n_samples));

        for i in 0..n_queries {
            let query_indices = &indices[i];
            let query_distances = distances_opt.as_ref().map(|d| &d[i]);

            for (j, &neighbor_idx) in query_indices.iter().enumerate() {
                // Skip self-connections if not included
                if !include_self && neighbor_idx == i {
                    continue;
                }

                let value = match mode {
                    "distance" => {
                        if let Some(distances) = query_distances {
                            distances[j]
                        } else {
                            1.0
                        }
                    }
                    "connectivity" => 1.0,
                    _ => 1.0,
                };

                graph[[i, neighbor_idx]] = value;
            }
        }

        Ok(graph)
    }
}

/// Compute a graph of k-neighbors
pub fn kneighbors_graph(
    x: &Array2<Float>,
    n_neighbors: usize,
    metric: Distance,
    mode: Option<&str>,
    include_self: bool,
) -> Array2<Float> {
    let nn = NearestNeighbors::new(n_neighbors).with_metric(metric);

    let fitted = nn.fit(x, &()).unwrap();
    fitted
        .kneighbors_graph(x, Some(n_neighbors), mode, include_self)
        .unwrap()
}

/// Compute a graph of neighbors within a given radius
pub fn radius_neighbors_graph(
    x: &Array2<Float>,
    radius: Float,
    metric: Distance,
    mode: Option<&str>,
    include_self: bool,
) -> Array2<Float> {
    let nn = NearestNeighbors::new(1) // n_neighbors doesn't matter for radius
        .with_radius(radius)
        .with_metric(metric);

    let fitted = nn.fit(x, &()).unwrap();
    fitted
        .radius_neighbors_graph(x, Some(radius), mode, include_self)
        .unwrap()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_nearest_neighbors_basic() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

        let nn = NearestNeighbors::new(2);
        let fitted = nn.fit(&x, &()).unwrap();

        // Test kneighbors
        let (distances, indices) = fitted.kneighbors(&x, Some(2), true).unwrap();

        assert!(distances.is_some());
        let dists = distances.unwrap();
        assert_eq!(dists.shape(), &[4, 2]);
        assert_eq!(indices.shape(), &[4, 2]);

        // First row: closest should be itself (distance 0) and next point
        assert_eq!(indices[[0, 0]], 0); // Self
        assert_eq!(indices[[0, 1]], 1); // Next closest
        assert!((dists[[0, 0]] - 0.0).abs() < 1e-10); // Distance to self
    }

    #[test]
    fn test_radius_neighbors() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 5.0]).unwrap();

        let nn = NearestNeighbors::new(1).with_radius(1.5);
        let fitted = nn.fit(&x, &()).unwrap();

        let (distances, indices) = fitted.radius_neighbors(&x, Some(1.5), true).unwrap();

        assert!(distances.is_some());
        let _dists = distances.unwrap();

        // First point (1.0) should have neighbors [0, 1] (itself and point at 2.0)
        assert_eq!(indices[0].len(), 2);
        assert_eq!(indices[0][0], 0); // Self
        assert_eq!(indices[0][1], 1); // Point at 2.0

        // Third point (5.0) should only have itself as neighbor
        assert_eq!(indices[2].len(), 1);
        assert_eq!(indices[2][0], 2); // Self
    }

    #[test]
    fn test_kneighbors_graph() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();

        let graph = kneighbors_graph(&x, 2, Distance::Euclidean, Some("connectivity"), true);

        assert_eq!(graph.shape(), &[3, 3]);

        // Each row should have exactly 2 connections (k=2)
        for row in 0..3 {
            let connections: usize = graph
                .row(row)
                .iter()
                .map(|&x| if x > 0.0 { 1 } else { 0 })
                .sum();
            assert_eq!(connections, 2);
        }
    }

    #[test]
    fn test_radius_neighbors_graph() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 5.0]).unwrap();

        let graph =
            radius_neighbors_graph(&x, 1.5, Distance::Euclidean, Some("connectivity"), true);

        assert_eq!(graph.shape(), &[3, 3]);

        // First row should connect to points 0 and 1
        assert!(graph[[0, 0]] > 0.0); // Self
        assert!(graph[[0, 1]] > 0.0); // Close neighbor
        assert_eq!(graph[[0, 2]], 0.0); // Far neighbor

        // Third row should only connect to itself
        assert_eq!(graph[[2, 0]], 0.0); // Far
        assert_eq!(graph[[2, 1]], 0.0); // Far
        assert!(graph[[2, 2]] > 0.0); // Self
    }
}

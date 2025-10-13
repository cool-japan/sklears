//! Neighbor-based transformers for building graphs and embeddings

use crate::{Distance, NeighborsError};
use scirs2_core::ndarray::{Array2, Axis};
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Transform};
use sklears_core::types::{Features, Float};

/// Transformer for building k-neighbors graphs
#[derive(Debug, Clone)]
pub struct KNeighborsTransformer<State = sklears_core::traits::Untrained> {
    /// Number of neighbors for each sample
    pub n_neighbors: usize,
    /// Distance metric to use
    pub metric: Distance,
    /// Type of returned matrix: 'connectivity' or 'distance'
    pub mode: String,
    /// Algorithm to use for neighbors search
    pub algorithm: crate::knn::Algorithm,
    /// Training data (only available after fitting)
    pub(crate) x_train: Option<Array2<Float>>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

/// Transformer for building radius-based neighbors graphs
#[derive(Debug, Clone)]
pub struct RadiusNeighborsTransformer<State = sklears_core::traits::Untrained> {
    pub radius: Float,
    pub metric: Distance,
    pub mode: String,
    pub algorithm: crate::knn::Algorithm,
    pub(crate) x_train: Option<Array2<Float>>,
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl KNeighborsTransformer {
    /// Create a new KNeighborsTransformer
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            metric: Distance::default(),
            mode: "connectivity".to_string(),
            algorithm: crate::knn::Algorithm::Brute,
            x_train: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: Distance) -> Self {
        self.metric = metric;
        self
    }

    /// Set the mode ('connectivity' or 'distance')
    pub fn with_mode(mut self, mode: &str) -> Self {
        self.mode = mode.to_string();
        self
    }

    /// Set the algorithm
    pub fn with_algorithm(mut self, algorithm: crate::knn::Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

impl RadiusNeighborsTransformer {
    /// Create a new RadiusNeighborsTransformer
    pub fn new(radius: Float) -> Self {
        Self {
            radius,
            metric: Distance::default(),
            mode: "connectivity".to_string(),
            algorithm: crate::knn::Algorithm::Brute,
            x_train: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: Distance) -> Self {
        self.metric = metric;
        self
    }

    /// Set the mode ('connectivity' or 'distance')
    pub fn with_mode(mut self, mode: &str) -> Self {
        self.mode = mode.to_string();
        self
    }

    /// Set the algorithm
    pub fn with_algorithm(mut self, algorithm: crate::knn::Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

impl Estimator for KNeighborsTransformer {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for RadiusNeighborsTransformer {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, ()> for KNeighborsTransformer {
    type Fitted = KNeighborsTransformer<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, _y: &()) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if self.n_neighbors == 0 {
            return Err(NeighborsError::InvalidNeighbors(self.n_neighbors).into());
        }

        Ok(KNeighborsTransformer {
            n_neighbors: self.n_neighbors,
            metric: self.metric,
            mode: self.mode,
            algorithm: self.algorithm,
            x_train: Some(x.clone()),
            _state: std::marker::PhantomData,
        })
    }
}

impl Fit<Features, ()> for RadiusNeighborsTransformer {
    type Fitted = RadiusNeighborsTransformer<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, _y: &()) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if self.radius <= 0.0 {
            return Err(NeighborsError::InvalidRadius(self.radius).into());
        }

        Ok(RadiusNeighborsTransformer {
            radius: self.radius,
            metric: self.metric,
            mode: self.mode,
            algorithm: self.algorithm,
            x_train: Some(x.clone()),
            _state: std::marker::PhantomData,
        })
    }
}

impl Transform<Features> for KNeighborsTransformer<sklears_core::traits::Trained> {
    fn transform(&self, x: &Features) -> Result<Array2<Float>> {
        let x_train = self.x_train.as_ref().unwrap();

        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            }
            .into());
        }

        let n_queries = x.nrows();
        let n_samples = x_train.nrows();
        let mut graph = Array2::zeros((n_queries, n_samples));

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
            let k = self.n_neighbors.min(neighbors.len());
            for (dist, neighbor_idx) in neighbors.iter().take(k) {
                let (dist, neighbor_idx) = (*dist, *neighbor_idx);

                let value = match self.mode.as_str() {
                    "distance" => dist,
                    "connectivity" => 1.0,
                    _ => 1.0,
                };

                graph[[query_idx, neighbor_idx]] = value;
            }
        }

        Ok(graph)
    }
}

impl Transform<Features> for RadiusNeighborsTransformer<sklears_core::traits::Trained> {
    fn transform(&self, x: &Features) -> Result<Array2<Float>> {
        let x_train = self.x_train.as_ref().unwrap();

        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            }
            .into());
        }

        let n_queries = x.nrows();
        let n_samples = x_train.nrows();
        let mut graph = Array2::zeros((n_queries, n_samples));

        for (query_idx, sample) in x.axis_iter(Axis(0)).enumerate() {
            let distances = self.metric.to_matrix(&sample, &x_train.view());

            for (neighbor_idx, &dist) in distances.iter().enumerate() {
                if dist <= self.radius {
                    let value = match self.mode.as_str() {
                        "distance" => dist,
                        "connectivity" => 1.0,
                        _ => 1.0,
                    };

                    graph[[query_idx, neighbor_idx]] = value;
                }
            }
        }

        Ok(graph)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_kneighbors_transformer() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

        let transformer = KNeighborsTransformer::new(2);
        let fitted = transformer.fit(&x, &()).unwrap();

        let graph = fitted.transform(&x).unwrap();
        assert_eq!(graph.shape(), &[4, 4]);

        // Each row should have exactly 2 connections (k=2)
        for row in 0..4 {
            let connections: usize = graph
                .row(row)
                .iter()
                .map(|&x| if x > 0.0 { 1 } else { 0 })
                .sum();
            assert_eq!(connections, 2);
        }

        // First point should be connected to itself and the second point
        assert!(graph[[0, 0]] > 0.0); // Self
        assert!(graph[[0, 1]] > 0.0); // Closest neighbor
    }

    #[test]
    fn test_kneighbors_transformer_distance_mode() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();

        let transformer = KNeighborsTransformer::new(2).with_mode("distance");
        let fitted = transformer.fit(&x, &()).unwrap();

        let graph = fitted.transform(&x).unwrap();
        assert_eq!(graph.shape(), &[3, 3]);

        // In distance mode, values should be actual distances, not 1.0
        assert!((graph[[0, 0]] - 0.0).abs() < 1e-10); // Distance to self
        assert!((graph[[0, 1]] - 1.0).abs() < 1e-10); // Distance to next point
    }

    #[test]
    fn test_radius_neighbors_transformer() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 5.0]).unwrap();

        let transformer = RadiusNeighborsTransformer::new(1.5);
        let fitted = transformer.fit(&x, &()).unwrap();

        let graph = fitted.transform(&x).unwrap();
        assert_eq!(graph.shape(), &[3, 3]);

        // First point should connect to points 0 and 1 (within radius 1.5)
        assert!(graph[[0, 0]] > 0.0); // Self
        assert!(graph[[0, 1]] > 0.0); // Close neighbor
        assert_eq!(graph[[0, 2]], 0.0); // Far neighbor

        // Third point should only connect to itself
        assert_eq!(graph[[2, 0]], 0.0); // Far
        assert_eq!(graph[[2, 1]], 0.0); // Far
        assert!(graph[[2, 2]] > 0.0); // Self
    }

    #[test]
    fn test_radius_neighbors_transformer_distance_mode() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();

        let transformer = RadiusNeighborsTransformer::new(2.0).with_mode("distance");
        let fitted = transformer.fit(&x, &()).unwrap();

        let graph = fitted.transform(&x).unwrap();
        assert_eq!(graph.shape(), &[2, 2]);

        // In distance mode, values should be actual distances
        assert!((graph[[0, 0]] - 0.0).abs() < 1e-10); // Distance to self
        assert!((graph[[0, 1]] - 1.0).abs() < 1e-10); // Distance to other point
    }

    #[test]
    fn test_transformers_empty_input() {
        let x = Array2::zeros((0, 2));

        let knn_transformer = KNeighborsTransformer::new(2);
        let result = knn_transformer.fit(&x, &());
        assert!(result.is_err());

        let radius_transformer = RadiusNeighborsTransformer::new(1.0);
        let result = radius_transformer.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_transformers_invalid_parameters() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();

        // Invalid n_neighbors
        let knn_transformer = KNeighborsTransformer::new(0);
        let result = knn_transformer.fit(&x, &());
        assert!(result.is_err());

        // Invalid radius
        let radius_transformer = RadiusNeighborsTransformer::new(0.0);
        let result = radius_transformer.fit(&x, &());
        assert!(result.is_err());
    }
}

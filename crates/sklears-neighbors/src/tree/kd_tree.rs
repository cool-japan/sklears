//! KD-Tree implementation for efficient nearest neighbor search

use crate::{Distance, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::types::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A node in the KD-tree
#[derive(Debug, Clone)]
pub struct KdTreeNode {
    /// The point stored at this node
    pub point: Array1<Float>,
    /// Index of the original data point
    pub index: usize,
    /// The dimension used to split at this level
    pub split_dimension: usize,
    /// Left child (points with smaller values in split dimension)
    pub left: Option<Box<KdTreeNode>>,
    /// Right child (points with larger or equal values in split dimension)
    pub right: Option<Box<KdTreeNode>>,
}

/// KD-Tree for efficient nearest neighbor search in low-dimensional spaces
#[derive(Debug, Clone)]
pub struct KdTree {
    /// Root node of the tree
    root: Option<Box<KdTreeNode>>,
    /// Number of dimensions
    n_dimensions: usize,
    /// Number of points in the tree
    n_points: usize,
    /// Distance metric (currently only Euclidean supported for trees)
    metric: Distance,
}

/// Helper struct for k-nearest neighbors priority queue
#[derive(Debug, Clone, PartialEq)]
struct NeighborCandidate {
    distance: Float,
    index: usize,
}

impl Eq for NeighborCandidate {}

impl Ord for NeighborCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for max-heap (we want smallest distances)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for NeighborCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl KdTree {
    /// Create a new KD-tree from the given data points
    pub fn new(data: &Array2<Float>, metric: Distance) -> NeighborsResult<Self> {
        if data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        // For KD-trees, we currently only support Euclidean distance
        if !matches!(metric, Distance::Euclidean) {
            return Err(NeighborsError::InvalidInput(
                "KD-tree currently only supports Euclidean distance".to_string(),
            ));
        }

        let n_points = data.nrows();
        let n_dimensions = data.ncols();

        // Create indexed data points
        let mut indexed_points: Vec<(Array1<Float>, usize)> = data
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, row)| (row.to_owned(), i))
            .collect();

        let root = Self::build_tree(&mut indexed_points, 0);

        Ok(Self {
            root,
            n_dimensions,
            n_points,
            metric,
        })
    }

    /// Recursively build the KD-tree
    fn build_tree(points: &mut [(Array1<Float>, usize)], depth: usize) -> Option<Box<KdTreeNode>> {
        if points.is_empty() {
            return None;
        }

        if points.len() == 1 {
            let (point, index) = points[0].clone();
            let n_dimensions = point.len();
            return Some(Box::new(KdTreeNode {
                point,
                index,
                split_dimension: depth % n_dimensions,
                left: None,
                right: None,
            }));
        }

        let n_dimensions = points[0].0.len();
        let split_dimension = depth % n_dimensions;

        // Sort points by the split dimension
        points.sort_by(|a, b| {
            a.0[split_dimension]
                .partial_cmp(&b.0[split_dimension])
                .unwrap_or(Ordering::Equal)
        });

        let median_idx = points.len() / 2;
        let (median_point, median_index) = points[median_idx].clone();

        // Split points into left and right
        let (left_points, rest) = points.split_at_mut(median_idx);
        let right_points = &mut rest[1..]; // Skip the median point

        let left_child = Self::build_tree(left_points, depth + 1);
        let right_child = Self::build_tree(right_points, depth + 1);

        Some(Box::new(KdTreeNode {
            point: median_point,
            index: median_index,
            split_dimension,
            left: left_child,
            right: right_child,
        }))
    }

    /// Find k-nearest neighbors for a query point
    pub fn kneighbors(
        &self,
        query: &ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<Float>, Vec<usize>)> {
        if k == 0 {
            return Err(NeighborsError::InvalidNeighbors(k));
        }

        if query.len() != self.n_dimensions {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![self.n_dimensions],
                actual: vec![query.len()],
            });
        }

        let mut candidates = BinaryHeap::new();

        if let Some(ref root) = self.root {
            self.search_knn(root, query, k, &mut candidates);
        }

        // Convert to sorted vectors (closest first)
        let mut neighbors: Vec<_> = candidates.into_sorted_vec();
        neighbors.reverse(); // BinaryHeap gives us largest first, we want smallest

        let distances: Vec<Float> = neighbors.iter().map(|c| c.distance).collect();
        let indices: Vec<usize> = neighbors.iter().map(|c| c.index).collect();

        Ok((distances, indices))
    }

    /// Recursive k-nearest neighbors search
    fn search_knn(
        &self,
        node: &KdTreeNode,
        query: &ArrayView1<Float>,
        k: usize,
        candidates: &mut BinaryHeap<NeighborCandidate>,
    ) {
        // Calculate distance to current node
        let distance = self.calculate_distance(&node.point.view(), query);

        // Add current node as candidate
        if candidates.len() < k {
            candidates.push(NeighborCandidate {
                distance,
                index: node.index,
            });
        } else if distance < candidates.peek().unwrap().distance {
            candidates.pop();
            candidates.push(NeighborCandidate {
                distance,
                index: node.index,
            });
        }

        // Determine which side to search first
        let split_coord = query[node.split_dimension];
        let node_coord = node.point[node.split_dimension];
        let diff = split_coord - node_coord;

        // Search the closer side first
        let (primary, secondary) = if diff <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(child) = primary {
            self.search_knn(child, query, k, candidates);
        }

        // Check if we need to search the other side
        let worst_distance = if candidates.len() < k {
            Float::INFINITY
        } else {
            candidates.peek().unwrap().distance
        };

        if diff.abs() < worst_distance {
            if let Some(child) = secondary {
                self.search_knn(child, query, k, candidates);
            }
        }
    }

    /// Find all neighbors within a given radius
    pub fn radius_neighbors(
        &self,
        query: &ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<(Vec<Float>, Vec<usize>)> {
        if radius <= 0.0 {
            return Err(NeighborsError::InvalidRadius(radius));
        }

        if query.len() != self.n_dimensions {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![self.n_dimensions],
                actual: vec![query.len()],
            });
        }

        let mut neighbors = Vec::new();

        if let Some(ref root) = self.root {
            self.search_radius(root, query, radius, &mut neighbors);
        }

        // Sort by distance
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let distances: Vec<Float> = neighbors.iter().map(|(d, _)| *d).collect();
        let indices: Vec<usize> = neighbors.iter().map(|(_, i)| *i).collect();

        Ok((distances, indices))
    }

    /// Recursive radius neighbors search
    fn search_radius(
        &self,
        node: &KdTreeNode,
        query: &ArrayView1<Float>,
        radius: Float,
        neighbors: &mut Vec<(Float, usize)>,
    ) {
        let distance = self.calculate_distance(&node.point.view(), query);

        if distance <= radius {
            neighbors.push((distance, node.index));
        }

        let split_coord = query[node.split_dimension];
        let node_coord = node.point[node.split_dimension];
        let diff = split_coord - node_coord;

        // Search both sides if the splitting hyperplane intersects the search sphere
        if diff <= radius {
            if let Some(ref left) = node.left {
                self.search_radius(left, query, radius, neighbors);
            }
        }

        if diff >= -radius {
            if let Some(ref right) = node.right {
                self.search_radius(right, query, radius, neighbors);
            }
        }
    }

    /// Calculate distance between two points using the tree's metric
    fn calculate_distance(&self, point1: &ArrayView1<Float>, point2: &ArrayView1<Float>) -> Float {
        self.metric.calculate(point1, point2)
    }

    /// Get the number of points in the tree
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Get the number of dimensions
    pub fn n_dimensions(&self) -> usize {
        self.n_dimensions
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_kdtree_construction() {
        let data =
            Array2::from_shape_vec((4, 2), vec![2.0, 3.0, 5.0, 4.0, 9.0, 6.0, 4.0, 7.0]).unwrap();

        let tree = KdTree::new(&data, Distance::Euclidean).unwrap();
        assert_eq!(tree.n_points(), 4);
        assert_eq!(tree.n_dimensions(), 2);
    }

    #[test]
    fn test_kdtree_kneighbors() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 1.0, // Point 0
                2.0, 2.0, // Point 1
                3.0, 3.0, // Point 2
                10.0, 10.0, // Point 3
                11.0, 11.0, // Point 4
            ],
        )
        .unwrap();

        let tree = KdTree::new(&data, Distance::Euclidean).unwrap();
        let query = array![1.5, 1.5];

        let (distances, indices) = tree.kneighbors(&query.view(), 3).unwrap();

        assert_eq!(distances.len(), 3);
        assert_eq!(indices.len(), 3);

        // Check distances are sorted
        assert!(distances[0] <= distances[1]);
        assert!(distances[1] <= distances[2]);

        // Points 0 and 1 should be the two closest (both equidistant)
        // We don't care about their specific order since they're equidistant
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));

        // First two distances should be approximately equal (both are sqrt(0.5))
        assert_abs_diff_eq!(distances[0], distances[1], epsilon = 1e-10);
    }

    #[test]
    fn test_kdtree_radius_neighbors() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0, // Point 0 - distance = sqrt(0.5) ≈ 0.707
                2.0, 2.0, // Point 1 - distance = sqrt(0.5) ≈ 0.707
                3.0, 3.0, // Point 2 - distance = sqrt(4.5) ≈ 2.121 (outside radius 2.0)
                10.0, 10.0, // Point 3 - distance = sqrt(144.5) ≈ 12.02 (outside radius 2.0)
            ],
        )
        .unwrap();

        let tree = KdTree::new(&data, Distance::Euclidean).unwrap();
        let query = array![1.5, 1.5];

        let (distances, indices) = tree.radius_neighbors(&query.view(), 2.0).unwrap();

        // Should find points 0 and 1 within radius 2.0 (point 2 is at distance ~2.121 > 2.0)
        assert_eq!(distances.len(), 2);
        assert_eq!(indices.len(), 2);

        // All distances should be <= 2.0
        for &dist in &distances {
            assert!(dist <= 2.0);
        }

        // Should contain points 0 and 1
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }

    #[test]
    fn test_kdtree_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let tree = KdTree::new(&data, Distance::Euclidean).unwrap();

        let query = array![1.0, 2.0];
        let (distances, indices) = tree.kneighbors(&query.view(), 1).unwrap();

        assert_eq!(distances.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_abs_diff_eq!(distances[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kdtree_empty_input() {
        let data = Array2::zeros((0, 2));
        let result = KdTree::new(&data, Distance::Euclidean);
        assert!(result.is_err());
    }

    #[test]
    fn test_kdtree_invalid_metric() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let result = KdTree::new(&data, Distance::Manhattan);
        assert!(result.is_err());
    }

    #[test]
    fn test_kdtree_shape_mismatch() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let tree = KdTree::new(&data, Distance::Euclidean).unwrap();

        let query = array![1.0]; // Wrong dimension
        let result = tree.kneighbors(&query.view(), 1);
        assert!(result.is_err());
    }
}

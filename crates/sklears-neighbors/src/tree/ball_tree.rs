//! Ball Tree implementation for efficient nearest neighbor search

use crate::{Distance, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::types::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A node in the Ball tree
#[derive(Debug, Clone)]
pub struct BallTreeNode {
    /// Center point of the ball
    pub center: Array1<Float>,
    /// Radius of the ball
    pub radius: Float,
    /// Data points and indices stored in this node (only for leaf nodes)
    pub data_points: Option<Vec<(Array1<Float>, usize)>>,
    /// Left child (closer points)
    pub left: Option<Box<BallTreeNode>>,
    /// Right child (further points)
    pub right: Option<Box<BallTreeNode>>,
    /// Whether this is a leaf node
    pub is_leaf: bool,
}

/// Ball Tree for efficient nearest neighbor search in high-dimensional spaces
#[derive(Debug, Clone)]
pub struct BallTree {
    /// Root node of the tree
    root: Option<Box<BallTreeNode>>,
    /// Number of points in the tree
    n_points: usize,
    /// Number of dimensions
    n_dimensions: usize,
    /// Distance metric
    metric: Distance,
    /// Leaf size (maximum points in a leaf)
    leaf_size: usize,
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

impl BallTree {
    /// Create a new Ball tree from the given data points
    pub fn new(data: &Array2<Float>, metric: Distance) -> NeighborsResult<Self> {
        Self::with_leaf_size(data, metric, 10)
    }

    /// Create a new Ball tree with specified leaf size
    pub fn with_leaf_size(
        data: &Array2<Float>,
        metric: Distance,
        leaf_size: usize,
    ) -> NeighborsResult<Self> {
        if data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        if leaf_size == 0 {
            return Err(NeighborsError::InvalidInput(
                "Leaf size must be greater than 0".to_string(),
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

        let root = Self::build_tree(&mut indexed_points, &metric, leaf_size);

        Ok(Self {
            root,
            n_points,
            n_dimensions,
            metric,
            leaf_size,
        })
    }

    /// Recursively build the Ball tree
    fn build_tree(
        points: &mut [(Array1<Float>, usize)],
        metric: &Distance,
        leaf_size: usize,
    ) -> Option<Box<BallTreeNode>> {
        if points.is_empty() {
            return None;
        }

        // If we have few enough points, create a leaf node
        if points.len() <= leaf_size {
            let center = Self::compute_centroid(points);
            let radius = Self::compute_radius(&center, points, metric);

            return Some(Box::new(BallTreeNode {
                center,
                radius,
                data_points: Some(points.to_vec()),
                left: None,
                right: None,
                is_leaf: true,
            }));
        }

        // Find the two points that are farthest apart
        let (pivot1_idx, pivot2_idx) = Self::find_furthest_pair(points, metric);
        let pivot1 = points[pivot1_idx].0.clone();
        let pivot2 = points[pivot2_idx].0.clone();

        // Split points based on which pivot they're closer to
        let mut left_points = Vec::new();
        let mut right_points = Vec::new();

        for (point, index) in points.iter() {
            let dist1 = metric.calculate(&point.view(), &pivot1.view());
            let dist2 = metric.calculate(&point.view(), &pivot2.view());

            if dist1 <= dist2 {
                left_points.push((point.clone(), *index));
            } else {
                right_points.push((point.clone(), *index));
            }
        }

        // Ensure both sides have at least one point
        if left_points.is_empty() || right_points.is_empty() {
            // Fall back to median split if pivoting fails
            points.sort_by(|a, b| a.0[0].partial_cmp(&b.0[0]).unwrap_or(Ordering::Equal));
            let mid = points.len() / 2;
            left_points = points[..mid].to_vec();
            right_points = points[mid..].to_vec();
        }

        // Compute the center and radius for this ball
        let center = Self::compute_centroid(points);
        let radius = Self::compute_radius(&center, points, metric);

        // Recursively build subtrees
        let left_child = Self::build_tree(&mut left_points, metric, leaf_size);
        let right_child = Self::build_tree(&mut right_points, metric, leaf_size);

        Some(Box::new(BallTreeNode {
            center,
            radius,
            data_points: None,
            left: left_child,
            right: right_child,
            is_leaf: false,
        }))
    }

    /// Find the two points that are farthest apart
    fn find_furthest_pair(points: &[(Array1<Float>, usize)], metric: &Distance) -> (usize, usize) {
        let mut max_distance = 0.0;
        let mut best_pair = (0, 1);

        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let distance = metric.calculate(&points[i].0.view(), &points[j].0.view());
                if distance > max_distance {
                    max_distance = distance;
                    best_pair = (i, j);
                }
            }
        }

        best_pair
    }

    /// Compute the centroid of a set of points
    fn compute_centroid(points: &[(Array1<Float>, usize)]) -> Array1<Float> {
        if points.is_empty() {
            return Array1::zeros(0);
        }

        let n_dimensions = points[0].0.len();
        let mut centroid = Array1::zeros(n_dimensions);

        for (point, _) in points {
            centroid = &centroid + point;
        }

        centroid / (points.len() as Float)
    }

    /// Compute the radius of a ball (maximum distance from center to any point)
    fn compute_radius(
        center: &Array1<Float>,
        points: &[(Array1<Float>, usize)],
        metric: &Distance,
    ) -> Float {
        points
            .iter()
            .map(|(point, _)| metric.calculate(&center.view(), &point.view()))
            .fold(0.0, |max_dist, dist| max_dist.max(dist))
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
        node: &BallTreeNode,
        query: &ArrayView1<Float>,
        k: usize,
        candidates: &mut BinaryHeap<NeighborCandidate>,
    ) {
        if node.is_leaf {
            // If it's a leaf, check all points stored in this node
            if let Some(ref data_points) = node.data_points {
                for (point, index) in data_points {
                    let distance = self.metric.calculate(&point.view(), query);

                    if candidates.len() < k {
                        candidates.push(NeighborCandidate {
                            distance,
                            index: *index,
                        });
                    } else if distance < candidates.peek().unwrap().distance {
                        candidates.pop();
                        candidates.push(NeighborCandidate {
                            distance,
                            index: *index,
                        });
                    }
                }
            }
            return;
        }

        // Calculate distance to ball center
        let dist_to_center = self.metric.calculate(&node.center.view(), query);

        // Get current worst distance
        let worst_distance = if candidates.len() < k {
            Float::INFINITY
        } else {
            candidates.peek().unwrap().distance
        };

        // If the ball is too far away, prune this branch
        if dist_to_center - node.radius >= worst_distance {
            return;
        }

        // Determine which child to search first
        let (primary, secondary) =
            if let (Some(ref left), Some(ref right)) = (&node.left, &node.right) {
                let dist_left = self.metric.calculate(&left.center.view(), query);
                let dist_right = self.metric.calculate(&right.center.view(), query);

                if dist_left <= dist_right {
                    (left, right)
                } else {
                    (right, left)
                }
            } else if let Some(ref left) = &node.left {
                (left, left) // Only left child exists
            } else if let Some(ref right) = &node.right {
                (right, right) // Only right child exists
            } else {
                return; // No children
            };

        // Search the closer child first
        self.search_knn(primary, query, k, candidates);

        // Check if we need to search the other child
        let updated_worst = if candidates.len() < k {
            Float::INFINITY
        } else {
            candidates.peek().unwrap().distance
        };

        if !std::ptr::eq(primary.as_ref(), secondary.as_ref()) {
            let dist_to_secondary = self.metric.calculate(&secondary.center.view(), query);
            if dist_to_secondary - secondary.radius < updated_worst {
                self.search_knn(secondary, query, k, candidates);
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
        node: &BallTreeNode,
        query: &ArrayView1<Float>,
        radius: Float,
        neighbors: &mut Vec<(Float, usize)>,
    ) {
        let dist_to_center = self.metric.calculate(&node.center.view(), query);

        // If the ball is too far away, prune this branch
        if dist_to_center - node.radius > radius {
            return;
        }

        if node.is_leaf {
            // If it's a leaf, check all points stored in this node
            if let Some(ref data_points) = node.data_points {
                for (point, index) in data_points {
                    let distance = self.metric.calculate(&point.view(), query);
                    if distance <= radius {
                        neighbors.push((distance, *index));
                    }
                }
            }
            return;
        }

        // Search children
        if let Some(ref left) = node.left {
            self.search_radius(left, query, radius, neighbors);
        }
        if let Some(ref right) = node.right {
            self.search_radius(right, query, radius, neighbors);
        }
    }

    /// Get the number of points in the tree
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Get the number of dimensions
    pub fn n_dimensions(&self) -> usize {
        self.n_dimensions
    }

    /// Get the leaf size
    pub fn leaf_size(&self) -> usize {
        self.leaf_size
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_balltree_construction() {
        let data =
            Array2::from_shape_vec((4, 2), vec![2.0, 3.0, 5.0, 4.0, 9.0, 6.0, 4.0, 7.0]).unwrap();

        let tree = BallTree::new(&data, Distance::Euclidean).unwrap();
        assert_eq!(tree.n_points(), 4);
        assert_eq!(tree.n_dimensions(), 2);
        assert_eq!(tree.leaf_size(), 10);
    }

    #[test]
    fn test_balltree_kneighbors() {
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

        let tree = BallTree::new(&data, Distance::Euclidean).unwrap();
        let query = array![1.5, 1.5];

        let (distances, indices) = tree.kneighbors(&query.view(), 3).unwrap();

        assert_eq!(distances.len(), 3);
        assert_eq!(indices.len(), 3);

        // Check distances are sorted
        assert!(distances[0] <= distances[1]);
        assert!(distances[1] <= distances[2]);

        // Points 0 and 1 should be the two closest (both equidistant)
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }

    #[test]
    fn test_balltree_radius_neighbors() {
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

        let tree = BallTree::new(&data, Distance::Euclidean).unwrap();
        let query = array![1.5, 1.5];

        let (distances, indices) = tree.radius_neighbors(&query.view(), 2.0).unwrap();

        // Should find points 0 and 1 within radius 2.0
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
    fn test_balltree_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let tree = BallTree::new(&data, Distance::Euclidean).unwrap();

        let query = array![1.0, 2.0];
        let (distances, indices) = tree.kneighbors(&query.view(), 1).unwrap();

        assert_eq!(distances.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_abs_diff_eq!(distances[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_balltree_empty_input() {
        let data = Array2::zeros((0, 2));
        let result = BallTree::new(&data, Distance::Euclidean);
        assert!(result.is_err());
    }

    #[test]
    fn test_balltree_shape_mismatch() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let tree = BallTree::new(&data, Distance::Euclidean).unwrap();

        let query = array![1.0]; // Wrong dimension
        let result = tree.kneighbors(&query.view(), 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_balltree_with_leaf_size() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
        )
        .unwrap();

        let tree = BallTree::with_leaf_size(&data, Distance::Euclidean, 2).unwrap();
        assert_eq!(tree.leaf_size(), 2);

        let query = array![3.5, 3.5];
        let (distances, indices) = tree.kneighbors(&query.view(), 2).unwrap();

        assert_eq!(distances.len(), 2);
        assert_eq!(indices.len(), 2);
    }
}

//! Vantage Point Tree implementation for efficient metric space searching
//!
//! VP-trees are particularly effective for high-dimensional data and arbitrary metric spaces
//! where the triangle inequality holds. They work by recursively partitioning the data
//! based on distance from a "vantage point" (pivot).

use crate::{Distance, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array2, ArrayView1};
use sklears_core::types::Float;
use std::cmp::Ordering;

/// A node in the VP-tree
#[derive(Debug, Clone)]
enum VpNode {
    /// Internal node containing a vantage point and threshold
    Internal {
        /// Index of the vantage point in the original data
        vantage_point: usize,
        /// Threshold distance for partitioning
        threshold: Float,
        /// Left subtree (points closer than threshold)
        left: Box<VpNode>,
        /// Right subtree (points farther than threshold)
        right: Box<VpNode>,
    },
    /// Leaf node containing point indices
    Leaf {
        /// Indices of points in this leaf
        points: Vec<usize>,
    },
}

/// Vantage Point Tree for efficient nearest neighbor search in metric spaces
#[derive(Debug, Clone)]
pub struct VpTree {
    /// Root node of the tree
    root: Option<VpNode>,
    /// Training data
    data: Array2<Float>,
    /// Distance metric
    metric: Distance,
    /// Maximum leaf size
    leaf_size: usize,
}

/// A priority queue entry for k-NN search
#[derive(Debug, Clone)]
struct HeapItem {
    distance: Float,
    index: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for max-heap behavior
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl VpTree {
    /// Create a new VP-tree from training data
    pub fn new(data: &Array2<Float>, metric: Distance) -> NeighborsResult<Self> {
        if data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let mut tree = Self {
            root: None,
            data: data.clone(),
            metric,
            leaf_size: 40, // Default leaf size
        };

        let indices: Vec<usize> = (0..data.nrows()).collect();
        tree.root = Some(tree.build_tree(indices)?);
        Ok(tree)
    }

    /// Set the leaf size parameter
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size;
        self
    }

    /// Build the VP-tree recursively
    fn build_tree(&self, mut indices: Vec<usize>) -> NeighborsResult<VpNode> {
        if indices.len() <= self.leaf_size {
            return Ok(VpNode::Leaf { points: indices });
        }

        // Choose vantage point (for simplicity, use the first point)
        // In practice, random selection or more sophisticated heuristics work better
        let vp_idx = 0;
        let vp_global_idx = indices.swap_remove(vp_idx);
        let vp_point = self.data.row(vp_global_idx);

        if indices.is_empty() {
            return Ok(VpNode::Leaf {
                points: vec![vp_global_idx],
            });
        }

        // Calculate distances from vantage point to all other points
        let mut distances: Vec<(Float, usize)> = indices
            .iter()
            .map(|&idx| {
                let point = self.data.row(idx);
                let dist = self.metric.calculate(&vp_point, &point);
                (dist, idx)
            })
            .collect();

        // Sort by distance to find median
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Use median as threshold
        let median_idx = distances.len() / 2;
        let threshold = distances[median_idx].0;

        // Partition points
        let left_indices: Vec<usize> = distances[..median_idx]
            .iter()
            .map(|&(_, idx)| idx)
            .collect();
        let right_indices: Vec<usize> = distances[median_idx..]
            .iter()
            .map(|&(_, idx)| idx)
            .collect();

        // Recursively build subtrees
        let left = Box::new(self.build_tree(left_indices)?);
        let right = Box::new(self.build_tree(right_indices)?);

        Ok(VpNode::Internal {
            vantage_point: vp_global_idx,
            threshold,
            left,
            right,
        })
    }

    /// Find k nearest neighbors
    pub fn kneighbors(
        &self,
        query: &ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<Float>, Vec<usize>)> {
        if k == 0 {
            return Err(NeighborsError::InvalidNeighbors(k));
        }

        let mut heap = std::collections::BinaryHeap::new();

        if let Some(ref root) = self.root {
            self.search_node(root, query, k, &mut heap)?;
        }

        // Extract results from heap
        let mut results: Vec<_> = heap.into_sorted_vec();
        results.reverse(); // Sort by increasing distance

        let distances: Vec<Float> = results.iter().map(|item| item.distance).collect();
        let indices: Vec<usize> = results.iter().map(|item| item.index).collect();

        Ok((distances, indices))
    }

    /// Search a node recursively
    fn search_node(
        &self,
        node: &VpNode,
        query: &ArrayView1<Float>,
        k: usize,
        heap: &mut std::collections::BinaryHeap<HeapItem>,
    ) -> NeighborsResult<()> {
        match node {
            VpNode::Leaf { points } => {
                // Search all points in the leaf
                for &idx in points {
                    let point = self.data.row(idx);
                    let distance = self.metric.calculate(query, &point);

                    if heap.len() < k {
                        heap.push(HeapItem {
                            distance,
                            index: idx,
                        });
                    } else if let Some(furthest) = heap.peek() {
                        if distance < furthest.distance {
                            heap.pop();
                            heap.push(HeapItem {
                                distance,
                                index: idx,
                            });
                        }
                    }
                }
            }
            VpNode::Internal {
                vantage_point,
                threshold,
                left,
                right,
            } => {
                // Calculate distance to vantage point
                let vp_point = self.data.row(*vantage_point);
                let vp_distance = self.metric.calculate(query, &vp_point);

                // Consider the vantage point itself
                if heap.len() < k {
                    heap.push(HeapItem {
                        distance: vp_distance,
                        index: *vantage_point,
                    });
                } else if let Some(furthest) = heap.peek() {
                    if vp_distance < furthest.distance {
                        heap.pop();
                        heap.push(HeapItem {
                            distance: vp_distance,
                            index: *vantage_point,
                        });
                    }
                }

                // Determine which subtree to search first
                let search_left_first = vp_distance < *threshold;

                // Search the closer subtree first
                if search_left_first {
                    self.search_node(left, query, k, heap)?;
                } else {
                    self.search_node(right, query, k, heap)?;
                }

                // Check if we need to search the other subtree
                let max_distance = heap
                    .peek()
                    .map(|item| item.distance)
                    .unwrap_or(Float::INFINITY);

                let should_search_other = heap.len() < k
                    || (vp_distance - max_distance).abs() <= *threshold
                    || (search_left_first && vp_distance + max_distance >= *threshold)
                    || (!search_left_first && vp_distance <= *threshold + max_distance);

                if should_search_other {
                    if search_left_first {
                        self.search_node(right, query, k, heap)?;
                    } else {
                        self.search_node(left, query, k, heap)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Find all neighbors within a given radius
    pub fn radius_neighbors(
        &self,
        query: &ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<(Vec<Float>, Vec<usize>)> {
        if radius < 0.0 {
            return Err(NeighborsError::InvalidRadius(radius));
        }

        let mut results = Vec::new();

        if let Some(ref root) = self.root {
            self.radius_search_node(root, query, radius, &mut results)?;
        }

        // Sort by distance
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let distances: Vec<Float> = results.iter().map(|&(dist, _)| dist).collect();
        let indices: Vec<usize> = results.iter().map(|&(_, idx)| idx).collect();

        Ok((distances, indices))
    }

    /// Search a node for radius neighbors
    fn radius_search_node(
        &self,
        node: &VpNode,
        query: &ArrayView1<Float>,
        radius: Float,
        results: &mut Vec<(Float, usize)>,
    ) -> NeighborsResult<()> {
        match node {
            VpNode::Leaf { points } => {
                // Search all points in the leaf
                for &idx in points {
                    let point = self.data.row(idx);
                    let distance = self.metric.calculate(query, &point);

                    if distance <= radius {
                        results.push((distance, idx));
                    }
                }
            }
            VpNode::Internal {
                vantage_point,
                threshold,
                left,
                right,
            } => {
                // Calculate distance to vantage point
                let vp_point = self.data.row(*vantage_point);
                let vp_distance = self.metric.calculate(query, &vp_point);

                // Consider the vantage point itself
                if vp_distance <= radius {
                    results.push((vp_distance, *vantage_point));
                }

                // Check which subtrees might contain points within radius
                if vp_distance <= *threshold + radius {
                    self.radius_search_node(left, query, radius, results)?;
                }

                if vp_distance + radius >= *threshold {
                    self.radius_search_node(right, query, radius, results)?;
                }
            }
        }

        Ok(())
    }

    /// Get the number of points in the tree
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_vp_tree_construction() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let tree = VpTree::new(&data, Distance::default()).unwrap();
        assert_eq!(tree.len(), 4);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_vp_tree_kneighbors() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // Point 0
                2.0, 2.0, // Point 1
                3.0, 3.0, // Point 2
                10.0, 10.0, // Point 3 (far)
                11.0, 11.0, // Point 4 (far)
                12.0, 12.0, // Point 5 (far)
            ],
        )
        .unwrap();

        let tree = VpTree::new(&data, Distance::default()).unwrap();

        // Query near the first cluster
        let query = array![1.5, 1.5];
        let (distances, indices) = tree.kneighbors(&query.view(), 3).unwrap();

        assert_eq!(distances.len(), 3);
        assert_eq!(indices.len(), 3);

        // First few neighbors should be from the close cluster
        let first_neighbor_dist =
            ((1.5 as Float - 1.0 as Float).powi(2) + (1.5 as Float - 1.0 as Float).powi(2)).sqrt();
        assert_abs_diff_eq!(distances[0], first_neighbor_dist, epsilon = 1e-6);
    }

    #[test]
    fn test_vp_tree_radius_neighbors() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, // Point 0
                1.0, 0.0, // Point 1
                0.0, 1.0, // Point 2
                5.0, 5.0, // Point 3 (far)
            ],
        )
        .unwrap();

        let tree = VpTree::new(&data, Distance::default()).unwrap();

        // Query at origin with radius 1.5
        let query = array![0.0, 0.0];
        let (distances, indices) = tree.radius_neighbors(&query.view(), 1.5).unwrap();

        // Should find points 0, 1, 2 but not 3
        assert!(distances.len() >= 3);
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));

        // All returned distances should be <= radius
        for &dist in &distances {
            assert!(dist <= 1.5 + 1e-10);
        }
    }

    #[test]
    fn test_vp_tree_with_different_metrics() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();

        // Test with Manhattan distance
        let tree = VpTree::new(&data, Distance::Manhattan).unwrap();
        let query = array![0.5, 0.5];
        let (distances, indices) = tree.kneighbors(&query.view(), 2).unwrap();

        assert_eq!(distances.len(), 2);
        assert_eq!(indices.len(), 2);

        // Check that we get reasonable results
        assert!(distances[0] <= distances[1]);
    }

    #[test]
    fn test_vp_tree_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let tree = VpTree::new(&data, Distance::default()).unwrap();

        let query = array![1.0, 2.0];
        let (distances, indices) = tree.kneighbors(&query.view(), 1).unwrap();

        assert_eq!(distances.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_abs_diff_eq!(distances[0], 0.0, epsilon = 1e-10);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_vp_tree_empty_input() {
        let data = Array2::zeros((0, 2));
        let result = VpTree::new(&data, Distance::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_vp_tree_invalid_k() {
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let tree = VpTree::new(&data, Distance::default()).unwrap();

        let query = array![0.0, 0.0];
        let result = tree.kneighbors(&query.view(), 0);
        assert!(result.is_err());
    }
}

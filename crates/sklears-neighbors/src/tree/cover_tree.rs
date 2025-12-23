//! Cover Tree implementation for efficient nearest neighbor search
//!
//! Cover trees provide theoretical guarantees for nearest neighbor search in metric spaces.
//! They maintain three key invariants:
//! 1. Nesting: C_i ⊆ C_{i-1}
//! 2. Covering: For every p ∈ C_{i-1}, there exists q ∈ C_i such that d(p,q) ≤ 2^i
//! 3. Separation: For any p, q ∈ C_i, d(p,q) > 2^i

use crate::{Distance, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array2, ArrayView1};
use sklears_core::types::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A node in the cover tree
#[derive(Debug, Clone)]
pub struct CoverNode {
    /// Index of the point in the original data
    pub point_idx: usize,
    /// Scale level of this node
    pub level: i32,
    /// Children at the next level down
    pub children: Vec<CoverNode>,
    /// Maximum distance to any descendant
    pub max_distance: Float,
}

impl CoverNode {
    /// Create a new cover node
    pub fn new(point_idx: usize, level: i32) -> Self {
        Self {
            point_idx,
            level,
            children: Vec::new(),
            max_distance: 0.0,
        }
    }

    /// Update the maximum distance to descendants
    pub fn update_max_distance(&mut self, data: &Array2<Float>, metric: &Distance) {
        if self.children.is_empty() {
            self.max_distance = 0.0;
            return;
        }

        let self_point = data.row(self.point_idx);
        self.max_distance = self
            .children
            .iter()
            .map(|child| {
                let child_point = data.row(child.point_idx);
                let dist = metric.calculate(&self_point, &child_point);
                dist + child.max_distance
            })
            .fold(0.0, Float::max);
    }
}

/// Priority queue item for k-NN search
#[derive(Debug, Clone)]
struct SearchItem {
    distance: Float,
    node_idx: usize,
    is_point: bool,
}

impl PartialEq for SearchItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchItem {}

impl PartialOrd for SearchItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Cover Tree for efficient nearest neighbor search in metric spaces
#[derive(Debug, Clone)]
pub struct CoverTree {
    /// Training data
    data: Array2<Float>,
    /// Distance metric
    metric: Distance,
    /// Root node of the tree
    root: Option<CoverNode>,
    /// Base of the tree (typically 2.0)
    base: Float,
    /// Minimum scale level
    min_level: i32,
    /// Maximum scale level
    max_level: i32,
}

impl CoverTree {
    /// Create a new cover tree
    pub fn new(data: Array2<Float>, metric: Distance) -> NeighborsResult<Self> {
        if data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let mut tree = Self {
            data: data.clone(),
            metric,
            root: None,
            base: 2.0,
            min_level: -10, // Arbitrary minimum level
            max_level: 10,  // Arbitrary maximum level
        };

        tree.build_tree()?;
        Ok(tree)
    }

    /// Build the cover tree from the data
    fn build_tree(&mut self) -> NeighborsResult<()> {
        let n_points = self.data.nrows();
        if n_points == 0 {
            return Ok(());
        }

        // Start with all points at the top level
        let mut point_indices: Vec<usize> = (0..n_points).collect();

        // Find appropriate starting level
        let max_distance = self.compute_max_distance(&point_indices);
        self.max_level = (max_distance.log2().ceil() as i32).max(0);

        // Pick the first point as root
        let root_idx = point_indices.remove(0);
        let mut root = CoverNode::new(root_idx, self.max_level);

        // Recursively build the tree
        self.build_recursive(&mut root, &mut point_indices)?;
        root.update_max_distance(&self.data, &self.metric);

        self.root = Some(root);
        Ok(())
    }

    /// Recursively build the cover tree
    #[allow(clippy::only_used_in_recursion)]
    fn build_recursive(
        &self,
        node: &mut CoverNode,
        remaining_points: &mut Vec<usize>,
    ) -> NeighborsResult<()> {
        if remaining_points.is_empty() || node.level <= self.min_level {
            return Ok(());
        }

        let node_point = self.data.row(node.point_idx);
        let scale = self.base.powi(node.level);

        // Separate points into those within scale distance and those outside
        let mut close_points = Vec::new();
        let mut far_points = Vec::new();

        for &point_idx in remaining_points.iter() {
            let point = self.data.row(point_idx);
            let distance = self.metric.calculate(&node_point, &point);

            if distance <= scale {
                close_points.push(point_idx);
            } else {
                far_points.push(point_idx);
            }
        }

        // Create children for the next level
        if !close_points.is_empty() && node.level > self.min_level {
            // Use a greedy algorithm to select cover points
            let children_indices = self.select_cover_points(&close_points, node.level - 1);

            for child_idx in children_indices {
                let mut child = CoverNode::new(child_idx, node.level - 1);

                // Remove this child from close_points
                if let Some(pos) = close_points.iter().position(|&x| x == child_idx) {
                    close_points.remove(pos);
                }

                // Recursively build the child's subtree
                self.build_recursive(&mut child, &mut close_points)?;
                child.update_max_distance(&self.data, &self.metric);
                node.children.push(child);
            }
        }

        // Update remaining points to be the far points
        *remaining_points = far_points;
        Ok(())
    }

    /// Select cover points using a greedy algorithm
    fn select_cover_points(&self, points: &[usize], level: i32) -> Vec<usize> {
        if points.is_empty() {
            return Vec::new();
        }

        let mut cover_points = Vec::new();
        let mut remaining: Vec<usize> = points.to_vec();
        let separation_distance = self.base.powi(level);

        while !remaining.is_empty() {
            // Pick the first remaining point as a cover point
            let cover_idx = remaining.remove(0);
            cover_points.push(cover_idx);

            let cover_point = self.data.row(cover_idx);

            // Remove all points within separation distance
            remaining.retain(|&point_idx| {
                let point = self.data.row(point_idx);
                let distance = self.metric.calculate(&cover_point, &point);
                distance > separation_distance
            });
        }

        cover_points
    }

    /// Compute maximum distance between any pair of points
    fn compute_max_distance(&self, points: &[usize]) -> Float {
        if points.len() < 2 {
            return 0.0;
        }

        let mut max_dist: Float = 0.0;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let p1 = self.data.row(points[i]);
                let p2 = self.data.row(points[j]);
                let dist = self.metric.calculate(&p1, &p2);
                max_dist = max_dist.max(dist);
            }
        }
        max_dist
    }

    /// Find k nearest neighbors
    pub fn kneighbors(
        &self,
        query: &ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<Float>, Vec<usize>)> {
        let root = self.root.as_ref().ok_or(NeighborsError::EmptyInput)?;

        let mut candidates = BinaryHeap::new();
        let mut visited = Vec::new();

        // Start search from root
        self.search_node(query, root, &mut candidates, &mut visited)?;

        // Sort candidates by distance and take k nearest
        let mut all_candidates: Vec<(Float, usize)> = visited;
        all_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let k = k.min(all_candidates.len());
        let distances = all_candidates.iter().take(k).map(|(d, _)| *d).collect();
        let indices = all_candidates.iter().take(k).map(|(_, i)| *i).collect();

        Ok((distances, indices))
    }

    /// Recursively search a node and its children
    #[allow(clippy::only_used_in_recursion)]
    fn search_node(
        &self,
        query: &ArrayView1<Float>,
        node: &CoverNode,
        candidates: &mut BinaryHeap<SearchItem>,
        visited: &mut Vec<(Float, usize)>,
    ) -> NeighborsResult<()> {
        let node_point = self.data.row(node.point_idx);
        let distance = self.metric.calculate(query, &node_point);

        // Add this node's point to visited
        visited.push((distance, node.point_idx));

        // Search children that could contain closer points
        for child in &node.children {
            let child_point = self.data.row(child.point_idx);
            let child_distance = self.metric.calculate(query, &child_point);

            // Check if we need to explore this child
            // We explore if the query could be within the child's covering radius
            let covering_radius = self.base.powi(child.level) + child.max_distance;

            if child_distance <= covering_radius {
                self.search_node(query, child, candidates, visited)?;
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

    /// Get the data
    pub fn data(&self) -> &Array2<Float> {
        &self.data
    }

    /// Get the metric
    pub fn metric(&self) -> &Distance {
        &self.metric
    }

    /// Get tree statistics
    pub fn stats(&self) -> CoverTreeStats {
        let root = self.root.as_ref();
        CoverTreeStats {
            n_points: self.len(),
            max_level: self.max_level,
            min_level: self.min_level,
            depth: root.map(|r| self.compute_depth(r)).unwrap_or(0),
            n_nodes: root.map(|r| self.count_nodes(r)).unwrap_or(0),
        }
    }

    /// Compute the depth of the tree
    #[allow(clippy::only_used_in_recursion)]
    fn compute_depth(&self, node: &CoverNode) -> usize {
        if node.children.is_empty() {
            1
        } else {
            1 + node
                .children
                .iter()
                .map(|c| self.compute_depth(c))
                .max()
                .unwrap_or(0)
        }
    }

    /// Count the total number of nodes
    #[allow(clippy::only_used_in_recursion)]
    fn count_nodes(&self, node: &CoverNode) -> usize {
        1 + node
            .children
            .iter()
            .map(|c| self.count_nodes(c))
            .sum::<usize>()
    }
}

/// Statistics about the cover tree
#[derive(Debug, Clone)]
pub struct CoverTreeStats {
    pub n_points: usize,
    pub max_level: i32,
    pub min_level: i32,
    pub depth: usize,
    pub n_nodes: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_cover_tree_construction() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let tree = CoverTree::new(data, Distance::Euclidean).unwrap();
        assert_eq!(tree.len(), 4);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_cover_tree_kneighbors() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, // Point 0
                1.0, 0.0, // Point 1
                0.0, 1.0, // Point 2
                3.0, 3.0, // Point 3 (far)
                0.1, 0.1, // Point 4 (close to 0)
            ],
        )
        .unwrap();

        let tree = CoverTree::new(data, Distance::Euclidean).unwrap();

        // Query point close to origin
        let query = array![0.05, 0.05];
        let (distances, indices) = tree.kneighbors(&query.view(), 3).unwrap();

        assert_eq!(distances.len(), 3);
        assert_eq!(indices.len(), 3);

        // Should find the 3 nearest points
        // The closest should be point 4 or 0
        assert!(indices.contains(&0) || indices.contains(&4));
    }

    #[test]
    fn test_cover_tree_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let tree = CoverTree::new(data, Distance::Euclidean).unwrap();

        let query = array![1.1, 2.1];
        let (distances, indices) = tree.kneighbors(&query.view(), 1).unwrap();

        assert_eq!(distances.len(), 1);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_cover_tree_empty_data() {
        let data = Array2::zeros((0, 2));
        let result = CoverTree::new(data, Distance::Euclidean);
        assert!(result.is_err());
    }

    #[test]
    fn test_cover_tree_stats() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )
        .unwrap();

        let tree = CoverTree::new(data, Distance::Euclidean).unwrap();
        let stats = tree.stats();

        assert_eq!(stats.n_points, 6);
        assert!(stats.depth > 0);
        assert!(stats.n_nodes > 0);
        assert!(stats.max_level >= stats.min_level);
    }

    #[test]
    fn test_cover_tree_different_metrics() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        // Test with Manhattan distance
        let tree_manhattan = CoverTree::new(data.clone(), Distance::Manhattan).unwrap();
        let query = array![0.5, 0.5];
        let (_, _) = tree_manhattan.kneighbors(&query.view(), 2).unwrap();

        // Test with Cosine distance
        let tree_cosine = CoverTree::new(data, Distance::Cosine).unwrap();
        let (_, _) = tree_cosine.kneighbors(&query.view(), 2).unwrap();
    }

    #[test]
    fn test_cover_tree_large_k() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        let tree = CoverTree::new(data, Distance::Euclidean).unwrap();
        let query = array![0.0, 0.0];

        // Request more neighbors than available points
        let (distances, indices) = tree.kneighbors(&query.view(), 10).unwrap();

        // Should return all available points
        assert_eq!(distances.len(), 3);
        assert_eq!(indices.len(), 3);
    }
}

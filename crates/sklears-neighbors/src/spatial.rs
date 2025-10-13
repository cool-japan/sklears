//! Spatial data structures for efficient nearest neighbor search
//!
//! This module provides specialized data structures for spatial data including
//! R-trees for general spatial indexing and Quad-trees for 2D spatial data.

use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float as FloatTrait;
use sklears_core::types::Float;

/// Rectangle/bounding box for spatial indexing
#[derive(Clone, Debug, PartialEq)]
pub struct Rectangle {
    /// Minimum bounds for each dimension
    pub min_bounds: Array1<Float>,
    /// Maximum bounds for each dimension
    pub max_bounds: Array1<Float>,
}

impl Rectangle {
    /// Create a new rectangle
    pub fn new(min_bounds: Array1<Float>, max_bounds: Array1<Float>) -> NeighborsResult<Self> {
        if min_bounds.len() != max_bounds.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![min_bounds.len()],
                actual: vec![max_bounds.len()],
            });
        }

        for (min, max) in min_bounds.iter().zip(max_bounds.iter()) {
            if min > max {
                return Err(NeighborsError::InvalidInput(
                    "Min bounds must be less than or equal to max bounds".to_string(),
                ));
            }
        }

        Ok(Rectangle {
            min_bounds,
            max_bounds,
        })
    }

    /// Create a rectangle from a single point
    pub fn from_point(point: &ArrayView1<Float>) -> Self {
        Rectangle {
            min_bounds: point.to_owned(),
            max_bounds: point.to_owned(),
        }
    }

    /// Check if a point is contained within this rectangle
    pub fn contains_point(&self, point: &ArrayView1<Float>) -> bool {
        for ((min, max), &coord) in self
            .min_bounds
            .iter()
            .zip(self.max_bounds.iter())
            .zip(point.iter())
        {
            if coord < *min || coord > *max {
                return false;
            }
        }
        true
    }

    /// Check if this rectangle intersects with another rectangle
    pub fn intersects(&self, other: &Rectangle) -> bool {
        for ((min1, max1), (min2, max2)) in self
            .min_bounds
            .iter()
            .zip(self.max_bounds.iter())
            .zip(other.min_bounds.iter().zip(other.max_bounds.iter()))
        {
            if *max1 < *min2 || *max2 < *min1 {
                return false;
            }
        }
        true
    }

    /// Compute the union of two rectangles
    pub fn union(&self, other: &Rectangle) -> Rectangle {
        let min_bounds = self
            .min_bounds
            .iter()
            .zip(other.min_bounds.iter())
            .map(|(a, b)| a.min(*b))
            .collect::<Array1<Float>>();

        let max_bounds = self
            .max_bounds
            .iter()
            .zip(other.max_bounds.iter())
            .map(|(a, b)| a.max(*b))
            .collect::<Array1<Float>>();

        Rectangle {
            min_bounds,
            max_bounds,
        }
    }

    /// Compute the area/volume of the rectangle
    pub fn area(&self) -> Float {
        self.min_bounds
            .iter()
            .zip(self.max_bounds.iter())
            .map(|(min, max)| max - min)
            .product()
    }

    /// Compute the minimum distance from a point to this rectangle
    pub fn min_distance_to_point(&self, point: &ArrayView1<Float>) -> Float {
        let mut distance_sq = 0.0;

        for ((min, max), &coord) in self
            .min_bounds
            .iter()
            .zip(self.max_bounds.iter())
            .zip(point.iter())
        {
            if coord < *min {
                distance_sq += (min - coord).powi(2);
            } else if coord > *max {
                distance_sq += (coord - max).powi(2);
            }
        }

        distance_sq.sqrt()
    }

    /// Compute the maximum distance from a point to this rectangle
    pub fn max_distance_to_point(&self, point: &ArrayView1<Float>) -> Float {
        let mut distance_sq = 0.0;

        for ((min, max), &coord) in self
            .min_bounds
            .iter()
            .zip(self.max_bounds.iter())
            .zip(point.iter())
        {
            let dist_to_min = (coord - min).abs();
            let dist_to_max = (coord - max).abs();
            distance_sq += dist_to_min.max(dist_to_max).powi(2);
        }

        distance_sq.sqrt()
    }
}

/// R-tree node for spatial indexing
#[derive(Clone, Debug)]
enum RTreeNode {
    /// Internal node containing child nodes
    Internal {
        /// Bounding rectangle
        bounds: Rectangle,
        /// Child nodes
        children: Vec<RTreeNode>,
    },
    /// Leaf node containing data points
    Leaf {
        /// Bounding rectangle
        bounds: Rectangle,
        /// Data points and their indices
        points: Vec<(Array1<Float>, usize)>,
    },
}

impl RTreeNode {
    /// Get the bounding rectangle of this node
    fn bounds(&self) -> &Rectangle {
        match self {
            RTreeNode::Internal { bounds, .. } => bounds,
            RTreeNode::Leaf { bounds, .. } => bounds,
        }
    }

    /// Check if this is a leaf node
    fn is_leaf(&self) -> bool {
        matches!(self, RTreeNode::Leaf { .. })
    }

    /// Get the number of entries in this node
    fn len(&self) -> usize {
        match self {
            RTreeNode::Internal { children, .. } => children.len(),
            RTreeNode::Leaf { points, .. } => points.len(),
        }
    }
}

/// R-tree for efficient spatial indexing and range queries
pub struct RTree {
    /// Root node of the R-tree
    root: Option<RTreeNode>,
    /// Maximum number of entries per node
    max_entries: usize,
    /// Minimum number of entries per node
    min_entries: usize,
    /// Dimensionality of the space
    dimensions: usize,
}

impl RTree {
    /// Create a new R-tree
    pub fn new(max_entries: usize, dimensions: usize) -> Self {
        let min_entries = max_entries / 2;
        Self {
            root: None,
            max_entries,
            min_entries,
            dimensions,
        }
    }

    /// Insert a point into the R-tree
    pub fn insert(&mut self, point: Array1<Float>, index: usize) -> NeighborsResult<()> {
        if point.len() != self.dimensions {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![self.dimensions],
                actual: vec![point.len()],
            });
        }

        if self.root.is_none() {
            // Create root as leaf
            let bounds = Rectangle::from_point(&point.view());
            self.root = Some(RTreeNode::Leaf {
                bounds,
                points: vec![(point, index)],
            });
        } else {
            let root = self.root.take().unwrap();
            let new_root = self.insert_recursive(root, point, index)?;
            self.root = Some(new_root);
        }

        Ok(())
    }

    /// Recursive insertion into R-tree
    fn insert_recursive(
        &self,
        node: RTreeNode,
        point: Array1<Float>,
        index: usize,
    ) -> NeighborsResult<RTreeNode> {
        match node {
            RTreeNode::Leaf { bounds, mut points } => {
                // Add point to leaf
                points.push((point.clone(), index));
                let new_bounds = bounds.union(&Rectangle::from_point(&point.view()));

                if points.len() <= self.max_entries {
                    // Leaf not full
                    Ok(RTreeNode::Leaf {
                        bounds: new_bounds,
                        points,
                    })
                } else {
                    // Split leaf
                    self.split_leaf(points)
                }
            }
            RTreeNode::Internal {
                bounds,
                mut children,
            } => {
                // Choose child with minimum area increase
                let best_child_idx = self.choose_child(&children, &point);

                // Insert into chosen child
                let child = children.remove(best_child_idx);
                let new_child = self.insert_recursive(child, point, index)?;
                children.insert(best_child_idx, new_child);

                // Update bounds
                let new_bounds = children
                    .iter()
                    .fold(None, |acc, child| match acc {
                        None => Some(child.bounds().clone()),
                        Some(rect) => Some(rect.union(child.bounds())),
                    })
                    .unwrap_or(bounds);

                if children.len() <= self.max_entries {
                    // Internal node not full
                    Ok(RTreeNode::Internal {
                        bounds: new_bounds,
                        children,
                    })
                } else {
                    // Split internal node
                    self.split_internal(children)
                }
            }
        }
    }

    /// Choose the best child for insertion
    fn choose_child(&self, children: &[RTreeNode], point: &Array1<Float>) -> usize {
        let mut best_idx = 0;
        let mut min_area_increase = Float::infinity();

        for (i, child) in children.iter().enumerate() {
            let current_area = child.bounds().area();
            let new_bounds = child.bounds().union(&Rectangle::from_point(&point.view()));
            let new_area = new_bounds.area();
            let area_increase = new_area - current_area;

            if area_increase < min_area_increase {
                min_area_increase = area_increase;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Split a leaf node (simplified version)
    fn split_leaf(&self, points: Vec<(Array1<Float>, usize)>) -> NeighborsResult<RTreeNode> {
        let mid = points.len() / 2;
        let (left_points, right_points) = points.split_at(mid);

        let left_bounds = self.compute_bounds_for_points(left_points);
        let right_bounds = self.compute_bounds_for_points(right_points);

        let left_leaf = RTreeNode::Leaf {
            bounds: left_bounds,
            points: left_points.to_vec(),
        };

        let right_leaf = RTreeNode::Leaf {
            bounds: right_bounds,
            points: right_points.to_vec(),
        };

        let combined_bounds = left_leaf.bounds().union(right_leaf.bounds());

        Ok(RTreeNode::Internal {
            bounds: combined_bounds,
            children: vec![left_leaf, right_leaf],
        })
    }

    /// Split an internal node (simplified version)
    fn split_internal(&self, children: Vec<RTreeNode>) -> NeighborsResult<RTreeNode> {
        let mid = children.len() / 2;
        let (left_children, right_children) = children.split_at(mid);

        let left_bounds = self.compute_bounds_for_nodes(left_children);
        let right_bounds = self.compute_bounds_for_nodes(right_children);

        let left_internal = RTreeNode::Internal {
            bounds: left_bounds,
            children: left_children.to_vec(),
        };

        let right_internal = RTreeNode::Internal {
            bounds: right_bounds,
            children: right_children.to_vec(),
        };

        let combined_bounds = left_internal.bounds().union(right_internal.bounds());

        Ok(RTreeNode::Internal {
            bounds: combined_bounds,
            children: vec![left_internal, right_internal],
        })
    }

    /// Compute bounds for a set of points
    fn compute_bounds_for_points(&self, points: &[(Array1<Float>, usize)]) -> Rectangle {
        if points.is_empty() {
            return Rectangle::new(
                Array1::zeros(self.dimensions),
                Array1::zeros(self.dimensions),
            )
            .unwrap();
        }

        let first_point = &points[0].0;
        let mut min_bounds = first_point.clone();
        let mut max_bounds = first_point.clone();

        for (point, _) in points.iter().skip(1) {
            for (i, &coord) in point.iter().enumerate() {
                min_bounds[i] = min_bounds[i].min(coord);
                max_bounds[i] = max_bounds[i].max(coord);
            }
        }

        Rectangle::new(min_bounds, max_bounds).unwrap()
    }

    /// Compute bounds for a set of nodes
    fn compute_bounds_for_nodes(&self, nodes: &[RTreeNode]) -> Rectangle {
        if nodes.is_empty() {
            return Rectangle::new(
                Array1::zeros(self.dimensions),
                Array1::zeros(self.dimensions),
            )
            .unwrap();
        }

        let first_bounds = nodes[0].bounds().clone();
        nodes
            .iter()
            .skip(1)
            .fold(first_bounds, |acc, node| acc.union(node.bounds()))
    }

    /// Range query: find all points within a rectangle
    pub fn range_query(&self, query_rect: &Rectangle) -> Vec<(Array1<Float>, usize)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.range_query_recursive(root, query_rect, &mut results);
        }
        results
    }

    /// Recursive range query
    fn range_query_recursive(
        &self,
        node: &RTreeNode,
        query_rect: &Rectangle,
        results: &mut Vec<(Array1<Float>, usize)>,
    ) {
        if !node.bounds().intersects(query_rect) {
            return;
        }

        match node {
            RTreeNode::Leaf { points, .. } => {
                for (point, index) in points {
                    if query_rect.contains_point(&point.view()) {
                        results.push((point.clone(), *index));
                    }
                }
            }
            RTreeNode::Internal { children, .. } => {
                for child in children {
                    self.range_query_recursive(child, query_rect, results);
                }
            }
        }
    }

    /// K-nearest neighbors query
    pub fn knn_query(
        &self,
        query_point: &ArrayView1<Float>,
        k: usize,
    ) -> Vec<(Array1<Float>, usize, Float)> {
        let mut candidates = Vec::new();
        if let Some(ref root) = self.root {
            self.knn_query_recursive(root, query_point, &mut candidates);
        }

        // Sort by distance and take k nearest
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        candidates.into_iter().take(k).collect()
    }

    /// Recursive k-nearest neighbors query
    fn knn_query_recursive(
        &self,
        node: &RTreeNode,
        query_point: &ArrayView1<Float>,
        candidates: &mut Vec<(Array1<Float>, usize, Float)>,
    ) {
        match node {
            RTreeNode::Leaf { points, .. } => {
                for (point, index) in points {
                    let distance = self.euclidean_distance(query_point, &point.view());
                    candidates.push((point.clone(), *index, distance));
                }
            }
            RTreeNode::Internal { children, .. } => {
                // Sort children by minimum distance to query point
                let mut child_distances: Vec<(usize, Float)> = children
                    .iter()
                    .enumerate()
                    .map(|(i, child)| (i, child.bounds().min_distance_to_point(query_point)))
                    .collect();
                child_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                for (child_idx, _) in child_distances {
                    self.knn_query_recursive(&children[child_idx], query_point, candidates);
                }
            }
        }
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<Float>()
            .sqrt()
    }
}

/// Point for quad-tree
#[derive(Clone, Debug)]
pub struct QuadPoint {
    /// Coordinates (x, y)
    pub coords: Array1<Float>,
    /// Data index
    pub index: usize,
}

/// Quad-tree node for 2D spatial indexing
#[derive(Clone, Debug)]
enum QuadTreeNode {
    /// Internal node with four children
    Internal {
        /// Bounding rectangle
        bounds: Rectangle,
        /// Four children: NW, NE, SW, SE
        children: [Option<Box<QuadTreeNode>>; 4],
    },
    /// Leaf node containing points
    Leaf {
        /// Bounding rectangle
        bounds: Rectangle,
        /// Points in this leaf
        points: Vec<QuadPoint>,
    },
}

/// Quad-tree for efficient 2D spatial indexing
pub struct QuadTree {
    /// Root node
    root: Option<QuadTreeNode>,
    /// Maximum points per leaf
    max_points: usize,
    /// Maximum depth
    max_depth: usize,
}

impl QuadTree {
    /// Create a new quad-tree
    pub fn new(bounds: Rectangle, max_points: usize, max_depth: usize) -> NeighborsResult<Self> {
        if bounds.min_bounds.len() != 2 || bounds.max_bounds.len() != 2 {
            return Err(NeighborsError::InvalidInput(
                "QuadTree requires 2D bounds".to_string(),
            ));
        }

        Ok(Self {
            root: Some(QuadTreeNode::Leaf {
                bounds,
                points: Vec::new(),
            }),
            max_points,
            max_depth,
        })
    }

    /// Insert a point into the quad-tree
    pub fn insert(&mut self, point: Array1<Float>, index: usize) -> NeighborsResult<()> {
        if point.len() != 2 {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![2],
                actual: vec![point.len()],
            });
        }

        let quad_point = QuadPoint {
            coords: point,
            index,
        };

        if let Some(root) = self.root.take() {
            self.root = Some(self.insert_recursive(root, quad_point, 0)?);
        }

        Ok(())
    }

    /// Recursive insertion
    fn insert_recursive(
        &self,
        node: QuadTreeNode,
        point: QuadPoint,
        depth: usize,
    ) -> NeighborsResult<QuadTreeNode> {
        match node {
            QuadTreeNode::Leaf { bounds, mut points } => {
                if !bounds.contains_point(&point.coords.view()) {
                    return Err(NeighborsError::InvalidInput(
                        "Point outside tree bounds".to_string(),
                    ));
                }

                points.push(point);

                if points.len() <= self.max_points || depth >= self.max_depth {
                    // Leaf not full or at max depth
                    Ok(QuadTreeNode::Leaf { bounds, points })
                } else {
                    // Split leaf
                    self.split_leaf(bounds, points, depth)
                }
            }
            QuadTreeNode::Internal {
                bounds,
                mut children,
            } => {
                let quadrant = self.get_quadrant(&bounds, &point.coords.view())?;

                if let Some(child) = children[quadrant].take() {
                    let new_child = self.insert_recursive(*child, point, depth + 1)?;
                    children[quadrant] = Some(Box::new(new_child));
                } else {
                    // Create new child
                    let child_bounds = self.get_quadrant_bounds(&bounds, quadrant);
                    let new_child = QuadTreeNode::Leaf {
                        bounds: child_bounds,
                        points: vec![point],
                    };
                    children[quadrant] = Some(Box::new(new_child));
                }

                Ok(QuadTreeNode::Internal { bounds, children })
            }
        }
    }

    /// Split a leaf node into four children
    fn split_leaf(
        &self,
        bounds: Rectangle,
        points: Vec<QuadPoint>,
        depth: usize,
    ) -> NeighborsResult<QuadTreeNode> {
        let mut children: [Option<Box<QuadTreeNode>>; 4] = [None, None, None, None];

        // Distribute points to children
        for point in points {
            let quadrant = self.get_quadrant(&bounds, &point.coords.view())?;

            if children[quadrant].is_none() {
                let child_bounds = self.get_quadrant_bounds(&bounds, quadrant);
                children[quadrant] = Some(Box::new(QuadTreeNode::Leaf {
                    bounds: child_bounds,
                    points: Vec::new(),
                }));
            }

            if let Some(ref mut child) = children[quadrant] {
                if let QuadTreeNode::Leaf { ref mut points, .. } = **child {
                    points.push(point);
                }
            }
        }

        Ok(QuadTreeNode::Internal { bounds, children })
    }

    /// Get the quadrant (0: NW, 1: NE, 2: SW, 3: SE) for a point
    fn get_quadrant(
        &self,
        bounds: &Rectangle,
        point: &ArrayView1<Float>,
    ) -> NeighborsResult<usize> {
        let center_x = (bounds.min_bounds[0] + bounds.max_bounds[0]) / 2.0;
        let center_y = (bounds.min_bounds[1] + bounds.max_bounds[1]) / 2.0;

        let x = point[0];
        let y = point[1];

        let quadrant = match (x >= center_x, y >= center_y) {
            (false, true) => 0,  // NW
            (true, true) => 1,   // NE
            (false, false) => 2, // SW
            (true, false) => 3,  // SE
        };

        Ok(quadrant)
    }

    /// Get bounds for a specific quadrant
    fn get_quadrant_bounds(&self, bounds: &Rectangle, quadrant: usize) -> Rectangle {
        let center_x = (bounds.min_bounds[0] + bounds.max_bounds[0]) / 2.0;
        let center_y = (bounds.min_bounds[1] + bounds.max_bounds[1]) / 2.0;

        let (min_x, max_x, min_y, max_y) = match quadrant {
            0 => (
                bounds.min_bounds[0],
                center_x,
                center_y,
                bounds.max_bounds[1],
            ), // NW
            1 => (
                center_x,
                bounds.max_bounds[0],
                center_y,
                bounds.max_bounds[1],
            ), // NE
            2 => (
                bounds.min_bounds[0],
                center_x,
                bounds.min_bounds[1],
                center_y,
            ), // SW
            3 => (
                center_x,
                bounds.max_bounds[0],
                bounds.min_bounds[1],
                center_y,
            ), // SE
            _ => unreachable!(),
        };

        Rectangle::new(
            Array1::from_vec(vec![min_x, min_y]),
            Array1::from_vec(vec![max_x, max_y]),
        )
        .unwrap()
    }

    /// Range query: find all points within a rectangle
    pub fn range_query(&self, query_rect: &Rectangle) -> Vec<QuadPoint> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.range_query_recursive(root, query_rect, &mut results);
        }
        results
    }

    /// Recursive range query
    fn range_query_recursive(
        &self,
        node: &QuadTreeNode,
        query_rect: &Rectangle,
        results: &mut Vec<QuadPoint>,
    ) {
        match node {
            QuadTreeNode::Leaf { bounds, points } => {
                if bounds.intersects(query_rect) {
                    for point in points {
                        if query_rect.contains_point(&point.coords.view()) {
                            results.push(point.clone());
                        }
                    }
                }
            }
            QuadTreeNode::Internal { bounds, children } => {
                if bounds.intersects(query_rect) {
                    for child in children.iter().flatten() {
                        self.range_query_recursive(child, query_rect, results);
                    }
                }
            }
        }
    }

    /// K-nearest neighbors query
    pub fn knn_query(&self, query_point: &ArrayView1<Float>, k: usize) -> Vec<(QuadPoint, Float)> {
        let mut candidates = Vec::new();
        if let Some(ref root) = self.root {
            self.knn_query_recursive(root, query_point, &mut candidates);
        }

        // Sort by distance and take k nearest
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.into_iter().take(k).collect()
    }

    /// Recursive k-nearest neighbors query
    fn knn_query_recursive(
        &self,
        node: &QuadTreeNode,
        query_point: &ArrayView1<Float>,
        candidates: &mut Vec<(QuadPoint, Float)>,
    ) {
        match node {
            QuadTreeNode::Leaf { points, .. } => {
                for point in points {
                    let distance = self.euclidean_distance(query_point, &point.coords.view());
                    candidates.push((point.clone(), distance));
                }
            }
            QuadTreeNode::Internal { bounds, children } => {
                // Sort children by minimum distance to query point
                let mut child_distances: Vec<(usize, Float)> = children
                    .iter()
                    .enumerate()
                    .filter_map(|(i, child_opt)| {
                        child_opt.as_ref().map(|child| {
                            let min_dist = match child.as_ref() {
                                QuadTreeNode::Leaf { bounds, .. }
                                | QuadTreeNode::Internal { bounds, .. } => {
                                    bounds.min_distance_to_point(query_point)
                                }
                            };
                            (i, min_dist)
                        })
                    })
                    .collect();

                child_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                for (child_idx, _) in child_distances {
                    if let Some(ref child) = children[child_idx] {
                        self.knn_query_recursive(child, query_point, candidates);
                    }
                }
            }
        }
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<Float>()
            .sqrt()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_rectangle_operations() {
        let rect1 = Rectangle::new(array![0.0, 0.0], array![2.0, 2.0]).unwrap();

        let rect2 = Rectangle::new(array![1.0, 1.0], array![3.0, 3.0]).unwrap();

        // Test intersection
        assert!(rect1.intersects(&rect2));

        // Test union
        let union = rect1.union(&rect2);
        assert_eq!(union.min_bounds, array![0.0, 0.0]);
        assert_eq!(union.max_bounds, array![3.0, 3.0]);

        // Test point containment
        let point = array![1.5, 1.5];
        assert!(rect1.contains_point(&point.view()));
        assert!(rect2.contains_point(&point.view()));
    }

    #[test]
    fn test_rtree_basic() {
        let mut rtree = RTree::new(4, 2);

        // Insert some points
        rtree.insert(array![1.0, 1.0], 0).unwrap();
        rtree.insert(array![2.0, 2.0], 1).unwrap();
        rtree.insert(array![3.0, 3.0], 2).unwrap();
        rtree.insert(array![4.0, 4.0], 3).unwrap();

        // Range query
        let query_rect = Rectangle::new(array![0.5, 0.5], array![2.5, 2.5]).unwrap();

        let results = rtree.range_query(&query_rect);
        assert_eq!(results.len(), 2); // Points 0 and 1 should be in range

        // KNN query
        let query_point = array![1.5, 1.5];
        let knn_results = rtree.knn_query(&query_point.view(), 2);
        assert_eq!(knn_results.len(), 2);
    }

    #[test]
    fn test_quadtree_basic() {
        let bounds = Rectangle::new(array![0.0, 0.0], array![10.0, 10.0]).unwrap();

        let mut qtree = QuadTree::new(bounds, 2, 5).unwrap();

        // Insert some points
        qtree.insert(array![1.0, 1.0], 0).unwrap();
        qtree.insert(array![2.0, 2.0], 1).unwrap();
        qtree.insert(array![8.0, 8.0], 2).unwrap();
        qtree.insert(array![9.0, 9.0], 3).unwrap();

        // Range query
        let query_rect = Rectangle::new(array![0.5, 0.5], array![2.5, 2.5]).unwrap();

        let results = qtree.range_query(&query_rect);
        assert_eq!(results.len(), 2);

        // KNN query
        let query_point = array![1.5, 1.5];
        let knn_results = qtree.knn_query(&query_point.view(), 2);
        assert_eq!(knn_results.len(), 2);
    }

    #[test]
    fn test_rectangle_distance_functions() {
        let rect = Rectangle::new(array![1.0, 1.0], array![3.0, 3.0]).unwrap();

        // Point inside rectangle
        let inside_point = array![2.0, 2.0];
        assert_abs_diff_eq!(
            rect.min_distance_to_point(&inside_point.view()),
            0.0,
            epsilon = 1e-10
        );

        // Point outside rectangle
        let outside_point = array![0.0, 0.0];
        let expected_distance = (1.0_f64.powi(2) + 1.0_f64.powi(2)).sqrt();
        assert_abs_diff_eq!(
            rect.min_distance_to_point(&outside_point.view()),
            expected_distance,
            epsilon = 1e-10
        );

        // Test area calculation
        assert_abs_diff_eq!(rect.area(), 4.0, epsilon = 1e-10);
    }
}

/// Point in 3D space for OctTree
#[derive(Clone, Debug)]
pub struct OctPoint {
    /// 3D coordinates
    pub coords: Array1<Float>,
    /// Index in original data
    pub index: usize,
}

impl OctPoint {
    /// Create a new 3D point
    pub fn new(x: Float, y: Float, z: Float, index: usize) -> Self {
        OctPoint {
            coords: Array1::from_vec(vec![x, y, z]),
            index,
        }
    }

    /// Create from array view
    pub fn from_coords(coords: &ArrayView1<Float>, index: usize) -> NeighborsResult<Self> {
        if coords.len() != 3 {
            return Err(NeighborsError::InvalidInput(
                "OctPoint requires exactly 3 dimensions".to_string(),
            ));
        }
        Ok(OctPoint {
            coords: coords.to_owned(),
            index,
        })
    }
}

/// Node in an OctTree
#[derive(Clone, Debug)]
pub enum OctTreeNode {
    /// Internal node with 8 children (octants)
    Internal {
        /// Bounding box for this node
        bounds: Rectangle,
        /// Children nodes (8 octants)
        /// Ordering: [NWU, NEU, SWU, SEU, NWD, NED, SWD, SED]
        /// N=North, S=South, W=West, E=East, U=Up, D=Down
        children: [Option<Box<OctTreeNode>>; 8],
    },
    /// Leaf node containing points
    Leaf {
        /// Bounding box for this node
        bounds: Rectangle,
        /// Points in this leaf
        points: Vec<OctPoint>,
    },
}

impl OctTreeNode {
    /// Get the bounding box of this node
    pub fn bounds(&self) -> &Rectangle {
        match self {
            OctTreeNode::Internal { bounds, .. } => bounds,
            OctTreeNode::Leaf { bounds, .. } => bounds,
        }
    }
}

/// OctTree for efficient 3D spatial indexing
#[derive(Clone, Debug)]
pub struct OctTree {
    /// Root node of the tree
    root: Option<Box<OctTreeNode>>,
    /// Maximum number of points per leaf node
    max_points_per_leaf: usize,
    /// Maximum depth of the tree
    max_depth: usize,
}

impl OctTree {
    /// Create a new OctTree
    pub fn new(max_points_per_leaf: usize, max_depth: usize) -> Self {
        OctTree {
            root: None,
            max_points_per_leaf,
            max_depth,
        }
    }

    /// Insert a batch of points into the octree
    pub fn insert_batch(&mut self, data: &ArrayView2<Float>) -> NeighborsResult<()> {
        if data.ncols() != 3 {
            return Err(NeighborsError::InvalidInput(
                "OctTree requires exactly 3 dimensions".to_string(),
            ));
        }

        if data.is_empty() {
            return Ok(());
        }

        // Calculate overall bounds
        let mut min_bounds = Array1::from_elem(3, Float::INFINITY);
        let mut max_bounds = Array1::from_elem(3, Float::NEG_INFINITY);

        for row in data.axis_iter(scirs2_core::ndarray::Axis(0)) {
            for (i, &val) in row.iter().enumerate() {
                min_bounds[i] = min_bounds[i].min(val);
                max_bounds[i] = max_bounds[i].max(val);
            }
        }

        // Add small padding to avoid edge cases
        let padding = 0.0001;
        for i in 0..3 {
            min_bounds[i] -= padding;
            max_bounds[i] += padding;
        }

        let bounds = Rectangle::new(min_bounds, max_bounds)?;

        // Create points
        let mut points = Vec::new();
        for (index, row) in data.axis_iter(scirs2_core::ndarray::Axis(0)).enumerate() {
            points.push(OctPoint::from_coords(&row, index)?);
        }

        // Build the tree
        self.root = Some(Box::new(self.build_tree(points, bounds, 0)?));
        Ok(())
    }

    /// Build the tree recursively
    fn build_tree(
        &self,
        points: Vec<OctPoint>,
        bounds: Rectangle,
        depth: usize,
    ) -> NeighborsResult<OctTreeNode> {
        // Base case: create leaf if we have few points or reached max depth
        if points.len() <= self.max_points_per_leaf || depth >= self.max_depth {
            return Ok(OctTreeNode::Leaf { bounds, points });
        }

        // Create internal node with 8 children
        let mut children: [Option<Box<OctTreeNode>>; 8] = Default::default();

        // Group points by octant
        let mut octant_points: [Vec<OctPoint>; 8] = Default::default();

        for point in points {
            let octant = self.get_octant(&bounds, &point.coords.view())?;
            octant_points[octant].push(point);
        }

        // Recursively build children for non-empty octants
        for (octant, points) in octant_points.into_iter().enumerate() {
            if !points.is_empty() {
                let octant_bounds = self.get_octant_bounds(&bounds, octant);
                children[octant] = Some(Box::new(self.build_tree(
                    points,
                    octant_bounds,
                    depth + 1,
                )?));
            }
        }

        Ok(OctTreeNode::Internal { bounds, children })
    }

    /// Range query: find all points within a 3D bounding box
    pub fn range_query(&self, query_bounds: &Rectangle) -> Vec<(Array1<Float>, usize)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.range_query_recursive(root, query_bounds, &mut results);
        }
        results
    }

    /// Recursive range query
    fn range_query_recursive(
        &self,
        node: &OctTreeNode,
        query_bounds: &Rectangle,
        results: &mut Vec<(Array1<Float>, usize)>,
    ) {
        if !node.bounds().intersects(query_bounds) {
            return;
        }

        match node {
            OctTreeNode::Leaf { points, .. } => {
                for point in points {
                    if query_bounds.contains_point(&point.coords.view()) {
                        results.push((point.coords.clone(), point.index));
                    }
                }
            }
            OctTreeNode::Internal { children, .. } => {
                for child in children.iter().flatten() {
                    self.range_query_recursive(child, query_bounds, results);
                }
            }
        }
    }

    /// K-nearest neighbors query
    pub fn knn_query(
        &self,
        query_point: &ArrayView1<Float>,
        k: usize,
    ) -> Vec<(Array1<Float>, usize, Float)> {
        if query_point.len() != 3 {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        if let Some(ref root) = self.root {
            self.knn_query_recursive(root, query_point, &mut candidates);
        }

        // Sort by distance and take k nearest
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        candidates.into_iter().take(k).collect()
    }

    /// Recursive k-nearest neighbors query
    fn knn_query_recursive(
        &self,
        node: &OctTreeNode,
        query_point: &ArrayView1<Float>,
        candidates: &mut Vec<(Array1<Float>, usize, Float)>,
    ) {
        match node {
            OctTreeNode::Leaf { points, .. } => {
                for point in points {
                    let distance = self.euclidean_distance(query_point, &point.coords.view());
                    candidates.push((point.coords.clone(), point.index, distance));
                }
            }
            OctTreeNode::Internal { children, .. } => {
                // Sort children by minimum distance to query point
                let mut child_distances: Vec<(usize, Float)> = children
                    .iter()
                    .enumerate()
                    .filter_map(|(i, child)| {
                        child
                            .as_ref()
                            .map(|c| (i, c.bounds().min_distance_to_point(query_point)))
                    })
                    .collect();
                child_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                // Visit children in order of increasing distance
                for (child_idx, _) in child_distances {
                    if let Some(ref child) = children[child_idx] {
                        self.knn_query_recursive(child, query_point, candidates);
                    }
                }
            }
        }
    }

    /// Get the octant (0-7) for a point in 3D space
    /// Ordering: [NWU, NEU, SWU, SEU, NWD, NED, SWD, SED]
    fn get_octant(&self, bounds: &Rectangle, point: &ArrayView1<Float>) -> NeighborsResult<usize> {
        let center_x = (bounds.min_bounds[0] + bounds.max_bounds[0]) / 2.0;
        let center_y = (bounds.min_bounds[1] + bounds.max_bounds[1]) / 2.0;
        let center_z = (bounds.min_bounds[2] + bounds.max_bounds[2]) / 2.0;

        let x = point[0];
        let y = point[1];
        let z = point[2];

        let octant = match (x >= center_x, y >= center_y, z >= center_z) {
            (false, true, true) => 0,   // NWU (North-West-Up)
            (true, true, true) => 1,    // NEU (North-East-Up)
            (false, false, true) => 2,  // SWU (South-West-Up)
            (true, false, true) => 3,   // SEU (South-East-Up)
            (false, true, false) => 4,  // NWD (North-West-Down)
            (true, true, false) => 5,   // NED (North-East-Down)
            (false, false, false) => 6, // SWD (South-West-Down)
            (true, false, false) => 7,  // SED (South-East-Down)
        };

        Ok(octant)
    }

    /// Get bounds for a specific octant
    fn get_octant_bounds(&self, bounds: &Rectangle, octant: usize) -> Rectangle {
        let center_x = (bounds.min_bounds[0] + bounds.max_bounds[0]) / 2.0;
        let center_y = (bounds.min_bounds[1] + bounds.max_bounds[1]) / 2.0;
        let center_z = (bounds.min_bounds[2] + bounds.max_bounds[2]) / 2.0;

        let (min_x, max_x, min_y, max_y, min_z, max_z) = match octant {
            0 => (
                bounds.min_bounds[0],
                center_x,
                center_y,
                bounds.max_bounds[1],
                center_z,
                bounds.max_bounds[2],
            ), // NWU
            1 => (
                center_x,
                bounds.max_bounds[0],
                center_y,
                bounds.max_bounds[1],
                center_z,
                bounds.max_bounds[2],
            ), // NEU
            2 => (
                bounds.min_bounds[0],
                center_x,
                bounds.min_bounds[1],
                center_y,
                center_z,
                bounds.max_bounds[2],
            ), // SWU
            3 => (
                center_x,
                bounds.max_bounds[0],
                bounds.min_bounds[1],
                center_y,
                center_z,
                bounds.max_bounds[2],
            ), // SEU
            4 => (
                bounds.min_bounds[0],
                center_x,
                center_y,
                bounds.max_bounds[1],
                bounds.min_bounds[2],
                center_z,
            ), // NWD
            5 => (
                center_x,
                bounds.max_bounds[0],
                center_y,
                bounds.max_bounds[1],
                bounds.min_bounds[2],
                center_z,
            ), // NED
            6 => (
                bounds.min_bounds[0],
                center_x,
                bounds.min_bounds[1],
                center_y,
                bounds.min_bounds[2],
                center_z,
            ), // SWD
            7 => (
                center_x,
                bounds.max_bounds[0],
                bounds.min_bounds[1],
                center_y,
                bounds.min_bounds[2],
                center_z,
            ), // SED
            _ => unreachable!(),
        };

        Rectangle::new(
            Array1::from_vec(vec![min_x, min_y, min_z]),
            Array1::from_vec(vec![max_x, max_y, max_z]),
        )
        .unwrap()
    }

    /// Calculate Euclidean distance between two 3D points
    fn euclidean_distance(&self, p1: &ArrayView1<Float>, p2: &ArrayView1<Float>) -> Float {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<Float>()
            .sqrt()
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Get the height of the tree
    pub fn height(&self) -> usize {
        match &self.root {
            Some(root) => self.node_height(root),
            None => 0,
        }
    }

    /// Calculate height of a subtree
    fn node_height(&self, node: &OctTreeNode) -> usize {
        match node {
            OctTreeNode::Leaf { .. } => 1,
            OctTreeNode::Internal { children, .. } => {
                1 + children
                    .iter()
                    .filter_map(|child| child.as_ref())
                    .map(|child| self.node_height(child))
                    .max()
                    .unwrap_or(0)
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod oct_tree_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_octree_construction() {
        let data = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0,
                1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            ],
        )
        .unwrap();

        let mut octree = OctTree::new(4, 5);
        octree.insert_batch(&data.view()).unwrap();

        assert!(!octree.is_empty());
        assert!(octree.height() > 0);
    }

    #[test]
    fn test_octree_range_query() {
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 0.5, 0.5, 0.5, 10.0, 10.0, 10.0,
            ],
        )
        .unwrap();

        let mut octree = OctTree::new(2, 5);
        octree.insert_batch(&data.view()).unwrap();

        // Query for points in range [0, 2.5] for all dimensions
        let query_bounds = Rectangle::new(
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![2.5, 2.5, 2.5]),
        )
        .unwrap();

        let results = octree.range_query(&query_bounds);
        assert_eq!(results.len(), 3); // Should find points 0, 1, 3
    }

    #[test]
    fn test_octree_knn_query() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 10.0, 10.0, 10.0, 1.1, 1.1, 1.1,
            ],
        )
        .unwrap();

        let mut octree = OctTree::new(2, 5);
        octree.insert_batch(&data.view()).unwrap();

        let query_point = array![1.0, 1.0, 1.0];
        let results = octree.knn_query(&query_point.view(), 2);

        assert_eq!(results.len(), 2);
        // First result should be the closest point (index 0)
        assert_eq!(results[0].1, 0);
        assert_abs_diff_eq!(results[0].2, 0.0, epsilon = 1e-10);

        // Second result should be index 3 (1.1, 1.1, 1.1)
        assert_eq!(results[1].1, 3);
    }

    #[test]
    fn test_octree_empty_input() {
        let mut octree = OctTree::new(4, 5);
        let empty_data = Array2::zeros((0, 3));

        let result = octree.insert_batch(&empty_data.view());
        assert!(result.is_ok());
        assert!(octree.is_empty());
    }

    #[test]
    fn test_octree_invalid_dimensions() {
        let mut octree = OctTree::new(4, 5);
        let invalid_data = Array2::zeros((5, 2)); // Wrong number of dimensions

        let result = octree.insert_batch(&invalid_data.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_octpoint_creation() {
        let point = OctPoint::new(1.0, 2.0, 3.0, 5);
        assert_eq!(point.coords[0], 1.0);
        assert_eq!(point.coords[1], 2.0);
        assert_eq!(point.coords[2], 3.0);
        assert_eq!(point.index, 5);

        let coords = array![4.0, 5.0, 6.0];
        let point2 = OctPoint::from_coords(&coords.view(), 10).unwrap();
        assert_eq!(point2.coords[0], 4.0);
        assert_eq!(point2.coords[1], 5.0);
        assert_eq!(point2.coords[2], 6.0);
        assert_eq!(point2.index, 10);
    }
}

/// Spatial hashing for approximate nearest neighbor search
///
/// Spatial hashing divides space into a grid and hashes points into buckets
/// based on their spatial location. This enables fast approximate neighbor
/// queries by only searching relevant buckets.
#[derive(Clone, Debug)]
pub struct SpatialHash {
    /// Hash table mapping grid coordinates to point indices
    hash_table: std::collections::HashMap<Vec<i32>, Vec<usize>>,
    /// Grid cell size for each dimension
    cell_size: Array1<Float>,
    /// Minimum bounds of the data space
    min_bounds: Array1<Float>,
    /// Maximum bounds of the data space  
    max_bounds: Array1<Float>,
    /// Original data points
    points: Array2<Float>,
    /// Number of dimensions
    dimensions: usize,
}

impl SpatialHash {
    /// Create a new spatial hash
    pub fn new(cell_size: Array1<Float>) -> NeighborsResult<Self> {
        if cell_size.iter().any(|&s| s <= 0.0) {
            return Err(NeighborsError::InvalidInput(
                "Cell size must be positive for all dimensions".to_string(),
            ));
        }

        Ok(SpatialHash {
            hash_table: std::collections::HashMap::new(),
            cell_size,
            min_bounds: Array1::zeros(0),
            max_bounds: Array1::zeros(0),
            points: Array2::zeros((0, 0)),
            dimensions: 0,
        })
    }

    /// Create spatial hash with automatic cell size based on data
    pub fn from_data(
        data: &ArrayView2<Float>,
        target_points_per_cell: usize,
    ) -> NeighborsResult<Self> {
        if data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let n_points = data.nrows();
        let dimensions = data.ncols();

        // Calculate bounds
        let mut min_bounds = Array1::from_elem(dimensions, Float::INFINITY);
        let mut max_bounds = Array1::from_elem(dimensions, Float::NEG_INFINITY);

        for row in data.axis_iter(scirs2_core::ndarray::Axis(0)) {
            for (i, &val) in row.iter().enumerate() {
                min_bounds[i] = min_bounds[i].min(val);
                max_bounds[i] = max_bounds[i].max(val);
            }
        }

        // Calculate cell size to achieve target points per cell
        let total_volume = min_bounds
            .iter()
            .zip(max_bounds.iter())
            .map(|(min, max)| max - min)
            .product::<Float>();

        let target_cells = (n_points as Float / target_points_per_cell as Float).max(1.0);
        let cell_volume = total_volume / target_cells;
        let cell_size_scalar = cell_volume.powf(1.0 / dimensions as Float);

        let cell_size = Array1::from_elem(dimensions, cell_size_scalar.max(0.001));

        let mut hash = SpatialHash::new(cell_size)?;
        hash.insert_batch(data)?;
        Ok(hash)
    }

    /// Insert a batch of points into the spatial hash
    pub fn insert_batch(&mut self, data: &ArrayView2<Float>) -> NeighborsResult<()> {
        if data.is_empty() {
            return Ok(());
        }

        self.dimensions = data.ncols();
        self.points = data.to_owned();

        // Update cell size if it doesn't match dimensions
        if self.cell_size.len() != self.dimensions {
            let default_size = 1.0;
            self.cell_size = Array1::from_elem(self.dimensions, default_size);
        }

        // Calculate bounds
        self.min_bounds = Array1::from_elem(self.dimensions, Float::INFINITY);
        self.max_bounds = Array1::from_elem(self.dimensions, Float::NEG_INFINITY);

        for row in data.axis_iter(scirs2_core::ndarray::Axis(0)) {
            for (i, &val) in row.iter().enumerate() {
                self.min_bounds[i] = self.min_bounds[i].min(val);
                self.max_bounds[i] = self.max_bounds[i].max(val);
            }
        }

        // Clear existing hash table
        self.hash_table.clear();

        // Insert all points
        for (point_idx, row) in data.axis_iter(scirs2_core::ndarray::Axis(0)).enumerate() {
            let grid_coords = self.point_to_grid_coords(&row);
            self.hash_table
                .entry(grid_coords)
                .or_default()
                .push(point_idx);
        }

        Ok(())
    }

    /// Find approximate nearest neighbors using spatial hashing
    pub fn approximate_neighbors(
        &self,
        query_point: &ArrayView1<Float>,
        k: usize,
        search_radius: Option<usize>,
    ) -> NeighborsResult<Vec<(usize, Float)>> {
        if query_point.len() != self.dimensions {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![self.dimensions],
                actual: vec![query_point.len()],
            });
        }

        let query_grid_coords = self.point_to_grid_coords(query_point);
        let search_radius = search_radius.unwrap_or(1);

        let mut candidates = Vec::new();

        // Search in neighboring cells
        self.search_neighboring_cells(&query_grid_coords, search_radius as i32, &mut candidates);

        // Calculate distances and sort
        let mut distances: Vec<(usize, Float)> = candidates
            .into_iter()
            .map(|point_idx| {
                let point = self.points.row(point_idx);
                let distance = self.euclidean_distance(query_point, &point);
                (point_idx, distance)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        Ok(distances)
    }

    /// Find all points within a given radius using spatial hashing
    pub fn radius_neighbors(
        &self,
        query_point: &ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<Vec<(usize, Float)>> {
        if query_point.len() != self.dimensions {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![self.dimensions],
                actual: vec![query_point.len()],
            });
        }

        let query_grid_coords = self.point_to_grid_coords(query_point);

        // Calculate search radius in grid cells
        let cell_radius = (radius
            / self
                .cell_size
                .iter()
                .fold(Float::INFINITY, |a, &b| a.min(b)))
        .ceil() as i32;

        let mut candidates = Vec::new();
        self.search_neighboring_cells(&query_grid_coords, cell_radius, &mut candidates);

        // Filter by actual distance
        let mut neighbors: Vec<(usize, Float)> = candidates
            .into_iter()
            .map(|point_idx| {
                let point = self.points.row(point_idx);
                let distance = self.euclidean_distance(query_point, &point);
                (point_idx, distance)
            })
            .filter(|(_, distance)| *distance <= radius)
            .collect();

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(neighbors)
    }

    /// Convert a point to grid coordinates
    fn point_to_grid_coords(&self, point: &ArrayView1<Float>) -> Vec<i32> {
        point
            .iter()
            .zip(self.min_bounds.iter())
            .zip(self.cell_size.iter())
            .map(|((&coord, &min_bound), &cell_size)| {
                ((coord - min_bound) / cell_size).floor() as i32
            })
            .collect()
    }

    /// Search in neighboring cells within a given radius
    fn search_neighboring_cells(
        &self,
        center_coords: &[i32],
        radius: i32,
        candidates: &mut Vec<usize>,
    ) {
        // Generate all combinations of offsets in the given radius
        let mut offsets = vec![vec![]];

        for dim in 0..self.dimensions {
            let mut new_offsets = Vec::new();
            for offset in offsets {
                for delta in -radius..=radius {
                    let mut new_offset = offset.clone();
                    new_offset.push(delta);
                    new_offsets.push(new_offset);
                }
            }
            offsets = new_offsets;
        }

        // Search each neighboring cell
        for offset in offsets {
            let coords: Vec<i32> = center_coords
                .iter()
                .zip(offset.iter())
                .map(|(&center, &delta)| center + delta)
                .collect();

            if let Some(point_indices) = self.hash_table.get(&coords) {
                candidates.extend(point_indices.iter());
            }
        }
    }

    /// Calculate Euclidean distance between two points
    fn euclidean_distance(&self, p1: &ArrayView1<Float>, p2: &ArrayView1<Float>) -> Float {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<Float>()
            .sqrt()
    }

    /// Get the number of non-empty cells
    pub fn num_cells(&self) -> usize {
        self.hash_table.len()
    }

    /// Get the average number of points per cell
    pub fn avg_points_per_cell(&self) -> Float {
        if self.hash_table.is_empty() {
            0.0
        } else {
            self.points.nrows() as Float / self.hash_table.len() as Float
        }
    }

    /// Get statistics about the spatial hash
    pub fn get_stats(&self) -> SpatialHashStats {
        let mut cell_sizes = Vec::new();
        for points in self.hash_table.values() {
            cell_sizes.push(points.len());
        }

        let min_cell_size = cell_sizes.iter().min().copied().unwrap_or(0);
        let max_cell_size = cell_sizes.iter().max().copied().unwrap_or(0);
        let avg_cell_size = if cell_sizes.is_empty() {
            0.0
        } else {
            cell_sizes.iter().sum::<usize>() as Float / cell_sizes.len() as Float
        };

        SpatialHashStats {
            num_cells: self.hash_table.len(),
            total_points: self.points.nrows(),
            min_points_per_cell: min_cell_size,
            max_points_per_cell: max_cell_size,
            avg_points_per_cell: avg_cell_size,
            cell_size: self.cell_size.clone(),
        }
    }
}

/// Statistics about a spatial hash
#[derive(Clone, Debug)]
pub struct SpatialHashStats {
    /// Number of non-empty cells
    pub num_cells: usize,
    /// Total number of points
    pub total_points: usize,
    /// Minimum points in any cell
    pub min_points_per_cell: usize,
    /// Maximum points in any cell
    pub max_points_per_cell: usize,
    /// Average points per cell
    pub avg_points_per_cell: Float,
    /// Cell size for each dimension
    pub cell_size: Array1<Float>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod spatial_hash_tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_spatial_hash_creation() {
        let cell_size = Array1::from_vec(vec![1.0, 1.0]);
        let hash = SpatialHash::new(cell_size).unwrap();
        assert_eq!(hash.num_cells(), 0);
    }

    #[test]
    fn test_spatial_hash_invalid_cell_size() {
        let cell_size = Array1::from_vec(vec![0.0, 1.0]); // Invalid: contains zero
        let result = SpatialHash::new(cell_size);
        assert!(result.is_err());
    }

    #[test]
    fn test_spatial_hash_from_data() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1, 10.0, 10.0, 10.1, 10.1,
            ],
        )
        .unwrap();

        let hash = SpatialHash::from_data(&data.view(), 2).unwrap();
        assert!(hash.num_cells() > 0);
        assert_eq!(hash.points.nrows(), 6);
    }

    #[test]
    fn test_spatial_hash_approximate_neighbors() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 10.0, 10.0, 10.1, 10.1],
        )
        .unwrap();

        let hash = SpatialHash::from_data(&data.view(), 2).unwrap();

        let query = array![1.05, 1.05];
        let neighbors = hash
            .approximate_neighbors(&query.view(), 2, Some(1))
            .unwrap();

        assert!(neighbors.len() >= 1);
        // Should find nearby points (indices 0 and 1)
        assert!(neighbors[0].1 < 0.1); // Distance should be small
    }

    #[test]
    fn test_spatial_hash_radius_neighbors() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.2, 1.2, 5.0, 5.0, 10.0, 10.0]).unwrap();

        let hash = SpatialHash::from_data(&data.view(), 2).unwrap();

        let query = array![1.1, 1.1];
        let neighbors = hash.radius_neighbors(&query.view(), 0.5).unwrap();

        // Should find points within radius 0.5
        assert!(neighbors.len() >= 1);
        for (_, distance) in &neighbors {
            assert!(*distance <= 0.5);
        }
    }

    #[test]
    fn test_spatial_hash_3d() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0,
            ],
        )
        .unwrap();

        let hash = SpatialHash::from_data(&data.view(), 2).unwrap();

        let query = array![1.05, 1.05, 1.05];
        let neighbors = hash
            .approximate_neighbors(&query.view(), 2, Some(1))
            .unwrap();

        assert!(neighbors.len() >= 1);
    }

    #[test]
    fn test_spatial_hash_stats() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1, 10.0, 10.0, 10.1, 10.1,
            ],
        )
        .unwrap();

        let hash = SpatialHash::from_data(&data.view(), 2).unwrap();
        let stats = hash.get_stats();

        assert_eq!(stats.total_points, 6);
        assert!(stats.num_cells > 0);
        assert!(stats.avg_points_per_cell > 0.0);
    }

    #[test]
    fn test_spatial_hash_empty_input() {
        let empty_data = Array2::zeros((0, 2));
        let result = SpatialHash::from_data(&empty_data.view(), 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_spatial_hash_dimension_mismatch() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();

        let hash = SpatialHash::from_data(&data.view(), 2).unwrap();

        let query_3d = array![1.0, 1.0, 1.0]; // Wrong dimensions
        let result = hash.approximate_neighbors(&query_3d.view(), 1, Some(1));
        assert!(result.is_err());
    }
}

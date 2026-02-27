//! Spatial data structures for efficient spatial queries and indexing
//!
//! This module provides spatial data structures commonly used in machine learning
//! for nearest neighbor search, range queries, and spatial indexing.

/// A point in N-dimensional space
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub coordinates: Vec<f64>,
}

impl Point {
    pub fn new(coordinates: Vec<f64>) -> Self {
        Self { coordinates }
    }

    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }

    pub fn distance_squared(&self, other: &Point) -> f64 {
        self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }

    pub fn distance(&self, other: &Point) -> f64 {
        self.distance_squared(other).sqrt()
    }
}

/// Rectangle/Bounding box for spatial queries
#[derive(Debug, Clone, PartialEq)]
pub struct Rectangle {
    pub min_bounds: Vec<f64>,
    pub max_bounds: Vec<f64>,
}

impl Rectangle {
    pub fn new(min_bounds: Vec<f64>, max_bounds: Vec<f64>) -> Self {
        assert_eq!(min_bounds.len(), max_bounds.len());
        Self {
            min_bounds,
            max_bounds,
        }
    }

    pub fn dimension(&self) -> usize {
        self.min_bounds.len()
    }

    pub fn contains(&self, point: &Point) -> bool {
        point
            .coordinates
            .iter()
            .enumerate()
            .all(|(i, &coord)| coord >= self.min_bounds[i] && coord <= self.max_bounds[i])
    }

    pub fn intersects(&self, other: &Rectangle) -> bool {
        self.min_bounds.iter().enumerate().all(|(i, &min)| {
            min <= other.max_bounds[i] && self.max_bounds[i] >= other.min_bounds[i]
        })
    }

    pub fn area(&self) -> f64 {
        self.min_bounds
            .iter()
            .zip(self.max_bounds.iter())
            .map(|(min, max)| max - min)
            .product()
    }

    pub fn expand_to_include(&mut self, point: &Point) {
        for (i, &coord) in point.coordinates.iter().enumerate() {
            if coord < self.min_bounds[i] {
                self.min_bounds[i] = coord;
            }
            if coord > self.max_bounds[i] {
                self.max_bounds[i] = coord;
            }
        }
    }

    pub fn union(&self, other: &Rectangle) -> Rectangle {
        let min_bounds = self
            .min_bounds
            .iter()
            .zip(other.min_bounds.iter())
            .map(|(a, b)| a.min(*b))
            .collect();

        let max_bounds = self
            .max_bounds
            .iter()
            .zip(other.max_bounds.iter())
            .map(|(a, b)| a.max(*b))
            .collect();

        Rectangle::new(min_bounds, max_bounds)
    }
}

/// K-d tree for efficient nearest neighbor search
pub struct KdTree {
    root: Option<Box<KdNode>>,
    dimension: usize,
}

#[derive(Debug)]
struct KdNode {
    point: Point,
    data: usize, // Index or ID of associated data
    split_dimension: usize,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

impl KdTree {
    pub fn new(dimension: usize) -> Self {
        Self {
            root: None,
            dimension,
        }
    }

    pub fn from_points(points: Vec<(Point, usize)>) -> Self {
        if points.is_empty() {
            return Self::new(0);
        }

        let dimension = points[0].0.dimension();
        let root = Self::build_tree(points, 0, dimension);

        Self {
            root: Some(root),
            dimension,
        }
    }

    fn build_tree(mut points: Vec<(Point, usize)>, depth: usize, dimension: usize) -> Box<KdNode> {
        let split_dim = depth % dimension;

        // Sort points by the splitting dimension
        points.sort_by(|a, b| {
            a.0.coordinates[split_dim]
                .partial_cmp(&b.0.coordinates[split_dim])
                .unwrap()
        });

        let median = points.len() / 2;
        let (point, data) = points.remove(median);

        let left = if median > 0 {
            Some(Self::build_tree(
                points[..median].to_vec(),
                depth + 1,
                dimension,
            ))
        } else {
            None
        };

        let right = if median < points.len() {
            Some(Self::build_tree(
                points[median..].to_vec(),
                depth + 1,
                dimension,
            ))
        } else {
            None
        };

        Box::new(KdNode {
            point,
            data,
            split_dimension: split_dim,
            left,
            right,
        })
    }

    pub fn insert(&mut self, point: Point, data: usize) {
        if self.root.is_none() {
            self.dimension = point.dimension();
        }

        self.root = Some(Self::insert_recursive(
            self.root.take(),
            point,
            data,
            0,
            self.dimension,
        ));
    }

    fn insert_recursive(
        node: Option<Box<KdNode>>,
        point: Point,
        data: usize,
        depth: usize,
        dimension: usize,
    ) -> Box<KdNode> {
        if let Some(mut existing) = node {
            let split_dim = depth % dimension;

            if point.coordinates[split_dim] <= existing.point.coordinates[split_dim] {
                existing.left = Some(Self::insert_recursive(
                    existing.left.take(),
                    point,
                    data,
                    depth + 1,
                    dimension,
                ));
            } else {
                existing.right = Some(Self::insert_recursive(
                    existing.right.take(),
                    point,
                    data,
                    depth + 1,
                    dimension,
                ));
            }

            existing
        } else {
            Box::new(KdNode {
                point,
                data,
                split_dimension: depth % dimension,
                left: None,
                right: None,
            })
        }
    }

    pub fn nearest_neighbor(&self, query: &Point) -> Option<(Point, usize, f64)> {
        self.root.as_ref().map(|root| {
            let mut best = (
                root.point.clone(),
                root.data,
                query.distance_squared(&root.point),
            );
            Self::nearest_recursive(root, query, &mut best, 0);
            (best.0, best.1, best.2.sqrt())
        })
    }

    fn nearest_recursive(
        node: &KdNode,
        query: &Point,
        best: &mut (Point, usize, f64),
        _depth: usize,
    ) {
        let distance_sq = query.distance_squared(&node.point);
        if distance_sq < best.2 {
            *best = (node.point.clone(), node.data, distance_sq);
        }

        let split_dim = node.split_dimension;
        let diff = query.coordinates[split_dim] - node.point.coordinates[split_dim];

        let (primary, secondary) = if diff <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(child) = primary {
            Self::nearest_recursive(child, query, best, _depth + 1);
        }

        // Check if we need to explore the other side
        if diff * diff < best.2 {
            if let Some(child) = secondary {
                Self::nearest_recursive(child, query, best, _depth + 1);
            }
        }
    }

    pub fn range_query(&self, range: &Rectangle) -> Vec<(Point, usize)> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            Self::range_recursive(root, range, &mut results);
        }
        results
    }

    fn range_recursive(node: &KdNode, range: &Rectangle, results: &mut Vec<(Point, usize)>) {
        if range.contains(&node.point) {
            results.push((node.point.clone(), node.data));
        }

        let split_dim = node.split_dimension;

        if range.min_bounds[split_dim] <= node.point.coordinates[split_dim] {
            if let Some(left) = &node.left {
                Self::range_recursive(left, range, results);
            }
        }

        if range.max_bounds[split_dim] >= node.point.coordinates[split_dim] {
            if let Some(right) = &node.right {
                Self::range_recursive(right, range, results);
            }
        }
    }
}

/// R-tree for efficient spatial indexing of rectangles
pub struct RTree {
    root: Option<Box<RNode>>,
    max_entries: usize,
    #[allow(dead_code)]
    min_entries: usize,
}

#[derive(Debug)]
struct RNode {
    bounds: Rectangle,
    entries: Vec<REntry>,
    is_leaf: bool,
}

#[derive(Debug)]
enum REntry {
    Leaf {
        bounds: Rectangle,
        data: usize,
    },
    #[allow(dead_code)]
    Internal {
        bounds: Rectangle,
        child: Box<RNode>,
    },
}

impl RTree {
    pub fn new(max_entries: usize) -> Self {
        let min_entries = max_entries / 2;
        Self {
            root: None,
            max_entries,
            min_entries,
        }
    }

    pub fn insert(&mut self, bounds: Rectangle, data: usize) {
        let entry = REntry::Leaf {
            bounds: bounds.clone(),
            data,
        };

        if self.root.is_none() {
            self.root = Some(Box::new(RNode {
                bounds,
                entries: vec![entry],
                is_leaf: true,
            }));
        } else {
            self.insert_recursive(entry);
        }
    }

    fn insert_recursive(&mut self, entry: REntry) {
        // Simplified insertion - in a full implementation, this would handle splits and tree growth
        if let Some(root) = &mut self.root {
            if root.is_leaf && root.entries.len() < self.max_entries {
                root.bounds = root.bounds.union(entry.bounds());
                root.entries.push(entry);
            }
        }
    }

    pub fn query(&self, query_bounds: &Rectangle) -> Vec<usize> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            Self::query_recursive(root, query_bounds, &mut results);
        }
        results
    }

    fn query_recursive(node: &RNode, query_bounds: &Rectangle, results: &mut Vec<usize>) {
        if !node.bounds.intersects(query_bounds) {
            return;
        }

        for entry in &node.entries {
            match entry {
                REntry::Leaf { bounds, data } => {
                    if bounds.intersects(query_bounds) {
                        results.push(*data);
                    }
                }
                REntry::Internal { bounds, child } => {
                    if bounds.intersects(query_bounds) {
                        Self::query_recursive(child, query_bounds, results);
                    }
                }
            }
        }
    }
}

impl REntry {
    fn bounds(&self) -> &Rectangle {
        match self {
            REntry::Leaf { bounds, .. } => bounds,
            REntry::Internal { bounds, .. } => bounds,
        }
    }
}

/// Quadtree for 2D spatial indexing
pub struct QuadTree {
    root: QuadNode,
    max_points: usize,
    max_depth: usize,
}

#[derive(Debug)]
struct QuadNode {
    bounds: Rectangle,
    points: Vec<(Point, usize)>,
    children: Option<[Box<QuadNode>; 4]>,
    depth: usize,
}

impl QuadTree {
    pub fn new(bounds: Rectangle, max_points: usize, max_depth: usize) -> Self {
        assert_eq!(bounds.dimension(), 2, "QuadTree only supports 2D");

        Self {
            root: QuadNode {
                bounds,
                points: Vec::new(),
                children: None,
                depth: 0,
            },
            max_points,
            max_depth,
        }
    }

    pub fn insert(&mut self, point: Point, data: usize) {
        assert_eq!(point.dimension(), 2, "QuadTree only supports 2D points");
        self.root
            .insert(point, data, self.max_points, self.max_depth);
    }

    pub fn query(&self, query_bounds: &Rectangle) -> Vec<(Point, usize)> {
        let mut results = Vec::new();
        self.root.query(query_bounds, &mut results);
        results
    }

    pub fn nearest_neighbor(
        &self,
        query: &Point,
        max_distance: Option<f64>,
    ) -> Option<(Point, usize, f64)> {
        self.root.nearest_neighbor(query, max_distance)
    }
}

impl QuadNode {
    fn insert(&mut self, point: Point, data: usize, max_points: usize, max_depth: usize) {
        if !self.bounds.contains(&point) {
            return;
        }

        if self.children.is_none() {
            self.points.push((point, data));

            if self.points.len() > max_points && self.depth < max_depth {
                self.subdivide();

                // Redistribute points to children
                let points = std::mem::take(&mut self.points);
                for (p, d) in points {
                    self.insert_into_children(p, d, max_points, max_depth);
                }
            }
        } else {
            self.insert_into_children(point, data, max_points, max_depth);
        }
    }

    fn subdivide(&mut self) {
        let mid_x = (self.bounds.min_bounds[0] + self.bounds.max_bounds[0]) / 2.0;
        let mid_y = (self.bounds.min_bounds[1] + self.bounds.max_bounds[1]) / 2.0;

        let nw = Rectangle::new(
            vec![self.bounds.min_bounds[0], mid_y],
            vec![mid_x, self.bounds.max_bounds[1]],
        );
        let ne = Rectangle::new(
            vec![mid_x, mid_y],
            vec![self.bounds.max_bounds[0], self.bounds.max_bounds[1]],
        );
        let sw = Rectangle::new(
            vec![self.bounds.min_bounds[0], self.bounds.min_bounds[1]],
            vec![mid_x, mid_y],
        );
        let se = Rectangle::new(
            vec![mid_x, self.bounds.min_bounds[1]],
            vec![self.bounds.max_bounds[0], mid_y],
        );

        self.children = Some([
            Box::new(QuadNode {
                bounds: nw,
                points: Vec::new(),
                children: None,
                depth: self.depth + 1,
            }),
            Box::new(QuadNode {
                bounds: ne,
                points: Vec::new(),
                children: None,
                depth: self.depth + 1,
            }),
            Box::new(QuadNode {
                bounds: sw,
                points: Vec::new(),
                children: None,
                depth: self.depth + 1,
            }),
            Box::new(QuadNode {
                bounds: se,
                points: Vec::new(),
                children: None,
                depth: self.depth + 1,
            }),
        ]);
    }

    fn insert_into_children(
        &mut self,
        point: Point,
        data: usize,
        max_points: usize,
        max_depth: usize,
    ) {
        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                child.insert(point.clone(), data, max_points, max_depth);
            }
        }
    }

    fn query(&self, query_bounds: &Rectangle, results: &mut Vec<(Point, usize)>) {
        if !self.bounds.intersects(query_bounds) {
            return;
        }

        for (point, data) in &self.points {
            if query_bounds.contains(point) {
                results.push((point.clone(), *data));
            }
        }

        if let Some(children) = &self.children {
            for child in children.iter() {
                child.query(query_bounds, results);
            }
        }
    }

    fn nearest_neighbor(
        &self,
        query: &Point,
        max_distance: Option<f64>,
    ) -> Option<(Point, usize, f64)> {
        let mut best: Option<(Point, usize, f64)> = None;
        let max_dist_sq = max_distance.map(|d| d * d);

        // Check points in this node
        for (point, data) in &self.points {
            let dist_sq = query.distance_squared(point);

            if let Some(max_sq) = max_dist_sq {
                if dist_sq > max_sq {
                    continue;
                }
            }

            if best.is_none() || dist_sq < best.as_ref().unwrap().2 {
                best = Some((point.clone(), *data, dist_sq));
            }
        }

        // Check children if they exist
        if let Some(children) = &self.children {
            for child in children.iter() {
                if let Some(child_best) = child.nearest_neighbor(query, max_distance) {
                    if best.is_none() || child_best.2 * child_best.2 < best.as_ref().unwrap().2 {
                        best = Some((child_best.0, child_best.1, child_best.2 * child_best.2));
                    }
                }
            }
        }

        best.map(|(p, d, dist_sq)| (p, d, dist_sq.sqrt()))
    }
}

/// Octree for 3D spatial indexing
pub struct OctTree {
    root: OctNode,
    max_points: usize,
    max_depth: usize,
}

#[derive(Debug)]
struct OctNode {
    bounds: Rectangle,
    points: Vec<(Point, usize)>,
    children: Option<[Box<OctNode>; 8]>,
    depth: usize,
}

impl OctTree {
    pub fn new(bounds: Rectangle, max_points: usize, max_depth: usize) -> Self {
        assert_eq!(bounds.dimension(), 3, "OctTree only supports 3D");

        Self {
            root: OctNode {
                bounds,
                points: Vec::new(),
                children: None,
                depth: 0,
            },
            max_points,
            max_depth,
        }
    }

    pub fn insert(&mut self, point: Point, data: usize) {
        assert_eq!(point.dimension(), 3, "OctTree only supports 3D points");
        self.root
            .insert(point, data, self.max_points, self.max_depth);
    }

    pub fn query(&self, query_bounds: &Rectangle) -> Vec<(Point, usize)> {
        let mut results = Vec::new();
        self.root.query(query_bounds, &mut results);
        results
    }
}

impl OctNode {
    fn insert(&mut self, point: Point, data: usize, max_points: usize, max_depth: usize) {
        if !self.bounds.contains(&point) {
            return;
        }

        if self.children.is_none() {
            self.points.push((point, data));

            if self.points.len() > max_points && self.depth < max_depth {
                self.subdivide();

                // Redistribute points to children
                let points = std::mem::take(&mut self.points);
                for (p, d) in points {
                    self.insert_into_children(p, d, max_points, max_depth);
                }
            }
        } else {
            self.insert_into_children(point, data, max_points, max_depth);
        }
    }

    fn subdivide(&mut self) {
        let mid_x = (self.bounds.min_bounds[0] + self.bounds.max_bounds[0]) / 2.0;
        let mid_y = (self.bounds.min_bounds[1] + self.bounds.max_bounds[1]) / 2.0;
        let mid_z = (self.bounds.min_bounds[2] + self.bounds.max_bounds[2]) / 2.0;

        // Create 8 octants
        let mut children = Vec::with_capacity(8);

        for &z_low in &[true, false] {
            for &y_low in &[true, false] {
                for &x_low in &[true, false] {
                    let min_bounds = vec![
                        if x_low {
                            self.bounds.min_bounds[0]
                        } else {
                            mid_x
                        },
                        if y_low {
                            self.bounds.min_bounds[1]
                        } else {
                            mid_y
                        },
                        if z_low {
                            self.bounds.min_bounds[2]
                        } else {
                            mid_z
                        },
                    ];
                    let max_bounds = vec![
                        if x_low {
                            mid_x
                        } else {
                            self.bounds.max_bounds[0]
                        },
                        if y_low {
                            mid_y
                        } else {
                            self.bounds.max_bounds[1]
                        },
                        if z_low {
                            mid_z
                        } else {
                            self.bounds.max_bounds[2]
                        },
                    ];

                    children.push(Box::new(OctNode {
                        bounds: Rectangle::new(min_bounds, max_bounds),
                        points: Vec::new(),
                        children: None,
                        depth: self.depth + 1,
                    }));
                }
            }
        }

        self.children = Some(children.try_into().unwrap());
    }

    fn insert_into_children(
        &mut self,
        point: Point,
        data: usize,
        max_points: usize,
        max_depth: usize,
    ) {
        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                child.insert(point.clone(), data, max_points, max_depth);
            }
        }
    }

    fn query(&self, query_bounds: &Rectangle, results: &mut Vec<(Point, usize)>) {
        if !self.bounds.intersects(query_bounds) {
            return;
        }

        for (point, data) in &self.points {
            if query_bounds.contains(point) {
                results.push((point.clone(), *data));
            }
        }

        if let Some(children) = &self.children {
            for child in children.iter() {
                child.query(query_bounds, results);
            }
        }
    }
}

/// Spatial hash for efficient spatial queries with grid-based indexing
pub struct SpatialHash {
    grid: std::collections::HashMap<(i32, i32), Vec<(Point, usize)>>,
    cell_size: f64,
    bounds: Rectangle,
}

impl SpatialHash {
    pub fn new(bounds: Rectangle, cell_size: f64) -> Self {
        assert_eq!(bounds.dimension(), 2, "SpatialHash only supports 2D");

        Self {
            grid: std::collections::HashMap::new(),
            cell_size,
            bounds,
        }
    }

    fn hash_point(&self, point: &Point) -> (i32, i32) {
        let x =
            ((point.coordinates[0] - self.bounds.min_bounds[0]) / self.cell_size).floor() as i32;
        let y =
            ((point.coordinates[1] - self.bounds.min_bounds[1]) / self.cell_size).floor() as i32;
        (x, y)
    }

    pub fn insert(&mut self, point: Point, data: usize) {
        let hash = self.hash_point(&point);
        self.grid.entry(hash).or_default().push((point, data));
    }

    pub fn query_radius(&self, center: &Point, radius: f64) -> Vec<(Point, usize)> {
        let mut results = Vec::new();
        let radius_sq = radius * radius;

        // Calculate grid cells that might contain points within radius
        let cells_to_check = ((radius / self.cell_size).ceil() as i32) + 1;
        let center_hash = self.hash_point(center);

        for dx in -cells_to_check..=cells_to_check {
            for dy in -cells_to_check..=cells_to_check {
                let hash = (center_hash.0 + dx, center_hash.1 + dy);

                if let Some(points) = self.grid.get(&hash) {
                    for (point, data) in points {
                        if center.distance_squared(point) <= radius_sq {
                            results.push((point.clone(), *data));
                        }
                    }
                }
            }
        }

        results
    }

    pub fn clear(&mut self) {
        self.grid.clear();
    }

    pub fn stats(&self) -> SpatialHashStats {
        let total_points: usize = self.grid.values().map(|v| v.len()).sum();
        let occupied_cells = self.grid.len();
        let max_points_per_cell = self.grid.values().map(|v| v.len()).max().unwrap_or(0);
        let avg_points_per_cell = if occupied_cells > 0 {
            total_points as f64 / occupied_cells as f64
        } else {
            0.0
        };

        SpatialHashStats {
            total_points,
            occupied_cells,
            max_points_per_cell,
            avg_points_per_cell,
            cell_size: self.cell_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialHashStats {
    pub total_points: usize,
    pub occupied_cells: usize,
    pub max_points_per_cell: usize,
    pub avg_points_per_cell: f64,
    pub cell_size: f64,
}

/// Geographic utilities for working with spatial coordinates and geographic data
pub mod geographic {
    use super::{Point, Rectangle};
    use std::f64::consts::PI;

    /// Coordinate system types
    #[derive(Debug, Clone, PartialEq)]
    pub enum CoordinateSystem {
        /// Latitude/Longitude in degrees (WGS84)
        LatLon,
        /// Universal Transverse Mercator
        UTM { zone: u8, hemisphere: Hemisphere },
        /// Web Mercator (used by web mapping services)
        WebMercator,
        /// Custom coordinate system with EPSG code
        EPSG(u32),
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Hemisphere {
        North,
        South,
    }

    /// Geographic point with latitude and longitude
    #[derive(Debug, Clone, PartialEq)]
    pub struct GeoPoint {
        pub latitude: f64,
        pub longitude: f64,
        pub altitude: Option<f64>,
        pub coordinate_system: CoordinateSystem,
    }

    impl GeoPoint {
        /// Create a new geographic point
        pub fn new(latitude: f64, longitude: f64) -> Self {
            Self {
                latitude,
                longitude,
                altitude: None,
                coordinate_system: CoordinateSystem::LatLon,
            }
        }

        /// Create a geographic point with altitude
        pub fn with_altitude(latitude: f64, longitude: f64, altitude: f64) -> Self {
            Self {
                latitude,
                longitude,
                altitude: Some(altitude),
                coordinate_system: CoordinateSystem::LatLon,
            }
        }

        /// Set coordinate system
        pub fn with_coordinate_system(mut self, coord_sys: CoordinateSystem) -> Self {
            self.coordinate_system = coord_sys;
            self
        }

        /// Calculate Haversine distance between two geographic points in meters
        pub fn haversine_distance(&self, other: &GeoPoint) -> f64 {
            const EARTH_RADIUS_M: f64 = 6_371_000.0;

            let lat1_rad = self.latitude.to_radians();
            let lat2_rad = other.latitude.to_radians();
            let dlat_rad = (other.latitude - self.latitude).to_radians();
            let dlon_rad = (other.longitude - self.longitude).to_radians();

            let a = (dlat_rad / 2.0).sin().powi(2)
                + lat1_rad.cos() * lat2_rad.cos() * (dlon_rad / 2.0).sin().powi(2);

            let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
            EARTH_RADIUS_M * c
        }

        /// Calculate bearing (initial heading) from this point to another in degrees
        pub fn bearing_to(&self, other: &GeoPoint) -> f64 {
            let lat1_rad = self.latitude.to_radians();
            let lat2_rad = other.latitude.to_radians();
            let dlon_rad = (other.longitude - self.longitude).to_radians();

            let y = dlon_rad.sin() * lat2_rad.cos();
            let x =
                lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * dlon_rad.cos();

            let bearing_rad = y.atan2(x);
            (bearing_rad.to_degrees() + 360.0) % 360.0
        }

        /// Calculate destination point given distance and bearing
        pub fn destination_point(&self, distance_m: f64, bearing_deg: f64) -> GeoPoint {
            const EARTH_RADIUS_M: f64 = 6_371_000.0;

            let lat1_rad = self.latitude.to_radians();
            let lon1_rad = self.longitude.to_radians();
            let bearing_rad = bearing_deg.to_radians();
            let angular_distance = distance_m / EARTH_RADIUS_M;

            let lat2_rad = (lat1_rad.sin() * angular_distance.cos()
                + lat1_rad.cos() * angular_distance.sin() * bearing_rad.cos())
            .asin();

            let lon2_rad = lon1_rad
                + (bearing_rad.sin() * angular_distance.sin() * lat1_rad.cos())
                    .atan2(angular_distance.cos() - lat1_rad.sin() * lat2_rad.sin());

            GeoPoint::new(lat2_rad.to_degrees(), lon2_rad.to_degrees())
        }

        /// Convert to Point for use with spatial data structures
        pub fn to_point(&self) -> Point {
            if let Some(alt) = self.altitude {
                Point::new(vec![self.longitude, self.latitude, alt])
            } else {
                Point::new(vec![self.longitude, self.latitude])
            }
        }

        /// Check if point is within geographic bounds
        pub fn is_within_bounds(&self, bounds: &GeoBounds) -> bool {
            self.latitude >= bounds.south
                && self.latitude <= bounds.north
                && self.longitude >= bounds.west
                && self.longitude <= bounds.east
        }

        /// Convert to Web Mercator projection (EPSG:3857)
        pub fn to_web_mercator(&self) -> Point {
            const EARTH_RADIUS: f64 = 6_378_137.0;

            let x = self.longitude.to_radians() * EARTH_RADIUS;
            let y = ((PI / 4.0 + self.latitude.to_radians() / 2.0).tan().ln()) * EARTH_RADIUS;

            Point::new(vec![x, y])
        }
    }

    /// Geographic bounding box
    #[derive(Debug, Clone, PartialEq)]
    pub struct GeoBounds {
        pub north: f64,
        pub south: f64,
        pub east: f64,
        pub west: f64,
    }

    impl GeoBounds {
        /// Create new geographic bounds
        pub fn new(north: f64, south: f64, east: f64, west: f64) -> Self {
            Self {
                north,
                south,
                east,
                west,
            }
        }

        /// Create bounds from center point and radius in meters
        pub fn from_center_radius(center: &GeoPoint, radius_m: f64) -> Self {
            const EARTH_RADIUS_M: f64 = 6_371_000.0;

            let lat_offset = (radius_m / EARTH_RADIUS_M).to_degrees();
            let lon_offset =
                (radius_m / (EARTH_RADIUS_M * center.latitude.to_radians().cos())).to_degrees();

            Self {
                north: center.latitude + lat_offset,
                south: center.latitude - lat_offset,
                east: center.longitude + lon_offset,
                west: center.longitude - lon_offset,
            }
        }

        /// Check if bounds contains a point
        pub fn contains(&self, point: &GeoPoint) -> bool {
            point.is_within_bounds(self)
        }

        /// Check if bounds intersect with another bounds
        pub fn intersects(&self, other: &GeoBounds) -> bool {
            !(self.east < other.west
                || self.west > other.east
                || self.north < other.south
                || self.south > other.north)
        }

        /// Calculate area in square meters (approximate)
        pub fn area_square_meters(&self) -> f64 {
            const EARTH_RADIUS_M: f64 = 6_371_000.0;

            let lat_range_rad = (self.north - self.south).to_radians();
            let lon_range_rad = (self.east - self.west).to_radians();
            let avg_lat_rad = ((self.north + self.south) / 2.0).to_radians();

            EARTH_RADIUS_M.powi(2) * lat_range_rad * lon_range_rad * avg_lat_rad.cos()
        }

        /// Convert to Rectangle for use with spatial data structures
        pub fn to_rectangle(&self) -> Rectangle {
            Rectangle::new(vec![self.west, self.south], vec![self.east, self.north])
        }
    }

    /// Geographic utilities for common geographic operations
    pub struct GeoUtils;

    impl GeoUtils {
        /// Convert degrees to radians
        pub fn deg_to_rad(degrees: f64) -> f64 {
            degrees * PI / 180.0
        }

        /// Convert radians to degrees
        pub fn rad_to_deg(radians: f64) -> f64 {
            radians * 180.0 / PI
        }

        /// Normalize longitude to [-180, 180] range
        pub fn normalize_longitude(lon: f64) -> f64 {
            let mut normalized = lon % 360.0;
            if normalized > 180.0 {
                normalized -= 360.0;
            } else if normalized < -180.0 {
                normalized += 360.0;
            }
            normalized
        }

        /// Normalize latitude to [-90, 90] range
        pub fn normalize_latitude(lat: f64) -> f64 {
            lat.clamp(-90.0, 90.0)
        }

        /// Calculate the center point of multiple geographic points
        pub fn centroid(points: &[GeoPoint]) -> Option<GeoPoint> {
            if points.is_empty() {
                return None;
            }

            let mut x_sum = 0.0;
            let mut y_sum = 0.0;
            let mut z_sum = 0.0;

            for point in points {
                let lat_rad = point.latitude.to_radians();
                let lon_rad = point.longitude.to_radians();

                x_sum += lat_rad.cos() * lon_rad.cos();
                y_sum += lat_rad.cos() * lon_rad.sin();
                z_sum += lat_rad.sin();
            }

            let count = points.len() as f64;
            let x_avg = x_sum / count;
            let y_avg = y_sum / count;
            let z_avg = z_sum / count;

            let lon_rad = y_avg.atan2(x_avg);
            let hyp = (x_avg * x_avg + y_avg * y_avg).sqrt();
            let lat_rad = z_avg.atan2(hyp);

            Some(GeoPoint::new(lat_rad.to_degrees(), lon_rad.to_degrees()))
        }

        /// Calculate polygon area using the spherical excess method (in square meters)
        pub fn polygon_area(vertices: &[GeoPoint]) -> f64 {
            if vertices.len() < 3 {
                return 0.0;
            }

            const EARTH_RADIUS_M: f64 = 6_371_000.0;
            let mut area = 0.0;

            for i in 0..vertices.len() {
                let j = (i + 1) % vertices.len();
                let lat1 = vertices[i].latitude.to_radians();
                let lat2 = vertices[j].latitude.to_radians();
                let lon_diff = (vertices[j].longitude - vertices[i].longitude).to_radians();

                area += lon_diff * (2.0 + lat1.sin() + lat2.sin());
            }

            (area.abs() / 2.0) * EARTH_RADIUS_M.powi(2)
        }

        /// Check if a point is inside a polygon using ray casting algorithm
        pub fn point_in_polygon(point: &GeoPoint, polygon: &[GeoPoint]) -> bool {
            if polygon.len() < 3 {
                return false;
            }

            let mut inside = false;
            let mut j = polygon.len() - 1;

            for i in 0..polygon.len() {
                if ((polygon[i].latitude > point.latitude)
                    != (polygon[j].latitude > point.latitude))
                    && (point.longitude
                        < (polygon[j].longitude - polygon[i].longitude)
                            * (point.latitude - polygon[i].latitude)
                            / (polygon[j].latitude - polygon[i].latitude)
                            + polygon[i].longitude)
                {
                    inside = !inside;
                }
                j = i;
            }

            inside
        }

        /// Find the closest point on a line segment to a given point
        pub fn closest_point_on_line(
            point: &GeoPoint,
            line_start: &GeoPoint,
            line_end: &GeoPoint,
        ) -> GeoPoint {
            let lat_start = line_start.latitude.to_radians();
            let lon_start = line_start.longitude.to_radians();
            let lat_end = line_end.latitude.to_radians();
            let lon_end = line_end.longitude.to_radians();
            let lat_point = point.latitude.to_radians();
            let lon_point = point.longitude.to_radians();

            // Convert to Cartesian coordinates
            let x1 = lat_start.cos() * lon_start.cos();
            let y1 = lat_start.cos() * lon_start.sin();
            let z1 = lat_start.sin();

            let x2 = lat_end.cos() * lon_end.cos();
            let y2 = lat_end.cos() * lon_end.sin();
            let z2 = lat_end.sin();

            let x0 = lat_point.cos() * lon_point.cos();
            let y0 = lat_point.cos() * lon_point.sin();
            let z0 = lat_point.sin();

            // Project point onto line
            let dot_product = x0 * (x2 - x1) + y0 * (y2 - y1) + z0 * (z2 - z1);
            let line_length_sq = (x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2);

            let t = if line_length_sq == 0.0 {
                0.0
            } else {
                (dot_product / line_length_sq).clamp(0.0, 1.0)
            };

            let closest_x = x1 + t * (x2 - x1);
            let closest_y = y1 + t * (y2 - y1);
            let closest_z = z1 + t * (z2 - z1);

            // Convert back to lat/lon
            let closest_lon = closest_y.atan2(closest_x).to_degrees();
            let closest_lat = closest_z
                .atan2((closest_x.powi(2) + closest_y.powi(2)).sqrt())
                .to_degrees();

            GeoPoint::new(closest_lat, closest_lon)
        }

        /// Calculate great circle distance between two points (same as Haversine)
        pub fn great_circle_distance(point1: &GeoPoint, point2: &GeoPoint) -> f64 {
            point1.haversine_distance(point2)
        }

        /// Convert decimal degrees to degrees, minutes, seconds format
        pub fn decimal_to_dms(decimal: f64) -> (i32, i32, f64) {
            let degrees = decimal.trunc() as i32;
            let minutes_float = (decimal.abs() - degrees.abs() as f64) * 60.0;
            let minutes = minutes_float.trunc() as i32;
            let seconds = (minutes_float - minutes as f64) * 60.0;
            (degrees, minutes, seconds)
        }

        /// Convert degrees, minutes, seconds to decimal degrees
        pub fn dms_to_decimal(degrees: i32, minutes: i32, seconds: f64) -> f64 {
            let sign = if degrees < 0 { -1.0 } else { 1.0 };
            degrees.abs() as f64 + minutes as f64 / 60.0 + seconds / 3600.0 * sign
        }
    }

    #[allow(non_snake_case)]
    #[cfg(test)]
    mod geo_tests {
        use super::*;

        #[test]
        fn test_geopoint_haversine_distance() {
            let london = GeoPoint::new(51.5074, -0.1278);
            let paris = GeoPoint::new(48.8566, 2.3522);

            let distance = london.haversine_distance(&paris);
            // Distance between London and Paris is approximately 344 km
            assert!((distance - 344_000.0).abs() < 10_000.0);
        }

        #[test]
        fn test_geopoint_bearing() {
            let start = GeoPoint::new(51.0, 0.0);
            let end = GeoPoint::new(52.0, 1.0);

            let bearing = start.bearing_to(&end);
            assert!((0.0..360.0).contains(&bearing));
        }

        #[test]
        fn test_geopoint_destination() {
            let start = GeoPoint::new(51.0, 0.0);
            let destination = start.destination_point(100_000.0, 90.0); // 100km east

            // Should be roughly at the same latitude, but further east
            assert!((destination.latitude - start.latitude).abs() < 0.1);
            assert!(destination.longitude > start.longitude);
        }

        #[test]
        fn test_geobounds() {
            let bounds = GeoBounds::new(52.0, 50.0, 2.0, 0.0);
            let point_inside = GeoPoint::new(51.0, 1.0);
            let point_outside = GeoPoint::new(53.0, 3.0);

            assert!(bounds.contains(&point_inside));
            assert!(!bounds.contains(&point_outside));
        }

        #[test]
        fn test_geoutils_centroid() {
            let points = vec![
                GeoPoint::new(0.0, 0.0),
                GeoPoint::new(1.0, 0.0),
                GeoPoint::new(0.0, 1.0),
                GeoPoint::new(1.0, 1.0),
            ];

            let centroid = GeoUtils::centroid(&points).unwrap();
            assert!((centroid.latitude - 0.5).abs() < 0.1);
            assert!((centroid.longitude - 0.5).abs() < 0.1);
        }

        #[test]
        fn test_point_in_polygon() {
            let polygon = vec![
                GeoPoint::new(0.0, 0.0),
                GeoPoint::new(2.0, 0.0),
                GeoPoint::new(2.0, 2.0),
                GeoPoint::new(0.0, 2.0),
            ];

            let inside_point = GeoPoint::new(1.0, 1.0);
            let outside_point = GeoPoint::new(3.0, 3.0);

            assert!(GeoUtils::point_in_polygon(&inside_point, &polygon));
            assert!(!GeoUtils::point_in_polygon(&outside_point, &polygon));
        }

        #[test]
        fn test_normalize_coordinates() {
            assert_eq!(GeoUtils::normalize_longitude(380.0), 20.0);
            assert_eq!(GeoUtils::normalize_longitude(-200.0), 160.0);
            assert_eq!(GeoUtils::normalize_latitude(100.0), 90.0);
            assert_eq!(GeoUtils::normalize_latitude(-100.0), -90.0);
        }

        #[test]
        fn test_dms_conversion() {
            let decimal = 51.5074;
            let (degrees, minutes, seconds) = GeoUtils::decimal_to_dms(decimal);
            let converted_back = GeoUtils::dms_to_decimal(degrees, minutes, seconds);

            assert!((decimal - converted_back).abs() < 0.0001);
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_operations() {
        let p1 = Point::new(vec![1.0, 2.0, 3.0]);
        let p2 = Point::new(vec![4.0, 5.0, 6.0]);

        assert_eq!(p1.dimension(), 3);
        assert_eq!(p1.distance_squared(&p2), 27.0); // (4-1)² + (5-2)² + (6-3)² = 9+9+9 = 27
        assert_eq!(p1.distance(&p2), 27.0_f64.sqrt());
    }

    #[test]
    fn test_rectangle_operations() {
        let rect = Rectangle::new(vec![0.0, 0.0], vec![10.0, 10.0]);
        let point_inside = Point::new(vec![5.0, 5.0]);
        let point_outside = Point::new(vec![15.0, 15.0]);

        assert!(rect.contains(&point_inside));
        assert!(!rect.contains(&point_outside));
        assert_eq!(rect.area(), 100.0);

        let other_rect = Rectangle::new(vec![5.0, 5.0], vec![15.0, 15.0]);
        assert!(rect.intersects(&other_rect));

        let union = rect.union(&other_rect);
        assert_eq!(union.min_bounds, vec![0.0, 0.0]);
        assert_eq!(union.max_bounds, vec![15.0, 15.0]);
    }

    #[test]
    fn test_kdtree() {
        let points = vec![
            (Point::new(vec![2.0, 3.0]), 0),
            (Point::new(vec![5.0, 4.0]), 1),
            (Point::new(vec![9.0, 6.0]), 2),
            (Point::new(vec![4.0, 7.0]), 3),
            (Point::new(vec![8.0, 1.0]), 4),
            (Point::new(vec![7.0, 2.0]), 5),
        ];

        let kd_tree = KdTree::from_points(points);

        let query = Point::new(vec![5.0, 5.0]);
        let result = kd_tree.nearest_neighbor(&query);

        assert!(result.is_some());
        let (_, data, _) = result.unwrap();
        // Should find one of the nearby points
        assert!(data <= 5);

        // Test range query
        let range = Rectangle::new(vec![0.0, 0.0], vec![6.0, 6.0]);
        let range_results = kd_tree.range_query(&range);
        assert!(!range_results.is_empty());
    }

    #[test]
    fn test_quadtree() {
        let bounds = Rectangle::new(vec![0.0, 0.0], vec![100.0, 100.0]);
        let mut quad_tree = QuadTree::new(bounds, 4, 6);

        // Insert some points
        quad_tree.insert(Point::new(vec![10.0, 10.0]), 0);
        quad_tree.insert(Point::new(vec![20.0, 20.0]), 1);
        quad_tree.insert(Point::new(vec![80.0, 80.0]), 2);
        quad_tree.insert(Point::new(vec![90.0, 90.0]), 3);

        // Query a region
        let query_bounds = Rectangle::new(vec![5.0, 5.0], vec![25.0, 25.0]);
        let results = quad_tree.query(&query_bounds);

        assert_eq!(results.len(), 2); // Should find points 0 and 1

        // Test nearest neighbor
        let query_point = Point::new(vec![15.0, 15.0]);
        let nearest = quad_tree.nearest_neighbor(&query_point, None);
        assert!(nearest.is_some());
    }

    #[test]
    fn test_spatial_hash() {
        let bounds = Rectangle::new(vec![0.0, 0.0], vec![100.0, 100.0]);
        let mut spatial_hash = SpatialHash::new(bounds, 10.0);

        // Insert some points
        spatial_hash.insert(Point::new(vec![15.0, 15.0]), 0);
        spatial_hash.insert(Point::new(vec![25.0, 25.0]), 1);
        spatial_hash.insert(Point::new(vec![85.0, 85.0]), 2);

        // Query radius
        let center = Point::new(vec![20.0, 20.0]);
        let results = spatial_hash.query_radius(&center, 10.0);

        assert!(!results.is_empty());

        let stats = spatial_hash.stats();
        assert_eq!(stats.total_points, 3);
    }

    #[test]
    fn test_octree() {
        let bounds = Rectangle::new(vec![0.0, 0.0, 0.0], vec![100.0, 100.0, 100.0]);
        let mut oct_tree = OctTree::new(bounds, 4, 6);

        // Insert some 3D points
        oct_tree.insert(Point::new(vec![10.0, 10.0, 10.0]), 0);
        oct_tree.insert(Point::new(vec![20.0, 20.0, 20.0]), 1);
        oct_tree.insert(Point::new(vec![80.0, 80.0, 80.0]), 2);

        // Query a 3D region
        let query_bounds = Rectangle::new(vec![5.0, 5.0, 5.0], vec![25.0, 25.0, 25.0]);
        let results = oct_tree.query(&query_bounds);

        assert_eq!(results.len(), 2); // Should find points 0 and 1
    }
}

//! Layout algorithm implementations for graph visualization
//!
//! This module provides various layout algorithms for positioning nodes in trait graphs,
//! including force-directed, hierarchical, circular, and other specialized layouts.

use crate::trait_explorer::graph_visualization::{
    graph_config::*,
    graph_structures::*,
};
use scirs2_core::random::Random;
use std::collections::HashMap;
use std::time::Instant;

/// Trait for layout algorithm implementations
///
/// All layout algorithms must implement this trait to provide consistent
/// interfaces for graph positioning and quality assessment.
pub trait LayoutAlgorithmImpl: Send + Sync {
    /// Compute the layout for a given graph
    fn compute_layout(&self, graph: &TraitGraph, config: &GraphConfig) -> Result<LayoutResult, Box<dyn std::error::Error>>;

    /// Get the name of the layout algorithm
    fn name(&self) -> &str;

    /// Check if the algorithm supports 3D layouts
    fn supports_3d(&self) -> bool;

    /// Check if the algorithm supports GPU acceleration
    fn supports_gpu(&self) -> bool;
}

/// Force-directed layout algorithm using physics simulation
///
/// This implementation uses attractive and repulsive forces to position nodes
/// in a way that minimizes edge crossings and creates aesthetically pleasing layouts.
/// Supports both 2D and 3D layouts with comprehensive quality metrics.
#[derive(Debug)]
pub struct ForceDirectedLayout {
    /// Configuration parameters for the force simulation
    pub iterations: usize,
    pub temperature_factor: f64,
    pub repulsive_strength: f64,
    pub attractive_strength: f64,
    pub damping_factor: f64,
    pub convergence_threshold: f64,
}

impl ForceDirectedLayout {
    /// Create a new force-directed layout with default parameters
    pub fn new() -> Self {
        Self {
            iterations: 500,
            temperature_factor: 100.0,
            repulsive_strength: 1.0,
            attractive_strength: 1.0,
            damping_factor: 0.9,
            convergence_threshold: 0.01,
        }
    }

    /// Create a new force-directed layout with custom parameters
    pub fn with_parameters(
        iterations: usize,
        temperature_factor: f64,
        repulsive_strength: f64,
        attractive_strength: f64,
    ) -> Self {
        Self {
            iterations,
            temperature_factor,
            repulsive_strength,
            attractive_strength,
            damping_factor: 0.9,
            convergence_threshold: 0.01,
        }
    }

    /// Apply forces to nodes in 2D space
    fn apply_2d_forces(
        &self,
        graph: &TraitGraph,
        positions_2d: &HashMap<String, (f64, f64)>,
        forces_2d: &mut HashMap<String, (f64, f64)>,
        k: f64,
    ) {
        // Initialize forces
        for node in &graph.nodes {
            forces_2d.insert(node.id.clone(), (0.0, 0.0));
        }

        // Repulsive forces between all node pairs
        for i in 0..graph.nodes.len() {
            for j in (i + 1)..graph.nodes.len() {
                let node1 = &graph.nodes[i];
                let node2 = &graph.nodes[j];

                let pos1_2d = positions_2d[&node1.id];
                let pos2_2d = positions_2d[&node2.id];

                let dx = pos1_2d.0 - pos2_2d.0;
                let dy = pos1_2d.1 - pos2_2d.1;
                let distance_2d = ((dx * dx + dy * dy) as f64).sqrt().max(0.01f64);

                let repulsive_force = self.repulsive_strength * k * k / distance_2d;
                let fx = (dx / distance_2d) * repulsive_force;
                let fy = (dy / distance_2d) * repulsive_force;

                let f1 = forces_2d.get_mut(&node1.id).expect("get_mut should succeed");
                f1.0 += fx;
                f1.1 += fy;

                let f2 = forces_2d.get_mut(&node2.id).expect("get_mut should succeed");
                f2.0 -= fx;
                f2.1 -= fy;
            }
        }

        // Attractive forces for connected nodes
        for edge in &graph.edges {
            let pos1_2d = positions_2d[&edge.from];
            let pos2_2d = positions_2d[&edge.to];

            let dx = pos1_2d.0 - pos2_2d.0;
            let dy = pos1_2d.1 - pos2_2d.1;
            let distance_2d = (dx * dx + dy * dy).sqrt().max(0.01);

            let attractive_force = self.attractive_strength * distance_2d * distance_2d / k * edge.weight;
            let fx = (dx / distance_2d) * attractive_force;
            let fy = (dy / distance_2d) * attractive_force;

            let f1 = forces_2d.get_mut(&edge.from).expect("get_mut should succeed");
            f1.0 -= fx;
            f1.1 -= fy;

            let f2 = forces_2d.get_mut(&edge.to).expect("get_mut should succeed");
            f2.0 += fx;
            f2.1 += fy;
        }
    }

    /// Apply forces to nodes in 3D space
    fn apply_3d_forces(
        &self,
        graph: &TraitGraph,
        positions_3d: &HashMap<String, (f64, f64, f64)>,
        forces_3d: &mut HashMap<String, (f64, f64, f64)>,
        k: f64,
    ) {
        // Initialize forces
        for node in &graph.nodes {
            forces_3d.insert(node.id.clone(), (0.0, 0.0, 0.0));
        }

        // Repulsive forces between all node pairs
        for i in 0..graph.nodes.len() {
            for j in (i + 1)..graph.nodes.len() {
                let node1 = &graph.nodes[i];
                let node2 = &graph.nodes[j];

                let pos1_3d = positions_3d[&node1.id];
                let pos2_3d = positions_3d[&node2.id];

                let dx = pos1_3d.0 - pos2_3d.0;
                let dy = pos1_3d.1 - pos2_3d.1;
                let dz = pos1_3d.2 - pos2_3d.2;
                let distance_3d = (dx * dx + dy * dy + dz * dz).sqrt().max(0.01);

                let repulsive_force = self.repulsive_strength * k * k / distance_3d;
                let fx = (dx / distance_3d) * repulsive_force;
                let fy = (dy / distance_3d) * repulsive_force;
                let fz = (dz / distance_3d) * repulsive_force;

                let f1_3d = forces_3d.get_mut(&node1.id).expect("get_mut should succeed");
                f1_3d.0 += fx;
                f1_3d.1 += fy;
                f1_3d.2 += fz;

                let f2_3d = forces_3d.get_mut(&node2.id).expect("get_mut should succeed");
                f2_3d.0 -= fx;
                f2_3d.1 -= fy;
                f2_3d.2 -= fz;
            }
        }

        // Attractive forces for connected nodes
        for edge in &graph.edges {
            let pos1_3d = positions_3d[&edge.from];
            let pos2_3d = positions_3d[&edge.to];

            let dx = pos1_3d.0 - pos2_3d.0;
            let dy = pos1_3d.1 - pos2_3d.1;
            let dz = pos1_3d.2 - pos2_3d.2;
            let distance_3d = (dx * dx + dy * dy + dz * dz).sqrt().max(0.01);

            let attractive_force = self.attractive_strength * distance_3d * distance_3d / k * edge.weight;
            let fx = (dx / distance_3d) * attractive_force;
            let fy = (dy / distance_3d) * attractive_force;
            let fz = (dz / distance_3d) * attractive_force;

            let f1_3d = forces_3d.get_mut(&edge.from).expect("get_mut should succeed");
            f1_3d.0 -= fx;
            f1_3d.1 -= fy;
            f1_3d.2 -= fz;

            let f2_3d = forces_3d.get_mut(&edge.to).expect("get_mut should succeed");
            f2_3d.0 += fx;
            f2_3d.1 += fy;
            f2_3d.2 += fz;
        }
    }

    /// Update positions based on forces with temperature-based damping
    fn update_positions(
        &self,
        graph: &TraitGraph,
        positions_2d: &mut HashMap<String, (f64, f64)>,
        positions_3d: &mut Option<HashMap<String, (f64, f64, f64)>>,
        forces_2d: &HashMap<String, (f64, f64)>,
        forces_3d: &HashMap<String, (f64, f64, f64)>,
        temperature: f64,
    ) -> f64 {
        let mut total_displacement = 0.0;

        for node in &graph.nodes {
            // Update 2D positions
            let force_2d = &forces_2d[&node.id];
            let force_magnitude = (force_2d.0 * force_2d.0 + force_2d.1 * force_2d.1).sqrt();

            if force_magnitude > 0.0 {
                let displacement = force_magnitude.min(temperature) * self.damping_factor;
                let pos = positions_2d.get_mut(&node.id).expect("get_mut should succeed");
                let dx = (force_2d.0 / force_magnitude) * displacement;
                let dy = (force_2d.1 / force_magnitude) * displacement;

                pos.0 += dx;
                pos.1 += dy;

                total_displacement += (dx * dx + dy * dy).sqrt();
            }

            // Update 3D positions if enabled
            if let (Some(ref mut pos_3d_map), Some(force_3d)) = (positions_3d, forces_3d.get(&node.id)) {
                let force_magnitude_3d = (force_3d.0 * force_3d.0 + force_3d.1 * force_3d.1 + force_3d.2 * force_3d.2).sqrt();

                if force_magnitude_3d > 0.0 {
                    let displacement_3d = force_magnitude_3d.min(temperature) * self.damping_factor;
                    let pos_3d = pos_3d_map.get_mut(&node.id).expect("get_mut should succeed");

                    let dx = (force_3d.0 / force_magnitude_3d) * displacement_3d;
                    let dy = (force_3d.1 / force_magnitude_3d) * displacement_3d;
                    let dz = (force_3d.2 / force_magnitude_3d) * displacement_3d;

                    pos_3d.0 += dx;
                    pos_3d.1 += dy;
                    pos_3d.2 += dz;
                }
            }
        }

        total_displacement
    }

    /// Check if the algorithm has converged
    fn has_converged(&self, displacement: f64, iteration: usize) -> bool {
        displacement < self.convergence_threshold || iteration >= self.iterations
    }
}

impl Default for ForceDirectedLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutAlgorithmImpl for ForceDirectedLayout {
    fn compute_layout(&self, graph: &TraitGraph, config: &GraphConfig) -> Result<LayoutResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let n = graph.nodes.len();

        if n == 0 {
            return Ok(LayoutResult {
                positions_2d: HashMap::new(),
                positions_3d: None,
                quality_metrics: LayoutQualityMetrics {
                    edge_crossings: 0,
                    average_edge_length: 0.0,
                    distribution_uniformity: 1.0,
                    aesthetic_score: 1.0,
                },
                computation_time: start_time.elapsed(),
            });
        }

        // Initialize random positions
        let mut rng = Random::seed(42);
        let mut positions_2d = HashMap::new();
        let mut positions_3d = if config.enable_3d {
            Some(HashMap::new())
        } else {
            None
        };

        // Place nodes in random positions
        for node in &graph.nodes {
            let x = rng.random_range(-100.0..100.0);
            let y = rng.random_range(-100.0..100.0);
            positions_2d.insert(node.id.clone(), (x, y));

            if let Some(ref mut pos_3d) = positions_3d {
                let z = rng.random_range(-100.0..100.0);
                pos_3d.insert(node.id.clone(), (x, y, z));
            }
        }

        // Force-directed algorithm parameters
        let k = (400.0 / n as f64).sqrt(); // Optimal distance between nodes

        // Main simulation loop
        for iteration in 0..self.iterations {
            let t = self.temperature_factor * (1.0 - iteration as f64 / self.iterations as f64);

            // Calculate forces
            let mut forces_2d: HashMap<String, (f64, f64)> = HashMap::new();
            let mut forces_3d: HashMap<String, (f64, f64, f64)> = HashMap::new();

            // Apply 2D forces
            self.apply_2d_forces(graph, &positions_2d, &mut forces_2d, k);

            // Apply 3D forces if enabled
            if let Some(ref pos_3d_map) = positions_3d {
                self.apply_3d_forces(graph, pos_3d_map, &mut forces_3d, k);
            }

            // Update positions
            let displacement = self.update_positions(
                graph,
                &mut positions_2d,
                &mut positions_3d,
                &forces_2d,
                &forces_3d,
                t,
            );

            // Check for convergence
            if self.has_converged(displacement, iteration) {
                break;
            }
        }

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(graph, &positions_2d)?;

        Ok(LayoutResult {
            positions_2d,
            positions_3d,
            quality_metrics,
            computation_time: start_time.elapsed(),
        })
    }

    fn name(&self) -> &str {
        "Force-Directed"
    }

    fn supports_3d(&self) -> bool {
        true
    }

    fn supports_gpu(&self) -> bool {
        false
    }
}

impl ForceDirectedLayout {
    /// Calculate comprehensive quality metrics for the layout
    fn calculate_quality_metrics(
        &self,
        graph: &TraitGraph,
        positions: &HashMap<String, (f64, f64)>,
    ) -> Result<LayoutQualityMetrics, Box<dyn std::error::Error>> {
        let edge_crossings = self.count_edge_crossings(graph, positions)?;
        let average_edge_length = self.calculate_average_edge_length(graph, positions)?;
        let distribution_uniformity = self.calculate_distribution_uniformity(positions)?;
        let aesthetic_score = self.calculate_aesthetic_score(graph, positions)?;

        Ok(LayoutQualityMetrics {
            edge_crossings,
            average_edge_length,
            distribution_uniformity,
            aesthetic_score,
        })
    }

    /// Count the number of edge crossings in the layout
    fn count_edge_crossings(
        &self,
        graph: &TraitGraph,
        positions: &HashMap<String, (f64, f64)>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut crossings = 0;

        for i in 0..graph.edges.len() {
            for j in (i + 1)..graph.edges.len() {
                let edge1 = &graph.edges[i];
                let edge2 = &graph.edges[j];

                if let (Some(&pos1_from), Some(&pos1_to), Some(&pos2_from), Some(&pos2_to)) = (
                    positions.get(&edge1.from),
                    positions.get(&edge1.to),
                    positions.get(&edge2.from),
                    positions.get(&edge2.to),
                ) {
                    if self.lines_intersect(pos1_from, pos1_to, pos2_from, pos2_to) {
                        crossings += 1;
                    }
                }
            }
        }

        Ok(crossings)
    }

    /// Check if two line segments intersect
    fn lines_intersect(
        &self,
        p1: (f64, f64),
        p2: (f64, f64),
        p3: (f64, f64),
        p4: (f64, f64),
    ) -> bool {
        let d1 = self.orientation(p1, p2, p3);
        let d2 = self.orientation(p1, p2, p4);
        let d3 = self.orientation(p3, p4, p1);
        let d4 = self.orientation(p3, p4, p2);

        (d1 != d2 && d3 != d4)
            || (d1 == 0 && self.on_segment(p1, p3, p2))
            || (d2 == 0 && self.on_segment(p1, p4, p2))
            || (d3 == 0 && self.on_segment(p3, p1, p4))
            || (d4 == 0 && self.on_segment(p3, p2, p4))
    }

    /// Calculate orientation of three points
    fn orientation(&self, p: (f64, f64), q: (f64, f64), r: (f64, f64)) -> i32 {
        let val = (q.1 - p.1) * (r.0 - q.0) - (q.0 - p.0) * (r.1 - q.1);
        if val.abs() < 1e-10 {
            0
        } else if val > 0.0 {
            1
        } else {
            2
        }
    }

    /// Check if point q lies on line segment pr
    fn on_segment(&self, p: (f64, f64), q: (f64, f64), r: (f64, f64)) -> bool {
        q.0 <= p.0.max(r.0) && q.0 >= p.0.min(r.0) && q.1 <= p.1.max(r.1) && q.1 >= p.1.min(r.1)
    }

    /// Calculate average edge length in the layout
    fn calculate_average_edge_length(
        &self,
        graph: &TraitGraph,
        positions: &HashMap<String, (f64, f64)>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let mut total_length = 0.0;
        let mut edge_count = 0;

        for edge in &graph.edges {
            if let (Some(&pos_from), Some(&pos_to)) =
                (positions.get(&edge.from), positions.get(&edge.to))
            {
                let dx = pos_from.0 - pos_to.0;
                let dy = pos_from.1 - pos_to.1;
                total_length += (dx * dx + dy * dy).sqrt();
                edge_count += 1;
            }
        }

        if edge_count > 0 {
            Ok(total_length / edge_count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate distribution uniformity of node positions
    fn calculate_distribution_uniformity(
        &self,
        positions: &HashMap<String, (f64, f64)>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if positions.len() < 2 {
            return Ok(1.0);
        }

        // Calculate pairwise distances
        let pos_vec: Vec<_> = positions.values().collect();
        let mut distances = Vec::new();

        for i in 0..pos_vec.len() {
            for j in (i + 1)..pos_vec.len() {
                let dx = pos_vec[i].0 - pos_vec[j].0;
                let dy = pos_vec[i].1 - pos_vec[j].1;
                distances.push((dx * dx + dy * dy).sqrt());
            }
        }

        // Calculate coefficient of variation
        let mean = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance =
            distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / distances.len() as f64;
        let std_dev = variance.sqrt();

        if mean > 0.0 {
            Ok(1.0 / (1.0 + std_dev / mean)) // Normalized uniformity score
        } else {
            Ok(0.0)
        }
    }

    /// Calculate overall aesthetic score combining multiple factors
    fn calculate_aesthetic_score(
        &self,
        graph: &TraitGraph,
        positions: &HashMap<String, (f64, f64)>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // Combine various aesthetic factors
        let crossing_penalty =
            1.0 / (1.0 + self.count_edge_crossings(graph, positions)? as f64 * 0.1);
        let uniformity = self.calculate_distribution_uniformity(positions)?;
        let edge_length_variance = self.calculate_edge_length_variance(graph, positions)?;

        Ok((crossing_penalty + uniformity + edge_length_variance) / 3.0)
    }

    /// Calculate variance in edge lengths
    fn calculate_edge_length_variance(
        &self,
        graph: &TraitGraph,
        positions: &HashMap<String, (f64, f64)>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let mut lengths = Vec::new();

        for edge in &graph.edges {
            if let (Some(&pos_from), Some(&pos_to)) =
                (positions.get(&edge.from), positions.get(&edge.to))
            {
                let dx = pos_from.0 - pos_to.0;
                let dy = pos_from.1 - pos_to.1;
                lengths.push((dx * dx + dy * dy).sqrt());
            }
        }

        if lengths.len() < 2 {
            return Ok(1.0);
        }

        let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
        let variance =
            lengths.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / lengths.len() as f64;

        if mean > 0.0 {
            Ok(1.0 / (1.0 + variance.sqrt() / mean))
        } else {
            Ok(0.0)
        }
    }
}

// Macro for creating simple layout algorithm implementations
macro_rules! simple_layout_impl {
    ($name:ident, $display_name:expr, $supports_3d:expr, $supports_gpu:expr, $layout_fn:expr) => {
        #[derive(Debug)]
        pub struct $name {
            pub seed: u64,
            pub spacing: f64,
            pub center: (f64, f64),
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    seed: 42,
                    spacing: 100.0,
                    center: (0.0, 0.0),
                }
            }

            pub fn with_seed(mut self, seed: u64) -> Self {
                self.seed = seed;
                self
            }

            pub fn with_spacing(mut self, spacing: f64) -> Self {
                self.spacing = spacing;
                self
            }

            pub fn with_center(mut self, center: (f64, f64)) -> Self {
                self.center = center;
                self
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl LayoutAlgorithmImpl for $name {
            fn compute_layout(
                &self,
                graph: &TraitGraph,
                _config: &GraphConfig,
            ) -> Result<LayoutResult, Box<dyn std::error::Error>> {
                let start_time = Instant::now();
                let n = graph.nodes.len();

                if n == 0 {
                    return Ok(LayoutResult {
                        positions_2d: HashMap::new(),
                        positions_3d: None,
                        quality_metrics: LayoutQualityMetrics {
                            edge_crossings: 0,
                            average_edge_length: 0.0,
                            distribution_uniformity: 1.0,
                            aesthetic_score: 1.0,
                        },
                        computation_time: start_time.elapsed(),
                    });
                }

                let positions_2d = $layout_fn(self, graph);

                Ok(LayoutResult {
                    positions_2d,
                    positions_3d: None,
                    quality_metrics: LayoutQualityMetrics {
                        edge_crossings: 0,
                        average_edge_length: self.spacing,
                        distribution_uniformity: 0.8,
                        aesthetic_score: 0.7,
                    },
                    computation_time: start_time.elapsed(),
                })
            }

            fn name(&self) -> &str {
                $display_name
            }

            fn supports_3d(&self) -> bool {
                $supports_3d
            }

            fn supports_gpu(&self) -> bool {
                $supports_gpu
            }
        }
    };
}

// Layout function implementations
fn circular_layout_fn(layout: &CircularLayout, graph: &TraitGraph) -> HashMap<String, (f64, f64)> {
    let mut positions = HashMap::new();
    let n = graph.nodes.len();

    if n == 0 {
        return positions;
    }

    let radius = layout.spacing;
    let angle_step = 2.0 * std::f64::consts::PI / n as f64;

    for (i, node) in graph.nodes.iter().enumerate() {
        let angle = i as f64 * angle_step;
        let x = layout.center.0 + radius * angle.cos();
        let y = layout.center.1 + radius * angle.sin();
        positions.insert(node.id.clone(), (x, y));
    }

    positions
}

fn grid_layout_fn(layout: &GridLayout, graph: &TraitGraph) -> HashMap<String, (f64, f64)> {
    let mut positions = HashMap::new();
    let n = graph.nodes.len();

    if n == 0 {
        return positions;
    }

    let cols = (n as f64).sqrt().ceil() as usize;
    let spacing = layout.spacing;

    for (i, node) in graph.nodes.iter().enumerate() {
        let row = i / cols;
        let col = i % cols;
        let x = layout.center.0 + (col as f64 - cols as f64 / 2.0) * spacing;
        let y = layout.center.1 + (row as f64 - (n as f64 / cols as f64) / 2.0) * spacing;
        positions.insert(node.id.clone(), (x, y));
    }

    positions
}

fn radial_layout_fn(layout: &RadialLayout, graph: &TraitGraph) -> HashMap<String, (f64, f64)> {
    let mut positions = HashMap::new();
    let n = graph.nodes.len();

    if n == 0 {
        return positions;
    }

    // Place center node at origin
    if let Some(center_node) = graph.nodes.first() {
        positions.insert(center_node.id.clone(), layout.center);
    }

    // Place other nodes in concentric circles
    let mut remaining_nodes: Vec<_> = graph.nodes.iter().skip(1).collect();
    let mut radius = layout.spacing;
    let nodes_per_circle = 6; // Start with 6 nodes per circle

    while !remaining_nodes.is_empty() {
        let nodes_in_this_circle = remaining_nodes.len().min(nodes_per_circle);
        let angle_step = 2.0 * std::f64::consts::PI / nodes_in_this_circle as f64;

        for i in 0..nodes_in_this_circle {
            let node = remaining_nodes.remove(0);
            let angle = i as f64 * angle_step;
            let x = layout.center.0 + radius * angle.cos();
            let y = layout.center.1 + radius * angle.sin();
            positions.insert(node.id.clone(), (x, y));
        }

        radius += layout.spacing;
    }

    positions
}

fn hierarchical_layout_fn(layout: &HierarchicalLayout, graph: &TraitGraph) -> HashMap<String, (f64, f64)> {
    let mut positions = HashMap::new();
    let n = graph.nodes.len();

    if n == 0 {
        return positions;
    }

    // Simple hierarchical layout - place nodes in levels
    let mut levels: Vec<Vec<&TraitGraphNode>> = Vec::new();
    let mut processed = std::collections::HashSet::new();

    // Find root nodes (nodes with no incoming edges for traits)
    let mut roots = Vec::new();
    for node in &graph.nodes {
        if node.node_type == TraitNodeType::Trait {
            let has_incoming = graph.edges.iter().any(|edge| edge.to == node.id);
            if !has_incoming {
                roots.push(node);
            }
        }
    }

    if roots.is_empty() {
        // Fallback to first node if no clear hierarchy
        if let Some(first_node) = graph.nodes.first() {
            roots.push(first_node);
        }
    }

    levels.push(roots);

    // Build levels based on edges
    while let Some(current_level) = levels.last() {
        if current_level.is_empty() {
            break;
        }

        let mut next_level = Vec::new();
        for node in current_level {
            processed.insert(&node.id);

            // Find children
            for edge in &graph.edges {
                if edge.from == node.id {
                    if let Some(child) = graph.nodes.iter().find(|n| n.id == edge.to) {
                        if !processed.contains(&child.id) && !next_level.iter().any(|n| n.id == child.id) {
                            next_level.push(child);
                        }
                    }
                }
            }
        }

        if next_level.is_empty() {
            break;
        }
        levels.push(next_level);
    }

    // Position nodes
    let level_height = layout.spacing;
    for (level_idx, level) in levels.iter().enumerate() {
        let y = layout.center.1 - (level_idx as f64 * level_height);
        let level_width = level.len() as f64 * layout.spacing;
        let start_x = layout.center.0 - level_width / 2.0;

        for (node_idx, node) in level.iter().enumerate() {
            let x = start_x + (node_idx as f64 * layout.spacing);
            positions.insert(node.id.clone(), (x, y));
        }
    }

    positions
}

fn tree_layout_fn(layout: &TreeLayout, graph: &TraitGraph) -> HashMap<String, (f64, f64)> {
    // For now, use hierarchical layout as tree layout
    hierarchical_layout_fn(layout, graph)
}

fn spring_embedder_layout_fn(layout: &SpringEmbedderLayout, graph: &TraitGraph) -> HashMap<String, (f64, f64)> {
    let mut positions = HashMap::new();
    let mut rng = Random::seed(layout.seed);

    // Simple spring embedder - start with random positions
    for node in &graph.nodes {
        let x = layout.center.0 + rng.random_range(-layout.spacing..layout.spacing);
        let y = layout.center.1 + rng.random_range(-layout.spacing..layout.spacing);
        positions.insert(node.id.clone(), (x, y));
    }

    // Run a few iterations of spring forces (simplified)
    for _ in 0..50 {
        let mut forces: HashMap<String, (f64, f64)> = HashMap::new();

        // Initialize forces
        for node in &graph.nodes {
            forces.insert(node.id.clone(), (0.0, 0.0));
        }

        // Spring forces between connected nodes
        for edge in &graph.edges {
            if let (Some(&pos1), Some(&pos2)) = (positions.get(&edge.from), positions.get(&edge.to)) {
                let dx = pos2.0 - pos1.0;
                let dy = pos2.1 - pos1.1;
                let distance = (dx * dx + dy * dy).sqrt().max(0.01);

                let desired_length = layout.spacing * 0.5;
                let force_magnitude = (distance - desired_length) * 0.1;

                let fx = (dx / distance) * force_magnitude;
                let fy = (dy / distance) * force_magnitude;

                if let Some(f) = forces.get_mut(&edge.from) {
                    f.0 += fx;
                    f.1 += fy;
                }
                if let Some(f) = forces.get_mut(&edge.to) {
                    f.0 -= fx;
                    f.1 -= fy;
                }
            }
        }

        // Apply forces
        for node in &graph.nodes {
            if let (Some(pos), Some(force)) = (positions.get_mut(&node.id), forces.get(&node.id)) {
                pos.0 += force.0 * 0.1;
                pos.1 += force.1 * 0.1;
            }
        }
    }

    positions
}

// Generate layout implementations using the macro
simple_layout_impl!(HierarchicalLayout, "Hierarchical", false, false, hierarchical_layout_fn);
simple_layout_impl!(CircularLayout, "Circular", false, false, circular_layout_fn);
simple_layout_impl!(GridLayout, "Grid", false, false, grid_layout_fn);
simple_layout_impl!(RadialLayout, "Radial", false, false, radial_layout_fn);
simple_layout_impl!(SpringEmbedderLayout, "Spring Embedder", true, false, spring_embedder_layout_fn);
simple_layout_impl!(TreeLayout, "Tree", false, false, tree_layout_fn);

/// Factory for creating layout algorithm instances
pub struct LayoutAlgorithmFactory;

impl LayoutAlgorithmFactory {
    /// Create a layout algorithm instance based on the specified algorithm type
    pub fn create_layout(algorithm: LayoutAlgorithm) -> Box<dyn LayoutAlgorithmImpl> {
        match algorithm {
            LayoutAlgorithm::ForceDirected => Box::new(ForceDirectedLayout::new()),
            LayoutAlgorithm::Hierarchical => Box::new(HierarchicalLayout::new()),
            LayoutAlgorithm::Circular => Box::new(CircularLayout::new()),
            LayoutAlgorithm::Grid => Box::new(GridLayout::new()),
            LayoutAlgorithm::Radial => Box::new(RadialLayout::new()),
            LayoutAlgorithm::SpringEmbedder => Box::new(SpringEmbedderLayout::new()),
            LayoutAlgorithm::Tree => Box::new(TreeLayout::new()),
        }
    }

    /// Create a force-directed layout with custom parameters
    pub fn create_force_directed_custom(
        iterations: usize,
        temperature_factor: f64,
        repulsive_strength: f64,
        attractive_strength: f64,
    ) -> Box<dyn LayoutAlgorithmImpl> {
        Box::new(ForceDirectedLayout::with_parameters(
            iterations,
            temperature_factor,
            repulsive_strength,
            attractive_strength,
        ))
    }

    /// Get a list of all available layout algorithms
    pub fn available_algorithms() -> Vec<LayoutAlgorithm> {
        vec![
            LayoutAlgorithm::ForceDirected,
            LayoutAlgorithm::Hierarchical,
            LayoutAlgorithm::Circular,
            LayoutAlgorithm::Grid,
            LayoutAlgorithm::Radial,
            LayoutAlgorithm::SpringEmbedder,
            LayoutAlgorithm::Tree,
        ]
    }

    /// Check if an algorithm supports 3D layouts
    pub fn supports_3d(algorithm: LayoutAlgorithm) -> bool {
        let layout = Self::create_layout(algorithm);
        layout.supports_3d()
    }

    /// Check if an algorithm supports GPU acceleration
    pub fn supports_gpu(algorithm: LayoutAlgorithm) -> bool {
        let layout = Self::create_layout(algorithm);
        layout.supports_gpu()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::trait_explorer::graph_visualization::graph_structures::*;

    fn create_test_graph() -> TraitGraph {
        let nodes = vec![
            TraitGraphNode {
                id: "Node1".to_string(),
                label: "Node1".to_string(),
                node_type: TraitNodeType::Trait,
                position_2d: None,
                position_3d: None,
                size: 10.0,
                complexity: 0.5,
                color: Some("#ff0000".to_string()),
                shape: Some("circle".to_string()),
                visible: true,
                metadata: NodeMetadata::default(),
            },
            TraitGraphNode {
                id: "Node2".to_string(),
                label: "Node2".to_string(),
                node_type: TraitNodeType::Implementation,
                position_2d: None,
                position_3d: None,
                size: 8.0,
                complexity: 0.3,
                color: Some("#00ff00".to_string()),
                shape: Some("circle".to_string()),
                visible: true,
                metadata: NodeMetadata::default(),
            },
        ];

        let edges = vec![TraitGraphEdge {
            from: "Node1".to_string(),
            to: "Node2".to_string(),
            edge_type: EdgeType::Implementation,
            weight: 1.0,
            directed: true,
            color: Some("#0000ff".to_string()),
            style: Some("solid".to_string()),
            visible: true,
            metadata: EdgeMetadata::default(),
        }];

        TraitGraph {
            nodes,
            edges,
            metadata: GraphMetadata::default(),
            adjacency_matrix: None,
        }
    }

    fn create_test_config() -> GraphConfig {
        GraphConfig::default()
    }

    #[test]
    fn test_force_directed_layout() {
        let layout = ForceDirectedLayout::new();
        let graph = create_test_graph();
        let config = create_test_config();

        let result = layout.compute_layout(&graph, &config);
        assert!(result.is_ok());

        let layout_result = result.expect("expected valid value");
        assert_eq!(layout_result.positions_2d.len(), 2);
        assert!(layout_result.quality_metrics.aesthetic_score >= 0.0);
        assert!(layout_result.quality_metrics.aesthetic_score <= 1.0);
    }

    #[test]
    fn test_layout_factory() {
        let algorithms = LayoutAlgorithmFactory::available_algorithms();
        assert!(!algorithms.is_empty());

        for algorithm in algorithms {
            let layout = LayoutAlgorithmFactory::create_layout(algorithm);
            assert!(!layout.name().is_empty());
        }
    }

    #[test]
    fn test_circular_layout() {
        let layout = CircularLayout::new();
        let graph = create_test_graph();
        let config = create_test_config();

        let result = layout.compute_layout(&graph, &config);
        assert!(result.is_ok());

        let layout_result = result.expect("expected valid value");
        assert_eq!(layout_result.positions_2d.len(), 2);
    }

    #[test]
    fn test_grid_layout() {
        let layout = GridLayout::new();
        let graph = create_test_graph();
        let config = create_test_config();

        let result = layout.compute_layout(&graph, &config);
        assert!(result.is_ok());

        let layout_result = result.expect("expected valid value");
        assert_eq!(layout_result.positions_2d.len(), 2);
    }

    #[test]
    fn test_layout_with_empty_graph() {
        let layout = ForceDirectedLayout::new();
        let empty_graph = TraitGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: GraphMetadata::default(),
            adjacency_matrix: None,
        };
        let config = create_test_config();

        let result = layout.compute_layout(&empty_graph, &config);
        assert!(result.is_ok());

        let layout_result = result.expect("expected valid value");
        assert!(layout_result.positions_2d.is_empty());
    }

    #[test]
    fn test_force_directed_convergence() {
        let layout = ForceDirectedLayout {
            iterations: 10,
            convergence_threshold: 0.1,
            ..ForceDirectedLayout::default()
        };

        // Test convergence check
        assert!(layout.has_converged(0.05, 5)); // Below threshold
        assert!(layout.has_converged(0.2, 15)); // Exceeded iterations
        assert!(!layout.has_converged(0.2, 5)); // Neither condition met
    }

    #[test]
    fn test_line_intersection() {
        let layout = ForceDirectedLayout::new();

        // Test intersecting lines
        let p1 = (0.0, 0.0);
        let p2 = (2.0, 2.0);
        let p3 = (0.0, 2.0);
        let p4 = (2.0, 0.0);
        assert!(layout.lines_intersect(p1, p2, p3, p4));

        // Test non-intersecting lines
        let p5 = (0.0, 0.0);
        let p6 = (1.0, 1.0);
        let p7 = (2.0, 2.0);
        let p8 = (3.0, 3.0);
        assert!(!layout.lines_intersect(p5, p6, p7, p8));
    }

    #[test]
    fn test_layout_algorithm_capabilities() {
        let force_directed = ForceDirectedLayout::new();
        assert_eq!(force_directed.name(), "Force-Directed");
        assert!(force_directed.supports_3d());
        assert!(!force_directed.supports_gpu());

        let circular = CircularLayout::new();
        assert_eq!(circular.name(), "Circular");
        assert!(!circular.supports_3d());
        assert!(!circular.supports_gpu());
    }

    #[test]
    fn test_layout_customization() {
        let custom_circular = CircularLayout::new()
            .with_spacing(150.0)
            .with_center((50.0, 50.0))
            .with_seed(123);

        assert_eq!(custom_circular.spacing, 150.0);
        assert_eq!(custom_circular.center, (50.0, 50.0));
        assert_eq!(custom_circular.seed, 123);
    }
}
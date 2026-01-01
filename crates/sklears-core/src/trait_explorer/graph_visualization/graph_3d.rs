//! 3D visualization for trait relationship graphs
//!
//! This module provides advanced 3D visualization capabilities using Three.js
//! and WebGL, supporting interactive 3D graph exploration, physics simulation,
//! and immersive visualization experiences.

use super::graph_config::{GraphConfig, VisualizationTheme};
use super::graph_structures::{TraitGraph, TraitGraphNode, TraitGraphEdge, PerformanceMetrics};
use crate::error::{Result, SklearsError};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Configuration for 3D visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeDConfig {
    /// Enable physics simulation
    pub enable_physics: bool,
    /// Enable camera controls
    pub enable_camera_controls: bool,
    /// Enable VR/AR support
    pub enable_vr: bool,
    /// Field of view for the camera
    pub camera_fov: f64,
    /// Near clipping plane
    pub camera_near: f64,
    /// Far clipping plane
    pub camera_far: f64,
    /// Initial camera position
    pub camera_position: (f64, f64, f64),
    /// Camera target/look-at point
    pub camera_target: (f64, f64, f64),
    /// Ambient light intensity
    pub ambient_light: f64,
    /// Directional light intensity
    pub directional_light: f64,
    /// Fog enabled
    pub enable_fog: bool,
    /// Fog color
    pub fog_color: String,
    /// Fog density
    pub fog_density: f64,
    /// Animation speed
    pub animation_speed: f64,
    /// Physics gravity
    pub physics_gravity: f64,
    /// Physics damping
    pub physics_damping: f64,
    /// Node depth separation
    pub depth_separation: f64,
    /// Enable shadows
    pub enable_shadows: bool,
    /// Background type
    pub background_type: BackgroundType,
    /// Background color or texture
    pub background_value: String,
}

impl Default for ThreeDConfig {
    fn default() -> Self {
        Self {
            enable_physics: true,
            enable_camera_controls: true,
            enable_vr: false,
            camera_fov: 75.0,
            camera_near: 0.1,
            camera_far: 1000.0,
            camera_position: (0.0, 0.0, 500.0),
            camera_target: (0.0, 0.0, 0.0),
            ambient_light: 0.4,
            directional_light: 0.8,
            enable_fog: true,
            fog_color: "#ffffff".to_string(),
            fog_density: 0.0025,
            animation_speed: 1.0,
            physics_gravity: 0.1,
            physics_damping: 0.9,
            depth_separation: 100.0,
            enable_shadows: true,
            background_type: BackgroundType::SolidColor,
            background_value: "#f0f0f0".to_string(),
        }
    }
}

/// Background types for 3D scenes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackgroundType {
    /// Solid color background
    SolidColor,
    /// Gradient background
    Gradient,
    /// Skybox with cube map
    Skybox,
    /// Environment map
    Environment,
    /// Procedural background
    Procedural,
}

/// Builder for 3D configuration
#[derive(Debug)]
pub struct ThreeDConfigBuilder {
    config: ThreeDConfig,
}

impl ThreeDConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ThreeDConfig::default(),
        }
    }

    /// Enable or disable physics simulation
    pub fn with_physics_simulation(mut self, enable: bool) -> Self {
        self.config.enable_physics = enable;
        self
    }

    /// Enable or disable camera controls
    pub fn with_camera_controls(mut self, enable: bool) -> Self {
        self.config.enable_camera_controls = enable;
        self
    }

    /// Set camera field of view
    pub fn with_camera_fov(mut self, fov: f64) -> Self {
        self.config.camera_fov = fov;
        self
    }

    /// Set camera position
    pub fn with_camera_position(mut self, x: f64, y: f64, z: f64) -> Self {
        self.config.camera_position = (x, y, z);
        self
    }

    /// Set lighting configuration
    pub fn with_lighting(mut self, ambient: f64, directional: f64) -> Self {
        self.config.ambient_light = ambient;
        self.config.directional_light = directional;
        self
    }

    /// Enable fog with density
    pub fn with_fog(mut self, enable: bool, color: String, density: f64) -> Self {
        self.config.enable_fog = enable;
        self.config.fog_color = color;
        self.config.fog_density = density;
        self
    }

    /// Set physics parameters
    pub fn with_physics_params(mut self, gravity: f64, damping: f64) -> Self {
        self.config.physics_gravity = gravity;
        self.config.physics_damping = damping;
        self
    }

    /// Set background
    pub fn with_background(mut self, bg_type: BackgroundType, value: String) -> Self {
        self.config.background_type = bg_type;
        self.config.background_value = value;
        self
    }

    /// Build the configuration
    pub fn build(self) -> ThreeDConfig {
        self.config
    }
}

/// 3D visualization generator
pub struct Graph3DGenerator {
    config: ThreeDConfig,
    theme: VisualizationTheme,
    node_geometries: HashMap<String, NodeGeometry>,
    material_templates: HashMap<String, MaterialTemplate>,
}

/// 3D node geometry configuration
#[derive(Debug, Clone)]
pub struct NodeGeometry {
    /// Geometry type
    pub geometry_type: GeometryType,
    /// Size parameters
    pub size: (f64, f64, f64),
    /// Segments for curved geometries
    pub segments: u32,
    /// Custom parameters
    pub parameters: HashMap<String, f64>,
}

/// Types of 3D geometries for nodes
#[derive(Debug, Clone)]
pub enum GeometryType {
    /// Sphere geometry
    Sphere,
    /// Box/cube geometry
    Box,
    /// Cylinder geometry
    Cylinder,
    /// Cone geometry
    Cone,
    /// Octahedron geometry
    Octahedron,
    /// Icosahedron geometry
    Icosahedron,
    /// Dodecahedron geometry
    Dodecahedron,
    /// Torus geometry
    Torus,
    /// Custom geometry with vertices
    Custom,
}

/// Material template for 3D objects
#[derive(Debug, Clone)]
pub struct MaterialTemplate {
    /// Material type
    pub material_type: MaterialType,
    /// Base color
    pub color: String,
    /// Emission color
    pub emissive: String,
    /// Metalness factor (0.0-1.0)
    pub metalness: f64,
    /// Roughness factor (0.0-1.0)
    pub roughness: f64,
    /// Opacity (0.0-1.0)
    pub opacity: f64,
    /// Whether material is transparent
    pub transparent: bool,
    /// Texture URL
    pub texture: Option<String>,
    /// Normal map URL
    pub normal_map: Option<String>,
    /// Environment map URL
    pub env_map: Option<String>,
}

/// Types of 3D materials
#[derive(Debug, Clone)]
pub enum MaterialType {
    /// Basic material (unaffected by lights)
    Basic,
    /// Lambert material (diffuse reflection)
    Lambert,
    /// Phong material (specular reflection)
    Phong,
    /// Standard material (PBR)
    Standard,
    /// Physical material (advanced PBR)
    Physical,
    /// Toon material (cartoon shading)
    Toon,
    /// Points material
    Points,
    /// Line material
    Line,
}

/// 3D layout algorithms
#[derive(Debug, Clone)]
pub enum Layout3DAlgorithm {
    Force3D,
    Spherical,
    Cylindrical,
    Helical,
    Layered,
    Tree3D,
    Custom,
}

/// Result of 3D layout computation
#[derive(Debug, Clone)]
pub struct Layout3DResult {
    /// 3D positions for nodes
    pub positions: HashMap<String, (f64, f64, f64)>,
    /// Rotations for nodes
    pub rotations: HashMap<String, (f64, f64, f64)>,
    /// Quality metrics
    pub quality_metrics: Layout3DQualityMetrics,
    /// Computation time
    pub computation_time: std::time::Duration,
}

/// Quality metrics for 3D layouts
#[derive(Debug, Clone)]
pub struct Layout3DQualityMetrics {
    /// Volume utilization (0.0-1.0)
    pub volume_utilization: f64,
    /// Average 3D edge length
    pub average_edge_length: f64,
    /// 3D edge crossings/intersections
    pub edge_intersections: usize,
    /// Spatial distribution uniformity
    pub distribution_uniformity: f64,
    /// Visual clarity score
    pub visual_clarity: f64,
    /// Depth perception score
    pub depth_perception: f64,
}

impl Graph3DGenerator {
    /// Create a new 3D generator
    pub fn new(config: ThreeDConfig, theme: VisualizationTheme) -> Self {
        let mut generator = Self {
            config,
            theme,
            node_geometries: HashMap::new(),
            material_templates: HashMap::new(),
        };

        generator.initialize_default_geometries();
        generator.initialize_material_templates();
        generator
    }

    /// Generate 3D interactive HTML
    pub fn generate_3d_html(&self, graph: &TraitGraph) -> Result<String> {
        let layout_result = self.compute_3d_layout(graph)?;
        let three_js_code = self.generate_threejs_code(graph, &layout_result)?;
        let html_template = self.create_html_template(&three_js_code);

        Ok(html_template)
    }

    /// Compute 3D layout for the graph
    pub fn compute_3d_layout(&self, graph: &TraitGraph) -> Result<Layout3DResult> {
        let start_time = Instant::now();
        let algorithm = Layout3DAlgorithm::Force3D; // Default algorithm

        let positions = match algorithm {
            Layout3DAlgorithm::Force3D => self.compute_force_3d_layout(graph)?,
            Layout3DAlgorithm::Spherical => self.compute_spherical_layout(graph)?,
            Layout3DAlgorithm::Cylindrical => self.compute_cylindrical_layout(graph)?,
            Layout3DAlgorithm::Helical => self.compute_helical_layout(graph)?,
            Layout3DAlgorithm::Layered => self.compute_layered_layout(graph)?,
            Layout3DAlgorithm::Tree3D => self.compute_tree_3d_layout(graph)?,
            Layout3DAlgorithm::Custom => self.compute_custom_layout(graph)?,
        };

        let rotations = self.compute_node_rotations(graph, &positions);
        let quality_metrics = self.evaluate_layout_quality(graph, &positions);

        Ok(Layout3DResult {
            positions,
            rotations,
            quality_metrics,
            computation_time: start_time.elapsed(),
        })
    }

    /// Initialize default geometries for different node types
    fn initialize_default_geometries(&mut self) {
        use super::graph_config::TraitNodeType;

        // Trait nodes - spheres
        self.node_geometries.insert(
            TraitNodeType::Trait.display_name().to_string(),
            NodeGeometry {
                geometry_type: GeometryType::Sphere,
                size: (1.0, 1.0, 1.0),
                segments: 32,
                parameters: HashMap::new(),
            },
        );

        // Implementation nodes - boxes
        self.node_geometries.insert(
            TraitNodeType::Implementation.display_name().to_string(),
            NodeGeometry {
                geometry_type: GeometryType::Box,
                size: (1.0, 1.0, 1.0),
                segments: 1,
                parameters: HashMap::new(),
            },
        );

        // Associated type nodes - octahedrons
        self.node_geometries.insert(
            TraitNodeType::AssociatedType.display_name().to_string(),
            NodeGeometry {
                geometry_type: GeometryType::Octahedron,
                size: (1.0, 1.0, 1.0),
                segments: 0,
                parameters: HashMap::new(),
            },
        );

        // Method nodes - cylinders
        self.node_geometries.insert(
            TraitNodeType::Method.display_name().to_string(),
            NodeGeometry {
                geometry_type: GeometryType::Cylinder,
                size: (0.5, 1.0, 0.5),
                segments: 16,
                parameters: HashMap::new(),
            },
        );
    }

    /// Initialize material templates
    fn initialize_material_templates(&mut self) {
        use super::graph_config::TraitNodeType;

        let base_materials = match self.theme {
            VisualizationTheme::Dark => vec![
                (TraitNodeType::Trait, "#66b3ff", 0.2, 0.3),
                (TraitNodeType::Implementation, "#99ff99", 0.1, 0.4),
                (TraitNodeType::AssociatedType, "#ffcc99", 0.3, 0.2),
                (TraitNodeType::Method, "#ff99cc", 0.0, 0.5),
            ],
            _ => vec![
                (TraitNodeType::Trait, "#007bff", 0.2, 0.3),
                (TraitNodeType::Implementation, "#28a745", 0.1, 0.4),
                (TraitNodeType::AssociatedType, "#fd7e14", 0.3, 0.2),
                (TraitNodeType::Method, "#e83e8c", 0.0, 0.5),
            ],
        };

        for (node_type, color, metalness, roughness) in base_materials {
            self.material_templates.insert(
                node_type.display_name().to_string(),
                MaterialTemplate {
                    material_type: MaterialType::Standard,
                    color: color.to_string(),
                    emissive: "#000000".to_string(),
                    metalness,
                    roughness,
                    opacity: 1.0,
                    transparent: false,
                    texture: None,
                    normal_map: None,
                    env_map: None,
                },
            );
        }
    }

    /// Compute force-directed 3D layout
    fn compute_force_3d_layout(&self, graph: &TraitGraph) -> Result<HashMap<String, (f64, f64, f64)>> {
        let mut positions = HashMap::new();
        let n = graph.nodes.len();

        if n == 0 {
            return Ok(positions);
        }

        // Initialize random positions in 3D space
        let mut rng = scirs2_core::random::Random::seed(42);
        for node in &graph.nodes {
            let x = rng.random_range(-200.0..200.0);
            let y = rng.random_range(-200.0..200.0);
            let z = rng.random_range(-200.0..200.0);
            positions.insert(node.id.clone(), (x, y, z));
        }

        // Force-directed simulation in 3D
        let iterations = 300;
        let k = (1000.0 / n as f64).cbrt(); // Optimal 3D distance

        for iteration in 0..iterations {
            let temperature = 1.0 - (iteration as f64 / iterations as f64);
            let mut forces: HashMap<String, (f64, f64, f64)> = HashMap::new();

            // Initialize forces
            for node in &graph.nodes {
                forces.insert(node.id.clone(), (0.0, 0.0, 0.0));
            }

            // Repulsive forces
            for i in 0..graph.nodes.len() {
                for j in i+1..graph.nodes.len() {
                    let node1 = &graph.nodes[i];
                    let node2 = &graph.nodes[j];

                    if let (Some(&(x1, y1, z1)), Some(&(x2, y2, z2))) = (
                        positions.get(&node1.id),
                        positions.get(&node2.id)
                    ) {
                        let dx = x1 - x2;
                        let dy = y1 - y2;
                        let dz = z1 - z2;
                        let distance = (dx*dx + dy*dy + dz*dz).sqrt().max(0.1);

                        let force = k * k / distance;
                        let fx = force * dx / distance;
                        let fy = force * dy / distance;
                        let fz = force * dz / distance;

                        // Apply force to both nodes
                        let f1 = forces.get_mut(&node1.id).unwrap();
                        f1.0 += fx;
                        f1.1 += fy;
                        f1.2 += fz;

                        let f2 = forces.get_mut(&node2.id).unwrap();
                        f2.0 -= fx;
                        f2.1 -= fy;
                        f2.2 -= fz;
                    }
                }
            }

            // Attractive forces for connected nodes
            for edge in &graph.edges {
                if let (Some(&(x1, y1, z1)), Some(&(x2, y2, z2))) = (
                    positions.get(&edge.from),
                    positions.get(&edge.to)
                ) {
                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    let dz = z2 - z1;
                    let distance = (dx*dx + dy*dy + dz*dz).sqrt().max(0.1);

                    let force = distance * distance / k;
                    let fx = force * dx / distance;
                    let fy = force * dy / distance;
                    let fz = force * dz / distance;

                    // Apply attractive force
                    let f1 = forces.get_mut(&edge.from).unwrap();
                    f1.0 += fx;
                    f1.1 += fy;
                    f1.2 += fz;

                    let f2 = forces.get_mut(&edge.to).unwrap();
                    f2.0 -= fx;
                    f2.1 -= fy;
                    f2.2 -= fz;
                }
            }

            // Apply forces with cooling
            let cooling = temperature * 50.0;
            for node in &graph.nodes {
                if let Some(&(fx, fy, fz)) = forces.get(&node.id) {
                    let force_magnitude = (fx*fx + fy*fy + fz*fz).sqrt();
                    if force_magnitude > 0.0 {
                        let displacement = cooling.min(force_magnitude);
                        let scale = displacement / force_magnitude;

                        let pos = positions.get_mut(&node.id).unwrap();
                        pos.0 += fx * scale;
                        pos.1 += fy * scale;
                        pos.2 += fz * scale;
                    }
                }
            }
        }

        Ok(positions)
    }

    /// Compute spherical layout
    fn compute_spherical_layout(&self, graph: &TraitGraph) -> Result<HashMap<String, (f64, f64, f64)>> {
        let mut positions = HashMap::new();
        let n = graph.nodes.len();

        if n == 0 {
            return Ok(positions);
        }

        let radius = 150.0;

        // Use golden spiral distribution for even spacing on sphere
        let golden_angle = std::f64::consts::PI * (3.0 - (5.0_f64).sqrt());

        for (i, node) in graph.nodes.iter().enumerate() {
            let y = 1.0 - (i as f64 / (n - 1) as f64) * 2.0; // y from 1 to -1
            let radius_at_y = (1.0 - y * y).sqrt();

            let theta = golden_angle * i as f64;

            let x = radius * radius_at_y * theta.cos();
            let z = radius * radius_at_y * theta.sin();
            let y = radius * y;

            positions.insert(node.id.clone(), (x, y, z));
        }

        Ok(positions)
    }

    /// Compute cylindrical layout
    fn compute_cylindrical_layout(&self, graph: &TraitGraph) -> Result<HashMap<String, (f64, f64, f64)>> {
        let mut positions = HashMap::new();
        let n = graph.nodes.len();

        if n == 0 {
            return Ok(positions);
        }

        let radius = 120.0;
        let height_per_level = 80.0;

        // Group nodes by type for different levels
        let mut trait_nodes = Vec::new();
        let mut impl_nodes = Vec::new();
        let mut other_nodes = Vec::new();

        for node in &graph.nodes {
            match node.node_type {
                super::graph_config::TraitNodeType::Trait => trait_nodes.push(node),
                super::graph_config::TraitNodeType::Implementation => impl_nodes.push(node),
                _ => other_nodes.push(node),
            }
        }

        let levels = vec![&trait_nodes, &impl_nodes, &other_nodes];

        for (level_idx, level_nodes) in levels.iter().enumerate() {
            let y = level_idx as f64 * height_per_level - height_per_level;
            let nodes_in_level = level_nodes.len();

            if nodes_in_level > 0 {
                let angle_step = 2.0 * std::f64::consts::PI / nodes_in_level as f64;

                for (i, node) in level_nodes.iter().enumerate() {
                    let angle = i as f64 * angle_step;
                    let x = radius * angle.cos();
                    let z = radius * angle.sin();

                    positions.insert(node.id.clone(), (x, y, z));
                }
            }
        }

        Ok(positions)
    }

    /// Compute helical layout
    fn compute_helical_layout(&self, graph: &TraitGraph) -> Result<HashMap<String, (f64, f64, f64)>> {
        let mut positions = HashMap::new();
        let n = graph.nodes.len();

        if n == 0 {
            return Ok(positions);
        }

        let radius = 100.0;
        let height_per_turn = 200.0;
        let turns = (n as f64 / 20.0).max(1.0); // At least 1 turn, more for larger graphs

        for (i, node) in graph.nodes.iter().enumerate() {
            let t = i as f64 / (n - 1) as f64; // Parameter from 0 to 1
            let angle = t * turns * 2.0 * std::f64::consts::PI;

            let x = radius * angle.cos();
            let z = radius * angle.sin();
            let y = t * height_per_turn - height_per_turn / 2.0;

            positions.insert(node.id.clone(), (x, y, z));
        }

        Ok(positions)
    }

    /// Compute layered 3D layout
    fn compute_layered_layout(&self, graph: &TraitGraph) -> Result<HashMap<String, (f64, f64, f64)>> {
        let mut positions = HashMap::new();

        // Simple layered approach based on node types
        let layer_spacing = 150.0;
        let node_spacing = 80.0;

        let mut layers: HashMap<String, Vec<&TraitGraphNode>> = HashMap::new();

        // Group nodes by type
        for node in &graph.nodes {
            let layer_key = node.node_type.display_name().to_string();
            layers.entry(layer_key).or_insert_with(Vec::new).push(node);
        }

        for (layer_idx, (_layer_name, layer_nodes)) in layers.iter().enumerate() {
            let z = layer_idx as f64 * layer_spacing;
            let nodes_per_row = (layer_nodes.len() as f64).sqrt().ceil() as usize;

            for (i, node) in layer_nodes.iter().enumerate() {
                let row = i / nodes_per_row;
                let col = i % nodes_per_row;

                let x = (col as f64 - nodes_per_row as f64 / 2.0) * node_spacing;
                let y = (row as f64 - (layer_nodes.len() as f64 / nodes_per_row as f64) / 2.0) * node_spacing;

                positions.insert(node.id.clone(), (x, y, z));
            }
        }

        Ok(positions)
    }

    /// Compute tree 3D layout
    fn compute_tree_3d_layout(&self, graph: &TraitGraph) -> Result<HashMap<String, (f64, f64, f64)>> {
        let mut positions = HashMap::new();

        // Find root nodes (nodes with no incoming edges of type "inherits")
        let mut root_nodes = Vec::new();
        for node in &graph.nodes {
            let has_incoming_inherits = graph.edges.iter().any(|edge|
                edge.to == node.id && edge.edge_type == super::graph_config::EdgeType::Inherits
            );
            if !has_incoming_inherits {
                root_nodes.push(node);
            }
        }

        if root_nodes.is_empty() && !graph.nodes.is_empty() {
            root_nodes.push(&graph.nodes[0]); // Fallback to first node
        }

        // Simple tree layout with depth-based positioning
        let level_height = 120.0;
        let sibling_spacing = 100.0;

        for (root_idx, root) in root_nodes.iter().enumerate() {
            let root_x = root_idx as f64 * 300.0 - (root_nodes.len() as f64 - 1.0) * 150.0;
            positions.insert(root.id.clone(), (root_x, 0.0, 0.0));

            self.position_tree_children(graph, &root.id, &mut positions, root_x, -level_height, 0, sibling_spacing);
        }

        Ok(positions)
    }

    /// Helper for tree layout - position children recursively
    fn position_tree_children(
        &self,
        graph: &TraitGraph,
        parent_id: &str,
        positions: &mut HashMap<String, (f64, f64, f64)>,
        parent_x: f64,
        level_y: f64,
        child_offset: i32,
        spacing: f64,
    ) {
        let children: Vec<_> = graph.edges.iter()
            .filter(|edge| edge.from == parent_id && edge.edge_type == super::graph_config::EdgeType::Inherits)
            .map(|edge| &edge.to)
            .collect();

        for (child_idx, child_id) in children.iter().enumerate() {
            let child_x = parent_x + (child_idx as f64 - (children.len() as f64 - 1.0) / 2.0) * spacing;
            let child_z = child_offset as f64 * 50.0;

            positions.insert(child_id.to_string(), (child_x, level_y, child_z));

            // Recursively position grandchildren
            self.position_tree_children(
                graph,
                child_id,
                positions,
                child_x,
                level_y - 120.0,
                child_offset + 1,
                spacing * 0.8,
            );
        }
    }

    /// Compute custom layout (placeholder)
    fn compute_custom_layout(&self, _graph: &TraitGraph) -> Result<HashMap<String, (f64, f64, f64)>> {
        // Placeholder for custom layout algorithms
        Ok(HashMap::new())
    }

    /// Compute node rotations based on positions and connections
    fn compute_node_rotations(
        &self,
        graph: &TraitGraph,
        positions: &HashMap<String, (f64, f64, f64)>,
    ) -> HashMap<String, (f64, f64, f64)> {
        let mut rotations = HashMap::new();

        for node in &graph.nodes {
            // Find connected nodes to determine orientation
            let connected_nodes: Vec<_> = graph.edges.iter()
                .filter_map(|edge| {
                    if edge.from == node.id {
                        Some(&edge.to)
                    } else if edge.to == node.id {
                        Some(&edge.from)
                    } else {
                        None
                    }
                })
                .collect();

            if connected_nodes.is_empty() {
                rotations.insert(node.id.clone(), (0.0, 0.0, 0.0));
                continue;
            }

            // Calculate average direction to connected nodes
            let node_pos = positions.get(&node.id).unwrap_or(&(0.0, 0.0, 0.0));
            let mut avg_direction = (0.0, 0.0, 0.0);

            for connected_id in connected_nodes {
                if let Some(connected_pos) = positions.get(connected_id) {
                    avg_direction.0 += connected_pos.0 - node_pos.0;
                    avg_direction.1 += connected_pos.1 - node_pos.1;
                    avg_direction.2 += connected_pos.2 - node_pos.2;
                }
            }

            // Convert direction to rotation (simplified)
            let magnitude = (avg_direction.0.powi(2) + avg_direction.1.powi(2) + avg_direction.2.powi(2)).sqrt();
            if magnitude > 0.0 {
                let normalized = (
                    avg_direction.0 / magnitude,
                    avg_direction.1 / magnitude,
                    avg_direction.2 / magnitude,
                );

                // Convert to Euler angles (simplified approximation)
                let rotation_y = normalized.0.atan2(normalized.2);
                let rotation_x = (-normalized.1).asin();

                rotations.insert(node.id.clone(), (rotation_x, rotation_y, 0.0));
            } else {
                rotations.insert(node.id.clone(), (0.0, 0.0, 0.0));
            }
        }

        rotations
    }

    /// Evaluate 3D layout quality
    fn evaluate_layout_quality(
        &self,
        graph: &TraitGraph,
        positions: &HashMap<String, (f64, f64, f64)>,
    ) -> Layout3DQualityMetrics {
        let mut total_edge_length = 0.0;
        let mut edge_count = 0;

        // Calculate average edge length
        for edge in &graph.edges {
            if let (Some(pos1), Some(pos2)) = (positions.get(&edge.from), positions.get(&edge.to)) {
                let distance = ((pos1.0 - pos2.0).powi(2) + (pos1.1 - pos2.1).powi(2) + (pos1.2 - pos2.2).powi(2)).sqrt();
                total_edge_length += distance;
                edge_count += 1;
            }
        }

        let average_edge_length = if edge_count > 0 {
            total_edge_length / edge_count as f64
        } else {
            0.0
        };

        // Calculate volume utilization
        let mut min_bounds = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max_bounds = (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

        for pos in positions.values() {
            min_bounds.0 = min_bounds.0.min(pos.0);
            min_bounds.1 = min_bounds.1.min(pos.1);
            min_bounds.2 = min_bounds.2.min(pos.2);
            max_bounds.0 = max_bounds.0.max(pos.0);
            max_bounds.1 = max_bounds.1.max(pos.1);
            max_bounds.2 = max_bounds.2.max(pos.2);
        }

        let volume = (max_bounds.0 - min_bounds.0) * (max_bounds.1 - min_bounds.1) * (max_bounds.2 - min_bounds.2);
        let volume_utilization = if volume > 0.0 {
            (positions.len() as f64 * 1000.0) / volume // Normalize by approximate node volume
        } else {
            0.0
        }.min(1.0);

        Layout3DQualityMetrics {
            volume_utilization,
            average_edge_length,
            edge_intersections: 0, // Simplified - would require complex 3D intersection tests
            distribution_uniformity: 0.7, // Placeholder
            visual_clarity: 0.8, // Placeholder
            depth_perception: 0.75, // Placeholder
        }
    }

    /// Generate Three.js code for the 3D visualization
    fn generate_threejs_code(&self, graph: &TraitGraph, layout_result: &Layout3DResult) -> Result<String> {
        let mut code = String::new();

        // Scene setup
        code.push_str(&format!(r#"
        // Three.js 3D Visualization Setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera({}, window.innerWidth / window.innerHeight, {}, {});
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});

        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = {};
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.getElementById('canvas-container').appendChild(renderer.domElement);

        // Camera setup
        camera.position.set({}, {}, {});
        camera.lookAt({}, {}, {});

        // Lighting setup
        const ambientLight = new THREE.AmbientLight(0xffffff, {});
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, {});
        directionalLight.position.set(200, 200, 200);
        directionalLight.castShadow = {};
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        scene.add(directionalLight);
        "#,
            self.config.camera_fov,
            self.config.camera_near,
            self.config.camera_far,
            self.config.enable_shadows,
            self.config.camera_position.0,
            self.config.camera_position.1,
            self.config.camera_position.2,
            self.config.camera_target.0,
            self.config.camera_target.1,
            self.config.camera_target.2,
            self.config.ambient_light,
            self.config.directional_light,
            self.config.enable_shadows
        ));

        // Background setup
        match self.config.background_type {
            BackgroundType::SolidColor => {
                code.push_str(&format!(
                    "scene.background = new THREE.Color('{}');\n",
                    self.config.background_value
                ));
            },
            BackgroundType::Gradient => {
                code.push_str(&format!(r#"
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = 256;
                canvas.height = 256;
                const gradient = context.createLinearGradient(0, 0, 0, 256);
                gradient.addColorStop(0, '{}');
                gradient.addColorStop(1, '#ffffff');
                context.fillStyle = gradient;
                context.fillRect(0, 0, 256, 256);
                scene.background = new THREE.CanvasTexture(canvas);
                "#, self.config.background_value));
            },
            _ => {
                // Other background types would require more complex setup
                code.push_str(&format!(
                    "scene.background = new THREE.Color('{}');\n",
                    self.config.background_value
                ));
            }
        }

        // Fog setup
        if self.config.enable_fog {
            code.push_str(&format!(
                "scene.fog = new THREE.Fog('{}', 0.1, {});\n",
                self.config.fog_color,
                1.0 / self.config.fog_density
            ));
        }

        // Node creation
        code.push_str("\n// Create nodes\nconst nodes = new Map();\nconst nodeObjects = [];\n\n");

        for node in &graph.nodes {
            if let Some(position) = layout_result.positions.get(&node.id) {
                let geometry = self.node_geometries.get(node.node_type.display_name())
                    .cloned()
                    .unwrap_or_else(|| NodeGeometry {
                        geometry_type: GeometryType::Sphere,
                        size: (1.0, 1.0, 1.0),
                        segments: 16,
                        parameters: HashMap::new(),
                    });

                let material = self.material_templates.get(node.node_type.display_name())
                    .cloned()
                    .unwrap_or_else(|| MaterialTemplate {
                        material_type: MaterialType::Standard,
                        color: "#007bff".to_string(),
                        emissive: "#000000".to_string(),
                        metalness: 0.2,
                        roughness: 0.3,
                        opacity: 1.0,
                        transparent: false,
                        texture: None,
                        normal_map: None,
                        env_map: None,
                    });

                let geometry_code = self.generate_geometry_code(&geometry);
                let material_code = self.generate_material_code(&material);

                code.push_str(&format!(r#"
                {{
                    const geometry = {};
                    const material = {};
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.position.set({}, {}, {});
                    mesh.scale.set({}, {}, {});
                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                    mesh.userData = {{
                        id: '{}',
                        label: '{}',
                        type: '{}',
                        nodeData: {}
                    }};
                    scene.add(mesh);
                    nodes.set('{}', mesh);
                    nodeObjects.push(mesh);
                }}
                "#,
                    geometry_code,
                    material_code,
                    position.0, position.1, position.2,
                    node.size, node.size, node.size,
                    Self::escape_js_string(&node.id),
                    Self::escape_js_string(&node.label),
                    node.node_type.display_name(),
                    "null", // Placeholder for additional node data
                    Self::escape_js_string(&node.id)
                ));
            }
        }

        // Edge creation
        code.push_str("\n// Create edges\nconst edges = [];\n\n");

        for edge in &graph.edges {
            if let (Some(from_pos), Some(to_pos)) = (
                layout_result.positions.get(&edge.from),
                layout_result.positions.get(&edge.to)
            ) {
                let color = edge.color.as_deref().unwrap_or("#666666");
                let thickness = edge.thickness.unwrap_or(1.0);

                code.push_str(&format!(r#"
                {{
                    const points = [
                        new THREE.Vector3({}, {}, {}),
                        new THREE.Vector3({}, {}, {})
                    ];
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const material = new THREE.LineBasicMaterial({{
                        color: '{}',
                        linewidth: {}
                    }});
                    const line = new THREE.Line(geometry, material);
                    line.userData = {{
                        from: '{}',
                        to: '{}',
                        type: '{}',
                        weight: {}
                    }};
                    scene.add(line);
                    edges.push(line);
                }}
                "#,
                    from_pos.0, from_pos.1, from_pos.2,
                    to_pos.0, to_pos.1, to_pos.2,
                    color, thickness,
                    Self::escape_js_string(&edge.from),
                    Self::escape_js_string(&edge.to),
                    edge.edge_type.display_name(),
                    edge.weight
                ));
            }
        }

        // Controls and interaction
        if self.config.enable_camera_controls {
            code.push_str(r#"
            // Camera controls
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = false;
            controls.minDistance = 10;
            controls.maxDistance = 1000;
            controls.maxPolarAngle = Math.PI;
            "#);
        }

        // Animation and rendering
        code.push_str(&format!(r#"
        // Animation loop
        let animationId;
        const clock = new THREE.Clock();

        function animate() {{
            animationId = requestAnimationFrame(animate);

            const deltaTime = clock.getDelta();

            // Physics simulation
            if ({}) {{
                updatePhysics(deltaTime);
            }}

            // Update controls
            if (typeof controls !== 'undefined') {{
                controls.update();
            }}

            // Render
            renderer.render(scene, camera);
        }}

        function updatePhysics(deltaTime) {{
            // Simple physics simulation
            const gravity = {};
            const damping = {};

            nodeObjects.forEach(node => {{
                if (node.userData.velocity) {{
                    node.userData.velocity.y -= gravity * deltaTime;
                    node.position.add(node.userData.velocity.clone().multiplyScalar(deltaTime));
                    node.userData.velocity.multiplyScalar(damping);
                }}
            }});
        }}

        // Start animation
        animate();

        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // Cleanup function
        function cleanup() {{
            if (animationId) {{
                cancelAnimationFrame(animationId);
            }}
            renderer.dispose();
        }}

        window.addEventListener('beforeunload', cleanup);
        "#,
            self.config.enable_physics,
            self.config.physics_gravity,
            self.config.physics_damping
        ));

        Ok(code)
    }

    /// Generate geometry code for Three.js
    fn generate_geometry_code(&self, geometry: &NodeGeometry) -> String {
        match geometry.geometry_type {
            GeometryType::Sphere => format!(
                "new THREE.SphereGeometry({}, {}, {})",
                geometry.size.0, geometry.segments, geometry.segments / 2
            ),
            GeometryType::Box => format!(
                "new THREE.BoxGeometry({}, {}, {})",
                geometry.size.0, geometry.size.1, geometry.size.2
            ),
            GeometryType::Cylinder => format!(
                "new THREE.CylinderGeometry({}, {}, {}, {})",
                geometry.size.0, geometry.size.0, geometry.size.1, geometry.segments
            ),
            GeometryType::Cone => format!(
                "new THREE.ConeGeometry({}, {}, {})",
                geometry.size.0, geometry.size.1, geometry.segments
            ),
            GeometryType::Octahedron => format!(
                "new THREE.OctahedronGeometry({})",
                geometry.size.0
            ),
            GeometryType::Icosahedron => format!(
                "new THREE.IcosahedronGeometry({})",
                geometry.size.0
            ),
            GeometryType::Dodecahedron => format!(
                "new THREE.DodecahedronGeometry({})",
                geometry.size.0
            ),
            GeometryType::Torus => format!(
                "new THREE.TorusGeometry({}, {}, {}, {})",
                geometry.size.0, geometry.size.1, geometry.segments / 2, geometry.segments
            ),
            GeometryType::Custom => "new THREE.SphereGeometry(1, 16, 8)".to_string(), // Fallback
        }
    }

    /// Generate material code for Three.js
    fn generate_material_code(&self, material: &MaterialTemplate) -> String {
        let material_type = match material.material_type {
            MaterialType::Basic => "MeshBasicMaterial",
            MaterialType::Lambert => "MeshLambertMaterial",
            MaterialType::Phong => "MeshPhongMaterial",
            MaterialType::Standard => "MeshStandardMaterial",
            MaterialType::Physical => "MeshPhysicalMaterial",
            MaterialType::Toon => "MeshToonMaterial",
            MaterialType::Points => "PointsMaterial",
            MaterialType::Line => "LineBasicMaterial",
        };

        format!(
            "new THREE.{}({{ color: '{}', emissive: '{}', metalness: {}, roughness: {}, opacity: {}, transparent: {} }})",
            material_type,
            material.color,
            material.emissive,
            material.metalness,
            material.roughness,
            material.opacity,
            material.transparent
        )
    }

    /// Create HTML template for 3D visualization
    fn create_html_template(&self, three_js_code: &str) -> String {
        format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Trait Graph Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background: {};
        }}

        #canvas-container {{
            width: 100vw;
            height: 100vh;
        }}

        #ui-overlay {{
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1000;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 15px;
            border-radius: 8px;
            max-width: 300px;
        }}

        #controls {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}

        button {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: #007bff;
            color: white;
            cursor: pointer;
            font-size: 14px;
        }}

        button:hover {{
            background: #0056b3;
        }}

        #loading {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 2000;
            font-size: 18px;
            color: white;
            background: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 8px;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="loading">Loading 3D visualization...</div>

    <div id="canvas-container"></div>

    <div id="ui-overlay">
        <h3>3D Trait Graph</h3>
        <p>Use mouse to navigate:</p>
        <ul>
            <li>Left click + drag: Rotate</li>
            <li>Right click + drag: Pan</li>
            <li>Scroll: Zoom</li>
        </ul>
        <p>Physics: {}</p>
        <p>VR Support: {}</p>
    </div>

    <div id="controls">
        <button onclick="resetCamera()">Reset View</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
        <button onclick="changeLayout()">Change Layout</button>
        <button onclick="toggleFullscreen()">Fullscreen</button>
    </div>

    <script>
        // Remove loading indicator after a delay
        setTimeout(() => {{
            const loading = document.getElementById('loading');
            if (loading) loading.style.display = 'none';
        }}, 1000);

        {}

        // Additional control functions
        function resetCamera() {{
            camera.position.set({}, {}, {});
            camera.lookAt({}, {}, {});
            if (typeof controls !== 'undefined') {{
                controls.reset();
            }}
        }}

        let physicsEnabled = {};
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            console.log('Physics:', physicsEnabled ? 'enabled' : 'disabled');
        }}

        function changeLayout() {{
            // Cycle through different layouts
            console.log('Layout change not implemented yet');
        }}

        function toggleFullscreen() {{
            if (!document.fullscreenElement) {{
                document.documentElement.requestFullscreen();
            }} else {{
                document.exitFullscreen();
            }}
        }}

        // Add mouse interaction for nodes
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        function onMouseClick(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(nodeObjects);

            if (intersects.length > 0) {{
                const node = intersects[0].object;
                console.log('Clicked node:', node.userData);

                // Highlight node
                node.material.emissive.setHex(0x444444);
                setTimeout(() => {{
                    node.material.emissive.setHex(0x000000);
                }}, 500);
            }}
        }}

        renderer.domElement.addEventListener('click', onMouseClick);
    </script>
</body>
</html>"#,
            self.theme.background_color(),
            if self.config.enable_physics { "Enabled" } else { "Disabled" },
            if self.config.enable_vr { "Available" } else { "Not Available" },
            three_js_code,
            self.config.camera_position.0,
            self.config.camera_position.1,
            self.config.camera_position.2,
            self.config.camera_target.0,
            self.config.camera_target.1,
            self.config.camera_target.2,
            self.config.enable_physics
        )
    }

    /// Escape JavaScript string
    fn escape_js_string(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('\'', "\\'")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    }

    /// Get configuration
    pub fn get_config(&self) -> &ThreeDConfig {
        &self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: ThreeDConfig) {
        self.config = config;
    }

    /// Add custom geometry
    pub fn add_geometry(&mut self, name: String, geometry: NodeGeometry) {
        self.node_geometries.insert(name, geometry);
    }

    /// Add custom material
    pub fn add_material(&mut self, name: String, material: MaterialTemplate) {
        self.material_templates.insert(name, material);
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph_structures::{TraitGraph, TraitGraphNode, TraitGraphEdge};

    fn create_test_graph() -> TraitGraph {
        let mut graph = TraitGraph::new();

        let node1 = TraitGraphNode::new_trait("Trait1".to_string(), "Trait1".to_string());
        let node2 = TraitGraphNode::new_implementation("Impl1".to_string(), "Impl1".to_string(), "Trait1".to_string());

        graph.add_node(node1);
        graph.add_node(node2);

        let edge = TraitGraphEdge::new_implementation("Trait1".to_string(), "Impl1".to_string());
        graph.add_edge(edge);

        graph
    }

    #[test]
    fn test_3d_config_default() {
        let config = ThreeDConfig::default();
        assert!(config.enable_physics);
        assert!(config.enable_camera_controls);
        assert_eq!(config.camera_fov, 75.0);
    }

    #[test]
    fn test_3d_config_builder() {
        let config = ThreeDConfigBuilder::new()
            .with_physics_simulation(false)
            .with_camera_fov(60.0)
            .with_lighting(0.3, 0.7)
            .build();

        assert!(!config.enable_physics);
        assert_eq!(config.camera_fov, 60.0);
        assert_eq!(config.ambient_light, 0.3);
        assert_eq!(config.directional_light, 0.7);
    }

    #[test]
    fn test_3d_generator_creation() {
        let config = ThreeDConfig::default();
        let generator = Graph3DGenerator::new(config, VisualizationTheme::Dark);

        assert!(!generator.node_geometries.is_empty());
        assert!(!generator.material_templates.is_empty());
    }

    #[test]
    fn test_spherical_layout() {
        let config = ThreeDConfig::default();
        let generator = Graph3DGenerator::new(config, VisualizationTheme::Light);
        let graph = create_test_graph();

        let positions = generator.compute_spherical_layout(&graph);
        assert!(positions.is_ok());

        let positions = positions.unwrap();
        assert_eq!(positions.len(), graph.nodes.len());
    }

    #[test]
    fn test_cylindrical_layout() {
        let config = ThreeDConfig::default();
        let generator = Graph3DGenerator::new(config, VisualizationTheme::Light);
        let graph = create_test_graph();

        let positions = generator.compute_cylindrical_layout(&graph);
        assert!(positions.is_ok());

        let positions = positions.unwrap();
        assert_eq!(positions.len(), graph.nodes.len());
    }

    #[test]
    fn test_force_3d_layout() {
        let config = ThreeDConfig::default();
        let generator = Graph3DGenerator::new(config, VisualizationTheme::Light);
        let graph = create_test_graph();

        let positions = generator.compute_force_3d_layout(&graph);
        assert!(positions.is_ok());

        let positions = positions.unwrap();
        assert_eq!(positions.len(), graph.nodes.len());
    }

    #[test]
    fn test_3d_layout_computation() {
        let config = ThreeDConfig::default();
        let generator = Graph3DGenerator::new(config, VisualizationTheme::Light);
        let graph = create_test_graph();

        let layout_result = generator.compute_3d_layout(&graph);
        assert!(layout_result.is_ok());

        let result = layout_result.unwrap();
        assert_eq!(result.positions.len(), graph.nodes.len());
        assert_eq!(result.rotations.len(), graph.nodes.len());
    }

    #[test]
    fn test_geometry_code_generation() {
        let config = ThreeDConfig::default();
        let generator = Graph3DGenerator::new(config, VisualizationTheme::Light);

        let sphere_geom = NodeGeometry {
            geometry_type: GeometryType::Sphere,
            size: (2.0, 2.0, 2.0),
            segments: 16,
            parameters: HashMap::new(),
        };

        let code = generator.generate_geometry_code(&sphere_geom);
        assert!(code.contains("SphereGeometry"));
        assert!(code.contains("2"));
        assert!(code.contains("16"));
    }

    #[test]
    fn test_material_code_generation() {
        let config = ThreeDConfig::default();
        let generator = Graph3DGenerator::new(config, VisualizationTheme::Light);

        let material = MaterialTemplate {
            material_type: MaterialType::Standard,
            color: "#ff0000".to_string(),
            emissive: "#000000".to_string(),
            metalness: 0.5,
            roughness: 0.3,
            opacity: 1.0,
            transparent: false,
            texture: None,
            normal_map: None,
            env_map: None,
        };

        let code = generator.generate_material_code(&material);
        assert!(code.contains("MeshStandardMaterial"));
        assert!(code.contains("#ff0000"));
        assert!(code.contains("0.5"));
        assert!(code.contains("0.3"));
    }

    #[test]
    fn test_html_generation() {
        let config = ThreeDConfig::default();
        let generator = Graph3DGenerator::new(config, VisualizationTheme::Light);
        let graph = create_test_graph();

        let html_result = generator.generate_3d_html(&graph);
        assert!(html_result.is_ok());

        let html = html_result.unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("three.min.js"));
        assert!(html.contains("OrbitControls"));
    }

    #[test]
    fn test_javascript_string_escaping() {
        assert_eq!(
            Graph3DGenerator::escape_js_string("test\n\"quote\""),
            "test\\n\\\"quote\\\""
        );
        assert_eq!(
            Graph3DGenerator::escape_js_string("path\\to\\file"),
            "path\\\\to\\\\file"
        );
    }
}
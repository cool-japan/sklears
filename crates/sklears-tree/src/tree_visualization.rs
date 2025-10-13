//! Tree visualization utilities for creating visual representations of decision trees
//!
//! This module provides functionality to generate ASCII art, DOT graphs, and other
//! visual representations of tree structures for better interpretability.

use crate::tree_interpretation::{TreeNode, TreeStructure};
use std::collections::HashMap;
use std::fmt;

/// Tree visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Feature names for readable output
    pub feature_names: Option<Vec<String>>,
    /// Class names for classification trees
    pub class_names: Option<Vec<String>>,
    /// Maximum depth to visualize
    pub max_depth: Option<usize>,
    /// Whether to show node statistics
    pub show_stats: bool,
    /// Whether to show impurity values
    pub show_impurity: bool,
    /// Precision for floating point values
    pub precision: usize,
    /// Whether to use compact format
    pub compact: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            feature_names: None,
            class_names: None,
            max_depth: None,
            show_stats: true,
            show_impurity: true,
            precision: 3,
            compact: false,
        }
    }
}

/// ASCII art tree visualizer
pub struct AsciiTreeVisualizer {
    /// Configuration for visualization
    pub config: VisualizationConfig,
}

impl AsciiTreeVisualizer {
    /// Create a new ASCII tree visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate ASCII representation of a tree
    pub fn visualize(&self, tree: &TreeStructure) -> String {
        let mut output = String::new();
        output.push_str("Decision Tree Visualization\n");
        output.push_str("==========================\n\n");

        self.visualize_node(&tree.root, &mut output, "", true, 0);

        output
    }

    /// Recursively visualize tree nodes
    fn visualize_node(
        &self,
        node: &TreeNode,
        output: &mut String,
        prefix: &str,
        is_last: bool,
        depth: usize,
    ) {
        // Check max depth
        if let Some(max_depth) = self.config.max_depth {
            if depth >= max_depth {
                output.push_str(&format!(
                    "{}{}...\n",
                    prefix,
                    if is_last { "└── " } else { "├── " }
                ));
                return;
            }
        }

        let connector = if is_last { "└── " } else { "├── " };

        match node {
            TreeNode::Internal {
                feature_idx,
                threshold,
                left,
                right,
                n_samples,
                impurity,
            } => {
                let feature_name = self.get_feature_name(*feature_idx);
                let node_info =
                    self.format_internal_node(&feature_name, *threshold, *n_samples, *impurity);

                output.push_str(&format!("{}{}{}\n", prefix, connector, node_info));

                let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

                // Left child (<=)
                output.push_str(&format!(
                    "{}├── {} <= {:.prec$}\n",
                    child_prefix,
                    feature_name,
                    threshold,
                    prec = self.config.precision
                ));
                self.visualize_node(
                    left,
                    output,
                    &format!("{}│   ", child_prefix),
                    false,
                    depth + 1,
                );

                // Right child (>)
                output.push_str(&format!(
                    "{}└── {} > {:.prec$}\n",
                    child_prefix,
                    feature_name,
                    threshold,
                    prec = self.config.precision
                ));
                self.visualize_node(
                    right,
                    output,
                    &format!("{}    ", child_prefix),
                    true,
                    depth + 1,
                );
            }
            TreeNode::Leaf {
                prediction,
                confidence,
                n_samples,
                impurity,
                class_distribution,
            } => {
                let node_info = self.format_leaf_node(
                    *prediction,
                    *confidence,
                    *n_samples,
                    *impurity,
                    class_distribution,
                );
                output.push_str(&format!("{}{}{}\n", prefix, connector, node_info));
            }
        }
    }

    /// Format internal node information
    fn format_internal_node(
        &self,
        feature_name: &str,
        threshold: f64,
        n_samples: Option<usize>,
        impurity: Option<f64>,
    ) -> String {
        if self.config.compact {
            return format!(
                "{} <= {:.prec$}",
                feature_name,
                threshold,
                prec = self.config.precision
            );
        }

        let mut info = format!(
            "Split: {} <= {:.prec$}",
            feature_name,
            threshold,
            prec = self.config.precision
        );

        if self.config.show_stats {
            if let Some(samples) = n_samples {
                info.push_str(&format!(" | samples: {}", samples));
            }
        }

        if self.config.show_impurity {
            if let Some(imp) = impurity {
                info.push_str(&format!(
                    " | impurity: {:.prec$}",
                    imp,
                    prec = self.config.precision
                ));
            }
        }

        info
    }

    /// Format leaf node information
    fn format_leaf_node(
        &self,
        prediction: f64,
        confidence: Option<f64>,
        n_samples: Option<usize>,
        impurity: Option<f64>,
        class_distribution: &Option<HashMap<usize, usize>>,
    ) -> String {
        let class_name = self.get_class_name(prediction);

        if self.config.compact {
            return format!("Predict: {}", class_name);
        }

        let mut info = format!("Predict: {}", class_name);

        if self.config.show_stats {
            if let Some(conf) = confidence {
                info.push_str(&format!(
                    " | confidence: {:.prec$}",
                    conf,
                    prec = self.config.precision
                ));
            }

            if let Some(samples) = n_samples {
                info.push_str(&format!(" | samples: {}", samples));
            }

            if let Some(ref dist) = class_distribution {
                let dist_str = dist
                    .iter()
                    .map(|(class, count)| {
                        format!("{}:{}", self.get_class_name(*class as f64), count)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                info.push_str(&format!(" | distribution: [{}]", dist_str));
            }
        }

        if self.config.show_impurity {
            if let Some(imp) = impurity {
                info.push_str(&format!(
                    " | impurity: {:.prec$}",
                    imp,
                    prec = self.config.precision
                ));
            }
        }

        info
    }

    /// Get feature name or default
    fn get_feature_name(&self, feature_idx: usize) -> String {
        self.config
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_idx))
            .cloned()
            .unwrap_or_else(|| format!("feature_{}", feature_idx))
    }

    /// Get class name or default
    fn get_class_name(&self, class_value: f64) -> String {
        let class_idx = class_value as usize;
        self.config
            .class_names
            .as_ref()
            .and_then(|names| names.get(class_idx))
            .cloned()
            .unwrap_or_else(|| {
                if class_value.fract() == 0.0 {
                    format!("class_{}", class_idx)
                } else {
                    format!("{:.prec$}", class_value, prec = self.config.precision)
                }
            })
    }
}

/// DOT graph generator for tree visualization
pub struct DotGraphGenerator {
    /// Configuration for visualization
    pub config: VisualizationConfig,
    /// Node counter for unique IDs
    node_counter: usize,
}

impl DotGraphGenerator {
    /// Create a new DOT graph generator
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            config,
            node_counter: 0,
        }
    }

    /// Generate DOT graph representation of a tree
    pub fn generate_dot(&mut self, tree: &TreeStructure) -> String {
        self.node_counter = 0;

        let mut dot = String::new();
        dot.push_str("digraph DecisionTree {\n");
        dot.push_str("    node [shape=box, style=\"rounded,filled\", fontname=\"Arial\"];\n");
        dot.push_str("    edge [fontname=\"Arial\"];\n\n");

        self.generate_node_dot(&tree.root, &mut dot, 0);

        dot.push_str("}\n");
        dot
    }

    /// Generate DOT representation for a node
    fn generate_node_dot(&mut self, node: &TreeNode, dot: &mut String, depth: usize) -> usize {
        let node_id = self.node_counter;
        self.node_counter += 1;

        // Check max depth
        if let Some(max_depth) = self.config.max_depth {
            if depth >= max_depth {
                dot.push_str(&format!(
                    "    node{} [label=\"...\", fillcolor=\"lightgray\"];\n",
                    node_id
                ));
                return node_id;
            }
        }

        match node {
            TreeNode::Internal {
                feature_idx,
                threshold,
                left,
                right,
                n_samples,
                impurity,
            } => {
                let feature_name = self.get_feature_name(*feature_idx);
                let label =
                    self.format_internal_node_dot(&feature_name, *threshold, *n_samples, *impurity);

                dot.push_str(&format!(
                    "    node{} [label=\"{}\", fillcolor=\"lightblue\"];\n",
                    node_id, label
                ));

                // Left child
                let left_id = self.generate_node_dot(left, dot, depth + 1);
                dot.push_str(&format!(
                    "    node{} -> node{} [label=\"<= {:.prec$}\", color=\"blue\"];\n",
                    node_id,
                    left_id,
                    threshold,
                    prec = self.config.precision
                ));

                // Right child
                let right_id = self.generate_node_dot(right, dot, depth + 1);
                dot.push_str(&format!(
                    "    node{} -> node{} [label=\"> {:.prec$}\", color=\"red\"];\n",
                    node_id,
                    right_id,
                    threshold,
                    prec = self.config.precision
                ));
            }
            TreeNode::Leaf {
                prediction,
                confidence,
                n_samples,
                impurity,
                class_distribution,
            } => {
                let label = self.format_leaf_node_dot(
                    *prediction,
                    *confidence,
                    *n_samples,
                    *impurity,
                    class_distribution,
                );
                let color = self.get_leaf_color(*prediction);

                dot.push_str(&format!(
                    "    node{} [label=\"{}\", fillcolor=\"{}\"];\n",
                    node_id, label, color
                ));
            }
        }

        node_id
    }

    /// Format internal node for DOT
    fn format_internal_node_dot(
        &self,
        feature_name: &str,
        threshold: f64,
        n_samples: Option<usize>,
        impurity: Option<f64>,
    ) -> String {
        let mut label = format!("{}", feature_name);

        if self.config.show_stats {
            if let Some(samples) = n_samples {
                label.push_str(&format!("\\nsamples: {}", samples));
            }
        }

        if self.config.show_impurity {
            if let Some(imp) = impurity {
                label.push_str(&format!(
                    "\\nimpurity: {:.prec$}",
                    imp,
                    prec = self.config.precision
                ));
            }
        }

        label
    }

    /// Format leaf node for DOT
    fn format_leaf_node_dot(
        &self,
        prediction: f64,
        confidence: Option<f64>,
        n_samples: Option<usize>,
        impurity: Option<f64>,
        class_distribution: &Option<HashMap<usize, usize>>,
    ) -> String {
        let class_name = self.get_class_name(prediction);
        let mut label = format!("Predict: {}", class_name);

        if self.config.show_stats {
            if let Some(conf) = confidence {
                label.push_str(&format!(
                    "\\nconfidence: {:.prec$}",
                    conf,
                    prec = self.config.precision
                ));
            }

            if let Some(samples) = n_samples {
                label.push_str(&format!("\\nsamples: {}", samples));
            }

            if let Some(ref dist) = class_distribution {
                let dist_str = dist
                    .iter()
                    .map(|(class, count)| {
                        format!("{}:{}", self.get_class_name(*class as f64), count)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                label.push_str(&format!("\\n[{}]", dist_str));
            }
        }

        if self.config.show_impurity {
            if let Some(imp) = impurity {
                label.push_str(&format!(
                    "\\nimpurity: {:.prec$}",
                    imp,
                    prec = self.config.precision
                ));
            }
        }

        label
    }

    /// Get color for leaf node based on prediction
    fn get_leaf_color(&self, prediction: f64) -> &'static str {
        // Simple color scheme based on prediction value
        if prediction < 0.0 {
            "lightcoral"
        } else if prediction < 0.5 {
            "lightgreen"
        } else if prediction < 1.0 {
            "lightyellow"
        } else {
            "lightpink"
        }
    }

    /// Get feature name or default
    fn get_feature_name(&self, feature_idx: usize) -> String {
        self.config
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_idx))
            .cloned()
            .unwrap_or_else(|| format!("X_{}", feature_idx))
    }

    /// Get class name or default
    fn get_class_name(&self, class_value: f64) -> String {
        let class_idx = class_value as usize;
        self.config
            .class_names
            .as_ref()
            .and_then(|names| names.get(class_idx))
            .cloned()
            .unwrap_or_else(|| {
                if class_value.fract() == 0.0 {
                    format!("{}", class_idx)
                } else {
                    format!("{:.prec$}", class_value, prec = self.config.precision)
                }
            })
    }
}

/// Tree statistics calculator
pub struct TreeStatistics {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of internal nodes
    pub internal_nodes: usize,
    /// Number of leaf nodes
    pub leaf_nodes: usize,
    /// Maximum depth of the tree
    pub max_depth: usize,
    /// Average depth of leaves
    pub avg_leaf_depth: f64,
    /// Total number of samples
    pub total_samples: usize,
    /// Feature usage count
    pub feature_usage: HashMap<usize, usize>,
}

impl TreeStatistics {
    /// Calculate statistics for a tree
    pub fn calculate(tree: &TreeStructure) -> Self {
        let mut stats = Self {
            total_nodes: 0,
            internal_nodes: 0,
            leaf_nodes: 0,
            max_depth: 0,
            avg_leaf_depth: 0.0,
            total_samples: 0,
            feature_usage: HashMap::new(),
        };

        let mut leaf_depths = Vec::new();
        stats.calculate_recursive(&tree.root, 0, &mut leaf_depths);

        if !leaf_depths.is_empty() {
            stats.avg_leaf_depth =
                leaf_depths.iter().sum::<usize>() as f64 / leaf_depths.len() as f64;
            stats.max_depth = *leaf_depths.iter().max().unwrap();
        }

        stats
    }

    /// Recursively calculate statistics
    fn calculate_recursive(&mut self, node: &TreeNode, depth: usize, leaf_depths: &mut Vec<usize>) {
        self.total_nodes += 1;

        match node {
            TreeNode::Internal {
                feature_idx,
                left,
                right,
                n_samples,
                ..
            } => {
                self.internal_nodes += 1;
                *self.feature_usage.entry(*feature_idx).or_insert(0) += 1;

                if let Some(samples) = n_samples {
                    self.total_samples += samples;
                }

                self.calculate_recursive(left, depth + 1, leaf_depths);
                self.calculate_recursive(right, depth + 1, leaf_depths);
            }
            TreeNode::Leaf { n_samples, .. } => {
                self.leaf_nodes += 1;
                leaf_depths.push(depth);

                if let Some(samples) = n_samples {
                    self.total_samples += samples;
                }
            }
        }
    }
}

impl fmt::Display for TreeStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tree Statistics:")?;
        writeln!(f, "- Total nodes: {}", self.total_nodes)?;
        writeln!(f, "- Internal nodes: {}", self.internal_nodes)?;
        writeln!(f, "- Leaf nodes: {}", self.leaf_nodes)?;
        writeln!(f, "- Max depth: {}", self.max_depth)?;
        writeln!(f, "- Average leaf depth: {:.2}", self.avg_leaf_depth)?;
        writeln!(f, "- Total samples: {}", self.total_samples)?;

        if !self.feature_usage.is_empty() {
            writeln!(f, "- Feature usage:")?;
            let mut usage_vec: Vec<(usize, usize)> =
                self.feature_usage.iter().map(|(k, v)| (*k, *v)).collect();
            usage_vec.sort_by(|a, b| b.1.cmp(&a.1));

            for (feature_idx, count) in usage_vec.iter().take(10) {
                writeln!(f, "  - feature_{}: {} times", feature_idx, count)?;
            }
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_interpretation::TreeStructure;

    #[test]
    fn test_ascii_visualization() {
        let config = VisualizationConfig {
            feature_names: Some(vec!["age".to_string()]),
            class_names: Some(vec!["young".to_string(), "old".to_string()]),
            ..Default::default()
        };

        let visualizer = AsciiTreeVisualizer::new(config);
        let tree = TreeStructure::create_simple_tree();

        let output = visualizer.visualize(&tree);
        assert!(output.contains("Decision Tree Visualization"));
        assert!(output.contains("age"));
        assert!(output.contains("<="));
        assert!(output.contains("Predict"));
    }

    #[test]
    fn test_dot_generation() {
        let config = VisualizationConfig {
            feature_names: Some(vec!["feature_0".to_string()]),
            class_names: Some(vec!["class_0".to_string(), "class_1".to_string()]),
            ..Default::default()
        };

        let mut generator = DotGraphGenerator::new(config);
        let tree = TreeStructure::create_simple_tree();

        let dot = generator.generate_dot(&tree);
        assert!(dot.contains("digraph DecisionTree"));
        assert!(dot.contains("node"));
        assert!(dot.contains("edge"));
        assert!(dot.contains("feature_0"));
    }

    #[test]
    fn test_tree_statistics() {
        let tree = TreeStructure::create_simple_tree();
        let stats = TreeStatistics::calculate(&tree);

        assert_eq!(stats.total_nodes, 3); // 1 internal + 2 leaves
        assert_eq!(stats.internal_nodes, 1);
        assert_eq!(stats.leaf_nodes, 2);
        assert_eq!(stats.max_depth, 1);
        assert!((stats.avg_leaf_depth - 1.0).abs() < f64::EPSILON);
        assert_eq!(stats.feature_usage.get(&0), Some(&1)); // Feature 0 used once
    }

    #[test]
    fn test_compact_visualization() {
        let config = VisualizationConfig {
            compact: true,
            show_stats: false,
            show_impurity: false,
            ..Default::default()
        };

        let visualizer = AsciiTreeVisualizer::new(config);
        let tree = TreeStructure::create_simple_tree();

        let output = visualizer.visualize(&tree);
        assert!(!output.contains("samples:"));
        assert!(!output.contains("impurity:"));
    }
}

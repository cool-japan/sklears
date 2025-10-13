//! Lazy Evaluation for Preprocessing Chains
//!
//! This module provides lazy evaluation capabilities for preprocessing operations,
//! enabling memory-efficient processing of large datasets by deferring computations
//! until needed and optimizing across entire preprocessing chains.
//!
//! # Features
//!
//! - Lazy computation graphs for preprocessing operations
//! - Automatic operation fusion for improved performance
//! - Memory-efficient streaming evaluation
//! - Graph optimization (dead code elimination, operation merging)
//! - Parallel lazy evaluation support
//! - Zero-copy operation chaining where possible
//!
//! # Examples
//!
//! ```rust
//! use sklears_preprocessing::lazy_evaluation::{LazyPreprocessor, LazyConfig, LazyOp};
//! use scirs2_core::ndarray::Array2;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = LazyConfig::new()
//!         .with_optimization_level(2)
//!         .with_memory_budget(1024 * 1024 * 512); // 512MB
//!
//!     let mut lazy_processor = LazyPreprocessor::new(config);
//!
//!     // Build lazy computation graph
//!     lazy_processor
//!         .add_operation(LazyOp::StandardScale)
//!         .add_operation(LazyOp::PolynomialFeatures { degree: 2 })
//!         .add_operation(LazyOp::FeatureSelection { k: 100 });
//!
//!     let data = Array2::from_shape_vec((10000, 50),
//!         (0..500000).map(|x| x as f64).collect())?;
//!
//!     // Operations are not executed until evaluate is called
//!     let result = lazy_processor.evaluate(&data)?;
//!
//!     println!("Result shape: {:?}", result.dim());
//!     Ok(())
//! }
//! ```

use scirs2_core::memory::BufferPool;
use scirs2_core::ndarray::{s, Array2, Axis};
// FIXME: Some scirs2_core modules don't exist - using placeholders
// use scirs2_core::memory::{ChunkBuffer, MemoryOptimizer};
// use scirs2_core::parallel::{ParallelExecutor, ChunkStrategy};
// use scirs2_core::profiling::Profiler;

/// Placeholder MemoryOptimizer
#[derive(Debug, Clone, Default)]
pub struct MemoryOptimizer;

impl MemoryOptimizer {
    pub fn new() -> Self {
        Self
    }
}

/// Placeholder Profiler
#[derive(Debug, Clone, Default)]
pub struct Profiler;

impl Profiler {
    pub fn new() -> Self {
        Self
    }

    pub fn start(&self, _name: &str) {}
    pub fn end(&self, _name: &str) {}

    pub fn get_stats(&self) -> std::collections::HashMap<String, Float> {
        std::collections::HashMap::new()
    }
}
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::{HashMap, VecDeque};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Lazy operation types supported by the preprocessing pipeline
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LazyOp {
    /// Standard scaling operation
    StandardScale,
    /// Min-max scaling operation
    MinMaxScale { feature_range: (Float, Float) },
    /// Robust scaling operation
    RobustScale { quantile_range: (Float, Float) },
    /// Normalization operation
    Normalize { norm: String },
    /// Polynomial features generation
    PolynomialFeatures { degree: usize },
    /// Principal Component Analysis
    PCA { n_components: usize },
    /// Feature selection
    FeatureSelection { k: usize },
    /// Outlier detection and removal
    OutlierDetection { method: String, threshold: Float },
    /// Missing value imputation
    Imputation { strategy: String },
    /// Custom transformation function
    Custom {
        name: String,
        params: HashMap<String, Float>,
    },
}

impl LazyOp {
    /// Get memory footprint estimate for this operation
    pub fn memory_footprint(&self, input_shape: (usize, usize)) -> usize {
        let (n_samples, n_features) = input_shape;
        let element_size = std::mem::size_of::<Float>();

        match self {
            LazyOp::StandardScale | LazyOp::MinMaxScale { .. } | LazyOp::RobustScale { .. } => {
                // Input + output + statistics
                2 * n_samples * n_features * element_size + n_features * element_size * 4
            }
            LazyOp::Normalize { .. } => {
                // Input + output + row norms
                2 * n_samples * n_features * element_size + n_samples * element_size
            }
            LazyOp::PolynomialFeatures { degree } => {
                let output_features = polynomial_feature_count(n_features, *degree);
                (n_samples * n_features + n_samples * output_features) * element_size
            }
            LazyOp::PCA { n_components } => {
                (n_samples * n_features + n_samples * n_components + n_features * n_components)
                    * element_size
            }
            LazyOp::FeatureSelection { k } => {
                (n_samples * n_features + n_samples * k.min(&n_features)) * element_size
            }
            LazyOp::OutlierDetection { .. } => {
                // Input + output + outlier mask
                2 * n_samples * n_features * element_size + n_samples
            }
            LazyOp::Imputation { .. } => {
                // Input + output + missing mask
                2 * n_samples * n_features * element_size + n_samples * n_features
            }
            LazyOp::Custom { .. } => {
                // Conservative estimate: 2x input size
                2 * n_samples * n_features * element_size
            }
        }
    }

    /// Check if this operation can be fused with another operation
    pub fn can_fuse_with(&self, other: &LazyOp) -> bool {
        use LazyOp::*;
        match (self, other) {
            // Scaling operations can be fused
            (StandardScale, MinMaxScale { .. })
            | (MinMaxScale { .. }, RobustScale { .. })
            | (RobustScale { .. }, StandardScale) => true,

            // Normalize can be fused with scaling
            (StandardScale, Normalize { .. })
            | (MinMaxScale { .. }, Normalize { .. })
            | (Normalize { .. }, StandardScale)
            | (Normalize { .. }, MinMaxScale { .. }) => true,

            // Feature selection can be fused with dimensionality reduction
            (FeatureSelection { .. }, PCA { .. }) | (PCA { .. }, FeatureSelection { .. }) => true,

            _ => false,
        }
    }

    /// Create a fused operation from two compatible operations
    pub fn fuse(&self, other: &LazyOp) -> Option<LazyOp> {
        if !self.can_fuse_with(other) {
            return None;
        }

        use LazyOp::*;
        match (self, other) {
            (StandardScale, Normalize { norm: _ }) => Some(Custom {
                name: "standard_normalize".to_string(),
                params: HashMap::new(),
            }),
            _ => None, // More fusion rules can be added
        }
    }
}

/// Configuration for lazy evaluation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LazyConfig {
    /// Optimization level (0-3, higher = more aggressive optimization)
    pub optimization_level: usize,
    /// Memory budget in bytes
    pub memory_budget: usize,
    /// Enable operation fusion
    pub enable_fusion: bool,
    /// Enable parallel evaluation
    pub enable_parallel: bool,
    /// Chunk size for streaming evaluation
    pub chunk_size: usize,
    /// Enable intermediate result caching
    pub enable_caching: bool,
    /// Maximum cache size in bytes
    pub cache_size: usize,
}

impl Default for LazyConfig {
    fn default() -> Self {
        Self {
            optimization_level: 1,
            memory_budget: 1024 * 1024 * 1024, // 1GB
            enable_fusion: true,
            enable_parallel: true,
            chunk_size: 10000,
            enable_caching: true,
            cache_size: 256 * 1024 * 1024, // 256MB
        }
    }
}

impl LazyConfig {
    /// Create new lazy configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: usize) -> Self {
        self.optimization_level = level.min(3);
        self
    }

    /// Set memory budget
    pub fn with_memory_budget(mut self, budget: usize) -> Self {
        self.memory_budget = budget;
        self
    }

    /// Enable or disable operation fusion
    pub fn with_fusion(mut self, enabled: bool) -> Self {
        self.enable_fusion = enabled;
        self
    }

    /// Enable or disable parallel evaluation
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.enable_parallel = enabled;
        self
    }

    /// Set chunk size for streaming evaluation
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set caching configuration
    pub fn with_caching(mut self, enabled: bool, cache_size: usize) -> Self {
        self.enable_caching = enabled;
        self.cache_size = cache_size;
        self
    }
}

/// Node in the lazy computation graph
#[derive(Debug, Clone)]
pub struct LazyNode {
    /// Unique node identifier
    pub id: usize,
    /// Operation to perform
    pub operation: LazyOp,
    /// Input node dependencies
    pub inputs: Vec<usize>,
    /// Estimated output shape (rows, cols)
    pub output_shape: Option<(usize, usize)>,
    /// Estimated memory requirement
    pub memory_requirement: usize,
    /// Whether this node has been optimized
    pub optimized: bool,
}

impl LazyNode {
    /// Create a new lazy node
    pub fn new(id: usize, operation: LazyOp, inputs: Vec<usize>) -> Self {
        Self {
            id,
            operation,
            inputs,
            output_shape: None,
            memory_requirement: 0,
            optimized: false,
        }
    }

    /// Update shape and memory estimates
    pub fn update_estimates(&mut self, input_shape: (usize, usize)) {
        self.output_shape = Some(self.estimate_output_shape(input_shape));
        self.memory_requirement = self.operation.memory_footprint(input_shape);
    }

    /// Estimate output shape based on operation type
    fn estimate_output_shape(&self, input_shape: (usize, usize)) -> (usize, usize) {
        let (n_samples, n_features) = input_shape;

        match &self.operation {
            LazyOp::PolynomialFeatures { degree } => {
                (n_samples, polynomial_feature_count(n_features, *degree))
            }
            LazyOp::PCA { n_components } => (n_samples, *n_components),
            LazyOp::FeatureSelection { k } => (n_samples, (*k).min(n_features)),
            LazyOp::OutlierDetection { .. } => {
                // Assuming most samples are kept (conservative estimate: 90%)
                ((n_samples as Float * 0.9) as usize, n_features)
            }
            _ => input_shape, // Most operations preserve shape
        }
    }
}

/// Lazy preprocessing computation graph
pub struct LazyGraph {
    /// Nodes in the computation graph
    nodes: Vec<LazyNode>,
    /// Next available node ID
    next_id: usize,
    /// Input shape
    input_shape: Option<(usize, usize)>,
    /// Total memory requirement estimate
    total_memory: usize,
}

impl LazyGraph {
    /// Create a new empty lazy graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
            input_shape: None,
            total_memory: 0,
        }
    }

    /// Add an operation to the graph
    pub fn add_operation(&mut self, operation: LazyOp) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let inputs = if self.nodes.is_empty() {
            Vec::new() // First node has no inputs
        } else {
            vec![self.nodes.len() - 1] // Linear chain by default
        };

        let mut node = LazyNode::new(id, operation, inputs);

        // Update estimates if input shape is known
        if let Some(input_shape) = self.get_input_shape_for_node(id) {
            node.update_estimates(input_shape);
            self.total_memory += node.memory_requirement;
        }

        self.nodes.push(node);
        id
    }

    /// Get input shape for a specific node
    fn get_input_shape_for_node(&self, node_id: usize) -> Option<(usize, usize)> {
        if node_id == 0 {
            self.input_shape
        } else if let Some(prev_node) = self.nodes.get(node_id - 1) {
            prev_node.output_shape
        } else {
            None
        }
    }

    /// Set input shape and update all estimates
    pub fn set_input_shape(&mut self, shape: (usize, usize)) {
        self.input_shape = Some(shape);
        self.total_memory = 0;

        let mut current_shape = shape;
        for node in &mut self.nodes {
            node.update_estimates(current_shape);
            self.total_memory += node.memory_requirement;
            if let Some(output_shape) = node.output_shape {
                current_shape = output_shape;
            }
        }
    }

    /// Optimize the computation graph
    pub fn optimize(&mut self, config: &LazyConfig) {
        if config.optimization_level == 0 {
            return;
        }

        // Level 1: Dead code elimination
        if config.optimization_level >= 1 {
            self.eliminate_dead_code();
        }

        // Level 2: Operation fusion
        if config.optimization_level >= 2 && config.enable_fusion {
            self.fuse_operations();
        }

        // Level 3: Advanced optimizations
        if config.optimization_level >= 3 {
            self.reorder_operations();
            self.parallelize_independent_operations();
        }
    }

    /// Remove unused operations from the graph
    fn eliminate_dead_code(&mut self) {
        // Mark all nodes as potentially dead
        let mut used = vec![false; self.nodes.len()];

        // Mark the final node as used
        if !self.nodes.is_empty() {
            used[self.nodes.len() - 1] = true;

            // Trace backward to mark all dependencies as used
            let mut queue = VecDeque::new();
            queue.push_back(self.nodes.len() - 1);

            while let Some(node_idx) = queue.pop_front() {
                if let Some(node) = self.nodes.get(node_idx) {
                    for &input_id in &node.inputs {
                        if let Some(input_idx) = self.nodes.iter().position(|n| n.id == input_id) {
                            if !used[input_idx] {
                                used[input_idx] = true;
                                queue.push_back(input_idx);
                            }
                        }
                    }
                }
            }
        }

        // Remove unused nodes
        let mut removed = 0;
        self.nodes.retain(|_| {
            let keep = used[removed];
            removed += 1;
            keep
        });
    }

    /// Fuse compatible adjacent operations
    fn fuse_operations(&mut self) {
        let mut i = 0;
        while i + 1 < self.nodes.len() {
            if self.nodes[i]
                .operation
                .can_fuse_with(&self.nodes[i + 1].operation)
            {
                if let Some(fused_op) = self.nodes[i].operation.fuse(&self.nodes[i + 1].operation) {
                    // Replace first node with fused operation
                    self.nodes[i].operation = fused_op;
                    self.nodes[i].optimized = true;

                    // Remove second node
                    self.nodes.remove(i + 1);

                    // Update memory estimates
                    if let Some(input_shape) = self.get_input_shape_for_node(i) {
                        self.nodes[i].update_estimates(input_shape);
                    }

                    continue; // Check if we can fuse more
                }
            }
            i += 1;
        }
    }

    /// Reorder operations for better cache locality and parallelism
    fn reorder_operations(&mut self) {
        // Simple heuristic: move memory-intensive operations later
        // This is a placeholder for more sophisticated reordering algorithms
        self.nodes.sort_by_key(|node| node.memory_requirement);
    }

    /// Identify operations that can be parallelized
    fn parallelize_independent_operations(&mut self) {
        // Mark operations that can be executed in parallel
        // This is a placeholder for dependency analysis
        for node in &mut self.nodes {
            if matches!(
                node.operation,
                LazyOp::StandardScale | LazyOp::MinMaxScale { .. } | LazyOp::Normalize { .. }
            ) {
                node.optimized = true; // Mark as parallelizable
            }
        }
    }

    /// Get topological ordering of nodes
    pub fn topological_sort(&self) -> Vec<usize> {
        // For linear chains, this is straightforward
        (0..self.nodes.len()).collect()
    }

    /// Get total estimated memory requirement
    pub fn total_memory_requirement(&self) -> usize {
        self.total_memory
    }

    /// Check if graph fits within memory budget
    pub fn fits_in_memory(&self, budget: usize) -> bool {
        self.total_memory <= budget
    }
}

/// Lazy preprocessing pipeline
pub struct LazyPreprocessor {
    /// Configuration
    config: LazyConfig,
    /// Computation graph
    graph: LazyGraph,
    /// Memory optimizer
    memory_optimizer: MemoryOptimizer,
    /// Buffer pool for intermediate results
    buffer_pool: BufferPool<u8>,
    /// Performance profiler
    profiler: Profiler,
    /// Result cache
    cache: HashMap<String, Array2<Float>>,
}

impl LazyPreprocessor {
    /// Create a new lazy preprocessor
    pub fn new(config: LazyConfig) -> Self {
        let buffer_pool = BufferPool::new();
        let memory_optimizer = MemoryOptimizer::new();
        let profiler = Profiler::new();

        Self {
            config,
            graph: LazyGraph::new(),
            memory_optimizer,
            buffer_pool,
            profiler,
            cache: HashMap::new(),
        }
    }

    /// Add an operation to the lazy pipeline
    pub fn add_operation(&mut self, operation: LazyOp) -> &mut Self {
        self.graph.add_operation(operation);
        self
    }

    /// Evaluate the lazy computation graph
    pub fn evaluate(&mut self, input: &Array2<Float>) -> Result<Array2<Float>> {
        // Set input shape and optimize graph
        self.graph.set_input_shape((input.nrows(), input.ncols()));
        self.graph.optimize(&self.config);

        // Check memory requirements
        if !self.graph.fits_in_memory(self.config.memory_budget) {
            return self.evaluate_streaming(input);
        }

        // Execute the optimized graph
        self.execute_graph(input)
    }

    /// Execute the computation graph
    fn execute_graph(&mut self, input: &Array2<Float>) -> Result<Array2<Float>> {
        let execution_order = self.graph.topological_sort();
        let mut current_data = input.clone();

        self.profiler.start("graph_execution");

        for &node_idx in &execution_order {
            if let Some(node) = self.graph.nodes.get(node_idx) {
                current_data = self.execute_operation(&node.operation, &current_data)?;
            }
        }

        self.profiler.end("graph_execution");
        Ok(current_data)
    }

    /// Execute a single operation
    fn execute_operation(
        &self,
        operation: &LazyOp,
        input: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        match operation {
            LazyOp::StandardScale => {
                let mean = input.mean_axis(Axis(0)).unwrap();
                let std = input.std_axis(Axis(0), 1.0);
                Ok((input - &mean) / &std.mapv(|s| s.max(Float::EPSILON)))
            }

            LazyOp::MinMaxScale { feature_range } => {
                let min_vals = input.fold_axis(Axis(0), Float::INFINITY, |&acc, &val| acc.min(val));
                let max_vals =
                    input.fold_axis(Axis(0), Float::NEG_INFINITY, |&acc, &val| acc.max(val));
                let data_range = &max_vals - &min_vals;
                let feature_range_size = feature_range.1 - feature_range.0;

                let scale = data_range.mapv(|range| {
                    if range.abs() < Float::EPSILON {
                        1.0
                    } else {
                        feature_range_size / range
                    }
                });

                Ok((input - &min_vals) * &scale + feature_range.0)
            }

            LazyOp::Normalize { norm } => match norm.as_str() {
                "l2" => {
                    let norms = input
                        .mapv(|x| x * x)
                        .sum_axis(Axis(1))
                        .mapv(|x| x.sqrt().max(Float::EPSILON));
                    Ok(input / &norms.insert_axis(Axis(1)))
                }
                "l1" => {
                    let norms = input
                        .mapv(|x| x.abs())
                        .sum_axis(Axis(1))
                        .mapv(|x| x.max(Float::EPSILON));
                    Ok(input / &norms.insert_axis(Axis(1)))
                }
                _ => Err(SklearsError::InvalidInput(format!(
                    "Unknown norm: {}",
                    norm
                ))),
            },

            LazyOp::PolynomialFeatures { degree } => {
                self.generate_polynomial_features(input, *degree)
            }

            LazyOp::FeatureSelection { k } => {
                // Simple variance-based feature selection
                let variances = input.var_axis(Axis(0), 1.0);
                let mut indices: Vec<usize> = (0..input.ncols()).collect();
                indices.sort_by(|&a, &b| variances[b].partial_cmp(&variances[a]).unwrap());
                indices.truncate(*k);
                indices.sort();

                let selected_cols: Vec<_> = indices
                    .iter()
                    .map(|&i| input.column(i).to_owned())
                    .collect();

                if selected_cols.is_empty() {
                    return Err(SklearsError::InvalidInput(
                        "No features selected".to_string(),
                    ));
                }

                let mut result = Array2::zeros((input.nrows(), selected_cols.len()));
                for (col_idx, col) in selected_cols.iter().enumerate() {
                    result.column_mut(col_idx).assign(col);
                }

                Ok(result)
            }

            LazyOp::Custom { name, params: _ } => {
                match name.as_str() {
                    "standard_normalize" => {
                        // Fused standard scaling + L2 normalization
                        let mean = input.mean_axis(Axis(0)).unwrap();
                        let std = input.std_axis(Axis(0), 1.0);
                        let standardized = (input - &mean) / &std.mapv(|s| s.max(Float::EPSILON));
                        let norms = standardized
                            .mapv(|x| x * x)
                            .sum_axis(Axis(1))
                            .mapv(|x| x.sqrt().max(Float::EPSILON));
                        Ok(standardized / &norms.insert_axis(Axis(1)))
                    }
                    _ => Err(SklearsError::InvalidInput(format!(
                        "Unknown custom operation: {}",
                        name
                    ))),
                }
            }

            _ => Err(SklearsError::InvalidInput(format!(
                "Operation not implemented: {:?}",
                operation
            ))),
        }
    }

    /// Generate polynomial features
    fn generate_polynomial_features(
        &self,
        input: &Array2<Float>,
        degree: usize,
    ) -> Result<Array2<Float>> {
        if degree == 0 {
            return Ok(Array2::ones((input.nrows(), 1)));
        }
        if degree == 1 {
            return Ok(input.clone());
        }

        let n_samples = input.nrows();
        let n_features = input.ncols();
        let output_features = polynomial_feature_count(n_features, degree);

        let mut result = Array2::zeros((n_samples, output_features));
        let mut col_idx = 0;

        // Constant term
        result.column_mut(col_idx).fill(1.0);
        col_idx += 1;

        // Linear terms
        for j in 0..n_features {
            result.column_mut(col_idx).assign(&input.column(j));
            col_idx += 1;
        }

        // Higher degree terms (simplified - only degree 2 for now)
        if degree >= 2 {
            for i in 0..n_features {
                for j in i..n_features {
                    let col_i = input.column(i);
                    let col_j = input.column(j);
                    result.column_mut(col_idx).assign(&(&col_i * &col_j));
                    col_idx += 1;
                }
            }
        }

        Ok(result)
    }

    /// Evaluate using streaming for large datasets
    fn evaluate_streaming(&mut self, input: &Array2<Float>) -> Result<Array2<Float>> {
        let chunk_size = self.config.chunk_size;
        let n_samples = input.nrows();
        let mut results = Vec::new();

        for start in (0..n_samples).step_by(chunk_size) {
            let end = (start + chunk_size).min(n_samples);
            let chunk = input.slice(s![start..end, ..]).to_owned();
            let chunk_result = self.execute_graph(&chunk)?;
            results.push(chunk_result);
        }

        // Concatenate results
        if results.is_empty() {
            return Err(SklearsError::InvalidInput("No data to process".to_string()));
        }

        let total_rows: usize = results.iter().map(|r| r.nrows()).sum();
        let n_cols = results[0].ncols();
        let mut final_result = Array2::zeros((total_rows, n_cols));

        let mut row_offset = 0;
        for result in results {
            let chunk_rows = result.nrows();
            final_result
                .slice_mut(s![row_offset..row_offset + chunk_rows, ..])
                .assign(&result);
            row_offset += chunk_rows;
        }

        Ok(final_result)
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> HashMap<String, Float> {
        self.profiler.get_stats()
    }

    /// Clear the computation graph
    pub fn clear(&mut self) {
        self.graph = LazyGraph::new();
        self.cache.clear();
    }
}

/// Calculate the number of polynomial features for given input features and degree
fn polynomial_feature_count(n_features: usize, degree: usize) -> usize {
    match degree {
        0 => 1,
        1 => 1 + n_features,
        2 => 1 + n_features + (n_features * (n_features + 1)) / 2,
        _ => {
            // General formula for polynomial feature count with degree d and n features:
            // C(n + d, d) = (n + d)! / (n! * d!)
            // This is a simplified approximation
            let mut count = 1 + n_features; // constant + linear terms
            for d in 2..=degree {
                count += n_features.pow(d as u32) / factorial(d);
            }
            count
        }
    }
}

/// Calculate factorial (helper function)
fn factorial(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_lazy_config() {
        let config = LazyConfig::new()
            .with_optimization_level(2)
            .with_memory_budget(512 * 1024 * 1024)
            .with_fusion(true)
            .with_parallel(false);

        assert_eq!(config.optimization_level, 2);
        assert_eq!(config.memory_budget, 512 * 1024 * 1024);
        assert!(config.enable_fusion);
        assert!(!config.enable_parallel);
    }

    #[test]
    fn test_lazy_operation_fusion() {
        let op1 = LazyOp::StandardScale;
        let op2 = LazyOp::Normalize {
            norm: "l2".to_string(),
        };

        assert!(op1.can_fuse_with(&op2));

        let fused = op1.fuse(&op2);
        assert!(fused.is_some());
    }

    #[test]
    fn test_lazy_graph_construction() {
        let mut graph = LazyGraph::new();

        let id1 = graph.add_operation(LazyOp::StandardScale);
        let id2 = graph.add_operation(LazyOp::PolynomialFeatures { degree: 2 });
        let id3 = graph.add_operation(LazyOp::FeatureSelection { k: 10 });

        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[0].id, id1);
        assert_eq!(graph.nodes[1].id, id2);
        assert_eq!(graph.nodes[2].id, id3);
    }

    #[test]
    fn test_lazy_preprocessor_simple() -> Result<()> {
        let config = LazyConfig::new().with_optimization_level(0); // Disable optimizations for test
        let mut processor = LazyPreprocessor::new(config);

        processor.add_operation(LazyOp::StandardScale);

        let data = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0],
        )
        .unwrap();

        let result = processor.evaluate(&data)?;

        // Check that result has approximately zero mean and unit variance
        let mean = result.mean_axis(Axis(0)).unwrap();
        let std = result.std_axis(Axis(0), 1.0);

        for &m in mean.iter() {
            assert_abs_diff_eq!(m, 0.0, epsilon = 1e-10);
        }

        for &s in std.iter() {
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_polynomial_feature_count() {
        assert_eq!(polynomial_feature_count(3, 0), 1);
        assert_eq!(polynomial_feature_count(3, 1), 4); // 1 + 3
        assert_eq!(polynomial_feature_count(3, 2), 10); // 1 + 3 + 6
    }

    #[test]
    fn test_memory_estimation() {
        let op = LazyOp::PolynomialFeatures { degree: 2 };
        let memory = op.memory_footprint((1000, 10));

        // Should be reasonable memory estimate (not zero, not impossibly large)
        assert!(memory > 1000);
        assert!(memory < 1_000_000_000); // Less than 1GB for this small example
    }

    #[test]
    fn test_lazy_minmax_scaling() -> Result<()> {
        let config = LazyConfig::new().with_optimization_level(0);
        let mut processor = LazyPreprocessor::new(config);

        processor.add_operation(LazyOp::MinMaxScale {
            feature_range: (0.0, 1.0),
        });

        let data = Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
            .unwrap();

        let result = processor.evaluate(&data)?;

        // Check that all values are between 0 and 1
        for &val in result.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }

        // Check that min and max are approximately 0 and 1
        let min_vals = result.fold_axis(Axis(0), Float::INFINITY, |&acc, &val| acc.min(val));
        let max_vals = result.fold_axis(Axis(0), Float::NEG_INFINITY, |&acc, &val| acc.max(val));

        for &min_val in min_vals.iter() {
            assert_abs_diff_eq!(min_val, 0.0, epsilon = 1e-10);
        }

        for &max_val in max_vals.iter() {
            assert_abs_diff_eq!(max_val, 1.0, epsilon = 1e-10);
        }

        Ok(())
    }
}

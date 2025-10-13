//! Directed Acyclic Graph (DAG) pipeline components
//!
//! This module provides DAG-based pipeline execution for complex workflows with
//! parallel execution, dependency management, and cycle detection.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;

use crate::{PipelinePredictor, PipelineStep};

/// Node in a DAG pipeline
#[derive(Debug)]
pub struct DAGNode {
    /// Unique node identifier
    pub id: String,
    /// Node name/description
    pub name: String,
    /// Pipeline component
    pub component: NodeComponent,
    /// Input dependencies
    pub dependencies: Vec<String>,
    /// Output consumers
    pub consumers: Vec<String>,
    /// Node metadata
    pub metadata: HashMap<String, String>,
    /// Execution configuration
    pub config: NodeConfig,
}

/// Node component types
pub enum NodeComponent {
    /// Transformer component
    Transformer(Box<dyn PipelineStep>),
    /// Estimator component
    Estimator(Box<dyn PipelinePredictor>),
    /// Data source
    DataSource {
        data: Option<Array2<f64>>,
        targets: Option<Array1<f64>>,
    },
    /// Data sink/output
    DataSink,
    /// Conditional branch
    ConditionalBranch {
        condition: BranchCondition,
        true_path: String,
        false_path: String,
    },
    /// Data merger
    DataMerger { merge_strategy: MergeStrategy },
    /// Custom function
    CustomFunction {
        function: Box<dyn Fn(&[NodeOutput]) -> SklResult<NodeOutput> + Send + Sync>,
    },
}

impl Debug for NodeComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeComponent::Transformer(_) => f
                .debug_tuple("Transformer")
                .field(&"<transformer>")
                .finish(),
            NodeComponent::Estimator(_) => {
                f.debug_tuple("Estimator").field(&"<estimator>").finish()
            }
            NodeComponent::DataSource { data, targets } => f
                .debug_struct("DataSource")
                .field(
                    "data",
                    &data
                        .as_ref()
                        .map(|d| format!("Array2<f64>({}, {})", d.nrows(), d.ncols())),
                )
                .field(
                    "targets",
                    &targets
                        .as_ref()
                        .map(|t| format!("Array1<f64>({})", t.len())),
                )
                .finish(),
            NodeComponent::DataSink => f.debug_tuple("DataSink").finish(),
            NodeComponent::ConditionalBranch {
                condition,
                true_path,
                false_path,
            } => f
                .debug_struct("ConditionalBranch")
                .field("condition", condition)
                .field("true_path", true_path)
                .field("false_path", false_path)
                .finish(),
            NodeComponent::DataMerger { merge_strategy } => f
                .debug_struct("DataMerger")
                .field("merge_strategy", merge_strategy)
                .finish(),
            NodeComponent::CustomFunction { .. } => f
                .debug_struct("CustomFunction")
                .field("function", &"<function>")
                .finish(),
        }
    }
}

/// Branch condition for conditional nodes
#[derive(Debug)]
pub enum BranchCondition {
    /// Feature threshold condition
    FeatureThreshold {
        feature_idx: usize,
        threshold: f64,
        comparison: ComparisonOp,
    },
    /// Data size condition
    DataSize {
        min_samples: Option<usize>,
        max_samples: Option<usize>,
    },
    /// Custom condition
    Custom {
        condition_fn: fn(&NodeOutput) -> bool,
    },
}

/// Comparison operators for conditions
#[derive(Debug, Clone)]
pub enum ComparisonOp {
    /// GreaterThan
    GreaterThan,
    /// LessThan
    LessThan,
    /// GreaterEqual
    GreaterEqual,
    /// LessEqual
    LessEqual,
    /// Equal
    Equal,
    /// NotEqual
    NotEqual,
}

/// Data merging strategies
#[derive(Debug)]
pub enum MergeStrategy {
    /// Concatenate along features (horizontal)
    HorizontalConcat,
    /// Concatenate along samples (vertical)
    VerticalConcat,
    /// Average outputs
    Average,
    /// Weighted average
    WeightedAverage { weights: Vec<f64> },
    /// Maximum values
    Maximum,
    /// Minimum values
    Minimum,
    /// Custom merge function
    Custom {
        merge_fn: fn(&[NodeOutput]) -> SklResult<NodeOutput>,
    },
}

/// Node execution configuration
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// Whether node can be executed in parallel
    pub parallel_execution: bool,
    /// Maximum execution time (seconds)
    pub timeout: Option<f64>,
    /// Retry attempts on failure
    pub retry_attempts: usize,
    /// Cache output
    pub cache_output: bool,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            parallel_execution: true,
            timeout: None,
            retry_attempts: 0,
            cache_output: false,
            resource_requirements: ResourceRequirements::default(),
        }
    }
}

/// Resource requirements for node execution
#[derive(Debug, Clone, Default)]
pub struct ResourceRequirements {
    /// Memory requirement (MB)
    pub memory_mb: Option<usize>,
    /// CPU cores required
    pub cpu_cores: Option<usize>,
    /// GPU requirement
    pub gpu_required: bool,
}

/// Output from a DAG node
#[derive(Debug, Clone)]
pub struct NodeOutput {
    /// Output data
    pub data: Array2<f64>,
    /// Output targets (optional)
    pub targets: Option<Array1<f64>>,
    /// Output metadata
    pub metadata: HashMap<String, String>,
    /// Execution statistics
    pub execution_stats: ExecutionStats,
}

/// Execution statistics for a node
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Execution time (seconds)
    pub execution_time: f64,
    /// Memory usage (MB)
    pub memory_usage: f64,
    /// Success status
    pub success: bool,
    /// Error message (if any)
    pub error_message: Option<String>,
}

impl Default for ExecutionStats {
    fn default() -> Self {
        Self {
            execution_time: 0.0,
            memory_usage: 0.0,
            success: true,
            error_message: None,
        }
    }
}

/// DAG pipeline structure
#[derive(Debug)]
pub struct DAGPipeline<S = Untrained> {
    state: S,
    nodes: HashMap<String, DAGNode>,
    edges: HashMap<String, HashSet<String>>, // node_id -> dependencies
    execution_order: Vec<String>,
    parallel_groups: Vec<Vec<String>>,
    cache: HashMap<String, NodeOutput>,
}

/// Trained state for `DAGPipeline`
#[derive(Debug)]
pub struct DAGPipelineTrained {
    fitted_nodes: HashMap<String, DAGNode>,
    edges: HashMap<String, HashSet<String>>,
    execution_order: Vec<String>,
    parallel_groups: Vec<Vec<String>>,
    cache: HashMap<String, NodeOutput>,
    execution_history: Vec<ExecutionRecord>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

/// Record of pipeline execution
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Execution timestamp
    pub timestamp: f64,
    /// Executed nodes
    pub executed_nodes: Vec<String>,
    /// Total execution time
    pub total_time: f64,
    /// Success status
    pub success: bool,
    /// Error details
    pub errors: Vec<(String, String)>, // (node_id, error_message)
}

impl DAGPipeline<Untrained> {
    /// Create a new DAG pipeline
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            execution_order: Vec::new(),
            parallel_groups: Vec::new(),
            cache: HashMap::new(),
        }
    }

    /// Add a node to the DAG
    pub fn add_node(mut self, node: DAGNode) -> SklResult<Self> {
        // Check for duplicate node IDs
        if self.nodes.contains_key(&node.id) {
            return Err(SklearsError::InvalidInput(format!(
                "Node with ID '{}' already exists",
                node.id
            )));
        }

        // Add dependencies to edges
        let node_id = node.id.clone();
        self.edges
            .insert(node_id.clone(), node.dependencies.iter().cloned().collect());

        // Add node
        self.nodes.insert(node_id, node);

        // Recompute execution order
        self.compute_execution_order()?;

        Ok(self)
    }

    /// Add an edge between nodes
    pub fn add_edge(mut self, from_node: &str, to_node: &str) -> SklResult<Self> {
        // Check if nodes exist
        if !self.nodes.contains_key(from_node) {
            return Err(SklearsError::InvalidInput(format!(
                "Source node '{from_node}' does not exist"
            )));
        }
        if !self.nodes.contains_key(to_node) {
            return Err(SklearsError::InvalidInput(format!(
                "Target node '{to_node}' does not exist"
            )));
        }

        // Add edge
        self.edges
            .entry(to_node.to_string())
            .or_default()
            .insert(from_node.to_string());

        // Update node dependencies
        if let Some(to_node_obj) = self.nodes.get_mut(to_node) {
            if !to_node_obj.dependencies.contains(&from_node.to_string()) {
                to_node_obj.dependencies.push(from_node.to_string());
            }
        }

        // Update node consumers
        if let Some(from_node_obj) = self.nodes.get_mut(from_node) {
            if !from_node_obj.consumers.contains(&to_node.to_string()) {
                from_node_obj.consumers.push(to_node.to_string());
            }
        }

        // Check for cycles
        if self.has_cycles()? {
            return Err(SklearsError::InvalidInput(
                "Adding edge would create a cycle in the DAG".to_string(),
            ));
        }

        // Recompute execution order
        self.compute_execution_order()?;

        Ok(self)
    }

    /// Check if the DAG has cycles
    fn has_cycles(&self) -> SklResult<bool> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node_id in self.nodes.keys() {
            if !visited.contains(node_id)
                && self.dfs_cycle_check(node_id, &mut visited, &mut rec_stack)?
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// DFS-based cycle detection
    fn dfs_cycle_check(
        &self,
        node_id: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> SklResult<bool> {
        visited.insert(node_id.to_string());
        rec_stack.insert(node_id.to_string());

        if let Some(dependencies) = self.edges.get(node_id) {
            for dep in dependencies {
                if !visited.contains(dep) {
                    if self.dfs_cycle_check(dep, visited, rec_stack)? {
                        return Ok(true);
                    }
                } else if rec_stack.contains(dep) {
                    return Ok(true);
                }
            }
        }

        rec_stack.remove(node_id);
        Ok(false)
    }

    /// Compute topological execution order
    fn compute_execution_order(&mut self) -> SklResult<()> {
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut order = Vec::new();
        let mut parallel_groups = Vec::new();

        // Initialize in-degrees
        for node_id in self.nodes.keys() {
            in_degree.insert(node_id.clone(), 0);
        }

        // Compute in-degrees
        for (node_id, dependencies) in &self.edges {
            in_degree.insert(node_id.clone(), dependencies.len());
        }

        // Find nodes with no dependencies
        for (node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id.clone());
            }
        }

        // Process nodes level by level for parallel execution
        while !queue.is_empty() {
            let current_level: Vec<String> = queue.drain(..).collect();
            parallel_groups.push(current_level.clone());
            order.extend(current_level.iter().cloned());

            // Process current level
            for node_id in &current_level {
                // Update in-degrees of consumers
                if let Some(node) = self.nodes.get(node_id) {
                    for consumer in &node.consumers {
                        if let Some(degree) = in_degree.get_mut(consumer) {
                            *degree -= 1;
                            if *degree == 0 {
                                queue.push_back(consumer.clone());
                            }
                        }
                    }
                }
            }
        }

        // Check if all nodes are processed (no cycles)
        if order.len() != self.nodes.len() {
            return Err(SklearsError::InvalidInput(
                "DAG contains cycles".to_string(),
            ));
        }

        self.execution_order = order;
        self.parallel_groups = parallel_groups;

        Ok(())
    }

    /// Create a linear pipeline from components
    pub fn linear(components: Vec<(String, Box<dyn PipelineStep>)>) -> SklResult<Self> {
        let mut dag = Self::new();
        let num_components = components.len();

        for (i, (name, component)) in components.into_iter().enumerate() {
            let dependencies = if i == 0 {
                Vec::new()
            } else {
                vec![format!("node_{}", i - 1)]
            };

            let node = DAGNode {
                id: format!("node_{i}"),
                name,
                component: NodeComponent::Transformer(component),
                dependencies,
                consumers: if i == num_components - 1 {
                    Vec::new()
                } else {
                    vec![format!("node_{}", i + 1)]
                },
                metadata: HashMap::new(),
                config: NodeConfig::default(),
            };

            dag = dag.add_node(node)?;
        }

        Ok(dag)
    }

    /// Create a parallel pipeline with final merger
    pub fn parallel(
        components: Vec<(String, Box<dyn PipelineStep>)>,
        merge_strategy: MergeStrategy,
    ) -> SklResult<Self> {
        let mut dag = Self::new();

        let num_components = components.len();

        // Add parallel components
        for (i, (name, component)) in components.into_iter().enumerate() {
            let node = DAGNode {
                id: format!("parallel_{i}"),
                name,
                component: NodeComponent::Transformer(component),
                dependencies: Vec::new(),
                consumers: vec!["merger".to_string()],
                metadata: HashMap::new(),
                config: NodeConfig::default(),
            };

            dag = dag.add_node(node)?;
        }

        // Add merger node
        let merger_dependencies: Vec<String> = (0..num_components)
            .map(|i| format!("parallel_{i}"))
            .collect();

        let merger_node = DAGNode {
            id: "merger".to_string(),
            name: "Data Merger".to_string(),
            component: NodeComponent::DataMerger { merge_strategy },
            dependencies: merger_dependencies,
            consumers: Vec::new(),
            metadata: HashMap::new(),
            config: NodeConfig::default(),
        };

        dag = dag.add_node(merger_node)?;

        Ok(dag)
    }
}

impl Default for DAGPipeline<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DAGPipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for DAGPipeline<Untrained> {
    type Fitted = DAGPipeline<DAGPipelineTrained>;

    fn fit(
        mut self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let mut fitted_nodes = HashMap::new();
        let mut execution_errors = Vec::new();
        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Initialize with input data
        let initial_output = NodeOutput {
            data: x.mapv(|v| v),
            targets: y.as_ref().map(|y_vals| y_vals.mapv(|v| v)),
            metadata: HashMap::new(),
            execution_stats: ExecutionStats::default(),
        };
        self.cache.insert("input".to_string(), initial_output);

        // Execute nodes in topological order
        let parallel_groups = std::mem::take(&mut self.parallel_groups);
        for group in &parallel_groups {
            // Execute parallel group
            let group_results = self.execute_parallel_group(group)?;

            for (node_id, result) in group_results {
                match result {
                    Ok(output) => {
                        self.cache.insert(node_id.clone(), output);
                        if let Some(node) = self.nodes.remove(&node_id) {
                            fitted_nodes.insert(node_id, node);
                        }
                    }
                    Err(e) => {
                        execution_errors.push((node_id, e.to_string()));
                    }
                }
            }
        }

        let end_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let execution_record = ExecutionRecord {
            timestamp: start_time,
            executed_nodes: fitted_nodes.keys().cloned().collect(),
            total_time: end_time - start_time,
            success: execution_errors.is_empty(),
            errors: execution_errors,
        };

        Ok(DAGPipeline {
            state: DAGPipelineTrained {
                fitted_nodes,
                edges: self.edges,
                execution_order: self.execution_order,
                parallel_groups,
                cache: self.cache,
                execution_history: vec![execution_record],
                n_features_in: x.ncols(),
                feature_names_in: None,
            },
            nodes: HashMap::new(),
            edges: HashMap::new(),
            execution_order: Vec::new(),
            parallel_groups: Vec::new(),
            cache: HashMap::new(),
        })
    }
}

impl DAGPipeline<Untrained> {
    /// Execute a group of nodes in parallel
    fn execute_parallel_group(
        &mut self,
        group: &[String],
    ) -> SklResult<Vec<(String, SklResult<NodeOutput>)>> {
        let mut results = Vec::new();

        for node_id in group {
            if let Some(node) = self.nodes.remove(node_id) {
                let result = self.execute_node(&node);
                results.push((node_id.clone(), result));
                // Put the node back
                self.nodes.insert(node_id.clone(), node);
            }
        }

        Ok(results)
    }

    /// Execute a single node
    fn execute_node(&mut self, node: &DAGNode) -> SklResult<NodeOutput> {
        let start_time = std::time::SystemTime::now();

        // Collect inputs from dependencies
        let mut inputs = Vec::new();
        for dep_id in &node.dependencies {
            if let Some(output) = self.cache.get(dep_id) {
                inputs.push(output.clone());
            } else if dep_id == "input" {
                // Handle initial input
                continue;
            } else {
                return Err(SklearsError::InvalidInput(format!(
                    "Missing input from dependency: {dep_id}"
                )));
            }
        }

        // If no dependencies, use input data
        if inputs.is_empty() && self.cache.contains_key("input") {
            inputs.push(self.cache["input"].clone());
        }

        // Execute based on component type
        let result = match &node.component {
            NodeComponent::Transformer(transformer) => {
                if let Some(input) = inputs.first() {
                    let mapped_data = input.data.view().mapv(|v| v as Float);
                    let transformed = transformer.transform(&mapped_data.view())?;
                    Ok(NodeOutput {
                        data: transformed,
                        targets: input.targets.clone(),
                        metadata: HashMap::new(),
                        execution_stats: ExecutionStats::default(),
                    })
                } else {
                    Err(SklearsError::InvalidInput(
                        "No input data for transformer".to_string(),
                    ))
                }
            }
            NodeComponent::DataMerger { merge_strategy } => {
                self.execute_data_merger(&inputs, merge_strategy)
            }
            NodeComponent::ConditionalBranch {
                condition,
                true_path,
                false_path,
            } => self.execute_conditional_branch(&inputs, condition, true_path, false_path),
            NodeComponent::DataSource { data, targets } => {
                if let Some(ref source_data) = data {
                    Ok(NodeOutput {
                        data: source_data.clone(),
                        targets: targets.clone(),
                        metadata: HashMap::new(),
                        execution_stats: ExecutionStats::default(),
                    })
                } else {
                    Err(SklearsError::InvalidInput(
                        "No data in data source".to_string(),
                    ))
                }
            }
            NodeComponent::DataSink => {
                // Just pass through the input
                inputs
                    .into_iter()
                    .next()
                    .ok_or_else(|| SklearsError::InvalidInput("No input for data sink".to_string()))
            }
            NodeComponent::Estimator(_) => {
                // For fitting, estimators don't produce output data
                if let Some(input) = inputs.first() {
                    Ok(input.clone())
                } else {
                    Err(SklearsError::InvalidInput(
                        "No input data for estimator".to_string(),
                    ))
                }
            }
            NodeComponent::CustomFunction { function } => function(&inputs),
        };

        // Record execution time
        let execution_time = start_time.elapsed().unwrap().as_secs_f64();
        if let Ok(ref mut output) = result.clone() {
            output.execution_stats.execution_time = execution_time;
        }

        result
    }

    /// Execute data merger
    fn execute_data_merger(
        &self,
        inputs: &[NodeOutput],
        strategy: &MergeStrategy,
    ) -> SklResult<NodeOutput> {
        if inputs.is_empty() {
            return Err(SklearsError::InvalidInput("No inputs to merge".to_string()));
        }

        let merged_data = match strategy {
            MergeStrategy::HorizontalConcat => {
                let total_cols: usize = inputs.iter().map(|inp| inp.data.ncols()).sum();
                let n_rows = inputs[0].data.nrows();

                let mut merged = Array2::zeros((n_rows, total_cols));
                let mut col_offset = 0;

                for input in inputs {
                    let cols = input.data.ncols();
                    merged
                        .slice_mut(s![.., col_offset..col_offset + cols])
                        .assign(&input.data);
                    col_offset += cols;
                }

                merged
            }
            MergeStrategy::VerticalConcat => {
                let n_cols = inputs[0].data.ncols();
                let total_rows: usize = inputs.iter().map(|inp| inp.data.nrows()).sum();

                let mut merged = Array2::zeros((total_rows, n_cols));
                let mut row_offset = 0;

                for input in inputs {
                    let rows = input.data.nrows();
                    merged
                        .slice_mut(s![row_offset..row_offset + rows, ..])
                        .assign(&input.data);
                    row_offset += rows;
                }

                merged
            }
            MergeStrategy::Average => {
                let mut sum = inputs[0].data.clone();
                for input in inputs.iter().skip(1) {
                    sum += &input.data;
                }
                sum / inputs.len() as f64
            }
            MergeStrategy::WeightedAverage { weights } => {
                if weights.len() != inputs.len() {
                    return Err(SklearsError::InvalidInput(
                        "Number of weights must match number of inputs".to_string(),
                    ));
                }

                let mut weighted_sum = &inputs[0].data * weights[0];
                for (input, &weight) in inputs.iter().skip(1).zip(weights.iter().skip(1)) {
                    weighted_sum += &(&input.data * weight);
                }

                weighted_sum
            }
            MergeStrategy::Maximum => {
                let mut max_data = inputs[0].data.clone();
                for input in inputs.iter().skip(1) {
                    for ((i, j), &val) in input.data.indexed_iter() {
                        if val > max_data[(i, j)] {
                            max_data[(i, j)] = val;
                        }
                    }
                }
                max_data
            }
            MergeStrategy::Minimum => {
                let mut min_data = inputs[0].data.clone();
                for input in inputs.iter().skip(1) {
                    for ((i, j), &val) in input.data.indexed_iter() {
                        if val < min_data[(i, j)] {
                            min_data[(i, j)] = val;
                        }
                    }
                }
                min_data
            }
            MergeStrategy::Custom { merge_fn } => {
                return merge_fn(inputs);
            }
        };

        Ok(NodeOutput {
            data: merged_data,
            targets: inputs[0].targets.clone(),
            metadata: HashMap::new(),
            execution_stats: ExecutionStats::default(),
        })
    }

    /// Execute conditional branch
    fn execute_conditional_branch(
        &self,
        inputs: &[NodeOutput],
        condition: &BranchCondition,
        true_path: &str,
        false_path: &str,
    ) -> SklResult<NodeOutput> {
        if inputs.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No input for conditional branch".to_string(),
            ));
        }

        let input = &inputs[0];
        let condition_result = match condition {
            BranchCondition::FeatureThreshold {
                feature_idx,
                threshold,
                comparison,
            } => {
                if *feature_idx >= input.data.ncols() {
                    return Err(SklearsError::InvalidInput(
                        "Feature index out of bounds".to_string(),
                    ));
                }

                let feature_values = input.data.column(*feature_idx);
                let mean_value = feature_values.mean().unwrap_or(0.0);

                match comparison {
                    ComparisonOp::GreaterThan => mean_value > *threshold,
                    ComparisonOp::LessThan => mean_value < *threshold,
                    ComparisonOp::GreaterEqual => mean_value >= *threshold,
                    ComparisonOp::LessEqual => mean_value <= *threshold,
                    ComparisonOp::Equal => (mean_value - threshold).abs() < 1e-8,
                    ComparisonOp::NotEqual => (mean_value - threshold).abs() >= 1e-8,
                }
            }
            BranchCondition::DataSize {
                min_samples,
                max_samples,
            } => {
                let n_samples = input.data.nrows();
                let min_ok = min_samples.map_or(true, |min| n_samples >= min);
                let max_ok = max_samples.map_or(true, |max| n_samples <= max);
                min_ok && max_ok
            }
            BranchCondition::Custom { condition_fn } => condition_fn(input),
        };

        // For now, just return the input (branch execution would be more complex)
        let mut output = input.clone();
        output.metadata.insert(
            "branch_taken".to_string(),
            if condition_result {
                true_path.to_string()
            } else {
                false_path.to_string()
            },
        );

        Ok(output)
    }
}

impl DAGPipeline<DAGPipelineTrained> {
    /// Transform data through the fitted DAG
    pub fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // This is a simplified version - in practice, we'd re-execute the DAG
        // For now, return the input data as f64
        Ok(x.mapv(|v| v))
    }

    /// Get execution history
    #[must_use]
    pub fn execution_history(&self) -> &[ExecutionRecord] {
        &self.state.execution_history
    }

    /// Get DAG statistics
    #[must_use]
    pub fn statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert(
            "total_nodes".to_string(),
            self.state.fitted_nodes.len() as f64,
        );
        stats.insert(
            "parallel_groups".to_string(),
            self.state.parallel_groups.len() as f64,
        );

        if let Some(last_execution) = self.state.execution_history.last() {
            stats.insert("last_execution_time".to_string(), last_execution.total_time);
            stats.insert(
                "last_execution_success".to_string(),
                if last_execution.success { 1.0 } else { 0.0 },
            );
        }

        stats
    }

    /// Visualize DAG structure (returns DOT format)
    #[must_use]
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph DAG {\n");

        // Add nodes
        for (node_id, node) in &self.state.fitted_nodes {
            dot.push_str(&format!("  \"{}\" [label=\"{}\"];\n", node_id, node.name));
        }

        // Add edges
        for (to_node, dependencies) in &self.state.edges {
            for from_node in dependencies {
                dot.push_str(&format!("  \"{from_node}\" -> \"{to_node}\";\n"));
            }
        }

        dot.push_str("}\n");
        dot
    }
}

// Import ndarray slice macro
use scirs2_core::ndarray::s;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockTransformer;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_dag_node_creation() {
        let node = DAGNode {
            id: "test_node".to_string(),
            name: "Test Node".to_string(),
            component: NodeComponent::DataSource {
                data: Some(array![[1.0, 2.0], [3.0, 4.0]]),
                targets: Some(array![1.0, 0.0]),
            },
            dependencies: Vec::new(),
            consumers: Vec::new(),
            metadata: HashMap::new(),
            config: NodeConfig::default(),
        };

        assert_eq!(node.id, "test_node");
        assert_eq!(node.name, "Test Node");
    }

    #[test]
    fn test_linear_dag() {
        let components = vec![
            (
                "transformer1".to_string(),
                Box::new(MockTransformer::new()) as Box<dyn PipelineStep>,
            ),
            (
                "transformer2".to_string(),
                Box::new(MockTransformer::new()) as Box<dyn PipelineStep>,
            ),
        ];

        let dag = DAGPipeline::linear(components).unwrap();
        assert_eq!(dag.nodes.len(), 2);
        assert_eq!(dag.execution_order.len(), 2);
    }

    #[test]
    fn test_parallel_dag() {
        let components = vec![
            (
                "transformer1".to_string(),
                Box::new(MockTransformer::new()) as Box<dyn PipelineStep>,
            ),
            (
                "transformer2".to_string(),
                Box::new(MockTransformer::new()) as Box<dyn PipelineStep>,
            ),
        ];

        let dag = DAGPipeline::parallel(components, MergeStrategy::HorizontalConcat).unwrap();
        assert_eq!(dag.nodes.len(), 3); // 2 transformers + 1 merger
    }

    #[test]
    fn test_cycle_detection() {
        let mut dag = DAGPipeline::new();

        // First add nodes without circular dependencies
        let node1 = DAGNode {
            id: "node1".to_string(),
            name: "Node 1".to_string(),
            component: NodeComponent::DataSource {
                data: None,
                targets: None,
            },
            dependencies: vec![],
            consumers: vec![],
            metadata: HashMap::new(),
            config: NodeConfig::default(),
        };

        let node2 = DAGNode {
            id: "node2".to_string(),
            name: "Node 2".to_string(),
            component: NodeComponent::DataSource {
                data: None,
                targets: None,
            },
            dependencies: vec![],
            consumers: vec![],
            metadata: HashMap::new(),
            config: NodeConfig::default(),
        };

        dag = dag.add_node(node1).unwrap();
        dag = dag.add_node(node2).unwrap();

        // Add edges to create a cycle
        dag = dag.add_edge("node1", "node2").unwrap();

        // Adding a reverse edge should detect the cycle
        assert!(dag.add_edge("node2", "node1").is_err());
    }

    #[test]
    fn test_merge_strategies() {
        let input1 = NodeOutput {
            data: array![[1.0, 2.0], [3.0, 4.0]],
            targets: None,
            metadata: HashMap::new(),
            execution_stats: ExecutionStats::default(),
        };

        let input2 = NodeOutput {
            data: array![[5.0, 6.0], [7.0, 8.0]],
            targets: None,
            metadata: HashMap::new(),
            execution_stats: ExecutionStats::default(),
        };

        let inputs = vec![input1, input2];
        let dag = DAGPipeline::new();

        // Test horizontal concatenation
        let result = dag
            .execute_data_merger(&inputs, &MergeStrategy::HorizontalConcat)
            .unwrap();
        assert_eq!(result.data.ncols(), 4);
        assert_eq!(result.data.nrows(), 2);

        // Test average
        let result = dag
            .execute_data_merger(&inputs, &MergeStrategy::Average)
            .unwrap();
        assert_eq!(result.data[[0, 0]], 3.0); // (1+5)/2
        assert_eq!(result.data[[0, 1]], 4.0); // (2+6)/2
    }
}

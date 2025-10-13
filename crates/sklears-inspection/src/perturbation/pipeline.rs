//! Perturbation pipeline implementation
//!
//! This module provides a flexible perturbation pipeline for chaining multiple
//! perturbation methods and managing their execution.

use super::core::{
    ExecutionCondition, ExecutionGraph, ExecutionMode, ExecutionNode, ExecutionStatus,
    PerturbationConfig, PerturbationStage, PerturbationStrategy, PipelineConfig, PipelineMetadata,
    PipelineResult, StageQualityMetrics, StageResult,
};
use super::strategies::generate_perturbations;
use crate::{Float, SklResult, SklearsError};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array2, ArrayView2};
use std::collections::HashMap;
use std::sync::Mutex;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Flexible perturbation pipeline for chaining multiple perturbation methods
#[derive(Debug, Clone)]
pub struct PerturbationPipeline {
    /// Pipeline stages to execute
    pub stages: Vec<PerturbationStage>,
    /// Pipeline configuration
    pub config: PipelineConfig,
    /// Execution metadata
    pub metadata: PipelineMetadata,
}

impl PerturbationPipeline {
    /// Create a new perturbation pipeline
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            stages: Vec::new(),
            config,
            metadata: PipelineMetadata {
                total_execution_time_ms: 0,
                peak_memory_usage_bytes: 0,
                stages_executed: 0,
                stages_skipped: 0,
                stages_failed: 0,
                success_rate: 0.0,
            },
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(&mut self, stage: PerturbationStage) {
        self.stages.push(stage);
    }

    /// Create a new pipeline with fluent API
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Execute the pipeline
    pub fn execute(&mut self, X: &ArrayView2<Float>) -> SklResult<PipelineResult> {
        let start_time = std::time::Instant::now();
        let mut stage_results = HashMap::new();
        let mut execution_graph = ExecutionGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
        };

        match self.config.execution_mode {
            ExecutionMode::Sequential => {
                self.execute_sequential(X, &mut stage_results, &mut execution_graph)?;
            }
            ExecutionMode::Parallel => {
                self.execute_parallel(X, &mut stage_results, &mut execution_graph)?;
            }
            ExecutionMode::Conditional => {
                self.execute_conditional(X, &mut stage_results, &mut execution_graph)?;
            }
            ExecutionMode::Branching => {
                self.execute_branching(X, &mut stage_results, &mut execution_graph)?;
            }
        }

        // Combine results from all stages
        let final_perturbed_data = self
            .combine_stage_results(&stage_results)
            .unwrap_or_else(|_| Vec::new());

        // Update metadata
        let total_time = start_time.elapsed().as_millis() as u64;
        self.metadata.total_execution_time_ms = total_time;
        self.metadata.stages_executed = stage_results.values().filter(|r| r.success).count();
        self.metadata.stages_failed = stage_results.values().filter(|r| !r.success).count();
        self.metadata.success_rate =
            self.metadata.stages_executed as Float / stage_results.len() as Float;

        Ok(PipelineResult {
            metadata: self.metadata.clone(),
            stage_results,
            final_perturbed_data,
            execution_graph,
        })
    }

    /// Execute stages sequentially
    fn execute_sequential(
        &self,
        X: &ArrayView2<Float>,
        stage_results: &mut HashMap<String, StageResult>,
        execution_graph: &mut ExecutionGraph,
    ) -> SklResult<()> {
        for stage in &self.stages {
            if !stage.enabled {
                continue;
            }

            let stage_start = std::time::Instant::now();
            execution_graph.nodes.push(ExecutionNode {
                stage_id: stage.id.clone(),
                status: ExecutionStatus::Running,
                start_time: stage_start,
                end_time: None,
            });

            let result = self.execute_stage(stage, X, stage_results)?;
            stage_results.insert(stage.id.clone(), result);

            // Update execution graph node
            if let Some(node) = execution_graph
                .nodes
                .iter_mut()
                .find(|n| n.stage_id == stage.id)
            {
                node.status = if stage_results[&stage.id].success {
                    ExecutionStatus::Completed
                } else {
                    ExecutionStatus::Failed
                };
                node.end_time = Some(std::time::Instant::now());
            }
        }
        Ok(())
    }

    /// Execute stages in parallel
    #[cfg(feature = "parallel")]
    fn execute_parallel(
        &self,
        X: &ArrayView2<Float>,
        stage_results: &mut HashMap<String, StageResult>,
        execution_graph: &mut ExecutionGraph,
    ) -> SklResult<()> {
        use std::sync::Arc;

        let results = Arc::new(Mutex::new(HashMap::<String, StageResult>::new()));
        let graph = Arc::new(Mutex::new(execution_graph.clone()));

        // Sort stages by priority for parallel execution
        let mut sorted_stages = self.stages.clone();
        sorted_stages.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Group stages that can run in parallel (no dependencies)
        let mut parallel_groups = Vec::new();
        let mut remaining_stages = sorted_stages;

        while !remaining_stages.is_empty() {
            let mut current_group = Vec::new();
            let mut i = 0;

            while i < remaining_stages.len() {
                let stage = &remaining_stages[i];
                let results_guard = results.lock().unwrap();

                // Check if dependencies are satisfied
                let dependencies_satisfied = stage
                    .dependencies
                    .iter()
                    .all(|dep| results_guard.contains_key(dep) && results_guard[dep].success);

                drop(results_guard);

                if dependencies_satisfied && current_group.len() < self.config.max_parallel_stages {
                    current_group.push(remaining_stages.remove(i));
                } else {
                    i += 1;
                }
            }

            if current_group.is_empty() && !remaining_stages.is_empty() {
                return Err(SklearsError::InvalidOperation(
                    "Circular dependency detected in pipeline stages".to_string(),
                ));
            }

            parallel_groups.push(current_group);
        }

        // Execute each group in parallel
        for group in parallel_groups {
            let parallel_results: Vec<_> = group
                .par_iter()
                .filter(|stage| stage.enabled)
                .map(|stage| {
                    let stage_start = std::time::Instant::now();

                    // Update execution graph
                    {
                        let mut graph_guard = graph.lock().unwrap();
                        graph_guard.nodes.push(ExecutionNode {
                            stage_id: stage.id.clone(),
                            status: ExecutionStatus::Running,
                            start_time: stage_start,
                            end_time: None,
                        });
                    }

                    let results_guard = results.lock().unwrap();
                    let result = self.execute_stage(stage, X, &results_guard);
                    drop(results_guard);

                    (stage.id.clone(), result, stage_start)
                })
                .collect();

            // Collect results
            for (stage_id, result, stage_start) in parallel_results {
                match result {
                    Ok(stage_result) => {
                        let mut results_guard = results.lock().unwrap();
                        results_guard.insert(stage_id.clone(), stage_result);
                        drop(results_guard);

                        // Update execution graph
                        let mut graph_guard = graph.lock().unwrap();
                        if let Some(node) = graph_guard
                            .nodes
                            .iter_mut()
                            .find(|n| n.stage_id == stage_id)
                        {
                            node.status = ExecutionStatus::Completed;
                            node.end_time = Some(std::time::Instant::now());
                        }
                    }
                    Err(_) => {
                        let mut graph_guard = graph.lock().unwrap();
                        if let Some(node) = graph_guard
                            .nodes
                            .iter_mut()
                            .find(|n| n.stage_id == stage_id)
                        {
                            node.status = ExecutionStatus::Failed;
                            node.end_time = Some(std::time::Instant::now());
                        }
                    }
                }
            }
        }

        let final_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        *stage_results = final_results;
        *execution_graph = Arc::try_unwrap(graph).unwrap().into_inner().unwrap();

        Ok(())
    }

    /// Execute stages in parallel (non-parallel fallback)
    #[cfg(not(feature = "parallel"))]
    fn execute_parallel(
        &self,
        X: &ArrayView2<Float>,
        stage_results: &mut HashMap<String, StageResult>,
        execution_graph: &mut ExecutionGraph,
    ) -> SklResult<()> {
        // Fallback to sequential execution when parallel feature is not enabled
        self.execute_sequential(X, stage_results, execution_graph)
    }

    /// Execute stages conditionally
    fn execute_conditional(
        &self,
        X: &ArrayView2<Float>,
        stage_results: &mut HashMap<String, StageResult>,
        execution_graph: &mut ExecutionGraph,
    ) -> SklResult<()> {
        for stage in &self.stages {
            if !stage.enabled {
                continue;
            }

            // Check execution condition
            if let Some(condition) = &stage.condition {
                if !self.evaluate_condition(condition, X, stage_results) {
                    // Skip this stage
                    execution_graph.nodes.push(ExecutionNode {
                        stage_id: stage.id.clone(),
                        status: ExecutionStatus::Skipped,
                        start_time: std::time::Instant::now(),
                        end_time: Some(std::time::Instant::now()),
                    });
                    continue;
                }
            }

            let stage_start = std::time::Instant::now();
            execution_graph.nodes.push(ExecutionNode {
                stage_id: stage.id.clone(),
                status: ExecutionStatus::Running,
                start_time: stage_start,
                end_time: None,
            });

            let result = self.execute_stage(stage, X, stage_results)?;
            stage_results.insert(stage.id.clone(), result);

            // Update execution graph node
            if let Some(node) = execution_graph
                .nodes
                .iter_mut()
                .find(|n| n.stage_id == stage.id)
            {
                node.status = if stage_results[&stage.id].success {
                    ExecutionStatus::Completed
                } else {
                    ExecutionStatus::Failed
                };
                node.end_time = Some(std::time::Instant::now());
            }
        }
        Ok(())
    }

    /// Execute stages in branching pattern
    fn execute_branching(
        &self,
        X: &ArrayView2<Float>,
        stage_results: &mut HashMap<String, StageResult>,
        execution_graph: &mut ExecutionGraph,
    ) -> SklResult<()> {
        // Create dependency graph
        let dependency_graph = self.build_dependency_graph();

        // Topological sort to determine execution order
        let execution_order = self.topological_sort(&dependency_graph)?;

        for stage_id in execution_order {
            let stage = self.stages.iter().find(|s| s.id == stage_id).unwrap();

            if !stage.enabled {
                continue;
            }

            // Check if dependencies are satisfied
            let dependencies_satisfied = stage
                .dependencies
                .iter()
                .all(|dep| stage_results.contains_key(dep) && stage_results[dep].success);

            if !dependencies_satisfied {
                execution_graph.nodes.push(ExecutionNode {
                    stage_id: stage.id.clone(),
                    status: ExecutionStatus::Skipped,
                    start_time: std::time::Instant::now(),
                    end_time: Some(std::time::Instant::now()),
                });
                continue;
            }

            let stage_start = std::time::Instant::now();
            execution_graph.nodes.push(ExecutionNode {
                stage_id: stage.id.clone(),
                status: ExecutionStatus::Running,
                start_time: stage_start,
                end_time: None,
            });

            let result = self.execute_stage(stage, X, stage_results)?;
            stage_results.insert(stage.id.clone(), result);

            // Update execution graph node
            if let Some(node) = execution_graph
                .nodes
                .iter_mut()
                .find(|n| n.stage_id == stage.id)
            {
                node.status = if stage_results[&stage.id].success {
                    ExecutionStatus::Completed
                } else {
                    ExecutionStatus::Failed
                };
                node.end_time = Some(std::time::Instant::now());
            }
        }

        Ok(())
    }

    /// Execute a single stage
    fn execute_stage(
        &self,
        stage: &PerturbationStage,
        X: &ArrayView2<Float>,
        _stage_results: &HashMap<String, StageResult>,
    ) -> SklResult<StageResult> {
        let start_time = std::time::Instant::now();
        let mut retry_count = 0;

        loop {
            match generate_perturbations(X, &stage.config) {
                Ok(perturbed_data) => {
                    let execution_time = start_time.elapsed().as_millis() as u64;
                    let memory_usage = std::mem::size_of_val(&perturbed_data);

                    // Calculate quality metrics
                    let quality_metrics = self.calculate_stage_quality_metrics(X, &perturbed_data);

                    return Ok(StageResult {
                        stage_id: stage.id.clone(),
                        success: true,
                        perturbed_data: Some(perturbed_data),
                        execution_time_ms: execution_time,
                        memory_usage_bytes: memory_usage,
                        quality_metrics,
                        error_message: None,
                        retry_count,
                    });
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count >= stage.max_retries {
                        let execution_time = start_time.elapsed().as_millis() as u64;
                        return Ok(StageResult {
                            stage_id: stage.id.clone(),
                            success: false,
                            perturbed_data: None,
                            execution_time_ms: execution_time,
                            memory_usage_bytes: 0,
                            quality_metrics: StageQualityMetrics {
                                perturbation_magnitude: 0.0,
                                diversity_score: 0.0,
                                robustness_score: 0.0,
                                coverage_score: 0.0,
                            },
                            error_message: Some(e.to_string()),
                            retry_count,
                        });
                    }
                }
            }
        }
    }

    /// Evaluate execution condition
    fn evaluate_condition(
        &self,
        condition: &ExecutionCondition,
        X: &ArrayView2<Float>,
        stage_results: &HashMap<String, StageResult>,
    ) -> bool {
        match condition {
            ExecutionCondition::DataCharacteristics {
                min_samples,
                max_samples,
                min_features,
                max_features,
                sparsity_threshold,
            } => {
                let n_samples = X.nrows();
                let n_features = X.ncols();

                if let Some(min) = min_samples {
                    if n_samples < *min {
                        return false;
                    }
                }
                if let Some(max) = max_samples {
                    if n_samples > *max {
                        return false;
                    }
                }
                if let Some(min) = min_features {
                    if n_features < *min {
                        return false;
                    }
                }
                if let Some(max) = max_features {
                    if n_features > *max {
                        return false;
                    }
                }
                if let Some(threshold) = sparsity_threshold {
                    let zero_count = X.iter().filter(|&&x| x == 0.0).count();
                    let sparsity = zero_count as Float / (n_samples * n_features) as Float;
                    if sparsity < *threshold {
                        return false;
                    }
                }
                true
            }
            ExecutionCondition::PreviousStageResult {
                stage_id,
                success_required,
                quality_threshold,
            } => {
                if let Some(result) = stage_results.get(stage_id) {
                    if *success_required && !result.success {
                        return false;
                    }
                    if let Some(threshold) = quality_threshold {
                        if result.quality_metrics.robustness_score < *threshold {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            }
            ExecutionCondition::Custom { condition_fn } => condition_fn(X, stage_results),
        }
    }

    /// Build dependency graph
    fn build_dependency_graph(&self) -> HashMap<String, Vec<String>> {
        let mut graph = HashMap::new();
        for stage in &self.stages {
            graph.insert(stage.id.clone(), stage.dependencies.clone());
        }
        graph
    }

    /// Topological sort for dependency resolution
    fn topological_sort(&self, graph: &HashMap<String, Vec<String>>) -> SklResult<Vec<String>> {
        let mut in_degree = HashMap::new();
        let mut adj_list = HashMap::new();

        // Initialize
        for stage in &self.stages {
            in_degree.insert(stage.id.clone(), 0);
            adj_list.insert(stage.id.clone(), Vec::new());
        }

        // Build adjacency list and calculate in-degrees
        for (stage_id, dependencies) in graph {
            for dep in dependencies {
                adj_list.get_mut(dep).unwrap().push(stage_id.clone());
                *in_degree.get_mut(stage_id).unwrap() += 1;
            }
        }

        // Kahn's algorithm
        let mut queue = std::collections::VecDeque::new();
        let mut result = Vec::new();

        // Add nodes with no incoming edges
        for (stage_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(stage_id.clone());
            }
        }

        while let Some(current) = queue.pop_front() {
            result.push(current.clone());

            for neighbor in &adj_list[&current] {
                *in_degree.get_mut(neighbor).unwrap() -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor.clone());
                }
            }
        }

        if result.len() != self.stages.len() {
            return Err(SklearsError::InvalidOperation(
                "Circular dependency detected in pipeline".to_string(),
            ));
        }

        Ok(result)
    }

    /// Calculate quality metrics for a stage
    fn calculate_stage_quality_metrics(
        &self,
        X: &ArrayView2<Float>,
        perturbed_data: &[Array2<Float>],
    ) -> StageQualityMetrics {
        let mut total_magnitude = 0.0;
        let mut diversity_scores = Vec::new();

        for perturbed in perturbed_data {
            // Calculate perturbation magnitude
            let magnitude = X
                .iter()
                .zip(perturbed.iter())
                .map(|(&orig, &pert)| (orig - pert).abs())
                .sum::<Float>()
                / (X.nrows() * X.ncols()) as Float;
            total_magnitude += magnitude;

            // Calculate diversity (simplified as variance)
            let mean = perturbed.mean().unwrap_or(0.0);
            let variance = perturbed.iter().map(|&x| (x - mean).powi(2)).sum::<Float>()
                / perturbed.len() as Float;
            diversity_scores.push(variance.sqrt());
        }

        let avg_magnitude = total_magnitude / perturbed_data.len() as Float;
        let avg_diversity =
            diversity_scores.iter().sum::<Float>() / diversity_scores.len() as Float;

        // Simplified quality metrics
        StageQualityMetrics {
            perturbation_magnitude: avg_magnitude,
            diversity_score: avg_diversity,
            robustness_score: 1.0 / (1.0 + avg_magnitude), // Higher magnitude = lower robustness
            coverage_score: avg_diversity / (avg_diversity + 1.0), // Normalize diversity
        }
    }

    /// Combine results from all stages
    fn combine_stage_results(
        &self,
        stage_results: &HashMap<String, StageResult>,
    ) -> SklResult<Vec<Array2<Float>>> {
        let mut combined_data = Vec::new();

        for result in stage_results.values() {
            if result.success {
                if let Some(data) = &result.perturbed_data {
                    combined_data.extend(data.clone());
                }
            }
        }

        if combined_data.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "No successful stage results to combine".to_string(),
            ));
        }

        Ok(combined_data)
    }
}

/// Builder for creating perturbation pipelines
pub struct PipelineBuilder {
    config: PipelineConfig,
    stages: Vec<PerturbationStage>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
            stages: Vec::new(),
        }
    }

    /// Set execution mode
    pub fn execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.config.execution_mode = mode;
        self
    }

    /// Set maximum parallel stages
    pub fn max_parallel_stages(mut self, max: usize) -> Self {
        self.config.max_parallel_stages = max;
        self
    }

    /// Enable caching
    pub fn enable_caching(mut self, enable: bool) -> Self {
        self.config.enable_caching = enable;
        self
    }

    /// Set memory limit
    pub fn memory_limit_mb(mut self, limit: usize) -> Self {
        self.config.memory_limit_mb = limit;
        self
    }

    /// Add a stage
    pub fn add_stage(mut self, stage: PerturbationStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Add a simple perturbation stage
    pub fn add_perturbation_stage(
        mut self,
        id: String,
        name: String,
        strategy: PerturbationStrategy,
        magnitude: Float,
        n_samples: usize,
    ) -> Self {
        let stage = PerturbationStage {
            id,
            name,
            config: PerturbationConfig {
                strategy,
                magnitude,
                n_samples,
                ..Default::default()
            },
            condition: None,
            dependencies: Vec::new(),
            enabled: true,
            priority: 0,
            max_retries: 3,
        };
        self.stages.push(stage);
        self
    }

    /// Build the pipeline
    pub fn build(self) -> PerturbationPipeline {
        let mut pipeline = PerturbationPipeline::new(self.config);
        for stage in self.stages {
            pipeline.add_stage(stage);
        }
        pipeline
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

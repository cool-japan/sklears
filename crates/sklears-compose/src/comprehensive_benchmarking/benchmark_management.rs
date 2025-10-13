//! Benchmark Management System
//!
//! This module provides comprehensive benchmark management capabilities including
//! benchmark registration, execution scheduling, suite organization, and task coordination.

use super::config_types::*;
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::thread;
use serde::{Serialize, Deserialize};

// ================================================================================================
// CORE BENCHMARK MANAGER
// ================================================================================================

/// Benchmark manager for organizing and executing benchmarks
pub struct BenchmarkManager {
    registered_benchmarks: HashMap<String, BenchmarkDefinition>,
    benchmark_suites: HashMap<String, BenchmarkSuite>,
    execution_queue: VecDeque<BenchmarkExecution>,
    scheduler: BenchmarkScheduler,
    result_store: BenchmarkResultStore,
}

impl BenchmarkManager {
    /// Create a new benchmark manager
    pub fn new() -> Self {
        Self {
            registered_benchmarks: HashMap::new(),
            benchmark_suites: HashMap::new(),
            execution_queue: VecDeque::new(),
            scheduler: BenchmarkScheduler::new(),
            result_store: BenchmarkResultStore::new(),
        }
    }

    /// Register a new benchmark definition
    pub fn register_benchmark(&mut self, benchmark_def: BenchmarkDefinition) -> Result<(), BenchmarkingError> {
        if self.registered_benchmarks.contains_key(&benchmark_def.benchmark_id) {
            return Err(BenchmarkingError::ConfigurationError(
                format!("Benchmark {} already registered", benchmark_def.benchmark_id)
            ));
        }

        // Validate benchmark definition
        self.validate_benchmark_definition(&benchmark_def)?;

        self.registered_benchmarks.insert(benchmark_def.benchmark_id.clone(), benchmark_def);
        Ok(())
    }

    /// Register a benchmark suite
    pub fn register_benchmark_suite(&mut self, suite: BenchmarkSuite) -> Result<(), BenchmarkingError> {
        // Validate that all referenced benchmarks exist
        for benchmark_id in &suite.benchmark_ids {
            if !self.registered_benchmarks.contains_key(benchmark_id) {
                return Err(BenchmarkingError::ConfigurationError(
                    format!("Benchmark {} referenced in suite {} not found", benchmark_id, suite.suite_id)
                ));
            }
        }

        // Validate dependencies
        self.validate_suite_dependencies(&suite)?;

        self.benchmark_suites.insert(suite.suite_id.clone(), suite);
        Ok(())
    }

    /// Execute a single benchmark
    pub fn execute_benchmark(&mut self, benchmark_id: &str, parameters: HashMap<String, String>) -> Result<String, BenchmarkingError> {
        let benchmark_def = self.registered_benchmarks.get(benchmark_id)
            .ok_or_else(|| BenchmarkingError::ConfigurationError(format!("Benchmark {} not found", benchmark_id)))?;

        // Validate parameters
        self.validate_benchmark_parameters(benchmark_def, &parameters)?;

        let execution_id = self.generate_execution_id(benchmark_id);
        let execution = BenchmarkExecution {
            execution_id: execution_id.clone(),
            benchmark_id: benchmark_id.to_string(),
            suite_id: None,
            start_time: SystemTime::now(),
            end_time: None,
            status: ExecutionStatus::Queued,
            parameters,
            results: None,
            error_info: None,
            resource_usage: ResourceUsageInfo::default(),
            execution_metadata: ExecutionMetadata::default(),
        };

        self.execution_queue.push_back(execution);
        self.scheduler.schedule_next_execution(&mut self.execution_queue)?;

        Ok(execution_id)
    }

    /// Execute a benchmark suite
    pub fn execute_benchmark_suite(&mut self, suite_id: &str, global_parameters: HashMap<String, String>) -> Result<Vec<String>, BenchmarkingError> {
        let suite = self.benchmark_suites.get(suite_id)
            .ok_or_else(|| BenchmarkingError::ConfigurationError(format!("Suite {} not found", suite_id)))?
            .clone();

        let mut execution_ids = Vec::new();

        match suite.execution_order {
            ExecutionOrder::Sequential => {
                for benchmark_id in &suite.benchmark_ids {
                    let execution_id = self.execute_benchmark_in_suite(benchmark_id, &suite, &global_parameters)?;
                    execution_ids.push(execution_id);
                }
            },
            ExecutionOrder::Parallel => {
                for benchmark_id in &suite.benchmark_ids {
                    let execution_id = self.execute_benchmark_in_suite(benchmark_id, &suite, &global_parameters)?;
                    execution_ids.push(execution_id);
                }
            },
            ExecutionOrder::Dependency => {
                let execution_order = self.resolve_dependency_order(&suite)?;
                for benchmark_id in execution_order {
                    let execution_id = self.execute_benchmark_in_suite(&benchmark_id, &suite, &global_parameters)?;
                    execution_ids.push(execution_id);
                }
            },
            ExecutionOrder::Priority => {
                let mut prioritized_benchmarks = suite.benchmark_ids.clone();
                prioritized_benchmarks.sort_by_key(|id| {
                    self.registered_benchmarks.get(id)
                        .map(|def| def.priority.unwrap_or(0))
                        .unwrap_or(0)
                });

                for benchmark_id in prioritized_benchmarks {
                    let execution_id = self.execute_benchmark_in_suite(&benchmark_id, &suite, &global_parameters)?;
                    execution_ids.push(execution_id);
                }
            },
            ExecutionOrder::Custom(_) => {
                return Err(BenchmarkingError::ConfigurationError("Custom execution order not implemented".to_string()));
            }
        }

        Ok(execution_ids)
    }

    /// Get execution result
    pub fn get_execution_result(&self, execution_id: &str) -> Result<BenchmarkResult, BenchmarkingError> {
        self.result_store.retrieve_result(execution_id)
    }

    /// List all registered benchmarks
    pub fn list_benchmarks(&self) -> Vec<&BenchmarkDefinition> {
        self.registered_benchmarks.values().collect()
    }

    /// List all registered suites
    pub fn list_suites(&self) -> Vec<&BenchmarkSuite> {
        self.benchmark_suites.values().collect()
    }

    /// Get benchmark execution status
    pub fn get_execution_status(&self, execution_id: &str) -> Option<ExecutionStatus> {
        self.execution_queue.iter()
            .find(|exec| exec.execution_id == execution_id)
            .map(|exec| exec.status.clone())
    }

    /// Cancel a benchmark execution
    pub fn cancel_execution(&mut self, execution_id: &str) -> Result<(), BenchmarkingError> {
        if let Some(execution) = self.execution_queue.iter_mut()
            .find(|exec| exec.execution_id == execution_id) {
            execution.status = ExecutionStatus::Cancelled;
            execution.end_time = Some(SystemTime::now());
            Ok(())
        } else {
            Err(BenchmarkingError::ExecutionError(format!("Execution {} not found", execution_id)))
        }
    }

    /// Get benchmark definition by ID
    pub fn get_benchmark_definition(&self, benchmark_id: &str) -> Option<&BenchmarkDefinition> {
        self.registered_benchmarks.get(benchmark_id)
    }

    /// Get suite definition by ID
    pub fn get_suite_definition(&self, suite_id: &str) -> Option<&BenchmarkSuite> {
        self.benchmark_suites.get(suite_id)
    }

    /// Update benchmark configuration
    pub fn update_benchmark_config(&mut self, benchmark_id: &str, config: ExecutionConfig) -> Result<(), BenchmarkingError> {
        if let Some(benchmark) = self.registered_benchmarks.get_mut(benchmark_id) {
            benchmark.execution_config = config;
            Ok(())
        } else {
            Err(BenchmarkingError::ConfigurationError(format!("Benchmark {} not found", benchmark_id)))
        }
    }

    // Private helper methods
    fn validate_benchmark_definition(&self, benchmark_def: &BenchmarkDefinition) -> Result<(), BenchmarkingError> {
        // Validate required fields
        if benchmark_def.benchmark_id.is_empty() {
            return Err(BenchmarkingError::ValidationError("Benchmark ID cannot be empty".to_string()));
        }

        if benchmark_def.name.is_empty() {
            return Err(BenchmarkingError::ValidationError("Benchmark name cannot be empty".to_string()));
        }

        // Validate parameters
        for (param_name, param_def) in &benchmark_def.parameters {
            if param_def.required && param_def.default_value.is_none() {
                // This is fine - required parameters don't need defaults
            }
        }

        Ok(())
    }

    fn validate_benchmark_parameters(&self, benchmark_def: &BenchmarkDefinition, parameters: &HashMap<String, String>) -> Result<(), BenchmarkingError> {
        // Check required parameters
        for (param_name, param_def) in &benchmark_def.parameters {
            if param_def.required && !parameters.contains_key(param_name) && param_def.default_value.is_none() {
                return Err(BenchmarkingError::ValidationError(
                    format!("Required parameter {} not provided", param_name)
                ));
            }
        }

        // Validate parameter types and ranges
        for (param_name, param_value) in parameters {
            if let Some(param_def) = benchmark_def.parameters.get(param_name) {
                self.validate_parameter_value(param_def, param_value)?;
            }
        }

        Ok(())
    }

    fn validate_parameter_value(&self, param_def: &ParameterDefinition, value: &str) -> Result<(), BenchmarkingError> {
        match &param_def.parameter_type {
            ParameterType::Integer => {
                let parsed: i64 = value.parse()
                    .map_err(|_| BenchmarkingError::ValidationError(format!("Invalid integer value: {}", value)))?;

                if let Some(ParameterRange::IntegerRange(min, max)) = &param_def.valid_range {
                    if parsed < *min || parsed > *max {
                        return Err(BenchmarkingError::ValidationError(
                            format!("Integer value {} out of range [{}, {}]", parsed, min, max)
                        ));
                    }
                }
            },
            ParameterType::Float => {
                let parsed: f64 = value.parse()
                    .map_err(|_| BenchmarkingError::ValidationError(format!("Invalid float value: {}", value)))?;

                if let Some(ParameterRange::FloatRange(min, max)) = &param_def.valid_range {
                    if parsed < *min || parsed > *max {
                        return Err(BenchmarkingError::ValidationError(
                            format!("Float value {} out of range [{}, {}]", parsed, min, max)
                        ));
                    }
                }
            },
            ParameterType::String => {
                if let Some(ParameterRange::StringLength(min, max)) = &param_def.valid_range {
                    if value.len() < *min || value.len() > *max {
                        return Err(BenchmarkingError::ValidationError(
                            format!("String length {} out of range [{}, {}]", value.len(), min, max)
                        ));
                    }
                }
            },
            ParameterType::Boolean => {
                value.parse::<bool>()
                    .map_err(|_| BenchmarkingError::ValidationError(format!("Invalid boolean value: {}", value)))?;
            },
            _ => {
                // For complex types, we'll do basic validation
            }
        }

        Ok(())
    }

    fn validate_suite_dependencies(&self, suite: &BenchmarkSuite) -> Result<(), BenchmarkingError> {
        // Check for circular dependencies
        let mut visited = std::collections::HashSet::new();
        let mut recursion_stack = std::collections::HashSet::new();

        for benchmark_id in &suite.benchmark_ids {
            if self.has_circular_dependency(benchmark_id, &suite.dependencies, &mut visited, &mut recursion_stack) {
                return Err(BenchmarkingError::ConfigurationError(
                    format!("Circular dependency detected in suite {}", suite.suite_id)
                ));
            }
        }

        Ok(())
    }

    fn has_circular_dependency(
        &self,
        benchmark_id: &str,
        dependencies: &HashMap<String, Vec<String>>,
        visited: &mut std::collections::HashSet<String>,
        recursion_stack: &mut std::collections::HashSet<String>
    ) -> bool {
        visited.insert(benchmark_id.to_string());
        recursion_stack.insert(benchmark_id.to_string());

        if let Some(deps) = dependencies.get(benchmark_id) {
            for dep in deps {
                if !visited.contains(dep) {
                    if self.has_circular_dependency(dep, dependencies, visited, recursion_stack) {
                        return true;
                    }
                } else if recursion_stack.contains(dep) {
                    return true;
                }
            }
        }

        recursion_stack.remove(benchmark_id);
        false
    }

    fn resolve_dependency_order(&self, suite: &BenchmarkSuite) -> Result<Vec<String>, BenchmarkingError> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for benchmark_id in &suite.benchmark_ids {
            if !visited.contains(benchmark_id) {
                self.dependency_dfs(benchmark_id, &suite.dependencies, &mut visited, &mut result);
            }
        }

        Ok(result)
    }

    fn dependency_dfs(
        &self,
        benchmark_id: &str,
        dependencies: &HashMap<String, Vec<String>>,
        visited: &mut std::collections::HashSet<String>,
        result: &mut Vec<String>
    ) {
        visited.insert(benchmark_id.to_string());

        if let Some(deps) = dependencies.get(benchmark_id) {
            for dep in deps {
                if !visited.contains(dep) {
                    self.dependency_dfs(dep, dependencies, visited, result);
                }
            }
        }

        result.push(benchmark_id.to_string());
    }

    fn execute_benchmark_in_suite(&mut self, benchmark_id: &str, suite: &BenchmarkSuite, global_parameters: &HashMap<String, String>) -> Result<String, BenchmarkingError> {
        let mut parameters = global_parameters.clone();

        // Add suite-specific parameters if any
        if let Some(suite_params) = suite.suite_parameters.get(benchmark_id) {
            parameters.extend(suite_params.clone());
        }

        let execution_id = self.generate_execution_id(benchmark_id);
        let execution = BenchmarkExecution {
            execution_id: execution_id.clone(),
            benchmark_id: benchmark_id.to_string(),
            suite_id: Some(suite.suite_id.clone()),
            start_time: SystemTime::now(),
            end_time: None,
            status: ExecutionStatus::Queued,
            parameters,
            results: None,
            error_info: None,
            resource_usage: ResourceUsageInfo::default(),
            execution_metadata: ExecutionMetadata {
                suite_context: Some(suite.suite_id.clone()),
                execution_environment: std::env::consts::OS.to_string(),
                git_commit: self.get_git_commit().unwrap_or_default(),
                tags: suite.tags.clone().unwrap_or_default(),
            },
        };

        self.execution_queue.push_back(execution);
        Ok(execution_id)
    }

    fn generate_execution_id(&self, benchmark_id: &str) -> String {
        format!("exec_{}_{}", benchmark_id, SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis())
    }

    fn get_git_commit(&self) -> Option<String> {
        // In a real implementation, this would get the current git commit
        None
    }
}

impl Default for BenchmarkManager {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// BENCHMARK DEFINITIONS AND SUITES
// ================================================================================================

/// Definition of a benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDefinition {
    pub benchmark_id: String,
    pub name: String,
    pub description: String,
    pub category: BenchmarkCategory,
    pub version: String,
    pub author: String,
    pub dependencies: Vec<String>,
    pub parameters: HashMap<String, ParameterDefinition>,
    pub expected_metrics: Vec<PerformanceMetric>,
    pub execution_config: ExecutionConfig,
    pub validation_rules: Vec<ValidationRule>,
    pub priority: Option<u32>,
    pub tags: Vec<String>,
    pub documentation_url: Option<String>,
}

impl BenchmarkDefinition {
    /// Create a new benchmark definition builder
    pub fn builder(benchmark_id: &str) -> BenchmarkDefinitionBuilder {
        BenchmarkDefinitionBuilder::new(benchmark_id)
    }

    /// Get parameter definition by name
    pub fn get_parameter(&self, name: &str) -> Option<&ParameterDefinition> {
        self.parameters.get(name)
    }

    /// Check if benchmark has required parameters
    pub fn has_required_parameters(&self) -> bool {
        self.parameters.values().any(|param| param.required)
    }

    /// Get required parameter names
    pub fn required_parameters(&self) -> Vec<&str> {
        self.parameters.iter()
            .filter_map(|(name, param)| if param.required { Some(name.as_str()) } else { None })
            .collect()
    }
}

/// Builder for benchmark definitions
pub struct BenchmarkDefinitionBuilder {
    definition: BenchmarkDefinition,
}

impl BenchmarkDefinitionBuilder {
    pub fn new(benchmark_id: &str) -> Self {
        Self {
            definition: BenchmarkDefinition {
                benchmark_id: benchmark_id.to_string(),
                name: benchmark_id.to_string(),
                description: String::new(),
                category: BenchmarkCategory::Performance,
                version: "1.0.0".to_string(),
                author: "Unknown".to_string(),
                dependencies: Vec::new(),
                parameters: HashMap::new(),
                expected_metrics: Vec::new(),
                execution_config: ExecutionConfig::default(),
                validation_rules: Vec::new(),
                priority: None,
                tags: Vec::new(),
                documentation_url: None,
            }
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.definition.name = name.to_string();
        self
    }

    pub fn description(mut self, description: &str) -> Self {
        self.definition.description = description.to_string();
        self
    }

    pub fn category(mut self, category: BenchmarkCategory) -> Self {
        self.definition.category = category;
        self
    }

    pub fn version(mut self, version: &str) -> Self {
        self.definition.version = version.to_string();
        self
    }

    pub fn author(mut self, author: &str) -> Self {
        self.definition.author = author.to_string();
        self
    }

    pub fn add_parameter(mut self, name: &str, parameter: ParameterDefinition) -> Self {
        self.definition.parameters.insert(name.to_string(), parameter);
        self
    }

    pub fn add_metric(mut self, metric: PerformanceMetric) -> Self {
        self.definition.expected_metrics.push(metric);
        self
    }

    pub fn execution_config(mut self, config: ExecutionConfig) -> Self {
        self.definition.execution_config = config;
        self
    }

    pub fn priority(mut self, priority: u32) -> Self {
        self.definition.priority = Some(priority);
        self
    }

    pub fn add_tag(mut self, tag: &str) -> Self {
        self.definition.tags.push(tag.to_string());
        self
    }

    pub fn build(self) -> BenchmarkDefinition {
        self.definition
    }
}

/// Benchmark suite combining multiple benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub suite_id: String,
    pub name: String,
    pub description: String,
    pub benchmark_ids: Vec<String>,
    pub execution_order: ExecutionOrder,
    pub dependencies: HashMap<String, Vec<String>>,
    pub suite_config: BenchmarkingConfig,
    pub suite_parameters: HashMap<String, HashMap<String, String>>,
    pub tags: Option<Vec<String>>,
    pub timeout: Option<Duration>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite builder
    pub fn builder(suite_id: &str) -> BenchmarkSuiteBuilder {
        BenchmarkSuiteBuilder::new(suite_id)
    }

    /// Get total estimated duration
    pub fn estimated_duration(&self) -> Option<Duration> {
        self.timeout
    }

    /// Check if benchmark is in suite
    pub fn contains_benchmark(&self, benchmark_id: &str) -> bool {
        self.benchmark_ids.contains(&benchmark_id.to_string())
    }
}

/// Builder for benchmark suites
pub struct BenchmarkSuiteBuilder {
    suite: BenchmarkSuite,
}

impl BenchmarkSuiteBuilder {
    pub fn new(suite_id: &str) -> Self {
        Self {
            suite: BenchmarkSuite {
                suite_id: suite_id.to_string(),
                name: suite_id.to_string(),
                description: String::new(),
                benchmark_ids: Vec::new(),
                execution_order: ExecutionOrder::Sequential,
                dependencies: HashMap::new(),
                suite_config: BenchmarkingConfig::default(),
                suite_parameters: HashMap::new(),
                tags: None,
                timeout: None,
            }
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.suite.name = name.to_string();
        self
    }

    pub fn description(mut self, description: &str) -> Self {
        self.suite.description = description.to_string();
        self
    }

    pub fn add_benchmark(mut self, benchmark_id: &str) -> Self {
        self.suite.benchmark_ids.push(benchmark_id.to_string());
        self
    }

    pub fn execution_order(mut self, order: ExecutionOrder) -> Self {
        self.suite.execution_order = order;
        self
    }

    pub fn add_dependency(mut self, benchmark_id: &str, dependency: &str) -> Self {
        self.suite.dependencies
            .entry(benchmark_id.to_string())
            .or_insert_with(Vec::new)
            .push(dependency.to_string());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.suite.timeout = Some(timeout);
        self
    }

    pub fn build(self) -> BenchmarkSuite {
        self.suite
    }
}

// ================================================================================================
// BENCHMARK EXECUTION TRACKING
// ================================================================================================

/// Benchmark execution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkExecution {
    pub execution_id: String,
    pub benchmark_id: String,
    pub suite_id: Option<String>,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub status: ExecutionStatus,
    pub parameters: HashMap<String, String>,
    pub results: Option<BenchmarkResult>,
    pub error_info: Option<ExecutionError>,
    pub resource_usage: ResourceUsageInfo,
    pub execution_metadata: ExecutionMetadata,
}

impl BenchmarkExecution {
    /// Get execution duration
    pub fn duration(&self) -> Option<Duration> {
        self.end_time.map(|end| end.duration_since(self.start_time).unwrap_or_default())
    }

    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.status, ExecutionStatus::Completed | ExecutionStatus::Failed | ExecutionStatus::Cancelled | ExecutionStatus::Timeout)
    }

    /// Mark execution as started
    pub fn mark_started(&mut self) {
        self.status = ExecutionStatus::Running;
        self.start_time = SystemTime::now();
    }

    /// Mark execution as completed with results
    pub fn mark_completed(&mut self, result: BenchmarkResult) {
        self.status = ExecutionStatus::Completed;
        self.end_time = Some(SystemTime::now());
        self.results = Some(result);
    }

    /// Mark execution as failed with error
    pub fn mark_failed(&mut self, error: ExecutionError) {
        self.status = ExecutionStatus::Failed;
        self.end_time = Some(SystemTime::now());
        self.error_info = Some(error);
    }
}

/// Execution metadata for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    pub suite_context: Option<String>,
    pub execution_environment: String,
    pub git_commit: String,
    pub tags: Vec<String>,
}

impl Default for ExecutionMetadata {
    fn default() -> Self {
        Self {
            suite_context: None,
            execution_environment: std::env::consts::OS.to_string(),
            git_commit: String::new(),
            tags: Vec::new(),
        }
    }
}

// ================================================================================================
// BENCHMARK SCHEDULER
// ================================================================================================

/// Benchmark scheduler for execution management
pub struct BenchmarkScheduler {
    scheduling_policy: SchedulingPolicy,
    resource_constraints: ResourceConstraints,
    execution_pools: HashMap<String, ExecutionPool>,
    priority_queue: BTreeMap<u32, VecDeque<String>>,
    active_executions: HashMap<String, BenchmarkExecution>,
}

impl BenchmarkScheduler {
    /// Create a new benchmark scheduler
    pub fn new() -> Self {
        Self {
            scheduling_policy: SchedulingPolicy::Priority,
            resource_constraints: ResourceConstraints::default(),
            execution_pools: HashMap::new(),
            priority_queue: BTreeMap::new(),
            active_executions: HashMap::new(),
        }
    }

    /// Configure scheduling policy
    pub fn set_scheduling_policy(&mut self, policy: SchedulingPolicy) {
        self.scheduling_policy = policy;
    }

    /// Set resource constraints
    pub fn set_resource_constraints(&mut self, constraints: ResourceConstraints) {
        self.resource_constraints = constraints;
    }

    /// Add execution pool
    pub fn add_execution_pool(&mut self, pool_name: &str, pool: ExecutionPool) {
        self.execution_pools.insert(pool_name.to_string(), pool);
    }

    /// Schedule next execution from queue
    pub fn schedule_next_execution(&mut self, execution_queue: &mut VecDeque<BenchmarkExecution>) -> Result<(), BenchmarkingError> {
        // Check if we can start more executions
        if self.active_executions.len() >= self.resource_constraints.max_concurrent_executions as usize {
            return Ok(()); // Wait for some executions to complete
        }

        match self.scheduling_policy {
            SchedulingPolicy::FIFO => {
                if let Some(execution) = execution_queue.pop_front() {
                    self.start_execution(execution)?;
                }
            },
            SchedulingPolicy::Priority => {
                self.schedule_by_priority(execution_queue)?;
            },
            SchedulingPolicy::ResourceBased => {
                self.schedule_by_resources(execution_queue)?;
            },
            SchedulingPolicy::Adaptive => {
                self.schedule_adaptively(execution_queue)?;
            },
            SchedulingPolicy::LoadBalanced => {
                self.schedule_load_balanced(execution_queue)?;
            },
            SchedulingPolicy::Custom(_) => {
                return Err(BenchmarkingError::ConfigurationError("Custom scheduling policy not implemented".to_string()));
            }
        }

        Ok(())
    }

    /// Get active executions count
    pub fn active_executions_count(&self) -> usize {
        self.active_executions.len()
    }

    /// Check if scheduler can accept more executions
    pub fn can_accept_execution(&self) -> bool {
        self.active_executions.len() < self.resource_constraints.max_concurrent_executions as usize
    }

    // Private helper methods
    fn start_execution(&mut self, mut execution: BenchmarkExecution) -> Result<(), BenchmarkingError> {
        execution.mark_started();
        let execution_id = execution.execution_id.clone();
        self.active_executions.insert(execution_id.clone(), execution);

        // In a real implementation, this would actually start the benchmark execution
        // For now, we'll just track it

        Ok(())
    }

    fn schedule_by_priority(&mut self, execution_queue: &mut VecDeque<BenchmarkExecution>) -> Result<(), BenchmarkingError> {
        // Find highest priority execution
        let mut highest_priority_index = None;
        let mut highest_priority = 0u32;

        for (index, execution) in execution_queue.iter().enumerate() {
            // Priority could be determined by benchmark priority, execution time, etc.
            let priority = 1; // Placeholder
            if priority > highest_priority {
                highest_priority = priority;
                highest_priority_index = Some(index);
            }
        }

        if let Some(index) = highest_priority_index {
            if let Some(execution) = execution_queue.remove(index) {
                self.start_execution(execution)?;
            }
        }

        Ok(())
    }

    fn schedule_by_resources(&mut self, execution_queue: &mut VecDeque<BenchmarkExecution>) -> Result<(), BenchmarkingError> {
        // Find execution that fits current resource availability
        for i in 0..execution_queue.len() {
            if self.check_resource_availability(&execution_queue[i]) {
                if let Some(execution) = execution_queue.remove(i) {
                    self.start_execution(execution)?;
                    break;
                }
            }
        }
        Ok(())
    }

    fn schedule_adaptively(&mut self, execution_queue: &mut VecDeque<BenchmarkExecution>) -> Result<(), BenchmarkingError> {
        // Adaptive scheduling based on current system load and execution history
        // This is a placeholder for a more sophisticated algorithm
        self.schedule_by_priority(execution_queue)
    }

    fn schedule_load_balanced(&mut self, execution_queue: &mut VecDeque<BenchmarkExecution>) -> Result<(), BenchmarkingError> {
        // Distribute executions across available execution pools
        if let Some(execution) = execution_queue.pop_front() {
            self.start_execution(execution)?;
        }
        Ok(())
    }

    fn check_resource_availability(&self, execution: &BenchmarkExecution) -> bool {
        // Check if system has enough resources for this execution
        // This is a placeholder - real implementation would check actual system resources
        true
    }
}

impl Default for BenchmarkScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// EXECUTION POOLS AND TASKS
// ================================================================================================

/// Execution pool for parallel benchmark execution
pub struct ExecutionPool {
    pool_name: String,
    max_workers: usize,
    active_executions: HashMap<String, BenchmarkExecution>,
    worker_threads: Vec<thread::JoinHandle<()>>,
    task_queue: VecDeque<BenchmarkTask>,
}

impl ExecutionPool {
    /// Create a new execution pool
    pub fn new(pool_name: &str, max_workers: usize) -> Self {
        Self {
            pool_name: pool_name.to_string(),
            max_workers,
            active_executions: HashMap::new(),
            worker_threads: Vec::new(),
            task_queue: VecDeque::new(),
        }
    }

    /// Submit a task to the pool
    pub fn submit_task(&mut self, task: BenchmarkTask) -> Result<(), BenchmarkingError> {
        if self.active_executions.len() >= self.max_workers {
            self.task_queue.push_back(task);
        } else {
            self.execute_task(task)?;
        }
        Ok(())
    }

    /// Get pool statistics
    pub fn get_statistics(&self) -> ExecutionPoolStatistics {
        ExecutionPoolStatistics {
            pool_name: self.pool_name.clone(),
            max_workers: self.max_workers,
            active_workers: self.active_executions.len(),
            queued_tasks: self.task_queue.len(),
            total_completed: 0, // Would track this in real implementation
        }
    }

    fn execute_task(&mut self, task: BenchmarkTask) -> Result<(), BenchmarkingError> {
        // In a real implementation, this would spawn a worker thread to execute the task
        // For now, we'll just track it
        Ok(())
    }
}

/// Statistics for execution pools
#[derive(Debug, Clone)]
pub struct ExecutionPoolStatistics {
    pub pool_name: String,
    pub max_workers: usize,
    pub active_workers: usize,
    pub queued_tasks: usize,
    pub total_completed: usize,
}

/// Benchmark task for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTask {
    pub task_id: String,
    pub benchmark_id: String,
    pub execution_id: String,
    pub parameters: HashMap<String, String>,
    pub priority: u32,
    pub estimated_duration: Option<Duration>,
    pub resource_requirements: ResourceRequirements,
    pub retry_count: u32,
    pub max_retries: u32,
}

impl BenchmarkTask {
    /// Create a new benchmark task
    pub fn new(
        task_id: &str,
        benchmark_id: &str,
        execution_id: &str,
        parameters: HashMap<String, String>
    ) -> Self {
        Self {
            task_id: task_id.to_string(),
            benchmark_id: benchmark_id.to_string(),
            execution_id: execution_id.to_string(),
            parameters,
            priority: 0,
            estimated_duration: None,
            resource_requirements: ResourceRequirements::default(),
            retry_count: 0,
            max_retries: 3,
        }
    }

    /// Check if task can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

// ================================================================================================
// RESULT STORAGE
// ================================================================================================

/// Benchmark result store for persistence
pub struct BenchmarkResultStore {
    storage_backend: Box<dyn StorageBackend>,
    indexing_strategy: IndexingStrategy,
    retention_policy: RetentionPolicy,
    compression_config: CompressionConfig,
}

impl BenchmarkResultStore {
    /// Create a new result store
    pub fn new() -> Self {
        Self {
            storage_backend: Box::new(InMemoryStorage::new()),
            indexing_strategy: IndexingStrategy::BTree,
            retention_policy: RetentionPolicy::default(),
            compression_config: CompressionConfig::default(),
        }
    }

    /// Store a benchmark result
    pub fn store_result(&mut self, result: &BenchmarkResult) -> Result<String, BenchmarkingError> {
        self.storage_backend.store_result(result)
            .map_err(|e| BenchmarkingError::StorageError(format!("Failed to store result: {:?}", e)))
    }

    /// Retrieve a benchmark result
    pub fn retrieve_result(&self, result_id: &str) -> Result<BenchmarkResult, BenchmarkingError> {
        self.storage_backend.retrieve_result(result_id)
            .map_err(|e| BenchmarkingError::StorageError(format!("Failed to retrieve result: {:?}", e)))
    }

    /// Query results with filters
    pub fn query_results(&self, query: &ResultQuery) -> Result<Vec<BenchmarkResult>, BenchmarkingError> {
        self.storage_backend.query_results(query)
            .map_err(|e| BenchmarkingError::StorageError(format!("Failed to query results: {:?}", e)))
    }

    /// Get storage information
    pub fn get_storage_info(&self) -> StorageInfo {
        self.storage_backend.get_storage_info()
    }
}

impl Default for BenchmarkResultStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage backend trait for benchmark results
pub trait StorageBackend: Send + Sync {
    fn store_result(&mut self, result: &BenchmarkResult) -> Result<String, StorageError>;
    fn retrieve_result(&self, result_id: &str) -> Result<BenchmarkResult, StorageError>;
    fn query_results(&self, query: &ResultQuery) -> Result<Vec<BenchmarkResult>, StorageError>;
    fn delete_result(&mut self, result_id: &str) -> Result<(), StorageError>;
    fn get_storage_info(&self) -> StorageInfo;
}

/// In-memory storage implementation for testing
pub struct InMemoryStorage {
    results: HashMap<String, BenchmarkResult>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }
}

impl StorageBackend for InMemoryStorage {
    fn store_result(&mut self, result: &BenchmarkResult) -> Result<String, StorageError> {
        let result_id = result.result_id.clone();
        self.results.insert(result_id.clone(), result.clone());
        Ok(result_id)
    }

    fn retrieve_result(&self, result_id: &str) -> Result<BenchmarkResult, StorageError> {
        self.results.get(result_id)
            .cloned()
            .ok_or_else(|| StorageError::NotFound(result_id.to_string()))
    }

    fn query_results(&self, query: &ResultQuery) -> Result<Vec<BenchmarkResult>, StorageError> {
        let mut results: Vec<BenchmarkResult> = self.results.values().cloned().collect();

        // Apply filters
        if let Some(benchmark_ids) = &query.benchmark_ids {
            results.retain(|r| benchmark_ids.contains(&r.benchmark_id));
        }

        if let Some(time_range) = &query.time_range {
            results.retain(|r| r.timestamp >= time_range.start_time && r.timestamp <= time_range.end_time);
        }

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit as usize);
        }

        Ok(results)
    }

    fn delete_result(&mut self, result_id: &str) -> Result<(), StorageError> {
        if self.results.remove(result_id).is_some() {
            Ok(())
        } else {
            Err(StorageError::NotFound(result_id.to_string()))
        }
    }

    fn get_storage_info(&self) -> StorageInfo {
        StorageInfo {
            total_size: 1024 * 1024 * 1024, // 1GB placeholder
            used_size: self.results.len() as u64 * 1024, // Rough estimate
            result_count: self.results.len() as u64,
            compression_ratio: 1.0,
        }
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// SUPPORTING TYPES
// ================================================================================================

/// Query structure for result retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultQuery {
    pub benchmark_ids: Option<Vec<String>>,
    pub time_range: Option<TimeRange>,
    pub metric_filters: Option<HashMap<String, MetricFilter>>,
    pub limit: Option<u32>,
    pub sort_by: Option<String>,
    pub sort_order: Option<SortOrder>,
}

/// Time range for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start_time: SystemTime,
    pub end_time: SystemTime,
}

/// Metric filter for result queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricFilter {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub operator: ComparisonOperator,
}

/// Comparison operators for filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Between,
}

/// Sort order for result queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Storage information
#[derive(Debug, Clone)]
pub struct StorageInfo {
    pub total_size: u64,
    pub used_size: u64,
    pub result_count: u64,
    pub compression_ratio: f64,
}

/// Storage errors
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Result not found: {0}")]
    NotFound(String),
    #[error("Storage error: {0}")]
    IOError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Compression error: {0}")]
    CompressionError(String),
}

/// Benchmark result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub result_id: String,
    pub benchmark_id: String,
    pub execution_id: String,
    pub timestamp: SystemTime,
    pub metrics: HashMap<String, f64>,
    pub execution_info: ExecutionInfo,
    pub raw_data: Option<Vec<u8>>,
    pub metadata: HashMap<String, String>,
}

/// Execution information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionInfo {
    pub duration: Duration,
    pub success: bool,
    pub error_message: Option<String>,
    pub resource_usage: ResourceUsageInfo,
    pub environment_info: EnvironmentInfo,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub hostname: String,
    pub os_version: String,
    pub cpu_info: String,
    pub memory_info: String,
    pub compiler_version: String,
}

impl Default for EnvironmentInfo {
    fn default() -> Self {
        Self {
            hostname: "unknown".to_string(),
            os_version: std::env::consts::OS.to_string(),
            cpu_info: "unknown".to_string(),
            memory_info: "unknown".to_string(),
            compiler_version: "unknown".to_string(),
        }
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_manager_creation() {
        let manager = BenchmarkManager::new();
        assert_eq!(manager.list_benchmarks().len(), 0);
        assert_eq!(manager.list_suites().len(), 0);
    }

    #[test]
    fn test_benchmark_definition_builder() {
        let definition = BenchmarkDefinition::builder("test_benchmark")
            .name("Test Benchmark")
            .description("A test benchmark")
            .category(BenchmarkCategory::Performance)
            .author("Test Author")
            .priority(100)
            .add_tag("test")
            .build();

        assert_eq!(definition.benchmark_id, "test_benchmark");
        assert_eq!(definition.name, "Test Benchmark");
        assert_eq!(definition.description, "A test benchmark");
        assert!(matches!(definition.category, BenchmarkCategory::Performance));
        assert_eq!(definition.priority, Some(100));
        assert!(definition.tags.contains(&"test".to_string()));
    }

    #[test]
    fn test_benchmark_suite_builder() {
        let suite = BenchmarkSuite::builder("test_suite")
            .name("Test Suite")
            .description("A test suite")
            .add_benchmark("benchmark1")
            .add_benchmark("benchmark2")
            .execution_order(ExecutionOrder::Parallel)
            .build();

        assert_eq!(suite.suite_id, "test_suite");
        assert_eq!(suite.name, "Test Suite");
        assert_eq!(suite.benchmark_ids.len(), 2);
        assert!(suite.contains_benchmark("benchmark1"));
        assert!(matches!(suite.execution_order, ExecutionOrder::Parallel));
    }

    #[test]
    fn test_benchmark_registration() {
        let mut manager = BenchmarkManager::new();
        let definition = BenchmarkDefinition::builder("test_benchmark")
            .name("Test Benchmark")
            .build();

        let result = manager.register_benchmark(definition);
        assert!(result.is_ok());
        assert_eq!(manager.list_benchmarks().len(), 1);
    }

    #[test]
    fn test_duplicate_benchmark_registration() {
        let mut manager = BenchmarkManager::new();
        let definition1 = BenchmarkDefinition::builder("test_benchmark")
            .name("Test Benchmark 1")
            .build();
        let definition2 = BenchmarkDefinition::builder("test_benchmark")
            .name("Test Benchmark 2")
            .build();

        assert!(manager.register_benchmark(definition1).is_ok());
        assert!(manager.register_benchmark(definition2).is_err());
    }

    #[test]
    fn test_benchmark_task_creation() {
        let mut parameters = HashMap::new();
        parameters.insert("param1".to_string(), "value1".to_string());

        let task = BenchmarkTask::new("task1", "benchmark1", "exec1", parameters);
        assert_eq!(task.task_id, "task1");
        assert_eq!(task.benchmark_id, "benchmark1");
        assert_eq!(task.execution_id, "exec1");
        assert!(task.can_retry());
    }

    #[test]
    fn test_execution_pool_creation() {
        let pool = ExecutionPool::new("test_pool", 4);
        let stats = pool.get_statistics();
        assert_eq!(stats.pool_name, "test_pool");
        assert_eq!(stats.max_workers, 4);
        assert_eq!(stats.active_workers, 0);
        assert_eq!(stats.queued_tasks, 0);
    }

    #[test]
    fn test_result_store_operations() {
        let mut store = BenchmarkResultStore::new();
        let result = BenchmarkResult {
            result_id: "result1".to_string(),
            benchmark_id: "benchmark1".to_string(),
            execution_id: "exec1".to_string(),
            timestamp: SystemTime::now(),
            metrics: HashMap::new(),
            execution_info: ExecutionInfo {
                duration: Duration::from_secs(1),
                success: true,
                error_message: None,
                resource_usage: ResourceUsageInfo::default(),
                environment_info: EnvironmentInfo::default(),
            },
            raw_data: None,
            metadata: HashMap::new(),
        };

        let store_result = store.store_result(&result);
        assert!(store_result.is_ok());

        let retrieved_result = store.retrieve_result("result1");
        assert!(retrieved_result.is_ok());
        assert_eq!(retrieved_result.unwrap().result_id, "result1");
    }

    #[test]
    fn test_parameter_validation() {
        let manager = BenchmarkManager::new();
        let mut param_def = ParameterDefinition {
            parameter_name: "test_param".to_string(),
            parameter_type: ParameterType::Integer,
            default_value: None,
            valid_range: Some(ParameterRange::IntegerRange(1, 100)),
            description: "Test parameter".to_string(),
            required: true,
        };

        // Valid value
        assert!(manager.validate_parameter_value(&param_def, "50").is_ok());

        // Invalid value (out of range)
        assert!(manager.validate_parameter_value(&param_def, "150").is_err());

        // Invalid value (not integer)
        assert!(manager.validate_parameter_value(&param_def, "not_a_number").is_err());
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = BenchmarkScheduler::new();
        assert!(scheduler.can_accept_execution());
        assert_eq!(scheduler.active_executions_count(), 0);
    }
}
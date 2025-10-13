//! Execution hooks and middleware for pipeline execution
//!
//! This module provides a flexible hook system that allows users to inject custom logic
//! at various stages of pipeline execution. Hooks can be used for logging, monitoring,
//! data validation, performance measurement, and custom preprocessing/postprocessing.

use std::any::Any;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Note: async_trait would normally be imported here for AsyncExecutionHook
// use async_trait::async_trait;

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Fit, Predict, SklearsError, Transform},
    traits::Estimator,
    types::{Float, FloatBounds},
};
use std::collections::HashMap;

/// Hook execution phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookPhase {
    /// Before pipeline execution starts
    BeforeExecution,
    /// Before each step in the pipeline
    BeforeStep,
    /// After each step in the pipeline
    AfterStep,
    /// After pipeline execution completes
    AfterExecution,
    /// When an error occurs during execution
    OnError,
    /// Before fitting the pipeline
    BeforeFit,
    /// After fitting the pipeline
    AfterFit,
    /// Before prediction
    BeforePredict,
    /// After prediction
    AfterPredict,
    /// Before transformation
    BeforeTransform,
    /// After transformation
    AfterTransform,
}

/// Execution context passed to hooks
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Unique execution ID
    pub execution_id: String,
    /// Current step name (if applicable)
    pub step_name: Option<String>,
    /// Current step index (if applicable)
    pub step_index: Option<usize>,
    /// Total number of steps
    pub total_steps: usize,
    /// Execution start time
    pub start_time: Instant,
    /// Current phase
    pub phase: HookPhase,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Performance metrics tracked during execution
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total execution time
    pub total_duration: Duration,
    /// Time spent in each step
    pub step_durations: HashMap<String, Duration>,
    /// Memory usage statistics
    pub memory_usage: MemoryUsage,
    /// Data shape information
    pub data_shapes: Vec<(usize, usize)>,
    /// Error count
    pub error_count: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Current memory usage in bytes
    pub current_memory: usize,
    /// Memory allocations count
    pub allocations: usize,
}

/// Hook execution result
#[derive(Debug, Clone)]
pub enum HookResult {
    /// Continue normal execution
    Continue,
    /// Skip the current step
    Skip,
    /// Abort execution with error
    Abort(String),
    /// Continue with modified data
    ContinueWithData(HookData),
}

/// Data that can be passed between hooks and pipeline steps
#[derive(Debug, Clone)]
pub enum HookData {
    /// Input features
    Features(Array2<Float>),
    /// Target values
    Targets(Array1<Float>),
    /// Predictions
    Predictions(Array1<Float>),
    /// Custom data
    Custom(Arc<dyn Any + Send + Sync>),
}

/// Trait for implementing execution hooks
pub trait ExecutionHook: Send + Sync + Debug {
    /// Execute the hook
    fn execute(
        &mut self,
        context: &ExecutionContext,
        data: Option<&HookData>,
    ) -> SklResult<HookResult>;

    /// Get hook name
    fn name(&self) -> &str;

    /// Get hook priority (higher values execute first)
    fn priority(&self) -> i32 {
        0
    }

    /// Check if hook should execute for given phase
    fn should_execute(&self, phase: HookPhase) -> bool;
}

/// Hook manager for managing and executing hooks
#[derive(Debug)]
pub struct HookManager {
    hooks: HashMap<HookPhase, Vec<Box<dyn ExecutionHook>>>,
    execution_stack: Vec<ExecutionContext>,
    global_metrics: Arc<Mutex<PerformanceMetrics>>,
}

impl HookManager {
    /// Create a new hook manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            hooks: HashMap::new(),
            execution_stack: Vec::new(),
            global_metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
        }
    }

    /// Register a hook for specific phases
    pub fn register_hook(&mut self, hook: Box<dyn ExecutionHook>, phases: Vec<HookPhase>) {
        // For now, we'll only add the hook to the first phase
        // In a real implementation, you'd need to handle multi-phase hooks differently
        if let Some(&first_phase) = phases.first() {
            self.hooks.entry(first_phase).or_default().push(hook);

            // Sort hooks by priority (descending)
            if let Some(hooks) = self.hooks.get_mut(&first_phase) {
                hooks.sort_by(|a, b| b.priority().cmp(&a.priority()));
            }
        }
    }

    /// Execute hooks for a specific phase
    pub fn execute_hooks(
        &mut self,
        phase: HookPhase,
        context: &mut ExecutionContext,
        data: Option<&HookData>,
    ) -> SklResult<HookResult> {
        context.phase = phase;

        if let Some(hooks) = self.hooks.get_mut(&phase) {
            for hook in hooks {
                if hook.should_execute(phase) {
                    match hook.execute(context, data)? {
                        HookResult::Continue => continue,
                        HookResult::Skip => return Ok(HookResult::Skip),
                        HookResult::Abort(msg) => return Ok(HookResult::Abort(msg)),
                        HookResult::ContinueWithData(modified_data) => {
                            return Ok(HookResult::ContinueWithData(modified_data));
                        }
                    }
                }
            }
        }

        Ok(HookResult::Continue)
    }

    /// Create a new execution context
    #[must_use]
    pub fn create_context(&self, execution_id: String, total_steps: usize) -> ExecutionContext {
        /// ExecutionContext
        ExecutionContext {
            execution_id,
            step_name: None,
            step_index: None,
            total_steps,
            start_time: Instant::now(),
            phase: HookPhase::BeforeExecution,
            metadata: HashMap::new(),
            metrics: PerformanceMetrics::default(),
        }
    }

    /// Push execution context onto stack
    pub fn push_context(&mut self, context: ExecutionContext) {
        self.execution_stack.push(context);
    }

    /// Pop execution context from stack
    pub fn pop_context(&mut self) -> Option<ExecutionContext> {
        self.execution_stack.pop()
    }

    /// Get current execution context
    #[must_use]
    pub fn current_context(&self) -> Option<&ExecutionContext> {
        self.execution_stack.last()
    }

    /// Get mutable current execution context
    pub fn current_context_mut(&mut self) -> Option<&mut ExecutionContext> {
        self.execution_stack.last_mut()
    }

    /// Update global metrics
    pub fn update_global_metrics<F>(&self, updater: F)
    where
        F: FnOnce(&mut PerformanceMetrics),
    {
        if let Ok(mut metrics) = self.global_metrics.lock() {
            updater(&mut metrics);
        }
    }

    /// Get global metrics snapshot
    #[must_use]
    pub fn global_metrics(&self) -> PerformanceMetrics {
        self.global_metrics.lock().unwrap().clone()
    }
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Logging hook for pipeline execution
#[derive(Debug, Clone)]
pub struct LoggingHook {
    name: String,
    log_level: LogLevel,
    include_data_shapes: bool,
    include_timing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogLevel {
    /// Debug
    Debug,
    /// Info
    Info,
    /// Warn
    Warn,
    /// Error
    Error,
}

impl LoggingHook {
    /// Create a new logging hook
    #[must_use]
    pub fn new(name: String, log_level: LogLevel) -> Self {
        Self {
            name,
            log_level,
            include_data_shapes: true,
            include_timing: true,
        }
    }

    /// Set whether to include data shapes in logs
    #[must_use]
    pub fn include_data_shapes(mut self, include: bool) -> Self {
        self.include_data_shapes = include;
        self
    }

    /// Set whether to include timing information
    #[must_use]
    pub fn include_timing(mut self, include: bool) -> Self {
        self.include_timing = include;
        self
    }
}

impl ExecutionHook for LoggingHook {
    fn execute(
        &mut self,
        context: &ExecutionContext,
        data: Option<&HookData>,
    ) -> SklResult<HookResult> {
        let mut log_message = format!(
            "[{}] Phase: {:?}, Execution: {}",
            self.name, context.phase, context.execution_id
        );

        if let Some(step_name) = &context.step_name {
            log_message.push_str(&format!(", Step: {step_name}"));
        }

        if self.include_timing {
            let elapsed = context.start_time.elapsed();
            log_message.push_str(&format!(", Elapsed: {elapsed:?}"));
        }

        if self.include_data_shapes {
            if let Some(data) = data {
                match data {
                    HookData::Features(array) => {
                        log_message.push_str(&format!(
                            ", Features: {}x{}",
                            array.nrows(),
                            array.ncols()
                        ));
                    }
                    HookData::Targets(array) => {
                        log_message.push_str(&format!(", Targets: {}", array.len()));
                    }
                    HookData::Predictions(array) => {
                        log_message.push_str(&format!(", Predictions: {}", array.len()));
                    }
                    HookData::Custom(_) => {
                        log_message.push_str(", Data: Custom");
                    }
                }
            }
        }

        match self.log_level {
            LogLevel::Debug => println!("DEBUG: {log_message}"),
            LogLevel::Info => println!("INFO: {log_message}"),
            LogLevel::Warn => println!("WARN: {log_message}"),
            LogLevel::Error => println!("ERROR: {log_message}"),
        }

        Ok(HookResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn should_execute(&self, _phase: HookPhase) -> bool {
        true
    }
}

/// Performance monitoring hook
#[derive(Debug, Clone)]
pub struct PerformanceHook {
    name: String,
    track_memory: bool,
    track_timing: bool,
    alert_threshold: Option<Duration>,
}

impl PerformanceHook {
    /// Create a new performance monitoring hook
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            track_memory: true,
            track_timing: true,
            alert_threshold: None,
        }
    }

    /// Set memory tracking
    #[must_use]
    pub fn track_memory(mut self, track: bool) -> Self {
        self.track_memory = track;
        self
    }

    /// Set timing tracking
    #[must_use]
    pub fn track_timing(mut self, track: bool) -> Self {
        self.track_timing = track;
        self
    }

    /// Set alert threshold for slow operations
    #[must_use]
    pub fn alert_threshold(mut self, threshold: Duration) -> Self {
        self.alert_threshold = Some(threshold);
        self
    }
}

impl ExecutionHook for PerformanceHook {
    fn execute(
        &mut self,
        context: &ExecutionContext,
        _data: Option<&HookData>,
    ) -> SklResult<HookResult> {
        if self.track_timing {
            let elapsed = context.start_time.elapsed();

            if let Some(threshold) = self.alert_threshold {
                if elapsed > threshold {
                    println!(
                        "PERFORMANCE ALERT [{}]: Slow operation detected - {:?} (threshold: {:?})",
                        self.name, elapsed, threshold
                    );
                }
            }
        }

        if self.track_memory {
            // In a real implementation, you would use a proper memory profiler
            let estimated_memory = context
                .metrics
                .data_shapes
                .iter()
                .map(|(rows, cols)| rows * cols * std::mem::size_of::<Float>())
                .sum::<usize>();

            println!(
                "MEMORY [{}]: Estimated usage: {} bytes",
                self.name, estimated_memory
            );
        }

        Ok(HookResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn should_execute(&self, phase: HookPhase) -> bool {
        matches!(
            phase,
            HookPhase::BeforeStep
                | HookPhase::AfterStep
                | HookPhase::BeforeExecution
                | HookPhase::AfterExecution
        )
    }
}

/// Data validation hook
#[derive(Debug, Clone)]
pub struct ValidationHook {
    name: String,
    check_nan: bool,
    check_inf: bool,
    check_shape: bool,
    expected_features: Option<usize>,
}

impl ValidationHook {
    /// Create a new validation hook
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            check_nan: true,
            check_inf: true,
            check_shape: true,
            expected_features: None,
        }
    }

    /// Set NaN checking
    #[must_use]
    pub fn check_nan(mut self, check: bool) -> Self {
        self.check_nan = check;
        self
    }

    /// Set infinity checking
    #[must_use]
    pub fn check_inf(mut self, check: bool) -> Self {
        self.check_inf = check;
        self
    }

    /// Set shape validation
    #[must_use]
    pub fn check_shape(mut self, check: bool) -> Self {
        self.check_shape = check;
        self
    }

    /// Set expected number of features
    #[must_use]
    pub fn expected_features(mut self, features: usize) -> Self {
        self.expected_features = Some(features);
        self
    }
}

impl ExecutionHook for ValidationHook {
    fn execute(
        &mut self,
        _context: &ExecutionContext,
        data: Option<&HookData>,
    ) -> SklResult<HookResult> {
        if let Some(data) = data {
            match data {
                HookData::Features(array) => {
                    if self.check_nan && array.iter().any(|&x| x.is_nan()) {
                        return Ok(HookResult::Abort(format!(
                            "[{}] NaN values detected in features",
                            self.name
                        )));
                    }

                    if self.check_inf && array.iter().any(|&x| x.is_infinite()) {
                        return Ok(HookResult::Abort(format!(
                            "[{}] Infinite values detected in features",
                            self.name
                        )));
                    }

                    if self.check_shape {
                        if let Some(expected) = self.expected_features {
                            if array.ncols() != expected {
                                return Ok(HookResult::Abort(format!(
                                    "[{}] Shape mismatch: expected {} features, got {}",
                                    self.name,
                                    expected,
                                    array.ncols()
                                )));
                            }
                        }
                    }
                }
                HookData::Targets(array) | HookData::Predictions(array) => {
                    if self.check_nan && array.iter().any(|&x| x.is_nan()) {
                        return Ok(HookResult::Abort(format!(
                            "[{}] NaN values detected",
                            self.name
                        )));
                    }

                    if self.check_inf && array.iter().any(|&x| x.is_infinite()) {
                        return Ok(HookResult::Abort(format!(
                            "[{}] Infinite values detected",
                            self.name
                        )));
                    }
                }
                HookData::Custom(_) => {
                    // Custom validation could be implemented here
                }
            }
        }

        Ok(HookResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn should_execute(&self, phase: HookPhase) -> bool {
        matches!(
            phase,
            HookPhase::BeforeStep | HookPhase::BeforePredict | HookPhase::BeforeTransform
        )
    }
}

/// Custom hook builder for creating application-specific hooks
pub struct CustomHookBuilder {
    name: String,
    phases: Vec<HookPhase>,
    priority: i32,
    execute_fn: Option<
        Box<dyn Fn(&ExecutionContext, Option<&HookData>) -> SklResult<HookResult> + Send + Sync>,
    >,
}

impl std::fmt::Debug for CustomHookBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomHookBuilder")
            .field("name", &self.name)
            .field("phases", &self.phases)
            .field("priority", &self.priority)
            .field("execute_fn", &"<function>")
            .finish()
    }
}

impl CustomHookBuilder {
    /// Create a new custom hook builder
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            phases: Vec::new(),
            priority: 0,
            execute_fn: None,
        }
    }

    /// Add phases where this hook should execute
    #[must_use]
    pub fn phases(mut self, phases: Vec<HookPhase>) -> Self {
        self.phases = phases;
        self
    }

    /// Set hook priority
    #[must_use]
    pub fn priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set execution function
    pub fn execute_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&ExecutionContext, Option<&HookData>) -> SklResult<HookResult>
            + Send
            + Sync
            + 'static,
    {
        self.execute_fn = Some(Box::new(f));
        self
    }

    /// Build the custom hook
    pub fn build(self) -> SklResult<CustomHook> {
        let execute_fn = self.execute_fn.ok_or_else(|| {
            SklearsError::InvalidInput("Execute function is required for custom hook".to_string())
        })?;

        Ok(CustomHook {
            name: self.name,
            phases: self.phases,
            priority: self.priority,
            execute_fn,
        })
    }
}

/// Custom hook implementation
pub struct CustomHook {
    name: String,
    phases: Vec<HookPhase>,
    priority: i32,
    execute_fn:
        Box<dyn Fn(&ExecutionContext, Option<&HookData>) -> SklResult<HookResult> + Send + Sync>,
}

impl std::fmt::Debug for CustomHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomHook")
            .field("name", &self.name)
            .field("phases", &self.phases)
            .field("priority", &self.priority)
            .field("execute_fn", &"<function>")
            .finish()
    }
}

impl ExecutionHook for CustomHook {
    fn execute(
        &mut self,
        context: &ExecutionContext,
        data: Option<&HookData>,
    ) -> SklResult<HookResult> {
        (self.execute_fn)(context, data)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    fn should_execute(&self, phase: HookPhase) -> bool {
        self.phases.contains(&phase)
    }
}

impl Clone for CustomHook {
    fn clone(&self) -> Self {
        // Note: This is a simplified clone that doesn't actually clone the function
        // In a real implementation, you might want to use Arc<> for the function
        panic!("CustomHook cannot be cloned due to function pointer")
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_hook_manager_creation() {
        let manager = HookManager::new();
        assert!(manager.hooks.is_empty());
        assert!(manager.execution_stack.is_empty());
    }

    #[test]
    fn test_logging_hook() {
        let mut hook = LoggingHook::new("test_hook".to_string(), LogLevel::Info);
        let context = ExecutionContext {
            execution_id: "test_exec".to_string(),
            step_name: Some("test_step".to_string()),
            step_index: Some(0),
            total_steps: 1,
            start_time: Instant::now(),
            phase: HookPhase::BeforeStep,
            metadata: HashMap::new(),
            metrics: PerformanceMetrics::default(),
        };

        let result = hook.execute(&context, None).unwrap();
        assert!(matches!(result, HookResult::Continue));
    }

    #[test]
    fn test_validation_hook() {
        let mut hook = ValidationHook::new("validation".to_string()).expected_features(2);

        let context = ExecutionContext {
            execution_id: "test_exec".to_string(),
            step_name: None,
            step_index: None,
            total_steps: 1,
            start_time: Instant::now(),
            phase: HookPhase::BeforeStep,
            metadata: HashMap::new(),
            metrics: PerformanceMetrics::default(),
        };

        // Test with valid data
        let valid_data = HookData::Features(array![[1.0, 2.0], [3.0, 4.0]]);
        let result = hook.execute(&context, Some(&valid_data)).unwrap();
        assert!(matches!(result, HookResult::Continue));

        // Test with invalid shape
        let invalid_data = HookData::Features(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = hook.execute(&context, Some(&invalid_data)).unwrap();
        assert!(matches!(result, HookResult::Abort(_)));
    }

    #[test]
    fn test_performance_hook() {
        let mut hook =
            PerformanceHook::new("perf".to_string()).alert_threshold(Duration::from_millis(1));

        let context = ExecutionContext {
            execution_id: "test_exec".to_string(),
            step_name: None,
            step_index: None,
            total_steps: 1,
            start_time: Instant::now() - Duration::from_millis(10),
            phase: HookPhase::AfterStep,
            metadata: HashMap::new(),
            metrics: PerformanceMetrics::default(),
        };

        let result = hook.execute(&context, None).unwrap();
        assert!(matches!(result, HookResult::Continue));
    }

    #[test]
    fn test_hook_phases() {
        let hook = LoggingHook::new("test".to_string(), LogLevel::Info);
        assert!(hook.should_execute(HookPhase::BeforeExecution));
        assert!(hook.should_execute(HookPhase::AfterStep));
    }

    #[test]
    fn test_execution_context() {
        let mut manager = HookManager::new();
        let context = manager.create_context("test_id".to_string(), 5);

        assert_eq!(context.execution_id, "test_id");
        assert_eq!(context.total_steps, 5);
        assert!(context.step_name.is_none());
    }

    #[test]
    fn test_hook_data_variants() {
        let features = HookData::Features(array![[1.0, 2.0], [3.0, 4.0]]);
        let targets = HookData::Targets(array![1.0, 2.0]);
        let predictions = HookData::Predictions(array![1.1, 2.1]);

        match features {
            HookData::Features(arr) => assert_eq!(arr.shape(), &[2, 2]),
            _ => panic!("Wrong variant"),
        }

        match targets {
            HookData::Targets(arr) => assert_eq!(arr.len(), 2),
            _ => panic!("Wrong variant"),
        }

        match predictions {
            HookData::Predictions(arr) => assert_eq!(arr.len(), 2),
            _ => panic!("Wrong variant"),
        }
    }
}

/// Advanced hook dependency management system
#[derive(Debug, Clone)]
pub struct HookDependency {
    /// Hook name that this depends on
    pub hook_name: String,
    /// Whether this is a strict dependency (execution fails if dependency fails)
    pub strict: bool,
    /// Minimum required priority of dependency
    pub min_priority: Option<i32>,
}

/// Hook with dependency management
pub trait DependentExecutionHook: ExecutionHook {
    /// Get hook dependencies
    fn dependencies(&self) -> Vec<HookDependency> {
        Vec::new()
    }

    /// Check if dependencies are satisfied
    fn dependencies_satisfied(&self, executed_hooks: &[String]) -> bool {
        self.dependencies()
            .iter()
            .all(|dep| executed_hooks.contains(&dep.hook_name))
    }
}

/// Async execution hook trait for non-blocking operations
/// Note: Would use #[`async_trait::async_trait`] in real implementation
pub trait AsyncExecutionHook: Send + Sync + Debug {
    fn execute_async(
        &mut self,
        context: &ExecutionContext,
        data: Option<&HookData>,
    ) -> SklResult<HookResult>;

    fn name(&self) -> &str;

    fn priority(&self) -> i32 {
        0
    }

    /// Check if hook should execute for given phase
    fn should_execute(&self, phase: HookPhase) -> bool;

    /// Maximum execution timeout
    fn timeout(&self) -> Option<Duration> {
        None
    }
}

/// Resource management hook for tracking and managing computational resources
#[derive(Debug, Clone)]
pub struct ResourceManagerHook {
    name: String,
    max_memory: Option<usize>,
    max_execution_time: Option<Duration>,
    cpu_limit: Option<f64>, // CPU utilization percentage
    resource_usage: Arc<Mutex<ResourceUsage>>,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    pub current_memory: usize,
    pub peak_memory: usize,
    pub cpu_usage: f64,
    pub execution_time: Duration,
    pub violations: Vec<ResourceViolation>,
}

#[derive(Debug, Clone)]
pub struct ResourceViolation {
    pub violation_type: ViolationType,
    pub timestamp: Instant,
    pub details: String,
}

#[derive(Debug, Clone)]
pub enum ViolationType {
    /// MemoryLimit
    MemoryLimit,
    /// TimeLimit
    TimeLimit,
    /// CpuLimit
    CpuLimit,
}

impl ResourceManagerHook {
    /// Create a new resource manager hook
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            max_memory: None,
            max_execution_time: None,
            cpu_limit: None,
            resource_usage: Arc::new(Mutex::new(ResourceUsage::default())),
        }
    }

    /// Set maximum memory limit in bytes
    #[must_use]
    pub fn max_memory(mut self, limit: usize) -> Self {
        self.max_memory = Some(limit);
        self
    }

    /// Set maximum execution time
    #[must_use]
    pub fn max_execution_time(mut self, limit: Duration) -> Self {
        self.max_execution_time = Some(limit);
        self
    }

    /// Set CPU usage limit (0.0 to 1.0)
    #[must_use]
    pub fn cpu_limit(mut self, limit: f64) -> Self {
        self.cpu_limit = Some(limit.min(1.0).max(0.0));
        self
    }

    /// Get current resource usage
    #[must_use]
    pub fn get_usage(&self) -> ResourceUsage {
        self.resource_usage.lock().unwrap().clone()
    }

    /// Check resource limits and record violations
    fn check_limits(&self, context: &ExecutionContext) -> SklResult<HookResult> {
        let mut usage = self.resource_usage.lock().unwrap();

        // Check execution time limit
        if let Some(time_limit) = self.max_execution_time {
            let elapsed = context.start_time.elapsed();
            usage.execution_time = elapsed;

            if elapsed > time_limit {
                let violation = ResourceViolation {
                    violation_type: ViolationType::TimeLimit,
                    timestamp: Instant::now(),
                    details: format!(
                        "Execution time {} exceeded limit {:?}",
                        elapsed.as_secs_f64(),
                        time_limit
                    ),
                };
                usage.violations.push(violation);
                return Ok(HookResult::Abort(format!(
                    "[{}] Execution time limit exceeded: {:?} > {:?}",
                    self.name, elapsed, time_limit
                )));
            }
        }

        // Check memory limit (simplified estimation)
        if let Some(memory_limit) = self.max_memory {
            let estimated_memory = context
                .metrics
                .data_shapes
                .iter()
                .map(|(rows, cols)| rows * cols * std::mem::size_of::<Float>())
                .sum::<usize>();

            usage.current_memory = estimated_memory;
            usage.peak_memory = usage.peak_memory.max(estimated_memory);

            if estimated_memory > memory_limit {
                let violation = ResourceViolation {
                    violation_type: ViolationType::MemoryLimit,
                    timestamp: Instant::now(),
                    details: format!(
                        "Memory usage {estimated_memory} exceeded limit {memory_limit}"
                    ),
                };
                usage.violations.push(violation);
                return Ok(HookResult::Abort(format!(
                    "[{}] Memory limit exceeded: {} bytes > {} bytes",
                    self.name, estimated_memory, memory_limit
                )));
            }
        }

        Ok(HookResult::Continue)
    }
}

impl ExecutionHook for ResourceManagerHook {
    fn execute(
        &mut self,
        context: &ExecutionContext,
        _data: Option<&HookData>,
    ) -> SklResult<HookResult> {
        self.check_limits(context)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> i32 {
        1000 // High priority to check limits early
    }

    fn should_execute(&self, phase: HookPhase) -> bool {
        matches!(
            phase,
            HookPhase::BeforeStep | HookPhase::AfterStep | HookPhase::BeforeExecution
        )
    }
}

/// Security and audit hook for tracking sensitive operations
#[derive(Debug, Clone)]
pub struct SecurityAuditHook {
    name: String,
    audit_log: Arc<Mutex<Vec<AuditEntry>>>,
    sensitive_operations: Vec<String>,
    require_authorization: bool,
}

#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub timestamp: Instant,
    pub execution_id: String,
    pub operation: String,
    pub user_id: Option<String>,
    pub data_summary: String,
    pub result: AuditResult,
}

#[derive(Debug, Clone)]
pub enum AuditResult {
    /// Success
    Success,
    /// Failed
    Failed(String),
    /// Unauthorized
    Unauthorized,
    /// Suspicious
    Suspicious(String),
}

impl SecurityAuditHook {
    /// Create a new security audit hook
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            audit_log: Arc::new(Mutex::new(Vec::new())),
            sensitive_operations: Vec::new(),
            require_authorization: false,
        }
    }

    /// Add sensitive operations that require auditing
    #[must_use]
    pub fn sensitive_operations(mut self, operations: Vec<String>) -> Self {
        self.sensitive_operations = operations;
        self
    }

    /// Require authorization for sensitive operations
    #[must_use]
    pub fn require_authorization(mut self, require: bool) -> Self {
        self.require_authorization = require;
        self
    }

    /// Get audit log
    #[must_use]
    pub fn get_audit_log(&self) -> Vec<AuditEntry> {
        self.audit_log.lock().unwrap().clone()
    }

    /// Check if operation is sensitive
    fn is_sensitive_operation(&self, context: &ExecutionContext) -> bool {
        if let Some(step_name) = &context.step_name {
            self.sensitive_operations
                .iter()
                .any(|op| step_name.contains(op))
        } else {
            false
        }
    }

    /// Create audit entry
    fn create_audit_entry(
        &self,
        context: &ExecutionContext,
        result: AuditResult,
        data_summary: String,
    ) -> AuditEntry {
        /// AuditEntry
        AuditEntry {
            timestamp: Instant::now(),
            execution_id: context.execution_id.clone(),
            operation: context
                .step_name
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            user_id: context.metadata.get("user_id").cloned(),
            data_summary,
            result,
        }
    }
}

impl ExecutionHook for SecurityAuditHook {
    fn execute(
        &mut self,
        context: &ExecutionContext,
        data: Option<&HookData>,
    ) -> SklResult<HookResult> {
        let is_sensitive = self.is_sensitive_operation(context);

        // Create data summary for audit log
        let data_summary = match data {
            Some(HookData::Features(arr)) => format!("Features: {}x{}", arr.nrows(), arr.ncols()),
            Some(HookData::Targets(arr)) => format!("Targets: {}", arr.len()),
            Some(HookData::Predictions(arr)) => format!("Predictions: {}", arr.len()),
            Some(HookData::Custom(_)) => "Custom data".to_string(),
            None => "No data".to_string(),
        };

        // Check authorization for sensitive operations
        if is_sensitive && self.require_authorization {
            let has_auth = context
                .metadata
                .get("authorized")
                .is_some_and(|v| v == "true");

            if !has_auth {
                let audit_entry =
                    self.create_audit_entry(context, AuditResult::Unauthorized, data_summary);
                self.audit_log.lock().unwrap().push(audit_entry);

                return Ok(HookResult::Abort(format!(
                    "[{}] Unauthorized access to sensitive operation: {}",
                    self.name,
                    context.step_name.as_deref().unwrap_or("unknown")
                )));
            }
        }

        // Log all operations (or just sensitive ones)
        if is_sensitive || !self.sensitive_operations.is_empty() {
            let result = if is_sensitive {
                // Additional checks for sensitive operations
                if data_summary.contains("empty") {
                    AuditResult::Suspicious("Empty data in sensitive operation".to_string())
                } else {
                    AuditResult::Success
                }
            } else {
                AuditResult::Success
            };

            let audit_entry = self.create_audit_entry(context, result, data_summary);
            self.audit_log.lock().unwrap().push(audit_entry);
        }

        Ok(HookResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> i32 {
        900 // High priority for security checks
    }

    fn should_execute(&self, phase: HookPhase) -> bool {
        matches!(
            phase,
            HookPhase::BeforeStep | HookPhase::BeforePredict | HookPhase::BeforeTransform
        )
    }
}

/// Error recovery hook for handling and recovering from execution errors
#[derive(Debug, Clone)]
pub struct ErrorRecoveryHook {
    name: String,
    retry_count: usize,
    retry_delay: Duration,
    fallback_strategies: Vec<FallbackStrategy>,
    error_history: Arc<Mutex<Vec<ErrorRecord>>>,
}

#[derive(Debug, Clone)]
pub struct ErrorRecord {
    pub timestamp: Instant,
    pub execution_id: String,
    pub error_type: String,
    pub error_message: String,
    pub recovery_attempted: bool,
    pub recovery_successful: bool,
}

#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    /// RetryWithDelay
    RetryWithDelay(Duration),
    /// UseDefaultValues
    UseDefaultValues,
    /// SkipStep
    SkipStep,
    /// AbortExecution
    AbortExecution,
    /// CustomRecovery
    CustomRecovery(String), // Custom recovery logic identifier
}

impl ErrorRecoveryHook {
    /// Create a new error recovery hook
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            retry_count: 3,
            retry_delay: Duration::from_millis(100),
            fallback_strategies: vec![
                FallbackStrategy::RetryWithDelay(Duration::from_millis(100)),
                FallbackStrategy::UseDefaultValues,
                FallbackStrategy::SkipStep,
            ],
            error_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Set retry configuration
    #[must_use]
    pub fn retry_config(mut self, count: usize, delay: Duration) -> Self {
        self.retry_count = count;
        self.retry_delay = delay;
        self
    }

    /// Set fallback strategies
    #[must_use]
    pub fn fallback_strategies(mut self, strategies: Vec<FallbackStrategy>) -> Self {
        self.fallback_strategies = strategies;
        self
    }

    /// Get error history
    #[must_use]
    pub fn get_error_history(&self) -> Vec<ErrorRecord> {
        self.error_history.lock().unwrap().clone()
    }

    /// Record error for analysis
    fn record_error(
        &self,
        context: &ExecutionContext,
        error: &str,
        recovery_attempted: bool,
        recovery_successful: bool,
    ) {
        let record = ErrorRecord {
            timestamp: Instant::now(),
            execution_id: context.execution_id.clone(),
            error_type: "execution_error".to_string(),
            error_message: error.to_string(),
            recovery_attempted,
            recovery_successful,
        };

        self.error_history.lock().unwrap().push(record);
    }
}

impl ExecutionHook for ErrorRecoveryHook {
    fn execute(
        &mut self,
        context: &ExecutionContext,
        _data: Option<&HookData>,
    ) -> SklResult<HookResult> {
        // This hook primarily responds to error phases
        if matches!(context.phase, HookPhase::OnError) {
            // Analyze error and attempt recovery
            let error_msg = context
                .metadata
                .get("error")
                .unwrap_or(&"Unknown error".to_string())
                .clone();

            // Try fallback strategies
            for strategy in &self.fallback_strategies {
                match strategy {
                    FallbackStrategy::RetryWithDelay(delay) => {
                        self.record_error(context, &error_msg, true, false);
                        std::thread::sleep(*delay);
                        // In real implementation, would trigger retry
                        println!("[{}] Retrying after delay: {:?}", self.name, delay);
                        return Ok(HookResult::Continue);
                    }
                    FallbackStrategy::UseDefaultValues => {
                        self.record_error(context, &error_msg, true, true);
                        println!("[{}] Using default values for recovery", self.name);
                        // Return default data
                        return Ok(HookResult::ContinueWithData(HookData::Features(
                            Array2::zeros((1, 1)),
                        )));
                    }
                    FallbackStrategy::SkipStep => {
                        self.record_error(context, &error_msg, true, true);
                        println!("[{}] Skipping step for recovery", self.name);
                        return Ok(HookResult::Skip);
                    }
                    FallbackStrategy::AbortExecution => {
                        self.record_error(context, &error_msg, false, false);
                        return Ok(HookResult::Abort(format!(
                            "[{}] Unrecoverable error: {}",
                            self.name, error_msg
                        )));
                    }
                    FallbackStrategy::CustomRecovery(name) => {
                        println!("[{}] Attempting custom recovery: {}", self.name, name);
                        // Custom recovery logic would be implemented here
                        self.record_error(context, &error_msg, true, false);
                    }
                }
            }
        }

        Ok(HookResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> i32 {
        500 // Medium priority for error handling
    }

    fn should_execute(&self, phase: HookPhase) -> bool {
        matches!(phase, HookPhase::OnError)
    }
}

/// Hook composition system for chaining multiple hooks
#[derive(Debug)]
pub struct HookComposition {
    name: String,
    hooks: Vec<Box<dyn ExecutionHook>>,
    execution_strategy: CompositionStrategy,
}

#[derive(Debug, Clone)]
pub enum CompositionStrategy {
    /// Execute all hooks in sequence
    Sequential,
    /// Execute hooks in parallel (conceptually - actual implementation would need async)
    Parallel,
    /// Execute until first hook returns non-Continue
    FirstMatch,
    /// Execute all hooks and combine results
    Aggregate,
}

impl HookComposition {
    /// Create a new hook composition
    #[must_use]
    pub fn new(name: String, strategy: CompositionStrategy) -> Self {
        Self {
            name,
            hooks: Vec::new(),
            execution_strategy: strategy,
        }
    }

    /// Add a hook to the composition
    pub fn add_hook(&mut self, hook: Box<dyn ExecutionHook>) {
        self.hooks.push(hook);
        // Sort by priority
        self.hooks.sort_by(|a, b| b.priority().cmp(&a.priority()));
    }
}

impl ExecutionHook for HookComposition {
    fn execute(
        &mut self,
        context: &ExecutionContext,
        data: Option<&HookData>,
    ) -> SklResult<HookResult> {
        match self.execution_strategy {
            CompositionStrategy::Sequential => {
                for hook in &mut self.hooks {
                    if hook.should_execute(context.phase) {
                        let result = hook.execute(context, data)?;
                        if !matches!(result, HookResult::Continue) {
                            return Ok(result);
                        }
                    }
                }
                Ok(HookResult::Continue)
            }
            CompositionStrategy::FirstMatch => {
                for hook in &mut self.hooks {
                    if hook.should_execute(context.phase) {
                        let result = hook.execute(context, data)?;
                        if !matches!(result, HookResult::Continue) {
                            return Ok(result);
                        }
                    }
                }
                Ok(HookResult::Continue)
            }
            CompositionStrategy::Parallel => {
                // Simplified parallel execution (real implementation would use async)
                let mut results = Vec::new();
                for hook in &mut self.hooks {
                    if hook.should_execute(context.phase) {
                        results.push(hook.execute(context, data)?);
                    }
                }

                // Return first non-Continue result, or Continue if all continue
                for result in results {
                    if !matches!(result, HookResult::Continue) {
                        return Ok(result);
                    }
                }
                Ok(HookResult::Continue)
            }
            CompositionStrategy::Aggregate => {
                // Execute all and combine results (simplified)
                for hook in &mut self.hooks {
                    if hook.should_execute(context.phase) {
                        let _result = hook.execute(context, data)?;
                        // In real implementation, would aggregate results
                    }
                }
                Ok(HookResult::Continue)
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> i32 {
        // Return highest priority among constituent hooks
        self.hooks.iter().map(|h| h.priority()).max().unwrap_or(0)
    }

    fn should_execute(&self, phase: HookPhase) -> bool {
        self.hooks.iter().any(|h| h.should_execute(phase))
    }
}

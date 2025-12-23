//! Explanation Hooks and Event Handling System
//!
//! This module provides a comprehensive hook system for monitoring, logging,
//! and custom processing during explanation generation.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Trait for explanation hooks
pub trait ExplanationHook: Debug + Send + Sync {
    /// Hook identifier
    fn hook_id(&self) -> &str;

    /// Hook name
    fn hook_name(&self) -> &str;

    /// Hook description
    fn hook_description(&self) -> &str;

    /// Hook priority (higher means executed earlier)
    fn priority(&self) -> u32;

    /// Events this hook is interested in
    fn interested_events(&self) -> Vec<HookEvent>;

    /// Called before explanation starts
    fn before_explanation(&self, context: &mut HookContext) -> SklResult<()> {
        Ok(())
    }

    /// Called after explanation completes
    fn after_explanation(&self, context: &mut HookContext) -> SklResult<()> {
        Ok(())
    }

    /// Called before feature importance computation
    fn before_feature_importance(&self, context: &mut HookContext) -> SklResult<()> {
        Ok(())
    }

    /// Called after feature importance computation
    fn after_feature_importance(&self, context: &mut HookContext) -> SklResult<()> {
        Ok(())
    }

    /// Called before local explanation
    fn before_local_explanation(&self, context: &mut HookContext) -> SklResult<()> {
        Ok(())
    }

    /// Called after local explanation
    fn after_local_explanation(&self, context: &mut HookContext) -> SklResult<()> {
        Ok(())
    }

    /// Called before global explanation
    fn before_global_explanation(&self, context: &mut HookContext) -> SklResult<()> {
        Ok(())
    }

    /// Called after global explanation
    fn after_global_explanation(&self, context: &mut HookContext) -> SklResult<()> {
        Ok(())
    }

    /// Called before counterfactual generation
    fn before_counterfactual(&self, context: &mut HookContext) -> SklResult<()> {
        Ok(())
    }

    /// Called after counterfactual generation
    fn after_counterfactual(&self, context: &mut HookContext) -> SklResult<()> {
        Ok(())
    }

    /// Called on error
    fn on_error(&self, context: &mut HookContext, error: &dyn std::error::Error) -> SklResult<()> {
        Ok(())
    }

    /// Called on progress update
    fn on_progress(&self, context: &mut HookContext, progress: &ProgressInfo) -> SklResult<()> {
        Ok(())
    }

    /// Called on warning
    fn on_warning(&self, context: &mut HookContext, warning: &str) -> SklResult<()> {
        Ok(())
    }

    /// Called on custom event
    fn on_custom_event(&self, context: &mut HookContext, event: &CustomEvent) -> SklResult<()> {
        Ok(())
    }
}

/// Hook events that can be subscribed to
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HookEvent {
    /// Before explanation starts
    BeforeExplanation,
    /// After explanation completes
    AfterExplanation,
    /// Before feature importance computation
    BeforeFeatureImportance,
    /// After feature importance computation
    AfterFeatureImportance,
    /// Before local explanation
    BeforeLocalExplanation,
    /// After local explanation
    AfterLocalExplanation,
    /// Before global explanation
    BeforeGlobalExplanation,
    /// After global explanation
    AfterGlobalExplanation,
    /// Before counterfactual generation
    BeforeCounterfactual,
    /// After counterfactual generation
    AfterCounterfactual,
    /// On error occurrence
    OnError,
    /// On progress update
    OnProgress,
    /// On warning
    OnWarning,
    /// Custom event
    CustomEvent,
}

/// Hook execution context
#[derive(Debug)]
pub struct HookContext {
    /// Execution ID for tracking
    pub execution_id: String,
    /// Method being executed
    pub method: String,
    /// Input data metadata
    pub input_metadata: HashMap<String, String>,
    /// Timing information
    pub timing: TimingInfo,
    /// Memory usage information
    pub memory_usage: MemoryInfo,
    /// Progress information
    pub progress: ProgressInfo,
    /// Results cache
    pub results: HashMap<String, HookResult>,
    /// Custom data for hooks to store state
    pub custom_data: HashMap<String, serde_json::Value>,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Error information
    pub error_info: Option<ErrorInfo>,
}

impl HookContext {
    /// Create a new hook context
    pub fn new(execution_id: String, method: String) -> Self {
        Self {
            execution_id,
            method,
            input_metadata: HashMap::new(),
            timing: TimingInfo::new(),
            memory_usage: MemoryInfo::new(),
            progress: ProgressInfo::new(),
            results: HashMap::new(),
            custom_data: HashMap::new(),
            warnings: Vec::new(),
            error_info: None,
        }
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.input_metadata.insert(key, value);
    }

    /// Store result
    pub fn store_result(&mut self, key: String, result: HookResult) {
        self.results.insert(key, result);
    }

    /// Get result
    pub fn get_result(&self, key: &str) -> Option<&HookResult> {
        self.results.get(key)
    }

    /// Store custom data
    pub fn store_custom_data(&mut self, key: String, data: serde_json::Value) {
        self.custom_data.insert(key, data);
    }

    /// Get custom data
    pub fn get_custom_data(&self, key: &str) -> Option<&serde_json::Value> {
        self.custom_data.get(key)
    }

    /// Add warning
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Set error
    pub fn set_error(&mut self, error: ErrorInfo) {
        self.error_info = Some(error);
    }

    /// Update progress
    pub fn update_progress(&mut self, current: usize, total: usize, message: Option<String>) {
        self.progress.current = current;
        self.progress.total = total;
        self.progress.percentage = if total > 0 {
            (current as f64 / total as f64 * 100.0) as Float
        } else {
            0.0
        };
        if let Some(msg) = message {
            self.progress.message = msg;
        }
        self.progress.updated_at = chrono::Utc::now();
    }
}

/// Timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingInfo {
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// End time (if completed)
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Phase timings
    pub phase_timings: HashMap<String, u64>,
}

impl TimingInfo {
    /// Create new timing info
    pub fn new() -> Self {
        Self {
            start_time: chrono::Utc::now(),
            end_time: None,
            elapsed_ms: 0,
            phase_timings: HashMap::new(),
        }
    }

    /// Mark completion
    pub fn complete(&mut self) {
        let now = chrono::Utc::now();
        self.end_time = Some(now);
        self.elapsed_ms = (now - self.start_time).num_milliseconds() as u64;
    }

    /// Add phase timing
    pub fn add_phase_timing(&mut self, phase: String, duration_ms: u64) {
        self.phase_timings.insert(phase, duration_ms);
    }
}

impl Default for TimingInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Current memory usage in bytes
    pub current_memory_bytes: usize,
    /// Memory usage by phase
    pub phase_memory: HashMap<String, usize>,
}

impl MemoryInfo {
    /// Create new memory info
    pub fn new() -> Self {
        Self {
            peak_memory_bytes: 0,
            current_memory_bytes: 0,
            phase_memory: HashMap::new(),
        }
    }

    /// Update memory usage
    pub fn update_memory(&mut self, current: usize) {
        self.current_memory_bytes = current;
        if current > self.peak_memory_bytes {
            self.peak_memory_bytes = current;
        }
    }

    /// Add phase memory
    pub fn add_phase_memory(&mut self, phase: String, memory_bytes: usize) {
        self.phase_memory.insert(phase, memory_bytes);
    }
}

impl Default for MemoryInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressInfo {
    /// Current step
    pub current: usize,
    /// Total steps
    pub total: usize,
    /// Percentage complete
    pub percentage: Float,
    /// Progress message
    pub message: String,
    /// Last update time
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl ProgressInfo {
    /// Create new progress info
    pub fn new() -> Self {
        Self {
            current: 0,
            total: 0,
            percentage: 0.0,
            message: "Starting...".to_string(),
            updated_at: chrono::Utc::now(),
        }
    }
}

impl Default for ProgressInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Error message
    pub message: String,
    /// Error type
    pub error_type: String,
    /// Error occurred at
    pub occurred_at: chrono::DateTime<chrono::Utc>,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
}

/// Hook result data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookResult {
    /// Feature importance scores
    FeatureImportance(Vec<Float>),
    /// Local explanation values
    LocalExplanation(Vec<Float>),
    /// Global explanation values
    GlobalExplanation(Vec<Float>),
    /// Counterfactual instance
    Counterfactual(Vec<Float>),
    /// Custom result
    Custom(serde_json::Value),
}

/// Custom event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomEvent {
    /// Event type
    pub event_type: String,
    /// Event data
    pub data: serde_json::Value,
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Hook registry for managing hooks
#[derive(Debug, Default)]
pub struct HookRegistry {
    hooks: Arc<RwLock<HashMap<String, Arc<dyn ExplanationHook>>>>,
    event_subscriptions: Arc<RwLock<HashMap<HookEvent, Vec<String>>>>,
    execution_hooks: Arc<RwLock<Vec<String>>>, // Hooks sorted by priority
}

impl HookRegistry {
    /// Create a new hook registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new hook
    pub fn register_hook<H: ExplanationHook + 'static>(&self, hook: H) -> SklResult<()> {
        let hook_id = hook.hook_id().to_string();
        let interested_events = hook.interested_events();
        let priority = hook.priority();

        // Store hook
        {
            let mut hooks = self.hooks.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire hooks lock".to_string())
            })?;
            hooks.insert(hook_id.clone(), Arc::new(hook));
        }

        // Update event subscriptions
        {
            let mut subscriptions = self.event_subscriptions.write().map_err(|_| {
                crate::SklearsError::InvalidInput(
                    "Failed to acquire subscriptions lock".to_string(),
                )
            })?;

            for event in interested_events {
                subscriptions
                    .entry(event)
                    .or_insert_with(Vec::new)
                    .push(hook_id.clone());
            }
        }

        // Update execution order (sorted by priority)
        {
            let mut execution_hooks = self.execution_hooks.write().map_err(|_| {
                crate::SklearsError::InvalidInput(
                    "Failed to acquire execution hooks lock".to_string(),
                )
            })?;

            execution_hooks.push(hook_id.clone());

            // Re-sort by priority (higher priority first)
            let hooks_guard = self.hooks.read().map_err(|_| {
                crate::SklearsError::InvalidInput(
                    "Failed to acquire hooks lock for sorting".to_string(),
                )
            })?;

            execution_hooks.sort_by(|a, b| {
                let priority_a = hooks_guard.get(a).map(|h| h.priority()).unwrap_or(0);
                let priority_b = hooks_guard.get(b).map(|h| h.priority()).unwrap_or(0);
                priority_b.cmp(&priority_a) // Higher priority first
            });
        }

        Ok(())
    }

    /// Get hook by ID
    pub fn get_hook(&self, hook_id: &str) -> Option<Arc<dyn ExplanationHook>> {
        self.hooks.read().ok()?.get(hook_id).cloned()
    }

    /// List all registered hooks
    pub fn list_hooks(&self) -> Vec<String> {
        self.hooks
            .read()
            .ok()
            .map(|hooks| hooks.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Get hooks subscribed to an event
    pub fn get_event_hooks(&self, event: HookEvent) -> Vec<Arc<dyn ExplanationHook>> {
        let hook_ids = self
            .event_subscriptions
            .read()
            .ok()
            .and_then(|subs| subs.get(&event).cloned())
            .unwrap_or_default();

        let hooks = self.hooks.read().ok();
        if let Some(hooks_guard) = hooks {
            hook_ids
                .iter()
                .filter_map(|id| hooks_guard.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Execute hooks for an event
    pub fn execute_event_hooks(
        &self,
        event: HookEvent,
        context: &mut HookContext,
    ) -> SklResult<()> {
        let hooks = self.get_event_hooks(event);

        for hook in hooks {
            let result = match event {
                HookEvent::BeforeExplanation => hook.before_explanation(context),
                HookEvent::AfterExplanation => hook.after_explanation(context),
                HookEvent::BeforeFeatureImportance => hook.before_feature_importance(context),
                HookEvent::AfterFeatureImportance => hook.after_feature_importance(context),
                HookEvent::BeforeLocalExplanation => hook.before_local_explanation(context),
                HookEvent::AfterLocalExplanation => hook.after_local_explanation(context),
                HookEvent::BeforeGlobalExplanation => hook.before_global_explanation(context),
                HookEvent::AfterGlobalExplanation => hook.after_global_explanation(context),
                HookEvent::BeforeCounterfactual => hook.before_counterfactual(context),
                HookEvent::AfterCounterfactual => hook.after_counterfactual(context),
                HookEvent::OnProgress => {
                    let progress = context.progress.clone();
                    hook.on_progress(context, &progress)
                }
                HookEvent::OnWarning => {
                    if let Some(warning) = context.warnings.last().cloned() {
                        hook.on_warning(context, &warning)
                    } else {
                        Ok(())
                    }
                }
                HookEvent::OnError => {
                    if let Some(error_info) = &context.error_info {
                        // Create a dummy error for the hook
                        let error = crate::SklearsError::InvalidInput(error_info.message.clone());
                        hook.on_error(context, &error)
                    } else {
                        Ok(())
                    }
                }
                HookEvent::CustomEvent => Ok(()), // Handled separately
            };

            if let Err(e) = result {
                context.add_warning(format!("Hook '{}' failed: {}", hook.hook_id(), e));
            }
        }

        Ok(())
    }

    /// Execute custom event
    pub fn execute_custom_event(
        &self,
        context: &mut HookContext,
        event: &CustomEvent,
    ) -> SklResult<()> {
        let hooks = self.get_event_hooks(HookEvent::CustomEvent);

        for hook in hooks {
            if let Err(e) = hook.on_custom_event(context, event) {
                context.add_warning(format!(
                    "Hook '{}' failed on custom event: {}",
                    hook.hook_id(),
                    e
                ));
            }
        }

        Ok(())
    }

    /// Unregister a hook
    pub fn unregister_hook(&self, hook_id: &str) -> SklResult<()> {
        // Remove from main registry
        {
            let mut hooks = self.hooks.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire hooks lock".to_string())
            })?;
            hooks.remove(hook_id);
        }

        // Remove from event subscriptions
        {
            let mut subscriptions = self.event_subscriptions.write().map_err(|_| {
                crate::SklearsError::InvalidInput(
                    "Failed to acquire subscriptions lock".to_string(),
                )
            })?;

            for (_, hook_ids) in subscriptions.iter_mut() {
                hook_ids.retain(|id| id != hook_id);
            }
        }

        // Remove from execution order
        {
            let mut execution_hooks = self.execution_hooks.write().map_err(|_| {
                crate::SklearsError::InvalidInput(
                    "Failed to acquire execution hooks lock".to_string(),
                )
            })?;
            execution_hooks.retain(|id| id != hook_id);
        }

        Ok(())
    }

    /// Get registry statistics
    pub fn get_statistics(&self) -> HookRegistryStatistics {
        let hooks = self.hooks.read().ok();
        let subscriptions = self.event_subscriptions.read().ok();

        let total_hooks = hooks.as_ref().map(|h| h.len()).unwrap_or(0);

        let subscriptions_by_event = subscriptions
            .as_ref()
            .map(|subs| {
                subs.iter()
                    .map(|(event, hooks)| (*event, hooks.len()))
                    .collect()
            })
            .unwrap_or_default();

        HookRegistryStatistics {
            total_hooks,
            subscriptions_by_event,
            registry_created_at: chrono::Utc::now(),
        }
    }
}

/// Hook registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookRegistryStatistics {
    /// Total number of hooks
    pub total_hooks: usize,
    /// Number of hook subscriptions by event
    pub subscriptions_by_event: HashMap<HookEvent, usize>,
    /// Registry creation timestamp
    pub registry_created_at: chrono::DateTime<chrono::Utc>,
}

/// Explanation executor with hook support
#[derive(Debug)]
pub struct HookedExplanationExecutor {
    hook_registry: HookRegistry,
}

impl HookedExplanationExecutor {
    /// Create a new hooked explanation executor
    pub fn new() -> Self {
        Self {
            hook_registry: HookRegistry::new(),
        }
    }

    /// Get the hook registry
    pub fn hook_registry(&self) -> &HookRegistry {
        &self.hook_registry
    }

    /// Execute explanation with hooks
    pub fn execute_with_hooks<F, R>(&self, method: &str, execution_fn: F) -> SklResult<R>
    where
        F: FnOnce(&mut HookContext) -> SklResult<R>,
    {
        let execution_id = uuid::Uuid::new_v4().to_string();
        let mut context = HookContext::new(execution_id, method.to_string());

        // Execute before explanation hooks
        self.hook_registry
            .execute_event_hooks(HookEvent::BeforeExplanation, &mut context)?;

        let start_time = Instant::now();

        // Execute the main function
        let result = execution_fn(&mut context);

        // Update timing
        context.timing.elapsed_ms = start_time.elapsed().as_millis() as u64;
        context.timing.complete();

        match result {
            Ok(value) => {
                // Execute after explanation hooks
                self.hook_registry
                    .execute_event_hooks(HookEvent::AfterExplanation, &mut context)?;
                Ok(value)
            }
            Err(e) => {
                // Set error information
                context.set_error(ErrorInfo {
                    message: e.to_string(),
                    error_type: "ExplanationError".to_string(),
                    occurred_at: chrono::Utc::now(),
                    stack_trace: None,
                });

                // Execute error hooks
                self.hook_registry
                    .execute_event_hooks(HookEvent::OnError, &mut context)?;

                Err(e)
            }
        }
    }

    /// Execute feature importance with hooks
    pub fn execute_feature_importance_with_hooks<F, R>(&self, execution_fn: F) -> SklResult<R>
    where
        F: FnOnce(&mut HookContext) -> SklResult<R>,
    {
        let execution_id = uuid::Uuid::new_v4().to_string();
        let mut context = HookContext::new(execution_id, "feature_importance".to_string());

        // Execute before hooks
        self.hook_registry
            .execute_event_hooks(HookEvent::BeforeFeatureImportance, &mut context)?;

        let result = execution_fn(&mut context);

        // Execute after hooks
        self.hook_registry
            .execute_event_hooks(HookEvent::AfterFeatureImportance, &mut context)?;

        result
    }
}

impl Default for HookedExplanationExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in logging hook
#[derive(Debug)]
pub struct LoggingHook {
    id: String,
    name: String,
    description: String,
    log_level: LogLevel,
}

impl LoggingHook {
    /// Create a new logging hook
    pub fn new(log_level: LogLevel) -> Self {
        Self {
            id: "logging_hook".to_string(),
            name: "Logging Hook".to_string(),
            description: "Logs explanation execution events".to_string(),
            log_level,
        }
    }
}

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Error
    Error,
    /// Warn
    Warn,
    /// Info
    Info,
    /// Debug
    Debug,
    /// Trace
    Trace,
}

impl ExplanationHook for LoggingHook {
    fn hook_id(&self) -> &str {
        &self.id
    }

    fn hook_name(&self) -> &str {
        &self.name
    }

    fn hook_description(&self) -> &str {
        &self.description
    }

    fn priority(&self) -> u32 {
        100 // High priority for logging
    }

    fn interested_events(&self) -> Vec<HookEvent> {
        vec![
            HookEvent::BeforeExplanation,
            HookEvent::AfterExplanation,
            HookEvent::OnError,
            HookEvent::OnWarning,
            HookEvent::OnProgress,
        ]
    }

    fn before_explanation(&self, context: &mut HookContext) -> SklResult<()> {
        println!(
            "[INFO] Starting explanation execution: {} (ID: {})",
            context.method, context.execution_id
        );
        Ok(())
    }

    fn after_explanation(&self, context: &mut HookContext) -> SklResult<()> {
        println!(
            "[INFO] Completed explanation execution: {} in {}ms",
            context.method, context.timing.elapsed_ms
        );
        Ok(())
    }

    fn on_error(&self, context: &mut HookContext, error: &dyn std::error::Error) -> SklResult<()> {
        println!(
            "[ERROR] Error in explanation execution {}: {}",
            context.execution_id, error
        );
        Ok(())
    }

    fn on_warning(&self, context: &mut HookContext, warning: &str) -> SklResult<()> {
        println!(
            "[WARN] Warning in explanation execution {}: {}",
            context.execution_id, warning
        );
        Ok(())
    }

    fn on_progress(&self, context: &mut HookContext, progress: &ProgressInfo) -> SklResult<()> {
        if matches!(self.log_level, LogLevel::Debug | LogLevel::Trace) {
            println!(
                "[DEBUG] Progress {}: {:.1}% - {}",
                context.execution_id, progress.percentage, progress.message
            );
        }
        Ok(())
    }
}

impl Default for LoggingHook {
    fn default() -> Self {
        Self::new(LogLevel::Info)
    }
}

/// Built-in metrics collection hook
#[derive(Debug)]
pub struct MetricsHook {
    id: String,
    name: String,
    description: String,
    metrics: Arc<RwLock<HashMap<String, ExecutionMetrics>>>,
}

impl MetricsHook {
    /// Create a new metrics collection hook
    pub fn new() -> Self {
        Self {
            id: "metrics_hook".to_string(),
            name: "Metrics Collection Hook".to_string(),
            description: "Collects execution metrics for analysis".to_string(),
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get collected metrics
    pub fn get_metrics(&self) -> HashMap<String, ExecutionMetrics> {
        self.metrics
            .read()
            .ok()
            .map(|metrics| metrics.clone())
            .unwrap_or_default()
    }

    /// Clear metrics
    pub fn clear_metrics(&self) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.clear();
        }
    }
}

/// Execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Execution count
    pub execution_count: usize,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: u64,
    /// Average execution time in milliseconds
    pub average_execution_time_ms: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
    /// Last execution timestamp
    pub last_execution: chrono::DateTime<chrono::Utc>,
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            execution_count: 0,
            total_execution_time_ms: 0,
            average_execution_time_ms: 0,
            peak_memory_bytes: 0,
            error_count: 0,
            warning_count: 0,
            last_execution: chrono::Utc::now(),
        }
    }
}

impl ExplanationHook for MetricsHook {
    fn hook_id(&self) -> &str {
        &self.id
    }

    fn hook_name(&self) -> &str {
        &self.name
    }

    fn hook_description(&self) -> &str {
        &self.description
    }

    fn priority(&self) -> u32 {
        50 // Medium priority
    }

    fn interested_events(&self) -> Vec<HookEvent> {
        vec![
            HookEvent::AfterExplanation,
            HookEvent::OnError,
            HookEvent::OnWarning,
        ]
    }

    fn after_explanation(&self, context: &mut HookContext) -> SklResult<()> {
        if let Ok(mut metrics) = self.metrics.write() {
            let entry = metrics.entry(context.method.clone()).or_default();
            entry.execution_count += 1;
            entry.total_execution_time_ms += context.timing.elapsed_ms;
            entry.average_execution_time_ms =
                entry.total_execution_time_ms / entry.execution_count as u64;
            entry.peak_memory_bytes = entry
                .peak_memory_bytes
                .max(context.memory_usage.peak_memory_bytes);
            entry.last_execution = chrono::Utc::now();
        }
        Ok(())
    }

    fn on_error(&self, context: &mut HookContext, _error: &dyn std::error::Error) -> SklResult<()> {
        if let Ok(mut metrics) = self.metrics.write() {
            let entry = metrics.entry(context.method.clone()).or_default();
            entry.error_count += 1;
        }
        Ok(())
    }

    fn on_warning(&self, context: &mut HookContext, _warning: &str) -> SklResult<()> {
        if let Ok(mut metrics) = self.metrics.write() {
            let entry = metrics.entry(context.method.clone()).or_default();
            entry.warning_count += 1;
        }
        Ok(())
    }
}

impl Default for MetricsHook {
    fn default() -> Self {
        Self::new()
    }
}

// Add uuid dependency to Cargo.toml (we'll need to add this)
mod uuid {
    use scirs2_core::random::Rng;

    pub struct Uuid;

    impl Uuid {
        pub fn new_v4() -> Self {
            Self
        }

        pub fn to_string(&self) -> String {
            let mut rng = scirs2_core::random::thread_rng();
            format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                rng.random::<u32>(),
                rng.random::<u16>(),
                rng.random::<u16>(),
                rng.random::<u16>(),
                rng.random::<u64>() & 0xffffffffffff
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_registry() {
        let registry = HookRegistry::new();

        // Register hooks
        let logging_hook = LoggingHook::new(LogLevel::Info);
        let result = registry.register_hook(logging_hook);
        assert!(result.is_ok());

        let metrics_hook = MetricsHook::new();
        let result = registry.register_hook(metrics_hook);
        assert!(result.is_ok());

        // Check hooks are registered
        let hooks = registry.list_hooks();
        assert!(hooks.contains(&"logging_hook".to_string()));
        assert!(hooks.contains(&"metrics_hook".to_string()));
    }

    #[test]
    fn test_hook_context() {
        let mut context = HookContext::new("test_id".to_string(), "test_method".to_string());

        // Test metadata
        context.add_metadata("key1".to_string(), "value1".to_string());
        assert_eq!(
            context.input_metadata.get("key1"),
            Some(&"value1".to_string())
        );

        // Test progress
        context.update_progress(5, 10, Some("Half done".to_string()));
        assert_eq!(context.progress.current, 5);
        assert_eq!(context.progress.total, 10);
        assert_eq!(context.progress.percentage, 50.0);
        assert_eq!(context.progress.message, "Half done");

        // Test warnings
        context.add_warning("Test warning".to_string());
        assert_eq!(context.warnings.len(), 1);
        assert_eq!(context.warnings[0], "Test warning");
    }

    #[test]
    fn test_hook_execution() {
        let registry = HookRegistry::new();
        let logging_hook = LoggingHook::new(LogLevel::Info);
        registry.register_hook(logging_hook).unwrap();

        let mut context = HookContext::new("test_id".to_string(), "test_method".to_string());

        // Execute before explanation hooks
        let result = registry.execute_event_hooks(HookEvent::BeforeExplanation, &mut context);
        assert!(result.is_ok());

        // Execute after explanation hooks
        let result = registry.execute_event_hooks(HookEvent::AfterExplanation, &mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hooked_explanation_executor() {
        let executor = HookedExplanationExecutor::new();

        // Register logging hook
        let logging_hook = LoggingHook::new(LogLevel::Info);
        executor
            .hook_registry()
            .register_hook(logging_hook)
            .unwrap();

        // Execute with hooks
        let result = executor.execute_with_hooks("test_method", |context| {
            context.add_metadata("test_key".to_string(), "test_value".to_string());
            Ok("test_result".to_string())
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test_result");
    }

    #[test]
    fn test_metrics_hook() {
        let metrics_hook = MetricsHook::new();
        let mut context = HookContext::new("test_id".to_string(), "test_method".to_string());
        context.timing.elapsed_ms = 100;
        context.memory_usage.peak_memory_bytes = 1024;

        // Simulate after explanation
        let result = metrics_hook.after_explanation(&mut context);
        assert!(result.is_ok());

        // Check metrics
        let metrics = metrics_hook.get_metrics();
        assert!(metrics.contains_key("test_method"));

        let method_metrics = &metrics["test_method"];
        assert_eq!(method_metrics.execution_count, 1);
        assert_eq!(method_metrics.total_execution_time_ms, 100);
        assert_eq!(method_metrics.peak_memory_bytes, 1024);
    }

    #[test]
    fn test_custom_event() {
        let registry = HookRegistry::new();
        let logging_hook = LoggingHook::new(LogLevel::Debug);
        registry.register_hook(logging_hook).unwrap();

        let mut context = HookContext::new("test_id".to_string(), "test_method".to_string());
        let custom_event = CustomEvent {
            event_type: "test_event".to_string(),
            data: serde_json::json!({"key": "value"}),
            timestamp: chrono::Utc::now(),
        };

        // Execute custom event
        let result = registry.execute_custom_event(&mut context, &custom_event);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hook_priority() {
        let registry = HookRegistry::new();

        // Register hooks with different priorities
        let high_priority_hook = LoggingHook::new(LogLevel::Info); // Priority 100
        let low_priority_hook = MetricsHook::new(); // Priority 50

        registry.register_hook(low_priority_hook).unwrap();
        registry.register_hook(high_priority_hook).unwrap();

        // Check execution order (high priority first)
        let execution_hooks = registry.execution_hooks.read().unwrap();
        assert_eq!(execution_hooks[0], "logging_hook"); // Higher priority
        assert_eq!(execution_hooks[1], "metrics_hook"); // Lower priority
    }

    #[test]
    fn test_hook_event_subscriptions() {
        let registry = HookRegistry::new();
        let logging_hook = LoggingHook::new(LogLevel::Info);
        registry.register_hook(logging_hook).unwrap();

        // Check event subscriptions
        let before_hooks = registry.get_event_hooks(HookEvent::BeforeExplanation);
        assert_eq!(before_hooks.len(), 1);
        assert_eq!(before_hooks[0].hook_id(), "logging_hook");

        let after_hooks = registry.get_event_hooks(HookEvent::AfterExplanation);
        assert_eq!(after_hooks.len(), 1);

        let progress_hooks = registry.get_event_hooks(HookEvent::OnProgress);
        assert_eq!(progress_hooks.len(), 1);
    }
}

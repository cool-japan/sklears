//! Performance monitoring hooks for SIMD operations
//!
//! This module provides a hook system that allows users to register callbacks
//! that get executed during SIMD operations for performance monitoring and debugging.

#[cfg(not(feature = "no-std"))]
use std::collections::HashMap;
#[cfg(not(feature = "no-std"))]
use std::fmt;
#[cfg(not(feature = "no-std"))]
use std::string::ToString;
#[cfg(not(feature = "no-std"))]
use std::sync::{Arc, Mutex, RwLock};
#[cfg(not(feature = "no-std"))]
use std::thread::ThreadId;
#[cfg(not(feature = "no-std"))]
use std::time::{Duration, Instant};

#[cfg(feature = "no-std")]
use alloc::{
    collections::BTreeMap as HashMap,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
#[cfg(feature = "no-std")]
use core::fmt;
#[cfg(feature = "no-std")]
use spin::{Mutex, RwLock};

// Type aliases for conditional compilation

#[cfg(feature = "no-std")]
pub type ThreadId = u64; // Mock thread ID for no-std
#[cfg(feature = "no-std")]
#[derive(Debug, Clone, Copy)]
pub struct Duration(u64); // Mock duration in microseconds
#[cfg(feature = "no-std")]
#[derive(Debug, Clone, Copy)]
pub struct Instant(u64); // Mock instant as counter

#[cfg(feature = "no-std")]
impl Instant {
    pub fn now() -> Self {
        Instant(0) // Mock implementation
    }
}

#[cfg(feature = "no-std")]
impl Duration {
    pub fn from_millis(_millis: u64) -> Self {
        Duration(0) // Mock implementation
    }

    pub fn from_nanos(_nanos: u64) -> Self {
        Duration(0) // Mock implementation
    }

    pub fn as_nanos(&self) -> u128 {
        self.0 as u128 * 1000 // Mock implementation
    }
}

#[cfg(feature = "no-std")]
impl core::ops::Div<u32> for Duration {
    type Output = Duration;

    fn div(self, rhs: u32) -> Self::Output {
        Duration(if rhs == 0 {
            self.0
        } else {
            self.0 / rhs as u64
        })
    }
}

#[cfg(feature = "no-std")]
impl Default for Duration {
    fn default() -> Self {
        Duration(0)
    }
}

/// Hook execution phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookPhase {
    BeforeOperation,
    AfterOperation,
    OnError,
    OnOptimization,
}

/// Performance event information
#[derive(Debug, Clone)]
pub struct PerformanceEvent {
    pub operation_name: String,
    pub phase: HookPhase,
    pub timestamp: Instant,
    pub thread_id: ThreadId,
    pub input_size: usize,
    pub output_size: usize,
    pub execution_time: Option<Duration>,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl PerformanceEvent {
    pub fn new(
        operation_name: String,
        phase: HookPhase,
        input_size: usize,
        output_size: usize,
    ) -> Self {
        Self {
            operation_name,
            phase,
            timestamp: Instant::now(),
            #[cfg(not(feature = "no-std"))]
            thread_id: std::thread::current().id(),
            #[cfg(feature = "no-std")]
            thread_id: 0, // Mock thread ID
            input_size,
            output_size,
            execution_time: None,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_execution_time(mut self, duration: Duration) -> Self {
        self.execution_time = Some(duration);
        self
    }

    pub fn with_error(mut self, error: String) -> Self {
        self.error_message = Some(error);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Performance hook trait
pub trait PerformanceHook: Send + Sync {
    /// Called when a performance event occurs
    fn on_event(&self, event: &PerformanceEvent);

    /// Get the name of the hook
    fn name(&self) -> &str;

    /// Get the phases this hook is interested in
    fn interested_phases(&self) -> Vec<HookPhase> {
        vec![HookPhase::BeforeOperation, HookPhase::AfterOperation]
    }

    /// Check if this hook should be called for a specific operation
    fn should_handle(&self, operation_name: &str) -> bool {
        let _ = operation_name; // Default: handle all operations
        true
    }
}

/// Hook manager for registering and executing performance hooks
pub struct HookManager {
    hooks: RwLock<HashMap<String, Arc<dyn PerformanceHook>>>,
    enabled: RwLock<bool>,
    event_buffer: Mutex<Vec<PerformanceEvent>>,
    max_buffer_size: usize,
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

impl HookManager {
    /// Create a new hook manager
    pub fn new() -> Self {
        Self {
            hooks: RwLock::new(HashMap::new()),
            enabled: RwLock::new(true),
            event_buffer: Mutex::new(Vec::new()),
            max_buffer_size: 10000,
        }
    }

    /// Helper function to handle RwLock read locking in both std and no-std environments
    #[cfg(not(feature = "no-std"))]
    fn read_hooks(
        &self,
    ) -> std::sync::RwLockReadGuard<'_, HashMap<String, Arc<dyn PerformanceHook>>> {
        self.hooks.read().unwrap()
    }

    #[cfg(feature = "no-std")]
    fn read_hooks(&self) -> spin::RwLockReadGuard<'_, HashMap<String, Arc<dyn PerformanceHook>>> {
        self.hooks.read()
    }

    /// Helper function to handle RwLock write locking in both std and no-std environments
    #[cfg(not(feature = "no-std"))]
    fn write_hooks(
        &self,
    ) -> std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<dyn PerformanceHook>>> {
        self.hooks.write().unwrap()
    }

    #[cfg(feature = "no-std")]
    fn write_hooks(&self) -> spin::RwLockWriteGuard<'_, HashMap<String, Arc<dyn PerformanceHook>>> {
        self.hooks.write()
    }

    /// Helper function to handle enabled RwLock read locking in both std and no-std environments
    #[cfg(not(feature = "no-std"))]
    fn read_enabled(&self) -> std::sync::RwLockReadGuard<'_, bool> {
        self.enabled.read().unwrap()
    }

    #[cfg(feature = "no-std")]
    fn read_enabled(&self) -> spin::RwLockReadGuard<'_, bool> {
        self.enabled.read()
    }

    /// Helper function to handle enabled RwLock write locking in both std and no-std environments
    #[cfg(not(feature = "no-std"))]
    fn write_enabled(&self) -> std::sync::RwLockWriteGuard<'_, bool> {
        self.enabled.write().unwrap()
    }

    #[cfg(feature = "no-std")]
    fn write_enabled(&self) -> spin::RwLockWriteGuard<'_, bool> {
        self.enabled.write()
    }

    /// Helper function to handle event buffer Mutex locking in both std and no-std environments
    #[cfg(not(feature = "no-std"))]
    fn lock_event_buffer(&self) -> std::sync::MutexGuard<'_, Vec<PerformanceEvent>> {
        self.event_buffer.lock().unwrap()
    }

    #[cfg(feature = "no-std")]
    fn lock_event_buffer(&self) -> spin::MutexGuard<'_, Vec<PerformanceEvent>> {
        self.event_buffer.lock()
    }

    /// Register a performance hook
    pub fn register_hook(&self, hook: Arc<dyn PerformanceHook>) -> Result<(), HookError> {
        let name = hook.name().to_string();
        let mut hooks = self.write_hooks();

        if hooks.contains_key(&name) {
            return Err(HookError::AlreadyRegistered(name));
        }

        hooks.insert(name, hook);
        Ok(())
    }

    /// Unregister a performance hook
    pub fn unregister_hook(&self, name: &str) -> Result<(), HookError> {
        let mut hooks = self.write_hooks();
        if hooks.remove(name).is_none() {
            return Err(HookError::NotFound(name.to_string()));
        }
        Ok(())
    }

    /// Fire a performance event
    pub fn fire_event(&self, event: PerformanceEvent) {
        // Check if monitoring is enabled
        if !*self.read_enabled() {
            return;
        }

        // Store event in buffer
        {
            let mut buffer = self.lock_event_buffer();
            buffer.push(event.clone());

            // Limit buffer size
            if buffer.len() > self.max_buffer_size {
                buffer.remove(0);
            }
        }

        // Execute hooks
        let hooks = self.read_hooks();
        for hook in hooks.values() {
            if hook.should_handle(&event.operation_name)
                && hook.interested_phases().contains(&event.phase)
            {
                hook.on_event(&event);
            }
        }
    }

    /// Enable or disable performance monitoring
    pub fn set_enabled(&self, enabled: bool) {
        *self.write_enabled() = enabled;
    }

    /// Check if performance monitoring is enabled
    pub fn is_enabled(&self) -> bool {
        *self.read_enabled()
    }

    /// Get the event buffer
    pub fn get_events(&self) -> Vec<PerformanceEvent> {
        self.lock_event_buffer().clone()
    }

    /// Clear the event buffer
    pub fn clear_events(&self) {
        self.lock_event_buffer().clear();
    }

    /// Get statistics about registered hooks
    pub fn get_hook_stats(&self) -> HookStats {
        let hooks = self.read_hooks();
        let buffer = self.lock_event_buffer();

        HookStats {
            total_hooks: hooks.len(),
            hook_names: hooks.keys().cloned().collect(),
            total_events: buffer.len(),
            is_enabled: self.is_enabled(),
        }
    }

    /// Set maximum buffer size
    pub fn set_max_buffer_size(&mut self, size: usize) {
        self.max_buffer_size = size;
    }
}

/// Hook manager statistics
#[derive(Debug, Clone)]
pub struct HookStats {
    pub total_hooks: usize,
    pub hook_names: Vec<String>,
    pub total_events: usize,
    pub is_enabled: bool,
}

/// Hook error types
#[derive(Debug, Clone)]
pub enum HookError {
    AlreadyRegistered(String),
    NotFound(String),
    ExecutionFailed(String),
}

impl fmt::Display for HookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HookError::AlreadyRegistered(name) => {
                write!(f, "Hook '{}' is already registered", name)
            }
            HookError::NotFound(name) => write!(f, "Hook '{}' not found", name),
            HookError::ExecutionFailed(msg) => write!(f, "Hook execution failed: {}", msg),
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl std::error::Error for HookError {}

#[cfg(feature = "no-std")]
impl core::error::Error for HookError {}

/// Global hook manager instance
pub static GLOBAL_HOOK_MANAGER: once_cell::sync::Lazy<HookManager> =
    once_cell::sync::Lazy::new(HookManager::new);

/// Convenience functions for global hook manager
pub mod global {
    use super::*;

    /// Register a hook globally
    pub fn register_hook(hook: Arc<dyn PerformanceHook>) -> Result<(), HookError> {
        GLOBAL_HOOK_MANAGER.register_hook(hook)
    }

    /// Fire an event globally
    pub fn fire_event(event: PerformanceEvent) {
        GLOBAL_HOOK_MANAGER.fire_event(event);
    }

    /// Enable/disable global monitoring
    pub fn set_enabled(enabled: bool) {
        GLOBAL_HOOK_MANAGER.set_enabled(enabled);
    }

    /// Get global hook statistics
    pub fn get_stats() -> HookStats {
        GLOBAL_HOOK_MANAGER.get_hook_stats()
    }

    /// Clear global event buffer
    pub fn clear_events() {
        GLOBAL_HOOK_MANAGER.clear_events();
    }
}

/// Macro for creating performance monitoring scopes
#[macro_export]
macro_rules! perf_scope {
    ($operation_name:expr, $input_size:expr, $output_size:expr, $body:expr) => {{
        use $crate::performance_hooks::{global, HookPhase, PerformanceEvent};

        // Fire before event
        let before_event = PerformanceEvent::new(
            $operation_name.to_string(),
            HookPhase::BeforeOperation,
            $input_size,
            $output_size,
        );
        global::fire_event(before_event);

        // Execute operation
        let start_time = Instant::now();
        let result = $body;
        let execution_time = start_time.elapsed();

        // Fire after event
        let after_event = PerformanceEvent::new(
            $operation_name.to_string(),
            HookPhase::AfterOperation,
            $input_size,
            $output_size,
        )
        .with_execution_time(execution_time);

        global::fire_event(after_event);

        result
    }};
}

/// Built-in performance hooks
pub mod builtin_hooks {
    use super::*;
    #[cfg(feature = "no-std")]
    use core::sync::atomic::{AtomicU64, Ordering};
    #[cfg(not(feature = "no-std"))]
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Simple logging hook that prints events to console
    pub struct LoggingHook {
        name: String,
    }

    impl LoggingHook {
        pub fn new(name: String) -> Self {
            Self { name }
        }
    }

    impl PerformanceHook for LoggingHook {
        fn on_event(&self, event: &PerformanceEvent) {
            match event.phase {
                HookPhase::BeforeOperation => {
                    #[cfg(not(feature = "no-std"))]
                    println!(
                        "[{}] Starting {} (input: {}, output: {})",
                        self.name, event.operation_name, event.input_size, event.output_size
                    );
                }
                HookPhase::AfterOperation =>
                {
                    #[cfg(not(feature = "no-std"))]
                    if let Some(time) = event.execution_time {
                        println!(
                            "[{}] Finished {} in {:?}",
                            self.name, event.operation_name, time
                        );
                    }
                }
                HookPhase::OnError =>
                {
                    #[cfg(not(feature = "no-std"))]
                    if let Some(error) = &event.error_message {
                        println!(
                            "[{}] Error in {}: {}",
                            self.name, event.operation_name, error
                        );
                    }
                }
                HookPhase::OnOptimization => {
                    #[cfg(not(feature = "no-std"))]
                    println!(
                        "[{}] Optimization applied to {}",
                        self.name, event.operation_name
                    );
                }
            }
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    /// Statistics collection hook
    pub struct StatsHook {
        name: String,
        operation_counts: RwLock<HashMap<String, AtomicU64>>,
        total_execution_time: RwLock<HashMap<String, AtomicU64>>, // nanoseconds
        total_elements_processed: RwLock<HashMap<String, AtomicU64>>,
    }

    impl StatsHook {
        pub fn new(name: String) -> Self {
            Self {
                name,
                operation_counts: RwLock::new(HashMap::new()),
                total_execution_time: RwLock::new(HashMap::new()),
                total_elements_processed: RwLock::new(HashMap::new()),
            }
        }

        /// Helper function to handle operation counts RwLock read locking
        #[cfg(not(feature = "no-std"))]
        fn read_counts(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, AtomicU64>> {
            self.operation_counts.read().unwrap()
        }

        #[cfg(feature = "no-std")]
        fn read_counts(&self) -> spin::RwLockReadGuard<'_, HashMap<String, AtomicU64>> {
            self.operation_counts.read()
        }

        /// Helper function to handle operation counts RwLock write locking
        #[cfg(not(feature = "no-std"))]
        fn write_counts(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<String, AtomicU64>> {
            self.operation_counts.write().unwrap()
        }

        #[cfg(feature = "no-std")]
        fn write_counts(&self) -> spin::RwLockWriteGuard<'_, HashMap<String, AtomicU64>> {
            self.operation_counts.write()
        }

        /// Helper function to handle execution time RwLock read locking
        #[cfg(not(feature = "no-std"))]
        fn read_times(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, AtomicU64>> {
            self.total_execution_time.read().unwrap()
        }

        #[cfg(feature = "no-std")]
        fn read_times(&self) -> spin::RwLockReadGuard<'_, HashMap<String, AtomicU64>> {
            self.total_execution_time.read()
        }

        /// Helper function to handle execution time RwLock write locking
        #[cfg(not(feature = "no-std"))]
        fn write_times(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<String, AtomicU64>> {
            self.total_execution_time.write().unwrap()
        }

        #[cfg(feature = "no-std")]
        fn write_times(&self) -> spin::RwLockWriteGuard<'_, HashMap<String, AtomicU64>> {
            self.total_execution_time.write()
        }

        /// Helper function to handle elements processed RwLock read locking
        #[cfg(not(feature = "no-std"))]
        fn read_elements(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, AtomicU64>> {
            self.total_elements_processed.read().unwrap()
        }

        #[cfg(feature = "no-std")]
        fn read_elements(&self) -> spin::RwLockReadGuard<'_, HashMap<String, AtomicU64>> {
            self.total_elements_processed.read()
        }

        /// Helper function to handle elements processed RwLock write locking
        #[cfg(not(feature = "no-std"))]
        fn write_elements(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<String, AtomicU64>> {
            self.total_elements_processed.write().unwrap()
        }

        #[cfg(feature = "no-std")]
        fn write_elements(&self) -> spin::RwLockWriteGuard<'_, HashMap<String, AtomicU64>> {
            self.total_elements_processed.write()
        }

        pub fn get_stats(&self) -> HashMap<String, OperationStats> {
            let counts = self.read_counts();
            let times = self.read_times();
            let elements = self.read_elements();

            let mut stats = HashMap::new();
            for (op_name, count) in counts.iter() {
                let count_val = count.load(Ordering::Relaxed);
                let time_val = times
                    .get(op_name)
                    .map(|t| Duration::from_nanos(t.load(Ordering::Relaxed)))
                    .unwrap_or_default();
                let elements_val = elements
                    .get(op_name)
                    .map(|e| e.load(Ordering::Relaxed))
                    .unwrap_or(0);

                stats.insert(
                    op_name.clone(),
                    OperationStats {
                        call_count: count_val,
                        total_time: time_val,
                        total_elements: elements_val,
                        avg_time: if count_val > 0 {
                            time_val / count_val as u32
                        } else {
                            Duration::default()
                        },
                    },
                );
            }
            stats
        }

        pub fn reset_stats(&self) {
            self.write_counts().clear();
            self.write_times().clear();
            self.write_elements().clear();
        }
    }

    impl PerformanceHook for StatsHook {
        fn on_event(&self, event: &PerformanceEvent) {
            if event.phase == HookPhase::AfterOperation {
                let op_name = &event.operation_name;

                // Update call count
                {
                    let mut counts = self.write_counts();
                    counts
                        .entry(op_name.clone())
                        .or_insert_with(|| AtomicU64::new(0))
                        .fetch_add(1, Ordering::Relaxed);
                }

                // Update execution time
                if let Some(time) = event.execution_time {
                    let mut times = self.write_times();
                    times
                        .entry(op_name.clone())
                        .or_insert_with(|| AtomicU64::new(0))
                        .fetch_add(time.as_nanos() as u64, Ordering::Relaxed);
                }

                // Update elements processed
                {
                    let mut elements = self.write_elements();
                    elements
                        .entry(op_name.clone())
                        .or_insert_with(|| AtomicU64::new(0))
                        .fetch_add(event.input_size as u64, Ordering::Relaxed);
                }
            }
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn interested_phases(&self) -> Vec<HookPhase> {
            vec![HookPhase::AfterOperation]
        }
    }

    #[derive(Debug, Clone)]
    pub struct OperationStats {
        pub call_count: u64,
        pub total_time: Duration,
        pub total_elements: u64,
        pub avg_time: Duration,
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::builtin_hooks::*;
    use super::*;

    #[test]
    fn test_hook_registration() {
        let manager = HookManager::new();
        let hook = Arc::new(LoggingHook::new("test_hook".to_string()));

        assert!(manager.register_hook(hook).is_ok());

        let stats = manager.get_hook_stats();
        assert_eq!(stats.total_hooks, 1);
        assert!(stats.hook_names.contains(&"test_hook".to_string()));
    }

    #[test]
    fn test_event_firing() {
        let manager = HookManager::new();
        let hook = Arc::new(LoggingHook::new("test_hook".to_string()));
        manager.register_hook(hook).unwrap();

        let event = PerformanceEvent::new(
            "test_operation".to_string(),
            HookPhase::BeforeOperation,
            100,
            100,
        );

        manager.fire_event(event);

        let events = manager.get_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].operation_name, "test_operation");
    }

    #[test]
    fn test_stats_hook() {
        let manager = HookManager::new();
        let stats_hook = Arc::new(StatsHook::new("stats".to_string()));
        let stats_hook_clone = stats_hook.clone();

        manager.register_hook(stats_hook).unwrap();

        // Fire some events
        for i in 0..5 {
            let event =
                PerformanceEvent::new("test_op".to_string(), HookPhase::AfterOperation, 100, 100)
                    .with_execution_time(Duration::from_millis(i));

            manager.fire_event(event);
        }

        let stats = stats_hook_clone.get_stats();
        assert!(stats.contains_key("test_op"));

        let op_stats = &stats["test_op"];
        assert_eq!(op_stats.call_count, 5);
        assert_eq!(op_stats.total_elements, 500);
    }

    #[test]
    fn test_global_hooks() {
        let hook = Arc::new(LoggingHook::new("global_test".to_string()));
        global::register_hook(hook).unwrap();

        let event = PerformanceEvent::new(
            "global_test_op".to_string(),
            HookPhase::BeforeOperation,
            50,
            50,
        );

        global::fire_event(event);

        let stats = global::get_stats();
        assert!(stats.hook_names.contains(&"global_test".to_string()));
    }

    #[test]
    fn test_enable_disable() {
        let manager = HookManager::new();
        assert!(manager.is_enabled());

        manager.set_enabled(false);
        assert!(!manager.is_enabled());

        manager.set_enabled(true);
        assert!(manager.is_enabled());
    }

    #[test]
    fn test_error_handling() {
        let manager = HookManager::new();
        let hook1 = Arc::new(LoggingHook::new("duplicate".to_string()));
        let hook2 = Arc::new(LoggingHook::new("duplicate".to_string()));

        assert!(manager.register_hook(hook1).is_ok());
        assert!(matches!(
            manager.register_hook(hook2),
            Err(HookError::AlreadyRegistered(_))
        ));

        assert!(matches!(
            manager.unregister_hook("nonexistent"),
            Err(HookError::NotFound(_))
        ));
    }
}

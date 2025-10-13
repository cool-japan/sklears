//! Component Lifecycle Management System
//!
//! This module provides comprehensive lifecycle management capabilities for modular
//! components including state tracking, dependency-aware initialization, graceful
//! shutdown, and lifecycle event coordination.

use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime};

/// Lifecycle manager for component state coordination
///
/// Manages component lifecycle states, dependency resolution, initialization
/// ordering, and graceful shutdown procedures across the modular system.
#[derive(Debug)]
pub struct LifecycleManager {
    /// Component states tracking
    component_states: HashMap<String, ComponentLifecycleState>,
    /// Component dependencies
    dependencies: HashMap<String, Vec<String>>,
    /// Lifecycle listeners
    listeners: HashMap<LifecycleEvent, Vec<String>>,
    /// Initialization order cache
    initialization_order: Vec<String>,
    /// Lifecycle configuration
    config: LifecycleConfig,
    /// Lifecycle metrics
    metrics: LifecycleMetrics,
}

impl LifecycleManager {
    /// Create a new lifecycle manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            component_states: HashMap::new(),
            dependencies: HashMap::new(),
            listeners: HashMap::new(),
            initialization_order: Vec::new(),
            config: LifecycleConfig::default(),
            metrics: LifecycleMetrics::new(),
        }
    }

    /// Register a component with the lifecycle manager
    pub fn register_component(
        &mut self,
        component_id: &str,
        dependencies: Vec<String>,
    ) -> SklResult<()> {
        if self.component_states.contains_key(component_id) {
            return Err(SklearsError::InvalidInput(format!(
                "Component {component_id} already registered"
            )));
        }

        self.component_states
            .insert(component_id.to_string(), ComponentLifecycleState::Created);
        self.dependencies
            .insert(component_id.to_string(), dependencies);
        self.metrics.total_components += 1;

        // Invalidate initialization order cache
        self.initialization_order.clear();

        Ok(())
    }

    /// Unregister a component
    pub fn unregister_component(&mut self, component_id: &str) -> SklResult<()> {
        if let Some(state) = self.component_states.get(component_id) {
            if matches!(state, ComponentLifecycleState::Running) {
                return Err(SklearsError::InvalidInput(format!(
                    "Cannot unregister running component {component_id}"
                )));
            }
        }

        self.component_states.remove(component_id);
        self.dependencies.remove(component_id);
        self.initialization_order.retain(|id| id != component_id);
        self.metrics.total_components = self.metrics.total_components.saturating_sub(1);

        Ok(())
    }

    /// Set component state
    pub fn set_component_state(
        &mut self,
        component_id: &str,
        state: ComponentLifecycleState,
    ) -> SklResult<()> {
        if !self.component_states.contains_key(component_id) {
            return Err(SklearsError::InvalidInput(format!(
                "Component {component_id} not registered"
            )));
        }

        let old_state = self.component_states.get(component_id).unwrap().clone();

        // Validate state transition
        if !self.is_valid_transition(&old_state, &state) {
            return Err(SklearsError::InvalidInput(format!(
                "Invalid state transition from {old_state:?} to {state:?}"
            )));
        }

        self.component_states
            .insert(component_id.to_string(), state.clone());

        // Update metrics
        match &state {
            ComponentLifecycleState::Ready => self.metrics.ready_components += 1,
            ComponentLifecycleState::Running => {
                self.metrics.running_components += 1;
                self.metrics.ready_components = self.metrics.ready_components.saturating_sub(1);
            }
            ComponentLifecycleState::Stopped => {
                self.metrics.running_components = self.metrics.running_components.saturating_sub(1);
            }
            ComponentLifecycleState::Error(_) => self.metrics.failed_components += 1,
            _ => {}
        }

        // Notify listeners
        self.notify_lifecycle_event(LifecycleEvent::StateChanged {
            component_id: component_id.to_string(),
            old_state,
            new_state: state,
        })?;

        Ok(())
    }

    /// Get component state
    #[must_use]
    pub fn get_component_state(&self, component_id: &str) -> Option<&ComponentLifecycleState> {
        self.component_states.get(component_id)
    }

    /// Initialize all components in dependency order
    pub fn initialize_all_components(&mut self) -> SklResult<InitializationResult> {
        let start_time = SystemTime::now();
        let mut result = InitializationResult {
            total_components: self.component_states.len(),
            successful_initializations: 0,
            failed_initializations: 0,
            initialization_order: Vec::new(),
            duration: Duration::from_secs(0),
            errors: Vec::new(),
        };

        // Calculate initialization order if not cached
        if self.initialization_order.is_empty() {
            self.initialization_order = self.calculate_initialization_order()?;
        }

        result.initialization_order = self.initialization_order.clone();

        // Initialize components in order
        let initialization_order = self.initialization_order.clone();
        for component_id in &initialization_order {
            match self.initialize_component(component_id) {
                Ok(()) => {
                    result.successful_initializations += 1;
                    self.notify_lifecycle_event(LifecycleEvent::ComponentInitialized {
                        component_id: component_id.clone(),
                    })?;
                }
                Err(e) => {
                    result.failed_initializations += 1;
                    result
                        .errors
                        .push(format!("Failed to initialize {component_id}: {e}"));

                    if self.config.fail_fast_initialization {
                        result.duration = start_time.elapsed().unwrap_or(Duration::from_secs(0));
                        return Ok(result);
                    }
                }
            }
        }

        result.duration = start_time.elapsed().unwrap_or(Duration::from_secs(0));
        self.metrics.last_initialization_time = result.duration;

        Ok(result)
    }

    /// Initialize a specific component
    pub fn initialize_component(&mut self, component_id: &str) -> SklResult<()> {
        // Check if component exists
        let current_state = self.component_states.get(component_id).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Component {component_id} not found"))
        })?;

        // Check if already initialized
        if matches!(
            current_state,
            ComponentLifecycleState::Ready | ComponentLifecycleState::Running
        ) {
            return Ok(());
        }

        // Check dependencies are ready
        if let Some(deps) = self.dependencies.get(component_id) {
            for dep in deps {
                let dep_state = self.component_states.get(dep).ok_or_else(|| {
                    SklearsError::InvalidInput(format!("Dependency {dep} not found"))
                })?;

                if !matches!(
                    dep_state,
                    ComponentLifecycleState::Ready | ComponentLifecycleState::Running
                ) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Dependency {dep} not ready for {component_id}"
                    )));
                }
            }
        }

        // Set initializing state
        self.set_component_state(component_id, ComponentLifecycleState::Initializing)?;

        // Simulate initialization (in real implementation, this would call component.initialize())
        std::thread::sleep(Duration::from_millis(10)); // Simulate work

        // Set ready state
        self.set_component_state(component_id, ComponentLifecycleState::Ready)?;

        Ok(())
    }

    /// Shutdown all components in reverse dependency order
    pub fn shutdown_all_components(&mut self) -> SklResult<ShutdownResult> {
        let start_time = SystemTime::now();
        let mut result = ShutdownResult {
            total_components: self.component_states.len(),
            successful_shutdowns: 0,
            failed_shutdowns: 0,
            shutdown_order: Vec::new(),
            duration: Duration::from_secs(0),
            errors: Vec::new(),
        };

        // Reverse the initialization order for shutdown
        let shutdown_order: Vec<String> = self.initialization_order.iter().rev().cloned().collect();
        result.shutdown_order = shutdown_order.clone();

        // Shutdown components in reverse order
        for component_id in &shutdown_order {
            match self.shutdown_component(component_id) {
                Ok(()) => {
                    result.successful_shutdowns += 1;
                    self.notify_lifecycle_event(LifecycleEvent::ComponentShutdown {
                        component_id: component_id.clone(),
                    })?;
                }
                Err(e) => {
                    result.failed_shutdowns += 1;
                    result
                        .errors
                        .push(format!("Failed to shutdown {component_id}: {e}"));
                }
            }
        }

        result.duration = start_time.elapsed().unwrap_or(Duration::from_secs(0));
        self.metrics.last_shutdown_time = result.duration;

        Ok(result)
    }

    /// Shutdown a specific component
    pub fn shutdown_component(&mut self, component_id: &str) -> SklResult<()> {
        let current_state = self.component_states.get(component_id).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Component {component_id} not found"))
        })?;

        if matches!(current_state, ComponentLifecycleState::Stopped) {
            return Ok(());
        }

        // Set stopping state
        self.set_component_state(component_id, ComponentLifecycleState::Stopping)?;

        // Simulate shutdown (in real implementation, this would call component.shutdown())
        std::thread::sleep(Duration::from_millis(5)); // Simulate work

        // Set stopped state
        self.set_component_state(component_id, ComponentLifecycleState::Stopped)?;

        Ok(())
    }

    /// Add a lifecycle event listener
    pub fn add_listener(&mut self, event: LifecycleEvent, listener_id: &str) {
        self.listeners
            .entry(event)
            .or_default()
            .push(listener_id.to_string());
    }

    /// Remove a lifecycle event listener
    pub fn remove_listener(&mut self, event: &LifecycleEvent, listener_id: &str) {
        if let Some(listeners) = self.listeners.get_mut(event) {
            listeners.retain(|id| id != listener_id);
        }
    }

    /// Get lifecycle metrics
    #[must_use]
    pub fn get_metrics(&self) -> &LifecycleMetrics {
        &self.metrics
    }

    /// Get all component states
    #[must_use]
    pub fn get_all_states(&self) -> &HashMap<String, ComponentLifecycleState> {
        &self.component_states
    }

    /// Check if all components are in a specific state
    #[must_use]
    pub fn all_components_in_state(&self, state: &ComponentLifecycleState) -> bool {
        self.component_states.values().all(|s| s == state)
    }

    /// Get components in a specific state
    #[must_use]
    pub fn get_components_in_state(&self, state: &ComponentLifecycleState) -> Vec<String> {
        self.component_states
            .iter()
            .filter_map(|(id, s)| if s == state { Some(id.clone()) } else { None })
            .collect()
    }

    fn calculate_initialization_order(&self) -> SklResult<Vec<String>> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();

        for component_id in self.component_states.keys() {
            if !visited.contains(component_id) {
                self.topological_sort(component_id, &mut visited, &mut visiting, &mut order)?;
            }
        }

        Ok(order)
    }

    fn topological_sort(
        &self,
        component_id: &str,
        visited: &mut HashSet<String>,
        visiting: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) -> SklResult<()> {
        if visiting.contains(component_id) {
            return Err(SklearsError::InvalidInput(format!(
                "Circular dependency detected involving {component_id}"
            )));
        }

        if visited.contains(component_id) {
            return Ok(());
        }

        visiting.insert(component_id.to_string());

        if let Some(deps) = self.dependencies.get(component_id) {
            for dep in deps {
                self.topological_sort(dep, visited, visiting, order)?;
            }
        }

        visiting.remove(component_id);
        visited.insert(component_id.to_string());
        order.push(component_id.to_string());

        Ok(())
    }

    fn is_valid_transition(
        &self,
        from: &ComponentLifecycleState,
        to: &ComponentLifecycleState,
    ) -> bool {
        use ComponentLifecycleState::{
            Created, Error, Initializing, Paused, Ready, Running, Stopped, Stopping,
        };

        match (from, to) {
            (Created, Initializing) => true,
            (Initializing, Ready) => true,
            (Initializing, Error(_)) => true,
            (Ready, Running) => true,
            (Ready, Paused) => true,
            (Ready, Stopping) => true,
            (Running, Paused) => true,
            (Running, Stopping) => true,
            (Paused, Running) => true,
            (Paused, Stopping) => true,
            (Stopping, Stopped) => true,
            (Error(_), Initializing) => true, // Allow retry
            _ => false,
        }
    }

    fn notify_lifecycle_event(&mut self, event: LifecycleEvent) -> SklResult<()> {
        if let Some(listeners) = self.listeners.get(&event) {
            for listener in listeners {
                // In real implementation, would notify actual listener
                // For now, just track that notification would be sent
                self.metrics.total_notifications += 1;
            }
        }
        Ok(())
    }
}

/// Component lifecycle state
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentLifecycleState {
    /// Component has been created but not initialized
    Created,
    /// Component is currently initializing
    Initializing,
    /// Component is ready for use
    Ready,
    /// Component is actively running
    Running,
    /// Component is paused
    Paused,
    /// Component is shutting down
    Stopping,
    /// Component has been stopped
    Stopped,
    /// Component encountered an error
    Error(String),
}

/// Lifecycle events for monitoring and coordination
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LifecycleEvent {
    StateChanged {
        component_id: String,
        old_state: ComponentLifecycleState,
        new_state: ComponentLifecycleState,
    },
    /// Component was initialized
    ComponentInitialized { component_id: String },
    /// Component was shutdown
    ComponentShutdown { component_id: String },
    /// All components initialized
    AllComponentsInitialized,
    /// All components shutdown
    AllComponentsShutdown,
}

/// Lifecycle configuration
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Fail fast on initialization errors
    pub fail_fast_initialization: bool,
    /// Maximum initialization timeout per component
    pub initialization_timeout: Duration,
    /// Maximum shutdown timeout per component
    pub shutdown_timeout: Duration,
    /// Enable automatic restart on failures
    pub auto_restart_on_failure: bool,
    /// Maximum restart attempts
    pub max_restart_attempts: u32,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            fail_fast_initialization: true,
            initialization_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(10),
            auto_restart_on_failure: false,
            max_restart_attempts: 3,
        }
    }
}

/// Lifecycle metrics for monitoring
#[derive(Debug, Clone)]
pub struct LifecycleMetrics {
    /// Total number of components
    pub total_components: usize,
    /// Number of ready components
    pub ready_components: usize,
    /// Number of running components
    pub running_components: usize,
    /// Number of failed components
    pub failed_components: usize,
    /// Last initialization time
    pub last_initialization_time: Duration,
    /// Last shutdown time
    pub last_shutdown_time: Duration,
    /// Total lifecycle notifications sent
    pub total_notifications: u64,
}

impl Default for LifecycleMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl LifecycleMetrics {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_components: 0,
            ready_components: 0,
            running_components: 0,
            failed_components: 0,
            last_initialization_time: Duration::from_secs(0),
            last_shutdown_time: Duration::from_secs(0),
            total_notifications: 0,
        }
    }
}

/// Result of initialization process
#[derive(Debug, Clone)]
pub struct InitializationResult {
    /// Total number of components
    pub total_components: usize,
    /// Number of successful initializations
    pub successful_initializations: usize,
    /// Number of failed initializations
    pub failed_initializations: usize,
    /// Order in which components were initialized
    pub initialization_order: Vec<String>,
    /// Total initialization duration
    pub duration: Duration,
    /// Initialization errors
    pub errors: Vec<String>,
}

/// Result of shutdown process
#[derive(Debug, Clone)]
pub struct ShutdownResult {
    /// Total number of components
    pub total_components: usize,
    /// Number of successful shutdowns
    pub successful_shutdowns: usize,
    /// Number of failed shutdowns
    pub failed_shutdowns: usize,
    /// Order in which components were shutdown
    pub shutdown_order: Vec<String>,
    /// Total shutdown duration
    pub duration: Duration,
    /// Shutdown errors
    pub errors: Vec<String>,
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle_manager_creation() {
        let manager = LifecycleManager::new();
        assert_eq!(manager.get_metrics().total_components, 0);
    }

    #[test]
    fn test_component_registration() {
        let mut manager = LifecycleManager::new();

        let result = manager.register_component("component_a", vec![]);
        assert!(result.is_ok());
        assert_eq!(manager.get_metrics().total_components, 1);

        let state = manager.get_component_state("component_a");
        assert_eq!(state, Some(&ComponentLifecycleState::Created));
    }

    #[test]
    fn test_state_transitions() {
        let mut manager = LifecycleManager::new();
        manager.register_component("component_a", vec![]).unwrap();

        // Valid transition
        let result =
            manager.set_component_state("component_a", ComponentLifecycleState::Initializing);
        assert!(result.is_ok());

        // Invalid transition
        let result = manager.set_component_state("component_a", ComponentLifecycleState::Stopped);
        assert!(result.is_err());
    }

    #[test]
    fn test_dependency_resolution() {
        let mut manager = LifecycleManager::new();

        manager.register_component("component_a", vec![]).unwrap();
        manager
            .register_component("component_b", vec!["component_a".to_string()])
            .unwrap();

        let order = manager.calculate_initialization_order().unwrap();
        assert_eq!(order, vec!["component_a", "component_b"]);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut manager = LifecycleManager::new();

        manager
            .register_component("component_a", vec!["component_b".to_string()])
            .unwrap();
        manager
            .register_component("component_b", vec!["component_a".to_string()])
            .unwrap();

        let result = manager.calculate_initialization_order();
        assert!(result.is_err());
    }

    #[test]
    fn test_component_initialization() {
        let mut manager = LifecycleManager::new();
        manager.register_component("component_a", vec![]).unwrap();

        let result = manager.initialize_component("component_a");
        assert!(result.is_ok());

        let state = manager.get_component_state("component_a");
        assert_eq!(state, Some(&ComponentLifecycleState::Ready));
    }

    #[test]
    fn test_full_lifecycle() {
        let mut manager = LifecycleManager::new();

        manager.register_component("component_a", vec![]).unwrap();
        manager
            .register_component("component_b", vec!["component_a".to_string()])
            .unwrap();

        let init_result = manager.initialize_all_components().unwrap();
        assert_eq!(init_result.successful_initializations, 2);
        assert_eq!(init_result.failed_initializations, 0);

        let shutdown_result = manager.shutdown_all_components().unwrap();
        assert_eq!(shutdown_result.successful_shutdowns, 2);
        assert_eq!(shutdown_result.failed_shutdowns, 0);
    }
}

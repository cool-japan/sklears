//! Event System for Component Communication
//!
//! This module provides comprehensive event-driven communication capabilities for
//! modular components including event buses, component events, and subscription
//! management for decoupled component interaction.

use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;

/// Event bus for component communication
///
/// Provides publish-subscribe messaging between components with event routing,
/// subscription management, and event queuing for asynchronous processing.
pub struct EventBus {
    /// Subscribers mapped by event type
    subscribers: HashMap<String, Vec<String>>,
    /// Event queue for processing
    event_queue: Vec<ComponentEvent>,
    /// Event handlers for different event types
    event_handlers: HashMap<String, Box<dyn EventHandler>>,
    /// Event routing configuration
    routing_config: EventRoutingConfig,
    /// Event statistics for monitoring
    event_stats: EventStatistics,
}

impl EventBus {
    /// Create a new event bus
    #[must_use]
    pub fn new() -> Self {
        Self {
            subscribers: HashMap::new(),
            event_queue: Vec::new(),
            event_handlers: HashMap::new(),
            routing_config: EventRoutingConfig::default(),
            event_stats: EventStatistics::new(),
        }
    }

    /// Subscribe a component to an event type
    pub fn subscribe(&mut self, event_type: &str, component_id: &str) -> SklResult<()> {
        self.subscribers
            .entry(event_type.to_string())
            .or_default()
            .push(component_id.to_string());

        self.event_stats.total_subscriptions += 1;
        Ok(())
    }

    /// Unsubscribe a component from an event type
    pub fn unsubscribe(&mut self, event_type: &str, component_id: &str) -> SklResult<()> {
        if let Some(subscribers) = self.subscribers.get_mut(event_type) {
            subscribers.retain(|id| id != component_id);
            self.event_stats.total_unsubscriptions += 1;
        }
        Ok(())
    }

    /// Publish an event to the bus
    pub fn publish(&mut self, event: ComponentEvent) -> SklResult<()> {
        self.event_queue.push(event.clone());
        self.event_stats.total_events_published += 1;
        self.event_stats
            .events_by_type
            .entry(event.event_type.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);

        // Route event immediately if synchronous routing is enabled
        if self.routing_config.synchronous_routing {
            self.route_event(&event)?;
        }

        Ok(())
    }

    /// Emit an event (alias for publish for backward compatibility)
    pub fn emit_event(&mut self, event: ComponentEvent) -> SklResult<()> {
        self.publish(event)
    }

    /// Process all queued events
    pub fn process_events(&mut self) -> SklResult<Vec<EventProcessingResult>> {
        let mut results = Vec::new();
        let events_to_process = self.event_queue.drain(..).collect::<Vec<_>>();

        for event in events_to_process {
            let result = self.process_single_event(&event)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Process a single event
    fn process_single_event(&mut self, event: &ComponentEvent) -> SklResult<EventProcessingResult> {
        let start_time = std::time::SystemTime::now();
        let mut delivery_count = 0;
        let mut failed_deliveries = 0;

        // Route to target if specified
        if let Some(target) = &event.target {
            match self.deliver_to_target(event, target) {
                Ok(()) => delivery_count += 1,
                Err(_) => failed_deliveries += 1,
            }
        } else {
            // Broadcast to all subscribers of this event type
            if let Some(subscribers) = self.subscribers.get(&event.event_type) {
                for subscriber in subscribers {
                    match self.deliver_to_target(event, subscriber) {
                        Ok(()) => delivery_count += 1,
                        Err(_) => failed_deliveries += 1,
                    }
                }
            }
        }

        let processing_time = std::time::SystemTime::now()
            .duration_since(start_time)
            .unwrap_or_default();
        self.event_stats.total_processing_time += processing_time;

        Ok(EventProcessingResult {
            event_id: event.event_id.clone(),
            event_type: event.event_type.clone(),
            delivery_count,
            failed_deliveries,
            processing_time,
            success: failed_deliveries == 0,
        })
    }

    /// Route an event to its subscribers
    fn route_event(&mut self, event: &ComponentEvent) -> SklResult<()> {
        if let Some(subscribers) = self.subscribers.get(&event.event_type) {
            for subscriber in subscribers {
                if let Some(handler) = self.event_handlers.get(&event.event_type) {
                    handler.handle_event(event, subscriber)?;
                }
            }
        }
        Ok(())
    }

    /// Deliver event to specific target
    fn deliver_to_target(&self, event: &ComponentEvent, target: &str) -> SklResult<()> {
        // In a real implementation, this would deliver to the actual component
        // For now, we'll just validate the target exists
        if self
            .subscribers
            .values()
            .any(|subs| subs.contains(&target.to_string()))
        {
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Target component {target} not found"
            )))
        }
    }

    /// Register an event handler
    pub fn register_handler(&mut self, event_type: &str, handler: Box<dyn EventHandler>) {
        self.event_handlers.insert(event_type.to_string(), handler);
    }

    /// Get event statistics
    #[must_use]
    pub fn get_statistics(&self) -> &EventStatistics {
        &self.event_stats
    }

    /// Get subscribers for an event type
    #[must_use]
    pub fn get_subscribers(&self, event_type: &str) -> Vec<String> {
        self.subscribers
            .get(event_type)
            .cloned()
            .unwrap_or_default()
    }

    /// Clear all events from the queue
    pub fn clear_queue(&mut self) {
        self.event_queue.clear();
    }

    /// Get current queue size
    #[must_use]
    pub fn queue_size(&self) -> usize {
        self.event_queue.len()
    }

    /// Configure event routing
    pub fn configure_routing(&mut self, config: EventRoutingConfig) {
        self.routing_config = config;
    }
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBus")
            .field("subscribers", &self.subscribers)
            .field(
                "event_queue",
                &format!("<{} events>", self.event_queue.len()),
            )
            .field(
                "event_handlers",
                &format!("<{} handlers>", self.event_handlers.len()),
            )
            .field("routing_config", &self.routing_config)
            .field("event_stats", &self.event_stats)
            .finish()
    }
}

/// Component event for inter-component communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentEvent {
    /// Unique event identifier
    pub event_id: String,
    /// Source component identifier
    pub source: String,
    /// Target component (optional, if None broadcasts to all subscribers)
    pub target: Option<String>,
    /// Event type identifier
    pub event_type: String,
    /// Event data payload
    pub data: HashMap<String, String>,
    /// Event timestamp
    pub timestamp: std::time::SystemTime,
    /// Event priority level
    pub priority: EventPriority,
    /// Event metadata
    pub metadata: EventMetadata,
}

impl ComponentEvent {
    /// Create a new component event
    #[must_use]
    pub fn new(source: &str, event_type: &str) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            source: source.to_string(),
            target: None,
            event_type: event_type.to_string(),
            data: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
            priority: EventPriority::Normal,
            metadata: EventMetadata::new(),
        }
    }

    /// Set target component
    #[must_use]
    pub fn with_target(mut self, target: &str) -> Self {
        self.target = Some(target.to_string());
        self
    }

    /// Add data to the event
    #[must_use]
    pub fn with_data(mut self, key: &str, value: &str) -> Self {
        self.data.insert(key.to_string(), value.to_string());
        self
    }

    /// Set event priority
    #[must_use]
    pub fn with_priority(mut self, priority: EventPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Add metadata to the event
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata
            .custom_fields
            .insert(key.to_string(), value.to_string());
        self
    }
}

/// Event priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventPriority {
    /// Low
    Low,
    /// Normal
    Normal,
    /// High
    High,
    /// Critical
    Critical,
    /// Emergency
    Emergency,
}

/// Event metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Event category
    pub category: EventCategory,
    /// Correlation ID for related events
    pub correlation_id: Option<String>,
    /// Event sequence number
    pub sequence_number: Option<u64>,
    /// Custom metadata fields
    pub custom_fields: HashMap<String, String>,
}

impl EventMetadata {
    #[must_use]
    pub fn new() -> Self {
        Self {
            category: EventCategory::General,
            correlation_id: None,
            sequence_number: None,
            custom_fields: HashMap::new(),
        }
    }
}

/// Event categories for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventCategory {
    /// General
    General,
    /// Lifecycle
    Lifecycle,
    /// Error
    Error,
    /// Performance
    Performance,
    /// Security
    Security,
    /// Configuration
    Configuration,
    /// Custom
    Custom(String),
}

/// Event handler trait for processing events
pub trait EventHandler: Send + Sync {
    /// Handle an event for a specific component
    fn handle_event(&self, event: &ComponentEvent, target: &str) -> SklResult<()>;

    /// Get handler identifier
    fn handler_id(&self) -> &str;

    /// Check if handler can process the event type
    fn can_handle(&self, event_type: &str) -> bool;
}

/// Event routing configuration
#[derive(Debug, Clone)]
pub struct EventRoutingConfig {
    /// Enable synchronous event routing
    pub synchronous_routing: bool,
    /// Maximum event queue size
    pub max_queue_size: usize,
    /// Event timeout duration
    pub event_timeout: std::time::Duration,
    /// Enable event persistence
    pub enable_persistence: bool,
    /// Routing rules for specific event types
    pub routing_rules: HashMap<String, RoutingRule>,
}

impl Default for EventRoutingConfig {
    fn default() -> Self {
        Self {
            synchronous_routing: false,
            max_queue_size: 1000,
            event_timeout: std::time::Duration::from_secs(30),
            enable_persistence: false,
            routing_rules: HashMap::new(),
        }
    }
}

/// Routing rule for specific event types
#[derive(Debug, Clone)]
pub struct RoutingRule {
    /// Target components for this event type
    pub targets: Vec<String>,
    /// Routing strategy
    pub strategy: RoutingStrategy,
    /// Whether to require acknowledgment
    pub require_acknowledgment: bool,
    /// Maximum retry attempts
    pub max_retries: u32,
}

/// Routing strategies
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    /// Broadcast to all targets
    Broadcast,
    /// Round-robin among targets
    RoundRobin,
    /// Send to first available target
    FirstAvailable,
    /// Load balance based on target capacity
    LoadBalanced,
}

/// Event processing result
#[derive(Debug, Clone)]
pub struct EventProcessingResult {
    /// Event identifier
    pub event_id: String,
    /// Event type
    pub event_type: String,
    /// Number of successful deliveries
    pub delivery_count: usize,
    /// Number of failed deliveries
    pub failed_deliveries: usize,
    /// Processing time
    pub processing_time: std::time::Duration,
    /// Overall success status
    pub success: bool,
}

/// Event statistics for monitoring
#[derive(Debug, Clone)]
pub struct EventStatistics {
    /// Total events published
    pub total_events_published: u64,
    /// Total subscriptions
    pub total_subscriptions: u64,
    /// Total unsubscriptions
    pub total_unsubscriptions: u64,
    /// Events by type
    pub events_by_type: HashMap<String, u64>,
    /// Total processing time
    pub total_processing_time: std::time::Duration,
    /// Average processing time per event
    pub average_processing_time: std::time::Duration,
}

impl Default for EventStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl EventStatistics {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_events_published: 0,
            total_subscriptions: 0,
            total_unsubscriptions: 0,
            events_by_type: HashMap::new(),
            total_processing_time: std::time::Duration::from_secs(0),
            average_processing_time: std::time::Duration::from_secs(0),
        }
    }

    /// Update average processing time
    pub fn update_average_processing_time(&mut self) {
        if self.total_events_published > 0 {
            self.average_processing_time =
                self.total_processing_time / self.total_events_published as u32;
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EventMetadata {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_bus_creation() {
        let event_bus = EventBus::new();
        assert_eq!(event_bus.queue_size(), 0);
        assert_eq!(event_bus.get_statistics().total_events_published, 0);
    }

    #[test]
    fn test_subscription_management() {
        let mut event_bus = EventBus::new();

        let result = event_bus.subscribe("test_event", "component_1");
        assert!(result.is_ok());

        let subscribers = event_bus.get_subscribers("test_event");
        assert_eq!(subscribers.len(), 1);
        assert!(subscribers.contains(&"component_1".to_string()));

        let result = event_bus.unsubscribe("test_event", "component_1");
        assert!(result.is_ok());

        let subscribers = event_bus.get_subscribers("test_event");
        assert_eq!(subscribers.len(), 0);
    }

    #[test]
    fn test_event_publishing() {
        let mut event_bus = EventBus::new();

        let event = ComponentEvent::new("source_component", "test_event")
            .with_data("key", "value")
            .with_priority(EventPriority::High);

        let result = event_bus.publish(event);
        assert!(result.is_ok());
        assert_eq!(event_bus.queue_size(), 1);
        assert_eq!(event_bus.get_statistics().total_events_published, 1);
    }

    #[test]
    fn test_event_processing() {
        let mut event_bus = EventBus::new();
        event_bus.subscribe("test_event", "component_1").unwrap();

        let event = ComponentEvent::new("source_component", "test_event");
        event_bus.publish(event).unwrap();

        let results = event_bus.process_events().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(event_bus.queue_size(), 0);
    }

    #[test]
    fn test_targeted_events() {
        let mut event_bus = EventBus::new();
        event_bus.subscribe("test_event", "component_1").unwrap();
        event_bus.subscribe("test_event", "component_2").unwrap();

        let event =
            ComponentEvent::new("source_component", "test_event").with_target("component_1");

        event_bus.publish(event).unwrap();
        let results = event_bus.process_events().unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].delivery_count, 1);
    }

    #[test]
    fn test_event_statistics() {
        let mut event_bus = EventBus::new();
        event_bus.subscribe("event_type_1", "component_1").unwrap();
        event_bus.subscribe("event_type_2", "component_2").unwrap();

        let event1 = ComponentEvent::new("source", "event_type_1");
        let event2 = ComponentEvent::new("source", "event_type_1");
        let event3 = ComponentEvent::new("source", "event_type_2");

        event_bus.publish(event1).unwrap();
        event_bus.publish(event2).unwrap();
        event_bus.publish(event3).unwrap();

        let stats = event_bus.get_statistics();
        assert_eq!(stats.total_events_published, 3);
        assert_eq!(stats.total_subscriptions, 2);
        assert_eq!(stats.events_by_type.get("event_type_1"), Some(&2));
        assert_eq!(stats.events_by_type.get("event_type_2"), Some(&1));
    }
}

//! Event-driven execution strategy for reactive systems.

use crate::task_definitions::ExecutionTask;
use sklears_core::error::Result as SklResult;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

use super::core::{StrategyConfig, StrategyMetrics, StrategyState};

/// Delivery guarantee levels
#[derive(Debug, Clone)]
pub enum DeliveryGuarantees {
    /// AtMostOnce
    AtMostOnce,
    /// AtLeastOnce
    AtLeastOnce,
    /// ExactlyOnce
    ExactlyOnce,
}
/// Event for reactive processing
#[derive(Debug, Clone)]
pub struct Event {
    /// Event ID
    pub id: String,
    /// Event type
    pub event_type: String,
    /// Event data
    pub data: EventData,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event source
    pub source: String,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}
/// Event bus for message routing
#[derive(Debug)]
pub struct EventBus {
    /// Subscriptions
    pub subscriptions: HashMap<String, Vec<String>>,
    /// Event history
    pub event_history: VecDeque<Event>,
    /// Bus configuration
    pub config: EventBusConfig,
}
/// Event bus configuration
#[derive(Debug, Clone)]
pub struct EventBusConfig {
    /// Maximum event history size
    pub max_history_size: usize,
    /// Event TTL
    pub event_ttl: Duration,
    /// Enable persistence
    pub persistence: bool,
    /// Delivery guarantees
    pub delivery_guarantees: DeliveryGuarantees,
}
/// Event data types
#[derive(Debug, Clone)]
pub enum EventData {
    /// Task
    Task(Box<ExecutionTask>),
    /// Metric
    Metric(String, f64),
    /// Status
    Status(String, String),
    /// Custom
    Custom(HashMap<String, String>),
}
/// Event-driven execution strategy for reactive systems
pub struct EventDrivenExecutionStrategy {
    /// Strategy configuration
    pub(super) config: StrategyConfig,
    /// Event bus
    pub(super) event_bus: Arc<Mutex<EventBus>>,
    /// Event handlers
    pub(super) handlers: Arc<Mutex<HashMap<String, EventHandler>>>,
    /// Event queue
    pub(super) event_queue: Arc<Mutex<VecDeque<Event>>>,
    /// Execution metrics
    pub(super) metrics: Arc<Mutex<StrategyMetrics>>,
    /// Strategy state
    pub(super) state: Arc<RwLock<StrategyState>>,
}
impl std::fmt::Debug for EventDrivenExecutionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventDrivenExecutionStrategy")
            .field("config", &self.config)
            .field("event_bus", &self.event_bus)
            .field(
                "handlers",
                &format!(
                    "<{} handlers>",
                    self.handlers.lock().map(|h| h.len()).unwrap_or(0)
                ),
            )
            .field("event_queue", &self.event_queue)
            .field("metrics", &self.metrics)
            .field("state", &self.state)
            .finish()
    }
}
/// Event handler for processing events
pub type EventHandler =
    Arc<dyn Fn(Event) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send>> + Send + Sync>;

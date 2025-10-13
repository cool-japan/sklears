//! Event Tracking and Processing
//!
//! This module provides comprehensive event tracking capabilities for the execution monitoring
//! framework. It includes event types, processing, filtering, enrichment, and buffering
//! mechanisms for capturing and analyzing execution-related events.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use std::sync::{Arc, RwLock, Mutex};
use sklears_core::error::{Result as SklResult, SklearsError};
use crate::monitoring_config::*;
use crate::resource_management::ResourceUtilization;

/// Task execution events for tracking
///
/// Represents a significant event that occurs during task execution,
/// providing detailed context and metadata for monitoring and analysis.
#[derive(Debug, Clone)]
pub struct TaskExecutionEvent {
    /// Unique event identifier
    pub event_id: String,

    /// Task identifier this event relates to
    pub task_id: String,

    /// Type of execution event
    pub event_type: TaskEventType,

    /// Event timestamp
    pub timestamp: SystemTime,

    /// Detailed event information
    pub details: TaskEventDetails,

    /// Additional event metadata
    pub metadata: HashMap<String, String>,

    /// Event severity level
    pub severity: SeverityLevel,

    /// Event source information
    pub source: EventSource,

    /// Event correlation ID for tracing
    pub correlation_id: Option<String>,

    /// Event tags for categorization
    pub tags: Vec<String>,
}

/// Task event types
///
/// Enumeration of all possible event types that can occur during task execution.
#[derive(Debug, Clone, PartialEq)]
pub enum TaskEventType {
    /// Task started execution
    TaskStarted,

    /// Task completed successfully
    TaskCompleted,

    /// Task failed with error
    TaskFailed,

    /// Task was cancelled
    TaskCancelled,

    /// Task was paused
    TaskPaused,

    /// Task was resumed
    TaskResumed,

    /// Task was retried
    TaskRetried,

    /// Resource was allocated to task
    ResourceAllocated,

    /// Resource was released from task
    ResourceReleased,

    /// Performance threshold was exceeded
    ThresholdExceeded,

    /// Checkpoint was created
    CheckpointCreated,

    /// State was restored from checkpoint
    StateRestored,

    /// Custom event with specific name
    Custom { event_name: String },
}

/// Task event details
///
/// Contains detailed information about the event including performance data,
/// resource usage, and contextual information.
#[derive(Debug, Clone)]
pub struct TaskEventDetails {
    /// Execution duration (for completion events)
    pub duration: Option<Duration>,

    /// Error message (for failure events)
    pub error_message: Option<String>,

    /// Error code (for failure events)
    pub error_code: Option<String>,

    /// Stack trace (for failure events)
    pub stack_trace: Option<String>,

    /// Resource utilization at event time
    pub resource_utilization: Option<ResourceUtilization>,

    /// Performance metrics at event time
    pub performance_metrics: Vec<EventPerformanceMetric>,

    /// Additional contextual information
    pub context: HashMap<String, String>,

    /// Related events (cause/effect relationships)
    pub related_events: Vec<String>,

    /// Event payload (for custom events)
    pub payload: Option<EventPayload>,
}

/// Event source information
///
/// Identifies the source or component that generated the event.
#[derive(Debug, Clone)]
pub struct EventSource {
    /// Source component name
    pub component: String,

    /// Source instance identifier
    pub instance: Option<String>,

    /// Source host/node information
    pub host: Option<String>,

    /// Source process ID
    pub process_id: Option<u32>,

    /// Source thread ID
    pub thread_id: Option<String>,
}

/// Event-specific performance metrics
///
/// Performance metrics captured at the time of event occurrence.
#[derive(Debug, Clone)]
pub struct EventPerformanceMetric {
    /// Metric name
    pub name: String,

    /// Metric value
    pub value: f64,

    /// Metric unit
    pub unit: String,

    /// Metric timestamp
    pub timestamp: SystemTime,
}

/// Event payload for custom events
///
/// Flexible payload structure for custom events with typed data.
#[derive(Debug, Clone)]
pub enum EventPayload {
    /// Text payload
    Text(String),

    /// JSON payload
    Json(String),

    /// Binary payload
    Binary(Vec<u8>),

    /// Structured payload
    Structured(HashMap<String, EventValue>),
}

/// Event value types for structured payloads
#[derive(Debug, Clone)]
pub enum EventValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<EventValue>),
    Object(HashMap<String, EventValue>),
}

/// Event buffer for collecting and processing events
///
/// Thread-safe buffer that collects events and processes them in batches
/// according to the configured buffer policy.
#[derive(Debug)]
pub struct EventBuffer {
    /// Internal event storage
    events: Arc<Mutex<VecDeque<TaskExecutionEvent>>>,

    /// Buffer configuration
    config: EventBufferConfig,

    /// Buffer statistics
    stats: Arc<RwLock<BufferStats>>,

    /// Last flush time
    last_flush: Arc<RwLock<SystemTime>>,
}

/// Buffer statistics
#[derive(Debug, Clone)]
pub struct BufferStats {
    /// Total events received
    pub total_events: u64,

    /// Events dropped due to overflow
    pub dropped_events: u64,

    /// Events processed successfully
    pub processed_events: u64,

    /// Events failed to process
    pub failed_events: u64,

    /// Current buffer size
    pub current_size: usize,

    /// Buffer high water mark
    pub high_water_mark: usize,

    /// Last flush time
    pub last_flush: SystemTime,

    /// Average processing time
    pub avg_processing_time: Duration,
}

impl EventBuffer {
    /// Create new event buffer with configuration
    pub fn new(config: EventBufferConfig) -> Self {
        Self {
            events: Arc::new(Mutex::new(VecDeque::with_capacity(config.size))),
            config,
            stats: Arc::new(RwLock::new(BufferStats::default())),
            last_flush: Arc::new(RwLock::new(SystemTime::now())),
        }
    }

    /// Add event to buffer
    pub fn add_event(&self, event: TaskExecutionEvent) -> SklResult<()> {
        let mut events = self.events.lock().unwrap();
        let mut stats = self.stats.write().unwrap();

        stats.total_events += 1;

        // Handle buffer overflow
        if events.len() >= self.config.size {
            match self.config.overflow_policy {
                BufferOverflowPolicy::DropNew => {
                    stats.dropped_events += 1;
                    return Ok(());
                }
                BufferOverflowPolicy::DropOld => {
                    events.pop_front();
                    stats.dropped_events += 1;
                }
                BufferOverflowPolicy::Block => {
                    // Block until space is available (simplified implementation)
                    while events.len() >= self.config.size {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
                BufferOverflowPolicy::FlushImmediate => {
                    // Trigger immediate flush (simplified implementation)
                    self.flush_events()?;
                }
            }
        }

        events.push_back(event);
        stats.current_size = events.len();
        stats.high_water_mark = stats.high_water_mark.max(events.len());

        Ok(())
    }

    /// Flush events from buffer
    pub fn flush_events(&self) -> SklResult<Vec<TaskExecutionEvent>> {
        let mut events = self.events.lock().unwrap();
        let mut last_flush = self.last_flush.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let flushed_events: Vec<TaskExecutionEvent> = events.drain(..).collect();
        let count = flushed_events.len();

        stats.processed_events += count as u64;
        stats.current_size = 0;
        *last_flush = SystemTime::now();
        stats.last_flush = *last_flush;

        Ok(flushed_events)
    }

    /// Get buffer statistics
    pub fn get_stats(&self) -> BufferStats {
        self.stats.read().unwrap().clone()
    }

    /// Check if buffer should be flushed
    pub fn should_flush(&self) -> bool {
        let last_flush = *self.last_flush.read().unwrap();
        let time_since_flush = SystemTime::now().duration_since(last_flush).unwrap_or(Duration::ZERO);

        time_since_flush >= self.config.flush_interval ||
        self.events.lock().unwrap().len() >= self.config.size
    }
}

impl Default for BufferStats {
    fn default() -> Self {
        Self {
            total_events: 0,
            dropped_events: 0,
            processed_events: 0,
            failed_events: 0,
            current_size: 0,
            high_water_mark: 0,
            last_flush: SystemTime::now(),
            avg_processing_time: Duration::ZERO,
        }
    }
}

/// Event processor for handling and transforming events
///
/// Processes events according to configured rules including filtering,
/// enrichment, and routing to appropriate handlers.
#[derive(Debug)]
pub struct EventProcessor {
    /// Event filters
    filters: Vec<EventFilter>,

    /// Event enrichers
    enrichers: Vec<EventEnricher>,

    /// Event handlers
    handlers: HashMap<String, Box<dyn EventHandler>>,

    /// Processing statistics
    stats: Arc<RwLock<ProcessingStats>>,
}

/// Event filter for selective event processing
#[derive(Debug, Clone)]
pub struct EventFilter {
    /// Filter name
    pub name: String,

    /// Filter criteria
    pub criteria: FilterCriteria,

    /// Filter action
    pub action: FilterAction,

    /// Filter priority
    pub priority: i32,

    /// Filter enabled status
    pub enabled: bool,
}

/// Event enricher for adding contextual information
#[derive(Debug, Clone)]
pub struct EventEnricher {
    /// Enricher name
    pub name: String,

    /// Target event types
    pub target_events: Vec<TaskEventType>,

    /// Enrichment source
    pub source: EnrichmentSource,

    /// Field mappings
    pub field_mappings: HashMap<String, String>,

    /// Enricher enabled status
    pub enabled: bool,
}

/// Event handler trait for processing events
pub trait EventHandler: Send + Sync {
    /// Handle an event
    fn handle_event(&mut self, event: &TaskExecutionEvent) -> SklResult<()>;

    /// Handle batch of events
    fn handle_batch(&mut self, events: &[TaskExecutionEvent]) -> SklResult<()> {
        for event in events {
            self.handle_event(event)?;
        }
        Ok(())
    }

    /// Get handler name
    fn name(&self) -> &str;

    /// Check if handler can process event type
    fn can_handle(&self, event_type: &TaskEventType) -> bool;
}

/// Processing statistics
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    /// Events processed
    pub events_processed: u64,

    /// Events filtered out
    pub events_filtered: u64,

    /// Events enriched
    pub events_enriched: u64,

    /// Events failed processing
    pub events_failed: u64,

    /// Average processing time per event
    pub avg_processing_time: Duration,

    /// Processing throughput (events per second)
    pub throughput: f64,
}

impl Default for ProcessingStats {
    fn default() -> Self {
        Self {
            events_processed: 0,
            events_filtered: 0,
            events_enriched: 0,
            events_failed: 0,
            avg_processing_time: Duration::ZERO,
            throughput: 0.0,
        }
    }
}

impl EventProcessor {
    /// Create new event processor
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            enrichers: Vec::new(),
            handlers: HashMap::new(),
            stats: Arc::new(RwLock::new(ProcessingStats::default())),
        }
    }

    /// Add event filter
    pub fn add_filter(&mut self, filter: EventFilter) {
        self.filters.push(filter);
        self.filters.sort_by_key(|f| f.priority);
    }

    /// Add event enricher
    pub fn add_enricher(&mut self, enricher: EventEnricher) {
        self.enrichers.push(enricher);
    }

    /// Add event handler
    pub fn add_handler(&mut self, name: String, handler: Box<dyn EventHandler>) {
        self.handlers.insert(name, handler);
    }

    /// Process single event
    pub fn process_event(&mut self, mut event: TaskExecutionEvent) -> SklResult<()> {
        let start_time = SystemTime::now();
        let mut stats = self.stats.write().unwrap();

        // Apply filters
        if !self.apply_filters(&event)? {
            stats.events_filtered += 1;
            return Ok(());
        }

        // Apply enrichment
        self.enrich_event(&mut event)?;
        stats.events_enriched += 1;

        // Route to handlers
        self.route_event(&event)?;

        stats.events_processed += 1;
        let processing_time = SystemTime::now().duration_since(start_time).unwrap_or(Duration::ZERO);
        stats.avg_processing_time = (stats.avg_processing_time * (stats.events_processed - 1) as u32 + processing_time) / stats.events_processed as u32;

        Ok(())
    }

    /// Process batch of events
    pub fn process_batch(&mut self, events: Vec<TaskExecutionEvent>) -> SklResult<()> {
        for event in events {
            if let Err(e) = self.process_event(event) {
                self.stats.write().unwrap().events_failed += 1;
                // Log error but continue processing
                eprintln!("Failed to process event: {}", e);
            }
        }
        Ok(())
    }

    /// Apply filters to event
    fn apply_filters(&self, event: &TaskExecutionEvent) -> SklResult<bool> {
        for filter in &self.filters {
            if !filter.enabled {
                continue;
            }

            if self.matches_criteria(&filter.criteria, event)? {
                match filter.action {
                    FilterAction::Include => return Ok(true),
                    FilterAction::Exclude => return Ok(false),
                    FilterAction::Modify { .. } => {
                        // Apply transformation (simplified)
                        continue;
                    }
                    FilterAction::Route { .. } => {
                        // Route to specific handler (simplified)
                        continue;
                    }
                }
            }
        }
        Ok(true)
    }

    /// Check if event matches filter criteria
    fn matches_criteria(&self, criteria: &FilterCriteria, event: &TaskExecutionEvent) -> SklResult<bool> {
        match criteria {
            FilterCriteria::EventType(event_type) => {
                Ok(std::mem::discriminant(&event.event_type) == std::mem::discriminant(event_type))
            }
            FilterCriteria::Severity(severity) => Ok(event.severity == *severity),
            FilterCriteria::Source(source) => Ok(event.source.component == *source),
            FilterCriteria::TimeRange(range) => Ok(range.contains(event.timestamp)),
            FilterCriteria::Custom(expression) => {
                // Simplified custom filter evaluation
                Ok(true)
            }
            FilterCriteria::Combination { operator, criteria } => {
                match operator {
                    LogicalOperator::And => {
                        for criterion in criteria {
                            if !self.matches_criteria(criterion, event)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    LogicalOperator::Or => {
                        for criterion in criteria {
                            if self.matches_criteria(criterion, event)? {
                                return Ok(true);
                            }
                        }
                        Ok(false)
                    }
                    LogicalOperator::Not => {
                        if criteria.len() == 1 {
                            Ok(!self.matches_criteria(&criteria[0], event)?)
                        } else {
                            Err(SklearsError::InvalidInput("NOT operator requires exactly one criterion".to_string()))
                        }
                    }
                }
            }
        }
    }

    /// Enrich event with additional information
    fn enrich_event(&self, event: &mut TaskExecutionEvent) -> SklResult<()> {
        for enricher in &self.enrichers {
            if !enricher.enabled {
                continue;
            }

            // Check if enricher applies to this event type
            if enricher.target_events.is_empty() || enricher.target_events.contains(&event.event_type) {
                self.apply_enrichment(enricher, event)?;
            }
        }
        Ok(())
    }

    /// Apply specific enrichment to event
    fn apply_enrichment(&self, enricher: &EventEnricher, event: &mut TaskExecutionEvent) -> SklResult<()> {
        match &enricher.source {
            EnrichmentSource::ExecutionContext => {
                // Add execution context information
                event.metadata.insert("enriched_by".to_string(), enricher.name.clone());
            }
            EnrichmentSource::ResourceInfo => {
                // Add resource information
                event.metadata.insert("resource_enriched".to_string(), "true".to_string());
            }
            EnrichmentSource::Environment => {
                // Add environment information
                event.metadata.insert("environment".to_string(), "production".to_string());
            }
            EnrichmentSource::ExternalApi { endpoint, .. } => {
                // Query external API for enrichment (simplified)
                event.metadata.insert("external_enriched".to_string(), endpoint.clone());
            }
            EnrichmentSource::Database { connection, .. } => {
                // Query database for enrichment (simplified)
                event.metadata.insert("db_enriched".to_string(), connection.clone());
            }
        }
        Ok(())
    }

    /// Route event to appropriate handlers
    fn route_event(&mut self, event: &TaskExecutionEvent) -> SklResult<()> {
        for (name, handler) in &mut self.handlers {
            if handler.can_handle(&event.event_type) {
                handler.handle_event(event)?;
            }
        }
        Ok(())
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        self.stats.read().unwrap().clone()
    }
}

impl Default for EventProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Event tracking system for comprehensive event management
///
/// High-level system that combines event buffering, processing, and analytics
/// for complete event lifecycle management.
#[derive(Debug)]
pub struct EventTrackingSystem {
    /// Event buffer
    buffer: EventBuffer,

    /// Event processor
    processor: EventProcessor,

    /// Configuration
    config: EventTrackingConfig,

    /// System statistics
    stats: Arc<RwLock<SystemStats>>,
}

/// System-level statistics
#[derive(Debug, Clone)]
pub struct SystemStats {
    /// System start time
    pub start_time: SystemTime,

    /// System uptime
    pub uptime: Duration,

    /// Total events received
    pub total_events: u64,

    /// Events per second (current rate)
    pub current_rate: f64,

    /// Peak events per second
    pub peak_rate: f64,

    /// System health score
    pub health_score: f64,
}

impl Default for SystemStats {
    fn default() -> Self {
        Self {
            start_time: SystemTime::now(),
            uptime: Duration::ZERO,
            total_events: 0,
            current_rate: 0.0,
            peak_rate: 0.0,
            health_score: 1.0,
        }
    }
}

impl EventTrackingSystem {
    /// Create new event tracking system
    pub fn new(config: EventTrackingConfig) -> Self {
        Self {
            buffer: EventBuffer::new(config.buffer.clone()),
            processor: EventProcessor::new(),
            config,
            stats: Arc::new(RwLock::new(SystemStats::default())),
        }
    }

    /// Initialize session with configuration
    pub fn initialize_session(&mut self, session_id: &str, config: &EventTrackingConfig) -> SklResult<()> {
        // Initialize session-specific configuration
        Ok(())
    }

    /// Record event
    pub fn record_event(&mut self, event: TaskExecutionEvent) -> SklResult<()> {
        self.buffer.add_event(event)?;
        self.stats.write().unwrap().total_events += 1;

        // Process events if buffer should be flushed
        if self.buffer.should_flush() {
            self.flush_and_process()?;
        }

        Ok(())
    }

    /// Flush and process buffered events
    pub fn flush_and_process(&mut self) -> SklResult<()> {
        let events = self.buffer.flush_events()?;
        self.processor.process_batch(events)?;
        Ok(())
    }

    /// Get recent events for a session
    pub fn get_recent_events(&self, session_id: &str, count: usize) -> SklResult<Vec<TaskExecutionEvent>> {
        // In a real implementation, this would query stored events for the session
        Ok(Vec::new())
    }

    /// Finalize session
    pub fn finalize_session(&mut self, session_id: &str) -> SklResult<Vec<TaskExecutionEvent>> {
        // Flush any remaining events and return final event list
        self.flush_and_process()?;
        self.get_recent_events(session_id, 1000)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: EventTrackingConfig) -> SklResult<()> {
        self.config = config;
        Ok(())
    }

    /// Get system health status
    pub fn get_health_status(&self) -> SklResult<ComponentHealth> {
        let buffer_stats = self.buffer.get_stats();
        let processing_stats = self.processor.get_stats();

        // Calculate health score based on various factors
        let mut health_score = 1.0;
        let mut issues = Vec::new();

        // Check buffer health
        if buffer_stats.dropped_events > 0 {
            health_score -= 0.2;
            issues.push("Buffer overflow detected".to_string());
        }

        // Check processing health
        if processing_stats.events_failed > 0 {
            health_score -= 0.3;
            issues.push("Event processing failures detected".to_string());
        }

        let status = if health_score >= 0.8 {
            crate::monitoring_core::HealthStatus::Healthy
        } else if health_score >= 0.5 {
            crate::monitoring_core::HealthStatus::Warning
        } else {
            crate::monitoring_core::HealthStatus::Critical
        };

        Ok(crate::monitoring_core::ComponentHealth {
            component: "event_tracking".to_string(),
            status,
            score: health_score,
            last_check: SystemTime::now(),
            issues,
        })
    }

    /// Get system statistics
    pub fn get_system_stats(&self) -> SystemStats {
        let mut stats = self.stats.write().unwrap();
        stats.uptime = SystemTime::now().duration_since(stats.start_time).unwrap_or(Duration::ZERO);
        stats.clone()
    }
}

// Event creation utilities

impl TaskExecutionEvent {
    /// Create new task started event
    pub fn task_started(task_id: String, source: EventSource) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            task_id,
            event_type: TaskEventType::TaskStarted,
            timestamp: SystemTime::now(),
            details: TaskEventDetails::default(),
            metadata: HashMap::new(),
            severity: SeverityLevel::Info,
            source,
            correlation_id: None,
            tags: Vec::new(),
        }
    }

    /// Create new task completed event
    pub fn task_completed(task_id: String, duration: Duration, source: EventSource) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            task_id,
            event_type: TaskEventType::TaskCompleted,
            timestamp: SystemTime::now(),
            details: TaskEventDetails {
                duration: Some(duration),
                ..Default::default()
            },
            metadata: HashMap::new(),
            severity: SeverityLevel::Info,
            source,
            correlation_id: None,
            tags: Vec::new(),
        }
    }

    /// Create new task failed event
    pub fn task_failed(task_id: String, error_message: String, source: EventSource) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            task_id,
            event_type: TaskEventType::TaskFailed,
            timestamp: SystemTime::now(),
            details: TaskEventDetails {
                error_message: Some(error_message),
                ..Default::default()
            },
            metadata: HashMap::new(),
            severity: SeverityLevel::Error,
            source,
            correlation_id: None,
            tags: Vec::new(),
        }
    }

    /// Create custom event
    pub fn custom(task_id: String, event_name: String, source: EventSource) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            task_id,
            event_type: TaskEventType::Custom { event_name },
            timestamp: SystemTime::now(),
            details: TaskEventDetails::default(),
            metadata: HashMap::new(),
            severity: SeverityLevel::Info,
            source,
            correlation_id: None,
            tags: Vec::new(),
        }
    }

    /// Add metadata to event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set severity level
    pub fn with_severity(mut self, severity: SeverityLevel) -> Self {
        self.severity = severity;
        self
    }

    /// Add correlation ID
    pub fn with_correlation_id(mut self, correlation_id: String) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

impl Default for TaskEventDetails {
    fn default() -> Self {
        Self {
            duration: None,
            error_message: None,
            error_code: None,
            stack_trace: None,
            resource_utilization: None,
            performance_metrics: Vec::new(),
            context: HashMap::new(),
            related_events: Vec::new(),
            payload: None,
        }
    }
}

impl Default for EventSource {
    fn default() -> Self {
        Self {
            component: "unknown".to_string(),
            instance: None,
            host: None,
            process_id: Some(std::process::id()),
            thread_id: None,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let source = EventSource::default();
        let event = TaskExecutionEvent::task_started("task_1".to_string(), source);

        assert_eq!(event.task_id, "task_1");
        assert!(matches!(event.event_type, TaskEventType::TaskStarted));
        assert_eq!(event.severity, SeverityLevel::Info);
    }

    #[test]
    fn test_event_buffer() {
        let config = EventBufferConfig {
            size: 10,
            flush_interval: Duration::from_secs(1),
            overflow_policy: BufferOverflowPolicy::DropOld,
            batch_processing: BatchProcessingConfig::default(),
        };

        let buffer = EventBuffer::new(config);
        let source = EventSource::default();

        // Add events
        for i in 0..5 {
            let event = TaskExecutionEvent::task_started(format!("task_{}", i), source.clone());
            buffer.add_event(event).unwrap();
        }

        let stats = buffer.get_stats();
        assert_eq!(stats.total_events, 5);
        assert_eq!(stats.current_size, 5);
    }

    #[test]
    fn test_event_processor() {
        let mut processor = EventProcessor::new();
        let source = EventSource::default();
        let event = TaskExecutionEvent::task_started("task_1".to_string(), source);

        processor.process_event(event).unwrap();

        let stats = processor.get_stats();
        assert_eq!(stats.events_processed, 1);
    }

    #[test]
    fn test_event_tracking_system() {
        let config = EventTrackingConfig::default();
        let mut system = EventTrackingSystem::new(config);
        let source = EventSource::default();

        let event = TaskExecutionEvent::task_started("task_1".to_string(), source);
        system.record_event(event).unwrap();

        let stats = system.get_system_stats();
        assert_eq!(stats.total_events, 1);
    }

    #[test]
    fn test_event_filtering() {
        let processor = EventProcessor::new();
        let source = EventSource::default();
        let event = TaskExecutionEvent::task_started("task_1".to_string(), source);

        let criteria = FilterCriteria::EventType(TaskEventType::TaskStarted);
        assert!(processor.matches_criteria(&criteria, &event).unwrap());

        let criteria = FilterCriteria::EventType(TaskEventType::TaskCompleted);
        assert!(!processor.matches_criteria(&criteria, &event).unwrap());
    }

    #[test]
    fn test_event_builder_pattern() {
        let source = EventSource::default();
        let event = TaskExecutionEvent::task_started("task_1".to_string(), source)
            .with_metadata("key".to_string(), "value".to_string())
            .with_severity(SeverityLevel::Critical)
            .with_correlation_id("corr_123".to_string());

        assert_eq!(event.metadata.get("key").unwrap(), "value");
        assert_eq!(event.severity, SeverityLevel::Critical);
        assert_eq!(event.correlation_id.unwrap(), "corr_123");
    }
}
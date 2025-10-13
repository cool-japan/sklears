//! Event Tracking and Processing
//!
//! This module provides comprehensive event tracking capabilities for monitoring
//! task execution, lifecycle events, and system behaviors. It includes event
//! buffering, filtering, enrichment, and processing functionality.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, Instant};
use std::fmt;

use crate::monitoring_core::TimeRange;
use crate::metrics_collection::PerformanceMetric;
use crate::resource_management::ResourceUtilization;
use crate::configuration_management::EventTrackingConfig;

/// Task execution event with comprehensive details
///
/// Represents a single event in the task execution lifecycle with detailed
/// information for tracking, analysis, and debugging.
#[derive(Debug, Clone)]
pub struct TaskExecutionEvent {
    /// Unique event identifier
    pub event_id: String,

    /// Associated task identifier
    pub task_id: String,

    /// Type of event
    pub event_type: TaskEventType,

    /// Event timestamp
    pub timestamp: SystemTime,

    /// Detailed event information
    pub details: TaskEventDetails,

    /// Event metadata and context
    pub metadata: HashMap<String, String>,

    /// Event source information
    pub source: EventSource,

    /// Event severity level
    pub severity: EventSeverity,

    /// Event tags for categorization
    pub tags: Vec<String>,

    /// Event correlation ID for tracking related events
    pub correlation_id: Option<String>,

    /// Parent event ID for hierarchical tracking
    pub parent_event_id: Option<String>,

    /// Event sequence number within session
    pub sequence_number: u64,
}

impl TaskExecutionEvent {
    /// Create a new task execution event
    pub fn new(event_id: String, task_id: String, event_type: TaskEventType) -> Self {
        Self {
            event_id,
            task_id,
            event_type,
            timestamp: SystemTime::now(),
            details: TaskEventDetails::default(),
            metadata: HashMap::new(),
            source: EventSource::System,
            severity: EventSeverity::Info,
            tags: Vec::new(),
            correlation_id: None,
            parent_event_id: None,
            sequence_number: 0,
        }
    }

    /// Builder pattern for event creation
    pub fn builder() -> EventBuilder {
        EventBuilder::new()
    }

    /// Add metadata to event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set event severity
    pub fn with_severity(mut self, severity: EventSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Add tag to event
    pub fn with_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    /// Set correlation ID
    pub fn with_correlation_id(mut self, correlation_id: String) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    /// Set parent event ID
    pub fn with_parent(mut self, parent_event_id: String) -> Self {
        self.parent_event_id = Some(parent_event_id);
        self
    }

    /// Check if event matches filters
    pub fn matches_filter(&self, filter: &EventFilter) -> bool {
        // Check event type filter
        if !filter.event_types.is_empty() && !filter.event_types.contains(&self.event_type) {
            return false;
        }

        // Check severity filter
        if let Some(min_severity) = &filter.min_severity {
            if self.severity < *min_severity {
                return false;
            }
        }

        // Check tag filters
        if !filter.required_tags.is_empty() {
            if !filter.required_tags.iter().all(|tag| self.tags.contains(tag)) {
                return false;
            }
        }

        // Check metadata filters
        for (key, value) in &filter.metadata_filters {
            if self.metadata.get(key) != Some(value) {
                return false;
            }
        }

        // Check time range
        if let Some(time_range) = &filter.time_range {
            if !time_range.contains(self.timestamp) {
                return false;
            }
        }

        true
    }

    /// Get event age from current time
    pub fn age(&self) -> Duration {
        SystemTime::now().duration_since(self.timestamp).unwrap_or(Duration::from_secs(0))
    }

    /// Check if event is an error event
    pub fn is_error(&self) -> bool {
        matches!(self.event_type, TaskEventType::TaskFailed | TaskEventType::TaskError) ||
        matches!(self.severity, EventSeverity::Error | EventSeverity::Critical)
    }

    /// Get event duration if applicable
    pub fn duration(&self) -> Option<Duration> {
        self.details.duration
    }

    /// Generate unique event signature for deduplication
    pub fn signature(&self) -> String {
        format!("{}:{}:{:?}", self.task_id, self.event_type, self.timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap_or(Duration::from_secs(0)))
    }
}

/// Event builder for fluent construction
#[derive(Debug)]
pub struct EventBuilder {
    event_id: Option<String>,
    task_id: Option<String>,
    event_type: Option<TaskEventType>,
    timestamp: Option<SystemTime>,
    details: TaskEventDetails,
    metadata: HashMap<String, String>,
    source: EventSource,
    severity: EventSeverity,
    tags: Vec<String>,
    correlation_id: Option<String>,
    parent_event_id: Option<String>,
}

impl EventBuilder {
    /// Create new event builder
    pub fn new() -> Self {
        Self {
            event_id: None,
            task_id: None,
            event_type: None,
            timestamp: None,
            details: TaskEventDetails::default(),
            metadata: HashMap::new(),
            source: EventSource::System,
            severity: EventSeverity::Info,
            tags: Vec::new(),
            correlation_id: None,
            parent_event_id: None,
        }
    }

    /// Set event ID
    pub fn event_id(mut self, event_id: String) -> Self {
        self.event_id = Some(event_id);
        self
    }

    /// Set task ID
    pub fn task_id(mut self, task_id: String) -> Self {
        self.task_id = Some(task_id);
        self
    }

    /// Set event type
    pub fn event_type(mut self, event_type: TaskEventType) -> Self {
        self.event_type = Some(event_type);
        self
    }

    /// Set timestamp
    pub fn timestamp(mut self, timestamp: SystemTime) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Set duration
    pub fn duration(mut self, duration: Duration) -> Self {
        self.details.duration = Some(duration);
        self
    }

    /// Set error message
    pub fn error_message(mut self, error_message: String) -> Self {
        self.details.error_message = Some(error_message);
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set source
    pub fn source(mut self, source: EventSource) -> Self {
        self.source = source;
        self
    }

    /// Set severity
    pub fn severity(mut self, severity: EventSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Add tag
    pub fn tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    /// Build the event
    pub fn build(self) -> SklResult<TaskExecutionEvent> {
        let event_id = self.event_id.ok_or_else(|| SklearsError::InvalidInput("Event ID is required".to_string()))?;
        let task_id = self.task_id.ok_or_else(|| SklearsError::InvalidInput("Task ID is required".to_string()))?;
        let event_type = self.event_type.ok_or_else(|| SklearsError::InvalidInput("Event type is required".to_string()))?;

        Ok(TaskExecutionEvent {
            event_id,
            task_id,
            event_type,
            timestamp: self.timestamp.unwrap_or_else(SystemTime::now),
            details: self.details,
            metadata: self.metadata,
            source: self.source,
            severity: self.severity,
            tags: self.tags,
            correlation_id: self.correlation_id,
            parent_event_id: self.parent_event_id,
            sequence_number: 0,
        })
    }
}

impl Default for EventBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of task execution events
#[derive(Debug, Clone, PartialEq)]
pub enum TaskEventType {
    /// Task has been created
    TaskCreated,

    /// Task execution started
    TaskStarted,

    /// Task execution completed successfully
    TaskCompleted,

    /// Task execution failed
    TaskFailed,

    /// Task was cancelled
    TaskCancelled,

    /// Task was paused
    TaskPaused,

    /// Task was resumed
    TaskResumed,

    /// Task progress update
    TaskProgress,

    /// Task error occurred (non-fatal)
    TaskError,

    /// Task warning issued
    TaskWarning,

    /// Task resource allocation
    TaskResourceAllocated,

    /// Task resource deallocation
    TaskResourceDeallocated,

    /// Task checkpoint created
    TaskCheckpoint,

    /// Task retried
    TaskRetried,

    /// Task timeout occurred
    TaskTimeout,

    /// Custom event type
    Custom { name: String },
}

impl TaskEventType {
    /// Check if event type indicates completion
    pub fn is_completion(&self) -> bool {
        matches!(self, TaskEventType::TaskCompleted | TaskEventType::TaskFailed | TaskEventType::TaskCancelled)
    }

    /// Check if event type indicates error
    pub fn is_error(&self) -> bool {
        matches!(self, TaskEventType::TaskFailed | TaskEventType::TaskError | TaskEventType::TaskTimeout)
    }

    /// Get event type priority for ordering
    pub fn priority(&self) -> u32 {
        match self {
            TaskEventType::TaskCreated => 1,
            TaskEventType::TaskStarted => 2,
            TaskEventType::TaskProgress => 3,
            TaskEventType::TaskCompleted => 10,
            TaskEventType::TaskFailed => 11,
            TaskEventType::TaskCancelled => 12,
            TaskEventType::TaskError => 15,
            TaskEventType::TaskWarning => 5,
            _ => 7,
        }
    }
}

/// Event source information
#[derive(Debug, Clone, PartialEq)]
pub enum EventSource {
    /// System-generated event
    System,

    /// Application-generated event
    Application,

    /// User-generated event
    User,

    /// External system event
    External { system_name: String },

    /// Monitor-generated event
    Monitor,

    /// Framework-generated event
    Framework,
}

/// Event severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum EventSeverity {
    /// Debug information
    Debug,

    /// Informational event
    Info,

    /// Warning event
    Warning,

    /// Error event
    Error,

    /// Critical event
    Critical,
}

/// Detailed event information
#[derive(Debug, Clone)]
pub struct TaskEventDetails {
    /// Event duration (for timed events)
    pub duration: Option<Duration>,

    /// Error message (for error events)
    pub error_message: Option<String>,

    /// Resource utilization snapshot
    pub resource_utilization: Option<ResourceUtilization>,

    /// Associated performance metrics
    pub performance_metrics: Vec<PerformanceMetric>,

    /// Event context information
    pub context: HashMap<String, String>,

    /// Stack trace (for error events)
    pub stack_trace: Option<String>,

    /// Progress information (for progress events)
    pub progress: Option<ProgressInfo>,

    /// Retry information (for retry events)
    pub retry_info: Option<RetryInfo>,
}

impl Default for TaskEventDetails {
    fn default() -> Self {
        Self {
            duration: None,
            error_message: None,
            resource_utilization: None,
            performance_metrics: Vec::new(),
            context: HashMap::new(),
            stack_trace: None,
            progress: None,
            retry_info: None,
        }
    }
}

/// Progress information for progress events
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Current progress (0.0 to 1.0)
    pub percentage: f64,

    /// Current step
    pub current_step: u32,

    /// Total steps
    pub total_steps: u32,

    /// Progress message
    pub message: Option<String>,

    /// Estimated time remaining
    pub eta: Option<Duration>,
}

/// Retry information for retry events
#[derive(Debug, Clone)]
pub struct RetryInfo {
    /// Current retry attempt
    pub attempt: u32,

    /// Maximum retry attempts
    pub max_attempts: u32,

    /// Retry delay
    pub delay: Duration,

    /// Retry reason
    pub reason: String,

    /// Backoff strategy
    pub backoff_strategy: BackoffStrategy,
}

/// Backoff strategies for retries
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed,

    /// Exponential backoff
    Exponential { multiplier: f64 },

    /// Linear backoff
    Linear { increment: Duration },

    /// Custom backoff
    Custom { strategy_name: String },
}

/// Event buffer for managing event streams
#[derive(Debug)]
pub struct EventBuffer {
    /// Buffered events
    events: VecDeque<TaskExecutionEvent>,

    /// Buffer configuration
    config: EventBufferConfig,

    /// Buffer statistics
    stats: BufferStatistics,

    /// Thread safety lock
    lock: Arc<Mutex<()>>,
}

impl EventBuffer {
    /// Create new event buffer
    pub fn new(config: EventBufferConfig) -> Self {
        Self {
            events: VecDeque::with_capacity(config.size),
            config,
            stats: BufferStatistics::new(),
            lock: Arc::new(Mutex::new(())),
        }
    }

    /// Add event to buffer
    pub fn add_event(&mut self, event: TaskExecutionEvent) -> SklResult<()> {
        let _lock = self.lock.lock().unwrap();

        // Check buffer overflow
        if self.events.len() >= self.config.size {
            match self.config.overflow_policy {
                BufferOverflowPolicy::DropNew => {
                    self.stats.events_dropped += 1;
                    return Ok(());
                }
                BufferOverflowPolicy::DropOld => {
                    if let Some(dropped) = self.events.pop_front() {
                        self.stats.events_dropped += 1;
                        log::debug!("Dropped old event: {}", dropped.event_id);
                    }
                }
                BufferOverflowPolicy::Block => {
                    return Err(SklearsError::ResourceExhausted("Event buffer is full".to_string()));
                }
                BufferOverflowPolicy::FlushImmediate => {
                    // Trigger immediate flush (implementation would depend on processor)
                    log::warn!("Buffer overflow, immediate flush required");
                }
            }
        }

        self.events.push_back(event);
        self.stats.events_added += 1;
        self.stats.last_update = SystemTime::now();

        Ok(())
    }

    /// Get events from buffer
    pub fn get_events(&mut self, count: usize) -> Vec<TaskExecutionEvent> {
        let _lock = self.lock.lock().unwrap();
        let mut events = Vec::new();

        for _ in 0..count.min(self.events.len()) {
            if let Some(event) = self.events.pop_front() {
                events.push(event);
                self.stats.events_retrieved += 1;
            }
        }

        events
    }

    /// Drain all events from buffer
    pub fn drain_events(&mut self) -> Vec<TaskExecutionEvent> {
        let _lock = self.lock.lock().unwrap();
        let events: Vec<TaskExecutionEvent> = self.events.drain(..).collect();
        self.stats.events_retrieved += events.len() as u64;
        events
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.events.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.events.len() >= self.config.size
    }

    /// Get buffer statistics
    pub fn statistics(&self) -> &BufferStatistics {
        &self.stats
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        let _lock = self.lock.lock().unwrap();
        let dropped_count = self.events.len();
        self.events.clear();
        self.stats.events_dropped += dropped_count as u64;
    }
}

/// Event buffer configuration
#[derive(Debug, Clone)]
pub struct EventBufferConfig {
    /// Maximum buffer size (number of events)
    pub size: usize,

    /// Buffer flush interval
    pub flush_interval: Duration,

    /// Buffer overflow policy
    pub overflow_policy: BufferOverflowPolicy,

    /// Batch processing configuration
    pub batch_processing: BatchProcessingConfig,

    /// Enable event deduplication
    pub enable_deduplication: bool,

    /// Deduplication window
    pub deduplication_window: Duration,
}

impl Default for EventBufferConfig {
    fn default() -> Self {
        Self {
            size: 10000,
            flush_interval: Duration::from_secs(10),
            overflow_policy: BufferOverflowPolicy::DropOld,
            batch_processing: BatchProcessingConfig::default(),
            enable_deduplication: false,
            deduplication_window: Duration::from_secs(60),
        }
    }
}

/// Buffer overflow policies
#[derive(Debug, Clone, PartialEq)]
pub enum BufferOverflowPolicy {
    /// Drop new incoming events
    DropNew,

    /// Drop oldest events
    DropOld,

    /// Block until buffer space is available
    Block,

    /// Flush buffer immediately when full
    FlushImmediate,
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchProcessingConfig {
    /// Batch size
    pub batch_size: usize,

    /// Batch timeout
    pub timeout: Duration,

    /// Enable parallel processing
    pub parallel: bool,

    /// Error handling strategy
    pub error_handling: BatchErrorHandling,

    /// Maximum concurrent batches
    pub max_concurrent_batches: usize,
}

impl Default for BatchProcessingConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            timeout: Duration::from_secs(5),
            parallel: true,
            error_handling: BatchErrorHandling::Continue,
            max_concurrent_batches: 4,
        }
    }
}

/// Batch error handling strategies
#[derive(Debug, Clone, PartialEq)]
pub enum BatchErrorHandling {
    /// Continue processing other batches
    Continue,

    /// Stop batch processing on error
    Stop,

    /// Retry failed batches
    Retry { max_attempts: usize },

    /// Dead letter queue for failed events
    DeadLetter,
}

/// Buffer statistics
#[derive(Debug, Clone)]
pub struct BufferStatistics {
    /// Number of events added
    pub events_added: u64,

    /// Number of events retrieved
    pub events_retrieved: u64,

    /// Number of events dropped
    pub events_dropped: u64,

    /// Last update timestamp
    pub last_update: SystemTime,

    /// Average event size (bytes)
    pub avg_event_size: usize,

    /// Buffer utilization (0.0 to 1.0)
    pub utilization: f64,
}

impl BufferStatistics {
    fn new() -> Self {
        Self {
            events_added: 0,
            events_retrieved: 0,
            events_dropped: 0,
            last_update: SystemTime::now(),
            avg_event_size: 0,
            utilization: 0.0,
        }
    }

    /// Calculate drop rate
    pub fn drop_rate(&self) -> f64 {
        let total = self.events_added + self.events_dropped;
        if total > 0 {
            self.events_dropped as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Event filter for selecting events
#[derive(Debug, Clone)]
pub struct EventFilter {
    /// Event types to include (empty = all)
    pub event_types: Vec<TaskEventType>,

    /// Minimum severity level
    pub min_severity: Option<EventSeverity>,

    /// Required tags
    pub required_tags: Vec<String>,

    /// Metadata filters
    pub metadata_filters: HashMap<String, String>,

    /// Time range filter
    pub time_range: Option<TimeRange>,

    /// Task ID filter
    pub task_ids: Vec<String>,

    /// Correlation ID filter
    pub correlation_ids: Vec<String>,

    /// Maximum events to return
    pub limit: Option<usize>,
}

impl Default for EventFilter {
    fn default() -> Self {
        Self {
            event_types: Vec::new(),
            min_severity: None,
            required_tags: Vec::new(),
            metadata_filters: HashMap::new(),
            time_range: None,
            task_ids: Vec::new(),
            correlation_ids: Vec::new(),
            limit: None,
        }
    }
}

/// Event processor for handling event streams
#[derive(Debug)]
pub struct EventProcessor {
    /// Processing configuration
    config: EventProcessingConfig,

    /// Registered event handlers
    handlers: Vec<Box<dyn EventHandler>>,

    /// Event enrichers
    enrichers: Vec<Box<dyn EventEnricher>>,

    /// Processing statistics
    stats: ProcessingStatistics,
}

impl EventProcessor {
    /// Create new event processor
    pub fn new(config: EventProcessingConfig) -> Self {
        Self {
            config,
            handlers: Vec::new(),
            enrichers: Vec::new(),
            stats: ProcessingStatistics::new(),
        }
    }

    /// Register event handler
    pub fn register_handler(&mut self, handler: Box<dyn EventHandler>) {
        self.handlers.push(handler);
    }

    /// Register event enricher
    pub fn register_enricher(&mut self, enricher: Box<dyn EventEnricher>) {
        self.enrichers.push(enricher);
    }

    /// Process events from buffer
    pub fn process_events(&mut self, events: Vec<TaskExecutionEvent>) -> SklResult<()> {
        let start_time = Instant::now();

        for mut event in events {
            // Enrich event
            for enricher in &self.enrichers {
                if let Err(e) = enricher.enrich(&mut event) {
                    log::warn!("Event enrichment failed: {}", e);
                    self.stats.enrichment_failures += 1;
                }
            }

            // Process event with handlers
            for handler in &self.handlers {
                match handler.handle(&event) {
                    Ok(_) => self.stats.events_processed += 1,
                    Err(e) => {
                        log::error!("Event processing failed: {}", e);
                        self.stats.processing_failures += 1;
                    }
                }
            }
        }

        let processing_time = start_time.elapsed();
        self.stats.total_processing_time += processing_time;
        self.stats.last_processing = SystemTime::now();

        Ok(())
    }

    /// Get processing statistics
    pub fn statistics(&self) -> &ProcessingStatistics {
        &self.stats
    }
}

/// Event processing configuration
#[derive(Debug, Clone)]
pub struct EventProcessingConfig {
    /// Enable parallel processing
    pub parallel_processing: bool,

    /// Maximum processing threads
    pub max_threads: usize,

    /// Processing timeout
    pub processing_timeout: Duration,

    /// Enable event ordering
    pub preserve_order: bool,

    /// Maximum processing queue size
    pub max_queue_size: usize,
}

impl Default for EventProcessingConfig {
    fn default() -> Self {
        Self {
            parallel_processing: true,
            max_threads: 4,
            processing_timeout: Duration::from_secs(30),
            preserve_order: false,
            max_queue_size: 10000,
        }
    }
}

/// Event handler trait
pub trait EventHandler: Send + Sync {
    /// Handle an event
    fn handle(&self, event: &TaskExecutionEvent) -> SklResult<()>;

    /// Get handler name
    fn name(&self) -> &str;

    /// Check if handler can handle event type
    fn can_handle(&self, event_type: &TaskEventType) -> bool;
}

/// Event enricher trait
pub trait EventEnricher: Send + Sync {
    /// Enrich an event with additional information
    fn enrich(&self, event: &mut TaskExecutionEvent) -> SklResult<()>;

    /// Get enricher name
    fn name(&self) -> &str;
}

/// Processing statistics
#[derive(Debug, Clone)]
pub struct ProcessingStatistics {
    /// Number of events processed successfully
    pub events_processed: u64,

    /// Number of processing failures
    pub processing_failures: u64,

    /// Number of enrichment failures
    pub enrichment_failures: u64,

    /// Total processing time
    pub total_processing_time: Duration,

    /// Last processing timestamp
    pub last_processing: SystemTime,

    /// Average processing time per event
    pub avg_processing_time: Duration,
}

impl ProcessingStatistics {
    fn new() -> Self {
        Self {
            events_processed: 0,
            processing_failures: 0,
            enrichment_failures: 0,
            total_processing_time: Duration::from_millis(0),
            last_processing: SystemTime::now(),
            avg_processing_time: Duration::from_millis(0),
        }
    }

    /// Calculate processing success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.events_processed + self.processing_failures;
        if total > 0 {
            self.events_processed as f64 / total as f64
        } else {
            1.0
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_execution_event_creation() {
        let event = TaskExecutionEvent::new(
            "event_1".to_string(),
            "task_1".to_string(),
            TaskEventType::TaskStarted,
        );

        assert_eq!(event.event_id, "event_1");
        assert_eq!(event.task_id, "task_1");
        assert_eq!(event.event_type, TaskEventType::TaskStarted);
        assert!(!event.is_error());
    }

    #[test]
    fn test_event_builder() {
        let event = TaskExecutionEvent::builder()
            .event_id("event_1".to_string())
            .task_id("task_1".to_string())
            .event_type(TaskEventType::TaskCompleted)
            .duration(Duration::from_millis(1500))
            .severity(EventSeverity::Info)
            .tag("component".to_string())
            .metadata("key".to_string(), "value".to_string())
            .build()
            .unwrap();

        assert_eq!(event.event_id, "event_1");
        assert_eq!(event.event_type, TaskEventType::TaskCompleted);
        assert_eq!(event.duration(), Some(Duration::from_millis(1500)));
        assert!(event.tags.contains(&"component".to_string()));
    }

    #[test]
    fn test_event_buffer() {
        let config = EventBufferConfig {
            size: 3,
            overflow_policy: BufferOverflowPolicy::DropOld,
            ..Default::default()
        };

        let mut buffer = EventBuffer::new(config);

        // Add events
        for i in 0..5 {
            let event = TaskExecutionEvent::new(
                format!("event_{}", i),
                "task_1".to_string(),
                TaskEventType::TaskProgress,
            );
            buffer.add_event(event).unwrap();
        }

        // Should have 3 events (dropped 2 oldest)
        assert_eq!(buffer.size(), 3);
        assert_eq!(buffer.statistics().events_dropped, 2);

        // Get events
        let events = buffer.get_events(2);
        assert_eq!(events.len(), 2);
        assert_eq!(buffer.size(), 1);
    }

    #[test]
    fn test_event_filter() {
        let event = TaskExecutionEvent::new(
            "event_1".to_string(),
            "task_1".to_string(),
            TaskEventType::TaskError,
        )
        .with_severity(EventSeverity::Error)
        .with_tag("critical".to_string());

        let filter = EventFilter {
            event_types: vec![TaskEventType::TaskError],
            min_severity: Some(EventSeverity::Warning),
            required_tags: vec!["critical".to_string()],
            ..Default::default()
        };

        assert!(event.matches_filter(&filter));

        let filter_no_match = EventFilter {
            event_types: vec![TaskEventType::TaskCompleted],
            ..Default::default()
        };

        assert!(!event.matches_filter(&filter_no_match));
    }

    #[test]
    fn test_event_type_properties() {
        assert!(TaskEventType::TaskCompleted.is_completion());
        assert!(TaskEventType::TaskFailed.is_completion());
        assert!(TaskEventType::TaskFailed.is_error());
        assert!(!TaskEventType::TaskStarted.is_completion());

        assert_eq!(TaskEventType::TaskCompleted.priority(), 10);
        assert_eq!(TaskEventType::TaskStarted.priority(), 2);
    }

    #[test]
    fn test_event_severity_ordering() {
        assert!(EventSeverity::Critical > EventSeverity::Error);
        assert!(EventSeverity::Error > EventSeverity::Warning);
        assert!(EventSeverity::Warning > EventSeverity::Info);
        assert!(EventSeverity::Info > EventSeverity::Debug);
    }

    #[test]
    fn test_progress_info() {
        let progress = ProgressInfo {
            percentage: 0.75,
            current_step: 3,
            total_steps: 4,
            message: Some("Processing data".to_string()),
            eta: Some(Duration::from_secs(60)),
        };

        assert_eq!(progress.percentage, 0.75);
        assert_eq!(progress.current_step, 3);
        assert_eq!(progress.total_steps, 4);
    }

    #[test]
    fn test_buffer_statistics() {
        let mut stats = BufferStatistics::new();
        stats.events_added = 100;
        stats.events_dropped = 10;

        assert_eq!(stats.drop_rate(), 0.09090909090909091); // 10/110
    }
}
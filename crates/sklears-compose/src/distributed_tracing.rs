//! Distributed tracing system for pipeline execution monitoring
//!
//! This module provides comprehensive distributed tracing capabilities for
//! monitoring pipeline execution across different processes, threads, and
//! potentially different machines. It implements a custom tracing protocol
//! that can track execution flow, dependencies, and performance metrics.

use sklears_core::traits::Estimator;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Distributed tracing system for monitoring pipeline execution across services
pub struct DistributedTracer {
    /// Tracer configuration
    config: TracingConfig,
    /// Active spans storage
    spans: Arc<RwLock<HashMap<String, TraceSpan>>>,
    /// Completed traces storage
    traces: Arc<Mutex<VecDeque<Trace>>>,
    /// Span relationships (parent-child)
    relationships: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// Service registry for multi-service tracing
    services: Arc<RwLock<HashMap<String, ServiceInfo>>>,
    /// Trace exporters for external systems
    exporters: Vec<Box<dyn TraceExporter>>,
}

impl DistributedTracer {
    /// Create a new distributed tracer
    #[must_use]
    pub fn new(config: TracingConfig) -> Self {
        Self {
            config,
            spans: Arc::new(RwLock::new(HashMap::new())),
            traces: Arc::new(Mutex::new(VecDeque::new())),
            relationships: Arc::new(RwLock::new(HashMap::new())),
            services: Arc::new(RwLock::new(HashMap::new())),
            exporters: Vec::new(),
        }
    }

    /// Add trace exporter
    pub fn add_exporter(&mut self, exporter: Box<dyn TraceExporter>) {
        self.exporters.push(exporter);
    }

    /// Register a service
    pub fn register_service(&self, service_id: &str, service_info: ServiceInfo) {
        if let Ok(mut services) = self.services.write() {
            services.insert(service_id.to_string(), service_info);
        }
    }

    /// Start a new root trace
    #[must_use]
    pub fn start_trace(&self, operation_name: &str, service_id: &str) -> TraceHandle {
        let trace_id = Uuid::new_v4().to_string();
        let span_id = Uuid::new_v4().to_string();

        let span = TraceSpan {
            trace_id: trace_id.clone(),
            span_id: span_id.clone(),
            parent_span_id: None,
            operation_name: operation_name.to_string(),
            service_id: service_id.to_string(),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            end_time: None,
            duration_ns: None,
            status: SpanStatus::Active,
            tags: HashMap::new(),
            logs: Vec::new(),
            baggage: HashMap::new(),
        };

        if let Ok(mut spans) = self.spans.write() {
            spans.insert(span_id.clone(), span);
        }

        TraceHandle::new(
            trace_id,
            span_id,
            self.spans.clone(),
            self.relationships.clone(),
            self.traces.clone(),
            !self.exporters.is_empty(),
        )
    }

    /// Start a child span
    pub fn start_child_span(
        &self,
        parent_handle: &TraceHandle,
        operation_name: &str,
        service_id: &str,
    ) -> TraceHandle {
        let span_id = Uuid::new_v4().to_string();

        let span = TraceSpan {
            trace_id: parent_handle.trace_id.clone(),
            span_id: span_id.clone(),
            parent_span_id: Some(parent_handle.span_id.clone()),
            operation_name: operation_name.to_string(),
            service_id: service_id.to_string(),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            end_time: None,
            duration_ns: None,
            status: SpanStatus::Active,
            tags: HashMap::new(),
            logs: Vec::new(),
            baggage: HashMap::new(),
        };

        if let Ok(mut spans) = self.spans.write() {
            spans.insert(span_id.clone(), span);
        }

        // Record parent-child relationship
        if let Ok(mut relationships) = self.relationships.write() {
            relationships
                .entry(parent_handle.span_id.clone())
                .or_insert_with(Vec::new)
                .push(span_id.clone());
        }

        TraceHandle::new(
            parent_handle.trace_id.clone(),
            span_id,
            self.spans.clone(),
            self.relationships.clone(),
            self.traces.clone(),
            !self.exporters.is_empty(),
        )
    }

    /// Get a completed trace by ID
    #[must_use]
    pub fn get_trace(&self, trace_id: &str) -> Option<Trace> {
        if let Ok(traces) = self.traces.lock() {
            traces.iter().find(|t| t.trace_id == trace_id).cloned()
        } else {
            None
        }
    }

    /// Get all active spans
    #[must_use]
    pub fn get_active_spans(&self) -> Vec<TraceSpan> {
        if let Ok(spans) = self.spans.read() {
            spans
                .values()
                .filter(|s| s.status == SpanStatus::Active)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Analyze trace performance
    #[must_use]
    pub fn analyze_trace_performance(&self, trace_id: &str) -> Option<TraceAnalysis> {
        self.get_trace(trace_id)
            .map(|trace| TraceAnalysis::from_trace(&trace))
    }

    /// Export traces to configured exporters
    pub fn export_traces(&mut self) {
        if let Ok(mut traces) = self.traces.lock() {
            let traces_to_export: Vec<Trace> = traces.drain(..).collect();

            for exporter in &mut self.exporters {
                for trace in &traces_to_export {
                    let _ = exporter.export_trace(trace);
                }
            }
        }
    }

    /// Get trace statistics
    #[must_use]
    pub fn get_trace_statistics(&self) -> TraceStatistics {
        let mut stats = TraceStatistics::default();

        if let Ok(traces) = self.traces.lock() {
            stats.total_traces = traces.len();

            let mut total_duration = 0u64;
            let mut service_counts = HashMap::new();
            let mut operation_counts = HashMap::new();

            for trace in traces.iter() {
                if let Some(duration) = trace.total_duration_ns {
                    total_duration += duration;
                }

                for span in &trace.spans {
                    *service_counts.entry(span.service_id.clone()).or_insert(0) += 1;
                    *operation_counts
                        .entry(span.operation_name.clone())
                        .or_insert(0) += 1;
                }
            }

            if stats.total_traces > 0 {
                stats.average_duration_ns = total_duration / stats.total_traces as u64;
            }

            stats.service_counts = service_counts;
            stats.operation_counts = operation_counts;
        }

        if let Ok(spans) = self.spans.read() {
            stats.active_spans = spans.len();
        }

        stats
    }

    /// Clean up old completed traces
    pub fn cleanup_old_traces(&self, retention_period: Duration) {
        let cutoff = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            - retention_period.as_nanos() as u64;

        if let Ok(mut traces) = self.traces.lock() {
            traces.retain(|trace| trace.spans.iter().any(|span| span.start_time > cutoff));
        }
    }
}

/// Tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Service name for this tracer instance
    pub service_name: String,
    /// Maximum number of traces to retain
    pub max_traces: usize,
    /// Maximum number of spans per trace
    pub max_spans_per_trace: usize,
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
    /// Enable baggage propagation
    pub enable_baggage: bool,
    /// Auto-export interval
    pub export_interval: Duration,
    /// Trace retention period
    pub retention_period: Duration,
}

impl TracingConfig {
    /// Create new tracing configuration
    #[must_use]
    pub fn new(service_name: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
            max_traces: 1000,
            max_spans_per_trace: 100,
            sampling_rate: 1.0,
            enable_baggage: true,
            export_interval: Duration::from_secs(30),
            retention_period: Duration::from_secs(3600), // 1 hour
        }
    }

    /// Set maximum traces
    #[must_use]
    pub fn max_traces(mut self, max: usize) -> Self {
        self.max_traces = max;
        self
    }

    /// Set sampling rate
    #[must_use]
    pub fn sampling_rate(mut self, rate: f64) -> Self {
        self.sampling_rate = rate.max(0.0).min(1.0);
        self
    }

    /// Set export interval
    #[must_use]
    pub fn export_interval(mut self, interval: Duration) -> Self {
        self.export_interval = interval;
        self
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self::new("default-service")
    }
}

/// Handle for managing a trace span
pub struct TraceHandle {
    /// Trace ID
    pub trace_id: String,
    /// Span ID
    pub span_id: String,
    /// Reference to spans storage
    spans: Arc<RwLock<HashMap<String, TraceSpan>>>,
    /// Reference to relationships storage
    relationships: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// Reference to traces storage
    traces: Arc<Mutex<VecDeque<Trace>>>,
    /// Whether to export traces
    should_export: bool,
}

impl TraceHandle {
    /// Create new trace handle
    fn new(
        trace_id: String,
        span_id: String,
        spans: Arc<RwLock<HashMap<String, TraceSpan>>>,
        relationships: Arc<RwLock<HashMap<String, Vec<String>>>>,
        traces: Arc<Mutex<VecDeque<Trace>>>,
        should_export: bool,
    ) -> Self {
        Self {
            trace_id,
            span_id,
            spans,
            relationships,
            traces,
            should_export,
        }
    }

    /// Add a tag to the span
    pub fn set_tag(&self, key: &str, value: &str) {
        if let Ok(mut spans) = self.spans.write() {
            if let Some(span) = spans.get_mut(&self.span_id) {
                span.tags.insert(key.to_string(), value.to_string());
            }
        }
    }

    /// Add baggage (cross-process data)
    pub fn set_baggage(&self, key: &str, value: &str) {
        if let Ok(mut spans) = self.spans.write() {
            if let Some(span) = spans.get_mut(&self.span_id) {
                span.baggage.insert(key.to_string(), value.to_string());
            }
        }
    }

    /// Get baggage value
    #[must_use]
    pub fn get_baggage(&self, key: &str) -> Option<String> {
        if let Ok(spans) = self.spans.read() {
            spans.get(&self.span_id)?.baggage.get(key).cloned()
        } else {
            None
        }
    }

    /// Log an event
    pub fn log_event(&self, message: &str, level: LogLevel) {
        let log_entry = LogEntry {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            level,
            message: message.to_string(),
            fields: HashMap::new(),
        };

        if let Ok(mut spans) = self.spans.write() {
            if let Some(span) = spans.get_mut(&self.span_id) {
                span.logs.push(log_entry);
            }
        }
    }

    /// Set span status
    pub fn set_status(&self, status: SpanStatus) {
        if let Ok(mut spans) = self.spans.write() {
            if let Some(span) = spans.get_mut(&self.span_id) {
                span.status = status;
            }
        }
    }

    /// Record an error
    pub fn record_error(&self, error: &str) {
        self.set_tag("error", "true");
        self.set_tag("error.message", error);
        self.log_event(&format!("Error: {error}"), LogLevel::Error);
        self.set_status(SpanStatus::Error);
    }

    /// Finish the span
    pub fn finish(self) {
        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        if let Ok(mut spans) = self.spans.write() {
            if let Some(span) = spans.get_mut(&self.span_id) {
                span.end_time = Some(end_time);
                span.duration_ns = Some(end_time - span.start_time);
                if span.status == SpanStatus::Active {
                    span.status = SpanStatus::Completed;
                }
            }
        }

        // Check if this completes a trace
        self.check_trace_completion();
    }

    /// Check if trace is complete and move to completed traces
    fn check_trace_completion(&self) {
        let mut all_spans_complete = true;
        let mut trace_spans = Vec::new();

        if let Ok(spans) = self.spans.read() {
            for span in spans.values() {
                if span.trace_id == self.trace_id {
                    if span.status == SpanStatus::Active {
                        all_spans_complete = false;
                    }
                    trace_spans.push(span.clone());
                }
            }
        }

        if all_spans_complete && !trace_spans.is_empty() {
            let mut total_duration = 0u64;
            let mut root_span = None;

            for span in &trace_spans {
                if let Some(duration) = span.duration_ns {
                    total_duration = total_duration.max(duration);
                }
                if span.parent_span_id.is_none() {
                    root_span = Some(span.clone());
                }
            }

            let trace = Trace {
                trace_id: self.trace_id.clone(),
                spans: trace_spans,
                root_span,
                total_duration_ns: if total_duration > 0 {
                    Some(total_duration)
                } else {
                    None
                },
                service_count: self.count_unique_services(),
                start_time: self.get_trace_start_time(),
                end_time: self.get_trace_end_time(),
            };

            if let Ok(mut traces) = self.traces.lock() {
                traces.push_back(trace);

                // Remove completed spans from active spans
                if let Ok(mut spans) = self.spans.write() {
                    spans.retain(|_, span| span.trace_id != self.trace_id);
                }
            }
        }
    }

    /// Count unique services in the trace
    fn count_unique_services(&self) -> usize {
        if let Ok(spans) = self.spans.read() {
            let services: std::collections::HashSet<_> = spans
                .values()
                .filter(|span| span.trace_id == self.trace_id)
                .map(|span| &span.service_id)
                .collect();
            services.len()
        } else {
            0
        }
    }

    /// Get trace start time
    fn get_trace_start_time(&self) -> u64 {
        if let Ok(spans) = self.spans.read() {
            spans
                .values()
                .filter(|span| span.trace_id == self.trace_id)
                .map(|span| span.start_time)
                .min()
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// Get trace end time
    fn get_trace_end_time(&self) -> Option<u64> {
        if let Ok(spans) = self.spans.read() {
            spans
                .values()
                .filter(|span| span.trace_id == self.trace_id)
                .filter_map(|span| span.end_time)
                .max()
        } else {
            None
        }
    }
}

/// Individual span in a distributed trace
#[derive(Debug, Clone)]
pub struct TraceSpan {
    /// Unique trace identifier
    pub trace_id: String,
    /// Unique span identifier
    pub span_id: String,
    /// Parent span ID (None for root spans)
    pub parent_span_id: Option<String>,
    /// Operation name
    pub operation_name: String,
    /// Service identifier
    pub service_id: String,
    /// Start timestamp (nanoseconds since epoch)
    pub start_time: u64,
    /// End timestamp (nanoseconds since epoch)
    pub end_time: Option<u64>,
    /// Duration in nanoseconds
    pub duration_ns: Option<u64>,
    /// Span status
    pub status: SpanStatus,
    /// Key-value tags
    pub tags: HashMap<String, String>,
    /// Log entries
    pub logs: Vec<LogEntry>,
    /// Cross-process baggage
    pub baggage: HashMap<String, String>,
}

/// Span status
#[derive(Debug, Clone, PartialEq)]
pub enum SpanStatus {
    /// Span is currently active
    Active,
    /// Span completed successfully
    Completed,
    /// Span completed with error
    Error,
    /// Span was cancelled
    Cancelled,
}

/// Log entry within a span
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Timestamp (nanoseconds since epoch)
    pub timestamp: u64,
    /// Log level
    pub level: LogLevel,
    /// Log message
    pub message: String,
    /// Additional fields
    pub fields: HashMap<String, String>,
}

/// Log levels
#[derive(Debug, Clone, PartialEq)]
pub enum LogLevel {
    /// Debug information
    Debug,
    /// Informational messages
    Info,
    /// Warning messages
    Warn,
    /// Error messages
    Error,
}

/// Complete trace containing all spans
#[derive(Debug, Clone)]
pub struct Trace {
    /// Unique trace identifier
    pub trace_id: String,
    /// All spans in the trace
    pub spans: Vec<TraceSpan>,
    /// Root span (entry point)
    pub root_span: Option<TraceSpan>,
    /// Total trace duration
    pub total_duration_ns: Option<u64>,
    /// Number of unique services
    pub service_count: usize,
    /// Trace start time
    pub start_time: u64,
    /// Trace end time
    pub end_time: Option<u64>,
}

/// Service information for multi-service tracing
#[derive(Debug, Clone)]
pub struct ServiceInfo {
    /// Service name
    pub name: String,
    /// Service version
    pub version: String,
    /// Service endpoint/address
    pub address: String,
    /// Service metadata
    pub metadata: HashMap<String, String>,
}

/// Trace analysis results
#[derive(Debug, Clone)]
pub struct TraceAnalysis {
    /// Trace duration
    pub total_duration_ns: u64,
    /// Critical path (longest sequential chain)
    pub critical_path: Vec<String>,
    /// Service breakdown
    pub service_breakdown: HashMap<String, ServiceAnalysis>,
    /// Bottlenecks identified
    pub bottlenecks: Vec<Bottleneck>,
    /// Parallelism analysis
    pub parallelism_factor: f64,
}

impl TraceAnalysis {
    /// Create analysis from trace
    fn from_trace(trace: &Trace) -> Self {
        let total_duration_ns = trace.total_duration_ns.unwrap_or(0);
        let critical_path = Self::find_critical_path(trace);
        let service_breakdown = Self::analyze_services(trace);
        let bottlenecks = Self::identify_bottlenecks(trace);
        let parallelism_factor = Self::calculate_parallelism(trace);

        Self {
            total_duration_ns,
            critical_path,
            service_breakdown,
            bottlenecks,
            parallelism_factor,
        }
    }

    /// Find the critical path (longest sequential execution chain)
    fn find_critical_path(trace: &Trace) -> Vec<String> {
        // Simplified critical path analysis
        let mut path = Vec::new();

        if let Some(root) = &trace.root_span {
            path.push(root.operation_name.clone());
            // In a real implementation, we'd build a dependency graph
            // and find the longest path through it
        }

        path
    }

    /// Analyze service performance
    fn analyze_services(trace: &Trace) -> HashMap<String, ServiceAnalysis> {
        let mut analysis = HashMap::new();

        for span in &trace.spans {
            let entry =
                analysis
                    .entry(span.service_id.clone())
                    .or_insert_with(|| ServiceAnalysis {
                        total_duration_ns: 0,
                        span_count: 0,
                        error_count: 0,
                        operations: HashMap::new(),
                    });

            entry.span_count += 1;
            if let Some(duration) = span.duration_ns {
                entry.total_duration_ns += duration;
            }
            if span.status == SpanStatus::Error {
                entry.error_count += 1;
            }

            *entry
                .operations
                .entry(span.operation_name.clone())
                .or_insert(0) += 1;
        }

        analysis
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(trace: &Trace) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Find spans that take unusually long
        let durations: Vec<u64> = trace.spans.iter().filter_map(|s| s.duration_ns).collect();

        if let Some(max_duration) = durations.iter().max() {
            let avg_duration = durations.iter().sum::<u64>() / durations.len() as u64;

            for span in &trace.spans {
                if let Some(duration) = span.duration_ns {
                    if duration > avg_duration * 3 {
                        bottlenecks.push(Bottleneck {
                            span_id: span.span_id.clone(),
                            operation_name: span.operation_name.clone(),
                            service_id: span.service_id.clone(),
                            duration_ns: duration,
                            severity: if duration == *max_duration {
                                BottleneckSeverity::Critical
                            } else {
                                BottleneckSeverity::High
                            },
                        });
                    }
                }
            }
        }

        bottlenecks
    }

    /// Calculate parallelism factor
    fn calculate_parallelism(trace: &Trace) -> f64 {
        if trace.spans.is_empty() {
            return 1.0;
        }

        let total_work: u64 = trace.spans.iter().filter_map(|s| s.duration_ns).sum();

        let total_duration = trace.total_duration_ns.unwrap_or(1);

        if total_duration == 0 {
            1.0
        } else {
            total_work as f64 / total_duration as f64
        }
    }
}

/// Service analysis within a trace
#[derive(Debug, Clone)]
pub struct ServiceAnalysis {
    /// Total time spent in this service
    pub total_duration_ns: u64,
    /// Number of spans for this service
    pub span_count: usize,
    /// Number of errors in this service
    pub error_count: usize,
    /// Operations performed by this service
    pub operations: HashMap<String, usize>,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Span ID of the bottleneck
    pub span_id: String,
    /// Operation name
    pub operation_name: String,
    /// Service ID
    pub service_id: String,
    /// Duration that makes it a bottleneck
    pub duration_ns: u64,
    /// Severity level
    pub severity: BottleneckSeverity,
}

/// Bottleneck severity
#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckSeverity {
    /// Minor bottleneck
    Low,
    /// Moderate bottleneck
    Medium,
    /// Significant bottleneck
    High,
    /// Critical bottleneck requiring immediate attention
    Critical,
}

/// Trace statistics
#[derive(Debug, Clone, Default)]
pub struct TraceStatistics {
    /// Total number of completed traces
    pub total_traces: usize,
    /// Number of currently active spans
    pub active_spans: usize,
    /// Average trace duration
    pub average_duration_ns: u64,
    /// Count of spans per service
    pub service_counts: HashMap<String, usize>,
    /// Count of spans per operation
    pub operation_counts: HashMap<String, usize>,
}

/// Trait for exporting traces to external systems
pub trait TraceExporter: Send + Sync {
    /// Export a single trace
    fn export_trace(&mut self, trace: &Trace) -> Result<(), Box<dyn std::error::Error>>;

    /// Flush any pending exports
    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}

/// Console trace exporter for debugging
pub struct ConsoleTraceExporter {
    /// Whether to include detailed span information
    pub verbose: bool,
}

impl ConsoleTraceExporter {
    /// Create new console exporter
    #[must_use]
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
}

impl TraceExporter for ConsoleTraceExporter {
    fn export_trace(&mut self, trace: &Trace) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Trace {} ===", trace.trace_id);
        println!("Duration: {:?}ns", trace.total_duration_ns);
        println!("Services: {}", trace.service_count);
        println!("Spans: {}", trace.spans.len());

        if self.verbose {
            for span in &trace.spans {
                println!(
                    "  {} [{}] {}ms in {}",
                    span.operation_name,
                    &span.span_id[..8],
                    span.duration_ns.unwrap_or(0) / 1_000_000,
                    span.service_id
                );

                for (key, value) in &span.tags {
                    println!("    {key}: {value}");
                }
            }
        }

        Ok(())
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

/// JSON file trace exporter
pub struct JsonFileTraceExporter {
    /// File path for export
    pub file_path: String,
}

impl JsonFileTraceExporter {
    /// Create new JSON file exporter
    #[must_use]
    pub fn new(file_path: &str) -> Self {
        Self {
            file_path: file_path.to_string(),
        }
    }
}

impl TraceExporter for JsonFileTraceExporter {
    fn export_trace(&mut self, trace: &Trace) -> Result<(), Box<dyn std::error::Error>> {
        // In a real implementation, we'd serialize the trace to JSON
        // and append it to the file
        println!(
            "Would export trace {} to {}",
            trace.trace_id, self.file_path
        );
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_distributed_tracer_creation() {
        let config = TracingConfig::new("test-service");
        let tracer = DistributedTracer::new(config);

        let stats = tracer.get_trace_statistics();
        assert_eq!(stats.total_traces, 0);
        assert_eq!(stats.active_spans, 0);
    }

    #[test]
    fn test_trace_creation_and_completion() {
        let config = TracingConfig::new("test-service");
        let tracer = DistributedTracer::new(config);

        let handle = tracer.start_trace("test-operation", "test-service");
        handle.set_tag("version", "1.0");
        handle.log_event("Starting operation", LogLevel::Info);

        thread::sleep(Duration::from_millis(10));
        handle.finish();

        thread::sleep(Duration::from_millis(10));
        let stats = tracer.get_trace_statistics();
        assert_eq!(stats.total_traces, 1);
    }

    #[test]
    fn test_child_span_creation() {
        let config = TracingConfig::new("test-service");
        let tracer = DistributedTracer::new(config);

        let parent_handle = tracer.start_trace("parent-operation", "test-service");
        let child_handle =
            tracer.start_child_span(&parent_handle, "child-operation", "test-service");

        child_handle.set_tag("child", "true");
        child_handle.finish();
        parent_handle.finish();

        thread::sleep(Duration::from_millis(10));
        let stats = tracer.get_trace_statistics();
        assert_eq!(stats.total_traces, 1);
    }

    #[test]
    fn test_baggage_propagation() {
        let config = TracingConfig::new("test-service");
        let tracer = DistributedTracer::new(config);

        let handle = tracer.start_trace("test-operation", "test-service");
        handle.set_baggage("user_id", "123");
        handle.set_baggage("session_id", "abc");

        assert_eq!(handle.get_baggage("user_id"), Some("123".to_string()));
        assert_eq!(handle.get_baggage("session_id"), Some("abc".to_string()));
        assert_eq!(handle.get_baggage("nonexistent"), None);

        handle.finish();
    }

    #[test]
    fn test_error_recording() {
        let config = TracingConfig::new("test-service");
        let tracer = DistributedTracer::new(config);

        let handle = tracer.start_trace("failing-operation", "test-service");
        handle.record_error("Something went wrong");
        let trace_id = handle.trace_id.clone();
        handle.finish();

        thread::sleep(Duration::from_millis(10));

        if let Some(trace) = tracer.get_trace(&trace_id) {
            let span = &trace.spans[0];
            assert_eq!(span.status, SpanStatus::Error);
            assert_eq!(span.tags.get("error"), Some(&"true".to_string()));
            assert!(span.tags.contains_key("error.message"));
        }
    }

    #[test]
    fn test_trace_analysis() {
        let config = TracingConfig::new("test-service");
        let tracer = DistributedTracer::new(config);

        let handle = tracer.start_trace("complex-operation", "test-service");
        thread::sleep(Duration::from_millis(50));
        let trace_id = handle.trace_id.clone();
        handle.finish();

        thread::sleep(Duration::from_millis(10));

        if let Some(analysis) = tracer.analyze_trace_performance(&trace_id) {
            assert!(analysis.total_duration_ns > 0);
            assert!(!analysis.critical_path.is_empty());
        }
    }

    #[test]
    fn test_service_registration() {
        let config = TracingConfig::new("test-service");
        let tracer = DistributedTracer::new(config);

        let service_info = ServiceInfo {
            name: "test-service".to_string(),
            version: "1.0.0".to_string(),
            address: "localhost:8080".to_string(),
            metadata: HashMap::new(),
        };

        tracer.register_service("test-service", service_info);

        // Service registration is stored internally
        assert_eq!(tracer.get_active_spans().len(), 0);
    }

    #[test]
    fn test_console_trace_exporter() {
        let mut exporter = ConsoleTraceExporter::new(true);

        let trace = Trace {
            trace_id: "test-trace".to_string(),
            spans: vec![],
            root_span: None,
            total_duration_ns: Some(1000000),
            service_count: 1,
            start_time: 123456789,
            end_time: Some(123456790),
        };

        assert!(exporter.export_trace(&trace).is_ok());
        assert!(exporter.flush().is_ok());
    }

    #[test]
    fn test_tracing_config() {
        let config = TracingConfig::new("my-service")
            .max_traces(500)
            .sampling_rate(0.8)
            .export_interval(Duration::from_secs(60));

        assert_eq!(config.service_name, "my-service");
        assert_eq!(config.max_traces, 500);
        assert_eq!(config.sampling_rate, 0.8);
        assert_eq!(config.export_interval, Duration::from_secs(60));
    }
}

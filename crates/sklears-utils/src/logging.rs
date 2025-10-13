//! Comprehensive logging framework for sklears
//!
//! This module provides structured logging with configurable levels, performance logging,
//! distributed logging support, and log analysis utilities.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Log levels in order of verbosity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

impl Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Trace => write!(f, "TRACE"),
        }
    }
}

impl FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "ERROR" => Ok(LogLevel::Error),
            "WARN" => Ok(LogLevel::Warn),
            "INFO" => Ok(LogLevel::Info),
            "DEBUG" => Ok(LogLevel::Debug),
            "TRACE" => Ok(LogLevel::Trace),
            _ => Err(format!("Invalid log level: {s}")),
        }
    }
}

/// Structured log entry
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: SystemTime,
    pub level: LogLevel,
    pub message: String,
    pub module: String,
    pub file: String,
    pub line: u32,
    pub thread_id: String,
    pub fields: HashMap<String, Value>,
}

impl LogEntry {
    pub fn new(level: LogLevel, message: String, module: String, file: String, line: u32) -> Self {
        Self {
            timestamp: SystemTime::now(),
            level,
            message,
            module,
            file,
            line,
            thread_id: format!("{:?}", std::thread::current().id()),
            fields: HashMap::new(),
        }
    }

    pub fn with_field<V: Into<Value>>(mut self, key: String, value: V) -> Self {
        self.fields.insert(key, value.into());
        self
    }

    pub fn to_json(&self) -> Value {
        let timestamp_ms = self
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        let mut json = json!({
            "timestamp": timestamp_ms,
            "level": self.level.to_string(),
            "message": self.message,
            "module": self.module,
            "file": self.file,
            "line": self.line,
            "thread_id": self.thread_id,
        });

        if let Value::Object(ref mut map) = json {
            for (key, value) in &self.fields {
                map.insert(key.clone(), value.clone());
            }
        }

        json
    }

    pub fn to_text(&self) -> String {
        let timestamp = self
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        format!(
            "[{}] {} [{}:{}] [{}] {} {}",
            timestamp, self.level, self.file, self.line, self.thread_id, self.module, self.message
        )
    }
}

/// Log formatter trait
pub trait LogFormatter: Send + Sync {
    fn format(&self, entry: &LogEntry) -> String;
}

/// JSON formatter
pub struct JsonFormatter;

impl LogFormatter for JsonFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        entry.to_json().to_string()
    }
}

/// Text formatter
pub struct TextFormatter;

impl LogFormatter for TextFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        entry.to_text()
    }
}

/// Log output destination trait
pub trait LogOutput: Send + Sync {
    fn write(&mut self, formatted_log: &str) -> Result<(), std::io::Error>;
    fn flush(&mut self) -> Result<(), std::io::Error>;
}

/// Console output
pub struct ConsoleOutput;

impl LogOutput for ConsoleOutput {
    fn write(&mut self, formatted_log: &str) -> Result<(), std::io::Error> {
        println!("{formatted_log}");
        Ok(())
    }

    fn flush(&mut self) -> Result<(), std::io::Error> {
        std::io::stdout().flush()
    }
}

/// File output
pub struct FileOutput {
    writer: BufWriter<File>,
}

impl FileOutput {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl LogOutput for FileOutput {
    fn write(&mut self, formatted_log: &str) -> Result<(), std::io::Error> {
        writeln!(self.writer, "{formatted_log}")?;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), std::io::Error> {
        self.writer.flush()
    }
}

/// Logger configuration
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    pub level: LogLevel,
    pub module_filters: HashMap<String, LogLevel>,
    pub enable_performance_logging: bool,
    pub buffer_size: usize,
    pub auto_flush: bool,
    pub include_caller_info: bool,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            module_filters: HashMap::new(),
            enable_performance_logging: false,
            buffer_size: 1000,
            auto_flush: true,
            include_caller_info: true,
        }
    }
}

/// Main logger implementation
pub struct Logger {
    config: Arc<RwLock<LoggerConfig>>,
    outputs: Arc<Mutex<Vec<Box<dyn LogOutput>>>>,
    formatter: Arc<dyn LogFormatter>,
    buffer: Arc<Mutex<Vec<LogEntry>>>,
    stats: Arc<Mutex<LogStats>>,
}

#[derive(Debug, Default)]
pub struct LogStats {
    pub total_logs: u64,
    pub logs_by_level: HashMap<LogLevel, u64>,
    pub logs_by_module: HashMap<String, u64>,
    pub buffer_overflows: u64,
    pub write_errors: u64,
}

impl Logger {
    pub fn new(config: LoggerConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            outputs: Arc::new(Mutex::new(Vec::new())),
            formatter: Arc::new(TextFormatter),
            buffer: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(LogStats::default())),
        }
    }

    pub fn with_formatter(mut self, formatter: Arc<dyn LogFormatter>) -> Self {
        self.formatter = formatter;
        self
    }

    pub fn add_output(&self, output: Box<dyn LogOutput>) {
        let mut outputs = self.outputs.lock().unwrap();
        outputs.push(output);
    }

    pub fn log(&self, entry: LogEntry) {
        let config = self.config.read().unwrap();

        // Check if we should log this entry
        if !self.should_log(&entry.level, &entry.module, &config) {
            return;
        }

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_logs += 1;
            *stats.logs_by_level.entry(entry.level).or_insert(0) += 1;
            *stats
                .logs_by_module
                .entry(entry.module.clone())
                .or_insert(0) += 1;
        }

        // Add to buffer
        {
            let mut buffer = self.buffer.lock().unwrap();
            if buffer.len() >= config.buffer_size {
                buffer.remove(0); // Remove oldest entry
                let mut stats = self.stats.lock().unwrap();
                stats.buffer_overflows += 1;
            }
            buffer.push(entry.clone());
        }

        // Write immediately if auto_flush is enabled
        if config.auto_flush {
            self.flush_entry(&entry);
        }
    }

    fn should_log(&self, level: &LogLevel, module: &str, config: &LoggerConfig) -> bool {
        // Check module-specific filter first
        if let Some(module_level) = config.module_filters.get(module) {
            return level <= module_level;
        }

        // Fall back to global level
        level <= &config.level
    }

    fn flush_entry(&self, entry: &LogEntry) {
        let formatted = self.formatter.format(entry);
        let mut outputs = self.outputs.lock().unwrap();

        for output in outputs.iter_mut() {
            if output.write(&formatted).is_err() {
                let mut stats = self.stats.lock().unwrap();
                stats.write_errors += 1;
            }
        }
    }

    pub fn flush(&self) {
        let buffer = {
            let mut buffer = self.buffer.lock().unwrap();
            let entries = buffer.clone();
            buffer.clear();
            entries
        };

        for entry in buffer {
            self.flush_entry(&entry);
        }

        // Flush all outputs
        let mut outputs = self.outputs.lock().unwrap();
        for output in outputs.iter_mut() {
            let _ = output.flush();
        }
    }

    pub fn set_level(&self, level: LogLevel) {
        let mut config = self.config.write().unwrap();
        config.level = level;
    }

    pub fn set_module_level(&self, module: String, level: LogLevel) {
        let mut config = self.config.write().unwrap();
        config.module_filters.insert(module, level);
    }

    pub fn stats(&self) -> LogStats {
        self.stats.lock().unwrap().clone()
    }

    pub fn clear_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = LogStats::default();
    }
}

impl Clone for LogStats {
    fn clone(&self) -> Self {
        Self {
            total_logs: self.total_logs,
            logs_by_level: self.logs_by_level.clone(),
            logs_by_module: self.logs_by_module.clone(),
            buffer_overflows: self.buffer_overflows,
            write_errors: self.write_errors,
        }
    }
}

/// Performance logger for tracking operation timings
pub struct PerformanceLogger {
    logger: Arc<Logger>,
    operations: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
}

impl PerformanceLogger {
    pub fn new(logger: Arc<Logger>) -> Self {
        Self {
            logger,
            operations: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn time_operation<F, R>(&self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();

        // Record timing
        {
            let mut operations = self.operations.lock().unwrap();
            operations
                .entry(name.to_string())
                .or_default()
                .push(duration);
        }

        // Log performance
        let entry = LogEntry::new(
            LogLevel::Debug,
            format!("Operation '{name}' completed"),
            "performance".to_string(),
            "performance_logger.rs".to_string(),
            0,
        )
        .with_field("operation".to_string(), name.to_string())
        .with_field("duration_ms".to_string(), duration.as_millis() as f64);

        self.logger.log(entry);

        result
    }

    pub fn get_operation_stats(&self, name: &str) -> Option<OperationStats> {
        let operations = self.operations.lock().unwrap();
        if let Some(durations) = operations.get(name) {
            if durations.is_empty() {
                return None;
            }

            let total_ms: f64 = durations.iter().map(|d| d.as_millis() as f64).sum();
            let count = durations.len();
            let avg_ms = total_ms / count as f64;

            let mut sorted_durations = durations.clone();
            sorted_durations.sort();

            let min_ms = sorted_durations.first().unwrap().as_millis() as f64;
            let max_ms = sorted_durations.last().unwrap().as_millis() as f64;

            let median_ms = if count % 2 == 0 {
                let mid = count / 2;
                (sorted_durations[mid - 1].as_millis() + sorted_durations[mid].as_millis()) as f64
                    / 2.0
            } else {
                sorted_durations[count / 2].as_millis() as f64
            };

            Some(OperationStats {
                name: name.to_string(),
                count,
                total_ms,
                avg_ms,
                min_ms,
                max_ms,
                median_ms,
            })
        } else {
            None
        }
    }

    pub fn clear_operation_stats(&self, name: &str) {
        let mut operations = self.operations.lock().unwrap();
        operations.remove(name);
    }

    pub fn get_all_operations(&self) -> Vec<String> {
        let operations = self.operations.lock().unwrap();
        operations.keys().cloned().collect()
    }
}

#[derive(Debug, Clone)]
pub struct OperationStats {
    pub name: String,
    pub count: usize,
    pub total_ms: f64,
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub median_ms: f64,
}

/// Distributed logging coordinator
pub struct DistributedLogger {
    local_logger: Arc<Logger>,
    node_id: String,
    cluster_nodes: Arc<RwLock<Vec<String>>>,
}

impl DistributedLogger {
    pub fn new(local_logger: Arc<Logger>, node_id: String) -> Self {
        Self {
            local_logger,
            node_id,
            cluster_nodes: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn add_node(&self, node_id: String) {
        let mut nodes = self.cluster_nodes.write().unwrap();
        if !nodes.contains(&node_id) {
            nodes.push(node_id);
        }
    }

    pub fn remove_node(&self, node_id: &str) {
        let mut nodes = self.cluster_nodes.write().unwrap();
        nodes.retain(|id| id != node_id);
    }

    pub fn log_distributed(&self, mut entry: LogEntry) {
        // Add node information
        entry = entry.with_field("node_id".to_string(), self.node_id.clone());

        // Log locally
        self.local_logger.log(entry);

        // In a real implementation, you would send logs to other nodes here
        // This is a placeholder for distributed logging functionality
    }

    pub fn get_cluster_nodes(&self) -> Vec<String> {
        self.cluster_nodes.read().unwrap().clone()
    }
}

/// Log analysis utilities
pub struct LogAnalyzer {
    entries: Vec<LogEntry>,
}

impl LogAnalyzer {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add_entries(&mut self, entries: Vec<LogEntry>) {
        self.entries.extend(entries);
    }

    pub fn analyze_patterns(&self) -> LogAnalysis {
        let mut analysis = LogAnalysis::default();

        for entry in &self.entries {
            analysis.total_entries += 1;
            *analysis.entries_by_level.entry(entry.level).or_insert(0) += 1;
            *analysis
                .entries_by_module
                .entry(entry.module.clone())
                .or_insert(0) += 1;

            // Detect error patterns
            if entry.level == LogLevel::Error {
                *analysis
                    .error_patterns
                    .entry(entry.message.clone())
                    .or_insert(0) += 1;
            }
        }

        analysis
    }

    pub fn find_errors_in_timeframe(&self, start: SystemTime, end: SystemTime) -> Vec<LogEntry> {
        self.entries
            .iter()
            .filter(|entry| {
                entry.level == LogLevel::Error && entry.timestamp >= start && entry.timestamp <= end
            })
            .cloned()
            .collect()
    }

    pub fn get_module_activity(&self, module: &str) -> Vec<LogEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.module == module)
            .cloned()
            .collect()
    }
}

#[derive(Debug, Default)]
pub struct LogAnalysis {
    pub total_entries: u64,
    pub entries_by_level: HashMap<LogLevel, u64>,
    pub entries_by_module: HashMap<String, u64>,
    pub error_patterns: HashMap<String, u64>,
}

impl Default for LogAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

lazy_static::lazy_static! {
    /// Global logger instance
    static ref GLOBAL_LOGGER: Arc<Logger> = {
        let config = LoggerConfig::default();
        let logger = Arc::new(Logger::new(config));
        logger.add_output(Box::new(ConsoleOutput));
        logger
    };
}

/// Logging macros
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        $crate::logging::log_with_level($crate::logging::LogLevel::Error, format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        $crate::logging::log_with_level($crate::logging::LogLevel::Warn, format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        $crate::logging::log_with_level($crate::logging::LogLevel::Info, format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {
        $crate::logging::log_with_level($crate::logging::LogLevel::Debug, format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_trace {
    ($($arg:tt)*) => {
        $crate::logging::log_with_level($crate::logging::LogLevel::Trace, format!($($arg)*))
    };
}

pub fn log_with_level(level: LogLevel, message: String) {
    let entry = LogEntry::new(
        level,
        message,
        "global".to_string(),
        "unknown".to_string(),
        0,
    );
    GLOBAL_LOGGER.log(entry);
}

pub fn get_global_logger() -> Arc<Logger> {
    GLOBAL_LOGGER.clone()
}

pub fn set_global_level(level: LogLevel) {
    GLOBAL_LOGGER.set_level(level);
}

pub fn flush_global_logger() {
    GLOBAL_LOGGER.flush();
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    #[test]
    fn test_log_levels() {
        assert!(LogLevel::Error < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Trace);
    }

    #[test]
    fn test_log_entry_creation() {
        let entry = LogEntry::new(
            LogLevel::Info,
            "Test message".to_string(),
            "test_module".to_string(),
            "test.rs".to_string(),
            42,
        );

        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.message, "Test message");
        assert_eq!(entry.module, "test_module");
        assert_eq!(entry.file, "test.rs");
        assert_eq!(entry.line, 42);
    }

    #[test]
    fn test_log_entry_with_fields() {
        let entry = LogEntry::new(
            LogLevel::Debug,
            "Debug message".to_string(),
            "test".to_string(),
            "test.rs".to_string(),
            1,
        )
        .with_field("key1".to_string(), "value1".to_string())
        .with_field("key2".to_string(), 42);

        assert_eq!(entry.fields.len(), 2);
        assert_eq!(
            entry.fields.get("key1").unwrap(),
            &Value::String("value1".to_string())
        );
        assert_eq!(entry.fields.get("key2").unwrap(), &Value::Number(42.into()));
    }

    #[test]
    fn test_logger_creation() {
        let config = LoggerConfig::default();
        let logger = Logger::new(config);

        let stats = logger.stats();
        assert_eq!(stats.total_logs, 0);
    }

    #[test]
    fn test_logger_with_file_output() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = LoggerConfig::default();
        let logger = Logger::new(config);

        let file_output = FileOutput::new(temp_file.path()).unwrap();
        logger.add_output(Box::new(file_output));

        let entry = LogEntry::new(
            LogLevel::Info,
            "Test log".to_string(),
            "test".to_string(),
            "test.rs".to_string(),
            1,
        );

        logger.log(entry);
        logger.flush();

        let stats = logger.stats();
        assert_eq!(stats.total_logs, 1);
    }

    #[test]
    fn test_performance_logger() {
        let config = LoggerConfig::default();
        let logger = Arc::new(Logger::new(config));
        let perf_logger = PerformanceLogger::new(logger);

        let result = perf_logger.time_operation("test_op", || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);

        let stats = perf_logger.get_operation_stats("test_op").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.avg_ms >= 10.0);
    }

    #[test]
    fn test_log_analyzer() {
        let mut analyzer = LogAnalyzer::new();

        let entries = vec![
            LogEntry::new(
                LogLevel::Info,
                "Info message".to_string(),
                "module1".to_string(),
                "test.rs".to_string(),
                1,
            ),
            LogEntry::new(
                LogLevel::Error,
                "Error message".to_string(),
                "module1".to_string(),
                "test.rs".to_string(),
                2,
            ),
            LogEntry::new(
                LogLevel::Debug,
                "Debug message".to_string(),
                "module2".to_string(),
                "test.rs".to_string(),
                3,
            ),
        ];

        analyzer.add_entries(entries);
        let analysis = analyzer.analyze_patterns();

        assert_eq!(analysis.total_entries, 3);
        assert_eq!(*analysis.entries_by_level.get(&LogLevel::Info).unwrap(), 1);
        assert_eq!(*analysis.entries_by_level.get(&LogLevel::Error).unwrap(), 1);
        assert_eq!(*analysis.entries_by_module.get("module1").unwrap(), 2);
        assert_eq!(*analysis.entries_by_module.get("module2").unwrap(), 1);
    }

    #[test]
    fn test_distributed_logger() {
        let config = LoggerConfig::default();
        let local_logger = Arc::new(Logger::new(config));
        let dist_logger = DistributedLogger::new(local_logger, "node1".to_string());

        dist_logger.add_node("node2".to_string());
        dist_logger.add_node("node3".to_string());

        let nodes = dist_logger.get_cluster_nodes();
        assert_eq!(nodes.len(), 2);
        assert!(nodes.contains(&"node2".to_string()));
        assert!(nodes.contains(&"node3".to_string()));
    }
}

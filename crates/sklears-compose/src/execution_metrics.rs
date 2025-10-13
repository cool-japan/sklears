use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use std::fmt;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use scirs2_core::error::SklResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub task_id: String,
    pub strategy_name: String,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub duration: Option<Duration>,
    pub status: TaskStatus,
    pub resource_usage: ResourceUsage,
    pub performance_data: PerformanceData,
    pub error_details: Option<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Retrying,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub memory_peak_bytes: u64,
    pub gpu_usage_percent: Option<f64>,
    pub gpu_memory_bytes: Option<u64>,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub disk_bytes_read: u64,
    pub disk_bytes_written: u64,
    pub file_handles_used: u32,
    pub thread_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    pub throughput_ops_per_sec: f64,
    pub latency_percentiles: LatencyPercentiles,
    pub queue_depth: u32,
    pub batch_size: u32,
    pub parallelism_factor: f32,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub retry_count: u32,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: Duration,
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub p999: Duration,
    pub min: Duration,
    pub max: Duration,
    pub mean: Duration,
}

#[derive(Debug, Clone)]
pub struct MetricsCollector {
    metrics_store: Arc<RwLock<VecDeque<ExecutionMetrics>>>,
    active_tasks: Arc<Mutex<HashMap<String, TaskMetrics>>>,
    aggregated_stats: Arc<RwLock<AggregatedStats>>,
    event_sender: broadcast::Sender<MetricsEvent>,
    config: MetricsConfig,
}

#[derive(Debug)]
struct TaskMetrics {
    start_time: Instant,
    start_system_time: SystemTime,
    strategy_name: String,
    resource_tracker: ResourceTracker,
    performance_tracker: PerformanceTracker,
    custom_data: HashMap<String, String>,
}

#[derive(Debug)]
struct ResourceTracker {
    initial_memory: u64,
    peak_memory: u64,
    cpu_samples: VecDeque<f64>,
    network_start_sent: u64,
    network_start_received: u64,
    disk_start_read: u64,
    disk_start_written: u64,
}

#[derive(Debug)]
struct PerformanceTracker {
    operation_count: u64,
    latency_samples: VecDeque<Duration>,
    error_count: u64,
    retry_count: u32,
    cache_hits: u64,
    cache_misses: u64,
    custom_counters: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct MetricsConfig {
    pub max_stored_metrics: usize,
    pub sampling_interval: Duration,
    pub latency_histogram_buckets: Vec<Duration>,
    pub enable_detailed_profiling: bool,
    pub enable_resource_tracking: bool,
    pub enable_network_metrics: bool,
    pub retention_duration: Duration,
    pub export_interval: Duration,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            max_stored_metrics: 10000,
            sampling_interval: Duration::from_millis(100),
            latency_histogram_buckets: vec![
                Duration::from_micros(100),
                Duration::from_millis(1),
                Duration::from_millis(10),
                Duration::from_millis(100),
                Duration::from_secs(1),
                Duration::from_secs(10),
            ],
            enable_detailed_profiling: true,
            enable_resource_tracking: true,
            enable_network_metrics: false,
            retention_duration: Duration::from_secs(3600), // 1 hour
            export_interval: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AggregatedStats {
    pub total_tasks: u64,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub average_duration: Duration,
    pub total_cpu_time: Duration,
    pub peak_memory_usage: u64,
    pub total_throughput: f64,
    pub average_latency: Duration,
    pub error_rate: f64,
    pub last_updated: SystemTime,
}

impl Default for AggregatedStats {
    fn default() -> Self {
        Self {
            total_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            average_duration: Duration::ZERO,
            total_cpu_time: Duration::ZERO,
            peak_memory_usage: 0,
            total_throughput: 0.0,
            average_latency: Duration::ZERO,
            error_rate: 0.0,
            last_updated: SystemTime::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum MetricsEvent {
    TaskStarted { task_id: String, strategy: String },
    TaskCompleted { task_id: String, metrics: ExecutionMetrics },
    TaskFailed { task_id: String, error: String },
    PerformanceAlert { alert_type: AlertType, details: String },
    ResourceThresholdExceeded { resource: String, current: f64, threshold: f64 },
    StatsUpdated { stats: AggregatedStats },
}

#[derive(Debug, Clone, Serialize)]
pub enum AlertType {
    HighLatency,
    HighErrorRate,
    MemoryLeak,
    CpuSpike,
    DiskSpaceWarning,
    NetworkSaturation,
    Custom(String),
}

impl MetricsCollector {
    pub fn new(config: MetricsConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            metrics_store: Arc::new(RwLock::new(VecDeque::with_capacity(config.max_stored_metrics))),
            active_tasks: Arc::new(Mutex::new(HashMap::new())),
            aggregated_stats: Arc::new(RwLock::new(AggregatedStats::default())),
            event_sender,
            config,
        }
    }

    pub fn start_task(&self, task_id: String, strategy_name: String) -> SklResult<()> {
        let now = Instant::now();
        let system_now = SystemTime::now();

        let task_metrics = TaskMetrics {
            start_time: now,
            start_system_time: system_now,
            strategy_name: strategy_name.clone(),
            resource_tracker: ResourceTracker::new()?,
            performance_tracker: PerformanceTracker::new(),
            custom_data: HashMap::new(),
        };

        self.active_tasks.lock().unwrap().insert(task_id.clone(), task_metrics);

        let _ = self.event_sender.send(MetricsEvent::TaskStarted {
            task_id,
            strategy: strategy_name,
        });

        Ok(())
    }

    pub fn complete_task(&self, task_id: String, status: TaskStatus, error_details: Option<String>) -> SklResult<ExecutionMetrics> {
        let mut active_tasks = self.active_tasks.lock().unwrap();
        let task_metrics = active_tasks.remove(&task_id)
            .ok_or_else(|| scirs2_core::error::CoreError::InvalidInput(format!("Task {} not found", task_id)))?;

        let end_time = SystemTime::now();
        let duration = end_time.duration_since(task_metrics.start_system_time).ok();

        let resource_usage = task_metrics.resource_tracker.finalize()?;
        let performance_data = task_metrics.performance_tracker.finalize(task_metrics.start_time.elapsed())?;

        let metrics = ExecutionMetrics {
            task_id: task_id.clone(),
            strategy_name: task_metrics.strategy_name,
            start_time: task_metrics.start_system_time,
            end_time: Some(end_time),
            duration,
            status: status.clone(),
            resource_usage,
            performance_data,
            error_details: error_details.clone(),
            metadata: task_metrics.custom_data,
        };

        self.store_metrics(metrics.clone())?;
        self.update_aggregated_stats(&metrics)?;

        let event = match status {
            TaskStatus::Completed => MetricsEvent::TaskCompleted { task_id, metrics: metrics.clone() },
            TaskStatus::Failed => MetricsEvent::TaskFailed {
                task_id,
                error: error_details.unwrap_or_else(|| "Unknown error".to_string())
            },
            _ => MetricsEvent::TaskCompleted { task_id, metrics: metrics.clone() },
        };

        let _ = self.event_sender.send(event);

        Ok(metrics)
    }

    pub fn record_operation(&self, task_id: &str, operation_count: u64, latency: Duration) -> SklResult<()> {
        let mut active_tasks = self.active_tasks.lock().unwrap();
        if let Some(task_metrics) = active_tasks.get_mut(task_id) {
            task_metrics.performance_tracker.record_operation(operation_count, latency);
        }
        Ok(())
    }

    pub fn record_error(&self, task_id: &str) -> SklResult<()> {
        let mut active_tasks = self.active_tasks.lock().unwrap();
        if let Some(task_metrics) = active_tasks.get_mut(task_id) {
            task_metrics.performance_tracker.record_error();
        }
        Ok(())
    }

    pub fn record_retry(&self, task_id: &str) -> SklResult<()> {
        let mut active_tasks = self.active_tasks.lock().unwrap();
        if let Some(task_metrics) = active_tasks.get_mut(task_id) {
            task_metrics.performance_tracker.record_retry();
        }
        Ok(())
    }

    pub fn record_cache_hit(&self, task_id: &str) -> SklResult<()> {
        let mut active_tasks = self.active_tasks.lock().unwrap();
        if let Some(task_metrics) = active_tasks.get_mut(task_id) {
            task_metrics.performance_tracker.record_cache_hit();
        }
        Ok(())
    }

    pub fn record_cache_miss(&self, task_id: &str) -> SklResult<()> {
        let mut active_tasks = self.active_tasks.lock().unwrap();
        if let Some(task_metrics) = active_tasks.get_mut(task_id) {
            task_metrics.performance_tracker.record_cache_miss();
        }
        Ok(())
    }

    pub fn add_custom_metric(&self, task_id: &str, key: String, value: f64) -> SklResult<()> {
        let mut active_tasks = self.active_tasks.lock().unwrap();
        if let Some(task_metrics) = active_tasks.get_mut(task_id) {
            task_metrics.performance_tracker.add_custom_metric(key, value);
        }
        Ok(())
    }

    pub fn add_custom_data(&self, task_id: &str, key: String, value: String) -> SklResult<()> {
        let mut active_tasks = self.active_tasks.lock().unwrap();
        if let Some(task_metrics) = active_tasks.get_mut(task_id) {
            task_metrics.custom_data.insert(key, value);
        }
        Ok(())
    }

    pub fn get_aggregated_stats(&self) -> AggregatedStats {
        self.aggregated_stats.read().unwrap().clone()
    }

    pub fn get_recent_metrics(&self, count: usize) -> Vec<ExecutionMetrics> {
        let store = self.metrics_store.read().unwrap();
        store.iter().rev().take(count).cloned().collect()
    }

    pub fn get_metrics_by_strategy(&self, strategy_name: &str) -> Vec<ExecutionMetrics> {
        let store = self.metrics_store.read().unwrap();
        store.iter()
            .filter(|m| m.strategy_name == strategy_name)
            .cloned()
            .collect()
    }

    pub fn get_metrics_in_range(&self, start: SystemTime, end: SystemTime) -> Vec<ExecutionMetrics> {
        let store = self.metrics_store.read().unwrap();
        store.iter()
            .filter(|m| m.start_time >= start && m.start_time <= end)
            .cloned()
            .collect()
    }

    pub fn subscribe_to_events(&self) -> broadcast::Receiver<MetricsEvent> {
        self.event_sender.subscribe()
    }

    pub fn export_metrics(&self) -> SklResult<String> {
        let store = self.metrics_store.read().unwrap();
        let stats = self.aggregated_stats.read().unwrap();

        let export_data = MetricsExport {
            aggregated_stats: stats.clone(),
            recent_metrics: store.iter().cloned().collect(),
            export_timestamp: SystemTime::now(),
        };

        serde_json::to_string_pretty(&export_data)
            .map_err(|e| scirs2_core::error::CoreError::SerializationError(e.to_string()))
    }

    pub fn clear_old_metrics(&self) -> SklResult<usize> {
        let mut store = self.metrics_store.write().unwrap();
        let cutoff = SystemTime::now() - self.config.retention_duration;
        let initial_len = store.len();

        store.retain(|m| m.start_time >= cutoff);

        Ok(initial_len - store.len())
    }

    fn store_metrics(&self, metrics: ExecutionMetrics) -> SklResult<()> {
        let mut store = self.metrics_store.write().unwrap();

        if store.len() >= self.config.max_stored_metrics {
            store.pop_front();
        }

        store.push_back(metrics);
        Ok(())
    }

    fn update_aggregated_stats(&self, metrics: &ExecutionMetrics) -> SklResult<()> {
        let mut stats = self.aggregated_stats.write().unwrap();

        stats.total_tasks += 1;

        match metrics.status {
            TaskStatus::Completed => stats.completed_tasks += 1,
            TaskStatus::Failed => stats.failed_tasks += 1,
            _ => {}
        }

        if let Some(duration) = metrics.duration {
            let total_duration = stats.average_duration * (stats.total_tasks - 1) as u32 + duration;
            stats.average_duration = total_duration / stats.total_tasks as u32;
        }

        stats.peak_memory_usage = stats.peak_memory_usage.max(metrics.resource_usage.memory_peak_bytes);

        let new_latency = metrics.performance_data.latency_percentiles.mean;
        stats.average_latency = (stats.average_latency * (stats.total_tasks - 1) as u32 + new_latency) / stats.total_tasks as u32;

        stats.error_rate = stats.failed_tasks as f64 / stats.total_tasks as f64;
        stats.total_throughput += metrics.performance_data.throughput_ops_per_sec;
        stats.last_updated = SystemTime::now();

        let _ = self.event_sender.send(MetricsEvent::StatsUpdated { stats: stats.clone() });

        Ok(())
    }
}

impl ResourceTracker {
    fn new() -> SklResult<Self> {
        Ok(Self {
            initial_memory: Self::get_current_memory_usage()?,
            peak_memory: 0,
            cpu_samples: VecDeque::with_capacity(100),
            network_start_sent: Self::get_network_bytes_sent()?,
            network_start_received: Self::get_network_bytes_received()?,
            disk_start_read: Self::get_disk_bytes_read()?,
            disk_start_written: Self::get_disk_bytes_written()?,
        })
    }

    fn finalize(mut self) -> SklResult<ResourceUsage> {
        let current_memory = Self::get_current_memory_usage()?;
        self.peak_memory = self.peak_memory.max(current_memory);

        let avg_cpu = if self.cpu_samples.is_empty() {
            0.0
        } else {
            self.cpu_samples.iter().sum::<f64>() / self.cpu_samples.len() as f64
        };

        Ok(ResourceUsage {
            cpu_usage_percent: avg_cpu,
            memory_usage_bytes: current_memory,
            memory_peak_bytes: self.peak_memory,
            gpu_usage_percent: Self::get_gpu_usage()?,
            gpu_memory_bytes: Self::get_gpu_memory()?,
            network_bytes_sent: Self::get_network_bytes_sent()? - self.network_start_sent,
            network_bytes_received: Self::get_network_bytes_received()? - self.network_start_received,
            disk_bytes_read: Self::get_disk_bytes_read()? - self.disk_start_read,
            disk_bytes_written: Self::get_disk_bytes_written()? - self.disk_start_written,
            file_handles_used: Self::get_file_handle_count()?,
            thread_count: Self::get_thread_count()?,
        })
    }

    fn get_current_memory_usage() -> SklResult<u64> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let contents = fs::read_to_string("/proc/self/status")
                .map_err(|e| scirs2_core::error::CoreError::IoError(e.to_string()))?;

            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse::<u64>()
                            .map(|kb| kb * 1024)
                            .map_err(|e| scirs2_core::error::CoreError::ParseError(e.to_string()));
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            let output = Command::new("ps")
                .args(&["-o", "rss=", "-p"])
                .arg(std::process::id().to_string())
                .output()
                .map_err(|e| scirs2_core::error::CoreError::IoError(e.to_string()))?;

            let rss_str = String::from_utf8_lossy(&output.stdout);
            return rss_str.trim().parse::<u64>()
                .map(|kb| kb * 1024)
                .map_err(|e| scirs2_core::error::CoreError::ParseError(e.to_string()));
        }

        Ok(0) // Fallback for unsupported platforms
    }

    fn get_gpu_usage() -> SklResult<Option<f64>> {
        Ok(None) // Placeholder - would integrate with CUDA/OpenCL APIs
    }

    fn get_gpu_memory() -> SklResult<Option<u64>> {
        Ok(None) // Placeholder - would integrate with CUDA/OpenCL APIs
    }

    fn get_network_bytes_sent() -> SklResult<u64> {
        Ok(0) // Placeholder - would read from /proc/net/dev or equivalent
    }

    fn get_network_bytes_received() -> SklResult<u64> {
        Ok(0) // Placeholder - would read from /proc/net/dev or equivalent
    }

    fn get_disk_bytes_read() -> SklResult<u64> {
        Ok(0) // Placeholder - would read from /proc/self/io or equivalent
    }

    fn get_disk_bytes_written() -> SklResult<u64> {
        Ok(0) // Placeholder - would read from /proc/self/io or equivalent
    }

    fn get_file_handle_count() -> SklResult<u32> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let entries = fs::read_dir("/proc/self/fd")
                .map_err(|e| scirs2_core::error::CoreError::IoError(e.to_string()))?;
            return Ok(entries.count() as u32);
        }

        Ok(0) // Fallback for unsupported platforms
    }

    fn get_thread_count() -> SklResult<u32> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let contents = fs::read_to_string("/proc/self/status")
                .map_err(|e| scirs2_core::error::CoreError::IoError(e.to_string()))?;

            for line in contents.lines() {
                if line.starts_with("Threads:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse::<u32>()
                            .map_err(|e| scirs2_core::error::CoreError::ParseError(e.to_string()));
                    }
                }
            }
        }

        Ok(1) // Fallback
    }
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            operation_count: 0,
            latency_samples: VecDeque::with_capacity(1000),
            error_count: 0,
            retry_count: 0,
            cache_hits: 0,
            cache_misses: 0,
            custom_counters: HashMap::new(),
        }
    }

    fn record_operation(&mut self, count: u64, latency: Duration) {
        self.operation_count += count;
        if self.latency_samples.len() >= 1000 {
            self.latency_samples.pop_front();
        }
        self.latency_samples.push_back(latency);
    }

    fn record_error(&mut self) {
        self.error_count += 1;
    }

    fn record_retry(&mut self) {
        self.retry_count += 1;
    }

    fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    fn add_custom_metric(&mut self, key: String, value: f64) {
        self.custom_counters.insert(key, value);
    }

    fn finalize(self, total_duration: Duration) -> SklResult<PerformanceData> {
        let throughput = if total_duration.as_secs_f64() > 0.0 {
            self.operation_count as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let latency_percentiles = self.calculate_latency_percentiles()?;

        let total_cache_ops = self.cache_hits + self.cache_misses;
        let cache_hit_rate = if total_cache_ops > 0 {
            self.cache_hits as f64 / total_cache_ops as f64
        } else {
            0.0
        };

        let error_rate = if self.operation_count > 0 {
            self.error_count as f64 / self.operation_count as f64
        } else {
            0.0
        };

        Ok(PerformanceData {
            throughput_ops_per_sec: throughput,
            latency_percentiles,
            queue_depth: 0, // Would be tracked separately
            batch_size: 0,  // Would be tracked separately
            parallelism_factor: 1.0, // Would be tracked separately
            cache_hit_rate,
            error_rate,
            retry_count: self.retry_count,
            custom_metrics: self.custom_counters,
        })
    }

    fn calculate_latency_percentiles(&self) -> SklResult<LatencyPercentiles> {
        if self.latency_samples.is_empty() {
            return Ok(LatencyPercentiles {
                p50: Duration::ZERO,
                p90: Duration::ZERO,
                p95: Duration::ZERO,
                p99: Duration::ZERO,
                p999: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                mean: Duration::ZERO,
            });
        }

        let mut sorted_samples: Vec<Duration> = self.latency_samples.iter().cloned().collect();
        sorted_samples.sort();

        let len = sorted_samples.len();
        let p50_idx = (len as f64 * 0.50) as usize;
        let p90_idx = (len as f64 * 0.90) as usize;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;
        let p999_idx = (len as f64 * 0.999) as usize;

        let total_nanos: u64 = sorted_samples.iter().map(|d| d.as_nanos() as u64).sum();
        let mean_nanos = total_nanos / len as u64;

        Ok(LatencyPercentiles {
            p50: sorted_samples[p50_idx.min(len - 1)],
            p90: sorted_samples[p90_idx.min(len - 1)],
            p95: sorted_samples[p95_idx.min(len - 1)],
            p99: sorted_samples[p99_idx.min(len - 1)],
            p999: sorted_samples[p999_idx.min(len - 1)],
            min: sorted_samples[0],
            max: sorted_samples[len - 1],
            mean: Duration::from_nanos(mean_nanos),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct MetricsExport {
    aggregated_stats: AggregatedStats,
    recent_metrics: Vec<ExecutionMetrics>,
    export_timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub struct MetricsReporter {
    collector: Arc<MetricsCollector>,
    config: ReporterConfig,
}

#[derive(Debug, Clone)]
pub struct ReporterConfig {
    pub console_output: bool,
    pub file_output: Option<String>,
    pub http_endpoint: Option<String>,
    pub prometheus_format: bool,
    pub update_interval: Duration,
}

impl Default for ReporterConfig {
    fn default() -> Self {
        Self {
            console_output: true,
            file_output: None,
            http_endpoint: None,
            prometheus_format: false,
            update_interval: Duration::from_secs(30),
        }
    }
}

impl MetricsReporter {
    pub fn new(collector: Arc<MetricsCollector>, config: ReporterConfig) -> Self {
        Self { collector, config }
    }

    pub async fn start_reporting(&self) -> SklResult<()> {
        let mut interval = tokio::time::interval(self.config.update_interval);
        let mut event_receiver = self.collector.subscribe_to_events();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    self.generate_periodic_report().await?;
                }
                event = event_receiver.recv() => {
                    match event {
                        Ok(MetricsEvent::PerformanceAlert { alert_type, details }) => {
                            self.handle_alert(alert_type, details).await?;
                        }
                        Ok(_) => {} // Handle other events if needed
                        Err(_) => break, // Channel closed
                    }
                }
            }
        }

        Ok(())
    }

    async fn generate_periodic_report(&self) -> SklResult<()> {
        let stats = self.collector.get_aggregated_stats();
        let recent_metrics = self.collector.get_recent_metrics(100);

        if self.config.console_output {
            self.print_console_report(&stats, &recent_metrics);
        }

        if let Some(file_path) = &self.config.file_output {
            self.write_file_report(file_path, &stats, &recent_metrics).await?;
        }

        if let Some(endpoint) = &self.config.http_endpoint {
            self.send_http_report(endpoint, &stats, &recent_metrics).await?;
        }

        Ok(())
    }

    fn print_console_report(&self, stats: &AggregatedStats, recent_metrics: &[ExecutionMetrics]) {
        println!("\n=== Execution Metrics Report ===");
        println!("Total Tasks: {}", stats.total_tasks);
        println!("Completed: {} ({:.1}%)",
                stats.completed_tasks,
                (stats.completed_tasks as f64 / stats.total_tasks as f64) * 100.0);
        println!("Failed: {} ({:.1}%)",
                stats.failed_tasks,
                (stats.failed_tasks as f64 / stats.total_tasks as f64) * 100.0);
        println!("Average Duration: {:.2}s", stats.average_duration.as_secs_f64());
        println!("Average Latency: {:.2}ms", stats.average_latency.as_millis());
        println!("Peak Memory: {:.2} MB", stats.peak_memory_usage as f64 / 1024.0 / 1024.0);
        println!("Total Throughput: {:.2} ops/sec", stats.total_throughput);
        println!("Error Rate: {:.2}%", stats.error_rate * 100.0);

        if !recent_metrics.is_empty() {
            println!("\n=== Recent Task Summary ===");
            for metric in recent_metrics.iter().take(5) {
                let duration_str = metric.duration
                    .map(|d| format!("{:.2}s", d.as_secs_f64()))
                    .unwrap_or_else(|| "N/A".to_string());
                println!("{}: {} - {} ({})",
                        metric.task_id,
                        metric.strategy_name,
                        format!("{:?}", metric.status),
                        duration_str);
            }
        }
        println!("================================\n");
    }

    async fn write_file_report(&self, file_path: &str, _stats: &AggregatedStats, _recent_metrics: &[ExecutionMetrics]) -> SklResult<()> {
        let export_data = self.collector.export_metrics()?;
        tokio::fs::write(file_path, export_data).await
            .map_err(|e| scirs2_core::error::CoreError::IoError(e.to_string()))?;
        Ok(())
    }

    async fn send_http_report(&self, _endpoint: &str, _stats: &AggregatedStats, _recent_metrics: &[ExecutionMetrics]) -> SklResult<()> {
        // Placeholder for HTTP reporting implementation
        Ok(())
    }

    async fn handle_alert(&self, alert_type: AlertType, details: String) -> SklResult<()> {
        println!("🚨 PERFORMANCE ALERT: {:?} - {}", alert_type, details);

        // Additional alert handling logic (notifications, logging, etc.)

        Ok(())
    }
}

pub fn create_default_metrics_system() -> SklResult<(Arc<MetricsCollector>, MetricsReporter)> {
    let config = MetricsConfig::default();
    let collector = Arc::new(MetricsCollector::new(config));

    let reporter_config = ReporterConfig::default();
    let reporter = MetricsReporter::new(collector.clone(), reporter_config);

    Ok((collector, reporter))
}
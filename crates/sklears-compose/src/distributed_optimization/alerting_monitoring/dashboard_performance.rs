use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Rendering engine
pub struct RenderingEngine {
    /// Rendering workers
    pub workers: Vec<RenderingWorker>,
    /// Render queue
    pub render_queue: VecDeque<RenderTask>,
    /// Template cache
    pub template_cache: HashMap<String, Template>,
    /// Asset cache
    pub asset_cache: HashMap<String, Asset>,
}

/// Rendering worker
pub struct RenderingWorker {
    pub worker_id: String,
    pub status: WorkerStatus,
    pub current_task: Option<String>,
    pub performance_stats: RenderingStats,
}

/// Worker status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStatus {
    Idle,
    Busy,
    Error,
    Maintenance,
}

/// Rendering statistics
#[derive(Debug, Clone)]
pub struct RenderingStats {
    pub tasks_completed: u64,
    pub average_render_time: Duration,
    pub error_count: u64,
    pub cache_hit_rate: f64,
}

/// Render task
#[derive(Debug, Clone)]
pub struct RenderTask {
    pub task_id: String,
    pub task_type: RenderTaskType,
    pub dashboard_id: String,
    pub widget_id: Option<String>,
    pub priority: u32,
    pub created_at: SystemTime,
    pub deadline: Option<SystemTime>,
}

/// Render task types
#[derive(Debug, Clone)]
pub enum RenderTaskType {
    Dashboard,
    Widget,
    Chart,
    Export,
    Thumbnail,
}

/// Template
#[derive(Debug, Clone)]
pub struct Template {
    pub template_id: String,
    pub template_type: TemplateType,
    pub content: String,
    pub compiled: bool,
    pub created_at: SystemTime,
    pub last_used: SystemTime,
}

/// Template types
#[derive(Debug, Clone)]
pub enum TemplateType {
    Dashboard,
    Widget,
    Chart,
    Email,
    Report,
}

/// Asset
#[derive(Debug, Clone)]
pub struct Asset {
    pub asset_id: String,
    pub asset_type: AssetType,
    pub content: Vec<u8>,
    pub content_type: String,
    pub size: u64,
    pub last_accessed: SystemTime,
}

/// Asset types
#[derive(Debug, Clone)]
pub enum AssetType {
    Image,
    Font,
    Script,
    Stylesheet,
    Icon,
}

/// Cache manager
pub struct CacheManager {
    /// Memory cache
    pub memory_cache: HashMap<String, CacheEntry>,
    /// Disk cache
    pub disk_cache: HashMap<String, String>,
    /// Cache statistics
    pub cache_stats: CacheStatistics,
    /// Cache configuration
    pub cache_config: CacheConfiguration,
}

/// Cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub created_at: SystemTime,
    pub expires_at: Option<SystemTime>,
    pub access_count: u64,
    pub last_accessed: SystemTime,
    pub size: u64,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub total_entries: u64,
    pub memory_usage: u64,
    pub disk_usage: u64,
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub eviction_count: u64,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfiguration {
    pub max_memory_size: u64,
    pub max_disk_size: u64,
    pub default_ttl: Duration,
    pub eviction_policy: String,
    pub compression_enabled: bool,
}

/// Performance monitor
pub struct PerformanceMonitor {
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Alert thresholds
    pub thresholds: PerformanceThresholds,
    /// Monitoring configuration
    pub config: MonitoringConfiguration,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub response_times: Vec<Duration>,
    pub throughput: f64,
    pub error_rates: HashMap<String, f64>,
    pub resource_usage: ResourceUsageMetrics,
    pub user_experience: UserExperienceMetrics,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceUsageMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_usage: f64,
    pub database_connections: u32,
}

/// User experience metrics
#[derive(Debug, Clone)]
pub struct UserExperienceMetrics {
    pub page_load_time: Duration,
    pub time_to_interactive: Duration,
    pub first_contentful_paint: Duration,
    pub cumulative_layout_shift: f64,
    pub bounce_rate: f64,
}

/// Performance thresholds
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub max_response_time: Duration,
    pub max_error_rate: f64,
    pub max_cpu_usage: f64,
    pub max_memory_usage: f64,
    pub min_availability: f64,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfiguration {
    pub enabled: bool,
    pub collection_interval: Duration,
    pub retention_period: Duration,
    pub alert_enabled: bool,
    pub dashboard_monitoring: bool,
}

impl RenderingEngine {
    pub fn new() -> Self {
        Self {
            workers: Vec::new(),
            render_queue: VecDeque::new(),
            template_cache: HashMap::new(),
            asset_cache: HashMap::new(),
        }
    }

    pub fn add_worker(&mut self, worker: RenderingWorker) {
        self.workers.push(worker);
    }

    pub fn submit_task(&mut self, task: RenderTask) {
        self.render_queue.push_back(task);
    }

    pub fn get_next_task(&mut self) -> Option<RenderTask> {
        self.render_queue.pop_front()
    }

    pub fn cache_template(&mut self, template: Template) {
        self.template_cache.insert(template.template_id.clone(), template);
    }

    pub fn get_template(&self, template_id: &str) -> Option<&Template> {
        self.template_cache.get(template_id)
    }

    pub fn cache_asset(&mut self, asset: Asset) {
        self.asset_cache.insert(asset.asset_id.clone(), asset);
    }

    pub fn get_asset(&self, asset_id: &str) -> Option<&Asset> {
        self.asset_cache.get(asset_id)
    }

    pub fn get_worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn get_queue_size(&self) -> usize {
        self.render_queue.len()
    }

    pub fn get_idle_workers(&self) -> Vec<&RenderingWorker> {
        self.workers.iter().filter(|w| w.status == WorkerStatus::Idle).collect()
    }

    pub fn get_busy_workers(&self) -> Vec<&RenderingWorker> {
        self.workers.iter().filter(|w| w.status == WorkerStatus::Busy).collect()
    }
}

impl RenderingWorker {
    pub fn new(worker_id: String) -> Self {
        Self {
            worker_id,
            status: WorkerStatus::Idle,
            current_task: None,
            performance_stats: RenderingStats {
                tasks_completed: 0,
                average_render_time: Duration::from_secs(0),
                error_count: 0,
                cache_hit_rate: 0.0,
            },
        }
    }

    pub fn assign_task(&mut self, task_id: String) {
        self.status = WorkerStatus::Busy;
        self.current_task = Some(task_id);
    }

    pub fn complete_task(&mut self, render_time: Duration) {
        self.status = WorkerStatus::Idle;
        self.current_task = None;
        self.performance_stats.tasks_completed += 1;

        // Update average render time
        let total_time = self.performance_stats.average_render_time
            .mul_f64(self.performance_stats.tasks_completed as f64 - 1.0)
            + render_time;
        self.performance_stats.average_render_time = total_time
            .div_f64(self.performance_stats.tasks_completed as f64);
    }

    pub fn report_error(&mut self) {
        self.status = WorkerStatus::Error;
        self.current_task = None;
        self.performance_stats.error_count += 1;
    }

    pub fn reset(&mut self) {
        self.status = WorkerStatus::Idle;
        self.current_task = None;
    }
}

impl CacheManager {
    pub fn new() -> Self {
        Self {
            memory_cache: HashMap::new(),
            disk_cache: HashMap::new(),
            cache_stats: CacheStatistics {
                total_entries: 0,
                memory_usage: 0,
                disk_usage: 0,
                hit_rate: 0.0,
                miss_rate: 0.0,
                eviction_count: 0,
            },
            cache_config: CacheConfiguration {
                max_memory_size: 1024 * 1024 * 1024, // 1GB
                max_disk_size: 10 * 1024 * 1024 * 1024, // 10GB
                default_ttl: Duration::from_secs(3600), // 1 hour
                eviction_policy: "LRU".to_string(),
                compression_enabled: true,
            },
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        if let Some(entry) = self.memory_cache.get_mut(key) {
            entry.access_count += 1;
            entry.last_accessed = SystemTime::now();

            // Check if expired
            if let Some(expires_at) = entry.expires_at {
                if expires_at < SystemTime::now() {
                    self.memory_cache.remove(key);
                    return None;
                }
            }

            Some(entry)
        } else {
            None
        }
    }

    pub fn put(&mut self, key: String, value: Vec<u8>, ttl: Option<Duration>) {
        let expires_at = ttl.map(|t| SystemTime::now() + t);
        let size = value.len() as u64;

        let entry = CacheEntry {
            key: key.clone(),
            value,
            created_at: SystemTime::now(),
            expires_at,
            access_count: 0,
            last_accessed: SystemTime::now(),
            size,
        };

        // Check memory limits
        if self.cache_stats.memory_usage + size > self.cache_config.max_memory_size {
            self.evict_entries();
        }

        self.memory_cache.insert(key, entry);
        self.cache_stats.total_entries += 1;
        self.cache_stats.memory_usage += size;
    }

    pub fn remove(&mut self, key: &str) -> Option<CacheEntry> {
        if let Some(entry) = self.memory_cache.remove(key) {
            self.cache_stats.total_entries -= 1;
            self.cache_stats.memory_usage -= entry.size;
            Some(entry)
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.memory_cache.clear();
        self.disk_cache.clear();
        self.cache_stats.total_entries = 0;
        self.cache_stats.memory_usage = 0;
        self.cache_stats.disk_usage = 0;
    }

    pub fn evict_entries(&mut self) {
        // Simple LRU eviction
        let mut entries: Vec<_> = self.memory_cache.iter().collect();
        entries.sort_by_key(|(_, entry)| entry.last_accessed);

        // Remove oldest 25% of entries
        let remove_count = entries.len() / 4;
        for (key, _) in entries.iter().take(remove_count) {
            if let Some(entry) = self.memory_cache.remove(*key) {
                self.cache_stats.memory_usage -= entry.size;
                self.cache_stats.total_entries -= 1;
                self.cache_stats.eviction_count += 1;
            }
        }
    }

    pub fn get_stats(&self) -> &CacheStatistics {
        &self.cache_stats
    }

    pub fn update_hit_rate(&mut self, hits: u64, misses: u64) {
        let total = hits + misses;
        if total > 0 {
            self.cache_stats.hit_rate = hits as f64 / total as f64;
            self.cache_stats.miss_rate = misses as f64 / total as f64;
        }
    }
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: PerformanceMetrics {
                response_times: Vec::new(),
                throughput: 0.0,
                error_rates: HashMap::new(),
                resource_usage: ResourceUsageMetrics {
                    cpu_usage: 0.0,
                    memory_usage: 0.0,
                    disk_usage: 0.0,
                    network_usage: 0.0,
                    database_connections: 0,
                },
                user_experience: UserExperienceMetrics {
                    page_load_time: Duration::from_secs(0),
                    time_to_interactive: Duration::from_secs(0),
                    first_contentful_paint: Duration::from_secs(0),
                    cumulative_layout_shift: 0.0,
                    bounce_rate: 0.0,
                },
            },
            thresholds: PerformanceThresholds {
                max_response_time: Duration::from_secs(5),
                max_error_rate: 0.01,
                max_cpu_usage: 80.0,
                max_memory_usage: 80.0,
                min_availability: 99.9,
            },
            config: MonitoringConfiguration {
                enabled: true,
                collection_interval: Duration::from_secs(60),
                retention_period: Duration::from_secs(86400 * 7), // 7 days
                alert_enabled: true,
                dashboard_monitoring: true,
            },
        }
    }

    pub fn record_response_time(&mut self, response_time: Duration) {
        self.metrics.response_times.push(response_time);

        // Keep only recent measurements (e.g., last 1000)
        if self.metrics.response_times.len() > 1000 {
            self.metrics.response_times.remove(0);
        }
    }

    pub fn update_throughput(&mut self, throughput: f64) {
        self.metrics.throughput = throughput;
    }

    pub fn record_error(&mut self, error_type: String) {
        *self.metrics.error_rates.entry(error_type).or_insert(0.0) += 1.0;
    }

    pub fn update_resource_usage(&mut self, resource_usage: ResourceUsageMetrics) {
        self.metrics.resource_usage = resource_usage;
    }

    pub fn update_user_experience(&mut self, user_experience: UserExperienceMetrics) {
        self.metrics.user_experience = user_experience;
    }

    pub fn get_average_response_time(&self) -> Duration {
        if self.metrics.response_times.is_empty() {
            return Duration::from_secs(0);
        }

        let total: Duration = self.metrics.response_times.iter().sum();
        total / self.metrics.response_times.len() as u32
    }

    pub fn get_95th_percentile_response_time(&self) -> Duration {
        if self.metrics.response_times.is_empty() {
            return Duration::from_secs(0);
        }

        let mut sorted_times = self.metrics.response_times.clone();
        sorted_times.sort();

        let index = (sorted_times.len() as f64 * 0.95) as usize;
        sorted_times.get(index).copied().unwrap_or(Duration::from_secs(0))
    }

    pub fn check_thresholds(&self) -> Vec<PerformanceAlert> {
        let mut alerts = Vec::new();

        // Check response time
        let avg_response_time = self.get_average_response_time();
        if avg_response_time > self.thresholds.max_response_time {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::ResponseTime,
                message: format!("Average response time ({:?}) exceeds threshold ({:?})",
                    avg_response_time, self.thresholds.max_response_time),
                severity: AlertSeverity::Warning,
                timestamp: SystemTime::now(),
            });
        }

        // Check CPU usage
        if self.metrics.resource_usage.cpu_usage > self.thresholds.max_cpu_usage {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::CpuUsage,
                message: format!("CPU usage ({:.1}%) exceeds threshold ({:.1}%)",
                    self.metrics.resource_usage.cpu_usage, self.thresholds.max_cpu_usage),
                severity: AlertSeverity::Warning,
                timestamp: SystemTime::now(),
            });
        }

        // Check memory usage
        if self.metrics.resource_usage.memory_usage > self.thresholds.max_memory_usage {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::MemoryUsage,
                message: format!("Memory usage ({:.1}%) exceeds threshold ({:.1}%)",
                    self.metrics.resource_usage.memory_usage, self.thresholds.max_memory_usage),
                severity: AlertSeverity::Warning,
                timestamp: SystemTime::now(),
            });
        }

        alerts
    }

    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    pub fn get_thresholds(&self) -> &PerformanceThresholds {
        &self.thresholds
    }

    pub fn update_thresholds(&mut self, thresholds: PerformanceThresholds) {
        self.thresholds = thresholds;
    }
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub alert_type: AlertType,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: SystemTime,
}

/// Alert types
#[derive(Debug, Clone)]
pub enum AlertType {
    ResponseTime,
    CpuUsage,
    MemoryUsage,
    DiskUsage,
    NetworkUsage,
    ErrorRate,
    Availability,
}

/// Alert severity
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl Default for RenderingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}
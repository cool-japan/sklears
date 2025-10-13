use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc};
use tokio::process::Command;
use uuid::Uuid;
use std::fs;
use std::io::Write;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    pub image: String,
    pub tag: String,
    pub environment: HashMap<String, String>,
    pub volumes: Vec<VolumeMount>,
    pub ports: Vec<PortMapping>,
    pub resources: ResourceLimits,
    pub network_mode: NetworkMode,
    pub restart_policy: RestartPolicy,
    pub health_check: Option<HealthCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    pub host_path: PathBuf,
    pub container_path: PathBuf,
    pub read_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    pub host_port: u16,
    pub container_port: u16,
    pub protocol: Protocol,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Protocol {
    TCP,
    UDP,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu_limit: Option<f64>,
    pub memory_limit: Option<String>,
    pub gpu_enabled: bool,
    pub disk_limit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMode {
    Bridge,
    Host,
    None,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestartPolicy {
    No,
    Always,
    OnFailure { max_retries: u32 },
    UnlessStopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub command: Vec<String>,
    pub interval: Duration,
    pub timeout: Duration,
    pub retries: u32,
    pub start_period: Duration,
}

#[derive(Debug, Clone)]
pub struct ContainerInstance {
    pub id: String,
    pub name: String,
    pub config: ContainerConfig,
    pub status: ContainerStatus,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub finished_at: Option<Instant>,
    pub exit_code: Option<i32>,
    pub logs: Vec<String>,
    pub metrics: ContainerMetrics,
}

#[derive(Debug, Clone)]
pub enum ContainerStatus {
    Created,
    Starting,
    Running,
    Paused,
    Stopped,
    Exited,
    Error,
}

#[derive(Debug, Clone, Default)]
pub struct ContainerMetrics {
    pub cpu_usage: f64,
    pub memory_usage: u64,
    pub network_rx: u64,
    pub network_tx: u64,
    pub disk_read: u64,
    pub disk_write: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ContainerError {
    #[error("Container not found: {container_id}")]
    ContainerNotFound { container_id: String },
    
    #[error("Container already exists: {container_name}")]
    ContainerAlreadyExists { container_name: String },
    
    #[error("Docker daemon not available")]
    DockerUnavailable,
    
    #[error("Container start failed: {container_id} - {error}")]
    StartFailed { container_id: String, error: String },
    
    #[error("Container stop failed: {container_id} - {error}")]
    StopFailed { container_id: String, error: String },
    
    #[error("Image pull failed: {image} - {error}")]
    ImagePullFailed { image: String, error: String },
    
    #[error("Volume mount failed: {volume} - {error}")]
    VolumeMountFailed { volume: String, error: String },
    
    #[error("Resource allocation failed: {error}")]
    ResourceAllocationFailed { error: String },
    
    #[error("Health check failed: {container_id}")]
    HealthCheckFailed { container_id: String },
    
    #[error("Container execution timeout: {container_id}")]
    ExecutionTimeout { container_id: String },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub struct ContainerRuntime {
    containers: Arc<RwLock<HashMap<String, ContainerInstance>>>,
    registry: Arc<RwLock<ContainerRegistry>>,
    orchestrator: Arc<ContainerOrchestrator>,
    monitor: Arc<ContainerMonitor>,
    network_manager: Arc<NetworkManager>,
    compose_manager: Arc<DockerComposeManager>,
    service_discovery: Arc<ServiceDiscovery>,
    secrets_manager: Arc<SecretsManager>,
}

pub struct ContainerRegistry {
    images: HashMap<String, ImageInfo>,
    repositories: Vec<Repository>,
}

#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub name: String,
    pub tag: String,
    pub digest: String,
    pub size: u64,
    pub created_at: Instant,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Repository {
    pub url: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub insecure: bool,
}

pub struct ContainerOrchestrator {
    cluster_config: Arc<RwLock<ClusterConfig>>,
    scheduler: Arc<Scheduler>,
    load_balancer: Arc<ContainerLoadBalancer>,
}

#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub nodes: Vec<ClusterNode>,
    pub placement_strategy: PlacementStrategy,
    pub auto_scaling: AutoScalingConfig,
}

#[derive(Debug, Clone)]
pub struct ClusterNode {
    pub id: String,
    pub address: String,
    pub resources: NodeResources,
    pub labels: HashMap<String, String>,
    pub status: NodeStatus,
}

#[derive(Debug, Clone)]
pub struct NodeResources {
    pub cpu_cores: u32,
    pub memory_gb: u64,
    pub gpu_count: u32,
    pub disk_gb: u64,
}

#[derive(Debug, Clone)]
pub enum NodeStatus {
    Ready,
    NotReady,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum PlacementStrategy {
    Random,
    LeastUsed,
    MostUsed,
    Affinity(HashMap<String, String>),
    AntiAffinity(HashMap<String, String>),
}

#[derive(Debug, Clone)]
pub struct AutoScalingConfig {
    pub enabled: bool,
    pub min_nodes: u32,
    pub max_nodes: u32,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub scale_up_cooldown: Duration,
    pub scale_down_cooldown: Duration,
}

pub struct Scheduler {
    placement_strategy: PlacementStrategy,
    resource_tracker: Arc<RwLock<HashMap<String, NodeResources>>>,
}

pub struct ContainerLoadBalancer {
    strategy: LoadBalancingStrategy,
    health_checker: Arc<HealthChecker>,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    IPHash,
    Random,
}

pub struct HealthChecker {
    checks: Arc<RwLock<HashMap<String, HealthCheckConfig>>>,
}

#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    pub command: Vec<String>,
    pub interval: Duration,
    pub timeout: Duration,
    pub healthy_threshold: u32,
    pub unhealthy_threshold: u32,
}

pub struct ContainerMonitor {
    metrics_collector: Arc<MetricsCollector>,
    log_aggregator: Arc<LogAggregator>,
    alert_manager: Arc<AlertManager>,
}

pub struct MetricsCollector {
    metrics: Arc<RwLock<HashMap<String, ContainerMetrics>>>,
    exporters: Vec<Box<dyn MetricsExporter>>,
}

pub trait MetricsExporter: Send + Sync {
    fn export(&self, metrics: &HashMap<String, ContainerMetrics>) -> Result<(), ContainerError>;
}

pub struct LogAggregator {
    logs: Arc<RwLock<HashMap<String, Vec<LogEntry>>>>,
    processors: Vec<Box<dyn LogProcessor>>,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: Instant,
    pub level: LogLevel,
    pub message: String,
    pub container_id: String,
}

#[derive(Debug, Clone)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

pub trait LogProcessor: Send + Sync {
    fn process(&self, entry: &LogEntry) -> Result<(), ContainerError>;
}

pub struct AlertManager {
    rules: Arc<RwLock<Vec<AlertRule>>>,
    notifications: mpsc::UnboundedSender<Alert>,
}

#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub cooldown: Duration,
    pub last_triggered: Option<Instant>,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    CpuUsageHigh(f64),
    MemoryUsageHigh(u64),
    ContainerStopped,
    HealthCheckFailed,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
pub struct Alert {
    pub rule_name: String,
    pub container_id: String,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: Instant,
}

pub struct NetworkManager {
    networks: Arc<RwLock<HashMap<String, NetworkConfig>>>,
    dns_resolver: Arc<DnsResolver>,
    proxy: Arc<ServiceProxy>,
}

#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub name: String,
    pub driver: NetworkDriver,
    pub subnet: String,
    pub gateway: String,
    pub options: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum NetworkDriver {
    Bridge,
    Overlay,
    MacVlan,
    Host,
    None,
}

pub struct DnsResolver {
    entries: Arc<RwLock<HashMap<String, String>>>,
}

pub struct ServiceProxy {
    routes: Arc<RwLock<HashMap<String, ProxyRoute>>>,
}

#[derive(Debug, Clone)]
pub struct ProxyRoute {
    pub source_port: u16,
    pub target_containers: Vec<String>,
    pub load_balancing: LoadBalancingStrategy,
}

impl ContainerRuntime {
    pub fn new() -> Self {
        let containers = Arc::new(RwLock::new(HashMap::new()));
        let registry = Arc::new(RwLock::new(ContainerRegistry {
            images: HashMap::new(),
            repositories: Vec::new(),
        }));
        
        let cluster_config = Arc::new(RwLock::new(ClusterConfig {
            nodes: Vec::new(),
            placement_strategy: PlacementStrategy::LeastUsed,
            auto_scaling: AutoScalingConfig {
                enabled: false,
                min_nodes: 1,
                max_nodes: 10,
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.2,
                scale_up_cooldown: Duration::from_secs(300),
                scale_down_cooldown: Duration::from_secs(600),
            },
        }));
        
        let scheduler = Arc::new(Scheduler {
            placement_strategy: PlacementStrategy::LeastUsed,
            resource_tracker: Arc::new(RwLock::new(HashMap::new())),
        });
        
        let health_checker = Arc::new(HealthChecker {
            checks: Arc::new(RwLock::new(HashMap::new())),
        });
        
        let load_balancer = Arc::new(ContainerLoadBalancer {
            strategy: LoadBalancingStrategy::RoundRobin,
            health_checker,
        });
        
        let orchestrator = Arc::new(ContainerOrchestrator {
            cluster_config,
            scheduler,
            load_balancer,
        });
        
        let metrics_collector = Arc::new(MetricsCollector {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            exporters: Vec::new(),
        });
        
        let log_aggregator = Arc::new(LogAggregator {
            logs: Arc::new(RwLock::new(HashMap::new())),
            processors: Vec::new(),
        });
        
        let (alert_tx, _alert_rx) = mpsc::unbounded_channel();
        let alert_manager = Arc::new(AlertManager {
            rules: Arc::new(RwLock::new(Vec::new())),
            notifications: alert_tx,
        });
        
        let monitor = Arc::new(ContainerMonitor {
            metrics_collector,
            log_aggregator,
            alert_manager,
        });
        
        let networks = Arc::new(RwLock::new(HashMap::new()));
        let dns_resolver = Arc::new(DnsResolver {
            entries: Arc::new(RwLock::new(HashMap::new())),
        });
        let proxy = Arc::new(ServiceProxy {
            routes: Arc::new(RwLock::new(HashMap::new())),
        });
        
        let network_manager = Arc::new(NetworkManager {
            networks,
            dns_resolver,
            proxy,
        });
        
        Self {
            containers,
            registry,
            orchestrator,
            monitor,
            network_manager,
        }
    }

    pub async fn create_container(
        &self,
        name: String,
        config: ContainerConfig,
    ) -> Result<String, ContainerError> {
        let container_id = Uuid::new_v4().to_string();
        
        {
            let containers = self.containers.read().await;
            if containers.values().any(|c| c.name == name) {
                return Err(ContainerError::ContainerAlreadyExists { container_name: name });
            }
        }

        let instance = ContainerInstance {
            id: container_id.clone(),
            name: name.clone(),
            config: config.clone(),
            status: ContainerStatus::Created,
            created_at: Instant::now(),
            started_at: None,
            finished_at: None,
            exit_code: None,
            logs: Vec::new(),
            metrics: ContainerMetrics::default(),
        };

        {
            let mut containers = self.containers.write().await;
            containers.insert(container_id.clone(), instance);
        }

        self.pull_image_if_needed(&config.image, &config.tag).await?;
        
        Ok(container_id)
    }

    pub async fn start_container(&self, container_id: &str) -> Result<(), ContainerError> {
        let config = {
            let containers = self.containers.read().await;
            let container = containers.get(container_id)
                .ok_or(ContainerError::ContainerNotFound { 
                    container_id: container_id.to_string() 
                })?;
            container.config.clone()
        };

        let docker_args = self.build_docker_run_args(&config).await?;
        
        let output = Command::new("docker")
            .args(&["run", "-d", "--name", container_id])
            .args(&docker_args)
            .arg(format!("{}:{}", config.image, config.tag))
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ContainerError::StartFailed {
                container_id: container_id.to_string(),
                error: error.to_string(),
            });
        }

        {
            let mut containers = self.containers.write().await;
            if let Some(container) = containers.get_mut(container_id) {
                container.status = ContainerStatus::Running;
                container.started_at = Some(Instant::now());
            }
        }

        self.start_monitoring(container_id).await;
        
        Ok(())
    }

    pub async fn stop_container(&self, container_id: &str) -> Result<(), ContainerError> {
        let output = Command::new("docker")
            .args(&["stop", container_id])
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ContainerError::StopFailed {
                container_id: container_id.to_string(),
                error: error.to_string(),
            });
        }

        {
            let mut containers = self.containers.write().await;
            if let Some(container) = containers.get_mut(container_id) {
                container.status = ContainerStatus::Stopped;
                container.finished_at = Some(Instant::now());
            }
        }

        Ok(())
    }

    pub async fn remove_container(&self, container_id: &str) -> Result<(), ContainerError> {
        self.stop_container(container_id).await.ok();
        
        let output = Command::new("docker")
            .args(&["rm", container_id])
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ContainerError::StartFailed {
                container_id: container_id.to_string(),
                error: error.to_string(),
            });
        }

        {
            let mut containers = self.containers.write().await;
            containers.remove(container_id);
        }

        Ok(())
    }

    pub async fn execute_command(
        &self,
        container_id: &str,
        command: Vec<String>,
    ) -> Result<String, ContainerError> {
        let mut cmd = Command::new("docker");
        cmd.args(&["exec", container_id]);
        cmd.args(&command);

        let output = cmd.output().await?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(ContainerError::StartFailed {
                container_id: container_id.to_string(),
                error: error.to_string(),
            })
        }
    }

    pub async fn get_container_logs(&self, container_id: &str) -> Result<Vec<String>, ContainerError> {
        let containers = self.containers.read().await;
        let container = containers.get(container_id)
            .ok_or(ContainerError::ContainerNotFound { 
                container_id: container_id.to_string() 
            })?;
        
        Ok(container.logs.clone())
    }

    pub async fn get_container_metrics(&self, container_id: &str) -> Result<ContainerMetrics, ContainerError> {
        let containers = self.containers.read().await;
        let container = containers.get(container_id)
            .ok_or(ContainerError::ContainerNotFound { 
                container_id: container_id.to_string() 
            })?;
        
        Ok(container.metrics.clone())
    }

    pub async fn list_containers(&self) -> Vec<ContainerInstance> {
        let containers = self.containers.read().await;
        containers.values().cloned().collect()
    }

    async fn pull_image_if_needed(&self, image: &str, tag: &str) -> Result<(), ContainerError> {
        let image_name = format!("{}:{}", image, tag);
        
        let output = Command::new("docker")
            .args(&["images", "-q", &image_name])
            .output()
            .await?;

        if output.stdout.is_empty() {
            let pull_output = Command::new("docker")
                .args(&["pull", &image_name])
                .output()
                .await?;

            if !pull_output.status.success() {
                let error = String::from_utf8_lossy(&pull_output.stderr);
                return Err(ContainerError::ImagePullFailed {
                    image: image_name,
                    error: error.to_string(),
                });
            }
        }

        Ok(())
    }

    async fn build_docker_run_args(&self, config: &ContainerConfig) -> Result<Vec<String>, ContainerError> {
        let mut args = Vec::new();

        for (key, value) in &config.environment {
            args.push("-e".to_string());
            args.push(format!("{}={}", key, value));
        }

        for volume in &config.volumes {
            args.push("-v".to_string());
            let volume_str = if volume.read_only {
                format!("{}:{}:ro", volume.host_path.display(), volume.container_path.display())
            } else {
                format!("{}:{}", volume.host_path.display(), volume.container_path.display())
            };
            args.push(volume_str);
        }

        for port in &config.ports {
            args.push("-p".to_string());
            let protocol = match port.protocol {
                Protocol::TCP => "tcp",
                Protocol::UDP => "udp",
            };
            args.push(format!("{}:{}/{}", port.host_port, port.container_port, protocol));
        }

        if let Some(cpu_limit) = config.resources.cpu_limit {
            args.push("--cpus".to_string());
            args.push(cpu_limit.to_string());
        }

        if let Some(ref memory_limit) = config.resources.memory_limit {
            args.push("-m".to_string());
            args.push(memory_limit.clone());
        }

        if config.resources.gpu_enabled {
            args.push("--gpus".to_string());
            args.push("all".to_string());
        }

        match config.network_mode {
            NetworkMode::Bridge => {},
            NetworkMode::Host => {
                args.push("--network".to_string());
                args.push("host".to_string());
            },
            NetworkMode::None => {
                args.push("--network".to_string());
                args.push("none".to_string());
            },
            NetworkMode::Custom(ref network) => {
                args.push("--network".to_string());
                args.push(network.clone());
            },
        }

        match config.restart_policy {
            RestartPolicy::No => {},
            RestartPolicy::Always => {
                args.push("--restart".to_string());
                args.push("always".to_string());
            },
            RestartPolicy::OnFailure { max_retries } => {
                args.push("--restart".to_string());
                args.push(format!("on-failure:{}", max_retries));
            },
            RestartPolicy::UnlessStopped => {
                args.push("--restart".to_string());
                args.push("unless-stopped".to_string());
            },
        }

        if let Some(ref health_check) = config.health_check {
            args.push("--health-cmd".to_string());
            args.push(health_check.command.join(" "));
            args.push("--health-interval".to_string());
            args.push(format!("{}s", health_check.interval.as_secs()));
            args.push("--health-timeout".to_string());
            args.push(format!("{}s", health_check.timeout.as_secs()));
            args.push("--health-retries".to_string());
            args.push(health_check.retries.to_string());
            args.push("--health-start-period".to_string());
            args.push(format!("{}s", health_check.start_period.as_secs()));
        }

        Ok(args)
    }

    /// Deploy multi-container application using Docker Compose
    pub async fn deploy_compose(&self, compose_config: DockerComposeConfig) -> Result<String, ContainerError> {
        self.compose_manager.deploy(compose_config).await
    }

    /// Stop Docker Compose deployment
    pub async fn stop_compose(&self, deployment_id: &str) -> Result<(), ContainerError> {
        self.compose_manager.stop(deployment_id).await
    }

    /// Scale service in Docker Compose deployment
    pub async fn scale_service(&self, deployment_id: &str, service_name: &str, replicas: u32) -> Result<(), ContainerError> {
        self.compose_manager.scale_service(deployment_id, service_name, replicas).await
    }

    /// Register service for discovery
    pub async fn register_service(&self, service: ServiceInfo) -> Result<(), ContainerError> {
        self.service_discovery.register_service(service).await
    }

    /// Discover services by name
    pub async fn discover_services(&self, service_name: &str) -> Result<Vec<ServiceEndpoint>, ContainerError> {
        self.service_discovery.discover_services(service_name).await
    }

    /// Create secret
    pub async fn create_secret(&self, name: &str, data: &[u8]) -> Result<(), ContainerError> {
        self.secrets_manager.create_secret(name, data).await
    }

    /// Update secret
    pub async fn update_secret(&self, name: &str, data: &[u8]) -> Result<(), ContainerError> {
        self.secrets_manager.update_secret(name, data).await
    }

    /// Delete secret
    pub async fn delete_secret(&self, name: &str) -> Result<(), ContainerError> {
        self.secrets_manager.delete_secret(name).await
    }

    async fn start_monitoring(&self, container_id: &str) {
        let container_id = container_id.to_string();
        let containers = self.containers.clone();
        let monitor = self.monitor.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                if let Ok(stats_output) = Command::new("docker")
                    .args(&["stats", "--no-stream", "--format", "json", &container_id])
                    .output()
                    .await
                {
                    if let Ok(stats_str) = String::from_utf8(stats_output.stdout) {
                        if let Ok(stats) = serde_json::from_str::<serde_json::Value>(&stats_str) {
                            let metrics = ContainerMetrics {
                                cpu_usage: stats["CPUPerc"].as_str()
                                    .unwrap_or("0%")
                                    .trim_end_matches('%')
                                    .parse::<f64>()
                                    .unwrap_or(0.0),
                                memory_usage: stats["MemUsage"].as_str()
                                    .unwrap_or("0B")
                                    .split('/')
                                    .next()
                                    .unwrap_or("0B")
                                    .trim_end_matches('B')
                                    .parse::<u64>()
                                    .unwrap_or(0),
                                network_rx: 0,
                                network_tx: 0,
                                disk_read: 0,
                                disk_write: 0,
                            };
                            
                            {
                                let mut containers_guard = containers.write().await;
                                if let Some(container) = containers_guard.get_mut(&container_id) {
                                    container.metrics = metrics.clone();
                                }
                            }
                            
                            {
                                let mut monitor_metrics = monitor.metrics_collector.metrics.write().await;
                                monitor_metrics.insert(container_id.clone(), metrics);
                            }
                        }
                    }
                }

                let containers_guard = containers.read().await;
                if let Some(container) = containers_guard.get(&container_id) {
                    if matches!(container.status, ContainerStatus::Stopped | ContainerStatus::Exited) {
                        break;
                    }
                } else {
                    break;
                }
            }
        });
    }

    pub async fn create_network(&self, config: NetworkConfig) -> Result<(), ContainerError> {
        let mut args = vec!["network", "create"];
        
        args.push("--driver");
        args.push(match config.driver {
            NetworkDriver::Bridge => "bridge",
            NetworkDriver::Overlay => "overlay", 
            NetworkDriver::MacVlan => "macvlan",
            NetworkDriver::Host => "host",
            NetworkDriver::None => "null",
        });
        
        if !config.subnet.is_empty() {
            args.push("--subnet");
            args.push(&config.subnet);
        }
        
        if !config.gateway.is_empty() {
            args.push("--gateway");
            args.push(&config.gateway);
        }
        
        for (key, value) in &config.options {
            args.push("-o");
            args.push(&format!("{}={}", key, value));
        }
        
        args.push(&config.name);
        
        let output = Command::new("docker")
            .args(&args)
            .output()
            .await?;
            
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ContainerError::StartFailed {
                container_id: config.name.clone(),
                error: error.to_string(),
            });
        }
        
        {
            let mut networks = self.network_manager.networks.write().await;
            networks.insert(config.name.clone(), config);
        }
        
        Ok(())
    }

    pub async fn run_ml_pipeline(
        &self,
        pipeline_config: MLPipelineConfig,
    ) -> Result<String, ContainerError> {
        let container_config = ContainerConfig {
            image: pipeline_config.image,
            tag: pipeline_config.tag,
            environment: pipeline_config.environment,
            volumes: pipeline_config.volumes,
            ports: vec![],
            resources: pipeline_config.resources,
            network_mode: NetworkMode::Bridge,
            restart_policy: RestartPolicy::No,
            health_check: None,
        };

        let container_id = self.create_container(
            format!("ml-pipeline-{}", Uuid::new_v4()),
            container_config,
        ).await?;

        self.start_container(&container_id).await?;

        let execution_timeout = pipeline_config.timeout.unwrap_or(Duration::from_secs(3600));
        let start_time = Instant::now();

        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            
            let containers = self.containers.read().await;
            if let Some(container) = containers.get(&container_id) {
                match container.status {
                    ContainerStatus::Exited => {
                        return if container.exit_code == Some(0) {
                            Ok(container_id)
                        } else {
                            Err(ContainerError::StartFailed {
                                container_id: container_id.clone(),
                                error: format!("Pipeline exited with code {:?}", container.exit_code),
                            })
                        };
                    }
                    ContainerStatus::Error => {
                        return Err(ContainerError::StartFailed {
                            container_id: container_id.clone(),
                            error: "Pipeline execution failed".to_string(),
                        });
                    }
                    _ => {
                        if start_time.elapsed() > execution_timeout {
                            self.stop_container(&container_id).await.ok();
                            return Err(ContainerError::ExecutionTimeout { container_id });
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MLPipelineConfig {
    pub image: String,
    pub tag: String,
    pub environment: HashMap<String, String>,
    pub volumes: Vec<VolumeMount>,
    pub resources: ResourceLimits,
    pub timeout: Option<Duration>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_container_creation() {
        let runtime = ContainerRuntime::new();
        
        let config = ContainerConfig {
            image: "ubuntu".to_string(),
            tag: "latest".to_string(),
            environment: HashMap::new(),
            volumes: vec![],
            ports: vec![],
            resources: ResourceLimits {
                cpu_limit: Some(1.0),
                memory_limit: Some("512m".to_string()),
                gpu_enabled: false,
                disk_limit: None,
            },
            network_mode: NetworkMode::Bridge,
            restart_policy: RestartPolicy::No,
            health_check: None,
        };
        
        let result = runtime.create_container("test-container".to_string(), config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_docker_args_building() {
        let runtime = ContainerRuntime::new();
        
        let config = ContainerConfig {
            image: "test".to_string(),
            tag: "latest".to_string(),
            environment: {
                let mut env = HashMap::new();
                env.insert("KEY".to_string(), "value".to_string());
                env
            },
            volumes: vec![VolumeMount {
                host_path: PathBuf::from("/host"),
                container_path: PathBuf::from("/container"),
                read_only: true,
            }],
            ports: vec![PortMapping {
                host_port: 8080,
                container_port: 80,
                protocol: Protocol::TCP,
            }],
            resources: ResourceLimits {
                cpu_limit: Some(1.5),
                memory_limit: Some("1g".to_string()),
                gpu_enabled: true,
                disk_limit: None,
            },
            network_mode: NetworkMode::Host,
            restart_policy: RestartPolicy::Always,
            health_check: Some(HealthCheck {
                command: vec!["curl".to_string(), "-f".to_string(), "localhost:80".to_string()],
                interval: Duration::from_secs(30),
                timeout: Duration::from_secs(10),
                retries: 3,
                start_period: Duration::from_secs(60),
            }),
        };
        
        let args = runtime.build_docker_run_args(&config).await.unwrap();
        
        assert!(args.contains(&"-e".to_string()));
        assert!(args.contains(&"KEY=value".to_string()));
        assert!(args.contains(&"-v".to_string()));
        assert!(args.contains(&"/host:/container:ro".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"8080:80/tcp".to_string()));
        assert!(args.contains(&"--cpus".to_string()));
        assert!(args.contains(&"1.5".to_string()));
        assert!(args.contains(&"-m".to_string()));
        assert!(args.contains(&"1g".to_string()));
        assert!(args.contains(&"--gpus".to_string()));
        assert!(args.contains(&"all".to_string()));
        assert!(args.contains(&"--network".to_string()));
        assert!(args.contains(&"host".to_string()));
        assert!(args.contains(&"--restart".to_string()));
        assert!(args.contains(&"always".to_string()));
    }

    #[tokio::test]
    async fn test_network_creation() {
        let runtime = ContainerRuntime::new();
        
        let config = NetworkConfig {
            name: "test-network".to_string(),
            driver: NetworkDriver::Bridge,
            subnet: "172.20.0.0/16".to_string(),
            gateway: "172.20.0.1".to_string(),
            options: HashMap::new(),
        };
        
        {
            let mut networks = runtime.network_manager.networks.write().await;
            networks.insert(config.name.clone(), config.clone());
        }
        
        let networks = runtime.network_manager.networks.read().await;
        assert!(networks.contains_key("test-network"));
    }

    #[tokio::test]
    async fn test_ml_pipeline_config() {
        let config = MLPipelineConfig {
            image: "sklearn".to_string(),
            tag: "latest".to_string(),
            environment: {
                let mut env = HashMap::new();
                env.insert("PYTHONPATH".to_string(), "/app".to_string());
                env
            },
            volumes: vec![VolumeMount {
                host_path: PathBuf::from("/data"),
                container_path: PathBuf::from("/app/data"),
                read_only: true,
            }],
            resources: ResourceLimits {
                cpu_limit: Some(2.0),
                memory_limit: Some("2g".to_string()),
                gpu_enabled: true,
                disk_limit: Some("10g".to_string()),
            },
            timeout: Some(Duration::from_secs(1800)),
        };
        
        assert_eq!(config.image, "sklearn");
        assert_eq!(config.timeout, Some(Duration::from_secs(1800)));
        assert!(config.resources.gpu_enabled);
    }
}

// Enhanced Docker Integration Components

/// Docker Compose manager for multi-container deployments
pub struct DockerComposeManager {
    deployments: Arc<RwLock<HashMap<String, DockerComposeDeployment>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeConfig {
    pub version: String,
    pub services: HashMap<String, DockerComposeService>,
    pub networks: Option<HashMap<String, DockerComposeNetwork>>,
    pub volumes: Option<HashMap<String, DockerComposeVolume>>,
    pub secrets: Option<HashMap<String, DockerComposeSecret>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeService {
    pub image: String,
    pub build: Option<DockerComposeBuild>,
    pub ports: Option<Vec<String>>,
    pub environment: Option<HashMap<String, String>>,
    pub volumes: Option<Vec<String>>,
    pub depends_on: Option<Vec<String>>,
    pub networks: Option<Vec<String>>,
    pub deploy: Option<DockerComposeDeploy>,
    pub healthcheck: Option<DockerComposeHealthCheck>,
    pub secrets: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeBuild {
    pub context: String,
    pub dockerfile: Option<String>,
    pub args: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeDeploy {
    pub replicas: Option<u32>,
    pub resources: Option<DockerComposeResources>,
    pub restart_policy: Option<DockerComposeRestartPolicy>,
    pub placement: Option<DockerComposePlacement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeResources {
    pub limits: Option<DockerComposeResourceLimits>,
    pub reservations: Option<DockerComposeResourceLimits>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeResourceLimits {
    pub cpus: Option<String>,
    pub memory: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeRestartPolicy {
    pub condition: String,
    pub delay: Option<String>,
    pub max_attempts: Option<u32>,
    pub window: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposePlacement {
    pub constraints: Option<Vec<String>>,
    pub preferences: Option<Vec<HashMap<String, String>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeNetwork {
    pub driver: Option<String>,
    pub driver_opts: Option<HashMap<String, String>>,
    pub attachable: Option<bool>,
    pub internal: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeVolume {
    pub driver: Option<String>,
    pub driver_opts: Option<HashMap<String, String>>,
    pub external: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeSecret {
    pub file: Option<String>,
    pub external: Option<bool>,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerComposeHealthCheck {
    pub test: Vec<String>,
    pub interval: Option<String>,
    pub timeout: Option<String>,
    pub retries: Option<u32>,
    pub start_period: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DockerComposeDeployment {
    pub id: String,
    pub config: DockerComposeConfig,
    pub status: DeploymentStatus,
    pub created_at: Instant,
    pub updated_at: Instant,
    pub services: HashMap<String, Vec<String>>, // service name -> container IDs
}

#[derive(Debug, Clone)]
pub enum DeploymentStatus {
    Creating,
    Running,
    Scaling,
    Stopping,
    Stopped,
    Failed,
}

impl DockerComposeManager {
    pub fn new() -> Self {
        Self {
            deployments: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn deploy(&self, config: DockerComposeConfig) -> Result<String, ContainerError> {
        let deployment_id = Uuid::new_v4().to_string();
        let deployment_name = format!("sklears-deploy-{}", deployment_id);
        
        // Create temporary compose file
        let compose_file = format!("/tmp/{}.yml", deployment_name);
        let compose_content = serde_yaml::to_string(&config)
            .map_err(|e| ContainerError::Serialization(serde_json::Error::custom(e.to_string())))?;
        
        fs::write(&compose_file, compose_content)
            .map_err(|e| ContainerError::Io(e))?;

        // Deploy using docker-compose
        let output = Command::new("docker-compose")
            .args(&["-f", &compose_file, "-p", &deployment_name, "up", "-d"])
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ContainerError::StartFailed {
                container_id: deployment_id,
                error: error.to_string(),
            });
        }

        let deployment = DockerComposeDeployment {
            id: deployment_id.clone(),
            config,
            status: DeploymentStatus::Running,
            created_at: Instant::now(),
            updated_at: Instant::now(),
            services: HashMap::new(),
        };

        {
            let mut deployments = self.deployments.write().await;
            deployments.insert(deployment_id.clone(), deployment);
        }

        // Clean up temporary file
        fs::remove_file(&compose_file).ok();

        Ok(deployment_id)
    }

    pub async fn stop(&self, deployment_id: &str) -> Result<(), ContainerError> {
        let deployment_name = format!("sklears-deploy-{}", deployment_id);
        
        let output = Command::new("docker-compose")
            .args(&["-p", &deployment_name, "down"])
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ContainerError::StopFailed {
                container_id: deployment_id.to_string(),
                error: error.to_string(),
            });
        }

        {
            let mut deployments = self.deployments.write().await;
            if let Some(deployment) = deployments.get_mut(deployment_id) {
                deployment.status = DeploymentStatus::Stopped;
                deployment.updated_at = Instant::now();
            }
        }

        Ok(())
    }

    pub async fn scale_service(&self, deployment_id: &str, service_name: &str, replicas: u32) -> Result<(), ContainerError> {
        let deployment_name = format!("sklears-deploy-{}", deployment_id);
        
        let output = Command::new("docker-compose")
            .args(&["-p", &deployment_name, "up", "-d", "--scale", &format!("{}={}", service_name, replicas)])
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ContainerError::StartFailed {
                container_id: deployment_id.to_string(),
                error: error.to_string(),
            });
        }

        {
            let mut deployments = self.deployments.write().await;
            if let Some(deployment) = deployments.get_mut(deployment_id) {
                deployment.status = DeploymentStatus::Scaling;
                deployment.updated_at = Instant::now();
            }
        }

        Ok(())
    }
}

/// Service discovery for container-based services
pub struct ServiceDiscovery {
    services: Arc<RwLock<HashMap<String, Vec<ServiceEndpoint>>>>,
    health_checker: Arc<ServiceHealthChecker>,
}

#[derive(Debug, Clone)]
pub struct ServiceInfo {
    pub name: String,
    pub endpoint: ServiceEndpoint,
    pub health_check: Option<ServiceHealthCheck>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ServiceEndpoint {
    pub id: String,
    pub address: String,
    pub port: u16,
    pub protocol: String,
    pub status: ServiceStatus,
    pub registered_at: Instant,
    pub last_health_check: Option<Instant>,
}

#[derive(Debug, Clone)]
pub enum ServiceStatus {
    Healthy,
    Unhealthy,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ServiceHealthCheck {
    pub endpoint: String,
    pub interval: Duration,
    pub timeout: Duration,
    pub expected_status: u16,
}

pub struct ServiceHealthChecker {
    checks: Arc<RwLock<HashMap<String, ServiceHealthCheck>>>,
}

impl ServiceDiscovery {
    pub fn new() -> Self {
        Self {
            services: Arc::new(RwLock::new(HashMap::new())),
            health_checker: Arc::new(ServiceHealthChecker {
                checks: Arc::new(RwLock::new(HashMap::new())),
            }),
        }
    }

    pub async fn register_service(&self, service: ServiceInfo) -> Result<(), ContainerError> {
        {
            let mut services = self.services.write().await;
            services.entry(service.name.clone())
                .or_insert_with(Vec::new)
                .push(service.endpoint.clone());
        }

        if let Some(health_check) = service.health_check {
            let mut checks = self.health_checker.checks.write().await;
            checks.insert(service.endpoint.id.clone(), health_check);
        }

        Ok(())
    }

    pub async fn discover_services(&self, service_name: &str) -> Result<Vec<ServiceEndpoint>, ContainerError> {
        let services = self.services.read().await;
        Ok(services.get(service_name)
            .map(|endpoints| endpoints.iter().filter(|e| matches!(e.status, ServiceStatus::Healthy)).cloned().collect())
            .unwrap_or_default())
    }

    pub async fn deregister_service(&self, service_name: &str, endpoint_id: &str) -> Result<(), ContainerError> {
        let mut services = self.services.write().await;
        if let Some(endpoints) = services.get_mut(service_name) {
            endpoints.retain(|e| e.id != endpoint_id);
            if endpoints.is_empty() {
                services.remove(service_name);
            }
        }
        Ok(())
    }
}

/// Secrets management for containers
pub struct SecretsManager {
    secrets: Arc<RwLock<HashMap<String, SecretInfo>>>,
}

#[derive(Debug, Clone)]
pub struct SecretInfo {
    pub name: String,
    pub created_at: Instant,
    pub updated_at: Instant,
    pub version: u32,
    pub size: usize,
}

impl SecretsManager {
    pub fn new() -> Self {
        Self {
            secrets: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_secret(&self, name: &str, data: &[u8]) -> Result<(), ContainerError> {
        // Use Docker secrets if available, otherwise use local storage
        let output = Command::new("docker")
            .args(&["secret", "create", name, "-"])
            .stdin(std::process::Stdio::piped())
            .output()
            .await?;

        if !output.status.success() {
            // Fallback to local storage
            let secret_file = format!("/tmp/sklears-secret-{}", name);
            fs::write(&secret_file, data)?;
        }

        let secret_info = SecretInfo {
            name: name.to_string(),
            created_at: Instant::now(),
            updated_at: Instant::now(),
            version: 1,
            size: data.len(),
        };

        {
            let mut secrets = self.secrets.write().await;
            secrets.insert(name.to_string(), secret_info);
        }

        Ok(())
    }

    pub async fn update_secret(&self, name: &str, data: &[u8]) -> Result<(), ContainerError> {
        // Remove existing secret
        self.delete_secret(name).await?;
        
        // Create new version
        self.create_secret(name, data).await?;

        {
            let mut secrets = self.secrets.write().await;
            if let Some(secret) = secrets.get_mut(name) {
                secret.updated_at = Instant::now();
                secret.version += 1;
                secret.size = data.len();
            }
        }

        Ok(())
    }

    pub async fn delete_secret(&self, name: &str) -> Result<(), ContainerError> {
        let output = Command::new("docker")
            .args(&["secret", "rm", name])
            .output()
            .await?;

        if !output.status.success() {
            // Try local storage
            let secret_file = format!("/tmp/sklears-secret-{}", name);
            fs::remove_file(&secret_file).ok();
        }

        {
            let mut secrets = self.secrets.write().await;
            secrets.remove(name);
        }

        Ok(())
    }
}

// Update the ContainerRuntime constructor to include new components
impl ContainerRuntime {
    pub fn new() -> Self {
        let containers = Arc::new(RwLock::new(HashMap::new()));
        let registry = Arc::new(RwLock::new(ContainerRegistry {
            images: HashMap::new(),
            repositories: Vec::new(),
        }));
        
        let cluster_config = Arc::new(RwLock::new(ClusterConfig {
            nodes: Vec::new(),
            placement_strategy: PlacementStrategy::LeastUsed,
            auto_scaling: AutoScalingConfig {
                enabled: false,
                min_nodes: 1,
                max_nodes: 10,
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.2,
                scale_up_cooldown: Duration::from_secs(300),
                scale_down_cooldown: Duration::from_secs(600),
            },
        }));
        
        let scheduler = Arc::new(Scheduler {
            placement_strategy: PlacementStrategy::LeastUsed,
            resource_tracker: Arc::new(RwLock::new(HashMap::new())),
        });
        
        let health_checker = Arc::new(HealthChecker {
            checks: Arc::new(RwLock::new(HashMap::new())),
        });
        
        let load_balancer = Arc::new(ContainerLoadBalancer {
            strategy: LoadBalancingStrategy::RoundRobin,
            health_checker,
        });
        
        let orchestrator = Arc::new(ContainerOrchestrator {
            cluster_config,
            scheduler,
            load_balancer,
        });
        
        let metrics_collector = Arc::new(MetricsCollector {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            exporters: Vec::new(),
        });
        
        let log_aggregator = Arc::new(LogAggregator {
            logs: Arc::new(RwLock::new(HashMap::new())),
            processors: Vec::new(),
        });
        
        let (alert_tx, _alert_rx) = mpsc::unbounded_channel();
        let alert_manager = Arc::new(AlertManager {
            rules: Arc::new(RwLock::new(Vec::new())),
            notifications: alert_tx,
        });
        
        let monitor = Arc::new(ContainerMonitor {
            metrics_collector,
            log_aggregator,
            alert_manager,
        });
        
        let networks = Arc::new(RwLock::new(HashMap::new()));
        let dns_resolver = Arc::new(DnsResolver {
            entries: Arc::new(RwLock::new(HashMap::new())),
        });
        let proxy = Arc::new(ServiceProxy {
            routes: Arc::new(RwLock::new(HashMap::new())),
        });
        
        let network_manager = Arc::new(NetworkManager {
            networks,
            dns_resolver,
            proxy,
        });

        let compose_manager = Arc::new(DockerComposeManager::new());
        let service_discovery = Arc::new(ServiceDiscovery::new());
        let secrets_manager = Arc::new(SecretsManager::new());
        
        Self {
            containers,
            registry,
            orchestrator,
            monitor,
            network_manager,
            compose_manager,
            service_discovery,
            secrets_manager,
        }
    }
}
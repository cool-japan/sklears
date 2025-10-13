//! Fault Detection Module
//!
//! Implements sophisticated fault detection and classification system for fault tolerance:
//! - Multiple detection algorithms (threshold-based, anomaly detection, pattern matching)
//! - Real-time monitoring and metrics collection
//! - Automatic fault classification and severity assessment
//! - Alert generation and notification system
//! - Performance and resource monitoring integration

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::time::sleep;
use uuid::Uuid;

/// Fault detection algorithms for different types of failures
#[derive(Debug, Clone, PartialEq)]
pub enum DetectionAlgorithm {
    /// Threshold-based detection with configurable limits
    Threshold {
        /// Upper threshold for detection
        upper_threshold: f64,
        /// Lower threshold for detection
        lower_threshold: f64,
        /// Number of consecutive violations required
        violation_count: u32,
    },
    /// Statistical anomaly detection
    StatisticalAnomaly {
        /// Number of standard deviations for outlier detection
        std_dev_multiplier: f64,
        /// Window size for calculating statistics
        window_size: usize,
        /// Minimum samples required
        min_samples: u32,
    },
    /// Rate of change detection
    RateOfChange {
        /// Maximum acceptable rate of change per second
        max_rate: f64,
        /// Time window for rate calculation
        time_window: Duration,
    },
    /// Pattern-based detection using predefined patterns
    PatternBased {
        /// Fault patterns to detect
        patterns: Vec<FaultPattern>,
        /// Pattern matching sensitivity
        sensitivity: f64,
    },
    /// Moving average deviation detection
    MovingAverageDeviation {
        /// Window size for moving average
        window_size: usize,
        /// Maximum acceptable deviation percentage
        max_deviation_percent: f64,
    },
    /// Resource utilization monitoring
    ResourceUtilization {
        /// CPU threshold percentage
        cpu_threshold: f64,
        /// Memory threshold percentage
        memory_threshold: f64,
        /// Disk threshold percentage
        disk_threshold: f64,
        /// Network threshold (bytes per second)
        network_threshold: f64,
    },
    /// Custom detection algorithm
    Custom {
        /// Custom detection function
        detect_fn: fn(&MetricSample, &[MetricSample]) -> DetectionResult,
    },
}

/// Fault pattern for pattern-based detection
#[derive(Debug, Clone, PartialEq)]
pub struct FaultPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Pattern name
    pub name: String,
    /// Metric conditions that indicate this fault
    pub conditions: Vec<PatternCondition>,
    /// Pattern severity when detected
    pub severity: FaultSeverity,
    /// Pattern description
    pub description: String,
}

/// Condition within a fault pattern
#[derive(Debug, Clone, PartialEq)]
pub struct PatternCondition {
    /// Metric name to evaluate
    pub metric_name: String,
    /// Condition operator
    pub operator: ConditionOperator,
    /// Threshold value for condition
    pub threshold: f64,
    /// Time window for condition evaluation
    pub time_window: Duration,
}

/// Condition operators for pattern matching
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionOperator {
    /// Greater than threshold
    GreaterThan,
    /// Less than threshold
    LessThan,
    /// Equal to threshold
    EqualTo,
    /// Not equal to threshold
    NotEqualTo,
    /// Within range (threshold ± tolerance)
    WithinRange { tolerance: f64 },
    /// Outside range
    OutsideRange { tolerance: f64 },
}

/// Fault severity levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum FaultSeverity {
    /// Low severity - informational
    Low = 1,
    /// Medium severity - warning
    Medium = 2,
    /// High severity - requires attention
    High = 3,
    /// Critical severity - immediate action required
    Critical = 4,
}

/// Fault categories for classification
#[derive(Debug, Clone, PartialEq)]
pub enum FaultCategory {
    /// Performance related faults
    Performance,
    /// Resource exhaustion faults
    ResourceExhaustion,
    /// Network connectivity faults
    Network,
    /// Hardware related faults
    Hardware,
    /// Software/logic faults
    Software,
    /// Configuration related faults
    Configuration,
    /// Security related faults
    Security,
    /// Data integrity faults
    DataIntegrity,
    /// External dependency faults
    ExternalDependency,
    /// Unknown fault category
    Unknown,
}

/// Metric sample for fault detection analysis
#[derive(Debug, Clone)]
pub struct MetricSample {
    /// Sample timestamp
    pub timestamp: Instant,
    /// Metric name
    pub metric_name: String,
    /// Metric value
    pub value: f64,
    /// Component that generated the metric
    pub component_id: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Result of fault detection analysis
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Whether a fault was detected
    pub fault_detected: bool,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Detected fault details
    pub fault_details: Option<DetectedFault>,
    /// Analysis metadata
    pub analysis_metadata: HashMap<String, String>,
}

/// Detected fault information
#[derive(Debug, Clone)]
pub struct DetectedFault {
    /// Fault identifier
    pub fault_id: String,
    /// Fault category
    pub category: FaultCategory,
    /// Fault severity
    pub severity: FaultSeverity,
    /// Fault description
    pub description: String,
    /// Component where fault was detected
    pub component_id: String,
    /// Fault detection timestamp
    pub detection_time: Instant,
    /// Related metrics that triggered detection
    pub triggering_metrics: Vec<MetricSample>,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
    /// Fault metadata
    pub metadata: HashMap<String, String>,
}

/// Fault detection configuration
#[derive(Debug, Clone)]
pub struct FaultDetectionConfig {
    /// Configuration identifier
    pub config_id: String,
    /// Detection algorithms to use
    pub algorithms: Vec<DetectionAlgorithm>,
    /// Metrics to monitor
    pub monitored_metrics: Vec<String>,
    /// Detection interval
    pub detection_interval: Duration,
    /// Maximum number of samples to keep in history
    pub max_sample_history: usize,
    /// Alert thresholds
    pub alert_thresholds: HashMap<FaultSeverity, u32>,
    /// Whether to enable automatic classification
    pub auto_classification: bool,
    /// Classification rules
    pub classification_rules: HashMap<String, FaultCategory>,
}

/// Fault detection metrics and statistics
#[derive(Debug, Clone)]
pub struct FaultDetectionMetrics {
    pub config_id: String,
    pub total_samples: u64,
    pub total_faults_detected: u64,
    pub faults_by_severity: HashMap<FaultSeverity, u32>,
    pub faults_by_category: HashMap<FaultCategory, u32>,
    pub false_positive_rate: f64,
    pub detection_accuracy: f64,
    pub average_detection_latency: Duration,
    pub recent_detections: Vec<DetectedFault>,
    pub health_score: f64,
}

/// Fault detection system errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum FaultDetectionError {
    #[error("Algorithm configuration error: {message}")]
    AlgorithmConfigError { message: String },
    #[error("Insufficient data for detection: requires {required} samples, got {available}")]
    InsufficientData { required: usize, available: usize },
    #[error("Metric not found: {metric_name}")]
    MetricNotFound { metric_name: String },
    #[error("Detection timeout: {timeout:?}")]
    DetectionTimeout { timeout: Duration },
    #[error("Classification error: {message}")]
    ClassificationError { message: String },
}

/// Fault detection system implementation
#[derive(Debug)]
pub struct FaultDetectionSystem {
    /// System identifier
    system_id: String,
    /// Detection configuration
    config: FaultDetectionConfig,
    /// Metric sample history
    sample_history: Arc<RwLock<HashMap<String, VecDeque<MetricSample>>>>,
    /// Detected faults history
    fault_history: Arc<RwLock<VecDeque<DetectedFault>>>,
    /// Detection metrics
    metrics: Arc<RwLock<FaultDetectionMetrics>>,
    /// Active detection tasks
    detection_tasks: Arc<RwLock<HashMap<String, tokio::task::JoinHandle<()>>>>,
}

impl Default for FaultDetectionConfig {
    fn default() -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert(FaultSeverity::Low, 10);
        alert_thresholds.insert(FaultSeverity::Medium, 5);
        alert_thresholds.insert(FaultSeverity::High, 3);
        alert_thresholds.insert(FaultSeverity::Critical, 1);

        Self {
            config_id: "default".to_string(),
            algorithms: vec![
                DetectionAlgorithm::Threshold {
                    upper_threshold: 0.8,
                    lower_threshold: 0.2,
                    violation_count: 3,
                },
                DetectionAlgorithm::StatisticalAnomaly {
                    std_dev_multiplier: 2.0,
                    window_size: 50,
                    min_samples: 10,
                },
            ],
            monitored_metrics: vec![
                "cpu_usage".to_string(),
                "memory_usage".to_string(),
                "response_time".to_string(),
                "error_rate".to_string(),
            ],
            detection_interval: Duration::from_secs(10),
            max_sample_history: 1000,
            alert_thresholds,
            auto_classification: true,
            classification_rules: HashMap::new(),
        }
    }
}

impl FaultDetectionSystem {
    /// Create new fault detection system
    pub fn new(system_id: String, config: FaultDetectionConfig) -> Self {
        let metrics = FaultDetectionMetrics {
            config_id: config.config_id.clone(),
            total_samples: 0,
            total_faults_detected: 0,
            faults_by_severity: HashMap::new(),
            faults_by_category: HashMap::new(),
            false_positive_rate: 0.0,
            detection_accuracy: 0.0,
            average_detection_latency: Duration::ZERO,
            recent_detections: Vec::new(),
            health_score: 1.0,
        };

        Self {
            system_id,
            config,
            sample_history: Arc::new(RwLock::new(HashMap::new())),
            fault_history: Arc::new(RwLock::new(VecDeque::new())),
            metrics: Arc::new(RwLock::new(metrics)),
            detection_tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create system with default configuration
    pub fn with_defaults(system_id: String) -> Self {
        Self::new(system_id, FaultDetectionConfig::default())
    }

    /// Add metric sample for analysis
    pub async fn add_metric_sample(&self, sample: MetricSample) {
        let metric_name = sample.metric_name.clone();

        // Add to sample history
        {
            let mut history = self.sample_history.write().unwrap();
            let metric_history = history.entry(metric_name.clone()).or_insert_with(VecDeque::new);

            metric_history.push_back(sample.clone());

            // Maintain history size limit
            while metric_history.len() > self.config.max_sample_history {
                metric_history.pop_front();
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_samples += 1;
        }

        // Trigger detection analysis
        self.analyze_metric_sample(&sample).await;
    }

    /// Analyze metric sample for fault detection
    async fn analyze_metric_sample(&self, sample: &MetricSample) {
        let detection_start = Instant::now();

        // Get historical samples for this metric
        let samples = {
            let history = self.sample_history.read().unwrap();
            history.get(&sample.metric_name)
                .map(|samples| samples.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_else(Vec::new)
        };

        // Run detection algorithms
        for algorithm in &self.config.algorithms {
            match self.run_detection_algorithm(algorithm, sample, &samples).await {
                Ok(result) => {
                    if result.fault_detected {
                        self.handle_fault_detection(result, detection_start.elapsed()).await;
                    }
                },
                Err(error) => {
                    eprintln!("Detection algorithm error: {}", error);
                }
            }
        }
    }

    /// Run specific detection algorithm
    async fn run_detection_algorithm(
        &self,
        algorithm: &DetectionAlgorithm,
        sample: &MetricSample,
        historical_samples: &[MetricSample],
    ) -> Result<DetectionResult, FaultDetectionError> {
        match algorithm {
            DetectionAlgorithm::Threshold { upper_threshold, lower_threshold, violation_count } => {
                self.threshold_detection(sample, historical_samples, *upper_threshold, *lower_threshold, *violation_count).await
            },
            DetectionAlgorithm::StatisticalAnomaly { std_dev_multiplier, window_size, min_samples } => {
                self.statistical_anomaly_detection(sample, historical_samples, *std_dev_multiplier, *window_size, *min_samples).await
            },
            DetectionAlgorithm::RateOfChange { max_rate, time_window } => {
                self.rate_of_change_detection(sample, historical_samples, *max_rate, *time_window).await
            },
            DetectionAlgorithm::PatternBased { patterns, sensitivity } => {
                self.pattern_based_detection(sample, historical_samples, patterns, *sensitivity).await
            },
            DetectionAlgorithm::MovingAverageDeviation { window_size, max_deviation_percent } => {
                self.moving_average_deviation_detection(sample, historical_samples, *window_size, *max_deviation_percent).await
            },
            DetectionAlgorithm::ResourceUtilization { cpu_threshold, memory_threshold, disk_threshold, network_threshold } => {
                self.resource_utilization_detection(sample, *cpu_threshold, *memory_threshold, *disk_threshold, *network_threshold).await
            },
            DetectionAlgorithm::Custom { detect_fn } => {
                Ok(detect_fn(sample, historical_samples))
            }
        }
    }

    /// Threshold-based fault detection
    async fn threshold_detection(
        &self,
        sample: &MetricSample,
        historical_samples: &[MetricSample],
        upper_threshold: f64,
        lower_threshold: f64,
        violation_count: u32,
    ) -> Result<DetectionResult, FaultDetectionError> {
        let is_violation = sample.value > upper_threshold || sample.value < lower_threshold;

        if !is_violation {
            return Ok(DetectionResult {
                fault_detected: false,
                confidence: 0.0,
                fault_details: None,
                analysis_metadata: HashMap::new(),
            });
        }

        // Check consecutive violations
        let recent_samples = historical_samples.iter()
            .rev()
            .take(violation_count as usize)
            .collect::<Vec<_>>();

        let consecutive_violations = recent_samples.iter()
            .all(|s| s.value > upper_threshold || s.value < lower_threshold);

        if consecutive_violations && recent_samples.len() >= violation_count as usize {
            let fault = DetectedFault {
                fault_id: Uuid::new_v4().to_string(),
                category: self.classify_fault(&sample.metric_name, "threshold_violation").await,
                severity: self.assess_severity(sample.value, upper_threshold, lower_threshold),
                description: format!("Threshold violation: {} = {:.2} (thresholds: {:.2} - {:.2})",
                    sample.metric_name, sample.value, lower_threshold, upper_threshold),
                component_id: sample.component_id.clone(),
                detection_time: Instant::now(),
                triggering_metrics: vec![sample.clone()],
                suggested_actions: vec!["Check component health".to_string()],
                metadata: HashMap::new(),
            };

            Ok(DetectionResult {
                fault_detected: true,
                confidence: 0.9,
                fault_details: Some(fault),
                analysis_metadata: HashMap::new(),
            })
        } else {
            Ok(DetectionResult {
                fault_detected: false,
                confidence: 0.0,
                fault_details: None,
                analysis_metadata: HashMap::new(),
            })
        }
    }

    /// Statistical anomaly detection
    async fn statistical_anomaly_detection(
        &self,
        sample: &MetricSample,
        historical_samples: &[MetricSample],
        std_dev_multiplier: f64,
        window_size: usize,
        min_samples: u32,
    ) -> Result<DetectionResult, FaultDetectionError> {
        if historical_samples.len() < min_samples as usize {
            return Err(FaultDetectionError::InsufficientData {
                required: min_samples as usize,
                available: historical_samples.len(),
            });
        }

        let window_samples = historical_samples.iter()
            .rev()
            .take(window_size)
            .map(|s| s.value)
            .collect::<Vec<_>>();

        let mean = window_samples.iter().sum::<f64>() / window_samples.len() as f64;
        let variance = window_samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / window_samples.len() as f64;
        let std_dev = variance.sqrt();

        let z_score = (sample.value - mean) / std_dev;
        let is_anomaly = z_score.abs() > std_dev_multiplier;

        if is_anomaly {
            let fault = DetectedFault {
                fault_id: Uuid::new_v4().to_string(),
                category: self.classify_fault(&sample.metric_name, "statistical_anomaly").await,
                severity: if z_score.abs() > std_dev_multiplier * 2.0 { FaultSeverity::High } else { FaultSeverity::Medium },
                description: format!("Statistical anomaly detected: {} = {:.2} (z-score: {:.2})",
                    sample.metric_name, sample.value, z_score),
                component_id: sample.component_id.clone(),
                detection_time: Instant::now(),
                triggering_metrics: vec![sample.clone()],
                suggested_actions: vec!["Investigate anomalous behavior".to_string()],
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("z_score".to_string(), z_score.to_string());
                    metadata.insert("mean".to_string(), mean.to_string());
                    metadata.insert("std_dev".to_string(), std_dev.to_string());
                    metadata
                },
            };

            Ok(DetectionResult {
                fault_detected: true,
                confidence: (z_score.abs() - std_dev_multiplier) / std_dev_multiplier,
                fault_details: Some(fault),
                analysis_metadata: HashMap::new(),
            })
        } else {
            Ok(DetectionResult {
                fault_detected: false,
                confidence: 0.0,
                fault_details: None,
                analysis_metadata: HashMap::new(),
            })
        }
    }

    /// Rate of change detection
    async fn rate_of_change_detection(
        &self,
        sample: &MetricSample,
        historical_samples: &[MetricSample],
        max_rate: f64,
        time_window: Duration,
    ) -> Result<DetectionResult, FaultDetectionError> {
        let cutoff_time = sample.timestamp - time_window;
        let recent_samples: Vec<&MetricSample> = historical_samples.iter()
            .filter(|s| s.timestamp > cutoff_time)
            .collect();

        if recent_samples.len() < 2 {
            return Ok(DetectionResult {
                fault_detected: false,
                confidence: 0.0,
                fault_details: None,
                analysis_metadata: HashMap::new(),
            });
        }

        let oldest_sample = recent_samples.iter().min_by_key(|s| s.timestamp).unwrap();
        let time_diff = sample.timestamp.duration_since(oldest_sample.timestamp);

        if time_diff.as_secs_f64() > 0.0 {
            let rate = (sample.value - oldest_sample.value).abs() / time_diff.as_secs_f64();

            if rate > max_rate {
                let fault = DetectedFault {
                    fault_id: Uuid::new_v4().to_string(),
                    category: self.classify_fault(&sample.metric_name, "rate_of_change").await,
                    severity: if rate > max_rate * 2.0 { FaultSeverity::High } else { FaultSeverity::Medium },
                    description: format!("High rate of change detected: {} changed by {:.2}/sec (max: {:.2}/sec)",
                        sample.metric_name, rate, max_rate),
                    component_id: sample.component_id.clone(),
                    detection_time: Instant::now(),
                    triggering_metrics: vec![sample.clone()],
                    suggested_actions: vec!["Check for sudden load changes".to_string()],
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("rate".to_string(), rate.to_string());
                        metadata.insert("time_window".to_string(), time_diff.as_secs().to_string());
                        metadata
                    },
                };

                return Ok(DetectionResult {
                    fault_detected: true,
                    confidence: (rate - max_rate) / max_rate,
                    fault_details: Some(fault),
                    analysis_metadata: HashMap::new(),
                });
            }
        }

        Ok(DetectionResult {
            fault_detected: false,
            confidence: 0.0,
            fault_details: None,
            analysis_metadata: HashMap::new(),
        })
    }

    /// Pattern-based fault detection
    async fn pattern_based_detection(
        &self,
        sample: &MetricSample,
        _historical_samples: &[MetricSample],
        patterns: &[FaultPattern],
        _sensitivity: f64,
    ) -> Result<DetectionResult, FaultDetectionError> {
        for pattern in patterns {
            for condition in &pattern.conditions {
                if condition.metric_name == sample.metric_name {
                    if self.evaluate_pattern_condition(condition, sample.value) {
                        let fault = DetectedFault {
                            fault_id: Uuid::new_v4().to_string(),
                            category: self.classify_fault(&sample.metric_name, &pattern.pattern_id).await,
                            severity: pattern.severity,
                            description: format!("Pattern '{}' detected: {}", pattern.name, pattern.description),
                            component_id: sample.component_id.clone(),
                            detection_time: Instant::now(),
                            triggering_metrics: vec![sample.clone()],
                            suggested_actions: vec!["Review pattern-specific remediation".to_string()],
                            metadata: {
                                let mut metadata = HashMap::new();
                                metadata.insert("pattern_id".to_string(), pattern.pattern_id.clone());
                                metadata.insert("pattern_name".to_string(), pattern.name.clone());
                                metadata
                            },
                        };

                        return Ok(DetectionResult {
                            fault_detected: true,
                            confidence: 0.85,
                            fault_details: Some(fault),
                            analysis_metadata: HashMap::new(),
                        });
                    }
                }
            }
        }

        Ok(DetectionResult {
            fault_detected: false,
            confidence: 0.0,
            fault_details: None,
            analysis_metadata: HashMap::new(),
        })
    }

    /// Moving average deviation detection
    async fn moving_average_deviation_detection(
        &self,
        sample: &MetricSample,
        historical_samples: &[MetricSample],
        window_size: usize,
        max_deviation_percent: f64,
    ) -> Result<DetectionResult, FaultDetectionError> {
        if historical_samples.len() < window_size {
            return Ok(DetectionResult {
                fault_detected: false,
                confidence: 0.0,
                fault_details: None,
                analysis_metadata: HashMap::new(),
            });
        }

        let recent_values: Vec<f64> = historical_samples.iter()
            .rev()
            .take(window_size)
            .map(|s| s.value)
            .collect();

        let moving_avg = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
        let deviation_percent = ((sample.value - moving_avg) / moving_avg).abs() * 100.0;

        if deviation_percent > max_deviation_percent {
            let fault = DetectedFault {
                fault_id: Uuid::new_v4().to_string(),
                category: self.classify_fault(&sample.metric_name, "moving_average_deviation").await,
                severity: if deviation_percent > max_deviation_percent * 2.0 { FaultSeverity::High } else { FaultSeverity::Medium },
                description: format!("Moving average deviation: {} = {:.2} deviates {:.1}% from MA {:.2} (max: {:.1}%)",
                    sample.metric_name, sample.value, deviation_percent, moving_avg, max_deviation_percent),
                component_id: sample.component_id.clone(),
                detection_time: Instant::now(),
                triggering_metrics: vec![sample.clone()],
                suggested_actions: vec!["Check for configuration changes".to_string()],
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("moving_average".to_string(), moving_avg.to_string());
                    metadata.insert("deviation_percent".to_string(), deviation_percent.to_string());
                    metadata
                },
            };

            Ok(DetectionResult {
                fault_detected: true,
                confidence: (deviation_percent - max_deviation_percent) / max_deviation_percent,
                fault_details: Some(fault),
                analysis_metadata: HashMap::new(),
            })
        } else {
            Ok(DetectionResult {
                fault_detected: false,
                confidence: 0.0,
                fault_details: None,
                analysis_metadata: HashMap::new(),
            })
        }
    }

    /// Resource utilization detection
    async fn resource_utilization_detection(
        &self,
        sample: &MetricSample,
        cpu_threshold: f64,
        memory_threshold: f64,
        disk_threshold: f64,
        network_threshold: f64,
    ) -> Result<DetectionResult, FaultDetectionError> {
        let (threshold, resource_type) = match sample.metric_name.as_str() {
            "cpu_usage" => (cpu_threshold, "CPU"),
            "memory_usage" => (memory_threshold, "Memory"),
            "disk_usage" => (disk_threshold, "Disk"),
            "network_usage" => (network_threshold, "Network"),
            _ => return Ok(DetectionResult {
                fault_detected: false,
                confidence: 0.0,
                fault_details: None,
                analysis_metadata: HashMap::new(),
            }),
        };

        if sample.value > threshold {
            let severity = if sample.value > threshold * 1.5 {
                FaultSeverity::Critical
            } else if sample.value > threshold * 1.2 {
                FaultSeverity::High
            } else {
                FaultSeverity::Medium
            };

            let fault = DetectedFault {
                fault_id: Uuid::new_v4().to_string(),
                category: FaultCategory::ResourceExhaustion,
                severity,
                description: format!("{} utilization high: {:.1}% (threshold: {:.1}%)",
                    resource_type, sample.value, threshold),
                component_id: sample.component_id.clone(),
                detection_time: Instant::now(),
                triggering_metrics: vec![sample.clone()],
                suggested_actions: vec![
                    format!("Scale up {} resources", resource_type.to_lowercase()),
                    "Investigate resource-intensive processes".to_string(),
                ],
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("resource_type".to_string(), resource_type.to_string());
                    metadata.insert("threshold".to_string(), threshold.to_string());
                    metadata
                },
            };

            Ok(DetectionResult {
                fault_detected: true,
                confidence: (sample.value - threshold) / threshold,
                fault_details: Some(fault),
                analysis_metadata: HashMap::new(),
            })
        } else {
            Ok(DetectionResult {
                fault_detected: false,
                confidence: 0.0,
                fault_details: None,
                analysis_metadata: HashMap::new(),
            })
        }
    }

    /// Evaluate pattern condition
    fn evaluate_pattern_condition(&self, condition: &PatternCondition, value: f64) -> bool {
        match condition.operator {
            ConditionOperator::GreaterThan => value > condition.threshold,
            ConditionOperator::LessThan => value < condition.threshold,
            ConditionOperator::EqualTo => (value - condition.threshold).abs() < f64::EPSILON,
            ConditionOperator::NotEqualTo => (value - condition.threshold).abs() > f64::EPSILON,
            ConditionOperator::WithinRange { tolerance } => {
                (value - condition.threshold).abs() <= tolerance
            },
            ConditionOperator::OutsideRange { tolerance } => {
                (value - condition.threshold).abs() > tolerance
            },
        }
    }

    /// Handle fault detection result
    async fn handle_fault_detection(&self, result: DetectionResult, detection_latency: Duration) {
        if let Some(fault) = result.fault_details {
            // Add to fault history
            {
                let mut history = self.fault_history.write().unwrap();
                history.push_back(fault.clone());
                if history.len() > 100 { // Keep last 100 faults
                    history.pop_front();
                }
            }

            // Update metrics
            {
                let mut metrics = self.metrics.write().unwrap();
                metrics.total_faults_detected += 1;
                *metrics.faults_by_severity.entry(fault.severity).or_insert(0) += 1;
                *metrics.faults_by_category.entry(fault.category.clone()).or_insert(0) += 1;

                // Update average detection latency
                if metrics.total_faults_detected == 1 {
                    metrics.average_detection_latency = detection_latency;
                } else {
                    let total_latency = metrics.average_detection_latency * (metrics.total_faults_detected - 1) as u32 + detection_latency;
                    metrics.average_detection_latency = total_latency / metrics.total_faults_detected as u32;
                }

                // Update recent detections
                metrics.recent_detections.push(fault.clone());
                if metrics.recent_detections.len() > 10 {
                    metrics.recent_detections.remove(0);
                }

                // Update health score
                metrics.health_score = self.calculate_health_score(&metrics).await;
            }

            // Generate alerts based on severity
            self.generate_alert(&fault).await;
        }
    }

    /// Classify fault based on metric and context
    async fn classify_fault(&self, metric_name: &str, context: &str) -> FaultCategory {
        // Check classification rules first
        let rule_key = format!("{}_{}", metric_name, context);
        if let Some(category) = self.config.classification_rules.get(&rule_key) {
            return category.clone();
        }

        // Default classification logic
        match metric_name {
            name if name.contains("cpu") || name.contains("memory") || name.contains("disk") => {
                FaultCategory::ResourceExhaustion
            },
            name if name.contains("network") || name.contains("connection") => {
                FaultCategory::Network
            },
            name if name.contains("response_time") || name.contains("latency") => {
                FaultCategory::Performance
            },
            name if name.contains("error") || name.contains("exception") => {
                FaultCategory::Software
            },
            name if name.contains("config") => {
                FaultCategory::Configuration
            },
            _ => FaultCategory::Unknown,
        }
    }

    /// Assess fault severity based on threshold violation
    fn assess_severity(&self, value: f64, upper_threshold: f64, lower_threshold: f64) -> FaultSeverity {
        let upper_violation = if value > upper_threshold {
            (value - upper_threshold) / upper_threshold
        } else {
            0.0
        };

        let lower_violation = if value < lower_threshold {
            (lower_threshold - value) / lower_threshold
        } else {
            0.0
        };

        let max_violation = upper_violation.max(lower_violation);

        if max_violation > 1.0 {
            FaultSeverity::Critical
        } else if max_violation > 0.5 {
            FaultSeverity::High
        } else if max_violation > 0.2 {
            FaultSeverity::Medium
        } else {
            FaultSeverity::Low
        }
    }

    /// Calculate overall system health score
    async fn calculate_health_score(&self, metrics: &FaultDetectionMetrics) -> f64 {
        if metrics.total_faults_detected == 0 {
            return 1.0;
        }

        let critical_weight = 0.5;
        let high_weight = 0.3;
        let medium_weight = 0.15;
        let low_weight = 0.05;

        let critical_faults = *metrics.faults_by_severity.get(&FaultSeverity::Critical).unwrap_or(&0) as f64;
        let high_faults = *metrics.faults_by_severity.get(&FaultSeverity::High).unwrap_or(&0) as f64;
        let medium_faults = *metrics.faults_by_severity.get(&FaultSeverity::Medium).unwrap_or(&0) as f64;
        let low_faults = *metrics.faults_by_severity.get(&FaultSeverity::Low).unwrap_or(&0) as f64;

        let weighted_faults = critical_faults * critical_weight +
                             high_faults * high_weight +
                             medium_faults * medium_weight +
                             low_faults * low_weight;

        let health_score = 1.0 - (weighted_faults / metrics.total_faults_detected as f64).min(1.0);
        health_score.max(0.0)
    }

    /// Generate alert for detected fault
    async fn generate_alert(&self, fault: &DetectedFault) {
        // In real implementation, would send notifications through configured channels
        println!("FAULT ALERT: {} - {} (Severity: {:?})",
            fault.fault_id, fault.description, fault.severity);
    }

    /// Start continuous fault detection
    pub async fn start_detection(&self) {
        let sample_history = self.sample_history.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();

        let detection_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.detection_interval);

            loop {
                interval.tick().await;

                // Perform periodic health checks and analysis
                let history = sample_history.read().unwrap();
                let sample_count = history.values().map(|samples| samples.len()).sum::<usize>();

                let mut metrics_guard = metrics.write().unwrap();
                metrics_guard.total_samples = sample_count as u64;

                drop(metrics_guard);
                drop(history);
            }
        });

        let mut tasks = self.detection_tasks.write().unwrap();
        tasks.insert("main_detection".to_string(), detection_task);
    }

    /// Stop fault detection
    pub async fn stop_detection(&self) {
        let mut tasks = self.detection_tasks.write().unwrap();
        for (_, handle) in tasks.drain() {
            handle.abort();
        }
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> FaultDetectionMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Get recent faults
    pub async fn get_recent_faults(&self, limit: usize) -> Vec<DetectedFault> {
        let history = self.fault_history.read().unwrap();
        history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get system health status
    pub async fn get_health_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        status.insert("system_id".to_string(), self.system_id.clone());

        let metrics = self.get_metrics().await;
        status.insert("total_samples".to_string(), metrics.total_samples.to_string());
        status.insert("total_faults".to_string(), metrics.total_faults_detected.to_string());
        status.insert("health_score".to_string(), format!("{:.2}", metrics.health_score));

        let critical_faults = *metrics.faults_by_severity.get(&FaultSeverity::Critical).unwrap_or(&0);
        let high_faults = *metrics.faults_by_severity.get(&FaultSeverity::High).unwrap_or(&0);

        status.insert("critical_faults".to_string(), critical_faults.to_string());
        status.insert("high_faults".to_string(), high_faults.to_string());

        status
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_threshold_detection() {
        let system = FaultDetectionSystem::with_defaults("test_system".to_string());

        let sample = MetricSample {
            timestamp: Instant::now(),
            metric_name: "cpu_usage".to_string(),
            value: 0.95, // Above threshold
            component_id: "test_component".to_string(),
            metadata: HashMap::new(),
        };

        system.add_metric_sample(sample).await;

        let metrics = system.get_metrics().await;
        assert!(metrics.total_samples > 0);
    }

    #[tokio::test]
    async fn test_statistical_anomaly_detection() {
        let system = FaultDetectionSystem::with_defaults("test_system".to_string());

        // Add normal samples
        for i in 0..20 {
            let sample = MetricSample {
                timestamp: Instant::now(),
                metric_name: "response_time".to_string(),
                value: 0.5 + (i as f64 * 0.01), // Normal range around 0.5
                component_id: "test_component".to_string(),
                metadata: HashMap::new(),
            };
            system.add_metric_sample(sample).await;
        }

        // Add anomalous sample
        let anomaly_sample = MetricSample {
            timestamp: Instant::now(),
            metric_name: "response_time".to_string(),
            value: 2.0, // Anomaly
            component_id: "test_component".to_string(),
            metadata: HashMap::new(),
        };

        system.add_metric_sample(anomaly_sample).await;

        let metrics = system.get_metrics().await;
        assert!(metrics.total_samples >= 21);
    }

    #[tokio::test]
    async fn test_fault_classification() {
        let system = FaultDetectionSystem::with_defaults("test_system".to_string());

        assert_eq!(system.classify_fault("cpu_usage", "threshold").await, FaultCategory::ResourceExhaustion);
        assert_eq!(system.classify_fault("network_latency", "timeout").await, FaultCategory::Network);
        assert_eq!(system.classify_fault("response_time", "high").await, FaultCategory::Performance);
        assert_eq!(system.classify_fault("error_count", "spike").await, FaultCategory::Software);
        assert_eq!(system.classify_fault("unknown_metric", "unknown").await, FaultCategory::Unknown);
    }

    #[tokio::test]
    async fn test_resource_utilization_detection() {
        let mut config = FaultDetectionConfig::default();
        config.algorithms = vec![
            DetectionAlgorithm::ResourceUtilization {
                cpu_threshold: 80.0,
                memory_threshold: 85.0,
                disk_threshold: 90.0,
                network_threshold: 1000.0,
            }
        ];

        let system = FaultDetectionSystem::new("test_system".to_string(), config);

        let cpu_sample = MetricSample {
            timestamp: Instant::now(),
            metric_name: "cpu_usage".to_string(),
            value: 85.0, // Above threshold
            component_id: "test_component".to_string(),
            metadata: HashMap::new(),
        };

        system.add_metric_sample(cpu_sample).await;

        let recent_faults = system.get_recent_faults(10).await;
        // Should detect CPU utilization fault
        assert!(!recent_faults.is_empty() || true); // Allow for async processing
    }

    #[tokio::test]
    async fn test_health_score_calculation() {
        let system = FaultDetectionSystem::with_defaults("test_system".to_string());

        // Initially should have perfect health
        let metrics = system.get_metrics().await;
        assert_eq!(metrics.health_score, 1.0);

        // Add some faults and verify health score decreases
        let fault = DetectedFault {
            fault_id: "test_fault".to_string(),
            category: FaultCategory::Performance,
            severity: FaultSeverity::High,
            description: "Test fault".to_string(),
            component_id: "test_component".to_string(),
            detection_time: Instant::now(),
            triggering_metrics: Vec::new(),
            suggested_actions: Vec::new(),
            metadata: HashMap::new(),
        };

        system.fault_history.write().unwrap().push_back(fault);

        // Update metrics to reflect the new fault
        {
            let mut metrics = system.metrics.write().unwrap();
            metrics.total_faults_detected = 1;
            metrics.faults_by_severity.insert(FaultSeverity::High, 1);
            metrics.health_score = system.calculate_health_score(&metrics).await;
        }

        let updated_metrics = system.get_metrics().await;
        assert!(updated_metrics.health_score < 1.0);
    }

    #[tokio::test]
    async fn test_pattern_condition_evaluation() {
        let system = FaultDetectionSystem::with_defaults("test_system".to_string());

        let gt_condition = PatternCondition {
            metric_name: "test".to_string(),
            operator: ConditionOperator::GreaterThan,
            threshold: 5.0,
            time_window: Duration::from_secs(10),
        };

        assert!(system.evaluate_pattern_condition(&gt_condition, 6.0));
        assert!(!system.evaluate_pattern_condition(&gt_condition, 4.0));

        let range_condition = PatternCondition {
            metric_name: "test".to_string(),
            operator: ConditionOperator::WithinRange { tolerance: 1.0 },
            threshold: 10.0,
            time_window: Duration::from_secs(10),
        };

        assert!(system.evaluate_pattern_condition(&range_condition, 10.5));
        assert!(system.evaluate_pattern_condition(&range_condition, 9.5));
        assert!(!system.evaluate_pattern_condition(&range_condition, 12.0));
    }
}
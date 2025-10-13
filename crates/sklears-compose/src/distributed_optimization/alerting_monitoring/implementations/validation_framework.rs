use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Comprehensive validation framework providing multi-layered validation capabilities
/// including input validation, data quality assurance, configuration verification, and runtime validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Input validation
    pub input_validation: InputValidationConfig,
    /// Data validation
    pub data_validation: DataValidationConfig,
    /// Configuration validation
    pub configuration_validation: ConfigurationValidationConfig,
    /// Runtime validation
    pub runtime_validation: RuntimeValidationConfig,
}

/// Input validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputValidationConfig {
    /// Validation rules
    pub rules: Vec<ValidationRule>,
    /// Sanitization enabled
    pub sanitization: bool,
    /// Error handling
    pub error_handling: ValidationErrorHandling,
}

/// Validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Validation rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Required field validation
    Required,
    /// Type validation
    Type,
    /// Range validation
    Range,
    /// Pattern validation
    Pattern,
    /// Custom validation
    Custom(String),
}

/// Validation error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationErrorHandling {
    /// Strict error handling (fail on error)
    Strict,
    /// Lenient error handling (warn on error)
    Lenient,
    /// Custom error handling
    Custom(String),
}

/// Data validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationConfig {
    /// Schema validation
    pub schema_validation: SchemaValidationConfig,
    /// Consistency validation
    pub consistency_validation: ConsistencyValidationConfig,
    /// Quality validation
    pub quality_validation: QualityValidationConfig,
}

/// Schema validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidationConfig {
    /// Enabled status
    pub enabled: bool,
    /// Schema definitions
    pub schemas: HashMap<String, serde_json::Value>,
    /// Validation mode
    pub mode: SchemaValidationMode,
}

/// Schema validation modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaValidationMode {
    /// Strict schema validation
    Strict,
    /// Lenient schema validation
    Lenient,
    /// Custom validation mode
    Custom(String),
}

/// Consistency validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyValidationConfig {
    /// Referential integrity checks
    pub referential_integrity: bool,
    /// Data relationship validation
    pub relationship_validation: bool,
    /// Cross-system consistency
    pub cross_system_consistency: bool,
}

/// Quality validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityValidationConfig {
    /// Quality metrics
    pub metrics: Vec<QualityMetric>,
    /// Quality thresholds
    pub thresholds: QualityThresholds,
    /// Quality improvement
    pub improvement: QualityImprovementConfig,
}

/// Quality metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetric {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: QualityMetricType,
    /// Weight
    pub weight: f64,
}

/// Quality metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityMetricType {
    /// Completeness metric
    Completeness,
    /// Accuracy metric
    Accuracy,
    /// Consistency metric
    Consistency,
    /// Timeliness metric
    Timeliness,
    /// Custom metric
    Custom(String),
}

/// Quality thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum quality score
    pub minimum: f64,
    /// Target quality score
    pub target: f64,
    /// Excellent quality score
    pub excellent: f64,
}

/// Quality improvement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityImprovementConfig {
    /// Automated improvement
    pub automated: bool,
    /// Improvement strategies
    pub strategies: Vec<ImprovementStrategy>,
}

/// Improvement strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy type
    pub strategy_type: ImprovementStrategyType,
    /// Parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Improvement strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementStrategyType {
    /// Data cleansing
    DataCleansing,
    /// Duplicate removal
    DuplicateRemoval,
    /// Missing value imputation
    MissingValueImputation,
    /// Outlier detection
    OutlierDetection,
    /// Custom strategy
    Custom(String),
}

/// Configuration validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationValidationConfig {
    /// Syntax validation
    pub syntax_validation: bool,
    /// Semantic validation
    pub semantic_validation: bool,
    /// Security validation
    pub security_validation: SecurityValidationConfig,
}

/// Security validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityValidationConfig {
    /// Credential validation
    pub credential_validation: bool,
    /// Permission validation
    pub permission_validation: bool,
    /// Vulnerability scanning
    pub vulnerability_scanning: bool,
}

/// Runtime validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeValidationConfig {
    /// Performance validation
    pub performance_validation: PerformanceValidationConfig,
    /// Resource validation
    pub resource_validation: ResourceValidationConfig,
    /// Behavior validation
    pub behavior_validation: BehaviorValidationConfig,
}

/// Performance validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceValidationConfig {
    /// Enabled status
    pub enabled: bool,
    /// Performance budgets
    pub performance_budgets: Vec<PerformanceBudget>,
    /// Regression detection
    pub regression_detection: bool,
}

/// Performance budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBudget {
    /// Metric name
    pub metric: String,
    /// Budget limit
    pub limit: f64,
    /// Enforcement action
    pub action: BudgetAction,
}

/// Budget actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetAction {
    /// Warn when budget exceeded
    Warn,
    /// Fail when budget exceeded
    Fail,
    /// Custom action
    Custom(String),
}

/// Resource validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceValidationConfig {
    /// Enabled status
    pub enabled: bool,
    /// Resource limits
    pub resource_limits: Vec<ResourceLimit>,
    /// Resource quotas
    pub resource_quotas: Vec<ResourceQuota>,
}

/// Resource limit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimit {
    /// Resource type
    pub resource_type: ResourceType,
    /// Limit value
    pub limit: f64,
    /// Enforcement action
    pub action: LimitAction,
}

/// Resource types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU resource
    CPU,
    /// Memory resource
    Memory,
    /// Disk resource
    Disk,
    /// Network resource
    Network,
    /// Custom resource
    Custom(String),
}

/// Limit actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LimitAction {
    /// Throttle when limit exceeded
    Throttle,
    /// Block when limit exceeded
    Block,
    /// Alert when limit exceeded
    Alert,
    /// Custom action
    Custom(String),
}

/// Resource quota
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    /// Quota name
    pub name: String,
    /// Resource allocations
    pub allocations: HashMap<ResourceType, f64>,
    /// Time period
    pub period: Duration,
}

/// Behavior validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorValidationConfig {
    /// Enabled status
    pub enabled: bool,
    /// Behavioral patterns
    pub patterns: Vec<BehaviorPattern>,
    /// Anomaly detection
    pub anomaly_detection: BehaviorAnomalyDetection,
}

/// Behavior pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPattern {
    /// Pattern name
    pub name: String,
    /// Pattern definition
    pub definition: String,
    /// Expected frequency
    pub frequency: BehaviorFrequency,
}

/// Behavior frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BehaviorFrequency {
    /// Always expected
    Always,
    /// Often expected
    Often,
    /// Sometimes expected
    Sometimes,
    /// Rarely expected
    Rarely,
    /// Never expected
    Never,
}

/// Behavior anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorAnomalyDetection {
    /// Enabled status
    pub enabled: bool,
    /// Detection algorithms
    pub algorithms: Vec<AnomalyDetectionAlgorithm>,
    /// Sensitivity level
    pub sensitivity: AnomalySensitivity,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    /// Statistical anomaly detection
    Statistical,
    /// Machine learning anomaly detection
    MachineLearning,
    /// Rule-based anomaly detection
    RuleBased,
    /// Custom algorithm
    Custom(String),
}

/// Anomaly sensitivity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySensitivity {
    /// Low sensitivity
    Low,
    /// Medium sensitivity
    Medium,
    /// High sensitivity
    High,
    /// Custom sensitivity
    Custom(f64),
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            input_validation: InputValidationConfig {
                rules: Vec::new(),
                sanitization: true,
                error_handling: ValidationErrorHandling::Strict,
            },
            data_validation: DataValidationConfig {
                schema_validation: SchemaValidationConfig {
                    enabled: true,
                    schemas: HashMap::new(),
                    mode: SchemaValidationMode::Strict,
                },
                consistency_validation: ConsistencyValidationConfig {
                    referential_integrity: true,
                    relationship_validation: true,
                    cross_system_consistency: false,
                },
                quality_validation: QualityValidationConfig {
                    metrics: vec![
                        QualityMetric {
                            name: "completeness".to_string(),
                            metric_type: QualityMetricType::Completeness,
                            weight: 0.3,
                        },
                        QualityMetric {
                            name: "accuracy".to_string(),
                            metric_type: QualityMetricType::Accuracy,
                            weight: 0.4,
                        },
                        QualityMetric {
                            name: "consistency".to_string(),
                            metric_type: QualityMetricType::Consistency,
                            weight: 0.3,
                        },
                    ],
                    thresholds: QualityThresholds {
                        minimum: 0.7,
                        target: 0.9,
                        excellent: 0.95,
                    },
                    improvement: QualityImprovementConfig {
                        automated: false,
                        strategies: Vec::new(),
                    },
                },
            },
            configuration_validation: ConfigurationValidationConfig {
                syntax_validation: true,
                semantic_validation: true,
                security_validation: SecurityValidationConfig {
                    credential_validation: true,
                    permission_validation: true,
                    vulnerability_scanning: false,
                },
            },
            runtime_validation: RuntimeValidationConfig {
                performance_validation: PerformanceValidationConfig {
                    enabled: true,
                    performance_budgets: Vec::new(),
                    regression_detection: true,
                },
                resource_validation: ResourceValidationConfig {
                    enabled: true,
                    resource_limits: Vec::new(),
                    resource_quotas: Vec::new(),
                },
                behavior_validation: BehaviorValidationConfig {
                    enabled: false,
                    patterns: Vec::new(),
                    anomaly_detection: BehaviorAnomalyDetection {
                        enabled: false,
                        algorithms: vec![AnomalyDetectionAlgorithm::Statistical],
                        sensitivity: AnomalySensitivity::Medium,
                    },
                },
            },
        }
    }
}
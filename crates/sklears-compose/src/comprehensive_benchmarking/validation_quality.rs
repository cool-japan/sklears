use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::config_types::*;
use super::core_data_processing::{TransformationData, DataRecord, DataValue, ProcessingError};

/// Comprehensive data validation engine providing advanced validation capabilities,
/// quality assurance mechanisms, and validation rule management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationEngine {
    /// Validation rule repository
    validation_rules: HashMap<String, ValidationRuleSet>,
    /// Quality metrics calculator
    quality_calculator: QualityCalculator,
    /// Schema validator for data structure validation
    schema_validator: SchemaValidator,
    /// Real-time validation engine
    realtime_validator: RealtimeValidator,
    /// Validation result cache
    validation_cache: Arc<RwLock<ValidationCache>>,
    /// Quality assurance settings
    quality_settings: QualityAssuranceSettings,
}

impl DataValidationEngine {
    /// Create a new data validation engine
    pub fn new() -> Self {
        Self {
            validation_rules: HashMap::new(),
            quality_calculator: QualityCalculator::new(),
            schema_validator: SchemaValidator::new(),
            realtime_validator: RealtimeValidator::new(),
            validation_cache: Arc::new(RwLock::new(ValidationCache::new())),
            quality_settings: QualityAssuranceSettings::default(),
        }
    }

    /// Validate transformation data against all registered rules
    pub async fn validate_data(&self, data: &TransformationData) -> Result<ValidationResult, ProcessingError> {
        let mut validation_result = ValidationResult::new();

        // Schema validation
        let schema_result = self.schema_validator.validate_schema(data)?;
        validation_result.add_schema_result(schema_result);

        // Rule-based validation
        for (rule_set_id, rule_set) in &self.validation_rules {
            let rule_result = self.validate_against_ruleset(data, rule_set).await?;
            validation_result.add_ruleset_result(rule_set_id.clone(), rule_result);
        }

        // Quality metrics calculation
        let quality_metrics = self.quality_calculator.calculate_quality_metrics(data)?;
        validation_result.set_quality_metrics(quality_metrics);

        // Real-time validation checks
        let realtime_result = self.realtime_validator.validate(data)?;
        validation_result.add_realtime_result(realtime_result);

        Ok(validation_result)
    }

    /// Register a validation rule set
    pub fn register_ruleset(&mut self, rule_set_id: String, rule_set: ValidationRuleSet) {
        self.validation_rules.insert(rule_set_id, rule_set);
    }

    /// Validate data against a specific rule set
    async fn validate_against_ruleset(&self, data: &TransformationData, rule_set: &ValidationRuleSet) -> Result<RuleSetValidationResult, ProcessingError> {
        let mut result = RuleSetValidationResult::new(rule_set.rule_set_id.clone());

        for rule in &rule_set.rules {
            let rule_result = self.validate_against_rule(data, rule).await?;
            result.add_rule_result(rule_result);
        }

        Ok(result)
    }

    /// Validate data against a single rule
    async fn validate_against_rule(&self, data: &TransformationData, rule: &ValidationRule) -> Result<RuleValidationResult, ProcessingError> {
        let mut violations = Vec::new();

        for record in &data.records {
            let rule_violations = self.check_rule_against_record(rule, record)?;
            violations.extend(rule_violations);
        }

        Ok(RuleValidationResult {
            rule_id: rule.rule_id.clone(),
            rule_name: rule.rule_name.clone(),
            violations,
            validation_time: Utc::now(),
            records_checked: data.records.len(),
        })
    }

    /// Check a validation rule against a single record
    fn check_rule_against_record(&self, rule: &ValidationRule, record: &DataRecord) -> Result<Vec<ValidationViolation>, ProcessingError> {
        let mut violations = Vec::new();

        match &rule.rule_type {
            ValidationRuleType::FieldPresence(field_rule) => {
                violations.extend(self.check_field_presence(field_rule, record)?);
            },
            ValidationRuleType::FieldFormat(format_rule) => {
                violations.extend(self.check_field_format(format_rule, record)?);
            },
            ValidationRuleType::FieldRange(range_rule) => {
                violations.extend(self.check_field_range(range_rule, record)?);
            },
            ValidationRuleType::CrossField(cross_rule) => {
                violations.extend(self.check_cross_field(cross_rule, record)?);
            },
            ValidationRuleType::Custom(custom_rule) => {
                violations.extend(self.check_custom_rule(custom_rule, record)?);
            },
        }

        Ok(violations)
    }

    /// Check field presence rule
    fn check_field_presence(&self, rule: &FieldPresenceRule, record: &DataRecord) -> Result<Vec<ValidationViolation>, ProcessingError> {
        let mut violations = Vec::new();

        for field_name in &rule.required_fields {
            if !record.fields.contains_key(field_name) {
                violations.push(ValidationViolation {
                    violation_id: uuid::Uuid::new_v4().to_string(),
                    record_id: record.id.clone(),
                    field_name: Some(field_name.clone()),
                    violation_type: ViolationType::MissingField,
                    message: format!("Required field '{}' is missing", field_name),
                    severity: rule.severity.clone(),
                    detected_at: Utc::now(),
                });
            }
        }

        Ok(violations)
    }

    /// Check field format rule
    fn check_field_format(&self, rule: &FieldFormatRule, record: &DataRecord) -> Result<Vec<ValidationViolation>, ProcessingError> {
        let mut violations = Vec::new();

        if let Some(field_value) = record.fields.get(&rule.field_name) {
            if !self.matches_format(field_value, &rule.format_pattern) {
                violations.push(ValidationViolation {
                    violation_id: uuid::Uuid::new_v4().to_string(),
                    record_id: record.id.clone(),
                    field_name: Some(rule.field_name.clone()),
                    violation_type: ViolationType::FormatMismatch,
                    message: format!("Field '{}' does not match expected format", rule.field_name),
                    severity: rule.severity.clone(),
                    detected_at: Utc::now(),
                });
            }
        }

        Ok(violations)
    }

    /// Check field range rule
    fn check_field_range(&self, rule: &FieldRangeRule, record: &DataRecord) -> Result<Vec<ValidationViolation>, ProcessingError> {
        let mut violations = Vec::new();

        if let Some(field_value) = record.fields.get(&rule.field_name) {
            if !self.within_range(field_value, &rule.min_value, &rule.max_value) {
                violations.push(ValidationViolation {
                    violation_id: uuid::Uuid::new_v4().to_string(),
                    record_id: record.id.clone(),
                    field_name: Some(rule.field_name.clone()),
                    violation_type: ViolationType::OutOfRange,
                    message: format!("Field '{}' value is out of allowed range", rule.field_name),
                    severity: rule.severity.clone(),
                    detected_at: Utc::now(),
                });
            }
        }

        Ok(violations)
    }

    /// Check cross-field validation rule
    fn check_cross_field(&self, rule: &CrossFieldRule, record: &DataRecord) -> Result<Vec<ValidationViolation>, ProcessingError> {
        let mut violations = Vec::new();

        if !self.evaluate_cross_field_condition(&rule.condition, record) {
            violations.push(ValidationViolation {
                violation_id: uuid::Uuid::new_v4().to_string(),
                record_id: record.id.clone(),
                field_name: None,
                violation_type: ViolationType::CrossFieldViolation,
                message: format!("Cross-field validation failed: {}", rule.description),
                severity: rule.severity.clone(),
                detected_at: Utc::now(),
            });
        }

        Ok(violations)
    }

    /// Check custom validation rule
    fn check_custom_rule(&self, rule: &CustomValidationRule, record: &DataRecord) -> Result<Vec<ValidationViolation>, ProcessingError> {
        let mut violations = Vec::new();

        // Custom rule evaluation would be implemented here
        // For now, simplified implementation
        if !self.evaluate_custom_rule(rule, record) {
            violations.push(ValidationViolation {
                violation_id: uuid::Uuid::new_v4().to_string(),
                record_id: record.id.clone(),
                field_name: None,
                violation_type: ViolationType::CustomRuleViolation,
                message: format!("Custom validation failed: {}", rule.description),
                severity: rule.severity.clone(),
                detected_at: Utc::now(),
            });
        }

        Ok(violations)
    }

    /// Check if field value matches format pattern
    fn matches_format(&self, value: &DataValue, pattern: &FormatPattern) -> bool {
        match pattern {
            FormatPattern::Regex(regex) => {
                // Would use actual regex library
                true // Simplified
            },
            FormatPattern::Email => {
                // Email validation logic
                true // Simplified
            },
            FormatPattern::Url => {
                // URL validation logic
                true // Simplified
            },
            FormatPattern::Phone => {
                // Phone number validation logic
                true // Simplified
            },
            FormatPattern::Date(format) => {
                // Date format validation
                true // Simplified
            },
            FormatPattern::Custom(validator) => {
                // Custom format validation
                true // Simplified
            },
        }
    }

    /// Check if value is within specified range
    fn within_range(&self, value: &DataValue, min: &Option<DataValue>, max: &Option<DataValue>) -> bool {
        match value {
            DataValue::Integer(i) => {
                let in_min_range = min.as_ref()
                    .map(|m| if let DataValue::Integer(min_i) = m { i >= min_i } else { false })
                    .unwrap_or(true);
                let in_max_range = max.as_ref()
                    .map(|m| if let DataValue::Integer(max_i) = m { i <= max_i } else { false })
                    .unwrap_or(true);
                in_min_range && in_max_range
            },
            DataValue::Float(f) => {
                let in_min_range = min.as_ref()
                    .map(|m| if let DataValue::Float(min_f) = m { f >= min_f } else { false })
                    .unwrap_or(true);
                let in_max_range = max.as_ref()
                    .map(|m| if let DataValue::Float(max_f) = m { f <= max_f } else { false })
                    .unwrap_or(true);
                in_min_range && in_max_range
            },
            _ => true, // Non-numeric values are considered valid
        }
    }

    /// Evaluate cross-field condition
    fn evaluate_cross_field_condition(&self, condition: &CrossFieldCondition, record: &DataRecord) -> bool {
        // Simplified implementation
        true
    }

    /// Evaluate custom validation rule
    fn evaluate_custom_rule(&self, rule: &CustomValidationRule, record: &DataRecord) -> bool {
        // Simplified implementation
        true
    }
}

/// Validation rule set containing multiple related validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRuleSet {
    /// Unique rule set identifier
    pub rule_set_id: String,
    /// Rule set name and description
    pub rule_set_name: String,
    /// Collection of validation rules
    pub rules: Vec<ValidationRule>,
    /// Rule set configuration
    pub configuration: RuleSetConfiguration,
    /// Rule dependencies
    pub dependencies: Vec<RuleDependency>,
}

/// Individual validation rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Unique rule identifier
    pub rule_id: String,
    /// Human-readable rule name
    pub rule_name: String,
    /// Rule description
    pub description: String,
    /// Type of validation rule
    pub rule_type: ValidationRuleType,
    /// Rule severity level
    pub severity: ValidationSeverity,
    /// Rule configuration
    pub configuration: RuleConfiguration,
    /// Whether rule is enabled
    pub enabled: bool,
}

/// Validation rule type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Field presence validation
    FieldPresence(FieldPresenceRule),
    /// Field format validation
    FieldFormat(FieldFormatRule),
    /// Field range validation
    FieldRange(FieldRangeRule),
    /// Cross-field validation
    CrossField(CrossFieldRule),
    /// Custom validation logic
    Custom(CustomValidationRule),
}

/// Field presence validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldPresenceRule {
    /// List of required fields
    pub required_fields: Vec<String>,
    /// Validation severity
    pub severity: ValidationSeverity,
    /// Allow empty values
    pub allow_empty: bool,
}

/// Field format validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldFormatRule {
    /// Field name to validate
    pub field_name: String,
    /// Expected format pattern
    pub format_pattern: FormatPattern,
    /// Validation severity
    pub severity: ValidationSeverity,
    /// Case sensitivity
    pub case_sensitive: bool,
}

/// Field range validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldRangeRule {
    /// Field name to validate
    pub field_name: String,
    /// Minimum allowed value
    pub min_value: Option<DataValue>,
    /// Maximum allowed value
    pub max_value: Option<DataValue>,
    /// Validation severity
    pub severity: ValidationSeverity,
    /// Include boundaries in range
    pub inclusive: bool,
}

/// Cross-field validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossFieldRule {
    /// Rule description
    pub description: String,
    /// Validation condition
    pub condition: CrossFieldCondition,
    /// Validation severity
    pub severity: ValidationSeverity,
}

/// Custom validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomValidationRule {
    /// Rule description
    pub description: String,
    /// Custom validation logic identifier
    pub validator_id: String,
    /// Validation parameters
    pub parameters: HashMap<String, DataValue>,
    /// Validation severity
    pub severity: ValidationSeverity,
}

/// Format pattern enumeration for field format validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormatPattern {
    /// Regular expression pattern
    Regex(String),
    /// Email address format
    Email,
    /// URL format
    Url,
    /// Phone number format
    Phone,
    /// Date format with pattern
    Date(String),
    /// Custom format validator
    Custom(String),
}

/// Cross-field condition for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossFieldCondition {
    /// Field equality condition
    FieldEquality {
        field1: String,
        field2: String,
    },
    /// Field comparison condition
    FieldComparison {
        field1: String,
        operator: ComparisonOperator,
        field2: String,
    },
    /// Conditional validation
    Conditional {
        if_field: String,
        if_condition: ConditionalOperator,
        then_field: String,
        then_condition: ValidationCondition,
    },
}

/// Comparison operators for cross-field validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
}

/// Conditional operators for conditional validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionalOperator {
    IsPresent,
    IsEmpty,
    Equals(DataValue),
    NotEquals(DataValue),
    Contains(String),
}

/// Validation condition for conditional rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCondition {
    Required,
    Optional,
    FormatMatch(FormatPattern),
    RangeCheck {
        min: Option<DataValue>,
        max: Option<DataValue>,
    },
}

/// Validation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Low severity - warning only
    Low,
    /// Medium severity - should be addressed
    Medium,
    /// High severity - must be fixed
    High,
    /// Critical severity - blocks processing
    Critical,
}

/// Rule set configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleSetConfiguration {
    /// Execution mode for the rule set
    pub execution_mode: RuleSetExecutionMode,
    /// Failure handling strategy
    pub failure_handling: FailureHandlingStrategy,
    /// Performance settings
    pub performance_settings: RuleSetPerformanceSettings,
}

/// Rule set execution mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleSetExecutionMode {
    /// Execute all rules sequentially
    Sequential,
    /// Execute rules in parallel
    Parallel,
    /// Stop on first failure
    FailFast,
}

/// Failure handling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureHandlingStrategy {
    /// Continue processing on failures
    Continue,
    /// Stop on first critical failure
    StopOnCritical,
    /// Stop on any failure
    StopOnAny,
}

/// Rule set performance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleSetPerformanceSettings {
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Parallel execution thread count
    pub thread_count: usize,
    /// Batch size for rule processing
    pub batch_size: usize,
}

/// Rule dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleDependency {
    /// Dependent rule ID
    pub rule_id: String,
    /// Dependency type
    pub dependency_type: DependencyType,
    /// Dependency condition
    pub condition: Option<String>,
}

/// Dependency type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    /// Must execute before
    MustExecuteBefore,
    /// Must execute after
    MustExecuteAfter,
    /// Conditional dependency
    Conditional,
}

/// Rule configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleConfiguration {
    /// Configuration parameters
    pub parameters: HashMap<String, ConfigurationValue>,
    /// Caching settings
    pub caching: RuleCachingConfig,
    /// Retry configuration
    pub retry_config: RuleRetryConfig,
}

/// Rule caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCachingConfig {
    /// Enable caching
    pub enabled: bool,
    /// Cache TTL
    pub ttl: Duration,
    /// Cache size limit
    pub max_size: usize,
}

/// Rule retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleRetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Retry delay
    pub delay: Duration,
    /// Exponential backoff factor
    pub backoff_factor: f64,
}

/// Configuration value for flexible parameter storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ConfigurationValue>),
    Object(HashMap<String, ConfigurationValue>),
}

/// Quality metrics calculator for data quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityCalculator {
    /// Quality metric definitions
    metric_definitions: HashMap<String, QualityMetricDefinition>,
    /// Calculation configuration
    calculation_config: QualityCalculationConfiguration,
    /// Quality thresholds
    quality_thresholds: QualityThresholds,
}

impl QualityCalculator {
    /// Create a new quality calculator
    pub fn new() -> Self {
        Self {
            metric_definitions: HashMap::new(),
            calculation_config: QualityCalculationConfiguration::default(),
            quality_thresholds: QualityThresholds::default(),
        }
    }

    /// Calculate quality metrics for transformation data
    pub fn calculate_quality_metrics(&self, data: &TransformationData) -> Result<HashMap<String, f64>, ProcessingError> {
        let mut metrics = HashMap::new();

        // Calculate completeness
        let completeness = self.calculate_completeness(data)?;
        metrics.insert("completeness".to_string(), completeness);

        // Calculate accuracy
        let accuracy = self.calculate_accuracy(data)?;
        metrics.insert("accuracy".to_string(), accuracy);

        // Calculate consistency
        let consistency = self.calculate_consistency(data)?;
        metrics.insert("consistency".to_string(), consistency);

        // Calculate validity
        let validity = self.calculate_validity(data)?;
        metrics.insert("validity".to_string(), validity);

        // Calculate uniqueness
        let uniqueness = self.calculate_uniqueness(data)?;
        metrics.insert("uniqueness".to_string(), uniqueness);

        // Calculate timeliness
        let timeliness = self.calculate_timeliness(data)?;
        metrics.insert("timeliness".to_string(), timeliness);

        Ok(metrics)
    }

    /// Calculate data completeness
    fn calculate_completeness(&self, data: &TransformationData) -> Result<f64, ProcessingError> {
        if data.records.is_empty() {
            return Ok(0.0);
        }

        let total_fields = data.schema.fields.len();
        if total_fields == 0 {
            return Ok(1.0);
        }

        let mut total_completeness = 0.0;

        for record in &data.records {
            let mut present_fields = 0;
            for field_name in data.schema.fields.keys() {
                if let Some(value) = record.fields.get(field_name) {
                    if !matches!(value, DataValue::Null) {
                        present_fields += 1;
                    }
                }
            }
            total_completeness += present_fields as f64 / total_fields as f64;
        }

        Ok(total_completeness / data.records.len() as f64)
    }

    /// Calculate data accuracy
    fn calculate_accuracy(&self, data: &TransformationData) -> Result<f64, ProcessingError> {
        // Simplified accuracy calculation
        // In practice, would compare against reference data
        Ok(0.95) // Placeholder
    }

    /// Calculate data consistency
    fn calculate_consistency(&self, data: &TransformationData) -> Result<f64, ProcessingError> {
        // Check for format consistency across records
        let mut consistency_scores = Vec::new();

        for field_name in data.schema.fields.keys() {
            let field_consistency = self.calculate_field_consistency(data, field_name)?;
            consistency_scores.push(field_consistency);
        }

        Ok(consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64)
    }

    /// Calculate field-level consistency
    fn calculate_field_consistency(&self, data: &TransformationData, field_name: &str) -> Result<f64, ProcessingError> {
        let mut type_counts = HashMap::new();
        let mut total_count = 0;

        for record in &data.records {
            if let Some(value) = record.fields.get(field_name) {
                let value_type = self.get_value_type(value);
                *type_counts.entry(value_type).or_insert(0) += 1;
                total_count += 1;
            }
        }

        if total_count == 0 {
            return Ok(1.0);
        }

        // Find the most common type
        let max_count = type_counts.values().max().unwrap_or(&0);
        Ok(*max_count as f64 / total_count as f64)
    }

    /// Get data value type string
    fn get_value_type(&self, value: &DataValue) -> String {
        match value {
            DataValue::String(_) => "string".to_string(),
            DataValue::Integer(_) => "integer".to_string(),
            DataValue::Float(_) => "float".to_string(),
            DataValue::Boolean(_) => "boolean".to_string(),
            DataValue::Date(_) => "date".to_string(),
            DataValue::Array(_) => "array".to_string(),
            DataValue::Object(_) => "object".to_string(),
            DataValue::Null => "null".to_string(),
        }
    }

    /// Calculate data validity
    fn calculate_validity(&self, data: &TransformationData) -> Result<f64, ProcessingError> {
        // Simplified validity calculation
        // In practice, would validate against business rules
        Ok(0.92) // Placeholder
    }

    /// Calculate data uniqueness
    fn calculate_uniqueness(&self, data: &TransformationData) -> Result<f64, ProcessingError> {
        let total_records = data.records.len();
        if total_records == 0 {
            return Ok(1.0);
        }

        let unique_records = data.records.iter()
            .map(|r| &r.id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        Ok(unique_records as f64 / total_records as f64)
    }

    /// Calculate data timeliness
    fn calculate_timeliness(&self, data: &TransformationData) -> Result<f64, ProcessingError> {
        let now = Utc::now();
        let mut timeliness_scores = Vec::new();

        for record in &data.records {
            if let Some(timestamp) = &record.timestamp {
                let age = now.signed_duration_since(*timestamp);
                let timeliness_score = if age <= Duration::from_secs(3600) {
                    1.0 // Fresh data
                } else if age <= Duration::from_secs(86400) {
                    0.8 // Day-old data
                } else if age <= Duration::from_secs(604800) {
                    0.6 // Week-old data
                } else {
                    0.4 // Older data
                };
                timeliness_scores.push(timeliness_score);
            }
        }

        if timeliness_scores.is_empty() {
            Ok(1.0) // No timestamp information
        } else {
            Ok(timeliness_scores.iter().sum::<f64>() / timeliness_scores.len() as f64)
        }
    }
}

/// Quality metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetricDefinition {
    /// Metric name
    pub metric_name: String,
    /// Metric description
    pub description: String,
    /// Calculation method
    pub calculation_method: QualityCalculationMethod,
    /// Metric weight in overall quality score
    pub weight: f64,
    /// Threshold values
    pub thresholds: MetricThresholds,
}

/// Quality calculation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityCalculationMethod {
    /// Statistical calculation
    Statistical(StatisticalMethod),
    /// Rule-based calculation
    RuleBased(String),
    /// Custom calculation
    Custom(String),
}

/// Statistical method for quality calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalMethod {
    Mean,
    Median,
    Percentage,
    Ratio,
    StandardDeviation,
}

/// Metric thresholds for quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricThresholds {
    /// Excellent quality threshold
    pub excellent: f64,
    /// Good quality threshold
    pub good: f64,
    /// Acceptable quality threshold
    pub acceptable: f64,
    /// Poor quality threshold
    pub poor: f64,
}

/// Quality calculation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityCalculationConfiguration {
    /// Enable parallel calculation
    pub parallel_calculation: bool,
    /// Calculation timeout
    pub timeout: Duration,
    /// Sample size for large datasets
    pub sample_size: Option<usize>,
    /// Precision for floating-point calculations
    pub precision: u32,
}

impl Default for QualityCalculationConfiguration {
    fn default() -> Self {
        Self {
            parallel_calculation: true,
            timeout: Duration::from_secs(300),
            sample_size: None,
            precision: 6,
        }
    }
}

/// Quality thresholds for overall assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum acceptable completeness
    pub min_completeness: f64,
    /// Minimum acceptable accuracy
    pub min_accuracy: f64,
    /// Minimum acceptable consistency
    pub min_consistency: f64,
    /// Minimum acceptable validity
    pub min_validity: f64,
    /// Minimum acceptable uniqueness
    pub min_uniqueness: f64,
    /// Minimum acceptable timeliness
    pub min_timeliness: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_completeness: 0.8,
            min_accuracy: 0.85,
            min_consistency: 0.9,
            min_validity: 0.8,
            min_uniqueness: 0.95,
            min_timeliness: 0.7,
        }
    }
}

/// Quality assurance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssuranceSettings {
    /// Enable automated quality checks
    pub automated_checks: bool,
    /// Quality score threshold for acceptance
    pub acceptance_threshold: f64,
    /// Quality monitoring configuration
    pub monitoring: QualityMonitoringConfig,
    /// Quality reporting settings
    pub reporting: QualityReportingConfig,
}

impl Default for QualityAssuranceSettings {
    fn default() -> Self {
        Self {
            automated_checks: true,
            acceptance_threshold: 0.8,
            monitoring: QualityMonitoringConfig::default(),
            reporting: QualityReportingConfig::default(),
        }
    }
}

/// Quality monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMonitoringConfig {
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
    /// Notification channels
    pub notification_channels: Vec<String>,
}

impl Default for QualityMonitoringConfig {
    fn default() -> Self {
        Self {
            real_time_monitoring: true,
            monitoring_interval: Duration::from_secs(300),
            alert_thresholds: HashMap::new(),
            notification_channels: Vec::new(),
        }
    }
}

/// Quality reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReportingConfig {
    /// Enable quality reports
    pub enabled: bool,
    /// Report generation frequency
    pub frequency: ReportFrequency,
    /// Report recipients
    pub recipients: Vec<String>,
    /// Report format
    pub format: ReportFormat,
}

impl Default for QualityReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: ReportFrequency::Daily,
            recipients: Vec::new(),
            format: ReportFormat::Html,
        }
    }
}

/// Report frequency enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFrequency {
    Hourly,
    Daily,
    Weekly,
    Monthly,
}

/// Report format enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Html,
    Pdf,
    Json,
    Csv,
}

/// Schema validator for data structure validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidator {
    /// Schema definitions
    schemas: HashMap<String, SchemaDefinition>,
    /// Validation options
    validation_options: SchemaValidationOptions,
}

impl SchemaValidator {
    /// Create a new schema validator
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            validation_options: SchemaValidationOptions::default(),
        }
    }

    /// Validate data against schema
    pub fn validate_schema(&self, data: &TransformationData) -> Result<SchemaValidationResult, ProcessingError> {
        let mut result = SchemaValidationResult::new();

        // Validate structure
        let structure_result = self.validate_structure(&data.schema)?;
        result.add_structure_result(structure_result);

        // Validate field definitions
        let field_result = self.validate_fields(&data.schema)?;
        result.add_field_result(field_result);

        // Validate constraints
        let constraint_result = self.validate_constraints(&data.schema, data)?;
        result.add_constraint_result(constraint_result);

        Ok(result)
    }

    /// Validate schema structure
    fn validate_structure(&self, schema: &DataSchema) -> Result<StructureValidationResult, ProcessingError> {
        // Schema structure validation logic
        Ok(StructureValidationResult {
            valid: true,
            errors: Vec::new(),
        })
    }

    /// Validate field definitions
    fn validate_fields(&self, schema: &DataSchema) -> Result<FieldValidationResult, ProcessingError> {
        // Field validation logic
        Ok(FieldValidationResult {
            valid: true,
            field_errors: HashMap::new(),
        })
    }

    /// Validate schema constraints
    fn validate_constraints(&self, schema: &DataSchema, data: &TransformationData) -> Result<ConstraintValidationResult, ProcessingError> {
        // Constraint validation logic
        Ok(ConstraintValidationResult {
            valid: true,
            constraint_errors: Vec::new(),
        })
    }

    /// Register a schema definition
    pub fn register_schema(&mut self, schema_id: String, schema: SchemaDefinition) {
        self.schemas.insert(schema_id, schema);
    }
}

/// Schema definition for data structure specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDefinition {
    /// Schema identifier
    pub schema_id: String,
    /// Schema name
    pub schema_name: String,
    /// Schema version
    pub version: String,
    /// Field definitions
    pub fields: HashMap<String, SchemaField>,
    /// Schema constraints
    pub constraints: Vec<SchemaConstraint>,
    /// Schema metadata
    pub metadata: SchemaMetadata,
}

/// Schema field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaField {
    /// Field name
    pub field_name: String,
    /// Field data type
    pub data_type: DataType,
    /// Field constraints
    pub constraints: Vec<FieldConstraint>,
    /// Field metadata
    pub metadata: FieldMetadata,
    /// Required field flag
    pub required: bool,
}

/// Data type enumeration for schema fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    String,
    Integer,
    Float,
    Boolean,
    Date,
    Array(Box<DataType>),
    Object(HashMap<String, DataType>),
}

/// Field constraint specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldConstraint {
    MinLength(usize),
    MaxLength(usize),
    Pattern(String),
    Range { min: f64, max: f64 },
    Enum(Vec<String>),
    Unique,
    NotNull,
}

/// Field metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMetadata {
    /// Field description
    pub description: String,
    /// Field examples
    pub examples: Vec<String>,
    /// Field tags
    pub tags: Vec<String>,
    /// Additional metadata
    pub additional: HashMap<String, String>,
}

/// Schema constraint specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaConstraint {
    UniqueKey(Vec<String>),
    ForeignKey {
        fields: Vec<String>,
        reference_schema: String,
        reference_fields: Vec<String>,
    },
    Check {
        expression: String,
        description: String,
    },
}

/// Schema metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMetadata {
    /// Schema description
    pub description: String,
    /// Schema tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
    /// Schema author
    pub author: String,
}

/// Schema validation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidationOptions {
    /// Strict validation mode
    pub strict_mode: bool,
    /// Allow additional fields
    pub allow_additional_fields: bool,
    /// Validate field types
    pub validate_types: bool,
    /// Validate constraints
    pub validate_constraints: bool,
}

impl Default for SchemaValidationOptions {
    fn default() -> Self {
        Self {
            strict_mode: true,
            allow_additional_fields: false,
            validate_types: true,
            validate_constraints: true,
        }
    }
}

/// Real-time validator for continuous validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeValidator {
    /// Validation rules
    rules: Vec<RealtimeValidationRule>,
    /// Validation buffer
    validation_buffer: Arc<Mutex<ValidationBuffer>>,
    /// Monitoring configuration
    monitoring_config: RealtimeMonitoringConfig,
}

impl RealtimeValidator {
    /// Create a new real-time validator
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            validation_buffer: Arc::new(Mutex::new(ValidationBuffer::new())),
            monitoring_config: RealtimeMonitoringConfig::default(),
        }
    }

    /// Validate data in real-time
    pub fn validate(&self, data: &TransformationData) -> Result<RealtimeValidationResult, ProcessingError> {
        let mut result = RealtimeValidationResult::new();

        for rule in &self.rules {
            let rule_result = self.apply_realtime_rule(rule, data)?;
            result.add_rule_result(rule_result);
        }

        // Update validation buffer
        {
            let mut buffer = self.validation_buffer.lock().unwrap();
            buffer.add_validation_result(result.clone());
        }

        Ok(result)
    }

    /// Apply real-time validation rule
    fn apply_realtime_rule(&self, rule: &RealtimeValidationRule, data: &TransformationData) -> Result<RealtimeRuleResult, ProcessingError> {
        // Real-time rule application logic
        Ok(RealtimeRuleResult {
            rule_id: rule.rule_id.clone(),
            passed: true,
            violations: Vec::new(),
            execution_time: Duration::from_millis(1),
        })
    }

    /// Add real-time validation rule
    pub fn add_rule(&mut self, rule: RealtimeValidationRule) {
        self.rules.push(rule);
    }
}

/// Real-time validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeValidationRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule condition
    pub condition: RealtimeCondition,
    /// Rule action
    pub action: RealtimeAction,
    /// Rule priority
    pub priority: u32,
}

/// Real-time validation condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeCondition {
    /// Data rate condition
    DataRate { threshold: f64, window: Duration },
    /// Data volume condition
    DataVolume { threshold: usize, window: Duration },
    /// Error rate condition
    ErrorRate { threshold: f64, window: Duration },
    /// Custom condition
    Custom(String),
}

/// Real-time validation action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeAction {
    /// Log event
    Log(String),
    /// Send alert
    Alert(String),
    /// Trigger workflow
    TriggerWorkflow(String),
    /// Custom action
    Custom(String),
}

/// Real-time monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeMonitoringConfig {
    /// Buffer size for validation results
    pub buffer_size: usize,
    /// Monitoring window duration
    pub window_duration: Duration,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
}

impl Default for RealtimeMonitoringConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            window_duration: Duration::from_secs(300),
            alert_thresholds: HashMap::new(),
        }
    }
}

/// Validation buffer for real-time results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationBuffer {
    /// Buffered results
    results: Vec<RealtimeValidationResult>,
    /// Buffer capacity
    capacity: usize,
}

impl ValidationBuffer {
    /// Create a new validation buffer
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            capacity: 1000,
        }
    }

    /// Add validation result to buffer
    pub fn add_validation_result(&mut self, result: RealtimeValidationResult) {
        self.results.push(result);

        // Maintain buffer size
        if self.results.len() > self.capacity {
            self.results.drain(0..self.results.len() - self.capacity);
        }
    }

    /// Get recent validation results
    pub fn get_recent_results(&self, count: usize) -> Vec<&RealtimeValidationResult> {
        let start = if self.results.len() > count {
            self.results.len() - count
        } else {
            0
        };
        self.results[start..].iter().collect()
    }
}

/// Validation cache for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCache {
    /// Cached validation results
    cache: HashMap<String, CachedValidationResult>,
    /// Cache configuration
    config: ValidationCacheConfig,
}

impl ValidationCache {
    /// Create a new validation cache
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            config: ValidationCacheConfig::default(),
        }
    }

    /// Get cached validation result
    pub fn get(&self, key: &str) -> Option<&CachedValidationResult> {
        self.cache.get(key)
    }

    /// Store validation result in cache
    pub fn store(&mut self, key: String, result: ValidationResult) {
        let cached_result = CachedValidationResult {
            result,
            cached_at: Utc::now(),
            access_count: 0,
        };
        self.cache.insert(key, cached_result);

        // Maintain cache size
        if self.cache.len() > self.config.max_size {
            self.evict_entries();
        }
    }

    /// Evict cache entries based on policy
    fn evict_entries(&mut self) {
        let evict_count = self.cache.len() - self.config.max_size + 100;
        let mut entries: Vec<_> = self.cache.iter().collect();

        // Sort by access count and age
        entries.sort_by(|a, b| {
            let a_score = a.1.access_count as f64 +
                (Utc::now().signed_duration_since(a.1.cached_at).num_seconds() as f64 / 3600.0);
            let b_score = b.1.access_count as f64 +
                (Utc::now().signed_duration_since(b.1.cached_at).num_seconds() as f64 / 3600.0);
            a_score.partial_cmp(&b_score).unwrap()
        });

        for (key, _) in entries.into_iter().take(evict_count) {
            self.cache.remove(key);
        }
    }
}

/// Cached validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedValidationResult {
    /// Validation result
    pub result: ValidationResult,
    /// Cache timestamp
    pub cached_at: DateTime<Utc>,
    /// Access count
    pub access_count: u32,
}

/// Validation cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCacheConfig {
    /// Maximum cache size
    pub max_size: usize,
    /// Cache TTL
    pub ttl: Duration,
    /// Enable cache compression
    pub compression: bool,
}

impl Default for ValidationCacheConfig {
    fn default() -> Self {
        Self {
            max_size: 10000,
            ttl: Duration::from_secs(3600),
            compression: true,
        }
    }
}

/// Comprehensive validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation timestamp
    pub validation_time: DateTime<Utc>,
    /// Overall validation status
    pub overall_status: ValidationStatus,
    /// Schema validation result
    pub schema_result: Option<SchemaValidationResult>,
    /// Rule set validation results
    pub ruleset_results: HashMap<String, RuleSetValidationResult>,
    /// Quality metrics
    pub quality_metrics: Option<HashMap<String, f64>>,
    /// Real-time validation result
    pub realtime_result: Option<RealtimeValidationResult>,
    /// Validation summary
    pub summary: ValidationSummary,
}

impl ValidationResult {
    /// Create a new validation result
    pub fn new() -> Self {
        Self {
            validation_time: Utc::now(),
            overall_status: ValidationStatus::Pending,
            schema_result: None,
            ruleset_results: HashMap::new(),
            quality_metrics: None,
            realtime_result: None,
            summary: ValidationSummary::default(),
        }
    }

    /// Add schema validation result
    pub fn add_schema_result(&mut self, result: SchemaValidationResult) {
        self.schema_result = Some(result);
        self.update_overall_status();
    }

    /// Add rule set validation result
    pub fn add_ruleset_result(&mut self, ruleset_id: String, result: RuleSetValidationResult) {
        self.ruleset_results.insert(ruleset_id, result);
        self.update_overall_status();
    }

    /// Set quality metrics
    pub fn set_quality_metrics(&mut self, metrics: HashMap<String, f64>) {
        self.quality_metrics = Some(metrics);
    }

    /// Add real-time validation result
    pub fn add_realtime_result(&mut self, result: RealtimeValidationResult) {
        self.realtime_result = Some(result);
        self.update_overall_status();
    }

    /// Update overall validation status
    fn update_overall_status(&mut self) {
        let mut has_errors = false;
        let mut has_warnings = false;

        // Check schema result
        if let Some(ref schema_result) = self.schema_result {
            if !schema_result.is_valid() {
                has_errors = true;
            }
        }

        // Check rule set results
        for result in self.ruleset_results.values() {
            if result.has_critical_violations() {
                has_errors = true;
            } else if result.has_warnings() {
                has_warnings = true;
            }
        }

        // Check real-time result
        if let Some(ref realtime_result) = self.realtime_result {
            if realtime_result.has_violations() {
                has_warnings = true;
            }
        }

        self.overall_status = if has_errors {
            ValidationStatus::Failed
        } else if has_warnings {
            ValidationStatus::WarningsPresent
        } else {
            ValidationStatus::Passed
        };

        // Update summary
        self.update_summary();
    }

    /// Update validation summary
    fn update_summary(&mut self) {
        let mut total_violations = 0;
        let mut critical_violations = 0;
        let mut warning_violations = 0;

        for result in self.ruleset_results.values() {
            for rule_result in &result.rule_results {
                total_violations += rule_result.violations.len();
                for violation in &rule_result.violations {
                    match violation.severity {
                        ValidationSeverity::Critical => critical_violations += 1,
                        ValidationSeverity::High => critical_violations += 1,
                        ValidationSeverity::Medium => warning_violations += 1,
                        ValidationSeverity::Low => warning_violations += 1,
                    }
                }
            }
        }

        self.summary = ValidationSummary {
            total_violations,
            critical_violations,
            warning_violations,
            validation_duration: Utc::now().signed_duration_since(self.validation_time),
        };
    }
}

/// Validation status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Pending,
    Passed,
    WarningsPresent,
    Failed,
}

/// Schema validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidationResult {
    /// Structure validation result
    pub structure_result: StructureValidationResult,
    /// Field validation result
    pub field_result: FieldValidationResult,
    /// Constraint validation result
    pub constraint_result: ConstraintValidationResult,
}

impl SchemaValidationResult {
    /// Create a new schema validation result
    pub fn new() -> Self {
        Self {
            structure_result: StructureValidationResult { valid: true, errors: Vec::new() },
            field_result: FieldValidationResult { valid: true, field_errors: HashMap::new() },
            constraint_result: ConstraintValidationResult { valid: true, constraint_errors: Vec::new() },
        }
    }

    /// Add structure validation result
    pub fn add_structure_result(&mut self, result: StructureValidationResult) {
        self.structure_result = result;
    }

    /// Add field validation result
    pub fn add_field_result(&mut self, result: FieldValidationResult) {
        self.field_result = result;
    }

    /// Add constraint validation result
    pub fn add_constraint_result(&mut self, result: ConstraintValidationResult) {
        self.constraint_result = result;
    }

    /// Check if schema validation is valid
    pub fn is_valid(&self) -> bool {
        self.structure_result.valid &&
        self.field_result.valid &&
        self.constraint_result.valid
    }
}

/// Structure validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureValidationResult {
    /// Validation status
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
}

/// Field validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldValidationResult {
    /// Validation status
    pub valid: bool,
    /// Field-specific errors
    pub field_errors: HashMap<String, Vec<String>>,
}

/// Constraint validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintValidationResult {
    /// Validation status
    pub valid: bool,
    /// Constraint errors
    pub constraint_errors: Vec<String>,
}

/// Rule set validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleSetValidationResult {
    /// Rule set identifier
    pub rule_set_id: String,
    /// Individual rule results
    pub rule_results: Vec<RuleValidationResult>,
    /// Rule set execution time
    pub execution_time: Duration,
}

impl RuleSetValidationResult {
    /// Create a new rule set validation result
    pub fn new(rule_set_id: String) -> Self {
        Self {
            rule_set_id,
            rule_results: Vec::new(),
            execution_time: Duration::from_millis(0),
        }
    }

    /// Add rule validation result
    pub fn add_rule_result(&mut self, result: RuleValidationResult) {
        self.rule_results.push(result);
    }

    /// Check if rule set has critical violations
    pub fn has_critical_violations(&self) -> bool {
        self.rule_results.iter().any(|r| {
            r.violations.iter().any(|v| matches!(v.severity, ValidationSeverity::Critical | ValidationSeverity::High))
        })
    }

    /// Check if rule set has warnings
    pub fn has_warnings(&self) -> bool {
        self.rule_results.iter().any(|r| {
            r.violations.iter().any(|v| matches!(v.severity, ValidationSeverity::Medium | ValidationSeverity::Low))
        })
    }
}

/// Rule validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleValidationResult {
    /// Rule identifier
    pub rule_id: String,
    /// Rule name
    pub rule_name: String,
    /// Validation violations
    pub violations: Vec<ValidationViolation>,
    /// Validation timestamp
    pub validation_time: DateTime<Utc>,
    /// Number of records checked
    pub records_checked: usize,
}

/// Validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// Violation identifier
    pub violation_id: String,
    /// Record identifier
    pub record_id: String,
    /// Field name (if applicable)
    pub field_name: Option<String>,
    /// Violation type
    pub violation_type: ViolationType,
    /// Violation message
    pub message: String,
    /// Violation severity
    pub severity: ValidationSeverity,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
}

/// Violation type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Missing required field
    MissingField,
    /// Format mismatch
    FormatMismatch,
    /// Value out of range
    OutOfRange,
    /// Cross-field validation failure
    CrossFieldViolation,
    /// Custom rule violation
    CustomRuleViolation,
    /// Schema violation
    SchemaViolation,
}

/// Real-time validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeValidationResult {
    /// Validation timestamp
    pub validation_time: DateTime<Utc>,
    /// Rule results
    pub rule_results: Vec<RealtimeRuleResult>,
    /// Overall status
    pub status: RealtimeStatus,
}

impl RealtimeValidationResult {
    /// Create a new real-time validation result
    pub fn new() -> Self {
        Self {
            validation_time: Utc::now(),
            rule_results: Vec::new(),
            status: RealtimeStatus::Normal,
        }
    }

    /// Add rule result
    pub fn add_rule_result(&mut self, result: RealtimeRuleResult) {
        self.rule_results.push(result);
        self.update_status();
    }

    /// Check if has violations
    pub fn has_violations(&self) -> bool {
        self.rule_results.iter().any(|r| !r.violations.is_empty())
    }

    /// Update overall status
    fn update_status(&mut self) {
        if self.rule_results.iter().any(|r| !r.passed) {
            self.status = RealtimeStatus::Alert;
        } else {
            self.status = RealtimeStatus::Normal;
        }
    }
}

/// Real-time rule result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeRuleResult {
    /// Rule identifier
    pub rule_id: String,
    /// Rule passed status
    pub passed: bool,
    /// Rule violations
    pub violations: Vec<String>,
    /// Execution time
    pub execution_time: Duration,
}

/// Real-time status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeStatus {
    Normal,
    Warning,
    Alert,
    Critical,
}

/// Validation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total number of violations
    pub total_violations: usize,
    /// Critical violations count
    pub critical_violations: usize,
    /// Warning violations count
    pub warning_violations: usize,
    /// Validation duration
    pub validation_duration: Duration,
}

impl Default for ValidationSummary {
    fn default() -> Self {
        Self {
            total_violations: 0,
            critical_violations: 0,
            warning_violations: 0,
            validation_duration: Duration::from_millis(0),
        }
    }
}

/// Validation error types for comprehensive error handling
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Schema validation failed: {0}")]
    SchemaValidationFailed(String),

    #[error("Rule validation failed: {0}")]
    RuleValidationFailed(String),

    #[error("Quality calculation failed: {0}")]
    QualityCalculationFailed(String),

    #[error("Real-time validation failed: {0}")]
    RealtimeValidationFailed(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Type alias for validation results
pub type ValidationResultType<T> = Result<T, ValidationError>;
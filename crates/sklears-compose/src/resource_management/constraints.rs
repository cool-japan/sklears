//! Resource constraint validation and enforcement
//!
//! This module provides constraint checking capabilities for resource allocations,
//! ensuring that security, performance, and policy requirements are met.

use super::monitoring::AlertSeverity;
use super::resource_types::{
    AllocationPriority, ResourceAllocation, ResourceUsage, SecurityConstraints,
};
use crate::execution_config::ResourceConstraints;
use crate::task_definitions::TaskRequirements;
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Resource constraint checker for validating allocation requests
#[derive(Debug)]
pub struct ResourceConstraintChecker {
    /// Global resource constraints
    global_constraints: ResourceConstraints,
    /// Policy rules
    policy_rules: Vec<PolicyRule>,
    /// Constraint validation cache
    validation_cache: HashMap<String, ValidationResult>,
    /// Checker configuration
    config: ConstraintCheckerConfig,
    /// Validation statistics
    stats: ValidationStats,
}

/// Configuration for constraint checker
#[derive(Debug, Clone)]
pub struct ConstraintCheckerConfig {
    /// Enable constraint caching
    pub enable_caching: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Strict validation mode
    pub strict_mode: bool,
    /// Enable security validation
    pub enable_security_validation: bool,
    /// Enable performance validation
    pub enable_performance_validation: bool,
    /// Validation timeout
    pub validation_timeout: Duration,
}

/// Policy rule for resource allocation
#[derive(Debug, Clone)]
pub struct PolicyRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule condition
    pub condition: PolicyCondition,
    /// Rule action
    pub action: PolicyAction,
    /// Rule priority
    pub priority: u32,
    /// Rule enabled
    pub enabled: bool,
    /// Rule metadata
    pub metadata: HashMap<String, String>,
}

/// Policy condition that must be met
#[derive(Debug, Clone)]
pub enum PolicyCondition {
    /// Resource usage threshold
    ResourceThreshold {
        resource_type: String,
        threshold: f64,
        operator: ComparisonOperator,
    },
    /// User or group constraint
    UserConstraint {
        users: Vec<String>,
        groups: Vec<String>,
        operation: SetOperation,
    },
    /// Time-based constraint
    TimeConstraint {
        start_time: u32,       // seconds since midnight
        end_time: u32,         // seconds since midnight
        days_of_week: Vec<u8>, // 0-6, Sunday = 0
    },
    /// Security constraint
    SecurityConstraint {
        security_level: SecurityLevel,
        isolation_required: bool,
        encryption_required: bool,
    },
    /// Performance constraint
    PerformanceConstraint {
        min_performance: f64,
        max_latency: Duration,
        min_throughput: f64,
    },
    /// Custom constraint
    Custom {
        expression: String,
        parameters: HashMap<String, String>,
    },
    /// Logical combination of conditions
    Composite {
        operator: LogicalOperator,
        conditions: Vec<PolicyCondition>,
    },
}

/// Comparison operators for constraints
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOperator {
    /// Equal
    Equal,
    /// NotEqual
    NotEqual,
    /// GreaterThan
    GreaterThan,
    /// GreaterThanOrEqual
    GreaterThanOrEqual,
    /// LessThan
    LessThan,
    /// LessThanOrEqual
    LessThanOrEqual,
}

/// Set operations for user constraints
#[derive(Debug, Clone, PartialEq)]
pub enum SetOperation {
    /// Include
    Include,
    /// Exclude
    Exclude,
    /// Any
    Any,
    /// All
    All,
}

/// Security levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SecurityLevel {
    /// Public
    Public,
    /// Internal
    Internal,
    /// Confidential
    Confidential,
    /// Secret
    Secret,
    /// TopSecret
    TopSecret,
}

/// Logical operators for combining conditions
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOperator {
    /// And
    And,
    /// Or
    Or,
    /// Not
    Not,
}

/// Actions to take when policy conditions are met
#[derive(Debug, Clone)]
pub enum PolicyAction {
    /// Allow the allocation
    Allow,
    /// Deny the allocation
    Deny { reason: String },
    /// Modify the allocation
    Modify {
        modifications: AllocationModifications,
    },
    /// Require approval
    RequireApproval {
        approvers: Vec<String>,
        timeout: Duration,
    },
    /// Log the allocation
    Log { level: LogLevel, message: String },
    /// Alert on the allocation
    Alert {
        severity: AlertSeverity,
        message: String,
        channels: Vec<String>,
    },
    /// Rate limit the allocation
    RateLimit {
        max_allocations: u32,
        time_window: Duration,
    },
}

/// Modifications to apply to allocations
#[derive(Debug, Clone)]
pub struct AllocationModifications {
    /// Reduce CPU allocation
    pub reduce_cpu: Option<f64>,
    /// Reduce memory allocation
    pub reduce_memory: Option<f64>,
    /// Change priority
    pub new_priority: Option<AllocationPriority>,
    /// Add security constraints
    pub add_security_constraints: Option<SecurityConstraints>,
    /// Modify time limits
    pub time_limit: Option<Duration>,
}

/// Log levels for policy actions
#[derive(Debug, Clone, PartialEq)]
pub enum LogLevel {
    /// Debug
    Debug,
    /// Info
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Result of constraint validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Is allocation valid
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Applied policy actions
    pub applied_actions: Vec<PolicyAction>,
    /// Validation timestamp
    pub timestamp: SystemTime,
    /// Validation duration
    pub duration: Duration,
    /// Modified allocation (if any)
    pub modified_allocation: Option<ResourceAllocation>,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Related policy rule
    pub policy_rule_id: Option<String>,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ErrorSeverity {
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Critical
    Critical,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Related policy rule
    pub policy_rule_id: Option<String>,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStats {
    /// Total validations performed
    pub total_validations: u64,
    /// Successful validations
    pub successful_validations: u64,
    /// Failed validations
    pub failed_validations: u64,
    /// Average validation time
    pub avg_validation_time: Duration,
    /// Policy rule hit counts
    pub policy_rule_hits: HashMap<String, u64>,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Context for constraint validation
#[derive(Debug, Clone)]
pub struct ValidationContext {
    /// User requesting allocation
    pub user: Option<String>,
    /// User groups
    pub groups: Vec<String>,
    /// Current time
    pub current_time: SystemTime,
    /// Current resource usage
    pub current_usage: ResourceUsage,
    /// Available resources
    pub available_resources: ResourceUsage,
    /// System load
    pub system_load: f64,
    /// Security context
    pub security_context: SecurityContext,
}

/// Security context for validation
#[derive(Debug, Clone)]
pub struct SecurityContext {
    /// Security level
    pub level: SecurityLevel,
    /// Security clearance
    pub clearance: Option<String>,
    /// Access tokens
    pub tokens: Vec<String>,
    /// Trusted execution environment
    pub tee_available: bool,
    /// Encryption capabilities
    pub encryption_available: bool,
}

impl Default for ResourceConstraintChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceConstraintChecker {
    /// Create a new constraint checker
    #[must_use]
    pub fn new() -> Self {
        Self {
            global_constraints: ResourceConstraints::default(),
            policy_rules: Vec::new(),
            validation_cache: HashMap::new(),
            config: ConstraintCheckerConfig::default(),
            stats: ValidationStats::default(),
        }
    }

    /// Set global resource constraints
    pub fn set_global_constraints(&mut self, constraints: ResourceConstraints) {
        self.global_constraints = constraints;
    }

    /// Add a policy rule
    pub fn add_policy_rule(&mut self, rule: PolicyRule) {
        self.policy_rules.push(rule);
        // Sort by priority (higher priority first)
        self.policy_rules
            .sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Remove a policy rule
    pub fn remove_policy_rule(&mut self, rule_id: &str) -> bool {
        if let Some(pos) = self.policy_rules.iter().position(|r| r.id == rule_id) {
            self.policy_rules.remove(pos);
            true
        } else {
            false
        }
    }

    /// Validate a resource allocation request
    pub fn validate_allocation(
        &mut self,
        requirements: &TaskRequirements,
        context: &ValidationContext,
    ) -> SklResult<ValidationResult> {
        let start_time = SystemTime::now();

        // Check cache first
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(requirements, context);
            if let Some(cached_result) = self.validation_cache.get(&cache_key) {
                if start_time
                    .duration_since(cached_result.timestamp)
                    .unwrap_or(Duration::MAX)
                    < self.config.cache_ttl
                {
                    return Ok(cached_result.clone());
                }
            }
        }

        let mut result = ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            applied_actions: Vec::new(),
            timestamp: start_time,
            duration: Duration::from_secs(0),
            modified_allocation: None,
        };

        // Validate global constraints
        self.validate_global_constraints(requirements, context, &mut result)?;

        // Apply policy rules
        self.apply_policy_rules(requirements, context, &mut result)?;

        // Security validation
        if self.config.enable_security_validation {
            self.validate_security_constraints(requirements, context, &mut result)?;
        }

        // Performance validation
        if self.config.enable_performance_validation {
            self.validate_performance_constraints(requirements, context, &mut result)?;
        }

        // Update duration
        result.duration = SystemTime::now()
            .duration_since(start_time)
            .unwrap_or(Duration::from_secs(0));

        // Update statistics
        self.update_stats(&result);

        // Cache result
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(requirements, context);
            self.validation_cache.insert(cache_key, result.clone());
        }

        Ok(result)
    }

    /// Validate global resource constraints
    fn validate_global_constraints(
        &self,
        requirements: &TaskRequirements,
        context: &ValidationContext,
        result: &mut ValidationResult,
    ) -> SklResult<()> {
        // Check CPU constraints
        if let Some(cpu_cores) = requirements.cpu_cores {
            if let Some(max_cpu) = self.global_constraints.max_cpu_cores {
                if cpu_cores > max_cpu {
                    result.valid = false;
                    result.errors.push(ValidationError {
                        code: "CPU_LIMIT_EXCEEDED".to_string(),
                        message: format!(
                            "Requested {cpu_cores} CPU cores exceeds limit of {max_cpu}"
                        ),
                        policy_rule_id: None,
                        severity: ErrorSeverity::High,
                        context: HashMap::new(),
                    });
                }
            }
        }

        // Check memory constraints
        if let Some(memory) = requirements.memory {
            if let Some(max_memory) = self.global_constraints.max_memory {
                if memory > max_memory {
                    result.valid = false;
                    result.errors.push(ValidationError {
                        code: "MEMORY_LIMIT_EXCEEDED".to_string(),
                        message: format!(
                            "Requested {memory} bytes memory exceeds limit of {max_memory}"
                        ),
                        policy_rule_id: None,
                        severity: ErrorSeverity::High,
                        context: HashMap::new(),
                    });
                }
            }
        }

        // Check GPU constraints
        if !requirements.gpu_devices.is_empty() {
            if let Some(max_gpus) = self.global_constraints.max_gpu_devices {
                if requirements.gpu_devices.len() > max_gpus {
                    result.valid = false;
                    result.errors.push(ValidationError {
                        code: "GPU_LIMIT_EXCEEDED".to_string(),
                        message: format!(
                            "Requested {} GPU devices exceeds limit of {}",
                            requirements.gpu_devices.len(),
                            max_gpus
                        ),
                        policy_rule_id: None,
                        severity: ErrorSeverity::High,
                        context: HashMap::new(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Apply policy rules to the allocation request
    fn apply_policy_rules(
        &mut self,
        requirements: &TaskRequirements,
        context: &ValidationContext,
        result: &mut ValidationResult,
    ) -> SklResult<()> {
        for rule in &self.policy_rules {
            if !rule.enabled {
                continue;
            }

            // Evaluate rule condition
            if self.evaluate_condition(&rule.condition, requirements, context)? {
                // Update hit count
                *self
                    .stats
                    .policy_rule_hits
                    .entry(rule.id.clone())
                    .or_insert(0) += 1;

                // Apply rule action
                match &rule.action {
                    PolicyAction::Allow => {
                        // No action needed, allocation is allowed
                    }
                    PolicyAction::Deny { reason } => {
                        result.valid = false;
                        result.errors.push(ValidationError {
                            code: "POLICY_DENIED".to_string(),
                            message: reason.clone(),
                            policy_rule_id: Some(rule.id.clone()),
                            severity: ErrorSeverity::High,
                            context: HashMap::new(),
                        });
                    }
                    PolicyAction::Modify { modifications } => {
                        // Apply modifications - this would modify the allocation
                        result.warnings.push(ValidationWarning {
                            code: "ALLOCATION_MODIFIED".to_string(),
                            message: "Allocation was modified by policy rule".to_string(),
                            policy_rule_id: Some(rule.id.clone()),
                            context: HashMap::new(),
                        });
                    }
                    PolicyAction::RequireApproval {
                        approvers: _,
                        timeout: _,
                    } => {
                        result.warnings.push(ValidationWarning {
                            code: "APPROVAL_REQUIRED".to_string(),
                            message: "Allocation requires approval".to_string(),
                            policy_rule_id: Some(rule.id.clone()),
                            context: HashMap::new(),
                        });
                    }
                    PolicyAction::Log { level: _, message } => {
                        // Log the allocation (would integrate with logging system)
                        result.warnings.push(ValidationWarning {
                            code: "LOGGED_ALLOCATION".to_string(),
                            message: message.clone(),
                            policy_rule_id: Some(rule.id.clone()),
                            context: HashMap::new(),
                        });
                    }
                    PolicyAction::Alert {
                        severity: _,
                        message,
                        channels: _,
                    } => {
                        result.warnings.push(ValidationWarning {
                            code: "ALERT_TRIGGERED".to_string(),
                            message: message.clone(),
                            policy_rule_id: Some(rule.id.clone()),
                            context: HashMap::new(),
                        });
                    }
                    PolicyAction::RateLimit {
                        max_allocations: _,
                        time_window: _,
                    } => {
                        // Would implement rate limiting logic
                        result.warnings.push(ValidationWarning {
                            code: "RATE_LIMITED".to_string(),
                            message: "Allocation is subject to rate limiting".to_string(),
                            policy_rule_id: Some(rule.id.clone()),
                            context: HashMap::new(),
                        });
                    }
                }

                result.applied_actions.push(rule.action.clone());
            }
        }

        Ok(())
    }

    /// Evaluate a policy condition
    fn evaluate_condition(
        &self,
        condition: &PolicyCondition,
        requirements: &TaskRequirements,
        context: &ValidationContext,
    ) -> SklResult<bool> {
        match condition {
            PolicyCondition::ResourceThreshold {
                resource_type,
                threshold,
                operator,
            } => {
                let current_usage = match resource_type.as_str() {
                    "cpu" => context.current_usage.cpu_percent,
                    "memory" => {
                        (context.current_usage.memory_usage.used as f64
                            / context.current_usage.memory_usage.total as f64)
                            * 100.0
                    }
                    _ => return Ok(false),
                };

                Ok(self.compare_values(current_usage, *threshold, operator))
            }
            PolicyCondition::UserConstraint {
                users,
                groups,
                operation,
            } => {
                if let Some(user) = &context.user {
                    let user_match = users.contains(user);
                    let group_match = groups.iter().any(|g| context.groups.contains(g));

                    Ok(match operation {
                        SetOperation::Include => user_match || group_match,
                        SetOperation::Exclude => !user_match && !group_match,
                        SetOperation::Any => user_match || group_match,
                        SetOperation::All => {
                            user_match && groups.iter().all(|g| context.groups.contains(g))
                        }
                    })
                } else {
                    Ok(false)
                }
            }
            PolicyCondition::TimeConstraint {
                start_time,
                end_time,
                days_of_week,
            } => {
                // Simplified time checking - would need proper timezone handling
                let now = context
                    .current_time
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or(Duration::from_secs(0));
                let seconds_today = (now.as_secs() % (24 * 3600)) as u32;
                let day_of_week = ((now.as_secs() / (24 * 3600)) % 7) as u8; // Approximate

                let time_match = if start_time <= end_time {
                    seconds_today >= *start_time && seconds_today <= *end_time
                } else {
                    seconds_today >= *start_time || seconds_today <= *end_time
                };

                let day_match = days_of_week.is_empty() || days_of_week.contains(&day_of_week);

                Ok(time_match && day_match)
            }
            PolicyCondition::SecurityConstraint {
                security_level,
                isolation_required,
                encryption_required,
            } => {
                let level_match = context.security_context.level >= *security_level;
                let isolation_match = !isolation_required || context.security_context.tee_available;
                let encryption_match =
                    !encryption_required || context.security_context.encryption_available;

                Ok(level_match && isolation_match && encryption_match)
            }
            PolicyCondition::PerformanceConstraint {
                min_performance: _,
                max_latency: _,
                min_throughput: _,
            } => {
                // Would need performance metrics to evaluate
                Ok(true) // Simplified for now
            }
            PolicyCondition::Custom {
                expression: _,
                parameters: _,
            } => {
                // Would need a rule engine to evaluate custom expressions
                Ok(false) // Simplified for now
            }
            PolicyCondition::Composite {
                operator,
                conditions,
            } => match operator {
                LogicalOperator::And => {
                    for cond in conditions {
                        if !self.evaluate_condition(cond, requirements, context)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                LogicalOperator::Or => {
                    for cond in conditions {
                        if self.evaluate_condition(cond, requirements, context)? {
                            return Ok(true);
                        }
                    }
                    Ok(false)
                }
                LogicalOperator::Not => {
                    if conditions.len() != 1 {
                        return Err(SklearsError::ResourceAllocationError(
                            "NOT operator requires exactly one condition".to_string(),
                        ));
                    }
                    Ok(!self.evaluate_condition(&conditions[0], requirements, context)?)
                }
            },
        }
    }

    /// Compare two values using the specified operator
    fn compare_values(&self, left: f64, right: f64, operator: &ComparisonOperator) -> bool {
        match operator {
            ComparisonOperator::Equal => (left - right).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (left - right).abs() >= f64::EPSILON,
            ComparisonOperator::GreaterThan => left > right,
            ComparisonOperator::GreaterThanOrEqual => left >= right,
            ComparisonOperator::LessThan => left < right,
            ComparisonOperator::LessThanOrEqual => left <= right,
        }
    }

    /// Validate security constraints
    fn validate_security_constraints(
        &self,
        requirements: &TaskRequirements,
        context: &ValidationContext,
        result: &mut ValidationResult,
    ) -> SklResult<()> {
        // Check if security requirements can be met
        // TODO: Add security_constraints field to TaskRequirements
        // TODO: Implement security validation when security_constraints field is added to TaskRequirements
        /*
        if let Some(security_constraints) = &requirements.security_constraints {
            if security_constraints.isolation_required && !context.security_context.tee_available {
                result.errors.push(ValidationError {
                    code: "ISOLATION_NOT_AVAILABLE".to_string(),
                    message: "Isolation required but TEE not available".to_string(),
                    policy_rule_id: None,
                    severity: ErrorSeverity::High,
                    context: HashMap::new(),
                });
                result.valid = false;
            }
        }
        */

        Ok(())
    }

    /// Validate performance constraints
    fn validate_performance_constraints(
        &self,
        _requirements: &TaskRequirements,
        context: &ValidationContext,
        result: &mut ValidationResult,
    ) -> SklResult<()> {
        // Check if system can meet performance requirements
        if context.system_load > 0.9 {
            result.warnings.push(ValidationWarning {
                code: "HIGH_SYSTEM_LOAD".to_string(),
                message: "System load is high, performance may be degraded".to_string(),
                policy_rule_id: None,
                context: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Generate cache key for validation result
    fn generate_cache_key(
        &self,
        requirements: &TaskRequirements,
        context: &ValidationContext,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        requirements.cpu_cores.hash(&mut hasher);
        requirements.memory.hash(&mut hasher);
        requirements.gpu_devices.hash(&mut hasher);
        context.user.hash(&mut hasher);
        context.groups.hash(&mut hasher);

        format!("validation_{}", hasher.finish())
    }

    /// Update validation statistics
    fn update_stats(&mut self, result: &ValidationResult) {
        self.stats.total_validations += 1;
        if result.valid {
            self.stats.successful_validations += 1;
        } else {
            self.stats.failed_validations += 1;
        }

        // Update average validation time
        let total_time = self.stats.avg_validation_time.as_nanos() as f64
            * (self.stats.total_validations - 1) as f64
            + result.duration.as_nanos() as f64;
        self.stats.avg_validation_time =
            Duration::from_nanos((total_time / self.stats.total_validations as f64) as u64);
    }

    /// Get validation statistics
    #[must_use]
    pub fn get_stats(&self) -> &ValidationStats {
        &self.stats
    }
}

impl Default for ConstraintCheckerConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl: Duration::from_secs(300), // 5 minutes
            strict_mode: false,
            enable_security_validation: true,
            enable_performance_validation: true,
            validation_timeout: Duration::from_secs(10),
        }
    }
}

impl Default for ValidationStats {
    fn default() -> Self {
        Self {
            total_validations: 0,
            successful_validations: 0,
            failed_validations: 0,
            avg_validation_time: Duration::from_millis(0),
            policy_rule_hits: HashMap::new(),
            cache_hit_rate: 0.0,
        }
    }
}

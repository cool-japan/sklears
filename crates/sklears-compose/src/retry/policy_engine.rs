//! Policy Engine for Rule-Based Retry Management
//!
//! This module provides sophisticated policy-based retry management including
//! rule engines, policy evaluation, optimization strategies, and adaptive
//! policy adjustment based on performance feedback and machine learning.

use super::core::*;
use super::machine_learning::*;
use sklears_core::error::Result as SklResult;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
    time::{Duration, SystemTime},
};

/// Retry policy engine
#[derive(Debug)]
pub struct RetryPolicyEngine {
    /// Policy rules
    policies: Arc<RwLock<HashMap<String, RetryPolicy>>>,
    /// Rule engine
    rule_engine: Arc<RuleEngine>,
    /// Policy evaluator
    evaluator: Arc<PolicyEvaluator>,
    /// Policy optimizer
    optimizer: Arc<PolicyOptimizer>,
}

impl RetryPolicyEngine {
    /// Create new retry policy engine
    pub fn new() -> Self {
        Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
            rule_engine: Arc::new(RuleEngine::new()),
            evaluator: Arc::new(PolicyEvaluator::new()),
            optimizer: Arc::new(PolicyOptimizer::new()),
        }
    }

    /// Register a retry policy
    pub fn register_policy(&self, policy: RetryPolicy) -> SklResult<()> {
        let mut policies = self.policies.write().unwrap();
        policies.insert(policy.id.clone(), policy);
        Ok(())
    }

    /// Evaluate policies for a given context
    pub fn evaluate_policies(&self, context: &RetryContext) -> SklResult<Vec<PolicyEvaluationResult>> {
        let policies = self.policies.read().unwrap();
        let mut results = Vec::new();

        for policy in policies.values() {
            if let Ok(result) = self.evaluator.evaluate_policy(policy, context) {
                results.push(result);
            }
        }

        // Sort by priority and confidence
        results.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Get applicable policies for context
    pub fn get_applicable_policies(&self, context: &RetryContext) -> SklResult<Vec<RetryPolicy>> {
        let evaluation_results = self.evaluate_policies(context)?;
        let applicable_policies = evaluation_results
            .iter()
            .filter(|result| result.applies && result.confidence > 0.5)
            .map(|result| {
                let policies = self.policies.read().unwrap();
                policies.values()
                    .find(|p| p.id == result.policy_id)
                    .cloned()
            })
            .filter_map(|policy| policy)
            .collect();

        Ok(applicable_policies)
    }

    /// Optimize policies based on performance
    pub fn optimize_policies(&self, performance_data: &[PerformanceDataPoint]) -> SklResult<()> {
        let mut optimization_results = Vec::new();

        {
            let policies = self.policies.read().unwrap();
            for policy in policies.values() {
                if let Ok(optimization_result) = self.optimizer.optimize_policy(policy, performance_data) {
                    optimization_results.push((policy.id.clone(), optimization_result.policy));
                }
            }
        }

        // Update policies with optimized parameters
        if !optimization_results.is_empty() {
            let mut policies_write = self.policies.write().unwrap();
            for (policy_id, optimized_policy) in optimization_results {
                policies_write.insert(policy_id, optimized_policy);
            }
        }

        Ok(())
    }

    /// Get policy statistics
    pub fn get_statistics(&self) -> PolicyEngineStatistics {
        let policies = self.policies.read().unwrap();
        PolicyEngineStatistics {
            total_policies: policies.len(),
            active_policies: policies.values().filter(|p| p.enabled).count(),
            avg_rules_per_policy: if !policies.is_empty() {
                policies.values().map(|p| p.rules.len()).sum::<usize>() as f64 / policies.len() as f64
            } else {
                0.0
            },
        }
    }
}

/// Policy engine statistics
#[derive(Debug, Clone)]
pub struct PolicyEngineStatistics {
    /// Total number of policies
    pub total_policies: usize,
    /// Number of active policies
    pub active_policies: usize,
    /// Average rules per policy
    pub avg_rules_per_policy: f64,
}

/// Retry policy
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Policy identifier
    pub id: String,
    /// Policy name
    pub name: String,
    /// Policy rules
    pub rules: Vec<PolicyRule>,
    /// Policy conditions
    pub conditions: Vec<PolicyCondition>,
    /// Policy actions
    pub actions: Vec<PolicyAction>,
    /// Policy priority
    pub priority: i32,
    /// Policy enabled
    pub enabled: bool,
    /// Policy metadata
    pub metadata: HashMap<String, String>,
}

/// Policy rule
#[derive(Debug, Clone)]
pub struct PolicyRule {
    /// Rule identifier
    pub id: String,
    /// Rule condition
    pub condition: String,
    /// Rule action
    pub action: String,
    /// Rule priority
    pub priority: i32,
    /// Rule enabled
    pub enabled: bool,
    /// Rule metadata
    pub metadata: HashMap<String, String>,
}

/// Policy condition
#[derive(Debug, Clone)]
pub struct PolicyCondition {
    /// Condition identifier
    pub id: String,
    /// Condition expression
    pub expression: String,
    /// Condition parameters
    pub parameters: HashMap<String, String>,
    /// Condition enabled
    pub enabled: bool,
}

/// Policy action
#[derive(Debug, Clone)]
pub struct PolicyAction {
    /// Action identifier
    pub id: String,
    /// Action type
    pub action_type: ActionType,
    /// Action parameters
    pub parameters: HashMap<String, String>,
    /// Action priority
    pub priority: Priority,
    /// Action enabled
    pub enabled: bool,
}

/// Rule engine for evaluating policy conditions
#[derive(Debug)]
pub struct RuleEngine {
    /// Rule evaluators
    evaluators: HashMap<String, Box<dyn RuleEvaluator + Send + Sync>>,
    /// Engine configuration
    config: RuleEngineConfig,
}

/// Rule engine configuration
#[derive(Debug, Clone)]
pub struct RuleEngineConfig {
    /// Engine enabled
    pub enabled: bool,
    /// Evaluation timeout
    pub timeout: Duration,
    /// Maximum rules per evaluation
    pub max_rules: usize,
    /// Enable caching
    pub caching: bool,
}

impl RuleEngine {
    /// Create new rule engine
    pub fn new() -> Self {
        let mut engine = Self {
            evaluators: HashMap::new(),
            config: RuleEngineConfig {
                enabled: true,
                timeout: Duration::from_secs(5),
                max_rules: 100,
                caching: true,
            },
        };

        // Register default evaluators
        engine.register_evaluator("success_rate", Box::new(SuccessRateEvaluator));
        engine.register_evaluator("error_type", Box::new(ErrorTypeEvaluator));
        engine.register_evaluator("duration", Box::new(DurationEvaluator));
        engine.register_evaluator("attempt_count", Box::new(AttemptCountEvaluator));

        engine
    }

    /// Register rule evaluator
    pub fn register_evaluator(&mut self, name: &str, evaluator: Box<dyn RuleEvaluator + Send + Sync>) {
        self.evaluators.insert(name.to_string(), evaluator);
    }

    /// Evaluate rule condition
    pub fn evaluate_rule(&self, rule: &PolicyRule, context: &RetryContext) -> SklResult<bool> {
        if !rule.enabled {
            return Ok(false);
        }

        // Parse rule condition to determine evaluator
        let evaluator_name = self.extract_evaluator_name(&rule.condition);

        if let Some(evaluator) = self.evaluators.get(&evaluator_name) {
            evaluator.evaluate(rule, context)
        } else {
            // Default evaluation based on simple conditions
            Ok(self.evaluate_simple_condition(&rule.condition, context))
        }
    }

    /// Extract evaluator name from condition
    fn extract_evaluator_name(&self, condition: &str) -> String {
        // Simple parsing - in practice would use proper expression parser
        if condition.contains("success_rate") {
            "success_rate".to_string()
        } else if condition.contains("error_type") {
            "error_type".to_string()
        } else if condition.contains("duration") {
            "duration".to_string()
        } else if condition.contains("attempt") {
            "attempt_count".to_string()
        } else {
            "default".to_string()
        }
    }

    /// Simple condition evaluation fallback
    fn evaluate_simple_condition(&self, condition: &str, context: &RetryContext) -> bool {
        // Simplified condition evaluation
        if condition.contains("always_retry") {
            true
        } else if condition.contains("never_retry") {
            false
        } else if condition.contains("max_attempts_reached") {
            context.current_attempt >= 5 // Default threshold
        } else {
            false
        }
    }
}

/// Rule evaluator trait
pub trait RuleEvaluator: Send + Sync {
    /// Evaluate rule against context
    fn evaluate(&self, rule: &PolicyRule, context: &RetryContext) -> SklResult<bool>;

    /// Get evaluator name
    fn name(&self) -> &str;
}

/// Success rate evaluator
pub struct SuccessRateEvaluator;

impl RuleEvaluator for SuccessRateEvaluator {
    fn evaluate(&self, rule: &PolicyRule, context: &RetryContext) -> SklResult<bool> {
        if context.attempts.is_empty() {
            return Ok(false);
        }

        let success_count = context.attempts.iter()
            .filter(|a| a.result == AttemptResult::Success)
            .count();
        let success_rate = success_count as f64 / context.attempts.len() as f64;

        // Parse threshold from rule condition
        let threshold = self.extract_threshold(&rule.condition).unwrap_or(0.5);

        Ok(if rule.condition.contains("less_than") {
            success_rate < threshold
        } else if rule.condition.contains("greater_than") {
            success_rate > threshold
        } else {
            success_rate >= threshold
        })
    }

    fn name(&self) -> &str {
        "success_rate"
    }
}

impl SuccessRateEvaluator {
    fn extract_threshold(&self, condition: &str) -> Option<f64> {
        // Simple threshold extraction - would use proper parsing in practice
        condition.split_whitespace()
            .find_map(|token| token.parse::<f64>().ok())
    }
}

/// Error type evaluator
pub struct ErrorTypeEvaluator;

impl RuleEvaluator for ErrorTypeEvaluator {
    fn evaluate(&self, rule: &PolicyRule, context: &RetryContext) -> SklResult<bool> {
        let target_error_type = self.extract_error_type(&rule.condition);

        for error in &context.errors {
            let error_type = match error {
                RetryError::Network { .. } => "network",
                RetryError::Service { .. } => "service",
                RetryError::Timeout { .. } => "timeout",
                RetryError::ResourceExhaustion { .. } => "resource",
                RetryError::Auth { .. } => "auth",
                RetryError::Configuration { .. } => "config",
                RetryError::RateLimit { .. } => "rate_limit",
                RetryError::CircuitOpen { .. } => "circuit_open",
                RetryError::Custom { .. } => "custom",
            };

            if error_type == target_error_type {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn name(&self) -> &str {
        "error_type"
    }
}

impl ErrorTypeEvaluator {
    fn extract_error_type(&self, condition: &str) -> &str {
        if condition.contains("network") {
            "network"
        } else if condition.contains("service") {
            "service"
        } else if condition.contains("timeout") {
            "timeout"
        } else if condition.contains("resource") {
            "resource"
        } else if condition.contains("auth") {
            "auth"
        } else {
            "unknown"
        }
    }
}

/// Duration evaluator
pub struct DurationEvaluator;

impl RuleEvaluator for DurationEvaluator {
    fn evaluate(&self, rule: &PolicyRule, context: &RetryContext) -> SklResult<bool> {
        if context.attempts.is_empty() {
            return Ok(false);
        }

        let avg_duration = context.attempts.iter()
            .map(|a| a.duration)
            .sum::<Duration>() / context.attempts.len() as u32;

        let threshold_ms = self.extract_duration_threshold(&rule.condition).unwrap_or(1000);
        let threshold = Duration::from_millis(threshold_ms);

        Ok(if rule.condition.contains("greater_than") {
            avg_duration > threshold
        } else {
            avg_duration < threshold
        })
    }

    fn name(&self) -> &str {
        "duration"
    }
}

impl DurationEvaluator {
    fn extract_duration_threshold(&self, condition: &str) -> Option<u64> {
        condition.split_whitespace()
            .find_map(|token| token.parse::<u64>().ok())
    }
}

/// Attempt count evaluator
pub struct AttemptCountEvaluator;

impl RuleEvaluator for AttemptCountEvaluator {
    fn evaluate(&self, rule: &PolicyRule, context: &RetryContext) -> SklResult<bool> {
        let threshold = self.extract_count_threshold(&rule.condition).unwrap_or(3);

        Ok(if rule.condition.contains("greater_than") {
            context.current_attempt > threshold
        } else if rule.condition.contains("equals") {
            context.current_attempt == threshold
        } else {
            context.current_attempt < threshold
        })
    }

    fn name(&self) -> &str {
        "attempt_count"
    }
}

impl AttemptCountEvaluator {
    fn extract_count_threshold(&self, condition: &str) -> Option<u32> {
        condition.split_whitespace()
            .find_map(|token| token.parse::<u32>().ok())
    }
}

/// Policy evaluator
#[derive(Debug)]
pub struct PolicyEvaluator {
    /// Evaluation strategies
    strategies: HashMap<String, Box<dyn EvaluationStrategy + Send + Sync>>,
    /// Evaluation cache
    cache: Arc<Mutex<EvaluationCache>>,
    /// Evaluator configuration
    config: EvaluatorConfig,
}

impl PolicyEvaluator {
    /// Create new policy evaluator
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            cache: Arc::new(Mutex::new(EvaluationCache::default())),
            config: EvaluatorConfig {
                enabled: true,
                timeout: Duration::from_secs(10),
                cache: CacheConfiguration {
                    ttl: Duration::from_secs(300),
                    max_size: 1000,
                    enabled: true,
                },
                parallel: true,
            },
        }
    }

    /// Evaluate policy against context
    pub fn evaluate_policy(&self, policy: &RetryPolicy, context: &RetryContext) -> SklResult<PolicyEvaluationResult> {
        if !policy.enabled {
            return Ok(PolicyEvaluationResult {
                policy_id: policy.id.clone(),
                applies: false,
                confidence: 0.0,
                actions: Vec::new(),
                metadata: HashMap::new(),
            });
        }

        // Check cache
        let cache_key = format!("{}_{}", policy.id, context.id);
        if self.config.cache.enabled {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.evaluations.get(&cache_key) {
                if SystemTime::now().duration_since(cached.cached_at).unwrap_or(Duration::MAX) < cached.ttl {
                    return Ok(cached.result.clone());
                }
            }
        }

        // Evaluate all conditions
        let mut condition_results = Vec::new();
        for condition in &policy.conditions {
            if condition.enabled {
                let result = self.evaluate_condition(condition, context)?;
                condition_results.push(result);
            }
        }

        // Policy applies if all conditions are met
        let applies = condition_results.iter().all(|&result| result);
        let confidence = if condition_results.is_empty() {
            0.5
        } else {
            condition_results.iter().map(|&r| if r { 1.0 } else { 0.0 }).sum::<f64>() / condition_results.len() as f64
        };

        // Generate recommended actions
        let mut actions = Vec::new();
        if applies {
            for policy_action in &policy.actions {
                if policy_action.enabled {
                    actions.push(RecommendedAction {
                        action_type: policy_action.action_type.clone(),
                        parameters: policy_action.parameters.clone(),
                        priority: policy_action.priority.clone(),
                        expected_impact: 0.5, // Simplified
                    });
                }
            }
        }

        let result = PolicyEvaluationResult {
            policy_id: policy.id.clone(),
            applies,
            confidence,
            actions,
            metadata: policy.metadata.clone(),
        };

        // Update cache
        if self.config.cache.enabled {
            let mut cache = self.cache.lock().unwrap();
            cache.evaluations.insert(cache_key.clone(), CachedEvaluation {
                result: result.clone(),
                cached_at: SystemTime::now(),
                ttl: self.config.cache.ttl,
            });
            cache.timestamps.insert(cache_key, SystemTime::now());
        }

        Ok(result)
    }

    /// Evaluate individual condition
    fn evaluate_condition(&self, condition: &PolicyCondition, context: &RetryContext) -> SklResult<bool> {
        // Simple condition evaluation - would be more sophisticated in practice
        if condition.expression.contains("success_rate_low") {
            let success_rate = if !context.attempts.is_empty() {
                let success_count = context.attempts.iter()
                    .filter(|a| a.result == AttemptResult::Success)
                    .count();
                success_count as f64 / context.attempts.len() as f64
            } else {
                0.0
            };
            Ok(success_rate < 0.3)
        } else if condition.expression.contains("max_attempts_reached") {
            Ok(context.current_attempt >= 5)
        } else if condition.expression.contains("timeout_errors") {
            Ok(context.errors.iter().any(|e| matches!(e, RetryError::Timeout { .. })))
        } else {
            Ok(false)
        }
    }
}

/// Evaluation strategy trait
pub trait EvaluationStrategy: Send + Sync {
    /// Evaluate policy
    fn evaluate(&self, policy: &RetryPolicy, context: &RetryContext) -> PolicyEvaluationResult;

    /// Get strategy name
    fn name(&self) -> &str;
}

/// Policy evaluation result
#[derive(Debug, Clone)]
pub struct PolicyEvaluationResult {
    /// Policy identifier
    pub policy_id: String,
    /// Policy applies
    pub applies: bool,
    /// Evaluation confidence
    pub confidence: f64,
    /// Recommended actions
    pub actions: Vec<RecommendedAction>,
    /// Evaluation metadata
    pub metadata: HashMap<String, String>,
}

/// Recommended action
#[derive(Debug, Clone)]
pub struct RecommendedAction {
    /// Action type
    pub action_type: ActionType,
    /// Action parameters
    pub parameters: HashMap<String, String>,
    /// Action priority
    pub priority: Priority,
    /// Expected impact
    pub expected_impact: f64,
}

/// Evaluation cache
#[derive(Debug, Default)]
pub struct EvaluationCache {
    /// Cached evaluations
    pub evaluations: HashMap<String, CachedEvaluation>,
    /// Cache timestamps
    pub timestamps: HashMap<String, SystemTime>,
    /// Cache statistics
    pub statistics: CacheStatistics,
}

/// Cached evaluation
#[derive(Debug, Clone)]
pub struct CachedEvaluation {
    /// Evaluation result
    pub result: PolicyEvaluationResult,
    /// Cache timestamp
    pub cached_at: SystemTime,
    /// Cache TTL
    pub ttl: Duration,
}

/// Evaluator configuration
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Enable evaluator
    pub enabled: bool,
    /// Evaluation timeout
    pub timeout: Duration,
    /// Cache configuration
    pub cache: CacheConfiguration,
    /// Parallel evaluation
    pub parallel: bool,
}

/// Policy optimizer for adaptive policy improvement
#[derive(Debug)]
pub struct PolicyOptimizer {
    /// Optimization strategies
    strategies: HashMap<String, Box<dyn PolicyOptimizationStrategy + Send + Sync>>,
    /// Optimization history
    history: Arc<Mutex<Vec<PolicyOptimizationResult>>>,
    /// Optimizer configuration
    config: PolicyOptimizerConfig,
}

impl PolicyOptimizer {
    /// Create new policy optimizer
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            history: Arc::new(Mutex::new(Vec::new())),
            config: PolicyOptimizerConfig {
                enabled: true,
                interval: Duration::from_secs(3600),
                min_data_points: 100,
                objectives: Vec::new(),
            },
        }
    }

    /// Optimize policy based on performance data
    pub fn optimize_policy(
        &self,
        policy: &RetryPolicy,
        performance_data: &[PerformanceDataPoint]
    ) -> SklResult<PolicyOptimizationResult> {
        if performance_data.len() < self.config.min_data_points {
            return Ok(PolicyOptimizationResult {
                policy: policy.clone(),
                improvement: 0.0,
                confidence: 0.0,
                timestamp: SystemTime::now(),
                metadata: HashMap::new(),
            });
        }

        // Simple optimization - adjust priority based on success rate
        let avg_success_rate: f64 = performance_data.iter()
            .map(|pd| pd.success_rate)
            .sum::<f64>() / performance_data.len() as f64;

        let mut optimized_policy = policy.clone();
        let improvement = if avg_success_rate < 0.3 {
            // Poor performance - lower priority
            optimized_policy.priority = (policy.priority - 1).max(1);
            0.1
        } else if avg_success_rate > 0.8 {
            // Good performance - higher priority
            optimized_policy.priority = policy.priority + 1;
            0.2
        } else {
            // No change needed
            0.0
        };

        let result = PolicyOptimizationResult {
            policy: optimized_policy,
            improvement,
            confidence: if performance_data.len() >= 100 { 0.8 } else { 0.4 },
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        };

        // Store in history
        {
            let mut history = self.history.lock().unwrap();
            history.push(result.clone());
            if history.len() > 1000 {
                history.remove(0);
            }
        }

        Ok(result)
    }
}

/// Policy optimization strategy trait
pub trait PolicyOptimizationStrategy: Send + Sync {
    /// Optimize policy
    fn optimize(&self, policy: &RetryPolicy, performance_data: &[PerformanceDataPoint]) -> PolicyOptimizationResult;

    /// Get strategy name
    fn name(&self) -> &str;
}

/// Policy optimization result
#[derive(Debug, Clone)]
pub struct PolicyOptimizationResult {
    /// Optimized policy
    pub policy: RetryPolicy,
    /// Expected improvement
    pub improvement: f64,
    /// Optimization confidence
    pub confidence: f64,
    /// Optimization timestamp
    pub timestamp: SystemTime,
    /// Optimization metadata
    pub metadata: HashMap<String, String>,
}

/// Policy optimizer configuration
#[derive(Debug, Clone)]
pub struct PolicyOptimizerConfig {
    /// Enable optimization
    pub enabled: bool,
    /// Optimization interval
    pub interval: Duration,
    /// Minimum performance data
    pub min_data_points: usize,
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
}

/// Default policy factory
pub struct DefaultPolicyFactory;

impl DefaultPolicyFactory {
    /// Create default retry policies
    pub fn create_default_policies() -> Vec<RetryPolicy> {
        vec![
            Self::create_network_error_policy(),
            Self::create_timeout_policy(),
            Self::create_rate_limit_policy(),
            Self::create_circuit_breaker_policy(),
        ]
    }

    /// Network error policy
    fn create_network_error_policy() -> RetryPolicy {
        RetryPolicy {
            id: "network_error_policy".to_string(),
            name: "Network Error Retry Policy".to_string(),
            rules: vec![
                PolicyRule {
                    id: "network_error_rule".to_string(),
                    condition: "error_type contains network".to_string(),
                    action: "retry_with_exponential_backoff".to_string(),
                    priority: 10,
                    enabled: true,
                    metadata: HashMap::new(),
                }
            ],
            conditions: vec![
                PolicyCondition {
                    id: "network_error_condition".to_string(),
                    expression: "error_type == network".to_string(),
                    parameters: HashMap::new(),
                    enabled: true,
                }
            ],
            actions: vec![
                PolicyAction {
                    id: "retry_action".to_string(),
                    action_type: ActionType::Retry,
                    parameters: HashMap::from([
                        ("strategy".to_string(), "exponential".to_string()),
                        ("max_attempts".to_string(), "5".to_string()),
                    ]),
                    priority: Priority::Medium,
                    enabled: true,
                }
            ],
            priority: 10,
            enabled: true,
            metadata: HashMap::new(),
        }
    }

    /// Timeout policy
    fn create_timeout_policy() -> RetryPolicy {
        RetryPolicy {
            id: "timeout_policy".to_string(),
            name: "Timeout Retry Policy".to_string(),
            rules: vec![
                PolicyRule {
                    id: "timeout_rule".to_string(),
                    condition: "error_type contains timeout".to_string(),
                    action: "retry_with_linear_backoff".to_string(),
                    priority: 8,
                    enabled: true,
                    metadata: HashMap::new(),
                }
            ],
            conditions: vec![
                PolicyCondition {
                    id: "timeout_condition".to_string(),
                    expression: "timeout_errors".to_string(),
                    parameters: HashMap::new(),
                    enabled: true,
                }
            ],
            actions: vec![
                PolicyAction {
                    id: "timeout_retry_action".to_string(),
                    action_type: ActionType::Retry,
                    parameters: HashMap::from([
                        ("strategy".to_string(), "linear".to_string()),
                        ("max_attempts".to_string(), "3".to_string()),
                    ]),
                    priority: Priority::Medium,
                    enabled: true,
                }
            ],
            priority: 8,
            enabled: true,
            metadata: HashMap::new(),
        }
    }

    /// Rate limit policy
    fn create_rate_limit_policy() -> RetryPolicy {
        RetryPolicy {
            id: "rate_limit_policy".to_string(),
            name: "Rate Limit Policy".to_string(),
            rules: vec![],
            conditions: vec![
                PolicyCondition {
                    id: "rate_limit_condition".to_string(),
                    expression: "error_type == rate_limit".to_string(),
                    parameters: HashMap::new(),
                    enabled: true,
                }
            ],
            actions: vec![
                PolicyAction {
                    id: "rate_limit_action".to_string(),
                    action_type: ActionType::RateLimit,
                    parameters: HashMap::from([
                        ("wait_time".to_string(), "60000".to_string()),
                    ]),
                    priority: Priority::High,
                    enabled: true,
                }
            ],
            priority: 15,
            enabled: true,
            metadata: HashMap::new(),
        }
    }

    /// Circuit breaker policy
    fn create_circuit_breaker_policy() -> RetryPolicy {
        RetryPolicy {
            id: "circuit_breaker_policy".to_string(),
            name: "Circuit Breaker Policy".to_string(),
            rules: vec![],
            conditions: vec![
                PolicyCondition {
                    id: "circuit_condition".to_string(),
                    expression: "success_rate_low".to_string(),
                    parameters: HashMap::new(),
                    enabled: true,
                }
            ],
            actions: vec![
                PolicyAction {
                    id: "circuit_action".to_string(),
                    action_type: ActionType::CircuitBreak,
                    parameters: HashMap::new(),
                    priority: Priority::Critical,
                    enabled: true,
                }
            ],
            priority: 20,
            enabled: true,
            metadata: HashMap::new(),
        }
    }
}
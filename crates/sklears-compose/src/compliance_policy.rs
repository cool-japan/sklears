//! Policy engine, rules, and violation management
//!
//! This module provides comprehensive policy management capabilities including
//! policy definition, rule evaluation, violation tracking, and automated
//! enforcement for regulatory compliance frameworks.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::{Duration, SystemTime},
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::context_core::ContextResult;
use crate::compliance_core::RegulatoryFramework;
use crate::compliance_regulatory::ComplianceSeverity;

/// Compliance policy engine
#[derive(Debug)]
pub struct CompliancePolicyEngine {
    /// Active policies
    pub policies: HashMap<String, CompliancePolicy>,
    /// Policy evaluator
    pub evaluator: Box<dyn CompliancePolicyEvaluator>,
    /// Policy violations
    pub violations: VecDeque<PolicyViolation>,
    /// Configuration
    pub config: PolicyEngineConfig,
}

/// Compliance policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompliancePolicy {
    /// Policy ID
    pub id: String,
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: String,
    /// Applicable frameworks
    pub frameworks: HashSet<RegulatoryFramework>,
    /// Policy rules
    pub rules: Vec<PolicyRule>,
    /// Policy status
    pub status: PolicyStatus,
    /// Effective date
    pub effective_date: SystemTime,
    /// Expiration date
    pub expiration_date: Option<SystemTime>,
    /// Policy owner
    pub owner: String,
}

/// Policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Rule ID
    pub id: String,
    /// Rule condition
    pub condition: String,
    /// Rule action
    pub action: PolicyAction,
    /// Rule priority
    pub priority: u32,
    /// Rule enabled
    pub enabled: bool,
}

/// Policy actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyAction {
    /// Allow the action
    Allow,
    /// Deny the action
    Deny,
    /// Require approval
    RequireApproval,
    /// Log and continue
    LogAndContinue,
    /// Quarantine
    Quarantine,
    /// Custom action
    Custom(String),
}

/// Policy status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyStatus {
    /// Draft
    Draft,
    /// Active
    Active,
    /// Suspended
    Suspended,
    /// Deprecated
    Deprecated,
    /// Archived
    Archived,
}

/// Policy violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyViolation {
    /// Violation ID
    pub id: Uuid,
    /// Policy ID
    pub policy_id: String,
    /// Rule ID
    pub rule_id: String,
    /// Violation timestamp
    pub timestamp: SystemTime,
    /// Actor
    pub actor: String,
    /// Resource
    pub resource: String,
    /// Action attempted
    pub action: String,
    /// Violation details
    pub details: HashMap<String, serde_json::Value>,
    /// Severity
    pub severity: ComplianceSeverity,
    /// Status
    pub status: ViolationStatus,
}

/// Violation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationStatus {
    /// Open
    Open,
    /// Investigating
    Investigating,
    /// Resolved
    Resolved,
    /// False positive
    FalsePositive,
    /// Accepted risk
    AcceptedRisk,
}

/// Compliance policy evaluator trait
pub trait CompliancePolicyEvaluator: Send + Sync + std::fmt::Debug {
    /// Evaluate policy
    fn evaluate(&self, context: &PolicyEvaluationContext) -> ContextResult<PolicyEvaluationResult>;

    /// Get evaluator name
    fn name(&self) -> &str;
}

/// Policy evaluation context
#[derive(Debug, Clone)]
pub struct PolicyEvaluationContext {
    /// Actor performing the action
    pub actor: String,
    /// Resource being accessed
    pub resource: String,
    /// Action being performed
    pub action: String,
    /// Request context
    pub context: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Policy evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluationResult {
    /// Evaluation decision
    pub decision: PolicyAction,
    /// Applicable policies
    pub policies: Vec<String>,
    /// Decision reason
    pub reason: String,
    /// Confidence score
    pub confidence: f64,
    /// Additional obligations
    pub obligations: Vec<String>,
}

/// Policy engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEngineConfig {
    /// Enable policy enforcement
    pub enforcement_enabled: bool,
    /// Default policy action
    pub default_action: PolicyAction,
    /// Policy cache TTL
    pub cache_ttl: Duration,
    /// Violation alert threshold
    pub violation_threshold: usize,
}

impl Default for PolicyEngineConfig {
    fn default() -> Self {
        Self {
            enforcement_enabled: true,
            default_action: PolicyAction::Deny,
            cache_ttl: Duration::from_secs(5 * 60), // 5 minutes
            violation_threshold: 10,
        }
    }
}

/// Default policy evaluator implementation
#[derive(Debug)]
pub struct DefaultPolicyEvaluator;

impl CompliancePolicyEvaluator for DefaultPolicyEvaluator {
    fn evaluate(&self, _context: &PolicyEvaluationContext) -> ContextResult<PolicyEvaluationResult> {
        Ok(PolicyEvaluationResult {
            decision: PolicyAction::Allow,
            policies: vec![],
            reason: "Default evaluation - no matching policies".to_string(),
            confidence: 0.5,
            obligations: vec![],
        })
    }

    fn name(&self) -> &str {
        "default"
    }
}

impl CompliancePolicyEngine {
    /// Create a new compliance policy engine
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            evaluator: Box::new(DefaultPolicyEvaluator),
            violations: VecDeque::new(),
            config: PolicyEngineConfig::default(),
        }
    }

    /// Create with custom evaluator
    pub fn with_evaluator(evaluator: Box<dyn CompliancePolicyEvaluator>) -> Self {
        Self {
            policies: HashMap::new(),
            evaluator,
            violations: VecDeque::new(),
            config: PolicyEngineConfig::default(),
        }
    }

    /// Add a policy
    pub fn add_policy(&mut self, policy: CompliancePolicy) {
        self.policies.insert(policy.id.clone(), policy);
    }

    /// Remove a policy
    pub fn remove_policy(&mut self, policy_id: &str) -> Option<CompliancePolicy> {
        self.policies.remove(policy_id)
    }

    /// Get policy by ID
    pub fn get_policy(&self, policy_id: &str) -> Option<&CompliancePolicy> {
        self.policies.get(policy_id)
    }

    /// Get active policies
    pub fn get_active_policies(&self) -> Vec<&CompliancePolicy> {
        self.policies
            .values()
            .filter(|policy| policy.status == PolicyStatus::Active)
            .collect()
    }

    /// Get policies by framework
    pub fn get_policies_by_framework(&self, framework: &RegulatoryFramework) -> Vec<&CompliancePolicy> {
        self.policies
            .values()
            .filter(|policy| policy.frameworks.contains(framework))
            .collect()
    }

    /// Evaluate policy for given context
    pub fn evaluate_policy(&self, context: &PolicyEvaluationContext) -> ContextResult<PolicyEvaluationResult> {
        if !self.config.enforcement_enabled {
            return Ok(PolicyEvaluationResult {
                decision: PolicyAction::Allow,
                policies: vec![],
                reason: "Policy enforcement disabled".to_string(),
                confidence: 1.0,
                obligations: vec![],
            });
        }

        // Find applicable policies
        let applicable_policies = self.find_applicable_policies(context);

        if applicable_policies.is_empty() {
            return Ok(PolicyEvaluationResult {
                decision: self.config.default_action.clone(),
                policies: vec![],
                reason: "No applicable policies found".to_string(),
                confidence: 0.8,
                obligations: vec![],
            });
        }

        // Evaluate against applicable policies
        self.evaluator.evaluate(context)
    }

    /// Find applicable policies for context
    fn find_applicable_policies(&self, context: &PolicyEvaluationContext) -> Vec<&CompliancePolicy> {
        self.get_active_policies()
            .into_iter()
            .filter(|policy| self.matches_policy_conditions(policy, context))
            .collect()
    }

    /// Check if policy conditions match context
    fn matches_policy_conditions(&self, _policy: &CompliancePolicy, _context: &PolicyEvaluationContext) -> bool {
        // Simplified matching logic - in a real implementation,
        // this would evaluate the policy rules against the context
        true
    }

    /// Record policy violation
    pub fn record_violation(&mut self, violation: PolicyViolation) {
        // Check if violation count exceeds threshold
        if self.violations.len() >= self.config.violation_threshold {
            // Handle threshold exceeded (could trigger alerts)
            self.handle_violation_threshold_exceeded();
        }

        self.violations.push_back(violation);
    }

    /// Get violations by status
    pub fn get_violations_by_status(&self, status: ViolationStatus) -> Vec<&PolicyViolation> {
        self.violations
            .iter()
            .filter(|violation| violation.status == status)
            .collect()
    }

    /// Get violations by policy
    pub fn get_violations_by_policy(&self, policy_id: &str) -> Vec<&PolicyViolation> {
        self.violations
            .iter()
            .filter(|violation| violation.policy_id == policy_id)
            .collect()
    }

    /// Get violations by severity
    pub fn get_violations_by_severity(&self, severity: ComplianceSeverity) -> Vec<&PolicyViolation> {
        self.violations
            .iter()
            .filter(|violation| violation.severity == severity)
            .collect()
    }

    /// Update violation status
    pub fn update_violation_status(&mut self, violation_id: &Uuid, status: ViolationStatus) -> bool {
        if let Some(violation) = self.violations.iter_mut().find(|v| v.id == *violation_id) {
            violation.status = status;
            true
        } else {
            false
        }
    }

    /// Enable or disable policy
    pub fn set_policy_status(&mut self, policy_id: &str, status: PolicyStatus) -> bool {
        if let Some(policy) = self.policies.get_mut(policy_id) {
            policy.status = status;
            true
        } else {
            false
        }
    }

    /// Get policy statistics
    pub fn get_policy_statistics(&self) -> PolicyStatistics {
        let total_policies = self.policies.len();
        let active_policies = self.get_active_policies().len();
        let total_violations = self.violations.len();
        let open_violations = self.get_violations_by_status(ViolationStatus::Open).len();

        let violation_rate = if total_policies > 0 {
            total_violations as f64 / total_policies as f64
        } else {
            0.0
        };

        PolicyStatistics {
            total_policies,
            active_policies,
            total_violations,
            open_violations,
            violation_rate,
        }
    }

    /// Clean up old violations
    pub fn cleanup_old_violations(&mut self, cutoff_time: SystemTime) {
        self.violations.retain(|violation| violation.timestamp > cutoff_time);
    }

    /// Handle violation threshold exceeded
    fn handle_violation_threshold_exceeded(&self) {
        // Placeholder for alert handling
        // In a real implementation, this might send notifications,
        // escalate to security teams, or trigger automated responses
    }

    /// Set policy evaluator
    pub fn set_evaluator(&mut self, evaluator: Box<dyn CompliancePolicyEvaluator>) {
        self.evaluator = evaluator;
    }
}

/// Policy statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyStatistics {
    /// Total policies
    pub total_policies: usize,
    /// Active policies
    pub active_policies: usize,
    /// Total violations
    pub total_violations: usize,
    /// Open violations
    pub open_violations: usize,
    /// Violation rate
    pub violation_rate: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_engine_creation() {
        let engine = CompliancePolicyEngine::new();
        assert_eq!(engine.policies.len(), 0);
        assert_eq!(engine.violations.len(), 0);
        assert!(engine.config.enforcement_enabled);
    }

    #[test]
    fn test_policy_management() {
        let mut engine = CompliancePolicyEngine::new();

        let mut frameworks = HashSet::new();
        frameworks.insert(RegulatoryFramework::Gdpr);

        let policy = CompliancePolicy {
            id: "test-policy".to_string(),
            name: "Test Policy".to_string(),
            description: "A test policy".to_string(),
            frameworks,
            rules: Vec::new(),
            status: PolicyStatus::Active,
            effective_date: SystemTime::now(),
            expiration_date: None,
            owner: "test-owner".to_string(),
        };

        // Add policy
        engine.add_policy(policy);
        assert_eq!(engine.policies.len(), 1);

        // Get policy
        assert!(engine.get_policy("test-policy").is_some());

        // Get active policies
        let active_policies = engine.get_active_policies();
        assert_eq!(active_policies.len(), 1);

        // Remove policy
        let removed_policy = engine.remove_policy("test-policy");
        assert!(removed_policy.is_some());
        assert_eq!(engine.policies.len(), 0);
    }

    #[test]
    fn test_policy_violation_recording() {
        let mut engine = CompliancePolicyEngine::new();

        let violation = PolicyViolation {
            id: Uuid::new_v4(),
            policy_id: "test-policy".to_string(),
            rule_id: "test-rule".to_string(),
            timestamp: SystemTime::now(),
            actor: "test-user".to_string(),
            resource: "test-resource".to_string(),
            action: "test-action".to_string(),
            details: HashMap::new(),
            severity: ComplianceSeverity::Medium,
            status: ViolationStatus::Open,
        };

        engine.record_violation(violation.clone());
        assert_eq!(engine.violations.len(), 1);

        // Get violations by status
        let open_violations = engine.get_violations_by_status(ViolationStatus::Open);
        assert_eq!(open_violations.len(), 1);

        // Update violation status
        let success = engine.update_violation_status(&violation.id, ViolationStatus::Resolved);
        assert!(success);

        let resolved_violations = engine.get_violations_by_status(ViolationStatus::Resolved);
        assert_eq!(resolved_violations.len(), 1);
    }

    #[test]
    fn test_policy_evaluation_context() {
        let context = PolicyEvaluationContext {
            actor: "test-user".to_string(),
            resource: "test-resource".to_string(),
            action: "read".to_string(),
            context: HashMap::new(),
            timestamp: SystemTime::now(),
        };

        assert_eq!(context.actor, "test-user");
        assert_eq!(context.resource, "test-resource");
        assert_eq!(context.action, "read");
    }

    #[test]
    fn test_policy_actions() {
        assert_eq!(PolicyAction::Allow, PolicyAction::Allow);
        assert_ne!(PolicyAction::Allow, PolicyAction::Deny);
        assert_eq!(PolicyAction::Custom("test".to_string()), PolicyAction::Custom("test".to_string()));
    }

    #[test]
    fn test_policy_statistics() {
        let mut engine = CompliancePolicyEngine::new();
        let stats = engine.get_policy_statistics();

        assert_eq!(stats.total_policies, 0);
        assert_eq!(stats.active_policies, 0);
        assert_eq!(stats.total_violations, 0);
        assert_eq!(stats.violation_rate, 0.0);

        // Add a violation
        let violation = PolicyViolation {
            id: Uuid::new_v4(),
            policy_id: "test-policy".to_string(),
            rule_id: "test-rule".to_string(),
            timestamp: SystemTime::now(),
            actor: "test-user".to_string(),
            resource: "test-resource".to_string(),
            action: "test-action".to_string(),
            details: HashMap::new(),
            severity: ComplianceSeverity::Low,
            status: ViolationStatus::Open,
        };

        engine.record_violation(violation);
        let stats = engine.get_policy_statistics();
        assert_eq!(stats.total_violations, 1);
        assert_eq!(stats.open_violations, 1);
    }

    #[test]
    fn test_default_policy_evaluator() {
        let evaluator = DefaultPolicyEvaluator;
        assert_eq!(evaluator.name(), "default");

        let context = PolicyEvaluationContext {
            actor: "test-user".to_string(),
            resource: "test-resource".to_string(),
            action: "read".to_string(),
            context: HashMap::new(),
            timestamp: SystemTime::now(),
        };

        let result = evaluator.evaluate(&context).unwrap();
        assert_eq!(result.decision, PolicyAction::Allow);
    }
}
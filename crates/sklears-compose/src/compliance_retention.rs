//! Data retention policies, schedules, and lifecycle management
//!
//! This module provides comprehensive data retention management capabilities
//! including retention policy enforcement, scheduling, legal holds,
//! and automated deletion workflows for regulatory compliance.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::{Duration, SystemTime},
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::compliance_governance::DeletionMethod;
use crate::compliance_policy::PolicyStatus;

/// Data retention manager
#[derive(Debug)]
pub struct DataRetentionManager {
    /// Retention policies
    pub policies: HashMap<String, RetentionPolicy>,
    /// Retention schedules
    pub schedules: VecDeque<RetentionSchedule>,
    /// Legal holds
    pub legal_holds: HashMap<String, LegalHold>,
    /// Deletion logs
    pub deletion_logs: VecDeque<DeletionLog>,
    /// Configuration
    pub config: RetentionManagerConfig,
}

/// Retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Policy ID
    pub id: String,
    /// Policy name
    pub name: String,
    /// Applicable data types
    pub data_types: HashSet<String>,
    /// Retention period
    pub retention_period: Duration,
    /// Deletion method
    pub deletion_method: DeletionMethod,
    /// Policy status
    pub status: PolicyStatus,
    /// Legal requirements
    pub legal_requirements: Vec<String>,
    /// Exceptions
    pub exceptions: Vec<RetentionException>,
}

/// Retention exception
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionException {
    /// Exception ID
    pub id: String,
    /// Exception condition
    pub condition: String,
    /// Modified retention period
    pub modified_retention: Duration,
    /// Exception reason
    pub reason: String,
    /// Approval required
    pub requires_approval: bool,
}

/// Retention schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionSchedule {
    /// Schedule ID
    pub id: Uuid,
    /// Data asset ID
    pub asset_id: String,
    /// Policy ID
    pub policy_id: String,
    /// Scheduled deletion date
    pub deletion_date: SystemTime,
    /// Schedule status
    pub status: ScheduleStatus,
    /// Creation date
    pub created_at: SystemTime,
    /// Last updated
    pub updated_at: SystemTime,
}

/// Schedule status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScheduleStatus {
    /// Scheduled
    Scheduled,
    /// On hold
    OnHold,
    /// Processing
    Processing,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
}

/// Legal hold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalHold {
    /// Hold ID
    pub id: String,
    /// Hold name
    pub name: String,
    /// Hold reason
    pub reason: String,
    /// Custodian
    pub custodian: String,
    /// Matter description
    pub matter_description: String,
    /// Data scope
    pub data_scope: DataScope,
    /// Hold status
    pub status: LegalHoldStatus,
    /// Created date
    pub created_at: SystemTime,
    /// Release date
    pub released_at: Option<SystemTime>,
    /// Hold instructions
    pub instructions: String,
}

/// Data scope for legal hold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataScope {
    /// Date range
    pub date_range: (SystemTime, SystemTime),
    /// Custodians
    pub custodians: Vec<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// Data sources
    pub data_sources: Vec<String>,
    /// File types
    pub file_types: Vec<String>,
}

/// Legal hold status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegalHoldStatus {
    /// Active
    Active,
    /// Released
    Released,
    /// Expired
    Expired,
    /// Cancelled
    Cancelled,
}

/// Deletion log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionLog {
    /// Log ID
    pub id: Uuid,
    /// Asset ID
    pub asset_id: String,
    /// Deletion timestamp
    pub deleted_at: SystemTime,
    /// Deletion method
    pub method: DeletionMethod,
    /// Reason for deletion
    pub reason: String,
    /// Approver
    pub approver: Option<String>,
    /// Verification status
    pub verification_status: VerificationStatus,
    /// Certificate of destruction
    pub certificate: Option<String>,
}

/// Verification status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    /// Pending verification
    Pending,
    /// Verified
    Verified,
    /// Failed verification
    Failed,
    /// Not required
    NotRequired,
}

/// Retention manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionManagerConfig {
    /// Enable automatic deletion
    pub auto_deletion: bool,
    /// Deletion buffer period
    pub buffer_period: Duration,
    /// Require approval for manual deletion
    pub require_approval: bool,
    /// Enable verification
    pub enable_verification: bool,
    /// Verification method
    pub verification_method: String,
}

impl Default for RetentionManagerConfig {
    fn default() -> Self {
        Self {
            auto_deletion: true,
            buffer_period: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            require_approval: true,
            enable_verification: true,
            verification_method: "hash_verification".to_string(),
        }
    }
}

impl DataRetentionManager {
    /// Create a new data retention manager
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            schedules: VecDeque::new(),
            legal_holds: HashMap::new(),
            deletion_logs: VecDeque::new(),
            config: RetentionManagerConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: RetentionManagerConfig) -> Self {
        Self {
            policies: HashMap::new(),
            schedules: VecDeque::new(),
            legal_holds: HashMap::new(),
            deletion_logs: VecDeque::new(),
            config,
        }
    }

    /// Add retention policy
    pub fn add_policy(&mut self, policy: RetentionPolicy) {
        self.policies.insert(policy.id.clone(), policy);
    }

    /// Get retention policy
    pub fn get_policy(&self, policy_id: &str) -> Option<&RetentionPolicy> {
        self.policies.get(policy_id)
    }

    /// Get active policies
    pub fn get_active_policies(&self) -> Vec<&RetentionPolicy> {
        self.policies
            .values()
            .filter(|policy| policy.status == PolicyStatus::Active)
            .collect()
    }

    /// Schedule deletion for asset
    pub fn schedule_deletion(&mut self, asset_id: &str, policy_id: &str) -> Option<Uuid> {
        if let Some(policy) = self.get_policy(policy_id) {
            // Check if asset is under legal hold
            if self.is_asset_under_legal_hold(asset_id) {
                return None; // Cannot schedule deletion for assets under legal hold
            }

            let deletion_date = SystemTime::now() + policy.retention_period + self.config.buffer_period;

            let schedule = RetentionSchedule {
                id: Uuid::new_v4(),
                asset_id: asset_id.to_string(),
                policy_id: policy_id.to_string(),
                deletion_date,
                status: ScheduleStatus::Scheduled,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
            };

            let schedule_id = schedule.id;
            self.schedules.push_back(schedule);
            Some(schedule_id)
        } else {
            None
        }
    }

    /// Get due schedules
    pub fn get_due_schedules(&self) -> Vec<&RetentionSchedule> {
        let now = SystemTime::now();
        self.schedules
            .iter()
            .filter(|schedule| {
                schedule.status == ScheduleStatus::Scheduled && schedule.deletion_date <= now
            })
            .collect()
    }

    /// Process deletion schedule
    pub fn process_deletion_schedule(&mut self, schedule_id: &Uuid) -> bool {
        if let Some(schedule) = self.schedules.iter_mut().find(|s| s.id == *schedule_id) {
            // Check for legal hold again before processing
            if self.is_asset_under_legal_hold(&schedule.asset_id) {
                schedule.status = ScheduleStatus::OnHold;
                schedule.updated_at = SystemTime::now();
                return false;
            }

            schedule.status = ScheduleStatus::Processing;
            schedule.updated_at = SystemTime::now();

            // Perform deletion
            let deletion_result = self.delete_asset(&schedule.asset_id, &schedule.policy_id);

            if deletion_result {
                schedule.status = ScheduleStatus::Completed;
                schedule.updated_at = SystemTime::now();
                true
            } else {
                schedule.status = ScheduleStatus::Failed;
                schedule.updated_at = SystemTime::now();
                false
            }
        } else {
            false
        }
    }

    /// Delete asset
    fn delete_asset(&mut self, asset_id: &str, policy_id: &str) -> bool {
        if let Some(policy) = self.get_policy(policy_id) {
            let deletion_log = DeletionLog {
                id: Uuid::new_v4(),
                asset_id: asset_id.to_string(),
                deleted_at: SystemTime::now(),
                method: policy.deletion_method.clone(),
                reason: format!("Scheduled deletion per policy {}", policy_id),
                approver: None,
                verification_status: if self.config.enable_verification {
                    VerificationStatus::Pending
                } else {
                    VerificationStatus::NotRequired
                },
                certificate: None,
            };

            self.deletion_logs.push_back(deletion_log);

            // In a real implementation, this would perform the actual deletion
            // based on the deletion method
            true
        } else {
            false
        }
    }

    /// Add legal hold
    pub fn add_legal_hold(&mut self, hold: LegalHold) {
        // Put related schedules on hold
        for schedule in &mut self.schedules {
            if self.is_asset_in_scope(&schedule.asset_id, &hold.data_scope) {
                schedule.status = ScheduleStatus::OnHold;
                schedule.updated_at = SystemTime::now();
            }
        }

        self.legal_holds.insert(hold.id.clone(), hold);
    }

    /// Release legal hold
    pub fn release_legal_hold(&mut self, hold_id: &str) -> bool {
        if let Some(hold) = self.legal_holds.get_mut(hold_id) {
            hold.status = LegalHoldStatus::Released;
            hold.released_at = Some(SystemTime::now());

            // Resume related schedules
            for schedule in &mut self.schedules {
                if schedule.status == ScheduleStatus::OnHold
                    && self.is_asset_in_scope(&schedule.asset_id, &hold.data_scope) {
                    schedule.status = ScheduleStatus::Scheduled;
                    schedule.updated_at = SystemTime::now();
                }
            }

            true
        } else {
            false
        }
    }

    /// Check if asset is under legal hold
    pub fn is_asset_under_legal_hold(&self, asset_id: &str) -> bool {
        self.legal_holds
            .values()
            .any(|hold| {
                hold.status == LegalHoldStatus::Active
                    && self.is_asset_in_scope(asset_id, &hold.data_scope)
            })
    }

    /// Check if asset is in legal hold scope
    fn is_asset_in_scope(&self, asset_id: &str, scope: &DataScope) -> bool {
        // Simplified scope checking - in a real implementation,
        // this would perform more comprehensive matching
        scope.data_sources.contains(&asset_id.to_string())
    }

    /// Get schedules by status
    pub fn get_schedules_by_status(&self, status: ScheduleStatus) -> Vec<&RetentionSchedule> {
        self.schedules
            .iter()
            .filter(|schedule| schedule.status == status)
            .collect()
    }

    /// Get active legal holds
    pub fn get_active_legal_holds(&self) -> Vec<&LegalHold> {
        self.legal_holds
            .values()
            .filter(|hold| hold.status == LegalHoldStatus::Active)
            .collect()
    }

    /// Update verification status
    pub fn update_verification_status(&mut self, deletion_id: &Uuid, status: VerificationStatus) -> bool {
        if let Some(log) = self.deletion_logs.iter_mut().find(|log| log.id == *deletion_id) {
            log.verification_status = status;
            true
        } else {
            false
        }
    }

    /// Get unverified deletions
    pub fn get_unverified_deletions(&self) -> Vec<&DeletionLog> {
        self.deletion_logs
            .iter()
            .filter(|log| log.verification_status == VerificationStatus::Pending)
            .collect()
    }

    /// Get retention statistics
    pub fn get_retention_statistics(&self) -> RetentionStatistics {
        let total_policies = self.policies.len();
        let active_policies = self.get_active_policies().len();
        let total_schedules = self.schedules.len();
        let due_schedules = self.get_due_schedules().len();
        let completed_deletions = self.get_schedules_by_status(ScheduleStatus::Completed).len();
        let active_legal_holds = self.get_active_legal_holds().len();
        let unverified_deletions = self.get_unverified_deletions().len();

        let completion_rate = if total_schedules > 0 {
            completed_deletions as f64 / total_schedules as f64
        } else {
            0.0
        };

        RetentionStatistics {
            total_policies,
            active_policies,
            total_schedules,
            due_schedules,
            completed_deletions,
            active_legal_holds,
            unverified_deletions,
            completion_rate,
        }
    }

    /// Process due schedules automatically
    pub fn process_due_schedules(&mut self) -> usize {
        if !self.config.auto_deletion {
            return 0;
        }

        let due_schedule_ids: Vec<Uuid> = self.get_due_schedules()
            .iter()
            .map(|schedule| schedule.id)
            .collect();

        let mut processed_count = 0;
        for schedule_id in due_schedule_ids {
            if self.process_deletion_schedule(&schedule_id) {
                processed_count += 1;
            }
        }

        processed_count
    }

    /// Clean up old deletion logs
    pub fn cleanup_old_deletion_logs(&mut self, cutoff_time: SystemTime) {
        self.deletion_logs.retain(|log| log.deleted_at > cutoff_time);
    }

    /// Cancel schedule
    pub fn cancel_schedule(&mut self, schedule_id: &Uuid) -> bool {
        if let Some(schedule) = self.schedules.iter_mut().find(|s| s.id == *schedule_id) {
            schedule.status = ScheduleStatus::Cancelled;
            schedule.updated_at = SystemTime::now();
            true
        } else {
            false
        }
    }
}

/// Retention statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionStatistics {
    /// Total policies
    pub total_policies: usize,
    /// Active policies
    pub active_policies: usize,
    /// Total schedules
    pub total_schedules: usize,
    /// Due schedules
    pub due_schedules: usize,
    /// Completed deletions
    pub completed_deletions: usize,
    /// Active legal holds
    pub active_legal_holds: usize,
    /// Unverified deletions
    pub unverified_deletions: usize,
    /// Completion rate
    pub completion_rate: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retention_manager_creation() {
        let manager = DataRetentionManager::new();
        assert_eq!(manager.policies.len(), 0);
        assert_eq!(manager.schedules.len(), 0);
        assert_eq!(manager.legal_holds.len(), 0);
        assert!(manager.config.auto_deletion);
    }

    #[test]
    fn test_retention_policy_management() {
        let mut manager = DataRetentionManager::new();

        let policy = RetentionPolicy {
            id: "test-policy".to_string(),
            name: "Test Policy".to_string(),
            data_types: ["user_data".to_string()].into(),
            retention_period: Duration::from_secs(365 * 24 * 60 * 60), // 1 year
            deletion_method: DeletionMethod::Soft,
            status: PolicyStatus::Active,
            legal_requirements: vec!["GDPR".to_string()],
            exceptions: Vec::new(),
        };

        manager.add_policy(policy);
        assert_eq!(manager.policies.len(), 1);
        assert!(manager.get_policy("test-policy").is_some());

        let active_policies = manager.get_active_policies();
        assert_eq!(active_policies.len(), 1);
    }

    #[test]
    fn test_deletion_scheduling() {
        let mut manager = DataRetentionManager::new();

        let policy = RetentionPolicy {
            id: "test-policy".to_string(),
            name: "Test Policy".to_string(),
            data_types: ["user_data".to_string()].into(),
            retention_period: Duration::from_secs(365 * 24 * 60 * 60), // 1 year
            deletion_method: DeletionMethod::Soft,
            status: PolicyStatus::Active,
            legal_requirements: Vec::new(),
            exceptions: Vec::new(),
        };

        manager.add_policy(policy);

        let schedule_id = manager.schedule_deletion("asset-123", "test-policy");
        assert!(schedule_id.is_some());
        assert_eq!(manager.schedules.len(), 1);

        let schedule = &manager.schedules[0];
        assert_eq!(schedule.asset_id, "asset-123");
        assert_eq!(schedule.policy_id, "test-policy");
        assert_eq!(schedule.status, ScheduleStatus::Scheduled);
    }

    #[test]
    fn test_legal_hold() {
        let mut manager = DataRetentionManager::new();

        let hold = LegalHold {
            id: "test-hold".to_string(),
            name: "Test Hold".to_string(),
            reason: "Litigation".to_string(),
            custodian: "Legal Team".to_string(),
            matter_description: "Contract dispute".to_string(),
            data_scope: DataScope {
                date_range: (SystemTime::UNIX_EPOCH, SystemTime::now()),
                custodians: vec!["user-123".to_string()],
                keywords: vec!["contract".to_string()],
                data_sources: vec!["asset-123".to_string()],
                file_types: vec!["pdf".to_string()],
            },
            status: LegalHoldStatus::Active,
            created_at: SystemTime::now(),
            released_at: None,
            instructions: "Preserve all related documents".to_string(),
        };

        manager.add_legal_hold(hold);
        assert_eq!(manager.legal_holds.len(), 1);
        assert!(manager.is_asset_under_legal_hold("asset-123"));

        // Release hold
        let success = manager.release_legal_hold("test-hold");
        assert!(success);

        let hold = manager.legal_holds.get("test-hold").unwrap();
        assert_eq!(hold.status, LegalHoldStatus::Released);
        assert!(hold.released_at.is_some());
    }

    #[test]
    fn test_schedule_status_changes() {
        assert_eq!(ScheduleStatus::Scheduled, ScheduleStatus::Scheduled);
        assert_ne!(ScheduleStatus::Scheduled, ScheduleStatus::Completed);
    }

    #[test]
    fn test_legal_hold_status_changes() {
        assert_eq!(LegalHoldStatus::Active, LegalHoldStatus::Active);
        assert_ne!(LegalHoldStatus::Active, LegalHoldStatus::Released);
    }

    #[test]
    fn test_verification_status() {
        assert_eq!(VerificationStatus::Pending, VerificationStatus::Pending);
        assert_ne!(VerificationStatus::Pending, VerificationStatus::Verified);
    }

    #[test]
    fn test_retention_statistics() {
        let manager = DataRetentionManager::new();
        let stats = manager.get_retention_statistics();

        assert_eq!(stats.total_policies, 0);
        assert_eq!(stats.active_policies, 0);
        assert_eq!(stats.total_schedules, 0);
        assert_eq!(stats.completion_rate, 0.0);
    }

    #[test]
    fn test_schedule_cancellation() {
        let mut manager = DataRetentionManager::new();

        let policy = RetentionPolicy {
            id: "test-policy".to_string(),
            name: "Test Policy".to_string(),
            data_types: ["user_data".to_string()].into(),
            retention_period: Duration::from_secs(365 * 24 * 60 * 60),
            deletion_method: DeletionMethod::Soft,
            status: PolicyStatus::Active,
            legal_requirements: Vec::new(),
            exceptions: Vec::new(),
        };

        manager.add_policy(policy);
        let schedule_id = manager.schedule_deletion("asset-123", "test-policy").unwrap();

        let success = manager.cancel_schedule(&schedule_id);
        assert!(success);

        let schedule = &manager.schedules[0];
        assert_eq!(schedule.status, ScheduleStatus::Cancelled);
    }
}
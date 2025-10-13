//! Consent management, privacy rights, and legal basis tracking
//!
//! This module provides comprehensive consent management capabilities including
//! consent recording, withdrawal processing, privacy rights management,
//! and legal basis tracking for compliance with privacy regulations.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::{Duration, SystemTime},
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// Consent manager
#[derive(Debug)]
pub struct ConsentManager {
    /// Consent records
    pub consents: HashMap<String, ConsentRecord>,
    /// Consent requests
    pub consent_requests: VecDeque<ConsentRequest>,
    /// Consent withdrawals
    pub withdrawals: VecDeque<ConsentWithdrawal>,
    /// Configuration
    pub config: ConsentManagerConfig,
}

/// Consent record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Consent ID
    pub id: String,
    /// Data subject ID
    pub data_subject_id: String,
    /// Consent type
    pub consent_type: ConsentType,
    /// Purposes
    pub purposes: HashSet<ProcessingPurpose>,
    /// Data categories
    pub data_categories: HashSet<DataCategory>,
    /// Consent status
    pub status: ConsentStatus,
    /// Granted date
    pub granted_date: SystemTime,
    /// Expiry date
    pub expiry_date: Option<SystemTime>,
    /// Last updated
    pub last_updated: SystemTime,
    /// Withdrawal date
    pub withdrawal_date: Option<SystemTime>,
    /// Legal basis
    pub legal_basis: LegalBasis,
    /// Consent evidence
    pub evidence: ConsentEvidence,
}

/// Consent types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsentType {
    /// Explicit consent
    Explicit,
    /// Implicit consent
    Implicit,
    /// Opt-in consent
    OptIn,
    /// Opt-out consent
    OptOut,
    /// Blanket consent
    Blanket,
    /// Granular consent
    Granular,
    /// Custom consent type
    Custom(String),
}

/// Processing purposes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessingPurpose {
    /// Service provision
    ServiceProvision,
    /// Marketing
    Marketing,
    /// Analytics
    Analytics,
    /// Personalization
    Personalization,
    /// Security
    Security,
    /// Legal compliance
    LegalCompliance,
    /// Research
    Research,
    /// Custom purpose
    Custom(String),
}

/// Data categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataCategory {
    /// Personal data
    Personal,
    /// Sensitive personal data
    SensitivePersonal,
    /// Behavioral data
    Behavioral,
    /// Transactional data
    Transactional,
    /// Technical data
    Technical,
    /// Marketing data
    Marketing,
    /// Custom category
    Custom(String),
}

/// Consent status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentStatus {
    /// Granted
    Granted,
    /// Withdrawn
    Withdrawn,
    /// Expired
    Expired,
    /// Pending
    Pending,
    /// Refused
    Refused,
    /// Invalid
    Invalid,
}

/// Legal basis for processing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegalBasis {
    /// Consent
    Consent,
    /// Contract
    Contract,
    /// Legal obligation
    LegalObligation,
    /// Vital interests
    VitalInterests,
    /// Public task
    PublicTask,
    /// Legitimate interests
    LegitimateInterests,
    /// Custom legal basis
    Custom(String),
}

/// Consent evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentEvidence {
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Evidence data
    pub evidence_data: serde_json::Value,
    /// Collection method
    pub collection_method: String,
    /// IP address
    pub ip_address: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Digital signature
    pub digital_signature: Option<String>,
}

impl Default for ConsentEvidence {
    fn default() -> Self {
        Self {
            evidence_type: EvidenceType::WebForm,
            evidence_data: serde_json::Value::Null,
            collection_method: "web_form".to_string(),
            ip_address: None,
            user_agent: None,
            timestamp: SystemTime::now(),
            digital_signature: None,
        }
    }
}

/// Evidence types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Web form submission
    WebForm,
    /// API call
    ApiCall,
    /// Email confirmation
    EmailConfirmation,
    /// SMS confirmation
    SmsConfirmation,
    /// Digital signature
    DigitalSignature,
    /// Custom evidence type
    Custom(String),
}

/// Consent request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRequest {
    /// Request ID
    pub id: Uuid,
    /// Data subject ID
    pub data_subject_id: String,
    /// Requested purposes
    pub purposes: HashSet<ProcessingPurpose>,
    /// Requested data categories
    pub data_categories: HashSet<DataCategory>,
    /// Request timestamp
    pub requested_at: SystemTime,
    /// Request status
    pub status: RequestStatus,
    /// Response timestamp
    pub responded_at: Option<SystemTime>,
    /// Request context
    pub context: HashMap<String, serde_json::Value>,
}

/// Consent withdrawal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentWithdrawal {
    /// Withdrawal ID
    pub id: Uuid,
    /// Consent ID
    pub consent_id: String,
    /// Data subject ID
    pub data_subject_id: String,
    /// Withdrawal timestamp
    pub withdrawn_at: SystemTime,
    /// Withdrawal reason
    pub reason: Option<String>,
    /// Withdrawal method
    pub method: WithdrawalMethod,
    /// Processing status
    pub processing_status: WithdrawalStatus,
}

/// Request status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestStatus {
    /// Pending
    Pending,
    /// Granted
    Granted,
    /// Denied
    Denied,
    /// Expired
    Expired,
    /// Cancelled
    Cancelled,
}

/// Withdrawal methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WithdrawalMethod {
    /// Web interface
    Web,
    /// Email
    Email,
    /// Phone
    Phone,
    /// Mail
    Mail,
    /// API
    Api,
    /// Custom method
    Custom(String),
}

/// Withdrawal status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WithdrawalStatus {
    /// Pending processing
    Pending,
    /// Processing
    Processing,
    /// Completed
    Completed,
    /// Failed
    Failed,
}

/// Consent manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentManagerConfig {
    /// Default consent expiry
    pub default_expiry: Duration,
    /// Require double opt-in
    pub require_double_opt_in: bool,
    /// Enable consent refresh
    pub enable_consent_refresh: bool,
    /// Refresh interval
    pub refresh_interval: Duration,
    /// Withdrawal processing timeout
    pub withdrawal_timeout: Duration,
}

impl Default for ConsentManagerConfig {
    fn default() -> Self {
        Self {
            default_expiry: Duration::from_secs(2 * 365 * 24 * 60 * 60), // 2 years
            require_double_opt_in: false,
            enable_consent_refresh: true,
            refresh_interval: Duration::from_secs(365 * 24 * 60 * 60), // Annually
            withdrawal_timeout: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
        }
    }
}

impl ConsentManager {
    /// Create a new consent manager
    pub fn new() -> Self {
        Self {
            consents: HashMap::new(),
            consent_requests: VecDeque::new(),
            withdrawals: VecDeque::new(),
            config: ConsentManagerConfig::default(),
        }
    }

    /// Create consent manager with custom config
    pub fn with_config(config: ConsentManagerConfig) -> Self {
        Self {
            consents: HashMap::new(),
            consent_requests: VecDeque::new(),
            withdrawals: VecDeque::new(),
            config,
        }
    }

    /// Record a new consent
    pub fn record_consent(&mut self, consent: ConsentRecord) {
        self.consents.insert(consent.id.clone(), consent);
    }

    /// Get consent by ID
    pub fn get_consent(&self, consent_id: &str) -> Option<&ConsentRecord> {
        self.consents.get(consent_id)
    }

    /// Get consents for data subject
    pub fn get_consents_for_subject(&self, data_subject_id: &str) -> Vec<&ConsentRecord> {
        self.consents
            .values()
            .filter(|consent| consent.data_subject_id == data_subject_id)
            .collect()
    }

    /// Get active consents for data subject
    pub fn get_active_consents_for_subject(&self, data_subject_id: &str) -> Vec<&ConsentRecord> {
        self.get_consents_for_subject(data_subject_id)
            .into_iter()
            .filter(|consent| consent.status == ConsentStatus::Granted)
            .filter(|consent| !self.is_consent_expired(consent))
            .collect()
    }

    /// Check if subject has consent for purpose
    pub fn has_consent_for_purpose(&self, data_subject_id: &str, purpose: &ProcessingPurpose) -> bool {
        self.get_active_consents_for_subject(data_subject_id)
            .iter()
            .any(|consent| consent.purposes.contains(purpose))
    }

    /// Check if subject has consent for data category
    pub fn has_consent_for_category(&self, data_subject_id: &str, category: &DataCategory) -> bool {
        self.get_active_consents_for_subject(data_subject_id)
            .iter()
            .any(|consent| consent.data_categories.contains(category))
    }

    /// Submit consent request
    pub fn submit_consent_request(&mut self, request: ConsentRequest) {
        self.consent_requests.push_back(request);
    }

    /// Process consent request
    pub fn process_consent_request(&mut self, request_id: &Uuid, granted: bool) -> Option<ConsentRequest> {
        if let Some(pos) = self.consent_requests.iter().position(|req| req.id == *request_id) {
            let mut request = self.consent_requests.remove(pos).unwrap();
            request.status = if granted { RequestStatus::Granted } else { RequestStatus::Denied };
            request.responded_at = Some(SystemTime::now());

            if granted {
                // Create consent record
                let consent = ConsentRecord {
                    id: format!("consent-{}", Uuid::new_v4()),
                    data_subject_id: request.data_subject_id.clone(),
                    consent_type: ConsentType::Explicit,
                    purposes: request.purposes.clone(),
                    data_categories: request.data_categories.clone(),
                    status: ConsentStatus::Granted,
                    granted_date: SystemTime::now(),
                    expiry_date: Some(SystemTime::now() + self.config.default_expiry),
                    last_updated: SystemTime::now(),
                    withdrawal_date: None,
                    legal_basis: LegalBasis::Consent,
                    evidence: ConsentEvidence::default(),
                };

                self.record_consent(consent);
            }

            Some(request)
        } else {
            None
        }
    }

    /// Withdraw consent
    pub fn withdraw_consent(&mut self, consent_id: &str, withdrawal: ConsentWithdrawal) -> bool {
        if let Some(consent) = self.consents.get_mut(consent_id) {
            consent.status = ConsentStatus::Withdrawn;
            consent.withdrawal_date = Some(withdrawal.withdrawn_at);
            consent.last_updated = SystemTime::now();

            self.withdrawals.push_back(withdrawal);
            true
        } else {
            false
        }
    }

    /// Update withdrawal status
    pub fn update_withdrawal_status(&mut self, withdrawal_id: &Uuid, status: WithdrawalStatus) -> bool {
        if let Some(withdrawal) = self.withdrawals.iter_mut().find(|w| w.id == *withdrawal_id) {
            withdrawal.processing_status = status;
            true
        } else {
            false
        }
    }

    /// Get expired consents
    pub fn get_expired_consents(&self) -> Vec<&ConsentRecord> {
        let now = SystemTime::now();
        self.consents
            .values()
            .filter(|consent| {
                if let Some(expiry_date) = consent.expiry_date {
                    expiry_date <= now
                } else {
                    false
                }
            })
            .collect()
    }

    /// Mark expired consents
    pub fn mark_expired_consents(&mut self) -> usize {
        let expired_consent_ids: Vec<String> = self.get_expired_consents()
            .iter()
            .map(|consent| consent.id.clone())
            .collect();

        for consent_id in &expired_consent_ids {
            if let Some(consent) = self.consents.get_mut(consent_id) {
                consent.status = ConsentStatus::Expired;
                consent.last_updated = SystemTime::now();
            }
        }

        expired_consent_ids.len()
    }

    /// Get consents requiring refresh
    pub fn get_consents_requiring_refresh(&self) -> Vec<&ConsentRecord> {
        if !self.config.enable_consent_refresh {
            return Vec::new();
        }

        let refresh_cutoff = SystemTime::now()
            .checked_sub(self.config.refresh_interval)
            .unwrap_or(SystemTime::UNIX_EPOCH);

        self.consents
            .values()
            .filter(|consent| consent.status == ConsentStatus::Granted)
            .filter(|consent| consent.granted_date < refresh_cutoff)
            .collect()
    }

    /// Check if consent is expired
    fn is_consent_expired(&self, consent: &ConsentRecord) -> bool {
        if let Some(expiry_date) = consent.expiry_date {
            expiry_date <= SystemTime::now()
        } else {
            false
        }
    }

    /// Get consent statistics
    pub fn get_consent_statistics(&self) -> ConsentStatistics {
        let total_consents = self.consents.len();
        let granted_consents = self.consents
            .values()
            .filter(|c| c.status == ConsentStatus::Granted)
            .count();
        let withdrawn_consents = self.consents
            .values()
            .filter(|c| c.status == ConsentStatus::Withdrawn)
            .count();
        let expired_consents = self.get_expired_consents().len();
        let pending_requests = self.consent_requests
            .iter()
            .filter(|r| r.status == RequestStatus::Pending)
            .count();
        let pending_withdrawals = self.withdrawals
            .iter()
            .filter(|w| w.processing_status == WithdrawalStatus::Pending)
            .count();

        let consent_rate = if total_consents > 0 {
            granted_consents as f64 / total_consents as f64
        } else {
            0.0
        };

        ConsentStatistics {
            total_consents,
            granted_consents,
            withdrawn_consents,
            expired_consents,
            pending_requests,
            pending_withdrawals,
            consent_rate,
        }
    }

    /// Clean up old consent requests
    pub fn cleanup_old_requests(&mut self, cutoff_time: SystemTime) {
        self.consent_requests.retain(|request| request.requested_at > cutoff_time);
    }

    /// Clean up processed withdrawals
    pub fn cleanup_processed_withdrawals(&mut self) {
        self.withdrawals.retain(|withdrawal| withdrawal.processing_status != WithdrawalStatus::Completed);
    }
}

/// Consent statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentStatistics {
    /// Total consents
    pub total_consents: usize,
    /// Granted consents
    pub granted_consents: usize,
    /// Withdrawn consents
    pub withdrawn_consents: usize,
    /// Expired consents
    pub expired_consents: usize,
    /// Pending requests
    pub pending_requests: usize,
    /// Pending withdrawals
    pub pending_withdrawals: usize,
    /// Consent rate
    pub consent_rate: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consent_manager_creation() {
        let manager = ConsentManager::new();
        assert_eq!(manager.consents.len(), 0);
        assert_eq!(manager.consent_requests.len(), 0);
        assert_eq!(manager.withdrawals.len(), 0);
    }

    #[test]
    fn test_consent_recording() {
        let mut manager = ConsentManager::new();

        let consent = ConsentRecord {
            id: "test-consent".to_string(),
            data_subject_id: "subject-123".to_string(),
            consent_type: ConsentType::Explicit,
            purposes: [ProcessingPurpose::ServiceProvision].into(),
            data_categories: [DataCategory::Personal].into(),
            status: ConsentStatus::Granted,
            granted_date: SystemTime::now(),
            expiry_date: None,
            last_updated: SystemTime::now(),
            withdrawal_date: None,
            legal_basis: LegalBasis::Consent,
            evidence: ConsentEvidence::default(),
        };

        manager.record_consent(consent);
        assert_eq!(manager.consents.len(), 1);
        assert!(manager.get_consent("test-consent").is_some());
    }

    #[test]
    fn test_consent_purpose_checking() {
        let mut manager = ConsentManager::new();

        let consent = ConsentRecord {
            id: "test-consent".to_string(),
            data_subject_id: "subject-123".to_string(),
            consent_type: ConsentType::Explicit,
            purposes: [ProcessingPurpose::Marketing, ProcessingPurpose::Analytics].into(),
            data_categories: [DataCategory::Personal].into(),
            status: ConsentStatus::Granted,
            granted_date: SystemTime::now(),
            expiry_date: None,
            last_updated: SystemTime::now(),
            withdrawal_date: None,
            legal_basis: LegalBasis::Consent,
            evidence: ConsentEvidence::default(),
        };

        manager.record_consent(consent);

        assert!(manager.has_consent_for_purpose("subject-123", &ProcessingPurpose::Marketing));
        assert!(manager.has_consent_for_purpose("subject-123", &ProcessingPurpose::Analytics));
        assert!(!manager.has_consent_for_purpose("subject-123", &ProcessingPurpose::Security));
    }

    #[test]
    fn test_consent_request_processing() {
        let mut manager = ConsentManager::new();

        let request = ConsentRequest {
            id: Uuid::new_v4(),
            data_subject_id: "subject-123".to_string(),
            purposes: [ProcessingPurpose::Marketing].into(),
            data_categories: [DataCategory::Personal].into(),
            requested_at: SystemTime::now(),
            status: RequestStatus::Pending,
            responded_at: None,
            context: HashMap::new(),
        };

        let request_id = request.id;
        manager.submit_consent_request(request);
        assert_eq!(manager.consent_requests.len(), 1);

        // Process request (grant)
        let processed = manager.process_consent_request(&request_id, true);
        assert!(processed.is_some());
        assert_eq!(processed.unwrap().status, RequestStatus::Granted);
        assert_eq!(manager.consents.len(), 1);
    }

    #[test]
    fn test_consent_withdrawal() {
        let mut manager = ConsentManager::new();

        let consent = ConsentRecord {
            id: "test-consent".to_string(),
            data_subject_id: "subject-123".to_string(),
            consent_type: ConsentType::Explicit,
            purposes: [ProcessingPurpose::Marketing].into(),
            data_categories: [DataCategory::Personal].into(),
            status: ConsentStatus::Granted,
            granted_date: SystemTime::now(),
            expiry_date: None,
            last_updated: SystemTime::now(),
            withdrawal_date: None,
            legal_basis: LegalBasis::Consent,
            evidence: ConsentEvidence::default(),
        };

        manager.record_consent(consent);

        let withdrawal = ConsentWithdrawal {
            id: Uuid::new_v4(),
            consent_id: "test-consent".to_string(),
            data_subject_id: "subject-123".to_string(),
            withdrawn_at: SystemTime::now(),
            reason: Some("No longer needed".to_string()),
            method: WithdrawalMethod::Web,
            processing_status: WithdrawalStatus::Pending,
        };

        let success = manager.withdraw_consent("test-consent", withdrawal);
        assert!(success);

        let consent = manager.get_consent("test-consent").unwrap();
        assert_eq!(consent.status, ConsentStatus::Withdrawn);
        assert!(consent.withdrawal_date.is_some());
        assert_eq!(manager.withdrawals.len(), 1);
    }

    #[test]
    fn test_consent_types() {
        assert_eq!(ConsentType::Explicit, ConsentType::Explicit);
        assert_ne!(ConsentType::Explicit, ConsentType::Implicit);
    }

    #[test]
    fn test_processing_purposes() {
        assert_eq!(ProcessingPurpose::Marketing, ProcessingPurpose::Marketing);
        assert_ne!(ProcessingPurpose::Marketing, ProcessingPurpose::Analytics);
    }

    #[test]
    fn test_legal_basis() {
        assert_eq!(LegalBasis::Consent, LegalBasis::Consent);
        assert_ne!(LegalBasis::Consent, LegalBasis::Contract);
    }

    #[test]
    fn test_consent_statistics() {
        let mut manager = ConsentManager::new();
        let stats = manager.get_consent_statistics();

        assert_eq!(stats.total_consents, 0);
        assert_eq!(stats.granted_consents, 0);
        assert_eq!(stats.consent_rate, 0.0);

        // Add a granted consent
        let consent = ConsentRecord {
            id: "test-consent".to_string(),
            data_subject_id: "subject-123".to_string(),
            consent_type: ConsentType::Explicit,
            purposes: [ProcessingPurpose::Marketing].into(),
            data_categories: [DataCategory::Personal].into(),
            status: ConsentStatus::Granted,
            granted_date: SystemTime::now(),
            expiry_date: None,
            last_updated: SystemTime::now(),
            withdrawal_date: None,
            legal_basis: LegalBasis::Consent,
            evidence: ConsentEvidence::default(),
        };

        manager.record_consent(consent);
        let stats = manager.get_consent_statistics();

        assert_eq!(stats.total_consents, 1);
        assert_eq!(stats.granted_consents, 1);
        assert_eq!(stats.consent_rate, 1.0);
    }
}
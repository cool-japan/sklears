//! Compliance context for regulatory and data governance
//!
//! This module provides comprehensive compliance management including
//! regulatory frameworks, data governance, policy enforcement,
//! and audit reporting capabilities.
//!
//! ## Architecture
//!
//! The compliance context is organized into specialized modules:
//!
//! - **Core**: Foundational traits, types, and the main ComplianceContext
//! - **Governance**: Data governance, classification, lineage, and quality management
//! - **Regulatory**: Regulatory frameworks, standards, and compliance rules
//! - **Audit**: Audit trails, reporting, and compliance auditing
//! - **Policy**: Policy engine, rules, and violation management
//! - **Consent**: Consent management, privacy rights, and legal basis tracking
//! - **Retention**: Data retention policies, schedules, and lifecycle management
//! - **Breach**: Breach detection, incident response, and investigation management
//! - **Metrics**: Compliance metrics, analytics, and reporting
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use crate::context_compliance::*;
//!
//! // Create compliance configuration
//! let config = ComplianceConfig::default();
//!
//! // Create compliance context
//! let context = ComplianceContext::new("main-compliance".to_string(), config);
//!
//! // Initialize the context
//! context.initialize()?;
//!
//! // Register a data asset
//! let asset = DataAsset {
//!     id: "user-data".to_string(),
//!     name: "User Personal Data".to_string(),
//!     asset_type: DataAssetType::Database,
//!     location: "prod-db".to_string(),
//!     owner: "data-team".to_string(),
//!     classification: DataClassification {
//!         level: ClassificationLevel::Confidential,
//!         sensitive_types: [SensitiveDataType::Pii].into(),
//!         regulatory_requirements: [RegulatoryFramework::Gdpr].into(),
//!         access_restrictions: AccessRestrictions::default(),
//!         retention_requirements: RetentionRequirements::default(),
//!         geographic_restrictions: GeographicRestrictions::default(),
//!     },
//!     metadata: HashMap::new(),
//!     created_at: SystemTime::now(),
//!     updated_at: SystemTime::now(),
//!     status: AssetStatus::Active,
//! };
//! context.register_data_asset(asset)?;
//!
//! // Create consent record
//! let consent = ConsentRecord {
//!     id: "user-123-consent".to_string(),
//!     data_subject_id: "user-123".to_string(),
//!     consent_type: ConsentType::Explicit,
//!     purposes: [ProcessingPurpose::ServiceProvision].into(),
//!     data_categories: [DataCategory::Personal].into(),
//!     status: ConsentStatus::Granted,
//!     granted_date: SystemTime::now(),
//!     expiry_date: None,
//!     last_updated: SystemTime::now(),
//!     withdrawal_date: None,
//!     legal_basis: LegalBasis::Consent,
//!     evidence: ConsentEvidence::default(),
//! };
//! context.create_consent(consent)?;
//!
//! // Log audit event
//! let audit_entry = AuditEntry {
//!     id: Uuid::new_v4(),
//!     timestamp: SystemTime::now(),
//!     event_type: AuditEventType::DataAccess,
//!     actor: "user-123".to_string(),
//!     resource: "user-data".to_string(),
//!     action: "read".to_string(),
//!     result: AuditResult::Success,
//!     details: HashMap::new(),
//!     risk_score: Some(0.2),
//! };
//! context.log_audit_event(audit_entry)?;
//!
//! // Get compliance metrics
//! let metrics = context.get_metrics()?;
//! println!("Compliance score: {}", metrics.compliance_score);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Re-export specialized modules
pub mod compliance_core;
pub mod compliance_governance;
pub mod compliance_regulatory;
pub mod compliance_audit;
pub mod compliance_policy;
pub mod compliance_consent;
pub mod compliance_retention;
pub mod compliance_breach;
pub mod compliance_metrics;

// Re-export all public types for easy access
pub use compliance_core::*;
pub use compliance_governance::*;
pub use compliance_regulatory::*;
pub use compliance_audit::*;
pub use compliance_policy::*;
pub use compliance_consent::*;
pub use compliance_retention::*;
pub use compliance_breach::*;
pub use compliance_metrics::*;
//! Data governance, classification, and quality management
//!
//! This module provides comprehensive data governance capabilities including
//! data asset management, classification, lineage tracking, quality monitoring,
//! and catalog management for compliance frameworks.

use std::{
    collections::{HashMap, HashSet, VecDeque, BTreeMap},
    time::{Duration, SystemTime},
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::compliance_core::RegulatoryFramework;

/// Data governance manager
#[derive(Debug)]
pub struct DataGovernanceManager {
    /// Data assets registry
    pub data_assets: HashMap<String, DataAsset>,
    /// Data classifications
    pub classifications: HashMap<String, DataClassification>,
    /// Data lineage graph
    pub lineage_graph: DataLineageGraph,
    /// Data quality monitor
    pub quality_monitor: DataQualityMonitor,
    /// Data catalog
    pub catalog: DataCatalog,
}

/// Data asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAsset {
    /// Asset ID
    pub id: String,
    /// Asset name
    pub name: String,
    /// Asset type
    pub asset_type: DataAssetType,
    /// Asset location
    pub location: String,
    /// Asset owner
    pub owner: String,
    /// Asset classification
    pub classification: DataClassification,
    /// Asset metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last updated
    pub updated_at: SystemTime,
    /// Asset status
    pub status: AssetStatus,
}

/// Data asset types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataAssetType {
    /// Database
    Database,
    /// Table
    Table,
    /// File
    File,
    /// API endpoint
    Api,
    /// Stream
    Stream,
    /// Report
    Report,
    /// Custom asset type
    Custom(String),
}

/// Asset status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetStatus {
    /// Asset is active
    Active,
    /// Asset is deprecated
    Deprecated,
    /// Asset is archived
    Archived,
    /// Asset is under review
    UnderReview,
    /// Asset is retired
    Retired,
}

/// Data classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataClassification {
    /// Classification level
    pub level: ClassificationLevel,
    /// Sensitive data types
    pub sensitive_types: HashSet<SensitiveDataType>,
    /// Regulatory requirements
    pub regulatory_requirements: HashSet<RegulatoryFramework>,
    /// Access restrictions
    pub access_restrictions: AccessRestrictions,
    /// Retention requirements
    pub retention_requirements: RetentionRequirements,
    /// Geographic restrictions
    pub geographic_restrictions: GeographicRestrictions,
}

impl Default for DataClassification {
    fn default() -> Self {
        Self {
            level: ClassificationLevel::Internal,
            sensitive_types: HashSet::new(),
            regulatory_requirements: HashSet::new(),
            access_restrictions: AccessRestrictions::default(),
            retention_requirements: RetentionRequirements::default(),
            geographic_restrictions: GeographicRestrictions::default(),
        }
    }
}

/// Classification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ClassificationLevel {
    /// Public data
    Public = 0,
    /// Internal data
    Internal = 1,
    /// Confidential data
    Confidential = 2,
    /// Restricted data
    Restricted = 3,
    /// Top secret data
    TopSecret = 4,
}

impl Display for ClassificationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassificationLevel::Public => write!(f, "PUBLIC"),
            ClassificationLevel::Internal => write!(f, "INTERNAL"),
            ClassificationLevel::Confidential => write!(f, "CONFIDENTIAL"),
            ClassificationLevel::Restricted => write!(f, "RESTRICTED"),
            ClassificationLevel::TopSecret => write!(f, "TOP_SECRET"),
        }
    }
}

/// Sensitive data types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensitiveDataType {
    /// Personally Identifiable Information
    Pii,
    /// Personal Health Information
    Phi,
    /// Financial information
    Financial,
    /// Credit card information
    CreditCard,
    /// Social security number
    Ssn,
    /// Email address
    Email,
    /// Phone number
    Phone,
    /// IP address
    IpAddress,
    /// Biometric data
    Biometric,
    /// Location data
    Location,
    /// Custom sensitive type
    Custom(String),
}

impl Display for SensitiveDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SensitiveDataType::Pii => write!(f, "PII"),
            SensitiveDataType::Phi => write!(f, "PHI"),
            SensitiveDataType::Financial => write!(f, "FINANCIAL"),
            SensitiveDataType::CreditCard => write!(f, "CREDIT_CARD"),
            SensitiveDataType::Ssn => write!(f, "SSN"),
            SensitiveDataType::Email => write!(f, "EMAIL"),
            SensitiveDataType::Phone => write!(f, "PHONE"),
            SensitiveDataType::IpAddress => write!(f, "IP_ADDRESS"),
            SensitiveDataType::Biometric => write!(f, "BIOMETRIC"),
            SensitiveDataType::Location => write!(f, "LOCATION"),
            SensitiveDataType::Custom(name) => write!(f, "CUSTOM_{}", name.to_uppercase()),
        }
    }
}

/// Access restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRestrictions {
    /// Required roles
    pub required_roles: HashSet<String>,
    /// Required permissions
    pub required_permissions: HashSet<String>,
    /// Time-based restrictions
    pub time_restrictions: Option<TimeRestrictions>,
    /// Location-based restrictions
    pub location_restrictions: Option<LocationRestrictions>,
    /// Device-based restrictions
    pub device_restrictions: Option<DeviceRestrictions>,
}

impl Default for AccessRestrictions {
    fn default() -> Self {
        Self {
            required_roles: HashSet::new(),
            required_permissions: HashSet::new(),
            time_restrictions: None,
            location_restrictions: None,
            device_restrictions: None,
        }
    }
}

/// Time restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRestrictions {
    /// Allowed hours (24-hour format)
    pub allowed_hours: Vec<(u8, u8)>,
    /// Allowed days of week (0=Sunday, 6=Saturday)
    pub allowed_days: Vec<u8>,
    /// Timezone
    pub timezone: String,
}

/// Location restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationRestrictions {
    /// Allowed countries
    pub allowed_countries: HashSet<String>,
    /// Blocked countries
    pub blocked_countries: HashSet<String>,
    /// Allowed IP ranges
    pub allowed_ip_ranges: Vec<String>,
    /// Blocked IP ranges
    pub blocked_ip_ranges: Vec<String>,
}

/// Device restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRestrictions {
    /// Allowed device types
    pub allowed_device_types: HashSet<String>,
    /// Required device compliance
    pub require_compliance: bool,
    /// Required encryption
    pub require_encryption: bool,
    /// Allowed operating systems
    pub allowed_os: HashSet<String>,
}

/// Retention requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRequirements {
    /// Minimum retention period
    pub min_retention: Duration,
    /// Maximum retention period
    pub max_retention: Option<Duration>,
    /// Legal hold enabled
    pub legal_hold: bool,
    /// Deletion method
    pub deletion_method: DeletionMethod,
    /// Verification required
    pub verification_required: bool,
}

impl Default for RetentionRequirements {
    fn default() -> Self {
        Self {
            min_retention: Duration::from_secs(365 * 24 * 60 * 60), // 1 year
            max_retention: Some(Duration::from_secs(7 * 365 * 24 * 60 * 60)), // 7 years
            legal_hold: false,
            deletion_method: DeletionMethod::Soft,
            verification_required: true,
        }
    }
}

/// Deletion methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeletionMethod {
    /// Soft delete (mark as deleted)
    Soft,
    /// Hard delete (remove completely)
    Hard,
    /// Secure delete (overwrite multiple times)
    Secure,
    /// Cryptographic deletion (destroy keys)
    Cryptographic,
    /// Custom deletion method
    Custom(String),
}

/// Geographic restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicRestrictions {
    /// Data residency requirements
    pub residency_requirements: HashSet<String>,
    /// Cross-border transfer restrictions
    pub transfer_restrictions: HashMap<String, TransferRestriction>,
    /// Sovereignty requirements
    pub sovereignty_requirements: SovereigntyRequirements,
}

impl Default for GeographicRestrictions {
    fn default() -> Self {
        Self {
            residency_requirements: HashSet::new(),
            transfer_restrictions: HashMap::new(),
            sovereignty_requirements: SovereigntyRequirements::default(),
        }
    }
}

/// Transfer restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferRestriction {
    /// Allowed destination countries
    pub allowed_destinations: HashSet<String>,
    /// Blocked destination countries
    pub blocked_destinations: HashSet<String>,
    /// Required adequacy decision
    pub require_adequacy: bool,
    /// Required safeguards
    pub required_safeguards: HashSet<String>,
}

/// Sovereignty requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereigntyRequirements {
    /// Data must remain in country
    pub data_localization: bool,
    /// Government access restrictions
    pub government_access_restrictions: bool,
    /// Local jurisdiction required
    pub local_jurisdiction: bool,
}

impl Default for SovereigntyRequirements {
    fn default() -> Self {
        Self {
            data_localization: false,
            government_access_restrictions: false,
            local_jurisdiction: false,
        }
    }
}

/// Data lineage graph
#[derive(Debug, Clone)]
pub struct DataLineageGraph {
    /// Graph nodes (data assets)
    pub nodes: HashMap<String, LineageNode>,
    /// Graph edges (data flows)
    pub edges: Vec<LineageEdge>,
}

/// Lineage node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    /// Node ID
    pub id: String,
    /// Asset reference
    pub asset_id: String,
    /// Node type
    pub node_type: LineageNodeType,
    /// Node metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Lineage node types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineageNodeType {
    /// Source node
    Source,
    /// Transform node
    Transform,
    /// Sink node
    Sink,
    /// Storage node
    Storage,
}

/// Lineage edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge type
    pub edge_type: LineageEdgeType,
    /// Transform description
    pub transform: Option<String>,
    /// Edge metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Lineage edge types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineageEdgeType {
    /// Data read operation
    Read,
    /// Data write operation
    Write,
    /// Data transformation
    Transform,
    /// Data derivation
    Derive,
    /// Data copy
    Copy,
}

/// Data quality monitor
#[derive(Debug, Clone)]
pub struct DataQualityMonitor {
    /// Quality rules
    pub quality_rules: Vec<DataQualityRule>,
    /// Quality metrics
    pub quality_metrics: HashMap<String, QualityMetric>,
    /// Quality reports
    pub quality_reports: VecDeque<QualityReport>,
}

/// Data quality rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule type
    pub rule_type: QualityRuleType,
    /// Target asset
    pub target_asset: String,
    /// Rule condition
    pub condition: String,
    /// Threshold
    pub threshold: f64,
    /// Severity
    pub severity: QualitySeverity,
}

/// Quality rule types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityRuleType {
    /// Completeness check
    Completeness,
    /// Accuracy check
    Accuracy,
    /// Consistency check
    Consistency,
    /// Validity check
    Validity,
    /// Uniqueness check
    Uniqueness,
    /// Timeliness check
    Timeliness,
    /// Custom rule
    Custom(String),
}

/// Quality severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QualitySeverity {
    /// Low severity
    Low = 1,
    /// Medium severity
    Medium = 2,
    /// High severity
    High = 3,
    /// Critical severity
    Critical = 4,
}

/// Quality metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetric {
    /// Metric name
    pub name: String,
    /// Current value
    pub current_value: f64,
    /// Target value
    pub target_value: f64,
    /// Metric trend
    pub trend: QualityTrend,
    /// Last updated
    pub last_updated: SystemTime,
}

/// Quality trend
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTrend {
    /// Improving quality
    Improving,
    /// Stable quality
    Stable,
    /// Declining quality
    Declining,
    /// Unknown trend
    Unknown,
}

/// Quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Report ID
    pub id: Uuid,
    /// Asset ID
    pub asset_id: String,
    /// Report timestamp
    pub timestamp: SystemTime,
    /// Quality score
    pub quality_score: f64,
    /// Rule violations
    pub violations: Vec<QualityViolation>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Quality violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityViolation {
    /// Rule ID
    pub rule_id: String,
    /// Violation description
    pub description: String,
    /// Affected records
    pub affected_records: usize,
    /// Severity
    pub severity: QualitySeverity,
}

/// Data catalog
#[derive(Debug, Clone)]
pub struct DataCatalog {
    /// Catalog entries
    pub entries: HashMap<String, CatalogEntry>,
    /// Schema registry
    pub schema_registry: SchemaRegistry,
    /// Metadata store
    pub metadata_store: MetadataStore,
}

/// Catalog entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Entry ID
    pub id: String,
    /// Asset reference
    pub asset_id: String,
    /// Entry metadata
    pub metadata: CatalogMetadata,
    /// Schema reference
    pub schema_id: Option<String>,
    /// Tags
    pub tags: HashSet<String>,
    /// Annotations
    pub annotations: HashMap<String, String>,
}

/// Catalog metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogMetadata {
    /// Title
    pub title: String,
    /// Description
    pub description: Option<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// Business glossary terms
    pub business_terms: Vec<String>,
    /// Technical metadata
    pub technical_metadata: HashMap<String, serde_json::Value>,
}

/// Schema registry
#[derive(Debug, Clone)]
pub struct SchemaRegistry {
    /// Registered schemas
    pub schemas: HashMap<String, DataSchema>,
    /// Schema evolution history
    pub evolution_history: HashMap<String, Vec<SchemaVersion>>,
}

/// Data schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    /// Schema ID
    pub id: String,
    /// Schema name
    pub name: String,
    /// Schema version
    pub version: String,
    /// Schema definition
    pub definition: serde_json::Value,
    /// Schema format
    pub format: SchemaFormat,
    /// Compatibility mode
    pub compatibility: CompatibilityMode,
}

/// Schema formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaFormat {
    /// Avro schema
    Avro,
    /// JSON schema
    JsonSchema,
    /// Protobuf schema
    Protobuf,
    /// XML schema
    XmlSchema,
    /// Custom schema format
    Custom(String),
}

/// Compatibility modes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityMode {
    /// Backward compatibility
    Backward,
    /// Forward compatibility
    Forward,
    /// Full compatibility
    Full,
    /// No compatibility checking
    None,
}

/// Schema version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Version number
    pub version: String,
    /// Schema definition
    pub definition: serde_json::Value,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Changes from previous version
    pub changes: Vec<SchemaChange>,
}

/// Schema change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaChange {
    /// Change type
    pub change_type: SchemaChangeType,
    /// Change description
    pub description: String,
    /// Breaking change
    pub breaking: bool,
}

/// Schema change types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaChangeType {
    /// Field added
    FieldAdded,
    /// Field removed
    FieldRemoved,
    /// Field modified
    FieldModified,
    /// Type changed
    TypeChanged,
    /// Constraint added
    ConstraintAdded,
    /// Constraint removed
    ConstraintRemoved,
}

/// Metadata store
#[derive(Debug, Clone)]
pub struct MetadataStore {
    /// Stored metadata
    pub metadata: HashMap<String, AssetMetadata>,
    /// Metadata lineage
    pub lineage: HashMap<String, MetadataLineage>,
}

/// Asset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMetadata {
    /// Asset ID
    pub asset_id: String,
    /// Business metadata
    pub business: BusinessMetadata,
    /// Technical metadata
    pub technical: TechnicalMetadata,
    /// Operational metadata
    pub operational: OperationalMetadata,
    /// Custom metadata
    pub custom: HashMap<String, serde_json::Value>,
}

/// Business metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessMetadata {
    /// Business owner
    pub owner: String,
    /// Business purpose
    pub purpose: String,
    /// Business rules
    pub rules: Vec<String>,
    /// Stakeholders
    pub stakeholders: Vec<String>,
    /// Business criticality
    pub criticality: BusinessCriticality,
}

/// Business criticality levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusinessCriticality {
    /// Low criticality
    Low,
    /// Medium criticality
    Medium,
    /// High criticality
    High,
    /// Mission critical
    MissionCritical,
}

/// Technical metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalMetadata {
    /// Data type
    pub data_type: String,
    /// Format
    pub format: String,
    /// Size
    pub size: Option<usize>,
    /// Encoding
    pub encoding: Option<String>,
    /// Compression
    pub compression: Option<String>,
    /// Schema reference
    pub schema_reference: Option<String>,
}

/// Operational metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalMetadata {
    /// Creation time
    pub created_at: SystemTime,
    /// Last modified
    pub modified_at: SystemTime,
    /// Last accessed
    pub accessed_at: Option<SystemTime>,
    /// Access frequency
    pub access_frequency: AccessFrequency,
    /// Performance metrics
    pub performance: PerformanceMetrics,
}

/// Access frequency
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessFrequency {
    /// Never accessed
    Never,
    /// Rarely accessed
    Rare,
    /// Occasionally accessed
    Occasional,
    /// Frequently accessed
    Frequent,
    /// Constantly accessed
    Constant,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average query time
    pub avg_query_time: Duration,
    /// Throughput (queries per second)
    pub throughput: f64,
    /// Error rate
    pub error_rate: f64,
    /// Availability
    pub availability: f64,
}

/// Metadata lineage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataLineage {
    /// Source metadata
    pub sources: Vec<String>,
    /// Derived metadata
    pub derived: Vec<String>,
    /// Transformation logic
    pub transformations: Vec<MetadataTransformation>,
}

/// Metadata transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataTransformation {
    /// Transformation ID
    pub id: String,
    /// Transformation type
    pub transformation_type: String,
    /// Input metadata
    pub inputs: Vec<String>,
    /// Output metadata
    pub outputs: Vec<String>,
    /// Transformation logic
    pub logic: String,
}

impl DataGovernanceManager {
    /// Create a new data governance manager
    pub fn new() -> Self {
        Self {
            data_assets: HashMap::new(),
            classifications: HashMap::new(),
            lineage_graph: DataLineageGraph {
                nodes: HashMap::new(),
                edges: Vec::new(),
            },
            quality_monitor: DataQualityMonitor {
                quality_rules: Vec::new(),
                quality_metrics: HashMap::new(),
                quality_reports: VecDeque::new(),
            },
            catalog: DataCatalog {
                entries: HashMap::new(),
                schema_registry: SchemaRegistry {
                    schemas: HashMap::new(),
                    evolution_history: HashMap::new(),
                },
                metadata_store: MetadataStore {
                    metadata: HashMap::new(),
                    lineage: HashMap::new(),
                },
            },
        }
    }

    /// Register a new data asset
    pub fn register_asset(&mut self, asset: DataAsset) {
        self.data_assets.insert(asset.id.clone(), asset);
    }

    /// Get data asset by ID
    pub fn get_asset(&self, asset_id: &str) -> Option<&DataAsset> {
        self.data_assets.get(asset_id)
    }

    /// Update asset classification
    pub fn update_classification(&mut self, asset_id: &str, classification: DataClassification) {
        self.classifications.insert(asset_id.to_string(), classification);
    }

    /// Add lineage edge
    pub fn add_lineage_edge(&mut self, edge: LineageEdge) {
        self.lineage_graph.edges.push(edge);
    }

    /// Add quality rule
    pub fn add_quality_rule(&mut self, rule: DataQualityRule) {
        self.quality_monitor.quality_rules.push(rule);
    }

    /// Generate quality report
    pub fn generate_quality_report(&mut self, asset_id: &str) -> QualityReport {
        QualityReport {
            id: Uuid::new_v4(),
            asset_id: asset_id.to_string(),
            timestamp: SystemTime::now(),
            quality_score: 0.8, // Placeholder
            violations: Vec::new(),
            recommendations: Vec::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_levels() {
        assert!(ClassificationLevel::TopSecret > ClassificationLevel::Restricted);
        assert!(ClassificationLevel::Restricted > ClassificationLevel::Confidential);
        assert_eq!(ClassificationLevel::Public.to_string(), "PUBLIC");
    }

    #[test]
    fn test_sensitive_data_types() {
        assert_eq!(SensitiveDataType::Pii.to_string(), "PII");
        assert_eq!(SensitiveDataType::Phi.to_string(), "PHI");
        assert_eq!(SensitiveDataType::Custom("TEST".to_string()).to_string(), "CUSTOM_TEST");
    }

    #[test]
    fn test_data_governance_manager() {
        let mut manager = DataGovernanceManager::new();
        assert_eq!(manager.data_assets.len(), 0);

        let asset = DataAsset {
            id: "test-asset".to_string(),
            name: "Test Asset".to_string(),
            asset_type: DataAssetType::Database,
            location: "test-db".to_string(),
            owner: "test-owner".to_string(),
            classification: DataClassification::default(),
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            status: AssetStatus::Active,
        };

        manager.register_asset(asset);
        assert_eq!(manager.data_assets.len(), 1);
        assert!(manager.get_asset("test-asset").is_some());
    }

    #[test]
    fn test_quality_severity_ordering() {
        assert!(QualitySeverity::Critical > QualitySeverity::High);
        assert!(QualitySeverity::High > QualitySeverity::Medium);
        assert!(QualitySeverity::Medium > QualitySeverity::Low);
    }

    #[test]
    fn test_deletion_methods() {
        assert_eq!(DeletionMethod::Soft, DeletionMethod::Soft);
        assert_ne!(DeletionMethod::Soft, DeletionMethod::Hard);
    }
}
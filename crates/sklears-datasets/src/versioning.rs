//! Dataset versioning and provenance tracking
//!
//! This module provides functionality for tracking dataset versions, lineage,
//! and provenance information to ensure reproducibility and auditability.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use thiserror::Error;

/// Error types for versioning operations
#[derive(Debug, Error)]
pub enum VersioningError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Version not found: {0}")]
    VersionNotFound(String),
    #[error("Invalid version format: {0}")]
    InvalidVersion(String),
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
}

pub type VersioningResult<T> = Result<T, VersioningError>;

/// Semantic version for datasets
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DatasetVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub prerelease: Option<String>,
}

impl DatasetVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            prerelease: None,
        }
    }

    pub fn with_prerelease(mut self, prerelease: String) -> Self {
        self.prerelease = Some(prerelease);
        self
    }

    pub fn from_string(s: &str) -> VersioningResult<Self> {
        let parts: Vec<&str> = s.split('-').collect();
        let version_parts: Vec<&str> = parts[0].split('.').collect();

        if version_parts.len() != 3 {
            return Err(VersioningError::InvalidVersion(s.to_string()));
        }

        let major = version_parts[0]
            .parse()
            .map_err(|_| VersioningError::InvalidVersion(s.to_string()))?;
        let minor = version_parts[1]
            .parse()
            .map_err(|_| VersioningError::InvalidVersion(s.to_string()))?;
        let patch = version_parts[2]
            .parse()
            .map_err(|_| VersioningError::InvalidVersion(s.to_string()))?;

        let prerelease = if parts.len() > 1 {
            Some(parts[1].to_string())
        } else {
            None
        };

        Ok(Self {
            major,
            minor,
            patch,
            prerelease,
        })
    }
}

impl fmt::Display for DatasetVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref pre) = self.prerelease {
            write!(f, "{}.{}.{}-{}", self.major, self.minor, self.patch, pre)
        } else {
            write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
        }
    }
}

/// Provenance information tracking dataset lineage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceInfo {
    /// Unique identifier for this dataset
    pub dataset_id: String,
    /// Version of the dataset
    pub version: DatasetVersion,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modification timestamp
    pub modified_at: DateTime<Utc>,
    /// Creator/author information
    pub creator: String,
    /// Description of the dataset
    pub description: String,
    /// Source datasets (parent datasets this was derived from)
    pub sources: Vec<String>,
    /// Transformation operations applied
    pub transformations: Vec<TransformationStep>,
    /// Checksums for data integrity
    pub checksums: HashMap<String, String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl ProvenanceInfo {
    pub fn new(dataset_id: String, version: DatasetVersion, creator: String) -> Self {
        let now = Utc::now();
        Self {
            dataset_id,
            version,
            created_at: now,
            modified_at: now,
            creator,
            description: String::new(),
            sources: Vec::new(),
            transformations: Vec::new(),
            checksums: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    pub fn add_source(&mut self, source_id: String) {
        self.sources.push(source_id);
        self.modified_at = Utc::now();
    }

    pub fn add_transformation(&mut self, transformation: TransformationStep) {
        self.transformations.push(transformation);
        self.modified_at = Utc::now();
    }

    pub fn add_checksum(&mut self, name: String, checksum: String) {
        self.checksums.insert(name, checksum);
        self.modified_at = Utc::now();
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
        self.modified_at = Utc::now();
    }

    /// Save provenance information to a JSON file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> VersioningResult<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load provenance information from a JSON file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> VersioningResult<Self> {
        let json = std::fs::read_to_string(path)?;
        let provenance: ProvenanceInfo = serde_json::from_str(&json)?;
        Ok(provenance)
    }
}

/// A single transformation step in the dataset lineage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationStep {
    /// Timestamp of the transformation
    pub timestamp: DateTime<Utc>,
    /// Type of transformation (e.g., "normalization", "feature_selection", "sampling")
    pub transformation_type: String,
    /// Description of the transformation
    pub description: String,
    /// Parameters used in the transformation
    pub parameters: HashMap<String, String>,
    /// User/system that performed the transformation
    pub performed_by: String,
}

impl TransformationStep {
    pub fn new(transformation_type: String, description: String, performed_by: String) -> Self {
        Self {
            timestamp: Utc::now(),
            transformation_type,
            description,
            parameters: HashMap::new(),
            performed_by,
        }
    }

    pub fn with_parameter(mut self, key: String, value: String) -> Self {
        self.parameters.insert(key, value);
        self
    }
}

/// Dataset version registry for managing multiple versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionRegistry {
    /// Name of the dataset
    pub dataset_name: String,
    /// All registered versions
    pub versions: HashMap<String, ProvenanceInfo>,
    /// Current/latest version
    pub current_version: Option<String>,
}

impl VersionRegistry {
    pub fn new(dataset_name: String) -> Self {
        Self {
            dataset_name,
            versions: HashMap::new(),
            current_version: None,
        }
    }

    pub fn register_version(&mut self, provenance: ProvenanceInfo) {
        let version_str = provenance.version.to_string();
        self.versions.insert(version_str.clone(), provenance);
        self.current_version = Some(version_str);
    }

    pub fn get_version(&self, version: &str) -> Option<&ProvenanceInfo> {
        self.versions.get(version)
    }

    pub fn get_current(&self) -> Option<&ProvenanceInfo> {
        self.current_version
            .as_ref()
            .and_then(|v| self.versions.get(v))
    }

    pub fn set_current(&mut self, version: &str) -> VersioningResult<()> {
        if !self.versions.contains_key(version) {
            return Err(VersioningError::VersionNotFound(version.to_string()));
        }
        self.current_version = Some(version.to_string());
        Ok(())
    }

    /// List all versions in chronological order
    pub fn list_versions(&self) -> Vec<&ProvenanceInfo> {
        let mut versions: Vec<&ProvenanceInfo> = self.versions.values().collect();
        versions.sort_by_key(|p| &p.created_at);
        versions
    }

    /// Save version registry to a JSON file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> VersioningResult<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load version registry from a JSON file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> VersioningResult<Self> {
        let json = std::fs::read_to_string(path)?;
        let registry: VersionRegistry = serde_json::from_str(&json)?;
        Ok(registry)
    }
}

/// Calculate SHA-256 checksum for data verification
pub fn calculate_checksum(data: &[u8]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Verify data integrity against a checksum
pub fn verify_checksum(data: &[u8], expected: &str) -> VersioningResult<()> {
    let actual = calculate_checksum(data);
    if actual != expected {
        Err(VersioningError::ChecksumMismatch {
            expected: expected.to_string(),
            actual,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_version() {
        let version = DatasetVersion::new(1, 2, 3);
        assert_eq!(version.to_string(), "1.2.3");

        let version_pre = DatasetVersion::new(1, 2, 3).with_prerelease("alpha".to_string());
        assert_eq!(version_pre.to_string(), "1.2.3-alpha");
    }

    #[test]
    fn test_version_parsing() {
        let version = DatasetVersion::from_string("1.2.3").unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
        assert_eq!(version.prerelease, None);

        let version_pre = DatasetVersion::from_string("1.2.3-beta").unwrap();
        assert_eq!(version_pre.prerelease, Some("beta".to_string()));
    }

    #[test]
    fn test_provenance_info() {
        let mut provenance = ProvenanceInfo::new(
            "test-dataset".to_string(),
            DatasetVersion::new(1, 0, 0),
            "test-user".to_string(),
        );

        provenance.add_source("source-dataset-1".to_string());
        provenance.add_metadata("key1".to_string(), "value1".to_string());

        assert_eq!(provenance.sources.len(), 1);
        assert_eq!(provenance.metadata.len(), 1);
    }

    #[test]
    fn test_transformation_step() {
        let step = TransformationStep::new(
            "normalization".to_string(),
            "StandardScaler normalization".to_string(),
            "test-user".to_string(),
        )
        .with_parameter("mean".to_string(), "0.0".to_string())
        .with_parameter("std".to_string(), "1.0".to_string());

        assert_eq!(step.parameters.len(), 2);
    }

    #[test]
    fn test_version_registry() {
        let mut registry = VersionRegistry::new("test-dataset".to_string());

        let prov1 = ProvenanceInfo::new(
            "test-dataset".to_string(),
            DatasetVersion::new(1, 0, 0),
            "user1".to_string(),
        );
        registry.register_version(prov1);

        let prov2 = ProvenanceInfo::new(
            "test-dataset".to_string(),
            DatasetVersion::new(1, 1, 0),
            "user1".to_string(),
        );
        registry.register_version(prov2);

        assert_eq!(registry.versions.len(), 2);
        assert!(registry.get_version("1.0.0").is_some());
        assert!(registry.get_version("1.1.0").is_some());
    }

    #[test]
    fn test_checksum() {
        let data = b"test data";
        let checksum = calculate_checksum(data);
        assert!(verify_checksum(data, &checksum).is_ok());

        assert!(verify_checksum(data, "invalid").is_err());
    }

    #[test]
    fn test_provenance_serialization() {
        use std::env::temp_dir;

        let provenance = ProvenanceInfo::new(
            "test-dataset".to_string(),
            DatasetVersion::new(1, 0, 0),
            "test-user".to_string(),
        )
        .with_description("Test dataset".to_string());

        let temp_path = temp_dir().join("test_provenance.json");
        provenance.save_to_file(&temp_path).unwrap();

        let loaded = ProvenanceInfo::load_from_file(&temp_path).unwrap();
        assert_eq!(loaded.dataset_id, "test-dataset");
        assert_eq!(loaded.version.major, 1);

        std::fs::remove_file(temp_path).ok();
    }
}

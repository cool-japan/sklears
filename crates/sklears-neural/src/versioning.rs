use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Result type for versioning operations
pub type VersioningResult<T> = Result<T, SklearsError>;

/// Model version identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ModelVersion {
    /// Major version (breaking changes)
    pub major: u32,
    /// Minor version (new features, backward compatible)
    pub minor: u32,
    /// Patch version (bug fixes, backward compatible)
    pub patch: u32,
    /// Optional pre-release identifier
    pub pre_release: Option<String>,
    /// Build metadata
    pub build: Option<String>,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
            build: None,
        }
    }

    /// Create a version with pre-release identifier
    pub fn with_pre_release(mut self, pre_release: String) -> Self {
        self.pre_release = Some(pre_release);
        self
    }

    /// Create a version with build metadata
    pub fn with_build(mut self, build: String) -> Self {
        self.build = Some(build);
        self
    }

    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &ModelVersion) -> bool {
        // Same major version means compatible
        self.major == other.major
    }

    /// Check if this version is newer than another
    pub fn is_newer_than(&self, other: &ModelVersion) -> bool {
        if self.major != other.major {
            return self.major > other.major;
        }
        if self.minor != other.minor {
            return self.minor > other.minor;
        }
        self.patch > other.patch
    }

    /// Check if this version requires migration from another
    pub fn requires_migration_from(&self, other: &ModelVersion) -> bool {
        self.major != other.major || (self.major == other.major && self.minor > other.minor)
    }
}

impl std::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(ref pre) = self.pre_release {
            write!(f, "-{}", pre)?;
        }
        if let Some(ref build) = self.build {
            write!(f, "+{}", build)?;
        }
        Ok(())
    }
}

impl std::str::FromStr for ModelVersion {
    type Err = SklearsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() < 3 {
            return Err(SklearsError::InvalidParameter {
                name: "version".to_string(),
                reason: "Version must have at least major.minor.patch".to_string(),
            });
        }

        let major = parts[0]
            .parse()
            .map_err(|_| SklearsError::InvalidParameter {
                name: "major_version".to_string(),
                reason: "Major version must be a number".to_string(),
            })?;

        let minor = parts[1]
            .parse()
            .map_err(|_| SklearsError::InvalidParameter {
                name: "minor_version".to_string(),
                reason: "Minor version must be a number".to_string(),
            })?;

        let patch_part = parts[2];
        let (patch_str, pre_release, build) = if let Some(build_pos) = patch_part.find('+') {
            let (patch_pre, build_str) = patch_part.split_at(build_pos);
            let build_str = &build_str[1..]; // Remove '+'

            if let Some(pre_pos) = patch_pre.find('-') {
                let (patch_str, pre_str) = patch_pre.split_at(pre_pos);
                let pre_str = &pre_str[1..]; // Remove '-'
                (
                    patch_str,
                    Some(pre_str.to_string()),
                    Some(build_str.to_string()),
                )
            } else {
                (patch_pre, None, Some(build_str.to_string()))
            }
        } else if let Some(pre_pos) = patch_part.find('-') {
            let (patch_str, pre_str) = patch_part.split_at(pre_pos);
            let pre_str = &pre_str[1..]; // Remove '-'
            (patch_str, Some(pre_str.to_string()), None)
        } else {
            (patch_part, None, None)
        };

        let patch = patch_str
            .parse()
            .map_err(|_| SklearsError::InvalidParameter {
                name: "patch_version".to_string(),
                reason: "Patch version must be a number".to_string(),
            })?;

        Ok(ModelVersion {
            major,
            minor,
            patch,
            pre_release,
            build,
        })
    }
}

/// Migration strategy for model updates
#[derive(Debug, Clone)]
pub enum MigrationStrategy {
    /// No migration needed
    None,
    /// Automatic migration with parameter mapping
    Automatic {
        parameter_mapping: HashMap<String, String>,
        default_values: HashMap<String, f64>,
    },
    /// Custom migration function
    Custom {
        migration_name: String,
        description: String,
    },
    /// Manual migration required
    Manual { instructions: String },
}

/// Model metadata for versioning
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ModelMetadata {
    /// Model version
    pub version: ModelVersion,
    /// Timestamp of creation
    pub created_at: u64,
    /// Model architecture description
    pub architecture: String,
    /// Training configuration hash
    pub config_hash: Option<String>,
    /// Performance metrics at time of saving
    pub performance_metrics: HashMap<String, f64>,
    /// Model size in parameters
    pub parameter_count: usize,
    /// Framework version used
    pub framework_version: String,
    /// Custom tags
    pub tags: Vec<String>,
    /// Model description
    pub description: Option<String>,
}

impl ModelMetadata {
    pub fn new(version: ModelVersion, architecture: String) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            version,
            created_at,
            architecture,
            config_hash: None,
            performance_metrics: HashMap::new(),
            parameter_count: 0,
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            tags: Vec::new(),
            description: None,
        }
    }

    pub fn with_config_hash(mut self, hash: String) -> Self {
        self.config_hash = Some(hash);
        self
    }

    pub fn with_performance_metrics(mut self, metrics: HashMap<String, f64>) -> Self {
        self.performance_metrics = metrics;
        self
    }

    pub fn with_parameter_count(mut self, count: usize) -> Self {
        self.parameter_count = count;
        self
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn add_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }
}

/// Version compatibility checker
#[derive(Debug, Clone)]
pub struct CompatibilityChecker {
    /// Migration strategies for different version transitions
    migration_strategies: HashMap<(ModelVersion, ModelVersion), MigrationStrategy>,
    /// Deprecated features by version
    deprecated_features: HashMap<ModelVersion, Vec<String>>,
    /// Breaking changes by version
    breaking_changes: HashMap<ModelVersion, Vec<String>>,
}

impl CompatibilityChecker {
    pub fn new() -> Self {
        Self {
            migration_strategies: HashMap::new(),
            deprecated_features: HashMap::new(),
            breaking_changes: HashMap::new(),
        }
    }

    /// Register a migration strategy for version transition
    pub fn register_migration(
        &mut self,
        from: ModelVersion,
        to: ModelVersion,
        strategy: MigrationStrategy,
    ) {
        self.migration_strategies.insert((from, to), strategy);
    }

    /// Register deprecated features for a version
    pub fn register_deprecated_features(&mut self, version: ModelVersion, features: Vec<String>) {
        self.deprecated_features.insert(version, features);
    }

    /// Register breaking changes for a version
    pub fn register_breaking_changes(&mut self, version: ModelVersion, changes: Vec<String>) {
        self.breaking_changes.insert(version, changes);
    }

    /// Check compatibility between two versions
    pub fn check_compatibility(
        &self,
        from: &ModelVersion,
        to: &ModelVersion,
    ) -> VersioningResult<CompatibilityReport> {
        let mut report = CompatibilityReport {
            compatible: from.is_compatible_with(to),
            migration_required: to.requires_migration_from(from),
            migration_strategy: MigrationStrategy::None,
            warnings: Vec::new(),
            errors: Vec::new(),
        };

        // Check for breaking changes
        if to.major > from.major {
            if let Some(changes) = self.breaking_changes.get(to) {
                report.errors.extend(changes.iter().cloned());
                report.compatible = false;
            }
        }

        // Check for deprecated features
        if let Some(deprecated) = self.deprecated_features.get(from) {
            for feature in deprecated {
                report
                    .warnings
                    .push(format!("Feature '{}' is deprecated", feature));
            }
        }

        // Find migration strategy
        if let Some(strategy) = self.migration_strategies.get(&(from.clone(), to.clone())) {
            report.migration_strategy = strategy.clone();
        } else if report.migration_required {
            report.migration_strategy = MigrationStrategy::Manual {
                instructions: "Manual migration required - no automatic strategy available"
                    .to_string(),
            };
        }

        Ok(report)
    }
}

impl Default for CompatibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility report between model versions
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    /// Whether versions are compatible
    pub compatible: bool,
    /// Whether migration is required
    pub migration_required: bool,
    /// Migration strategy to use
    pub migration_strategy: MigrationStrategy,
    /// Compatibility warnings
    pub warnings: Vec<String>,
    /// Compatibility errors
    pub errors: Vec<String>,
}

/// Model version manager
#[derive(Debug, Clone)]
pub struct ModelVersionManager {
    compatibility_checker: CompatibilityChecker,
    version_history: Vec<ModelMetadata>,
    current_version: Option<ModelVersion>,
}

impl ModelVersionManager {
    pub fn new() -> Self {
        let mut compatibility_checker = CompatibilityChecker::new();

        // Register some default migration strategies
        Self::register_default_migrations(&mut compatibility_checker);

        Self {
            compatibility_checker,
            version_history: Vec::new(),
            current_version: None,
        }
    }

    /// Register default migration strategies for common version transitions
    fn register_default_migrations(checker: &mut CompatibilityChecker) {
        // Example: Migration from v1.0.0 to v1.1.0 (minor version bump)
        let from_v1_0 = ModelVersion::new(1, 0, 0);
        let to_v1_1 = ModelVersion::new(1, 1, 0);

        checker.register_migration(
            from_v1_0.clone(),
            to_v1_1.clone(),
            MigrationStrategy::Automatic {
                parameter_mapping: HashMap::new(),
                default_values: HashMap::new(),
            },
        );

        // Example: Major version migration (breaking changes)
        let from_v1_x = ModelVersion::new(1, 9, 0);
        let to_v2_0 = ModelVersion::new(2, 0, 0);

        checker.register_migration(
            from_v1_x,
            to_v2_0.clone(),
            MigrationStrategy::Manual {
                instructions: "Major version upgrade requires manual review of model architecture"
                    .to_string(),
            },
        );

        // Register deprecated features
        checker.register_deprecated_features(
            ModelVersion::new(1, 5, 0),
            vec![
                "old_activation_function".to_string(),
                "legacy_optimizer".to_string(),
            ],
        );

        // Register breaking changes
        checker.register_breaking_changes(
            to_v2_0,
            vec![
                "Changed default activation from sigmoid to relu".to_string(),
                "Removed support for legacy file format".to_string(),
            ],
        );
    }

    /// Set the current model version
    pub fn set_current_version(&mut self, version: ModelVersion, metadata: ModelMetadata) {
        self.current_version = Some(version);
        self.version_history.push(metadata);
    }

    /// Get the current model version
    pub fn get_current_version(&self) -> Option<&ModelVersion> {
        self.current_version.as_ref()
    }

    /// Get version history
    pub fn get_version_history(&self) -> &[ModelMetadata] {
        &self.version_history
    }

    /// Check if a version can be loaded
    pub fn can_load_version(
        &self,
        version: &ModelVersion,
    ) -> VersioningResult<CompatibilityReport> {
        if let Some(current) = &self.current_version {
            self.compatibility_checker
                .check_compatibility(version, current)
        } else {
            Ok(CompatibilityReport {
                compatible: true,
                migration_required: false,
                migration_strategy: MigrationStrategy::None,
                warnings: Vec::new(),
                errors: Vec::new(),
            })
        }
    }

    /// Apply automatic migration between versions
    pub fn apply_migration<T: FloatBounds>(
        &self,
        from_version: &ModelVersion,
        to_version: &ModelVersion,
        parameters: &mut HashMap<String, Array2<T>>,
        biases: &mut HashMap<String, Array1<T>>,
    ) -> VersioningResult<()> {
        let report = self
            .compatibility_checker
            .check_compatibility(from_version, to_version)?;

        if !report.compatible {
            return Err(SklearsError::InvalidParameter {
                name: "version_compatibility".to_string(),
                reason: format!(
                    "Versions {} and {} are not compatible",
                    from_version, to_version
                ),
            });
        }

        match report.migration_strategy {
            MigrationStrategy::None => {
                // No migration needed
            }
            MigrationStrategy::Automatic {
                parameter_mapping,
                default_values,
            } => {
                // Apply parameter mapping
                for (old_name, new_name) in parameter_mapping {
                    if let Some(param) = parameters.remove(&old_name) {
                        parameters.insert(new_name.clone(), param);
                    }
                    if let Some(bias) = biases.remove(&old_name) {
                        biases.insert(new_name, bias);
                    }
                }

                // Add default values for new parameters
                for (name, default_val) in default_values {
                    if !parameters.contains_key(&name) {
                        // Create default parameter with appropriate shape
                        // This is a simplified example - real implementation would need shape info
                        let default_param =
                            Array2::from_elem((1, 1), T::from(default_val).unwrap());
                        parameters.insert(name.clone(), default_param);
                    }
                }
            }
            MigrationStrategy::Custom { migration_name, .. } => {
                return Err(SklearsError::InvalidParameter {
                    name: "migration".to_string(),
                    reason: format!("Custom migration '{}' not implemented", migration_name),
                });
            }
            MigrationStrategy::Manual { instructions } => {
                return Err(SklearsError::InvalidParameter {
                    name: "migration".to_string(),
                    reason: format!("Manual migration required: {}", instructions),
                });
            }
        }

        Ok(())
    }

    /// Get migration path between versions
    pub fn get_migration_path(&self, from: &ModelVersion, to: &ModelVersion) -> Vec<ModelVersion> {
        // For simplicity, this returns direct path
        // A more sophisticated implementation would find optimal migration path
        if from.is_compatible_with(to) {
            vec![from.clone(), to.clone()]
        } else {
            // Would need to implement pathfinding through compatible versions
            vec![]
        }
    }

    /// Validate model metadata
    pub fn validate_metadata(&self, metadata: &ModelMetadata) -> VersioningResult<()> {
        if metadata.architecture.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "architecture".to_string(),
                reason: "Architecture description cannot be empty".to_string(),
            });
        }

        if metadata.parameter_count == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "parameter_count".to_string(),
                reason: "Parameter count must be greater than 0".to_string(),
            });
        }

        Ok(())
    }
}

impl Default for ModelVersionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for versioned models
pub trait VersionedModel {
    /// Get model version
    fn get_version(&self) -> &ModelVersion;

    /// Get model metadata
    fn get_metadata(&self) -> &ModelMetadata;

    /// Check if model is compatible with a version
    fn is_compatible_with(&self, version: &ModelVersion) -> bool;

    /// Migrate model to a new version
    fn migrate_to_version(&mut self, version: ModelVersion) -> VersioningResult<()>;
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_version_creation() {
        let version = ModelVersion::new(1, 2, 3);
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
        assert_eq!(version.to_string(), "1.2.3");
    }

    #[test]
    fn test_version_parsing() {
        let version: ModelVersion = "1.2.3".parse().unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);

        let version_with_pre: ModelVersion = "1.2.3-alpha".parse().unwrap();
        assert_eq!(version_with_pre.pre_release, Some("alpha".to_string()));

        let version_with_build: ModelVersion = "1.2.3+20240101".parse().unwrap();
        assert_eq!(version_with_build.build, Some("20240101".to_string()));
    }

    #[test]
    fn test_version_compatibility() {
        let v1_0_0 = ModelVersion::new(1, 0, 0);
        let v1_1_0 = ModelVersion::new(1, 1, 0);
        let v2_0_0 = ModelVersion::new(2, 0, 0);

        assert!(v1_0_0.is_compatible_with(&v1_1_0));
        assert!(!v1_0_0.is_compatible_with(&v2_0_0));
        assert!(v1_1_0.is_newer_than(&v1_0_0));
        assert!(v1_1_0.requires_migration_from(&v1_0_0));
    }

    #[test]
    fn test_model_metadata() {
        let version = ModelVersion::new(1, 0, 0);
        let metadata = ModelMetadata::new(version.clone(), "MLP".to_string())
            .with_parameter_count(1000)
            .with_description("Test model".to_string())
            .add_tag("test".to_string());

        assert_eq!(metadata.version, version);
        assert_eq!(metadata.architecture, "MLP");
        assert_eq!(metadata.parameter_count, 1000);
        assert_eq!(metadata.description, Some("Test model".to_string()));
        assert!(metadata.tags.contains(&"test".to_string()));
    }

    #[test]
    fn test_compatibility_checker() {
        let mut checker = CompatibilityChecker::new();
        let v1_0 = ModelVersion::new(1, 0, 0);
        let v1_1 = ModelVersion::new(1, 1, 0);

        checker.register_migration(
            v1_0.clone(),
            v1_1.clone(),
            MigrationStrategy::Automatic {
                parameter_mapping: HashMap::new(),
                default_values: HashMap::new(),
            },
        );

        let report = checker.check_compatibility(&v1_0, &v1_1).unwrap();
        assert!(report.compatible);
        assert!(report.migration_required);
        assert!(matches!(
            report.migration_strategy,
            MigrationStrategy::Automatic { .. }
        ));
    }

    #[test]
    fn test_version_manager() {
        let mut manager = ModelVersionManager::new();
        let version = ModelVersion::new(1, 0, 0);
        let metadata = ModelMetadata::new(version.clone(), "MLP".to_string());

        manager.set_current_version(version.clone(), metadata);
        assert_eq!(manager.get_current_version(), Some(&version));
        assert_eq!(manager.get_version_history().len(), 1);
    }

    #[test]
    fn test_migration_application() {
        let manager = ModelVersionManager::new();
        let from_version = ModelVersion::new(1, 0, 0);
        let to_version = ModelVersion::new(1, 0, 1);

        let mut parameters: HashMap<String, Array2<f64>> = HashMap::new();
        let mut biases: HashMap<String, Array1<f64>> = HashMap::new();

        // Should succeed with no migration needed
        let result =
            manager.apply_migration(&from_version, &to_version, &mut parameters, &mut biases);
        assert!(result.is_ok());
    }
}

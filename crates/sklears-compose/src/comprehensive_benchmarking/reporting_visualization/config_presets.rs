//! User preset configuration management
//!
//! This module handles user-defined presets, sharing configurations, and preset metadata
//! for personalized export settings and collaborative workflows.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use super::format_definitions::ExportFormat;

/// User preset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreset {
    /// Unique preset identifier
    pub preset_id: String,
    /// User-defined preset name
    pub preset_name: String,
    /// Preset description
    pub description: Option<String>,
    /// Export format for this preset
    pub export_format: ExportFormat,
    /// Custom configuration settings
    pub custom_settings: HashMap<String, String>,
    /// Preset metadata
    pub metadata: PresetMetadata,
    /// Sharing settings
    pub sharing: PresetSharing,
    /// Preset category
    pub category: PresetCategory,
    /// Base template (if derived from template)
    pub base_template: Option<String>,
    /// Preset validation status
    pub validation_status: ValidationStatus,
}

/// Preset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetMetadata {
    /// User who created the preset
    pub created_by: String,
    /// Creation timestamp
    pub created_date: DateTime<Utc>,
    /// Last modification timestamp
    pub modified_date: DateTime<Utc>,
    /// Preset version
    pub version: String,
    /// Usage count
    pub usage_count: u64,
    /// Last used timestamp
    pub last_used: Option<DateTime<Utc>>,
    /// Preset tags
    pub tags: Vec<String>,
    /// Favorite status
    pub is_favorite: bool,
}

/// Preset sharing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetSharing {
    /// Sharing enabled
    pub enabled: bool,
    /// Sharing level
    pub sharing_level: SharingLevel,
    /// Allowed users (for user-level sharing)
    pub allowed_users: Vec<String>,
    /// Allowed groups (for group-level sharing)
    pub allowed_groups: Vec<String>,
    /// Public sharing settings
    pub public_settings: Option<PublicSharingSettings>,
}

/// Sharing levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharingLevel {
    /// Private (owner only)
    Private,
    /// User-specific sharing
    User,
    /// Group-level sharing
    Group,
    /// Organization-wide sharing
    Organization,
    /// Public sharing
    Public,
}

/// Public sharing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicSharingSettings {
    /// Allow downloads
    pub allow_downloads: bool,
    /// Allow modifications
    pub allow_modifications: bool,
    /// Require attribution
    pub require_attribution: bool,
    /// License type
    pub license: String,
}

/// Preset categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PresetCategory {
    /// Work/professional presets
    Work,
    /// Personal presets
    Personal,
    /// Project-specific presets
    Project(String),
    /// Shared/team presets
    Shared,
    /// Experimental presets
    Experimental,
    /// Custom category
    Custom(String),
}

/// Validation status for presets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Validation passed
    Valid,
    /// Validation failed
    Invalid(Vec<ValidationError>),
    /// Validation pending
    Pending,
    /// Not validated
    NotValidated,
}

/// Validation error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error severity
    pub severity: ValidationSeverity,
    /// Field that caused the error
    pub field: Option<String>,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Validation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Information level
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical level
    Critical,
}

/// User preset management operations
impl UserPreset {
    /// Creates a new user preset
    pub fn new(
        preset_id: String,
        preset_name: String,
        created_by: String,
        export_format: ExportFormat,
        category: PresetCategory,
    ) -> Self {
        let now = Utc::now();
        Self {
            preset_id,
            preset_name,
            description: None,
            export_format,
            custom_settings: HashMap::new(),
            metadata: PresetMetadata {
                created_by,
                created_date: now,
                modified_date: now,
                version: "1.0.0".to_string(),
                usage_count: 0,
                last_used: None,
                tags: vec![],
                is_favorite: false,
            },
            sharing: PresetSharing {
                enabled: false,
                sharing_level: SharingLevel::Private,
                allowed_users: vec![],
                allowed_groups: vec![],
                public_settings: None,
            },
            category,
            base_template: None,
            validation_status: ValidationStatus::NotValidated,
        }
    }

    /// Sets the preset description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self.metadata.modified_date = Utc::now();
        self
    }

    /// Sets custom configuration settings
    pub fn with_custom_settings(mut self, settings: HashMap<String, String>) -> Self {
        self.custom_settings = settings;
        self.metadata.modified_date = Utc::now();
        self
    }

    /// Sets the base template
    pub fn with_base_template(mut self, template_id: String) -> Self {
        self.base_template = Some(template_id);
        self.metadata.modified_date = Utc::now();
        self
    }

    /// Sets preset tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.metadata.tags = tags;
        self.metadata.modified_date = Utc::now();
        self
    }

    /// Sets sharing configuration
    pub fn with_sharing(mut self, sharing: PresetSharing) -> Self {
        self.sharing = sharing;
        self.metadata.modified_date = Utc::now();
        self
    }

    /// Marks preset as favorite
    pub fn set_favorite(&mut self, is_favorite: bool) {
        self.metadata.is_favorite = is_favorite;
        self.metadata.modified_date = Utc::now();
    }

    /// Updates usage statistics
    pub fn update_usage(&mut self) {
        self.metadata.usage_count += 1;
        self.metadata.last_used = Some(Utc::now());
        self.metadata.modified_date = Utc::now();
    }

    /// Updates the preset version
    pub fn update_version(&mut self, version: String) {
        self.metadata.version = version;
        self.metadata.modified_date = Utc::now();
    }

    /// Adds a tag to the preset
    pub fn add_tag(&mut self, tag: String) {
        if !self.metadata.tags.contains(&tag) {
            self.metadata.tags.push(tag);
            self.metadata.modified_date = Utc::now();
        }
    }

    /// Removes a tag from the preset
    pub fn remove_tag(&mut self, tag: &str) {
        if let Some(pos) = self.metadata.tags.iter().position(|t| t == tag) {
            self.metadata.tags.remove(pos);
            self.metadata.modified_date = Utc::now();
        }
    }

    /// Checks if preset has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.metadata.tags.iter().any(|t| t == tag)
    }

    /// Gets preset age in days
    pub fn get_age_days(&self) -> i64 {
        (Utc::now() - self.metadata.created_date).num_days()
    }

    /// Checks if preset is recently used (within specified days)
    pub fn is_recently_used(&self, days: i64) -> bool {
        if let Some(last_used) = self.metadata.last_used {
            (Utc::now() - last_used).num_days() <= days
        } else {
            false
        }
    }

    /// Validates preset configuration
    pub fn validate(&mut self) -> ValidationStatus {
        let mut errors = vec![];

        // Validate preset name
        if self.preset_name.trim().is_empty() {
            errors.push(ValidationError {
                code: "EMPTY_NAME".to_string(),
                message: "Preset name cannot be empty".to_string(),
                severity: ValidationSeverity::Error,
                field: Some("preset_name".to_string()),
                suggested_fix: Some("Provide a meaningful preset name".to_string()),
            });
        }

        // Validate export format settings
        if self.custom_settings.is_empty() && self.base_template.is_none() {
            errors.push(ValidationError {
                code: "NO_SETTINGS".to_string(),
                message: "Preset must have custom settings or base template".to_string(),
                severity: ValidationSeverity::Warning,
                field: Some("custom_settings".to_string()),
                suggested_fix: Some("Add custom settings or select a base template".to_string()),
            });
        }

        // Validate sharing settings
        if self.sharing.enabled {
            match self.sharing.sharing_level {
                SharingLevel::User if self.sharing.allowed_users.is_empty() => {
                    errors.push(ValidationError {
                        code: "USER_SHARING_NO_USERS".to_string(),
                        message: "User-level sharing enabled but no users specified".to_string(),
                        severity: ValidationSeverity::Warning,
                        field: Some("sharing.allowed_users".to_string()),
                        suggested_fix: Some("Add allowed users or change sharing level".to_string()),
                    });
                }
                SharingLevel::Group if self.sharing.allowed_groups.is_empty() => {
                    errors.push(ValidationError {
                        code: "GROUP_SHARING_NO_GROUPS".to_string(),
                        message: "Group-level sharing enabled but no groups specified".to_string(),
                        severity: ValidationSeverity::Warning,
                        field: Some("sharing.allowed_groups".to_string()),
                        suggested_fix: Some("Add allowed groups or change sharing level".to_string()),
                    });
                }
                SharingLevel::Public if self.sharing.public_settings.is_none() => {
                    errors.push(ValidationError {
                        code: "PUBLIC_SHARING_NO_SETTINGS".to_string(),
                        message: "Public sharing enabled but no public settings specified".to_string(),
                        severity: ValidationSeverity::Error,
                        field: Some("sharing.public_settings".to_string()),
                        suggested_fix: Some("Configure public sharing settings".to_string()),
                    });
                }
                _ => {}
            }
        }

        self.validation_status = if errors.is_empty() {
            ValidationStatus::Valid
        } else {
            ValidationStatus::Invalid(errors.clone())
        };

        self.validation_status.clone()
    }
}

impl PresetSharing {
    /// Creates a private sharing configuration
    pub fn private() -> Self {
        Self {
            enabled: false,
            sharing_level: SharingLevel::Private,
            allowed_users: vec![],
            allowed_groups: vec![],
            public_settings: None,
        }
    }

    /// Creates a user-level sharing configuration
    pub fn user_sharing(allowed_users: Vec<String>) -> Self {
        Self {
            enabled: true,
            sharing_level: SharingLevel::User,
            allowed_users,
            allowed_groups: vec![],
            public_settings: None,
        }
    }

    /// Creates a group-level sharing configuration
    pub fn group_sharing(allowed_groups: Vec<String>) -> Self {
        Self {
            enabled: true,
            sharing_level: SharingLevel::Group,
            allowed_users: vec![],
            allowed_groups,
            public_settings: None,
        }
    }

    /// Creates an organization-wide sharing configuration
    pub fn organization_sharing() -> Self {
        Self {
            enabled: true,
            sharing_level: SharingLevel::Organization,
            allowed_users: vec![],
            allowed_groups: vec![],
            public_settings: None,
        }
    }

    /// Creates a public sharing configuration
    pub fn public_sharing(public_settings: PublicSharingSettings) -> Self {
        Self {
            enabled: true,
            sharing_level: SharingLevel::Public,
            allowed_users: vec![],
            allowed_groups: vec![],
            public_settings: Some(public_settings),
        }
    }

    /// Checks if user has access to the preset
    pub fn has_user_access(&self, user_id: &str) -> bool {
        if !self.enabled {
            return false;
        }

        match self.sharing_level {
            SharingLevel::Private => false,
            SharingLevel::User => self.allowed_users.contains(&user_id.to_string()),
            SharingLevel::Group => {
                // Note: In a real implementation, you'd check group membership
                !self.allowed_groups.is_empty()
            }
            SharingLevel::Organization | SharingLevel::Public => true,
        }
    }

    /// Checks if preset allows downloads
    pub fn allows_downloads(&self) -> bool {
        match &self.public_settings {
            Some(settings) => settings.allow_downloads,
            None => self.enabled && matches!(self.sharing_level, SharingLevel::Organization | SharingLevel::Public),
        }
    }

    /// Checks if preset allows modifications
    pub fn allows_modifications(&self) -> bool {
        match &self.public_settings {
            Some(settings) => settings.allow_modifications,
            None => false,
        }
    }
}

impl PublicSharingSettings {
    /// Creates permissive public sharing settings
    pub fn permissive(license: String) -> Self {
        Self {
            allow_downloads: true,
            allow_modifications: true,
            require_attribution: false,
            license,
        }
    }

    /// Creates restrictive public sharing settings
    pub fn restrictive(license: String) -> Self {
        Self {
            allow_downloads: true,
            allow_modifications: false,
            require_attribution: true,
            license,
        }
    }

    /// Creates read-only public sharing settings
    pub fn read_only(license: String) -> Self {
        Self {
            allow_downloads: false,
            allow_modifications: false,
            require_attribution: true,
            license,
        }
    }
}

impl ValidationError {
    /// Creates a new validation error
    pub fn new(
        code: String,
        message: String,
        severity: ValidationSeverity,
    ) -> Self {
        Self {
            code,
            message,
            severity,
            field: None,
            suggested_fix: None,
        }
    }

    /// Sets the field that caused the error
    pub fn with_field(mut self, field: String) -> Self {
        self.field = Some(field);
        self
    }

    /// Sets a suggested fix for the error
    pub fn with_suggested_fix(mut self, fix: String) -> Self {
        self.suggested_fix = Some(fix);
        self
    }

    /// Checks if the error is critical
    pub fn is_critical(&self) -> bool {
        matches!(self.severity, ValidationSeverity::Critical | ValidationSeverity::Error)
    }

    /// Gets the error priority score (higher = more critical)
    pub fn get_priority_score(&self) -> u8 {
        match self.severity {
            ValidationSeverity::Info => 1,
            ValidationSeverity::Warning => 2,
            ValidationSeverity::Error => 3,
            ValidationSeverity::Critical => 4,
        }
    }
}
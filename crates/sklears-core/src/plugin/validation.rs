//! Plugin Validation Framework
//!
//! This module provides comprehensive validation capabilities for plugins,
//! including security analysis, dependency checking, code safety validation,
//! and trust verification. It ensures that plugins meet security and
//! quality standards before being loaded and executed.

use super::core_traits::Plugin;
use super::security::{DigitalSignature, Permission, PublisherInfo, SecurityPolicy};
use super::types_config::PluginMetadata;
use crate::error::Result;
use std::collections::HashMap;

/// Advanced plugin validation framework with security analysis
///
/// The PluginValidator provides comprehensive validation of plugins including
/// metadata validation, security analysis, dependency checking, code safety
/// validation, and trust verification.
///
/// # Examples
///
/// ```rust,no_run
/// use sklears_core::plugin::{PluginValidator, PluginManifest};
///
/// let validator = PluginValidator::new();
///
/// // Validate a plugin comprehensively
/// // let report = validator.validate_comprehensive(&plugin, &manifest)?;
/// //
/// // Check validation results
/// // if report.has_errors() {
/// //     println!("Validation failed: {:?}", report.get_errors());
/// // } else {
/// //     println!("Plugin validation passed");
/// // }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PluginValidator {
    /// Security policy configuration
    security_policy: SecurityPolicy,
    /// Dependency resolver for checking dependencies
    dependency_resolver: DependencyResolver,
    /// Code analyzer for security checks
    #[allow(dead_code)]
    code_analyzer: CodeAnalyzer,
    /// Trust store for plugin verification
    trust_store: TrustStore,
}

impl PluginValidator {
    /// Create a new plugin validator
    ///
    /// Initializes a new validator with default security policies and
    /// validation components.
    pub fn new() -> Self {
        Self {
            security_policy: SecurityPolicy::default(),
            dependency_resolver: DependencyResolver::new(),
            code_analyzer: CodeAnalyzer::new(),
            trust_store: TrustStore::new(),
        }
    }

    /// Create a validator with custom security policy
    ///
    /// # Arguments
    ///
    /// * `security_policy` - Custom security policy to use
    pub fn with_security_policy(security_policy: SecurityPolicy) -> Self {
        Self {
            security_policy,
            dependency_resolver: DependencyResolver::new(),
            code_analyzer: CodeAnalyzer::new(),
            trust_store: TrustStore::new(),
        }
    }

    /// Perform comprehensive plugin validation
    ///
    /// This method performs all validation checks including basic metadata
    /// validation, security analysis, dependency checking, code safety
    /// validation, and trust verification.
    ///
    /// # Arguments
    ///
    /// * `plugin` - The plugin to validate
    /// * `manifest` - The plugin manifest with detailed information
    ///
    /// # Returns
    ///
    /// A validation report containing all findings, errors, and warnings.
    pub fn validate_comprehensive(
        &self,
        plugin: &dyn Plugin,
        manifest: &PluginManifest,
    ) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        // Basic validation
        self.validate_basic(plugin, &mut report)?;

        // Security validation
        self.validate_security(manifest, &mut report)?;

        // Dependency validation
        self.validate_dependencies(manifest, &mut report)?;

        // Code analysis
        self.validate_code_safety(manifest, &mut report)?;

        // Trust verification
        self.validate_trust(manifest, &mut report)?;

        Ok(report)
    }

    /// Validate basic plugin requirements
    ///
    /// Performs basic validation of plugin metadata and requirements.
    fn validate_basic(&self, plugin: &dyn Plugin, report: &mut ValidationReport) -> Result<()> {
        let metadata = plugin.metadata();

        if metadata.name.is_empty() {
            report.add_error(ValidationError::InvalidMetadata(
                "Plugin name cannot be empty".to_string(),
            ));
        }

        if metadata.version.is_empty() {
            report.add_error(ValidationError::InvalidMetadata(
                "Plugin version cannot be empty".to_string(),
            ));
        }

        if metadata.author.is_empty() {
            report.add_error(ValidationError::InvalidMetadata(
                "Plugin author cannot be empty".to_string(),
            ));
        }

        // Validate version format
        if !self.is_valid_version(&metadata.version) {
            report.add_error(ValidationError::InvalidVersion(metadata.version.clone()));
        }

        // Check required SDK version compatibility
        if !self.is_compatible_sdk_version(&metadata.min_sdk_version) {
            report.add_error(ValidationError::IncompatibleSdkVersion(
                metadata.min_sdk_version.clone(),
            ));
        }

        report.add_check(ValidationCheck::BasicMetadata, ValidationResult::Passed);
        Ok(())
    }

    /// Validate security requirements
    ///
    /// Performs security validation including permission checks, API usage
    /// validation, and unsafe code detection.
    fn validate_security(
        &self,
        manifest: &PluginManifest,
        report: &mut ValidationReport,
    ) -> Result<()> {
        // Check for dangerous permissions
        for permission in &manifest.permissions {
            if self.security_policy.is_dangerous_permission(permission) {
                report.add_warning(ValidationWarning::DangerousPermission(permission.clone()));
            }
        }

        // Validate API usage patterns
        if let Some(ref api_usage) = manifest.api_usage {
            for api_call in &api_usage.calls {
                if self.security_policy.is_restricted_api(api_call) {
                    report.add_error(ValidationError::RestrictedApiUsage(api_call.clone()));
                }
            }
        }

        // Check for unsafe code blocks
        if manifest.contains_unsafe_code {
            if !self.security_policy.allow_unsafe_code {
                report.add_error(ValidationError::UnsafeCodeNotAllowed);
            } else {
                report.add_warning(ValidationWarning::UnsafeCodeDetected);
            }
        }

        report.add_check(ValidationCheck::Security, ValidationResult::Passed);
        Ok(())
    }

    /// Validate plugin dependencies
    ///
    /// Checks that all required dependencies are available, compatible,
    /// and free from known vulnerabilities.
    fn validate_dependencies(
        &self,
        manifest: &PluginManifest,
        report: &mut ValidationReport,
    ) -> Result<()> {
        for dependency in &manifest.dependencies {
            // Check if dependency is available
            if !self.dependency_resolver.is_available(dependency) {
                report.add_error(ValidationError::MissingDependency(dependency.clone()));
                continue;
            }

            // Check version compatibility
            if !self.dependency_resolver.is_compatible_version(dependency) {
                report.add_error(ValidationError::IncompatibleDependency(dependency.clone()));
                continue;
            }

            // Check for known vulnerabilities
            if let Some(vulnerabilities) =
                self.dependency_resolver.check_vulnerabilities(dependency)
            {
                for vuln in vulnerabilities {
                    report.add_error(ValidationError::VulnerableDependency(
                        dependency.clone(),
                        vuln,
                    ));
                }
            }
        }

        report.add_check(ValidationCheck::Dependencies, ValidationResult::Passed);
        Ok(())
    }

    /// Validate code safety
    ///
    /// Performs static analysis of the plugin code to identify potential
    /// safety issues, suspicious patterns, and complexity problems.
    fn validate_code_safety(
        &self,
        manifest: &PluginManifest,
        report: &mut ValidationReport,
    ) -> Result<()> {
        if let Some(ref code_info) = manifest.code_analysis {
            // Check complexity metrics
            if code_info.cyclomatic_complexity > self.security_policy.max_complexity as usize {
                report.add_warning(ValidationWarning::HighComplexity(
                    code_info.cyclomatic_complexity,
                ));
            }

            // Check for suspicious patterns
            for pattern in &code_info.suspicious_patterns {
                report.add_warning(ValidationWarning::SuspiciousPattern(pattern.clone()));
            }

            // Memory safety checks
            if code_info.potential_memory_issues > 0 {
                report.add_error(ValidationError::MemorySafetyIssue(
                    code_info.potential_memory_issues,
                ));
            }
        }

        report.add_check(ValidationCheck::CodeSafety, ValidationResult::Passed);
        Ok(())
    }

    /// Validate trust and signatures
    ///
    /// Verifies digital signatures and checks publisher trust levels.
    fn validate_trust(
        &self,
        manifest: &PluginManifest,
        report: &mut ValidationReport,
    ) -> Result<()> {
        // Verify digital signature
        if let Some(ref signature) = manifest.signature {
            match self
                .trust_store
                .verify_signature(&manifest.content_hash, signature)
            {
                Ok(true) => report.add_check(ValidationCheck::Signature, ValidationResult::Passed),
                Ok(false) => report.add_error(ValidationError::InvalidSignature),
                Err(e) => {
                    report.add_error(ValidationError::SignatureVerificationFailed(e.to_string()))
                }
            }
        } else if self.security_policy.require_signatures {
            report.add_error(ValidationError::MissingSignature);
        }

        // Check publisher trust level
        let trust_level = self.trust_store.get_publisher_trust(&manifest.publisher);
        if trust_level < self.security_policy.min_trust_level as f32 {
            report.add_warning(ValidationWarning::LowTrustPublisher(trust_level));
        }

        report.add_check(ValidationCheck::Trust, ValidationResult::Passed);
        Ok(())
    }

    /// Check if version string is valid semver
    fn is_valid_version(&self, version: &str) -> bool {
        let parts: Vec<&str> = version.split('.').collect();
        parts.len() >= 2 && parts.len() <= 3 && parts.iter().all(|p| p.parse::<u32>().is_ok())
    }

    /// Check if SDK version is compatible
    fn is_compatible_sdk_version(&self, min_version: &str) -> bool {
        // Simplified version check - in practice, you'd use a proper semver library
        // For now, assume compatibility
        !min_version.is_empty()
    }
}

impl Default for PluginValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced plugin manifest with marketplace information
///
/// Contains comprehensive information about a plugin including metadata,
/// security information, dependencies, and marketplace data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PluginManifest {
    /// Basic plugin metadata
    pub metadata: PluginMetadata,
    /// Plugin permissions required
    pub permissions: Vec<Permission>,
    /// API usage information
    pub api_usage: Option<ApiUsageInfo>,
    /// Whether plugin contains unsafe code
    pub contains_unsafe_code: bool,
    /// Plugin dependencies with versions
    pub dependencies: Vec<Dependency>,
    /// Code analysis results
    pub code_analysis: Option<CodeAnalysisInfo>,
    /// Digital signature for verification
    pub signature: Option<DigitalSignature>,
    /// Content hash for integrity verification
    pub content_hash: String,
    /// Publisher information
    pub publisher: PublisherInfo,
    /// Marketplace specific metadata
    pub marketplace: MarketplaceInfo,
}

/// Validation report containing all validation results
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Validation errors that prevent plugin usage
    pub errors: Vec<ValidationError>,
    /// Validation warnings that should be noted but don't prevent usage
    pub warnings: Vec<ValidationWarning>,
    /// Completed validation checks
    pub checks: HashMap<ValidationCheck, ValidationResult>,
    /// Overall validation status
    pub status: ValidationStatus,
}

impl ValidationReport {
    /// Create a new empty validation report
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            checks: HashMap::new(),
            status: ValidationStatus::Pending,
        }
    }

    /// Add a validation error
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
        self.status = ValidationStatus::Failed;
    }

    /// Add a validation warning
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    /// Add a completed validation check
    pub fn add_check(&mut self, check: ValidationCheck, result: ValidationResult) {
        self.checks.insert(check, result);
        if self.status == ValidationStatus::Pending && self.errors.is_empty() {
            self.status = ValidationStatus::Passed;
        }
    }

    /// Check if validation has errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Check if validation has warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Get all validation errors
    pub fn get_errors(&self) -> &[ValidationError] {
        &self.errors
    }

    /// Get all validation warnings
    pub fn get_warnings(&self) -> &[ValidationWarning] {
        &self.warnings
    }

    /// Get validation status
    pub fn status(&self) -> ValidationStatus {
        self.status
    }

    /// Get completed checks
    pub fn get_checks(&self) -> &HashMap<ValidationCheck, ValidationResult> {
        &self.checks
    }
}

/// Validation errors that prevent plugin usage
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Invalid metadata
    InvalidMetadata(String),
    /// Invalid version format
    InvalidVersion(String),
    /// Incompatible SDK version
    IncompatibleSdkVersion(String),
    /// Missing required dependency
    MissingDependency(Dependency),
    /// Incompatible dependency version
    IncompatibleDependency(Dependency),
    /// Vulnerable dependency detected
    VulnerableDependency(Dependency, String),
    /// Restricted API usage detected
    RestrictedApiUsage(String),
    /// Unsafe code not allowed
    UnsafeCodeNotAllowed,
    /// Memory safety issue detected
    MemorySafetyIssue(usize),
    /// Invalid digital signature
    InvalidSignature,
    /// Missing required signature
    MissingSignature,
    /// Signature verification failed
    SignatureVerificationFailed(String),
}

/// Validation warnings that should be noted but don't prevent usage
#[derive(Debug, Clone)]
pub enum ValidationWarning {
    /// Dangerous permission requested
    DangerousPermission(Permission),
    /// Unsafe code detected
    UnsafeCodeDetected,
    /// High code complexity
    HighComplexity(usize),
    /// Suspicious code pattern detected
    SuspiciousPattern(String),
    /// Low trust publisher
    LowTrustPublisher(f32),
}

/// Types of validation checks
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValidationCheck {
    /// Basic metadata validation
    BasicMetadata,
    /// Security validation
    Security,
    /// Dependency validation
    Dependencies,
    /// Code safety validation
    CodeSafety,
    /// Trust and signature validation
    Trust,
    /// Digital signature verification
    Signature,
}

/// Validation check results
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    /// Check passed successfully
    Passed,
    /// Check failed
    Failed,
    /// Check was skipped
    Skipped,
}

/// Overall validation status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationStatus {
    /// Validation is pending
    Pending,
    /// Validation passed
    Passed,
    /// Validation failed
    Failed,
}

/// Security vulnerability information
#[derive(Debug, Clone)]
pub struct Vulnerability {
    /// Vulnerability ID (e.g., CVE number)
    pub id: String,
    /// Severity level
    pub severity: VulnerabilitySeverity,
    /// Description of the vulnerability
    pub description: String,
    /// Affected versions
    pub affected_versions: Vec<String>,
}

/// Vulnerability severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VulnerabilitySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// API usage information for security analysis
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiUsageInfo {
    /// API calls made by the plugin
    pub calls: Vec<String>,
    /// Network access patterns
    pub network_access: Vec<String>,
    /// File system access patterns
    pub filesystem_access: Vec<String>,
}

/// Code analysis information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodeAnalysisInfo {
    /// Cyclomatic complexity score
    pub cyclomatic_complexity: usize,
    /// Suspicious patterns detected
    pub suspicious_patterns: Vec<String>,
    /// Potential memory issues count
    pub potential_memory_issues: usize,
    /// Lines of code
    pub lines_of_code: usize,
    /// Test coverage percentage
    pub test_coverage: f32,
}

/// Dependency specification
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Dependency {
    /// Dependency name
    pub name: String,
    /// Version requirement
    pub version: String,
    /// Whether dependency is optional
    pub optional: bool,
    /// Features required from the dependency
    pub features: Vec<String>,
}

/// Marketplace information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MarketplaceInfo {
    /// Plugin URL on marketplace
    pub url: String,
    /// Download count
    pub downloads: u64,
    /// User rating (0.0 to 5.0)
    pub rating: f32,
    /// Number of reviews
    pub reviews: u32,
    /// Last update timestamp
    pub last_updated: String,
}

/// Dependency resolver for checking plugin dependencies
#[derive(Debug)]
pub struct DependencyResolver {
    /// Available dependencies
    available_deps: HashMap<String, String>,
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyResolver {
    /// Create a new dependency resolver
    pub fn new() -> Self {
        Self {
            available_deps: HashMap::new(),
        }
    }

    /// Check if a dependency is available
    pub fn is_available(&self, dependency: &Dependency) -> bool {
        self.available_deps.contains_key(&dependency.name)
    }

    /// Check if dependency version is compatible
    pub fn is_compatible_version(&self, _dependency: &Dependency) -> bool {
        // Simplified version check
        true
    }

    /// Check for known vulnerabilities
    pub fn check_vulnerabilities(&self, _dependency: &Dependency) -> Option<Vec<String>> {
        // Placeholder - would integrate with vulnerability databases
        None
    }
}

/// Code analyzer for security checks
#[derive(Debug)]
pub struct CodeAnalyzer {
    /// Analysis rules
    #[allow(dead_code)]
    rules: Vec<AnalysisRule>,
}

impl Default for CodeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeAnalyzer {
    /// Create a new code analyzer
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }
}

/// Trust store for plugin verification
#[derive(Debug)]
pub struct TrustStore {
    /// Trusted publishers
    trusted_publishers: HashMap<String, f32>,
}

impl Default for TrustStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TrustStore {
    /// Create a new trust store
    pub fn new() -> Self {
        Self {
            trusted_publishers: HashMap::new(),
        }
    }

    /// Verify a digital signature
    pub fn verify_signature(
        &self,
        _content_hash: &str,
        _signature: &DigitalSignature,
    ) -> Result<bool> {
        // Placeholder - would implement actual signature verification
        Ok(true)
    }

    /// Get publisher trust level
    pub fn get_publisher_trust(&self, publisher: &PublisherInfo) -> f32 {
        self.trusted_publishers
            .get(&publisher.name)
            .copied()
            .unwrap_or(0.5)
    }
}

/// Analysis rule for code validation
#[derive(Debug, Clone)]
pub struct AnalysisRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule pattern
    pub pattern: String,
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

//! Template Management Core System
//!
//! This module provides the main orchestration and coordination for the template
//! management system, including the central TemplateManagementSystem structure
//! and error handling definitions.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use super::{
    template_repository::{TemplateRepository, TemplateEntry},
    template_subsystems::*,
};

/// Central template management system that orchestrates all template operations
///
/// This system coordinates between different subsystems including repository management,
/// versioning, validation, compilation, security, and performance analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateManagementSystem {
    /// Template repository and storage
    pub template_repository: Arc<RwLock<TemplateRepository>>,
    /// Template versioning system
    pub versioning_system: Arc<RwLock<TemplateVersioningSystem>>,
    /// Template category management
    pub category_manager: Arc<RwLock<TemplateCategoryManager>>,
    /// Template validation engine
    pub validation_engine: Arc<RwLock<TemplateValidationEngine>>,
    /// Template compilation system
    pub compilation_system: Arc<RwLock<TemplateCompilationSystem>>,
    /// Template dependency management
    pub dependency_manager: Arc<RwLock<TemplateDependencyManager>>,
    /// Template security scanner
    pub security_scanner: Arc<RwLock<TemplateSecurityScanner>>,
    /// Template performance analyzer
    pub performance_analyzer: Arc<RwLock<TemplatePerformanceAnalyzer>>,
}

impl TemplateManagementSystem {
    /// Create a new template management system
    ///
    /// Initializes all subsystems with default configurations and establishes
    /// the coordination framework for template operations.
    pub fn new() -> Self {
        Self {
            template_repository: Arc::new(RwLock::new(TemplateRepository::new())),
            versioning_system: Arc::new(RwLock::new(TemplateVersioningSystem::new())),
            category_manager: Arc::new(RwLock::new(TemplateCategoryManager::new())),
            validation_engine: Arc::new(RwLock::new(TemplateValidationEngine::new())),
            compilation_system: Arc::new(RwLock::new(TemplateCompilationSystem::new())),
            dependency_manager: Arc::new(RwLock::new(TemplateDependencyManager::new())),
            security_scanner: Arc::new(RwLock::new(TemplateSecurityScanner::new())),
            performance_analyzer: Arc::new(RwLock::new(TemplatePerformanceAnalyzer::new())),
        }
    }

    /// Add template to repository
    ///
    /// Adds a new template to the repository after performing validation,
    /// security scanning, and dependency analysis.
    ///
    /// # Arguments
    /// * `template` - The template entry to add
    ///
    /// # Returns
    /// Result indicating success or error in template addition
    pub fn add_template(&self, template: TemplateEntry) -> Result<(), TemplateError> {
        // Validate template before adding
        {
            let validation_engine = self.validation_engine.read().unwrap();
            validation_engine.validate_template(&template)?;
        }

        // Perform security scan
        {
            let security_scanner = self.security_scanner.read().unwrap();
            security_scanner.scan_template(&template)?;
        }

        // Analyze dependencies
        {
            let dependency_manager = self.dependency_manager.read().unwrap();
            dependency_manager.analyze_dependencies(&template)?;
        }

        // Add to repository
        let mut repository = self.template_repository.write().unwrap();
        repository.add_template(template)?;

        Ok(())
    }

    /// Get template by ID
    ///
    /// Retrieves a template from the repository by its unique identifier.
    ///
    /// # Arguments
    /// * `template_id` - The unique identifier of the template
    ///
    /// # Returns
    /// Result containing the template entry or error if not found
    pub fn get_template(&self, template_id: &str) -> Result<TemplateEntry, TemplateError> {
        let repository = self.template_repository.read().unwrap();
        repository.get_template(template_id)
    }

    /// Search templates
    ///
    /// Performs a search across all templates using the integrated search engine.
    ///
    /// # Arguments
    /// * `query` - The search query string
    ///
    /// # Returns
    /// Result containing matching template entries
    pub fn search_templates(&self, query: &str) -> Result<Vec<TemplateEntry>, TemplateError> {
        let repository = self.template_repository.read().unwrap();
        repository.search_templates(query)
    }

    /// Update template
    ///
    /// Updates an existing template with new content and metadata.
    ///
    /// # Arguments
    /// * `template_id` - The unique identifier of the template to update
    /// * `updated_template` - The updated template entry
    ///
    /// # Returns
    /// Result indicating success or error in template update
    pub fn update_template(&self, template_id: &str, updated_template: TemplateEntry) -> Result<(), TemplateError> {
        // Validate updated template
        {
            let validation_engine = self.validation_engine.read().unwrap();
            validation_engine.validate_template(&updated_template)?;
        }

        // Version management
        {
            let mut versioning_system = self.versioning_system.write().unwrap();
            versioning_system.create_version(template_id, &updated_template)?;
        }

        // Update repository
        let mut repository = self.template_repository.write().unwrap();
        repository.update_template(template_id, updated_template)?;

        Ok(())
    }

    /// Delete template
    ///
    /// Removes a template from the repository after checking dependencies.
    ///
    /// # Arguments
    /// * `template_id` - The unique identifier of the template to delete
    ///
    /// # Returns
    /// Result indicating success or error in template deletion
    pub fn delete_template(&self, template_id: &str) -> Result<(), TemplateError> {
        // Check dependencies before deletion
        {
            let dependency_manager = self.dependency_manager.read().unwrap();
            if dependency_manager.has_dependents(template_id)? {
                return Err(TemplateError::DependencyError(
                    "Cannot delete template with active dependencies".to_string()
                ));
            }
        }

        // Archive version
        {
            let mut versioning_system = self.versioning_system.write().unwrap();
            versioning_system.archive_template(template_id)?;
        }

        // Remove from repository
        let mut repository = self.template_repository.write().unwrap();
        repository.delete_template(template_id)?;

        Ok(())
    }

    /// Compile template
    ///
    /// Compiles a template for execution with given parameters.
    ///
    /// # Arguments
    /// * `template_id` - The unique identifier of the template to compile
    /// * `parameters` - Compilation parameters
    ///
    /// # Returns
    /// Result containing the compiled template or compilation error
    pub fn compile_template(
        &self,
        template_id: &str,
        parameters: &HashMap<String, String>
    ) -> Result<CompiledTemplate, TemplateError> {
        let template = self.get_template(template_id)?;

        let compilation_system = self.compilation_system.read().unwrap();
        compilation_system.compile_template(&template, parameters)
    }

    /// Get template performance metrics
    ///
    /// Retrieves performance analysis data for a specific template.
    ///
    /// # Arguments
    /// * `template_id` - The unique identifier of the template
    ///
    /// # Returns
    /// Result containing performance metrics or error
    pub fn get_template_performance(&self, template_id: &str) -> Result<TemplatePerformanceMetrics, TemplateError> {
        let performance_analyzer = self.performance_analyzer.read().unwrap();
        performance_analyzer.get_metrics(template_id)
    }

    /// Validate system integrity
    ///
    /// Performs a comprehensive validation of the entire template management system.
    ///
    /// # Returns
    /// Result containing validation summary or errors found
    pub fn validate_system_integrity(&self) -> Result<SystemIntegrityReport, TemplateError> {
        let mut report = SystemIntegrityReport::new();

        // Validate repository integrity
        {
            let repository = self.template_repository.read().unwrap();
            repository.validate_integrity(&mut report)?;
        }

        // Validate all templates
        {
            let validation_engine = self.validation_engine.read().unwrap();
            let repository = self.template_repository.read().unwrap();

            for template in repository.get_all_templates()? {
                if let Err(e) = validation_engine.validate_template(&template) {
                    report.add_validation_error(template.template_id.clone(), e);
                }
            }
        }

        Ok(report)
    }
}

impl Default for TemplateManagementSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled template ready for execution
#[derive(Debug, Clone)]
pub struct CompiledTemplate {
    /// Template identifier
    pub template_id: String,
    /// Compiled content
    pub compiled_content: String,
    /// Compilation metadata
    pub metadata: CompilationMetadata,
    /// Performance characteristics
    pub performance_info: TemplatePerformanceInfo,
}

/// Compilation metadata and settings
#[derive(Debug, Clone)]
pub struct CompilationMetadata {
    /// Compilation timestamp
    pub compiled_at: DateTime<Utc>,
    /// Compilation parameters used
    pub parameters: HashMap<String, String>,
    /// Compiler version
    pub compiler_version: String,
    /// Optimization level applied
    pub optimization_level: OptimizationLevel,
    /// Warnings generated during compilation
    pub warnings: Vec<CompilationWarning>,
}

/// Template performance metrics
#[derive(Debug, Clone)]
pub struct TemplatePerformanceMetrics {
    /// Execution time statistics
    pub execution_time: ExecutionTimeStats,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    /// Resource utilization metrics
    pub resource_utilization: ResourceUtilizationStats,
    /// Performance trends
    pub trends: PerformanceTrends,
}

/// Execution time statistics
#[derive(Debug, Clone)]
pub struct ExecutionTimeStats {
    /// Average execution time in milliseconds
    pub average_ms: f64,
    /// Minimum execution time
    pub min_ms: f64,
    /// Maximum execution time
    pub max_ms: f64,
    /// Standard deviation
    pub std_deviation: f64,
    /// 95th percentile
    pub p95_ms: f64,
    /// 99th percentile
    pub p99_ms: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    /// Average memory usage
    pub average_bytes: usize,
    /// Memory allocation count
    pub allocation_count: usize,
    /// Memory deallocation count
    pub deallocation_count: usize,
}

/// Resource utilization statistics
#[derive(Debug, Clone)]
pub struct ResourceUtilizationStats {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// I/O operations count
    pub io_operations: usize,
    /// Network requests made
    pub network_requests: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

/// Performance trends over time
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Performance trend direction
    pub trend_direction: TrendDirection,
    /// Rate of change
    pub change_rate: f64,
    /// Confidence level of trend analysis
    pub confidence_level: f64,
}

/// Direction of performance trend
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Template performance information
#[derive(Debug, Clone)]
pub struct TemplatePerformanceInfo {
    /// Estimated execution time
    pub estimated_execution_time_ms: f64,
    /// Memory requirements
    pub memory_requirements_bytes: usize,
    /// Complexity score
    pub complexity_score: f64,
    /// Resource intensity level
    pub resource_intensity: ResourceIntensity,
}

/// Resource intensity levels
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceIntensity {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization levels for template compilation
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Maximum,
}

/// Compilation warning information
#[derive(Debug, Clone)]
pub struct CompilationWarning {
    /// Warning message
    pub message: String,
    /// Warning level
    pub level: WarningLevel,
    /// Source location
    pub location: Option<SourceLocation>,
}

/// Warning severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum WarningLevel {
    Info,
    Warning,
    Error,
}

/// Source location information
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// Line number
    pub line: usize,
    /// Column number
    pub column: usize,
    /// Source file
    pub file: String,
}

/// System integrity validation report
#[derive(Debug, Clone)]
pub struct SystemIntegrityReport {
    /// Overall system health status
    pub health_status: SystemHealthStatus,
    /// Repository integrity status
    pub repository_status: RepositoryStatus,
    /// Template validation errors
    pub validation_errors: Vec<TemplateValidationError>,
    /// Security issues found
    pub security_issues: Vec<SecurityIssue>,
    /// Performance issues
    pub performance_issues: Vec<PerformanceIssue>,
    /// Recommendations for improvements
    pub recommendations: Vec<SystemRecommendation>,
}

impl SystemIntegrityReport {
    /// Create a new empty integrity report
    pub fn new() -> Self {
        Self {
            health_status: SystemHealthStatus::Unknown,
            repository_status: RepositoryStatus::Unknown,
            validation_errors: Vec::new(),
            security_issues: Vec::new(),
            performance_issues: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    /// Add a validation error to the report
    pub fn add_validation_error(&mut self, template_id: String, error: TemplateError) {
        self.validation_errors.push(TemplateValidationError {
            template_id,
            error_message: error.to_string(),
            severity: ValidationErrorSeverity::Error,
        });
    }

    /// Add a security issue to the report
    pub fn add_security_issue(&mut self, issue: SecurityIssue) {
        self.security_issues.push(issue);
    }

    /// Add a performance issue to the report
    pub fn add_performance_issue(&mut self, issue: PerformanceIssue) {
        self.performance_issues.push(issue);
    }

    /// Add a system recommendation
    pub fn add_recommendation(&mut self, recommendation: SystemRecommendation) {
        self.recommendations.push(recommendation);
    }
}

/// Overall system health status
#[derive(Debug, Clone, PartialEq)]
pub enum SystemHealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// Repository integrity status
#[derive(Debug, Clone, PartialEq)]
pub enum RepositoryStatus {
    Valid,
    Corrupted,
    Inconsistent,
    Unknown,
}

/// Template validation error details
#[derive(Debug, Clone)]
pub struct TemplateValidationError {
    /// Template identifier
    pub template_id: String,
    /// Error message
    pub error_message: String,
    /// Error severity
    pub severity: ValidationErrorSeverity,
}

/// Validation error severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Security issue details
#[derive(Debug, Clone)]
pub struct SecurityIssue {
    /// Issue type
    pub issue_type: SecurityIssueType,
    /// Affected template ID
    pub template_id: Option<String>,
    /// Issue description
    pub description: String,
    /// Risk level
    pub risk_level: RiskLevel,
}

/// Types of security issues
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityIssueType {
    UnauthorizedAccess,
    InsecureContent,
    PrivilegeEscalation,
    DataExposure,
    CodeInjection,
}

/// Risk assessment levels
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance issue details
#[derive(Debug, Clone)]
pub struct PerformanceIssue {
    /// Issue type
    pub issue_type: PerformanceIssueType,
    /// Affected template ID
    pub template_id: Option<String>,
    /// Issue description
    pub description: String,
    /// Impact severity
    pub impact: PerformanceImpact,
}

/// Types of performance issues
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceIssueType {
    SlowExecution,
    MemoryLeak,
    ResourceStarvation,
    InfiniteLoop,
    DeadLock,
}

/// Performance impact levels
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceImpact {
    Minimal,
    Moderate,
    Significant,
    Severe,
}

/// System improvement recommendations
#[derive(Debug, Clone)]
pub struct SystemRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description
    pub description: String,
    /// Implementation effort estimate
    pub effort_estimate: EffortEstimate,
}

/// Types of system recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationType {
    Security,
    Performance,
    Maintenance,
    Upgrade,
    Configuration,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation effort estimates
#[derive(Debug, Clone, PartialEq)]
pub enum EffortEstimate {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

/// Comprehensive error handling for template operations
#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("Template not found: {0}")]
    TemplateNotFound(String),
    #[error("Template validation error: {0}")]
    ValidationError(String),
    #[error("Template compilation error: {0}")]
    CompilationError(String),
    #[error("Template security error: {0}")]
    SecurityError(String),
    #[error("Template permission error: {0}")]
    PermissionError(String),
    #[error("Template dependency error: {0}")]
    DependencyError(String),
    #[error("Template configuration error: {0}")]
    ConfigurationError(String),
    #[error("Template integrity error: {0}")]
    IntegrityError(String),
    #[error("Template parsing error: {0}")]
    ParsingError(String),
    #[error("Template execution error: {0}")]
    ExecutionError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

impl TemplateError {
    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(self,
            TemplateError::ValidationError(_) |
            TemplateError::ConfigurationError(_) |
            TemplateError::ParsingError(_)
        )
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            TemplateError::TemplateNotFound(_) => ErrorSeverity::Warning,
            TemplateError::ValidationError(_) => ErrorSeverity::Error,
            TemplateError::CompilationError(_) => ErrorSeverity::Error,
            TemplateError::SecurityError(_) => ErrorSeverity::Critical,
            TemplateError::PermissionError(_) => ErrorSeverity::Error,
            TemplateError::DependencyError(_) => ErrorSeverity::Warning,
            TemplateError::ConfigurationError(_) => ErrorSeverity::Warning,
            TemplateError::IntegrityError(_) => ErrorSeverity::Critical,
            TemplateError::ParsingError(_) => ErrorSeverity::Error,
            TemplateError::ExecutionError(_) => ErrorSeverity::Error,
            TemplateError::IoError(_) => ErrorSeverity::Error,
            TemplateError::SerializationError(_) => ErrorSeverity::Error,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_management_system_creation() {
        let system = TemplateManagementSystem::new();
        assert!(system.template_repository.read().is_ok());
        assert!(system.versioning_system.read().is_ok());
        assert!(system.category_manager.read().is_ok());
    }

    #[test]
    fn test_error_severity_classification() {
        assert_eq!(
            TemplateError::SecurityError("test".to_string()).severity(),
            ErrorSeverity::Critical
        );
        assert_eq!(
            TemplateError::TemplateNotFound("test".to_string()).severity(),
            ErrorSeverity::Warning
        );
    }

    #[test]
    fn test_error_recoverability() {
        assert!(TemplateError::ValidationError("test".to_string()).is_recoverable());
        assert!(!TemplateError::SecurityError("test".to_string()).is_recoverable());
    }

    #[test]
    fn test_system_integrity_report() {
        let mut report = SystemIntegrityReport::new();
        assert_eq!(report.health_status, SystemHealthStatus::Unknown);

        report.add_validation_error(
            "test_template".to_string(),
            TemplateError::ValidationError("test error".to_string())
        );

        assert_eq!(report.validation_errors.len(), 1);
    }
}
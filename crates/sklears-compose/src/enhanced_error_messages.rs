//! Enhanced error messages with actionable suggestions and context-aware diagnostics
//!
//! This module provides comprehensive error enhancement capabilities for ML pipelines including:
//! - Context-aware error messages with relevant diagnostic information
//! - Actionable suggestions for fixing common issues
//! - Learning-based error pattern recognition and resolution
//! - Error recovery strategies and automated fixes
//! - Comprehensive error reporting and analysis
//! - Integration with pipeline debugging and monitoring systems

use crate::error::{Result, SklearsComposeError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Enhanced error message system for ML pipelines
#[derive(Debug)]
pub struct ErrorMessageEnhancer {
    /// Error pattern analyzer for learning from historical errors
    pattern_analyzer: Arc<RwLock<ErrorPatternAnalyzer>>,

    /// Context collector for gathering relevant diagnostic information
    context_collector: Arc<RwLock<ErrorContextCollector>>,

    /// Suggestion engine for generating actionable recommendations
    suggestion_engine: Arc<RwLock<SuggestionEngine>>,

    /// Recovery advisor for automated error recovery strategies
    recovery_advisor: Arc<RwLock<RecoveryAdvisor>>,

    /// Error formatter for creating user-friendly error messages
    error_formatter: Arc<RwLock<ErrorFormatter>>,

    /// Configuration for error enhancement behavior
    config: ErrorEnhancementConfig,
}

/// Configuration for error message enhancement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnhancementConfig {
    /// Enable pattern-based error analysis
    pub enable_pattern_analysis: bool,

    /// Enable context collection for diagnostics
    pub enable_context_collection: bool,

    /// Enable automated suggestion generation
    pub enable_auto_suggestions: bool,

    /// Enable recovery strategy recommendations
    pub enable_recovery_strategies: bool,

    /// Maximum number of suggestions per error
    pub max_suggestions_per_error: usize,

    /// Confidence threshold for suggestions (0.0 to 1.0)
    pub suggestion_confidence_threshold: f64,

    /// Enable learning from error resolution outcomes
    pub enable_learning: bool,

    /// Error history size for pattern analysis
    pub error_history_size: usize,

    /// Enable detailed diagnostic information
    pub enable_detailed_diagnostics: bool,
}

/// Error pattern analyzer for learning from historical errors
#[derive(Debug)]
pub struct ErrorPatternAnalyzer {
    /// Historical error patterns
    error_patterns: HashMap<String, ErrorPattern>,

    /// Error frequency tracking
    error_frequency: HashMap<String, usize>,

    /// Resolution success tracking
    resolution_success: HashMap<String, f64>,

    /// Pattern learning configuration
    config: PatternAnalysisConfig,
}

/// Error pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Pattern identifier
    pub pattern_id: String,

    /// Error type or category
    pub error_type: String,

    /// Common error message patterns
    pub message_patterns: Vec<String>,

    /// Associated context patterns
    pub context_patterns: Vec<ContextPattern>,

    /// Successful resolution strategies
    pub resolution_strategies: Vec<ResolutionStrategy>,

    /// Pattern frequency and recency
    pub frequency: usize,

    /// Last occurrence timestamp
    pub last_occurrence: SystemTime,

    /// Success rate of suggested fixes
    pub success_rate: f64,
}

/// Context pattern for error classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPattern {
    /// Context type (data, configuration, environment, etc.)
    pub context_type: ContextType,

    /// Pattern description
    pub pattern: String,

    /// Pattern matching confidence
    pub confidence: f64,
}

/// Types of error context
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextType {
    /// DataShape
    DataShape,
    /// DataType
    DataType,
    /// Configuration
    Configuration,
    /// Environment
    Environment,
    /// Dependencies
    Dependencies,
    /// Resources
    Resources,
    /// Pipeline
    Pipeline,
    /// Model
    Model,
    /// Performance
    Performance,
    /// Network
    Network,
}

/// Error resolution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionStrategy {
    /// Strategy identifier
    pub strategy_id: String,

    /// Strategy description
    pub description: String,

    /// Implementation steps
    pub steps: Vec<ResolutionStep>,

    /// Expected success rate
    pub success_rate: f64,

    /// Difficulty level
    pub difficulty: DifficultyLevel,

    /// Required expertise level
    pub expertise_level: ExpertiseLevel,
}

/// Individual resolution step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionStep {
    /// Step description
    pub description: String,

    /// Code example or command
    pub code_example: Option<String>,

    /// Documentation reference
    pub documentation_link: Option<String>,

    /// Validation method
    pub validation: Option<String>,
}

/// Difficulty level for resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyLevel {
    /// Trivial
    Trivial,
    /// Easy
    Easy,
    /// Medium
    Medium,
    /// Hard
    Hard,
    /// Expert
    Expert,
}

/// Required expertise level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    /// Beginner
    Beginner,
    /// Intermediate
    Intermediate,
    /// Advanced
    Advanced,
    /// Expert
    Expert,
}

/// Error context collector for diagnostic information
#[derive(Debug)]
pub struct ErrorContextCollector {
    /// Context providers for different types of information
    context_providers: HashMap<ContextType, Box<dyn ContextProvider>>,

    /// Cached context information
    context_cache: HashMap<String, EnhancedErrorContext>,

    /// Collection configuration
    config: ContextCollectionConfig,
}

/// Enhanced error context with comprehensive diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedErrorContext {
    /// Error occurrence timestamp
    pub timestamp: SystemTime,

    /// Pipeline context information
    pub pipeline_context: PipelineContext,

    /// Data context information
    pub data_context: DataContext,

    /// Environment context information
    pub environment_context: EnvironmentContext,

    /// Performance context information
    pub performance_context: PerformanceContext,

    /// Configuration context information
    pub configuration_context: ConfigurationContext,

    /// Error call stack
    pub call_stack: Vec<StackFrame>,

    /// Related errors and warnings
    pub related_issues: Vec<RelatedIssue>,
}

/// Pipeline execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineContext {
    /// Pipeline name
    pub pipeline_name: String,

    /// Current step
    pub current_step: String,

    /// Step index in pipeline
    pub step_index: usize,

    /// Total pipeline steps
    pub total_steps: usize,

    /// Previous successful steps
    pub completed_steps: Vec<String>,

    /// Pipeline configuration
    pub pipeline_config: HashMap<String, String>,
}

/// Data context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataContext {
    /// Input data shape
    pub input_shape: Vec<usize>,

    /// Input data type
    pub input_dtype: String,

    /// Expected data shape
    pub expected_shape: Option<Vec<usize>>,

    /// Expected data type
    pub expected_dtype: Option<String>,

    /// Data statistics
    pub data_statistics: DataStatistics,

    /// Missing value information
    pub missing_values: MissingValueInfo,

    /// Data quality metrics
    pub quality_metrics: DataQualityMetrics,
}

/// Missing value information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingValueInfo {
    /// Total missing values
    pub total_missing: usize,

    /// Missing values per feature
    pub missing_per_feature: Vec<usize>,

    /// Missing value patterns
    pub missing_patterns: Vec<String>,
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub quality_score: f64,

    /// Completeness score
    pub completeness: f64,

    /// Consistency score
    pub consistency: f64,

    /// Validity score
    pub validity: f64,

    /// Specific quality issues
    pub quality_issues: Vec<QualityIssue>,
}

/// Data quality issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: String,

    /// Issue description
    pub description: String,

    /// Severity level
    pub severity: SeverityLevel,

    /// Affected features or samples
    pub affected_elements: Vec<String>,
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Critical
    Critical,
}

/// Environment context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentContext {
    /// Operating system information
    pub os_info: String,

    /// Available memory
    pub available_memory: u64,

    /// CPU information
    pub cpu_info: String,

    /// Python/Rust version
    pub runtime_version: String,

    /// Installed packages and versions
    pub package_versions: HashMap<String, String>,

    /// Environment variables
    pub environment_variables: HashMap<String, String>,
}

/// Performance context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceContext {
    /// Memory usage at error time
    pub memory_usage: u64,

    /// CPU utilization
    pub cpu_utilization: f64,

    /// Execution time before error
    pub execution_time: Duration,

    /// Performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Performance bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck location
    pub location: String,

    /// Bottleneck type
    pub bottleneck_type: String,

    /// Impact level
    pub impact: f64,
}

/// Configuration context information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigurationContext {
    /// Model configuration
    pub model_config: HashMap<String, String>,

    /// Pipeline configuration
    pub pipeline_config: HashMap<String, String>,

    /// Training configuration
    pub training_config: HashMap<String, String>,

    /// Configuration validation results
    pub validation_results: Vec<ConfigValidationResult>,
}

/// Configuration validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigValidationResult {
    /// Configuration key
    pub config_key: String,

    /// Validation status
    pub is_valid: bool,

    /// Validation message
    pub message: String,

    /// Suggested value
    pub suggested_value: Option<String>,
}

/// Stack frame information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    /// Function or method name
    pub function_name: String,

    /// File name
    pub file_name: String,

    /// Line number
    pub line_number: usize,

    /// Module name
    pub module_name: String,
}

/// Related issue information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedIssue {
    /// Issue type
    pub issue_type: String,

    /// Issue message
    pub message: String,

    /// Relationship to main error
    pub relationship: IssueRelationship,
}

/// Relationship between issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueRelationship {
    /// CausedBy
    CausedBy,
    /// LeadsTo
    LeadsTo,
    /// RelatedTo
    RelatedTo,
    /// SimilarTo
    SimilarTo,
}

/// Suggestion engine for generating actionable recommendations
#[derive(Debug)]
pub struct SuggestionEngine {
    /// Suggestion generators for different error types
    generators: HashMap<String, Box<dyn SuggestionGenerator>>,

    /// Learning model for suggestion ranking
    learning_model: Option<SuggestionRankingModel>,

    /// Suggestion cache for performance
    suggestion_cache: HashMap<String, Vec<ActionableSuggestion>>,

    /// Configuration
    config: SuggestionEngineConfig,
}

/// Actionable suggestion for error resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableSuggestion {
    /// Suggestion identifier
    pub suggestion_id: String,

    /// Suggestion title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Implementation steps
    pub implementation_steps: Vec<ImplementationStep>,

    /// Code examples
    pub code_examples: Vec<CodeExample>,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Estimated time to implement
    pub estimated_time: Duration,

    /// Success probability
    pub success_probability: f64,

    /// Required expertise level
    pub expertise_level: ExpertiseLevel,

    /// Dependencies or prerequisites
    pub prerequisites: Vec<String>,

    /// Validation method
    pub validation_method: Option<String>,

    /// Follow-up suggestions
    pub follow_up_suggestions: Vec<String>,
}

/// Implementation step for suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationStep {
    /// Step number
    pub step_number: usize,

    /// Step description
    pub description: String,

    /// Code snippet
    pub code_snippet: Option<String>,

    /// Command to run
    pub command: Option<String>,

    /// Expected outcome
    pub expected_outcome: String,

    /// Validation check
    pub validation_check: Option<String>,
}

/// Code example for suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Programming language
    pub language: String,

    /// Code snippet
    pub code: String,

    /// Description
    pub description: String,

    /// File path (if applicable)
    pub file_path: Option<String>,
}

/// Recovery advisor for automated error recovery strategies
#[derive(Debug)]
pub struct RecoveryAdvisor {
    /// Recovery strategies by error type
    recovery_strategies: HashMap<String, Vec<RecoveryStrategy>>,

    /// Automatic recovery implementations
    auto_recovery: HashMap<String, Box<dyn AutoRecoveryHandler>>,

    /// Recovery history and success rates
    recovery_history: VecDeque<RecoveryAttempt>,

    /// Configuration
    config: RecoveryConfig,
}

/// Recovery strategy for errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStrategy {
    /// Strategy identifier
    pub strategy_id: String,

    /// Strategy name
    pub name: String,

    /// Strategy description
    pub description: String,

    /// Recovery actions
    pub actions: Vec<RecoveryAction>,

    /// Automatic or manual recovery
    pub is_automatic: bool,

    /// Success rate
    pub success_rate: f64,

    /// Risk level
    pub risk_level: RiskLevel,
}

/// Recovery action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAction {
    /// Action type
    pub action_type: RecoveryActionType,

    /// Action description
    pub description: String,

    /// Parameters for the action
    pub parameters: HashMap<String, String>,

    /// Rollback action (if applicable)
    pub rollback_action: Option<String>,
}

/// Types of recovery actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryActionType {
    /// DataTransformation
    DataTransformation,
    /// ConfigurationAdjustment
    ConfigurationAdjustment,
    /// ModelReset
    ModelReset,
    /// ResourceReallocation
    ResourceReallocation,
    /// ParameterTuning
    ParameterTuning,
    /// FallbackStrategy
    FallbackStrategy,
    /// Manual
    Manual,
}

/// Risk levels for recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Safe
    Safe,
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Dangerous
    Dangerous,
}

/// Recovery attempt record
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    /// Error that triggered recovery
    pub error_id: String,

    /// Recovery strategy used
    pub strategy_id: String,

    /// Recovery actions taken
    pub actions_taken: Vec<String>,

    /// Recovery outcome
    pub outcome: RecoveryOutcome,

    /// Time taken for recovery
    pub recovery_time: Duration,

    /// Timestamp
    pub timestamp: SystemTime,
}

/// Recovery attempt outcome
#[derive(Debug, Clone)]
pub enum RecoveryOutcome {
    /// Success
    Success,
    /// PartialSuccess
    PartialSuccess,
    /// Failure
    Failure,
    /// Manual
    Manual,
}

/// Error formatter for user-friendly error messages
#[derive(Debug)]
pub struct ErrorFormatter {
    /// Format templates for different error types
    format_templates: HashMap<String, ErrorTemplate>,

    /// Localization support
    localization: HashMap<String, HashMap<String, String>>,

    /// Configuration
    config: FormatterConfig,
}

/// Error message template
#[derive(Debug, Clone)]
pub struct ErrorTemplate {
    /// Template identifier
    pub template_id: String,

    /// Main error message template
    pub message_template: String,

    /// Context sections to include
    pub context_sections: Vec<ContextSection>,

    /// Suggestion formatting
    pub suggestion_format: SuggestionFormat,

    /// Output format (text, HTML, markdown)
    pub output_format: OutputFormat,
}

/// Context section in error message
#[derive(Debug, Clone)]
pub struct ContextSection {
    /// Section title
    pub title: String,

    /// Section content template
    pub content_template: String,

    /// Whether section is optional
    pub optional: bool,
}

/// Suggestion formatting options
#[derive(Debug, Clone)]
pub struct SuggestionFormat {
    /// Maximum suggestions to show
    pub max_suggestions: usize,

    /// Include code examples
    pub include_code_examples: bool,

    /// Include confidence scores
    pub include_confidence_scores: bool,

    /// Include time estimates
    pub include_time_estimates: bool,
}

/// Output format for error messages
#[derive(Debug, Clone)]
pub enum OutputFormat {
    /// PlainText
    PlainText,
    /// Markdown
    Markdown,
    /// Html
    Html,
    /// Json
    Json,
    /// Yaml
    Yaml,
}

/// Enhanced error message result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedErrorMessage {
    /// Original error message
    pub original_error: String,

    /// Enhanced error message
    pub enhanced_message: String,

    /// Error classification
    pub error_classification: ErrorClassification,

    /// Diagnostic information
    pub diagnostics: EnhancedErrorContext,

    /// Actionable suggestions
    pub suggestions: Vec<ActionableSuggestion>,

    /// Recovery strategies
    pub recovery_strategies: Vec<RecoveryStrategy>,

    /// Related documentation
    pub documentation_links: Vec<DocumentationLink>,

    /// Similar issues
    pub similar_issues: Vec<SimilarIssue>,
}

/// Error classification information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorClassification {
    /// Primary error category
    pub category: ErrorCategory,

    /// Error severity
    pub severity: SeverityLevel,

    /// Error frequency
    pub frequency: ErrorFrequency,

    /// Resolution difficulty
    pub resolution_difficulty: DifficultyLevel,
}

/// Error categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// DataError
    DataError,
    /// ConfigurationError
    ConfigurationError,
    /// EnvironmentError
    EnvironmentError,
    /// ModelError
    ModelError,
    /// PipelineError
    PipelineError,
    /// PerformanceError
    PerformanceError,
    /// NetworkError
    NetworkError,
    /// SecurityError
    SecurityError,
    /// Unknown
    Unknown,
}

/// Error frequency classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorFrequency {
    /// Rare
    Rare,
    /// Occasional
    Occasional,
    /// Common
    Common,
    /// Frequent
    Frequent,
    /// Constant
    Constant,
}

/// Documentation link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationLink {
    /// Link title
    pub title: String,

    /// URL
    pub url: String,

    /// Link description
    pub description: String,

    /// Relevance score
    pub relevance: f64,
}

/// Similar issue information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarIssue {
    /// Issue identifier
    pub issue_id: String,

    /// Issue description
    pub description: String,

    /// Similarity score
    pub similarity: f64,

    /// Resolution information
    pub resolution: Option<String>,
}

/// Data statistics for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatistics {
    /// Number of samples
    pub n_samples: usize,

    /// Number of features
    pub n_features: usize,

    /// Mean values per feature
    pub means: Vec<f64>,

    /// Standard deviations per feature
    pub stds: Vec<f64>,

    /// Minimum values per feature
    pub mins: Vec<f64>,

    /// Maximum values per feature
    pub maxs: Vec<f64>,
}

// Configuration structures
#[derive(Debug, Clone)]
pub struct PatternAnalysisConfig {
    pub max_patterns: usize,
    pub pattern_similarity_threshold: f64,
    pub learning_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ContextCollectionConfig {
    pub enable_detailed_context: bool,
    pub max_context_size: usize,
    pub context_timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct SuggestionEngineConfig {
    pub max_suggestions: usize,
    pub confidence_threshold: f64,
    pub enable_machine_learning: bool,
}

#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    pub enable_auto_recovery: bool,
    pub max_recovery_attempts: usize,
    pub recovery_timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct FormatterConfig {
    pub default_language: String,
    pub include_technical_details: bool,
    pub max_message_length: usize,
}

/// Trait for context providers
pub trait ContextProvider: std::fmt::Debug + Send + Sync {
    fn collect_context(&self, error: &SklearsComposeError) -> Result<HashMap<String, String>>;
    fn context_type(&self) -> ContextType;
}

/// Trait for suggestion generators
pub trait SuggestionGenerator: std::fmt::Debug + Send + Sync {
    fn generate_suggestions(
        &self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
    ) -> Result<Vec<ActionableSuggestion>>;
    fn error_types(&self) -> Vec<String>;
}

/// Trait for automatic recovery handlers
pub trait AutoRecoveryHandler: std::fmt::Debug + Send + Sync {
    fn can_recover(&self, error: &SklearsComposeError) -> bool;
    fn attempt_recovery(
        &self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
    ) -> Result<RecoveryOutcome>;
}

/// Machine learning model for suggestion ranking
#[derive(Debug)]
pub struct SuggestionRankingModel {
    /// Feature weights for ranking
    feature_weights: HashMap<String, f64>,

    /// Historical success rates
    success_rates: HashMap<String, f64>,

    /// Model parameters
    parameters: ModelParameters,
}

#[derive(Debug, Clone)]
pub struct ModelParameters {
    pub learning_rate: f64,
    pub regularization: f64,
    pub feature_count: usize,
}

impl Default for ErrorEnhancementConfig {
    fn default() -> Self {
        Self {
            enable_pattern_analysis: true,
            enable_context_collection: true,
            enable_auto_suggestions: true,
            enable_recovery_strategies: true,
            max_suggestions_per_error: 5,
            suggestion_confidence_threshold: 0.7,
            enable_learning: true,
            error_history_size: 1000,
            enable_detailed_diagnostics: true,
        }
    }
}

impl Default for ErrorMessageEnhancer {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorMessageEnhancer {
    /// Create a new error message enhancer with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(ErrorEnhancementConfig::default())
    }

    /// Create a new error message enhancer with custom configuration
    #[must_use]
    pub fn with_config(config: ErrorEnhancementConfig) -> Self {
        let pattern_config = PatternAnalysisConfig {
            max_patterns: 500,
            pattern_similarity_threshold: 0.8,
            learning_rate: 0.01,
        };

        let context_config = ContextCollectionConfig {
            enable_detailed_context: config.enable_detailed_diagnostics,
            max_context_size: 10_000,
            context_timeout: Duration::from_secs(5),
        };

        let suggestion_config = SuggestionEngineConfig {
            max_suggestions: config.max_suggestions_per_error,
            confidence_threshold: config.suggestion_confidence_threshold,
            enable_machine_learning: config.enable_learning,
        };

        let recovery_config = RecoveryConfig {
            enable_auto_recovery: config.enable_recovery_strategies,
            max_recovery_attempts: 3,
            recovery_timeout: Duration::from_secs(30),
        };

        let formatter_config = FormatterConfig {
            default_language: "en".to_string(),
            include_technical_details: config.enable_detailed_diagnostics,
            max_message_length: 5000,
        };

        Self {
            pattern_analyzer: Arc::new(RwLock::new(ErrorPatternAnalyzer::new(pattern_config))),
            context_collector: Arc::new(RwLock::new(ErrorContextCollector::new(context_config))),
            suggestion_engine: Arc::new(RwLock::new(SuggestionEngine::new(suggestion_config))),
            recovery_advisor: Arc::new(RwLock::new(RecoveryAdvisor::new(recovery_config))),
            error_formatter: Arc::new(RwLock::new(ErrorFormatter::new(formatter_config))),
            config,
        }
    }

    /// Enhance an error with comprehensive diagnostics and suggestions
    pub fn enhance_error(&self, error: &SklearsComposeError) -> Result<EnhancedErrorMessage> {
        // Collect comprehensive context
        let context = if self.config.enable_context_collection {
            self.context_collector
                .write()
                .unwrap()
                .collect_context(error)?
        } else {
            EnhancedErrorContext::default()
        };

        // Analyze error patterns
        let classification = if self.config.enable_pattern_analysis {
            self.pattern_analyzer
                .write()
                .unwrap()
                .analyze_error(error, &context)?
        } else {
            ErrorClassification::default()
        };

        // Generate actionable suggestions
        let suggestions = if self.config.enable_auto_suggestions {
            self.suggestion_engine
                .read()
                .unwrap()
                .generate_suggestions(error, &context)?
        } else {
            Vec::new()
        };

        // Generate recovery strategies
        let recovery_strategies = if self.config.enable_recovery_strategies {
            self.recovery_advisor
                .read()
                .unwrap()
                .generate_recovery_strategies(error, &context)?
        } else {
            Vec::new()
        };

        // Format enhanced error message
        let enhanced_message =
            self.error_formatter
                .read()
                .unwrap()
                .format_error(error, &context, &suggestions)?;

        // Generate documentation links
        let documentation_links = self.generate_documentation_links(error, &classification)?;

        // Find similar issues
        let similar_issues = self.find_similar_issues(error, &context)?;

        Ok(EnhancedErrorMessage {
            original_error: error.to_string(),
            enhanced_message,
            error_classification: classification,
            diagnostics: context,
            suggestions,
            recovery_strategies,
            documentation_links,
            similar_issues,
        })
    }

    /// Attempt automatic error recovery
    pub fn attempt_recovery(&self, error: &SklearsComposeError) -> Result<RecoveryOutcome> {
        let context = self
            .context_collector
            .write()
            .unwrap()
            .collect_context(error)?;
        self.recovery_advisor
            .write()
            .unwrap()
            .attempt_auto_recovery(error, &context)
    }

    /// Learn from error resolution outcomes
    pub fn learn_from_resolution(
        &self,
        error: &SklearsComposeError,
        suggestion_id: &str,
        outcome: bool,
    ) -> Result<()> {
        if self.config.enable_learning {
            self.pattern_analyzer
                .write()
                .unwrap()
                .update_success_rate(suggestion_id, outcome)?;
            self.suggestion_engine
                .write()
                .unwrap()
                .update_ranking_model(suggestion_id, outcome)?;
        }
        Ok(())
    }

    /// Export error enhancement statistics
    pub fn export_statistics(&self) -> Result<ErrorEnhancementStatistics> {
        let pattern_stats = self.pattern_analyzer.read().unwrap().get_statistics();
        let suggestion_stats = self.suggestion_engine.read().unwrap().get_statistics();
        let recovery_stats = self.recovery_advisor.read().unwrap().get_statistics();

        Ok(ErrorEnhancementStatistics {
            total_errors_analyzed: pattern_stats.total_patterns,
            suggestions_generated: suggestion_stats.total_suggestions,
            recovery_attempts: recovery_stats.total_attempts,
            success_rate: recovery_stats.success_rate,
            pattern_accuracy: pattern_stats.accuracy,
            suggestion_confidence: suggestion_stats.average_confidence,
        })
    }

    // Private helper methods
    fn generate_documentation_links(
        &self,
        error: &SklearsComposeError,
        classification: &ErrorClassification,
    ) -> Result<Vec<DocumentationLink>> {
        let mut links = Vec::new();

        // Generate links based on error type and classification
        match classification.category {
            ErrorCategory::DataError => {
                links.push(DocumentationLink {
                    title: "Data Input Guidelines".to_string(),
                    url: "https://docs.rs/sklears-compose/data-input".to_string(),
                    description: "Guide for proper data formatting and validation".to_string(),
                    relevance: 0.9,
                });
            }
            ErrorCategory::ConfigurationError => {
                links.push(DocumentationLink {
                    title: "Configuration Reference".to_string(),
                    url: "https://docs.rs/sklears-compose/configuration".to_string(),
                    description: "Complete reference for configuration options".to_string(),
                    relevance: 0.95,
                });
            }
            _ => {}
        }

        Ok(links)
    }

    fn find_similar_issues(
        &self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
    ) -> Result<Vec<SimilarIssue>> {
        // This would typically search a database of known issues
        // For now, returning empty vector
        Ok(Vec::new())
    }
}

/// Error enhancement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnhancementStatistics {
    pub total_errors_analyzed: usize,
    pub suggestions_generated: usize,
    pub recovery_attempts: usize,
    pub success_rate: f64,
    pub pattern_accuracy: f64,
    pub suggestion_confidence: f64,
}

// Implementation blocks for supporting components

impl ErrorPatternAnalyzer {
    fn new(config: PatternAnalysisConfig) -> Self {
        Self {
            error_patterns: HashMap::new(),
            error_frequency: HashMap::new(),
            resolution_success: HashMap::new(),
            config,
        }
    }

    fn analyze_error(
        &mut self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
    ) -> Result<ErrorClassification> {
        let error_type = format!("{error:?}");

        // Update frequency tracking
        *self.error_frequency.entry(error_type.clone()).or_insert(0) += 1;

        // Classify error based on patterns
        let category = self.classify_error_category(error);
        let severity = self.assess_severity(error, context);
        let frequency = self.assess_frequency(&error_type);
        let resolution_difficulty = self.assess_resolution_difficulty(error);

        Ok(ErrorClassification {
            category,
            severity,
            frequency,
            resolution_difficulty,
        })
    }

    fn classify_error_category(&self, error: &SklearsComposeError) -> ErrorCategory {
        match error {
            SklearsComposeError::InvalidData { .. } => ErrorCategory::DataError,
            SklearsComposeError::InvalidConfiguration(_) => ErrorCategory::ConfigurationError,
            SklearsComposeError::InvalidOperation(_) => ErrorCategory::PipelineError,
            SklearsComposeError::Serialization(_) | SklearsComposeError::Io(_) => {
                ErrorCategory::EnvironmentError
            }
            SklearsComposeError::Core(_) => ErrorCategory::ModelError,
            SklearsComposeError::Other(reason) => {
                let lower = reason.to_lowercase();
                if lower.contains("shape") || lower.contains("dimension") {
                    ErrorCategory::DataError
                } else if lower.contains("config") || lower.contains("parameter") {
                    ErrorCategory::ConfigurationError
                } else if lower.contains("memory") || lower.contains("resource") {
                    ErrorCategory::EnvironmentError
                } else if lower.contains("model") || lower.contains("fit") {
                    ErrorCategory::ModelError
                } else if lower.contains("pipeline") {
                    ErrorCategory::PipelineError
                } else {
                    ErrorCategory::Unknown
                }
            }
        }
    }

    fn assess_severity(
        &self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
    ) -> SeverityLevel {
        // Simple heuristic - would be more sophisticated in practice
        if context.performance_context.memory_usage > 1_000_000_000 {
            SeverityLevel::High
        } else {
            SeverityLevel::Medium
        }
    }

    fn assess_frequency(&self, error_type: &str) -> ErrorFrequency {
        let count = self.error_frequency.get(error_type).unwrap_or(&0);
        match count {
            0..=1 => ErrorFrequency::Rare,
            2..=5 => ErrorFrequency::Occasional,
            6..=15 => ErrorFrequency::Common,
            16..=50 => ErrorFrequency::Frequent,
            _ => ErrorFrequency::Constant,
        }
    }

    fn assess_resolution_difficulty(&self, error: &SklearsComposeError) -> DifficultyLevel {
        // Simple heuristic based on error type
        let error_str = error.to_string().to_lowercase();

        if error_str.contains("simple") || error_str.contains("basic") {
            DifficultyLevel::Easy
        } else if error_str.contains("complex") || error_str.contains("advanced") {
            DifficultyLevel::Hard
        } else {
            DifficultyLevel::Medium
        }
    }

    fn update_success_rate(&mut self, suggestion_id: &str, success: bool) -> Result<()> {
        let current_rate = self.resolution_success.get(suggestion_id).unwrap_or(&0.5);
        let new_rate = if success {
            (current_rate + 0.1).min(1.0)
        } else {
            (current_rate - 0.1).max(0.0)
        };
        self.resolution_success
            .insert(suggestion_id.to_string(), new_rate);
        Ok(())
    }

    fn get_statistics(&self) -> PatternStatistics {
        /// PatternStatistics
        PatternStatistics {
            total_patterns: self.error_patterns.len(),
            accuracy: 0.85, // Would be calculated based on historical data
        }
    }
}

#[derive(Debug)]
struct PatternStatistics {
    total_patterns: usize,
    accuracy: f64,
}

impl ErrorContextCollector {
    fn new(config: ContextCollectionConfig) -> Self {
        let mut context_providers: HashMap<ContextType, Box<dyn ContextProvider>> = HashMap::new();

        // Add default context providers
        context_providers.insert(
            ContextType::Environment,
            Box::new(EnvironmentContextProvider::new()),
        );
        context_providers.insert(
            ContextType::Performance,
            Box::new(PerformanceContextProvider::new()),
        );

        Self {
            context_providers,
            context_cache: HashMap::new(),
            config,
        }
    }

    fn collect_context(&mut self, error: &SklearsComposeError) -> Result<EnhancedErrorContext> {
        // Create default context
        let context = EnhancedErrorContext {
            timestamp: SystemTime::now(),
            pipeline_context: PipelineContext::default(),
            data_context: DataContext::default(),
            environment_context: EnvironmentContext::default(),
            performance_context: PerformanceContext::default(),
            configuration_context: ConfigurationContext::default(),
            call_stack: Vec::new(),
            related_issues: Vec::new(),
        };

        // Collect additional context from providers
        for (context_type, provider) in &self.context_providers {
            if let Ok(additional_context) = provider.collect_context(error) {
                // Merge additional context - implementation would be more sophisticated
            }
        }

        Ok(context)
    }
}

impl SuggestionEngine {
    fn new(config: SuggestionEngineConfig) -> Self {
        let mut generators: HashMap<String, Box<dyn SuggestionGenerator>> = HashMap::new();

        // Add default suggestion generators
        generators.insert(
            "DataError".to_string(),
            Box::new(DataErrorSuggestionGenerator::new()),
        );
        generators.insert(
            "ConfigurationError".to_string(),
            Box::new(ConfigurationErrorSuggestionGenerator::new()),
        );

        Self {
            generators,
            learning_model: None,
            suggestion_cache: HashMap::new(),
            config,
        }
    }

    fn generate_suggestions(
        &self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
    ) -> Result<Vec<ActionableSuggestion>> {
        let error_type = format!("{error:?}");

        let mut suggestions = Vec::new();

        // Generate suggestions from relevant generators
        for (generator_type, generator) in &self.generators {
            if error_type.contains(generator_type) {
                if let Ok(mut generated) = generator.generate_suggestions(error, context) {
                    suggestions.append(&mut generated);
                }
            }
        }

        // Filter by confidence threshold
        suggestions.retain(|s| s.confidence >= self.config.confidence_threshold);

        // Limit number of suggestions
        suggestions.truncate(self.config.max_suggestions);

        Ok(suggestions)
    }

    fn update_ranking_model(&mut self, suggestion_id: &str, success: bool) -> Result<()> {
        // Update machine learning model for suggestion ranking
        // Implementation would involve updating model weights
        Ok(())
    }

    fn get_statistics(&self) -> SuggestionStatistics {
        /// SuggestionStatistics
        SuggestionStatistics {
            total_suggestions: self.suggestion_cache.len(),
            average_confidence: 0.8, // Would be calculated
        }
    }
}

#[derive(Debug)]
struct SuggestionStatistics {
    total_suggestions: usize,
    average_confidence: f64,
}

impl RecoveryAdvisor {
    fn new(config: RecoveryConfig) -> Self {
        Self {
            recovery_strategies: HashMap::new(),
            auto_recovery: HashMap::new(),
            recovery_history: VecDeque::with_capacity(1000),
            config,
        }
    }

    fn generate_recovery_strategies(
        &self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
    ) -> Result<Vec<RecoveryStrategy>> {
        let error_type = format!("{error:?}");

        if let Some(strategies) = self.recovery_strategies.get(&error_type) {
            Ok(strategies.clone())
        } else {
            // Generate default recovery strategies
            Ok(vec![RecoveryStrategy {
                strategy_id: "manual_review".to_string(),
                name: "Manual Review".to_string(),
                description: "Manually review the error and fix the underlying issue".to_string(),
                actions: vec![RecoveryAction {
                    action_type: RecoveryActionType::Manual,
                    description: "Review error message and fix the issue".to_string(),
                    parameters: HashMap::new(),
                    rollback_action: None,
                }],
                is_automatic: false,
                success_rate: 0.9,
                risk_level: RiskLevel::Safe,
            }])
        }
    }

    fn attempt_auto_recovery(
        &mut self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
    ) -> Result<RecoveryOutcome> {
        if !self.config.enable_auto_recovery {
            return Ok(RecoveryOutcome::Manual);
        }

        let error_type = format!("{error:?}");

        if let Some(handler) = self.auto_recovery.get(&error_type) {
            if handler.can_recover(error) {
                let start_time = Instant::now();
                let outcome = handler.attempt_recovery(error, context)?;

                // Record recovery attempt
                let attempt = RecoveryAttempt {
                    error_id: format!(
                        "err_{}",
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis()
                    ),
                    strategy_id: "auto_recovery".to_string(),
                    actions_taken: vec!["automatic_recovery".to_string()],
                    outcome: outcome.clone(),
                    recovery_time: start_time.elapsed(),
                    timestamp: SystemTime::now(),
                };

                self.recovery_history.push_back(attempt);
                if self.recovery_history.len() > 1000 {
                    self.recovery_history.pop_front();
                }

                return Ok(outcome);
            }
        }

        Ok(RecoveryOutcome::Manual)
    }

    fn get_statistics(&self) -> RecoveryStatistics {
        let total_attempts = self.recovery_history.len();
        let successful_attempts = self
            .recovery_history
            .iter()
            .filter(|attempt| matches!(attempt.outcome, RecoveryOutcome::Success))
            .count();

        let success_rate = if total_attempts > 0 {
            successful_attempts as f64 / total_attempts as f64
        } else {
            0.0
        };

        /// RecoveryStatistics
        RecoveryStatistics {
            total_attempts,
            success_rate,
        }
    }
}

#[derive(Debug)]
struct RecoveryStatistics {
    total_attempts: usize,
    success_rate: f64,
}

impl ErrorFormatter {
    fn new(config: FormatterConfig) -> Self {
        let mut format_templates = HashMap::new();

        // Add default templates
        format_templates.insert("default".to_string(), ErrorTemplate::default());

        Self {
            format_templates,
            localization: HashMap::new(),
            config,
        }
    }

    fn format_error(
        &self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
        suggestions: &[ActionableSuggestion],
    ) -> Result<String> {
        let template = self.format_templates.get("default").unwrap();

        let mut formatted = format!("ERROR: {error}\n\n");

        // Add context information
        formatted.push_str("CONTEXT:\n");
        formatted.push_str(&format!(
            "  Pipeline: {}\n",
            context.pipeline_context.pipeline_name
        ));
        formatted.push_str(&format!(
            "  Step: {}\n",
            context.pipeline_context.current_step
        ));
        formatted.push_str(&format!(
            "  Data Shape: {:?}\n",
            context.data_context.input_shape
        ));
        formatted.push('\n');

        // Add suggestions
        if !suggestions.is_empty() {
            formatted.push_str("SUGGESTIONS:\n");
            for (i, suggestion) in suggestions.iter().enumerate() {
                formatted.push_str(&format!(
                    "{}. {} (confidence: {:.1}%)\n",
                    i + 1,
                    suggestion.title,
                    suggestion.confidence * 100.0
                ));
                formatted.push_str(&format!("   {}\n", suggestion.description));
            }
        }

        Ok(formatted)
    }
}

// Default implementations

impl Default for EnhancedErrorContext {
    fn default() -> Self {
        Self {
            timestamp: SystemTime::now(),
            pipeline_context: PipelineContext::default(),
            data_context: DataContext::default(),
            environment_context: EnvironmentContext::default(),
            performance_context: PerformanceContext::default(),
            configuration_context: ConfigurationContext::default(),
            call_stack: Vec::new(),
            related_issues: Vec::new(),
        }
    }
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self {
            pipeline_name: "unknown".to_string(),
            current_step: "unknown".to_string(),
            step_index: 0,
            total_steps: 0,
            completed_steps: Vec::new(),
            pipeline_config: HashMap::new(),
        }
    }
}

impl Default for DataContext {
    fn default() -> Self {
        Self {
            input_shape: Vec::new(),
            input_dtype: "unknown".to_string(),
            expected_shape: None,
            expected_dtype: None,
            data_statistics: DataStatistics {
                n_samples: 0,
                n_features: 0,
                means: Vec::new(),
                stds: Vec::new(),
                mins: Vec::new(),
                maxs: Vec::new(),
            },
            missing_values: MissingValueInfo {
                total_missing: 0,
                missing_per_feature: Vec::new(),
                missing_patterns: Vec::new(),
            },
            quality_metrics: DataQualityMetrics {
                quality_score: 0.0,
                completeness: 0.0,
                consistency: 0.0,
                validity: 0.0,
                quality_issues: Vec::new(),
            },
        }
    }
}

impl Default for EnvironmentContext {
    fn default() -> Self {
        Self {
            os_info: "unknown".to_string(),
            available_memory: 0,
            cpu_info: "unknown".to_string(),
            runtime_version: "unknown".to_string(),
            package_versions: HashMap::new(),
            environment_variables: HashMap::new(),
        }
    }
}

impl Default for PerformanceContext {
    fn default() -> Self {
        Self {
            memory_usage: 0,
            cpu_utilization: 0.0,
            execution_time: Duration::from_secs(0),
            bottlenecks: Vec::new(),
        }
    }
}

impl Default for ErrorClassification {
    fn default() -> Self {
        Self {
            category: ErrorCategory::Unknown,
            severity: SeverityLevel::Medium,
            frequency: ErrorFrequency::Occasional,
            resolution_difficulty: DifficultyLevel::Medium,
        }
    }
}

impl Default for ErrorTemplate {
    fn default() -> Self {
        Self {
            template_id: "default".to_string(),
            message_template: "{error}\n\nContext: {context}\n\nSuggestions: {suggestions}"
                .to_string(),
            context_sections: Vec::new(),
            suggestion_format: SuggestionFormat {
                max_suggestions: 5,
                include_code_examples: true,
                include_confidence_scores: true,
                include_time_estimates: false,
            },
            output_format: OutputFormat::PlainText,
        }
    }
}

// Example context providers

#[derive(Debug)]
struct EnvironmentContextProvider;

impl EnvironmentContextProvider {
    fn new() -> Self {
        Self
    }
}

impl ContextProvider for EnvironmentContextProvider {
    fn collect_context(&self, _error: &SklearsComposeError) -> Result<HashMap<String, String>> {
        let mut context = HashMap::new();
        context.insert("os".to_string(), std::env::consts::OS.to_string());
        context.insert("arch".to_string(), std::env::consts::ARCH.to_string());
        Ok(context)
    }

    fn context_type(&self) -> ContextType {
        ContextType::Environment
    }
}

#[derive(Debug)]
struct PerformanceContextProvider;

impl PerformanceContextProvider {
    fn new() -> Self {
        Self
    }
}

impl ContextProvider for PerformanceContextProvider {
    fn collect_context(&self, _error: &SklearsComposeError) -> Result<HashMap<String, String>> {
        let mut context = HashMap::new();
        // Would collect actual performance metrics
        context.insert("memory_usage".to_string(), "unknown".to_string());
        Ok(context)
    }

    fn context_type(&self) -> ContextType {
        ContextType::Performance
    }
}

// Example suggestion generators

#[derive(Debug)]
struct DataErrorSuggestionGenerator;

impl DataErrorSuggestionGenerator {
    fn new() -> Self {
        Self
    }
}

impl SuggestionGenerator for DataErrorSuggestionGenerator {
    fn generate_suggestions(
        &self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
    ) -> Result<Vec<ActionableSuggestion>> {
        let mut suggestions = Vec::new();

        let error_str = error.to_string().to_lowercase();

        if error_str.contains("shape") {
            suggestions.push(ActionableSuggestion {
                suggestion_id: "fix_data_shape".to_string(),
                title: "Fix Data Shape Mismatch".to_string(),
                description: "The input data shape doesn't match the expected shape. Check your data preprocessing steps.".to_string(),
                implementation_steps: vec![
                    /// ImplementationStep
                    ImplementationStep {
                        step_number: 1,
                        description: "Check the shape of your input data".to_string(),
                        code_snippet: Some("println!(\"Data shape: {:?}\", data.shape());".to_string()),
                        command: None,
                        expected_outcome: "Display the actual data shape".to_string(),
                        validation_check: None,
                    }
                ],
                code_examples: vec![
                    /// CodeExample
                    CodeExample {
                        language: "rust".to_string(),
                        code: "let data = data.into_shape((n_samples, n_features))?;".to_string(),
                        description: "Reshape data to correct dimensions".to_string(),
                        file_path: None,
                    }
                ],
                confidence: 0.9,
                estimated_time: <Duration as DurationExt>::from_mins(5),
                success_probability: 0.85,
                expertise_level: ExpertiseLevel::Beginner,
                prerequisites: Vec::new(),
                validation_method: Some("Check that data.shape() matches expected shape".to_string()),
                follow_up_suggestions: Vec::new(),
            });
        }

        Ok(suggestions)
    }

    fn error_types(&self) -> Vec<String> {
        vec!["DataError".to_string()]
    }
}

#[derive(Debug)]
struct ConfigurationErrorSuggestionGenerator;

impl ConfigurationErrorSuggestionGenerator {
    fn new() -> Self {
        Self
    }
}

impl SuggestionGenerator for ConfigurationErrorSuggestionGenerator {
    fn generate_suggestions(
        &self,
        error: &SklearsComposeError,
        _context: &EnhancedErrorContext,
    ) -> Result<Vec<ActionableSuggestion>> {
        let mut suggestions = Vec::new();

        suggestions.push(ActionableSuggestion {
            suggestion_id: "check_configuration".to_string(),
            title: "Review Configuration Settings".to_string(),
            description: "Check your configuration parameters for correct values and types."
                .to_string(),
            implementation_steps: vec![ImplementationStep {
                step_number: 1,
                description: "Review configuration documentation".to_string(),
                code_snippet: None,
                command: None,
                expected_outcome: "Understanding of correct configuration format".to_string(),
                validation_check: None,
            }],
            code_examples: Vec::new(),
            confidence: 0.7,
            estimated_time: <Duration as DurationExt>::from_mins(10),
            success_probability: 0.8,
            expertise_level: ExpertiseLevel::Intermediate,
            prerequisites: Vec::new(),
            validation_method: None,
            follow_up_suggestions: Vec::new(),
        });

        Ok(suggestions)
    }

    fn error_types(&self) -> Vec<String> {
        vec!["ConfigurationError".to_string()]
    }
}

// Extension trait for Duration
trait DurationExt {
    fn from_mins(minutes: u64) -> Duration;
}

impl DurationExt for Duration {
    fn from_mins(minutes: u64) -> Duration {
        Duration::from_secs(minutes * 60)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_enhancer_creation() {
        let enhancer = ErrorMessageEnhancer::new();
        assert!(enhancer.config.enable_pattern_analysis);
    }

    #[test]
    fn test_error_enhancement() {
        let enhancer = ErrorMessageEnhancer::new();
        let error = SklearsComposeError::InvalidConfiguration("Test error".to_string());

        let result = enhancer.enhance_error(&error);
        assert!(result.is_ok());

        let enhanced = result.unwrap();
        assert!(!enhanced.enhanced_message.is_empty());
        assert!(!enhanced.original_error.is_empty());
    }

    #[test]
    fn test_context_collection() {
        let config = ContextCollectionConfig {
            enable_detailed_context: true,
            max_context_size: 1000,
            context_timeout: Duration::from_secs(1),
        };

        let mut collector = ErrorContextCollector::new(config);
        let error = SklearsComposeError::InvalidConfiguration("Test".to_string());

        let result = collector.collect_context(&error);
        assert!(result.is_ok());
    }

    #[test]
    fn test_suggestion_generation() {
        let config = SuggestionEngineConfig {
            max_suggestions: 3,
            confidence_threshold: 0.5,
            enable_machine_learning: false,
        };

        let engine = SuggestionEngine::new(config);
        let error = SklearsComposeError::InvalidData {
            reason: "shape mismatch".to_string(),
        };
        let context = EnhancedErrorContext::default();

        let result = engine.generate_suggestions(&error, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_classification() {
        let config = PatternAnalysisConfig {
            max_patterns: 100,
            pattern_similarity_threshold: 0.8,
            learning_rate: 0.01,
        };

        let mut analyzer = ErrorPatternAnalyzer::new(config);
        let error = SklearsComposeError::InvalidData {
            reason: "test".to_string(),
        };
        let context = EnhancedErrorContext::default();

        let result = analyzer.analyze_error(&error, &context);
        assert!(result.is_ok());

        let classification = result.unwrap();
        assert!(matches!(classification.category, ErrorCategory::DataError));
    }

    #[test]
    fn test_recovery_strategies() {
        let config = RecoveryConfig {
            enable_auto_recovery: true,
            max_recovery_attempts: 3,
            recovery_timeout: Duration::from_secs(10),
        };

        let advisor = RecoveryAdvisor::new(config);
        let error = SklearsComposeError::InvalidConfiguration("test".to_string());
        let context = EnhancedErrorContext::default();

        let result = advisor.generate_recovery_strategies(&error, &context);
        assert!(result.is_ok());

        let strategies = result.unwrap();
        assert!(!strategies.is_empty());
    }

    #[test]
    fn test_error_formatting() {
        let config = FormatterConfig {
            default_language: "en".to_string(),
            include_technical_details: true,
            max_message_length: 1000,
        };

        let formatter = ErrorFormatter::new(config);
        let error = SklearsComposeError::InvalidConfiguration("test error".to_string());
        let context = EnhancedErrorContext::default();
        let suggestions = Vec::new();

        let result = formatter.format_error(&error, &context, &suggestions);
        assert!(result.is_ok());

        let formatted = result.unwrap();
        assert!(formatted.contains("ERROR:"));
        assert!(formatted.contains("test error"));
    }

    #[test]
    fn test_learning_from_resolution() {
        let enhancer = ErrorMessageEnhancer::new();
        let error = SklearsComposeError::InvalidConfiguration("test".to_string());

        let result = enhancer.learn_from_resolution(&error, "test_suggestion", true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_statistics_export() {
        let enhancer = ErrorMessageEnhancer::new();
        let result = enhancer.export_statistics();
        assert!(result.is_ok());

        let stats = result.unwrap();
        assert!(stats.success_rate >= 0.0 && stats.success_rate <= 1.0);
    }

    #[test]
    fn test_context_providers() {
        let provider = EnvironmentContextProvider::new();
        let error = SklearsComposeError::InvalidConfiguration("test".to_string());

        let result = provider.collect_context(&error);
        assert!(result.is_ok());

        let context = result.unwrap();
        assert!(context.contains_key("os"));
        assert_eq!(provider.context_type(), ContextType::Environment);
    }

    #[test]
    fn test_suggestion_generators() {
        let generator = DataErrorSuggestionGenerator::new();
        let error = SklearsComposeError::InvalidData {
            reason: "shape mismatch".to_string(),
        };
        let context = EnhancedErrorContext::default();

        let result = generator.generate_suggestions(&error, &context);
        assert!(result.is_ok());

        let suggestions = result.unwrap();
        assert!(!suggestions.is_empty());
        assert_eq!(generator.error_types(), vec!["DataError".to_string()]);
    }
}

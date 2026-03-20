use serde::{Serialize, Deserialize};
use std::time::{Duration, SystemTime};
use std::collections::HashMap;

/// Comprehensive dependency resolution system for distributed failover operations
///
/// This module provides sophisticated dependency graph analysis, circular dependency detection,
/// resolution algorithms, and comprehensive validation systems for complex distributed scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyResolver {
    /// Graph representation of all dependencies
    pub dependency_graph: DependencyGraph,
    /// Available resolution algorithms for dependency ordering
    pub resolution_algorithms: Vec<ResolutionAlgorithm>,
    /// Specialized handler for circular dependency detection and resolution
    pub circular_dependency_handler: CircularDependencyHandler,
    /// Comprehensive validation system for dependency integrity
    pub dependency_validation: DependencyValidation,
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self {
            dependency_graph: DependencyGraph::default(),
            resolution_algorithms: vec![
                ResolutionAlgorithm::TopologicalSort,
                ResolutionAlgorithm::CriticalPath,
                ResolutionAlgorithm::LayeredApproach,
            ],
            circular_dependency_handler: CircularDependencyHandler::default(),
            dependency_validation: DependencyValidation::default(),
        }
    }
}

/// Advanced dependency graph with comprehensive metrics and analysis
pub struct DependencyGraph {
    /// Map of all dependency nodes indexed by ID
    pub nodes: HashMap<String, DependencyNode>,
    /// Collection of directed edges representing dependencies
    pub edges: Vec<DependencyEdge>,
    /// Computed metrics for graph complexity analysis
    pub graph_metrics: GraphMetrics,
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            graph_metrics: GraphMetrics::default(),
        }
    }
}

/// Individual dependency node with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyNode {
    /// Unique identifier for the dependency node
    pub node_id: String,
    /// Type classification of the dependency
    pub node_type: DependencyNodeType,
    /// Flexible property system for node-specific metadata
    pub node_properties: HashMap<String, String>,
    /// Execution requirements and constraints
    pub execution_requirements: ExecutionRequirements,
}

impl Default for DependencyNode {
    fn default() -> Self {
        Self {
            node_id: String::new(),
            node_type: DependencyNodeType::Service,
            node_properties: HashMap::new(),
            execution_requirements: ExecutionRequirements::default(),
        }
    }
}

/// Classification types for dependency nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyNodeType {
    /// Service-level dependency
    Service,
    /// Resource-level dependency (CPU, memory, storage)
    Resource,
    /// Configuration or parameter dependency
    Configuration,
    /// Data or state dependency
    Data,
    /// Network connectivity dependency
    Network,
    /// Custom dependency type with string identifier
    Custom(String),
}

/// Comprehensive execution requirements for dependency nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequirements {
    /// List of prerequisite nodes that must be satisfied
    pub prerequisites: Vec<String>,
    /// Resource allocation requirements
    pub resource_needs: HashMap<String, f64>,
    /// Temporal constraints for execution
    pub timing_constraints: Vec<TimingConstraint>,
    /// Quality and reliability requirements
    pub quality_requirements: Vec<QualityRequirement>,
}

impl Default for ExecutionRequirements {
    fn default() -> Self {
        Self {
            prerequisites: Vec::new(),
            resource_needs: HashMap::new(),
            timing_constraints: Vec::new(),
            quality_requirements: Vec::new(),
        }
    }
}

/// Timing constraints for dependency execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConstraint {
    /// Type of timing constraint
    pub constraint_type: TimingConstraintType,
    /// Primary constraint value
    pub constraint_value: Duration,
    /// Acceptable tolerance for the constraint
    pub constraint_tolerance: Duration,
}

/// Types of timing constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimingConstraintType {
    /// Maximum allowed execution duration
    MaxDuration,
    /// Minimum required execution duration
    MinDuration,
    /// Absolute deadline for completion
    Deadline,
    /// Specific start time requirement
    StartTime,
    /// Custom timing constraint
    Custom(String),
}

/// Quality and reliability requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirement {
    /// Type of quality requirement
    pub requirement_type: QualityRequirementType,
    /// Numeric threshold or target value
    pub requirement_value: f64,
    /// Method for measuring the requirement
    pub measurement_method: String,
}

/// Types of quality requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityRequirementType {
    /// System availability requirement
    Availability,
    /// Reliability and fault tolerance
    Reliability,
    /// Performance and responsiveness
    Performance,
    /// Security and access control
    Security,
    /// Accuracy and correctness
    Accuracy,
    /// Custom quality requirement
    Custom(String),
}

/// Directed edges representing dependencies between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Unique identifier for the edge
    pub edge_id: String,
    /// Source node in the dependency relationship
    pub source_node: String,
    /// Target node that depends on the source
    pub target_node: String,
    /// Type and nature of the dependency
    pub dependency_type: DependencyType,
    /// Strength or criticality of the dependency (0.0 to 1.0)
    pub dependency_strength: f64,
}

/// Types of dependency relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    /// Start-to-start dependency
    StartToStart,
    /// Start-to-finish dependency
    StartToFinish,
    /// Finish-to-start dependency
    FinishToStart,
    /// Finish-to-finish dependency
    FinishToFinish,
    /// Custom dependency type
    Custom(String),
}

/// Comprehensive metrics for dependency graph analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// Total number of nodes in the graph
    pub node_count: u32,
    /// Total number of edges in the graph
    pub edge_count: u32,
    /// Overall complexity score of the graph
    pub complexity_score: f64,
    /// Length of the critical path through the graph
    pub critical_path_length: u32,
    /// Cyclomatic complexity measure
    pub cyclomatic_complexity: f64,
}

impl Default for GraphMetrics {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            complexity_score: 0.0,
            critical_path_length: 0,
            cyclomatic_complexity: 0.0,
        }
    }
}

/// Available algorithms for dependency resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionAlgorithm {
    /// Classic topological sorting algorithm
    TopologicalSort,
    /// Critical path method for scheduling
    CriticalPath,
    /// Layered approach for complex hierarchies
    LayeredApproach,
    /// Priority-based resolution strategy
    PriorityBased,
    /// Resource-constrained scheduling algorithm
    ResourceConstrained,
    /// Custom resolution algorithm
    Custom(String),
}

/// Specialized system for handling circular dependencies
pub struct CircularDependencyHandler {
    /// Algorithms for detecting circular dependencies
    pub detection_algorithms: Vec<CircularDetectionAlgorithm>,
    /// Strategies for breaking circular dependencies
    pub breaking_strategies: Vec<CircularBreakingStrategy>,
    /// Preventive measures for avoiding circular dependencies
    pub prevention_measures: Vec<CircularPreventionMeasure>,
}

impl Default for CircularDependencyHandler {
    fn default() -> Self {
        Self {
            detection_algorithms: vec![
                CircularDetectionAlgorithm::DepthFirstSearch,
                CircularDetectionAlgorithm::StronglyConnectedComponents,
            ],
            breaking_strategies: vec![
                CircularBreakingStrategy::WeakestLink,
                CircularBreakingStrategy::LeastCritical,
            ],
            prevention_measures: vec![
                CircularPreventionMeasure::LayeredArchitecture,
                CircularPreventionMeasure::DependencyInversion,
            ],
        }
    }
}

/// Algorithms for detecting circular dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircularDetectionAlgorithm {
    /// Depth-first search with cycle detection
    DepthFirstSearch,
    /// Strongly connected components analysis
    StronglyConnectedComponents,
    /// Johnson's algorithm for finding cycles
    JohnsonsAlgorithm,
    /// Custom detection algorithm
    Custom(String),
}

/// Strategies for breaking circular dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircularBreakingStrategy {
    /// Remove the weakest dependency link
    WeakestLink,
    /// Remove the least critical dependency
    LeastCritical,
    /// Introduce temporal separation
    TemporalSeparation,
    /// Insert intermediate service
    IntermediateService,
    /// Custom breaking strategy
    Custom(String),
}

/// Preventive measures for avoiding circular dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircularPreventionMeasure {
    /// Enforce layered architecture patterns
    LayeredArchitecture,
    /// Apply dependency inversion principle
    DependencyInversion,
    /// Use event-driven design patterns
    EventDrivenDesign,
    /// Implement service mesh architecture
    ServiceMesh,
    /// Custom prevention measure
    Custom(String),
}

/// Comprehensive validation system for dependency integrity
pub struct DependencyValidation {
    /// Set of validation rules to apply
    pub validation_rules: Vec<DependencyValidationRule>,
    /// Results from validation executions
    pub validation_results: Vec<ValidationResult>,
    /// Compliance checking system
    pub compliance_checking: ComplianceChecking,
}

impl Default for DependencyValidation {
    fn default() -> Self {
        Self {
            validation_rules: Vec::new(),
            validation_results: Vec::new(),
            compliance_checking: ComplianceChecking::default(),
        }
    }
}

/// Individual validation rules for dependency checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyValidationRule {
    /// Unique identifier for the validation rule
    pub rule_id: String,
    /// Category of validation rule
    pub rule_type: ValidationRuleType,
    /// Expression or logic for the rule
    pub rule_expression: String,
    /// Severity level of rule violations
    pub rule_severity: ValidationSeverity,
}

/// Categories of validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Structural integrity rules
    Structural,
    /// Behavioral consistency rules
    Behavioral,
    /// Performance constraint rules
    Performance,
    /// Security requirement rules
    Security,
    /// Compliance adherence rules
    Compliance,
    /// Custom validation rule type
    Custom(String),
}

/// Severity levels for validation rule violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Informational level
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical level requiring immediate attention
    Critical,
}

/// Results from validation rule execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Unique identifier for the validation result
    pub result_id: String,
    /// ID of the rule that was executed
    pub rule_id: String,
    /// Overall status of the validation
    pub validation_status: ValidationStatus,
    /// Detailed violation information if applicable
    pub violation_details: Option<ViolationDetails>,
    /// Suggested remediation actions
    pub remediation_suggestions: Vec<String>,
}

/// Status outcomes for validation execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Validation passed successfully
    Passed,
    /// Validation failed with violations
    Failed,
    /// Validation passed with warnings
    Warning,
    /// Validation was skipped
    Skipped,
}

/// Detailed information about validation violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationDetails {
    /// Type or category of the violation
    pub violation_type: String,
    /// Human-readable description of the violation
    pub violation_description: String,
    /// List of components affected by the violation
    pub affected_components: Vec<String>,
    /// Assessment of the violation's impact
    pub impact_assessment: String,
}

/// Comprehensive compliance checking system
pub struct ComplianceChecking {
    /// Available compliance frameworks
    pub compliance_frameworks: Vec<ComplianceFramework>,
    /// Historical compliance assessments
    pub compliance_assessments: Vec<ComplianceAssessment>,
    /// Reporting and documentation system
    pub compliance_reporting: ComplianceReporting,
}

impl Default for ComplianceChecking {
    fn default() -> Self {
        Self {
            compliance_frameworks: Vec::new(),
            compliance_assessments: Vec::new(),
            compliance_reporting: ComplianceReporting::default(),
        }
    }
}

/// Individual compliance frameworks with requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFramework {
    /// Unique identifier for the framework
    pub framework_id: String,
    /// Human-readable name of the framework
    pub framework_name: String,
    /// Version of the framework specification
    pub framework_version: String,
    /// List of compliance requirements
    pub compliance_requirements: Vec<ComplianceRequirement>,
    /// Criteria for assessing compliance
    pub assessment_criteria: Vec<AssessmentCriterion>,
}

/// Individual compliance requirements within a framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRequirement {
    /// Unique identifier for the requirement
    pub requirement_id: String,
    /// Description of the compliance requirement
    pub requirement_description: String,
    /// Category classification of the requirement
    pub requirement_category: RequirementCategory,
    /// Level of compliance required
    pub compliance_level: ComplianceLevel,
    /// Method for verifying compliance
    pub verification_method: VerificationMethod,
}

/// Categories for organizing compliance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequirementCategory {
    /// Security-related requirements
    Security,
    /// Privacy and data protection requirements
    Privacy,
    /// Availability and uptime requirements
    Availability,
    /// Performance and efficiency requirements
    Performance,
    /// Governance and oversight requirements
    Governance,
    /// Custom requirement category
    Custom(String),
}

/// Levels of compliance required
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceLevel {
    /// Required compliance, no exceptions
    Mandatory,
    /// Recommended but not required
    Recommended,
    /// Optional compliance
    Optional,
    /// Conditional based on circumstances
    Conditional,
}

/// Methods for verifying compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationMethod {
    /// Automated verification process
    Automated,
    /// Manual verification process
    Manual,
    /// External audit verification
    Audit,
    /// Testing-based verification
    Testing,
    /// Documentation-based verification
    Documentation,
    /// Custom verification method
    Custom(String),
}

/// Criteria for assessing compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentCriterion {
    /// Unique identifier for the criterion
    pub criterion_id: String,
    /// Description of the assessment criterion
    pub criterion_description: String,
    /// Method for measuring the criterion
    pub measurement_method: String,
    /// Threshold for passing assessment
    pub passing_threshold: f64,
    /// Weight of this criterion in overall assessment
    pub weight: f64,
}

/// Results from compliance assessments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceAssessment {
    /// Unique identifier for the assessment
    pub assessment_id: String,
    /// ID of the framework being assessed
    pub framework_id: String,
    /// Date when the assessment was performed
    pub assessment_date: SystemTime,
    /// Scope of the compliance assessment
    pub assessment_scope: String,
    /// Overall compliance score (0.0 to 1.0)
    pub compliance_score: f64,
    /// Detailed findings from the assessment
    pub findings: Vec<ComplianceFinding>,
}

/// Individual findings from compliance assessments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFinding {
    /// Unique identifier for the finding
    pub finding_id: String,
    /// ID of the requirement being assessed
    pub requirement_id: String,
    /// Type of finding (compliant, non-compliant, etc.)
    pub finding_type: FindingType,
    /// Severity level of the finding
    pub severity: FindingSeverity,
    /// Detailed description of the finding
    pub description: String,
    /// Plan for addressing the finding
    pub remediation_plan: String,
}

/// Types of compliance findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FindingType {
    /// Fully compliant with requirements
    Compliant,
    /// Not compliant with requirements
    NonCompliant,
    /// Partially compliant with requirements
    PartiallyCompliant,
    /// Requirement not applicable to current context
    NotApplicable,
    /// Assessment not performed for this requirement
    NotAssessed,
}

/// Severity levels for compliance findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FindingSeverity {
    /// Low impact finding
    Low,
    /// Medium impact finding
    Medium,
    /// High impact finding requiring attention
    High,
    /// Critical finding requiring immediate action
    Critical,
}

/// Comprehensive reporting system for compliance
pub struct ComplianceReporting {
    /// Available report templates
    pub report_templates: Vec<ReportTemplate>,
    /// Automated reporting configuration
    pub automated_reporting: AutomatedReporting,
    /// Interactive compliance dashboards
    pub compliance_dashboards: Vec<ComplianceDashboard>,
}

impl Default for ComplianceReporting {
    fn default() -> Self {
        Self {
            report_templates: Vec::new(),
            automated_reporting: AutomatedReporting::default(),
            compliance_dashboards: Vec::new(),
        }
    }
}

/// Templates for generating compliance reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplate {
    /// Unique identifier for the template
    pub template_id: String,
    /// Human-readable name of the template
    pub template_name: String,
    /// Output format for the report
    pub report_format: ReportFormat,
    /// Sections included in the report
    pub report_sections: Vec<ReportSection>,
    /// Available customization options
    pub customization_options: Vec<String>,
}

/// Supported formats for compliance reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    /// PDF document format
    PDF,
    /// HTML web format
    HTML,
    /// Microsoft Excel format
    Excel,
    /// JSON data format
    JSON,
    /// XML data format
    XML,
    /// Custom report format
    Custom(String),
}

/// Individual sections within compliance reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Unique identifier for the section
    pub section_id: String,
    /// Human-readable name of the section
    pub section_name: String,
    /// Type of content in the section
    pub section_type: SectionType,
    /// Template for section content
    pub content_template: String,
}

/// Types of report sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SectionType {
    /// Executive summary section
    Executive,
    /// Detailed findings section
    Detailed,
    /// Technical implementation section
    Technical,
    /// Recommendations and action items
    Recommendations,
    /// Supporting documentation and appendices
    Appendix,
    /// Custom section type
    Custom(String),
}

/// Configuration for automated compliance reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedReporting {
    /// Schedule for automated report generation
    pub reporting_schedule: ReportingSchedule,
    /// Distribution configuration for reports
    pub report_distribution: ReportDistribution,
    /// Automation settings for report workflow
    pub report_automation: ReportAutomation,
}

impl Default for AutomatedReporting {
    fn default() -> Self {
        Self {
            reporting_schedule: ReportingSchedule::default(),
            report_distribution: ReportDistribution::default(),
            report_automation: ReportAutomation::default(),
        }
    }
}

/// Schedule configuration for automated reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingSchedule {
    /// Frequency of automated report generation
    pub schedule_frequency: ScheduleFrequency,
    /// Time of day for report generation
    pub schedule_time: String,
    /// Timezone for scheduling
    pub schedule_timezone: String,
    /// Exceptions to the regular schedule
    pub schedule_exceptions: Vec<String>,
}

impl Default for ReportingSchedule {
    fn default() -> Self {
        Self {
            schedule_frequency: ScheduleFrequency::Monthly,
            schedule_time: "00:00".to_string(),
            schedule_timezone: "UTC".to_string(),
            schedule_exceptions: Vec::new(),
        }
    }
}

/// Frequency options for automated reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleFrequency {
    /// Generate reports daily
    Daily,
    /// Generate reports weekly
    Weekly,
    /// Generate reports monthly
    Monthly,
    /// Generate reports quarterly
    Quarterly,
    /// Generate reports annually
    Annually,
    /// Generate reports on demand only
    OnDemand,
    /// Custom frequency specification
    Custom(String),
}

/// Distribution configuration for automated reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDistribution {
    /// List of recipients for report distribution
    pub distribution_list: Vec<String>,
    /// Method for distributing reports
    pub distribution_method: DistributionMethod,
    /// Whether to require delivery confirmation
    pub delivery_confirmation: bool,
}

impl Default for ReportDistribution {
    fn default() -> Self {
        Self {
            distribution_list: Vec::new(),
            distribution_method: DistributionMethod::Email,
            delivery_confirmation: false,
        }
    }
}

/// Methods for distributing compliance reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionMethod {
    /// Email distribution
    Email,
    /// Shared file system distribution
    FileShare,
    /// API-based distribution
    API,
    /// Dashboard publication
    Dashboard,
    /// Scheduled printing
    PrintScheduled,
    /// Custom distribution method
    Custom(String),
}

/// Automation settings for report workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportAutomation {
    /// Whether to automate data collection
    pub data_collection: bool,
    /// Whether to automate report generation
    pub report_generation: bool,
    /// Whether to perform automated quality checks
    pub quality_checks: bool,
    /// Whether to include approval workflow
    pub approval_workflow: bool,
}

impl Default for ReportAutomation {
    fn default() -> Self {
        Self {
            data_collection: true,
            report_generation: true,
            quality_checks: true,
            approval_workflow: false,
        }
    }
}

/// Interactive dashboards for compliance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceDashboard {
    /// Unique identifier for the dashboard
    pub dashboard_id: String,
    /// Human-readable name of the dashboard
    pub dashboard_name: String,
    /// Widgets and visualizations in the dashboard
    pub dashboard_widgets: Vec<DashboardWidget>,
    /// Frequency for dashboard data refresh
    pub refresh_frequency: Duration,
    /// Access permissions for the dashboard
    pub access_permissions: Vec<String>,
}

/// Individual widgets within compliance dashboards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardWidget {
    /// Unique identifier for the widget
    pub widget_id: String,
    /// Type of visualization widget
    pub widget_type: WidgetType,
    /// Configuration settings for the widget
    pub widget_config: WidgetConfig,
    /// Data source for the widget
    pub data_source: String,
}

/// Types of dashboard widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    /// Chart visualization (line, bar, pie, etc.)
    Chart,
    /// Data table widget
    Table,
    /// Single metric display
    Metric,
    /// Gauge or progress indicator
    Gauge,
    /// Heatmap visualization
    Heatmap,
    /// Custom widget type
    Custom(String),
}

/// Configuration settings for dashboard widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    /// Display title for the widget
    pub title: String,
    /// Size dimensions of the widget
    pub size: WidgetSize,
    /// Position coordinates of the widget
    pub position: WidgetPosition,
    /// Styling and appearance settings
    pub styling: HashMap<String, String>,
    /// Optional data filtering criteria
    pub data_filter: Option<String>,
}

/// Size specification for dashboard widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetSize {
    /// Width in grid units or pixels
    pub width: u32,
    /// Height in grid units or pixels
    pub height: u32,
}

/// Position specification for dashboard widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetPosition {
    /// X coordinate or column position
    pub x: u32,
    /// Y coordinate or row position
    pub y: u32,
}
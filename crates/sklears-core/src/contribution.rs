/// Contribution guidelines and review process for sklears
///
/// This module provides comprehensive guidelines, tools, and processes for contributing
/// to the sklears machine learning library. It includes automated checks, quality gates,
/// and best practices to ensure consistent code quality and maintainability.
///
/// # Key Components
///
/// - **Code Review Guidelines**: Structured review criteria and checklists
/// - **Quality Gates**: Automated checks for code quality and compliance
/// - **Contribution Workflow**: Step-by-step contribution process
/// - **Best Practices**: Coding standards and patterns specific to ML in Rust
/// - **Documentation Standards**: Requirements for API documentation and examples
///
/// # Usage
///
/// ```rust
/// use sklears_core::contribution::{ContributionChecker, ReviewCriteria};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let checker = ContributionChecker::new();
/// let criteria = ReviewCriteria::default();
///
/// // Check a contribution against quality gates
/// let result = checker.check_contribution("path/to/code", &criteria)?;
/// println!("Contribution quality score: {:.2}", result.quality_score());
/// # Ok(())
/// # }
/// ```
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::SystemTime;

/// Main contribution checker and validator
#[derive(Debug)]
pub struct ContributionChecker {
    #[allow(dead_code)]
    config: ContributionConfig,
    quality_gates: Vec<QualityGate>,
}

/// Configuration for contribution checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionConfig {
    /// Minimum code coverage percentage required
    pub min_coverage: f64,
    /// Maximum cyclomatic complexity allowed
    pub max_complexity: u32,
    /// Require comprehensive documentation
    pub require_docs: bool,
    /// Require property-based tests for ML algorithms
    pub require_property_tests: bool,
    /// Maximum line length for code
    pub max_line_length: usize,
    /// Required clippy compliance level
    pub clippy_compliance: ClippyLevel,
    /// Performance regression threshold
    pub performance_threshold: f64,
}

impl Default for ContributionConfig {
    fn default() -> Self {
        Self {
            min_coverage: 90.0,
            max_complexity: 10,
            require_docs: true,
            require_property_tests: true,
            max_line_length: 100,
            clippy_compliance: ClippyLevel::AllowedByDefault,
            performance_threshold: 0.05, // 5% regression threshold
        }
    }
}

/// Clippy compliance levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClippyLevel {
    AllowedByDefault,
    Warn,
    Deny,
    Forbid,
}

/// Review criteria for contributions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReviewCriteria {
    /// Algorithmic correctness requirements
    pub algorithmic_correctness: AlgorithmicCriteria,
    /// Code quality requirements
    pub code_quality: CodeQualityCriteria,
    /// Documentation requirements
    pub documentation: DocumentationCriteria,
    /// Testing requirements
    pub testing: TestingCriteria,
    /// Performance requirements
    pub performance: PerformanceCriteria,
}

/// Algorithmic correctness criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmicCriteria {
    /// Mathematical foundations must be sound
    pub mathematical_soundness: bool,
    /// Algorithm must converge where expected
    pub convergence_guarantees: bool,
    /// Numerical stability must be ensured
    pub numerical_stability: bool,
    /// Edge cases must be handled properly
    pub edge_case_handling: bool,
    /// Reference implementations for comparison
    pub reference_validation: bool,
}

impl Default for AlgorithmicCriteria {
    fn default() -> Self {
        Self {
            mathematical_soundness: true,
            convergence_guarantees: true,
            numerical_stability: true,
            edge_case_handling: true,
            reference_validation: true,
        }
    }
}

/// Code quality criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeQualityCriteria {
    /// Type safety and generic usage
    pub type_safety: bool,
    /// Memory efficiency considerations
    pub memory_efficiency: bool,
    /// Error handling completeness
    pub error_handling: bool,
    /// API design consistency
    pub api_consistency: bool,
    /// Rust idioms and best practices
    pub rust_idioms: bool,
}

impl Default for CodeQualityCriteria {
    fn default() -> Self {
        Self {
            type_safety: true,
            memory_efficiency: true,
            error_handling: true,
            api_consistency: true,
            rust_idioms: true,
        }
    }
}

/// Documentation criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationCriteria {
    /// API documentation completeness
    pub api_docs: bool,
    /// Executable examples provided
    pub examples: bool,
    /// Mathematical background explained
    pub mathematical_background: bool,
    /// Performance characteristics documented
    pub performance_docs: bool,
    /// Usage patterns and best practices
    pub usage_patterns: bool,
}

impl Default for DocumentationCriteria {
    fn default() -> Self {
        Self {
            api_docs: true,
            examples: true,
            mathematical_background: true,
            performance_docs: true,
            usage_patterns: true,
        }
    }
}

/// Testing criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestingCriteria {
    /// Unit test coverage
    pub unit_tests: bool,
    /// Integration test coverage
    pub integration_tests: bool,
    /// Property-based testing for algorithms
    pub property_tests: bool,
    /// Edge case testing
    pub edge_case_tests: bool,
    /// Performance benchmarks
    pub benchmarks: bool,
    /// Regression tests
    pub regression_tests: bool,
}

impl Default for TestingCriteria {
    fn default() -> Self {
        Self {
            unit_tests: true,
            integration_tests: true,
            property_tests: true,
            edge_case_tests: true,
            benchmarks: true,
            regression_tests: true,
        }
    }
}

/// Performance criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCriteria {
    /// Performance compared to baseline
    pub baseline_comparison: bool,
    /// Memory usage analysis
    pub memory_analysis: bool,
    /// Scalability characteristics
    pub scalability: bool,
    /// Parallelization effectiveness
    pub parallelization: bool,
    /// Time complexity documentation
    pub complexity_analysis: bool,
}

impl Default for PerformanceCriteria {
    fn default() -> Self {
        Self {
            baseline_comparison: true,
            memory_analysis: true,
            scalability: true,
            parallelization: true,
            complexity_analysis: true,
        }
    }
}

/// Quality gate definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGate {
    pub name: String,
    pub description: String,
    pub gate_type: QualityGateType,
    pub threshold: f64,
    pub blocking: bool,
}

/// Types of quality gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityGateType {
    CodeCoverage,
    TestPassing,
    LintCompliance,
    DocumentationCoverage,
    PerformanceRegression,
    SecurityVulnerabilities,
    DependencyLicenses,
    APIBreaking,
}

/// Result of contribution check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionResult {
    pub overall_score: f64,
    pub gate_results: Vec<GateResult>,
    pub recommendations: Vec<String>,
    pub blocking_issues: Vec<String>,
    pub timestamp: SystemTime,
}

/// Result of individual quality gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub gate_name: String,
    pub passed: bool,
    pub score: f64,
    pub details: String,
    pub improvement_suggestions: Vec<String>,
}

impl ContributionChecker {
    /// Create a new contribution checker with default configuration
    pub fn new() -> Self {
        Self::with_config(ContributionConfig::default())
    }

    /// Create a contribution checker with custom configuration
    pub fn with_config(config: ContributionConfig) -> Self {
        let quality_gates = Self::create_default_quality_gates(&config);
        Self {
            config,
            quality_gates,
        }
    }

    /// Check a contribution against all quality gates
    pub fn check_contribution(
        &self,
        path: impl AsRef<Path>,
        criteria: &ReviewCriteria,
    ) -> Result<ContributionResult> {
        let path = path.as_ref();
        let mut gate_results = Vec::new();
        let mut blocking_issues = Vec::new();
        let mut recommendations = Vec::new();

        // Run all quality gates
        for gate in &self.quality_gates {
            let result = self.run_quality_gate(gate, path, criteria)?;

            if gate.blocking && !result.passed {
                blocking_issues.push(format!("Blocking issue: {}", result.details));
            }

            recommendations.extend(result.improvement_suggestions.clone());
            gate_results.push(result);
        }

        // Calculate overall score
        let overall_score = if gate_results.is_empty() {
            0.0
        } else {
            gate_results.iter().map(|r| r.score).sum::<f64>() / gate_results.len() as f64
        };

        Ok(ContributionResult {
            overall_score,
            gate_results,
            recommendations,
            blocking_issues,
            timestamp: SystemTime::now(),
        })
    }

    /// Run a specific quality gate
    fn run_quality_gate(
        &self,
        gate: &QualityGate,
        path: &Path,
        _criteria: &ReviewCriteria,
    ) -> Result<GateResult> {
        match gate.gate_type {
            QualityGateType::CodeCoverage => self.check_code_coverage(gate, path),
            QualityGateType::TestPassing => self.check_test_passing(gate, path),
            QualityGateType::LintCompliance => self.check_lint_compliance(gate, path),
            QualityGateType::DocumentationCoverage => self.check_documentation_coverage(gate, path),
            QualityGateType::PerformanceRegression => self.check_performance_regression(gate, path),
            QualityGateType::SecurityVulnerabilities => {
                self.check_security_vulnerabilities(gate, path)
            }
            QualityGateType::DependencyLicenses => self.check_dependency_licenses(gate, path),
            QualityGateType::APIBreaking => self.check_api_breaking(gate, path),
        }
    }

    fn check_code_coverage(&self, gate: &QualityGate, _path: &Path) -> Result<GateResult> {
        // Placeholder implementation - in real scenario would integrate with coverage tools
        let coverage = 85.0; // Mock coverage percentage
        let passed = coverage >= gate.threshold;

        Ok(GateResult {
            gate_name: gate.name.clone(),
            passed,
            score: if passed { 100.0 } else { coverage },
            details: format!(
                "Code coverage: {:.1}% (threshold: {:.1}%)",
                coverage, gate.threshold
            ),
            improvement_suggestions: if passed {
                vec![]
            } else {
                vec![
                    "Add more unit tests for uncovered code paths".to_string(),
                    "Consider adding property-based tests for algorithmic code".to_string(),
                    "Add integration tests for end-to-end workflows".to_string(),
                ]
            },
        })
    }

    fn check_test_passing(&self, gate: &QualityGate, _path: &Path) -> Result<GateResult> {
        // Placeholder implementation
        let all_tests_pass = true; // Mock test result
        let score = if all_tests_pass { 100.0 } else { 0.0 };

        Ok(GateResult {
            gate_name: gate.name.clone(),
            passed: all_tests_pass,
            score,
            details: if all_tests_pass {
                "All tests passing".to_string()
            } else {
                "Some tests failing".to_string()
            },
            improvement_suggestions: if all_tests_pass {
                vec![]
            } else {
                vec![
                    "Fix failing tests before submitting".to_string(),
                    "Ensure tests are deterministic and reproducible".to_string(),
                ]
            },
        })
    }

    fn check_lint_compliance(&self, gate: &QualityGate, _path: &Path) -> Result<GateResult> {
        // Placeholder implementation
        let lint_score = 95.0; // Mock lint compliance score
        let passed = lint_score >= gate.threshold;

        Ok(GateResult {
            gate_name: gate.name.clone(),
            passed,
            score: lint_score,
            details: format!("Lint compliance: {lint_score:.1}%"),
            improvement_suggestions: if passed {
                vec![]
            } else {
                vec![
                    "Fix clippy warnings and errors".to_string(),
                    "Follow Rust naming conventions".to_string(),
                    "Remove unused imports and variables".to_string(),
                ]
            },
        })
    }

    fn check_documentation_coverage(&self, gate: &QualityGate, _path: &Path) -> Result<GateResult> {
        // Placeholder implementation
        let doc_coverage = 88.0; // Mock documentation coverage
        let passed = doc_coverage >= gate.threshold;

        Ok(GateResult {
            gate_name: gate.name.clone(),
            passed,
            score: doc_coverage,
            details: format!("Documentation coverage: {doc_coverage:.1}%"),
            improvement_suggestions: if passed {
                vec![]
            } else {
                vec![
                    "Add doc comments for all public APIs".to_string(),
                    "Include executable examples in documentation".to_string(),
                    "Document mathematical foundations and algorithms".to_string(),
                    "Add performance characteristics documentation".to_string(),
                ]
            },
        })
    }

    fn check_performance_regression(&self, gate: &QualityGate, _path: &Path) -> Result<GateResult> {
        // Placeholder implementation
        let performance_change: f64 = -0.02; // Mock 2% improvement
        let passed = performance_change.abs() <= gate.threshold;

        Ok(GateResult {
            gate_name: gate.name.clone(),
            passed,
            score: if passed { 100.0 } else { 50.0 },
            details: format!(
                "Performance change: {:.1}% (threshold: ±{:.1}%)",
                performance_change * 100.0,
                gate.threshold * 100.0
            ),
            improvement_suggestions: if passed {
                vec![]
            } else {
                vec![
                    "Profile the code to identify performance bottlenecks".to_string(),
                    "Consider algorithmic optimizations".to_string(),
                    "Add benchmarks to track performance over time".to_string(),
                ]
            },
        })
    }

    fn check_security_vulnerabilities(
        &self,
        gate: &QualityGate,
        _path: &Path,
    ) -> Result<GateResult> {
        // Placeholder implementation
        let vulnerabilities_found = 0; // Mock security scan result
        let passed = vulnerabilities_found == 0;

        Ok(GateResult {
            gate_name: gate.name.clone(),
            passed,
            score: if passed { 100.0 } else { 0.0 },
            details: format!("Security vulnerabilities found: {vulnerabilities_found}"),
            improvement_suggestions: if passed {
                vec![]
            } else {
                vec![
                    "Update dependencies with known vulnerabilities".to_string(),
                    "Review unsafe code blocks for safety".to_string(),
                    "Validate all external inputs".to_string(),
                ]
            },
        })
    }

    fn check_dependency_licenses(&self, gate: &QualityGate, _path: &Path) -> Result<GateResult> {
        // Placeholder implementation
        let license_compatible = true; // Mock license check

        Ok(GateResult {
            gate_name: gate.name.clone(),
            passed: license_compatible,
            score: if license_compatible { 100.0 } else { 0.0 },
            details: if license_compatible {
                "All dependencies have compatible licenses".to_string()
            } else {
                "Some dependencies have incompatible licenses".to_string()
            },
            improvement_suggestions: if license_compatible {
                vec![]
            } else {
                vec![
                    "Replace dependencies with incompatible licenses".to_string(),
                    "Ensure all licenses are compatible with project license".to_string(),
                ]
            },
        })
    }

    fn check_api_breaking(&self, gate: &QualityGate, _path: &Path) -> Result<GateResult> {
        // Placeholder implementation
        let breaking_changes = false; // Mock API breaking change detection

        Ok(GateResult {
            gate_name: gate.name.clone(),
            passed: !breaking_changes,
            score: if breaking_changes { 0.0 } else { 100.0 },
            details: if breaking_changes {
                "Breaking API changes detected".to_string()
            } else {
                "No breaking API changes detected".to_string()
            },
            improvement_suggestions: if breaking_changes {
                vec![
                    "Consider deprecation warnings before removing APIs".to_string(),
                    "Use semantic versioning for breaking changes".to_string(),
                    "Provide migration guides for API changes".to_string(),
                ]
            } else {
                vec![]
            },
        })
    }

    fn create_default_quality_gates(config: &ContributionConfig) -> Vec<QualityGate> {
        vec![
            QualityGate {
                name: "Code Coverage".to_string(),
                description: "Minimum code coverage percentage".to_string(),
                gate_type: QualityGateType::CodeCoverage,
                threshold: config.min_coverage,
                blocking: true,
            },
            QualityGate {
                name: "Test Passing".to_string(),
                description: "All tests must pass".to_string(),
                gate_type: QualityGateType::TestPassing,
                threshold: 100.0,
                blocking: true,
            },
            QualityGate {
                name: "Lint Compliance".to_string(),
                description: "Code must pass linting checks".to_string(),
                gate_type: QualityGateType::LintCompliance,
                threshold: 95.0,
                blocking: true,
            },
            QualityGate {
                name: "Documentation Coverage".to_string(),
                description: "Minimum documentation coverage".to_string(),
                gate_type: QualityGateType::DocumentationCoverage,
                threshold: 85.0,
                blocking: false,
            },
            QualityGate {
                name: "Performance Regression".to_string(),
                description: "No significant performance regression".to_string(),
                gate_type: QualityGateType::PerformanceRegression,
                threshold: config.performance_threshold,
                blocking: false,
            },
            QualityGate {
                name: "Security Vulnerabilities".to_string(),
                description: "No security vulnerabilities".to_string(),
                gate_type: QualityGateType::SecurityVulnerabilities,
                threshold: 0.0,
                blocking: true,
            },
            QualityGate {
                name: "Dependency Licenses".to_string(),
                description: "All dependencies have compatible licenses".to_string(),
                gate_type: QualityGateType::DependencyLicenses,
                threshold: 100.0,
                blocking: true,
            },
            QualityGate {
                name: "API Breaking Changes".to_string(),
                description: "No breaking API changes without version bump".to_string(),
                gate_type: QualityGateType::APIBreaking,
                threshold: 0.0,
                blocking: true,
            },
        ]
    }
}

impl Default for ContributionChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl ContributionResult {
    /// Get the overall quality score
    pub fn quality_score(&self) -> f64 {
        self.overall_score
    }

    /// Check if contribution meets all requirements
    pub fn meets_requirements(&self) -> bool {
        self.blocking_issues.is_empty()
    }

    /// Get recommendations for improvement
    pub fn recommendations(&self) -> &[String] {
        &self.recommendations
    }

    /// Generate a detailed report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Contribution Review Report\n\n");
        report.push_str(&format!(
            "**Overall Score**: {:.1}/100\n",
            self.overall_score
        ));
        report.push_str(&format!(
            "**Status**: {}\n\n",
            if self.meets_requirements() {
                "✅ APPROVED"
            } else {
                "❌ NEEDS WORK"
            }
        ));

        if !self.blocking_issues.is_empty() {
            report.push_str("## 🚫 Blocking Issues\n\n");
            for issue in &self.blocking_issues {
                report.push_str(&format!("- {issue}\n"));
            }
            report.push('\n');
        }

        report.push_str("## 📊 Quality Gate Results\n\n");
        for result in &self.gate_results {
            let status = if result.passed { "✅" } else { "❌" };
            report.push_str(&format!(
                "### {} {} ({:.1}/100)\n",
                status, result.gate_name, result.score
            ));
            report.push_str(&format!("{}\n\n", result.details));

            if !result.improvement_suggestions.is_empty() {
                report.push_str("**Suggestions:**\n");
                for suggestion in &result.improvement_suggestions {
                    report.push_str(&format!("- {suggestion}\n"));
                }
                report.push('\n');
            }
        }

        if !self.recommendations.is_empty() {
            report.push_str("## 💡 General Recommendations\n\n");
            for recommendation in &self.recommendations {
                report.push_str(&format!("- {recommendation}\n"));
            }
        }

        report
    }
}

/// Contribution workflow helper
pub struct ContributionWorkflow {
    steps: Vec<WorkflowStep>,
}

/// Individual workflow step
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    pub name: String,
    pub description: String,
    pub commands: Vec<String>,
    pub automated: bool,
}

impl ContributionWorkflow {
    /// Create the standard contribution workflow
    pub fn standard() -> Self {
        let steps = vec![
            WorkflowStep {
                name: "Fork and Clone".to_string(),
                description: "Fork the repository and clone your fork".to_string(),
                commands: vec![
                    "# Fork on GitHub".to_string(),
                    "git clone https://github.com/cool-japan/sklears.git".to_string(),
                    "cd sklears".to_string(),
                    "git remote add upstream https://github.com/cool-japan/sklears.git".to_string(),
                ],
                automated: false,
            },
            WorkflowStep {
                name: "Create Feature Branch".to_string(),
                description: "Create a new branch for your feature".to_string(),
                commands: vec!["git checkout -b feature/your-feature-name".to_string()],
                automated: false,
            },
            WorkflowStep {
                name: "Development".to_string(),
                description: "Implement your changes following best practices".to_string(),
                commands: vec![
                    "# Make your changes".to_string(),
                    "cargo fmt".to_string(),
                    "cargo clippy -- -D warnings".to_string(),
                ],
                automated: false,
            },
            WorkflowStep {
                name: "Testing".to_string(),
                description: "Run comprehensive tests".to_string(),
                commands: vec![
                    "cargo nextest run --no-fail-fast".to_string(),
                    "cargo test --doc".to_string(),
                    "cargo bench".to_string(),
                ],
                automated: true,
            },
            WorkflowStep {
                name: "Documentation".to_string(),
                description: "Update documentation".to_string(),
                commands: vec![
                    "cargo doc --no-deps".to_string(),
                    "# Update CHANGELOG.md".to_string(),
                    "# Update relevant README files".to_string(),
                ],
                automated: false,
            },
            WorkflowStep {
                name: "Quality Checks".to_string(),
                description: "Run quality gates".to_string(),
                commands: vec![
                    "cargo audit".to_string(),
                    "cargo deny check".to_string(),
                    "# Run contribution checker".to_string(),
                ],
                automated: true,
            },
            WorkflowStep {
                name: "Commit and Push".to_string(),
                description: "Commit changes with descriptive message".to_string(),
                commands: vec![
                    "git add .".to_string(),
                    "git commit -m \"feat: add your feature description\"".to_string(),
                    "git push origin feature/your-feature-name".to_string(),
                ],
                automated: false,
            },
            WorkflowStep {
                name: "Pull Request".to_string(),
                description: "Create pull request for review".to_string(),
                commands: vec![
                    "# Create PR on GitHub".to_string(),
                    "# Fill out PR template".to_string(),
                    "# Request reviews".to_string(),
                ],
                automated: false,
            },
        ];

        Self { steps }
    }

    /// Get all workflow steps
    pub fn steps(&self) -> &[WorkflowStep] {
        &self.steps
    }

    /// Generate workflow documentation
    pub fn generate_guide(&self) -> String {
        let mut guide = String::new();

        guide.push_str("# Contribution Workflow Guide\n\n");
        guide.push_str("Follow these steps to contribute to sklears:\n\n");

        for (i, step) in self.steps.iter().enumerate() {
            guide.push_str(&format!("## {}. {}\n\n", i + 1, step.name));
            guide.push_str(&format!("{}\n\n", step.description));

            if !step.commands.is_empty() {
                guide.push_str("```bash\n");
                for command in &step.commands {
                    guide.push_str(&format!("{command}\n"));
                }
                guide.push_str("```\n\n");
            }

            if step.automated {
                guide.push_str("*This step can be automated in CI/CD.*\n\n");
            }
        }

        guide
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contribution_checker_creation() {
        let checker = ContributionChecker::new();
        assert_eq!(checker.config.min_coverage, 90.0);
        assert!(!checker.quality_gates.is_empty());
    }

    #[test]
    fn test_review_criteria_default() {
        let criteria = ReviewCriteria::default();
        assert!(criteria.algorithmic_correctness.mathematical_soundness);
        assert!(criteria.code_quality.type_safety);
        assert!(criteria.documentation.api_docs);
        assert!(criteria.testing.unit_tests);
        assert!(criteria.performance.baseline_comparison);
    }

    #[test]
    fn test_contribution_workflow() {
        let workflow = ContributionWorkflow::standard();
        assert!(!workflow.steps().is_empty());

        let guide = workflow.generate_guide();
        assert!(guide.contains("Contribution Workflow Guide"));
        assert!(guide.contains("Fork and Clone"));
    }

    #[test]
    fn test_quality_gate_types() {
        let config = ContributionConfig::default();
        let gates = ContributionChecker::create_default_quality_gates(&config);

        let gate_types: Vec<_> = gates.iter().map(|g| &g.gate_type).collect();
        assert!(gate_types
            .iter()
            .any(|t| matches!(t, QualityGateType::CodeCoverage)));
        assert!(gate_types
            .iter()
            .any(|t| matches!(t, QualityGateType::TestPassing)));
        assert!(gate_types
            .iter()
            .any(|t| matches!(t, QualityGateType::LintCompliance)));
    }

    #[test]
    fn test_contribution_result() {
        let result = ContributionResult {
            overall_score: 85.5,
            gate_results: vec![GateResult {
                gate_name: "Test".to_string(),
                passed: true,
                score: 100.0,
                details: "All tests pass".to_string(),
                improvement_suggestions: vec![],
            }],
            recommendations: vec!["Improve documentation".to_string()],
            blocking_issues: vec![],
            timestamp: SystemTime::now(),
        };

        assert_eq!(result.quality_score(), 85.5);
        assert!(result.meets_requirements());
        assert_eq!(result.recommendations().len(), 1);

        let report = result.generate_report();
        assert!(report.contains("Contribution Review Report"));
        assert!(report.contains("✅ APPROVED"));
    }
}

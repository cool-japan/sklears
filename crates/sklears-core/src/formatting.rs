/// Automated code formatting and linting utilities
///
/// This module provides tools for checking and enforcing code formatting standards
/// specific to machine learning code patterns in the sklears ecosystem.
use crate::error::SklearsError;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Result type alias for formatting operations
pub type Result<T> = std::result::Result<T, SklearsError>;

/// Configuration for code formatting checks
#[derive(Debug, Clone)]
pub struct FormattingConfig {
    /// Enable rustfmt checking
    pub check_rustfmt: bool,
    /// Enable clippy checking
    pub check_clippy: bool,
    /// Custom clippy lints to enable
    pub clippy_lints: Vec<String>,
    /// Paths to exclude from formatting checks
    pub exclude_paths: Vec<PathBuf>,
    /// Maximum allowed line length
    pub max_line_length: usize,
    /// Require documentation for public items
    pub require_docs: bool,
    /// ML-specific formatting rules
    pub ml_specific_rules: MLFormattingRules,
}

/// ML-specific formatting rules
#[derive(Debug, Clone)]
pub struct MLFormattingRules {
    /// Require type annotations for ML parameters
    pub require_param_types: bool,
    /// Enforce consistent naming for ML concepts
    pub enforce_ml_naming: bool,
    /// Require validation for ML inputs
    pub require_input_validation: bool,
    /// Maximum complexity for ML functions
    pub max_function_complexity: usize,
    /// Require error handling for ML operations
    pub require_error_handling: bool,
}

impl Default for FormattingConfig {
    fn default() -> Self {
        Self {
            check_rustfmt: true,
            check_clippy: true,
            clippy_lints: vec![
                "clippy::pedantic".to_string(),
                "clippy::cargo".to_string(),
                "clippy::nursery".to_string(),
            ],
            exclude_paths: vec![PathBuf::from("target"), PathBuf::from("*.lock")],
            max_line_length: 100,
            require_docs: true,
            ml_specific_rules: MLFormattingRules::default(),
        }
    }
}

impl Default for MLFormattingRules {
    fn default() -> Self {
        Self {
            require_param_types: true,
            enforce_ml_naming: true,
            require_input_validation: true,
            max_function_complexity: 10,
            require_error_handling: true,
        }
    }
}

/// Result of formatting checks
#[derive(Debug, Clone)]
pub struct FormattingReport {
    /// Overall formatting status
    pub passed: bool,
    /// Rustfmt check results
    pub rustfmt_result: Option<CheckResult>,
    /// Clippy check results
    pub clippy_result: Option<CheckResult>,
    /// ML-specific rule check results
    pub ml_rules_result: Option<MLRulesResult>,
    /// Summary of all issues found
    pub summary: FormattingSummary,
}

/// Result of a specific formatting check
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// Whether the check passed
    pub passed: bool,
    /// Issues found during the check
    pub issues: Vec<FormattingIssue>,
    /// Command output (stdout/stderr)
    pub output: String,
    /// Exit code from the formatting tool
    pub exit_code: i32,
}

/// ML-specific rules check result
#[derive(Debug, Clone)]
pub struct MLRulesResult {
    /// Whether all ML rules passed
    pub passed: bool,
    /// Parameter type annotation issues
    pub param_type_issues: Vec<FormattingIssue>,
    /// ML naming convention issues
    pub naming_issues: Vec<FormattingIssue>,
    /// Input validation issues
    pub validation_issues: Vec<FormattingIssue>,
    /// Function complexity issues
    pub complexity_issues: Vec<FormattingIssue>,
    /// Error handling issues
    pub error_handling_issues: Vec<FormattingIssue>,
}

/// Individual formatting issue
#[derive(Debug, Clone)]
pub struct FormattingIssue {
    /// File path where the issue was found
    pub file: PathBuf,
    /// Line number (if applicable)
    pub line: Option<usize>,
    /// Column number (if applicable)
    pub column: Option<usize>,
    /// Issue severity
    pub severity: IssueSeverity,
    /// Description of the issue
    pub message: String,
    /// Suggested fix (if available)
    pub suggestion: Option<String>,
    /// Rule that triggered this issue
    pub rule: String,
}

/// Severity level of formatting issues
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Error that must be fixed
    Error,
    /// Warning that should be addressed
    Warning,
    /// Information/suggestion
    Info,
}

/// Summary of all formatting checks
#[derive(Debug, Clone)]
pub struct FormattingSummary {
    /// Total number of files checked
    pub files_checked: usize,
    /// Number of errors found
    pub error_count: usize,
    /// Number of warnings found
    pub warning_count: usize,
    /// Number of info issues found
    pub info_count: usize,
    /// Files with issues
    pub files_with_issues: Vec<PathBuf>,
}

/// Main formatter for checking code quality
pub struct CodeFormatter {
    config: FormattingConfig,
}

impl CodeFormatter {
    /// Create a new code formatter with default configuration
    pub fn new() -> Self {
        Self {
            config: FormattingConfig::default(),
        }
    }

    /// Create a new code formatter with custom configuration
    pub fn with_config(config: FormattingConfig) -> Self {
        Self { config }
    }

    /// Run all formatting checks on the specified path
    pub fn check_all<P: AsRef<Path>>(&self, path: P) -> Result<FormattingReport> {
        let path = path.as_ref();

        let mut report = FormattingReport {
            passed: true,
            rustfmt_result: None,
            clippy_result: None,
            ml_rules_result: None,
            summary: FormattingSummary {
                files_checked: 0,
                error_count: 0,
                warning_count: 0,
                info_count: 0,
                files_with_issues: Vec::new(),
            },
        };

        // Run rustfmt check
        if self.config.check_rustfmt {
            match self.check_rustfmt(path) {
                Ok(result) => {
                    report.passed &= result.passed;
                    report.rustfmt_result = Some(result);
                }
                Err(e) => {
                    log::warn!("Failed to run rustfmt check: {e}");
                    report.passed = false;
                }
            }
        }

        // Run clippy check
        if self.config.check_clippy {
            match self.check_clippy(path) {
                Ok(result) => {
                    report.passed &= result.passed;
                    report.clippy_result = Some(result);
                }
                Err(e) => {
                    log::warn!("Failed to run clippy check: {e}");
                    report.passed = false;
                }
            }
        }

        // Run ML-specific rules check
        match self.check_ml_rules(path) {
            Ok(result) => {
                report.passed &= result.passed;
                report.ml_rules_result = Some(result);
            }
            Err(e) => {
                log::warn!("Failed to run ML rules check: {e}");
                report.passed = false;
            }
        }

        // Generate summary
        self.generate_summary(&mut report);

        Ok(report)
    }

    /// Check rustfmt formatting
    fn check_rustfmt<P: AsRef<Path>>(&self, path: P) -> Result<CheckResult> {
        let output = Command::new("rustfmt")
            .arg("--check")
            .arg("--config")
            .arg(format!("max_width={}", self.config.max_line_length))
            .arg(path.as_ref())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to run rustfmt: {e}")))?;

        let passed = output.status.success();
        let output_str = String::from_utf8_lossy(&output.stderr).to_string();
        let issues = self.parse_rustfmt_output(&output_str);

        Ok(CheckResult {
            passed,
            issues,
            output: output_str,
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    /// Check clippy lints
    fn check_clippy<P: AsRef<Path>>(&self, path: P) -> Result<CheckResult> {
        let mut cmd = Command::new("cargo");
        cmd.arg("clippy").arg("--").arg("-D").arg("warnings");

        // Add custom lints
        for lint in &self.config.clippy_lints {
            cmd.arg("-W").arg(lint);
        }

        let output = cmd
            .current_dir(path.as_ref())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to run clippy: {e}")))?;

        let passed = output.status.success();
        let output_str = String::from_utf8_lossy(&output.stderr).to_string();
        let issues = self.parse_clippy_output(&output_str);

        Ok(CheckResult {
            passed,
            issues,
            output: output_str,
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    /// Check ML-specific formatting rules
    fn check_ml_rules<P: AsRef<Path>>(&self, _path: P) -> Result<MLRulesResult> {
        // This is a simplified implementation
        // In a full implementation, this would parse Rust AST and check for ML-specific patterns

        let result = MLRulesResult {
            passed: true,
            param_type_issues: Vec::new(),
            naming_issues: Vec::new(),
            validation_issues: Vec::new(),
            complexity_issues: Vec::new(),
            error_handling_issues: Vec::new(),
        };

        Ok(result)
    }

    /// Parse rustfmt output into formatting issues
    fn parse_rustfmt_output(&self, output: &str) -> Vec<FormattingIssue> {
        let mut issues = Vec::new();

        for line in output.lines() {
            if line.contains("Diff in") {
                if let Some(file_path) = line.split_whitespace().nth(2) {
                    issues.push(FormattingIssue {
                        file: PathBuf::from(file_path),
                        line: None,
                        column: None,
                        severity: IssueSeverity::Error,
                        message: "File is not properly formatted".to_string(),
                        suggestion: Some("Run 'cargo fmt' to fix formatting".to_string()),
                        rule: "rustfmt".to_string(),
                    });
                }
            }
        }

        issues
    }

    /// Parse clippy output into formatting issues
    fn parse_clippy_output(&self, output: &str) -> Vec<FormattingIssue> {
        let mut issues = Vec::new();

        for line in output.lines() {
            if line.contains("warning:") || line.contains("error:") {
                // Parse clippy output format: "file:line:column: level: message"
                let parts: Vec<&str> = line.splitn(5, ':').collect();
                if parts.len() >= 5 {
                    let file = PathBuf::from(parts[0]);
                    let line = parts[1].parse().ok();
                    let column = parts[2].parse().ok();
                    let severity = match parts[3].trim() {
                        "error" => IssueSeverity::Error,
                        "warning" => IssueSeverity::Warning,
                        _ => IssueSeverity::Info,
                    };
                    let message = parts[4].trim().to_string();

                    issues.push(FormattingIssue {
                        file,
                        line,
                        column,
                        severity,
                        message,
                        suggestion: None,
                        rule: "clippy".to_string(),
                    });
                }
            }
        }

        issues
    }

    /// Generate summary statistics for the formatting report
    fn generate_summary(&self, report: &mut FormattingReport) {
        let mut files_with_issues = Vec::new();
        let mut error_count = 0;
        let mut warning_count = 0;
        let mut info_count = 0;

        // Count issues from all check results
        if let Some(ref result) = report.rustfmt_result {
            for issue in &result.issues {
                match issue.severity {
                    IssueSeverity::Error => error_count += 1,
                    IssueSeverity::Warning => warning_count += 1,
                    IssueSeverity::Info => info_count += 1,
                }
                if !files_with_issues.contains(&issue.file) {
                    files_with_issues.push(issue.file.clone());
                }
            }
        }

        if let Some(ref result) = report.clippy_result {
            for issue in &result.issues {
                match issue.severity {
                    IssueSeverity::Error => error_count += 1,
                    IssueSeverity::Warning => warning_count += 1,
                    IssueSeverity::Info => info_count += 1,
                }
                if !files_with_issues.contains(&issue.file) {
                    files_with_issues.push(issue.file.clone());
                }
            }
        }

        if let Some(ref result) = report.ml_rules_result {
            let all_ml_issues = [
                &result.param_type_issues,
                &result.naming_issues,
                &result.validation_issues,
                &result.complexity_issues,
                &result.error_handling_issues,
            ];

            for issues in all_ml_issues {
                for issue in issues {
                    match issue.severity {
                        IssueSeverity::Error => error_count += 1,
                        IssueSeverity::Warning => warning_count += 1,
                        IssueSeverity::Info => info_count += 1,
                    }
                    if !files_with_issues.contains(&issue.file) {
                        files_with_issues.push(issue.file.clone());
                    }
                }
            }
        }

        report.summary = FormattingSummary {
            files_checked: files_with_issues.len().max(1), // At least 1 if any checks were run
            error_count,
            warning_count,
            info_count,
            files_with_issues,
        };
    }

    /// Fix formatting issues automatically where possible
    pub fn fix_issues<P: AsRef<Path>>(&self, path: P) -> Result<FormattingReport> {
        let path = path.as_ref();

        // Run rustfmt to fix formatting
        if self.config.check_rustfmt {
            let _output = Command::new("rustfmt")
                .arg(path)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to run rustfmt: {e}")))?;
        }

        // Re-run checks to see what was fixed
        self.check_all(path)
    }

    /// Get the current configuration
    pub fn config(&self) -> &FormattingConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: FormattingConfig) {
        self.config = config;
    }
}

impl Default for CodeFormatter {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating formatting configurations
pub struct FormattingConfigBuilder {
    config: FormattingConfig,
}

impl FormattingConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: FormattingConfig::default(),
        }
    }

    /// Enable or disable rustfmt checking
    pub fn check_rustfmt(mut self, enable: bool) -> Self {
        self.config.check_rustfmt = enable;
        self
    }

    /// Enable or disable clippy checking
    pub fn check_clippy(mut self, enable: bool) -> Self {
        self.config.check_clippy = enable;
        self
    }

    /// Add custom clippy lints
    pub fn clippy_lints(mut self, lints: Vec<String>) -> Self {
        self.config.clippy_lints = lints;
        self
    }

    /// Set maximum line length
    pub fn max_line_length(mut self, length: usize) -> Self {
        self.config.max_line_length = length;
        self
    }

    /// Enable or disable documentation requirements
    pub fn require_docs(mut self, require: bool) -> Self {
        self.config.require_docs = require;
        self
    }

    /// Set ML-specific formatting rules
    pub fn ml_rules(mut self, rules: MLFormattingRules) -> Self {
        self.config.ml_specific_rules = rules;
        self
    }

    /// Build the configuration
    pub fn build(self) -> FormattingConfig {
        self.config
    }
}

impl Default for FormattingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formatting_config_default() {
        let config = FormattingConfig::default();
        assert!(config.check_rustfmt);
        assert!(config.check_clippy);
        assert_eq!(config.max_line_length, 100);
        assert!(config.require_docs);
    }

    #[test]
    fn test_formatting_config_builder() {
        let config = FormattingConfigBuilder::new()
            .check_rustfmt(false)
            .max_line_length(120)
            .require_docs(false)
            .build();

        assert!(!config.check_rustfmt);
        assert_eq!(config.max_line_length, 120);
        assert!(!config.require_docs);
    }

    #[test]
    fn test_code_formatter_creation() {
        let formatter = CodeFormatter::new();
        assert!(formatter.config().check_rustfmt);
        assert!(formatter.config().check_clippy);
    }

    #[test]
    fn test_formatting_issue_creation() {
        let issue = FormattingIssue {
            file: PathBuf::from("test.rs"),
            line: Some(10),
            column: Some(5),
            severity: IssueSeverity::Warning,
            message: "Test issue".to_string(),
            suggestion: Some("Fix it".to_string()),
            rule: "test_rule".to_string(),
        };

        assert_eq!(issue.file, PathBuf::from("test.rs"));
        assert_eq!(issue.line, Some(10));
        assert_eq!(issue.severity, IssueSeverity::Warning);
    }

    #[test]
    fn test_ml_formatting_rules_default() {
        let rules = MLFormattingRules::default();
        assert!(rules.require_param_types);
        assert!(rules.enforce_ml_naming);
        assert!(rules.require_input_validation);
        assert_eq!(rules.max_function_complexity, 10);
        assert!(rules.require_error_handling);
    }

    #[test]
    fn test_parse_rustfmt_output() {
        let formatter = CodeFormatter::new();
        let output = "Diff in src/test.rs at line 1:\n -old line\n +new line";
        let issues = formatter.parse_rustfmt_output(output);

        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Error);
        assert!(issues[0].message.contains("not properly formatted"));
    }

    #[test]
    fn test_parse_clippy_output() {
        let formatter = CodeFormatter::new();
        let output = "src/test.rs:10:5: warning: unused variable";
        let issues = formatter.parse_clippy_output(output);

        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].line, Some(10));
        assert_eq!(issues[0].column, Some(5));
        assert_eq!(issues[0].severity, IssueSeverity::Warning);
    }

    #[test]
    fn test_issue_severity_ordering() {
        assert_eq!(IssueSeverity::Error, IssueSeverity::Error);
        assert_ne!(IssueSeverity::Error, IssueSeverity::Warning);
        assert_ne!(IssueSeverity::Warning, IssueSeverity::Info);
    }
}

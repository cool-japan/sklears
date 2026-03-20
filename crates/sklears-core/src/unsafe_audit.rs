/// Unsafe code auditing and minimization utilities
///
/// This module provides tools for auditing, tracking, and minimizing unsafe code usage
/// in the sklears ecosystem, with a focus on safety and correctness.
use crate::error::SklearsError;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Result type alias for unsafe audit operations
pub type Result<T> = std::result::Result<T, SklearsError>;

/// Configuration for unsafe code auditing
#[derive(Debug, Clone)]
pub struct UnsafeAuditConfig {
    /// Paths to scan for unsafe code
    pub scan_paths: Vec<PathBuf>,
    /// Paths to exclude from auditing
    pub exclude_paths: Vec<PathBuf>,
    /// Maximum allowed unsafe blocks per file
    pub max_unsafe_per_file: usize,
    /// Whether to check for documented justifications
    pub require_justification: bool,
    /// Whether to flag all unsafe code as errors
    pub strict_mode: bool,
    /// Known safe patterns to allow
    pub allowed_patterns: Vec<UnsafePattern>,
}

/// Pattern for safe unsafe code usage
#[derive(Debug, Clone)]
pub struct UnsafePattern {
    /// Name/description of the pattern
    pub name: String,
    /// Function/method signatures that are considered safe
    pub signatures: Vec<String>,
    /// Justification for why this pattern is safe
    pub justification: String,
    /// Required preconditions for safety
    pub preconditions: Vec<String>,
}

/// Result of unsafe code audit
#[derive(Debug, Clone)]
pub struct UnsafeAuditReport {
    /// Whether the audit passed all checks
    pub passed: bool,
    /// Total number of files scanned
    pub files_scanned: usize,
    /// Number of files with unsafe code
    pub files_with_unsafe: usize,
    /// Total number of unsafe blocks found
    pub total_unsafe_blocks: usize,
    /// Unsafe code findings per file
    pub findings: HashMap<PathBuf, Vec<UnsafeFinding>>,
    /// Summary statistics
    pub summary: UnsafeSummary,
    /// Recommendations for improvement
    pub recommendations: Vec<SafetyRecommendation>,
}

/// Individual unsafe code finding
#[derive(Debug, Clone)]
pub struct UnsafeFinding {
    /// File path where unsafe code was found
    pub file: PathBuf,
    /// Line number of the unsafe block
    pub line: usize,
    /// Column number (if available)
    pub column: Option<usize>,
    /// Type of unsafe operation
    pub unsafe_type: UnsafeType,
    /// The actual unsafe code snippet
    pub code_snippet: String,
    /// Justification provided (if any)
    pub justification: Option<String>,
    /// Whether this pattern is known to be safe
    pub is_known_safe: bool,
    /// Severity of the safety concern
    pub severity: SafetySeverity,
    /// Suggested alternatives or improvements
    pub suggestions: Vec<String>,
}

/// Type of unsafe operation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnsafeType {
    /// Raw pointer dereferencing
    RawPointerDeref,
    /// Calling unsafe functions
    UnsafeFunctionCall,
    /// Mutable static access
    MutableStatic,
    /// Union field access
    UnionFieldAccess,
    /// Transmute operations
    Transmute,
    /// Inline assembly
    InlineAssembly,
    /// Generic unsafe block
    UnsafeBlock,
}

/// Severity of safety concerns
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SafetySeverity {
    /// Informational, pattern is known safe
    Info,
    /// Low risk, acceptable with documentation
    Low,
    /// Medium risk that should be justified
    Medium,
    /// High risk that should be reviewed
    High,
    /// Critical safety issue that must be addressed
    Critical,
}

impl PartialOrd for SafetySeverity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SafetySeverity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use SafetySeverity::*;
        match (self, other) {
            (Info, Info) => std::cmp::Ordering::Equal,
            (Info, _) => std::cmp::Ordering::Less,
            (_, Info) => std::cmp::Ordering::Greater,
            (Low, Low) => std::cmp::Ordering::Equal,
            (Low, _) => std::cmp::Ordering::Less,
            (_, Low) => std::cmp::Ordering::Greater,
            (Medium, Medium) => std::cmp::Ordering::Equal,
            (Medium, _) => std::cmp::Ordering::Less,
            (_, Medium) => std::cmp::Ordering::Greater,
            (High, High) => std::cmp::Ordering::Equal,
            (High, Critical) => std::cmp::Ordering::Less,
            (Critical, High) => std::cmp::Ordering::Greater,
            (Critical, Critical) => std::cmp::Ordering::Equal,
        }
    }
}

/// Summary statistics for unsafe code audit
#[derive(Debug, Clone)]
pub struct UnsafeSummary {
    /// Breakdown by unsafe operation type
    pub types_breakdown: HashMap<UnsafeType, usize>,
    /// Breakdown by severity
    pub severity_breakdown: HashMap<SafetySeverity, usize>,
    /// Files with the most unsafe code
    pub top_unsafe_files: Vec<(PathBuf, usize)>,
    /// Common unsafe patterns found
    pub common_patterns: Vec<String>,
}

/// Safety improvement recommendation
#[derive(Debug, Clone)]
pub struct SafetyRecommendation {
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Description of the recommendation
    pub description: String,
    /// Files that would benefit from this recommendation
    pub affected_files: Vec<PathBuf>,
    /// Estimated effort to implement
    pub effort: EffortLevel,
    /// Safety impact of implementing this recommendation
    pub safety_impact: SafetyImpact,
}

/// Type of safety recommendation
#[derive(Debug, Clone)]
pub enum RecommendationType {
    /// Replace unsafe code with safe alternatives
    ReplaceWithSafe,
    /// Add better documentation/justification
    ImproveDocumentation,
    /// Reduce scope of unsafe operations
    ReduceScope,
    /// Add safety assertions/checks
    AddSafetyChecks,
    /// Refactor to eliminate unsafe code
    Refactor,
    /// Use safer abstractions
    UseSaferAbstractions,
}

/// Effort level for implementing recommendations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum EffortLevel {
    /// Minimal effort (< 1 hour)
    Minimal,
    /// Low effort (1-4 hours)
    Low,
    /// Medium effort (4-16 hours)
    Medium,
    /// High effort (16+ hours)
    High,
}

/// Safety impact of recommendations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SafetyImpact {
    /// Critical safety improvement
    Critical,
    /// High safety improvement
    High,
    /// Medium safety improvement
    Medium,
    /// Low safety improvement
    Low,
}

impl Default for UnsafeAuditConfig {
    fn default() -> Self {
        Self {
            scan_paths: vec![PathBuf::from("src")],
            exclude_paths: vec![
                PathBuf::from("target"),
                PathBuf::from("benches"),
                PathBuf::from("examples"),
            ],
            max_unsafe_per_file: 5,
            require_justification: true,
            strict_mode: false,
            allowed_patterns: Self::default_safe_patterns(),
        }
    }
}

impl UnsafeAuditConfig {
    /// Get default set of known safe patterns
    fn default_safe_patterns() -> Vec<UnsafePattern> {
        vec![
            UnsafePattern {
                name: "SIMD Operations".to_string(),
                signatures: vec!["std::simd::".to_string(), "std::arch::".to_string()],
                justification: "SIMD operations are generally safe when used correctly".to_string(),
                preconditions: vec![
                    "Input arrays are properly aligned".to_string(),
                    "Array bounds are checked".to_string(),
                ],
            },
            UnsafePattern {
                name: "Slice from Raw Parts".to_string(),
                signatures: vec![
                    "std::slice::from_raw_parts".to_string(),
                    "std::slice::from_raw_parts_mut".to_string(),
                ],
                justification: "Safe when pointer and length are valid".to_string(),
                preconditions: vec![
                    "Pointer is non-null and properly aligned".to_string(),
                    "Length is accurate and doesn't overflow".to_string(),
                    "Memory is valid for the lifetime".to_string(),
                ],
            },
            UnsafePattern {
                name: "FFI Bindings".to_string(),
                signatures: vec!["extern".to_string()],
                justification: "FFI calls to well-tested C libraries".to_string(),
                preconditions: vec![
                    "C library is memory-safe".to_string(),
                    "Parameters are validated".to_string(),
                    "Return values are checked".to_string(),
                ],
            },
        ]
    }
}

/// Main unsafe code auditor
pub struct UnsafeAuditor {
    config: UnsafeAuditConfig,
}

impl UnsafeAuditor {
    /// Create a new auditor with default configuration
    pub fn new() -> Self {
        Self {
            config: UnsafeAuditConfig::default(),
        }
    }

    /// Create a new auditor with custom configuration
    pub fn with_config(config: UnsafeAuditConfig) -> Self {
        Self { config }
    }

    /// Run complete unsafe code audit
    pub fn audit<P: AsRef<Path>>(&self, root_path: P) -> Result<UnsafeAuditReport> {
        let root_path = root_path.as_ref();
        let mut findings = HashMap::new();
        let mut files_scanned = 0;
        let mut total_unsafe_blocks = 0;

        // Scan all Rust files in the specified paths
        for scan_path in &self.config.scan_paths {
            let full_path = root_path.join(scan_path);
            if full_path.exists() {
                self.scan_directory(
                    &full_path,
                    &mut findings,
                    &mut files_scanned,
                    &mut total_unsafe_blocks,
                )?;
            }
        }

        let files_with_unsafe = findings.len();
        let passed = self.evaluate_audit_results(&findings);
        let summary = self.generate_summary(&findings);
        let recommendations = self.generate_recommendations(&findings);

        Ok(UnsafeAuditReport {
            passed,
            files_scanned,
            files_with_unsafe,
            total_unsafe_blocks,
            findings,
            summary,
            recommendations,
        })
    }

    /// Scan a directory for unsafe code
    fn scan_directory(
        &self,
        dir: &Path,
        findings: &mut HashMap<PathBuf, Vec<UnsafeFinding>>,
        files_scanned: &mut usize,
        total_unsafe: &mut usize,
    ) -> Result<()> {
        if self.should_exclude(dir) {
            return Ok(());
        }

        let entries = fs::read_dir(dir)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read directory: {e}")))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to read entry: {e}")))?;
            let path = entry.path();

            if path.is_dir() {
                self.scan_directory(&path, findings, files_scanned, total_unsafe)?;
            } else if path.extension().map(|ext| ext == "rs").unwrap_or(false)
                && !self.should_exclude(&path)
            {
                *files_scanned += 1;
                let file_findings = self.scan_file(&path)?;
                *total_unsafe += file_findings.len();
                if !file_findings.is_empty() {
                    findings.insert(path, file_findings);
                }
            }
        }

        Ok(())
    }

    /// Scan a single file for unsafe code
    fn scan_file(&self, file_path: &Path) -> Result<Vec<UnsafeFinding>> {
        let content = fs::read_to_string(file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read file: {e}")))?;

        let mut findings = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            if let Some(finding) = self.analyze_line(file_path, line_num + 1, line) {
                findings.push(finding);
            }
        }

        // Check for block-level unsafe patterns
        findings.extend(self.analyze_unsafe_blocks(file_path, &content)?);

        Ok(findings)
    }

    /// Analyze a single line for unsafe patterns
    fn analyze_line(&self, file_path: &Path, line_num: usize, line: &str) -> Option<UnsafeFinding> {
        let trimmed = line.trim();

        // Check for unsafe keyword
        if trimmed.starts_with("unsafe") {
            let unsafe_type = self.determine_unsafe_type(line);
            let severity = self.assess_severity(&unsafe_type, line);
            let is_known_safe = self.is_known_safe_pattern(line);
            let justification = self.extract_justification(line);
            let suggestions = self.generate_suggestions(&unsafe_type, line);

            Some(UnsafeFinding {
                file: file_path.to_path_buf(),
                line: line_num,
                column: line.find("unsafe"),
                unsafe_type,
                code_snippet: line.to_string(),
                justification,
                is_known_safe,
                severity,
                suggestions,
            })
        } else {
            None
        }
    }

    /// Analyze unsafe blocks in the entire file
    fn analyze_unsafe_blocks(&self, file_path: &Path, content: &str) -> Result<Vec<UnsafeFinding>> {
        let mut findings = Vec::new();
        let mut in_unsafe_block = false;
        let mut block_start = 0;
        let mut brace_count = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.contains("unsafe {") {
                in_unsafe_block = true;
                block_start = line_num + 1;
                brace_count = 1;
            } else if in_unsafe_block {
                brace_count += line.matches('{').count();
                brace_count -= line.matches('}').count();

                if brace_count == 0 {
                    // End of unsafe block
                    in_unsafe_block = false;

                    // Extract the entire unsafe block
                    let block_lines: Vec<&str> = content
                        .lines()
                        .skip(block_start - 1)
                        .take(line_num - block_start + 2)
                        .collect();
                    let block_content = block_lines.join("\n");

                    let unsafe_type = UnsafeType::UnsafeBlock;
                    let severity = self.assess_block_severity(&block_content);
                    let is_known_safe = self.is_known_safe_pattern(&block_content);
                    let justification = self.extract_block_justification(&block_content);
                    let suggestions = self.generate_block_suggestions(&block_content);

                    findings.push(UnsafeFinding {
                        file: file_path.to_path_buf(),
                        line: block_start,
                        column: None,
                        unsafe_type,
                        code_snippet: block_content,
                        justification,
                        is_known_safe,
                        severity,
                        suggestions,
                    });
                }
            }
        }

        Ok(findings)
    }

    /// Determine the type of unsafe operation
    fn determine_unsafe_type(&self, line: &str) -> UnsafeType {
        if line.contains("transmute") {
            UnsafeType::Transmute
        } else if line.contains("asm!") {
            UnsafeType::InlineAssembly
        } else if line.contains("static mut") {
            UnsafeType::MutableStatic
        } else if line.contains("union") {
            UnsafeType::UnionFieldAccess
        } else if line.contains("*ptr")
            || (line.contains("*") && (line.contains("as *") || line.contains("->")))
        {
            UnsafeType::RawPointerDeref
        } else if line.contains("func()")
            || (line.contains("(")
                && line.contains(")")
                && !line.contains("asm!")
                && !line.contains("transmute"))
        {
            UnsafeType::UnsafeFunctionCall
        } else {
            UnsafeType::UnsafeBlock
        }
    }

    /// Assess the severity of an unsafe operation
    fn assess_severity(&self, unsafe_type: &UnsafeType, code: &str) -> SafetySeverity {
        match unsafe_type {
            UnsafeType::Transmute => SafetySeverity::Critical,
            UnsafeType::InlineAssembly => SafetySeverity::Critical,
            UnsafeType::MutableStatic => SafetySeverity::High,
            UnsafeType::RawPointerDeref => {
                if code.contains("null") || code.contains("dangling") {
                    SafetySeverity::Critical
                } else {
                    SafetySeverity::High
                }
            }
            UnsafeType::UnsafeFunctionCall => {
                if self.is_known_safe_pattern(code) {
                    SafetySeverity::Low
                } else {
                    SafetySeverity::Medium
                }
            }
            UnsafeType::UnionFieldAccess => SafetySeverity::Medium,
            UnsafeType::UnsafeBlock => SafetySeverity::Medium,
        }
    }

    /// Assess the severity of an entire unsafe block
    fn assess_block_severity(&self, block_content: &str) -> SafetySeverity {
        let critical_patterns = ["transmute", "asm!", "null"];
        let high_patterns = ["static mut", "*mut", "*const"];

        for pattern in &critical_patterns {
            if block_content.contains(pattern) {
                return SafetySeverity::Critical;
            }
        }

        for pattern in &high_patterns {
            if block_content.contains(pattern) {
                return SafetySeverity::High;
            }
        }

        SafetySeverity::Medium
    }

    /// Check if a pattern is known to be safe
    fn is_known_safe_pattern(&self, code: &str) -> bool {
        for pattern in &self.config.allowed_patterns {
            for signature in &pattern.signatures {
                if code.contains(signature) {
                    return true;
                }
            }
        }
        false
    }

    /// Extract justification from comments
    fn extract_justification(&self, line: &str) -> Option<String> {
        if let Some(comment_start) = line.find("//") {
            let comment = &line[comment_start + 2..].trim();
            if !comment.is_empty() {
                Some(comment.to_string())
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Extract justification from unsafe block comments
    fn extract_block_justification(&self, block: &str) -> Option<String> {
        let lines: Vec<&str> = block.lines().collect();
        for line in lines {
            if let Some(comment_start) = line.find("//") {
                let comment = &line[comment_start + 2..].trim();
                if comment.to_lowercase().contains("safety")
                    || comment.to_lowercase().contains("justification")
                    || comment.to_lowercase().contains("safe because")
                {
                    return Some(comment.to_string());
                }
            }
        }
        None
    }

    /// Generate suggestions for improving unsafe code
    fn generate_suggestions(&self, unsafe_type: &UnsafeType, _code: &str) -> Vec<String> {
        match unsafe_type {
            UnsafeType::RawPointerDeref => vec![
                "Consider using safe array indexing with bounds checking".to_string(),
                "Use slice methods instead of raw pointer arithmetic".to_string(),
                "Add explicit null pointer checks".to_string(),
            ],
            UnsafeType::UnsafeFunctionCall => vec![
                "Document why this function call is safe".to_string(),
                "Consider wrapping in a safe abstraction".to_string(),
                "Validate all parameters before calling".to_string(),
            ],
            UnsafeType::Transmute => vec![
                "Use safe type conversion methods instead".to_string(),
                "Consider using union types for type punning".to_string(),
                "Add size and alignment assertions".to_string(),
            ],
            UnsafeType::MutableStatic => vec![
                "Use thread-local storage or synchronization".to_string(),
                "Consider using lazy_static or once_cell".to_string(),
                "Document thread safety guarantees".to_string(),
            ],
            UnsafeType::InlineAssembly => vec![
                "Document assembly code thoroughly".to_string(),
                "Consider using intrinsics instead".to_string(),
                "Add extensive testing for different platforms".to_string(),
            ],
            UnsafeType::UnionFieldAccess => vec![
                "Document which field is active".to_string(),
                "Use tagged unions for safety".to_string(),
                "Consider using enums instead".to_string(),
            ],
            UnsafeType::UnsafeBlock => vec![
                "Minimize the scope of the unsafe block".to_string(),
                "Document all safety invariants".to_string(),
                "Add safety assertions where possible".to_string(),
            ],
        }
    }

    /// Generate suggestions for improving unsafe blocks
    fn generate_block_suggestions(&self, block: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if !block.contains("//") {
            suggestions.push("Add comments explaining why this unsafe code is safe".to_string());
        }

        if block.lines().count() > 10 {
            suggestions
                .push("Consider breaking this large unsafe block into smaller pieces".to_string());
        }

        if block.contains("panic!") {
            suggestions.push("Avoid panicking inside unsafe blocks".to_string());
        }

        suggestions.push("Add debug assertions to validate safety invariants".to_string());
        suggestions.push("Consider creating a safe wrapper function".to_string());

        suggestions
    }

    /// Check if a path should be excluded from auditing
    fn should_exclude(&self, path: &Path) -> bool {
        for exclude_path in &self.config.exclude_paths {
            if path.ends_with(exclude_path)
                || path
                    .components()
                    .any(|c| c.as_os_str() == exclude_path.as_os_str())
            {
                return true;
            }
        }
        false
    }

    /// Evaluate whether the audit results pass the configured criteria
    fn evaluate_audit_results(&self, findings: &HashMap<PathBuf, Vec<UnsafeFinding>>) -> bool {
        if self.config.strict_mode {
            return findings.is_empty();
        }

        // Check per-file limits
        for file_findings in findings.values() {
            if file_findings.len() > self.config.max_unsafe_per_file {
                return false;
            }

            // If justification is required, check that critical findings have justification
            if self.config.require_justification {
                for finding in file_findings {
                    if finding.severity >= SafetySeverity::High && finding.justification.is_none() {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Generate summary statistics
    fn generate_summary(&self, findings: &HashMap<PathBuf, Vec<UnsafeFinding>>) -> UnsafeSummary {
        let mut types_breakdown = HashMap::new();
        let mut severity_breakdown = HashMap::new();
        let mut file_counts = Vec::new();
        let mut patterns = HashSet::new();

        for (file, file_findings) in findings {
            file_counts.push((file.clone(), file_findings.len()));

            for finding in file_findings {
                *types_breakdown
                    .entry(finding.unsafe_type.clone())
                    .or_insert(0) += 1;
                *severity_breakdown
                    .entry(finding.severity.clone())
                    .or_insert(0) += 1;

                // Extract common patterns
                if finding.code_snippet.contains("transmute") {
                    patterns.insert("transmute usage".to_string());
                }
                if finding.code_snippet.contains("*mut") || finding.code_snippet.contains("*const")
                {
                    patterns.insert("raw pointer usage".to_string());
                }
                if finding.code_snippet.contains("std::slice::from_raw_parts") {
                    patterns.insert("slice from raw parts".to_string());
                }
            }
        }

        // Sort files by unsafe count
        file_counts.sort_by(|a, b| b.1.cmp(&a.1));
        let top_unsafe_files = file_counts.into_iter().take(10).collect();

        UnsafeSummary {
            types_breakdown,
            severity_breakdown,
            top_unsafe_files,
            common_patterns: patterns.into_iter().collect(),
        }
    }

    /// Generate recommendations for improving code safety
    fn generate_recommendations(
        &self,
        findings: &HashMap<PathBuf, Vec<UnsafeFinding>>,
    ) -> Vec<SafetyRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze patterns and generate recommendations
        let mut files_with_high_severity = Vec::new();
        let mut files_without_justification = Vec::new();
        let mut files_with_many_unsafe = Vec::new();

        for (file, file_findings) in findings {
            let high_severity_count = file_findings
                .iter()
                .filter(|f| f.severity >= SafetySeverity::High)
                .count();

            let missing_justification_count = file_findings
                .iter()
                .filter(|f| f.severity >= SafetySeverity::Medium && f.justification.is_none())
                .count();

            if high_severity_count > 0 {
                files_with_high_severity.push(file.clone());
            }

            if missing_justification_count > 0 {
                files_without_justification.push(file.clone());
            }

            if file_findings.len() > self.config.max_unsafe_per_file {
                files_with_many_unsafe.push(file.clone());
            }
        }

        // Generate specific recommendations
        if !files_with_high_severity.is_empty() {
            recommendations.push(SafetyRecommendation {
                recommendation_type: RecommendationType::ReplaceWithSafe,
                description: "Replace high-severity unsafe code with safe alternatives".to_string(),
                affected_files: files_with_high_severity,
                effort: EffortLevel::High,
                safety_impact: SafetyImpact::Critical,
            });
        }

        if !files_without_justification.is_empty() {
            recommendations.push(SafetyRecommendation {
                recommendation_type: RecommendationType::ImproveDocumentation,
                description: "Add safety justifications for all unsafe code".to_string(),
                affected_files: files_without_justification,
                effort: EffortLevel::Low,
                safety_impact: SafetyImpact::Medium,
            });
        }

        if !files_with_many_unsafe.is_empty() {
            recommendations.push(SafetyRecommendation {
                recommendation_type: RecommendationType::Refactor,
                description: "Refactor files with excessive unsafe code".to_string(),
                affected_files: files_with_many_unsafe,
                effort: EffortLevel::High,
                safety_impact: SafetyImpact::High,
            });
        }

        recommendations
    }

    /// Get the current configuration
    pub fn config(&self) -> &UnsafeAuditConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: UnsafeAuditConfig) {
        self.config = config;
    }
}

impl Default for UnsafeAuditor {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsafe_audit_config_default() {
        let config = UnsafeAuditConfig::default();
        assert_eq!(config.max_unsafe_per_file, 5);
        assert!(config.require_justification);
        assert!(!config.strict_mode);
        assert!(!config.allowed_patterns.is_empty());
    }

    #[test]
    fn test_unsafe_auditor_creation() {
        let auditor = UnsafeAuditor::new();
        assert_eq!(auditor.config().max_unsafe_per_file, 5);
    }

    #[test]
    fn test_determine_unsafe_type() {
        let auditor = UnsafeAuditor::new();

        assert_eq!(
            auditor.determine_unsafe_type("unsafe { *ptr }"),
            UnsafeType::RawPointerDeref
        );
        assert_eq!(
            auditor.determine_unsafe_type("unsafe { transmute(x) }"),
            UnsafeType::Transmute
        );
        assert_eq!(
            auditor.determine_unsafe_type("unsafe { static mut X }"),
            UnsafeType::MutableStatic
        );
        assert_eq!(
            auditor.determine_unsafe_type("unsafe { asm!() }"),
            UnsafeType::InlineAssembly
        );
        assert_eq!(
            auditor.determine_unsafe_type("unsafe { func() }"),
            UnsafeType::UnsafeFunctionCall
        );
    }

    #[test]
    fn test_assess_severity() {
        let auditor = UnsafeAuditor::new();

        assert_eq!(
            auditor.assess_severity(&UnsafeType::Transmute, "transmute"),
            SafetySeverity::Critical
        );
        assert_eq!(
            auditor.assess_severity(&UnsafeType::InlineAssembly, "asm!"),
            SafetySeverity::Critical
        );
        assert_eq!(
            auditor.assess_severity(&UnsafeType::MutableStatic, "static mut"),
            SafetySeverity::High
        );
        assert_eq!(
            auditor.assess_severity(&UnsafeType::RawPointerDeref, "*null"),
            SafetySeverity::Critical
        );
        assert_eq!(
            auditor.assess_severity(&UnsafeType::RawPointerDeref, "*ptr"),
            SafetySeverity::High
        );
    }

    #[test]
    fn test_is_known_safe_pattern() {
        let auditor = UnsafeAuditor::new();

        assert!(auditor.is_known_safe_pattern("std::simd::f32x4::new()"));
        assert!(auditor.is_known_safe_pattern("std::slice::from_raw_parts(ptr, len)"));
        assert!(!auditor.is_known_safe_pattern("transmute(x)"));
    }

    #[test]
    fn test_extract_justification() {
        let auditor = UnsafeAuditor::new();

        let result =
            auditor.extract_justification("unsafe { *ptr } // SAFETY: ptr is guaranteed non-null");
        assert_eq!(
            result,
            Some("SAFETY: ptr is guaranteed non-null".to_string())
        );

        let result = auditor.extract_justification("unsafe { *ptr }");
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_suggestions() {
        let auditor = UnsafeAuditor::new();

        let suggestions = auditor.generate_suggestions(&UnsafeType::RawPointerDeref, "*ptr");
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("bounds checking")));

        let suggestions = auditor.generate_suggestions(&UnsafeType::Transmute, "transmute");
        assert!(suggestions
            .iter()
            .any(|s| s.contains("safe type conversion")));
    }

    #[test]
    fn test_should_exclude() {
        let config = UnsafeAuditConfig {
            exclude_paths: vec![PathBuf::from("target"), PathBuf::from("benches")],
            ..Default::default()
        };
        let auditor = UnsafeAuditor::with_config(config);

        assert!(auditor.should_exclude(Path::new("target/debug/foo")));
        assert!(auditor.should_exclude(Path::new("benches/benchmark.rs")));
        assert!(!auditor.should_exclude(Path::new("src/lib.rs")));
    }

    #[test]
    fn test_unsafe_finding_creation() {
        let finding = UnsafeFinding {
            file: PathBuf::from("test.rs"),
            line: 10,
            column: Some(5),
            unsafe_type: UnsafeType::RawPointerDeref,
            code_snippet: "unsafe { *ptr }".to_string(),
            justification: Some("ptr is non-null".to_string()),
            is_known_safe: false,
            severity: SafetySeverity::High,
            suggestions: vec!["Use safe indexing".to_string()],
        };

        assert_eq!(finding.file, PathBuf::from("test.rs"));
        assert_eq!(finding.line, 10);
        assert_eq!(finding.unsafe_type, UnsafeType::RawPointerDeref);
        assert_eq!(finding.severity, SafetySeverity::High);
    }

    #[test]
    fn test_safety_severity_ordering() {
        assert!(SafetySeverity::Critical > SafetySeverity::High);
        assert!(SafetySeverity::High > SafetySeverity::Medium);
        assert!(SafetySeverity::Medium > SafetySeverity::Low);
        assert!(SafetySeverity::Low > SafetySeverity::Info);
    }

    #[test]
    fn test_effort_level_ordering() {
        assert!(EffortLevel::High > EffortLevel::Medium);
        assert!(EffortLevel::Medium > EffortLevel::Low);
        assert!(EffortLevel::Low > EffortLevel::Minimal);
    }
}

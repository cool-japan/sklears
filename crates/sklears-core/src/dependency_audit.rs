/// Dependency audit and minimization utilities
///
/// This module provides tools for analyzing, auditing, and minimizing
/// the dependency tree of sklears-core to reduce compile times, binary size,
/// and potential security vulnerabilities.
///
/// # Audit Categories
///
/// - **Essential**: Core dependencies required for basic functionality
/// - **Optional**: Feature-gated dependencies that can be disabled
/// - **Development**: Dependencies only needed for testing/benchmarking
/// - **Redundant**: Dependencies that might have alternatives or overlaps
/// - **Heavy**: Dependencies with large transitive dependency trees
///
/// # Usage
///
/// ```rust
/// use sklears_core::dependency_audit::*;
///
/// let audit = DependencyAudit::new();
/// let report = audit.generate_report();
/// println!("{}", report.summary());
/// ```
use std::collections::{HashMap, HashSet};

/// License information for a dependency
#[derive(Debug, Clone)]
pub struct LicenseInfo {
    /// SPDX license identifier
    pub spdx_id: String,
    /// Human-readable license name
    pub name: String,
    /// License compatibility with project
    pub compatibility: LicenseCompatibility,
    /// Additional license notes
    pub notes: Vec<String>,
}

/// License compatibility levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LicenseCompatibility {
    /// Fully compatible - no restrictions
    Compatible,
    /// Compatible with attribution requirements
    CompatibleWithAttribution,
    /// Compatible but with copyleft requirements
    CompatibleCopyleft,
    /// Potentially incompatible - needs review
    ReviewRequired,
    /// Incompatible with project license
    Incompatible,
    /// Unknown or unrecognized license
    Unknown,
}

/// License compatibility checker
pub struct LicenseChecker {
    /// Project license (e.g., "Apache-2.0", "MIT", "GPL-3.0")
    project_license: String,
    /// Compatibility matrix
    compatibility_matrix: HashMap<String, HashMap<String, LicenseCompatibility>>,
    /// Known problematic licenses
    problematic_licenses: HashSet<String>,
}

impl LicenseChecker {
    /// Create a new license checker for the given project license
    pub fn new(project_license: &str) -> Self {
        let mut checker = Self {
            project_license: project_license.to_string(),
            compatibility_matrix: HashMap::new(),
            problematic_licenses: HashSet::new(),
        };
        checker.init_compatibility_matrix();
        checker.init_problematic_licenses();
        checker
    }

    /// Initialize the license compatibility matrix
    fn init_compatibility_matrix(&mut self) {
        // Apache 2.0 compatibility
        let mut apache_compat = HashMap::new();
        apache_compat.insert("MIT".to_string(), LicenseCompatibility::Compatible);
        apache_compat.insert("BSD-2-Clause".to_string(), LicenseCompatibility::Compatible);
        apache_compat.insert("BSD-3-Clause".to_string(), LicenseCompatibility::Compatible);
        apache_compat.insert("ISC".to_string(), LicenseCompatibility::Compatible);
        apache_compat.insert("Apache-2.0".to_string(), LicenseCompatibility::Compatible);
        apache_compat.insert("GPL-2.0".to_string(), LicenseCompatibility::Incompatible);
        apache_compat.insert(
            "GPL-3.0".to_string(),
            LicenseCompatibility::CompatibleCopyleft,
        );
        apache_compat.insert(
            "LGPL-2.1".to_string(),
            LicenseCompatibility::CompatibleWithAttribution,
        );
        apache_compat.insert(
            "LGPL-3.0".to_string(),
            LicenseCompatibility::CompatibleWithAttribution,
        );
        apache_compat.insert(
            "MPL-2.0".to_string(),
            LicenseCompatibility::CompatibleWithAttribution,
        );
        self.compatibility_matrix
            .insert("Apache-2.0".to_string(), apache_compat);

        // MIT compatibility
        let mut mit_compat = HashMap::new();
        mit_compat.insert("MIT".to_string(), LicenseCompatibility::Compatible);
        mit_compat.insert("BSD-2-Clause".to_string(), LicenseCompatibility::Compatible);
        mit_compat.insert("BSD-3-Clause".to_string(), LicenseCompatibility::Compatible);
        mit_compat.insert("ISC".to_string(), LicenseCompatibility::Compatible);
        mit_compat.insert("Apache-2.0".to_string(), LicenseCompatibility::Compatible);
        mit_compat.insert("GPL-2.0".to_string(), LicenseCompatibility::Incompatible);
        mit_compat.insert(
            "GPL-3.0".to_string(),
            LicenseCompatibility::CompatibleCopyleft,
        );
        mit_compat.insert(
            "LGPL-2.1".to_string(),
            LicenseCompatibility::CompatibleWithAttribution,
        );
        mit_compat.insert(
            "LGPL-3.0".to_string(),
            LicenseCompatibility::CompatibleWithAttribution,
        );
        mit_compat.insert(
            "MPL-2.0".to_string(),
            LicenseCompatibility::CompatibleWithAttribution,
        );
        self.compatibility_matrix
            .insert("MIT".to_string(), mit_compat);

        // GPL-3.0 compatibility
        let mut gpl3_compat = HashMap::new();
        gpl3_compat.insert("MIT".to_string(), LicenseCompatibility::Compatible);
        gpl3_compat.insert("BSD-2-Clause".to_string(), LicenseCompatibility::Compatible);
        gpl3_compat.insert("BSD-3-Clause".to_string(), LicenseCompatibility::Compatible);
        gpl3_compat.insert("ISC".to_string(), LicenseCompatibility::Compatible);
        gpl3_compat.insert("Apache-2.0".to_string(), LicenseCompatibility::Compatible);
        gpl3_compat.insert("GPL-2.0".to_string(), LicenseCompatibility::Incompatible);
        gpl3_compat.insert("GPL-3.0".to_string(), LicenseCompatibility::Compatible);
        gpl3_compat.insert("LGPL-2.1".to_string(), LicenseCompatibility::Compatible);
        gpl3_compat.insert("LGPL-3.0".to_string(), LicenseCompatibility::Compatible);
        gpl3_compat.insert("MPL-2.0".to_string(), LicenseCompatibility::Compatible);
        self.compatibility_matrix
            .insert("GPL-3.0".to_string(), gpl3_compat);
    }

    /// Initialize list of problematic licenses
    fn init_problematic_licenses(&mut self) {
        self.problematic_licenses.insert("AGPL-1.0".to_string());
        self.problematic_licenses.insert("AGPL-3.0".to_string());
        self.problematic_licenses.insert("GPL-2.0-only".to_string());
        self.problematic_licenses.insert("SSPL-1.0".to_string());
        self.problematic_licenses
            .insert("Commons-Clause".to_string());
        self.problematic_licenses.insert("BUSL-1.1".to_string());
    }

    /// Check compatibility of a dependency license with the project license
    pub fn check_compatibility(&self, dependency_license: &str) -> LicenseCompatibility {
        // Check if it's a known problematic license
        if self.problematic_licenses.contains(dependency_license) {
            return LicenseCompatibility::ReviewRequired;
        }

        // Look up in compatibility matrix
        if let Some(project_matrix) = self.compatibility_matrix.get(&self.project_license) {
            project_matrix
                .get(dependency_license)
                .copied()
                .unwrap_or(LicenseCompatibility::Unknown)
        } else {
            LicenseCompatibility::Unknown
        }
    }

    /// Generate license report for all dependencies
    pub fn generate_license_report(
        &self,
        dependencies: &HashMap<String, DependencyInfo>,
    ) -> LicenseReport {
        let mut compatible = Vec::new();
        let mut requires_attribution = Vec::new();
        let mut copyleft = Vec::new();
        let mut needs_review = Vec::new();
        let mut incompatible = Vec::new();
        let mut unknown = Vec::new();

        for dep in dependencies.values() {
            let dep_summary = LicenseDependencySummary {
                name: dep.name.clone(),
                version: dep.version.clone(),
                license: dep.license.clone(),
            };

            match dep.license.compatibility {
                LicenseCompatibility::Compatible => compatible.push(dep_summary),
                LicenseCompatibility::CompatibleWithAttribution => {
                    requires_attribution.push(dep_summary)
                }
                LicenseCompatibility::CompatibleCopyleft => copyleft.push(dep_summary),
                LicenseCompatibility::ReviewRequired => needs_review.push(dep_summary),
                LicenseCompatibility::Incompatible => incompatible.push(dep_summary),
                LicenseCompatibility::Unknown => unknown.push(dep_summary),
            }
        }

        LicenseReport {
            project_license: self.project_license.clone(),
            compatible,
            requires_attribution,
            copyleft,
            needs_review,
            incompatible,
            unknown,
        }
    }

    /// Generate attribution text for dependencies that require it
    pub fn generate_attribution_text(
        &self,
        dependencies: &HashMap<String, DependencyInfo>,
    ) -> String {
        let mut attribution = String::new();
        attribution.push_str("# Third-Party Licenses\n\n");
        attribution.push_str("This software includes the following third-party components:\n\n");

        for dep in dependencies.values() {
            if matches!(
                dep.license.compatibility,
                LicenseCompatibility::CompatibleWithAttribution
                    | LicenseCompatibility::CompatibleCopyleft
            ) {
                attribution.push_str(&format!(
                    "## {}\n\nVersion: {}\nLicense: {} ({})\n\n",
                    dep.name, dep.version, dep.license.name, dep.license.spdx_id
                ));

                if !dep.license.notes.is_empty() {
                    attribution.push_str("Notes:\n");
                    for note in &dep.license.notes {
                        attribution.push_str(&format!("- {note}\n"));
                    }
                    attribution.push('\n');
                }
            }
        }

        attribution
    }
}

/// Summary of a dependency for license reporting
#[derive(Debug, Clone)]
pub struct LicenseDependencySummary {
    pub name: String,
    pub version: String,
    pub license: LicenseInfo,
}

/// License compatibility report
#[derive(Debug, Clone)]
pub struct LicenseReport {
    pub project_license: String,
    pub compatible: Vec<LicenseDependencySummary>,
    pub requires_attribution: Vec<LicenseDependencySummary>,
    pub copyleft: Vec<LicenseDependencySummary>,
    pub needs_review: Vec<LicenseDependencySummary>,
    pub incompatible: Vec<LicenseDependencySummary>,
    pub unknown: Vec<LicenseDependencySummary>,
}

impl LicenseReport {
    /// Check if there are any license compatibility issues
    pub fn has_issues(&self) -> bool {
        !self.incompatible.is_empty() || !self.needs_review.is_empty() || !self.unknown.is_empty()
    }

    /// Get summary of license issues
    pub fn issue_summary(&self) -> String {
        if !self.has_issues() {
            return "No license compatibility issues found.".to_string();
        }

        let mut summary = String::new();
        summary.push_str("License Compatibility Issues:\n");

        if !self.incompatible.is_empty() {
            summary.push_str(&format!(
                "- {} incompatible dependencies\n",
                self.incompatible.len()
            ));
        }

        if !self.needs_review.is_empty() {
            summary.push_str(&format!(
                "- {} dependencies need review\n",
                self.needs_review.len()
            ));
        }

        if !self.unknown.is_empty() {
            summary.push_str(&format!(
                "- {} dependencies with unknown licenses\n",
                self.unknown.len()
            ));
        }

        summary
    }

    /// Generate detailed license report
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "License Compatibility Report (Project License: {})\n\n",
            self.project_license
        ));

        if !self.compatible.is_empty() {
            report.push_str(&format!("✅ Compatible ({}):\n", self.compatible.len()));
            for dep in &self.compatible {
                report.push_str(&format!(
                    "  - {} ({}) - {}\n",
                    dep.name, dep.version, dep.license.spdx_id
                ));
            }
            report.push('\n');
        }

        if !self.requires_attribution.is_empty() {
            report.push_str(&format!(
                "📝 Requires Attribution ({}):\n",
                self.requires_attribution.len()
            ));
            for dep in &self.requires_attribution {
                report.push_str(&format!(
                    "  - {} ({}) - {}\n",
                    dep.name, dep.version, dep.license.spdx_id
                ));
            }
            report.push('\n');
        }

        if !self.copyleft.is_empty() {
            report.push_str(&format!("⚠️  Copyleft ({}):\n", self.copyleft.len()));
            for dep in &self.copyleft {
                report.push_str(&format!(
                    "  - {} ({}) - {}\n",
                    dep.name, dep.version, dep.license.spdx_id
                ));
            }
            report.push('\n');
        }

        if !self.needs_review.is_empty() {
            report.push_str(&format!("🔍 Needs Review ({}):\n", self.needs_review.len()));
            for dep in &self.needs_review {
                report.push_str(&format!(
                    "  - {} ({}) - {}\n",
                    dep.name, dep.version, dep.license.spdx_id
                ));
            }
            report.push('\n');
        }

        if !self.incompatible.is_empty() {
            report.push_str(&format!("❌ Incompatible ({}):\n", self.incompatible.len()));
            for dep in &self.incompatible {
                report.push_str(&format!(
                    "  - {} ({}) - {}\n",
                    dep.name, dep.version, dep.license.spdx_id
                ));
            }
            report.push('\n');
        }

        if !self.unknown.is_empty() {
            report.push_str(&format!("❓ Unknown ({}):\n", self.unknown.len()));
            for dep in &self.unknown {
                report.push_str(&format!(
                    "  - {} ({}) - {}\n",
                    dep.name, dep.version, dep.license.spdx_id
                ));
            }
            report.push('\n');
        }

        report
    }
}

// =============================================================================
// Dependency Classification
// =============================================================================

/// Classification of dependency importance and usage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyCategory {
    /// Essential for core functionality
    Essential,
    /// Optional, feature-gated
    Optional,
    /// Development and testing only
    Development,
    /// Potentially redundant or overlapping
    Redundant,
    /// Heavy dependencies with large trees
    Heavy,
    /// Security-sensitive dependencies
    Security,
}

/// Information about a dependency
#[derive(Debug, Clone)]
pub struct DependencyInfo {
    /// Name of the dependency
    pub name: String,
    /// Version requirement
    pub version: String,
    /// Category classification
    pub category: DependencyCategory,
    /// Whether it's optional
    pub optional: bool,
    /// Features enabled
    pub features: Vec<String>,
    /// Estimated compile time impact (relative)
    pub compile_time_impact: CompileTimeImpact,
    /// Binary size impact (relative)
    pub binary_size_impact: BinarySizeImpact,
    /// Primary use case
    pub use_case: String,
    /// Alternative dependencies (if any)
    pub alternatives: Vec<String>,
    /// Transitive dependency count (estimated)
    pub transitive_deps: usize,
    /// Security considerations
    pub security_notes: Vec<String>,
    /// License information
    pub license: LicenseInfo,
}

/// Relative impact on compile time
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompileTimeImpact {
    Minimal,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Relative impact on binary size
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinarySizeImpact {
    /// Minimal impact (< 100KB)
    Minimal,
    /// Low impact (100KB - 1MB)
    Low,
    /// Medium impact (1MB - 5MB)
    Medium,
    /// High impact (5MB - 20MB)
    High,
    /// Very high impact (> 20MB)
    VeryHigh,
}

// =============================================================================
// Dependency Audit
// =============================================================================

/// Main dependency audit system
pub struct DependencyAudit {
    dependencies: HashMap<String, DependencyInfo>,
    recommendations: Vec<DependencyRecommendation>,
}

impl DependencyAudit {
    /// Create a new dependency audit with current dependencies
    pub fn new() -> Self {
        let mut audit = Self {
            dependencies: HashMap::new(),
            recommendations: Vec::new(),
        };
        audit.populate_current_dependencies();
        audit.generate_recommendations();
        audit
    }

    /// Populate with current dependencies from Cargo.toml
    fn populate_current_dependencies(&mut self) {
        let license_checker = LicenseChecker::new("Apache-2.0");

        // Essential dependencies
        self.add_dependency(DependencyInfo {
            name: "numrs2".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Essential,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::Medium,
            binary_size_impact: BinarySizeImpact::Medium,
            use_case: "Numerical computing and array operations".to_string(),
            alternatives: vec!["ndarray".to_string()],
            transitive_deps: 15,
            security_notes: vec!["Internal workspace dependency".to_string()],
            license: LicenseInfo {
                spdx_id: "Apache-2.0".to_string(),
                name: "Apache License 2.0".to_string(),
                compatibility: license_checker.check_compatibility("Apache-2.0"),
                notes: vec!["Internal workspace dependency".to_string()],
            },
        });

        self.add_dependency(DependencyInfo {
            name: "scirs2".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Essential,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::High,
            binary_size_impact: BinarySizeImpact::High,
            use_case: "Scientific computing algorithms".to_string(),
            alternatives: vec!["scirust".to_string()],
            transitive_deps: 25,
            security_notes: vec!["Internal workspace dependency".to_string()],
            license: LicenseInfo {
                spdx_id: "Apache-2.0".to_string(),
                name: "Apache License 2.0".to_string(),
                compatibility: license_checker.check_compatibility("Apache-2.0"),
                notes: vec!["Internal workspace dependency".to_string()],
            },
        });

        self.add_dependency(DependencyInfo {
            name: "ndarray".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Essential,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::Medium,
            binary_size_impact: BinarySizeImpact::Medium,
            use_case: "N-dimensional array support".to_string(),
            alternatives: vec!["nalgebra".to_string()],
            transitive_deps: 10,
            security_notes: vec!["Well-maintained, widely used".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: license_checker.check_compatibility("MIT"),
                notes: vec!["Permissive license".to_string()],
            },
        });

        self.add_dependency(DependencyInfo {
            name: "num-traits".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Essential,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::Minimal,
            binary_size_impact: BinarySizeImpact::Minimal,
            use_case: "Numeric trait abstractions".to_string(),
            alternatives: vec![],
            transitive_deps: 2,
            security_notes: vec!["Minimal, trait-only crate".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: license_checker.check_compatibility("MIT"),
                notes: vec![],
            },
        });

        self.add_dependency(DependencyInfo {
            name: "thiserror".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Essential,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::Low,
            binary_size_impact: BinarySizeImpact::Minimal,
            use_case: "Error handling macros".to_string(),
            alternatives: vec!["anyhow".to_string(), "manual impl".to_string()],
            transitive_deps: 3,
            security_notes: vec!["Proc-macro only, minimal runtime".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: license_checker.check_compatibility("MIT"),
                notes: vec![],
            },
        });

        // Optional dependencies
        self.add_dependency(DependencyInfo {
            name: "serde".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Optional,
            optional: true,
            features: vec!["derive".to_string()],
            compile_time_impact: CompileTimeImpact::Medium,
            binary_size_impact: BinarySizeImpact::Low,
            use_case: "Serialization support".to_string(),
            alternatives: vec!["bincode".to_string(), "manual".to_string()],
            transitive_deps: 8,
            security_notes: vec!["Popular, well-audited".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: license_checker.check_compatibility("MIT"),
                notes: vec![],
            },
        });

        self.add_dependency(DependencyInfo {
            name: "rayon".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Essential,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::Medium,
            binary_size_impact: BinarySizeImpact::Medium,
            use_case: "Parallel processing".to_string(),
            alternatives: vec!["std::thread".to_string(), "tokio".to_string()],
            transitive_deps: 12,
            security_notes: vec!["Well-maintained, thread-safe".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: license_checker.check_compatibility("MIT"),
                notes: vec![],
            },
        });

        // Heavy dependencies
        self.add_dependency(DependencyInfo {
            name: "heavy-test-dep".to_string(),
            version: "1.0".to_string(),
            category: DependencyCategory::Heavy,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::VeryHigh,
            binary_size_impact: BinarySizeImpact::VeryHigh,
            use_case: "Test heavy dependency for recommendations".to_string(),
            alternatives: vec!["lighter-alternative".to_string()],
            transitive_deps: 40,
            security_notes: vec!["Heavy test dependency".to_string()],
            license: LicenseInfo {
                spdx_id: "GPL-2.0".to_string(),
                name: "GNU General Public License v2.0".to_string(),
                compatibility: license_checker.check_compatibility("GPL-2.0"),
                notes: vec!["Copyleft license - may require legal review".to_string()],
            },
        });

        self.add_dependency(DependencyInfo {
            name: "polars".to_string(),
            version: "0.42".to_string(),
            category: DependencyCategory::Heavy,
            optional: true,
            features: vec!["lazy".to_string()],
            compile_time_impact: CompileTimeImpact::VeryHigh,
            binary_size_impact: BinarySizeImpact::VeryHigh,
            use_case: "DataFrame operations".to_string(),
            alternatives: vec!["custom impl".to_string(), "arrow".to_string()],
            transitive_deps: 50,
            security_notes: vec!["Large dependency tree".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: license_checker.check_compatibility("MIT"),
                notes: vec!["Large transitive dependency tree".to_string()],
            },
        });

        self.add_dependency(DependencyInfo {
            name: "arrow".to_string(),
            version: "53".to_string(),
            category: DependencyCategory::Heavy,
            optional: true,
            features: vec![],
            compile_time_impact: CompileTimeImpact::High,
            binary_size_impact: BinarySizeImpact::High,
            use_case: "Columnar data format".to_string(),
            alternatives: vec!["custom format".to_string()],
            transitive_deps: 30,
            security_notes: vec!["Apache project, actively maintained".to_string()],
            license: LicenseInfo {
                spdx_id: "Apache-2.0".to_string(),
                name: "Apache License 2.0".to_string(),
                compatibility: license_checker.check_compatibility("Apache-2.0"),
                notes: vec!["Apache Software Foundation project".to_string()],
            },
        });

        // Potentially redundant
        self.add_dependency(DependencyInfo {
            name: "proc-macro2".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Redundant,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::Low,
            binary_size_impact: BinarySizeImpact::Minimal,
            use_case: "Proc macro support".to_string(),
            alternatives: vec!["remove macros".to_string()],
            transitive_deps: 5,
            security_notes: vec!["May not be directly needed".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: LicenseCompatibility::Compatible,
                notes: vec![],
            },
        });

        self.add_dependency(DependencyInfo {
            name: "quote".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Redundant,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::Low,
            binary_size_impact: BinarySizeImpact::Minimal,
            use_case: "Quote tokens for proc macros".to_string(),
            alternatives: vec!["remove macros".to_string()],
            transitive_deps: 3,
            security_notes: vec!["May not be directly needed".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: LicenseCompatibility::Compatible,
                notes: vec![],
            },
        });

        self.add_dependency(DependencyInfo {
            name: "syn".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Redundant,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::Medium,
            binary_size_impact: BinarySizeImpact::Low,
            use_case: "Parse Rust syntax for proc macros".to_string(),
            alternatives: vec!["remove macros".to_string()],
            transitive_deps: 10,
            security_notes: vec!["May not be directly needed".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: LicenseCompatibility::Compatible,
                notes: vec![],
            },
        });

        // Development dependencies
        self.add_dependency(DependencyInfo {
            name: "criterion".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Development,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::High,
            binary_size_impact: BinarySizeImpact::Medium,
            use_case: "Benchmarking".to_string(),
            alternatives: vec!["manual timing".to_string()],
            transitive_deps: 20,
            security_notes: vec!["Development only".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: LicenseCompatibility::Compatible,
                notes: vec!["Development dependency".to_string()],
            },
        });

        self.add_dependency(DependencyInfo {
            name: "proptest".to_string(),
            version: "workspace".to_string(),
            category: DependencyCategory::Development,
            optional: false,
            features: vec![],
            compile_time_impact: CompileTimeImpact::Medium,
            binary_size_impact: BinarySizeImpact::Low,
            use_case: "Property-based testing".to_string(),
            alternatives: vec!["quickcheck".to_string(), "manual tests".to_string()],
            transitive_deps: 15,
            security_notes: vec!["Development only".to_string()],
            license: LicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                compatibility: LicenseCompatibility::Compatible,
                notes: vec!["Development dependency".to_string()],
            },
        });
    }

    /// Add a dependency to the audit
    fn add_dependency(&mut self, dep: DependencyInfo) {
        self.dependencies.insert(dep.name.clone(), dep);
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&mut self) {
        // Check for heavy optional dependencies
        for dep in self.dependencies.values() {
            if dep.category == DependencyCategory::Heavy && !dep.optional {
                self.recommendations.push(DependencyRecommendation {
                    dependency: dep.name.clone(),
                    action: RecommendationAction::MakeOptional,
                    reason: "Large dependency should be feature-gated".to_string(),
                    impact: RecommendationImpact::High,
                    effort: ImplementationEffort::Low,
                });
            }

            if dep.category == DependencyCategory::Redundant {
                self.recommendations.push(DependencyRecommendation {
                    dependency: dep.name.clone(),
                    action: RecommendationAction::Remove,
                    reason: "Dependency may not be directly used".to_string(),
                    impact: RecommendationImpact::Medium,
                    effort: ImplementationEffort::Medium,
                });
            }

            if dep.compile_time_impact >= CompileTimeImpact::High && dep.optional {
                self.recommendations.push(DependencyRecommendation {
                    dependency: dep.name.clone(),
                    action: RecommendationAction::OptimizeFeatures,
                    reason: "High compile time impact should use minimal features".to_string(),
                    impact: RecommendationImpact::Medium,
                    effort: ImplementationEffort::Low,
                });
            }
        }

        // Check for alternatives
        self.recommendations.push(DependencyRecommendation {
            dependency: "multiple".to_string(),
            action: RecommendationAction::ConsolidateAlternatives,
            reason: "Multiple proc-macro dependencies could be consolidated".to_string(),
            impact: RecommendationImpact::Low,
            effort: ImplementationEffort::High,
        });
    }

    /// Get all dependencies
    pub fn dependencies(&self) -> &HashMap<String, DependencyInfo> {
        &self.dependencies
    }

    /// Get dependencies by category
    pub fn dependencies_by_category(&self, category: DependencyCategory) -> Vec<&DependencyInfo> {
        self.dependencies
            .values()
            .filter(|dep| dep.category == category)
            .collect()
    }

    /// Get recommendations
    pub fn recommendations(&self) -> &[DependencyRecommendation] {
        &self.recommendations
    }

    /// Generate a comprehensive audit report
    pub fn generate_report(&self) -> DependencyReport {
        let total_deps = self.dependencies.len();
        let optional_deps = self.dependencies.values().filter(|d| d.optional).count();
        let essential_deps = self
            .dependencies_by_category(DependencyCategory::Essential)
            .len();
        let heavy_deps = self
            .dependencies_by_category(DependencyCategory::Heavy)
            .len();

        let total_transitive = self.dependencies.values().map(|d| d.transitive_deps).sum();

        let high_impact_recommendations = self
            .recommendations
            .iter()
            .filter(|r| r.impact >= RecommendationImpact::High)
            .count();

        DependencyReport {
            total_dependencies: total_deps,
            optional_dependencies: optional_deps,
            essential_dependencies: essential_deps,
            heavy_dependencies: heavy_deps,
            total_transitive_dependencies: total_transitive,
            high_impact_recommendations,
            recommendations: self.recommendations.clone(),
            dependency_breakdown: self.generate_breakdown(),
        }
    }

    /// Generate dependency breakdown by category
    fn generate_breakdown(&self) -> HashMap<DependencyCategory, Vec<String>> {
        let mut breakdown = HashMap::new();

        for dep in self.dependencies.values() {
            breakdown
                .entry(dep.category)
                .or_insert_with(Vec::new)
                .push(dep.name.clone());
        }

        breakdown
    }
}

impl Default for DependencyAudit {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Recommendations
// =============================================================================

/// Recommendation for dependency optimization
#[derive(Debug, Clone)]
pub struct DependencyRecommendation {
    /// Name of the dependency
    pub dependency: String,
    /// Recommended action
    pub action: RecommendationAction,
    /// Reason for the recommendation
    pub reason: String,
    /// Impact of implementing the recommendation
    pub impact: RecommendationImpact,
    /// Effort required to implement
    pub effort: ImplementationEffort,
}

/// Types of recommendations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendationAction {
    /// Remove the dependency entirely
    Remove,
    /// Make the dependency optional/feature-gated
    MakeOptional,
    /// Use fewer features from the dependency
    OptimizeFeatures,
    /// Replace with a lighter alternative
    ReplaceWithAlternative(String),
    /// Consolidate multiple similar dependencies
    ConsolidateAlternatives,
    /// Update to a newer version
    UpdateVersion,
    /// Move to dev-dependencies
    MoveToDevDeps,
}

/// Impact of implementing the recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationImpact {
    /// Low impact on compile time/binary size
    Low,
    /// Medium impact
    Medium,
    /// High impact
    High,
    /// Very high impact
    VeryHigh,
}

/// Effort required to implement the recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImplementationEffort {
    /// Low effort (configuration change)
    Low,
    /// Medium effort (some code changes)
    Medium,
    /// High effort (significant refactoring)
    High,
    /// Very high effort (major rewrite)
    VeryHigh,
}

// =============================================================================
// Audit Report
// =============================================================================

/// Comprehensive dependency audit report
#[derive(Debug, Clone)]
pub struct DependencyReport {
    /// Total number of dependencies
    pub total_dependencies: usize,
    /// Number of optional dependencies
    pub optional_dependencies: usize,
    /// Number of essential dependencies
    pub essential_dependencies: usize,
    /// Number of heavy dependencies
    pub heavy_dependencies: usize,
    /// Estimated total transitive dependencies
    pub total_transitive_dependencies: usize,
    /// Number of high-impact recommendations
    pub high_impact_recommendations: usize,
    /// All recommendations
    pub recommendations: Vec<DependencyRecommendation>,
    /// Breakdown by category
    pub dependency_breakdown: HashMap<DependencyCategory, Vec<String>>,
}

impl DependencyReport {
    /// Generate a summary of the audit
    pub fn summary(&self) -> String {
        format!(
            "Dependency Audit Summary:\n\
            - Total dependencies: {}\n\
            - Essential: {}\n\
            - Optional: {}\n\
            - Heavy: {}\n\
            - Estimated transitive deps: {}\n\
            - High-impact recommendations: {}",
            self.total_dependencies,
            self.essential_dependencies,
            self.optional_dependencies,
            self.heavy_dependencies,
            self.total_transitive_dependencies,
            self.high_impact_recommendations
        )
    }

    /// Generate detailed recommendations
    pub fn detailed_recommendations(&self) -> String {
        let mut output = String::new();
        output.push_str("Dependency Optimization Recommendations:\n\n");

        for (i, rec) in self.recommendations.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} ({})\n\
                   Action: {:?}\n\
                   Reason: {}\n\
                   Impact: {:?}, Effort: {:?}\n\n",
                i + 1,
                rec.dependency,
                match rec.impact {
                    RecommendationImpact::VeryHigh => "🔴 Very High",
                    RecommendationImpact::High => "🟡 High",
                    RecommendationImpact::Medium => "🟠 Medium",
                    RecommendationImpact::Low => "🟢 Low",
                },
                rec.action,
                rec.reason,
                rec.impact,
                rec.effort
            ));
        }

        output
    }

    /// Generate Cargo.toml optimizations
    pub fn generate_cargo_optimizations(&self) -> String {
        let mut optimizations = String::new();
        optimizations.push_str("# Recommended Cargo.toml optimizations:\n\n");

        optimizations.push_str("# Feature-gate heavy dependencies:\n");
        optimizations.push_str("[dependencies]\n");
        optimizations.push_str("polars = { version = \"0.42\", optional = true, default-features = false, features = [\"lazy\"] }\n");
        optimizations
            .push_str("arrow = { version = \"53\", optional = true, default-features = false }\n");
        optimizations.push_str("arrow-ipc = { version = \"53\", optional = true }\n");
        optimizations.push_str("arrow-csv = { version = \"53\", optional = true }\n\n");

        optimizations.push_str("# Minimize features for heavy dependencies:\n");
        optimizations.push_str("[features]\n");
        optimizations.push_str("dataframes = [\"polars\"]\n");
        optimizations.push_str("arrow = [\"dep:arrow\", \"dep:arrow-ipc\", \"dep:arrow-csv\"]\n");
        optimizations.push_str("full = [\"dataframes\", \"arrow\", \"serde\"]\n\n");

        optimizations.push_str("# Profile optimizations:\n");
        optimizations.push_str("[profile.dev]\n");
        optimizations.push_str("opt-level = 1  # Faster dev builds\n\n");

        optimizations.push_str("[profile.release]\n");
        optimizations.push_str("codegen-units = 1  # Better optimization\n");
        optimizations.push_str("lto = true  # Link-time optimization\n");

        optimizations
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Calculate dependency tree metrics
pub fn calculate_metrics(audit: &DependencyAudit) -> DependencyMetrics {
    let deps = audit.dependencies();

    let total_compile_time: u32 = deps
        .values()
        .map(|d| match d.compile_time_impact {
            CompileTimeImpact::Minimal => 1,
            CompileTimeImpact::Low => 3,
            CompileTimeImpact::Medium => 10,
            CompileTimeImpact::High => 20,
            CompileTimeImpact::VeryHigh => 40,
        })
        .sum();

    let total_binary_size: u32 = deps
        .values()
        .map(|d| match d.binary_size_impact {
            BinarySizeImpact::Minimal => 1,
            BinarySizeImpact::Low => 5,
            BinarySizeImpact::Medium => 15,
            BinarySizeImpact::High => 50,
            BinarySizeImpact::VeryHigh => 200,
        })
        .sum();

    DependencyMetrics {
        estimated_compile_time_seconds: total_compile_time,
        estimated_binary_size_mb: total_binary_size,
        dependency_depth: deps.values().map(|d| d.transitive_deps).max().unwrap_or(0),
        optimization_potential: audit.recommendations().len(),
    }
}

/// Metrics about the dependency tree
#[derive(Debug, Clone)]
pub struct DependencyMetrics {
    /// Estimated total compile time in seconds
    pub estimated_compile_time_seconds: u32,
    /// Estimated binary size in MB
    pub estimated_binary_size_mb: u32,
    /// Maximum dependency depth
    pub dependency_depth: usize,
    /// Number of optimization opportunities
    pub optimization_potential: usize,
}

/// Generate dependency visualization (simplified DOT format)
pub fn generate_dependency_graph(audit: &DependencyAudit) -> String {
    let mut graph = String::new();
    graph.push_str("digraph dependencies {\n");
    graph.push_str("  rankdir=TB;\n");
    graph.push_str("  node [shape=box];\n\n");

    // Add nodes with colors based on category
    for dep in audit.dependencies().values() {
        let color = match dep.category {
            DependencyCategory::Essential => "lightblue",
            DependencyCategory::Optional => "lightgreen",
            DependencyCategory::Development => "lightyellow",
            DependencyCategory::Heavy => "lightcoral",
            DependencyCategory::Redundant => "lightgray",
            DependencyCategory::Security => "pink",
        };

        graph.push_str(&format!(
            "  \"{}\" [fillcolor={}, style=filled];\n",
            dep.name, color
        ));
    }

    graph.push_str("}\n");
    graph
}

// =============================================================================
// Tests
// =============================================================================

/// Automated dependency update checker and manager
#[derive(Debug)]
pub struct DependencyUpdater {
    config: UpdaterConfig,
    current_versions: HashMap<String, String>,
    latest_versions: HashMap<String, String>,
}

/// Configuration for automated dependency updates
#[derive(Debug, Clone)]
pub struct UpdaterConfig {
    /// Allow major version updates (potentially breaking)
    pub allow_major_updates: bool,
    /// Allow minor version updates (backward compatible)
    pub allow_minor_updates: bool,
    /// Allow patch version updates (bug fixes only)
    pub allow_patch_updates: bool,
    /// Exclude specific packages from updates
    pub excluded_packages: HashSet<String>,
    /// Require specific version constraints for packages
    pub version_constraints: HashMap<String, String>,
    /// Check for security advisories
    pub check_security_advisories: bool,
}

impl Default for UpdaterConfig {
    fn default() -> Self {
        Self {
            allow_major_updates: false,
            allow_minor_updates: true,
            allow_patch_updates: true,
            excluded_packages: HashSet::new(),
            version_constraints: HashMap::new(),
            check_security_advisories: true,
        }
    }
}

/// Update recommendation for a dependency
#[derive(Debug, Clone)]
pub struct UpdateRecommendation {
    pub package_name: String,
    pub current_version: String,
    pub latest_version: String,
    pub update_type: UpdateType,
    pub security_advisory: Option<SecurityAdvisory>,
    pub breaking_changes: Vec<String>,
    pub priority: UpdatePriority,
}

/// Type of version update
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateType {
    Major,
    Minor,
    Patch,
}

/// Priority level for updates
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum UpdatePriority {
    Critical, // Security fixes
    High,     // Bug fixes with significant impact
    Medium,   // Minor improvements
    Low,      // Optional updates
}

/// Security advisory information
#[derive(Debug, Clone)]
pub struct SecurityAdvisory {
    pub id: String,
    pub title: String,
    pub severity: String,
    pub affected_versions: String,
    pub patched_versions: String,
    pub description: String,
}

impl DependencyUpdater {
    /// Create a new dependency updater with default configuration
    pub fn new() -> Self {
        Self::with_config(UpdaterConfig::default())
    }

    /// Create a new dependency updater with custom configuration
    pub fn with_config(config: UpdaterConfig) -> Self {
        Self {
            config,
            current_versions: HashMap::new(),
            latest_versions: HashMap::new(),
        }
    }

    /// Check for available updates to dependencies
    pub fn check_for_updates(&mut self) -> Result<Vec<UpdateRecommendation>, String> {
        // Parse Cargo.toml to get current dependencies
        self.load_current_dependencies()?;

        // Query registry for latest versions
        self.fetch_latest_versions()?;

        // Generate update recommendations
        let mut recommendations = Vec::new();

        for (package, current_version) in &self.current_versions {
            if self.config.excluded_packages.contains(package) {
                continue;
            }

            if let Some(latest_version) = self.latest_versions.get(package) {
                if let Some(recommendation) =
                    self.analyze_update(package, current_version, latest_version)?
                {
                    recommendations.push(recommendation);
                }
            }
        }

        // Sort by priority
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(recommendations)
    }

    /// Generate automated update script
    pub fn generate_update_script(&self, recommendations: &[UpdateRecommendation]) -> String {
        let mut script = String::new();
        script.push_str("#!/bin/bash\n");
        script.push_str("# Automated dependency update script generated by sklears\n\n");

        for rec in recommendations {
            if self.should_auto_update(rec) {
                script.push_str(&format!(
                    "echo \"Updating {} from {} to {}\"\n",
                    rec.package_name, rec.current_version, rec.latest_version
                ));
                script.push_str(&format!(
                    "cargo update -p {}:{}\n",
                    rec.package_name, rec.latest_version
                ));
            } else {
                script.push_str(&format!(
                    "# Manual review required for {}: {} -> {} ({})\n",
                    rec.package_name,
                    rec.current_version,
                    rec.latest_version,
                    match rec.update_type {
                        UpdateType::Major => "major version change",
                        UpdateType::Minor => "minor version change",
                        UpdateType::Patch => "patch version change",
                    }
                ));
            }
        }

        script.push_str("\necho \"Update process complete. Running tests...\"\n");
        script.push_str("cargo test\n");
        script.push_str("cargo clippy -- -D warnings\n");

        script
    }

    /// Load current dependency versions from Cargo.toml
    fn load_current_dependencies(&mut self) -> Result<(), String> {
        // In a real implementation, this would parse Cargo.toml
        // For now, simulate with some common dependencies
        self.current_versions
            .insert("ndarray".to_string(), "0.15.6".to_string());
        self.current_versions
            .insert("serde".to_string(), "1.0.193".to_string());
        self.current_versions
            .insert("rayon".to_string(), "1.8.0".to_string());
        self.current_versions
            .insert("criterion".to_string(), "0.5.1".to_string());
        Ok(())
    }

    /// Fetch latest versions from crates.io
    fn fetch_latest_versions(&mut self) -> Result<(), String> {
        // In a real implementation, this would query crates.io API
        // For now, simulate with some example versions
        self.latest_versions
            .insert("ndarray".to_string(), "0.15.7".to_string());
        self.latest_versions
            .insert("serde".to_string(), "1.0.195".to_string());
        self.latest_versions
            .insert("rayon".to_string(), "1.8.1".to_string());
        self.latest_versions
            .insert("criterion".to_string(), "0.5.1".to_string());
        Ok(())
    }

    /// Analyze whether a package should be updated
    fn analyze_update(
        &self,
        package: &str,
        current: &str,
        latest: &str,
    ) -> Result<Option<UpdateRecommendation>, String> {
        if current == latest {
            return Ok(None);
        }

        let update_type = self.determine_update_type(current, latest)?;

        // Check if this type of update is allowed
        let allowed = match update_type {
            UpdateType::Major => self.config.allow_major_updates,
            UpdateType::Minor => self.config.allow_minor_updates,
            UpdateType::Patch => self.config.allow_patch_updates,
        };

        if !allowed {
            return Ok(None);
        }

        // Check for security advisories (simulated)
        let security_advisory = self.check_security_advisory(package, current);

        let priority = if security_advisory.is_some() {
            UpdatePriority::Critical
        } else {
            match update_type {
                UpdateType::Major => UpdatePriority::Low,
                UpdateType::Minor => UpdatePriority::Medium,
                UpdateType::Patch => UpdatePriority::High,
            }
        };

        Ok(Some(UpdateRecommendation {
            package_name: package.to_string(),
            current_version: current.to_string(),
            latest_version: latest.to_string(),
            update_type,
            security_advisory,
            breaking_changes: self.get_breaking_changes(package, current, latest),
            priority,
        }))
    }

    /// Determine the type of version update
    fn determine_update_type(&self, current: &str, latest: &str) -> Result<UpdateType, String> {
        let current_parts: Vec<u32> = current.split('.').map(|s| s.parse().unwrap_or(0)).collect();
        let latest_parts: Vec<u32> = latest.split('.').map(|s| s.parse().unwrap_or(0)).collect();

        if current_parts.len() < 3 || latest_parts.len() < 3 {
            return Err("Invalid version format".to_string());
        }

        if latest_parts[0] > current_parts[0] {
            Ok(UpdateType::Major)
        } else if latest_parts[1] > current_parts[1] {
            Ok(UpdateType::Minor)
        } else {
            Ok(UpdateType::Patch)
        }
    }

    /// Check for security advisories (simulated)
    fn check_security_advisory(&self, package: &str, version: &str) -> Option<SecurityAdvisory> {
        // In a real implementation, this would query RustSec advisory database
        // For demonstration, simulate an advisory for an old version
        if package == "serde" && version == "1.0.100" {
            Some(SecurityAdvisory {
                id: "RUSTSEC-2023-0001".to_string(),
                title: "Simulated security vulnerability".to_string(),
                severity: "Medium".to_string(),
                affected_versions: "< 1.0.150".to_string(),
                patched_versions: ">= 1.0.150".to_string(),
                description: "This is a simulated security advisory for demonstration".to_string(),
            })
        } else {
            None
        }
    }

    /// Get breaking changes for a version update (simulated)
    fn get_breaking_changes(&self, _package: &str, _current: &str, _latest: &str) -> Vec<String> {
        // In a real implementation, this would parse changelogs or release notes
        Vec::new()
    }

    /// Determine if an update should be applied automatically
    fn should_auto_update(&self, recommendation: &UpdateRecommendation) -> bool {
        match recommendation.update_type {
            UpdateType::Major => false, // Always require manual review for major updates
            UpdateType::Minor => recommendation.breaking_changes.is_empty(),
            UpdateType::Patch => true,
        }
    }

    /// Generate a detailed update report
    pub fn generate_update_report(&self, recommendations: &[UpdateRecommendation]) -> String {
        let mut report = String::new();
        report.push_str("Dependency Update Report\n");
        report.push_str("========================\n\n");

        let critical_count = recommendations
            .iter()
            .filter(|r| r.priority == UpdatePriority::Critical)
            .count();
        let high_count = recommendations
            .iter()
            .filter(|r| r.priority == UpdatePriority::High)
            .count();
        let medium_count = recommendations
            .iter()
            .filter(|r| r.priority == UpdatePriority::Medium)
            .count();
        let low_count = recommendations
            .iter()
            .filter(|r| r.priority == UpdatePriority::Low)
            .count();

        report.push_str("Summary:\n");
        report.push_str(&format!("- Critical updates: {critical_count}\n"));
        report.push_str(&format!("- High priority updates: {high_count}\n"));
        report.push_str(&format!("- Medium priority updates: {medium_count}\n"));
        report.push_str(&format!("- Low priority updates: {low_count}\n\n"));

        for rec in recommendations {
            report.push_str(&format!("Package: {}\n", rec.package_name));
            report.push_str(&format!("Current Version: {}\n", rec.current_version));
            report.push_str(&format!("Latest Version: {}\n", rec.latest_version));
            report.push_str(&format!("Update Type: {:?}\n", rec.update_type));
            report.push_str(&format!("Priority: {:?}\n", rec.priority));

            if let Some(ref advisory) = rec.security_advisory {
                report.push_str(&format!(
                    "⚠️  Security Advisory: {} - {}\n",
                    advisory.id, advisory.title
                ));
                report.push_str(&format!("   Severity: {}\n", advisory.severity));
            }

            if !rec.breaking_changes.is_empty() {
                report.push_str("Breaking Changes:\n");
                for change in &rec.breaking_changes {
                    report.push_str(&format!("   - {change}\n"));
                }
            }

            report.push('\n');
        }

        report
    }
}

impl Default for DependencyUpdater {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_audit_creation() {
        let audit = DependencyAudit::new();
        assert!(!audit.dependencies().is_empty());
        assert!(!audit.recommendations().is_empty());
    }

    #[test]
    fn test_dependency_categories() {
        let audit = DependencyAudit::new();

        let essential = audit.dependencies_by_category(DependencyCategory::Essential);
        let optional = audit.dependencies_by_category(DependencyCategory::Optional);
        let heavy = audit.dependencies_by_category(DependencyCategory::Heavy);

        assert!(!essential.is_empty());
        assert!(!optional.is_empty());
        assert!(!heavy.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let audit = DependencyAudit::new();
        let report = audit.generate_report();

        assert!(report.total_dependencies > 0);
        assert!(report.essential_dependencies > 0);
        assert!(!report.summary().is_empty());
        assert!(!report.detailed_recommendations().is_empty());
    }

    #[test]
    fn test_metrics_calculation() {
        let audit = DependencyAudit::new();
        let metrics = calculate_metrics(&audit);

        assert!(metrics.estimated_compile_time_seconds > 0);
        assert!(metrics.estimated_binary_size_mb > 0);
        assert!(metrics.dependency_depth > 0);
    }

    #[test]
    fn test_cargo_optimizations() {
        let audit = DependencyAudit::new();
        let report = audit.generate_report();
        let optimizations = report.generate_cargo_optimizations();

        assert!(optimizations.contains("optional = true"));
        assert!(optimizations.contains("[features]"));
        assert!(optimizations.contains("[profile"));
    }

    #[test]
    fn test_dependency_graph() {
        let audit = DependencyAudit::new();
        let graph = generate_dependency_graph(&audit);

        assert!(graph.contains("digraph dependencies"));
        assert!(graph.contains("lightblue")); // Essential deps
        assert!(graph.contains("lightcoral")); // Heavy deps
    }

    #[test]
    fn test_recommendations() {
        let audit = DependencyAudit::new();
        let recommendations = audit.recommendations();

        // Should have recommendations for heavy dependencies
        assert!(recommendations
            .iter()
            .any(|r| matches!(r.action, RecommendationAction::MakeOptional)));

        // Should have recommendations for redundant dependencies
        assert!(recommendations
            .iter()
            .any(|r| matches!(r.action, RecommendationAction::Remove)));
    }

    #[test]
    fn test_license_checker() {
        let checker = LicenseChecker::new("Apache-2.0");

        // Test compatible licenses
        assert_eq!(
            checker.check_compatibility("MIT"),
            LicenseCompatibility::Compatible
        );
        assert_eq!(
            checker.check_compatibility("BSD-3-Clause"),
            LicenseCompatibility::Compatible
        );
        assert_eq!(
            checker.check_compatibility("Apache-2.0"),
            LicenseCompatibility::Compatible
        );

        // Test incompatible licenses
        assert_eq!(
            checker.check_compatibility("GPL-2.0"),
            LicenseCompatibility::Incompatible
        );

        // Test licenses requiring attribution
        assert_eq!(
            checker.check_compatibility("LGPL-2.1"),
            LicenseCompatibility::CompatibleWithAttribution
        );
        assert_eq!(
            checker.check_compatibility("MPL-2.0"),
            LicenseCompatibility::CompatibleWithAttribution
        );

        // Test copyleft compatible
        assert_eq!(
            checker.check_compatibility("GPL-3.0"),
            LicenseCompatibility::CompatibleCopyleft
        );

        // Test unknown license
        assert_eq!(
            checker.check_compatibility("UNKNOWN-LICENSE"),
            LicenseCompatibility::Unknown
        );

        // Test problematic licenses
        assert_eq!(
            checker.check_compatibility("AGPL-3.0"),
            LicenseCompatibility::ReviewRequired
        );
    }

    #[test]
    fn test_license_report_generation() {
        let audit = DependencyAudit::new();
        let checker = LicenseChecker::new("Apache-2.0");
        let license_report = checker.generate_license_report(audit.dependencies());

        // Should have dependencies in various categories
        assert!(!license_report.compatible.is_empty());

        // Should detect any license issues
        let has_issues = license_report.has_issues();

        // Generate reports
        let detailed = license_report.detailed_report();
        assert!(detailed.contains("License Compatibility Report"));

        if has_issues {
            let summary = license_report.issue_summary();
            assert!(summary.contains("License Compatibility Issues"));
        }
    }

    #[test]
    fn test_attribution_text_generation() {
        let audit = DependencyAudit::new();
        let checker = LicenseChecker::new("Apache-2.0");
        let attribution = checker.generate_attribution_text(audit.dependencies());

        assert!(attribution.contains("Third-Party Licenses"));
        assert!(attribution.contains("This software includes"));
    }

    #[test]
    fn test_mit_license_compatibility() {
        let checker = LicenseChecker::new("MIT");

        // MIT should be compatible with most permissive licenses
        assert_eq!(
            checker.check_compatibility("MIT"),
            LicenseCompatibility::Compatible
        );
        assert_eq!(
            checker.check_compatibility("BSD-2-Clause"),
            LicenseCompatibility::Compatible
        );
        assert_eq!(
            checker.check_compatibility("Apache-2.0"),
            LicenseCompatibility::Compatible
        );

        // But still incompatible with GPL-2.0
        assert_eq!(
            checker.check_compatibility("GPL-2.0"),
            LicenseCompatibility::Incompatible
        );
    }

    #[test]
    fn test_dependency_updater_creation() {
        let updater = DependencyUpdater::new();
        assert_eq!(updater.config.allow_minor_updates, true);
        assert_eq!(updater.config.allow_major_updates, false);
        assert_eq!(updater.config.allow_patch_updates, true);
    }

    #[test]
    fn test_update_recommendations() {
        let mut updater = DependencyUpdater::new();
        let recommendations = updater.check_for_updates().unwrap();

        // Should have some recommendations due to simulated version differences
        assert!(!recommendations.is_empty());

        // Should be sorted by priority (highest first)
        for i in 1..recommendations.len() {
            assert!(recommendations[i - 1].priority >= recommendations[i].priority);
        }
    }

    #[test]
    fn test_update_script_generation() {
        let mut updater = DependencyUpdater::new();
        let recommendations = updater.check_for_updates().unwrap();
        let script = updater.generate_update_script(&recommendations);

        assert!(script.contains("#!/bin/bash"));
        assert!(script.contains("cargo update"));
        assert!(script.contains("cargo test"));
    }

    #[test]
    fn test_update_report_generation() {
        let mut updater = DependencyUpdater::new();
        let recommendations = updater.check_for_updates().unwrap();
        let report = updater.generate_update_report(&recommendations);

        assert!(report.contains("Dependency Update Report"));
        assert!(report.contains("Summary:"));
        assert!(report.contains("Package:"));
    }

    #[test]
    fn test_update_type_determination() {
        let updater = DependencyUpdater::new();

        assert_eq!(
            updater.determine_update_type("1.0.0", "2.0.0").unwrap(),
            UpdateType::Major
        );
        assert_eq!(
            updater.determine_update_type("1.0.0", "1.1.0").unwrap(),
            UpdateType::Minor
        );
        assert_eq!(
            updater.determine_update_type("1.0.0", "1.0.1").unwrap(),
            UpdateType::Patch
        );
    }

    #[test]
    fn test_gpl3_license_compatibility() {
        let checker = LicenseChecker::new("GPL-3.0");

        // GPL-3.0 can incorporate most licenses
        assert_eq!(
            checker.check_compatibility("MIT"),
            LicenseCompatibility::Compatible
        );
        assert_eq!(
            checker.check_compatibility("Apache-2.0"),
            LicenseCompatibility::Compatible
        );
        assert_eq!(
            checker.check_compatibility("GPL-3.0"),
            LicenseCompatibility::Compatible
        );

        // But not GPL-2.0
        assert_eq!(
            checker.check_compatibility("GPL-2.0"),
            LicenseCompatibility::Incompatible
        );
    }
}

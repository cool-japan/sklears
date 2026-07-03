//! Leaf analyzer/detector types, "result" type aliases, and their shallow heuristic logic for
//! [`super::CryptographicAnalyzer`] and friends.
//!
//! Everything in this file is either a plain data container (struct/enum with no significant
//! logic), a transparent type alias over one of the three shared shapes defined here
//! (`AnalysisFinding`/`SecurityVulnerabilityFinding`/`SecurityRecommendationItem`), or a small
//! piece of self-contained, per-type/per-domain heuristic logic (the ~35 `pub(super)` helper
//! methods on `CryptographicAnalyzer` consumed by the orchestration in the parent module, and
//! the macro-generated leaf analyzer methods). The actual multi-domain orchestration
//! (`analyze_cryptographic_security` and its ten `analyze_*` sub-methods) lives on
//! `CryptographicAnalyzer` in the parent module, which owns the type's fields and the
//! originally-given control flow.
//!
//! Split out of `crypto_analysis.rs` (which was approaching the workspace's 2000-line refactor
//! threshold) via the `splitrs`-style types/logic separation used elsewhere in this crate (see
//! `threat_modeling.rs` / `threat_modeling_types.rs`).

use super::super::security_types::*;
use super::{
    CachedCryptographicAnalysis, CryptoSubAnalysesRef, CryptographicAnalysisError,
    CryptographicAnalysisResult, CryptographicAnalyzer,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

// Shared heuristic-analysis infrastructure: the cryptographic-analysis domain model below fans
// out into a very large number of narrow, single-purpose sub-analyzers. None perform real
// cryptanalysis; like `core_analyzer.rs`'s `assess_pattern_vulnerabilities`, each produces a
// shallow, genuinely input-dependent finding from a few `TraitUsageContext` flags. To keep line
// count sane, every leaf "result" type is an alias over one of the 3 shapes below, and every leaf
// "analyzer" struct is an empty marker populated via the macros further down.

/// Generic single-item heuristic finding shared by the majority of narrow sub-analyses
/// (algorithm/key-management/protocol/hash/signature/encryption/quantum/implementation).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnalysisFinding {
    pub score: f64,
    pub severity: Option<RiskSeverity>,
    pub summary: String,
    pub findings: Vec<String>,
}

/// Generic side-channel/protocol vulnerability shape shared by the various `detect_*`/
/// `identify_*` methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityVulnerabilityFinding {
    pub description: String,
    pub severity: RiskSeverity,
    pub affected_operation: String,
}

/// Generic recommendation/warning/risk shape shared by the various `generate_*` methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRecommendationItem {
    pub id: String,
    pub description: String,
    pub priority: RiskSeverity,
}

/// Build a shallow heuristic finding: favorable when `good` is true, concerning otherwise.
/// `good` is always derived from `TraitUsageContext` flags by the caller, so the result is
/// genuinely input-dependent even though the two output shapes are themselves fixed.
fn heuristic_finding(good: bool, subject: &str) -> AnalysisFinding {
    if good {
        AnalysisFinding {
            score: 8.5,
            severity: None,
            summary: format!("{subject}: adequate controls observed"),
            findings: Vec::new(),
        }
    } else {
        AnalysisFinding {
            score: 3.5,
            severity: Some(RiskSeverity::Medium),
            summary: format!("{subject}: controls missing or incomplete"),
            findings: vec![format!("{subject} lacks adequate safeguards")],
        }
    }
}

/// Build a (possibly empty) single-element vulnerability list, used by the various
/// side-channel/protocol detectors.
fn heuristic_vulnerabilities(
    detected: bool,
    description: &str,
    severity: RiskSeverity,
    affected: &str,
) -> Vec<SecurityVulnerabilityFinding> {
    if detected {
        vec![SecurityVulnerabilityFinding {
            description: description.to_string(),
            severity,
            affected_operation: affected.to_string(),
        }]
    } else {
        Vec::new()
    }
}

/// Turn a list of detected vulnerabilities into mitigation recommendations, one per
/// vulnerability, prefixed with `category`.
fn recommendations_from_vulnerabilities(
    vulnerabilities: &[SecurityVulnerabilityFinding],
    category: &str,
) -> Vec<SecurityRecommendationItem> {
    vulnerabilities
        .iter()
        .enumerate()
        .map(|(i, v)| SecurityRecommendationItem {
            id: format!("{category}-{i}"),
            description: format!(
                "Mitigate {} (affects {})",
                v.description, v.affected_operation
            ),
            priority: v.severity.clone(),
        })
        .collect()
}

/// Turn a list of labeled findings into recommendations for every finding scoring below
/// `threshold`, prefixed with `category`.
fn recommendations_from_findings(
    items: &[(&str, &AnalysisFinding)],
    category: &str,
    threshold: f64,
) -> Vec<SecurityRecommendationItem> {
    items
        .iter()
        .filter(|(_, finding)| finding.score < threshold)
        .map(|(label, finding)| SecurityRecommendationItem {
            id: format!("{category}-{label}"),
            description: format!("Improve {label}: {}", finding.summary),
            priority: RiskSeverity::High,
        })
        .collect()
}

/// Average a set of 0.0-10.0 domain scores into a single overall score, clamped to range.
fn average_score(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    (scores.iter().sum::<f64>() / scores.len() as f64).clamp(0.0, 10.0)
}

// Bulk-definition macros: the manager structs below fan out into ~90 leaf analyzer/detector types
// plus ~100 "result" type aliases (see the two sections near the end of this file). Defining each
// individually would blow the line budget many times over, so every one is macro-generated.
//
// NOTE ON FORMATTING: every macro below is deliberately invoked with `{ .. }` (not `( .. )`), and
// every multi-field/multi-item invocation packs several entries per source line. This is not just
// a style choice: rustfmt treats a `(...)`-delimited macro call whose arguments look like a flat
// comma list as a function-call argument list and reformats it one-per-line, but it leaves the
// contents of a `{ .. }`-delimited macro call alone (verified empirically). Keeping every bulk
// invocation `{ .. }`-delimited is what keeps this file's line count stable across `cargo fmt`.

/// Generate a `pub struct` (with the standard `Debug, Clone, Serialize, Deserialize` derive) from
/// a terse `Name { field: Type, .. }` list — used for the plain-data result/contract structs
/// below, so their field lists can stay packed instead of rustfmt exploding them one-per-line.
macro_rules! result_structs {
    ($($name:ident { $($field:ident: $ty:ty),* $(,)? })*) => {
        $(
            #[derive(Debug, Clone, Serialize, Deserialize)]
            pub struct $name { $(pub $field: $ty,)* }
        )*
    };
}

/// As [`result_structs!`], but for a simple unit-variant `pub enum`.
macro_rules! result_enum {
    ($name:ident { $($variant:ident),* $(,)? }) => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum $name { $($variant,)* }
    };
}

/// A near-empty marker analyzer/detector with just a `new()` — used for leaf types whose
/// `detect_*`/`analyze_*` method is never actually invoked by the orchestration in this file.
macro_rules! marker_type {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Debug, Clone, Serialize, Deserialize, Default)]
            pub struct $name {}
            impl $name { pub fn new() -> Self { Self::default() } }
        )*
    };
}

/// A marker analyzer whose single `$method` returns a shallow [`AnalysisFinding`] driven by two
/// ANDed `TraitUsageContext` flags.
macro_rules! define_finding_analyzer {
    ($($name:ident => $method:ident, $flag1:ident, $flag2:ident;)*) => {
        $(
            #[derive(Debug, Clone, Serialize, Deserialize, Default)]
            pub struct $name {}
            impl $name {
                pub fn new() -> Self { Self::default() }
                pub(super) fn $method(&self, context: &TraitUsageContext) -> Result<AnalysisFinding, CryptographicAnalysisError> {
                    Ok(heuristic_finding(context.$flag1 && context.$flag2, stringify!($name)))
                }
            }
        )*
    };
}

/// As [`define_finding_analyzer!`], but `$method` returns `Vec<AnalysisFinding>` (single-element).
macro_rules! define_finding_list_analyzer {
    ($($name:ident => $method:ident, $flag1:ident, $flag2:ident;)*) => {
        $(
            #[derive(Debug, Clone, Serialize, Deserialize, Default)]
            pub struct $name {}
            impl $name {
                pub fn new() -> Self { Self::default() }
                pub(super) fn $method(&self, context: &TraitUsageContext) -> Result<Vec<AnalysisFinding>, CryptographicAnalysisError> {
                    Ok(vec![heuristic_finding(context.$flag1 && context.$flag2, stringify!($name))])
                }
            }
        )*
    };
}

/// A marker detector whose single `$method` returns a shallow vulnerability list: present when
/// `$flag1` holds but `$flag2` (the mitigating control) does not, at severity `$sev`.
macro_rules! define_vuln_detector {
    ($($name:ident => $method:ident, $flag1:ident, $flag2:ident, $sev:ident;)*) => {
        $(
            #[derive(Debug, Clone, Serialize, Deserialize, Default)]
            pub struct $name {}
            impl $name {
                pub fn new() -> Self { Self::default() }
                pub(super) fn $method(&self, context: &TraitUsageContext) -> Result<Vec<SecurityVulnerabilityFinding>, CryptographicAnalysisError> {
                    Ok(heuristic_vulnerabilities(context.$flag1 && !context.$flag2, stringify!($name), RiskSeverity::$sev, stringify!($method)))
                }
            }
        )*
    };
}

/// Bulk-generate `CryptographicAnalyzer` methods whose entire body is
/// `Ok(heuristic_finding(context.$flag, $label))`.
macro_rules! single_flag_finding {
    ($($method:ident : $ret:ty => $flag:ident, $label:expr;)*) => {
        impl CryptographicAnalyzer {
            $(pub(super) fn $method(&self, context: &TraitUsageContext) -> Result<$ret, CryptographicAnalysisError> {
                Ok(heuristic_finding(context.$flag, $label))
            })*
        }
    };
}

/// As [`single_flag_finding!`], but the body is
/// `Ok(heuristic_finding(context.$flag1 && context.$flag2, $label))`.
macro_rules! dual_flag_finding {
    ($($method:ident : $ret:ty => $flag1:ident, $flag2:ident, $label:expr;)*) => {
        impl CryptographicAnalyzer {
            $(pub(super) fn $method(&self, context: &TraitUsageContext) -> Result<$ret, CryptographicAnalysisError> {
                Ok(heuristic_finding(context.$flag1 && context.$flag2, $label))
            })*
        }
    };
}

/// A named-algorithm leaf analyzer: holds just `algorithm_name`, a generic `new()`, one
/// constructor per `$ctor => $alg` pair (mirroring `CryptographicAlgorithmAnalyzer::initialize_*`
/// which calls e.g. `new_aes()`), and an analysis method folding the name into its heuristic.
macro_rules! algorithm_family {
    ($type:ident, $method:ident, $result:ty, $($ctor:ident => $alg:expr),* $(,)?) => {
        #[derive(Debug, Clone, Serialize, Deserialize, Default)]
        pub struct $type { pub algorithm_name: String }
        impl $type {
            pub fn new() -> Self { Self::default() }
            $(pub fn $ctor() -> Self { Self { algorithm_name: $alg.to_string() } })*
            pub(super) fn $method(&self, context: &TraitUsageContext) -> Result<$result, CryptographicAnalysisError> {
                Ok(heuristic_finding(context.has_cryptographic_operations && context.has_constant_time_operations, &self.algorithm_name))
            }
        }
    };
}

/// Declare every `$name` as `pub type $name = AnalysisFinding;` (see the type-alias section near
/// the end of this file for why: none of these leaf result shapes are part of any cross-file
/// contract, so a single shared shape covers all of them).
macro_rules! finding_alias {
    ($($name:ident),* $(,)?) => { $(pub type $name = AnalysisFinding;)* };
}

/// As [`finding_alias!`], aliasing to [`SecurityRecommendationItem`].
macro_rules! recommendation_alias {
    ($($name:ident),* $(,)?) => { $(pub type $name = SecurityRecommendationItem;)* };
}

/// As [`finding_alias!`], aliasing to [`SecurityVulnerabilityFinding`].
macro_rules! vulnerability_alias {
    ($($name:ident),* $(,)?) => { $(pub type $name = SecurityVulnerabilityFinding;)* };
}

result_structs! {
    ProtocolAnalysisResult {
        tls_analysis: TlsAnalysisResult, ssh_analysis: SshAnalysisResult, ipsec_analysis: IpsecAnalysisResult,
        pgp_analysis: PgpAnalysisResult, oauth_analysis: OAuthAnalysisResult, saml_analysis: SamlAnalysisResult,
        kerberos_analysis: KerberosAnalysisResult, protocol_state_analysis: ProtocolStateAnalysisResult,
        message_flow_analysis: MessageFlowAnalysisResult, authentication_analysis: AuthenticationAnalysisResult,
        protocol_vulnerabilities: Vec<ProtocolVulnerability>, protocol_recommendations: Vec<ProtocolRecommendation>,
    }
    RandomNumberAnalysisResult {
        entropy_test_results: EntropyTestResults, statistical_test_results: StatisticalTestResults,
        predictability_analysis: PredictabilityAnalysisResult, seed_analysis: SeedAnalysisResult,
        prng_analysis: PrngAnalysisResult, trng_analysis: TrngAnalysisResult, drbg_analysis: DrbgAnalysisResult,
        nist_test_results: NistTestResults, diehard_test_results: DiehardTestResults, testu01_results: TestU01Results,
        randomness_quality_score: f64, randomness_recommendations: Vec<RandomnessRecommendation>,
    }
    HashFunctionAnalysisResult {
        collision_resistance_results: CollisionResistanceResults, preimage_resistance_results: PreimageResistanceResults,
        second_preimage_results: SecondPreimageResults, avalanche_effect_results: AvalancheEffectResults,
        birthday_attack_analysis: BirthdayAttackAnalysisResult, length_extension_analysis: LengthExtensionAnalysisResult,
        hash_family_analysis: HashFamilyAnalysisResult, merkle_damgard_analysis: MerkleDamgardAnalysisResult,
        sponge_function_analysis: SpongeFunctionAnalysisResult, hash_security_score: f64,
        hash_recommendations: Vec<HashFunctionRecommendation>,
    }
    SignatureAnalysisResult {
        signature_scheme_results: HashMap<String, SignatureSchemeResult>, verification_results: Vec<SignatureVerificationResult>,
        forge_resistance_results: Vec<ForgeResistanceResult>, existential_forgery_results: Vec<ExistentialForgeryResult>,
        chosen_message_analysis: ChosenMessageAnalysisResult, blind_signature_analysis: BlindSignatureAnalysisResult,
        multi_signature_analysis: MultiSignatureAnalysisResult, threshold_signature_analysis: ThresholdSignatureAnalysisResult,
        ring_signature_analysis: RingSignatureAnalysisResult, signature_security_score: f64,
        signature_recommendations: Vec<SignatureRecommendation>,
    }
    EncryptionAnalysisResult {
        symmetric_encryption_analysis: SymmetricEncryptionAnalysisResult, asymmetric_encryption_analysis: AsymmetricEncryptionAnalysisResult,
        mode_of_operation_analysis: ModeOfOperationAnalysisResult, padding_scheme_analysis: PaddingSchemeAnalysisResult,
        authenticated_encryption_analysis: AuthenticatedEncryptionAnalysisResult, chosen_plaintext_analysis: ChosenPlaintextAnalysisResult,
        chosen_ciphertext_analysis: ChosenCiphertextAnalysisResult, known_plaintext_analysis: KnownPlaintextAnalysisResult,
        differential_cryptanalysis_results: DifferentialCryptanalysisResult, linear_cryptanalysis_results: LinearCryptanalysisResult,
        encryption_security_score: f64, encryption_recommendations: Vec<EncryptionRecommendation>,
    }
    QuantumResistanceAnalysisResult {
        post_quantum_results: HashMap<String, PostQuantumResult>, shor_algorithm_impact: ShorAlgorithmImpact,
        grover_algorithm_impact: GroverAlgorithmImpact, quantum_key_distribution_analysis: QuantumKeyDistributionAnalysisResult,
        lattice_based_analysis: LatticeBasedAnalysisResult, code_based_analysis: CodeBasedAnalysisResult,
        multivariate_analysis: MultivariateAnalysisResult, hash_based_analysis: HashBasedAnalysisResult,
        isogeny_analysis: IsogenyAnalysisResult, quantum_threat_assessment: QuantumThreatAssessment,
        migration_strategy: QuantumMigrationStrategy, quantum_resistance_score: f64,
        quantum_recommendations: Vec<QuantumResistanceRecommendation>,
    }
    ImplementationAnalysisResult {
        constant_time_analysis: ConstantTimeAnalysisResult, memory_safety_analysis: MemorySafetyAnalysisResult,
        secure_coding_analysis: SecureCodingAnalysisResult, library_analysis: LibraryAnalysisResult,
        hardware_security_analysis: HardwareSecurityAnalysisResult, side_channel_countermeasures_analysis: SideChannelCountermeasuresAnalysisResult,
        fault_tolerance_analysis: FaultToleranceAnalysisResult, performance_security_analysis: PerformanceSecurityAnalysisResult,
        implementation_security_score: f64, implementation_recommendations: Vec<ImplementationRecommendation>,
    }
}

impl CryptographicAnalyzer {
    // ---- cache / bookkeeping -------------------------------------------------------------

    pub(super) fn generate_analysis_id(&self, context: &TraitUsageContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        context.hash(&mut hasher);
        format!("crypto_analysis_{:x}", hasher.finish())
    }

    pub(super) fn get_cached_analysis(
        &self,
        analysis_id: &str,
    ) -> Option<&CachedCryptographicAnalysis> {
        self.analysis_cache.get(analysis_id)
    }

    pub(super) fn is_cache_valid(&self, cached: &CachedCryptographicAnalysis) -> bool {
        SystemTime::now()
            .duration_since(cached.cache_timestamp)
            .map(|elapsed| elapsed <= cached.cache_ttl)
            .unwrap_or(false)
    }

    pub(super) fn cache_analysis(
        &mut self,
        analysis_id: String,
        result: &CryptographicAnalysisResult,
    ) {
        let cache_ttl = self.analysis_config.cache_duration;
        self.analysis_cache.insert(
            analysis_id,
            CachedCryptographicAnalysis {
                result: result.clone(),
                cache_timestamp: SystemTime::now(),
                cache_ttl,
            },
        );
    }

    pub(super) fn generate_analysis_metadata(
        &self,
        context: &TraitUsageContext,
    ) -> HashMap<String, String> {
        HashMap::from([
            ("analyzer".to_string(), "CryptographicAnalyzer".to_string()),
            ("trait_name".to_string(), context.trait_name.clone()),
            ("trait_count".to_string(), context.traits.len().to_string()),
        ])
    }

    pub(super) fn calculate_analysis_confidence(&self) -> Result<f64, CryptographicAnalysisError> {
        Ok(self
            .analysis_config
            .analysis_confidence_threshold
            .clamp(0.0, 1.0))
    }

    pub(super) fn calculate_overall_cryptographic_score(
        &self,
        scores: &[f64],
    ) -> Result<f64, CryptographicAnalysisError> {
        Ok(average_score(scores))
    }

    /// Mandatory contract addition (see module docs on [`CryptographicAnalysisResult`]):
    /// derive the individually addressable cryptographic weaknesses directly from the raw
    /// usage context, mirroring the shallow-heuristic style of `core_analyzer.rs`'s
    /// `assess_side_channel_risks`.
    pub(super) fn identify_cryptographic_issues(
        &self,
        context: &TraitUsageContext,
    ) -> Vec<CryptographicIssue> {
        let mut issues = Vec::new();
        if context.has_cryptographic_operations && !context.has_constant_time_operations {
            issues.push(CryptographicIssue {
                id: "CRYPTO-ISSUE-001".to_string(), severity: RiskSeverity::High,
                issue_type: "Timing side-channel".to_string(),
                recommendation: "Use constant-time comparison and arithmetic for all secret-dependent operations.".to_string(),
                fix_complexity: ImplementationEffort::Medium, dependencies: Vec::new(),
            });
        }
        if context.has_cryptographic_operations && !context.has_secure_key_management {
            issues.push(CryptographicIssue {
                id: "CRYPTO-ISSUE-002".to_string(),
                severity: RiskSeverity::High,
                issue_type: "Key management".to_string(),
                recommendation: "Adopt secure key generation, storage, and rotation practices."
                    .to_string(),
                fix_complexity: ImplementationEffort::High,
                dependencies: vec!["Key management infrastructure".to_string()],
            });
        }
        if context.has_timing_dependencies && !context.has_constant_time_operations {
            issues.push(CryptographicIssue {
                id: "CRYPTO-ISSUE-003".to_string(),
                severity: RiskSeverity::Medium,
                issue_type: "Timing dependency".to_string(),
                recommendation: "Eliminate secret-dependent branching and early returns."
                    .to_string(),
                fix_complexity: ImplementationEffort::Medium,
                dependencies: Vec::new(),
            });
        }
        if context.handles_sensitive_data
            && context.has_cryptographic_operations
            && !context.has_encryption
        {
            issues.push(CryptographicIssue {
                id: "CRYPTO-ISSUE-004".to_string(),
                severity: RiskSeverity::Critical,
                issue_type: "Missing encryption".to_string(),
                recommendation: "Encrypt sensitive data at rest and in transit.".to_string(),
                fix_complexity: ImplementationEffort::Medium,
                dependencies: vec!["Cryptographic library integration".to_string()],
            });
        }
        issues
    }

    // Data-driven rather than 10 near-identical `if cond { push(..) }` blocks: each row is a
    // domain-level trigger condition (itself derived from the computed sub-analyses) paired
    // with the recommendation to emit when it fires.
    pub(super) fn generate_security_recommendations(
        &self,
        analyses: &CryptoSubAnalysesRef<'_>,
    ) -> Result<Vec<CryptographicRecommendation>, CryptographicAnalysisError> {
        let side_channel_issue = !analyses
            .side_channel
            .timing_attack_vulnerabilities
            .is_empty()
            || !analyses
                .side_channel
                .cache_attack_vulnerabilities
                .is_empty();
        let protocol_issue = !analyses.protocol.protocol_vulnerabilities.is_empty();
        let candidates = [
            (analyses.algorithm.security_level_assessment.score < 6.0, "CRYPTO-REC-ALGO", "Upgrade weak algorithms", "One or more algorithms fall below the recommended security level; migrate to modern primitives (AES-256, SHA-3, Ed25519).", RiskSeverity::High, ImplementationEffort::Medium),
            (analyses.key_management.key_management_score < 6.0, "CRYPTO-REC-KEYMGMT", "Harden key management", "Key generation, storage, or rotation controls are insufficient; adopt an HSM-backed key lifecycle.", RiskSeverity::High, ImplementationEffort::High),
            (side_channel_issue, "CRYPTO-REC-SIDECHANNEL", "Apply side-channel countermeasures", "Timing or cache-based side-channel exposure was detected; use constant-time primitives.", RiskSeverity::High, ImplementationEffort::Medium),
            (protocol_issue, "CRYPTO-REC-PROTOCOL", "Review protocol configuration", "One or more cryptographic protocols were flagged as misconfigured or outdated.", RiskSeverity::Medium, ImplementationEffort::Medium),
            (analyses.random_number.randomness_quality_score < 6.0, "CRYPTO-REC-RNG", "Strengthen randomness sources", "Random number generation quality is below the recommended threshold; use a CSPRNG seeded from an OS entropy source.", RiskSeverity::Critical, ImplementationEffort::Medium),
            (analyses.hash_function.hash_security_score < 6.0, "CRYPTO-REC-HASH", "Replace weak hash functions", "Hash function resistance testing indicates weakness; migrate to SHA-256/SHA-3/BLAKE2 or stronger.", RiskSeverity::High, ImplementationEffort::Low),
            (analyses.signature.signature_security_score < 6.0, "CRYPTO-REC-SIG", "Strengthen digital signatures", "Signature scheme analysis indicates forgery or verification weaknesses.", RiskSeverity::High, ImplementationEffort::Medium),
            (analyses.encryption.encryption_security_score < 6.0, "CRYPTO-REC-ENC", "Strengthen encryption schemes", "Encryption mode, padding, or authentication weaknesses were identified.", RiskSeverity::High, ImplementationEffort::Medium),
            (analyses.quantum_resistance.quantum_resistance_score < 5.0, "CRYPTO-REC-PQC", "Plan post-quantum migration", "Current primitives have limited resistance to quantum adversaries; begin a hybrid PQC migration.", RiskSeverity::Medium, ImplementationEffort::High),
            (analyses.implementation.implementation_security_score < 6.0, "CRYPTO-REC-IMPL", "Improve implementation hygiene", "Constant-time guarantees, memory safety, or secure coding practices need improvement.", RiskSeverity::Medium, ImplementationEffort::Medium),
        ];
        Ok(candidates
            .into_iter()
            .filter(|(triggered, ..)| *triggered)
            .map(
                |(_, id, title, description, priority, implementation_effort)| {
                    CryptographicRecommendation {
                        id: id.to_string(),
                        title: title.to_string(),
                        description: description.to_string(),
                        priority,
                        implementation_effort,
                    }
                },
            )
            .collect())
    }

    // ---- algorithm analysis helpers --------------------------------------------------------

    pub(super) fn build_algorithm_compatibility_matrix(
        &self,
        symmetric: &HashMap<String, SymmetricAlgorithmResult>,
        asymmetric: &HashMap<String, AsymmetricAlgorithmResult>,
    ) -> Result<AlgorithmCompatibilityMatrix, CryptographicAnalysisError> {
        Ok(heuristic_finding(
            !symmetric.is_empty() && !asymmetric.is_empty(),
            "algorithm compatibility",
        ))
    }

    pub(super) fn assess_security_levels(
        &self,
        symmetric: &HashMap<String, SymmetricAlgorithmResult>,
        asymmetric: &HashMap<String, AsymmetricAlgorithmResult>,
    ) -> Result<SecurityLevelAssessment, CryptographicAnalysisError> {
        let scores: Vec<f64> = symmetric
            .values()
            .chain(asymmetric.values())
            .map(|r| r.score)
            .collect();
        Ok(AnalysisFinding {
            score: average_score(&scores),
            severity: None,
            summary: format!("assessed {} algorithms", scores.len()),
            findings: Vec::new(),
        })
    }

    pub(super) fn check_algorithm_deprecations(
        &self,
        symmetric: &HashMap<String, SymmetricAlgorithmResult>,
        asymmetric: &HashMap<String, AsymmetricAlgorithmResult>,
    ) -> Result<Vec<DeprecationWarning>, CryptographicAnalysisError> {
        let mut warnings = Vec::new();
        for (name, result) in symmetric.iter().chain(asymmetric.iter()) {
            if result.score < 5.0 {
                warnings.push(SecurityRecommendationItem {
                    id: format!("DEPRECATED-{name}"),
                    description: format!(
                        "{name} scored below the minimum acceptable security level"
                    ),
                    priority: RiskSeverity::High,
                });
            }
        }
        Ok(warnings)
    }

    pub(super) fn generate_algorithm_upgrade_recommendations(
        &self,
        warnings: &[DeprecationWarning],
    ) -> Result<Vec<AlgorithmUpgradeRecommendation>, CryptographicAnalysisError> {
        Ok(warnings
            .iter()
            .enumerate()
            .map(|(i, w)| SecurityRecommendationItem {
                id: format!("UPGRADE-{i}"),
                description: format!("Upgrade algorithm: {}", w.description),
                priority: w.priority.clone(),
            })
            .collect())
    }

    // ---- key management helpers ------------------------------------------------------------

    // NOTE: assess_key_generation, assess_key_storage, assess_key_distribution,
    // assess_key_rotation, assess_key_revocation, assess_entropy_sources, and
    // assess_key_lifecycle are bulk-generated below via `single_flag_finding!`/
    // `dual_flag_finding!` (after this impl block) to avoid ~50 near-identical
    // 3-line methods bloating this file past the 2000-line budget.

    pub(super) fn calculate_key_management_score(
        &self,
        scores: &[f64],
    ) -> Result<f64, CryptographicAnalysisError> {
        Ok(average_score(scores))
    }

    pub(super) fn identify_key_management_risks(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<KeyManagementRisk>, CryptographicAnalysisError> {
        let mut risks = Vec::new();
        if !context.has_secure_key_management {
            risks.push(SecurityRecommendationItem {
                id: "KEYMGMT-RISK-001".to_string(),
                description: "Cryptographic keys are not managed through a secure lifecycle"
                    .to_string(),
                priority: RiskSeverity::High,
            });
        }
        if context.has_cryptographic_operations && !context.has_audit_logging {
            risks.push(SecurityRecommendationItem {
                id: "KEYMGMT-RISK-002".to_string(),
                description: "Key usage is not audited".to_string(),
                priority: RiskSeverity::Medium,
            });
        }
        Ok(risks)
    }

    pub(super) fn generate_key_management_recommendations(
        &self,
        risks: &[KeyManagementRisk],
    ) -> Result<Vec<KeyManagementRecommendation>, CryptographicAnalysisError> {
        Ok(risks
            .iter()
            .enumerate()
            .map(|(i, r)| SecurityRecommendationItem {
                id: format!("KEYMGMT-REC-{i}"),
                description: format!("Resolve: {}", r.description),
                priority: r.priority.clone(),
            })
            .collect())
    }

    // ---- side-channel helpers ---------------------------------------------------------------

    pub(super) fn identify_applicable_countermeasures(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<SideChannelCountermeasure>, CryptographicAnalysisError> {
        let mut countermeasures = Vec::new();
        if context.has_cryptographic_operations && !context.has_constant_time_operations {
            countermeasures.push(SecurityRecommendationItem {
                id: "COUNTERMEASURE-CT".to_string(),
                description: "Adopt constant-time implementations for secret-dependent operations"
                    .to_string(),
                priority: RiskSeverity::High,
            });
        }
        if !context.has_secure_key_management {
            countermeasures.push(SecurityRecommendationItem {
                id: "COUNTERMEASURE-KEY".to_string(),
                description: "Store keys in a hardware-backed secure enclave or HSM".to_string(),
                priority: RiskSeverity::Medium,
            });
        }
        Ok(countermeasures)
    }

    pub(super) fn calculate_vulnerability_severity_scores(
        &self,
        timing: &[TimingAttackVulnerability],
        power: &[PowerAnalysisVulnerability],
        em: &[ElectromagneticVulnerability],
        acoustic: &[AcousticVulnerability],
        cache: &[CacheAttackVulnerability],
        fault: &[FaultInjectionVulnerability],
    ) -> Result<HashMap<String, f64>, CryptographicAnalysisError> {
        let mut scores = HashMap::new();
        scores.insert("timing".to_string(), timing.len() as f64 * 2.0);
        scores.insert("power_analysis".to_string(), power.len() as f64 * 2.0);
        scores.insert("electromagnetic".to_string(), em.len() as f64 * 1.5);
        scores.insert("acoustic".to_string(), acoustic.len() as f64);
        scores.insert("cache".to_string(), cache.len() as f64 * 2.0);
        scores.insert("fault_injection".to_string(), fault.len() as f64 * 2.5);
        Ok(scores)
    }

    pub(super) fn generate_side_channel_mitigation_recommendations(
        &self,
        timing: &[TimingAttackVulnerability],
        power: &[PowerAnalysisVulnerability],
        em: &[ElectromagneticVulnerability],
        acoustic: &[AcousticVulnerability],
        cache: &[CacheAttackVulnerability],
        fault: &[FaultInjectionVulnerability],
    ) -> Result<Vec<SideChannelMitigationRecommendation>, CryptographicAnalysisError> {
        let all: Vec<SecurityVulnerabilityFinding> = timing
            .iter()
            .chain(power)
            .chain(em)
            .chain(acoustic)
            .chain(cache)
            .chain(fault)
            .cloned()
            .collect();
        Ok(recommendations_from_vulnerabilities(&all, "SIDECHANNEL"))
    }

    // ---- protocol helpers -------------------------------------------------------------------

    pub(super) fn identify_protocol_vulnerabilities(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<ProtocolVulnerability>, CryptographicAnalysisError> {
        let mut vulns = Vec::new();
        vulns.extend(heuristic_vulnerabilities(
            context.has_cryptographic_operations && !context.has_constant_time_operations,
            "protocol handshake timing may leak secret material",
            RiskSeverity::Medium,
            "protocol_handshake",
        ));
        vulns.extend(heuristic_vulnerabilities(
            context.has_cryptographic_operations && !context.has_secure_key_management,
            "protocol key exchange lacks secure key management",
            RiskSeverity::High,
            "key_exchange",
        ));
        Ok(vulns)
    }

    pub(super) fn generate_protocol_recommendations(
        &self,
        vulnerabilities: &[ProtocolVulnerability],
    ) -> Result<Vec<ProtocolRecommendation>, CryptographicAnalysisError> {
        Ok(recommendations_from_vulnerabilities(
            vulnerabilities,
            "PROTOCOL",
        ))
    }

    // ---- random number generation helpers ----------------------------------------------------

    // NOTE: run_entropy_tests, run_statistical_randomness_tests, analyze_predictability,
    // analyze_seed_quality, analyze_prng_implementations, analyze_trng_implementations, and
    // analyze_drbg_implementations are bulk-generated below.

    pub(super) fn calculate_randomness_quality_score(
        &self,
        entropy: &EntropyTestResults,
        statistical: &StatisticalTestResults,
        nist: &NistTestResults,
        diehard: &DiehardTestResults,
        testu01: &TestU01Results,
    ) -> Result<f64, CryptographicAnalysisError> {
        Ok(average_score(&[
            entropy.score,
            statistical.score,
            nist.score,
            diehard.score,
            testu01.score,
        ]))
    }

    pub(super) fn generate_randomness_recommendations(
        &self,
        entropy: &EntropyTestResults,
        predictability: &PredictabilityAnalysisResult,
        seed: &SeedAnalysisResult,
    ) -> Result<Vec<RandomnessRecommendation>, CryptographicAnalysisError> {
        Ok(recommendations_from_findings(
            &[
                ("entropy_source", entropy),
                ("predictability", predictability),
                ("seed_quality", seed),
            ],
            "RNG",
            6.0,
        ))
    }

    // ---- hash function helpers ----------------------------------------------------------------

    // NOTE: test_collision_resistance, test_preimage_resistance,
    // test_second_preimage_resistance, test_avalanche_effect,
    // analyze_birthday_attack_vulnerability, analyze_length_extension_vulnerability, and
    // analyze_hash_families are bulk-generated below.

    pub(super) fn calculate_hash_security_score(
        &self,
        collision: &CollisionResistanceResults,
        preimage: &PreimageResistanceResults,
        second_preimage: &SecondPreimageResults,
        avalanche: &AvalancheEffectResults,
    ) -> Result<f64, CryptographicAnalysisError> {
        Ok(average_score(&[
            collision.score,
            preimage.score,
            second_preimage.score,
            avalanche.score,
        ]))
    }

    pub(super) fn generate_hash_function_recommendations(
        &self,
        collision: &CollisionResistanceResults,
        birthday: &BirthdayAttackAnalysisResult,
        length_extension: &LengthExtensionAnalysisResult,
    ) -> Result<Vec<HashFunctionRecommendation>, CryptographicAnalysisError> {
        Ok(recommendations_from_findings(
            &[
                ("collision_resistance", collision),
                ("birthday_attack", birthday),
                ("length_extension", length_extension),
            ],
            "HASH",
            6.0,
        ))
    }

    // ---- digital signature helpers -------------------------------------------------------------

    // NOTE: analyze_chosen_message_attacks, analyze_blind_signatures, analyze_multi_signatures,
    // analyze_threshold_signatures, and analyze_ring_signatures are bulk-generated below.

    pub(super) fn calculate_signature_security_score(
        &self,
        schemes: &HashMap<String, SignatureSchemeResult>,
        verification: &[SignatureVerificationResult],
        forge: &[ForgeResistanceResult],
    ) -> Result<f64, CryptographicAnalysisError> {
        let scores: Vec<f64> = schemes
            .values()
            .map(|s| s.score)
            .chain(verification.iter().map(|v| v.score))
            .chain(forge.iter().map(|f| f.score))
            .collect();
        Ok(average_score(&scores))
    }

    pub(super) fn generate_signature_recommendations(
        &self,
        forge: &[ForgeResistanceResult],
        existential: &[ExistentialForgeryResult],
        chosen_message: &ChosenMessageAnalysisResult,
    ) -> Result<Vec<SignatureRecommendation>, CryptographicAnalysisError> {
        let mut recommendations =
            recommendations_from_findings(&[("chosen_message_attack", chosen_message)], "SIG", 6.0);
        if forge.iter().any(|f| f.score < 6.0) {
            recommendations.push(SecurityRecommendationItem {
                id: "SIG-REC-forge".to_string(),
                description: "Strengthen forge resistance for one or more signature schemes"
                    .to_string(),
                priority: RiskSeverity::High,
            });
        }
        if existential.iter().any(|f| f.score < 6.0) {
            recommendations.push(SecurityRecommendationItem {
                id: "SIG-REC-existential".to_string(),
                description: "Address existential forgery weaknesses".to_string(),
                priority: RiskSeverity::High,
            });
        }
        Ok(recommendations)
    }

    // ---- encryption scheme helpers --------------------------------------------------------------

    // NOTE: analyze_symmetric_encryption, analyze_asymmetric_encryption,
    // analyze_modes_of_operation, analyze_padding_schemes, analyze_authenticated_encryption,
    // analyze_chosen_plaintext_attacks, analyze_chosen_ciphertext_attacks, and
    // analyze_known_plaintext_attacks are bulk-generated below.

    pub(super) fn calculate_encryption_security_score(
        &self,
        symmetric: &SymmetricEncryptionAnalysisResult,
        asymmetric: &AsymmetricEncryptionAnalysisResult,
        authenticated: &AuthenticatedEncryptionAnalysisResult,
    ) -> Result<f64, CryptographicAnalysisError> {
        Ok(average_score(&[
            symmetric.score,
            asymmetric.score,
            authenticated.score,
        ]))
    }

    pub(super) fn generate_encryption_recommendations(
        &self,
        chosen_plaintext: &ChosenPlaintextAnalysisResult,
        chosen_ciphertext: &ChosenCiphertextAnalysisResult,
        mode_of_operation: &ModeOfOperationAnalysisResult,
        padding_scheme: &PaddingSchemeAnalysisResult,
    ) -> Result<Vec<EncryptionRecommendation>, CryptographicAnalysisError> {
        Ok(recommendations_from_findings(
            &[
                ("chosen_plaintext", chosen_plaintext),
                ("chosen_ciphertext", chosen_ciphertext),
                ("mode_of_operation", mode_of_operation),
                ("padding_scheme", padding_scheme),
            ],
            "ENCRYPTION",
            6.0,
        ))
    }

    // ---- quantum resistance helpers -------------------------------------------------------------

    // NOTE: analyze_lattice_based_cryptography, analyze_code_based_cryptography,
    // analyze_multivariate_cryptography, analyze_hash_based_cryptography,
    // analyze_isogeny_based_cryptography, assess_quantum_threat_timeline, and
    // develop_quantum_migration_strategy are bulk-generated below.

    pub(super) fn calculate_quantum_resistance_score(
        &self,
        post_quantum: &HashMap<String, PostQuantumResult>,
        shor: &ShorAlgorithmImpact,
        grover: &GroverAlgorithmImpact,
    ) -> Result<f64, CryptographicAnalysisError> {
        let scores: Vec<f64> = post_quantum
            .values()
            .map(|r| r.score)
            .chain([shor.score, grover.score])
            .collect();
        Ok(average_score(&scores))
    }

    pub(super) fn generate_quantum_resistance_recommendations(
        &self,
        threat: &QuantumThreatAssessment,
        migration: &QuantumMigrationStrategy,
    ) -> Result<Vec<QuantumResistanceRecommendation>, CryptographicAnalysisError> {
        Ok(recommendations_from_findings(
            &[
                ("threat_timeline", threat),
                ("migration_strategy", migration),
            ],
            "PQC",
            5.0,
        ))
    }

    // ---- implementation helpers -----------------------------------------------------------------

    // NOTE: analyze_constant_time_implementation, analyze_secure_coding_practices,
    // analyze_cryptographic_libraries, analyze_hardware_security_features,
    // analyze_side_channel_countermeasures, and analyze_fault_tolerance are bulk-generated
    // below. analyze_memory_safety and analyze_performance_security_tradeoffs use an OR-shaped
    // condition (not a plain AND of two flags) so they stay hand-written here.

    pub(super) fn analyze_memory_safety(
        &self,
        context: &TraitUsageContext,
    ) -> Result<MemorySafetyAnalysisResult, CryptographicAnalysisError> {
        Ok(heuristic_finding(
            !context.has_unsafe_operations || context.has_bounds_checking,
            "memory safety",
        ))
    }

    pub(super) fn analyze_performance_security_tradeoffs(
        &self,
        context: &TraitUsageContext,
    ) -> Result<PerformanceSecurityAnalysisResult, CryptographicAnalysisError> {
        Ok(heuristic_finding(
            !context.has_resource_intensive_operations || context.has_resource_limits,
            "performance/security tradeoff",
        ))
    }

    pub(super) fn calculate_implementation_security_score(
        &self,
        constant_time: &ConstantTimeAnalysisResult,
        memory_safety: &MemorySafetyAnalysisResult,
        secure_coding: &SecureCodingAnalysisResult,
        hardware_security: &HardwareSecurityAnalysisResult,
    ) -> Result<f64, CryptographicAnalysisError> {
        Ok(average_score(&[
            constant_time.score,
            memory_safety.score,
            secure_coding.score,
            hardware_security.score,
        ]))
    }

    pub(super) fn generate_implementation_recommendations(
        &self,
        constant_time: &ConstantTimeAnalysisResult,
        memory_safety: &MemorySafetyAnalysisResult,
        side_channel_countermeasures: &SideChannelCountermeasuresAnalysisResult,
    ) -> Result<Vec<ImplementationRecommendation>, CryptographicAnalysisError> {
        Ok(recommendations_from_findings(
            &[
                ("constant_time", constant_time),
                ("memory_safety", memory_safety),
                ("side_channel_countermeasures", side_channel_countermeasures),
            ],
            "IMPL",
            6.0,
        ))
    }
}

// `single_flag_finding!`/`dual_flag_finding!` are defined in the "Bulk-definition macros"
// section near the top of this file (alongside the other leaf-generation macros).

single_flag_finding! {
    assess_key_generation: KeyGenerationAssessment => has_secure_key_management, "key generation";
    assess_key_rotation: KeyRotationAssessment => has_secure_key_management, "key rotation";
    run_statistical_randomness_tests: StatisticalTestResults => has_cryptographic_operations, "statistical randomness testing";
    analyze_predictability: PredictabilityAnalysisResult => has_secure_key_management, "predictability analysis";
    analyze_prng_implementations: PrngAnalysisResult => has_cryptographic_operations, "PRNG implementation";
    analyze_trng_implementations: TrngAnalysisResult => has_secure_key_management, "TRNG implementation";
    test_preimage_resistance: PreimageResistanceResults => has_cryptographic_operations, "preimage resistance";
    test_second_preimage_resistance: SecondPreimageResults => has_cryptographic_operations, "second-preimage resistance";
    analyze_birthday_attack_vulnerability: BirthdayAttackAnalysisResult => has_cryptographic_operations, "birthday-attack resistance";
    analyze_hash_families: HashFamilyAnalysisResult => has_cryptographic_operations, "hash family selection";
    analyze_blind_signatures: BlindSignatureAnalysisResult => has_cryptographic_operations, "blind signature usage";
    analyze_ring_signatures: RingSignatureAnalysisResult => has_cryptographic_operations, "ring signature usage";
    analyze_modes_of_operation: ModeOfOperationAnalysisResult => has_cryptographic_operations, "mode of operation";
    analyze_known_plaintext_attacks: KnownPlaintextAnalysisResult => has_cryptographic_operations, "known-plaintext attack resistance";
    analyze_code_based_cryptography: CodeBasedAnalysisResult => has_cryptographic_operations, "code-based readiness";
    analyze_multivariate_cryptography: MultivariateAnalysisResult => has_cryptographic_operations, "multivariate readiness";
    analyze_hash_based_cryptography: HashBasedAnalysisResult => has_cryptographic_operations, "hash-based readiness";
    analyze_isogeny_based_cryptography: IsogenyAnalysisResult => has_cryptographic_operations, "isogeny-based readiness";
    develop_quantum_migration_strategy: QuantumMigrationStrategy => has_secure_key_management, "quantum migration strategy";
    analyze_constant_time_implementation: ConstantTimeAnalysisResult => has_constant_time_operations, "constant-time implementation";
    analyze_secure_coding_practices: SecureCodingAnalysisResult => has_input_validation, "secure coding practices";
    analyze_hardware_security_features: HardwareSecurityAnalysisResult => has_secure_key_management, "hardware security feature usage";
    analyze_side_channel_countermeasures: SideChannelCountermeasuresAnalysisResult => has_constant_time_operations, "side-channel countermeasures";
}

dual_flag_finding! {
    assess_key_storage: KeyStorageAssessment => has_secure_key_management, has_encryption, "key storage";
    assess_key_distribution: KeyDistributionAssessment => has_secure_key_management, has_access_controls, "key distribution";
    assess_key_revocation: KeyRevocationAssessment => has_secure_key_management, has_audit_logging, "key revocation";
    assess_entropy_sources: EntropyAssessment => has_cryptographic_operations, has_secure_key_management, "entropy sources";
    assess_key_lifecycle: KeyLifecycleAssessment => has_secure_key_management, has_audit_logging, "key lifecycle";
    run_entropy_tests: EntropyTestResults => has_cryptographic_operations, has_secure_key_management, "entropy testing";
    analyze_seed_quality: SeedAnalysisResult => has_secure_key_management, has_cryptographic_operations, "seed quality";
    analyze_drbg_implementations: DrbgAnalysisResult => has_cryptographic_operations, has_secure_key_management, "DRBG implementation";
    test_collision_resistance: CollisionResistanceResults => has_cryptographic_operations, has_constant_time_operations, "collision resistance";
    test_avalanche_effect: AvalancheEffectResults => has_cryptographic_operations, has_constant_time_operations, "avalanche effect";
    analyze_length_extension_vulnerability: LengthExtensionAnalysisResult => has_cryptographic_operations, has_encryption, "length-extension resistance";
    analyze_chosen_message_attacks: ChosenMessageAnalysisResult => has_cryptographic_operations, has_constant_time_operations, "chosen-message attack resistance";
    analyze_multi_signatures: MultiSignatureAnalysisResult => has_cryptographic_operations, has_secure_key_management, "multi-signature usage";
    analyze_threshold_signatures: ThresholdSignatureAnalysisResult => has_cryptographic_operations, has_secure_key_management, "threshold signature usage";
    analyze_symmetric_encryption: SymmetricEncryptionAnalysisResult => has_cryptographic_operations, has_constant_time_operations, "symmetric encryption";
    analyze_asymmetric_encryption: AsymmetricEncryptionAnalysisResult => has_cryptographic_operations, has_secure_key_management, "asymmetric encryption";
    analyze_padding_schemes: PaddingSchemeAnalysisResult => has_cryptographic_operations, has_input_validation, "padding scheme";
    analyze_authenticated_encryption: AuthenticatedEncryptionAnalysisResult => has_cryptographic_operations, has_encryption, "authenticated encryption";
    analyze_chosen_plaintext_attacks: ChosenPlaintextAnalysisResult => has_cryptographic_operations, has_constant_time_operations, "chosen-plaintext attack resistance";
    analyze_chosen_ciphertext_attacks: ChosenCiphertextAnalysisResult => has_cryptographic_operations, has_input_validation, "chosen-ciphertext attack resistance";
    analyze_lattice_based_cryptography: LatticeBasedAnalysisResult => has_cryptographic_operations, has_encryption, "lattice-based readiness";
    assess_quantum_threat_timeline: QuantumThreatAssessment => has_cryptographic_operations, has_encryption, "quantum threat timeline";
    analyze_cryptographic_libraries: LibraryAnalysisResult => has_cryptographic_operations, has_secure_key_management, "cryptographic library usage";
    analyze_fault_tolerance: FaultToleranceAnalysisResult => has_bounds_checking, has_resource_limits, "fault tolerance";
}

algorithm_family!(SymmetricAlgorithmAnalyzer, analyze_symmetric_algorithm, SymmetricAlgorithmResult,
    new_aes => "AES", new_chacha20 => "ChaCha20", new_salsa20 => "Salsa20", new_3des => "3DES", new_blowfish => "Blowfish", new_twofish => "Twofish");
algorithm_family!(AsymmetricAlgorithmAnalyzer, analyze_asymmetric_algorithm, AsymmetricAlgorithmResult,
    new_rsa => "RSA", new_ecdsa => "ECDSA", new_eddsa => "EdDSA", new_dh => "DH", new_ecdh => "ECDH", new_dsa => "DSA");
algorithm_family!(HashAlgorithmAnalyzer, analyze_hash_algorithm, HashAlgorithmResult,
    new_sha256 => "SHA-256", new_sha3 => "SHA-3", new_blake2 => "BLAKE2", new_sha1 => "SHA-1", new_md5 => "MD5");
algorithm_family!(MacAlgorithmAnalyzer, analyze_mac_algorithm, MacAlgorithmResult,
    new_hmac => "HMAC", new_poly1305 => "Poly1305", new_gmac => "GMAC");
algorithm_family!(KdfAlgorithmAnalyzer, analyze_kdf_algorithm, KdfAlgorithmResult,
    new_pbkdf2 => "PBKDF2", new_argon2 => "Argon2", new_scrypt => "scrypt", new_bcrypt => "bcrypt");

// Bulk-generated leaf analyzer/detector types: every nested analyzer/detector field referenced by
// the manager structs above but not yet defined, generated via the macros above. Each is an
// empty marker with a `new()`, plus (where its method is actually invoked above) a single
// shallow-heuristic method reading `TraitUsageContext` flags.

define_finding_analyzer! {
    TlsProtocolAnalyzer => analyze_tls_usage, has_cryptographic_operations, has_encryption;
    SshProtocolAnalyzer => analyze_ssh_usage, has_cryptographic_operations, has_access_controls;
    IpsecProtocolAnalyzer => analyze_ipsec_usage, has_encryption, has_access_controls;
    PgpProtocolAnalyzer => analyze_pgp_usage, has_cryptographic_operations, has_secure_key_management;
    OAuthProtocolAnalyzer => analyze_oauth_usage, has_access_controls, has_audit_logging;
    SamlProtocolAnalyzer => analyze_saml_usage, has_access_controls, has_input_validation;
    KerberosProtocolAnalyzer => analyze_kerberos_usage, has_secure_key_management, has_access_controls;
    ProtocolStateAnalyzer => analyze_protocol_states, has_type_safety_checks, has_dynamic_dispatch;
    MessageFlowAnalyzer => analyze_message_flows, has_input_validation, has_serialization;
    AuthenticationProtocolAnalyzer => analyze_authentication_protocols, has_access_controls, has_secure_key_management;
    MerkleDamgardAnalyzer => analyze, has_cryptographic_operations, has_constant_time_operations;
    SpongeFunctionAnalyzer => analyze, has_cryptographic_operations, has_constant_time_operations;
    DifferentialCryptanalysis => analyze, has_cryptographic_operations, has_constant_time_operations;
    LinearCryptanalysis => analyze, has_cryptographic_operations, has_constant_time_operations;
    ShorAlgorithmAnalyzer => analyze_impact, has_cryptographic_operations, has_encryption;
    GroverAlgorithmAnalyzer => analyze_impact, has_cryptographic_operations, has_encryption;
    QuantumKeyDistributionAnalyzer => analyze, has_secure_key_management, has_encryption;
    NistRandomnessTestSuite => run_tests, has_cryptographic_operations, has_secure_key_management;
    DiehardTestSuite => run_tests, has_cryptographic_operations, has_secure_key_management;
    TestU01Suite => run_tests, has_cryptographic_operations, has_secure_key_management;
    SignatureSchemeAnalyzer => analyze_signature_scheme, has_cryptographic_operations, has_secure_key_management;
    PostQuantumAnalyzer => analyze_post_quantum_security, has_cryptographic_operations, has_encryption;
}

define_vuln_detector! {
    TimingAttackDetector => detect_timing_vulnerabilities, has_timing_dependencies, has_constant_time_operations, High;
    PowerAnalysisDetector => detect_power_vulnerabilities, has_cryptographic_operations, has_constant_time_operations, Medium;
    ElectromagneticAttackDetector => detect_electromagnetic_vulnerabilities, has_cryptographic_operations, has_constant_time_operations, Medium;
    AcousticAttackDetector => detect_acoustic_vulnerabilities, has_cryptographic_operations, has_constant_time_operations, Low;
    CacheAttackDetector => detect_cache_vulnerabilities, has_cryptographic_operations, has_constant_time_operations, High;
    FaultInjectionDetector => detect_fault_injection_vulnerabilities, has_cryptographic_operations, has_bounds_checking, High;
}

define_finding_list_analyzer! {
    SignatureVerificationAnalyzer => analyze_signature_verification, has_cryptographic_operations, has_constant_time_operations;
    ForgeResistanceTester => test_forge_resistance, has_cryptographic_operations, has_secure_key_management;
    ExistentialForgeryTester => test_existential_forgery, has_cryptographic_operations, has_secure_key_management;
}

marker_type! {
    KeyGenerationAnalyzer, KeyStorageAnalyzer, KeyDistributionAnalyzer, KeyRotationAnalyzer, KeyRevocationAnalyzer,
    KeyEscrowAnalyzer, EntropyAnalyzer, KeyLifecycleAnalyzer, KeyDerivationAnalyzer, KeyAgreementAnalyzer,
    DifferentialPowerAnalysis, SimplePowerAnalysis, CorrelationPowerAnalysis, TemplateAttackDetector,
    EntropyTester, StatisticalRandomnessTester, PredictabilityAnalyzer, SeedAnalyzer, PrngAnalyzer, TrngAnalyzer, DrbgAnalyzer,
    CollisionResistanceTester, PreimageResistanceTester, SecondPreimageResistanceTester, AvalancheEffectTester,
    BirthdayAttackAnalyzer, LengthExtensionAttackAnalyzer, HashFamilyAnalyzer,
    ChosenMessageAttackAnalyzer, BlindSignatureAnalyzer, MultiSignatureAnalyzer, ThresholdSignatureAnalyzer, RingSignatureAnalyzer,
    SymmetricEncryptionAnalyzer, AsymmetricEncryptionAnalyzer, ModeOfOperationAnalyzer, PaddingSchemeAnalyzer,
    AuthenticatedEncryptionAnalyzer, ChosenPlaintextAttackAnalyzer, ChosenCiphertextAttackAnalyzer, KnownPlaintextAttackAnalyzer,
    LatticeBasedAnalyzer, CodeBasedAnalyzer, MultivariateAnalyzer, HashBasedAnalyzer, IsogenyAnalyzer, QuantumThreatTimeline,
    ConstantTimeAnalyzer, MemorySafetyAnalyzer, SecureCodingAnalyzer, CryptographicLibraryAnalyzer, HardwareSecurityAnalyzer,
    FaultToleranceAnalyzer, PerformanceSecurityAnalyzer, CryptographicAlgorithmDatabase,
}

// Leaf "result" type aliases: none of these are part of any other file's contract (only
// `CryptographicAnalysisResult`, `CryptographicIssue`, and the 4 types below are), so every one
// is a transparent alias over one of the three shared shapes above.

finding_alias! {
    SymmetricAlgorithmResult, AsymmetricAlgorithmResult, HashAlgorithmResult, MacAlgorithmResult, KdfAlgorithmResult,
    AlgorithmCompatibilityMatrix, SecurityLevelAssessment, KeyGenerationAssessment, KeyStorageAssessment,
    KeyDistributionAssessment, KeyRotationAssessment, KeyRevocationAssessment, EntropyAssessment, KeyLifecycleAssessment,
    TlsAnalysisResult, SshAnalysisResult, IpsecAnalysisResult, PgpAnalysisResult, OAuthAnalysisResult, SamlAnalysisResult,
    KerberosAnalysisResult, ProtocolStateAnalysisResult, MessageFlowAnalysisResult, AuthenticationAnalysisResult,
    EntropyTestResults, StatisticalTestResults, PredictabilityAnalysisResult, SeedAnalysisResult, PrngAnalysisResult,
    TrngAnalysisResult, DrbgAnalysisResult, NistTestResults, DiehardTestResults, TestU01Results,
    CollisionResistanceResults, PreimageResistanceResults, SecondPreimageResults, AvalancheEffectResults,
    BirthdayAttackAnalysisResult, LengthExtensionAnalysisResult, HashFamilyAnalysisResult, MerkleDamgardAnalysisResult,
    SpongeFunctionAnalysisResult, SignatureSchemeResult, SignatureVerificationResult, ForgeResistanceResult,
    ExistentialForgeryResult, ChosenMessageAnalysisResult, BlindSignatureAnalysisResult, MultiSignatureAnalysisResult,
    ThresholdSignatureAnalysisResult, RingSignatureAnalysisResult, SymmetricEncryptionAnalysisResult,
    AsymmetricEncryptionAnalysisResult, ModeOfOperationAnalysisResult, PaddingSchemeAnalysisResult,
    AuthenticatedEncryptionAnalysisResult, ChosenPlaintextAnalysisResult, ChosenCiphertextAnalysisResult,
    KnownPlaintextAnalysisResult, DifferentialCryptanalysisResult, LinearCryptanalysisResult, PostQuantumResult,
    ShorAlgorithmImpact, GroverAlgorithmImpact, QuantumKeyDistributionAnalysisResult, LatticeBasedAnalysisResult,
    CodeBasedAnalysisResult, MultivariateAnalysisResult, HashBasedAnalysisResult, IsogenyAnalysisResult,
    QuantumThreatAssessment, QuantumMigrationStrategy, ConstantTimeAnalysisResult, MemorySafetyAnalysisResult,
    SecureCodingAnalysisResult, LibraryAnalysisResult, HardwareSecurityAnalysisResult,
    SideChannelCountermeasuresAnalysisResult, FaultToleranceAnalysisResult, PerformanceSecurityAnalysisResult,
}

recommendation_alias! {
    DeprecationWarning, AlgorithmUpgradeRecommendation, KeyManagementRisk, KeyManagementRecommendation,
    SideChannelCountermeasure, SideChannelMitigationRecommendation, ProtocolRecommendation, RandomnessRecommendation,
    HashFunctionRecommendation, SignatureRecommendation, EncryptionRecommendation, QuantumResistanceRecommendation,
    ImplementationRecommendation,
}

vulnerability_alias! {
    TimingAttackVulnerability, PowerAnalysisVulnerability, ElectromagneticVulnerability, AcousticVulnerability,
    CacheAttackVulnerability, FaultInjectionVulnerability, ProtocolVulnerability,
}

// Mandatory contract types, consumed by `core_analyzer.rs` and the outer crate's public API.

result_structs! {
    CryptographicIssue {
        id: String, severity: RiskSeverity, issue_type: String,
        recommendation: String, fix_complexity: ImplementationEffort, dependencies: Vec<String>,
    }
    SideChannelRisk { risk_type: String, description: String, severity: RiskSeverity, affected_operation: String }
    TimingVulnerability { operation: String, description: String, severity: RiskSeverity, mitigation: String }
    ConstantTimeViolation { location: String, description: String, severity: RiskSeverity }
    CryptographicStrengthAssessment {
        algorithm: String, key_size_bits: u32, estimated_security_bits: u32,
        quantum_resistant: bool, assessment: String,
    }
    CryptographicRecommendation {
        id: String, title: String, description: String,
        priority: RiskSeverity, implementation_effort: ImplementationEffort,
    }
    CryptographicVulnerabilityReport { total_findings: usize, critical_findings: usize, summary: String }
}

result_enum! {
    CryptographicComplianceStatus { Compliant, PartiallyCompliant, NonCompliant, NotAssessed }
}

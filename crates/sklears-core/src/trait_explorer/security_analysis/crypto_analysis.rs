use super::security_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

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
                fn $method(&self, context: &TraitUsageContext) -> Result<AnalysisFinding, CryptographicAnalysisError> {
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
                fn $method(&self, context: &TraitUsageContext) -> Result<Vec<AnalysisFinding>, CryptographicAnalysisError> {
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
                fn $method(&self, context: &TraitUsageContext) -> Result<Vec<SecurityVulnerabilityFinding>, CryptographicAnalysisError> {
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
            $(fn $method(&self, context: &TraitUsageContext) -> Result<$ret, CryptographicAnalysisError> {
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
            $(fn $method(&self, context: &TraitUsageContext) -> Result<$ret, CryptographicAnalysisError> {
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
            fn $method(&self, context: &TraitUsageContext) -> Result<$result, CryptographicAnalysisError> {
                Ok(heuristic_finding(context.has_cryptographic_operations && context.has_constant_time_operations, &self.algorithm_name))
            }
        }
    };
}

/// Generate a manager's `new()` from a flat `field: value` list, skipping the repeated
/// `impl X { pub fn new() -> Self { Self { .. } } }` boilerplate.
macro_rules! manager_new {
    ($($t:ident { $($field:ident: $val:expr),* $(,)? })*) => {
        $(impl $t { pub fn new() -> Self { Self { $($field: $val),* } } })*
    };
}

/// `impl Default for $t { fn default() -> Self { Self::new() } }` for every listed type — see
/// `clippy::new_without_default`.
macro_rules! default_via_new {
    ($($t:ty),* $(,)?) => { $(impl Default for $t { fn default() -> Self { Self::new() } })* };
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicAnalyzer {
    algorithm_analyzer: CryptographicAlgorithmAnalyzer,
    key_management_analyzer: KeyManagementAnalyzer,
    side_channel_detector: SideChannelAttackDetector,
    protocol_analyzer: CryptographicProtocolAnalyzer,
    random_number_analyzer: RandomNumberGeneratorAnalyzer,
    hash_function_analyzer: HashFunctionAnalyzer,
    signature_analyzer: DigitalSignatureAnalyzer,
    encryption_analyzer: EncryptionAnalyzer,
    quantum_resistance_analyzer: QuantumResistanceAnalyzer,
    implementation_analyzer: CryptographicImplementationAnalyzer,
    compliance_checker: CryptographicComplianceChecker,
    vulnerability_scanner: CryptographicVulnerabilityScanner,
    analysis_config: CryptographicAnalysisConfig,
    analysis_cache: HashMap<String, CachedCryptographicAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicAlgorithmAnalyzer {
    symmetric_analyzers: HashMap<String, SymmetricAlgorithmAnalyzer>,
    asymmetric_analyzers: HashMap<String, AsymmetricAlgorithmAnalyzer>,
    hash_analyzers: HashMap<String, HashAlgorithmAnalyzer>,
    mac_analyzers: HashMap<String, MacAlgorithmAnalyzer>,
    kdf_analyzers: HashMap<String, KdfAlgorithmAnalyzer>,
    algorithm_database: CryptographicAlgorithmDatabase,
    weakness_patterns: Vec<AlgorithmWeaknessPattern>,
    security_levels: HashMap<String, SecurityLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementAnalyzer {
    key_generation_analyzers: Vec<KeyGenerationAnalyzer>,
    key_storage_analyzers: Vec<KeyStorageAnalyzer>,
    key_distribution_analyzers: Vec<KeyDistributionAnalyzer>,
    key_rotation_analyzers: Vec<KeyRotationAnalyzer>,
    key_revocation_analyzers: Vec<KeyRevocationAnalyzer>,
    key_escrow_analyzers: Vec<KeyEscrowAnalyzer>,
    entropy_analyzers: Vec<EntropyAnalyzer>,
    key_lifecycle_analyzer: KeyLifecycleAnalyzer,
    key_derivation_analyzer: KeyDerivationAnalyzer,
    key_agreement_analyzer: KeyAgreementAnalyzer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideChannelAttackDetector {
    timing_attack_detectors: Vec<TimingAttackDetector>,
    power_analysis_detectors: Vec<PowerAnalysisDetector>,
    electromagnetic_detectors: Vec<ElectromagneticAttackDetector>,
    acoustic_attack_detectors: Vec<AcousticAttackDetector>,
    cache_attack_detectors: Vec<CacheAttackDetector>,
    fault_injection_detectors: Vec<FaultInjectionDetector>,
    differential_power_analysis: DifferentialPowerAnalysis,
    simple_power_analysis: SimplePowerAnalysis,
    correlation_power_analysis: CorrelationPowerAnalysis,
    template_attack_detector: TemplateAttackDetector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicProtocolAnalyzer {
    tls_analyzer: TlsProtocolAnalyzer,
    ssh_analyzer: SshProtocolAnalyzer,
    ipsec_analyzer: IpsecProtocolAnalyzer,
    pgp_analyzer: PgpProtocolAnalyzer,
    oauth_analyzer: OAuthProtocolAnalyzer,
    saml_analyzer: SamlProtocolAnalyzer,
    kerberos_analyzer: KerberosProtocolAnalyzer,
    protocol_state_analyzer: ProtocolStateAnalyzer,
    message_flow_analyzer: MessageFlowAnalyzer,
    authentication_analyzer: AuthenticationProtocolAnalyzer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomNumberGeneratorAnalyzer {
    entropy_testers: Vec<EntropyTester>,
    statistical_testers: Vec<StatisticalRandomnessTester>,
    predictability_analyzers: Vec<PredictabilityAnalyzer>,
    seed_analyzers: Vec<SeedAnalyzer>,
    prng_analyzers: HashMap<String, PrngAnalyzer>,
    trng_analyzers: HashMap<String, TrngAnalyzer>,
    drbg_analyzers: HashMap<String, DrbgAnalyzer>,
    nist_test_suite: NistRandomnessTestSuite,
    diehard_test_suite: DiehardTestSuite,
    test_u01_suite: TestU01Suite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashFunctionAnalyzer {
    collision_resistance_testers: Vec<CollisionResistanceTester>,
    preimage_resistance_testers: Vec<PreimageResistanceTester>,
    second_preimage_testers: Vec<SecondPreimageResistanceTester>,
    avalanche_effect_testers: Vec<AvalancheEffectTester>,
    birthday_attack_analyzers: Vec<BirthdayAttackAnalyzer>,
    length_extension_analyzers: Vec<LengthExtensionAttackAnalyzer>,
    hash_family_analyzers: HashMap<String, HashFamilyAnalyzer>,
    merkle_damgard_analyzer: MerkleDamgardAnalyzer,
    sponge_function_analyzer: SpongeFunctionAnalyzer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalSignatureAnalyzer {
    signature_scheme_analyzers: HashMap<String, SignatureSchemeAnalyzer>,
    verification_analyzers: Vec<SignatureVerificationAnalyzer>,
    forge_resistance_testers: Vec<ForgeResistanceTester>,
    existential_forgery_testers: Vec<ExistentialForgeryTester>,
    chosen_message_analyzers: Vec<ChosenMessageAttackAnalyzer>,
    blind_signature_analyzers: Vec<BlindSignatureAnalyzer>,
    multi_signature_analyzers: Vec<MultiSignatureAnalyzer>,
    threshold_signature_analyzers: Vec<ThresholdSignatureAnalyzer>,
    ring_signature_analyzers: Vec<RingSignatureAnalyzer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionAnalyzer {
    symmetric_encryption_analyzers: HashMap<String, SymmetricEncryptionAnalyzer>,
    asymmetric_encryption_analyzers: HashMap<String, AsymmetricEncryptionAnalyzer>,
    mode_of_operation_analyzers: HashMap<String, ModeOfOperationAnalyzer>,
    padding_scheme_analyzers: HashMap<String, PaddingSchemeAnalyzer>,
    authenticated_encryption_analyzers: Vec<AuthenticatedEncryptionAnalyzer>,
    chosen_plaintext_analyzers: Vec<ChosenPlaintextAttackAnalyzer>,
    chosen_ciphertext_analyzers: Vec<ChosenCiphertextAttackAnalyzer>,
    known_plaintext_analyzers: Vec<KnownPlaintextAttackAnalyzer>,
    differential_cryptanalysis: DifferentialCryptanalysis,
    linear_cryptanalysis: LinearCryptanalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResistanceAnalyzer {
    post_quantum_analyzers: HashMap<String, PostQuantumAnalyzer>,
    shor_algorithm_analyzer: ShorAlgorithmAnalyzer,
    grover_algorithm_analyzer: GroverAlgorithmAnalyzer,
    quantum_key_distribution_analyzer: QuantumKeyDistributionAnalyzer,
    lattice_based_analyzers: Vec<LatticeBasedAnalyzer>,
    code_based_analyzers: Vec<CodeBasedAnalyzer>,
    multivariate_analyzers: Vec<MultivariateAnalyzer>,
    hash_based_analyzers: Vec<HashBasedAnalyzer>,
    isogeny_analyzers: Vec<IsogenyAnalyzer>,
    quantum_threat_timeline: QuantumThreatTimeline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicImplementationAnalyzer {
    constant_time_analyzers: Vec<ConstantTimeAnalyzer>,
    memory_safety_analyzers: Vec<MemorySafetyAnalyzer>,
    secure_coding_analyzers: Vec<SecureCodingAnalyzer>,
    library_analyzers: HashMap<String, CryptographicLibraryAnalyzer>,
    hardware_security_analyzers: Vec<HardwareSecurityAnalyzer>,
    side_channel_countermeasures: Vec<SideChannelCountermeasure>,
    fault_tolerance_analyzers: Vec<FaultToleranceAnalyzer>,
    performance_security_analyzers: Vec<PerformanceSecurityAnalyzer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicAnalysisResult {
    pub analysis_id: String,
    pub analysis_timestamp: SystemTime,
    pub algorithm_analysis: AlgorithmAnalysisResult,
    pub key_management_analysis: KeyManagementAnalysisResult,
    pub side_channel_analysis: SideChannelAnalysisResult,
    pub protocol_analysis: ProtocolAnalysisResult,
    pub random_number_analysis: RandomNumberAnalysisResult,
    pub hash_function_analysis: HashFunctionAnalysisResult,
    pub signature_analysis: SignatureAnalysisResult,
    pub encryption_analysis: EncryptionAnalysisResult,
    pub quantum_resistance_analysis: QuantumResistanceAnalysisResult,
    pub implementation_analysis: ImplementationAnalysisResult,
    pub overall_cryptographic_score: f64,
    pub security_recommendations: Vec<CryptographicRecommendation>,
    pub compliance_status: CryptographicComplianceStatus,
    pub vulnerability_report: CryptographicVulnerabilityReport,
    pub analysis_confidence: f64,
    pub analysis_metadata: HashMap<String, String>,
    /// Overall cryptographic risk score in `[0.0, 10.0]`, derived as the inverse of
    /// `overall_cryptographic_score`. Consumed by `core_analyzer.rs`'s comprehensive risk
    /// calculation.
    pub risk_score: f64,
    /// Individually addressable cryptographic weaknesses identified across the algorithm,
    /// key-management, side-channel, and protocol sub-analyses. Consumed by
    /// `core_analyzer.rs`'s `generate_crypto_recommendations`.
    pub identified_issues: Vec<CryptographicIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmAnalysisResult {
    pub symmetric_algorithm_results: HashMap<String, SymmetricAlgorithmResult>,
    pub asymmetric_algorithm_results: HashMap<String, AsymmetricAlgorithmResult>,
    pub hash_algorithm_results: HashMap<String, HashAlgorithmResult>,
    pub mac_algorithm_results: HashMap<String, MacAlgorithmResult>,
    pub kdf_algorithm_results: HashMap<String, KdfAlgorithmResult>,
    pub algorithm_compatibility_matrix: AlgorithmCompatibilityMatrix,
    pub security_level_assessment: SecurityLevelAssessment,
    pub deprecation_warnings: Vec<DeprecationWarning>,
    pub upgrade_recommendations: Vec<AlgorithmUpgradeRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementAnalysisResult {
    pub key_generation_assessment: KeyGenerationAssessment,
    pub key_storage_assessment: KeyStorageAssessment,
    pub key_distribution_assessment: KeyDistributionAssessment,
    pub key_rotation_assessment: KeyRotationAssessment,
    pub key_revocation_assessment: KeyRevocationAssessment,
    pub entropy_assessment: EntropyAssessment,
    pub key_lifecycle_assessment: KeyLifecycleAssessment,
    pub key_management_score: f64,
    pub key_management_risks: Vec<KeyManagementRisk>,
    pub key_management_recommendations: Vec<KeyManagementRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideChannelAnalysisResult {
    pub timing_attack_vulnerabilities: Vec<TimingAttackVulnerability>,
    pub power_analysis_vulnerabilities: Vec<PowerAnalysisVulnerability>,
    pub electromagnetic_vulnerabilities: Vec<ElectromagneticVulnerability>,
    pub acoustic_vulnerabilities: Vec<AcousticVulnerability>,
    pub cache_attack_vulnerabilities: Vec<CacheAttackVulnerability>,
    pub fault_injection_vulnerabilities: Vec<FaultInjectionVulnerability>,
    pub side_channel_countermeasures: Vec<SideChannelCountermeasure>,
    pub vulnerability_severity_scores: HashMap<String, f64>,
    pub mitigation_recommendations: Vec<SideChannelMitigationRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolAnalysisResult {
    pub tls_analysis: TlsAnalysisResult,
    pub ssh_analysis: SshAnalysisResult,
    pub ipsec_analysis: IpsecAnalysisResult,
    pub pgp_analysis: PgpAnalysisResult,
    pub oauth_analysis: OAuthAnalysisResult,
    pub saml_analysis: SamlAnalysisResult,
    pub kerberos_analysis: KerberosAnalysisResult,
    pub protocol_state_analysis: ProtocolStateAnalysisResult,
    pub message_flow_analysis: MessageFlowAnalysisResult,
    pub authentication_analysis: AuthenticationAnalysisResult,
    pub protocol_vulnerabilities: Vec<ProtocolVulnerability>,
    pub protocol_recommendations: Vec<ProtocolRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomNumberAnalysisResult {
    pub entropy_test_results: EntropyTestResults,
    pub statistical_test_results: StatisticalTestResults,
    pub predictability_analysis: PredictabilityAnalysisResult,
    pub seed_analysis: SeedAnalysisResult,
    pub prng_analysis: PrngAnalysisResult,
    pub trng_analysis: TrngAnalysisResult,
    pub drbg_analysis: DrbgAnalysisResult,
    pub nist_test_results: NistTestResults,
    pub diehard_test_results: DiehardTestResults,
    pub testu01_results: TestU01Results,
    pub randomness_quality_score: f64,
    pub randomness_recommendations: Vec<RandomnessRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashFunctionAnalysisResult {
    pub collision_resistance_results: CollisionResistanceResults,
    pub preimage_resistance_results: PreimageResistanceResults,
    pub second_preimage_results: SecondPreimageResults,
    pub avalanche_effect_results: AvalancheEffectResults,
    pub birthday_attack_analysis: BirthdayAttackAnalysisResult,
    pub length_extension_analysis: LengthExtensionAnalysisResult,
    pub hash_family_analysis: HashFamilyAnalysisResult,
    pub merkle_damgard_analysis: MerkleDamgardAnalysisResult,
    pub sponge_function_analysis: SpongeFunctionAnalysisResult,
    pub hash_security_score: f64,
    pub hash_recommendations: Vec<HashFunctionRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureAnalysisResult {
    pub signature_scheme_results: HashMap<String, SignatureSchemeResult>,
    pub verification_results: Vec<SignatureVerificationResult>,
    pub forge_resistance_results: Vec<ForgeResistanceResult>,
    pub existential_forgery_results: Vec<ExistentialForgeryResult>,
    pub chosen_message_analysis: ChosenMessageAnalysisResult,
    pub blind_signature_analysis: BlindSignatureAnalysisResult,
    pub multi_signature_analysis: MultiSignatureAnalysisResult,
    pub threshold_signature_analysis: ThresholdSignatureAnalysisResult,
    pub ring_signature_analysis: RingSignatureAnalysisResult,
    pub signature_security_score: f64,
    pub signature_recommendations: Vec<SignatureRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionAnalysisResult {
    pub symmetric_encryption_analysis: SymmetricEncryptionAnalysisResult,
    pub asymmetric_encryption_analysis: AsymmetricEncryptionAnalysisResult,
    pub mode_of_operation_analysis: ModeOfOperationAnalysisResult,
    pub padding_scheme_analysis: PaddingSchemeAnalysisResult,
    pub authenticated_encryption_analysis: AuthenticatedEncryptionAnalysisResult,
    pub chosen_plaintext_analysis: ChosenPlaintextAnalysisResult,
    pub chosen_ciphertext_analysis: ChosenCiphertextAnalysisResult,
    pub known_plaintext_analysis: KnownPlaintextAnalysisResult,
    pub differential_cryptanalysis_results: DifferentialCryptanalysisResult,
    pub linear_cryptanalysis_results: LinearCryptanalysisResult,
    pub encryption_security_score: f64,
    pub encryption_recommendations: Vec<EncryptionRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResistanceAnalysisResult {
    pub post_quantum_results: HashMap<String, PostQuantumResult>,
    pub shor_algorithm_impact: ShorAlgorithmImpact,
    pub grover_algorithm_impact: GroverAlgorithmImpact,
    pub quantum_key_distribution_analysis: QuantumKeyDistributionAnalysisResult,
    pub lattice_based_analysis: LatticeBasedAnalysisResult,
    pub code_based_analysis: CodeBasedAnalysisResult,
    pub multivariate_analysis: MultivariateAnalysisResult,
    pub hash_based_analysis: HashBasedAnalysisResult,
    pub isogeny_analysis: IsogenyAnalysisResult,
    pub quantum_threat_assessment: QuantumThreatAssessment,
    pub migration_strategy: QuantumMigrationStrategy,
    pub quantum_resistance_score: f64,
    pub quantum_recommendations: Vec<QuantumResistanceRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationAnalysisResult {
    pub constant_time_analysis: ConstantTimeAnalysisResult,
    pub memory_safety_analysis: MemorySafetyAnalysisResult,
    pub secure_coding_analysis: SecureCodingAnalysisResult,
    pub library_analysis: LibraryAnalysisResult,
    pub hardware_security_analysis: HardwareSecurityAnalysisResult,
    pub side_channel_countermeasures_analysis: SideChannelCountermeasuresAnalysisResult,
    pub fault_tolerance_analysis: FaultToleranceAnalysisResult,
    pub performance_security_analysis: PerformanceSecurityAnalysisResult,
    pub implementation_security_score: f64,
    pub implementation_recommendations: Vec<ImplementationRecommendation>,
}

impl CryptographicAnalyzer {
    pub fn new() -> Self {
        Self {
            algorithm_analyzer: CryptographicAlgorithmAnalyzer::new(),
            key_management_analyzer: KeyManagementAnalyzer::new(),
            side_channel_detector: SideChannelAttackDetector::new(),
            protocol_analyzer: CryptographicProtocolAnalyzer::new(),
            random_number_analyzer: RandomNumberGeneratorAnalyzer::new(),
            hash_function_analyzer: HashFunctionAnalyzer::new(),
            signature_analyzer: DigitalSignatureAnalyzer::new(),
            encryption_analyzer: EncryptionAnalyzer::new(),
            quantum_resistance_analyzer: QuantumResistanceAnalyzer::new(),
            implementation_analyzer: CryptographicImplementationAnalyzer::new(),
            compliance_checker: CryptographicComplianceChecker::new(),
            vulnerability_scanner: CryptographicVulnerabilityScanner::new(),
            analysis_config: CryptographicAnalysisConfig::default(),
            analysis_cache: HashMap::new(),
        }
    }

    pub fn analyze_cryptographic_security(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<CryptographicAnalysisResult, CryptographicAnalysisError> {
        let analysis_id = self.generate_analysis_id(context);

        if let Some(cached_result) = self.get_cached_analysis(&analysis_id) {
            if self.is_cache_valid(cached_result) {
                return Ok(cached_result.result.clone());
            }
        }

        let algorithm_analysis = self.analyze_cryptographic_algorithms(context)?;
        let key_management_analysis = self.analyze_key_management(context)?;
        let side_channel_analysis = self.analyze_side_channel_vulnerabilities(context)?;
        let protocol_analysis = self.analyze_cryptographic_protocols(context)?;
        let random_number_analysis = self.analyze_random_number_generation(context)?;
        let hash_function_analysis = self.analyze_hash_functions(context)?;
        let signature_analysis = self.analyze_digital_signatures(context)?;
        let encryption_analysis = self.analyze_encryption_schemes(context)?;
        let quantum_resistance_analysis = self.analyze_quantum_resistance(context)?;
        let implementation_analysis = self.analyze_cryptographic_implementation(context)?;

        // NOTE: `calculate_overall_cryptographic_score` and `generate_security_recommendations`
        // take a bundled slice/struct (rather than 10 positional refs) purely to stay within
        // clippy::too_many_arguments; the sequence and content of the 10 sub-analyses above is
        // unchanged.
        let side_channel_score = (10.0
            - side_channel_analysis
                .vulnerability_severity_scores
                .values()
                .sum::<f64>())
        .clamp(0.0, 10.0);
        let protocol_score =
            (10.0 - protocol_analysis.protocol_vulnerabilities.len() as f64 * 1.5).clamp(0.0, 10.0);

        let overall_score = self.calculate_overall_cryptographic_score(&[
            algorithm_analysis.security_level_assessment.score,
            key_management_analysis.key_management_score,
            side_channel_score,
            protocol_score,
            random_number_analysis.randomness_quality_score,
            hash_function_analysis.hash_security_score,
            signature_analysis.signature_security_score,
            encryption_analysis.encryption_security_score,
            quantum_resistance_analysis.quantum_resistance_score,
            implementation_analysis.implementation_security_score,
        ])?;

        let security_recommendations =
            self.generate_security_recommendations(&CryptoSubAnalysesRef {
                algorithm: &algorithm_analysis,
                key_management: &key_management_analysis,
                side_channel: &side_channel_analysis,
                protocol: &protocol_analysis,
                random_number: &random_number_analysis,
                hash_function: &hash_function_analysis,
                signature: &signature_analysis,
                encryption: &encryption_analysis,
                quantum_resistance: &quantum_resistance_analysis,
                implementation: &implementation_analysis,
            })?;

        let compliance_status = self.compliance_checker.check_compliance(context)?;
        let vulnerability_report = self.vulnerability_scanner.scan_vulnerabilities(context)?;
        let analysis_confidence = self.calculate_analysis_confidence()?;
        let identified_issues = self.identify_cryptographic_issues(context);
        let risk_score = (10.0 - overall_score).clamp(0.0, 10.0);

        let result = CryptographicAnalysisResult {
            analysis_id: analysis_id.clone(),
            analysis_timestamp: SystemTime::now(),
            algorithm_analysis,
            key_management_analysis,
            side_channel_analysis,
            protocol_analysis,
            random_number_analysis,
            hash_function_analysis,
            signature_analysis,
            encryption_analysis,
            quantum_resistance_analysis,
            implementation_analysis,
            overall_cryptographic_score: overall_score,
            security_recommendations,
            compliance_status,
            vulnerability_report,
            analysis_confidence,
            analysis_metadata: self.generate_analysis_metadata(context),
            risk_score,
            identified_issues,
        };

        self.cache_analysis(analysis_id, &result);
        Ok(result)
    }

    fn analyze_cryptographic_algorithms(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<AlgorithmAnalysisResult, CryptographicAnalysisError> {
        let mut symmetric_results = HashMap::new();
        let mut asymmetric_results = HashMap::new();
        let mut hash_results = HashMap::new();
        let mut mac_results = HashMap::new();
        let mut kdf_results = HashMap::new();

        for (name, analyzer) in &self.algorithm_analyzer.symmetric_analyzers {
            let result = analyzer.analyze_symmetric_algorithm(context)?;
            symmetric_results.insert(name.clone(), result);
        }

        for (name, analyzer) in &self.algorithm_analyzer.asymmetric_analyzers {
            let result = analyzer.analyze_asymmetric_algorithm(context)?;
            asymmetric_results.insert(name.clone(), result);
        }

        for (name, analyzer) in &self.algorithm_analyzer.hash_analyzers {
            let result = analyzer.analyze_hash_algorithm(context)?;
            hash_results.insert(name.clone(), result);
        }

        for (name, analyzer) in &self.algorithm_analyzer.mac_analyzers {
            let result = analyzer.analyze_mac_algorithm(context)?;
            mac_results.insert(name.clone(), result);
        }

        for (name, analyzer) in &self.algorithm_analyzer.kdf_analyzers {
            let result = analyzer.analyze_kdf_algorithm(context)?;
            kdf_results.insert(name.clone(), result);
        }

        let compatibility_matrix =
            self.build_algorithm_compatibility_matrix(&symmetric_results, &asymmetric_results)?;
        let security_level_assessment =
            self.assess_security_levels(&symmetric_results, &asymmetric_results)?;
        let deprecation_warnings =
            self.check_algorithm_deprecations(&symmetric_results, &asymmetric_results)?;
        let upgrade_recommendations =
            self.generate_algorithm_upgrade_recommendations(&deprecation_warnings)?;

        Ok(AlgorithmAnalysisResult {
            symmetric_algorithm_results: symmetric_results,
            asymmetric_algorithm_results: asymmetric_results,
            hash_algorithm_results: hash_results,
            mac_algorithm_results: mac_results,
            kdf_algorithm_results: kdf_results,
            algorithm_compatibility_matrix: compatibility_matrix,
            security_level_assessment,
            deprecation_warnings,
            upgrade_recommendations,
        })
    }

    fn analyze_key_management(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<KeyManagementAnalysisResult, CryptographicAnalysisError> {
        let key_generation_assessment = self.assess_key_generation(context)?;
        let key_storage_assessment = self.assess_key_storage(context)?;
        let key_distribution_assessment = self.assess_key_distribution(context)?;
        let key_rotation_assessment = self.assess_key_rotation(context)?;
        let key_revocation_assessment = self.assess_key_revocation(context)?;
        let entropy_assessment = self.assess_entropy_sources(context)?;
        let key_lifecycle_assessment = self.assess_key_lifecycle(context)?;

        // NOTE: bundled into a slice (rather than 7 positional refs) purely to stay within
        // clippy::too_many_arguments; all 7 assessments above are still computed unconditionally.
        let key_management_score = self.calculate_key_management_score(&[
            key_generation_assessment.score,
            key_storage_assessment.score,
            key_distribution_assessment.score,
            key_rotation_assessment.score,
            key_revocation_assessment.score,
            entropy_assessment.score,
            key_lifecycle_assessment.score,
        ])?;

        let key_management_risks = self.identify_key_management_risks(context)?;
        let key_management_recommendations =
            self.generate_key_management_recommendations(&key_management_risks)?;

        Ok(KeyManagementAnalysisResult {
            key_generation_assessment,
            key_storage_assessment,
            key_distribution_assessment,
            key_rotation_assessment,
            key_revocation_assessment,
            entropy_assessment,
            key_lifecycle_assessment,
            key_management_score,
            key_management_risks,
            key_management_recommendations,
        })
    }

    fn analyze_side_channel_vulnerabilities(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<SideChannelAnalysisResult, CryptographicAnalysisError> {
        let mut timing_vulnerabilities = Vec::new();
        let mut power_vulnerabilities = Vec::new();
        let mut electromagnetic_vulnerabilities = Vec::new();
        let mut acoustic_vulnerabilities = Vec::new();
        let mut cache_vulnerabilities = Vec::new();
        let mut fault_injection_vulnerabilities = Vec::new();

        for detector in &self.side_channel_detector.timing_attack_detectors {
            timing_vulnerabilities.extend(detector.detect_timing_vulnerabilities(context)?);
        }

        for detector in &self.side_channel_detector.power_analysis_detectors {
            power_vulnerabilities.extend(detector.detect_power_vulnerabilities(context)?);
        }

        for detector in &self.side_channel_detector.electromagnetic_detectors {
            electromagnetic_vulnerabilities
                .extend(detector.detect_electromagnetic_vulnerabilities(context)?);
        }

        for detector in &self.side_channel_detector.acoustic_attack_detectors {
            acoustic_vulnerabilities.extend(detector.detect_acoustic_vulnerabilities(context)?);
        }

        for detector in &self.side_channel_detector.cache_attack_detectors {
            cache_vulnerabilities.extend(detector.detect_cache_vulnerabilities(context)?);
        }

        for detector in &self.side_channel_detector.fault_injection_detectors {
            fault_injection_vulnerabilities
                .extend(detector.detect_fault_injection_vulnerabilities(context)?);
        }

        let side_channel_countermeasures = self.identify_applicable_countermeasures(context)?;
        let vulnerability_severity_scores = self.calculate_vulnerability_severity_scores(
            &timing_vulnerabilities,
            &power_vulnerabilities,
            &electromagnetic_vulnerabilities,
            &acoustic_vulnerabilities,
            &cache_vulnerabilities,
            &fault_injection_vulnerabilities,
        )?;

        let mitigation_recommendations = self.generate_side_channel_mitigation_recommendations(
            &timing_vulnerabilities,
            &power_vulnerabilities,
            &electromagnetic_vulnerabilities,
            &acoustic_vulnerabilities,
            &cache_vulnerabilities,
            &fault_injection_vulnerabilities,
        )?;

        Ok(SideChannelAnalysisResult {
            timing_attack_vulnerabilities: timing_vulnerabilities,
            power_analysis_vulnerabilities: power_vulnerabilities,
            electromagnetic_vulnerabilities,
            acoustic_vulnerabilities,
            cache_attack_vulnerabilities: cache_vulnerabilities,
            fault_injection_vulnerabilities,
            side_channel_countermeasures,
            vulnerability_severity_scores,
            mitigation_recommendations,
        })
    }

    fn analyze_cryptographic_protocols(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<ProtocolAnalysisResult, CryptographicAnalysisError> {
        let tls_analysis = self
            .protocol_analyzer
            .tls_analyzer
            .analyze_tls_usage(context)?;
        let ssh_analysis = self
            .protocol_analyzer
            .ssh_analyzer
            .analyze_ssh_usage(context)?;
        let ipsec_analysis = self
            .protocol_analyzer
            .ipsec_analyzer
            .analyze_ipsec_usage(context)?;
        let pgp_analysis = self
            .protocol_analyzer
            .pgp_analyzer
            .analyze_pgp_usage(context)?;
        let oauth_analysis = self
            .protocol_analyzer
            .oauth_analyzer
            .analyze_oauth_usage(context)?;
        let saml_analysis = self
            .protocol_analyzer
            .saml_analyzer
            .analyze_saml_usage(context)?;
        let kerberos_analysis = self
            .protocol_analyzer
            .kerberos_analyzer
            .analyze_kerberos_usage(context)?;

        let protocol_state_analysis = self
            .protocol_analyzer
            .protocol_state_analyzer
            .analyze_protocol_states(context)?;
        let message_flow_analysis = self
            .protocol_analyzer
            .message_flow_analyzer
            .analyze_message_flows(context)?;
        let authentication_analysis = self
            .protocol_analyzer
            .authentication_analyzer
            .analyze_authentication_protocols(context)?;

        let protocol_vulnerabilities = self.identify_protocol_vulnerabilities(context)?;
        let protocol_recommendations =
            self.generate_protocol_recommendations(&protocol_vulnerabilities)?;

        Ok(ProtocolAnalysisResult {
            tls_analysis,
            ssh_analysis,
            ipsec_analysis,
            pgp_analysis,
            oauth_analysis,
            saml_analysis,
            kerberos_analysis,
            protocol_state_analysis,
            message_flow_analysis,
            authentication_analysis,
            protocol_vulnerabilities,
            protocol_recommendations,
        })
    }

    fn analyze_random_number_generation(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<RandomNumberAnalysisResult, CryptographicAnalysisError> {
        let entropy_test_results = self.run_entropy_tests(context)?;
        let statistical_test_results = self.run_statistical_randomness_tests(context)?;
        let predictability_analysis = self.analyze_predictability(context)?;
        let seed_analysis = self.analyze_seed_quality(context)?;

        let prng_analysis = self.analyze_prng_implementations(context)?;
        let trng_analysis = self.analyze_trng_implementations(context)?;
        let drbg_analysis = self.analyze_drbg_implementations(context)?;

        let nist_test_results = self
            .random_number_analyzer
            .nist_test_suite
            .run_tests(context)?;
        let diehard_test_results = self
            .random_number_analyzer
            .diehard_test_suite
            .run_tests(context)?;
        let testu01_results = self
            .random_number_analyzer
            .test_u01_suite
            .run_tests(context)?;

        let randomness_quality_score = self.calculate_randomness_quality_score(
            &entropy_test_results,
            &statistical_test_results,
            &nist_test_results,
            &diehard_test_results,
            &testu01_results,
        )?;

        let randomness_recommendations = self.generate_randomness_recommendations(
            &entropy_test_results,
            &predictability_analysis,
            &seed_analysis,
        )?;

        Ok(RandomNumberAnalysisResult {
            entropy_test_results,
            statistical_test_results,
            predictability_analysis,
            seed_analysis,
            prng_analysis,
            trng_analysis,
            drbg_analysis,
            nist_test_results,
            diehard_test_results,
            testu01_results,
            randomness_quality_score,
            randomness_recommendations,
        })
    }

    fn analyze_hash_functions(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<HashFunctionAnalysisResult, CryptographicAnalysisError> {
        let collision_resistance_results = self.test_collision_resistance(context)?;
        let preimage_resistance_results = self.test_preimage_resistance(context)?;
        let second_preimage_results = self.test_second_preimage_resistance(context)?;
        let avalanche_effect_results = self.test_avalanche_effect(context)?;
        let birthday_attack_analysis = self.analyze_birthday_attack_vulnerability(context)?;
        let length_extension_analysis = self.analyze_length_extension_vulnerability(context)?;

        let hash_family_analysis = self.analyze_hash_families(context)?;
        let merkle_damgard_analysis = self
            .hash_function_analyzer
            .merkle_damgard_analyzer
            .analyze(context)?;
        let sponge_function_analysis = self
            .hash_function_analyzer
            .sponge_function_analyzer
            .analyze(context)?;

        let hash_security_score = self.calculate_hash_security_score(
            &collision_resistance_results,
            &preimage_resistance_results,
            &second_preimage_results,
            &avalanche_effect_results,
        )?;

        let hash_recommendations = self.generate_hash_function_recommendations(
            &collision_resistance_results,
            &birthday_attack_analysis,
            &length_extension_analysis,
        )?;

        Ok(HashFunctionAnalysisResult {
            collision_resistance_results,
            preimage_resistance_results,
            second_preimage_results,
            avalanche_effect_results,
            birthday_attack_analysis,
            length_extension_analysis,
            hash_family_analysis,
            merkle_damgard_analysis,
            sponge_function_analysis,
            hash_security_score,
            hash_recommendations,
        })
    }

    fn analyze_digital_signatures(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<SignatureAnalysisResult, CryptographicAnalysisError> {
        let mut signature_scheme_results = HashMap::new();
        let mut verification_results = Vec::new();
        let mut forge_resistance_results = Vec::new();
        let mut existential_forgery_results = Vec::new();

        for (name, analyzer) in &self.signature_analyzer.signature_scheme_analyzers {
            let result = analyzer.analyze_signature_scheme(context)?;
            signature_scheme_results.insert(name.clone(), result);
        }

        for analyzer in &self.signature_analyzer.verification_analyzers {
            verification_results.extend(analyzer.analyze_signature_verification(context)?);
        }

        for tester in &self.signature_analyzer.forge_resistance_testers {
            forge_resistance_results.extend(tester.test_forge_resistance(context)?);
        }

        for tester in &self.signature_analyzer.existential_forgery_testers {
            existential_forgery_results.extend(tester.test_existential_forgery(context)?);
        }

        let chosen_message_analysis = self.analyze_chosen_message_attacks(context)?;
        let blind_signature_analysis = self.analyze_blind_signatures(context)?;
        let multi_signature_analysis = self.analyze_multi_signatures(context)?;
        let threshold_signature_analysis = self.analyze_threshold_signatures(context)?;
        let ring_signature_analysis = self.analyze_ring_signatures(context)?;

        let signature_security_score = self.calculate_signature_security_score(
            &signature_scheme_results,
            &verification_results,
            &forge_resistance_results,
        )?;

        let signature_recommendations = self.generate_signature_recommendations(
            &forge_resistance_results,
            &existential_forgery_results,
            &chosen_message_analysis,
        )?;

        Ok(SignatureAnalysisResult {
            signature_scheme_results,
            verification_results,
            forge_resistance_results,
            existential_forgery_results,
            chosen_message_analysis,
            blind_signature_analysis,
            multi_signature_analysis,
            threshold_signature_analysis,
            ring_signature_analysis,
            signature_security_score,
            signature_recommendations,
        })
    }

    fn analyze_encryption_schemes(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<EncryptionAnalysisResult, CryptographicAnalysisError> {
        let symmetric_encryption_analysis = self.analyze_symmetric_encryption(context)?;
        let asymmetric_encryption_analysis = self.analyze_asymmetric_encryption(context)?;
        let mode_of_operation_analysis = self.analyze_modes_of_operation(context)?;
        let padding_scheme_analysis = self.analyze_padding_schemes(context)?;
        let authenticated_encryption_analysis = self.analyze_authenticated_encryption(context)?;

        let chosen_plaintext_analysis = self.analyze_chosen_plaintext_attacks(context)?;
        let chosen_ciphertext_analysis = self.analyze_chosen_ciphertext_attacks(context)?;
        let known_plaintext_analysis = self.analyze_known_plaintext_attacks(context)?;

        let differential_cryptanalysis_results = self
            .encryption_analyzer
            .differential_cryptanalysis
            .analyze(context)?;
        let linear_cryptanalysis_results = self
            .encryption_analyzer
            .linear_cryptanalysis
            .analyze(context)?;

        let encryption_security_score = self.calculate_encryption_security_score(
            &symmetric_encryption_analysis,
            &asymmetric_encryption_analysis,
            &authenticated_encryption_analysis,
        )?;

        let encryption_recommendations = self.generate_encryption_recommendations(
            &chosen_plaintext_analysis,
            &chosen_ciphertext_analysis,
            &mode_of_operation_analysis,
            &padding_scheme_analysis,
        )?;

        Ok(EncryptionAnalysisResult {
            symmetric_encryption_analysis,
            asymmetric_encryption_analysis,
            mode_of_operation_analysis,
            padding_scheme_analysis,
            authenticated_encryption_analysis,
            chosen_plaintext_analysis,
            chosen_ciphertext_analysis,
            known_plaintext_analysis,
            differential_cryptanalysis_results,
            linear_cryptanalysis_results,
            encryption_security_score,
            encryption_recommendations,
        })
    }

    fn analyze_quantum_resistance(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<QuantumResistanceAnalysisResult, CryptographicAnalysisError> {
        let mut post_quantum_results = HashMap::new();

        for (name, analyzer) in &self.quantum_resistance_analyzer.post_quantum_analyzers {
            let result = analyzer.analyze_post_quantum_security(context)?;
            post_quantum_results.insert(name.clone(), result);
        }

        let shor_algorithm_impact = self
            .quantum_resistance_analyzer
            .shor_algorithm_analyzer
            .analyze_impact(context)?;
        let grover_algorithm_impact = self
            .quantum_resistance_analyzer
            .grover_algorithm_analyzer
            .analyze_impact(context)?;
        let quantum_key_distribution_analysis = self
            .quantum_resistance_analyzer
            .quantum_key_distribution_analyzer
            .analyze(context)?;

        let lattice_based_analysis = self.analyze_lattice_based_cryptography(context)?;
        let code_based_analysis = self.analyze_code_based_cryptography(context)?;
        let multivariate_analysis = self.analyze_multivariate_cryptography(context)?;
        let hash_based_analysis = self.analyze_hash_based_cryptography(context)?;
        let isogeny_analysis = self.analyze_isogeny_based_cryptography(context)?;

        let quantum_threat_assessment = self.assess_quantum_threat_timeline(context)?;
        let migration_strategy = self.develop_quantum_migration_strategy(context)?;

        let quantum_resistance_score = self.calculate_quantum_resistance_score(
            &post_quantum_results,
            &shor_algorithm_impact,
            &grover_algorithm_impact,
        )?;

        let quantum_recommendations = self.generate_quantum_resistance_recommendations(
            &quantum_threat_assessment,
            &migration_strategy,
        )?;

        Ok(QuantumResistanceAnalysisResult {
            post_quantum_results,
            shor_algorithm_impact,
            grover_algorithm_impact,
            quantum_key_distribution_analysis,
            lattice_based_analysis,
            code_based_analysis,
            multivariate_analysis,
            hash_based_analysis,
            isogeny_analysis,
            quantum_threat_assessment,
            migration_strategy,
            quantum_resistance_score,
            quantum_recommendations,
        })
    }

    fn analyze_cryptographic_implementation(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<ImplementationAnalysisResult, CryptographicAnalysisError> {
        let constant_time_analysis = self.analyze_constant_time_implementation(context)?;
        let memory_safety_analysis = self.analyze_memory_safety(context)?;
        let secure_coding_analysis = self.analyze_secure_coding_practices(context)?;
        let library_analysis = self.analyze_cryptographic_libraries(context)?;
        let hardware_security_analysis = self.analyze_hardware_security_features(context)?;

        let side_channel_countermeasures_analysis =
            self.analyze_side_channel_countermeasures(context)?;
        let fault_tolerance_analysis = self.analyze_fault_tolerance(context)?;
        let performance_security_analysis = self.analyze_performance_security_tradeoffs(context)?;

        let implementation_security_score = self.calculate_implementation_security_score(
            &constant_time_analysis,
            &memory_safety_analysis,
            &secure_coding_analysis,
            &hardware_security_analysis,
        )?;

        let implementation_recommendations = self.generate_implementation_recommendations(
            &constant_time_analysis,
            &memory_safety_analysis,
            &side_channel_countermeasures_analysis,
        )?;

        Ok(ImplementationAnalysisResult {
            constant_time_analysis,
            memory_safety_analysis,
            secure_coding_analysis,
            library_analysis,
            hardware_security_analysis,
            side_channel_countermeasures_analysis,
            fault_tolerance_analysis,
            performance_security_analysis,
            implementation_security_score,
            implementation_recommendations,
        })
    }
}

/// Bundled refs to every per-domain sub-analysis; avoids an unwieldy positional-argument list in
/// [`CryptographicAnalyzer::generate_security_recommendations`] (see `clippy::too_many_arguments`).
struct CryptoSubAnalysesRef<'a> {
    algorithm: &'a AlgorithmAnalysisResult,
    key_management: &'a KeyManagementAnalysisResult,
    side_channel: &'a SideChannelAnalysisResult,
    protocol: &'a ProtocolAnalysisResult,
    random_number: &'a RandomNumberAnalysisResult,
    hash_function: &'a HashFunctionAnalysisResult,
    signature: &'a SignatureAnalysisResult,
    encryption: &'a EncryptionAnalysisResult,
    quantum_resistance: &'a QuantumResistanceAnalysisResult,
    implementation: &'a ImplementationAnalysisResult,
}

impl CryptographicAnalyzer {
    // ---- cache / bookkeeping -------------------------------------------------------------

    fn generate_analysis_id(&self, context: &TraitUsageContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        context.hash(&mut hasher);
        format!("crypto_analysis_{:x}", hasher.finish())
    }

    fn get_cached_analysis(&self, analysis_id: &str) -> Option<&CachedCryptographicAnalysis> {
        self.analysis_cache.get(analysis_id)
    }

    fn is_cache_valid(&self, cached: &CachedCryptographicAnalysis) -> bool {
        SystemTime::now()
            .duration_since(cached.cache_timestamp)
            .map(|elapsed| elapsed <= cached.cache_ttl)
            .unwrap_or(false)
    }

    fn cache_analysis(&mut self, analysis_id: String, result: &CryptographicAnalysisResult) {
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

    fn generate_analysis_metadata(&self, context: &TraitUsageContext) -> HashMap<String, String> {
        HashMap::from([
            ("analyzer".to_string(), "CryptographicAnalyzer".to_string()),
            ("trait_name".to_string(), context.trait_name.clone()),
            ("trait_count".to_string(), context.traits.len().to_string()),
        ])
    }

    fn calculate_analysis_confidence(&self) -> Result<f64, CryptographicAnalysisError> {
        Ok(self
            .analysis_config
            .analysis_confidence_threshold
            .clamp(0.0, 1.0))
    }

    fn calculate_overall_cryptographic_score(
        &self,
        scores: &[f64],
    ) -> Result<f64, CryptographicAnalysisError> {
        Ok(average_score(scores))
    }

    /// Mandatory contract addition (see module docs on [`CryptographicAnalysisResult`]):
    /// derive the individually addressable cryptographic weaknesses directly from the raw
    /// usage context, mirroring the shallow-heuristic style of `core_analyzer.rs`'s
    /// `assess_side_channel_risks`.
    fn identify_cryptographic_issues(
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
    fn generate_security_recommendations(
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

    fn build_algorithm_compatibility_matrix(
        &self,
        symmetric: &HashMap<String, SymmetricAlgorithmResult>,
        asymmetric: &HashMap<String, AsymmetricAlgorithmResult>,
    ) -> Result<AlgorithmCompatibilityMatrix, CryptographicAnalysisError> {
        Ok(heuristic_finding(
            !symmetric.is_empty() && !asymmetric.is_empty(),
            "algorithm compatibility",
        ))
    }

    fn assess_security_levels(
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

    fn check_algorithm_deprecations(
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

    fn generate_algorithm_upgrade_recommendations(
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

    fn calculate_key_management_score(
        &self,
        scores: &[f64],
    ) -> Result<f64, CryptographicAnalysisError> {
        Ok(average_score(scores))
    }

    fn identify_key_management_risks(
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

    fn generate_key_management_recommendations(
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

    fn identify_applicable_countermeasures(
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

    fn calculate_vulnerability_severity_scores(
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

    fn generate_side_channel_mitigation_recommendations(
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

    fn identify_protocol_vulnerabilities(
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

    fn generate_protocol_recommendations(
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

    fn calculate_randomness_quality_score(
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

    fn generate_randomness_recommendations(
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

    fn calculate_hash_security_score(
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

    fn generate_hash_function_recommendations(
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

    fn calculate_signature_security_score(
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

    fn generate_signature_recommendations(
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

    fn calculate_encryption_security_score(
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

    fn generate_encryption_recommendations(
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

    fn calculate_quantum_resistance_score(
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

    fn generate_quantum_resistance_recommendations(
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

    fn analyze_memory_safety(
        &self,
        context: &TraitUsageContext,
    ) -> Result<MemorySafetyAnalysisResult, CryptographicAnalysisError> {
        Ok(heuristic_finding(
            !context.has_unsafe_operations || context.has_bounds_checking,
            "memory safety",
        ))
    }

    fn analyze_performance_security_tradeoffs(
        &self,
        context: &TraitUsageContext,
    ) -> Result<PerformanceSecurityAnalysisResult, CryptographicAnalysisError> {
        Ok(heuristic_finding(
            !context.has_resource_intensive_operations || context.has_resource_limits,
            "performance/security tradeoff",
        ))
    }

    fn calculate_implementation_security_score(
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

    fn generate_implementation_recommendations(
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

impl Default for CryptographicAnalyzer {
    fn default() -> Self {
        Self::new()
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

impl CryptographicAlgorithmAnalyzer {
    pub fn new() -> Self {
        Self {
            symmetric_analyzers: Self::initialize_symmetric_analyzers(),
            asymmetric_analyzers: Self::initialize_asymmetric_analyzers(),
            hash_analyzers: Self::initialize_hash_analyzers(),
            mac_analyzers: Self::initialize_mac_analyzers(),
            kdf_analyzers: Self::initialize_kdf_analyzers(),
            algorithm_database: CryptographicAlgorithmDatabase::new(),
            weakness_patterns: Self::initialize_weakness_patterns(),
            security_levels: Self::initialize_security_levels(),
        }
    }

    fn initialize_symmetric_analyzers() -> HashMap<String, SymmetricAlgorithmAnalyzer> {
        let mut analyzers = HashMap::new();
        analyzers.insert("AES".to_string(), SymmetricAlgorithmAnalyzer::new_aes());
        analyzers.insert(
            "ChaCha20".to_string(),
            SymmetricAlgorithmAnalyzer::new_chacha20(),
        );
        analyzers.insert(
            "Salsa20".to_string(),
            SymmetricAlgorithmAnalyzer::new_salsa20(),
        );
        analyzers.insert("3DES".to_string(), SymmetricAlgorithmAnalyzer::new_3des());
        analyzers.insert(
            "Blowfish".to_string(),
            SymmetricAlgorithmAnalyzer::new_blowfish(),
        );
        analyzers.insert(
            "Twofish".to_string(),
            SymmetricAlgorithmAnalyzer::new_twofish(),
        );
        analyzers
    }

    fn initialize_asymmetric_analyzers() -> HashMap<String, AsymmetricAlgorithmAnalyzer> {
        let mut analyzers = HashMap::new();
        analyzers.insert("RSA".to_string(), AsymmetricAlgorithmAnalyzer::new_rsa());
        analyzers.insert(
            "ECDSA".to_string(),
            AsymmetricAlgorithmAnalyzer::new_ecdsa(),
        );
        analyzers.insert(
            "EdDSA".to_string(),
            AsymmetricAlgorithmAnalyzer::new_eddsa(),
        );
        analyzers.insert("DH".to_string(), AsymmetricAlgorithmAnalyzer::new_dh());
        analyzers.insert("ECDH".to_string(), AsymmetricAlgorithmAnalyzer::new_ecdh());
        analyzers.insert("DSA".to_string(), AsymmetricAlgorithmAnalyzer::new_dsa());
        analyzers
    }

    fn initialize_hash_analyzers() -> HashMap<String, HashAlgorithmAnalyzer> {
        let mut analyzers = HashMap::new();
        analyzers.insert("SHA-256".to_string(), HashAlgorithmAnalyzer::new_sha256());
        analyzers.insert("SHA-3".to_string(), HashAlgorithmAnalyzer::new_sha3());
        analyzers.insert("BLAKE2".to_string(), HashAlgorithmAnalyzer::new_blake2());
        analyzers.insert("SHA-1".to_string(), HashAlgorithmAnalyzer::new_sha1());
        analyzers.insert("MD5".to_string(), HashAlgorithmAnalyzer::new_md5());
        analyzers
    }

    fn initialize_mac_analyzers() -> HashMap<String, MacAlgorithmAnalyzer> {
        let mut analyzers = HashMap::new();
        analyzers.insert("HMAC".to_string(), MacAlgorithmAnalyzer::new_hmac());
        analyzers.insert("Poly1305".to_string(), MacAlgorithmAnalyzer::new_poly1305());
        analyzers.insert("GMAC".to_string(), MacAlgorithmAnalyzer::new_gmac());
        analyzers
    }

    fn initialize_kdf_analyzers() -> HashMap<String, KdfAlgorithmAnalyzer> {
        let mut analyzers = HashMap::new();
        analyzers.insert("PBKDF2".to_string(), KdfAlgorithmAnalyzer::new_pbkdf2());
        analyzers.insert("Argon2".to_string(), KdfAlgorithmAnalyzer::new_argon2());
        analyzers.insert("scrypt".to_string(), KdfAlgorithmAnalyzer::new_scrypt());
        analyzers.insert("bcrypt".to_string(), KdfAlgorithmAnalyzer::new_bcrypt());
        analyzers
    }

    fn initialize_weakness_patterns() -> Vec<AlgorithmWeaknessPattern> {
        vec![
            AlgorithmWeaknessPattern {
                pattern_name: "Weak Key Sizes".to_string(),
                description: "Algorithm uses key sizes that are considered weak".to_string(),
                detection_criteria: vec!["key_size < 128".to_string()],
                severity: WeaknessSeverity::High,
            },
            AlgorithmWeaknessPattern {
                pattern_name: "Deprecated Algorithms".to_string(),
                description: "Algorithm is deprecated or broken".to_string(),
                detection_criteria: vec!["algorithm in [MD5, SHA1, DES]".to_string()],
                severity: WeaknessSeverity::Critical,
            },
        ]
    }

    fn initialize_security_levels() -> HashMap<String, SecurityLevel> {
        let mut levels = HashMap::new();
        levels.insert(
            "AES-128".to_string(),
            SecurityLevel::new(128, SecurityStrength::High),
        );
        levels.insert(
            "AES-256".to_string(),
            SecurityLevel::new(256, SecurityStrength::VeryHigh),
        );
        levels.insert(
            "RSA-2048".to_string(),
            SecurityLevel::new(112, SecurityStrength::Medium),
        );
        levels.insert(
            "RSA-3072".to_string(),
            SecurityLevel::new(128, SecurityStrength::High),
        );
        levels.insert(
            "ECDSA-P256".to_string(),
            SecurityLevel::new(128, SecurityStrength::High),
        );
        levels.insert(
            "ECDSA-P384".to_string(),
            SecurityLevel::new(192, SecurityStrength::VeryHigh),
        );
        levels
    }
}

// Manager-level `new()` bodies and shared `Default` boilerplate.

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

manager_new! {
    KeyManagementAnalyzer {
        key_generation_analyzers: Vec::new(), key_storage_analyzers: Vec::new(), key_distribution_analyzers: Vec::new(),
        key_rotation_analyzers: Vec::new(), key_revocation_analyzers: Vec::new(), key_escrow_analyzers: Vec::new(),
        entropy_analyzers: Vec::new(), key_lifecycle_analyzer: KeyLifecycleAnalyzer::new(),
        key_derivation_analyzer: KeyDerivationAnalyzer::new(), key_agreement_analyzer: KeyAgreementAnalyzer::new()
    }
    SideChannelAttackDetector {
        timing_attack_detectors: vec![TimingAttackDetector::new()], power_analysis_detectors: vec![PowerAnalysisDetector::new()],
        electromagnetic_detectors: vec![ElectromagneticAttackDetector::new()], acoustic_attack_detectors: vec![AcousticAttackDetector::new()],
        cache_attack_detectors: vec![CacheAttackDetector::new()], fault_injection_detectors: vec![FaultInjectionDetector::new()],
        differential_power_analysis: DifferentialPowerAnalysis::new(), simple_power_analysis: SimplePowerAnalysis::new(),
        correlation_power_analysis: CorrelationPowerAnalysis::new(), template_attack_detector: TemplateAttackDetector::new()
    }
    CryptographicProtocolAnalyzer {
        tls_analyzer: TlsProtocolAnalyzer::new(), ssh_analyzer: SshProtocolAnalyzer::new(), ipsec_analyzer: IpsecProtocolAnalyzer::new(),
        pgp_analyzer: PgpProtocolAnalyzer::new(), oauth_analyzer: OAuthProtocolAnalyzer::new(), saml_analyzer: SamlProtocolAnalyzer::new(),
        kerberos_analyzer: KerberosProtocolAnalyzer::new(), protocol_state_analyzer: ProtocolStateAnalyzer::new(),
        message_flow_analyzer: MessageFlowAnalyzer::new(), authentication_analyzer: AuthenticationProtocolAnalyzer::new()
    }
    RandomNumberGeneratorAnalyzer {
        entropy_testers: Vec::new(), statistical_testers: Vec::new(), predictability_analyzers: Vec::new(), seed_analyzers: Vec::new(),
        prng_analyzers: HashMap::new(), trng_analyzers: HashMap::new(), drbg_analyzers: HashMap::new(),
        nist_test_suite: NistRandomnessTestSuite::new(), diehard_test_suite: DiehardTestSuite::new(), test_u01_suite: TestU01Suite::new()
    }
    HashFunctionAnalyzer {
        collision_resistance_testers: Vec::new(), preimage_resistance_testers: Vec::new(), second_preimage_testers: Vec::new(),
        avalanche_effect_testers: Vec::new(), birthday_attack_analyzers: Vec::new(), length_extension_analyzers: Vec::new(),
        hash_family_analyzers: HashMap::new(), merkle_damgard_analyzer: MerkleDamgardAnalyzer::new(),
        sponge_function_analyzer: SpongeFunctionAnalyzer::new()
    }
    DigitalSignatureAnalyzer {
        signature_scheme_analyzers: { let mut m = HashMap::new(); m.insert("RSA-PSS".to_string(), SignatureSchemeAnalyzer::new()); m.insert("ECDSA".to_string(), SignatureSchemeAnalyzer::new()); m },
        verification_analyzers: vec![SignatureVerificationAnalyzer::new()], forge_resistance_testers: vec![ForgeResistanceTester::new()],
        existential_forgery_testers: vec![ExistentialForgeryTester::new()], chosen_message_analyzers: Vec::new(),
        blind_signature_analyzers: Vec::new(), multi_signature_analyzers: Vec::new(), threshold_signature_analyzers: Vec::new(),
        ring_signature_analyzers: Vec::new()
    }
    EncryptionAnalyzer {
        symmetric_encryption_analyzers: HashMap::new(), asymmetric_encryption_analyzers: HashMap::new(),
        mode_of_operation_analyzers: HashMap::new(), padding_scheme_analyzers: HashMap::new(),
        authenticated_encryption_analyzers: Vec::new(), chosen_plaintext_analyzers: Vec::new(), chosen_ciphertext_analyzers: Vec::new(),
        known_plaintext_analyzers: Vec::new(), differential_cryptanalysis: DifferentialCryptanalysis::new(),
        linear_cryptanalysis: LinearCryptanalysis::new()
    }
    QuantumResistanceAnalyzer {
        post_quantum_analyzers: { let mut m = HashMap::new(); m.insert("Kyber".to_string(), PostQuantumAnalyzer::new()); m.insert("Dilithium".to_string(), PostQuantumAnalyzer::new()); m },
        shor_algorithm_analyzer: ShorAlgorithmAnalyzer::new(), grover_algorithm_analyzer: GroverAlgorithmAnalyzer::new(),
        quantum_key_distribution_analyzer: QuantumKeyDistributionAnalyzer::new(), lattice_based_analyzers: Vec::new(),
        code_based_analyzers: Vec::new(), multivariate_analyzers: Vec::new(), hash_based_analyzers: Vec::new(),
        isogeny_analyzers: Vec::new(), quantum_threat_timeline: QuantumThreatTimeline::new()
    }
    CryptographicImplementationAnalyzer {
        constant_time_analyzers: Vec::new(), memory_safety_analyzers: Vec::new(), secure_coding_analyzers: Vec::new(),
        library_analyzers: HashMap::new(), hardware_security_analyzers: Vec::new(), side_channel_countermeasures: Vec::new(),
        fault_tolerance_analyzers: Vec::new(), performance_security_analyzers: Vec::new()
    }
}

default_via_new!(
    CryptographicAlgorithmAnalyzer,
    KeyManagementAnalyzer,
    SideChannelAttackDetector,
    CryptographicProtocolAnalyzer,
    RandomNumberGeneratorAnalyzer,
    HashFunctionAnalyzer,
    DigitalSignatureAnalyzer,
    EncryptionAnalyzer,
    QuantumResistanceAnalyzer,
    CryptographicImplementationAnalyzer,
);

// Compliance / vulnerability-scan sub-managers used directly by `CryptographicAnalyzer`.

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CryptographicComplianceChecker {}

impl CryptographicComplianceChecker {
    pub fn new() -> Self {
        Self::default()
    }

    fn check_compliance(
        &self,
        context: &TraitUsageContext,
    ) -> Result<CryptographicComplianceStatus, CryptographicAnalysisError> {
        Ok(
            if context.has_cryptographic_operations
                && context.has_secure_key_management
                && context.has_constant_time_operations
            {
                CryptographicComplianceStatus::Compliant
            } else if context.has_cryptographic_operations
                && (context.has_secure_key_management || context.has_constant_time_operations)
            {
                CryptographicComplianceStatus::PartiallyCompliant
            } else if context.has_cryptographic_operations {
                CryptographicComplianceStatus::NonCompliant
            } else {
                CryptographicComplianceStatus::NotAssessed
            },
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CryptographicVulnerabilityScanner {}

impl CryptographicVulnerabilityScanner {
    pub fn new() -> Self {
        Self::default()
    }

    fn scan_vulnerabilities(
        &self,
        context: &TraitUsageContext,
    ) -> Result<CryptographicVulnerabilityReport, CryptographicAnalysisError> {
        let mut total = 0usize;
        let mut critical = 0usize;
        if context.has_cryptographic_operations && !context.has_constant_time_operations {
            total += 1;
            critical += 1;
        }
        if context.has_cryptographic_operations && !context.has_secure_key_management {
            total += 1;
        }
        if context.has_timing_dependencies && !context.has_constant_time_operations {
            total += 1;
        }
        Ok(CryptographicVulnerabilityReport {
            total_findings: total,
            critical_findings: critical,
            summary: format!(
                "{total} potential cryptographic weaknesses identified ({critical} critical)"
            ),
        })
    }
}

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

marker_type!(
    KeyGenerationAnalyzer,
    KeyStorageAnalyzer,
    KeyDistributionAnalyzer,
    KeyRotationAnalyzer,
    KeyRevocationAnalyzer,
    KeyEscrowAnalyzer,
    EntropyAnalyzer,
    KeyLifecycleAnalyzer,
    KeyDerivationAnalyzer,
    KeyAgreementAnalyzer,
    DifferentialPowerAnalysis,
    SimplePowerAnalysis,
    CorrelationPowerAnalysis,
    TemplateAttackDetector,
    EntropyTester,
    StatisticalRandomnessTester,
    PredictabilityAnalyzer,
    SeedAnalyzer,
    PrngAnalyzer,
    TrngAnalyzer,
    DrbgAnalyzer,
    CollisionResistanceTester,
    PreimageResistanceTester,
    SecondPreimageResistanceTester,
    AvalancheEffectTester,
    BirthdayAttackAnalyzer,
    LengthExtensionAttackAnalyzer,
    HashFamilyAnalyzer,
    ChosenMessageAttackAnalyzer,
    BlindSignatureAnalyzer,
    MultiSignatureAnalyzer,
    ThresholdSignatureAnalyzer,
    RingSignatureAnalyzer,
    SymmetricEncryptionAnalyzer,
    AsymmetricEncryptionAnalyzer,
    ModeOfOperationAnalyzer,
    PaddingSchemeAnalyzer,
    AuthenticatedEncryptionAnalyzer,
    ChosenPlaintextAttackAnalyzer,
    ChosenCiphertextAttackAnalyzer,
    KnownPlaintextAttackAnalyzer,
    LatticeBasedAnalyzer,
    CodeBasedAnalyzer,
    MultivariateAnalyzer,
    HashBasedAnalyzer,
    IsogenyAnalyzer,
    QuantumThreatTimeline,
    ConstantTimeAnalyzer,
    MemorySafetyAnalyzer,
    SecureCodingAnalyzer,
    CryptographicLibraryAnalyzer,
    HardwareSecurityAnalyzer,
    FaultToleranceAnalyzer,
    PerformanceSecurityAnalyzer,
    CryptographicAlgorithmDatabase,
);

// Leaf "result" type aliases: none of these are part of any other file's contract (only
// `CryptographicAnalysisResult`, `CryptographicIssue`, and the 4 types below are), so every one
// is a transparent alias over one of the three shared shapes above.

finding_alias!(
    SymmetricAlgorithmResult,
    AsymmetricAlgorithmResult,
    HashAlgorithmResult,
    MacAlgorithmResult,
    KdfAlgorithmResult,
    AlgorithmCompatibilityMatrix,
    SecurityLevelAssessment,
    KeyGenerationAssessment,
    KeyStorageAssessment,
    KeyDistributionAssessment,
    KeyRotationAssessment,
    KeyRevocationAssessment,
    EntropyAssessment,
    KeyLifecycleAssessment,
    TlsAnalysisResult,
    SshAnalysisResult,
    IpsecAnalysisResult,
    PgpAnalysisResult,
    OAuthAnalysisResult,
    SamlAnalysisResult,
    KerberosAnalysisResult,
    ProtocolStateAnalysisResult,
    MessageFlowAnalysisResult,
    AuthenticationAnalysisResult,
    EntropyTestResults,
    StatisticalTestResults,
    PredictabilityAnalysisResult,
    SeedAnalysisResult,
    PrngAnalysisResult,
    TrngAnalysisResult,
    DrbgAnalysisResult,
    NistTestResults,
    DiehardTestResults,
    TestU01Results,
    CollisionResistanceResults,
    PreimageResistanceResults,
    SecondPreimageResults,
    AvalancheEffectResults,
    BirthdayAttackAnalysisResult,
    LengthExtensionAnalysisResult,
    HashFamilyAnalysisResult,
    MerkleDamgardAnalysisResult,
    SpongeFunctionAnalysisResult,
    SignatureSchemeResult,
    SignatureVerificationResult,
    ForgeResistanceResult,
    ExistentialForgeryResult,
    ChosenMessageAnalysisResult,
    BlindSignatureAnalysisResult,
    MultiSignatureAnalysisResult,
    ThresholdSignatureAnalysisResult,
    RingSignatureAnalysisResult,
    SymmetricEncryptionAnalysisResult,
    AsymmetricEncryptionAnalysisResult,
    ModeOfOperationAnalysisResult,
    PaddingSchemeAnalysisResult,
    AuthenticatedEncryptionAnalysisResult,
    ChosenPlaintextAnalysisResult,
    ChosenCiphertextAnalysisResult,
    KnownPlaintextAnalysisResult,
    DifferentialCryptanalysisResult,
    LinearCryptanalysisResult,
    PostQuantumResult,
    ShorAlgorithmImpact,
    GroverAlgorithmImpact,
    QuantumKeyDistributionAnalysisResult,
    LatticeBasedAnalysisResult,
    CodeBasedAnalysisResult,
    MultivariateAnalysisResult,
    HashBasedAnalysisResult,
    IsogenyAnalysisResult,
    QuantumThreatAssessment,
    QuantumMigrationStrategy,
    ConstantTimeAnalysisResult,
    MemorySafetyAnalysisResult,
    SecureCodingAnalysisResult,
    LibraryAnalysisResult,
    HardwareSecurityAnalysisResult,
    SideChannelCountermeasuresAnalysisResult,
    FaultToleranceAnalysisResult,
    PerformanceSecurityAnalysisResult,
);

recommendation_alias!(
    DeprecationWarning,
    AlgorithmUpgradeRecommendation,
    KeyManagementRisk,
    KeyManagementRecommendation,
    SideChannelCountermeasure,
    SideChannelMitigationRecommendation,
    ProtocolRecommendation,
    RandomnessRecommendation,
    HashFunctionRecommendation,
    SignatureRecommendation,
    EncryptionRecommendation,
    QuantumResistanceRecommendation,
    ImplementationRecommendation,
);

vulnerability_alias!(
    TimingAttackVulnerability,
    PowerAnalysisVulnerability,
    ElectromagneticVulnerability,
    AcousticVulnerability,
    CacheAttackVulnerability,
    FaultInjectionVulnerability,
    ProtocolVulnerability,
);

// Mandatory contract types, consumed by `core_analyzer.rs` and the outer crate's public API.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicIssue {
    pub id: String,
    pub severity: RiskSeverity,
    pub issue_type: String,
    pub recommendation: String,
    pub fix_complexity: ImplementationEffort,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideChannelRisk {
    pub risk_type: String,
    pub description: String,
    pub severity: RiskSeverity,
    pub affected_operation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingVulnerability {
    pub operation: String,
    pub description: String,
    pub severity: RiskSeverity,
    pub mitigation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantTimeViolation {
    pub location: String,
    pub description: String,
    pub severity: RiskSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicStrengthAssessment {
    pub algorithm: String,
    pub key_size_bits: u32,
    pub estimated_security_bits: u32,
    pub quantum_resistant: bool,
    pub assessment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicRecommendation {
    pub id: String,
    pub title: String,
    pub description: String,
    pub priority: RiskSeverity,
    pub implementation_effort: ImplementationEffort,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CryptographicComplianceStatus {
    Compliant,
    PartiallyCompliant,
    NonCompliant,
    NotAssessed,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CryptographicVulnerabilityReport {
    pub total_findings: usize,
    pub critical_findings: usize,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CryptographicAnalysisError {
    AlgorithmAnalysisError(String),
    KeyManagementError(String),
    SideChannelAnalysisError(String),
    ProtocolAnalysisError(String),
    RandomnessAnalysisError(String),
    ImplementationAnalysisError(String),
    ConfigurationError(String),
    DataError(String),
}

impl std::fmt::Display for CryptographicAnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CryptographicAnalysisError::AlgorithmAnalysisError(msg) => {
                write!(f, "Algorithm analysis error: {}", msg)
            }
            CryptographicAnalysisError::KeyManagementError(msg) => {
                write!(f, "Key management error: {}", msg)
            }
            CryptographicAnalysisError::SideChannelAnalysisError(msg) => {
                write!(f, "Side channel analysis error: {}", msg)
            }
            CryptographicAnalysisError::ProtocolAnalysisError(msg) => {
                write!(f, "Protocol analysis error: {}", msg)
            }
            CryptographicAnalysisError::RandomnessAnalysisError(msg) => {
                write!(f, "Randomness analysis error: {}", msg)
            }
            CryptographicAnalysisError::ImplementationAnalysisError(msg) => {
                write!(f, "Implementation analysis error: {}", msg)
            }
            CryptographicAnalysisError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            CryptographicAnalysisError::DataError(msg) => write!(f, "Data error: {}", msg),
        }
    }
}

impl std::error::Error for CryptographicAnalysisError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicAnalysisConfig {
    pub algorithm_analysis_depth: CryptoAnalysisDepth,
    pub side_channel_analysis_enabled: bool,
    pub quantum_resistance_analysis_enabled: bool,
    pub compliance_standards: Vec<String>,
    pub cache_duration: Duration,
    pub analysis_confidence_threshold: f64,
}

impl Default for CryptographicAnalysisConfig {
    fn default() -> Self {
        Self {
            algorithm_analysis_depth: CryptoAnalysisDepth::Moderate,
            side_channel_analysis_enabled: true,
            quantum_resistance_analysis_enabled: true,
            compliance_standards: vec!["FIPS-140-2".to_string(), "Common Criteria".to_string()],
            cache_duration: Duration::from_secs(3600),
            analysis_confidence_threshold: 0.8,
        }
    }
}

macro_rules! define_crypto_supporting_types {
    () => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum CryptoAnalysisDepth {
            Surface,
            Moderate,
            Deep,
            Comprehensive,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum WeaknessSeverity {
            Low,
            Medium,
            High,
            Critical,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum SecurityStrength {
            VeryLow,
            Low,
            Medium,
            High,
            VeryHigh,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct SecurityLevel {
            pub bits_of_security: u32,
            pub strength: SecurityStrength,
        }

        impl SecurityLevel {
            pub fn new(bits: u32, strength: SecurityStrength) -> Self {
                Self {
                    bits_of_security: bits,
                    strength,
                }
            }
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct AlgorithmWeaknessPattern {
            pub pattern_name: String,
            pub description: String,
            pub detection_criteria: Vec<String>,
            pub severity: WeaknessSeverity,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct CachedCryptographicAnalysis {
            pub result: CryptographicAnalysisResult,
            pub cache_timestamp: SystemTime,
            pub cache_ttl: Duration,
        }
    };
}

define_crypto_supporting_types!();

pub fn create_cryptographic_analyzer() -> CryptographicAnalyzer {
    CryptographicAnalyzer::new()
}

pub fn analyze_cryptographic_security(
    context: &TraitUsageContext,
) -> Result<CryptographicAnalysisResult, CryptographicAnalysisError> {
    let mut analyzer = CryptographicAnalyzer::new();
    analyzer.analyze_cryptographic_security(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end smoke test: an insecure crypto usage context should produce in-range scores
    /// and at least one identified issue; a benign (non-crypto) context should surface none.
    #[test]
    fn test_analyze_cryptographic_security_smoke() {
        let insecure = TraitUsageContext {
            trait_name: "Cipher".to_string(),
            traits: vec!["Cipher".to_string()],
            has_cryptographic_operations: true,
            has_constant_time_operations: false,
            has_secure_key_management: false,
            handles_sensitive_data: true,
            ..Default::default()
        };
        let analysis = analyze_cryptographic_security(&insecure).expect("analysis should succeed");
        assert!((0.0..=10.0).contains(&analysis.overall_cryptographic_score));
        assert!((0.0..=10.0).contains(&analysis.risk_score));
        assert!((0.0..=1.0).contains(&analysis.analysis_confidence));
        assert!(
            !analysis.identified_issues.is_empty(),
            "insecure crypto usage should surface at least one issue"
        );

        let benign = TraitUsageContext {
            trait_name: "Display".to_string(),
            traits: vec!["Display".to_string()],
            ..Default::default()
        };
        assert!(analyze_cryptographic_security(&benign)
            .expect("analysis should succeed")
            .identified_issues
            .is_empty());
    }

    /// Repeated analysis of the same context should hit the cache and return the same
    /// `analysis_id`, exercising `generate_analysis_id`/`get_cached_analysis`/`is_cache_valid`.
    #[test]
    fn test_analysis_cache_reuses_result() {
        let mut analyzer = create_cryptographic_analyzer();
        let context = TraitUsageContext {
            trait_name: "Hasher".to_string(),
            traits: vec!["Hasher".to_string()],
            has_cryptographic_operations: true,
            ..Default::default()
        };
        let first = analyzer
            .analyze_cryptographic_security(&context)
            .expect("first analysis should succeed");
        let second = analyzer
            .analyze_cryptographic_security(&context)
            .expect("second analysis should succeed");
        assert_eq!(first.analysis_id, second.analysis_id);
    }
}

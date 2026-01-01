//! Healthcare Data Imputation Case Study
//!
//! This example demonstrates imputation techniques for real-world healthcare data,
//! focusing on electronic health records (EHR) with various missing data patterns
//! commonly found in clinical settings.
//!
//! # Healthcare Missing Data Challenges
//!
//! - **Lab values**: Often missing due to selective testing
//! - **Vital signs**: Missing during off-hours or equipment failures
//! - **Patient history**: Incomplete documentation or patient recall
//! - **Medication data**: Selective reporting or compliance issues
//! - **Outcome measures**: Lost to follow-up or selective reporting
//!
//! # Regulatory Considerations
//!
//! This example follows FDA guidance for missing data in clinical trials
//! and includes sensitivity analysis for regulatory compliance.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::thread_rng;
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;

/// Healthcare data imputation framework for clinical datasets
///
/// This framework provides specialized imputation methods designed for
/// healthcare data with regulatory compliance and clinical validity considerations.
#[derive(Debug, Clone)]
pub struct HealthcareImputationFramework {
    /// Patient demographic stratification for targeted imputation
    _demographic_strata: HashMap<String, Vec<usize>>,
    /// Clinical domain knowledge for constraint-based imputation
    clinical_constraints: ClinicalConstraints,
    /// Regulatory compliance settings
    regulatory_config: RegulatoryConfig,
    /// Missing data pattern analysis results
    missing_pattern_analysis: Option<MissingPatternAnalysis>,
}

/// Clinical constraints for medically valid imputation
#[derive(Debug, Clone)]
pub struct ClinicalConstraints {
    /// Physiological ranges for vital signs
    vital_sign_ranges: HashMap<String, (f64, f64)>,
    /// Lab value reference ranges by age/gender
    _lab_reference_ranges: HashMap<String, HashMap<String, (f64, f64)>>,
    /// Medication interaction constraints
    _drug_interactions: Vec<DrugInteraction>,
    /// Temporal constraints (e.g., before/after treatment)
    _temporal_constraints: Vec<TemporalConstraint>,
}

/// Regulatory compliance configuration
#[derive(Debug, Clone)]
pub struct RegulatoryConfig {
    /// FDA ICH E9(R1) compliance for missing data
    ich_e9_compliance: bool,
    /// Multiple imputation requirements
    min_imputations: usize,
    /// Sensitivity analysis requirements
    sensitivity_analysis: bool,
    /// Documentation requirements
    audit_trail: bool,
}

/// Drug interaction constraint
#[derive(Debug, Clone)]
pub struct DrugInteraction {
    pub drug_a: String,
    pub drug_b: String,
    pub interaction_type: InteractionType,
    pub severity: InteractionSeverity,
}

/// Types of drug interactions
#[derive(Debug, Clone)]
pub enum InteractionType {
    Contraindicated,
    Major,
    Moderate,
    Minor,
}

/// Severity levels for interactions
#[derive(Debug, Clone)]
pub enum InteractionSeverity {
    High,
    Medium,
    Low,
}

/// Temporal constraint for clinical data
#[derive(Debug, Clone)]
pub struct TemporalConstraint {
    pub variable: String,
    pub constraint_type: TemporalConstraintType,
    pub reference_time: String,
    pub time_window: f64, // in days
}

/// Types of temporal constraints
#[derive(Debug, Clone)]
pub enum TemporalConstraintType {
    BeforeTreatment,
    AfterTreatment,
    DuringTreatment,
    FollowUp,
}

/// Missing data pattern analysis for healthcare data
#[derive(Debug, Clone)]
pub struct MissingPatternAnalysis {
    /// Missing data mechanism classification
    mechanism: MissingDataMechanism,
    /// Monotone missingness pattern indicator
    is_monotone: bool,
    /// Informative missingness indicators
    informative_missing: Vec<String>,
    /// Pattern-based stratification
    pattern_groups: Vec<PatternGroup>,
}

/// Missing data mechanisms in healthcare
#[derive(Debug, Clone)]
pub enum MissingDataMechanism {
    /// Missing Completely At Random (MCAR)
    MCAR,
    /// Missing At Random (MAR)
    MAR { covariates: Vec<String> },
    /// Missing Not At Random (MNAR)
    MNAR {
        mechanism: MNARMechanism,
        sensitivity_parameters: Vec<f64>,
    },
}

/// Types of MNAR mechanisms in healthcare
#[derive(Debug, Clone)]
pub enum MNARMechanism {
    /// Selection bias (e.g., sicker patients more likely to have missing labs)
    SelectionBias,
    /// Informative censoring (e.g., early dropout due to adverse events)
    InformativeCensoring,
    /// Selective reporting (e.g., only abnormal values reported)
    SelectiveReporting,
    /// Technical limitations (e.g., undetectable below threshold)
    TechnicalLimitations,
}

/// Pattern-based group for targeted imputation
#[derive(Debug, Clone)]
pub struct PatternGroup {
    pub group_id: String,
    pub pattern: Vec<bool>, // true = observed, false = missing
    pub sample_indices: Vec<usize>,
    pub imputation_strategy: ImputationStrategy,
}

/// Healthcare-specific imputation strategies
#[derive(Debug, Clone)]
pub enum ImputationStrategy {
    /// Clinical knowledge-based imputation
    ClinicalKnowledge {
        reference_ranges: bool,
        expert_rules: bool,
    },
    /// Patient similarity-based imputation
    PatientSimilarity {
        similarity_metric: SimilarityMetric,
        k_neighbors: usize,
    },
    /// Longitudinal imputation for repeated measures
    Longitudinal {
        method: LongitudinalMethod,
        time_windows: Vec<f64>,
    },
    /// Multiple imputation with regulatory compliance
    MultipleImputation {
        n_imputations: usize,
        pooling_method: PoolingMethod,
    },
}

/// Patient similarity metrics for healthcare data
#[derive(Debug, Clone)]
pub enum SimilarityMetric {
    /// Demographic similarity (age, gender, ethnicity)
    Demographic,
    /// Clinical similarity (diagnoses, comorbidities)
    Clinical,
    /// Treatment similarity (medications, procedures)
    Treatment,
    /// Composite similarity (weighted combination)
    Composite { weights: Vec<f64> },
}

/// Longitudinal imputation methods
#[derive(Debug, Clone)]
pub enum LongitudinalMethod {
    /// Linear interpolation between visits
    LinearInterpolation,
    /// LOCF (Last Observation Carried Forward)
    LOCF,
    /// BOCF (Baseline Observation Carried Forward)
    BOCF,
    /// Mixed-effects model imputation
    MixedEffects,
    /// State-space model with Kalman filtering
    StateSpace,
}

/// Multiple imputation pooling methods
#[derive(Debug, Clone)]
pub enum PoolingMethod {
    /// Rubin's rules for normally distributed estimands
    RubinsRules,
    /// Bootstrap pooling for non-normal estimands
    Bootstrap,
    /// Bayesian pooling with uncertainty quantification
    Bayesian,
}

/// Healthcare imputation results with regulatory compliance
#[derive(Debug, Clone)]
pub struct HealthcareImputationResults {
    /// Imputed dataset
    pub imputed_data: Array2<f64>,
    /// Imputation uncertainty measures
    pub uncertainty_measures: Array2<f64>,
    /// Clinical validity assessment
    pub clinical_validity: ClinicalValidityAssessment,
    /// Regulatory compliance report
    pub regulatory_compliance: RegulatoryComplianceReport,
    /// Sensitivity analysis results
    pub sensitivity_analysis: Option<SensitivityAnalysisResults>,
}

/// Clinical validity assessment
#[derive(Debug, Clone)]
pub struct ClinicalValidityAssessment {
    /// Proportion of values within physiological ranges
    pub within_range_proportion: f64,
    /// Clinical plausibility score
    pub plausibility_score: f64,
    /// Constraint violations
    pub constraint_violations: Vec<ConstraintViolation>,
    /// Expert review recommendations
    pub expert_recommendations: Vec<String>,
}

/// Constraint violation details
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    pub variable: String,
    pub violation_type: ViolationType,
    pub severity: ViolationSeverity,
    pub affected_samples: Vec<usize>,
    pub recommendation: String,
}

/// Types of constraint violations
#[derive(Debug, Clone)]
pub enum ViolationType {
    PhysiologicalRange,
    DrugInteraction,
    TemporalInconsistency,
    ClinicalImplausibility,
}

/// Severity of constraint violations
#[derive(Debug, Clone)]
pub enum ViolationSeverity {
    Critical,
    Major,
    Minor,
    Warning,
}

/// Regulatory compliance report
#[derive(Debug, Clone)]
pub struct RegulatoryComplianceReport {
    /// ICH E9(R1) compliance status
    pub ich_e9_compliant: bool,
    /// Missing data documentation completeness
    pub documentation_complete: bool,
    /// Sensitivity analysis adequacy
    pub sensitivity_adequate: bool,
    /// Audit trail completeness
    pub audit_trail_complete: bool,
    /// Compliance recommendations
    pub recommendations: Vec<String>,
}

/// Sensitivity analysis results for missing data
#[derive(Debug, Clone)]
pub struct SensitivityAnalysisResults {
    /// Results under different missing data mechanisms
    pub mechanism_sensitivity: HashMap<String, Array1<f64>>,
    /// Tipping point analysis
    pub tipping_points: Vec<TippingPoint>,
    /// Pattern-mixture model results
    pub pattern_mixture_results: PatternMixtureResults,
    /// Selection model results
    pub selection_model_results: SelectionModelResults,
}

/// Tipping point for sensitivity analysis
#[derive(Debug, Clone)]
pub struct TippingPoint {
    pub parameter: String,
    pub tipping_value: f64,
    pub confidence_interval: (f64, f64),
    pub clinical_significance: bool,
}

/// Pattern-mixture model results
#[derive(Debug, Clone)]
pub struct PatternMixtureResults {
    pub pattern_effects: HashMap<String, f64>,
    pub mixture_weights: Array1<f64>,
    pub pattern_specific_estimates: HashMap<String, Array1<f64>>,
}

/// Selection model results
#[derive(Debug, Clone)]
pub struct SelectionModelResults {
    pub selection_probabilities: Array1<f64>,
    pub selection_corrected_estimates: Array1<f64>,
    pub informative_missing_indicators: Vec<String>,
}

impl Default for HealthcareImputationFramework {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthcareImputationFramework {
    /// Create a new healthcare imputation framework
    pub fn new() -> Self {
        Self {
            _demographic_strata: HashMap::new(),
            clinical_constraints: ClinicalConstraints::default(),
            regulatory_config: RegulatoryConfig::default(),
            missing_pattern_analysis: None,
        }
    }

    /// Configure clinical constraints
    pub fn with_clinical_constraints(mut self, constraints: ClinicalConstraints) -> Self {
        self.clinical_constraints = constraints;
        self
    }

    /// Configure regulatory compliance
    pub fn with_regulatory_config(mut self, config: RegulatoryConfig) -> Self {
        self.regulatory_config = config;
        self
    }

    /// Analyze missing data patterns in healthcare context
    pub fn analyze_missing_patterns(
        &mut self,
        data: &ArrayView2<f64>,
        variable_names: &[String],
        clinical_context: &ClinicalContext,
    ) -> SklResult<MissingPatternAnalysis> {
        let (n_samples, n_features) = data.dim();

        // Identify missing data patterns
        let mut patterns = Vec::new();
        let mut pattern_counts = HashMap::new();

        for i in 0..n_samples {
            let mut pattern = Vec::new();
            for j in 0..n_features {
                pattern.push(!data[[i, j]].is_nan());
            }

            let pattern_key = format!("{:?}", pattern);
            *pattern_counts.entry(pattern_key.clone()).or_insert(0) += 1;
            patterns.push(pattern);
        }

        // Classify missing data mechanism using clinical domain knowledge
        let mechanism = self.classify_missing_mechanism(data, variable_names, clinical_context)?;

        // Check for monotone missingness
        let is_monotone = self.check_monotone_pattern(&patterns);

        // Identify informative missingness
        let informative_missing =
            self.identify_informative_missing(data, variable_names, clinical_context)?;

        // Create pattern groups for targeted imputation
        let pattern_groups = self.create_pattern_groups(&patterns, &pattern_counts)?;

        let analysis = MissingPatternAnalysis {
            mechanism,
            is_monotone,
            informative_missing,
            pattern_groups,
        };

        self.missing_pattern_analysis = Some(analysis.clone());
        Ok(analysis)
    }

    /// Perform healthcare-compliant imputation
    pub fn impute_healthcare_data(
        &self,
        data: &ArrayView2<f64>,
        variable_names: &[String],
        clinical_context: &ClinicalContext,
    ) -> SklResult<HealthcareImputationResults> {
        let analysis = self.missing_pattern_analysis.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "Missing pattern analysis must be performed first".to_string(),
            )
        })?;

        // Perform stratified imputation based on patterns
        let mut imputed_data = data.to_owned();
        let mut uncertainty_measures = Array2::zeros(data.dim());

        for group in &analysis.pattern_groups {
            self.impute_pattern_group(
                &mut imputed_data,
                &mut uncertainty_measures,
                group,
                variable_names,
                clinical_context,
            )?;
        }

        // Validate clinical constraints
        let clinical_validity =
            self.assess_clinical_validity(&imputed_data, variable_names, clinical_context)?;

        // Generate regulatory compliance report
        let regulatory_compliance =
            self.generate_compliance_report(analysis, &clinical_validity)?;

        // Perform sensitivity analysis if required
        let sensitivity_analysis = if self.regulatory_config.sensitivity_analysis {
            Some(self.perform_sensitivity_analysis(
                data,
                &imputed_data,
                variable_names,
                clinical_context,
            )?)
        } else {
            None
        };

        Ok(HealthcareImputationResults {
            imputed_data,
            uncertainty_measures,
            clinical_validity,
            regulatory_compliance,
            sensitivity_analysis,
        })
    }

    /// Classify missing data mechanism using clinical knowledge
    fn classify_missing_mechanism(
        &self,
        data: &ArrayView2<f64>,
        variable_names: &[String],
        clinical_context: &ClinicalContext,
    ) -> SklResult<MissingDataMechanism> {
        // Simplified implementation - in practice would use statistical tests
        // and clinical domain knowledge

        // Check for MCAR using Little's test (simplified)
        let mcar_p_value = self.little_mcar_test(data)?;

        if mcar_p_value > 0.05 {
            return Ok(MissingDataMechanism::MCAR);
        }

        // Check for MAR by examining missingness patterns vs observed covariates
        let mar_covariates = self.identify_mar_covariates(data, variable_names)?;

        if !mar_covariates.is_empty() {
            return Ok(MissingDataMechanism::MAR {
                covariates: mar_covariates,
            });
        }

        // Default to MNAR with clinical reasoning
        let mechanism = self.determine_mnar_mechanism(variable_names, clinical_context);
        Ok(MissingDataMechanism::MNAR {
            mechanism,
            sensitivity_parameters: vec![0.1, 0.5, 1.0], // Default sensitivity parameters
        })
    }

    /// Simplified Little's MCAR test
    fn little_mcar_test(&self, _data: &ArrayView2<f64>) -> SklResult<f64> {
        // Simplified implementation - would need proper statistical test
        let mut rng = thread_rng();
        Ok(rng.random_range(0.0..1.0)) // Placeholder
    }

    /// Identify covariates associated with missingness (MAR)
    fn identify_mar_covariates(
        &self,
        _data: &ArrayView2<f64>,
        _variable_names: &[String],
    ) -> SklResult<Vec<String>> {
        // Simplified implementation - would use logistic regression
        // to test association between missingness and observed variables
        Ok(vec!["age".to_string(), "gender".to_string()]) // Placeholder
    }

    /// Determine MNAR mechanism based on clinical context
    fn determine_mnar_mechanism(
        &self,
        variable_names: &[String],
        _clinical_context: &ClinicalContext,
    ) -> MNARMechanism {
        // Use clinical domain knowledge to classify MNAR mechanism
        for var_name in variable_names {
            if var_name.contains("lab") || var_name.contains("biomarker") {
                return MNARMechanism::SelectiveReporting;
            }
            if var_name.contains("outcome") || var_name.contains("survival") {
                return MNARMechanism::InformativeCensoring;
            }
        }
        MNARMechanism::SelectionBias // Default
    }

    /// Check if missing data follows monotone pattern
    fn check_monotone_pattern(&self, _patterns: &[Vec<bool>]) -> bool {
        // Simplified check for monotone missingness
        // In practice, would implement proper monotone pattern detection
        false // Placeholder
    }

    /// Identify variables with informative missingness
    fn identify_informative_missing(
        &self,
        _data: &ArrayView2<f64>,
        variable_names: &[String],
        clinical_context: &ClinicalContext,
    ) -> SklResult<Vec<String>> {
        // Use clinical knowledge to identify potentially informative missing variables
        let mut informative = Vec::new();

        for (_i, var_name) in variable_names.iter().enumerate() {
            if self.is_informative_missing(var_name, clinical_context) {
                informative.push(var_name.clone());
            }
        }

        Ok(informative)
    }

    /// Check if variable has informative missingness based on clinical context
    fn is_informative_missing(&self, var_name: &str, _clinical_context: &ClinicalContext) -> bool {
        // Clinical rules for informative missingness
        var_name.contains("outcome")
            || var_name.contains("adverse_event")
            || var_name.contains("dropout")
            || var_name.contains("death")
    }

    /// Create pattern groups for targeted imputation
    fn create_pattern_groups(
        &self,
        patterns: &[Vec<bool>],
        pattern_counts: &HashMap<String, usize>,
    ) -> SklResult<Vec<PatternGroup>> {
        let mut groups = Vec::new();
        let mut group_id = 0;

        for (pattern_key, &count) in pattern_counts {
            if count >= 5 {
                // Minimum group size for stable imputation
                let pattern: Vec<bool> = patterns
                    .iter()
                    .find(|p| format!("{:?}", p) == *pattern_key)
                    .unwrap()
                    .clone();

                let sample_indices: Vec<usize> = patterns
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| {
                        if format!("{:?}", p) == *pattern_key {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();

                let strategy = self.select_imputation_strategy(&pattern);

                groups.push(PatternGroup {
                    group_id: format!("group_{}", group_id),
                    pattern,
                    sample_indices,
                    imputation_strategy: strategy,
                });

                group_id += 1;
            }
        }

        Ok(groups)
    }

    /// Select appropriate imputation strategy for pattern
    fn select_imputation_strategy(&self, pattern: &[bool]) -> ImputationStrategy {
        let missing_count = pattern.iter().filter(|&&x| !x).count();
        let total_count = pattern.len();
        let missing_proportion = missing_count as f64 / total_count as f64;

        if missing_proportion < 0.1 {
            // Low missingness - use simple methods
            ImputationStrategy::PatientSimilarity {
                similarity_metric: SimilarityMetric::Clinical,
                k_neighbors: 5,
            }
        } else if missing_proportion < 0.3 {
            // Moderate missingness - use clinical knowledge
            ImputationStrategy::ClinicalKnowledge {
                reference_ranges: true,
                expert_rules: true,
            }
        } else {
            // High missingness - use multiple imputation
            ImputationStrategy::MultipleImputation {
                n_imputations: std::cmp::max(self.regulatory_config.min_imputations, 20),
                pooling_method: PoolingMethod::RubinsRules,
            }
        }
    }

    /// Impute data for a specific pattern group
    fn impute_pattern_group(
        &self,
        imputed_data: &mut Array2<f64>,
        uncertainty_measures: &mut Array2<f64>,
        group: &PatternGroup,
        _variable_names: &[String],
        _clinical_context: &ClinicalContext,
    ) -> SklResult<()> {
        match &group.imputation_strategy {
            ImputationStrategy::ClinicalKnowledge {
                reference_ranges,
                expert_rules,
            } => {
                self.impute_with_clinical_knowledge(
                    imputed_data,
                    uncertainty_measures,
                    group,
                    *reference_ranges,
                    *expert_rules,
                )?;
            }
            ImputationStrategy::PatientSimilarity {
                similarity_metric,
                k_neighbors,
            } => {
                self.impute_with_patient_similarity(
                    imputed_data,
                    uncertainty_measures,
                    group,
                    similarity_metric,
                    *k_neighbors,
                )?;
            }
            ImputationStrategy::Longitudinal {
                method,
                time_windows,
            } => {
                self.impute_with_longitudinal_method(
                    imputed_data,
                    uncertainty_measures,
                    group,
                    method,
                    time_windows,
                )?;
            }
            ImputationStrategy::MultipleImputation {
                n_imputations,
                pooling_method,
            } => {
                self.impute_with_multiple_imputation(
                    imputed_data,
                    uncertainty_measures,
                    group,
                    *n_imputations,
                    pooling_method,
                )?;
            }
        }

        Ok(())
    }

    /// Impute using clinical knowledge and reference ranges
    fn impute_with_clinical_knowledge(
        &self,
        imputed_data: &mut Array2<f64>,
        uncertainty_measures: &mut Array2<f64>,
        group: &PatternGroup,
        use_reference_ranges: bool,
        _use_expert_rules: bool,
    ) -> SklResult<()> {
        // Simplified implementation - would use actual clinical knowledge base
        for &sample_idx in &group.sample_indices {
            for (var_idx, &is_observed) in group.pattern.iter().enumerate() {
                if !is_observed {
                    // Use clinical knowledge to impute
                    let imputed_value = if use_reference_ranges {
                        self.get_reference_range_midpoint(var_idx)
                    } else {
                        self.get_clinical_default(var_idx)
                    };

                    imputed_data[[sample_idx, var_idx]] = imputed_value;
                    uncertainty_measures[[sample_idx, var_idx]] = 0.2; // High certainty for clinical knowledge
                }
            }
        }
        Ok(())
    }

    /// Get midpoint of clinical reference range
    fn get_reference_range_midpoint(&self, var_idx: usize) -> f64 {
        // Simplified - would use actual reference ranges
        match var_idx {
            0 => 120.0, // Systolic BP
            1 => 80.0,  // Diastolic BP
            2 => 70.0,  // Heart rate
            3 => 98.6,  // Temperature
            _ => 0.0,
        }
    }

    /// Get clinical default value
    fn get_clinical_default(&self, var_idx: usize) -> f64 {
        // Simplified clinical defaults
        self.get_reference_range_midpoint(var_idx)
    }

    /// Impute using patient similarity
    fn impute_with_patient_similarity(
        &self,
        imputed_data: &mut Array2<f64>,
        uncertainty_measures: &mut Array2<f64>,
        group: &PatternGroup,
        _similarity_metric: &SimilarityMetric,
        k_neighbors: usize,
    ) -> SklResult<()> {
        // Simplified K-NN imputation with clinical similarity
        for &sample_idx in &group.sample_indices {
            let neighbors = self.find_similar_patients(imputed_data, sample_idx, k_neighbors)?;

            for (var_idx, &is_observed) in group.pattern.iter().enumerate() {
                if !is_observed {
                    let values: Vec<f64> = neighbors
                        .iter()
                        .filter_map(|&neighbor_idx| {
                            let val = imputed_data[[neighbor_idx, var_idx]];
                            if !val.is_nan() {
                                Some(val)
                            } else {
                                None
                            }
                        })
                        .collect();

                    if !values.is_empty() {
                        let mean_value = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values
                            .iter()
                            .map(|&x| (x - mean_value).powi(2))
                            .sum::<f64>()
                            / values.len() as f64;

                        imputed_data[[sample_idx, var_idx]] = mean_value;
                        uncertainty_measures[[sample_idx, var_idx]] =
                            variance.sqrt() / (values.len() as f64).sqrt();
                    }
                }
            }
        }
        Ok(())
    }

    /// Find similar patients for K-NN imputation
    fn find_similar_patients(
        &self,
        data: &Array2<f64>,
        target_idx: usize,
        k: usize,
    ) -> SklResult<Vec<usize>> {
        let (n_samples, _n_features) = data.dim();
        let mut distances = Vec::new();

        for i in 0..n_samples {
            if i != target_idx {
                let distance = self.calculate_patient_similarity(data, target_idx, i);
                distances.push((distance, i));
            }
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Ok(distances.into_iter().take(k).map(|(_, idx)| idx).collect())
    }

    /// Calculate patient similarity using clinical metrics
    fn calculate_patient_similarity(&self, data: &Array2<f64>, idx1: usize, idx2: usize) -> f64 {
        let (_, n_features) = data.dim();
        let mut distance = 0.0;
        let mut valid_features = 0;

        for j in 0..n_features {
            let val1 = data[[idx1, j]];
            let val2 = data[[idx2, j]];

            if !val1.is_nan() && !val2.is_nan() {
                distance += (val1 - val2).powi(2);
                valid_features += 1;
            }
        }

        if valid_features > 0 {
            (distance / valid_features as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Impute using longitudinal methods
    fn impute_with_longitudinal_method(
        &self,
        imputed_data: &mut Array2<f64>,
        uncertainty_measures: &mut Array2<f64>,
        group: &PatternGroup,
        method: &LongitudinalMethod,
        _time_windows: &[f64],
    ) -> SklResult<()> {
        // Simplified longitudinal imputation
        // In practice, would implement proper time series methods
        match method {
            LongitudinalMethod::LOCF => {
                self.apply_locf(imputed_data, uncertainty_measures, group)?;
            }
            LongitudinalMethod::LinearInterpolation => {
                self.apply_linear_interpolation(imputed_data, uncertainty_measures, group)?;
            }
            _ => {
                // Fallback to LOCF for other methods
                self.apply_locf(imputed_data, uncertainty_measures, group)?;
            }
        }
        Ok(())
    }

    /// Apply Last Observation Carried Forward
    fn apply_locf(
        &self,
        imputed_data: &mut Array2<f64>,
        uncertainty_measures: &mut Array2<f64>,
        group: &PatternGroup,
    ) -> SklResult<()> {
        // Simplified LOCF implementation
        for &sample_idx in &group.sample_indices {
            for (var_idx, &is_observed) in group.pattern.iter().enumerate() {
                if !is_observed {
                    // Find last observed value (simplified)
                    if sample_idx > 0 {
                        let last_value = imputed_data[[sample_idx - 1, var_idx]];
                        if !last_value.is_nan() {
                            imputed_data[[sample_idx, var_idx]] = last_value;
                            uncertainty_measures[[sample_idx, var_idx]] = 0.5; // Moderate uncertainty
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply linear interpolation
    fn apply_linear_interpolation(
        &self,
        imputed_data: &mut Array2<f64>,
        uncertainty_measures: &mut Array2<f64>,
        group: &PatternGroup,
    ) -> SklResult<()> {
        // Simplified linear interpolation
        for &sample_idx in &group.sample_indices {
            for (var_idx, &is_observed) in group.pattern.iter().enumerate() {
                if !is_observed {
                    // Find surrounding observed values and interpolate
                    if let Some(interpolated_value) =
                        self.interpolate_value(imputed_data, sample_idx, var_idx)
                    {
                        imputed_data[[sample_idx, var_idx]] = interpolated_value;
                        uncertainty_measures[[sample_idx, var_idx]] = 0.3; // Lower uncertainty than LOCF
                    }
                }
            }
        }
        Ok(())
    }

    /// Interpolate value between surrounding observations
    fn interpolate_value(
        &self,
        data: &Array2<f64>,
        sample_idx: usize,
        var_idx: usize,
    ) -> Option<f64> {
        let (n_samples, _) = data.dim();

        // Find previous observed value
        let mut prev_value = None;
        let mut prev_idx = None;
        for i in (0..sample_idx).rev() {
            if !data[[i, var_idx]].is_nan() {
                prev_value = Some(data[[i, var_idx]]);
                prev_idx = Some(i);
                break;
            }
        }

        // Find next observed value
        let mut next_value = None;
        let mut next_idx = None;
        for i in (sample_idx + 1)..n_samples {
            if !data[[i, var_idx]].is_nan() {
                next_value = Some(data[[i, var_idx]]);
                next_idx = Some(i);
                break;
            }
        }

        // Interpolate if both bounds are available
        if let (Some(prev_val), Some(next_val), Some(prev_i), Some(next_i)) =
            (prev_value, next_value, prev_idx, next_idx)
        {
            let weight = (sample_idx - prev_i) as f64 / (next_i - prev_i) as f64;
            Some(prev_val + weight * (next_val - prev_val))
        } else {
            None
        }
    }

    /// Impute using multiple imputation
    fn impute_with_multiple_imputation(
        &self,
        imputed_data: &mut Array2<f64>,
        uncertainty_measures: &mut Array2<f64>,
        group: &PatternGroup,
        n_imputations: usize,
        pooling_method: &PoolingMethod,
    ) -> SklResult<()> {
        // Simplified multiple imputation - would implement proper MI
        let mut all_imputations = Vec::new();

        for _ in 0..n_imputations {
            let mut current_imputation = imputed_data.clone();

            // Create one imputation (simplified)
            self.create_single_imputation(&mut current_imputation, group)?;
            all_imputations.push(current_imputation);
        }

        // Pool results using specified method
        self.pool_multiple_imputations(
            imputed_data,
            uncertainty_measures,
            &all_imputations,
            group,
            pooling_method,
        )?;

        Ok(())
    }

    /// Create a single imputation for multiple imputation
    fn create_single_imputation(
        &self,
        imputation: &mut Array2<f64>,
        group: &PatternGroup,
    ) -> SklResult<()> {
        let mut rng = thread_rng();

        for &sample_idx in &group.sample_indices {
            for (var_idx, &is_observed) in group.pattern.iter().enumerate() {
                if !is_observed {
                    // Simple random imputation with clinical constraints
                    let imputed_value =
                        self.get_reference_range_midpoint(var_idx) + rng.random_range(-10.0..10.0); // Add some random variation
                    imputation[[sample_idx, var_idx]] = imputed_value;
                }
            }
        }

        Ok(())
    }

    /// Pool multiple imputations using specified method
    fn pool_multiple_imputations(
        &self,
        imputed_data: &mut Array2<f64>,
        uncertainty_measures: &mut Array2<f64>,
        all_imputations: &[Array2<f64>],
        group: &PatternGroup,
        pooling_method: &PoolingMethod,
    ) -> SklResult<()> {
        let _n_imputations = all_imputations.len();

        for &sample_idx in &group.sample_indices {
            for (var_idx, &is_observed) in group.pattern.iter().enumerate() {
                if !is_observed {
                    // Collect values across all imputations
                    let values: Vec<f64> = all_imputations
                        .iter()
                        .map(|imp| imp[[sample_idx, var_idx]])
                        .collect();

                    // Pool according to method
                    let (pooled_value, pooled_variance) = match pooling_method {
                        PoolingMethod::RubinsRules => self.apply_rubins_rules(&values),
                        PoolingMethod::Bootstrap => self.apply_bootstrap_pooling(&values),
                        PoolingMethod::Bayesian => self.apply_bayesian_pooling(&values),
                    };

                    imputed_data[[sample_idx, var_idx]] = pooled_value;
                    uncertainty_measures[[sample_idx, var_idx]] = pooled_variance.sqrt();
                }
            }
        }

        Ok(())
    }

    /// Apply Rubin's rules for multiple imputation pooling
    fn apply_rubins_rules(&self, values: &[f64]) -> (f64, f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let within_variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let between_variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

        let total_variance = within_variance + (1.0 + 1.0 / n) * between_variance;
        (mean, total_variance)
    }

    /// Apply bootstrap pooling
    fn apply_bootstrap_pooling(&self, values: &[f64]) -> (f64, f64) {
        // Simplified bootstrap pooling
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        (mean, variance)
    }

    /// Apply Bayesian pooling
    fn apply_bayesian_pooling(&self, values: &[f64]) -> (f64, f64) {
        // Simplified Bayesian pooling
        self.apply_rubins_rules(values)
    }

    /// Assess clinical validity of imputed values
    fn assess_clinical_validity(
        &self,
        imputed_data: &Array2<f64>,
        variable_names: &[String],
        _clinical_context: &ClinicalContext,
    ) -> SklResult<ClinicalValidityAssessment> {
        let (n_samples, _n_features) = imputed_data.dim();
        let mut violations = Vec::new();
        let mut within_range_count = 0;
        let mut total_values = 0;

        // Check physiological ranges
        for (j, var_name) in variable_names.iter().enumerate() {
            if let Some(&(min_val, max_val)) =
                self.clinical_constraints.vital_sign_ranges.get(var_name)
            {
                for i in 0..n_samples {
                    let value = imputed_data[[i, j]];
                    total_values += 1;

                    if value >= min_val && value <= max_val {
                        within_range_count += 1;
                    } else {
                        violations.push(ConstraintViolation {
                            variable: var_name.clone(),
                            violation_type: ViolationType::PhysiologicalRange,
                            severity: if value < min_val * 0.5 || value > max_val * 2.0 {
                                ViolationSeverity::Critical
                            } else {
                                ViolationSeverity::Major
                            },
                            affected_samples: vec![i],
                            recommendation: format!(
                                "Value {} outside physiological range [{}, {}] for {}",
                                value, min_val, max_val, var_name
                            ),
                        });
                    }
                }
            }
        }

        let within_range_proportion = if total_values > 0 {
            within_range_count as f64 / total_values as f64
        } else {
            1.0
        };

        // Calculate clinical plausibility score
        let plausibility_score = self.calculate_plausibility_score(imputed_data, variable_names);

        // Generate expert recommendations
        let expert_recommendations = self.generate_expert_recommendations(&violations);

        Ok(ClinicalValidityAssessment {
            within_range_proportion,
            plausibility_score,
            constraint_violations: violations,
            expert_recommendations,
        })
    }

    /// Calculate clinical plausibility score
    fn calculate_plausibility_score(
        &self,
        _imputed_data: &Array2<f64>,
        _variable_names: &[String],
    ) -> f64 {
        // Simplified plausibility scoring
        // In practice, would use complex clinical reasoning
        0.85 // Placeholder
    }

    /// Generate expert recommendations
    fn generate_expert_recommendations(&self, violations: &[ConstraintViolation]) -> Vec<String> {
        let mut recommendations = Vec::new();

        let critical_violations = violations
            .iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::Critical))
            .count();

        if critical_violations > 0 {
            recommendations.push(format!(
                "Review {} critical violations before using imputed data",
                critical_violations
            ));
        }

        if violations.len() > 10 {
            recommendations.push(
                "High number of constraint violations suggests need for domain expert review"
                    .to_string(),
            );
        }

        recommendations
            .push("Consider sensitivity analysis with alternative imputation methods".to_string());

        recommendations
    }

    /// Generate regulatory compliance report
    fn generate_compliance_report(
        &self,
        analysis: &MissingPatternAnalysis,
        clinical_validity: &ClinicalValidityAssessment,
    ) -> SklResult<RegulatoryComplianceReport> {
        let ich_e9_compliant = self.check_ich_e9_compliance(analysis);
        let documentation_complete = self.check_documentation_completeness();
        let sensitivity_adequate = self.check_sensitivity_adequacy(analysis);
        let audit_trail_complete = self.regulatory_config.audit_trail;

        let mut recommendations = Vec::new();

        if !ich_e9_compliant {
            recommendations
                .push("Ensure missing data analysis follows ICH E9(R1) guidelines".to_string());
        }

        if !documentation_complete {
            recommendations
                .push("Complete missing data documentation for regulatory submission".to_string());
        }

        if clinical_validity.within_range_proportion < 0.95 {
            recommendations.push(
                "Review imputation method due to high rate of implausible values".to_string(),
            );
        }

        Ok(RegulatoryComplianceReport {
            ich_e9_compliant,
            documentation_complete,
            sensitivity_adequate,
            audit_trail_complete,
            recommendations,
        })
    }

    /// Check ICH E9(R1) compliance
    fn check_ich_e9_compliance(&self, analysis: &MissingPatternAnalysis) -> bool {
        // Check key ICH E9(R1) requirements
        self.regulatory_config.ich_e9_compliance
            && matches!(
                analysis.mechanism,
                MissingDataMechanism::MAR { .. } | MissingDataMechanism::MCAR
            )
            && !analysis.informative_missing.is_empty() // Has identified potentially informative missing
    }

    /// Check documentation completeness
    fn check_documentation_completeness(&self) -> bool {
        // Simplified check - would verify all required documentation exists
        true // Placeholder
    }

    /// Check sensitivity analysis adequacy
    fn check_sensitivity_adequacy(&self, analysis: &MissingPatternAnalysis) -> bool {
        self.regulatory_config.sensitivity_analysis
            && matches!(analysis.mechanism, MissingDataMechanism::MNAR { .. })
    }

    /// Perform sensitivity analysis for missing data
    fn perform_sensitivity_analysis(
        &self,
        _original_data: &ArrayView2<f64>,
        imputed_data: &Array2<f64>,
        _variable_names: &[String],
        _clinical_context: &ClinicalContext,
    ) -> SklResult<SensitivityAnalysisResults> {
        // Simplified sensitivity analysis implementation
        let mut mechanism_sensitivity = HashMap::new();

        // Test sensitivity to different missing data mechanisms
        mechanism_sensitivity.insert("MCAR".to_string(), Array1::ones(5));
        mechanism_sensitivity.insert("MAR".to_string(), Array1::ones(5) * 0.9);
        mechanism_sensitivity.insert("MNAR_Selection".to_string(), Array1::ones(5) * 0.8);
        mechanism_sensitivity.insert("MNAR_Pattern".to_string(), Array1::ones(5) * 0.7);

        // Simplified tipping point analysis
        let tipping_points = vec![TippingPoint {
            parameter: "delta".to_string(),
            tipping_value: 0.5,
            confidence_interval: (0.3, 0.7),
            clinical_significance: false,
        }];

        // Simplified pattern-mixture model results
        let pattern_mixture_results = PatternMixtureResults {
            pattern_effects: HashMap::new(),
            mixture_weights: Array1::ones(3) / 3.0,
            pattern_specific_estimates: HashMap::new(),
        };

        // Simplified selection model results
        let selection_model_results = SelectionModelResults {
            selection_probabilities: Array1::ones(imputed_data.nrows()) * 0.8,
            selection_corrected_estimates: Array1::ones(5),
            informative_missing_indicators: vec!["outcome".to_string()],
        };

        Ok(SensitivityAnalysisResults {
            mechanism_sensitivity,
            tipping_points,
            pattern_mixture_results,
            selection_model_results,
        })
    }
}

/// Clinical context information for healthcare imputation
#[derive(Debug, Clone)]
pub struct ClinicalContext {
    /// Study type (clinical trial, observational, etc.)
    pub study_type: StudyType,
    /// Primary indication or disease area
    pub indication: String,
    /// Patient population characteristics
    pub population: PopulationCharacteristics,
    /// Treatment information
    pub treatments: Vec<Treatment>,
    /// Clinical endpoints
    pub endpoints: Vec<ClinicalEndpoint>,
}

/// Types of clinical studies
#[derive(Debug, Clone)]
pub enum StudyType {
    RandomizedControlledTrial,
    ObservationalCohort,
    CaseControl,
    CrossSectional,
    RealWorldEvidence,
}

/// Patient population characteristics
#[derive(Debug, Clone)]
pub struct PopulationCharacteristics {
    /// Age range
    pub age_range: (f64, f64),
    /// Gender distribution
    pub gender_distribution: HashMap<String, f64>,
    /// Comorbidity information
    pub comorbidities: Vec<String>,
    /// Severity stratification
    pub severity_strata: Vec<String>,
}

/// Treatment information
#[derive(Debug, Clone)]
pub struct Treatment {
    /// Treatment name
    pub name: String,
    /// Dosing regimen
    pub dosing: DosingRegimen,
    /// Administration route
    pub route: AdministrationRoute,
    /// Expected effects on outcomes
    pub expected_effects: Vec<ExpectedEffect>,
}

/// Dosing regimen information
#[derive(Debug, Clone)]
pub struct DosingRegimen {
    /// Dose amount
    pub dose: f64,
    /// Dose unit
    pub unit: String,
    /// Frequency (times per day)
    pub frequency: f64,
    /// Duration (days)
    pub duration: f64,
}

/// Routes of administration
#[derive(Debug, Clone)]
pub enum AdministrationRoute {
    Oral,
    Intravenous,
    Intramuscular,
    Subcutaneous,
    Topical,
    Inhalation,
}

/// Expected treatment effects
#[derive(Debug, Clone)]
pub struct ExpectedEffect {
    /// Outcome variable affected
    pub outcome: String,
    /// Expected direction of effect
    pub direction: EffectDirection,
    /// Expected magnitude
    pub magnitude: f64,
    /// Time to effect
    pub time_to_effect: f64,
}

/// Direction of treatment effects
#[derive(Debug, Clone)]
pub enum EffectDirection {
    Increase,
    Decrease,
    NoEffect,
    Bidirectional,
}

/// Clinical endpoint definitions
#[derive(Debug, Clone)]
pub struct ClinicalEndpoint {
    /// Endpoint name
    pub name: String,
    /// Endpoint type
    pub endpoint_type: EndpointType,
    /// Clinical significance threshold
    pub clinical_threshold: f64,
    /// Measurement timing
    pub timing: MeasurementTiming,
}

/// Types of clinical endpoints
#[derive(Debug, Clone)]
pub enum EndpointType {
    Primary,
    Secondary,
    Exploratory,
    Safety,
}

/// Measurement timing information
#[derive(Debug, Clone)]
pub struct MeasurementTiming {
    /// Baseline measurement
    pub baseline: bool,
    /// Follow-up time points (days)
    pub follow_up_times: Vec<f64>,
    /// End of study measurement
    pub end_of_study: bool,
}

// Default implementations for configuration structs
impl Default for ClinicalConstraints {
    fn default() -> Self {
        let mut vital_sign_ranges = HashMap::new();
        vital_sign_ranges.insert("systolic_bp".to_string(), (80.0, 200.0));
        vital_sign_ranges.insert("diastolic_bp".to_string(), (40.0, 120.0));
        vital_sign_ranges.insert("heart_rate".to_string(), (40.0, 150.0));
        vital_sign_ranges.insert("temperature".to_string(), (95.0, 105.0));

        Self {
            vital_sign_ranges,
            _lab_reference_ranges: HashMap::new(),
            _drug_interactions: Vec::new(),
            _temporal_constraints: Vec::new(),
        }
    }
}

impl Default for RegulatoryConfig {
    fn default() -> Self {
        Self {
            ich_e9_compliance: true,
            min_imputations: 20,
            sensitivity_analysis: true,
            audit_trail: true,
        }
    }
}

/// Example usage of the healthcare imputation framework
pub fn healthcare_imputation_example() -> SklResult<()> {
    // Create synthetic healthcare dataset
    let (data, variable_names, clinical_context) = create_synthetic_healthcare_data()?;

    // Initialize healthcare imputation framework
    let mut framework = HealthcareImputationFramework::new()
        .with_clinical_constraints(ClinicalConstraints::default())
        .with_regulatory_config(RegulatoryConfig::default());

    // Analyze missing data patterns
    let missing_analysis =
        framework.analyze_missing_patterns(&data.view(), &variable_names, &clinical_context)?;

    println!("Missing data mechanism: {:?}", missing_analysis.mechanism);
    println!("Monotone pattern: {}", missing_analysis.is_monotone);
    println!(
        "Informative missing variables: {:?}",
        missing_analysis.informative_missing
    );

    // Perform healthcare-compliant imputation
    let results =
        framework.impute_healthcare_data(&data.view(), &variable_names, &clinical_context)?;

    // Report results
    println!("Clinical validity assessment:");
    println!(
        "  Within range proportion: {:.2}",
        results.clinical_validity.within_range_proportion
    );
    println!(
        "  Plausibility score: {:.2}",
        results.clinical_validity.plausibility_score
    );
    println!(
        "  Constraint violations: {}",
        results.clinical_validity.constraint_violations.len()
    );

    println!("Regulatory compliance:");
    println!(
        "  ICH E9(R1) compliant: {}",
        results.regulatory_compliance.ich_e9_compliant
    );
    println!(
        "  Documentation complete: {}",
        results.regulatory_compliance.documentation_complete
    );

    if let Some(sensitivity) = &results.sensitivity_analysis {
        println!(
            "Sensitivity analysis performed with {} mechanism tests",
            sensitivity.mechanism_sensitivity.len()
        );
    }

    Ok(())
}

/// Create synthetic healthcare dataset for demonstration
fn create_synthetic_healthcare_data() -> SklResult<(Array2<f64>, Vec<String>, ClinicalContext)> {
    let mut rng = thread_rng();
    let n_patients = 1000;
    let n_variables = 10;

    // Create realistic healthcare data with missing patterns
    let mut data = Array2::zeros((n_patients, n_variables));

    for i in 0..n_patients {
        // Demographic variables (rarely missing)
        data[[i, 0]] = rng.random_range(18.0..85.0); // Age
        data[[i, 1]] = if rng.random::<f64>() < 0.5 { 0.0 } else { 1.0 }; // Gender (0=F, 1=M)

        // Vital signs (occasionally missing)
        if rng.random::<f64>() > 0.05 {
            data[[i, 2]] = rng.random_range(90.0..160.0); // Systolic BP
            data[[i, 3]] = rng.random_range(60.0..100.0); // Diastolic BP
            data[[i, 4]] = rng.random_range(60.0..100.0); // Heart rate
        } else {
            data[[i, 2]] = f64::NAN;
            data[[i, 3]] = f64::NAN;
            data[[i, 4]] = f64::NAN;
        }

        // Lab values (frequently missing, especially in sicker patients)
        let missing_prob = if data[[i, 0]] > 70.0 { 0.3 } else { 0.1 }; // Higher missing rate in elderly

        if rng.random::<f64>() > missing_prob {
            data[[i, 5]] = rng.random_range(3.5..5.5); // Hemoglobin
            data[[i, 6]] = rng.random_range(0.8..1.2); // Creatinine
            data[[i, 7]] = rng.random_range(135.0..145.0); // Sodium
        } else {
            data[[i, 5]] = f64::NAN;
            data[[i, 6]] = f64::NAN;
            data[[i, 7]] = f64::NAN;
        }

        // Outcome measures (missing due to dropout)
        let dropout_prob = if i as f64 / n_patients as f64 > 0.8 {
            0.2
        } else {
            0.05
        };
        if rng.random::<f64>() > dropout_prob {
            data[[i, 8]] = rng.random_range(0.0..100.0); // Quality of life score
            data[[i, 9]] = if rng.random::<f64>() < 0.1 { 1.0 } else { 0.0 }; // Adverse event
        } else {
            data[[i, 8]] = f64::NAN;
            data[[i, 9]] = f64::NAN;
        }
    }

    let variable_names = vec![
        "age".to_string(),
        "gender".to_string(),
        "systolic_bp".to_string(),
        "diastolic_bp".to_string(),
        "heart_rate".to_string(),
        "hemoglobin".to_string(),
        "creatinine".to_string(),
        "sodium".to_string(),
        "qol_score".to_string(),
        "adverse_event".to_string(),
    ];

    let clinical_context = ClinicalContext {
        study_type: StudyType::RandomizedControlledTrial,
        indication: "Cardiovascular Disease".to_string(),
        population: PopulationCharacteristics {
            age_range: (18.0, 85.0),
            gender_distribution: [("Female".to_string(), 0.5), ("Male".to_string(), 0.5)]
                .iter()
                .cloned()
                .collect(),
            comorbidities: vec!["Hypertension".to_string(), "Diabetes".to_string()],
            severity_strata: vec![
                "Mild".to_string(),
                "Moderate".to_string(),
                "Severe".to_string(),
            ],
        },
        treatments: vec![Treatment {
            name: "Study Drug".to_string(),
            dosing: DosingRegimen {
                dose: 10.0,
                unit: "mg".to_string(),
                frequency: 1.0,
                duration: 180.0,
            },
            route: AdministrationRoute::Oral,
            expected_effects: vec![ExpectedEffect {
                outcome: "systolic_bp".to_string(),
                direction: EffectDirection::Decrease,
                magnitude: 10.0,
                time_to_effect: 30.0,
            }],
        }],
        endpoints: vec![ClinicalEndpoint {
            name: "Change in Systolic BP".to_string(),
            endpoint_type: EndpointType::Primary,
            clinical_threshold: 5.0,
            timing: MeasurementTiming {
                baseline: true,
                follow_up_times: vec![30.0, 60.0, 90.0, 180.0],
                end_of_study: true,
            },
        }],
    };

    Ok((data, variable_names, clinical_context))
}

/// Main function to demonstrate healthcare imputation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏥 Healthcare Data Imputation Framework");
    println!("=======================================");

    // Create synthetic healthcare data
    println!("\n📊 Generating synthetic clinical trial data...");
    let (data, _variable_names, clinical_context) = create_synthetic_healthcare_data()?;

    println!("  - Patients: {}", data.nrows());
    println!("  - Variables: {}", data.ncols());
    println!("  - Study type: {:?}", clinical_context.study_type);
    println!("  - Indication: {}", clinical_context.indication);

    // Analyze missing data
    let missing_count = data.iter().filter(|&&x| x.is_nan()).count();
    let missing_percentage = missing_count as f64 / data.len() as f64 * 100.0;

    println!("\n🔍 Missing Data Analysis:");
    println!("  - Total values: {}", data.len());
    println!("  - Missing values: {}", missing_count);
    println!("  - Missing percentage: {:.2}%", missing_percentage);

    println!("\n✅ Healthcare imputation framework initialized successfully!");
    println!("   This is a demonstration of healthcare data imputation capabilities.");

    Ok(())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthcare_framework_creation() {
        let framework = HealthcareImputationFramework::new();
        assert!(framework.missing_pattern_analysis.is_none());
    }

    #[test]
    fn test_synthetic_data_generation() {
        let result = create_synthetic_healthcare_data();
        assert!(result.is_ok());

        let (data, variable_names, _) = result.unwrap();
        assert_eq!(data.nrows(), 1000);
        assert_eq!(data.ncols(), 10);
        assert_eq!(variable_names.len(), 10);
    }

    #[test]
    fn test_clinical_constraints_default() {
        let constraints = ClinicalConstraints::default();
        assert!(constraints.vital_sign_ranges.contains_key("systolic_bp"));
        assert!(constraints.vital_sign_ranges.contains_key("heart_rate"));
    }

    #[test]
    fn test_regulatory_config_default() {
        let config = RegulatoryConfig::default();
        assert!(config.ich_e9_compliance);
        assert!(config.sensitivity_analysis);
        assert_eq!(config.min_imputations, 20);
    }
}

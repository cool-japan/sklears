use super::security_types::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, SystemTime};

#[path = "threat_modeling_types.rs"]
mod threat_modeling_types;
pub use threat_modeling_types::*;
// `ThreatActor`/`ThreatActorType`/`AnalysisDepth`/`MitigationStrategy` are also defined in
// `security_types` (for the generic vulnerability/risk-assessment types, under the same
// names but different shapes); re-import the threat-modeling-specific versions explicitly
// here so they win unambiguously over the glob import above instead of triggering an
// "ambiguous glob re-exports" warning.
pub use threat_modeling_types::{AnalysisDepth, MitigationStrategy, ThreatActor, ThreatActorType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatModelingEngine {
    stride_analyzer: StrideAnalyzer,
    attack_tree_generator: AttackTreeGenerator,
    threat_scenarios: Vec<ThreatScenario>,
    threat_intelligence: ThreatIntelligenceManager,
    attack_vectors: Vec<AttackVector>,
    threat_landscape: ThreatLandscapeAssessment,
    modeling_config: ThreatModelingConfig,
    threat_cache: HashMap<String, CachedThreatModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrideAnalyzer {
    spoofing_detectors: Vec<SpoofingDetector>,
    tampering_detectors: Vec<TamperingDetector>,
    repudiation_detectors: Vec<RepudiationDetector>,
    information_disclosure_detectors: Vec<InformationDisclosureDetector>,
    denial_of_service_detectors: Vec<DenialOfServiceDetector>,
    elevation_of_privilege_detectors: Vec<ElevationOfPrivilegeDetector>,
    stride_weights: HashMap<StrideCategory, f64>,
    contextual_analyzers: HashMap<String, ContextualStrideAnalyzer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackTreeGenerator {
    attack_patterns: HashMap<String, AttackPattern>,
    tree_templates: Vec<AttackTreeTemplate>,
    node_generators: HashMap<String, AttackNodeGenerator>,
    tree_optimization: AttackTreeOptimization,
    probability_calculators: Vec<ProbabilityCalculator>,
    cost_benefit_analyzers: Vec<CostBenefitAnalyzer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIntelligenceManager {
    intelligence_sources: Vec<ThreatIntelligenceSource>,
    threat_feeds: HashMap<String, ThreatFeed>,
    indicators_of_compromise: Vec<IndicatorOfCompromise>,
    attack_campaigns: Vec<AttackCampaign>,
    threat_actor_profiles: HashMap<String, ThreatActorProfile>,
    intelligence_correlation: IntelligenceCorrelation,
    feed_aggregator: FeedAggregator,
    intelligence_scoring: IntelligenceScoring,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatLandscapeAssessment {
    threat_environment: ThreatEnvironment,
    emerging_threats: Vec<EmergingThreat>,
    threat_trends: Vec<ThreatTrend>,
    geographic_factors: HashMap<String, GeographicThreatFactor>,
    industry_specific_threats: HashMap<String, Vec<IndustryThreat>>,
    technology_threats: HashMap<String, Vec<TechnologyThreat>>,
    threat_evolution_models: Vec<ThreatEvolutionModel>,
    landscape_metrics: LandscapeMetrics,
}

impl ThreatModelingEngine {
    pub fn new() -> Self {
        Self {
            stride_analyzer: StrideAnalyzer::new(),
            attack_tree_generator: AttackTreeGenerator::new(),
            threat_scenarios: Vec::new(),
            threat_intelligence: ThreatIntelligenceManager::new(),
            attack_vectors: Vec::new(),
            threat_landscape: ThreatLandscapeAssessment::new(),
            modeling_config: ThreatModelingConfig::default(),
            threat_cache: HashMap::new(),
        }
    }

    pub fn analyze_threats(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<ThreatModelingResult, ThreatModelingError> {
        let model_id = self.generate_model_id(context);

        if let Some(cached_result) = self.get_cached_result(&model_id) {
            if self.is_cache_valid(&cached_result) {
                return Ok(cached_result.result.clone());
            }
        }

        let stride_analysis = self.perform_stride_analysis(context)?;
        let attack_trees = self.generate_attack_trees(context, &stride_analysis)?;
        let threat_scenarios = self.generate_threat_scenarios(context, &stride_analysis)?;
        let attack_vectors = self.identify_attack_vectors(context)?;
        let threat_landscape = self.assess_threat_landscape(context)?;
        let intelligence_insights = self.gather_intelligence_insights(context)?;
        let risk_prioritization = self.prioritize_threats(&stride_analysis, &attack_trees)?;
        let mitigation_recommendations =
            self.generate_mitigation_recommendations(&stride_analysis, &attack_trees)?;
        let identified_threats = self.extract_identified_threats(&stride_analysis, &attack_vectors);
        let overall_risk_score =
            self.calculate_overall_risk_score(&stride_analysis, &attack_vectors);

        let result = ThreatModelingResult {
            model_id: model_id.clone(),
            analysis_timestamp: SystemTime::now(),
            stride_analysis,
            attack_trees,
            threat_scenarios,
            attack_vectors,
            threat_landscape,
            intelligence_insights,
            risk_prioritization,
            mitigation_recommendations,
            model_confidence: self.calculate_model_confidence()?,
            model_metadata: self.generate_metadata(context),
            identified_threats,
            overall_risk_score,
        };

        self.cache_result(model_id, &result);
        Ok(result)
    }

    fn perform_stride_analysis(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<StrideAnalysisResult, ThreatModelingError> {
        let mut spoofing_threats = Vec::new();
        let mut tampering_threats = Vec::new();
        let mut repudiation_threats = Vec::new();
        let mut information_disclosure_threats = Vec::new();
        let mut denial_of_service_threats = Vec::new();
        let mut elevation_of_privilege_threats = Vec::new();

        for detector in &self.stride_analyzer.spoofing_detectors {
            spoofing_threats.extend(detector.detect_spoofing_threats(context)?);
        }

        for detector in &self.stride_analyzer.tampering_detectors {
            tampering_threats.extend(detector.detect_tampering_threats(context)?);
        }

        for detector in &self.stride_analyzer.repudiation_detectors {
            repudiation_threats.extend(detector.detect_repudiation_threats(context)?);
        }

        for detector in &self.stride_analyzer.information_disclosure_detectors {
            information_disclosure_threats.extend(detector.detect_disclosure_threats(context)?);
        }

        for detector in &self.stride_analyzer.denial_of_service_detectors {
            denial_of_service_threats.extend(detector.detect_dos_threats(context)?);
        }

        for detector in &self.stride_analyzer.elevation_of_privilege_detectors {
            elevation_of_privilege_threats.extend(detector.detect_escalation_threats(context)?);
        }

        let composite_threats = self.identify_composite_threats(
            &spoofing_threats,
            &tampering_threats,
            &repudiation_threats,
            &information_disclosure_threats,
            &denial_of_service_threats,
            &elevation_of_privilege_threats,
        )?;

        let stride_scores = self.calculate_stride_scores(
            &spoofing_threats,
            &tampering_threats,
            &repudiation_threats,
            &information_disclosure_threats,
            &denial_of_service_threats,
            &elevation_of_privilege_threats,
        )?;

        let overall_stride_rating = self.calculate_overall_stride_rating(&stride_scores)?;
        let confidence_intervals = self.calculate_confidence_intervals(&stride_scores)?;

        Ok(StrideAnalysisResult {
            analysis_id: format!(
                "stride_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .expect("duration_since should succeed")
                    .as_secs()
            ),
            spoofing_threats,
            tampering_threats,
            repudiation_threats,
            information_disclosure_threats,
            denial_of_service_threats,
            elevation_of_privilege_threats,
            composite_threats,
            stride_scores,
            overall_stride_rating,
            confidence_intervals,
        })
    }

    fn generate_attack_trees(
        &mut self,
        context: &TraitUsageContext,
        stride_analysis: &StrideAnalysisResult,
    ) -> Result<Vec<AttackTree>, ThreatModelingError> {
        let mut attack_trees = Vec::new();

        for pattern in self.attack_tree_generator.attack_patterns.values() {
            if self.is_pattern_applicable(pattern, context)? {
                let tree =
                    self.build_attack_tree_from_pattern(pattern, context, stride_analysis)?;
                attack_trees.push(tree);
            }
        }

        for template in &self.attack_tree_generator.tree_templates {
            if self.is_template_applicable(template, context)? {
                let tree =
                    self.build_attack_tree_from_template(template, context, stride_analysis)?;
                attack_trees.push(tree);
            }
        }

        let custom_trees = self.generate_custom_attack_trees(context, stride_analysis)?;
        attack_trees.extend(custom_trees);

        for tree in &mut attack_trees {
            self.optimize_attack_tree(tree)?;
            self.calculate_tree_metrics(tree)?;
        }

        Ok(attack_trees)
    }

    fn generate_threat_scenarios(
        &mut self,
        context: &TraitUsageContext,
        stride_analysis: &StrideAnalysisResult,
    ) -> Result<Vec<ThreatScenario>, ThreatModelingError> {
        let mut scenarios = Vec::new();

        let base_scenarios = self.generate_base_threat_scenarios(context, stride_analysis)?;
        scenarios.extend(base_scenarios);

        let advanced_scenarios = self.generate_advanced_persistent_threat_scenarios(context)?;
        scenarios.extend(advanced_scenarios);

        let insider_scenarios = self.generate_insider_threat_scenarios(context)?;
        scenarios.extend(insider_scenarios);

        let supply_chain_scenarios = self.generate_supply_chain_attack_scenarios(context)?;
        scenarios.extend(supply_chain_scenarios);

        let zero_day_scenarios = self.generate_zero_day_scenarios(context)?;
        scenarios.extend(zero_day_scenarios);

        for scenario in &mut scenarios {
            self.enrich_scenario_with_intelligence(scenario)?;
            self.calculate_scenario_likelihood(scenario, context)?;
            self.generate_scenario_variants(scenario, context)?;
        }

        Ok(scenarios)
    }

    fn identify_attack_vectors(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<Vec<AttackVector>, ThreatModelingError> {
        let mut vectors = Vec::new();

        let network_vectors = self.identify_network_attack_vectors(context)?;
        vectors.extend(network_vectors);

        let application_vectors = self.identify_application_attack_vectors(context)?;
        vectors.extend(application_vectors);

        let social_engineering_vectors = self.identify_social_engineering_vectors(context)?;
        vectors.extend(social_engineering_vectors);

        let physical_vectors = self.identify_physical_attack_vectors(context)?;
        vectors.extend(physical_vectors);

        let supply_chain_vectors = self.identify_supply_chain_vectors(context)?;
        vectors.extend(supply_chain_vectors);

        for vector in &mut vectors {
            self.analyze_vector_effectiveness(vector, context)?;
            self.identify_vector_dependencies(vector)?;
            self.generate_vector_variants(vector, context)?;
        }

        Ok(vectors)
    }

    fn assess_threat_landscape(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<ThreatLandscapeAssessment, ThreatModelingError> {
        let threat_environment = self.analyze_threat_environment(context)?;
        let emerging_threats = self.identify_emerging_threats(context)?;
        let threat_trends = self.analyze_threat_trends(context)?;
        let geographic_factors = self.assess_geographic_threat_factors(context)?;
        let industry_threats = self.assess_industry_specific_threats(context)?;
        let technology_threats = self.assess_technology_threats(context)?;
        let evolution_models = self.build_threat_evolution_models(context)?;
        let landscape_metrics = self.calculate_landscape_metrics()?;

        Ok(ThreatLandscapeAssessment {
            threat_environment,
            emerging_threats,
            threat_trends,
            geographic_factors,
            industry_specific_threats: industry_threats,
            technology_threats,
            threat_evolution_models: evolution_models,
            landscape_metrics,
        })
    }

    fn gather_intelligence_insights(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<Vec<IntelligenceInsight>, ThreatModelingError> {
        let mut insights = Vec::new();

        for source in &self.threat_intelligence.intelligence_sources {
            let source_insights = source.gather_insights(context)?;
            insights.extend(source_insights);
        }

        let correlated_insights = self.threat_intelligence.correlate_intelligence(&insights)?;
        insights.extend(correlated_insights);

        let scored_insights = self.threat_intelligence.score_intelligence(&insights)?;
        insights.extend(scored_insights);

        Ok(insights)
    }

    fn prioritize_threats(
        &self,
        stride_analysis: &StrideAnalysisResult,
        attack_trees: &[AttackTree],
    ) -> Result<Vec<ThreatRiskPriority>, ThreatModelingError> {
        let mut priorities = Vec::new();

        for (category, score) in &stride_analysis.stride_scores {
            let priority = ThreatRiskPriority {
                threat_category: format!("{:?}", category),
                risk_score: *score,
                priority_level: self.calculate_priority_level(*score)?,
                justification: self.generate_priority_justification(category, *score)?,
                recommended_timeline: self.calculate_response_timeline(*score)?,
            };
            priorities.push(priority);
        }

        for tree in attack_trees {
            for path in &tree.critical_paths {
                let priority = ThreatRiskPriority {
                    threat_category: format!("Attack Path: {}", path.path_id),
                    risk_score: path.risk_score,
                    priority_level: self.calculate_priority_level(path.risk_score)?,
                    justification: format!("Critical attack path with {} steps", path.steps.len()),
                    recommended_timeline: self.calculate_response_timeline(path.risk_score)?,
                };
                priorities.push(priority);
            }
        }

        priorities.sort_by(|a, b| {
            b.risk_score
                .partial_cmp(&a.risk_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(priorities)
    }

    fn generate_mitigation_recommendations(
        &self,
        stride_analysis: &StrideAnalysisResult,
        attack_trees: &[AttackTree],
    ) -> Result<Vec<MitigationRecommendation>, ThreatModelingError> {
        let mut recommendations = Vec::new();

        for (category, _threats) in [
            (
                StrideCategory::Spoofing,
                &stride_analysis.spoofing_threats as &dyn std::any::Any,
            ),
            (
                StrideCategory::Tampering,
                &stride_analysis.tampering_threats as &dyn std::any::Any,
            ),
            (
                StrideCategory::Repudiation,
                &stride_analysis.repudiation_threats as &dyn std::any::Any,
            ),
            (
                StrideCategory::InformationDisclosure,
                &stride_analysis.information_disclosure_threats as &dyn std::any::Any,
            ),
            (
                StrideCategory::DenialOfService,
                &stride_analysis.denial_of_service_threats as &dyn std::any::Any,
            ),
            (
                StrideCategory::ElevationOfPrivilege,
                &stride_analysis.elevation_of_privilege_threats as &dyn std::any::Any,
            ),
        ] {
            let category_recommendations = self.generate_category_mitigations(&category)?;
            recommendations.extend(category_recommendations);
        }

        for tree in attack_trees {
            let tree_recommendations = self.generate_attack_tree_mitigations(tree)?;
            recommendations.extend(tree_recommendations);
        }

        Ok(recommendations)
    }

    fn calculate_model_confidence(&self) -> Result<f64, ThreatModelingError> {
        let confidence_factors = [
            self.calculate_data_quality_confidence()?,
            self.calculate_intelligence_confidence()?,
            self.calculate_model_completeness_confidence()?,
            self.calculate_temporal_confidence()?,
        ];

        let weighted_confidence =
            confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64;
        Ok(weighted_confidence.clamp(0.0, 1.0))
    }
}

impl Default for ThreatModelingEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Private heuristic and support methods backing [`ThreatModelingEngine::analyze_threats`].
///
/// These implement the actual context-driven analysis: each method reads one or more
/// [`TraitUsageContext`] flags (or the engine's own configuration/collections) and derives
/// a plausible, internally-consistent result that varies with the input, in the same
/// shallow-but-genuine heuristic style as `core_analyzer.rs`.
impl ThreatModelingEngine {
    fn generate_model_id(&self, context: &TraitUsageContext) -> String {
        format!(
            "threat_model_{}_{}",
            context.trait_name,
            context.traits.len()
        )
    }

    fn get_cached_result(&self, model_id: &str) -> Option<CachedThreatModel> {
        self.threat_cache.get(model_id).cloned()
    }

    fn is_cache_valid(&self, cached: &CachedThreatModel) -> bool {
        cached
            .cache_timestamp
            .elapsed()
            .map(|elapsed| elapsed < cached.cache_ttl)
            .unwrap_or(false)
    }

    fn generate_metadata(&self, context: &TraitUsageContext) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("trait_name".to_string(), context.trait_name.clone());
        metadata.insert("trait_count".to_string(), context.traits.len().to_string());
        metadata.insert(
            "analysis_depth".to_string(),
            format!("{:?}", self.modeling_config.analysis_depth),
        );
        metadata
    }

    fn cache_result(&mut self, model_id: String, result: &ThreatModelingResult) {
        let cache_ttl = self.modeling_config.cache_duration;
        self.threat_cache.insert(
            model_id,
            CachedThreatModel {
                result: result.clone(),
                cache_timestamp: SystemTime::now(),
                cache_ttl,
            },
        );
    }

    fn identify_composite_threats(
        &self,
        spoofing: &[SpoofingThreat],
        tampering: &[TamperingThreat],
        repudiation: &[RepudiationThreat],
        disclosure: &[InformationDisclosureThreat],
        dos: &[DenialOfServiceThreat],
        elevation: &[ElevationOfPrivilegeThreat],
    ) -> Result<Vec<CompositeThreat>, ThreatModelingError> {
        let mut composites = Vec::new();
        if !spoofing.is_empty() && !elevation.is_empty() {
            composites.push(CompositeThreat {
                composite_id: "COMPOSITE-SPOOF-ELEVATE".to_string(),
                contributing_categories: vec![
                    StrideCategory::Spoofing,
                    StrideCategory::ElevationOfPrivilege,
                ],
                description: "Identity spoofing combined with privilege escalation".to_string(),
                combined_severity: ThreatSeverity::Critical,
                amplification_factor: 1.5,
            });
        }
        if !tampering.is_empty() && !disclosure.is_empty() {
            composites.push(CompositeThreat {
                composite_id: "COMPOSITE-TAMPER-DISCLOSE".to_string(),
                contributing_categories: vec![
                    StrideCategory::Tampering,
                    StrideCategory::InformationDisclosure,
                ],
                description: "Data tampering combined with information disclosure".to_string(),
                combined_severity: ThreatSeverity::High,
                amplification_factor: 1.3,
            });
        }
        if !dos.is_empty() && !repudiation.is_empty() {
            composites.push(CompositeThreat {
                composite_id: "COMPOSITE-DOS-REPUDIATE".to_string(),
                contributing_categories: vec![
                    StrideCategory::DenialOfService,
                    StrideCategory::Repudiation,
                ],
                description: "Denial of service enabling repudiation of availability failures"
                    .to_string(),
                combined_severity: ThreatSeverity::Medium,
                amplification_factor: 1.1,
            });
        }
        Ok(composites)
    }

    fn calculate_stride_scores(
        &self,
        spoofing: &[SpoofingThreat],
        tampering: &[TamperingThreat],
        repudiation: &[RepudiationThreat],
        disclosure: &[InformationDisclosureThreat],
        dos: &[DenialOfServiceThreat],
        elevation: &[ElevationOfPrivilegeThreat],
    ) -> Result<HashMap<StrideCategory, f64>, ThreatModelingError> {
        let weight = |c: &StrideCategory| {
            self.stride_analyzer
                .stride_weights
                .get(c)
                .copied()
                .unwrap_or(1.0)
        };
        let mut scores = HashMap::new();
        scores.insert(
            StrideCategory::Spoofing,
            spoofing.len() as f64 * weight(&StrideCategory::Spoofing),
        );
        scores.insert(
            StrideCategory::Tampering,
            tampering.len() as f64 * weight(&StrideCategory::Tampering),
        );
        scores.insert(
            StrideCategory::Repudiation,
            repudiation.len() as f64 * weight(&StrideCategory::Repudiation),
        );
        scores.insert(
            StrideCategory::InformationDisclosure,
            disclosure.len() as f64 * weight(&StrideCategory::InformationDisclosure),
        );
        scores.insert(
            StrideCategory::DenialOfService,
            dos.len() as f64 * weight(&StrideCategory::DenialOfService),
        );
        scores.insert(
            StrideCategory::ElevationOfPrivilege,
            elevation.len() as f64 * weight(&StrideCategory::ElevationOfPrivilege),
        );
        Ok(scores)
    }

    fn calculate_overall_stride_rating(
        &self,
        scores: &HashMap<StrideCategory, f64>,
    ) -> Result<f64, ThreatModelingError> {
        if scores.is_empty() {
            return Ok(0.0);
        }
        let total: f64 = scores.values().sum();
        Ok((total / scores.len() as f64).min(10.0))
    }

    fn calculate_confidence_intervals(
        &self,
        scores: &HashMap<StrideCategory, f64>,
    ) -> Result<HashMap<StrideCategory, (f64, f64)>, ThreatModelingError> {
        Ok(scores
            .iter()
            .map(|(category, score)| {
                let margin = (*score * 0.15).max(0.1);
                (
                    category.clone(),
                    ((*score - margin).max(0.0), *score + margin),
                )
            })
            .collect())
    }

    fn is_pattern_applicable(
        &self,
        pattern: &AttackPattern,
        context: &TraitUsageContext,
    ) -> Result<bool, ThreatModelingError> {
        Ok(pattern
            .required_capabilities
            .iter()
            .any(|cap| context.traits.contains(cap))
            || context.has_unsafe_operations)
    }

    fn build_attack_tree_from_pattern(
        &self,
        pattern: &AttackPattern,
        _context: &TraitUsageContext,
        stride_analysis: &StrideAnalysisResult,
    ) -> Result<AttackTree, ThreatModelingError> {
        let root = AttackNode {
            node_id: format!("root_{}", pattern.pattern_id),
            node_type: AttackNodeType::Goal,
            description: pattern.description.clone(),
            children: Vec::new(),
            gate_type: None,
            success_probability: 0.5,
            attack_cost: 1.0,
            skill_required: 0.5,
            detection_probability: 0.5,
            impact_level: (stride_analysis.overall_stride_rating / 10.0).clamp(0.0, 1.0),
        };
        Ok(AttackTree {
            tree_id: format!("tree_pattern_{}", pattern.pattern_id),
            success_probability: root.success_probability,
            attack_cost: root.attack_cost,
            detection_probability: root.detection_probability,
            root_node: root,
            attack_paths: Vec::new(),
            critical_paths: Vec::new(),
            tree_metrics: AttackTreeMetrics {
                total_nodes: 1,
                max_depth: 1,
                average_branching_factor: 0.0,
                complexity_score: 1.0,
            },
        })
    }

    fn is_template_applicable(
        &self,
        template: &AttackTreeTemplate,
        context: &TraitUsageContext,
    ) -> Result<bool, ThreatModelingError> {
        Ok(!template.node_templates.is_empty() && context.has_unsafe_operations)
    }

    fn build_attack_tree_from_template(
        &self,
        template: &AttackTreeTemplate,
        _context: &TraitUsageContext,
        stride_analysis: &StrideAnalysisResult,
    ) -> Result<AttackTree, ThreatModelingError> {
        let root = AttackNode {
            node_id: format!("root_{}", template.template_id),
            node_type: AttackNodeType::Goal,
            description: template.root_goal.clone(),
            children: Vec::new(),
            gate_type: None,
            success_probability: 0.5,
            attack_cost: template.node_templates.len() as f64,
            skill_required: 0.5,
            detection_probability: 0.5,
            impact_level: (stride_analysis.overall_stride_rating / 10.0).clamp(0.0, 1.0),
        };
        Ok(AttackTree {
            tree_id: format!("tree_template_{}", template.template_id),
            success_probability: root.success_probability,
            attack_cost: root.attack_cost,
            detection_probability: root.detection_probability,
            root_node: root,
            attack_paths: Vec::new(),
            critical_paths: Vec::new(),
            tree_metrics: AttackTreeMetrics {
                total_nodes: 1,
                max_depth: 1,
                average_branching_factor: 0.0,
                complexity_score: template.node_templates.len() as f64,
            },
        })
    }

    fn generate_custom_attack_trees(
        &self,
        context: &TraitUsageContext,
        stride_analysis: &StrideAnalysisResult,
    ) -> Result<Vec<AttackTree>, ThreatModelingError> {
        let mut trees = Vec::new();
        if context.has_unsafe_operations || context.requires_elevated_privileges {
            let access_probability = if context.has_bounds_checking {
                0.3
            } else {
                0.6
            };
            let access_detection = if context.has_audit_logging { 0.7 } else { 0.3 };
            let impact_probability = access_probability * 0.8;

            let steps = vec![
                AttackStep {
                    step_id: "step_access".to_string(),
                    description: "Reach the unsafe or privileged code path".to_string(),
                    required_skills: vec!["memory_exploitation".to_string()],
                    success_probability: access_probability,
                    detection_probability: access_detection,
                },
                AttackStep {
                    step_id: "step_impact".to_string(),
                    description: "Leverage access to achieve the attacker's goal".to_string(),
                    required_skills: vec!["privilege_escalation".to_string()],
                    success_probability: impact_probability,
                    detection_probability: access_detection,
                },
            ];
            let path_success = access_probability * impact_probability;
            let impact_level = (stride_analysis.overall_stride_rating / 10.0).max(0.4);
            let risk_score = (path_success * 10.0 * impact_level.max(0.3)).min(10.0);

            let leaf = AttackNode {
                node_id: "custom_access".to_string(),
                node_type: AttackNodeType::Action,
                description: "Reach the unsafe or privileged code path".to_string(),
                children: Vec::new(),
                gate_type: None,
                success_probability: access_probability,
                attack_cost: 2.0,
                skill_required: 0.6,
                detection_probability: access_detection,
                impact_level,
            };
            let root = AttackNode {
                node_id: "custom_root".to_string(),
                node_type: AttackNodeType::Goal,
                description: format!("Compromise {}", context.trait_name),
                children: vec![leaf.clone()],
                gate_type: Some(LogicGate::And),
                success_probability: path_success,
                attack_cost: leaf.attack_cost + 1.0,
                skill_required: leaf.skill_required,
                detection_probability: leaf.detection_probability,
                impact_level: leaf.impact_level,
            };

            trees.push(AttackTree {
                tree_id: format!("tree_custom_{}", context.trait_name),
                success_probability: root.success_probability,
                attack_cost: root.attack_cost,
                detection_probability: root.detection_probability,
                attack_paths: vec![AttackPath {
                    path_id: format!("path_{}", context.trait_name),
                    steps: steps.clone(),
                    success_probability: path_success,
                    total_cost: root.attack_cost,
                }],
                critical_paths: if risk_score >= 5.0 {
                    vec![CriticalPath {
                        path_id: format!("critical_{}", context.trait_name),
                        risk_score,
                        steps,
                    }]
                } else {
                    Vec::new()
                },
                root_node: root,
                tree_metrics: AttackTreeMetrics {
                    total_nodes: 2,
                    max_depth: 2,
                    average_branching_factor: 0.5,
                    complexity_score: 2.0,
                },
            });
        }
        Ok(trees)
    }

    fn optimize_attack_tree(&self, tree: &mut AttackTree) -> Result<(), ThreatModelingError> {
        let threshold = self
            .attack_tree_generator
            .tree_optimization
            .pruning_threshold;
        tree.root_node
            .children
            .retain(|child| child.success_probability > threshold);
        Ok(())
    }

    fn calculate_tree_metrics(&self, tree: &mut AttackTree) -> Result<(), ThreatModelingError> {
        let mut queue: VecDeque<(&AttackNode, usize)> = VecDeque::new();
        queue.push_back((&tree.root_node, 1));
        let mut total_nodes = 0usize;
        let mut max_depth = 0usize;
        while let Some((node, depth)) = queue.pop_front() {
            total_nodes += 1;
            max_depth = max_depth.max(depth);
            for child in &node.children {
                queue.push_back((child, depth + 1));
            }
        }
        let branching = if total_nodes > 1 {
            (total_nodes - 1) as f64 / total_nodes as f64
        } else {
            0.0
        };
        tree.tree_metrics = AttackTreeMetrics {
            total_nodes,
            max_depth,
            average_branching_factor: branching,
            complexity_score: total_nodes as f64 * max_depth as f64,
        };
        Ok(())
    }

    fn generate_base_threat_scenarios(
        &self,
        context: &TraitUsageContext,
        stride_analysis: &StrideAnalysisResult,
    ) -> Result<Vec<ThreatScenario>, ThreatModelingError> {
        let mut scenarios = Vec::new();
        if context.handles_sensitive_data {
            scenarios.push(ThreatScenario {
                scenario_id: "SCEN-BASE-DATA".to_string(),
                name: "Sensitive data compromise".to_string(),
                description: format!(
                    "Attacker targets sensitive data exposed via {}",
                    context.trait_name
                ),
                attack_vectors: vec!["data_exfiltration".to_string()],
                threat_actors: Vec::new(),
                assets_at_risk: context.traits.clone(),
                impact_assessment: ImpactAssessment {
                    confidentiality_impact: if context.has_encryption { 0.3 } else { 0.8 },
                    integrity_impact: 0.4,
                    availability_impact: 0.2,
                    financial_impact: 0.5,
                    reputational_impact: 0.6,
                },
                likelihood: (stride_analysis.overall_stride_rating / 10.0).clamp(0.0, 1.0),
                detection_methods: Vec::new(),
                mitigation_strategies: Vec::new(),
                timeline: Self::build_timeline(1.0),
                scenario_variants: Vec::new(),
            });
        }
        Ok(scenarios)
    }

    fn generate_advanced_persistent_threat_scenarios(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<ThreatScenario>, ThreatModelingError> {
        let mut scenarios = Vec::new();
        if context.requires_elevated_privileges && context.has_unsafe_operations {
            scenarios.push(ThreatScenario {
                scenario_id: "SCEN-APT".to_string(),
                name: "Advanced persistent threat".to_string(),
                description: "A well-resourced actor establishes long-term persistence through elevated, unsafe code paths".to_string(),
                attack_vectors: vec!["privilege_escalation".to_string(), "persistence".to_string()],
                threat_actors: vec![Self::build_actor(ThreatActorType::NationState, 0.9)],
                assets_at_risk: context.traits.clone(),
                impact_assessment: ImpactAssessment {
                    confidentiality_impact: 0.8,
                    integrity_impact: 0.7,
                    availability_impact: 0.4,
                    financial_impact: 0.7,
                    reputational_impact: 0.8,
                },
                likelihood: if context.has_privilege_separation { 0.2 } else { 0.5 },
                detection_methods: Vec::new(),
                mitigation_strategies: Vec::new(),
                timeline: Self::build_timeline(4.0),
                scenario_variants: Vec::new(),
            });
        }
        Ok(scenarios)
    }

    fn generate_insider_threat_scenarios(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<ThreatScenario>, ThreatModelingError> {
        let mut scenarios = Vec::new();
        if context.requires_elevated_privileges && !context.has_privilege_separation {
            scenarios.push(ThreatScenario {
                scenario_id: "SCEN-INSIDER".to_string(),
                name: "Insider threat".to_string(),
                description:
                    "A trusted insider abuses elevated privileges absent separation of duties"
                        .to_string(),
                attack_vectors: vec!["credential_abuse".to_string()],
                threat_actors: vec![Self::build_actor(ThreatActorType::InsiderThreat, 0.5)],
                assets_at_risk: context.traits.clone(),
                impact_assessment: ImpactAssessment {
                    confidentiality_impact: 0.7,
                    integrity_impact: 0.6,
                    availability_impact: 0.3,
                    financial_impact: 0.5,
                    reputational_impact: 0.5,
                },
                likelihood: if context.has_audit_logging { 0.3 } else { 0.6 },
                detection_methods: Vec::new(),
                mitigation_strategies: Vec::new(),
                timeline: Self::build_timeline(0.7),
                scenario_variants: Vec::new(),
            });
        }
        Ok(scenarios)
    }

    fn generate_supply_chain_attack_scenarios(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<ThreatScenario>, ThreatModelingError> {
        let mut scenarios = Vec::new();
        if context.has_dynamic_dispatch || context.has_serialization {
            scenarios.push(ThreatScenario {
                scenario_id: "SCEN-SUPPLY-CHAIN".to_string(),
                name: "Supply chain compromise".to_string(),
                description: "A dependency or plugin loaded through dynamic dispatch or deserialization is compromised".to_string(),
                attack_vectors: vec!["malicious_dependency".to_string()],
                threat_actors: vec![Self::build_actor(ThreatActorType::CriminalOrganization, 0.7)],
                assets_at_risk: context.traits.clone(),
                impact_assessment: ImpactAssessment {
                    confidentiality_impact: 0.6,
                    integrity_impact: 0.8,
                    availability_impact: 0.5,
                    financial_impact: 0.6,
                    reputational_impact: 0.7,
                },
                likelihood: if context.has_type_safety_checks { 0.25 } else { 0.55 },
                detection_methods: Vec::new(),
                mitigation_strategies: Vec::new(),
                timeline: Self::build_timeline(2.5),
                scenario_variants: Vec::new(),
            });
        }
        Ok(scenarios)
    }

    fn generate_zero_day_scenarios(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<ThreatScenario>, ThreatModelingError> {
        let mut scenarios = Vec::new();
        if context.has_unsafe_operations && !context.has_bounds_checking {
            scenarios.push(ThreatScenario {
                scenario_id: "SCEN-ZERO-DAY".to_string(),
                name: "Zero-day memory safety exploit".to_string(),
                description: "An unknown memory-safety flaw in unsafe, unchecked code is exploited before a patch exists".to_string(),
                attack_vectors: vec!["memory_corruption".to_string()],
                threat_actors: vec![Self::build_actor(ThreatActorType::Unknown, 0.95)],
                assets_at_risk: context.traits.clone(),
                impact_assessment: ImpactAssessment {
                    confidentiality_impact: 0.9,
                    integrity_impact: 0.9,
                    availability_impact: 0.6,
                    financial_impact: 0.8,
                    reputational_impact: 0.8,
                },
                likelihood: 0.15,
                detection_methods: Vec::new(),
                mitigation_strategies: Vec::new(),
                timeline: Self::build_timeline(0.5),
                scenario_variants: Vec::new(),
            });
        }
        Ok(scenarios)
    }

    fn enrich_scenario_with_intelligence(
        &self,
        scenario: &mut ThreatScenario,
    ) -> Result<(), ThreatModelingError> {
        if !self.threat_intelligence.attack_campaigns.is_empty() {
            scenario.detection_methods.push(DetectionMethod {
                method_name: "threat_intelligence_correlation".to_string(),
                detection_probability: 0.6,
                false_positive_rate: 0.1,
            });
        }
        Ok(())
    }

    fn calculate_scenario_likelihood(
        &self,
        scenario: &mut ThreatScenario,
        context: &TraitUsageContext,
    ) -> Result<(), ThreatModelingError> {
        let mut likelihood = scenario.likelihood;
        if context.has_rate_limiting {
            likelihood *= 0.8;
        }
        if context.has_access_controls {
            likelihood *= 0.8;
        }
        scenario.likelihood = likelihood.clamp(0.0, 1.0);
        Ok(())
    }

    fn generate_scenario_variants(
        &self,
        scenario: &mut ThreatScenario,
        context: &TraitUsageContext,
    ) -> Result<(), ThreatModelingError> {
        if context.has_user_input {
            scenario.scenario_variants.push(ScenarioVariant {
                variant_id: format!("{}-input-triggered", scenario.scenario_id),
                description: "Variant triggered directly through untrusted user input".to_string(),
                probability_modifier: 1.2,
            });
        }
        Ok(())
    }

    fn identify_network_attack_vectors(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<AttackVector>, ThreatModelingError> {
        let mut vectors = Vec::new();
        if context.has_user_input && !context.has_input_validation {
            vectors.push(Self::build_vector(
                "network_unvalidated_input",
                "Network-delivered unvalidated input",
                0.6,
                0.4,
            ));
        }
        Ok(vectors)
    }

    fn identify_application_attack_vectors(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<AttackVector>, ThreatModelingError> {
        let mut vectors = Vec::new();
        if context.has_sql_operations && !context.has_parameterized_queries {
            vectors.push(Self::build_vector(
                "app_sql_injection",
                "SQL injection via non-parameterized queries",
                0.7,
                0.3,
            ));
        }
        if context.has_serialization && !context.has_input_validation {
            vectors.push(Self::build_vector(
                "app_deserialization",
                "Insecure deserialization",
                0.6,
                0.4,
            ));
        }
        Ok(vectors)
    }

    fn identify_social_engineering_vectors(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<AttackVector>, ThreatModelingError> {
        let mut vectors = Vec::new();
        if context.requires_elevated_privileges {
            vectors.push(Self::build_vector(
                "social_privileged_phishing",
                "Phishing targeting privileged operators",
                0.5,
                0.5,
            ));
        }
        Ok(vectors)
    }

    fn identify_physical_attack_vectors(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<AttackVector>, ThreatModelingError> {
        let mut vectors = Vec::new();
        if context.has_cryptographic_operations && !context.has_secure_key_management {
            vectors.push(Self::build_vector(
                "physical_key_extraction",
                "Physical extraction of insecurely managed keys",
                0.3,
                0.7,
            ));
        }
        Ok(vectors)
    }

    fn identify_supply_chain_vectors(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<AttackVector>, ThreatModelingError> {
        let mut vectors = Vec::new();
        if context.has_dynamic_dispatch {
            vectors.push(Self::build_vector(
                "supply_chain_trait_object",
                "Malicious implementation injected via dynamic dispatch",
                0.4,
                0.6,
            ));
        }
        Ok(vectors)
    }

    fn analyze_vector_effectiveness(
        &self,
        vector: &mut AttackVector,
        context: &TraitUsageContext,
    ) -> Result<(), ThreatModelingError> {
        if context.has_rate_limiting {
            vector.success_probability *= 0.7;
        }
        if context.has_audit_logging {
            vector.detection_difficulty *= 0.8;
        }
        Ok(())
    }

    fn identify_vector_dependencies(
        &self,
        vector: &mut AttackVector,
    ) -> Result<(), ThreatModelingError> {
        let mut deps: HashSet<String> = vector.prerequisites.iter().cloned().collect();
        if vector.success_probability > 0.5 {
            deps.insert("network_access".to_string());
        }
        vector.prerequisites = deps.into_iter().collect();
        Ok(())
    }

    fn generate_vector_variants(
        &self,
        vector: &mut AttackVector,
        context: &TraitUsageContext,
    ) -> Result<(), ThreatModelingError> {
        if context.has_resource_intensive_operations {
            vector.vector_variants.push(VectorVariant {
                variant_id: format!("{}-resource-exhaustion", vector.vector_id),
                modifications: vec!["amplified via resource-intensive operations".to_string()],
                effectiveness_change: 0.1,
            });
        }
        Ok(())
    }

    fn analyze_threat_environment(
        &self,
        context: &TraitUsageContext,
    ) -> Result<ThreatEnvironment, ThreatModelingError> {
        let flags = [
            context.has_unsafe_operations,
            context.has_user_input,
            context.requires_elevated_privileges,
            context.has_sql_operations,
            context.has_dynamic_dispatch,
        ];
        let density = flags.iter().filter(|b| **b).count() as f64 / flags.len() as f64;
        Ok(ThreatEnvironment {
            environment_type: if context.requires_elevated_privileges {
                "privileged".to_string()
            } else {
                "standard".to_string()
            },
            threat_density: density,
            maturity_level: if context.has_audit_logging && context.has_access_controls {
                "mature".to_string()
            } else {
                "developing".to_string()
            },
        })
    }

    fn identify_emerging_threats(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<EmergingThreat>, ThreatModelingError> {
        let mut threats = Vec::new();
        if context.has_cryptographic_operations && !context.has_constant_time_operations {
            threats.push(EmergingThreat {
                threat_name: "Side-channel key recovery".to_string(),
                description: "Non-constant-time cryptographic operations are increasingly targeted by automated side-channel tooling"
                    .to_string(),
                growth_rate: 0.3,
                first_observed: SystemTime::now(),
            });
        }
        Ok(threats)
    }

    fn analyze_threat_trends(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<ThreatTrend>, ThreatModelingError> {
        Ok(vec![ThreatTrend {
            trend_name: "Automated exploitation tooling".to_string(),
            direction: if context.has_input_validation {
                TrendDirection::Stable
            } else {
                TrendDirection::Deteriorating
            },
            magnitude: if context.has_input_validation {
                0.2
            } else {
                0.6
            },
        }])
    }

    fn assess_geographic_threat_factors(
        &self,
        context: &TraitUsageContext,
    ) -> Result<HashMap<String, GeographicThreatFactor>, ThreatModelingError> {
        let mut factors = HashMap::new();
        factors.insert(
            "global".to_string(),
            GeographicThreatFactor {
                region: "global".to_string(),
                risk_multiplier: if context.handles_personal_data {
                    1.3
                } else {
                    1.0
                },
            },
        );
        Ok(factors)
    }

    fn assess_industry_specific_threats(
        &self,
        context: &TraitUsageContext,
    ) -> Result<HashMap<String, Vec<IndustryThreat>>, ThreatModelingError> {
        let mut threats = HashMap::new();
        if context.handles_sensitive_data {
            threats.insert(
                "general".to_string(),
                vec![IndustryThreat {
                    industry: "general".to_string(),
                    threat_description: "Targeted data theft from software handling sensitive data"
                        .to_string(),
                    prevalence: 0.5,
                }],
            );
        }
        Ok(threats)
    }

    fn assess_technology_threats(
        &self,
        context: &TraitUsageContext,
    ) -> Result<HashMap<String, Vec<TechnologyThreat>>, ThreatModelingError> {
        let mut threats = HashMap::new();
        if context.has_unsafe_operations {
            threats.insert(
                "memory_safety".to_string(),
                vec![TechnologyThreat {
                    technology: "unsafe_rust".to_string(),
                    vulnerability_class: "memory_corruption".to_string(),
                    exposure: if context.has_bounds_checking {
                        0.3
                    } else {
                        0.7
                    },
                }],
            );
        }
        Ok(threats)
    }

    fn build_threat_evolution_models(
        &self,
        context: &TraitUsageContext,
    ) -> Result<Vec<ThreatEvolutionModel>, ThreatModelingError> {
        Ok(vec![ThreatEvolutionModel {
            model_name: "linear_capability_growth".to_string(),
            projected_growth: if context.has_resource_limits {
                0.1
            } else {
                0.25
            },
            time_horizon: Duration::from_secs(86400 * 365),
        }])
    }

    fn calculate_landscape_metrics(&self) -> Result<LandscapeMetrics, ThreatModelingError> {
        Ok(LandscapeMetrics {
            total_threats_tracked: self.threat_intelligence.indicators_of_compromise.len() as u32,
            average_severity: 5.0,
            trend_score: 0.5,
        })
    }

    fn calculate_priority_level(
        &self,
        score: f64,
    ) -> Result<MitigationPriority, ThreatModelingError> {
        Ok(match score {
            s if s >= 7.5 => MitigationPriority::Critical,
            s if s >= 5.0 => MitigationPriority::High,
            s if s >= 2.5 => MitigationPriority::Medium,
            _ => MitigationPriority::Low,
        })
    }

    fn generate_priority_justification(
        &self,
        category: &StrideCategory,
        score: f64,
    ) -> Result<String, ThreatModelingError> {
        let weight = self
            .stride_analyzer
            .stride_weights
            .get(category)
            .copied()
            .unwrap_or(1.0);
        Ok(format!("{category:?} threats scored {score:.2}/10, weighted by configured STRIDE weight {weight:.1}"))
    }

    fn calculate_response_timeline(&self, score: f64) -> Result<Duration, ThreatModelingError> {
        Ok(match score {
            s if s >= 7.5 => Duration::from_secs(86400),
            s if s >= 5.0 => Duration::from_secs(86400 * 7),
            s if s >= 2.5 => Duration::from_secs(86400 * 30),
            _ => Duration::from_secs(86400 * 90),
        })
    }

    fn generate_category_mitigations(
        &self,
        category: &StrideCategory,
    ) -> Result<Vec<MitigationRecommendation>, ThreatModelingError> {
        let (title, effort) = match category {
            StrideCategory::Spoofing => (
                "Strengthen identity verification",
                ImplementationEffort::Medium,
            ),
            StrideCategory::Tampering => ("Add integrity checks", ImplementationEffort::Medium),
            StrideCategory::Repudiation => ("Expand audit logging", ImplementationEffort::Low),
            StrideCategory::InformationDisclosure => (
                "Classify and encrypt sensitive data",
                ImplementationEffort::High,
            ),
            StrideCategory::DenialOfService => (
                "Add rate limiting and resource quotas",
                ImplementationEffort::Medium,
            ),
            StrideCategory::ElevationOfPrivilege => (
                "Enforce least privilege and separation of duties",
                ImplementationEffort::High,
            ),
        };
        Ok(vec![MitigationRecommendation {
            recommendation_id: format!("MIT-{category:?}"),
            title: title.to_string(),
            description: format!("Mitigation for {category:?} threats: {title}"),
            priority: MitigationPriority::High,
            estimated_effort: effort,
        }])
    }

    fn generate_attack_tree_mitigations(
        &self,
        tree: &AttackTree,
    ) -> Result<Vec<MitigationRecommendation>, ThreatModelingError> {
        let mut recs = Vec::new();
        if tree.success_probability > 0.5 {
            recs.push(MitigationRecommendation {
                recommendation_id: format!("MIT-TREE-{}", tree.tree_id),
                title: "Reduce attack tree success probability".to_string(),
                description: format!(
                    "Attack tree '{}' has a high success probability ({:.2}); add compensating controls at high-probability nodes",
                    tree.tree_id, tree.success_probability
                ),
                priority: MitigationPriority::High,
                estimated_effort: ImplementationEffort::Medium,
            });
        }
        if tree.detection_probability < 0.4 {
            recs.push(MitigationRecommendation {
                recommendation_id: format!("MIT-TREE-DETECT-{}", tree.tree_id),
                title: "Improve attack detection coverage".to_string(),
                description: format!(
                    "Attack tree '{}' has low detection probability ({:.2}); add monitoring at key nodes",
                    tree.tree_id, tree.detection_probability
                ),
                priority: MitigationPriority::Medium,
                estimated_effort: ImplementationEffort::Low,
            });
        }
        Ok(recs)
    }

    fn calculate_data_quality_confidence(&self) -> Result<f64, ThreatModelingError> {
        let detector_count = self.stride_analyzer.spoofing_detectors.len()
            + self.stride_analyzer.tampering_detectors.len()
            + self.stride_analyzer.repudiation_detectors.len()
            + self.stride_analyzer.information_disclosure_detectors.len()
            + self.stride_analyzer.denial_of_service_detectors.len()
            + self.stride_analyzer.elevation_of_privilege_detectors.len();
        Ok(if detector_count >= 6 {
            0.85
        } else {
            0.4 + detector_count as f64 * 0.05
        })
    }

    fn calculate_intelligence_confidence(&self) -> Result<f64, ThreatModelingError> {
        let source_count = self.threat_intelligence.intelligence_sources.len();
        Ok(if source_count == 0 {
            0.3
        } else {
            0.5 + 0.1 * source_count.min(5) as f64
        })
    }

    fn calculate_model_completeness_confidence(&self) -> Result<f64, ThreatModelingError> {
        Ok(if self.attack_tree_generator.attack_patterns.is_empty() {
            0.5
        } else {
            0.9
        })
    }

    fn calculate_temporal_confidence(&self) -> Result<f64, ThreatModelingError> {
        let staleness = self.modeling_config.cache_duration.as_secs_f64() / (86400.0 * 30.0);
        Ok((1.0 - staleness.min(1.0)).max(0.1))
    }

    fn score_to_threat_severity(score: f64) -> ThreatSeverity {
        match score {
            s if s >= 7.5 => ThreatSeverity::Critical,
            s if s >= 5.0 => ThreatSeverity::High,
            s if s >= 2.5 => ThreatSeverity::Medium,
            _ => ThreatSeverity::Low,
        }
    }

    /// Distill the STRIDE analysis and attack vector assessment into a flat list of
    /// concrete, actionable threats. One entry is produced per STRIDE category that scored
    /// above zero, plus one per attack vector whose combined success/impact exceeds a
    /// notability threshold.
    fn extract_identified_threats(
        &self,
        stride_analysis: &StrideAnalysisResult,
        attack_vectors: &[AttackVector],
    ) -> Vec<IdentifiedThreat> {
        let mut threats = Vec::new();
        for (category, score) in &stride_analysis.stride_scores {
            if *score <= 0.0 {
                continue;
            }
            let (strategy, complexity) = match category {
                StrideCategory::Spoofing => (
                    "Strengthen identity verification and authentication",
                    ImplementationEffort::Medium,
                ),
                StrideCategory::Tampering => (
                    "Add integrity checks and input validation",
                    ImplementationEffort::Medium,
                ),
                StrideCategory::Repudiation => (
                    "Implement comprehensive audit logging",
                    ImplementationEffort::Low,
                ),
                StrideCategory::InformationDisclosure => (
                    "Encrypt sensitive data and classify access",
                    ImplementationEffort::High,
                ),
                StrideCategory::DenialOfService => (
                    "Add rate limiting and resource quotas",
                    ImplementationEffort::Medium,
                ),
                StrideCategory::ElevationOfPrivilege => (
                    "Enforce least privilege and privilege separation",
                    ImplementationEffort::High,
                ),
            };
            threats.push(IdentifiedThreat {
                id: format!("THREAT-{category:?}"),
                name: format!("{category:?} risk"),
                severity: Self::score_to_threat_severity(*score),
                mitigation_strategy: strategy.to_string(),
                mitigation_complexity: complexity,
                mitigation_dependencies: Vec::new(),
            });
        }
        for vector in attack_vectors {
            if vector.success_probability * vector.impact_potential > 0.3 {
                threats.push(IdentifiedThreat {
                    id: format!("THREAT-VECTOR-{}", vector.vector_id),
                    name: vector.name.clone(),
                    severity: Self::score_to_threat_severity(vector.success_probability * 10.0),
                    mitigation_strategy: format!("Mitigate attack vector: {}", vector.description),
                    mitigation_complexity: if vector.mitigation_complexity > 0.6 {
                        ImplementationEffort::High
                    } else {
                        ImplementationEffort::Medium
                    },
                    mitigation_dependencies: vector.prerequisites.clone(),
                });
            }
        }
        threats
    }

    /// Weighted combination of the overall STRIDE rating and the mean attack-vector risk
    /// contribution (success probability times impact potential), scaled to a 0.0-10.0 range.
    fn calculate_overall_risk_score(
        &self,
        stride_analysis: &StrideAnalysisResult,
        attack_vectors: &[AttackVector],
    ) -> f64 {
        let vector_component = if attack_vectors.is_empty() {
            0.0
        } else {
            attack_vectors
                .iter()
                .map(|v| v.success_probability * v.impact_potential)
                .sum::<f64>()
                / attack_vectors.len() as f64
                * 10.0
        };
        (stride_analysis.overall_stride_rating * 0.6 + vector_component * 0.4).clamp(0.0, 10.0)
    }

    fn build_timeline(scale: f64) -> ThreatTimeline {
        ThreatTimeline {
            reconnaissance_phase: Duration::from_secs((3600.0 * scale) as u64),
            initial_access_phase: Duration::from_secs((1800.0 * scale) as u64),
            persistence_phase: Duration::from_secs((7200.0 * scale) as u64),
            privilege_escalation_phase: Duration::from_secs((3600.0 * scale) as u64),
            lateral_movement_phase: Duration::from_secs((5400.0 * scale) as u64),
            data_collection_phase: Duration::from_secs((3600.0 * scale) as u64),
            exfiltration_phase: Duration::from_secs((1800.0 * scale) as u64),
            cleanup_phase: Duration::from_secs((900.0 * scale) as u64),
        }
    }

    fn build_actor(actor_type: ThreatActorType, sophistication: f64) -> ThreatActor {
        ThreatActor {
            actor_id: format!("actor_{actor_type:?}"),
            name: format!("{actor_type:?} adversary"),
            actor_type,
            motivation: vec!["financial_gain".to_string()],
            capabilities: ThreatCapabilities {
                technical_sophistication: sophistication,
                resource_availability: sophistication,
                stealth_capability: sophistication * 0.8,
                persistence_capability: sophistication * 0.9,
                social_engineering_skills: sophistication * 0.6,
                zero_day_access: sophistication > 0.8,
                insider_access: false,
            },
            resources: ThreatResources {
                financial_resources: sophistication * 100.0,
                technical_resources: sophistication * 100.0,
                human_resources: sophistication * 10.0,
                time_resources: sophistication * 1000.0,
            },
            target_preferences: Vec::new(),
            attack_patterns: Vec::new(),
            geographic_focus: Vec::new(),
        }
    }

    fn build_vector(
        id: &str,
        description: &str,
        success_probability: f64,
        detection_difficulty: f64,
    ) -> AttackVector {
        AttackVector {
            vector_id: id.to_string(),
            name: description.to_string(),
            description: description.to_string(),
            attack_surface: AttackSurface {
                network_surface: Vec::new(),
                application_surface: Vec::new(),
                physical_surface: Vec::new(),
                human_surface: Vec::new(),
            },
            entry_points: Vec::new(),
            prerequisites: Vec::new(),
            attack_steps: Vec::new(),
            success_probability,
            detection_difficulty,
            impact_potential: success_probability,
            mitigation_complexity: 0.5,
            vector_variants: Vec::new(),
        }
    }
}

impl StrideAnalyzer {
    pub fn new() -> Self {
        Self {
            spoofing_detectors: Self::initialize_spoofing_detectors(),
            tampering_detectors: Self::initialize_tampering_detectors(),
            repudiation_detectors: Self::initialize_repudiation_detectors(),
            information_disclosure_detectors: Self::initialize_information_disclosure_detectors(),
            denial_of_service_detectors: Self::initialize_denial_of_service_detectors(),
            elevation_of_privilege_detectors: Self::initialize_elevation_of_privilege_detectors(),
            stride_weights: Self::initialize_stride_weights(),
            contextual_analyzers: HashMap::new(),
        }
    }

    fn initialize_spoofing_detectors() -> Vec<SpoofingDetector> {
        vec![SpoofingDetector {
            name: "Identity Spoofing Detector".to_string(),
            detection_patterns: vec![
                "weak_authentication".to_string(),
                "missing_identity_verification".to_string(),
                "user_impersonation_risk".to_string(),
            ],
            trait_specific_checks: HashMap::new(),
            identity_verification_requirements: vec![
                "multi_factor_authentication".to_string(),
                "digital_certificates".to_string(),
                "biometric_verification".to_string(),
            ],
            authentication_bypass_patterns: vec![
                "default_credentials".to_string(),
                "credential_stuffing".to_string(),
                "session_hijacking".to_string(),
            ],
            spoofing_indicators: Vec::new(),
        }]
    }

    fn initialize_tampering_detectors() -> Vec<TamperingDetector> {
        vec![TamperingDetector {
            name: "Data Integrity Detector".to_string(),
            integrity_checks: vec![IntegrityCheck {
                check_type: "checksum_verification".to_string(),
                algorithm: "sha256".to_string(),
                scope: "data_in_transit".to_string(),
            }],
            modification_patterns: vec![
                "unauthorized_data_modification".to_string(),
                "code_injection".to_string(),
                "parameter_tampering".to_string(),
            ],
            data_tampering_vectors: vec![
                "man_in_the_middle".to_string(),
                "database_manipulation".to_string(),
                "file_system_modification".to_string(),
            ],
            code_injection_patterns: vec![
                "sql_injection".to_string(),
                "xss_injection".to_string(),
                "command_injection".to_string(),
            ],
            tampering_indicators: Vec::new(),
        }]
    }

    fn initialize_repudiation_detectors() -> Vec<RepudiationDetector> {
        vec![RepudiationDetector {
            name: "Non-Repudiation Detector".to_string(),
            audit_trail_requirements: vec![
                "comprehensive_logging".to_string(),
                "tamper_evident_logs".to_string(),
                "digital_signatures".to_string(),
            ],
            non_repudiation_mechanisms: vec![
                "digital_signatures".to_string(),
                "timestamping_services".to_string(),
                "audit_trails".to_string(),
            ],
            logging_patterns: vec![
                "transaction_logging".to_string(),
                "access_logging".to_string(),
                "error_logging".to_string(),
            ],
            evidence_collection_methods: vec![
                "forensic_imaging".to_string(),
                "chain_of_custody".to_string(),
                "witness_testimony".to_string(),
            ],
            repudiation_risks: Vec::new(),
        }]
    }

    fn initialize_information_disclosure_detectors() -> Vec<InformationDisclosureDetector> {
        vec![InformationDisclosureDetector {
            name: "Data Leakage Detector".to_string(),
            data_leakage_patterns: vec![
                "sensitive_data_exposure".to_string(),
                "information_disclosure".to_string(),
                "data_exfiltration".to_string(),
            ],
            privacy_violations: vec![
                "personal_data_exposure".to_string(),
                "unauthorized_access".to_string(),
                "privacy_breach".to_string(),
            ],
            information_exposure_vectors: vec![
                "error_messages".to_string(),
                "debug_information".to_string(),
                "configuration_files".to_string(),
            ],
            data_classification_requirements: vec![
                "confidential".to_string(),
                "restricted".to_string(),
                "public".to_string(),
            ],
            disclosure_indicators: Vec::new(),
        }]
    }

    fn initialize_denial_of_service_detectors() -> Vec<DenialOfServiceDetector> {
        vec![DenialOfServiceDetector {
            name: "Resource Exhaustion Detector".to_string(),
            resource_exhaustion_patterns: vec![
                "cpu_exhaustion".to_string(),
                "memory_exhaustion".to_string(),
                "network_flooding".to_string(),
            ],
            availability_requirements: vec![
                "99.9_percent_uptime".to_string(),
                "load_balancing".to_string(),
                "failover_mechanisms".to_string(),
            ],
            dos_vectors: vec![
                "distributed_dos".to_string(),
                "amplification_attacks".to_string(),
                "resource_consumption".to_string(),
            ],
            rate_limiting_requirements: vec![
                "request_throttling".to_string(),
                "connection_limits".to_string(),
                "bandwidth_limits".to_string(),
            ],
            dos_indicators: Vec::new(),
        }]
    }

    fn initialize_elevation_of_privilege_detectors() -> Vec<ElevationOfPrivilegeDetector> {
        vec![ElevationOfPrivilegeDetector {
            name: "Privilege Escalation Detector".to_string(),
            privilege_escalation_patterns: vec![
                "vertical_escalation".to_string(),
                "horizontal_escalation".to_string(),
                "role_confusion".to_string(),
            ],
            access_control_requirements: vec![
                "role_based_access".to_string(),
                "principle_of_least_privilege".to_string(),
                "mandatory_access_control".to_string(),
            ],
            authorization_bypass_patterns: vec![
                "direct_object_reference".to_string(),
                "path_traversal".to_string(),
                "privilege_escalation".to_string(),
            ],
            privilege_boundaries: vec![
                "user_space".to_string(),
                "kernel_space".to_string(),
                "administrative_space".to_string(),
            ],
            escalation_indicators: Vec::new(),
        }]
    }

    fn initialize_stride_weights() -> HashMap<StrideCategory, f64> {
        let mut weights = HashMap::new();
        weights.insert(StrideCategory::Spoofing, 1.0);
        weights.insert(StrideCategory::Tampering, 1.0);
        weights.insert(StrideCategory::Repudiation, 1.0);
        weights.insert(StrideCategory::InformationDisclosure, 1.0);
        weights.insert(StrideCategory::DenialOfService, 1.0);
        weights.insert(StrideCategory::ElevationOfPrivilege, 1.0);
        weights
    }
}

impl Default for StrideAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl AttackTreeGenerator {
    pub fn new() -> Self {
        Self {
            attack_patterns: HashMap::new(),
            tree_templates: Vec::new(),
            node_generators: HashMap::new(),
            tree_optimization: AttackTreeOptimization::new(),
            probability_calculators: Vec::new(),
            cost_benefit_analyzers: Vec::new(),
        }
    }
}

impl Default for AttackTreeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreatIntelligenceManager {
    pub fn new() -> Self {
        Self {
            intelligence_sources: Self::initialize_default_sources(),
            threat_feeds: HashMap::new(),
            indicators_of_compromise: Vec::new(),
            attack_campaigns: Vec::new(),
            threat_actor_profiles: HashMap::new(),
            intelligence_correlation: IntelligenceCorrelation::new(),
            feed_aggregator: FeedAggregator::new(),
            intelligence_scoring: IntelligenceScoring::new(),
        }
    }

    fn initialize_default_sources() -> Vec<ThreatIntelligenceSource> {
        vec![ThreatIntelligenceSource {
            source_id: "internal-heuristics".to_string(),
            name: "Internal Heuristic Feed".to_string(),
            reliability: 0.6,
            source_type: "heuristic".to_string(),
        }]
    }

    /// Combine independently-gathered insights into a single, higher-confidence correlated
    /// signal. Returns no additional insights when there is nothing to correlate.
    pub fn correlate_intelligence(
        &self,
        insights: &[IntelligenceInsight],
    ) -> Result<Vec<IntelligenceInsight>, ThreatModelingError> {
        if insights.len() < 2 {
            return Ok(Vec::new());
        }
        let average_confidence =
            insights.iter().map(|i| i.confidence).sum::<f64>() / insights.len() as f64;
        Ok(vec![IntelligenceInsight {
            insight_id: "correlated-0".to_string(),
            description: format!(
                "Correlated {} independent insight(s) into a single elevated-confidence signal",
                insights.len()
            ),
            confidence: (average_confidence * 1.1).min(1.0),
            relevance: insights.iter().map(|i| i.relevance).fold(0.0_f64, f64::max),
            source_ids: insights.iter().flat_map(|i| i.source_ids.clone()).collect(),
        }])
    }

    /// Summarize how many gathered insights clear the manager's configured confidence bar.
    pub fn score_intelligence(
        &self,
        insights: &[IntelligenceInsight],
    ) -> Result<Vec<IntelligenceInsight>, ThreatModelingError> {
        let threshold = self.intelligence_scoring.min_confidence_threshold;
        let high_confidence = insights
            .iter()
            .filter(|i| i.confidence >= threshold)
            .count();
        if high_confidence == 0 {
            return Ok(Vec::new());
        }
        Ok(vec![IntelligenceInsight {
            insight_id: "scoring-summary".to_string(),
            description: format!(
                "{high_confidence} insight(s) exceed the configured confidence threshold"
            ),
            confidence: threshold,
            relevance: 0.5,
            source_ids: Vec::new(),
        }])
    }
}

impl Default for ThreatIntelligenceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreatLandscapeAssessment {
    pub fn new() -> Self {
        Self {
            threat_environment: ThreatEnvironment::new(),
            emerging_threats: Vec::new(),
            threat_trends: Vec::new(),
            geographic_factors: HashMap::new(),
            industry_specific_threats: HashMap::new(),
            technology_threats: HashMap::new(),
            threat_evolution_models: Vec::new(),
            landscape_metrics: LandscapeMetrics::new(),
        }
    }
}

impl Default for ThreatLandscapeAssessment {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_threat_modeling_engine() -> ThreatModelingEngine {
    ThreatModelingEngine::new()
}

pub fn create_comprehensive_threat_model(
    context: &TraitUsageContext,
) -> Result<ThreatModelingResult, ThreatModelingError> {
    let mut engine = ThreatModelingEngine::new();
    engine.analyze_threats(context)
}

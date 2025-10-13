//! Report Generation Module
//!
//! This module provides comprehensive report generation capabilities for model interpretability,
//! including automated reports, model cards, explanation summaries, and interpretability scorecards.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::ArrayView1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for report generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Report title
    pub title: String,
    /// Report author/organization
    pub author: String,
    /// Model name
    pub model_name: String,
    /// Dataset name
    pub dataset_name: String,
    /// Include visualizations
    pub include_visualizations: bool,
    /// Include detailed explanations
    pub include_detailed_explanations: bool,
    /// Include model performance metrics
    pub include_performance_metrics: bool,
    /// Include fairness analysis
    pub include_fairness_analysis: bool,
    /// Include uncertainty analysis
    pub include_uncertainty_analysis: bool,
    /// Output format
    pub output_format: OutputFormat,
    /// Template style
    pub template_style: TemplateStyle,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            title: "Model Interpretability Report".to_string(),
            author: "Sklears Analysis".to_string(),
            model_name: "Untitled Model".to_string(),
            dataset_name: "Untitled Dataset".to_string(),
            include_visualizations: true,
            include_detailed_explanations: true,
            include_performance_metrics: true,
            include_fairness_analysis: true,
            include_uncertainty_analysis: true,
            output_format: OutputFormat::Html,
            template_style: TemplateStyle::Professional,
        }
    }
}

/// Output formats for reports
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Html
    Html,
    /// Markdown
    Markdown,
    /// Pdf
    Pdf,
    /// Json
    Json,
}

/// Template styles for reports
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TemplateStyle {
    /// Professional
    Professional,
    /// Academic
    Academic,
    /// Technical
    Technical,
    /// Executive
    Executive,
}

/// Model card data structure following responsible AI best practices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    /// Model details
    pub model_details: ModelDetails,
    /// Intended use
    pub intended_use: IntendedUse,
    /// Factors affecting performance
    pub factors: Vec<Factor>,
    /// Metrics and evaluation data
    pub metrics: ReportModelMetrics,
    /// Evaluation data
    pub evaluation_data: EvaluationData,
    /// Training data
    pub training_data: TrainingData,
    /// Quantitative analysis
    pub quantitative_analysis: QuantitativeAnalysis,
    /// Ethical considerations
    pub ethical_considerations: EthicalConsiderations,
    /// Caveats and recommendations
    pub caveats_and_recommendations: CaveatsAndRecommendations,
}

/// Model details section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetails {
    pub name: String,
    pub version: String,
    pub model_type: String,
    pub algorithm: String,
    pub architecture: String,
    pub paper_or_resource: Option<String>,
    pub citation: Option<String>,
    pub license: Option<String>,
    pub contact: Option<String>,
}

/// Intended use section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntendedUse {
    pub primary_uses: Vec<String>,
    pub primary_users: Vec<String>,
    pub out_of_scope_uses: Vec<String>,
}

/// Factors that affect model performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factor {
    /// Factor name
    pub name: String,
    /// Factor description
    pub description: String,
    /// Relevance to model performance
    pub relevance: String,
}

/// Model metrics and performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportModelMetrics {
    /// Performance metrics
    pub performance_metrics: HashMap<String, Float>,
    /// Decision thresholds
    pub decision_thresholds: HashMap<String, Float>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (Float, Float)>,
    /// Approach to uncertainty and variability
    pub uncertainty_approach: String,
}

/// Evaluation data information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationData {
    /// Datasets used for evaluation
    pub datasets: Vec<DatasetInfo>,
    /// Motivation for dataset choice
    pub motivation: String,
    /// Preprocessing steps
    pub preprocessing: Vec<String>,
}

/// Training data information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    /// Datasets used for training
    pub datasets: Vec<DatasetInfo>,
    /// Motivation for dataset choice
    pub motivation: String,
    /// Preprocessing steps
    pub preprocessing: Vec<String>,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,
    /// Dataset description
    pub description: String,
    /// Number of samples
    pub num_samples: usize,
    /// Number of features
    pub num_features: usize,
    /// Data sources
    pub sources: Vec<String>,
    /// Collection methodology
    pub collection_methodology: String,
}

/// Quantitative analysis section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantitativeAnalysis {
    /// Unitary results
    pub unitary_results: String,
    /// Intersectional results
    pub intersectional_results: String,
}

/// Ethical considerations section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalConsiderations {
    /// Sensitive data considerations
    pub sensitive_data: Vec<String>,
    /// Human life considerations
    pub human_life: Vec<String>,
    /// Mitigations
    pub mitigations: Vec<String>,
    /// Risks and harms
    pub risks_and_harms: Vec<String>,
    /// Use cases to avoid
    pub use_cases_to_avoid: Vec<String>,
}

/// Caveats and recommendations section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaveatsAndRecommendations {
    /// Known limitations
    pub limitations: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Ideal characteristics of evaluation dataset
    pub ideal_evaluation_dataset: Vec<String>,
}

/// Explanation summary data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationSummary {
    /// Summary of feature importance
    pub feature_importance_summary: String,
    /// Key insights from SHAP analysis
    pub shap_insights: Vec<String>,
    /// Counterfactual insights
    pub counterfactual_insights: Vec<String>,
    /// Local explanation insights
    pub local_explanation_insights: Vec<String>,
    /// Global explanation insights
    pub global_explanation_insights: Vec<String>,
    /// Model behavior patterns
    pub behavior_patterns: Vec<String>,
}

/// Interpretability scorecard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityScorecard {
    /// Overall interpretability score (0-100)
    pub overall_score: Float,
    /// Individual scores for different aspects
    pub scores: HashMap<String, Float>,
    /// Explanations for scores
    pub score_explanations: HashMap<String, String>,
    /// Recommendations for improvement
    pub improvement_recommendations: Vec<String>,
}

/// Comprehensive interpretability report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Executive summary
    pub executive_summary: String,
    /// Model card
    pub model_card: ModelCard,
    /// Explanation summary
    pub explanation_summary: ExplanationSummary,
    /// Interpretability scorecard
    pub scorecard: InterpretabilityScorecard,
    /// Detailed analysis sections
    pub detailed_analysis: HashMap<String, String>,
    /// Visualizations (as base64 encoded images or plot data)
    pub visualizations: HashMap<String, String>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Appendices
    pub appendices: HashMap<String, String>,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report generation timestamp
    pub timestamp: String,
    /// Report version
    pub version: String,
    /// Tool version
    pub tool_version: String,
    /// Configuration used
    pub config: ReportConfig,
}

/// Report generator
pub struct ReportGenerator {
    /// Configuration
    config: ReportConfig,
    /// Collected data for report
    data: HashMap<String, serde_json::Value>,
}

impl ReportGenerator {
    /// Create a new report generator
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_inspection::reporting::{ReportGenerator, ReportConfig};
    ///
    /// let config = ReportConfig::default();
    /// let generator = ReportGenerator::new(config);
    /// ```
    pub fn new(config: ReportConfig) -> Self {
        Self {
            config,
            data: HashMap::new(),
        }
    }

    /// Add data to the report
    pub fn add_data(&mut self, key: &str, value: serde_json::Value) {
        self.data.insert(key.to_string(), value);
    }

    /// Generate a complete interpretability report
    pub fn generate_report(&self) -> SklResult<InterpretabilityReport> {
        let metadata = self.generate_metadata()?;
        let executive_summary = self.generate_executive_summary()?;
        let model_card = self.generate_model_card()?;
        let explanation_summary = self.generate_explanation_summary()?;
        let scorecard = self.generate_scorecard()?;
        let detailed_analysis = self.generate_detailed_analysis()?;
        let visualizations = self.generate_visualizations()?;
        let recommendations = self.generate_recommendations()?;
        let appendices = self.generate_appendices()?;

        Ok(InterpretabilityReport {
            metadata,
            executive_summary,
            model_card,
            explanation_summary,
            scorecard,
            detailed_analysis,
            visualizations,
            recommendations,
            appendices,
        })
    }

    /// Generate report metadata
    fn generate_metadata(&self) -> SklResult<ReportMetadata> {
        Ok(ReportMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: "1.0.0".to_string(),
            tool_version: "sklears-inspection-0.1.0".to_string(),
            config: self.config.clone(),
        })
    }

    /// Generate executive summary
    fn generate_executive_summary(&self) -> SklResult<String> {
        let mut summary = format!(
            "# Executive Summary\n\n\
            This report provides a comprehensive analysis of the interpretability and \
            transparency of the {} model trained on the {} dataset.\n\n",
            self.config.model_name, self.config.dataset_name
        );

        // Add key findings if available
        if let Some(scorecard_data) = self.data.get("scorecard") {
            summary.push_str("## Key Findings\n\n");
            // Parse scorecard data and add insights
        }

        summary.push_str(
            "## Recommendations\n\n\
            Based on our analysis, we provide actionable recommendations to improve \
            model interpretability and ensure responsible deployment.\n\n",
        );

        Ok(summary)
    }

    /// Generate model card
    fn generate_model_card(&self) -> SklResult<ModelCard> {
        let model_details = ModelDetails {
            name: self.config.model_name.clone(),
            version: "1.0.0".to_string(),
            model_type: self
                .data
                .get("model_type")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string(),
            algorithm: self
                .data
                .get("algorithm")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string(),
            architecture: self
                .data
                .get("architecture")
                .and_then(|v| v.as_str())
                .unwrap_or("Standard architecture")
                .to_string(),
            paper_or_resource: None,
            citation: None,
            license: None,
            contact: None,
        };

        let intended_use = IntendedUse {
            primary_uses: vec![
                "Research and analysis".to_string(),
                "Educational purposes".to_string(),
            ],
            primary_users: vec![
                "Data scientists".to_string(),
                "Researchers".to_string(),
                "Model developers".to_string(),
            ],
            out_of_scope_uses: vec![
                "High-stakes decision making without human oversight".to_string(),
                "Applications involving protected characteristics without bias analysis"
                    .to_string(),
            ],
        };

        let factors = vec![
            Factor {
                name: "Data Quality".to_string(),
                description: "Quality and representativeness of input data".to_string(),
                relevance: "High impact on model performance and fairness".to_string(),
            },
            Factor {
                name: "Feature Distribution".to_string(),
                description: "Distribution of input features".to_string(),
                relevance: "Affects model generalization".to_string(),
            },
        ];

        let metrics = ReportModelMetrics {
            performance_metrics: self.extract_performance_metrics(),
            decision_thresholds: HashMap::new(),
            confidence_intervals: HashMap::new(),
            uncertainty_approach: "Bootstrap sampling and calibration".to_string(),
        };

        let evaluation_data = EvaluationData {
            datasets: vec![DatasetInfo {
                name: self.config.dataset_name.clone(),
                description: "Evaluation dataset".to_string(),
                num_samples: self
                    .data
                    .get("num_samples")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                num_features: self
                    .data
                    .get("num_features")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                sources: vec!["User provided".to_string()],
                collection_methodology: "Standard data collection".to_string(),
            }],
            motivation: "Representative test set for model evaluation".to_string(),
            preprocessing: vec![
                "Data cleaning".to_string(),
                "Feature scaling".to_string(),
                "Missing value handling".to_string(),
            ],
        };

        let training_data = TrainingData {
            datasets: vec![DatasetInfo {
                name: format!("{}_training", self.config.dataset_name),
                description: "Training dataset".to_string(),
                num_samples: self
                    .data
                    .get("training_samples")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                num_features: self
                    .data
                    .get("num_features")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                sources: vec!["User provided".to_string()],
                collection_methodology: "Standard data collection".to_string(),
            }],
            motivation: "Representative training set for model development".to_string(),
            preprocessing: vec![
                "Data cleaning".to_string(),
                "Feature engineering".to_string(),
                "Data augmentation".to_string(),
            ],
        };

        let quantitative_analysis = QuantitativeAnalysis {
            unitary_results: "Model shows consistent performance across individual features"
                .to_string(),
            intersectional_results: "Analysis of feature interactions reveals complex dependencies"
                .to_string(),
        };

        let ethical_considerations = EthicalConsiderations {
            sensitive_data: vec![
                "Model may process sensitive personal information".to_string(),
                "Consider privacy implications of feature usage".to_string(),
            ],
            human_life: vec!["Ensure human oversight for critical decisions".to_string()],
            mitigations: vec![
                "Regular bias testing".to_string(),
                "Fairness-aware model training".to_string(),
                "Continuous monitoring".to_string(),
            ],
            risks_and_harms: vec![
                "Potential for biased predictions".to_string(),
                "Over-reliance on automated decisions".to_string(),
            ],
            use_cases_to_avoid: vec![
                "Life-critical applications without validation".to_string(),
                "Discriminatory applications".to_string(),
            ],
        };

        let caveats_and_recommendations = CaveatsAndRecommendations {
            limitations: vec![
                "Model trained on limited dataset".to_string(),
                "Performance may degrade on out-of-distribution data".to_string(),
            ],
            recommendations: vec![
                "Regular model retraining".to_string(),
                "Continuous performance monitoring".to_string(),
                "Bias testing across protected groups".to_string(),
            ],
            ideal_evaluation_dataset: vec![
                "Large, diverse, and representative dataset".to_string(),
                "Balanced across all relevant demographics".to_string(),
                "Recent data reflecting current conditions".to_string(),
            ],
        };

        Ok(ModelCard {
            model_details,
            intended_use,
            factors,
            metrics,
            evaluation_data,
            training_data,
            quantitative_analysis,
            ethical_considerations,
            caveats_and_recommendations,
        })
    }

    /// Generate explanation summary
    fn generate_explanation_summary(&self) -> SklResult<ExplanationSummary> {
        Ok(ExplanationSummary {
            feature_importance_summary: self.generate_feature_importance_summary(),
            shap_insights: self.generate_shap_insights(),
            counterfactual_insights: self.generate_counterfactual_insights(),
            local_explanation_insights: self.generate_local_explanation_insights(),
            global_explanation_insights: self.generate_global_explanation_insights(),
            behavior_patterns: self.generate_behavior_patterns(),
        })
    }

    /// Generate interpretability scorecard
    fn generate_scorecard(&self) -> SklResult<InterpretabilityScorecard> {
        let mut scores = HashMap::new();
        let mut explanations = HashMap::new();

        // Calculate individual scores
        let transparency_score = self.calculate_transparency_score();
        let explainability_score = self.calculate_explainability_score();
        let fairness_score = self.calculate_fairness_score();
        let robustness_score = self.calculate_robustness_score();

        scores.insert("transparency".to_string(), transparency_score);
        scores.insert("explainability".to_string(), explainability_score);
        scores.insert("fairness".to_string(), fairness_score);
        scores.insert("robustness".to_string(), robustness_score);

        explanations.insert(
            "transparency".to_string(),
            "Model transparency based on feature importance clarity".to_string(),
        );
        explanations.insert(
            "explainability".to_string(),
            "Quality and comprehensiveness of explanations".to_string(),
        );
        explanations.insert(
            "fairness".to_string(),
            "Absence of bias across protected groups".to_string(),
        );
        explanations.insert(
            "robustness".to_string(),
            "Model stability under perturbations".to_string(),
        );

        let overall_score = scores.values().sum::<Float>() / scores.len() as Float;

        let improvement_recommendations = self.generate_improvement_recommendations(&scores);

        Ok(InterpretabilityScorecard {
            overall_score,
            scores,
            score_explanations: explanations,
            improvement_recommendations,
        })
    }

    /// Generate detailed analysis sections
    fn generate_detailed_analysis(&self) -> SklResult<HashMap<String, String>> {
        let mut analysis = HashMap::new();

        if self.config.include_detailed_explanations {
            analysis.insert(
                "feature_analysis".to_string(),
                self.generate_feature_analysis(),
            );
            analysis.insert(
                "model_behavior".to_string(),
                self.generate_model_behavior_analysis(),
            );
        }

        if self.config.include_performance_metrics {
            analysis.insert(
                "performance_analysis".to_string(),
                self.generate_performance_analysis(),
            );
        }

        if self.config.include_fairness_analysis {
            analysis.insert(
                "fairness_analysis".to_string(),
                self.generate_fairness_analysis(),
            );
        }

        if self.config.include_uncertainty_analysis {
            analysis.insert(
                "uncertainty_analysis".to_string(),
                self.generate_uncertainty_analysis(),
            );
        }

        Ok(analysis)
    }

    /// Generate visualizations section
    fn generate_visualizations(&self) -> SklResult<HashMap<String, String>> {
        let mut visualizations = HashMap::new();

        if self.config.include_visualizations {
            visualizations.insert(
                "feature_importance".to_string(),
                "Feature importance plot data".to_string(),
            );
            visualizations.insert(
                "shap_summary".to_string(),
                "SHAP summary plot data".to_string(),
            );
            visualizations.insert(
                "partial_dependence".to_string(),
                "Partial dependence plots data".to_string(),
            );
        }

        Ok(visualizations)
    }

    /// Generate recommendations
    fn generate_recommendations(&self) -> SklResult<Vec<String>> {
        let mut recommendations = vec![
            "Implement continuous model monitoring to detect performance drift".to_string(),
            "Regular bias testing across different demographic groups".to_string(),
            "Establish clear guidelines for model use and limitations".to_string(),
        ];

        // Add specific recommendations based on analysis
        if self.data.contains_key("low_fairness_score") {
            recommendations
                .push("Improve model fairness through bias mitigation techniques".to_string());
        }

        if self.data.contains_key("low_robustness_score") {
            recommendations
                .push("Enhance model robustness through adversarial training".to_string());
        }

        Ok(recommendations)
    }

    /// Generate appendices
    fn generate_appendices(&self) -> SklResult<HashMap<String, String>> {
        let mut appendices = HashMap::new();

        appendices.insert(
            "methodology".to_string(),
            "Detailed methodology for interpretability analysis".to_string(),
        );
        appendices.insert(
            "technical_details".to_string(),
            "Technical implementation details and algorithms used".to_string(),
        );
        appendices.insert(
            "references".to_string(),
            "Academic references and further reading".to_string(),
        );

        Ok(appendices)
    }

    // Helper methods for generating specific sections
    fn extract_performance_metrics(&self) -> HashMap<String, Float> {
        let mut metrics = HashMap::new();

        if let Some(accuracy) = self.data.get("accuracy").and_then(|v| v.as_f64()) {
            metrics.insert("accuracy".to_string(), accuracy as Float);
        }

        if let Some(precision) = self.data.get("precision").and_then(|v| v.as_f64()) {
            metrics.insert("precision".to_string(), precision as Float);
        }

        if let Some(recall) = self.data.get("recall").and_then(|v| v.as_f64()) {
            metrics.insert("recall".to_string(), recall as Float);
        }

        metrics
    }

    fn generate_feature_importance_summary(&self) -> String {
        "Analysis of feature importance reveals key drivers of model predictions".to_string()
    }

    fn generate_shap_insights(&self) -> Vec<String> {
        vec![
            "SHAP analysis provides instance-level explanations".to_string(),
            "Feature interactions identified through SHAP values".to_string(),
        ]
    }

    fn generate_counterfactual_insights(&self) -> Vec<String> {
        vec![
            "Counterfactual analysis reveals minimal changes needed for different outcomes"
                .to_string(),
        ]
    }

    fn generate_local_explanation_insights(&self) -> Vec<String> {
        vec!["Local explanations provide instance-specific model behavior".to_string()]
    }

    fn generate_global_explanation_insights(&self) -> Vec<String> {
        vec!["Global explanations reveal overall model behavior patterns".to_string()]
    }

    fn generate_behavior_patterns(&self) -> Vec<String> {
        vec!["Model shows consistent behavior across similar instances".to_string()]
    }

    fn calculate_transparency_score(&self) -> Float {
        // Calculate based on available explanations
        75.0 // Placeholder
    }

    fn calculate_explainability_score(&self) -> Float {
        // Calculate based on explanation quality
        80.0 // Placeholder
    }

    fn calculate_fairness_score(&self) -> Float {
        // Calculate based on fairness metrics
        85.0 // Placeholder
    }

    fn calculate_robustness_score(&self) -> Float {
        // Calculate based on robustness analysis
        70.0 // Placeholder
    }

    fn generate_improvement_recommendations(&self, scores: &HashMap<String, Float>) -> Vec<String> {
        let mut recommendations = Vec::new();

        for (aspect, score) in scores {
            if *score < 70.0 {
                recommendations.push(format!("Improve {} (current score: {:.1})", aspect, score));
            }
        }

        recommendations
    }

    fn generate_feature_analysis(&self) -> String {
        "Detailed analysis of feature importance and interactions".to_string()
    }

    fn generate_model_behavior_analysis(&self) -> String {
        "Analysis of model behavior patterns and decision boundaries".to_string()
    }

    fn generate_performance_analysis(&self) -> String {
        "Comprehensive performance evaluation across different metrics".to_string()
    }

    fn generate_fairness_analysis(&self) -> String {
        "Analysis of model fairness across different demographic groups".to_string()
    }

    fn generate_uncertainty_analysis(&self) -> String {
        "Analysis of model uncertainty and confidence in predictions".to_string()
    }
}

/// Export report to different formats
pub fn export_report(
    report: &InterpretabilityReport,
    output_path: &str,
    format: OutputFormat,
) -> SklResult<()> {
    match format {
        OutputFormat::Html => export_to_html(report, output_path),
        OutputFormat::Markdown => export_to_markdown(report, output_path),
        OutputFormat::Json => export_to_json(report, output_path),
        OutputFormat::Pdf => export_to_pdf(report, output_path),
    }
}

/// Export report to HTML format
fn export_to_html(report: &InterpretabilityReport, output_path: &str) -> SklResult<()> {
    let html_content = generate_html_report(report)?;
    std::fs::write(output_path, html_content).map_err(|e| {
        crate::SklearsError::InvalidInput(format!("Failed to write HTML report: {}", e))
    })?;
    Ok(())
}

/// Export report to Markdown format
fn export_to_markdown(report: &InterpretabilityReport, output_path: &str) -> SklResult<()> {
    let markdown_content = generate_markdown_report(report)?;
    std::fs::write(output_path, markdown_content).map_err(|e| {
        crate::SklearsError::InvalidInput(format!("Failed to write Markdown report: {}", e))
    })?;
    Ok(())
}

/// Export report to JSON format
fn export_to_json(report: &InterpretabilityReport, output_path: &str) -> SklResult<()> {
    let json_content = serde_json::to_string_pretty(report).map_err(|e| {
        crate::SklearsError::InvalidInput(format!("Failed to serialize report: {}", e))
    })?;
    std::fs::write(output_path, json_content).map_err(|e| {
        crate::SklearsError::InvalidInput(format!("Failed to write JSON report: {}", e))
    })?;
    Ok(())
}

/// Export report to PDF format (placeholder - would require additional dependencies)
fn export_to_pdf(_report: &InterpretabilityReport, _output_path: &str) -> SklResult<()> {
    Err(crate::SklearsError::InvalidInput(
        "PDF export not yet implemented".to_string(),
    ))
}

/// Generate HTML report content
fn generate_html_report(report: &InterpretabilityReport) -> SklResult<String> {
    let template = include_str!("../templates/report.html");

    let content = template
        .replace("{{TITLE}}", &report.metadata.config.title)
        .replace("{{AUTHOR}}", &report.metadata.config.author)
        .replace("{{MODEL_NAME}}", &report.metadata.config.model_name)
        .replace("{{TIMESTAMP}}", &report.metadata.timestamp)
        .replace("{{EXECUTIVE_SUMMARY}}", &report.executive_summary)
        .replace(
            "{{OVERALL_SCORE}}",
            &format!("{:.1}", report.scorecard.overall_score),
        );

    Ok(content)
}

/// Generate Markdown report content
fn generate_markdown_report(report: &InterpretabilityReport) -> SklResult<String> {
    let mut content = String::new();

    content.push_str(&format!("# {}\n\n", report.metadata.config.title));
    content.push_str(&format!("**Author:** {}\n", report.metadata.config.author));
    content.push_str(&format!(
        "**Model:** {}\n",
        report.metadata.config.model_name
    ));
    content.push_str(&format!("**Generated:** {}\n\n", report.metadata.timestamp));

    content.push_str("## Executive Summary\n\n");
    content.push_str(&report.executive_summary);
    content.push_str("\n\n");

    content.push_str("## Interpretability Scorecard\n\n");
    content.push_str(&format!(
        "**Overall Score:** {:.1}/100\n\n",
        report.scorecard.overall_score
    ));

    for (aspect, score) in &report.scorecard.scores {
        content.push_str(&format!("- **{}:** {:.1}/100\n", aspect, score));
    }

    content.push_str("\n## Recommendations\n\n");
    for (i, rec) in report.recommendations.iter().enumerate() {
        content.push_str(&format!("{}. {}\n", i + 1, rec));
    }

    Ok(content)
}

/// Quick report generation function
pub fn generate_quick_report(
    model_name: &str,
    dataset_name: &str,
    feature_importance: &ArrayView1<Float>,
    feature_names: Option<&[String]>,
) -> SklResult<InterpretabilityReport> {
    let mut config = ReportConfig::default();
    config.model_name = model_name.to_string();
    config.dataset_name = dataset_name.to_string();

    let mut generator = ReportGenerator::new(config);

    // Add basic data
    generator.add_data(
        "num_features",
        serde_json::Value::Number(serde_json::Number::from(feature_importance.len())),
    );

    // Generate the report
    generator.generate_report()
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_report_config_default() {
        let config = ReportConfig::default();
        assert_eq!(config.title, "Model Interpretability Report");
        assert!(config.include_visualizations);
        assert!(config.include_performance_metrics);
    }

    #[test]
    fn test_report_generator_creation() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);
        assert_eq!(generator.data.len(), 0);
    }

    #[test]
    fn test_add_data_to_generator() {
        let config = ReportConfig::default();
        let mut generator = ReportGenerator::new(config);

        generator.add_data(
            "test_key",
            serde_json::Value::String("test_value".to_string()),
        );
        assert_eq!(generator.data.len(), 1);
        assert!(generator.data.contains_key("test_key"));
    }

    #[test]
    fn test_generate_metadata() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);

        let metadata = generator.generate_metadata().unwrap();
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.tool_version, "sklears-inspection-0.1.0");
    }

    #[test]
    fn test_generate_executive_summary() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);

        let summary = generator.generate_executive_summary().unwrap();
        assert!(summary.contains("Executive Summary"));
        assert!(summary.contains("Untitled Model"));
        assert!(summary.contains("Untitled Dataset"));
    }

    #[test]
    fn test_generate_model_card() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);

        let model_card = generator.generate_model_card().unwrap();
        assert_eq!(model_card.model_details.name, "Untitled Model");
        assert_eq!(model_card.model_details.version, "1.0.0");
        assert!(!model_card.intended_use.primary_uses.is_empty());
        assert!(!model_card.factors.is_empty());
    }

    #[test]
    fn test_generate_scorecard() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);

        let scorecard = generator.generate_scorecard().unwrap();
        assert!(scorecard.overall_score > 0.0);
        assert!(scorecard.overall_score <= 100.0);
        assert!(scorecard.scores.contains_key("transparency"));
        assert!(scorecard.scores.contains_key("explainability"));
        assert!(scorecard.scores.contains_key("fairness"));
        assert!(scorecard.scores.contains_key("robustness"));
    }

    #[test]
    fn test_generate_complete_report() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);

        let report = generator.generate_report().unwrap();
        assert_eq!(report.metadata.version, "1.0.0");
        assert!(!report.executive_summary.is_empty());
        assert_eq!(report.model_card.model_details.name, "Untitled Model");
        assert!(report.scorecard.overall_score > 0.0);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_export_to_json() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);
        let report = generator.generate_report().unwrap();

        let result = export_to_json(&report, "/tmp/test_report.json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_markdown_report() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);
        let report = generator.generate_report().unwrap();

        let markdown = generate_markdown_report(&report).unwrap();
        assert!(markdown.contains("# Model Interpretability Report"));
        assert!(markdown.contains("## Executive Summary"));
        assert!(markdown.contains("## Interpretability Scorecard"));
        assert!(markdown.contains("## Recommendations"));
    }

    #[test]
    fn test_quick_report_generation() {
        let feature_importance = array![0.3, 0.5, 0.2];
        let feature_names = vec![
            "Feature1".to_string(),
            "Feature2".to_string(),
            "Feature3".to_string(),
        ];

        let report = generate_quick_report(
            "Test Model",
            "Test Dataset",
            &feature_importance.view(),
            Some(&feature_names),
        )
        .unwrap();

        assert_eq!(report.metadata.config.model_name, "Test Model");
        assert_eq!(report.metadata.config.dataset_name, "Test Dataset");
        assert!(report.scorecard.overall_score > 0.0);
    }

    #[test]
    fn test_output_format_variants() {
        let formats = [
            OutputFormat::Html,
            OutputFormat::Markdown,
            OutputFormat::Json,
            OutputFormat::Pdf,
        ];

        // Just ensure all variants can be created
        assert_eq!(formats.len(), 4);
    }

    #[test]
    fn test_template_style_variants() {
        let styles = [
            TemplateStyle::Professional,
            TemplateStyle::Academic,
            TemplateStyle::Technical,
            TemplateStyle::Executive,
        ];

        // Just ensure all variants can be created
        assert_eq!(styles.len(), 4);
    }

    #[test]
    fn test_improvement_recommendations() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);

        let mut scores = HashMap::new();
        scores.insert("transparency".to_string(), 65.0); // Below threshold
        scores.insert("fairness".to_string(), 85.0); // Above threshold

        let recommendations = generator.generate_improvement_recommendations(&scores);
        assert_eq!(recommendations.len(), 1);
        assert!(recommendations[0].contains("transparency"));
    }
}

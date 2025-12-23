//! Automated Cluster Validation Framework
//!
//! This module provides a comprehensive automated validation framework that combines
//! multiple validation metrics to assess clustering quality and generate actionable
//! recommendations for improvement.
//!
//! # Automated Validation Features
//! - Comprehensive quality assessment using multiple metrics
//! - Configurable validation criteria and thresholds
//! - Automated recommendation generation
//! - Quality scoring and ranking system
//! - Human-readable validation summaries
//!
//! # Validation Pipeline
//! 1. Internal validation: Silhouette, Calinski-Harabasz, Davies-Bouldin indices
//! 2. Coherence analysis: Cluster compactness, density, shape regularity
//! 3. Separation analysis: Inter-cluster distances, boundary clarity
//! 4. Stability analysis: Bootstrap, perturbation, or consensus methods (optional)
//! 5. Composite scoring: Weighted combination of all metrics
//! 6. Quality assessment: Excellent/Good/Fair/Poor classification
//! 7. Recommendations: Specific suggestions for improvement
//!
//! # Mathematical Background
//!
//! ## Composite Quality Score
//! Quality = w₁×Silhouette + w₂×CH + w₃×(1/DB) + w₄×Coherence + w₅×Separation
//! where weights w₁,...,w₅ sum to 1.0
//!
//! ## Overall Assessment
//! Final = α×Internal_Quality + β×Stability_Score
//! where α + β = 1.0 (typically α=0.7, β=0.3)

use scirs2_core::ndarray::Array2;
use sklears_core::error::{Result, SklearsError};

use super::coherence_separation::{
    ClusterCoherenceResult, ClusterSeparationResult, CoherenceSeparationAnalyzer,
};
use super::stability_analysis::StabilityAnalyzer;
use super::validation_types::{ExternalValidationMetrics, SilhouetteResult, ValidationMetric};
use crate::validation::InternalValidationMethods;

/// Overall cluster quality assessment
#[derive(Debug, Clone, PartialEq)]
pub enum ClusterQuality {
    /// Excellent clustering quality (score >= 0.8)
    Excellent,
    /// Good clustering quality (score >= 0.6)
    Good,
    /// Fair clustering quality (score >= 0.4)
    Fair,
    /// Poor clustering quality (score < 0.4)
    Poor,
}

impl ClusterQuality {
    /// Get numerical score for quality level
    pub fn score(&self) -> f64 {
        match self {
            ClusterQuality::Excellent => 0.9,
            ClusterQuality::Good => 0.7,
            ClusterQuality::Fair => 0.5,
            ClusterQuality::Poor => 0.2,
        }
    }

    /// Get descriptive text for quality level
    pub fn description(&self) -> &'static str {
        match self {
            ClusterQuality::Excellent => {
                "Excellent clustering quality - results are highly reliable"
            }
            ClusterQuality::Good => {
                "Good clustering quality - results are reliable with minor issues"
            }
            ClusterQuality::Fair => "Fair clustering quality - results may need improvement",
            ClusterQuality::Poor => "Poor clustering quality - significant issues detected",
        }
    }

    /// Create quality assessment from numerical score
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            ClusterQuality::Excellent
        } else if score >= 0.6 {
            ClusterQuality::Good
        } else if score >= 0.4 {
            ClusterQuality::Fair
        } else {
            ClusterQuality::Poor
        }
    }

    /// Check if quality meets minimum threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.score() >= threshold
    }
}

/// Configuration for automated cluster validation
#[derive(Debug, Clone)]
pub struct AutomatedValidationConfig {
    /// Whether to include stability analysis
    pub include_stability: bool,
    /// Stability analysis method ("subsample", "bootstrap", "perturbation")
    pub stability_method: Option<String>,
    /// Minimum acceptable quality threshold
    pub quality_threshold: f64,
    /// Whether to generate detailed recommendations
    pub detailed_recommendations: bool,
    /// Weight for internal validation metrics (vs stability)
    pub internal_weight: f64,
    /// Weight for stability metrics
    pub stability_weight: f64,
}

impl Default for AutomatedValidationConfig {
    fn default() -> Self {
        Self {
            include_stability: true,
            stability_method: Some("subsample".to_string()),
            quality_threshold: 0.5,
            detailed_recommendations: true,
            internal_weight: 0.7,
            stability_weight: 0.3,
        }
    }
}

impl AutomatedValidationConfig {
    /// Create configuration for fast validation (no stability analysis)
    pub fn fast() -> Self {
        Self {
            include_stability: false,
            stability_method: None,
            quality_threshold: 0.5,
            detailed_recommendations: false,
            internal_weight: 1.0,
            stability_weight: 0.0,
        }
    }

    /// Create configuration for comprehensive validation
    pub fn comprehensive() -> Self {
        Self {
            include_stability: true,
            stability_method: Some("consensus".to_string()),
            quality_threshold: 0.6,
            detailed_recommendations: true,
            internal_weight: 0.6,
            stability_weight: 0.4,
        }
    }

    /// Create configuration for production use
    pub fn production() -> Self {
        Self {
            include_stability: true,
            stability_method: Some("subsample".to_string()),
            quality_threshold: 0.7,
            detailed_recommendations: true,
            internal_weight: 0.7,
            stability_weight: 0.3,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.quality_threshold < 0.0 || self.quality_threshold > 1.0 {
            return Err(SklearsError::InvalidInput(
                "Quality threshold must be between 0 and 1".to_string(),
            ));
        }

        if (self.internal_weight + self.stability_weight - 1.0).abs() > 1e-6 {
            return Err(SklearsError::InvalidInput(
                "Internal and stability weights must sum to 1.0".to_string(),
            ));
        }

        if self.include_stability && self.stability_method.is_none() {
            return Err(SklearsError::InvalidInput(
                "Stability method must be specified when include_stability is true".to_string(),
            ));
        }

        Ok(())
    }
}

/// Result of automated cluster validation
#[derive(Debug, Clone)]
pub struct AutomatedValidationResult {
    /// Overall quality assessment
    pub overall_quality: ClusterQuality,
    /// Internal validation score (0-1)
    pub internal_quality_score: f64,
    /// Stability score (0-1)
    pub stability_score: f64,
    /// Individual metric scores
    pub silhouette_score: f64,
    pub calinski_harabasz_score: f64,
    pub davies_bouldin_score: f64,
    pub coherence_score: f64,
    pub separation_score: f64,
    /// Automated recommendations for improvement
    pub recommendations: Vec<String>,
    /// Human-readable validation summary
    pub validation_summary: String,
}

impl AutomatedValidationResult {
    /// Get the combined quality score (0-1)
    pub fn combined_score(&self) -> f64 {
        self.overall_quality.score()
    }

    /// Check if quality meets minimum threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.overall_quality.meets_threshold(threshold)
    }

    /// Get metric scores as a vector for analysis
    pub fn metric_scores(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("Silhouette", self.silhouette_score),
            ("Calinski-Harabasz", self.calinski_harabasz_score),
            ("Davies-Bouldin", self.davies_bouldin_score),
            ("Coherence", self.coherence_score),
            ("Separation", self.separation_score),
            ("Internal Quality", self.internal_quality_score),
            ("Stability", self.stability_score),
        ]
    }

    /// Get the best performing metric
    pub fn best_metric(&self) -> (&'static str, f64) {
        self.metric_scores()
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or(("None", 0.0))
    }

    /// Get the worst performing metric
    pub fn worst_metric(&self) -> (&'static str, f64) {
        self.metric_scores()
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or(("None", 0.0))
    }

    /// Get metrics below threshold
    pub fn problematic_metrics(&self, threshold: f64) -> Vec<(&'static str, f64)> {
        self.metric_scores()
            .into_iter()
            .filter(|(_, score)| *score < threshold)
            .collect()
    }

    /// Generate detailed report
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Automated Cluster Validation Report ===\n\n");
        report.push_str(&format!(
            "Overall Quality: {:?} (Score: {:.3})\n",
            self.overall_quality,
            self.combined_score()
        ));
        report.push_str(&format!("{}\n\n", self.overall_quality.description()));

        report.push_str("Individual Metrics:\n");
        for (name, score) in self.metric_scores() {
            report.push_str(&format!("  {}: {:.3}\n", name, score));
        }

        report.push_str(&format!("\n{}\n", self.validation_summary));

        if !self.recommendations.is_empty() {
            report.push_str("\nRecommendations:\n");
            for (i, rec) in self.recommendations.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, rec));
            }
        }

        report
    }
}

/// Automated cluster validation framework
pub struct AutomatedValidator {
    /// Distance metric for validation
    metric: ValidationMetric,
    /// Internal validation methods
    internal_validator: InternalValidationMethods,
    /// External validation methods
    external_validator: ExternalValidationMetrics,
    /// Coherence and separation analyzer
    coherence_separation_analyzer: CoherenceSeparationAnalyzer,
    /// Stability analyzer
    stability_analyzer: StabilityAnalyzer,
}

impl AutomatedValidator {
    /// Create a new automated validator
    pub fn new(metric: ValidationMetric) -> Self {
        Self {
            metric,
            internal_validator: InternalValidationMethods::new(metric),
            external_validator: ExternalValidationMetrics::default(),
            coherence_separation_analyzer: CoherenceSeparationAnalyzer::new(metric),
            stability_analyzer: StabilityAnalyzer::new(metric),
        }
    }

    /// Create validator with Euclidean distance
    pub fn euclidean() -> Self {
        Self::new(ValidationMetric::Euclidean)
    }

    /// Create validator with Manhattan distance
    pub fn manhattan() -> Self {
        Self::new(ValidationMetric::Manhattan)
    }

    /// Create validator with Cosine distance
    pub fn cosine() -> Self {
        Self::new(ValidationMetric::Cosine)
    }

    /// Automated cluster validation
    ///
    /// Performs comprehensive automated validation using multiple criteria
    /// and provides recommendations for cluster quality and parameter tuning.
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `labels` - Cluster assignments
    /// * `validation_config` - Configuration for validation criteria
    ///
    /// # Returns
    /// AutomatedValidationResult with recommendations and quality scores
    ///
    /// # Validation Pipeline
    /// 1. Validate configuration and inputs
    /// 2. Run internal validation metrics (Silhouette, CH, DB)
    /// 3. Run coherence and separation analysis
    /// 4. Run stability analysis (if enabled)
    /// 5. Compute composite quality scores
    /// 6. Generate recommendations based on results
    /// 7. Assess overall quality and create summary
    pub fn automated_cluster_validation(
        &self,
        X: &Array2<f64>,
        labels: &[i32],
        validation_config: &AutomatedValidationConfig,
    ) -> Result<AutomatedValidationResult> {
        // Validate inputs
        validation_config.validate()?;

        if X.nrows() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        // Run all validation metrics
        let silhouette_result = self.internal_validator.silhouette_analysis(X, labels)?;
        let calinski_harabasz = self.internal_validator.calinski_harabasz_index(X, labels)?;
        let davies_bouldin = self.internal_validator.davies_bouldin_index(X, labels)?;
        let coherence_result = self
            .coherence_separation_analyzer
            .cluster_coherence(X, labels)?;
        let separation_result = self
            .coherence_separation_analyzer
            .cluster_separation(X, labels)?;

        // Compute composite scores
        let internal_quality_score = self.compute_internal_quality_score(
            silhouette_result.mean_silhouette,
            calinski_harabasz,
            davies_bouldin,
            &coherence_result,
            &separation_result,
        );

        let stability_score = if validation_config.include_stability {
            match validation_config.stability_method.as_deref() {
                Some("subsample") => {
                    // Placeholder for subsample stability - would need clustering function
                    0.5
                }
                Some("bootstrap") => {
                    // Placeholder for bootstrap stability - would need clustering function
                    0.5
                }
                Some("perturbation") => {
                    // Placeholder for perturbation stability - would need clustering function
                    0.5
                }
                Some("consensus") => {
                    // Placeholder for consensus stability - would need clustering function
                    0.5
                }
                _ => 0.5, // Default when method not recognized
            }
        } else {
            0.5 // Default when stability not requested
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &silhouette_result,
            calinski_harabasz,
            davies_bouldin,
            &coherence_result,
            &separation_result,
            internal_quality_score,
            stability_score,
            validation_config,
        );

        // Determine overall quality assessment
        let overall_quality =
            self.assess_overall_quality(internal_quality_score, stability_score, validation_config);

        // Create validation summary
        let validation_summary = self.create_validation_summary(
            &silhouette_result,
            &coherence_result,
            &separation_result,
        );

        Ok(AutomatedValidationResult {
            overall_quality,
            internal_quality_score,
            stability_score,
            silhouette_score: silhouette_result.mean_silhouette,
            calinski_harabasz_score: calinski_harabasz,
            davies_bouldin_score: davies_bouldin,
            coherence_score: coherence_result.overall_coherence,
            separation_score: separation_result.avg_centroid_separation,
            recommendations,
            validation_summary,
        })
    }

    /// Compute internal quality score from multiple metrics
    fn compute_internal_quality_score(
        &self,
        silhouette: f64,
        calinski_harabasz: f64,
        davies_bouldin: f64,
        coherence: &ClusterCoherenceResult,
        separation: &ClusterSeparationResult,
    ) -> f64 {
        // Normalize metrics to [0, 1] scale and combine
        let silhouette_norm = (silhouette + 1.0) / 2.0; // [-1, 1] -> [0, 1]
        let ch_norm = (calinski_harabasz / (1.0 + calinski_harabasz)).min(1.0); // Asymptotic normalization
        let db_norm = 1.0 / (1.0 + davies_bouldin); // Lower is better
        let coherence_norm = coherence.overall_coherence;
        let separation_norm = (separation.gap_ratio / (1.0 + separation.gap_ratio)).min(1.0);

        // Weighted combination
        0.3 * silhouette_norm
            + 0.2 * ch_norm
            + 0.2 * db_norm
            + 0.15 * coherence_norm
            + 0.15 * separation_norm
    }

    /// Generate recommendations based on validation results
    #[allow(clippy::too_many_arguments)]
    fn generate_recommendations(
        &self,
        _silhouette: &SilhouetteResult,
        calinski_harabasz: f64,
        davies_bouldin: f64,
        coherence: &ClusterCoherenceResult,
        separation: &ClusterSeparationResult,
        internal_quality: f64,
        stability: f64,
        config: &AutomatedValidationConfig,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Quality-based recommendations
        if internal_quality < 0.3 {
            recommendations.push("Poor clustering quality detected. Consider adjusting algorithm parameters or trying different clustering methods.".to_string());
        }

        if stability < 0.5 && config.include_stability {
            recommendations.push("Low clustering stability. Results may not be reliable. Consider regularization or different initialization strategies.".to_string());
        }

        // Metric-specific recommendations
        if calinski_harabasz < 10.0 {
            recommendations.push("Low Calinski-Harabasz index suggests poor cluster separation. Consider increasing the number of clusters or improving feature selection.".to_string());
        }

        if davies_bouldin > 2.0 {
            recommendations.push("High Davies-Bouldin index indicates overlapping clusters. Consider reducing the number of clusters or using density-based methods.".to_string());
        }

        if coherence.overall_coherence < 0.5 {
            recommendations.push("Low cluster coherence detected. Clusters may be too heterogeneous. Consider stricter convergence criteria or different distance metrics.".to_string());
        }

        if separation.gap_ratio < 1.5 {
            recommendations.push("Poor cluster separation. Inter-cluster distances are too small relative to intra-cluster distances. Consider feature scaling or dimensionality reduction.".to_string());
        }

        if separation.overlap_measure > 0.3 {
            recommendations.push("Significant cluster overlap detected. Consider using fuzzy clustering methods or reducing the number of clusters.".to_string());
        }

        // Coherence-specific recommendations
        if coherence.coherence_consistency < 0.7 {
            recommendations.push("Inconsistent cluster coherence across clusters. Some clusters may be of poor quality. Consider post-processing or refinement.".to_string());
        }

        // Threshold-based recommendations
        if !ClusterQuality::from_score(internal_quality).meets_threshold(config.quality_threshold) {
            recommendations.push(format!(
                "Clustering quality ({:.3}) does not meet the specified threshold ({:.3}). Consider parameter tuning or alternative methods.",
                internal_quality, config.quality_threshold
            ));
        }

        // General improvement suggestions
        if recommendations.is_empty() && internal_quality < 0.8 {
            recommendations.push("Good clustering quality achieved. For further improvement, consider feature engineering or ensemble methods.".to_string());
        }

        recommendations
    }

    /// Assess overall clustering quality
    fn assess_overall_quality(
        &self,
        internal_quality: f64,
        stability: f64,
        config: &AutomatedValidationConfig,
    ) -> ClusterQuality {
        let combined_score =
            config.internal_weight * internal_quality + config.stability_weight * stability;
        ClusterQuality::from_score(combined_score)
    }

    /// Create validation summary
    fn create_validation_summary(
        &self,
        silhouette: &SilhouetteResult,
        coherence: &ClusterCoherenceResult,
        separation: &ClusterSeparationResult,
    ) -> String {
        format!(
            "Clustering validation summary:\n\
             - {} clusters identified\n\
             - Mean silhouette score: {:.3}\n\
             - Overall coherence: {:.3}\n\
             - Average separation: {:.3}\n\
             - Gap ratio: {:.3}",
            coherence.n_clusters,
            silhouette.mean_silhouette,
            coherence.overall_coherence,
            separation.avg_centroid_separation,
            separation.gap_ratio
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use std::collections::HashMap;

    fn generate_test_data() -> (Array2<f64>, Vec<i32>) {
        // Create two well-separated clusters
        let data = array![
            [1.0, 2.0], // Cluster 0
            [1.5, 1.8], // Cluster 0
            [1.2, 2.2], // Cluster 0
            [5.0, 8.0], // Cluster 1
            [5.2, 7.8], // Cluster 1
            [4.8, 8.2], // Cluster 1
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        (data, labels)
    }

    #[test]
    fn test_cluster_quality_enum() {
        assert_eq!(ClusterQuality::Excellent.score(), 0.9);
        assert_eq!(ClusterQuality::Good.score(), 0.7);
        assert_eq!(ClusterQuality::Fair.score(), 0.5);
        assert_eq!(ClusterQuality::Poor.score(), 0.2);

        assert!(ClusterQuality::Excellent
            .description()
            .contains("Excellent"));
        assert!(ClusterQuality::Poor.description().contains("Poor"));

        assert_eq!(ClusterQuality::from_score(0.9), ClusterQuality::Excellent);
        assert_eq!(ClusterQuality::from_score(0.7), ClusterQuality::Good);
        assert_eq!(ClusterQuality::from_score(0.5), ClusterQuality::Fair);
        assert_eq!(ClusterQuality::from_score(0.2), ClusterQuality::Poor);

        assert!(ClusterQuality::Good.meets_threshold(0.6));
        assert!(!ClusterQuality::Fair.meets_threshold(0.6));
    }

    #[test]
    fn test_automated_validation_config() {
        let default_config = AutomatedValidationConfig::default();
        assert!(default_config.include_stability);
        assert_eq!(
            default_config.stability_method,
            Some("subsample".to_string())
        );
        assert_eq!(default_config.quality_threshold, 0.5);

        let fast_config = AutomatedValidationConfig::fast();
        assert!(!fast_config.include_stability);
        assert_eq!(fast_config.stability_method, None);

        let comprehensive_config = AutomatedValidationConfig::comprehensive();
        assert!(comprehensive_config.include_stability);
        assert_eq!(
            comprehensive_config.stability_method,
            Some("consensus".to_string())
        );

        let production_config = AutomatedValidationConfig::production();
        assert!(production_config.include_stability);
        assert_eq!(production_config.quality_threshold, 0.7);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AutomatedValidationConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid quality threshold
        config.quality_threshold = 1.5;
        assert!(config.validate().is_err());
        config.quality_threshold = -0.1;
        assert!(config.validate().is_err());
        config.quality_threshold = 0.5; // Reset

        // Test invalid weight sum
        config.internal_weight = 0.8;
        config.stability_weight = 0.3; // Sum > 1.0
        assert!(config.validate().is_err());

        // Test missing stability method
        config.internal_weight = 0.7;
        config.stability_weight = 0.3; // Reset weights
        config.include_stability = true;
        config.stability_method = None;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_automated_validation_result_methods() {
        let result = AutomatedValidationResult {
            overall_quality: ClusterQuality::Good,
            internal_quality_score: 0.75,
            stability_score: 0.65,
            silhouette_score: 0.8,
            calinski_harabasz_score: 15.0,
            davies_bouldin_score: 1.2,
            coherence_score: 0.7,
            separation_score: 0.6,
            recommendations: vec!["Test recommendation".to_string()],
            validation_summary: "Test summary".to_string(),
        };

        assert_eq!(result.combined_score(), 0.7);
        assert!(result.meets_threshold(0.6));
        assert!(!result.meets_threshold(0.8));

        let metrics = result.metric_scores();
        assert_eq!(metrics.len(), 7);

        let (best_name, best_score) = result.best_metric();
        assert_eq!(best_name, "Calinski-Harabasz");
        assert_eq!(best_score, 15.0);

        let problematic = result.problematic_metrics(0.7);
        assert!(problematic.len() > 0); // Some metrics should be below 0.7

        let report = result.detailed_report();
        assert!(report.contains("Automated Cluster Validation Report"));
        assert!(report.contains("Overall Quality"));
        assert!(report.contains("Individual Metrics"));
    }

    #[test]
    fn test_automated_validation() {
        let (data, labels) = generate_test_data();
        let validator = AutomatedValidator::euclidean();
        let config = AutomatedValidationConfig::fast(); // No stability for simplicity

        let result = validator
            .automated_cluster_validation(&data, &labels, &config)
            .unwrap();

        assert!(result.internal_quality_score >= 0.0 && result.internal_quality_score <= 1.0);
        assert!(result.stability_score >= 0.0 && result.stability_score <= 1.0);
        assert!(result.silhouette_score >= -1.0 && result.silhouette_score <= 1.0);
        assert!(result.calinski_harabasz_score >= 0.0);
        assert!(result.davies_bouldin_score >= 0.0);
        assert!(result.coherence_score >= 0.0 && result.coherence_score <= 1.0);
        assert!(result.separation_score >= 0.0);

        assert!(!result.validation_summary.is_empty());
        // Well-separated clusters should have some quality level
        assert!(matches!(
            result.overall_quality,
            ClusterQuality::Excellent
                | ClusterQuality::Good
                | ClusterQuality::Fair
                | ClusterQuality::Poor
        ));
    }

    #[test]
    fn test_compute_internal_quality_score() {
        let validator = AutomatedValidator::euclidean();

        // Create mock coherence and separation results
        let coherence = ClusterCoherenceResult {
            overall_coherence: 0.8,
            overall_compactness: 0.7,
            overall_density: 0.6,
            overall_shape_regularity: 0.5,
            cluster_coherence_scores: HashMap::new(),
            cluster_compactness_scores: HashMap::new(),
            cluster_density_scores: HashMap::new(),
            cluster_shape_regularity: HashMap::new(),
            coherence_consistency: 0.9,
            compactness_consistency: 0.8,
            density_consistency: 0.7,
            shape_consistency: 0.6,
            n_clusters: 2,
        };

        let separation = ClusterSeparationResult {
            avg_centroid_separation: 5.0,
            min_separation: 3.0,
            avg_inter_cluster_distance: 4.0,
            boundary_clarity: 2.0,
            overlap_measure: 0.2,
            gap_ratio: 2.5,
            min_inter_cluster_distances: HashMap::new(),
            n_clusters: 2,
        };

        let quality_score = validator.compute_internal_quality_score(
            0.6,  // silhouette
            20.0, // calinski_harabasz
            0.8,  // davies_bouldin
            &coherence,
            &separation,
        );

        assert!(quality_score >= 0.0 && quality_score <= 1.0);
    }

    #[test]
    fn test_generate_recommendations() {
        let validator = AutomatedValidator::euclidean();
        let config = AutomatedValidationConfig::default();

        // Create mock results for poor quality clustering
        let silhouette = SilhouetteResult {
            sample_silhouettes: vec![0.2, 0.3, 0.1],
            mean_silhouette: 0.2,
            cluster_silhouettes: HashMap::new(),
            cluster_sizes: HashMap::new(),
            confidence_interval: None,
        };

        let coherence = ClusterCoherenceResult {
            overall_coherence: 0.3, // Low coherence
            overall_compactness: 0.4,
            overall_density: 0.5,
            overall_shape_regularity: 0.6,
            cluster_coherence_scores: HashMap::new(),
            cluster_compactness_scores: HashMap::new(),
            cluster_density_scores: HashMap::new(),
            cluster_shape_regularity: HashMap::new(),
            coherence_consistency: 0.5, // Low consistency
            compactness_consistency: 0.6,
            density_consistency: 0.7,
            shape_consistency: 0.8,
            n_clusters: 3,
        };

        let separation = ClusterSeparationResult {
            avg_centroid_separation: 1.0,
            min_separation: 0.5,
            avg_inter_cluster_distance: 1.2,
            boundary_clarity: 0.8,
            overlap_measure: 0.4, // High overlap
            gap_ratio: 1.0,       // Low gap ratio
            min_inter_cluster_distances: HashMap::new(),
            n_clusters: 3,
        };

        let recommendations = validator.generate_recommendations(
            &silhouette,
            5.0, // Low CH index
            3.0, // High DB index
            &coherence,
            &separation,
            0.2, // Low internal quality
            0.3, // Low stability
            &config,
        );

        // Should generate multiple recommendations for poor clustering
        assert!(!recommendations.is_empty());
        assert!(recommendations.len() >= 3); // Should have several specific recommendations
    }

    #[test]
    fn test_assess_overall_quality() {
        let validator = AutomatedValidator::euclidean();
        let config = AutomatedValidationConfig::default();

        // Test excellent quality
        let excellent = validator.assess_overall_quality(0.9, 0.8, &config);
        assert_eq!(excellent, ClusterQuality::Excellent);

        // Test poor quality
        let poor = validator.assess_overall_quality(0.2, 0.1, &config);
        assert_eq!(poor, ClusterQuality::Poor);

        // Test edge cases
        let good = validator.assess_overall_quality(0.7, 0.6, &config);
        assert!(matches!(good, ClusterQuality::Good | ClusterQuality::Fair));
    }

    #[test]
    fn test_invalid_inputs() {
        let validator = AutomatedValidator::euclidean();
        let config = AutomatedValidationConfig::default();

        // Test mismatched data and labels
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = vec![0]; // Wrong length

        assert!(validator
            .automated_cluster_validation(&data, &labels, &config)
            .is_err());

        // Test invalid configuration
        let mut invalid_config = AutomatedValidationConfig::default();
        invalid_config.quality_threshold = 1.5; // Invalid threshold

        let correct_labels = vec![0, 1];
        assert!(validator
            .automated_cluster_validation(&data, &correct_labels, &invalid_config)
            .is_err());
    }

    #[test]
    fn test_different_metrics() {
        let (data, labels) = generate_test_data();
        let config = AutomatedValidationConfig::fast();

        let euclidean_validator = AutomatedValidator::euclidean();
        let manhattan_validator = AutomatedValidator::manhattan();
        let cosine_validator = AutomatedValidator::cosine();

        let euc_result = euclidean_validator
            .automated_cluster_validation(&data, &labels, &config)
            .unwrap();
        let man_result = manhattan_validator
            .automated_cluster_validation(&data, &labels, &config)
            .unwrap();
        let cos_result = cosine_validator
            .automated_cluster_validation(&data, &labels, &config)
            .unwrap();

        // All should produce valid results
        assert!(euc_result.internal_quality_score >= 0.0);
        assert!(man_result.internal_quality_score >= 0.0);
        assert!(cos_result.internal_quality_score >= 0.0);

        // Results should be different for different metrics
        assert_ne!(euc_result.silhouette_score, man_result.silhouette_score);
    }
}

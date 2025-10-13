//! Quality assessment metrics for feature selection evaluation
//!
//! This module provides comprehensive quality assessment capabilities for evaluating
//! the overall quality of selected feature sets. All implementations follow the
//! SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
type Result<T> = SklResult<T>;

impl From<QualityError> for SklearsError {
    fn from(err: QualityError) -> Self {
        SklearsError::FitError(format!("Quality assessment error: {}", err))
    }
}
use thiserror::Error;

#[derive(Debug, Error)]
pub enum QualityError {
    #[error("Feature matrix is empty")]
    EmptyFeatureMatrix,
    #[error("Target array is empty")]
    EmptyTarget,
    #[error("Feature and target lengths do not match")]
    LengthMismatch,
    #[error("Invalid feature indices")]
    InvalidFeatureIndices,
    #[error("Insufficient data for quality assessment")]
    InsufficientData,
}

/// Selection quality metrics
#[derive(Debug, Clone)]
pub struct SelectionQuality {
    n_features_selected: usize,
    n_features_total: usize,
    selection_ratio: f64,
    feature_efficiency: f64,
    information_density: f64,
}

impl SelectionQuality {
    /// Create a new selection quality assessment
    pub fn new(
        n_features_selected: usize,
        n_features_total: usize,
        feature_efficiency: f64,
        information_density: f64,
    ) -> Self {
        let selection_ratio = if n_features_total > 0 {
            n_features_selected as f64 / n_features_total as f64
        } else {
            0.0
        };

        Self {
            n_features_selected,
            n_features_total,
            selection_ratio,
            feature_efficiency,
            information_density,
        }
    }

    /// Assess the quality of feature selection
    pub fn assess(&self) -> QualityAssessmentResult {
        let compactness_score = self.assess_compactness();
        let efficiency_score = self.assess_efficiency();
        let information_score = self.assess_information();
        let balance_score = self.assess_balance();

        let overall_score =
            (compactness_score + efficiency_score + information_score + balance_score) / 4.0;

        QualityAssessmentResult {
            overall_quality_score: overall_score,
            compactness_score,
            efficiency_score,
            information_score,
            balance_score,
            selection_ratio: self.selection_ratio,
            n_features_selected: self.n_features_selected,
            n_features_total: self.n_features_total,
        }
    }

    /// Assess compactness (fewer features is better)
    fn assess_compactness(&self) -> f64 {
        // Compactness increases as selection ratio decreases
        (1.0 - self.selection_ratio).max(0.0)
    }

    /// Assess feature efficiency
    fn assess_efficiency(&self) -> f64 {
        self.feature_efficiency.min(1.0).max(0.0)
    }

    /// Assess information density
    fn assess_information(&self) -> f64 {
        self.information_density.min(1.0).max(0.0)
    }

    /// Assess balance between number of features and performance
    fn assess_balance(&self) -> f64 {
        // Optimal balance around 10-30% of features selected
        let optimal_ratio = 0.2;
        let deviation = (self.selection_ratio - optimal_ratio).abs();
        (1.0 - deviation * 2.0).max(0.0)
    }
}

/// Predictive performance assessment
#[derive(Debug, Clone)]
pub struct PredictivePerformance {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub auc_roc: f64,
}

impl PredictivePerformance {
    /// Create a new predictive performance assessment
    pub fn new(accuracy: f64, precision: f64, recall: f64, f1_score: f64, auc_roc: f64) -> Self {
        Self {
            accuracy,
            precision,
            recall,
            f1_score,
            auc_roc,
        }
    }

    /// Compute overall performance score
    pub fn overall_score(&self) -> f64 {
        (self.accuracy + self.precision + self.recall + self.f1_score + self.auc_roc) / 5.0
    }

    /// Assess performance quality
    pub fn assess_quality(&self) -> &'static str {
        let overall = self.overall_score();
        match overall {
            x if x >= 0.9 => "Excellent",
            x if x >= 0.8 => "Very Good",
            x if x >= 0.7 => "Good",
            x if x >= 0.6 => "Acceptable",
            x if x >= 0.5 => "Poor",
            _ => "Very Poor",
        }
    }
}

/// Model complexity assessment
#[derive(Debug, Clone)]
pub struct ModelComplexity {
    pub n_features: usize,
    pub n_parameters: usize,
    pub training_time: f64,
    pub prediction_time: f64,
    pub memory_usage: usize,
}

impl ModelComplexity {
    /// Create a new model complexity assessment
    pub fn new(
        n_features: usize,
        n_parameters: usize,
        training_time: f64,
        prediction_time: f64,
        memory_usage: usize,
    ) -> Self {
        Self {
            n_features,
            n_parameters,
            training_time,
            prediction_time,
            memory_usage,
        }
    }

    /// Compute complexity score (lower is better)
    pub fn complexity_score(&self) -> f64 {
        let normalized_features = (self.n_features as f64 / 1000.0).min(1.0);
        let normalized_parameters = (self.n_parameters as f64 / 10000.0).min(1.0);
        let normalized_time = (self.training_time / 3600.0).min(1.0); // Normalize by 1 hour
        let normalized_memory = (self.memory_usage as f64 / 1_000_000_000.0).min(1.0); // Normalize by 1GB

        (normalized_features + normalized_parameters + normalized_time + normalized_memory) / 4.0
    }

    /// Assess complexity level
    pub fn assess_complexity(&self) -> &'static str {
        let score = self.complexity_score();
        match score {
            x if x >= 0.8 => "Very High Complexity",
            x if x >= 0.6 => "High Complexity",
            x if x >= 0.4 => "Moderate Complexity",
            x if x >= 0.2 => "Low Complexity",
            _ => "Very Low Complexity",
        }
    }
}

/// Interpretability metrics
#[derive(Debug, Clone)]
pub struct InterpretabilityMetrics {
    pub feature_importance_clarity: f64,
    pub feature_interaction_complexity: f64,
    pub model_transparency: f64,
    pub explanation_quality: f64,
}

impl InterpretabilityMetrics {
    /// Create a new interpretability assessment
    pub fn new(
        feature_importance_clarity: f64,
        feature_interaction_complexity: f64,
        model_transparency: f64,
        explanation_quality: f64,
    ) -> Self {
        Self {
            feature_importance_clarity,
            feature_interaction_complexity,
            model_transparency,
            explanation_quality,
        }
    }

    /// Compute overall interpretability score
    pub fn interpretability_score(&self) -> f64 {
        let clarity_score = self.feature_importance_clarity;
        let complexity_score = 1.0 - self.feature_interaction_complexity; // Lower complexity is better
        let transparency_score = self.model_transparency;
        let explanation_score = self.explanation_quality;

        (clarity_score + complexity_score + transparency_score + explanation_score) / 4.0
    }

    /// Assess interpretability level
    pub fn assess_interpretability(&self) -> &'static str {
        let score = self.interpretability_score();
        match score {
            x if x >= 0.8 => "Highly Interpretable",
            x if x >= 0.6 => "Moderately Interpretable",
            x if x >= 0.4 => "Somewhat Interpretable",
            x if x >= 0.2 => "Poorly Interpretable",
            _ => "Very Poorly Interpretable",
        }
    }
}

/// Quality assessment result
#[derive(Debug, Clone)]
pub struct QualityAssessmentResult {
    pub overall_quality_score: f64,
    pub compactness_score: f64,
    pub efficiency_score: f64,
    pub information_score: f64,
    pub balance_score: f64,
    pub selection_ratio: f64,
    pub n_features_selected: usize,
    pub n_features_total: usize,
}

impl QualityAssessmentResult {
    /// Generate quality assessment report
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Feature Selection Quality Assessment ===\n\n");

        report.push_str(&format!(
            "Features selected: {} out of {} ({:.1}%)\n",
            self.n_features_selected,
            self.n_features_total,
            self.selection_ratio * 100.0
        ));

        report.push_str(&"\nQuality Scores (0.0 - 1.0):\n".to_string());
        report.push_str(&format!(
            "  Overall Quality:     {:.4} - {}\n",
            self.overall_quality_score,
            self.interpret_overall()
        ));
        report.push_str(&format!(
            "  Compactness:         {:.4} - {}\n",
            self.compactness_score,
            self.interpret_compactness()
        ));
        report.push_str(&format!(
            "  Efficiency:          {:.4} - {}\n",
            self.efficiency_score,
            self.interpret_efficiency()
        ));
        report.push_str(&format!(
            "  Information Density: {:.4} - {}\n",
            self.information_score,
            self.interpret_information()
        ));
        report.push_str(&format!(
            "  Balance:             {:.4} - {}\n",
            self.balance_score,
            self.interpret_balance()
        ));

        report.push_str(&"\nRecommendations:\n".to_string());
        report.push_str(&self.generate_recommendations());

        report
    }

    fn interpret_overall(&self) -> &'static str {
        match self.overall_quality_score {
            x if x >= 0.8 => "Excellent",
            x if x >= 0.6 => "Good",
            x if x >= 0.4 => "Acceptable",
            x if x >= 0.2 => "Poor",
            _ => "Very Poor",
        }
    }

    fn interpret_compactness(&self) -> &'static str {
        match self.compactness_score {
            x if x >= 0.8 => "Very compact feature set",
            x if x >= 0.6 => "Reasonably compact",
            x if x >= 0.4 => "Moderately compact",
            x if x >= 0.2 => "Not very compact",
            _ => "Too many features selected",
        }
    }

    fn interpret_efficiency(&self) -> &'static str {
        match self.efficiency_score {
            x if x >= 0.8 => "Highly efficient features",
            x if x >= 0.6 => "Good feature efficiency",
            x if x >= 0.4 => "Moderate efficiency",
            x if x >= 0.2 => "Low efficiency",
            _ => "Very low efficiency",
        }
    }

    fn interpret_information(&self) -> &'static str {
        match self.information_score {
            x if x >= 0.8 => "Very high information density",
            x if x >= 0.6 => "Good information content",
            x if x >= 0.4 => "Moderate information",
            x if x >= 0.2 => "Low information content",
            _ => "Very low information",
        }
    }

    fn interpret_balance(&self) -> &'static str {
        match self.balance_score {
            x if x >= 0.8 => "Well-balanced selection",
            x if x >= 0.6 => "Good balance",
            x if x >= 0.4 => "Acceptable balance",
            x if x >= 0.2 => "Poor balance",
            _ => "Very poor balance",
        }
    }

    fn generate_recommendations(&self) -> String {
        let mut recommendations = String::new();

        if self.compactness_score < 0.5 {
            recommendations.push_str("- Consider reducing the number of selected features\n");
        }

        if self.efficiency_score < 0.5 {
            recommendations.push_str("- Review feature selection criteria to improve efficiency\n");
        }

        if self.information_score < 0.5 {
            recommendations.push_str("- Look for features with higher information content\n");
        }

        if self.balance_score < 0.5 {
            if self.selection_ratio < 0.1 {
                recommendations
                    .push_str("- Consider selecting more features for better coverage\n");
            } else if self.selection_ratio > 0.4 {
                recommendations
                    .push_str("- Consider selecting fewer features to avoid redundancy\n");
            }
        }

        if self.overall_quality_score >= 0.8 {
            recommendations
                .push_str("- Feature selection quality is excellent - no major changes needed\n");
        } else if recommendations.is_empty() {
            recommendations.push_str("- Overall quality is acceptable but could be improved\n");
        }

        recommendations
    }
}

/// Comprehensive quality assessment
#[derive(Debug, Clone)]
pub struct QualityAssessment;

impl QualityAssessment {
    /// Perform comprehensive quality assessment
    pub fn assess(
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
        performance: Option<PredictivePerformance>,
        complexity: Option<ModelComplexity>,
        interpretability: Option<InterpretabilityMetrics>,
    ) -> Result<ComprehensiveQualityAssessment> {
        if X.nrows() != y.len() {
            return Err(QualityError::LengthMismatch.into());
        }

        if X.is_empty() || y.is_empty() {
            return Err(QualityError::EmptyFeatureMatrix.into());
        }

        // Compute basic quality metrics
        let feature_efficiency = Self::compute_feature_efficiency(X, y, feature_indices)?;
        let information_density = Self::compute_information_density(X, y, feature_indices)?;

        let selection_quality = SelectionQuality::new(
            feature_indices.len(),
            X.ncols(),
            feature_efficiency,
            information_density,
        );

        let quality_result = selection_quality.assess();

        Ok(ComprehensiveQualityAssessment {
            selection_quality: quality_result,
            predictive_performance: performance,
            model_complexity: complexity,
            interpretability_metrics: interpretability,
        })
    }

    /// Compute feature efficiency (signal-to-noise ratio approximation)
    fn compute_feature_efficiency(
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<f64> {
        if feature_indices.is_empty() {
            return Ok(0.0);
        }

        let mut total_efficiency = 0.0;

        for &feature_idx in feature_indices {
            if feature_idx >= X.ncols() {
                return Err(QualityError::InvalidFeatureIndices.into());
            }

            let feature_column = X.column(feature_idx);

            // Compute signal-to-noise ratio approximation
            let signal = Self::compute_signal_strength(feature_column, y)?;
            let noise = Self::compute_noise_level(feature_column)?;

            let efficiency = if noise > 1e-10 {
                signal / noise
            } else {
                signal
            };

            total_efficiency += efficiency.min(1.0);
        }

        Ok(total_efficiency / feature_indices.len() as f64)
    }

    /// Compute signal strength (correlation with target)
    fn compute_signal_strength(feature: ArrayView1<f64>, target: ArrayView1<f64>) -> Result<f64> {
        let n = feature.len() as f64;
        if n < 2.0 {
            return Ok(0.0);
        }

        let mean_x = feature.mean().unwrap_or(0.0);
        let mean_y = target.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..feature.len() {
            let dx = feature[i] - mean_x;
            let dy = target[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom < 1e-10 {
            return Ok(0.0);
        }

        Ok((sum_xy / denom).abs())
    }

    /// Compute noise level (coefficient of variation)
    fn compute_noise_level(feature: ArrayView1<f64>) -> Result<f64> {
        let mean = feature.mean().unwrap_or(0.0);

        if mean.abs() < 1e-10 {
            return Ok(1.0); // High noise for zero-mean features
        }

        let variance = feature.var(1.0);
        let std_dev = variance.sqrt();

        Ok(std_dev / mean.abs())
    }

    /// Compute information density (entropy-based approximation)
    fn compute_information_density(
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<f64> {
        if feature_indices.is_empty() {
            return Ok(0.0);
        }

        let mut total_density = 0.0;

        for &feature_idx in feature_indices {
            if feature_idx >= X.ncols() {
                return Err(QualityError::InvalidFeatureIndices.into());
            }

            let feature_column = X.column(feature_idx);
            let density = Self::compute_feature_entropy(feature_column)?;
            total_density += density;
        }

        // Normalize by maximum possible entropy (log2 of number of samples)
        let max_entropy = (X.nrows() as f64).ln();
        Ok((total_density / feature_indices.len() as f64) / max_entropy.max(1.0))
    }

    /// Compute entropy of a feature (approximated through binning)
    fn compute_feature_entropy(feature: ArrayView1<f64>) -> Result<f64> {
        let n_bins = 10.min(feature.len());
        if n_bins <= 1 {
            return Ok(0.0);
        }

        let min_val = feature.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = feature.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        if (max_val - min_val).abs() < 1e-10 {
            return Ok(0.0); // Constant feature has zero entropy
        }

        let bin_width = (max_val - min_val) / n_bins as f64;
        let mut bin_counts = vec![0; n_bins];

        for &value in feature.iter() {
            let bin = ((value - min_val) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            bin_counts[bin] += 1;
        }

        let total = feature.len() as f64;
        let mut entropy = 0.0;

        for count in bin_counts {
            if count > 0 {
                let probability = count as f64 / total;
                entropy -= probability * probability.ln();
            }
        }

        Ok(entropy)
    }
}

/// Comprehensive quality assessment result
#[derive(Debug, Clone)]
pub struct ComprehensiveQualityAssessment {
    pub selection_quality: QualityAssessmentResult,
    pub predictive_performance: Option<PredictivePerformance>,
    pub model_complexity: Option<ModelComplexity>,
    pub interpretability_metrics: Option<InterpretabilityMetrics>,
}

impl ComprehensiveQualityAssessment {
    /// Generate comprehensive quality report
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Comprehensive Feature Selection Quality Assessment ===\n\n");

        // Selection quality
        report.push_str(&self.selection_quality.report());

        // Predictive performance
        if let Some(ref performance) = self.predictive_performance {
            report.push_str("\n=== Predictive Performance ===\n");
            report.push_str(&format!(
                "Overall Performance: {:.4} ({})\n",
                performance.overall_score(),
                performance.assess_quality()
            ));
            report.push_str(&format!("  Accuracy:   {:.4}\n", performance.accuracy));
            report.push_str(&format!("  Precision:  {:.4}\n", performance.precision));
            report.push_str(&format!("  Recall:     {:.4}\n", performance.recall));
            report.push_str(&format!("  F1 Score:   {:.4}\n", performance.f1_score));
            report.push_str(&format!("  AUC-ROC:    {:.4}\n", performance.auc_roc));
        }

        // Model complexity
        if let Some(ref complexity) = self.model_complexity {
            report.push_str("\n=== Model Complexity ===\n");
            report.push_str(&format!(
                "Complexity Level: {} (Score: {:.4})\n",
                complexity.assess_complexity(),
                complexity.complexity_score()
            ));
            report.push_str(&format!("  Features:         {}\n", complexity.n_features));
            report.push_str(&format!(
                "  Parameters:       {}\n",
                complexity.n_parameters
            ));
            report.push_str(&format!(
                "  Training Time:    {:.2}s\n",
                complexity.training_time
            ));
            report.push_str(&format!(
                "  Prediction Time:  {:.6}s\n",
                complexity.prediction_time
            ));
            report.push_str(&format!(
                "  Memory Usage:     {} bytes\n",
                complexity.memory_usage
            ));
        }

        // Interpretability
        if let Some(ref interpretability) = self.interpretability_metrics {
            report.push_str("\n=== Interpretability ===\n");
            report.push_str(&format!(
                "Interpretability Level: {} (Score: {:.4})\n",
                interpretability.assess_interpretability(),
                interpretability.interpretability_score()
            ));
            report.push_str(&format!(
                "  Feature Importance Clarity:    {:.4}\n",
                interpretability.feature_importance_clarity
            ));
            report.push_str(&format!(
                "  Feature Interaction Complexity: {:.4}\n",
                interpretability.feature_interaction_complexity
            ));
            report.push_str(&format!(
                "  Model Transparency:             {:.4}\n",
                interpretability.model_transparency
            ));
            report.push_str(&format!(
                "  Explanation Quality:            {:.4}\n",
                interpretability.explanation_quality
            ));
        }

        report.push_str(&format!("\n{}\n", self.overall_recommendation()));

        report
    }

    /// Generate overall recommendation
    fn overall_recommendation(&self) -> String {
        let mut recommendation = String::new();
        recommendation.push_str("=== Overall Recommendation ===\n");

        let quality_score = self.selection_quality.overall_quality_score;
        let performance_score = self
            .predictive_performance
            .as_ref()
            .map(|p| p.overall_score())
            .unwrap_or(0.5);
        let complexity_score = 1.0
            - self
                .model_complexity
                .as_ref()
                .map(|c| c.complexity_score())
                .unwrap_or(0.5);
        let interpretability_score = self
            .interpretability_metrics
            .as_ref()
            .map(|i| i.interpretability_score())
            .unwrap_or(0.5);

        let overall_score =
            (quality_score + performance_score + complexity_score + interpretability_score) / 4.0;

        match overall_score {
            x if x >= 0.8 => recommendation
                .push_str("EXCELLENT: Feature selection is of high quality across all dimensions"),
            x if x >= 0.6 => recommendation
                .push_str("GOOD: Feature selection is solid with minor room for improvement"),
            x if x >= 0.4 => recommendation.push_str(
                "ACCEPTABLE: Feature selection meets basic requirements but could be improved",
            ),
            x if x >= 0.2 => {
                recommendation.push_str("POOR: Feature selection needs significant improvement")
            }
            _ => recommendation.push_str("CRITICAL: Feature selection requires major revision"),
        }

        recommendation
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_selection_quality() {
        let quality = SelectionQuality::new(5, 20, 0.8, 0.7);
        let result = quality.assess();

        assert!(result.overall_quality_score >= 0.0 && result.overall_quality_score <= 1.0);
        assert_eq!(result.n_features_selected, 5);
        assert_eq!(result.n_features_total, 20);
        assert_eq!(result.selection_ratio, 0.25);
    }

    #[test]
    fn test_predictive_performance() {
        let performance = PredictivePerformance::new(0.85, 0.80, 0.90, 0.85, 0.88);
        assert!((performance.overall_score() - 0.856).abs() < 0.01);
        assert_eq!(performance.assess_quality(), "Very Good");
    }

    #[test]
    fn test_model_complexity() {
        let complexity = ModelComplexity::new(10, 100, 30.0, 0.01, 1000000);
        let score = complexity.complexity_score();
        assert!(score >= 0.0 && score <= 1.0);
        assert!(complexity.assess_complexity().len() > 0);
    }

    #[test]
    fn test_interpretability_metrics() {
        let interpretability = InterpretabilityMetrics::new(0.8, 0.3, 0.9, 0.7);
        let score = interpretability.interpretability_score();
        assert!(score >= 0.0 && score <= 1.0);
        assert!(interpretability.assess_interpretability().len() > 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_quality_assessment() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 1.0];

        let feature_indices = vec![0, 2];

        let performance = PredictivePerformance::new(0.8, 0.75, 0.85, 0.8, 0.82);
        let complexity = ModelComplexity::new(2, 20, 10.0, 0.001, 100000);
        let interpretability = InterpretabilityMetrics::new(0.9, 0.2, 0.8, 0.85);

        let assessment = QualityAssessment::assess(
            X.view(),
            y.view(),
            &feature_indices,
            Some(performance),
            Some(complexity),
            Some(interpretability),
        )
        .unwrap();

        assert!(assessment.selection_quality.overall_quality_score >= 0.0);
        assert!(assessment.predictive_performance.is_some());
        assert!(assessment.model_complexity.is_some());
        assert!(assessment.interpretability_metrics.is_some());

        let report = assessment.report();
        assert!(report.contains("Comprehensive"));
        assert!(report.contains("Recommendation"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_quality_assessment_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];
        let y = array![0.0, 1.0, 1.0];

        let feature_indices = vec![0];

        let assessment =
            QualityAssessment::assess(X.view(), y.view(), &feature_indices, None, None, None)
                .unwrap();

        assert!(assessment.selection_quality.overall_quality_score >= 0.0);
        assert!(assessment.predictive_performance.is_none());
        assert!(assessment.model_complexity.is_none());
        assert!(assessment.interpretability_metrics.is_none());

        let report = assessment.report();
        assert!(report.contains("Quality Assessment"));
    }
}

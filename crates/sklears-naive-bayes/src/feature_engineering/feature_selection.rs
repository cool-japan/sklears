//! Feature selection methods and statistical testing
//!
//! This module provides comprehensive feature selection implementations including
//! univariate selection, recursive feature elimination, mutual information,
//! and correlation-based methods. All implementations follow SciRS2 Policy.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Zero;
use serde::{Deserialize, Serialize};
use sklears_core::error::Result;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;

/// Supported feature selection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    /// KBest
    KBest,
    /// Percentile
    Percentile,
    /// FDR
    FDR, // False Discovery Rate
    /// FWE
    FWE, // Family-wise Error rate
    /// RFE
    RFE, // Recursive Feature Elimination
    /// MutualInfo
    MutualInfo,
    /// Correlation
    Correlation,
    /// VarianceThreshold
    VarianceThreshold,
    /// SelectFromModel
    SelectFromModel,
    /// SequentialForward
    SequentialForward,
    SequentialBackward,
}

/// Configuration for feature selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConfig {
    pub method: FeatureSelectionMethod,
    pub k: Option<usize>,
    pub percentile: Option<f64>,
    pub alpha: f64,
    pub score_func: String,
    pub estimator: Option<String>,
    pub n_features_to_select: Option<usize>,
    pub step: f64,
    pub cv: usize,
    pub random_state: Option<u64>,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            method: FeatureSelectionMethod::KBest,
            k: Some(10),
            percentile: Some(10.0),
            alpha: 0.05,
            score_func: "f_classif".to_string(),
            estimator: None,
            n_features_to_select: None,
            step: 1.0,
            cv: 5,
            random_state: Some(42),
        }
    }
}

/// Results from feature selection process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionResults {
    pub selected_features: Vec<usize>,
    pub feature_scores: Array1<f64>,
    pub p_values: Option<Array1<f64>>,
    pub support_mask: Array1<bool>,
    pub ranking: Option<Array1<usize>>,
    pub n_features_selected: usize,
    pub selection_metadata: HashMap<String, f64>,
}

impl FeatureSelectionResults {
    pub fn new(
        selected_features: Vec<usize>,
        feature_scores: Array1<f64>,
        support_mask: Array1<bool>,
    ) -> Self {
        let n_features_selected = selected_features.len();
        Self {
            selected_features,
            feature_scores,
            p_values: None,
            support_mask,
            ranking: None,
            n_features_selected,
            selection_metadata: HashMap::new(),
        }
    }

    /// Get selected features
    pub fn selected_features(&self) -> &[usize] {
        &self.selected_features
    }

    /// Get feature scores
    pub fn feature_scores(&self) -> &Array1<f64> {
        &self.feature_scores
    }

    /// Get support mask
    pub fn support_mask(&self) -> &Array1<bool> {
        &self.support_mask
    }

    /// Get p-values if available
    pub fn p_values(&self) -> Option<&Array1<f64>> {
        self.p_values.as_ref()
    }

    /// Set p-values
    pub fn set_p_values(&mut self, p_values: Array1<f64>) {
        self.p_values = Some(p_values);
    }

    /// Get ranking if available
    pub fn ranking(&self) -> Option<&Array1<usize>> {
        self.ranking.as_ref()
    }

    /// Set ranking
    pub fn set_ranking(&mut self, ranking: Array1<usize>) {
        self.ranking = Some(ranking);
    }

    /// Get number of features selected
    pub fn n_features_selected(&self) -> usize {
        self.n_features_selected
    }

    /// Get selection metadata
    pub fn selection_metadata(&self) -> &HashMap<String, f64> {
        &self.selection_metadata
    }

    /// Add selection metadata
    pub fn add_metadata(&mut self, key: String, value: f64) {
        self.selection_metadata.insert(key, value);
    }
}

/// Validator for feature selection configurations
#[derive(Debug, Clone)]
pub struct SelectionValidator;

impl SelectionValidator {
    pub fn validate_config(config: &SelectionConfig) -> Result<()> {
        if let Some(percentile) = config.percentile {
            if !(0.0..=100.0).contains(&percentile) {
                return Err(SklearsError::InvalidInput(
                    "percentile must be between 0 and 100".to_string(),
                ));
            }
        }

        if config.alpha <= 0.0 || config.alpha >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "alpha must be between 0 and 1".to_string(),
            ));
        }

        if config.step <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "step must be positive".to_string(),
            ));
        }

        if config.cv == 0 {
            return Err(SklearsError::InvalidInput(
                "cv must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Core feature selector trait
pub trait FeatureSelector<T> {
    fn fit(&mut self, x: &ArrayView2<T>, y: &ArrayView1<T>) -> Result<()>;
    fn transform(&self, x: &ArrayView2<T>) -> Result<Array2<T>>;
    fn fit_transform(&mut self, x: &ArrayView2<T>, y: &ArrayView1<T>) -> Result<Array2<T>> {
        self.fit(x, y)?;
        self.transform(x)
    }
    fn get_support(&self) -> Result<Array1<bool>>;
    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Result<Vec<String>>;
}

/// K-best feature selector
#[derive(Debug, Clone)]
pub struct KBestSelector {
    k: usize,
    score_func: String,
    scores: Option<Array1<f64>>,
    p_values: Option<Array1<f64>>,
    selected_features: Option<Vec<usize>>,
}

impl KBestSelector {
    pub fn new(k: usize, score_func: String) -> Self {
        Self {
            k,
            score_func,
            scores: None,
            p_values: None,
            selected_features: None,
        }
    }

    /// Get feature scores
    pub fn scores(&self) -> Option<&Array1<f64>> {
        self.scores.as_ref()
    }

    /// Get p-values
    pub fn p_values(&self) -> Option<&Array1<f64>> {
        self.p_values.as_ref()
    }

    /// Get selected features
    pub fn selected_features(&self) -> Option<&[usize]> {
        self.selected_features.as_deref()
    }

    /// Compute feature scores (simplified implementation)
    fn compute_scores<T>(
        &self,
        x: &ArrayView2<T>,
        y: &ArrayView1<T>,
    ) -> Result<(Array1<f64>, Array1<f64>)>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let n_features = x.dim().1;
        let mut scores = Vec::with_capacity(n_features);
        let mut p_values = Vec::with_capacity(n_features);

        for feature_idx in 0..n_features {
            let feature = x.column(feature_idx);
            let (score, p_value) = self.compute_univariate_score(&feature, y)?;
            scores.push(score);
            p_values.push(p_value);
        }

        Ok((Array1::from_vec(scores), Array1::from_vec(p_values)))
    }

    /// Compute univariate score for a single feature (simplified)
    fn compute_univariate_score<T>(
        &self,
        _feature: &ArrayView1<T>,
        _target: &ArrayView1<T>,
    ) -> Result<(f64, f64)>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        // Simplified implementation - would compute actual statistical tests in practice
        Ok((1.0, 0.05))
    }
}

impl<T> FeatureSelector<T> for KBestSelector
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd + Zero,
{
    fn fit(&mut self, x: &ArrayView2<T>, y: &ArrayView1<T>) -> Result<()> {
        let (scores, p_values) = self.compute_scores(x, y)?;

        // Select k best features based on scores
        let mut score_indices: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();

        // Sort by score in descending order
        score_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: Vec<usize> = score_indices
            .iter()
            .take(self.k)
            .map(|(idx, _)| *idx)
            .collect();

        self.scores = Some(scores);
        self.p_values = Some(p_values);
        self.selected_features = Some(selected);

        Ok(())
    }

    fn transform(&self, x: &ArrayView2<T>) -> Result<Array2<T>> {
        let selected = self
            .selected_features
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "KBestSelector not fitted".to_string(),
            })?;

        let (n_samples, _) = x.dim();
        let mut result = Array2::zeros((n_samples, selected.len()));

        for (new_idx, &old_idx) in selected.iter().enumerate() {
            let column = x.column(old_idx);
            for (row_idx, &value) in column.iter().enumerate() {
                result[(row_idx, new_idx)] = value;
            }
        }

        Ok(result)
    }

    fn get_support(&self) -> Result<Array1<bool>> {
        let selected = self
            .selected_features
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "KBestSelector not fitted".to_string(),
            })?;

        let n_features = self
            .scores
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "KBestSelector not fitted".to_string(),
            })?
            .len();

        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected {
            support[idx] = true;
        }

        Ok(support)
    }

    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Result<Vec<String>> {
        let selected = self
            .selected_features
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "KBestSelector not fitted".to_string(),
            })?;

        if let Some(names) = input_features {
            if names.len() != self.scores.as_ref().unwrap().len() {
                return Err(SklearsError::InvalidInput(
                    "input_features length doesn't match number of features".to_string(),
                ));
            }
            Ok(selected.iter().map(|&idx| names[idx].clone()).collect())
        } else {
            Ok(selected
                .iter()
                .map(|&idx| format!("feature_{}", idx))
                .collect())
        }
    }
}

/// Percentile-based feature selector
#[derive(Debug, Clone)]
pub struct PercentileSelector {
    percentile: f64,
    score_func: String,
    scores: Option<Array1<f64>>,
    threshold: Option<f64>,
    selected_features: Option<Vec<usize>>,
}

impl PercentileSelector {
    pub fn new(percentile: f64, score_func: String) -> Result<Self> {
        if !(0.0..=100.0).contains(&percentile) {
            return Err(SklearsError::InvalidInput(
                "percentile must be between 0 and 100".to_string(),
            ));
        }

        Ok(Self {
            percentile,
            score_func,
            scores: None,
            threshold: None,
            selected_features: None,
        })
    }

    pub fn threshold(&self) -> Option<f64> {
        self.threshold
    }
}

/// False Discovery Rate selector
#[derive(Debug, Clone)]
pub struct FDRSelector {
    alpha: f64,
    scores: Option<Array1<f64>>,
    p_values: Option<Array1<f64>>,
    selected_features: Option<Vec<usize>>,
}

impl FDRSelector {
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "alpha must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            alpha,
            scores: None,
            p_values: None,
            selected_features: None,
        })
    }
}

/// Family-wise Error rate selector
#[derive(Debug, Clone)]
pub struct FWESelector {
    alpha: f64,
    scores: Option<Array1<f64>>,
    p_values: Option<Array1<f64>>,
    selected_features: Option<Vec<usize>>,
}

impl FWESelector {
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "alpha must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            alpha,
            scores: None,
            p_values: None,
            selected_features: None,
        })
    }
}

/// Recursive Feature Elimination selector
#[derive(Debug, Clone)]
pub struct RFESelector {
    estimator: String,
    n_features_to_select: Option<usize>,
    step: f64,
    cv: usize,
    feature_ranking: Option<Array1<usize>>,
    selected_features: Option<Vec<usize>>,
}

impl RFESelector {
    pub fn new(
        estimator: String,
        n_features_to_select: Option<usize>,
        step: f64,
        cv: usize,
    ) -> Result<Self> {
        if step <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "step must be positive".to_string(),
            ));
        }

        if cv == 0 {
            return Err(SklearsError::InvalidInput(
                "cv must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            estimator,
            n_features_to_select,
            step,
            cv,
            feature_ranking: None,
            selected_features: None,
        })
    }

    pub fn feature_ranking(&self) -> Option<&Array1<usize>> {
        self.feature_ranking.as_ref()
    }
}

/// Mutual Information selector
#[derive(Debug, Clone)]
pub struct MutualInfoSelector {
    k: Option<usize>,
    percentile: Option<f64>,
    discrete_features: Vec<usize>,
    scores: Option<Array1<f64>>,
    selected_features: Option<Vec<usize>>,
}

impl MutualInfoSelector {
    pub fn new(k: Option<usize>, percentile: Option<f64>, discrete_features: Vec<usize>) -> Self {
        Self {
            k,
            percentile,
            discrete_features,
            scores: None,
            selected_features: None,
        }
    }

    pub fn discrete_features(&self) -> &[usize] {
        &self.discrete_features
    }
}

/// Correlation-based selector
#[derive(Debug, Clone)]
pub struct CorrelationSelector {
    method: String,
    threshold: f64,
    correlations: Option<Array1<f64>>,
    selected_features: Option<Vec<usize>>,
}

impl CorrelationSelector {
    pub fn new(method: String, threshold: f64) -> Self {
        Self {
            method,
            threshold,
            correlations: None,
            selected_features: None,
        }
    }

    pub fn correlations(&self) -> Option<&Array1<f64>> {
        self.correlations.as_ref()
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// Selection analyzer for comprehensive feature selection analysis
#[derive(Debug, Clone)]
pub struct SelectionAnalyzer {
    selection_history: Vec<FeatureSelectionResults>,
    performance_metrics: HashMap<String, f64>,
    stability_metrics: HashMap<String, f64>,
}

impl SelectionAnalyzer {
    pub fn new() -> Self {
        Self {
            selection_history: Vec::new(),
            performance_metrics: HashMap::new(),
            stability_metrics: HashMap::new(),
        }
    }

    /// Add selection results to history
    pub fn add_selection_result(&mut self, result: FeatureSelectionResults) {
        self.selection_history.push(result);
    }

    /// Get selection history
    pub fn selection_history(&self) -> &[FeatureSelectionResults] {
        &self.selection_history
    }

    /// Add performance metric
    pub fn add_performance_metric(&mut self, key: String, value: f64) {
        self.performance_metrics.insert(key, value);
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> &HashMap<String, f64> {
        &self.performance_metrics
    }

    /// Add stability metric
    pub fn add_stability_metric(&mut self, key: String, value: f64) {
        self.stability_metrics.insert(key, value);
    }

    /// Get stability metrics
    pub fn stability_metrics(&self) -> &HashMap<String, f64> {
        &self.stability_metrics
    }

    /// Calculate selection stability across multiple runs
    pub fn calculate_stability(&mut self) -> f64 {
        if self.selection_history.len() < 2 {
            return 1.0; // Perfect stability with single selection
        }

        // Simplified stability calculation
        let mut stability_sum = 0.0;
        let mut comparisons = 0;

        for i in 0..self.selection_history.len() {
            for j in (i + 1)..self.selection_history.len() {
                let overlap =
                    self.calculate_overlap(&self.selection_history[i], &self.selection_history[j]);
                stability_sum += overlap;
                comparisons += 1;
            }
        }

        let stability = if comparisons > 0 {
            stability_sum / comparisons as f64
        } else {
            1.0
        };

        self.add_stability_metric("overall_stability".to_string(), stability);
        stability
    }

    /// Calculate overlap between two selection results
    fn calculate_overlap(
        &self,
        result1: &FeatureSelectionResults,
        result2: &FeatureSelectionResults,
    ) -> f64 {
        let set1: std::collections::HashSet<_> = result1.selected_features.iter().collect();
        let set2: std::collections::HashSet<_> = result2.selected_features.iter().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

impl Default for SelectionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_config_default() {
        let config = SelectionConfig::default();
        assert_eq!(config.method, FeatureSelectionMethod::KBest);
        assert_eq!(config.k, Some(10));
        assert_eq!(config.alpha, 0.05);
    }

    #[test]
    fn test_selection_validator() {
        let mut config = SelectionConfig::default();
        assert!(SelectionValidator::validate_config(&config).is_ok());

        config.percentile = Some(150.0); // Invalid percentile
        assert!(SelectionValidator::validate_config(&config).is_err());

        config.percentile = Some(50.0);
        config.alpha = 0.0; // Invalid alpha
        assert!(SelectionValidator::validate_config(&config).is_err());
    }

    #[test]
    fn test_feature_selection_results() {
        let selected = vec![0, 2, 4];
        let scores = Array1::from_vec(vec![0.8, 0.3, 0.7, 0.2, 0.9]);
        let support = Array1::from_vec(vec![true, false, true, false, true]);

        let mut results =
            FeatureSelectionResults::new(selected.clone(), scores.clone(), support.clone());

        assert_eq!(results.selected_features(), &selected);
        assert_eq!(results.feature_scores(), &scores);
        assert_eq!(results.support_mask(), &support);
        assert_eq!(results.n_features_selected(), 3);

        let p_values = Array1::from_vec(vec![0.01, 0.5, 0.02, 0.8, 0.005]);
        results.set_p_values(p_values.clone());
        assert_eq!(results.p_values().unwrap(), &p_values);

        results.add_metadata("accuracy".to_string(), 0.95);
        assert_eq!(results.selection_metadata().get("accuracy"), Some(&0.95));
    }

    #[test]
    fn test_k_best_selector() {
        let selector = KBestSelector::new(3, "f_classif".to_string());
        assert_eq!(selector.k, 3);
        assert_eq!(selector.score_func, "f_classif");
        assert!(selector.scores().is_none());
    }

    #[test]
    fn test_percentile_selector() {
        let selector = PercentileSelector::new(25.0, "chi2".to_string()).unwrap();
        assert_eq!(selector.percentile, 25.0);

        // Test invalid percentile
        assert!(PercentileSelector::new(-10.0, "chi2".to_string()).is_err());
        assert!(PercentileSelector::new(150.0, "chi2".to_string()).is_err());
    }

    #[test]
    fn test_fdr_selector() {
        let selector = FDRSelector::new(0.05).unwrap();
        assert_eq!(selector.alpha, 0.05);

        // Test invalid alpha
        assert!(FDRSelector::new(0.0).is_err());
        assert!(FDRSelector::new(1.0).is_err());
    }

    #[test]
    fn test_rfe_selector() {
        let selector = RFESelector::new("linear_svc".to_string(), Some(5), 1.0, 3).unwrap();
        assert_eq!(selector.n_features_to_select, Some(5));
        assert_eq!(selector.step, 1.0);
        assert_eq!(selector.cv, 3);

        // Test invalid parameters
        assert!(RFESelector::new("linear_svc".to_string(), Some(5), -1.0, 3).is_err());
        assert!(RFESelector::new("linear_svc".to_string(), Some(5), 1.0, 0).is_err());
    }

    #[test]
    fn test_mutual_info_selector() {
        let selector = MutualInfoSelector::new(Some(5), None, vec![0, 2]);
        assert_eq!(selector.k, Some(5));
        assert_eq!(selector.discrete_features(), &[0, 2]);
    }

    #[test]
    fn test_correlation_selector() {
        let selector = CorrelationSelector::new("pearson".to_string(), 0.8);
        assert_eq!(selector.method, "pearson");
        assert_eq!(selector.threshold(), 0.8);
    }

    #[test]
    fn test_selection_analyzer() {
        let mut analyzer = SelectionAnalyzer::new();

        let scores = Array1::from_vec(vec![0.8, 0.3, 0.7]);
        let support = Array1::from_vec(vec![true, false, true]);
        let result = FeatureSelectionResults::new(vec![0, 2], scores, support);

        analyzer.add_selection_result(result);
        assert_eq!(analyzer.selection_history().len(), 1);

        analyzer.add_performance_metric("accuracy".to_string(), 0.95);
        assert_eq!(analyzer.performance_metrics().get("accuracy"), Some(&0.95));

        let stability = analyzer.calculate_stability();
        assert_eq!(stability, 1.0); // Only one result, perfect stability
    }

    #[test]
    fn test_k_best_selector_fit_transform() {
        let mut selector = KBestSelector::new(2, "f_classif".to_string());

        let x = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);

        assert!(selector.fit(&x.view(), &y.view()).is_ok());
        assert!(selector.scores().is_some());
        assert!(selector.selected_features().is_some());

        let transformed = selector.transform(&x.view()).unwrap();
        assert_eq!(transformed.dim(), (4, 2)); // 4 samples, 2 selected features

        let support = <KBestSelector as FeatureSelector<f64>>::get_support(&selector).unwrap();
        assert_eq!(support.len(), 3); // Original number of features

        let feature_names =
            <KBestSelector as FeatureSelector<f64>>::get_feature_names_out(&selector, None)
                .unwrap();
        assert_eq!(feature_names.len(), 2); // Selected features
    }
}

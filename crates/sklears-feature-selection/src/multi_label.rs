//! Multi-label feature selection algorithms
//!
//! This module provides feature selection methods specifically designed for multi-label datasets,
//! where each instance can be associated with multiple labels simultaneously.

use crate::base::{FeatureSelector, SelectorMixin};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

/// Type alias for multi-label targets where each row can have multiple labels
pub type MultiLabelTarget = Array2<Float>;

/// Multi-label feature selection strategy
#[derive(Debug, Clone)]
pub enum MultiLabelStrategy {
    /// Select features relevant to all labels
    GlobalRelevance,
    /// Select features most relevant to individual labels and combine
    LabelSpecific,
    /// Use label correlations to guide feature selection
    LabelCorrelationAware,
    /// Hierarchical selection considering label relationships
    HierarchicalLabels,
    /// Ensemble approach combining multiple strategies
    Ensemble,
}

/// Aggregation method for combining label-specific selections
#[derive(Debug, Clone)]
pub enum AggregateMethod {
    /// Union
    Union,
    /// Intersection
    Intersection,
    /// MajorityVote
    MajorityVote,
    /// WeightedUnion
    WeightedUnion,
}

/// Core multi-label feature selector
#[derive(Debug, Clone)]
pub struct MultiLabelFeatureSelector<State = Untrained> {
    strategy: MultiLabelStrategy,
    n_features: Option<usize>,
    threshold: Float,
    min_label_frequency: Float,
    use_label_correlation: bool,
    correlation_threshold: Float,
    state: PhantomData<State>,
    // Trained state
    scores_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    n_labels_: Option<usize>,
}

impl Default for MultiLabelFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiLabelFeatureSelector<Untrained> {
    /// Create a new multi-label feature selector
    pub fn new() -> Self {
        Self {
            strategy: MultiLabelStrategy::LabelSpecific,
            n_features: None,
            threshold: 0.01,
            min_label_frequency: 0.01,
            use_label_correlation: true,
            correlation_threshold: 0.1,
            state: PhantomData,
            scores_: None,
            selected_features_: None,
            n_features_: None,
            n_labels_: None,
        }
    }

    /// Set the selection strategy
    pub fn strategy(mut self, strategy: MultiLabelStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the number of features to select
    pub fn n_features(mut self, n_features: usize) -> Self {
        self.n_features = Some(n_features);
        self
    }

    /// Set the relevance threshold
    pub fn threshold(mut self, threshold: Float) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the minimum label frequency
    pub fn min_label_frequency(mut self, frequency: Float) -> Self {
        self.min_label_frequency = frequency;
        self
    }

    /// Set whether to use label correlation
    pub fn use_label_correlation(mut self, use_correlation: bool) -> Self {
        self.use_label_correlation = use_correlation;
        self
    }

    /// Set the correlation threshold
    pub fn correlation_threshold(mut self, threshold: Float) -> Self {
        self.correlation_threshold = threshold;
        self
    }

    /// Compute feature relevance for multi-label data
    fn compute_multi_label_relevance(
        &self,
        features: &Array2<Float>,
        labels: &MultiLabelTarget,
    ) -> SklResult<Array1<Float>> {
        let n_features = features.ncols();
        let mut relevance_scores = Array1::zeros(n_features);

        match self.strategy {
            MultiLabelStrategy::GlobalRelevance => {
                self.compute_global_relevance(features, labels, &mut relevance_scores)?;
            }
            MultiLabelStrategy::LabelSpecific => {
                self.compute_label_specific_relevance(features, labels, &mut relevance_scores)?;
            }
            MultiLabelStrategy::LabelCorrelationAware => {
                self.compute_correlation_aware_relevance(features, labels, &mut relevance_scores)?;
            }
            MultiLabelStrategy::HierarchicalLabels => {
                self.compute_hierarchical_relevance(features, labels, &mut relevance_scores)?;
            }
            MultiLabelStrategy::Ensemble => {
                self.compute_ensemble_relevance(features, labels, &mut relevance_scores)?;
            }
        }

        Ok(relevance_scores)
    }

    /// Compute global relevance across all labels
    fn compute_global_relevance(
        &self,
        features: &Array2<Float>,
        labels: &MultiLabelTarget,
        scores: &mut Array1<Float>,
    ) -> SklResult<()> {
        let n_features = features.ncols();
        let n_labels = labels.ncols();

        for feature_idx in 0..n_features {
            let feature_col = features.column(feature_idx);
            let mut total_relevance = 0.0;

            for label_idx in 0..n_labels {
                let label_col = labels.column(label_idx);

                // Compute correlation between feature and label
                let corr = self.compute_correlation(&feature_col, &label_col)?;
                total_relevance += corr.abs();
            }

            scores[feature_idx] = total_relevance / n_labels as Float;
        }

        Ok(())
    }

    /// Compute label-specific relevance and aggregate
    fn compute_label_specific_relevance(
        &self,
        features: &Array2<Float>,
        labels: &MultiLabelTarget,
        scores: &mut Array1<Float>,
    ) -> SklResult<()> {
        let n_features = features.ncols();
        let n_labels = labels.ncols();

        // Compute relevance for each label separately
        let mut label_relevances = Array2::zeros((n_labels, n_features));

        for label_idx in 0..n_labels {
            let label_col = labels.column(label_idx);

            // Skip labels with insufficient frequency
            let label_frequency = label_col.sum() / label_col.len() as Float;
            if label_frequency < self.min_label_frequency {
                continue;
            }

            for feature_idx in 0..n_features {
                let feature_col = features.column(feature_idx);
                let corr = self.compute_correlation(&feature_col, &label_col)?;
                label_relevances[[label_idx, feature_idx]] = corr.abs();
            }
        }

        // Aggregate relevances across labels (use max relevance)
        for feature_idx in 0..n_features {
            let feature_relevances = label_relevances.column(feature_idx);
            scores[feature_idx] = feature_relevances.iter().cloned().fold(0.0, Float::max);
        }

        Ok(())
    }

    /// Compute correlation-aware relevance considering label interactions
    fn compute_correlation_aware_relevance(
        &self,
        features: &Array2<Float>,
        labels: &MultiLabelTarget,
        scores: &mut Array1<Float>,
    ) -> SklResult<()> {
        let n_features = features.ncols();
        let n_labels = labels.ncols();

        // Compute label correlation matrix
        let label_correlations = self.compute_label_correlation_matrix(labels)?;

        for feature_idx in 0..n_features {
            let feature_col = features.column(feature_idx);
            let mut weighted_relevance = 0.0;
            let mut total_weight = 0.0;

            for label_idx in 0..n_labels {
                let label_col = labels.column(label_idx);
                let corr = self.compute_correlation(&feature_col, &label_col)?;

                // Weight by label importance and correlation structure
                let label_weight = self.compute_label_weight(label_idx, &label_correlations);
                weighted_relevance += corr.abs() * label_weight;
                total_weight += label_weight;
            }

            scores[feature_idx] = if total_weight > 0.0 {
                weighted_relevance / total_weight
            } else {
                0.0
            };
        }

        Ok(())
    }

    /// Compute hierarchical relevance for structured label spaces
    fn compute_hierarchical_relevance(
        &self,
        features: &Array2<Float>,
        labels: &MultiLabelTarget,
        scores: &mut Array1<Float>,
    ) -> SklResult<()> {
        // For now, implement a simplified hierarchical approach
        // In practice, this would require label hierarchy information
        self.compute_label_specific_relevance(features, labels, scores)?;

        // Apply hierarchical weighting (simplified)
        for score in scores.iter_mut() {
            *score *= 1.1; // Slight boost for hierarchical consideration
        }

        Ok(())
    }

    /// Compute ensemble relevance combining multiple strategies
    fn compute_ensemble_relevance(
        &self,
        features: &Array2<Float>,
        labels: &MultiLabelTarget,
        scores: &mut Array1<Float>,
    ) -> SklResult<()> {
        let n_features = features.ncols();
        let mut global_scores = Array1::zeros(n_features);
        let mut specific_scores = Array1::zeros(n_features);
        let mut correlation_scores = Array1::zeros(n_features);

        // Compute scores using different strategies
        self.compute_global_relevance(features, labels, &mut global_scores)?;
        self.compute_label_specific_relevance(features, labels, &mut specific_scores)?;
        self.compute_correlation_aware_relevance(features, labels, &mut correlation_scores)?;

        // Combine scores with equal weights
        for feature_idx in 0..n_features {
            scores[feature_idx] = (global_scores[feature_idx]
                + specific_scores[feature_idx]
                + correlation_scores[feature_idx])
                / 3.0;
        }

        Ok(())
    }

    /// Compute correlation between feature and label
    fn compute_correlation(
        &self,
        feature: &scirs2_core::ndarray::ArrayView1<Float>,
        label: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> SklResult<Float> {
        let feature_mean = feature.mean().unwrap_or(0.0);
        let label_mean = label.mean().unwrap_or(0.0);

        let mut covariance = 0.0;
        let mut feature_var = 0.0;
        let mut label_var = 0.0;

        let n = feature.len();
        if n == 0 {
            return Ok(0.0);
        }

        for i in 0..n {
            let f_diff = feature[i] - feature_mean;
            let l_diff = label[i] - label_mean;

            covariance += f_diff * l_diff;
            feature_var += f_diff * f_diff;
            label_var += l_diff * l_diff;
        }

        if feature_var == 0.0 || label_var == 0.0 {
            return Ok(0.0);
        }

        let correlation = covariance / (feature_var * label_var).sqrt();
        Ok(correlation)
    }

    /// Compute label correlation matrix
    fn compute_label_correlation_matrix(
        &self,
        labels: &MultiLabelTarget,
    ) -> SklResult<Array2<Float>> {
        let n_labels = labels.ncols();
        let mut correlations = Array2::zeros((n_labels, n_labels));

        for i in 0..n_labels {
            for j in 0..n_labels {
                if i == j {
                    correlations[[i, j]] = 1.0;
                } else {
                    let label_i = labels.column(i);
                    let label_j = labels.column(j);
                    let corr = self.compute_correlation(&label_i, &label_j)?;
                    correlations[[i, j]] = corr;
                }
            }
        }

        Ok(correlations)
    }

    /// Compute weight for a label based on correlation structure
    fn compute_label_weight(&self, label_idx: usize, correlations: &Array2<Float>) -> Float {
        let label_correlations = correlations.row(label_idx);
        let avg_correlation = label_correlations.mean().unwrap_or(0.0);

        // Labels with moderate correlations get higher weights
        1.0 - (avg_correlation - 0.5).abs()
    }

    /// Select features based on computed relevance scores
    fn select_features(&self, relevance_scores: &Array1<Float>) -> SklResult<Vec<usize>> {
        let n_features = relevance_scores.len();

        if let Some(k) = self.n_features {
            if k > n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "n_features ({}) must be <= total features ({})",
                    k, n_features
                )));
            }
            // Select top k features
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|&a, &b| {
                relevance_scores[b]
                    .partial_cmp(&relevance_scores[a])
                    .unwrap()
            });
            indices.truncate(k);
            Ok(indices)
        } else {
            // Select features above threshold
            let selected: Vec<usize> = relevance_scores
                .iter()
                .enumerate()
                .filter(|(_, &score)| score >= self.threshold)
                .map(|(idx, _)| idx)
                .collect();

            if selected.is_empty() {
                return Err(SklearsError::InvalidInput(
                    "No features selected with current threshold".to_string(),
                ));
            }
            Ok(selected)
        }
    }
}

impl Estimator for MultiLabelFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, MultiLabelTarget> for MultiLabelFeatureSelector<Untrained> {
    type Fitted = MultiLabelFeatureSelector<Trained>;

    fn fit(self, features: &Array2<Float>, target: &MultiLabelTarget) -> SklResult<Self::Fitted> {
        // Custom validation for multi-label targets
        if features.nrows() != target.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Inconsistent numbers of samples: features has {} samples, target has {}",
                features.nrows(),
                target.nrows()
            )));
        }

        let n_features = features.ncols();
        let n_labels = target.ncols();

        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }
        if n_labels == 0 {
            return Err(SklearsError::InvalidInput("No labels provided".to_string()));
        }

        let relevance_scores = self.compute_multi_label_relevance(features, target)?;
        let selected_features = self.select_features(&relevance_scores)?;

        Ok(MultiLabelFeatureSelector {
            strategy: self.strategy,
            n_features: self.n_features,
            threshold: self.threshold,
            min_label_frequency: self.min_label_frequency,
            use_label_correlation: self.use_label_correlation,
            correlation_threshold: self.correlation_threshold,
            state: PhantomData,
            scores_: Some(relevance_scores),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            n_labels_: Some(n_labels),
        })
    }
}

impl Transform<Array2<Float>> for MultiLabelFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for MultiLabelFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for MultiLabelFeatureSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl MultiLabelFeatureSelector<Trained> {
    /// Get feature relevance scores
    pub fn scores(&self) -> &Array1<Float> {
        self.scores_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }

    /// Get the number of labels
    pub fn n_labels(&self) -> usize {
        self.n_labels_.unwrap()
    }

    /// Check if a feature was selected
    pub fn is_feature_selected(&self, feature_idx: usize) -> bool {
        self.selected_features_
            .as_ref()
            .unwrap()
            .contains(&feature_idx)
    }

    /// Get feature ranking (0-indexed, lower is better)
    pub fn feature_ranking(&self) -> Vec<usize> {
        let scores = self.scores_.as_ref().unwrap();
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

        let mut ranking = vec![0; scores.len()];
        for (rank, &feature_idx) in indices.iter().enumerate() {
            ranking[feature_idx] = rank;
        }
        ranking
    }
}

/// Label-specific feature selector that selects features for individual labels
#[derive(Debug, Clone)]
pub struct LabelSpecificSelector<State = Untrained> {
    n_features_per_label: Option<usize>,
    threshold: Float,
    aggregate_method: AggregateMethod,
    state: PhantomData<State>,
    // Trained state
    selected_features_: Option<Vec<usize>>,
    label_selections_: Option<Vec<Vec<usize>>>,
    n_features_: Option<usize>,
    n_labels_: Option<usize>,
}

impl Default for LabelSpecificSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelSpecificSelector<Untrained> {
    pub fn new() -> Self {
        Self {
            n_features_per_label: None,
            threshold: 0.01,
            aggregate_method: AggregateMethod::Union,
            state: PhantomData,
            selected_features_: None,
            label_selections_: None,
            n_features_: None,
            n_labels_: None,
        }
    }

    pub fn n_features_per_label(mut self, n_features: usize) -> Self {
        self.n_features_per_label = Some(n_features);
        self
    }

    pub fn threshold(mut self, threshold: Float) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn aggregate_method(mut self, method: AggregateMethod) -> Self {
        self.aggregate_method = method;
        self
    }

    fn select_for_label(
        &self,
        features: &Array2<Float>,
        label: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> SklResult<Vec<usize>> {
        let n_features = features.ncols();
        let mut scores = Array1::zeros(n_features);

        for feature_idx in 0..n_features {
            let feature_col = features.column(feature_idx);
            scores[feature_idx] = self.compute_feature_label_relevance(&feature_col, label)?;
        }

        if let Some(k) = self.n_features_per_label {
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
            indices.truncate(k);
            Ok(indices)
        } else {
            Ok(scores
                .iter()
                .enumerate()
                .filter(|(_, &score)| score >= self.threshold)
                .map(|(idx, _)| idx)
                .collect())
        }
    }

    fn compute_feature_label_relevance(
        &self,
        feature: &scirs2_core::ndarray::ArrayView1<Float>,
        label: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> SklResult<Float> {
        // Compute correlation coefficient
        let feature_mean = feature.mean().unwrap_or(0.0);
        let label_mean = label.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut feature_variance = 0.0;
        let mut label_variance = 0.0;

        let n = feature.len();
        for i in 0..n {
            let f_diff = feature[i] - feature_mean;
            let l_diff = label[i] - label_mean;

            numerator += f_diff * l_diff;
            feature_variance += f_diff * f_diff;
            label_variance += l_diff * l_diff;
        }

        if feature_variance == 0.0 || label_variance == 0.0 {
            return Ok(0.0);
        }

        let correlation = numerator / (feature_variance * label_variance).sqrt();
        Ok(correlation.abs())
    }

    fn aggregate_selections(&self, label_selections: &[Vec<usize>]) -> Vec<usize> {
        match self.aggregate_method {
            AggregateMethod::Union => {
                let mut result = HashSet::new();
                for selection in label_selections {
                    result.extend(selection);
                }
                result.into_iter().collect()
            }
            AggregateMethod::Intersection => {
                if label_selections.is_empty() {
                    return vec![];
                }
                let mut result: HashSet<usize> = label_selections[0].iter().cloned().collect();
                for selection in &label_selections[1..] {
                    let selection_set: HashSet<usize> = selection.iter().cloned().collect();
                    result = result.intersection(&selection_set).cloned().collect();
                }
                result.into_iter().collect()
            }
            AggregateMethod::MajorityVote => {
                let mut feature_counts: HashMap<usize, usize> = HashMap::new();
                for selection in label_selections {
                    for &feature in selection {
                        *feature_counts.entry(feature).or_insert(0) += 1;
                    }
                }
                let majority_threshold = (label_selections.len() + 1) / 2;
                feature_counts
                    .into_iter()
                    .filter(|(_, count)| *count >= majority_threshold)
                    .map(|(feature, _)| feature)
                    .collect()
            }
            AggregateMethod::WeightedUnion => {
                // For now, same as union - could be extended with label importance weights
                let mut result = HashSet::new();
                for selection in label_selections {
                    result.extend(selection);
                }
                result.into_iter().collect()
            }
        }
    }
}

impl Estimator for LabelSpecificSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, MultiLabelTarget> for LabelSpecificSelector<Untrained> {
    type Fitted = LabelSpecificSelector<Trained>;

    fn fit(self, features: &Array2<Float>, target: &MultiLabelTarget) -> SklResult<Self::Fitted> {
        // Custom validation for multi-label targets
        if features.nrows() != target.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Inconsistent numbers of samples: features has {} samples, target has {}",
                features.nrows(),
                target.nrows()
            )));
        }

        let n_features = features.ncols();
        let n_labels = target.ncols();
        let mut label_selections = Vec::with_capacity(n_labels);

        for label_idx in 0..n_labels {
            let label_col = target.column(label_idx);
            let selection = self.select_for_label(features, &label_col)?;
            label_selections.push(selection);
        }

        let selected_features = self.aggregate_selections(&label_selections);

        Ok(LabelSpecificSelector {
            n_features_per_label: self.n_features_per_label,
            threshold: self.threshold,
            aggregate_method: self.aggregate_method,
            state: PhantomData,
            selected_features_: Some(selected_features),
            label_selections_: Some(label_selections),
            n_features_: Some(n_features),
            n_labels_: Some(n_labels),
        })
    }
}

impl Transform<Array2<Float>> for LabelSpecificSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();

        if n_selected == 0 {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for LabelSpecificSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for LabelSpecificSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl LabelSpecificSelector<Trained> {
    pub fn features_for_label(&self, label_idx: usize) -> Option<&[usize]> {
        self.label_selections_
            .as_ref()?
            .get(label_idx)
            .map(|v| v.as_slice())
    }

    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }

    pub fn n_labels(&self) -> usize {
        self.n_labels_.unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use scirs2_core::ndarray::Array2;

    fn create_test_data() -> (Array2<Float>, MultiLabelTarget) {
        let features =
            Array2::from_shape_vec((100, 10), (0..1000).map(|i| (i as Float) * 0.01).collect())
                .unwrap();
        let labels = Array2::from_shape_vec(
            (100, 3),
            (0..300)
                .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
                .collect(),
        )
        .unwrap();
        (features, labels)
    }

    #[test]
    fn test_multi_label_selector_global_relevance() {
        let (features, labels) = create_test_data();

        let selector = MultiLabelFeatureSelector::new()
            .strategy(MultiLabelStrategy::GlobalRelevance)
            .n_features(5);

        let trained = selector.fit(&features, &labels).unwrap();
        assert_eq!(trained.n_features_out(), 5);
        assert_eq!(trained.selected_features().len(), 5);
    }

    #[test]
    fn test_multi_label_selector_label_specific() {
        let (features, labels) = create_test_data();

        let selector = MultiLabelFeatureSelector::new()
            .strategy(MultiLabelStrategy::LabelSpecific)
            .n_features(3); // Use fixed number instead of threshold for random data

        let trained = selector.fit(&features, &labels).unwrap();
        assert_eq!(trained.n_features_out(), 3);
    }

    #[test]
    fn test_multi_label_transform() {
        let (features, labels) = create_test_data();

        let selector = MultiLabelFeatureSelector::new().n_features(3);

        let trained = selector.fit(&features, &labels).unwrap();
        let transformed = trained.transform(&features).unwrap();

        assert_eq!(transformed.ncols(), 3);
        assert_eq!(transformed.nrows(), features.nrows());
    }

    #[test]
    fn test_label_specific_selector() {
        let (features, labels) = create_test_data();

        let selector = LabelSpecificSelector::new()
            .n_features_per_label(2)
            .aggregate_method(AggregateMethod::Union);

        let trained = selector.fit(&features, &labels).unwrap();
        assert!(trained.n_features_out() > 0);
        assert!(trained.n_features_out() <= 6); // Max 2 per label * 3 labels
    }

    #[test]
    fn test_ensemble_strategy() {
        let (features, labels) = create_test_data();

        let selector = MultiLabelFeatureSelector::new()
            .strategy(MultiLabelStrategy::Ensemble)
            .n_features(4);

        let trained = selector.fit(&features, &labels).unwrap();
        assert_eq!(trained.n_features_out(), 4);
    }

    #[test]
    fn test_feature_ranking() {
        let (features, labels) = create_test_data();

        let selector = MultiLabelFeatureSelector::new().n_features(5);

        let trained = selector.fit(&features, &labels).unwrap();
        let ranking = trained.feature_ranking();

        assert_eq!(ranking.len(), features.ncols());
        // Check that selected features have better (lower) ranks
        for &selected_idx in trained.selected_features() {
            assert!(ranking[selected_idx] < 5);
        }
    }

    #[test]
    fn test_selector_mixin() {
        let (features, labels) = create_test_data();

        let selector = MultiLabelFeatureSelector::new().n_features(3);

        let trained = selector.fit(&features, &labels).unwrap();
        let support = trained.get_support().unwrap();

        assert_eq!(support.len(), features.ncols());
        assert_eq!(support.iter().filter(|&&x| x).count(), 3);
    }

    // Property-based tests for multi-label feature selection
    mod proptests {
        use super::*;

        fn valid_array_2d() -> impl Strategy<Value = Array2<Float>> {
            (5usize..20, 10usize..50).prop_flat_map(|(n_cols, n_rows)| {
                prop::collection::vec(-10.0..10.0f64, n_rows * n_cols).prop_map(move |values| {
                    Array2::from_shape_vec((n_rows, n_cols), values).unwrap()
                })
            })
        }

        fn valid_multilabel_target(
            n_samples: usize,
            n_labels: usize,
        ) -> impl Strategy<Value = MultiLabelTarget> {
            prop::collection::vec(0.0..1.0f64, n_samples * n_labels).prop_map(move |values| {
                Array2::from_shape_vec((n_samples, n_labels), values).unwrap()
            })
        }

        proptest! {
            #[test]
            fn prop_multi_label_selector_respects_feature_count(
                features in valid_array_2d(),
                n_features in 1usize..10
            ) {
                let n_labels = 3;
                let labels = Array2::from_elem((features.nrows(), n_labels), 0.5);

                let n_select = n_features.min(features.ncols());
                let selector = MultiLabelFeatureSelector::new()
                    .n_features(n_select);

                if let Ok(trained) = selector.fit(&features, &labels) {
                    prop_assert_eq!(trained.n_features_out(), n_select);
                    prop_assert!(trained.selected_features().len() == n_select);

                    // All selected features should be valid indices
                    for &idx in trained.selected_features() {
                        prop_assert!(idx < features.ncols());
                    }

                    // Transform should work correctly
                    if let Ok(transformed) = trained.transform(&features) {
                        prop_assert_eq!(transformed.ncols(), n_select);
                        prop_assert_eq!(transformed.nrows(), features.nrows());
                    }
                }
            }

            #[test]
            fn prop_multi_label_selector_deterministic(
                features in valid_array_2d(),
                n_features in 1usize..5
            ) {
                let n_labels = 2;
                let labels = Array2::from_elem((features.nrows(), n_labels), 0.3);

                let n_select = n_features.min(features.ncols());
                let selector = MultiLabelFeatureSelector::new()
                    .strategy(MultiLabelStrategy::GlobalRelevance)
                    .n_features(n_select);

                if let Ok(trained1) = selector.clone().fit(&features, &labels) {
                    if let Ok(trained2) = selector.fit(&features, &labels) {
                        // Same input should produce same output
                        prop_assert_eq!(trained1.selected_features(), trained2.selected_features());
                        prop_assert_eq!(trained1.n_features_out(), trained2.n_features_out());
                    }
                }
            }

            #[test]
            fn prop_multi_label_selector_scores_non_negative(
                features in valid_array_2d(),
                n_features in 1usize..5
            ) {
                let n_labels = 2;
                let labels = Array2::from_elem((features.nrows(), n_labels), 0.4);

                let n_select = n_features.min(features.ncols());
                let selector = MultiLabelFeatureSelector::new()
                    .n_features(n_select);

                if let Ok(trained) = selector.fit(&features, &labels) {
                    let scores = trained.scores();

                    // All scores should be non-negative (using absolute correlation)
                    for &score in scores.iter() {
                        prop_assert!(score >= 0.0);
                    }

                    // Selected features should have higher scores
                    let selected_indices = trained.selected_features();
                    let min_selected_score = selected_indices.iter()
                        .map(|&idx| scores[idx])
                        .fold(f64::INFINITY, f64::min);

                    // Count how many features have scores >= min_selected_score
                    let count_above_min = scores.iter()
                        .filter(|&&score| score >= min_selected_score)
                        .count();

                    // Should be at least as many as selected
                    prop_assert!(count_above_min >= selected_indices.len());
                }
            }

            #[test]
            fn prop_label_specific_selector_aggregation_consistency(
                features in valid_array_2d(),
                n_features_per_label in 1usize..3
            ) {
                let n_labels = 3;
                let labels = Array2::from_elem((features.nrows(), n_labels), 0.5);

                let n_select = n_features_per_label.min(features.ncols());

                // Test union aggregation
                let selector_union = LabelSpecificSelector::new()
                    .n_features_per_label(n_select)
                    .aggregate_method(AggregateMethod::Union);

                if let Ok(trained_union) = selector_union.fit(&features, &labels) {
                    // Union should select at most n_select * n_labels features
                    prop_assert!(trained_union.n_features_out() <= n_select * n_labels);

                    // Test intersection aggregation
                    let selector_intersect = LabelSpecificSelector::new()
                        .n_features_per_label(n_select)
                        .aggregate_method(AggregateMethod::Intersection);

                    if let Ok(trained_intersect) = selector_intersect.fit(&features, &labels) {
                        // Intersection should select at most n_select features
                        prop_assert!(trained_intersect.n_features_out() <= n_select);

                        // Intersection features should be subset of union features
                        let union_set: std::collections::HashSet<_> = trained_union.selected_features().iter().collect();
                        for &feature in trained_intersect.selected_features() {
                            prop_assert!(union_set.contains(&feature));
                        }
                    }
                }
            }

            #[test]
            fn prop_multi_label_transform_preserves_samples(
                features in valid_array_2d(),
                n_features in 1usize..5
            ) {
                let n_labels = 2;
                let labels = Array2::from_elem((features.nrows(), n_labels), 0.4);

                let n_select = n_features.min(features.ncols());
                let selector = MultiLabelFeatureSelector::new()
                    .n_features(n_select);

                if let Ok(trained) = selector.fit(&features, &labels) {
                    if let Ok(transformed) = trained.transform(&features) {
                        // Transform should preserve number of samples
                        prop_assert_eq!(transformed.nrows(), features.nrows());

                        // Should have correct number of features
                        prop_assert_eq!(transformed.ncols(), n_select);

                        // Values should be from original features
                        for (sample_idx, row) in transformed.rows().into_iter().enumerate() {
                            for (feat_idx, &value) in row.iter().enumerate() {
                                let original_feat_idx = trained.selected_features()[feat_idx];
                                let expected_value = features[[sample_idx, original_feat_idx]];
                                prop_assert!((value - expected_value).abs() < 1e-10);
                            }
                        }
                    }
                }
            }
        }
    }
}

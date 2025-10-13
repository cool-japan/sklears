//! Streaming and online feature selection methods
//!
//! This module provides algorithms for feature selection that can process data incrementally,
//! handle concept drift, and adapt their selection over time.

use crate::base::FeatureSelector;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;

/// Online feature selection using incremental statistics
///
/// This selector maintains running statistics for each feature and updates
/// feature selection based on these statistics as new data arrives.
#[derive(Debug, Clone)]
pub struct OnlineFeatureSelector<State = Untrained> {
    // Configuration
    k: usize,
    window_size: Option<usize>,
    decay_factor: f64,
    min_samples: usize,

    // Online statistics
    feature_means_: Option<Array1<Float>>,
    feature_vars_: Option<Array1<Float>>,
    sample_count_: usize,
    target_correlation_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,

    // Sliding window for concept drift detection
    window_data_: Option<VecDeque<Array1<Float>>>,
    window_targets_: Option<VecDeque<Float>>,

    state: PhantomData<State>,
}

impl OnlineFeatureSelector<Untrained> {
    /// Create a new online feature selector
    pub fn new(k: usize) -> Self {
        Self {
            k,
            window_size: None,
            decay_factor: 0.95,
            min_samples: 10,
            feature_means_: None,
            feature_vars_: None,
            sample_count_: 0,
            target_correlation_: None,
            selected_features_: None,
            n_features_: None,
            window_data_: None,
            window_targets_: None,
            state: PhantomData,
        }
    }

    /// Set the window size for concept drift detection
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = Some(window_size);
        self
    }

    /// Set the decay factor for exponential moving statistics
    pub fn decay_factor(mut self, decay_factor: f64) -> Self {
        if !(0.0..=1.0).contains(&decay_factor) {
            panic!("decay_factor must be between 0 and 1");
        }
        self.decay_factor = decay_factor;
        self
    }

    /// Set minimum samples before selection begins
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }
}

impl Default for OnlineFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Estimator for OnlineFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for OnlineFeatureSelector<Untrained> {
    type Fitted = OnlineFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(
                "k cannot be larger than number of features".to_string(),
            ));
        }

        let mut selector = OnlineFeatureSelector {
            k: self.k,
            window_size: self.window_size,
            decay_factor: self.decay_factor,
            min_samples: self.min_samples,
            feature_means_: Some(Array1::zeros(n_features)),
            feature_vars_: Some(Array1::zeros(n_features)),
            sample_count_: 0,
            target_correlation_: Some(Array1::zeros(n_features)),
            selected_features_: Some(Vec::new()),
            n_features_: Some(n_features),
            window_data_: if self.window_size.is_some() {
                Some(VecDeque::new())
            } else {
                None
            },
            window_targets_: if self.window_size.is_some() {
                Some(VecDeque::new())
            } else {
                None
            },
            state: PhantomData,
        };

        // Process initial data
        for (sample_idx, target) in x.axis_iter(Axis(0)).zip(y.iter()) {
            selector.partial_fit_sample(&sample_idx.to_owned(), *target)?;
        }

        Ok(selector)
    }
}

impl OnlineFeatureSelector<Trained> {
    /// Update the selector with a new sample
    pub fn partial_fit_sample(&mut self, sample: &Array1<Float>, target: Float) -> SklResult<()> {
        let n_features = sample.len();

        if let Some(expected_features) = self.n_features_ {
            if n_features != expected_features {
                return Err(SklearsError::InvalidInput(
                    "Sample has different number of features than expected".to_string(),
                ));
            }
        } else {
            self.n_features_ = Some(n_features);
            self.feature_means_ = Some(Array1::zeros(n_features));
            self.feature_vars_ = Some(Array1::zeros(n_features));
            self.target_correlation_ = Some(Array1::zeros(n_features));
        }

        // Update sliding window if enabled
        if let (Some(window_data), Some(window_targets)) =
            (self.window_data_.as_mut(), self.window_targets_.as_mut())
        {
            if let Some(window_size) = self.window_size {
                window_data.push_back(sample.clone());
                window_targets.push_back(target);

                if window_data.len() > window_size {
                    window_data.pop_front();
                    window_targets.pop_front();
                }
            }
        }

        // Update exponential moving statistics
        if let (Some(means), Some(vars), Some(correlations)) = (
            self.feature_means_.as_mut(),
            self.feature_vars_.as_mut(),
            self.target_correlation_.as_mut(),
        ) {
            self.sample_count_ += 1;
            let alpha = if self.sample_count_ == 1 {
                1.0
            } else {
                1.0 - self.decay_factor
            };

            for (i, &value) in sample.iter().enumerate() {
                // Update mean
                let old_mean = means[i];
                means[i] = alpha * value + (1.0 - alpha) * old_mean;

                // Update variance (using Welford's online algorithm)
                let delta = value - old_mean;
                let delta2 = value - means[i];
                vars[i] = (1.0 - alpha) * vars[i] + alpha * delta * delta2;

                // Update correlation with target (simplified)
                let target_mean = 0.0; // Simplified - in practice maintain running target mean
                let target_centered = target - target_mean;
                let feature_centered = value - means[i];
                correlations[i] =
                    alpha * (feature_centered * target_centered) + (1.0 - alpha) * correlations[i];
            }
        }

        // Update feature selection if we have enough samples
        if self.sample_count_ >= self.min_samples {
            self.update_feature_selection()?;
        }

        Ok(())
    }

    /// Update feature selection based on current statistics
    fn update_feature_selection(&mut self) -> SklResult<()> {
        if let Some(correlations) = &self.target_correlation_ {
            // Select features with highest absolute correlation
            let mut feature_scores: Vec<(usize, f64)> = correlations
                .iter()
                .enumerate()
                .map(|(i, &corr)| (i, corr.abs()))
                .collect();

            // Sort by score (descending)
            feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Select top k features
            let selected: Vec<usize> = feature_scores
                .into_iter()
                .take(self.k)
                .map(|(idx, _)| idx)
                .collect();

            self.selected_features_ = Some(selected);
        }

        Ok(())
    }

    /// Compute current target mean (simplified for this example)
    fn compute_target_mean(&self) -> Float {
        // This is a simplified implementation
        // In practice, you'd maintain a running mean of targets
        0.0
    }

    /// Detect concept drift using window statistics
    pub fn detect_concept_drift(&self) -> SklResult<bool> {
        if let (Some(window_data), Some(window_targets)) =
            (&self.window_data_, &self.window_targets_)
        {
            if window_data.len() < 20 {
                return Ok(false); // Not enough data
            }

            // Simple drift detection: compare first and second half of window
            let mid = window_data.len() / 2;
            let first_half_targets: Vec<Float> = window_targets.iter().take(mid).cloned().collect();
            let second_half_targets: Vec<Float> =
                window_targets.iter().skip(mid).cloned().collect();

            // Compute means
            let first_mean =
                first_half_targets.iter().sum::<Float>() / first_half_targets.len() as Float;
            let second_mean =
                second_half_targets.iter().sum::<Float>() / second_half_targets.len() as Float;

            // Simple threshold-based drift detection
            let drift_threshold = 0.5;
            Ok((first_mean - second_mean).abs() > drift_threshold)
        } else {
            Ok(false)
        }
    }

    /// Reset the selector (useful when concept drift is detected)
    pub fn reset(&mut self) -> SklResult<()> {
        if let Some(n_features) = self.n_features_ {
            self.feature_means_ = Some(Array1::zeros(n_features));
            self.feature_vars_ = Some(Array1::zeros(n_features));
            self.target_correlation_ = Some(Array1::zeros(n_features));
            self.sample_count_ = 0;

            if let Some(window_data) = self.window_data_.as_mut() {
                window_data.clear();
            }
            if let Some(window_targets) = self.window_targets_.as_mut() {
                window_targets.clear();
            }
        }
        Ok(())
    }
}

impl FeatureSelector for OnlineFeatureSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        match &self.selected_features_ {
            Some(features) => features,
            None => {
                static EMPTY: Vec<usize> = Vec::new();
                &EMPTY
            }
        }
    }
}

impl Transform<Array2<Float>, Array2<Float>> for OnlineFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        if let Some(selected) = &self.selected_features_ {
            if selected.is_empty() {
                return Err(SklearsError::InvalidData {
                    reason: "No features selected yet".to_string(),
                });
            }

            let selected_cols = x.select(Axis(1), selected);
            Ok(selected_cols)
        } else {
            Err(SklearsError::InvalidData {
                reason: "Selector not fitted yet".to_string(),
            })
        }
    }
}

/// Streaming feature importance calculator
///
/// Maintains running importance scores for features in a streaming fashion
#[derive(Debug, Clone)]
pub struct StreamingFeatureImportance {
    // Configuration
    decay_factor: f64,
    min_samples: usize,

    // State
    importance_scores_: HashMap<usize, Float>,
    sample_count_: usize,
    n_features_: Option<usize>,
}

impl StreamingFeatureImportance {
    /// Create a new streaming feature importance calculator
    pub fn new() -> Self {
        Self {
            decay_factor: 0.95,
            min_samples: 10,
            importance_scores_: HashMap::new(),
            sample_count_: 0,
            n_features_: None,
        }
    }

    /// Set decay factor for exponential moving average
    pub fn decay_factor(mut self, decay_factor: f64) -> Self {
        if !(0.0..=1.0).contains(&decay_factor) {
            panic!("decay_factor must be between 0 and 1");
        }
        self.decay_factor = decay_factor;
        self
    }

    /// Update importance scores with new sample
    pub fn update(
        &mut self,
        features: &Array1<Float>,
        target: Float,
        prediction: Float,
    ) -> SklResult<()> {
        let n_features = features.len();

        if let Some(expected) = self.n_features_ {
            if n_features != expected {
                return Err(SklearsError::InvalidInput(
                    "Inconsistent number of features".to_string(),
                ));
            }
        } else {
            self.n_features_ = Some(n_features);
        }

        self.sample_count_ += 1;
        let prediction_error = (target - prediction).abs();

        // Update importance based on feature values and prediction error
        for (i, &feature_value) in features.iter().enumerate() {
            let contribution = feature_value.abs() * prediction_error;

            let current_importance = self.importance_scores_.get(&i).cloned().unwrap_or(0.0);
            let alpha = 1.0 - self.decay_factor;
            let new_importance = alpha * contribution + self.decay_factor * current_importance;

            self.importance_scores_.insert(i, new_importance);
        }

        Ok(())
    }

    /// Get current importance scores
    pub fn get_importance_scores(&self) -> &HashMap<usize, Float> {
        &self.importance_scores_
    }

    /// Get top k most important features
    pub fn get_top_features(&self, k: usize) -> Vec<usize> {
        let mut scores: Vec<(usize, Float)> = self
            .importance_scores_
            .iter()
            .map(|(&idx, &score)| (idx, score))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.into_iter().take(k).map(|(idx, _)| idx).collect()
    }
}

impl Default for StreamingFeatureImportance {
    fn default() -> Self {
        Self::new()
    }
}

/// Concept drift-aware feature selector
///
/// Adapts feature selection when concept drift is detected
#[derive(Debug, Clone)]
pub struct ConceptDriftAwareSelector<State = Untrained> {
    base_selector: OnlineFeatureSelector<State>,
    drift_detection_window: usize,
    drift_threshold: f64,
    adaptation_rate: f64,

    // Drift detection state
    performance_history_: VecDeque<Float>,
    drift_detected_: bool,
}

impl ConceptDriftAwareSelector<Untrained> {
    /// Create a new concept drift-aware selector
    pub fn new(k: usize) -> Self {
        Self {
            base_selector: OnlineFeatureSelector::new(k),
            drift_detection_window: 100,
            drift_threshold: 0.05,
            adaptation_rate: 0.1,
            performance_history_: VecDeque::new(),
            drift_detected_: false,
        }
    }

    /// Set drift detection window size
    pub fn drift_detection_window(mut self, window_size: usize) -> Self {
        self.drift_detection_window = window_size;
        self
    }

    /// Set drift detection threshold
    pub fn drift_threshold(mut self, threshold: f64) -> Self {
        self.drift_threshold = threshold;
        self
    }

    /// Set minimum samples for feature selection
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.base_selector = self.base_selector.min_samples(min_samples);
        self
    }
}

impl Estimator for ConceptDriftAwareSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for ConceptDriftAwareSelector<Untrained> {
    type Fitted = ConceptDriftAwareSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        let fitted_base = self.base_selector.fit(x, y)?;

        Ok(ConceptDriftAwareSelector {
            base_selector: fitted_base,
            drift_detection_window: self.drift_detection_window,
            drift_threshold: self.drift_threshold,
            adaptation_rate: self.adaptation_rate,
            performance_history_: VecDeque::new(),
            drift_detected_: false,
        })
    }
}

impl ConceptDriftAwareSelector<Trained> {
    /// Update with new sample and check for drift
    pub fn partial_fit_with_performance(
        &mut self,
        sample: &Array1<Float>,
        target: Float,
        performance: Float,
    ) -> SklResult<()> {
        // Update base selector
        self.base_selector.partial_fit_sample(sample, target)?;

        // Update performance history
        self.performance_history_.push_back(performance);
        if self.performance_history_.len() > self.drift_detection_window {
            self.performance_history_.pop_front();
        }

        // Check for drift
        if self.performance_history_.len() >= self.drift_detection_window / 2 {
            self.drift_detected_ = self.detect_performance_drift()?;

            if self.drift_detected_ {
                // Adapt to drift by partially resetting the selector
                self.adapt_to_drift()?;
            }
        }

        Ok(())
    }

    /// Detect drift based on performance degradation
    fn detect_performance_drift(&self) -> SklResult<bool> {
        if self.performance_history_.len() < 20 {
            return Ok(false);
        }

        let mid = self.performance_history_.len() / 2;
        let recent_perf: Float = self.performance_history_.iter().skip(mid).sum::<Float>()
            / (self.performance_history_.len() - mid) as Float;

        let old_perf: Float =
            self.performance_history_.iter().take(mid).sum::<Float>() / mid as Float;

        // Drift detected if recent performance is significantly worse
        Ok(old_perf - recent_perf > self.drift_threshold)
    }

    /// Adapt to detected concept drift
    fn adapt_to_drift(&mut self) -> SklResult<()> {
        // Partially reset the base selector to adapt to new concept
        // This is a simplified adaptation strategy

        // Clear some of the history to focus on recent data
        let reset_fraction = self.adaptation_rate;
        let samples_to_keep =
            ((1.0 - reset_fraction) * self.performance_history_.len() as f64) as usize;

        while self.performance_history_.len() > samples_to_keep {
            self.performance_history_.pop_front();
        }

        self.drift_detected_ = false;
        Ok(())
    }

    /// Check if drift was recently detected
    pub fn drift_detected(&self) -> bool {
        self.drift_detected_
    }
}

impl FeatureSelector for ConceptDriftAwareSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.base_selector.selected_features()
    }
}

impl Transform<Array2<Float>, Array2<Float>> for ConceptDriftAwareSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        self.base_selector.transform(x)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_online_feature_selector_basic() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![1.0, 2.0, 3.0];

        let selector = OnlineFeatureSelector::new(2).min_samples(2);
        let fitted = selector.fit(&x, &y).unwrap();

        assert_eq!(fitted.selected_features().len(), 2);
        assert_eq!(fitted.sample_count_, 3);
    }

    #[test]
    fn test_online_selector_partial_fit() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = array![1.0, 2.0];

        let selector = OnlineFeatureSelector::new(2);
        let mut fitted = selector.fit(&x, &y).unwrap();

        // Add new sample
        let new_sample = array![10.0, 11.0, 12.0];
        fitted.partial_fit_sample(&new_sample, 3.0).unwrap();

        assert_eq!(fitted.sample_count_, 3);
    }

    #[test]
    fn test_streaming_importance() {
        let mut importance = StreamingFeatureImportance::new();

        let features = array![1.0, 2.0, 3.0];
        importance.update(&features, 5.0, 4.8).unwrap();

        let scores = importance.get_importance_scores();
        assert_eq!(scores.len(), 3);

        let top_features = importance.get_top_features(2);
        assert_eq!(top_features.len(), 2);
    }

    #[test]
    fn test_concept_drift_selector() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let selector = ConceptDriftAwareSelector::new(1).min_samples(2);
        let mut fitted = selector.fit(&x, &y).unwrap();

        // Add sample with performance
        let sample = array![7.0, 8.0];
        fitted
            .partial_fit_with_performance(&sample, 4.0, 0.9)
            .unwrap();

        assert_eq!(fitted.selected_features().len(), 1);
    }

    #[test]
    fn test_online_selector_transform() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = array![1.0, 2.0];

        let selector = OnlineFeatureSelector::new(2).min_samples(2);
        let fitted = selector.fit(&x, &y).unwrap();

        let test_x = array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
        let transformed = fitted.transform(&test_x).unwrap();

        assert_eq!(transformed.ncols(), 2);
        assert_eq!(transformed.nrows(), 2);
    }
}

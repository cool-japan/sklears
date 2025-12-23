//! Cross-validation strategies for feature selection evaluation
//!
//! This module implements comprehensive cross-validation methods specifically designed
//! for evaluating feature selection algorithms. All implementations follow the SciRS2
//! policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
type Result<T> = SklResult<T>;
use scirs2_core::random::{thread_rng, Rng};

impl From<CrossValidationError> for SklearsError {
    fn from(err: CrossValidationError) -> Self {
        SklearsError::FitError(format!("Cross-validation error: {}", err))
    }
}
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CrossValidationError {
    #[error("Insufficient data for cross-validation")]
    InsufficientData,
    #[error("Invalid fold configuration")]
    InvalidFoldConfiguration,
    #[error("Feature and target length mismatch")]
    LengthMismatch,
    #[error("Invalid feature indices")]
    InvalidFeatureIndices,
    #[error("Empty feature selection")]
    EmptyFeatureSelection,
}

/// Nested cross-validation for feature selection with inner and outer loops
#[derive(Debug, Clone)]
pub struct NestedCrossValidation {
    outer_folds: usize,
    inner_folds: usize,
    stratified: bool,
    random_state: Option<u64>,
}

impl NestedCrossValidation {
    /// Create a new nested cross-validation configuration
    pub fn new(
        outer_folds: usize,
        inner_folds: usize,
        stratified: bool,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            outer_folds,
            inner_folds,
            stratified,
            random_state,
        }
    }

    /// Perform nested cross-validation for feature selection evaluation
    #[allow(non_snake_case)]
    pub fn evaluate<F, G>(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_selector: F,
        performance_evaluator: G,
    ) -> Result<NestedCVResults>
    where
        F: Fn(ArrayView2<f64>, ArrayView1<f64>) -> Result<Vec<usize>> + Copy,
        G: Fn(
                ArrayView2<f64>,
                ArrayView1<f64>,
                ArrayView2<f64>,
                ArrayView1<f64>,
                &[usize],
            ) -> Result<f64>
            + Copy,
    {
        if X.nrows() != y.len() {
            return Err(CrossValidationError::LengthMismatch.into());
        }

        if X.nrows() < self.outer_folds * 2 {
            return Err(CrossValidationError::InsufficientData.into());
        }

        let n_samples = X.nrows();
        let indices: Vec<usize> = (0..n_samples).collect();

        // Create outer fold splits
        let outer_splits = if self.stratified {
            self.stratified_k_fold_split(&indices, y, self.outer_folds)?
        } else {
            self.k_fold_split(&indices, self.outer_folds)?
        };

        let mut outer_scores = Vec::with_capacity(self.outer_folds);
        let mut feature_selection_stability = Vec::new();
        let mut inner_cv_scores = Vec::new();

        for (outer_fold, (train_idx, test_idx)) in outer_splits.into_iter().enumerate() {
            // Extract outer training and test sets
            let X_outer_train = self.extract_samples(X, &train_idx);
            let y_outer_train = self.extract_targets(y, &train_idx);
            let X_outer_test = self.extract_samples(X, &test_idx);
            let y_outer_test = self.extract_targets(y, &test_idx);

            // Inner cross-validation for feature selection
            let inner_splits = if self.stratified {
                self.stratified_k_fold_split(
                    &(0..train_idx.len()).collect::<Vec<_>>(),
                    y_outer_train.view(),
                    self.inner_folds,
                )?
            } else {
                self.k_fold_split(&(0..train_idx.len()).collect::<Vec<_>>(), self.inner_folds)?
            };

            let mut inner_fold_scores = Vec::new();
            let mut inner_fold_features = Vec::new();

            for (inner_train_idx, inner_val_idx) in inner_splits {
                // Extract inner training and validation sets
                let X_inner_train = self.extract_samples(X_outer_train.view(), &inner_train_idx);
                let y_inner_train = self.extract_targets(y_outer_train.view(), &inner_train_idx);
                let X_inner_val = self.extract_samples(X_outer_train.view(), &inner_val_idx);
                let y_inner_val = self.extract_targets(y_outer_train.view(), &inner_val_idx);

                // Feature selection on inner training set
                let selected_features =
                    feature_selector(X_inner_train.view(), y_inner_train.view())?;

                if selected_features.is_empty() {
                    return Err(CrossValidationError::EmptyFeatureSelection.into());
                }

                // Evaluate on inner validation set
                let inner_score = performance_evaluator(
                    X_inner_train.view(),
                    y_inner_train.view(),
                    X_inner_val.view(),
                    y_inner_val.view(),
                    &selected_features,
                )?;

                inner_fold_scores.push(inner_score);
                inner_fold_features.push(selected_features);
            }

            // Store inner CV results
            let inner_cv_mean =
                inner_fold_scores.iter().sum::<f64>() / inner_fold_scores.len() as f64;
            inner_cv_scores.push(InnerCVResult {
                outer_fold,
                inner_scores: inner_fold_scores,
                mean_score: inner_cv_mean,
                selected_features: inner_fold_features,
            });

            // Select features using full outer training set
            let final_selected_features =
                feature_selector(X_outer_train.view(), y_outer_train.view())?;
            feature_selection_stability.push(final_selected_features.clone());

            // Evaluate on outer test set
            let outer_score = performance_evaluator(
                X_outer_train.view(),
                y_outer_train.view(),
                X_outer_test.view(),
                y_outer_test.view(),
                &final_selected_features,
            )?;

            outer_scores.push(outer_score);
        }

        // Compute stability metrics
        let stability_metrics = self.compute_stability_metrics(&feature_selection_stability)?;

        // Compute overall statistics
        let outer_mean = outer_scores.iter().sum::<f64>() / outer_scores.len() as f64;
        let outer_std = {
            let variance = outer_scores
                .iter()
                .map(|score| (score - outer_mean).powi(2))
                .sum::<f64>()
                / outer_scores.len() as f64;
            variance.sqrt()
        };

        let inner_mean = inner_cv_scores
            .iter()
            .map(|result| result.mean_score)
            .sum::<f64>()
            / inner_cv_scores.len() as f64;

        Ok(NestedCVResults {
            outer_scores,
            outer_mean_score: outer_mean,
            outer_std_score: outer_std,
            inner_cv_results: inner_cv_scores,
            inner_mean_score: inner_mean,
            feature_stability: stability_metrics,
            n_outer_folds: self.outer_folds,
            n_inner_folds: self.inner_folds,
        })
    }

    /// Create K-fold splits
    fn k_fold_split(
        &self,
        indices: &[usize],
        n_folds: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if indices.len() < n_folds {
            return Err(CrossValidationError::InvalidFoldConfiguration.into());
        }

        let mut shuffled_indices = indices.to_vec();

        // Shuffle if random state is provided
        if self.random_state.is_some() {
            self.shuffle_indices(&mut shuffled_indices);
        }

        let fold_size = indices.len() / n_folds;
        let remainder = indices.len() % n_folds;

        let mut splits = Vec::new();

        for fold in 0..n_folds {
            let start = fold * fold_size + fold.min(remainder);
            let end = start + fold_size + if fold < remainder { 1 } else { 0 };

            let test_indices = shuffled_indices[start..end].to_vec();
            let train_indices: Vec<usize> = shuffled_indices[..start]
                .iter()
                .chain(shuffled_indices[end..].iter())
                .cloned()
                .collect();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Create stratified K-fold splits
    fn stratified_k_fold_split(
        &self,
        indices: &[usize],
        y: ArrayView1<f64>,
        n_folds: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if indices.len() < n_folds {
            return Err(CrossValidationError::InvalidFoldConfiguration.into());
        }

        // Group indices by class
        let mut class_groups: HashMap<i32, Vec<usize>> = HashMap::new();
        for &idx in indices {
            let class = y[idx] as i32;
            class_groups.entry(class).or_default().push(idx);
        }

        // Shuffle each class group
        if self.random_state.is_some() {
            for group in class_groups.values_mut() {
                self.shuffle_indices(group);
            }
        }

        // Create folds maintaining class distribution
        let mut folds: Vec<Vec<usize>> = vec![Vec::new(); n_folds];

        for group in class_groups.values() {
            let group_fold_size = group.len() / n_folds;
            let group_remainder = group.len() % n_folds;

            for fold in 0..n_folds {
                let start = fold * group_fold_size + fold.min(group_remainder);
                let end = start + group_fold_size + if fold < group_remainder { 1 } else { 0 };
                folds[fold].extend_from_slice(&group[start..end]);
            }
        }

        // Create train/test splits
        let mut splits = Vec::new();
        for fold in 0..n_folds {
            let test_indices = folds[fold].clone();
            let train_indices: Vec<usize> = folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != fold)
                .flat_map(|(_, fold_indices)| fold_indices.iter())
                .cloned()
                .collect();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Simple shuffle implementation
    fn shuffle_indices(&self, indices: &mut [usize]) {
        for i in (1..indices.len()).rev() {
            let j = (thread_rng().gen::<f64>() * (i + 1) as f64) as usize;
            indices.swap(i, j);
        }
    }

    /// Extract samples by indices
    fn extract_samples(&self, X: ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
        let mut samples = Array2::zeros((indices.len(), X.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            samples.row_mut(i).assign(&X.row(idx));
        }
        samples
    }

    /// Extract targets by indices
    fn extract_targets(&self, y: ArrayView1<f64>, indices: &[usize]) -> Array1<f64> {
        let mut targets = Array1::zeros(indices.len());
        for (i, &idx) in indices.iter().enumerate() {
            targets[i] = y[idx];
        }
        targets
    }

    /// Compute feature selection stability metrics
    fn compute_stability_metrics(
        &self,
        feature_selections: &[Vec<usize>],
    ) -> Result<FeatureStabilityMetrics> {
        if feature_selections.is_empty() {
            return Ok(FeatureStabilityMetrics {
                jaccard_similarity: 0.0,
                intersection_stability: 0.0,
                average_selection_size: 0.0,
                unique_features_selected: 0,
                feature_frequencies: Vec::new(),
            });
        }

        // Compute pairwise Jaccard similarities
        let mut jaccard_similarities = Vec::new();
        for i in 0..feature_selections.len() {
            for j in (i + 1)..feature_selections.len() {
                let set1: std::collections::HashSet<_> = feature_selections[i].iter().collect();
                let set2: std::collections::HashSet<_> = feature_selections[j].iter().collect();

                let intersection = set1.intersection(&set2).count() as f64;
                let union = set1.union(&set2).count() as f64;

                let jaccard = if union > 0.0 {
                    intersection / union
                } else {
                    1.0
                };

                jaccard_similarities.push(jaccard);
            }
        }

        let mean_jaccard = if jaccard_similarities.is_empty() {
            1.0
        } else {
            jaccard_similarities.iter().sum::<f64>() / jaccard_similarities.len() as f64
        };

        // Compute feature frequencies
        let mut feature_counts: HashMap<usize, usize> = HashMap::new();
        let mut total_features = 0;

        for selection in feature_selections {
            total_features += selection.len();
            for &feature in selection {
                *feature_counts.entry(feature).or_insert(0) += 1;
            }
        }

        let average_selection_size = total_features as f64 / feature_selections.len() as f64;

        let mut feature_frequencies: Vec<(usize, f64)> = feature_counts
            .into_iter()
            .map(|(feature, count)| {
                let frequency = count as f64 / feature_selections.len() as f64;
                (feature, frequency)
            })
            .collect();

        feature_frequencies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Intersection stability (features selected in all folds)
        let all_features: std::collections::HashSet<_> = feature_selections[0].iter().collect();
        let intersection_features =
            feature_selections
                .iter()
                .skip(1)
                .fold(all_features, |acc, selection| {
                    let set: std::collections::HashSet<_> = selection.iter().collect();
                    acc.intersection(&set).cloned().collect()
                });

        let intersection_stability = intersection_features.len() as f64 / average_selection_size;

        Ok(FeatureStabilityMetrics {
            jaccard_similarity: mean_jaccard,
            intersection_stability,
            average_selection_size,
            unique_features_selected: feature_frequencies.len(),
            feature_frequencies,
        })
    }
}

/// Stratified K-Fold cross-validation
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl StratifiedKFold {
    /// Create a new stratified K-fold validator
    pub fn new(n_splits: usize, shuffle: bool, random_state: Option<u64>) -> Self {
        Self {
            n_splits,
            shuffle,
            random_state,
        }
    }

    /// Generate stratified splits
    pub fn split(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if X.nrows() != y.len() {
            return Err(CrossValidationError::LengthMismatch.into());
        }

        let indices: Vec<usize> = (0..X.nrows()).collect();
        self.stratified_split(&indices, y)
    }

    fn stratified_split(
        &self,
        indices: &[usize],
        y: ArrayView1<f64>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        // Group indices by class
        let mut class_groups: HashMap<i32, Vec<usize>> = HashMap::new();
        for &idx in indices {
            let class = y[idx] as i32;
            class_groups.entry(class).or_default().push(idx);
        }

        // Check minimum samples per class
        for (class, group) in &class_groups {
            if group.len() < self.n_splits {
                return Err(SklearsError::InvalidInput(format!(
                    "Class {} has only {} samples, need at least {}",
                    class,
                    group.len(),
                    self.n_splits
                )));
            }
        }

        // Shuffle within each class if requested
        if self.shuffle {
            for group in class_groups.values_mut() {
                self.shuffle_indices(group);
            }
        }

        // Create stratified folds
        let mut folds: Vec<Vec<usize>> = vec![Vec::new(); self.n_splits];

        for group in class_groups.values() {
            let fold_size = group.len() / self.n_splits;
            let remainder = group.len() % self.n_splits;

            for fold in 0..self.n_splits {
                let start = fold * fold_size + fold.min(remainder);
                let end = start + fold_size + if fold < remainder { 1 } else { 0 };
                folds[fold].extend_from_slice(&group[start..end]);
            }
        }

        // Create train/test splits
        let mut splits = Vec::new();
        for fold in 0..self.n_splits {
            let test_indices = folds[fold].clone();
            let train_indices: Vec<usize> = folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != fold)
                .flat_map(|(_, fold_indices)| fold_indices.iter())
                .cloned()
                .collect();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    fn shuffle_indices(&self, indices: &mut [usize]) {
        for i in (1..indices.len()).rev() {
            let j = (thread_rng().gen::<f64>() * (i + 1) as f64) as usize;
            indices.swap(i, j);
        }
    }
}

/// Time series cross-validation split
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    n_splits: usize,
    max_train_size: Option<usize>,
    test_size: Option<usize>,
}

impl TimeSeriesSplit {
    /// Create a new time series splitter
    pub fn new(n_splits: usize, max_train_size: Option<usize>, test_size: Option<usize>) -> Self {
        Self {
            n_splits,
            max_train_size,
            test_size,
        }
    }

    /// Generate time series splits
    pub fn split(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if n_samples < self.n_splits + 1 {
            return Err(CrossValidationError::InsufficientData.into());
        }

        let test_size = self.test_size.unwrap_or(n_samples / (self.n_splits + 1));
        let mut splits = Vec::new();

        for split in 0..self.n_splits {
            let test_start = (split + 1) * test_size;
            let test_end = test_start + test_size;

            if test_end > n_samples {
                break;
            }

            let train_end = test_start;
            let train_start = if let Some(max_size) = self.max_train_size {
                train_end.saturating_sub(max_size)
            } else {
                0
            };

            let train_indices: Vec<usize> = (train_start..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            if !train_indices.is_empty() && !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }

        Ok(splits)
    }
}

/// Group K-Fold cross-validation
#[derive(Debug, Clone)]
pub struct GroupKFold {
    n_splits: usize,
}

impl GroupKFold {
    /// Create a new group K-fold validator
    pub fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    /// Generate group-based splits
    pub fn split(&self, groups: &[usize]) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        // Get unique groups
        let mut unique_groups: Vec<usize> = groups.to_vec();
        unique_groups.sort_unstable();
        unique_groups.dedup();

        if unique_groups.len() < self.n_splits {
            return Err(CrossValidationError::InvalidFoldConfiguration.into());
        }

        // Create group index mapping
        let mut group_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &group) in groups.iter().enumerate() {
            group_indices.entry(group).or_default().push(idx);
        }

        // Distribute groups among folds
        let groups_per_fold = unique_groups.len() / self.n_splits;
        let remainder = unique_groups.len() % self.n_splits;

        let mut splits = Vec::new();

        for fold in 0..self.n_splits {
            let start = fold * groups_per_fold + fold.min(remainder);
            let end = start + groups_per_fold + if fold < remainder { 1 } else { 0 };

            let test_groups = &unique_groups[start..end];
            let train_groups: Vec<usize> = unique_groups[..start]
                .iter()
                .chain(unique_groups[end..].iter())
                .cloned()
                .collect();

            let test_indices: Vec<usize> = test_groups
                .iter()
                .flat_map(|&group| group_indices[&group].iter())
                .cloned()
                .collect();

            let train_indices: Vec<usize> = train_groups
                .iter()
                .flat_map(|&group| group_indices[&group].iter())
                .cloned()
                .collect();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }
}

/// Repeated K-Fold cross-validation
#[derive(Debug, Clone)]
pub struct RepeatedKFold {
    n_splits: usize,
    n_repeats: usize,
    random_state: Option<u64>,
}

impl RepeatedKFold {
    /// Create a new repeated K-fold validator
    pub fn new(n_splits: usize, n_repeats: usize, random_state: Option<u64>) -> Self {
        Self {
            n_splits,
            n_repeats,
            random_state,
        }
    }

    /// Generate repeated K-fold splits
    pub fn split(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut all_splits = Vec::new();

        for repeat in 0..self.n_repeats {
            let current_random_state = self.random_state.map(|s| s + repeat as u64);

            // Create K-fold with current random state
            let indices: Vec<usize> = (0..n_samples).collect();
            let kfold_splits = self.k_fold_split(&indices, current_random_state)?;

            all_splits.extend(kfold_splits);
        }

        Ok(all_splits)
    }

    fn k_fold_split(
        &self,
        indices: &[usize],
        random_state: Option<u64>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut shuffled_indices = indices.to_vec();

        if random_state.is_some() {
            self.shuffle_indices(&mut shuffled_indices);
        }

        let fold_size = indices.len() / self.n_splits;
        let remainder = indices.len() % self.n_splits;

        let mut splits = Vec::new();

        for fold in 0..self.n_splits {
            let start = fold * fold_size + fold.min(remainder);
            let end = start + fold_size + if fold < remainder { 1 } else { 0 };

            let test_indices = shuffled_indices[start..end].to_vec();
            let train_indices: Vec<usize> = shuffled_indices[..start]
                .iter()
                .chain(shuffled_indices[end..].iter())
                .cloned()
                .collect();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    fn shuffle_indices(&self, indices: &mut [usize]) {
        for i in (1..indices.len()).rev() {
            let j = (thread_rng().gen::<f64>() * (i + 1) as f64) as usize;
            indices.swap(i, j);
        }
    }
}

/// Results from nested cross-validation
#[derive(Debug, Clone)]
pub struct NestedCVResults {
    pub outer_scores: Vec<f64>,
    pub outer_mean_score: f64,
    pub outer_std_score: f64,
    pub inner_cv_results: Vec<InnerCVResult>,
    pub inner_mean_score: f64,
    pub feature_stability: FeatureStabilityMetrics,
    pub n_outer_folds: usize,
    pub n_inner_folds: usize,
}

impl NestedCVResults {
    /// Generate detailed report
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Nested Cross-Validation Results ===\n\n");

        report.push_str(&format!(
            "Configuration: {} outer folds, {} inner folds\n\n",
            self.n_outer_folds, self.n_inner_folds
        ));

        report.push_str("Outer CV Performance:\n");
        report.push_str(&format!(
            "  Mean Score: {:.4} ± {:.4}\n",
            self.outer_mean_score, self.outer_std_score
        ));
        report.push_str(&format!(
            "  Individual Scores: {:?}\n\n",
            self.outer_scores
                .iter()
                .map(|s| format!("{:.4}", s))
                .collect::<Vec<_>>()
        ));

        report.push_str("Inner CV Performance:\n");
        report.push_str(&format!("  Mean Score: {:.4}\n", self.inner_mean_score));

        for (i, inner_result) in self.inner_cv_results.iter().enumerate() {
            report.push_str(&format!(
                "  Outer Fold {}: {:.4} ± {:.4}\n",
                i,
                inner_result.mean_score,
                inner_result.std_score()
            ));
        }

        report.push_str("\nFeature Selection Stability:\n");
        report.push_str(&format!(
            "  Jaccard Similarity: {:.4}\n",
            self.feature_stability.jaccard_similarity
        ));
        report.push_str(&format!(
            "  Intersection Stability: {:.4}\n",
            self.feature_stability.intersection_stability
        ));
        report.push_str(&format!(
            "  Average Selection Size: {:.1}\n",
            self.feature_stability.average_selection_size
        ));
        report.push_str(&format!(
            "  Unique Features Selected: {}\n",
            self.feature_stability.unique_features_selected
        ));

        if !self.feature_stability.feature_frequencies.is_empty() {
            report.push_str("\nTop 10 Most Frequent Features:\n");
            for (feature, frequency) in self.feature_stability.feature_frequencies.iter().take(10) {
                report.push_str(&format!(
                    "  Feature {}: {:.1}%\n",
                    feature,
                    frequency * 100.0
                ));
            }
        }

        report
    }
}

/// Inner cross-validation result for one outer fold
#[derive(Debug, Clone)]
pub struct InnerCVResult {
    pub outer_fold: usize,
    pub inner_scores: Vec<f64>,
    pub mean_score: f64,
    pub selected_features: Vec<Vec<usize>>,
}

impl InnerCVResult {
    pub fn std_score(&self) -> f64 {
        if self.inner_scores.len() <= 1 {
            return 0.0;
        }

        let variance = self
            .inner_scores
            .iter()
            .map(|score| (score - self.mean_score).powi(2))
            .sum::<f64>()
            / self.inner_scores.len() as f64;
        variance.sqrt()
    }
}

/// Feature stability metrics from cross-validation
#[derive(Debug, Clone)]
pub struct FeatureStabilityMetrics {
    pub jaccard_similarity: f64,
    pub intersection_stability: f64,
    pub average_selection_size: f64,
    pub unique_features_selected: usize,
    pub feature_frequencies: Vec<(usize, f64)>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Mock feature selector for testing
    fn mock_feature_selector(X: ArrayView2<f64>, _y: ArrayView1<f64>) -> Result<Vec<usize>> {
        // Select first half of features
        let n_features = X.ncols();
        Ok((0..(n_features / 2)).collect())
    }

    // Mock performance evaluator for testing
    fn mock_performance_evaluator(
        _X_train: ArrayView2<f64>,
        _y_train: ArrayView1<f64>,
        _X_test: ArrayView2<f64>,
        _y_test: ArrayView1<f64>,
        _features: &[usize],
    ) -> Result<f64> {
        // Return random score between 0.7 and 0.9
        Ok(0.7 + thread_rng().gen::<f64>() * 0.2)
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_nested_cross_validation() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0],
            [7.0, 8.0, 9.0, 10.0],
            [8.0, 9.0, 10.0, 11.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];

        let nested_cv = NestedCrossValidation::new(3, 2, false, Some(42));
        let results = nested_cv
            .evaluate(
                X.view(),
                y.view(),
                mock_feature_selector,
                mock_performance_evaluator,
            )
            .unwrap();

        assert_eq!(results.outer_scores.len(), 3);
        assert_eq!(results.inner_cv_results.len(), 3);
        assert!(results.outer_mean_score >= 0.0 && results.outer_mean_score <= 1.0);
        assert!(results.feature_stability.jaccard_similarity >= 0.0);

        let report = results.report();
        assert!(report.contains("Nested Cross-Validation"));
        assert!(report.contains("Feature Selection Stability"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stratified_k_fold() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];

        let skf = StratifiedKFold::new(3, true, Some(42));
        let splits = skf.split(X.view(), y.view()).unwrap();

        assert_eq!(splits.len(), 3);

        for (train_idx, test_idx) in splits {
            assert!(!train_idx.is_empty());
            assert!(!test_idx.is_empty());
            assert_eq!(train_idx.len() + test_idx.len(), X.nrows());
        }
    }

    #[test]
    fn test_time_series_split() {
        let ts_split = TimeSeriesSplit::new(3, None, Some(2));
        let splits = ts_split.split(10).unwrap();

        assert_eq!(splits.len(), 3);

        for (train_idx, test_idx) in splits {
            assert!(!train_idx.is_empty());
            assert_eq!(test_idx.len(), 2);

            // Verify temporal order
            if !train_idx.is_empty() && !test_idx.is_empty() {
                let max_train = train_idx.iter().max().unwrap();
                let min_test = test_idx.iter().min().unwrap();
                assert!(max_train < min_test);
            }
        }
    }

    #[test]
    fn test_group_k_fold() {
        let groups = vec![0, 0, 1, 1, 2, 2];
        let gkf = GroupKFold::new(3);
        let splits = gkf.split(&groups).unwrap();

        assert_eq!(splits.len(), 3);

        for (train_idx, test_idx) in splits {
            assert!(!train_idx.is_empty());
            assert!(!test_idx.is_empty());
            assert_eq!(train_idx.len() + test_idx.len(), groups.len());
        }
    }

    #[test]
    fn test_repeated_k_fold() {
        let rkf = RepeatedKFold::new(3, 2, Some(42));
        let splits = rkf.split(9).unwrap();

        assert_eq!(splits.len(), 6); // 3 folds * 2 repeats

        for (train_idx, test_idx) in splits {
            assert!(!train_idx.is_empty());
            assert!(!test_idx.is_empty());
            assert_eq!(train_idx.len() + test_idx.len(), 9);
        }
    }
}

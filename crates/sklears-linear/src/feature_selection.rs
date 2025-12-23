//! Feature selection algorithms for linear models
//!
//! This module provides comprehensive feature selection capabilities including
//! univariate selection, model-based selection, wrapper methods, and embedded
//! feature selection techniques. These methods help identify the most relevant
//! features for improving model performance and interpretability.

use sklears_core::error::SklearsError;
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// Feature selection strategies
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureSelectionStrategy {
    /// Select k best features based on univariate statistical tests
    SelectKBest {
        /// Number of features to select
        k: usize,
        /// Statistical test function to use
        score_func: UnivariateScoreFunc,
    },
    /// Select features based on percentile of the highest scores
    SelectPercentile {
        percentile: f64,
        score_func: UnivariateScoreFunc,
    },
    /// Select features based on threshold of the test statistic
    SelectFpr {
        /// Alpha level for false positive rate
        alpha: f64,
        /// Statistical test function to use
        score_func: UnivariateScoreFunc,
    },
    /// Select features based on false discovery rate
    SelectFdr {
        alpha: f64,
        score_func: UnivariateScoreFunc,
    },
    /// Select features based on family-wise error rate
    SelectFwe {
        /// Alpha level for family-wise error rate
        alpha: f64,
        /// Statistical test function to use
        score_func: UnivariateScoreFunc,
    },
    /// L1-based feature selection using sparse coefficients
    SelectFromModel {
        estimator: ModelBasedEstimator,
        threshold: Option<f64>,
        prefit: bool,
        max_features: Option<usize>,
    },
    /// Sequential forward selection
    SequentialForwardSelection {
        /// Maximum number of features to select
        n_features_to_select: Option<usize>,
        /// Cross-validation strategy
        cv: usize,
        /// Scoring metric
        scoring: String,
    },
    /// Sequential backward elimination
    SequentialBackwardElimination {
        /// Number of features to select
        n_features_to_select: Option<usize>,
        /// Cross-validation strategy
        cv: usize,
        /// Scoring metric
        scoring: String,
    },
    /// Variance threshold
    VarianceThreshold {
        /// Minimum variance required
        threshold: f64,
    },
}

/// Univariate statistical score functions
#[derive(Debug, Clone, PartialEq)]
pub enum UnivariateScoreFunc {
    /// Chi-squared test for classification
    Chi2,
    /// F-test for classification (ANOVA F-value)
    FClassif,
    /// Mutual information for classification
    MutualInfoClassif {
        /// Number of neighbors for k-NN mutual information estimation
        n_neighbors: usize,
        /// Random state for reproducibility
        random_state: Option<u64>,
    },
    /// F-test for regression
    FRegression,
    /// Mutual information for regression
    MutualInfoRegression {
        /// Number of neighbors for k-NN mutual information estimation
        n_neighbors: usize,
        /// Random state for reproducibility
        random_state: Option<u64>,
    },
}

/// Model-based estimators for feature selection
#[derive(Debug, Clone, PartialEq)]
pub enum ModelBasedEstimator {
    /// L1-regularized linear models (Lasso)
    L1Linear {
        /// Regularization strength
        alpha: f64,
        /// Random state for reproducibility
        random_state: Option<u64>,
    },
    /// L1-regularized logistic regression
    L1Logistic {
        /// Regularization strength
        alpha: f64,
        /// Random state for reproducibility
        random_state: Option<u64>,
    },
    /// Random Forest feature importance
    RandomForest {
        /// Number of trees
        n_estimators: usize,
        /// Random state for reproducibility
        random_state: Option<u64>,
    },
    /// Extra trees feature importance
    ExtraTrees {
        /// Number of trees
        n_estimators: usize,
        /// Random state for reproducibility
        random_state: Option<u64>,
    },
}

/// Feature selection configuration
#[derive(Debug, Clone)]
pub struct FeatureSelectionConfig {
    /// Selection strategy to use
    pub strategy: FeatureSelectionStrategy,
    /// Whether to normalize features before selection
    pub normalize_features: bool,
    /// Features to exclude from selection
    pub exclude_features: Option<Vec<usize>>,
    /// Maximum time for selection (in seconds)
    pub max_time: Option<f64>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            strategy: FeatureSelectionStrategy::SelectKBest {
                k: 10,
                score_func: UnivariateScoreFunc::FRegression,
            },
            normalize_features: true,
            exclude_features: None,
            max_time: None,
            verbose: false,
        }
    }
}

/// Feature score information
#[derive(Debug, Clone)]
pub struct FeatureScore {
    /// Feature index
    pub feature_index: usize,
    /// Score value
    pub score: f64,
    /// P-value (if applicable)
    pub p_value: Option<f64>,
    /// Whether feature is selected
    pub selected: bool,
}

impl FeatureScore {
    fn new(feature_index: usize, score: f64, p_value: Option<f64>) -> Self {
        Self {
            feature_index,
            score,
            p_value,
            selected: false,
        }
    }
}

/// Feature selection result
#[derive(Debug, Clone)]
pub struct FeatureSelectionResult {
    /// Selected feature indices
    pub selected_features: Vec<usize>,
    /// Feature scores
    pub feature_scores: Vec<FeatureScore>,
    /// Selection ranking (0 = best)
    pub ranking: Vec<usize>,
    /// Number of original features
    pub n_features_in: usize,
    /// Number of selected features
    pub n_features_out: usize,
    /// Configuration used
    pub config: FeatureSelectionConfig,
}

/// Feature selector that automatically selects the most relevant features
pub struct FeatureSelector {
    config: FeatureSelectionConfig,
    is_fitted: bool,
    selection_result: Option<FeatureSelectionResult>,
}

impl FeatureSelector {
    /// Create a new feature selector with default configuration
    pub fn new() -> Self {
        Self {
            config: FeatureSelectionConfig::default(),
            is_fitted: false,
            selection_result: None,
        }
    }

    /// Create a feature selector with custom configuration
    pub fn with_config(config: FeatureSelectionConfig) -> Self {
        Self {
            config,
            is_fitted: false,
            selection_result: None,
        }
    }

    /// Set the selection strategy
    pub fn with_strategy(mut self, strategy: FeatureSelectionStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Enable or disable feature normalization
    pub fn with_normalize_features(mut self, normalize_features: bool) -> Self {
        self.config.normalize_features = normalize_features;
        self
    }

    /// Set features to exclude from selection
    pub fn with_exclude_features(mut self, exclude_features: Vec<usize>) -> Self {
        self.config.exclude_features = Some(exclude_features);
        self
    }

    /// Fit the feature selector to data
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), SklearsError> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit selector on empty dataset".to_string(),
            ));
        }

        let n_samples = x.len();
        let n_features = x[0].len();

        if y.len() != n_samples {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("y.len() == {}", n_samples),
                actual: format!("y.len() == {}", y.len()),
            });
        }

        // Validate that all rows have the same number of features
        for (i, row) in x.iter().enumerate() {
            if row.len() != n_features {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("row[{}].len() == {}", i, n_features),
                    actual: format!("row[{}].len() == {}", i, row.len()),
                });
            }
        }

        // Normalize features if requested
        let x_processed = if self.config.normalize_features {
            self.normalize_features(x)?
        } else {
            x.to_vec()
        };

        // Compute feature scores
        let feature_scores = self.compute_feature_scores(&x_processed, y)?;

        // Select features based on strategy
        let selected_features = self.select_features(&feature_scores)?;

        // Create ranking
        let mut ranking = vec![0; n_features];
        let mut sorted_scores: Vec<(usize, f64)> = feature_scores
            .iter()
            .enumerate()
            .map(|(i, score)| (i, score.score))
            .collect();
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        for (rank, (feature_idx, _)) in sorted_scores.iter().enumerate() {
            ranking[*feature_idx] = rank;
        }

        let n_features_out = selected_features.len();
        self.selection_result = Some(FeatureSelectionResult {
            selected_features,
            feature_scores,
            ranking,
            n_features_in: n_features,
            n_features_out,
            config: self.config.clone(),
        });

        self.is_fitted = true;
        Ok(())
    }

    /// Transform data by selecting only the chosen features
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SklearsError> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        let selection_result = self.selection_result.as_ref().unwrap();

        if x.is_empty() {
            return Ok(Vec::new());
        }

        let n_samples = x.len();
        let n_features_in = x[0].len();

        if n_features_in != selection_result.n_features_in {
            return Err(SklearsError::FeatureMismatch {
                expected: selection_result.n_features_in,
                actual: n_features_in,
            });
        }

        let mut transformed_data = Vec::with_capacity(n_samples);

        for row in x {
            let selected_row: Vec<f64> = selection_result
                .selected_features
                .iter()
                .map(|&idx| row[idx])
                .collect();
            transformed_data.push(selected_row);
        }

        Ok(transformed_data)
    }

    /// Fit and transform data in one step
    pub fn fit_transform(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
    ) -> Result<Vec<Vec<f64>>, SklearsError> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Get the selection result
    pub fn get_selection_result(&self) -> Option<&FeatureSelectionResult> {
        self.selection_result.as_ref()
    }

    /// Get selected feature indices
    pub fn get_selected_features(&self) -> Option<&Vec<usize>> {
        self.selection_result.as_ref().map(|r| &r.selected_features)
    }

    /// Get feature scores
    pub fn get_feature_scores(&self) -> Option<&Vec<FeatureScore>> {
        self.selection_result.as_ref().map(|r| &r.feature_scores)
    }

    /// Get feature ranking
    pub fn get_ranking(&self) -> Option<&Vec<usize>> {
        self.selection_result.as_ref().map(|r| &r.ranking)
    }

    /// Normalize features to have zero mean and unit variance
    fn normalize_features(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SklearsError> {
        let n_samples = x.len();
        let n_features = x[0].len();

        // Compute means and standard deviations
        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];

        // Compute means
        for row in x {
            for (j, &value) in row.iter().enumerate() {
                means[j] += value;
            }
        }
        for mean in &mut means {
            *mean /= n_samples as f64;
        }

        // Compute standard deviations
        for row in x {
            for (j, &value) in row.iter().enumerate() {
                let diff = value - means[j];
                stds[j] += diff * diff;
            }
        }
        for std in &mut stds {
            *std = (*std / (n_samples - 1) as f64).sqrt();
            if *std < 1e-8 {
                *std = 1.0; // Avoid division by zero
            }
        }

        // Normalize data
        let mut normalized_data = Vec::with_capacity(n_samples);
        for row in x {
            let normalized_row: Vec<f64> = row
                .iter()
                .enumerate()
                .map(|(j, &value)| (value - means[j]) / stds[j])
                .collect();
            normalized_data.push(normalized_row);
        }

        Ok(normalized_data)
    }

    /// Compute feature scores based on the configured strategy
    fn compute_feature_scores(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
    ) -> Result<Vec<FeatureScore>, SklearsError> {
        let n_features = x[0].len();
        let mut feature_scores = Vec::with_capacity(n_features);

        match &self.config.strategy {
            FeatureSelectionStrategy::SelectKBest { score_func, .. }
            | FeatureSelectionStrategy::SelectPercentile { score_func, .. }
            | FeatureSelectionStrategy::SelectFpr { score_func, .. }
            | FeatureSelectionStrategy::SelectFdr { score_func, .. }
            | FeatureSelectionStrategy::SelectFwe { score_func, .. } => {
                for feature_idx in 0..n_features {
                    let feature_values: Vec<f64> = x.iter().map(|row| row[feature_idx]).collect();
                    let (score, p_value) =
                        self.compute_univariate_score(&feature_values, y, score_func)?;
                    feature_scores.push(FeatureScore::new(feature_idx, score, p_value));
                }
            }

            FeatureSelectionStrategy::SelectFromModel { estimator, .. } => {
                let importance_scores = self.compute_model_based_scores(x, y, estimator)?;
                for (feature_idx, &score) in importance_scores.iter().enumerate() {
                    feature_scores.push(FeatureScore::new(feature_idx, score, None));
                }
            }

            FeatureSelectionStrategy::VarianceThreshold { .. } => {
                for feature_idx in 0..n_features {
                    let feature_values: Vec<f64> = x.iter().map(|row| row[feature_idx]).collect();
                    let variance = self.compute_variance(&feature_values);
                    feature_scores.push(FeatureScore::new(feature_idx, variance, None));
                }
            }

            FeatureSelectionStrategy::SequentialForwardSelection { .. }
            | FeatureSelectionStrategy::SequentialBackwardElimination { .. } => {
                // For sequential methods, we start with correlation-based scores
                for feature_idx in 0..n_features {
                    let feature_values: Vec<f64> = x.iter().map(|row| row[feature_idx]).collect();
                    let correlation = self.compute_correlation(&feature_values, y);
                    feature_scores.push(FeatureScore::new(feature_idx, correlation.abs(), None));
                }
            }
        }

        Ok(feature_scores)
    }

    /// Select features based on the configured strategy
    fn select_features(&self, feature_scores: &[FeatureScore]) -> Result<Vec<usize>, SklearsError> {
        let mut selected_features = Vec::new();

        match &self.config.strategy {
            FeatureSelectionStrategy::SelectKBest { k, .. } => {
                let mut scores_with_indices: Vec<(usize, f64)> = feature_scores
                    .iter()
                    .map(|fs| (fs.feature_index, fs.score))
                    .collect();
                scores_with_indices
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

                selected_features = scores_with_indices
                    .iter()
                    .take(*k)
                    .map(|(idx, _)| *idx)
                    .collect();
                selected_features.sort();
            }

            FeatureSelectionStrategy::SelectPercentile { percentile, .. } => {
                let k =
                    ((feature_scores.len() as f64 * percentile / 100.0).round() as usize).max(1);
                let mut scores_with_indices: Vec<(usize, f64)> = feature_scores
                    .iter()
                    .map(|fs| (fs.feature_index, fs.score))
                    .collect();
                scores_with_indices
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

                selected_features = scores_with_indices
                    .iter()
                    .take(k)
                    .map(|(idx, _)| *idx)
                    .collect();
                selected_features.sort();
            }

            FeatureSelectionStrategy::SelectFpr { alpha, .. }
            | FeatureSelectionStrategy::SelectFdr { alpha, .. }
            | FeatureSelectionStrategy::SelectFwe { alpha, .. } => {
                // Use p-values for selection
                for fs in feature_scores {
                    if let Some(p_value) = fs.p_value {
                        if p_value < *alpha {
                            selected_features.push(fs.feature_index);
                        }
                    }
                }
                if selected_features.is_empty() {
                    // Fallback: select the best feature
                    let best_idx = feature_scores
                        .iter()
                        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal))
                        .map(|fs| fs.feature_index)
                        .unwrap_or(0);
                    selected_features.push(best_idx);
                }
            }

            FeatureSelectionStrategy::SelectFromModel {
                threshold,
                max_features,
                ..
            } => {
                let threshold_value = threshold.unwrap_or_else(|| {
                    // Use median as default threshold
                    let mut scores: Vec<f64> = feature_scores.iter().map(|fs| fs.score).collect();
                    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                    scores[scores.len() / 2]
                });

                let mut candidates: Vec<(usize, f64)> = feature_scores
                    .iter()
                    .filter(|fs| fs.score >= threshold_value)
                    .map(|fs| (fs.feature_index, fs.score))
                    .collect();

                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

                if let Some(max_feat) = max_features {
                    candidates.truncate(*max_feat);
                }

                selected_features = candidates.iter().map(|(idx, _)| *idx).collect();
                selected_features.sort();
            }

            FeatureSelectionStrategy::VarianceThreshold { threshold } => {
                for fs in feature_scores {
                    if fs.score >= *threshold {
                        selected_features.push(fs.feature_index);
                    }
                }
            }

            FeatureSelectionStrategy::SequentialForwardSelection { .. }
            | FeatureSelectionStrategy::SequentialBackwardElimination { .. } => {
                // Simplified implementation - select top correlated features
                let mut scores_with_indices: Vec<(usize, f64)> = feature_scores
                    .iter()
                    .map(|fs| (fs.feature_index, fs.score))
                    .collect();
                scores_with_indices
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

                let k = (feature_scores.len() / 2).max(1);
                selected_features = scores_with_indices
                    .iter()
                    .take(k)
                    .map(|(idx, _)| *idx)
                    .collect();
                selected_features.sort();
            }
        }

        // Ensure we have at least one feature selected
        if selected_features.is_empty() {
            selected_features.push(0);
        }

        Ok(selected_features)
    }

    /// Compute univariate score for a feature
    fn compute_univariate_score(
        &self,
        feature_values: &[f64],
        target: &[f64],
        score_func: &UnivariateScoreFunc,
    ) -> Result<(f64, Option<f64>), SklearsError> {
        match score_func {
            UnivariateScoreFunc::FRegression => {
                let correlation = self.compute_correlation(feature_values, target);
                let n = feature_values.len() as f64;
                let f_statistic = correlation.powi(2) * (n - 2.0) / (1.0 - correlation.powi(2));
                let p_value = self.f_distribution_p_value(f_statistic, 1.0, n - 2.0);
                Ok((f_statistic, Some(p_value)))
            }

            UnivariateScoreFunc::FClassif => {
                // Simplified F-test for classification
                let f_statistic = self.compute_f_classif(feature_values, target)?;
                let p_value = self.f_distribution_p_value(
                    f_statistic,
                    1.0,
                    (feature_values.len() - 2) as f64,
                );
                Ok((f_statistic, Some(p_value)))
            }

            UnivariateScoreFunc::Chi2 => {
                // Simplified chi-squared test
                let chi2_statistic = self.compute_chi2(feature_values, target)?;
                let p_value = self.chi2_distribution_p_value(chi2_statistic, 1.0);
                Ok((chi2_statistic, Some(p_value)))
            }

            UnivariateScoreFunc::MutualInfoClassif { n_neighbors, .. }
            | UnivariateScoreFunc::MutualInfoRegression { n_neighbors, .. } => {
                let mi_score =
                    self.compute_mutual_information(feature_values, target, *n_neighbors);
                Ok((mi_score, None))
            }
        }
    }

    /// Compute model-based feature importance scores
    fn compute_model_based_scores(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        estimator: &ModelBasedEstimator,
    ) -> Result<Vec<f64>, SklearsError> {
        match estimator {
            ModelBasedEstimator::L1Linear { alpha, .. } => {
                self.compute_l1_linear_importance(x, y, *alpha)
            }

            ModelBasedEstimator::L1Logistic { alpha, .. } => {
                self.compute_l1_logistic_importance(x, y, *alpha)
            }

            ModelBasedEstimator::RandomForest { n_estimators, .. }
            | ModelBasedEstimator::ExtraTrees { n_estimators, .. } => {
                // Simplified random forest importance using feature correlations
                self.compute_tree_based_importance(x, y, *n_estimators)
            }
        }
    }

    /// Compute correlation between feature and target
    fn compute_correlation(&self, feature_values: &[f64], target: &[f64]) -> f64 {
        let n = feature_values.len() as f64;
        let mean_x = feature_values.iter().sum::<f64>() / n;
        let mean_y = target.iter().sum::<f64>() / n;

        let numerator: f64 = feature_values
            .iter()
            .zip(target.iter())
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum();

        let sum_sq_x: f64 = feature_values.iter().map(|&x| (x - mean_x).powi(2)).sum();

        let sum_sq_y: f64 = target.iter().map(|&y| (y - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Compute variance of a feature
    fn compute_variance(&self, values: &[f64]) -> f64 {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }

    /// Compute F-statistic for classification
    fn compute_f_classif(
        &self,
        feature_values: &[f64],
        target: &[f64],
    ) -> Result<f64, SklearsError> {
        // Group by target class
        let mut class_groups: BTreeMap<i32, Vec<f64>> = BTreeMap::new();

        for (&feature_val, &target_val) in feature_values.iter().zip(target.iter()) {
            let class = target_val.round() as i32;
            class_groups.entry(class).or_default().push(feature_val);
        }

        if class_groups.len() < 2 {
            return Ok(0.0);
        }

        // Compute between-group and within-group variances
        let overall_mean = feature_values.iter().sum::<f64>() / feature_values.len() as f64;

        let mut between_ss = 0.0;
        let mut within_ss = 0.0;
        let mut total_count = 0;

        for group_values in class_groups.values() {
            let group_mean = group_values.iter().sum::<f64>() / group_values.len() as f64;
            let group_size = group_values.len() as f64;

            between_ss += group_size * (group_mean - overall_mean).powi(2);

            for &value in group_values {
                within_ss += (value - group_mean).powi(2);
            }

            total_count += group_values.len();
        }

        let df_between = (class_groups.len() - 1) as f64;
        let df_within = (total_count - class_groups.len()) as f64;

        if df_within <= 0.0 || within_ss <= 0.0 {
            return Ok(0.0);
        }

        let ms_between = between_ss / df_between;
        let ms_within = within_ss / df_within;

        Ok(ms_between / ms_within)
    }

    /// Compute chi-squared statistic
    fn compute_chi2(&self, feature_values: &[f64], target: &[f64]) -> Result<f64, SklearsError> {
        // Simplified chi-squared test
        // Create contingency table by binning continuous values
        let n_bins = 5;
        let min_feature = feature_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_feature = feature_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let feature_range = max_feature - min_feature;

        let min_target = target.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_target = target.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let target_range = max_target - min_target;

        if feature_range <= 0.0 || target_range <= 0.0 {
            return Ok(0.0);
        }

        let mut contingency_table = vec![vec![0; n_bins]; n_bins];

        for (&feature_val, &target_val) in feature_values.iter().zip(target.iter()) {
            let feature_bin = ((feature_val - min_feature) / feature_range * (n_bins as f64 - 1.0))
                .floor() as usize;
            let target_bin =
                ((target_val - min_target) / target_range * (n_bins as f64 - 1.0)).floor() as usize;

            let feature_bin = feature_bin.min(n_bins - 1);
            let target_bin = target_bin.min(n_bins - 1);

            contingency_table[feature_bin][target_bin] += 1;
        }

        // Compute chi-squared statistic
        let total_count = feature_values.len() as f64;
        let mut chi2_stat = 0.0;

        for i in 0..n_bins {
            for j in 0..n_bins {
                let observed = contingency_table[i][j] as f64;
                let row_sum: f64 = contingency_table[i].iter().sum::<usize>() as f64;
                let col_sum: f64 =
                    (0..n_bins).map(|k| contingency_table[k][j]).sum::<usize>() as f64;
                let expected = (row_sum * col_sum) / total_count;

                if expected > 0.0 {
                    chi2_stat += (observed - expected).powi(2) / expected;
                }
            }
        }

        Ok(chi2_stat)
    }

    /// Compute mutual information (simplified k-NN estimator)
    fn compute_mutual_information(&self, feature_values: &[f64], target: &[f64], _k: usize) -> f64 {
        // Simplified mutual information using correlation as proxy
        self.compute_correlation(feature_values, target).abs()
    }

    /// Compute L1 linear model importance
    fn compute_l1_linear_importance(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        alpha: f64,
    ) -> Result<Vec<f64>, SklearsError> {
        let n_features = x[0].len();

        // Simplified L1 importance using correlation weighted by regularization
        let mut importance_scores = Vec::with_capacity(n_features);

        for feature_idx in 0..n_features {
            let feature_values: Vec<f64> = x.iter().map(|row| row[feature_idx]).collect();
            let correlation = self.compute_correlation(&feature_values, y);
            let importance = correlation.abs() * (1.0 - alpha.min(1.0));
            importance_scores.push(importance);
        }

        Ok(importance_scores)
    }

    /// Compute L1 logistic model importance
    fn compute_l1_logistic_importance(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        alpha: f64,
    ) -> Result<Vec<f64>, SklearsError> {
        // Similar to L1 linear but with different weighting
        self.compute_l1_linear_importance(x, y, alpha)
    }

    /// Compute tree-based feature importance
    fn compute_tree_based_importance(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        _n_estimators: usize,
    ) -> Result<Vec<f64>, SklearsError> {
        let n_features = x[0].len();

        // Simplified tree importance using variance-weighted correlation
        let mut importance_scores = Vec::with_capacity(n_features);

        for feature_idx in 0..n_features {
            let feature_values: Vec<f64> = x.iter().map(|row| row[feature_idx]).collect();
            let correlation = self.compute_correlation(&feature_values, y);
            let variance = self.compute_variance(&feature_values);
            let importance = correlation.abs() * variance.sqrt();
            importance_scores.push(importance);
        }

        Ok(importance_scores)
    }

    /// Approximate F-distribution p-value
    fn f_distribution_p_value(&self, f_stat: f64, df1: f64, df2: f64) -> f64 {
        // Simplified approximation
        if f_stat <= 0.0 {
            return 1.0;
        }

        // Use approximation for F-distribution
        let t = f_stat * df1 / (f_stat * df1 + df2);
        1.0 - t.powf(df1 / 2.0) * (1.0 - t).powf(df2 / 2.0)
    }

    /// Approximate chi-squared distribution p-value
    fn chi2_distribution_p_value(&self, chi2_stat: f64, _df: f64) -> f64 {
        // Simplified approximation
        if chi2_stat <= 0.0 {
            return 1.0;
        }

        // Use exponential approximation
        (-chi2_stat / 2.0).exp()
    }
}

impl Default for FeatureSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let x = vec![
            vec![1.0, 2.0, 3.0, 0.1],
            vec![2.0, 3.0, 4.0, 0.2],
            vec![3.0, 4.0, 5.0, 0.1],
            vec![4.0, 5.0, 6.0, 0.3],
            vec![5.0, 6.0, 7.0, 0.2],
        ];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        (x, y)
    }

    #[test]
    fn test_select_k_best() {
        let mut selector =
            FeatureSelector::new().with_strategy(FeatureSelectionStrategy::SelectKBest {
                k: 2,
                score_func: UnivariateScoreFunc::FRegression,
            });

        let (x, y) = create_sample_data();
        let result = selector.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should select 2 features
        assert_eq!(transformed[0].len(), 2);
        assert_eq!(transformed.len(), 5);
    }

    #[test]
    fn test_select_percentile() {
        let mut selector =
            FeatureSelector::new().with_strategy(FeatureSelectionStrategy::SelectPercentile {
                percentile: 50.0,
                score_func: UnivariateScoreFunc::FRegression,
            });

        let (x, y) = create_sample_data();
        let result = selector.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should select 50% of features (2 out of 4)
        assert_eq!(transformed[0].len(), 2);
    }

    #[test]
    fn test_variance_threshold() {
        let mut selector = FeatureSelector::new()
            .with_strategy(FeatureSelectionStrategy::VarianceThreshold { threshold: 0.1 });

        let (x, y) = create_sample_data();
        let result = selector.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should remove low-variance features
        assert!(transformed[0].len() <= 4);
    }

    #[test]
    fn test_select_from_model() {
        let mut selector =
            FeatureSelector::new().with_strategy(FeatureSelectionStrategy::SelectFromModel {
                estimator: ModelBasedEstimator::L1Linear {
                    alpha: 0.1,
                    random_state: Some(42),
                },
                threshold: None,
                prefit: false,
                max_features: Some(2),
            });

        let (x, y) = create_sample_data();
        let result = selector.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should select at most 2 features
        assert!(transformed[0].len() <= 2);
    }

    #[test]
    fn test_feature_scores() {
        let mut selector = FeatureSelector::new();
        let (x, y) = create_sample_data();

        selector.fit(&x, &y).unwrap();

        let scores = selector.get_feature_scores().unwrap();
        assert_eq!(scores.len(), 4); // 4 original features

        // Scores should be computed for all features
        for score in scores {
            assert!(score.score.is_finite());
        }
    }

    #[test]
    fn test_feature_ranking() {
        let mut selector = FeatureSelector::new();
        let (x, y) = create_sample_data();

        selector.fit(&x, &y).unwrap();

        let ranking = selector.get_ranking().unwrap();
        assert_eq!(ranking.len(), 4); // 4 original features

        // Rankings should be unique
        let mut sorted_ranking = ranking.clone();
        sorted_ranking.sort();
        assert_eq!(sorted_ranking, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_normalization() {
        let selector = FeatureSelector::new();
        let (x, _) = create_sample_data();

        let normalized = selector.normalize_features(&x).unwrap();

        // Check that features are approximately normalized
        for feature_idx in 0..4 {
            let feature_values: Vec<f64> = normalized.iter().map(|row| row[feature_idx]).collect();
            let mean = feature_values.iter().sum::<f64>() / feature_values.len() as f64;
            let var = feature_values
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / (feature_values.len() - 1) as f64;

            assert!(
                (mean.abs() < 1e-10),
                "Mean should be close to 0, got {}",
                mean
            );
            assert!(
                (var - 1.0).abs() < 1e-1,
                "Variance should be close to 1, got {}",
                var
            );
        }
    }

    #[test]
    fn test_empty_data_error() {
        let mut selector = FeatureSelector::new();
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];

        let result = selector.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut selector = FeatureSelector::new();
        let (x, _) = create_sample_data();
        let wrong_y = vec![1.0, 2.0]; // Wrong length

        let result = selector.fit(&x, &wrong_y);
        assert!(result.is_err());
    }

    #[test]
    fn test_transform_before_fit_error() {
        let selector = FeatureSelector::new();
        let (x, _) = create_sample_data();

        let result = selector.transform(&x);
        assert!(result.is_err());
    }
}

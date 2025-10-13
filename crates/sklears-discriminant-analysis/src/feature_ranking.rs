//! Discriminant Feature Ranking implementation
//!
//! This module implements feature ranking methods based on discriminant analysis
//! to identify the most informative features for classification.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{validate, Result},
    prelude::SklearsError,
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for Discriminant Feature Ranking
#[derive(Debug, Clone)]
pub struct DiscriminantFeatureRankingConfig {
    /// Ranking method ("fisher_score", "mutual_info", "chi2", "anova_f", "relief")
    pub ranking_method: String,
    /// Number of top features to select
    pub k: Option<usize>,
    /// Threshold for feature selection (alternative to k)
    pub threshold: Option<Float>,
    /// Whether to use absolute values for ranking
    pub use_absolute: bool,
    /// Whether to normalize features before ranking
    pub normalize: bool,
    /// Regularization parameter for numerical stability
    pub reg_param: Float,
    /// Number of neighbors for Relief algorithm
    pub n_neighbors: usize,
}

impl Default for DiscriminantFeatureRankingConfig {
    fn default() -> Self {
        Self {
            ranking_method: "fisher_score".to_string(),
            k: None,
            threshold: None,
            use_absolute: true,
            normalize: false,
            reg_param: 1e-6,
            n_neighbors: 10,
        }
    }
}

/// Feature ranking result
#[derive(Debug, Clone)]
pub struct FeatureRank {
    /// Feature index
    pub feature_idx: usize,
    /// Ranking score
    pub score: Float,
    /// Rank (1-based, lower is better)
    pub rank: usize,
}

/// Discriminant Feature Ranking
///
/// Ranks features based on their discriminative power for classification.
/// Supports multiple ranking methods:
/// - Fisher Score: ratio of between-class to within-class variance
/// - Mutual Information: information shared between feature and class
/// - Chi-squared: chi-squared statistic for categorical features
/// - ANOVA F-statistic: F-statistic from one-way ANOVA
/// - Relief: considers feature values of nearest neighbors
#[derive(Debug, Clone)]
pub struct DiscriminantFeatureRanking<State = Untrained> {
    config: DiscriminantFeatureRankingConfig,
    state: PhantomData<State>,
    // Trained state fields
    classes_: Option<Array1<i32>>,
    feature_scores_: Option<Array1<Float>>,
    feature_ranks_: Option<Vec<FeatureRank>>,
    selected_features_: Option<Array1<usize>>,
    n_features_: Option<usize>,
}

impl DiscriminantFeatureRanking<Untrained> {
    /// Create a new DiscriminantFeatureRanking instance
    pub fn new() -> Self {
        Self {
            config: DiscriminantFeatureRankingConfig::default(),
            state: PhantomData,
            classes_: None,
            feature_scores_: None,
            feature_ranks_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    /// Set the ranking method
    pub fn ranking_method(mut self, ranking_method: &str) -> Self {
        self.config.ranking_method = ranking_method.to_string();
        self
    }

    /// Set the number of top features to select
    pub fn k(mut self, k: Option<usize>) -> Self {
        self.config.k = k;
        self
    }

    /// Set the threshold for feature selection
    pub fn threshold(mut self, threshold: Option<Float>) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Set whether to use absolute values
    pub fn use_absolute(mut self, use_absolute: bool) -> Self {
        self.config.use_absolute = use_absolute;
        self
    }

    /// Set whether to normalize features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Set the regularization parameter
    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    /// Set the number of neighbors for Relief
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.config.n_neighbors = n_neighbors;
        self
    }

    /// Extract unique classes from labels
    fn unique_classes(&self, y: &Array1<i32>) -> Result<Array1<i32>> {
        let mut classes = Vec::new();
        for &label in y.iter() {
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        classes.sort_unstable();
        Ok(Array1::from_vec(classes))
    }

    /// Normalize features to zero mean and unit variance
    fn normalize_features(&self, x: &Array2<Float>) -> Array2<Float> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut normalized = x.clone();

        for j in 0..n_features {
            let column = x.column(j);
            let mean = column.sum() / n_samples as Float;
            let variance = column.mapv(|v| (v - mean).powi(2)).sum() / n_samples as Float;
            let std = (variance + self.config.reg_param).sqrt();

            for i in 0..n_samples {
                normalized[[i, j]] = (x[[i, j]] - mean) / std;
            }
        }

        normalized
    }

    /// Compute Fisher Score for each feature
    fn compute_fisher_score(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Array1<Float>> {
        let classes = self.unique_classes(y)?;
        let n_classes = classes.len();
        let n_features = x.ncols();
        let n_samples = x.nrows();

        let mut scores = Array1::zeros(n_features);

        for feature_idx in 0..n_features {
            let feature_values = x.column(feature_idx);

            // Compute overall mean
            let overall_mean = feature_values.sum() / n_samples as Float;

            // Compute class means and counts
            let mut class_means: Array1<Float> = Array1::zeros(n_classes);
            let mut class_counts: Array1<Float> = Array1::zeros(n_classes);

            for (sample_idx, &label) in y.iter().enumerate() {
                if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                    class_means[class_idx] += feature_values[sample_idx];
                    class_counts[class_idx] += 1.0;
                }
            }

            for class_idx in 0..n_classes {
                if class_counts[class_idx] > 0.0 {
                    class_means[class_idx] /= class_counts[class_idx];
                }
            }

            // Compute between-class variance
            let mut between_class_var = 0.0;
            for class_idx in 0..n_classes {
                if class_counts[class_idx] > 0.0 {
                    let diff = class_means[class_idx] - overall_mean;
                    between_class_var += class_counts[class_idx] * diff * diff;
                }
            }

            // Compute within-class variance
            let mut within_class_var = 0.0;
            for (sample_idx, &label) in y.iter().enumerate() {
                if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                    let diff = feature_values[sample_idx] - class_means[class_idx];
                    within_class_var += diff * diff;
                }
            }

            // Fisher score = between_class_var / within_class_var
            scores[feature_idx] = if within_class_var > self.config.reg_param {
                between_class_var / within_class_var
            } else {
                0.0
            };
        }

        Ok(scores)
    }

    /// Compute mutual information between features and labels
    fn compute_mutual_information(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<Array1<Float>> {
        let n_features = x.ncols();
        let n_samples = x.nrows();
        let classes = self.unique_classes(y)?;
        let n_classes = classes.len();

        let mut scores = Array1::zeros(n_features);

        for feature_idx in 0..n_features {
            let feature_values = x.column(feature_idx);

            // Discretize feature values into bins for MI calculation
            let n_bins = 10;
            let min_val = feature_values
                .iter()
                .fold(Float::INFINITY, |a, &b| a.min(b));
            let max_val = feature_values
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let bin_width = (max_val - min_val + self.config.reg_param) / n_bins as Float;

            // Create joint frequency matrix
            let mut joint_freq: Array2<Float> = Array2::zeros((n_bins, n_classes));
            let mut feature_freq: Array1<Float> = Array1::zeros(n_bins);
            let mut class_freq: Array1<Float> = Array1::zeros(n_classes);

            for (sample_idx, &label) in y.iter().enumerate() {
                if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                    let bin_idx = std::cmp::min(
                        ((feature_values[sample_idx] - min_val) / bin_width) as usize,
                        n_bins - 1,
                    );

                    joint_freq[[bin_idx, class_idx]] += 1.0;
                    feature_freq[bin_idx] += 1.0;
                    class_freq[class_idx] += 1.0;
                }
            }

            // Compute mutual information
            let mut mi: Float = 0.0;
            for bin_idx in 0..n_bins {
                for class_idx in 0..n_classes {
                    if joint_freq[[bin_idx, class_idx]] > 0.0 {
                        let p_xy = joint_freq[[bin_idx, class_idx]] / n_samples as Float;
                        let p_x = feature_freq[bin_idx] / n_samples as Float;
                        let p_y = class_freq[class_idx] / n_samples as Float;

                        if p_x > 0.0 && p_y > 0.0 {
                            let ratio: Float = p_xy / (p_x * p_y);
                            mi += p_xy * ratio.ln();
                        }
                    }
                }
            }

            scores[feature_idx] = mi;
        }

        Ok(scores)
    }

    /// Compute ANOVA F-statistic for each feature
    fn compute_anova_f(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Array1<Float>> {
        let classes = self.unique_classes(y)?;
        let n_classes = classes.len();
        let n_features = x.ncols();
        let n_samples = x.nrows();

        let mut scores = Array1::zeros(n_features);

        for feature_idx in 0..n_features {
            let feature_values = x.column(feature_idx);

            // Compute overall mean
            let overall_mean = feature_values.sum() / n_samples as Float;

            // Compute class statistics
            let mut class_means: Array1<Float> = Array1::zeros(n_classes);
            let mut class_counts: Array1<Float> = Array1::zeros(n_classes);

            for (sample_idx, &label) in y.iter().enumerate() {
                if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                    class_means[class_idx] += feature_values[sample_idx];
                    class_counts[class_idx] += 1.0;
                }
            }

            for class_idx in 0..n_classes {
                if class_counts[class_idx] > 0.0 {
                    class_means[class_idx] /= class_counts[class_idx];
                }
            }

            // Compute between-group sum of squares
            let mut ss_between = 0.0;
            for class_idx in 0..n_classes {
                if class_counts[class_idx] > 0.0 {
                    let diff = class_means[class_idx] - overall_mean;
                    ss_between += class_counts[class_idx] * diff * diff;
                }
            }

            // Compute within-group sum of squares
            let mut ss_within = 0.0;
            for (sample_idx, &label) in y.iter().enumerate() {
                if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                    let diff = feature_values[sample_idx] - class_means[class_idx];
                    ss_within += diff * diff;
                }
            }

            // Compute F-statistic
            let df_between = (n_classes - 1) as Float;
            let df_within = (n_samples - n_classes) as Float;

            let ms_between = ss_between / df_between;
            let ms_within = ss_within / df_within;

            scores[feature_idx] = if ms_within > self.config.reg_param {
                ms_between / ms_within
            } else {
                0.0
            };
        }

        Ok(scores)
    }

    /// Compute Relief algorithm scores
    fn compute_relief(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Array1<Float>> {
        let n_features = x.ncols();
        let n_samples = x.nrows();
        let classes = self.unique_classes(y)?;

        let mut scores = Array1::zeros(n_features);

        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);
            let sample_class = y[sample_idx];

            // Find nearest hit (same class) and nearest miss (different class)
            let mut nearest_hit_dist = Float::INFINITY;
            let mut nearest_miss_dist = Float::INFINITY;
            let mut nearest_hit_idx = None;
            let mut nearest_miss_idx = None;

            for other_idx in 0..n_samples {
                if other_idx == sample_idx {
                    continue;
                }

                let other_sample = x.row(other_idx);
                let other_class = y[other_idx];

                // Compute Euclidean distance
                let mut distance = 0.0;
                for feature_idx in 0..n_features {
                    let diff = sample[feature_idx] - other_sample[feature_idx];
                    distance += diff * diff;
                }
                distance = distance.sqrt();

                if other_class == sample_class {
                    // Same class (hit)
                    if distance < nearest_hit_dist {
                        nearest_hit_dist = distance;
                        nearest_hit_idx = Some(other_idx);
                    }
                } else {
                    // Different class (miss)
                    if distance < nearest_miss_dist {
                        nearest_miss_dist = distance;
                        nearest_miss_idx = Some(other_idx);
                    }
                }
            }

            // Update feature scores based on Relief formula
            if let (Some(hit_idx), Some(miss_idx)) = (nearest_hit_idx, nearest_miss_idx) {
                let hit_sample = x.row(hit_idx);
                let miss_sample = x.row(miss_idx);

                for feature_idx in 0..n_features {
                    let diff_hit = (sample[feature_idx] - hit_sample[feature_idx]).abs();
                    let diff_miss = (sample[feature_idx] - miss_sample[feature_idx]).abs();

                    // Relief score: decrease for similar instances of same class,
                    // increase for similar instances of different class
                    scores[feature_idx] += diff_miss - diff_hit;
                }
            }
        }

        // Normalize scores by number of samples
        scores /= n_samples as Float;

        Ok(scores)
    }

    /// Select features based on computed scores
    fn select_features(&self, scores: &Array1<Float>) -> Array1<usize> {
        let n_features = scores.len();

        // Create feature indices with scores
        let mut feature_scores: Vec<(usize, Float)> = (0..n_features)
            .map(|i| {
                (
                    i,
                    if self.config.use_absolute {
                        scores[i].abs()
                    } else {
                        scores[i]
                    },
                )
            })
            .collect();

        // Sort by score (descending)
        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select features based on k or threshold
        let selected_indices = if let Some(k) = self.config.k {
            feature_scores
                .into_iter()
                .take(k)
                .map(|(idx, _)| idx)
                .collect()
        } else if let Some(threshold) = self.config.threshold {
            feature_scores
                .into_iter()
                .filter(|(_, score)| *score >= threshold)
                .map(|(idx, _)| idx)
                .collect()
        } else {
            // Return all features if no selection criteria
            (0..n_features).collect()
        };

        Array1::from_vec(selected_indices)
    }

    /// Create feature ranking information
    fn create_feature_ranks(&self, scores: &Array1<Float>) -> Vec<FeatureRank> {
        let n_features = scores.len();

        // Create feature indices with scores
        let mut feature_scores: Vec<(usize, Float)> = (0..n_features)
            .map(|i| {
                (
                    i,
                    if self.config.use_absolute {
                        scores[i].abs()
                    } else {
                        scores[i]
                    },
                )
            })
            .collect();

        // Sort by score (descending)
        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create ranking information
        feature_scores
            .into_iter()
            .enumerate()
            .map(|(rank, (feature_idx, score))| {
                FeatureRank {
                    feature_idx,
                    score,
                    rank: rank + 1, // 1-based ranking
                }
            })
            .collect()
    }
}

impl Estimator for DiscriminantFeatureRanking<Untrained> {
    type Config = DiscriminantFeatureRankingConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for DiscriminantFeatureRanking<Untrained> {
    type Fitted = DiscriminantFeatureRanking<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_features = x.ncols();
        let classes = self.unique_classes(y)?;

        // Normalize features if requested
        let processed_x = if self.config.normalize {
            self.normalize_features(x)
        } else {
            x.clone()
        };

        // Compute feature scores based on selected method
        let scores = match self.config.ranking_method.as_str() {
            "fisher_score" => self.compute_fisher_score(&processed_x, y)?,
            "mutual_info" => self.compute_mutual_information(&processed_x, y)?,
            "anova_f" => self.compute_anova_f(&processed_x, y)?,
            "relief" => self.compute_relief(&processed_x, y)?,
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "ranking_method".to_string(),
                    reason: format!("Unknown ranking method: {}", self.config.ranking_method),
                })
            }
        };

        // Select features and create rankings
        let selected_features = self.select_features(&scores);
        let feature_ranks = self.create_feature_ranks(&scores);

        Ok(DiscriminantFeatureRanking {
            config: self.config,
            state: PhantomData,
            classes_: Some(classes),
            feature_scores_: Some(scores),
            feature_ranks_: Some(feature_ranks),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl DiscriminantFeatureRanking<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the feature scores
    pub fn feature_scores(&self) -> &Array1<Float> {
        self.feature_scores_.as_ref().unwrap()
    }

    /// Get the feature rankings
    pub fn feature_ranks(&self) -> &Vec<FeatureRank> {
        self.feature_ranks_.as_ref().unwrap()
    }

    /// Get the selected feature indices
    pub fn selected_features(&self) -> &Array1<usize> {
        self.selected_features_.as_ref().unwrap()
    }

    /// Get the number of original features
    pub fn n_features(&self) -> usize {
        self.n_features_.unwrap()
    }

    /// Get top k features
    pub fn get_top_k_features(&self, k: usize) -> Array1<usize> {
        let feature_ranks = self.feature_ranks();
        let top_features: Vec<usize> = feature_ranks
            .iter()
            .take(k)
            .map(|rank| rank.feature_idx)
            .collect();
        Array1::from_vec(top_features)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for DiscriminantFeatureRanking<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if self.selected_features_.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        if x.ncols() != self.n_features() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                x.ncols()
            )));
        }

        let selected_features = self.selected_features();
        let n_selected = selected_features.len();
        let n_samples = x.nrows();

        let mut transformed = Array2::zeros((n_samples, n_selected));

        for (new_idx, &original_idx) in selected_features.iter().enumerate() {
            for sample_idx in 0..n_samples {
                transformed[[sample_idx, new_idx]] = x[[sample_idx, original_idx]];
            }
        }

        Ok(transformed)
    }
}

impl Default for DiscriminantFeatureRanking<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

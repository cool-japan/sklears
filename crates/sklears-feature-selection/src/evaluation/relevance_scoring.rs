//! Relevance scoring methods for feature selection evaluation
//!
//! This module implements comprehensive relevance scoring methods to evaluate
//! how relevant selected features are to the target variable. All implementations
//! follow the SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
type Result<T> = SklResult<T>;

impl From<RelevanceError> for SklearsError {
    fn from(err: RelevanceError) -> Self {
        SklearsError::FitError(format!("Relevance analysis error: {}", err))
    }
}
use scirs2_core::random::{thread_rng, Rng};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RelevanceError {
    #[error("Feature matrix is empty")]
    EmptyFeatureMatrix,
    #[error("Target array is empty")]
    EmptyTarget,
    #[error("Feature and target lengths do not match")]
    LengthMismatch,
    #[error("Invalid feature indices")]
    InvalidFeatureIndices,
    #[error("Insufficient variance in data")]
    InsufficientVariance,
}

/// Information gain-based relevance scoring
#[derive(Debug, Clone)]
pub struct InformationGainScoring {
    n_bins: usize,
    use_equal_width: bool,
}

impl InformationGainScoring {
    /// Create a new information gain scorer
    pub fn new(n_bins: usize, use_equal_width: bool) -> Self {
        Self {
            n_bins,
            use_equal_width,
        }
    }

    /// Compute information gain for selected features
    pub fn compute(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<Vec<f64>> {
        if X.nrows() != y.len() {
            return Err(RelevanceError::LengthMismatch.into());
        }

        if X.is_empty() || y.is_empty() {
            return Err(RelevanceError::EmptyFeatureMatrix.into());
        }

        let mut scores = Vec::with_capacity(feature_indices.len());

        // Discretize target if it's continuous
        let discretized_target = self.discretize_target(y)?;

        for &feature_idx in feature_indices {
            if feature_idx >= X.ncols() {
                return Err(RelevanceError::InvalidFeatureIndices.into());
            }

            let feature_column = X.column(feature_idx);
            let discretized_feature = self.discretize_feature(feature_column)?;

            let ig = self.compute_information_gain(&discretized_feature, &discretized_target)?;
            scores.push(ig);
        }

        Ok(scores)
    }

    /// Discretize target variable
    fn discretize_target(&self, target: ArrayView1<f64>) -> Result<Array1<i32>> {
        // Check if target is already discrete (all integers)
        let is_discrete = target.iter().all(|&x| x.fract() == 0.0);

        if is_discrete {
            // Convert to integers
            return Ok(target.mapv(|x| x as i32));
        }

        // Discretize continuous target
        self.discretize_continuous(target)
    }

    /// Discretize feature variable
    fn discretize_feature(&self, feature: ArrayView1<f64>) -> Result<Array1<i32>> {
        // Check if feature is already discrete (all integers)
        let is_discrete = feature.iter().all(|&x| x.fract() == 0.0);

        if is_discrete {
            // Convert to integers, ensuring non-negative
            let min_val = feature.iter().fold(f64::INFINITY, |acc, &x| acc.min(x)) as i32;
            return Ok(feature.mapv(|x| x as i32 - min_val));
        }

        // Discretize continuous feature
        self.discretize_continuous(feature)
    }

    /// Discretize continuous variable using binning
    fn discretize_continuous(&self, values: ArrayView1<f64>) -> Result<Array1<i32>> {
        let min_val = values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        if (max_val - min_val).abs() < 1e-10 {
            // Constant feature
            return Ok(Array1::zeros(values.len()));
        }

        if self.use_equal_width {
            self.equal_width_binning(values, min_val, max_val)
        } else {
            self.equal_frequency_binning(values)
        }
    }

    /// Equal width binning
    fn equal_width_binning(
        &self,
        values: ArrayView1<f64>,
        min_val: f64,
        max_val: f64,
    ) -> Result<Array1<i32>> {
        let bin_width = (max_val - min_val) / self.n_bins as f64;
        let mut discretized = Array1::zeros(values.len());

        for (i, &value) in values.iter().enumerate() {
            let bin = ((value - min_val) / bin_width).floor() as i32;
            discretized[i] = bin.min((self.n_bins - 1) as i32).max(0);
        }

        Ok(discretized)
    }

    /// Equal frequency binning (quantile-based)
    fn equal_frequency_binning(&self, values: ArrayView1<f64>) -> Result<Array1<i32>> {
        let mut sorted_values: Vec<f64> = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = values.len();
        let bin_size = n / self.n_bins;
        let mut discretized = Array1::zeros(n);

        for (i, &value) in values.iter().enumerate() {
            // Find which quantile this value belongs to
            let rank = sorted_values.partition_point(|&x| x < value);
            let bin = (rank / bin_size.max(1)).min(self.n_bins - 1);
            discretized[i] = bin as i32;
        }

        Ok(discretized)
    }

    /// Compute information gain between feature and target
    fn compute_information_gain(&self, feature: &Array1<i32>, target: &Array1<i32>) -> Result<f64> {
        let target_entropy = self.compute_entropy(target)?;
        let conditional_entropy = self.compute_conditional_entropy(feature, target)?;

        Ok(target_entropy - conditional_entropy)
    }

    /// Compute entropy of a discrete variable
    fn compute_entropy(&self, values: &Array1<i32>) -> Result<f64> {
        let mut counts = HashMap::new();
        let total = values.len() as f64;

        for &value in values.iter() {
            *counts.entry(value).or_insert(0) += 1;
        }

        let mut entropy = 0.0;
        for count in counts.values() {
            if *count > 0 {
                let probability = *count as f64 / total;
                entropy -= probability * probability.ln();
            }
        }

        Ok(entropy)
    }

    /// Compute conditional entropy H(Y|X)
    fn compute_conditional_entropy(
        &self,
        feature: &Array1<i32>,
        target: &Array1<i32>,
    ) -> Result<f64> {
        let mut joint_counts = HashMap::new();
        let mut feature_counts = HashMap::new();
        let total = feature.len() as f64;

        // Count joint occurrences and marginal counts
        for i in 0..feature.len() {
            let x_val = feature[i];
            let y_val = target[i];

            *joint_counts.entry((x_val, y_val)).or_insert(0) += 1;
            *feature_counts.entry(x_val).or_insert(0) += 1;
        }

        let mut conditional_entropy = 0.0;

        // For each value of X
        for (&x_val, &x_count) in feature_counts.iter() {
            if x_count == 0 {
                continue;
            }

            let p_x = x_count as f64 / total;

            // Compute H(Y | X = x_val)
            let mut entropy_y_given_x = 0.0;
            for (&(joint_x, joint_y), &joint_count) in joint_counts.iter() {
                if joint_x == x_val && joint_count > 0 {
                    let p_y_given_x = joint_count as f64 / x_count as f64;
                    entropy_y_given_x -= p_y_given_x * p_y_given_x.ln();
                }
            }

            conditional_entropy += p_x * entropy_y_given_x;
        }

        Ok(conditional_entropy)
    }

    /// Compute average information gain for selected features
    pub fn average_score(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<f64> {
        let scores = self.compute(X, y, feature_indices)?;
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

/// Chi-square-based relevance scoring
#[derive(Debug, Clone)]
pub struct ChiSquareScoring {
    n_bins: usize,
}

impl ChiSquareScoring {
    /// Create a new chi-square scorer
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins }
    }

    /// Compute chi-square statistics for selected features
    pub fn compute(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<Vec<f64>> {
        if X.nrows() != y.len() {
            return Err(RelevanceError::LengthMismatch.into());
        }

        let mut scores = Vec::with_capacity(feature_indices.len());

        // Discretize target
        let discretized_target = self.discretize_variable(y)?;

        for &feature_idx in feature_indices {
            if feature_idx >= X.ncols() {
                return Err(RelevanceError::InvalidFeatureIndices.into());
            }

            let feature_column = X.column(feature_idx);
            let discretized_feature = self.discretize_variable(feature_column)?;

            let chi2 = self.compute_chi_square(&discretized_feature, &discretized_target)?;
            scores.push(chi2);
        }

        Ok(scores)
    }

    /// Discretize variable for chi-square test
    fn discretize_variable(&self, values: ArrayView1<f64>) -> Result<Array1<i32>> {
        let min_val = values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        if (max_val - min_val).abs() < 1e-10 {
            return Ok(Array1::zeros(values.len()));
        }

        let bin_width = (max_val - min_val) / self.n_bins as f64;
        let mut discretized = Array1::zeros(values.len());

        for (i, &value) in values.iter().enumerate() {
            let bin = ((value - min_val) / bin_width).floor() as i32;
            discretized[i] = bin.min((self.n_bins - 1) as i32).max(0);
        }

        Ok(discretized)
    }

    /// Compute chi-square statistic
    fn compute_chi_square(&self, feature: &Array1<i32>, target: &Array1<i32>) -> Result<f64> {
        // Create contingency table
        let mut joint_counts = HashMap::new();
        let mut feature_counts = HashMap::new();
        let mut target_counts = HashMap::new();
        let total = feature.len() as f64;

        for i in 0..feature.len() {
            let x = feature[i];
            let y = target[i];

            *joint_counts.entry((x, y)).or_insert(0) += 1;
            *feature_counts.entry(x).or_insert(0) += 1;
            *target_counts.entry(y).or_insert(0) += 1;
        }

        let mut chi_square = 0.0;

        for (&(x, y), &observed) in joint_counts.iter() {
            let x_count = *feature_counts.get(&x).unwrap_or(&0) as f64;
            let y_count = *target_counts.get(&y).unwrap_or(&0) as f64;

            let expected = (x_count * y_count) / total;

            if expected > 1e-10 {
                let diff = observed as f64 - expected;
                chi_square += (diff * diff) / expected;
            }
        }

        Ok(chi_square)
    }

    /// Compute average chi-square score
    pub fn average_score(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<f64> {
        let scores = self.compute(X, y, feature_indices)?;
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

/// F-statistic-based relevance scoring
#[derive(Debug, Clone)]
pub struct FStatisticScoring {
    classification: bool,
}

impl FStatisticScoring {
    /// Create a new F-statistic scorer
    pub fn new(classification: bool) -> Self {
        Self { classification }
    }

    /// Compute F-statistics for selected features
    pub fn compute(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<Vec<f64>> {
        if X.nrows() != y.len() {
            return Err(RelevanceError::LengthMismatch.into());
        }

        let mut scores = Vec::with_capacity(feature_indices.len());

        for &feature_idx in feature_indices {
            if feature_idx >= X.ncols() {
                return Err(RelevanceError::InvalidFeatureIndices.into());
            }

            let feature_column = X.column(feature_idx);

            let f_stat = if self.classification {
                self.compute_f_classif(feature_column, y)?
            } else {
                self.compute_f_regression(feature_column, y)?
            };

            scores.push(f_stat);
        }

        Ok(scores)
    }

    /// Compute F-statistic for classification
    fn compute_f_classif(&self, feature: ArrayView1<f64>, target: ArrayView1<f64>) -> Result<f64> {
        // Group feature values by class
        let mut class_groups: HashMap<i32, Vec<f64>> = HashMap::new();

        for i in 0..feature.len() {
            let class = target[i] as i32;
            class_groups.entry(class).or_default().push(feature[i]);
        }

        if class_groups.len() < 2 {
            return Ok(0.0); // No variation between classes
        }

        // Compute overall mean
        let overall_mean = feature.mean().unwrap_or(0.0);
        let total_n = feature.len() as f64;

        // Compute between-group sum of squares
        let mut ss_between = 0.0;
        for group in class_groups.values() {
            let group_mean = group.iter().sum::<f64>() / group.len() as f64;
            let n_group = group.len() as f64;
            ss_between += n_group * (group_mean - overall_mean).powi(2);
        }

        // Compute within-group sum of squares
        let mut ss_within = 0.0;
        for group in class_groups.values() {
            let group_mean = group.iter().sum::<f64>() / group.len() as f64;
            for &value in group {
                ss_within += (value - group_mean).powi(2);
            }
        }

        let df_between = (class_groups.len() - 1) as f64;
        let df_within = (total_n - class_groups.len() as f64).max(1.0);

        if ss_within < 1e-10 {
            return Ok(f64::INFINITY);
        }

        let ms_between = ss_between / df_between;
        let ms_within = ss_within / df_within;

        Ok(ms_between / ms_within)
    }

    /// Compute F-statistic for regression
    fn compute_f_regression(
        &self,
        feature: ArrayView1<f64>,
        target: ArrayView1<f64>,
    ) -> Result<f64> {
        let n = feature.len() as f64;
        if n < 3.0 {
            return Ok(0.0);
        }

        // Compute correlation coefficient
        let mean_x = feature.mean().unwrap_or(0.0);
        let mean_y = target.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;

        for i in 0..feature.len() {
            let dx = feature[i] - mean_x;
            let dy = target[i] - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        if sum_xx < 1e-10 || sum_yy < 1e-10 {
            return Ok(0.0);
        }

        let r = sum_xy / (sum_xx * sum_yy).sqrt();
        let r_squared = r * r;

        // F-statistic for regression: F = (r²/(1-r²)) * (n-2)
        if (1.0 - r_squared).abs() < 1e-10 {
            return Ok(f64::INFINITY);
        }

        Ok((r_squared / (1.0 - r_squared)) * (n - 2.0))
    }

    /// Compute average F-statistic score
    pub fn average_score(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<f64> {
        let scores = self.compute(X, y, feature_indices)?;
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

/// Correlation-based relevance scoring
#[derive(Debug, Clone)]
pub struct CorrelationScoring {
    use_absolute: bool,
}

impl CorrelationScoring {
    /// Create a new correlation scorer
    pub fn new(use_absolute: bool) -> Self {
        Self { use_absolute }
    }

    /// Compute correlations for selected features
    pub fn compute(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<Vec<f64>> {
        if X.nrows() != y.len() {
            return Err(RelevanceError::LengthMismatch.into());
        }

        let mut scores = Vec::with_capacity(feature_indices.len());

        for &feature_idx in feature_indices {
            if feature_idx >= X.ncols() {
                return Err(RelevanceError::InvalidFeatureIndices.into());
            }

            let feature_column = X.column(feature_idx);
            let correlation = self.compute_correlation(feature_column, y)?;

            scores.push(if self.use_absolute {
                correlation.abs()
            } else {
                correlation
            });
        }

        Ok(scores)
    }

    /// Compute Pearson correlation coefficient
    fn compute_correlation(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<f64> {
        let n = x.len() as f64;
        if n < 2.0 {
            return Ok(0.0);
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom < 1e-10 {
            return Ok(0.0);
        }

        Ok(sum_xy / denom)
    }

    /// Compute average correlation score
    pub fn average_score(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<f64> {
        let scores = self.compute(X, y, feature_indices)?;
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

/// Relief algorithm-based relevance scoring
#[derive(Debug, Clone)]
pub struct ReliefScoring {
    n_iterations: usize,
    k_neighbors: usize,
}

impl ReliefScoring {
    /// Create a new Relief scorer
    pub fn new(n_iterations: usize, k_neighbors: usize) -> Self {
        Self {
            n_iterations,
            k_neighbors,
        }
    }

    /// Compute Relief scores for selected features
    pub fn compute(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<Vec<f64>> {
        if X.nrows() != y.len() {
            return Err(RelevanceError::LengthMismatch.into());
        }

        if X.is_empty() || y.is_empty() {
            return Err(RelevanceError::EmptyFeatureMatrix.into());
        }

        let mut feature_weights = vec![0.0; feature_indices.len()];

        // Relief algorithm iterations
        for _ in 0..self.n_iterations {
            // Randomly select an instance
            let random_idx = (thread_rng().gen::<f64>() * X.nrows() as f64) as usize % X.nrows();

            let random_instance = X.row(random_idx);
            let random_class = y[random_idx];

            // Find nearest hit and miss
            let (nearest_hit_idx, nearest_miss_idx) =
                self.find_nearest_hit_miss(X, y, random_idx, random_class)?;

            if let Some(hit_idx) = nearest_hit_idx {
                let hit_instance = X.row(hit_idx);
                for (i, &feature_idx) in feature_indices.iter().enumerate() {
                    if feature_idx < X.ncols() {
                        let diff = (random_instance[feature_idx] - hit_instance[feature_idx]).abs();
                        feature_weights[i] -= diff;
                    }
                }
            }

            if let Some(miss_idx) = nearest_miss_idx {
                let miss_instance = X.row(miss_idx);
                for (i, &feature_idx) in feature_indices.iter().enumerate() {
                    if feature_idx < X.ncols() {
                        let diff =
                            (random_instance[feature_idx] - miss_instance[feature_idx]).abs();
                        feature_weights[i] += diff;
                    }
                }
            }
        }

        // Normalize weights
        let n_iterations = self.n_iterations as f64;
        for weight in &mut feature_weights {
            *weight /= n_iterations;
        }

        Ok(feature_weights)
    }

    /// Find nearest hit (same class) and miss (different class)
    fn find_nearest_hit_miss(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        instance_idx: usize,
        instance_class: f64,
    ) -> Result<(Option<usize>, Option<usize>)> {
        let mut nearest_hit_idx = None;
        let mut nearest_miss_idx = None;
        let mut min_hit_distance = f64::INFINITY;
        let mut min_miss_distance = f64::INFINITY;

        let instance = X.row(instance_idx);

        for i in 0..X.nrows() {
            if i == instance_idx {
                continue;
            }

            let other_instance = X.row(i);
            let other_class = y[i];

            // Compute Euclidean distance
            let mut distance = 0.0;
            for j in 0..X.ncols() {
                let diff = instance[j] - other_instance[j];
                distance += diff * diff;
            }
            distance = distance.sqrt();

            // Check if it's a hit or miss
            if (other_class - instance_class).abs() < 1e-10 {
                // Same class (hit)
                if distance < min_hit_distance {
                    min_hit_distance = distance;
                    nearest_hit_idx = Some(i);
                }
            } else {
                // Different class (miss)
                if distance < min_miss_distance {
                    min_miss_distance = distance;
                    nearest_miss_idx = Some(i);
                }
            }
        }

        Ok((nearest_hit_idx, nearest_miss_idx))
    }

    /// Compute average Relief score
    pub fn average_score(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<f64> {
        let scores = self.compute(X, y, feature_indices)?;
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

/// Comprehensive relevance scoring aggregator
#[derive(Debug, Clone)]
pub struct RelevanceScoring {
    classification: bool,
}

impl RelevanceScoring {
    /// Create a new relevance scorer
    pub fn new(classification: bool) -> Self {
        Self { classification }
    }

    /// Compute comprehensive relevance assessment
    pub fn compute(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> Result<RelevanceAssessment> {
        let information_gain = InformationGainScoring::new(10, true);
        let chi_square = ChiSquareScoring::new(10);
        let f_statistic = FStatisticScoring::new(self.classification);
        let correlation = CorrelationScoring::new(true);
        let relief = ReliefScoring::new(100, 5);

        let ig_scores = information_gain.compute(X, y, feature_indices)?;
        let chi2_scores = chi_square.compute(X, y, feature_indices)?;
        let f_scores = f_statistic.compute(X, y, feature_indices)?;
        let corr_scores = correlation.compute(X, y, feature_indices)?;
        let relief_scores = relief.compute(X, y, feature_indices)?;

        Ok(RelevanceAssessment {
            information_gain_scores: ig_scores,
            chi_square_scores: chi2_scores,
            f_statistic_scores: f_scores,
            correlation_scores: corr_scores,
            relief_scores,
            feature_indices: feature_indices.to_vec(),
            average_information_gain: information_gain.average_score(X, y, feature_indices)?,
            average_chi_square: chi_square.average_score(X, y, feature_indices)?,
            average_f_statistic: f_statistic.average_score(X, y, feature_indices)?,
            average_correlation: correlation.average_score(X, y, feature_indices)?,
            average_relief: relief.average_score(X, y, feature_indices)?,
        })
    }
}

/// Comprehensive relevance assessment results
#[derive(Debug, Clone)]
pub struct RelevanceAssessment {
    pub information_gain_scores: Vec<f64>,
    pub chi_square_scores: Vec<f64>,
    pub f_statistic_scores: Vec<f64>,
    pub correlation_scores: Vec<f64>,
    pub relief_scores: Vec<f64>,
    pub feature_indices: Vec<usize>,
    pub average_information_gain: f64,
    pub average_chi_square: f64,
    pub average_f_statistic: f64,
    pub average_correlation: f64,
    pub average_relief: f64,
}

impl RelevanceAssessment {
    /// Generate comprehensive relevance report
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Feature Relevance Assessment ===\n\n");

        report.push_str(&format!(
            "Number of features analyzed: {}\n\n",
            self.feature_indices.len()
        ));

        report.push_str(&format!(
            "Average Information Gain: {:.4}\n",
            self.average_information_gain
        ));
        report.push_str(&format!(
            "Average Chi-Square: {:.4}\n",
            self.average_chi_square
        ));
        report.push_str(&format!(
            "Average F-Statistic: {:.4}\n",
            self.average_f_statistic
        ));
        report.push_str(&format!(
            "Average Correlation: {:.4}\n",
            self.average_correlation
        ));
        report.push_str(&format!(
            "Average Relief Score: {:.4}\n\n",
            self.average_relief
        ));

        report.push_str("Per-Feature Relevance Scores:\n");
        report.push_str("Feature | InfoGain | Chi2     | F-Stat   | Corr     | Relief\n");
        report.push_str("--------|----------|----------|----------|----------|----------\n");

        for i in 0..self.feature_indices.len() {
            report.push_str(&format!(
                "{:7} | {:8.4} | {:8.4} | {:8.4} | {:8.4} | {:8.4}\n",
                self.feature_indices[i],
                self.information_gain_scores[i],
                self.chi_square_scores[i],
                self.f_statistic_scores[i],
                self.correlation_scores[i],
                self.relief_scores[i]
            ));
        }

        report.push_str(&format!(
            "\nOverall Relevance Assessment: {}\n",
            self.overall_assessment()
        ));

        report
    }

    fn overall_assessment(&self) -> &'static str {
        let scores = [
            self.average_information_gain,
            self.average_chi_square / 10.0,  // Normalize chi2
            self.average_f_statistic / 10.0, // Normalize F-stat
            self.average_correlation,
            self.average_relief,
        ];

        let average = scores.iter().sum::<f64>() / scores.len() as f64;

        match average {
            x if x >= 0.8 => "EXCELLENT: Features show very high relevance to target",
            x if x >= 0.6 => "GOOD: Features show good relevance to target",
            x if x >= 0.4 => "MODERATE: Features show moderate relevance to target",
            x if x >= 0.2 => "POOR: Features show low relevance to target",
            _ => {
                "CRITICAL: Features show very low relevance to target - consider different features"
            }
        }
    }

    /// Get top N features by relevance score (using average of normalized scores)
    pub fn get_top_features(&self, n: usize) -> Vec<(usize, f64)> {
        let mut feature_scores: Vec<(usize, f64)> = Vec::new();

        for i in 0..self.feature_indices.len() {
            let normalized_scores = [
                self.information_gain_scores[i],
                self.chi_square_scores[i] / 10.0,
                self.f_statistic_scores[i] / 10.0,
                self.correlation_scores[i],
                self.relief_scores[i],
            ];

            let average_score =
                normalized_scores.iter().sum::<f64>() / normalized_scores.len() as f64;
            feature_scores.push((self.feature_indices[i], average_score));
        }

        // Sort by score (descending)
        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        feature_scores.into_iter().take(n).collect()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_information_gain_scoring() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let ig_scorer = InformationGainScoring::new(3, true);
        let scores = ig_scorer.compute(X.view(), y.view(), &[0, 1]).unwrap();

        assert_eq!(scores.len(), 2);
        for score in &scores {
            assert!(score >= &0.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_chi_square_scoring() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let chi2_scorer = ChiSquareScoring::new(3);
        let scores = chi2_scorer.compute(X.view(), y.view(), &[0, 1]).unwrap();

        assert_eq!(scores.len(), 2);
        for score in &scores {
            assert!(score >= &0.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_f_statistic_scoring() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let f_scorer = FStatisticScoring::new(true);
        let scores = f_scorer.compute(X.view(), y.view(), &[0, 1]).unwrap();

        assert_eq!(scores.len(), 2);
        for score in &scores {
            assert!(score >= &0.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_correlation_scoring() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let corr_scorer = CorrelationScoring::new(true);
        let scores = corr_scorer.compute(X.view(), y.view(), &[0, 1]).unwrap();

        assert_eq!(scores.len(), 2);
        for score in &scores {
            assert!(score >= &0.0 && score <= &1.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_relief_scoring() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 1.0],];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let relief_scorer = ReliefScoring::new(10, 1);
        let scores = relief_scorer.compute(X.view(), y.view(), &[0, 1]).unwrap();

        assert_eq!(scores.len(), 2);
        // Relief scores can be negative
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_relevance_assessment() {
        let X = array![
            [1.0, 2.0, 10.0],
            [2.0, 3.0, 20.0],
            [3.0, 4.0, 30.0],
            [4.0, 5.0, 40.0],
            [5.0, 6.0, 50.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 1.0];

        let relevance_scorer = RelevanceScoring::new(true);
        let assessment = relevance_scorer
            .compute(X.view(), y.view(), &[0, 1, 2])
            .unwrap();

        assert_eq!(assessment.information_gain_scores.len(), 3);
        assert_eq!(assessment.chi_square_scores.len(), 3);
        assert_eq!(assessment.f_statistic_scores.len(), 3);
        assert_eq!(assessment.correlation_scores.len(), 3);
        assert_eq!(assessment.relief_scores.len(), 3);

        let report = assessment.report();
        assert!(report.contains("Relevance Assessment"));

        let top_features = assessment.get_top_features(2);
        assert_eq!(top_features.len(), 2);
    }
}

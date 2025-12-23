//! Redundancy measures for feature set quality assessment
//!
//! This module implements comprehensive redundancy assessment methods to evaluate
//! the degree of redundancy in selected feature sets. All implementations follow
//! the SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
type Result<T> = SklResult<T>;

impl From<RedundancyError> for SklearsError {
    fn from(err: RedundancyError) -> Self {
        SklearsError::FitError(format!("Redundancy analysis error: {}", err))
    }
}
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RedundancyError {
    #[error("Feature matrix is empty")]
    EmptyFeatureMatrix,
    #[error("Feature indices must be valid")]
    InvalidFeatureIndices,
    #[error("Insufficient variance for redundancy computation")]
    InsufficientVariance,
    #[error("Correlation matrix computation failed")]
    CorrelationComputationFailed,
}

/// Correlation-based redundancy measures
#[derive(Debug, Clone)]
pub struct CorrelationRedundancy {
    correlation_threshold: f64,
    absolute_correlation: bool,
}

impl CorrelationRedundancy {
    /// Create a new correlation redundancy analyzer
    pub fn new(correlation_threshold: f64, absolute_correlation: bool) -> Self {
        Self {
            correlation_threshold,
            absolute_correlation,
        }
    }

    /// Compute pairwise correlation redundancy for selected features
    pub fn compute(&self, X: ArrayView2<f64>, feature_indices: &[usize]) -> Result<f64> {
        if feature_indices.len() < 2 {
            return Ok(0.0); // No redundancy with less than 2 features
        }

        if X.is_empty() {
            return Err(RedundancyError::EmptyFeatureMatrix.into());
        }

        // Extract selected features
        let selected_features = self.extract_features(X, feature_indices)?;

        // Compute correlation matrix
        let correlation_matrix = self.compute_correlation_matrix(&selected_features)?;

        // Calculate redundancy score
        self.calculate_redundancy_score(&correlation_matrix)
    }

    /// Extract selected feature columns
    fn extract_features(
        &self,
        X: ArrayView2<f64>,
        feature_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let mut selected = Array2::zeros((n_samples, feature_indices.len()));

        for (col_idx, &feature_idx) in feature_indices.iter().enumerate() {
            if feature_idx >= X.ncols() {
                return Err(RedundancyError::InvalidFeatureIndices.into());
            }
            selected.column_mut(col_idx).assign(&X.column(feature_idx));
        }

        Ok(selected)
    }

    /// Compute correlation matrix for features
    fn compute_correlation_matrix(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let n_features = features.ncols();
        let mut correlation_matrix = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    correlation_matrix[[i, j]] = 1.0;
                } else {
                    let feature1 = features.column(i);
                    let feature2 = features.column(j);
                    let correlation = self.compute_correlation(feature1, feature2)?;
                    correlation_matrix[[i, j]] = if self.absolute_correlation {
                        correlation.abs()
                    } else {
                        correlation
                    };
                }
            }
        }

        Ok(correlation_matrix)
    }

    /// Compute Pearson correlation between two features
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

    /// Calculate redundancy score from correlation matrix
    fn calculate_redundancy_score(&self, correlation_matrix: &Array2<f64>) -> Result<f64> {
        let n_features = correlation_matrix.nrows();
        if n_features < 2 {
            return Ok(0.0);
        }

        let mut total_redundancy = 0.0;
        let mut _pair_count = 0;

        // Sum upper triangular correlations (excluding diagonal)
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let correlation = correlation_matrix[[i, j]];
                if correlation.abs() >= self.correlation_threshold {
                    total_redundancy += correlation.abs();
                    _pair_count += 1;
                }
            }
        }

        let total_pairs = (n_features * (n_features - 1)) / 2;
        Ok(total_redundancy / total_pairs as f64)
    }

    /// Identify highly correlated feature pairs
    pub fn identify_redundant_pairs(
        &self,
        X: ArrayView2<f64>,
        feature_indices: &[usize],
    ) -> Result<Vec<(usize, usize, f64)>> {
        let selected_features = self.extract_features(X, feature_indices)?;
        let correlation_matrix = self.compute_correlation_matrix(&selected_features)?;

        let mut redundant_pairs = Vec::new();

        for i in 0..feature_indices.len() {
            for j in (i + 1)..feature_indices.len() {
                let correlation = correlation_matrix[[i, j]];
                if correlation.abs() >= self.correlation_threshold {
                    redundant_pairs.push((feature_indices[i], feature_indices[j], correlation));
                }
            }
        }

        // Sort by correlation strength
        redundant_pairs.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

        Ok(redundant_pairs)
    }
}

/// Mutual information-based redundancy assessment
#[derive(Debug, Clone)]
pub struct MutualInformationRedundancy {
    mi_threshold: f64,
    n_bins: usize,
}

impl MutualInformationRedundancy {
    /// Create a new mutual information redundancy analyzer
    pub fn new(mi_threshold: f64, n_bins: usize) -> Self {
        Self {
            mi_threshold,
            n_bins,
        }
    }

    /// Compute mutual information-based redundancy
    pub fn compute(&self, X: ArrayView2<f64>, feature_indices: &[usize]) -> Result<f64> {
        if feature_indices.len() < 2 {
            return Ok(0.0);
        }

        let selected_features = self.extract_features(X, feature_indices)?;
        let mi_matrix = self.compute_mi_matrix(&selected_features)?;

        self.calculate_mi_redundancy_score(&mi_matrix)
    }

    fn extract_features(
        &self,
        X: ArrayView2<f64>,
        feature_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let mut selected = Array2::zeros((n_samples, feature_indices.len()));

        for (col_idx, &feature_idx) in feature_indices.iter().enumerate() {
            if feature_idx >= X.ncols() {
                return Err(RedundancyError::InvalidFeatureIndices.into());
            }
            selected.column_mut(col_idx).assign(&X.column(feature_idx));
        }

        Ok(selected)
    }

    /// Compute mutual information matrix
    fn compute_mi_matrix(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let n_features = features.ncols();
        let mut mi_matrix = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    mi_matrix[[i, j]] = self.compute_entropy(features.column(i))?;
                } else {
                    let mi =
                        self.compute_mutual_information(features.column(i), features.column(j))?;
                    mi_matrix[[i, j]] = mi;
                }
            }
        }

        Ok(mi_matrix)
    }

    /// Compute entropy of a feature
    fn compute_entropy(&self, feature: ArrayView1<f64>) -> Result<f64> {
        let histogram = self.create_histogram(feature);
        let total_count = feature.len() as f64;

        let mut entropy = 0.0;
        for &count in histogram.values() {
            if count > 0 {
                let probability = count as f64 / total_count;
                entropy -= probability * probability.ln();
            }
        }

        Ok(entropy)
    }

    /// Compute mutual information between two features
    fn compute_mutual_information(
        &self,
        feature1: ArrayView1<f64>,
        feature2: ArrayView1<f64>,
    ) -> Result<f64> {
        let hist1 = self.create_histogram(feature1);
        let hist2 = self.create_histogram(feature2);
        let joint_hist = self.create_joint_histogram(feature1, feature2);

        let n = feature1.len() as f64;
        let mut mi = 0.0;

        for (&(val1, val2), &joint_count) in joint_hist.iter() {
            if joint_count > 0 {
                let p_xy = joint_count as f64 / n;
                let p_x = *hist1.get(&val1).unwrap_or(&0) as f64 / n;
                let p_y = *hist2.get(&val2).unwrap_or(&0) as f64 / n;

                if p_x > 0.0 && p_y > 0.0 {
                    mi += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }

        Ok(mi)
    }

    /// Create histogram for discretization
    fn create_histogram(&self, feature: ArrayView1<f64>) -> HashMap<i32, usize> {
        let min_val = feature.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = feature.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let bin_width = (max_val - min_val) / self.n_bins as f64;

        let mut histogram = HashMap::new();

        for &value in feature.iter() {
            let bin = if bin_width > 0.0 {
                ((value - min_val) / bin_width).floor() as i32
            } else {
                0
            };
            let bin = bin.min((self.n_bins - 1) as i32).max(0);
            *histogram.entry(bin).or_insert(0) += 1;
        }

        histogram
    }

    /// Create joint histogram for two features
    fn create_joint_histogram(
        &self,
        feature1: ArrayView1<f64>,
        feature2: ArrayView1<f64>,
    ) -> HashMap<(i32, i32), usize> {
        let min1 = feature1.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max1 = feature1
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let min2 = feature2.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max2 = feature2
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        let bin_width1 = (max1 - min1) / self.n_bins as f64;
        let bin_width2 = (max2 - min2) / self.n_bins as f64;

        let mut joint_histogram = HashMap::new();

        for i in 0..feature1.len() {
            let bin1 = if bin_width1 > 0.0 {
                ((feature1[i] - min1) / bin_width1).floor() as i32
            } else {
                0
            };
            let bin2 = if bin_width2 > 0.0 {
                ((feature2[i] - min2) / bin_width2).floor() as i32
            } else {
                0
            };

            let bin1 = bin1.min((self.n_bins - 1) as i32).max(0);
            let bin2 = bin2.min((self.n_bins - 1) as i32).max(0);

            *joint_histogram.entry((bin1, bin2)).or_insert(0) += 1;
        }

        joint_histogram
    }

    fn calculate_mi_redundancy_score(&self, mi_matrix: &Array2<f64>) -> Result<f64> {
        let n_features = mi_matrix.nrows();
        if n_features < 2 {
            return Ok(0.0);
        }

        let mut total_redundancy = 0.0;
        let mut _pair_count = 0;

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let mi = mi_matrix[[i, j]];
                if mi >= self.mi_threshold {
                    total_redundancy += mi;
                    _pair_count += 1;
                }
            }
        }

        let total_pairs = (n_features * (n_features - 1)) / 2;
        Ok(total_redundancy / total_pairs as f64)
    }
}

/// Variance Inflation Factor for multicollinearity assessment
#[derive(Debug, Clone)]
pub struct VarianceInflationFactor {
    vif_threshold: f64,
}

impl VarianceInflationFactor {
    /// Create a new VIF analyzer
    pub fn new(vif_threshold: f64) -> Self {
        Self { vif_threshold }
    }

    /// Compute VIF for all features
    pub fn compute_all(&self, X: ArrayView2<f64>, feature_indices: &[usize]) -> Result<Vec<f64>> {
        if feature_indices.len() < 2 {
            return Ok(vec![1.0; feature_indices.len()]);
        }

        let selected_features = self.extract_features(X, feature_indices)?;
        let mut vif_scores = Vec::with_capacity(feature_indices.len());

        for i in 0..feature_indices.len() {
            let vif = self.compute_single_vif(&selected_features, i)?;
            vif_scores.push(vif);
        }

        Ok(vif_scores)
    }

    fn extract_features(
        &self,
        X: ArrayView2<f64>,
        feature_indices: &[usize],
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let mut selected = Array2::zeros((n_samples, feature_indices.len()));

        for (col_idx, &feature_idx) in feature_indices.iter().enumerate() {
            if feature_idx >= X.ncols() {
                return Err(RedundancyError::InvalidFeatureIndices.into());
            }
            selected.column_mut(col_idx).assign(&X.column(feature_idx));
        }

        Ok(selected)
    }

    /// Compute VIF for a single feature
    fn compute_single_vif(&self, features: &Array2<f64>, target_feature_idx: usize) -> Result<f64> {
        if target_feature_idx >= features.ncols() {
            return Err(RedundancyError::InvalidFeatureIndices.into());
        }

        let n_features = features.ncols();
        if n_features < 2 {
            return Ok(1.0);
        }

        // Extract target feature
        let target_feature = features.column(target_feature_idx);

        // Extract other features
        let mut other_features = Array2::zeros((features.nrows(), n_features - 1));
        let mut col_idx = 0;

        for i in 0..n_features {
            if i != target_feature_idx {
                other_features
                    .column_mut(col_idx)
                    .assign(&features.column(i));
                col_idx += 1;
            }
        }

        // Compute R-squared using simple linear regression approximation
        let r_squared = self.compute_r_squared(target_feature, other_features.view())?;

        // VIF = 1 / (1 - R²)
        if (1.0 - r_squared).abs() < 1e-10 {
            return Ok(f64::INFINITY);
        }

        Ok(1.0 / (1.0 - r_squared))
    }

    /// Compute R-squared approximation using correlation
    fn compute_r_squared(&self, target: ArrayView1<f64>, features: ArrayView2<f64>) -> Result<f64> {
        if features.ncols() == 0 {
            return Ok(0.0);
        }

        // Simple approximation: use maximum squared correlation
        let mut max_r_squared: f64 = 0.0;

        for i in 0..features.ncols() {
            let feature = features.column(i);
            let correlation = self.compute_correlation(target, feature)?;
            max_r_squared = max_r_squared.max(correlation * correlation);
        }

        Ok(max_r_squared)
    }

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

    /// Identify features with high VIF
    pub fn identify_high_vif_features(
        &self,
        X: ArrayView2<f64>,
        feature_indices: &[usize],
    ) -> Result<Vec<(usize, f64)>> {
        let vif_scores = self.compute_all(X, feature_indices)?;
        let mut high_vif_features = Vec::new();

        for (i, &vif) in vif_scores.iter().enumerate() {
            if vif >= self.vif_threshold {
                high_vif_features.push((feature_indices[i], vif));
            }
        }

        // Sort by VIF score (descending)
        high_vif_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(high_vif_features)
    }
}

/// Comprehensive redundancy matrix computation
#[derive(Debug, Clone)]
pub struct RedundancyMatrix {
    correlation_threshold: f64,
    mi_threshold: f64,
    vif_threshold: f64,
    n_bins: usize,
}

impl RedundancyMatrix {
    /// Create a new redundancy matrix analyzer
    pub fn new(
        correlation_threshold: f64,
        mi_threshold: f64,
        vif_threshold: f64,
        n_bins: usize,
    ) -> Self {
        Self {
            correlation_threshold,
            mi_threshold,
            vif_threshold,
            n_bins,
        }
    }

    /// Compute comprehensive redundancy assessment
    pub fn compute(
        &self,
        X: ArrayView2<f64>,
        feature_indices: &[usize],
    ) -> Result<RedundancyAssessment> {
        let correlation_redundancy = CorrelationRedundancy::new(self.correlation_threshold, true);
        let mi_redundancy = MutualInformationRedundancy::new(self.mi_threshold, self.n_bins);
        let vif_analyzer = VarianceInflationFactor::new(self.vif_threshold);

        let correlation_score = correlation_redundancy.compute(X, feature_indices)?;
        let redundant_pairs =
            correlation_redundancy.identify_redundant_pairs(X, feature_indices)?;

        let mi_score = mi_redundancy.compute(X, feature_indices)?;

        let vif_scores = vif_analyzer.compute_all(X, feature_indices)?;
        let high_vif_features = vif_analyzer.identify_high_vif_features(X, feature_indices)?;

        Ok(RedundancyAssessment {
            correlation_redundancy_score: correlation_score,
            mutual_information_redundancy_score: mi_score,
            average_vif: vif_scores.iter().sum::<f64>() / vif_scores.len() as f64,
            max_vif: vif_scores.iter().fold(0.0, |acc, &x| acc.max(x)),
            redundant_correlation_pairs: redundant_pairs,
            high_vif_features,
            vif_scores,
            n_features: feature_indices.len(),
        })
    }
}

/// Comprehensive redundancy assessment results
#[derive(Debug, Clone)]
pub struct RedundancyAssessment {
    pub correlation_redundancy_score: f64,
    pub mutual_information_redundancy_score: f64,
    pub average_vif: f64,
    pub max_vif: f64,
    pub redundant_correlation_pairs: Vec<(usize, usize, f64)>,
    pub high_vif_features: Vec<(usize, f64)>,
    pub vif_scores: Vec<f64>,
    pub n_features: usize,
}

impl RedundancyAssessment {
    /// Generate comprehensive redundancy report
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Feature Set Redundancy Assessment ===\n\n");

        report.push_str(&format!(
            "Number of features analyzed: {}\n\n",
            self.n_features
        ));

        report.push_str(&format!(
            "Correlation Redundancy Score: {:.4}\n",
            self.correlation_redundancy_score
        ));
        report.push_str(&format!(
            "  Interpretation: {}\n",
            self.interpret_correlation_redundancy()
        ));

        report.push_str(&format!(
            "\nMutual Information Redundancy Score: {:.4}\n",
            self.mutual_information_redundancy_score
        ));
        report.push_str(&format!(
            "  Interpretation: {}\n",
            self.interpret_mi_redundancy()
        ));

        report.push_str("\nVariance Inflation Factor Analysis:\n");
        report.push_str(&format!("  Average VIF: {:.4}\n", self.average_vif));
        report.push_str(&format!("  Maximum VIF: {:.4}\n", self.max_vif));
        report.push_str(&format!("  Interpretation: {}\n", self.interpret_vif()));

        if !self.redundant_correlation_pairs.is_empty() {
            report.push_str(&format!(
                "\nHighly Correlated Feature Pairs ({}):\n",
                self.redundant_correlation_pairs.len()
            ));
            for (i, &(feat1, feat2, corr)) in
                self.redundant_correlation_pairs.iter().take(10).enumerate()
            {
                report.push_str(&format!(
                    "  {}. Features {} and {}: correlation = {:.4}\n",
                    i + 1,
                    feat1,
                    feat2,
                    corr
                ));
            }
            if self.redundant_correlation_pairs.len() > 10 {
                report.push_str(&format!(
                    "  ... and {} more pairs\n",
                    self.redundant_correlation_pairs.len() - 10
                ));
            }
        }

        if !self.high_vif_features.is_empty() {
            report.push_str(&format!(
                "\nHigh VIF Features ({}):\n",
                self.high_vif_features.len()
            ));
            for (i, &(feat, vif)) in self.high_vif_features.iter().take(10).enumerate() {
                report.push_str(&format!(
                    "  {}. Feature {}: VIF = {:.4}\n",
                    i + 1,
                    feat,
                    vif
                ));
            }
            if self.high_vif_features.len() > 10 {
                report.push_str(&format!(
                    "  ... and {} more features\n",
                    self.high_vif_features.len() - 10
                ));
            }
        }

        report.push_str(&format!(
            "\nOverall Redundancy Assessment: {}\n",
            self.overall_assessment()
        ));

        report
    }

    fn interpret_correlation_redundancy(&self) -> &'static str {
        match self.correlation_redundancy_score {
            x if x >= 0.8 => "Very high correlation redundancy - many highly correlated features",
            x if x >= 0.6 => "High correlation redundancy - significant feature overlap",
            x if x >= 0.4 => "Moderate correlation redundancy - some correlated features",
            x if x >= 0.2 => "Low correlation redundancy - minimal feature overlap",
            _ => "Very low correlation redundancy - features are largely independent",
        }
    }

    fn interpret_mi_redundancy(&self) -> &'static str {
        match self.mutual_information_redundancy_score {
            x if x >= 0.8 => "Very high information redundancy - features share much information",
            x if x >= 0.6 => "High information redundancy - significant information overlap",
            x if x >= 0.4 => "Moderate information redundancy - some shared information",
            x if x >= 0.2 => "Low information redundancy - minimal information overlap",
            _ => "Very low information redundancy - features provide unique information",
        }
    }

    fn interpret_vif(&self) -> &'static str {
        match self.max_vif {
            x if x >= 10.0 => "Severe multicollinearity - high VIF values detected",
            x if x >= 5.0 => "Moderate multicollinearity - concerning VIF values",
            x if x >= 2.5 => "Mild multicollinearity - some elevated VIF values",
            _ => "No multicollinearity concerns - acceptable VIF values",
        }
    }

    fn overall_assessment(&self) -> &'static str {
        let redundancy_indicators = [
            self.correlation_redundancy_score >= 0.6,
            self.mutual_information_redundancy_score >= 0.6,
            self.max_vif >= 5.0,
            self.redundant_correlation_pairs.len() > self.n_features / 2,
        ];

        let high_redundancy_count = redundancy_indicators.iter().filter(|&&x| x).count();

        match high_redundancy_count {
            4 => "CRITICAL: Very high redundancy detected across all measures - major feature set cleanup needed",
            3 => "HIGH: High redundancy detected - consider feature reduction strategies",
            2 => "MODERATE: Some redundancy detected - review feature selection carefully",
            1 => "LOW: Minimal redundancy - feature set appears reasonable",
            _ => "EXCELLENT: Low redundancy across all measures - well-diversified feature set",
        }
    }
}

/// Comprehensive redundancy measures aggregator
#[derive(Debug, Clone)]
pub struct RedundancyMeasures;

impl RedundancyMeasures {
    /// Compute comprehensive redundancy assessment using default parameters
    pub fn compute(X: ArrayView2<f64>, feature_indices: &[usize]) -> Result<RedundancyAssessment> {
        let redundancy_matrix = RedundancyMatrix::new(0.7, 0.1, 5.0, 10);
        redundancy_matrix.compute(X, feature_indices)
    }

    /// Compute comprehensive redundancy assessment with custom parameters
    pub fn compute_with_params(
        X: ArrayView2<f64>,
        feature_indices: &[usize],
        correlation_threshold: f64,
        mi_threshold: f64,
        vif_threshold: f64,
        n_bins: usize,
    ) -> Result<RedundancyAssessment> {
        let redundancy_matrix =
            RedundancyMatrix::new(correlation_threshold, mi_threshold, vif_threshold, n_bins);
        redundancy_matrix.compute(X, feature_indices)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_correlation_redundancy() {
        let X = array![
            [1.0, 2.0, 1.1, 5.0],
            [2.0, 4.0, 2.1, 6.0],
            [3.0, 6.0, 3.1, 7.0],
            [4.0, 8.0, 4.1, 8.0],
        ];

        let feature_indices = vec![0, 1, 2]; // Features 0 and 1 are highly correlated, 2 is similar
        let redundancy = CorrelationRedundancy::new(0.5, true);
        let score = redundancy.compute(X.view(), &feature_indices).unwrap();

        assert!(score > 0.0);

        let pairs = redundancy
            .identify_redundant_pairs(X.view(), &feature_indices)
            .unwrap();
        assert!(!pairs.is_empty());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mi_redundancy() {
        let X = array![
            [1.0, 1.0, 5.0],
            [2.0, 2.0, 6.0],
            [3.0, 3.0, 7.0],
            [4.0, 4.0, 8.0],
        ];

        let feature_indices = vec![0, 1, 2];
        let redundancy = MutualInformationRedundancy::new(0.1, 3);
        let score = redundancy.compute(X.view(), &feature_indices).unwrap();

        assert!(score >= 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_vif() {
        let X = array![
            [1.0, 2.0, 1.5],
            [2.0, 4.0, 3.0],
            [3.0, 6.0, 4.5],
            [4.0, 8.0, 6.0],
        ];

        let feature_indices = vec![0, 1, 2];
        let vif_analyzer = VarianceInflationFactor::new(5.0);
        let vif_scores = vif_analyzer
            .compute_all(X.view(), &feature_indices)
            .unwrap();

        assert_eq!(vif_scores.len(), 3);
        for score in &vif_scores {
            assert!(score >= &1.0);
        }

        let high_vif = vif_analyzer
            .identify_high_vif_features(X.view(), &feature_indices)
            .unwrap();
        // Might be empty if VIF scores are below threshold
        assert!(high_vif.len() <= feature_indices.len());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_redundancy_measures() {
        let X = array![
            [1.0, 2.0, 1.1, 5.0, 0.5],
            [2.0, 4.0, 2.1, 6.0, 1.0],
            [3.0, 6.0, 3.1, 7.0, 1.5],
            [4.0, 8.0, 4.1, 8.0, 2.0],
            [5.0, 10.0, 5.1, 9.0, 2.5],
        ];

        let feature_indices = vec![0, 1, 2, 3, 4];
        let assessment = RedundancyMeasures::compute(X.view(), &feature_indices).unwrap();

        assert!(assessment.correlation_redundancy_score >= 0.0);
        assert!(assessment.mutual_information_redundancy_score >= 0.0);
        assert!(assessment.average_vif >= 1.0);
        assert!(assessment.max_vif >= 1.0);
        assert_eq!(assessment.vif_scores.len(), 5);

        let report = assessment.report();
        assert!(report.contains("Redundancy Assessment"));
        assert!(report.contains("Overall"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empty_feature_set() {
        let X = array![[1.0], [2.0], [3.0]];
        let feature_indices = vec![0];

        let redundancy = CorrelationRedundancy::new(0.5, true);
        let score = redundancy.compute(X.view(), &feature_indices).unwrap();
        assert_eq!(score, 0.0); // No redundancy with single feature
    }
}

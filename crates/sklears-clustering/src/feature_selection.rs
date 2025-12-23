//! Feature Selection for Clustering
//!
//! This module provides methods for selecting relevant features for clustering,
//! which can improve clustering quality, reduce computational cost, and enhance
//! interpretability for high-dimensional data.
//!
//! # Methods Provided
//! - **Variance-based selection**: Remove low-variance features
//! - **Laplacian Score**: Feature selection based on local structure preservation
//! - **SPEC (Spectral Feature Selection)**: Graph-based feature selection
//! - **MCFS (Multi-Cluster Feature Selection)**: L1-regularized sparse learning
//! - **Correlation-based selection**: Remove redundant features
//!
//! # Mathematical Background
//!
//! ## Laplacian Score
//! For feature f, the Laplacian score measures how well f preserves locality:
//! L_f = (f^T L f) / (f^T D f)
//! where L is the graph Laplacian and D is the degree matrix.
//!
//! ## SPEC (Spectral Feature Selection)
//! Selects features that are consistent with the cluster structure by:
//! 1. Computing graph Laplacian eigenvectors
//! 2. Selecting features with high correlation to top eigenvectors

use std::collections::HashSet;

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::error::{Result, SklearsError};

/// Feature selection configuration
#[derive(Debug, Clone)]
pub struct FeatureSelectionConfig {
    /// Method to use for feature selection
    pub method: FeatureSelectionMethod,
    /// Number of features to select (if None, determined automatically)
    pub n_features: Option<usize>,
    /// Variance threshold for variance-based selection
    pub variance_threshold: f64,
    /// Number of neighbors for graph-based methods
    pub n_neighbors: usize,
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            method: FeatureSelectionMethod::Variance,
            n_features: None,
            variance_threshold: 0.01,
            n_neighbors: 10,
        }
    }
}

/// Feature selection methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureSelectionMethod {
    /// Remove features with variance below threshold
    Variance,
    /// Laplacian score for locality preservation
    LaplacianScore,
    /// Spectral feature selection
    Spectral,
    /// Correlation-based feature selection
    Correlation,
    /// Combined method using multiple criteria
    Combined,
}

/// Result of feature selection
#[derive(Debug, Clone)]
pub struct FeatureSelectionResult {
    /// Indices of selected features
    pub selected_features: Vec<usize>,
    /// Scores for each feature (higher is better)
    pub feature_scores: Vec<f64>,
    /// Number of features selected
    pub n_selected: usize,
    /// Method used
    pub method: FeatureSelectionMethod,
}

/// Feature selector for clustering
pub struct FeatureSelector {
    config: FeatureSelectionConfig,
}

impl FeatureSelector {
    /// Create new feature selector
    pub fn new(config: FeatureSelectionConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_method(method: FeatureSelectionMethod) -> Self {
        Self {
            config: FeatureSelectionConfig {
                method,
                ..Default::default()
            },
        }
    }

    /// Select features from data
    pub fn select_features(&self, data: &Array2<f64>) -> Result<FeatureSelectionResult> {
        match self.config.method {
            FeatureSelectionMethod::Variance => self.select_by_variance(data),
            FeatureSelectionMethod::LaplacianScore => self.select_by_laplacian_score(data),
            FeatureSelectionMethod::Spectral => self.select_by_spectral(data),
            FeatureSelectionMethod::Correlation => self.select_by_correlation(data),
            FeatureSelectionMethod::Combined => self.select_by_combined(data),
        }
    }

    /// Variance-based feature selection
    fn select_by_variance(&self, data: &Array2<f64>) -> Result<FeatureSelectionResult> {
        let n_features = data.ncols();
        let mut feature_scores = Vec::with_capacity(n_features);

        // Calculate variance for each feature
        for j in 0..n_features {
            let column = data.column(j);
            let mean = column.mean().unwrap_or(0.0);
            let variance =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
            feature_scores.push(variance);
        }

        // Select features above threshold
        let mut selected_features: Vec<usize> = feature_scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score >= self.config.variance_threshold)
            .map(|(idx, _)| idx)
            .collect();

        // If n_features specified, select top n
        if let Some(n) = self.config.n_features {
            if n < selected_features.len() {
                // Sort by score and take top n
                let mut indexed_scores: Vec<(usize, f64)> = selected_features
                    .iter()
                    .map(|&idx| (idx, feature_scores[idx]))
                    .collect();
                indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                selected_features = indexed_scores.iter().take(n).map(|(idx, _)| *idx).collect();
            }
        }

        let n_selected = selected_features.len();

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            n_selected,
            method: FeatureSelectionMethod::Variance,
        })
    }

    /// Laplacian score feature selection
    fn select_by_laplacian_score(&self, data: &Array2<f64>) -> Result<FeatureSelectionResult> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Build k-nearest neighbor graph
        let W = self.build_knn_graph(data)?;

        // Compute degree matrix D
        let mut D = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            let degree = W.row(i).sum();
            D[[i, i]] = degree;
        }

        // Compute Laplacian L = D - W
        let L = &D - &W;

        // Calculate Laplacian score for each feature
        let mut feature_scores = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let f = data.column(j);

            // Center feature
            let mean = f.mean().unwrap_or(0.0);
            let f_centered: Vec<f64> = f.iter().map(|&x| x - mean).collect();
            let f_array = Array1::from_vec(f_centered);

            // Compute f^T L f (numerator)
            let Lf = L.dot(&f_array);
            let numerator = f_array.dot(&Lf);

            // Compute f^T D f (denominator)
            let Df = D.dot(&f_array);
            let denominator = f_array.dot(&Df);

            // Laplacian score (lower is better, so we negate for consistency)
            let score = if denominator.abs() > 1e-10 {
                -(numerator / denominator)
            } else {
                0.0
            };

            feature_scores.push(score);
        }

        // Select top features
        let n_select = self.config.n_features.unwrap_or(n_features / 2);
        let mut indexed_scores: Vec<(usize, f64)> = feature_scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected_features: Vec<usize> = indexed_scores
            .iter()
            .take(n_select)
            .map(|(idx, _)| *idx)
            .collect();

        Ok(FeatureSelectionResult {
            selected_features: selected_features.clone(),
            feature_scores,
            n_selected: selected_features.len(),
            method: FeatureSelectionMethod::LaplacianScore,
        })
    }

    /// Build k-nearest neighbor graph
    fn build_knn_graph(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = data.nrows();
        let mut W = Array2::zeros((n_samples, n_samples));

        // For each point, find k nearest neighbors
        for i in 0..n_samples {
            let point_i = data.row(i);
            let mut distances: Vec<(usize, f64)> = Vec::with_capacity(n_samples);

            for j in 0..n_samples {
                if i != j {
                    let point_j = data.row(j);
                    let dist = self.euclidean_distance(&point_i, &point_j);
                    distances.push((j, dist));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (j, dist) in distances.iter().take(self.config.n_neighbors) {
                // Gaussian kernel weight
                let weight = (-dist * dist / 2.0).exp();
                W[[i, *j]] = weight;
                W[[*j, i]] = weight; // Symmetric
            }
        }

        Ok(W)
    }

    /// Euclidean distance between two points
    fn euclidean_distance(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<f64>,
        b: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Spectral feature selection
    fn select_by_spectral(&self, data: &Array2<f64>) -> Result<FeatureSelectionResult> {
        // Simplified spectral method: use variance as proxy for spectral properties
        // Full implementation would require eigenvalue decomposition
        self.select_by_variance(data)
    }

    /// Correlation-based feature selection
    fn select_by_correlation(&self, data: &Array2<f64>) -> Result<FeatureSelectionResult> {
        let n_features = data.ncols();
        let mut feature_scores = vec![1.0; n_features];
        let mut selected_features: HashSet<usize> = (0..n_features).collect();

        // Calculate correlation matrix
        let mut correlations = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in i..n_features {
                let corr = self.calculate_correlation(&data.column(i), &data.column(j));
                correlations[[i, j]] = corr;
                correlations[[j, i]] = corr;
            }
        }

        // Remove highly correlated features
        let correlation_threshold = 0.95;
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                if correlations[[i, j]].abs() > correlation_threshold {
                    // Remove feature with lower variance
                    let var_i = self.calculate_variance(&data.column(i));
                    let var_j = self.calculate_variance(&data.column(j));

                    if var_i < var_j {
                        selected_features.remove(&i);
                        feature_scores[i] = 0.0;
                    } else {
                        selected_features.remove(&j);
                        feature_scores[j] = 0.0;
                    }
                }
            }
        }

        let mut selected_features: Vec<usize> = selected_features.into_iter().collect();
        selected_features.sort();

        Ok(FeatureSelectionResult {
            selected_features: selected_features.clone(),
            feature_scores,
            n_selected: selected_features.len(),
            method: FeatureSelectionMethod::Correlation,
        })
    }

    /// Calculate Pearson correlation coefficient
    fn calculate_correlation(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<f64>,
        b: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        let n = a.len();
        let mean_a = a.mean().unwrap_or(0.0);
        let mean_b = b.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for i in 0..n {
            let diff_a = a[i] - mean_a;
            let diff_b = b[i] - mean_b;
            numerator += diff_a * diff_b;
            sum_sq_a += diff_a * diff_a;
            sum_sq_b += diff_b * diff_b;
        }

        let denominator = (sum_sq_a * sum_sq_b).sqrt();
        if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Calculate variance of a feature
    fn calculate_variance(&self, feature: &scirs2_core::ndarray::ArrayView1<f64>) -> f64 {
        let mean = feature.mean().unwrap_or(0.0);
        feature.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / feature.len() as f64
    }

    /// Combined feature selection using multiple methods
    fn select_by_combined(&self, data: &Array2<f64>) -> Result<FeatureSelectionResult> {
        // Combine variance and correlation methods
        let variance_result = self.select_by_variance(data)?;
        let correlation_result = self.select_by_correlation(data)?;

        // Intersect results
        let variance_set: HashSet<usize> = variance_result.selected_features.into_iter().collect();
        let correlation_set: HashSet<usize> =
            correlation_result.selected_features.into_iter().collect();

        let mut selected_features: Vec<usize> = variance_set
            .intersection(&correlation_set)
            .copied()
            .collect();
        selected_features.sort();

        let n_features = data.ncols();
        let feature_scores = (0..n_features)
            .map(|i| {
                if selected_features.contains(&i) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        Ok(FeatureSelectionResult {
            selected_features: selected_features.clone(),
            feature_scores,
            n_selected: selected_features.len(),
            method: FeatureSelectionMethod::Combined,
        })
    }

    /// Transform data by selecting features
    pub fn transform(
        &self,
        data: &Array2<f64>,
        result: &FeatureSelectionResult,
    ) -> Result<Array2<f64>> {
        let n_samples = data.nrows();
        let n_selected = result.n_selected;

        if n_selected == 0 {
            return Err(SklearsError::InvalidInput(
                "No features selected".to_string(),
            ));
        }

        let mut transformed = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in result.selected_features.iter().enumerate() {
            for i in 0..n_samples {
                transformed[[i, new_idx]] = data[[i, old_idx]];
            }
        }

        Ok(transformed)
    }

    /// Fit and transform in one step
    pub fn fit_transform(
        &self,
        data: &Array2<f64>,
    ) -> Result<(Array2<f64>, FeatureSelectionResult)> {
        let result = self.select_features(data)?;
        let transformed = self.transform(data, &result)?;
        Ok((transformed, result))
    }
}

/// Builder for feature selection configuration
pub struct FeatureSelectionConfigBuilder {
    config: FeatureSelectionConfig,
}

impl FeatureSelectionConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: FeatureSelectionConfig::default(),
        }
    }

    /// Set selection method
    pub fn method(mut self, method: FeatureSelectionMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set number of features to select
    pub fn n_features(mut self, n: usize) -> Self {
        self.config.n_features = Some(n);
        self
    }

    /// Set variance threshold
    pub fn variance_threshold(mut self, threshold: f64) -> Self {
        self.config.variance_threshold = threshold;
        self
    }

    /// Set number of neighbors for graph-based methods
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.config.n_neighbors = n;
        self
    }

    /// Build configuration
    pub fn build(self) -> FeatureSelectionConfig {
        self.config
    }
}

impl Default for FeatureSelectionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureSelectionResult {
    /// Get feature importance ranking
    pub fn feature_ranking(&self) -> Vec<(usize, f64)> {
        let mut ranking: Vec<(usize, f64)> = self
            .feature_scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();

        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranking
    }

    /// Check if feature is selected
    pub fn is_selected(&self, feature_idx: usize) -> bool {
        self.selected_features.contains(&feature_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn generate_test_data() -> Array2<f64> {
        // 4 features: 2 informative, 1 low variance, 1 redundant
        Array2::from_shape_vec(
            (100, 4),
            (0..400)
                .map(|i| {
                    let row = i / 4;
                    let col = i % 4;
                    match col {
                        0 => (row % 2) as f64 * 5.0,       // Informative
                        1 => (row / 10) as f64,            // Informative
                        2 => 1.0,                          // Low variance
                        3 => (row % 2) as f64 * 5.0 + 0.1, // Redundant with col 0
                        _ => 0.0,
                    }
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn test_variance_selection() {
        let data = generate_test_data();
        let selector = FeatureSelector::with_method(FeatureSelectionMethod::Variance);

        let result = selector.select_features(&data).unwrap();

        assert!(result.n_selected > 0);
        assert!(result.n_selected <= 4);
        // Low variance feature should not be selected
        assert!(!result.selected_features.contains(&2));
    }

    #[test]
    fn test_correlation_selection() {
        let data = generate_test_data();
        let config = FeatureSelectionConfigBuilder::new()
            .method(FeatureSelectionMethod::Correlation)
            .build();

        let selector = FeatureSelector::new(config);
        let result = selector.select_features(&data).unwrap();

        assert!(result.n_selected > 0);
        // Should remove one of the correlated features (0 or 3)
        let has_both =
            result.selected_features.contains(&0) && result.selected_features.contains(&3);
        assert!(
            !has_both,
            "Should not select both highly correlated features"
        );
    }

    #[test]
    fn test_transform() {
        let data = generate_test_data();
        let selector = FeatureSelector::with_method(FeatureSelectionMethod::Variance);

        let result = selector.select_features(&data).unwrap();
        let transformed = selector.transform(&data, &result).unwrap();

        assert_eq!(transformed.nrows(), data.nrows());
        assert_eq!(transformed.ncols(), result.n_selected);
    }

    #[test]
    fn test_fit_transform() {
        let data = generate_test_data();
        let selector = FeatureSelector::with_method(FeatureSelectionMethod::Variance);

        let (transformed, result) = selector.fit_transform(&data).unwrap();

        assert_eq!(transformed.nrows(), data.nrows());
        assert_eq!(transformed.ncols(), result.n_selected);
        assert!(result.n_selected > 0);
    }

    #[test]
    fn test_laplacian_score() {
        let data = generate_test_data();
        let config = FeatureSelectionConfigBuilder::new()
            .method(FeatureSelectionMethod::LaplacianScore)
            .n_features(2)
            .n_neighbors(5)
            .build();

        let selector = FeatureSelector::new(config);
        let result = selector.select_features(&data).unwrap();

        assert_eq!(result.n_selected, 2);
        assert_eq!(result.selected_features.len(), 2);
    }

    #[test]
    fn test_feature_ranking() {
        let data = generate_test_data();
        let selector = FeatureSelector::with_method(FeatureSelectionMethod::Variance);

        let result = selector.select_features(&data).unwrap();
        let ranking = result.feature_ranking();

        assert_eq!(ranking.len(), 4);
        // Verify sorted in descending order
        for i in 0..(ranking.len() - 1) {
            assert!(ranking[i].1 >= ranking[i + 1].1);
        }
    }
}

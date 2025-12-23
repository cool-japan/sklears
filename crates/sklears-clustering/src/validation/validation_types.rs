//! Core types and validation metrics for clustering evaluation
//!
//! This module provides fundamental data structures and enumerations used
//! throughout the clustering validation framework, including distance metrics,
//! result structures, and configuration types.

use std::collections::HashMap;

/// Distance metrics available for clustering validation
///
/// Different distance metrics can significantly impact validation results,
/// particularly for high-dimensional data or data with varying scales.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ValidationMetric {
    /// Euclidean distance (L2 norm)
    ///
    /// Best for: Dense, continuous features with similar scales
    /// Formula: sqrt(Σ(xi - yi)²)
    #[default]
    Euclidean,

    /// Manhattan distance (L1 norm)
    ///
    /// Best for: High-dimensional sparse data, categorical data
    /// Formula: Σ|xi - yi|
    Manhattan,

    /// Cosine distance
    ///
    /// Best for: Text data, high-dimensional data where magnitude is less important
    /// Formula: 1 - (a·b)/(||a||·||b||)
    Cosine,

    /// Chebyshev distance (L∞ norm)
    ///
    /// Best for: When the maximum difference in any dimension is critical
    /// Formula: max|xi - yi|
    Chebyshev,

    /// Minkowski distance with custom p parameter
    ///
    /// Generalizes Euclidean (p=2) and Manhattan (p=1) distances
    /// Formula: (Σ|xi - yi|^p)^(1/p)
    Minkowski(f64),
}

impl ValidationMetric {
    /// Compute distance between two points using the selected metric
    pub fn compute_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return f64::NAN;
        }

        match self {
            ValidationMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt(),

            ValidationMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),

            ValidationMetric::Cosine => {
                let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0
                } else {
                    1.0 - (dot_product / (norm_a * norm_b))
                }
            }

            ValidationMetric::Chebyshev => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max),

            ValidationMetric::Minkowski(p) => {
                if *p <= 0.0 {
                    return f64::NAN;
                }
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs().powf(*p))
                    .sum::<f64>()
                    .powf(1.0 / p)
            }
        }
    }

    /// Get the name of the metric for display purposes
    pub fn name(&self) -> &str {
        match self {
            ValidationMetric::Euclidean => "Euclidean",
            ValidationMetric::Manhattan => "Manhattan",
            ValidationMetric::Cosine => "Cosine",
            ValidationMetric::Chebyshev => "Chebyshev",
            ValidationMetric::Minkowski(p) => {
                if *p == 1.0 {
                    "Minkowski (L1)"
                } else if *p == 2.0 {
                    "Minkowski (L2)"
                } else {
                    "Minkowski"
                }
            }
        }
    }

    /// Check if the metric is suitable for high-dimensional data
    pub fn is_high_dimensional_suitable(&self) -> bool {
        matches!(self, ValidationMetric::Manhattan | ValidationMetric::Cosine)
    }

    /// Check if the metric requires feature scaling
    pub fn requires_scaling(&self) -> bool {
        matches!(
            self,
            ValidationMetric::Euclidean
                | ValidationMetric::Chebyshev
                | ValidationMetric::Minkowski(_)
        )
    }
}

/// Result of silhouette analysis
///
/// The silhouette analysis provides a measure of how well each sample
/// fits within its assigned cluster compared to other clusters.
#[derive(Debug, Clone)]
pub struct SilhouetteResult {
    /// Individual silhouette coefficients for each sample
    ///
    /// Values range from -1 to 1, where:
    /// - 1: sample is far from neighboring clusters
    /// - 0: sample is on or very close to decision boundary
    /// - -1: sample might have been assigned to wrong cluster
    pub sample_silhouettes: Vec<f64>,

    /// Mean silhouette coefficient across all samples
    ///
    /// General interpretation:
    /// - 0.7-1.0: Strong cluster structure
    /// - 0.5-0.7: Reasonable cluster structure
    /// - 0.25-0.5: Weak cluster structure
    /// - <0.25: No substantial cluster structure
    pub mean_silhouette: f64,

    /// Average silhouette coefficient per cluster
    ///
    /// Helps identify which clusters are well-formed vs problematic
    pub cluster_silhouettes: HashMap<i32, f64>,

    /// Number of samples in each cluster
    pub cluster_sizes: HashMap<i32, usize>,

    /// Confidence intervals for mean silhouette (if computed)
    pub confidence_interval: Option<(f64, f64)>,
}

impl SilhouetteResult {
    /// Get the best performing cluster by silhouette score
    pub fn best_cluster(&self) -> Option<(i32, f64)> {
        self.cluster_silhouettes
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get the worst performing cluster by silhouette score
    pub fn worst_cluster(&self) -> Option<(i32, f64)> {
        self.cluster_silhouettes
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get problematic samples (silhouette < threshold)
    pub fn problematic_samples(&self, threshold: f64) -> Vec<usize> {
        self.sample_silhouettes
            .iter()
            .enumerate()
            .filter(|(_, &score)| score < threshold)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Calculate quality assessment based on silhouette scores
    pub fn quality_assessment(&self) -> SilhouetteQuality {
        if self.mean_silhouette >= 0.7 {
            SilhouetteQuality::Excellent
        } else if self.mean_silhouette >= 0.5 {
            SilhouetteQuality::Good
        } else if self.mean_silhouette >= 0.25 {
            SilhouetteQuality::Fair
        } else {
            SilhouetteQuality::Poor
        }
    }
}

/// Quality assessment based on silhouette scores
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilhouetteQuality {
    /// Excellent cluster structure (silhouette >= 0.7)
    Excellent,
    /// Good cluster structure (silhouette >= 0.5)
    Good,
    /// Fair cluster structure (silhouette >= 0.25)
    Fair,
    /// Poor cluster structure (silhouette < 0.25)
    Poor,
}

impl std::fmt::Display for SilhouetteQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SilhouetteQuality::Excellent => write!(f, "Excellent"),
            SilhouetteQuality::Good => write!(f, "Good"),
            SilhouetteQuality::Fair => write!(f, "Fair"),
            SilhouetteQuality::Poor => write!(f, "Poor"),
        }
    }
}

/// Gap statistic result for optimal cluster number selection
///
/// The gap statistic compares the within-cluster sum of squares
/// for different numbers of clusters against a null reference distribution.
#[derive(Debug, Clone)]
pub struct GapStatisticResult {
    /// Gap values for each tested k
    ///
    /// Higher gap values indicate better clustering
    pub gap_values: Vec<f64>,

    /// Standard errors for gap values
    ///
    /// Used to determine statistical significance of differences
    pub gap_std_errors: Vec<f64>,

    /// Optimal number of clusters according to gap statistic
    ///
    /// Determined using the "one standard error" rule:
    /// Choose the smallest k such that Gap(k) >= Gap(k+1) - se(k+1)
    pub optimal_k: usize,

    /// K values that were tested
    pub k_values: Vec<usize>,

    /// Within-cluster sum of squares for each k
    pub within_cluster_ss: Vec<f64>,

    /// Reference distribution statistics
    pub reference_statistics: Vec<ReferenceStatistics>,

    /// Number of reference datasets used per k
    pub n_references: usize,
}

impl GapStatisticResult {
    /// Get the gap value for a specific k
    pub fn gap_for_k(&self, k: usize) -> Option<f64> {
        self.k_values
            .iter()
            .position(|&x| x == k)
            .map(|idx| self.gap_values[idx])
    }

    /// Get the recommended k values (top candidates)
    pub fn recommended_k_values(&self, top_n: usize) -> Vec<(usize, f64)> {
        let mut k_gap_pairs: Vec<_> = self
            .k_values
            .iter()
            .zip(self.gap_values.iter())
            .map(|(&k, &gap)| (k, gap))
            .collect();

        k_gap_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        k_gap_pairs.truncate(top_n);
        k_gap_pairs
    }

    /// Check if the optimal k is statistically significant
    pub fn is_optimal_significant(&self) -> bool {
        if let Some(optimal_idx) = self.k_values.iter().position(|&k| k == self.optimal_k) {
            if optimal_idx + 1 < self.gap_values.len() {
                let gap_k = self.gap_values[optimal_idx];
                let gap_k_plus_1 = self.gap_values[optimal_idx + 1];
                let se_k_plus_1 = self.gap_std_errors[optimal_idx + 1];

                return gap_k >= gap_k_plus_1 - se_k_plus_1;
            }
        }
        false
    }
}

/// Reference distribution statistics for gap statistic
#[derive(Debug, Clone)]
pub struct ReferenceStatistics {
    /// K value
    pub k: usize,
    /// Mean log(W_k) across reference datasets
    pub mean_log_w: f64,
    /// Standard deviation of log(W_k) across reference datasets
    pub std_log_w: f64,
    /// Individual log(W_k) values from reference datasets
    pub reference_log_w_values: Vec<f64>,
}

/// Comprehensive clustering validation metrics
///
/// Combines multiple internal validation measures to provide
/// a holistic view of clustering quality.
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Silhouette analysis results
    pub silhouette: SilhouetteResult,

    /// Calinski-Harabasz Index (Variance Ratio Criterion)
    ///
    /// Higher values indicate better defined clusters
    /// Formula: (SSB/(k-1)) / (SSW/(n-k))
    /// where SSB = between-cluster sum of squares, SSW = within-cluster sum of squares
    pub calinski_harabasz: f64,

    /// Davies-Bouldin Index
    ///
    /// Lower values indicate better clustering
    /// Formula: (1/k) * Σ max((σi + σj) / d(ci, cj))
    /// where σi = avg distance to centroid, d(ci, cj) = centroid distance
    pub davies_bouldin: f64,

    /// Inertia (within-cluster sum of squared distances to centroids)
    ///
    /// Lower values indicate tighter clusters
    pub inertia: f64,

    /// Dunn Index (ratio of minimum inter-cluster distance to maximum intra-cluster distance)
    ///
    /// Higher values indicate better separation and compactness
    pub dunn_index: Option<f64>,

    /// Silhouette width variance
    ///
    /// Lower variance indicates more consistent cluster quality
    pub silhouette_variance: f64,

    /// Number of clusters
    pub n_clusters: usize,

    /// Number of samples
    pub n_samples: usize,

    /// Distance metric used
    pub metric_used: ValidationMetric,
}

impl ValidationMetrics {
    /// Create a summary score combining multiple metrics
    ///
    /// Returns a score between 0 and 1, where higher is better
    pub fn composite_score(&self) -> f64 {
        // Normalize individual metrics to [0, 1] scale
        let silhouette_norm = (self.silhouette.mean_silhouette + 1.0) / 2.0; // [-1, 1] -> [0, 1]
        let ch_norm = (self.calinski_harabasz / (1.0 + self.calinski_harabasz)).min(1.0);
        let db_norm = 1.0 / (1.0 + self.davies_bouldin); // Lower is better, so invert
        let dunn_norm = self.dunn_index.unwrap_or(0.0).min(1.0);

        // Weighted combination
        let weights = (0.4, 0.3, 0.2, 0.1); // (silhouette, CH, DB, Dunn)
        weights.0 * silhouette_norm
            + weights.1 * ch_norm
            + weights.2 * db_norm
            + weights.3 * dunn_norm
    }

    /// Get overall quality assessment
    pub fn overall_quality(&self) -> ClusterQuality {
        let score = self.composite_score();
        if score >= 0.8 {
            ClusterQuality::Excellent
        } else if score >= 0.6 {
            ClusterQuality::Good
        } else if score >= 0.4 {
            ClusterQuality::Fair
        } else {
            ClusterQuality::Poor
        }
    }

    /// Generate quality summary text
    pub fn quality_summary(&self) -> String {
        format!(
            "Clustering Quality Summary:\n\
             - {} clusters, {} samples\n\
             - Silhouette score: {:.3} ({})\n\
             - Calinski-Harabasz: {:.2}\n\
             - Davies-Bouldin: {:.3}\n\
             - Composite score: {:.3}\n\
             - Overall quality: {:?}",
            self.n_clusters,
            self.n_samples,
            self.silhouette.mean_silhouette,
            self.silhouette.quality_assessment(),
            self.calinski_harabasz,
            self.davies_bouldin,
            self.composite_score(),
            self.overall_quality()
        )
    }
}

/// External validation metrics when ground truth labels are available
///
/// These metrics compare clustering results against known true cluster assignments.
#[derive(Debug, Clone, Default)]
pub struct ExternalValidationMetrics {
    /// Adjusted Rand Index (ARI)
    ///
    /// Measures similarity between two clusterings, adjusted for chance
    /// Range: [-1, 1], where 1 = perfect agreement, 0 = random agreement
    pub adjusted_rand_index: f64,

    /// Normalized Mutual Information (NMI)
    ///
    /// Measures mutual dependence between cluster assignments
    /// Range: [0, 1], where 1 = perfect agreement, 0 = independent
    pub normalized_mutual_info: f64,

    /// V-measure (harmonic mean of homogeneity and completeness)
    ///
    /// Balanced measure ensuring both criteria are satisfied
    /// Range: [0, 1], where 1 = perfect clustering
    pub v_measure: f64,

    /// Homogeneity score
    ///
    /// Whether each cluster contains only members of a single class
    /// Range: [0, 1], where 1 = perfectly homogeneous
    pub homogeneity: f64,

    /// Completeness score
    ///
    /// Whether all members of a class are assigned to the same cluster
    /// Range: [0, 1], where 1 = perfectly complete
    pub completeness: f64,

    /// Fowlkes-Mallows Index (FM)
    ///
    /// Geometric mean of precision and recall
    /// Range: [0, 1], where 1 = perfect clustering
    pub fowlkes_mallows: f64,

    /// Jaccard Index
    ///
    /// Measures similarity as intersection over union of pairs
    /// Range: [0, 1], where 1 = identical clusterings
    pub jaccard_index: Option<f64>,

    /// Purity score
    ///
    /// Fraction of samples correctly clustered
    /// Range: [0, 1], where 1 = perfect clustering
    pub purity: Option<f64>,

    /// Inverse purity (coverage)
    ///
    /// Measures how well each true class is represented by clusters
    pub inverse_purity: Option<f64>,
}

impl ExternalValidationMetrics {
    /// Create a consensus score from multiple external metrics
    pub fn consensus_score(&self) -> f64 {
        let scores = [
            self.adjusted_rand_index.max(0.0), // Ensure non-negative for averaging
            self.normalized_mutual_info,
            self.v_measure,
            self.fowlkes_mallows,
        ];

        scores.iter().sum::<f64>() / scores.len() as f64
    }

    /// Check if clustering significantly matches ground truth
    pub fn is_significant_match(&self, threshold: f64) -> bool {
        self.consensus_score() >= threshold
    }

    /// Get the best performing metric
    pub fn best_metric(&self) -> (&str, f64) {
        let metrics = vec![
            ("ARI", self.adjusted_rand_index.max(0.0)),
            ("NMI", self.normalized_mutual_info),
            ("V-measure", self.v_measure),
            ("Fowlkes-Mallows", self.fowlkes_mallows),
        ];

        metrics
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("None", 0.0))
    }
}

/// Overall clustering quality assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterQuality {
    /// Excellent clustering quality (composite score >= 0.8)
    Excellent,
    /// Good clustering quality (composite score >= 0.6)
    Good,
    /// Fair clustering quality (composite score >= 0.4)
    Fair,
    /// Poor clustering quality (composite score < 0.4)
    Poor,
}

impl ClusterQuality {
    /// Get a human-readable description
    pub fn description(&self) -> &str {
        match self {
            ClusterQuality::Excellent => "Excellent: Strong, well-separated clusters",
            ClusterQuality::Good => "Good: Clear cluster structure with minor issues",
            ClusterQuality::Fair => "Fair: Some cluster structure, but improvements needed",
            ClusterQuality::Poor => "Poor: Weak or no discernible cluster structure",
        }
    }

    /// Get a numeric score representation
    pub fn score(&self) -> f64 {
        match self {
            ClusterQuality::Excellent => 0.9,
            ClusterQuality::Good => 0.7,
            ClusterQuality::Fair => 0.5,
            ClusterQuality::Poor => 0.2,
        }
    }
}

/// Configuration for validation computations
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Distance metric to use
    pub metric: ValidationMetric,

    /// Whether to compute confidence intervals
    pub compute_confidence_intervals: bool,

    /// Confidence level for intervals (e.g., 0.95 for 95%)
    pub confidence_level: f64,

    /// Whether to compute optional expensive metrics (e.g., Dunn index)
    pub compute_expensive_metrics: bool,

    /// Random seed for reproducible results
    pub random_seed: Option<u64>,

    /// Number of bootstrap samples for confidence intervals
    pub n_bootstrap_samples: usize,

    /// Whether to use parallel computation where possible
    pub use_parallel: bool,

    /// Threshold for considering samples as problematic
    pub problematic_threshold: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            metric: ValidationMetric::Euclidean,
            compute_confidence_intervals: false,
            confidence_level: 0.95,
            compute_expensive_metrics: false,
            random_seed: None,
            n_bootstrap_samples: 1000,
            use_parallel: true,
            problematic_threshold: 0.0,
        }
    }
}

impl ValidationConfig {
    /// Create a fast configuration for quick validation
    pub fn fast() -> Self {
        Self {
            compute_confidence_intervals: false,
            compute_expensive_metrics: false,
            n_bootstrap_samples: 100,
            ..Default::default()
        }
    }

    /// Create a comprehensive configuration for thorough analysis
    pub fn comprehensive() -> Self {
        Self {
            compute_confidence_intervals: true,
            compute_expensive_metrics: true,
            n_bootstrap_samples: 2000,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for high-dimensional data
    pub fn high_dimensional() -> Self {
        Self {
            metric: ValidationMetric::Cosine,
            compute_expensive_metrics: false, // Dunn index is expensive in high dimensions
            ..Default::default()
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_metric_distances() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // Test Euclidean distance
        let euclidean = ValidationMetric::Euclidean;
        let dist = euclidean.compute_distance(&a, &b);
        assert!((dist - 5.196152422706632).abs() < 1e-10);

        // Test Manhattan distance
        let manhattan = ValidationMetric::Manhattan;
        let dist = manhattan.compute_distance(&a, &b);
        assert!((dist - 9.0).abs() < 1e-10);

        // Test Cosine distance
        let cosine = ValidationMetric::Cosine;
        let dist = cosine.compute_distance(&a, &b);
        assert!(dist >= 0.0 && dist <= 2.0);
    }

    #[test]
    fn test_validation_metric_properties() {
        let euclidean = ValidationMetric::Euclidean;
        assert_eq!(euclidean.name(), "Euclidean");
        assert!(!euclidean.is_high_dimensional_suitable());
        assert!(euclidean.requires_scaling());

        let cosine = ValidationMetric::Cosine;
        assert_eq!(cosine.name(), "Cosine");
        assert!(cosine.is_high_dimensional_suitable());
        assert!(!cosine.requires_scaling());
    }

    #[test]
    fn test_silhouette_result_methods() {
        let sample_silhouettes = vec![0.8, 0.7, 0.9, 0.3, 0.6];
        let mut cluster_silhouettes = HashMap::new();
        cluster_silhouettes.insert(0, 0.8);
        cluster_silhouettes.insert(1, 0.5);

        let result = SilhouetteResult {
            sample_silhouettes: sample_silhouettes.clone(),
            mean_silhouette: 0.66,
            cluster_silhouettes,
            cluster_sizes: HashMap::new(),
            confidence_interval: None,
        };

        assert_eq!(result.quality_assessment(), SilhouetteQuality::Good);
        assert_eq!(result.best_cluster(), Some((0, 0.8)));
        assert_eq!(result.worst_cluster(), Some((1, 0.5)));

        let problematic = result.problematic_samples(0.4);
        assert_eq!(problematic, vec![3]);
    }

    #[test]
    fn test_cluster_quality_methods() {
        let excellent = ClusterQuality::Excellent;
        assert_eq!(excellent.score(), 0.9);
        assert!(excellent.description().contains("Excellent"));

        let poor = ClusterQuality::Poor;
        assert_eq!(poor.score(), 0.2);
        assert!(poor.description().contains("Poor"));
    }

    #[test]
    fn test_validation_config_presets() {
        let fast = ValidationConfig::fast();
        assert!(!fast.compute_confidence_intervals);
        assert_eq!(fast.n_bootstrap_samples, 100);

        let comprehensive = ValidationConfig::comprehensive();
        assert!(comprehensive.compute_confidence_intervals);
        assert_eq!(comprehensive.n_bootstrap_samples, 2000);

        let high_dim = ValidationConfig::high_dimensional();
        assert_eq!(high_dim.metric, ValidationMetric::Cosine);
        assert!(!high_dim.compute_expensive_metrics);
    }

    #[test]
    fn test_minkowski_distance() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];

        // Minkowski with p=1 should equal Manhattan
        let minkowski_1 = ValidationMetric::Minkowski(1.0);
        let manhattan = ValidationMetric::Manhattan;

        let dist_m1 = minkowski_1.compute_distance(&a, &b);
        let dist_man = manhattan.compute_distance(&a, &b);
        assert!((dist_m1 - dist_man).abs() < 1e-10);

        // Minkowski with p=2 should equal Euclidean
        let minkowski_2 = ValidationMetric::Minkowski(2.0);
        let euclidean = ValidationMetric::Euclidean;

        let dist_m2 = minkowski_2.compute_distance(&a, &b);
        let dist_euc = euclidean.compute_distance(&a, &b);
        assert!((dist_m2 - dist_euc).abs() < 1e-10);
    }

    #[test]
    fn test_gap_statistic_result_methods() {
        let result = GapStatisticResult {
            gap_values: vec![0.5, 0.8, 0.6, 0.4],
            gap_std_errors: vec![0.1, 0.15, 0.12, 0.08],
            optimal_k: 2,
            k_values: vec![1, 2, 3, 4],
            within_cluster_ss: vec![10.0, 5.0, 7.0, 8.0],
            reference_statistics: Vec::new(),
            n_references: 10,
        };

        assert_eq!(result.gap_for_k(2), Some(0.8));
        assert_eq!(result.gap_for_k(5), None);

        let recommended = result.recommended_k_values(2);
        assert_eq!(recommended.len(), 2);
        assert_eq!(recommended[0], (2, 0.8)); // Highest gap
    }

    #[test]
    fn test_external_validation_metrics() {
        let metrics = ExternalValidationMetrics {
            adjusted_rand_index: 0.8,
            normalized_mutual_info: 0.75,
            v_measure: 0.78,
            homogeneity: 0.8,
            completeness: 0.76,
            fowlkes_mallows: 0.82,
            jaccard_index: Some(0.7),
            purity: Some(0.85),
            inverse_purity: Some(0.8),
        };

        let consensus = metrics.consensus_score();
        assert!(consensus > 0.7 && consensus < 0.9);

        assert!(metrics.is_significant_match(0.7));
        assert!(!metrics.is_significant_match(0.9));

        let (best_name, _) = metrics.best_metric();
        assert_eq!(best_name, "Fowlkes-Mallows");
    }
}

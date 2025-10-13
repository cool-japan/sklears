//! Internal clustering validation methods
//!
//! This module implements internal validation metrics that evaluate clustering quality
//! without requiring ground truth labels. These metrics assess cluster compactness,
//! separation, and overall structure quality.

use super::validation_types::*;
use numrs2::prelude::*;
use scirs2_core::ndarray::Array2;
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

/// Main clustering validator for internal validation metrics
///
/// This struct provides methods to compute various internal validation metrics
/// including silhouette analysis, Calinski-Harabasz index, Davies-Bouldin index,
/// and other cluster quality measures.
pub struct ClusteringValidator {
    /// Distance metric to use for computations
    metric: ValidationMetric,
    /// Configuration for validation computations
    config: ValidationConfig,
}

impl ClusteringValidator {
    pub fn new(metric: ValidationMetric) -> Self {
        Self {
            metric,
            config: ValidationConfig::default(),
        }
    }

    /// Create a new clustering validator with configuration
    pub fn with_config(metric: ValidationMetric, config: ValidationConfig) -> Self {
        Self { metric, config }
    }

    /// Create validator with Euclidean distance
    pub fn euclidean() -> Self {
        Self::new(ValidationMetric::Euclidean)
    }

    /// Create validator with Manhattan distance
    pub fn manhattan() -> Self {
        Self::new(ValidationMetric::Manhattan)
    }

    /// Create validator with Cosine distance
    pub fn cosine() -> Self {
        Self::new(ValidationMetric::Cosine)
    }

    /// Create validator optimized for high-dimensional data
    pub fn high_dimensional() -> Self {
        Self::with_config(
            ValidationMetric::Cosine,
            ValidationConfig::high_dimensional(),
        )
    }

    /// Get the current distance metric
    pub fn metric(&self) -> ValidationMetric {
        self.metric
    }

    /// Update the distance metric
    pub fn set_metric(&mut self, metric: ValidationMetric) {
        self.metric = metric;
    }

    /// Get the current configuration
    pub fn config(&self) -> &ValidationConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: ValidationConfig) {
        self.config = config;
    }

    /// Compute comprehensive internal validation metrics
    ///
    /// This method combines multiple internal validation measures to provide
    /// a complete assessment of clustering quality.
    ///
    /// # Arguments
    /// * `X` - Input data matrix (n_samples x n_features)
    /// * `labels` - Cluster labels for each sample (-1 for noise points)
    ///
    /// # Returns
    /// Complete validation metrics including silhouette, CH index, DB index, and inertia
    pub fn validate(&self, X: &Array2<f64>, labels: &[i32]) -> Result<ValidationMetrics> {
        if X.nrows() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        // Compute individual metrics
        let silhouette = self.silhouette_analysis(X, labels)?;
        let calinski_harabasz = self.calinski_harabasz_index(X, labels)?;
        let davies_bouldin = self.davies_bouldin_index(X, labels)?;
        let inertia = self.compute_inertia(X, labels)?;

        // Compute optional expensive metrics
        let dunn_index = if self.config.compute_expensive_metrics {
            Some(self.dunn_index(X, labels)?)
        } else {
            None
        };

        // Compute silhouette variance
        let silhouette_variance = {
            let mean = silhouette.mean_silhouette;
            silhouette
                .sample_silhouettes
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / silhouette.sample_silhouettes.len() as f64
        };

        // Count valid clusters (excluding noise)
        let n_clusters = labels
            .iter()
            .filter(|&&label| label != -1)
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len();

        Ok(ValidationMetrics {
            silhouette,
            calinski_harabasz,
            davies_bouldin,
            inertia,
            dunn_index,
            silhouette_variance,
            n_clusters,
            n_samples: X.nrows(),
            metric_used: self.metric,
        })
    }

    /// Compute silhouette analysis for clustering evaluation
    ///
    /// The silhouette analysis measures how well each sample fits within its assigned cluster
    /// compared to other clusters. For each sample i:
    /// - a(i) = average distance to other points in the same cluster
    /// - b(i) = average distance to points in the nearest neighboring cluster
    /// - silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `labels` - Cluster labels (-1 for noise points)
    ///
    /// # Returns
    /// Comprehensive silhouette analysis results
    pub fn silhouette_analysis(&self, X: &Array2<f64>, labels: &[i32]) -> Result<SilhouetteResult> {
        if X.nrows() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let mut sample_silhouettes = Vec::with_capacity(n_samples);

        // Group points by cluster
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            clusters.entry(label).or_default().push(i);
        }

        // Remove noise points (label -1) from cluster analysis
        clusters.remove(&-1);

        if clusters.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 clusters for silhouette analysis".to_string(),
            ));
        }

        // Compute silhouette for each sample
        for i in 0..n_samples {
            let label = labels[i];

            // Noise points get silhouette score of 0
            if label == -1 {
                sample_silhouettes.push(0.0);
                continue;
            }

            let cluster_points = &clusters[&label];

            // Compute a(i) - mean intra-cluster distance
            let mut intra_distance = 0.0;
            let mut intra_count = 0;

            for &j in cluster_points {
                if i != j {
                    intra_distance += self
                        .metric
                        .compute_distance(&X.row(i).to_vec(), &X.row(j).to_vec());
                    intra_count += 1;
                }
            }

            let a_i = if intra_count > 0 {
                intra_distance / intra_count as f64
            } else {
                0.0
            };

            // Compute b(i) - mean nearest-cluster distance
            let mut min_inter_distance = f64::INFINITY;

            for (&other_label, other_points) in &clusters {
                if other_label != label {
                    let mut inter_distance = 0.0;

                    for &j in other_points {
                        inter_distance += self
                            .metric
                            .compute_distance(&X.row(i).to_vec(), &X.row(j).to_vec());
                    }

                    let avg_inter_distance = inter_distance / other_points.len() as f64;
                    if avg_inter_distance < min_inter_distance {
                        min_inter_distance = avg_inter_distance;
                    }
                }
            }

            let b_i = min_inter_distance;

            // Compute silhouette coefficient
            let silhouette = if a_i == 0.0 && b_i == 0.0 {
                0.0
            } else {
                (b_i - a_i) / f64::max(a_i, b_i)
            };

            sample_silhouettes.push(silhouette);
        }

        // Compute cluster-wise silhouettes and sizes
        let mut cluster_silhouettes = HashMap::new();
        let mut cluster_sizes = HashMap::new();

        for (&label, cluster_points) in &clusters {
            let cluster_sil: f64 = cluster_points
                .iter()
                .map(|&i| sample_silhouettes[i])
                .sum::<f64>()
                / cluster_points.len() as f64;

            cluster_silhouettes.insert(label, cluster_sil);
            cluster_sizes.insert(label, cluster_points.len());
        }

        let mean_silhouette =
            sample_silhouettes.iter().sum::<f64>() / sample_silhouettes.len() as f64;

        // Compute confidence interval if requested
        let confidence_interval = if self.config.compute_confidence_intervals {
            Some(self.compute_silhouette_confidence_interval(&sample_silhouettes)?)
        } else {
            None
        };

        Ok(SilhouetteResult {
            sample_silhouettes,
            mean_silhouette,
            cluster_silhouettes,
            cluster_sizes,
            confidence_interval,
        })
    }

    /// Compute Calinski-Harabasz Index (Variance Ratio Criterion)
    ///
    /// The CH index measures the ratio of between-cluster variance to within-cluster variance.
    /// Higher values indicate better clustering with well-separated, compact clusters.
    ///
    /// Formula: CH = (SSB / (k-1)) / (SSW / (n-k))
    /// where SSB = between-cluster sum of squares, SSW = within-cluster sum of squares
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `labels` - Cluster labels
    ///
    /// # Returns
    /// Calinski-Harabasz index value
    pub fn calinski_harabasz_index(&self, X: &Array2<f64>, labels: &[i32]) -> Result<f64> {
        if X.nrows() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Group points by cluster (exclude noise points)
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if label != -1 {
                clusters.entry(label).or_default().push(i);
            }
        }

        if clusters.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 clusters for Calinski-Harabasz index".to_string(),
            ));
        }

        let n_clusters = clusters.len();

        // Compute overall centroid
        let mut overall_centroid = vec![0.0; n_features];
        let mut n_valid_samples = 0;

        for i in 0..n_samples {
            if labels[i] != -1 {
                for j in 0..n_features {
                    overall_centroid[j] += X[(i, j)];
                }
                n_valid_samples += 1;
            }
        }

        for val in overall_centroid.iter_mut() {
            *val /= n_valid_samples as f64;
        }

        // Compute cluster centroids and between-cluster sum of squares (SSB)
        let mut ssb = 0.0;
        let mut ssw = 0.0;

        for cluster_points in clusters.values() {
            // Compute cluster centroid
            let mut cluster_centroid = vec![0.0; n_features];
            for &point_idx in cluster_points {
                for j in 0..n_features {
                    cluster_centroid[j] += X[(point_idx, j)];
                }
            }

            for val in cluster_centroid.iter_mut() {
                *val /= cluster_points.len() as f64;
            }

            // Add to between-cluster sum of squares
            let cluster_size = cluster_points.len() as f64;
            for j in 0..n_features {
                ssb += cluster_size * (cluster_centroid[j] - overall_centroid[j]).powi(2);
            }

            // Add to within-cluster sum of squares
            for &point_idx in cluster_points {
                for j in 0..n_features {
                    ssw += (X[(point_idx, j)] - cluster_centroid[j]).powi(2);
                }
            }
        }

        // Compute Calinski-Harabasz index
        if ssw == 0.0 {
            Ok(f64::INFINITY)
        } else {
            let ch_index =
                (ssb / (n_clusters - 1) as f64) / (ssw / (n_valid_samples - n_clusters) as f64);
            Ok(ch_index)
        }
    }

    /// Compute Davies-Bouldin Index
    ///
    /// The DB index measures the average similarity between clusters, where similarity
    /// is the ratio of within-cluster distances to between-cluster distances.
    /// Lower values indicate better clustering.
    ///
    /// Formula: DB = (1/k) * Σ max((σi + σj) / d(ci, cj))
    /// where σi = average distance to centroid, d(ci, cj) = distance between centroids
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `labels` - Cluster labels
    ///
    /// # Returns
    /// Davies-Bouldin index value
    pub fn davies_bouldin_index(&self, X: &Array2<f64>, labels: &[i32]) -> Result<f64> {
        if X.nrows() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        let n_features = X.ncols();

        // Group points by cluster (exclude noise points)
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if label != -1 {
                clusters.entry(label).or_default().push(i);
            }
        }

        if clusters.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 clusters for Davies-Bouldin index".to_string(),
            ));
        }

        // Compute cluster centroids and within-cluster dispersions
        let mut centroids = HashMap::new();
        let mut dispersions = HashMap::new();

        for (&label, cluster_points) in &clusters {
            // Compute cluster centroid
            let mut centroid = vec![0.0; n_features];
            for &point_idx in cluster_points {
                for j in 0..n_features {
                    centroid[j] += X[(point_idx, j)];
                }
            }

            for val in centroid.iter_mut() {
                *val /= cluster_points.len() as f64;
            }

            // Compute within-cluster dispersion (average distance to centroid)
            let mut total_distance = 0.0;
            for &point_idx in cluster_points {
                total_distance += self
                    .metric
                    .compute_distance(&X.row(point_idx).to_vec(), &centroid);
            }

            let dispersion = total_distance / cluster_points.len() as f64;

            centroids.insert(label, centroid);
            dispersions.insert(label, dispersion);
        }

        // Compute Davies-Bouldin index
        let mut db_sum = 0.0;
        let cluster_labels: Vec<_> = clusters.keys().cloned().collect();

        for &i in &cluster_labels {
            let mut max_ratio = 0.0;

            for &j in &cluster_labels {
                if i != j {
                    let centroid_distance =
                        self.metric.compute_distance(&centroids[&i], &centroids[&j]);

                    if centroid_distance > 0.0 {
                        let ratio = (dispersions[&i] + dispersions[&j]) / centroid_distance;
                        if ratio > max_ratio {
                            max_ratio = ratio;
                        }
                    }
                }
            }

            db_sum += max_ratio;
        }

        Ok(db_sum / cluster_labels.len() as f64)
    }

    /// Compute inertia (within-cluster sum of squared distances to centroids)
    ///
    /// Inertia measures how internally coherent clusters are. Lower values
    /// indicate tighter clusters.
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `labels` - Cluster labels
    ///
    /// # Returns
    /// Total inertia value
    pub fn compute_inertia(&self, X: &Array2<f64>, labels: &[i32]) -> Result<f64> {
        if X.nrows() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        let n_features = X.ncols();

        // Group points by cluster (exclude noise points)
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if label != -1 {
                clusters.entry(label).or_default().push(i);
            }
        }

        let mut total_inertia = 0.0;

        for cluster_points in clusters.values() {
            if cluster_points.is_empty() {
                continue;
            }

            // Compute cluster centroid
            let mut centroid = vec![0.0; n_features];
            for &point_idx in cluster_points {
                for j in 0..n_features {
                    centroid[j] += X[(point_idx, j)];
                }
            }

            for val in centroid.iter_mut() {
                *val /= cluster_points.len() as f64;
            }

            // Sum squared distances to centroid
            for &point_idx in cluster_points {
                let point = X.row(point_idx).to_vec();
                let distance = self.metric.compute_distance(&point, &centroid);
                total_inertia += distance * distance;
            }
        }

        Ok(total_inertia)
    }

    /// Compute Dunn Index
    ///
    /// The Dunn index is the ratio of the minimum inter-cluster distance
    /// to the maximum intra-cluster distance. Higher values indicate better clustering.
    /// This metric is computationally expensive for large datasets.
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `labels` - Cluster labels
    ///
    /// # Returns
    /// Dunn index value
    pub fn dunn_index(&self, X: &Array2<f64>, labels: &[i32]) -> Result<f64> {
        if X.nrows() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        // Group points by cluster (exclude noise points)
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if label != -1 {
                clusters.entry(label).or_default().push(i);
            }
        }

        if clusters.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 clusters for Dunn index".to_string(),
            ));
        }

        // Find minimum inter-cluster distance
        let mut min_inter_distance = f64::INFINITY;
        let cluster_labels: Vec<_> = clusters.keys().cloned().collect();

        for i in 0..cluster_labels.len() {
            for j in (i + 1)..cluster_labels.len() {
                let cluster1 = &clusters[&cluster_labels[i]];
                let cluster2 = &clusters[&cluster_labels[j]];

                for &point1 in cluster1 {
                    for &point2 in cluster2 {
                        let distance = self
                            .metric
                            .compute_distance(&X.row(point1).to_vec(), &X.row(point2).to_vec());
                        if distance < min_inter_distance {
                            min_inter_distance = distance;
                        }
                    }
                }
            }
        }

        // Find maximum intra-cluster distance
        let mut max_intra_distance = 0.0;

        for cluster_points in clusters.values() {
            for i in 0..cluster_points.len() {
                for j in (i + 1)..cluster_points.len() {
                    let distance = self.metric.compute_distance(
                        &X.row(cluster_points[i]).to_vec(),
                        &X.row(cluster_points[j]).to_vec(),
                    );
                    if distance > max_intra_distance {
                        max_intra_distance = distance;
                    }
                }
            }
        }

        // Compute Dunn index
        if max_intra_distance == 0.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(min_inter_distance / max_intra_distance)
        }
    }

    /// Compute confidence interval for silhouette scores using bootstrap
    fn compute_silhouette_confidence_interval(&self, silhouettes: &[f64]) -> Result<(f64, f64)> {
        use scirs2_core::random::Random;

        let n_samples = silhouettes.len();
        let n_bootstrap = self.config.n_bootstrap_samples;
        let mut bootstrap_means = Vec::with_capacity(n_bootstrap);

        let mut rng = Random::seed(self.config.random_seed.unwrap_or(42));

        for _ in 0..n_bootstrap {
            let mut bootstrap_sample = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                let idx = rng.gen_range(0..n_samples);
                bootstrap_sample.push(silhouettes[idx]);
            }

            let mean = bootstrap_sample.iter().sum::<f64>() / bootstrap_sample.len() as f64;
            bootstrap_means.push(mean);
        }

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha = 1.0 - self.config.confidence_level;
        let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;

        let lower_bound = bootstrap_means[lower_idx.min(n_bootstrap - 1)];
        let upper_bound = bootstrap_means[upper_idx.min(n_bootstrap - 1)];

        Ok((lower_bound, upper_bound))
    }

    /// Compute pairwise distances between all clusters
    pub fn compute_inter_cluster_distances(
        &self,
        X: &Array2<f64>,
        labels: &[i32],
    ) -> Result<HashMap<(i32, i32), f64>> {
        let n_features = X.ncols();

        // Group points by cluster
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if label != -1 {
                clusters.entry(label).or_default().push(i);
            }
        }

        // Compute cluster centroids
        let mut centroids = HashMap::new();
        for (&label, cluster_points) in &clusters {
            let mut centroid = vec![0.0; n_features];
            for &point_idx in cluster_points {
                for j in 0..n_features {
                    centroid[j] += X[(point_idx, j)];
                }
            }
            for val in centroid.iter_mut() {
                *val /= cluster_points.len() as f64;
            }
            centroids.insert(label, centroid);
        }

        // Compute pairwise distances
        let mut distances = HashMap::new();
        let cluster_labels: Vec<_> = clusters.keys().cloned().collect();

        for i in 0..cluster_labels.len() {
            for j in (i + 1)..cluster_labels.len() {
                let label1 = cluster_labels[i];
                let label2 = cluster_labels[j];

                let distance = self
                    .metric
                    .compute_distance(&centroids[&label1], &centroids[&label2]);

                distances.insert((label1, label2), distance);
                distances.insert((label2, label1), distance);
            }
        }

        Ok(distances)
    }

    /// Compute within-cluster distances for each cluster
    pub fn compute_intra_cluster_distances(
        &self,
        X: &Array2<f64>,
        labels: &[i32],
    ) -> Result<HashMap<i32, Vec<f64>>> {
        // Group points by cluster
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if label != -1 {
                clusters.entry(label).or_default().push(i);
            }
        }

        let mut cluster_distances = HashMap::new();

        for (&label, cluster_points) in &clusters {
            let mut distances = Vec::new();

            for i in 0..cluster_points.len() {
                for j in (i + 1)..cluster_points.len() {
                    let distance = self.metric.compute_distance(
                        &X.row(cluster_points[i]).to_vec(),
                        &X.row(cluster_points[j]).to_vec(),
                    );
                    distances.push(distance);
                }
            }

            cluster_distances.insert(label, distances);
        }

        Ok(cluster_distances)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data() -> (Array2<f64>, Vec<i32>) {
        // Create simple test data with clear cluster structure
        let data_vec = vec![
            // Cluster 0
            1.0, 1.0, 1.2, 1.1, 1.1, 1.2, // Cluster 1
            5.0, 5.0, 5.1, 5.2, 5.2, 5.1, // Cluster 2
            9.0, 9.0, 9.1, 9.2, 9.2, 9.1,
        ];
        let data = Array2::from_shape_vec((9, 2), data_vec).unwrap();
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        (data, labels)
    }

    #[test]
    fn test_validator_creation() {
        let validator = ClusteringValidator::euclidean();
        assert_eq!(validator.metric(), ValidationMetric::Euclidean);

        let validator = ClusteringValidator::manhattan();
        assert_eq!(validator.metric(), ValidationMetric::Manhattan);

        let validator = ClusteringValidator::cosine();
        assert_eq!(validator.metric(), ValidationMetric::Cosine);

        let validator = ClusteringValidator::high_dimensional();
        assert_eq!(validator.metric(), ValidationMetric::Cosine);
    }

    #[test]
    fn test_silhouette_analysis() {
        let (data, labels) = generate_test_data();
        let validator = ClusteringValidator::euclidean();

        let result = validator.silhouette_analysis(&data, &labels).unwrap();

        assert_eq!(result.sample_silhouettes.len(), 9);
        assert!(result.mean_silhouette >= 0.0);
        assert!(result.mean_silhouette <= 1.0);
        assert_eq!(result.cluster_silhouettes.len(), 3);
        assert_eq!(result.cluster_sizes.len(), 3);

        // Check that all cluster sizes are correct
        for (_, &size) in &result.cluster_sizes {
            assert_eq!(size, 3);
        }
    }

    #[test]
    fn test_calinski_harabasz_index() {
        let (data, labels) = generate_test_data();
        let validator = ClusteringValidator::euclidean();

        let ch_index = validator.calinski_harabasz_index(&data, &labels).unwrap();
        assert!(ch_index > 0.0);
        assert!(ch_index.is_finite());
    }

    #[test]
    fn test_davies_bouldin_index() {
        let (data, labels) = generate_test_data();
        let validator = ClusteringValidator::euclidean();

        let db_index = validator.davies_bouldin_index(&data, &labels).unwrap();
        assert!(db_index >= 0.0);
        assert!(db_index.is_finite());
    }

    #[test]
    fn test_inertia_computation() {
        let (data, labels) = generate_test_data();
        let validator = ClusteringValidator::euclidean();

        let inertia = validator.compute_inertia(&data, &labels).unwrap();
        assert!(inertia >= 0.0);
        assert!(inertia.is_finite());
    }

    #[test]
    fn test_dunn_index() {
        let (data, labels) = generate_test_data();
        let validator = ClusteringValidator::euclidean();

        let dunn = validator.dunn_index(&data, &labels).unwrap();
        assert!(dunn > 0.0);
        assert!(dunn.is_finite());
    }

    #[test]
    fn test_comprehensive_validation() {
        let (data, labels) = generate_test_data();
        let validator = ClusteringValidator::euclidean();

        let metrics = validator.validate(&data, &labels).unwrap();

        assert_eq!(metrics.n_clusters, 3);
        assert_eq!(metrics.n_samples, 9);
        assert_eq!(metrics.metric_used, ValidationMetric::Euclidean);
        assert!(metrics.silhouette.mean_silhouette >= 0.0);
        assert!(metrics.calinski_harabasz > 0.0);
        assert!(metrics.davies_bouldin >= 0.0);
        assert!(metrics.inertia >= 0.0);
        assert!(metrics.silhouette_variance >= 0.0);

        // Test composite score
        let score = metrics.composite_score();
        assert!(score >= 0.0 && score <= 1.0);

        // Test quality assessment
        let quality = metrics.overall_quality();
        assert!(matches!(
            quality,
            ClusterQuality::Excellent
                | ClusterQuality::Good
                | ClusterQuality::Fair
                | ClusterQuality::Poor
        ));
    }

    #[test]
    fn test_error_handling() {
        let validator = ClusteringValidator::euclidean();

        // Test dimension mismatch
        let data = Array2::zeros((5, 2));
        let labels = vec![0, 1, 2]; // Wrong length

        let result = validator.silhouette_analysis(&data, &labels);
        assert!(result.is_err());

        // Test insufficient clusters
        let labels = vec![0, 0, 0, 0, 0]; // Only one cluster
        let result = validator.silhouette_analysis(&data, &labels);
        assert!(result.is_err());
    }

    #[test]
    fn test_noise_handling() {
        let (mut data, mut labels) = generate_test_data();

        // Add noise points
        labels.push(-1);
        labels.push(-1);
        data = data.into_shape((9, 2)).unwrap().into_owned();
        let noise_data = Array2::from_shape_vec((2, 2), vec![10.0, 10.0, 15.0, 15.0]).unwrap();
        data = scirs2_core::ndarray::concatenate![
            scirs2_core::ndarray::Axis(0),
            data.view(),
            noise_data.view()
        ];

        let validator = ClusteringValidator::euclidean();
        let result = validator.silhouette_analysis(&data, &labels).unwrap();

        // Noise points should have silhouette score of 0
        assert_eq!(result.sample_silhouettes[9], 0.0);
        assert_eq!(result.sample_silhouettes[10], 0.0);

        // Should still have 3 clusters (noise excluded)
        assert_eq!(result.cluster_silhouettes.len(), 3);
    }

    #[test]
    fn test_different_metrics() {
        let (data, labels) = generate_test_data();

        let euclidean_validator = ClusteringValidator::euclidean();
        let manhattan_validator = ClusteringValidator::manhattan();
        let cosine_validator = ClusteringValidator::cosine();

        let euc_result = euclidean_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();
        let man_result = manhattan_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();
        let cos_result = cosine_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();

        // All should produce valid results
        assert!(euc_result.mean_silhouette.is_finite());
        assert!(man_result.mean_silhouette.is_finite());
        assert!(cos_result.mean_silhouette.is_finite());

        // Results may differ between metrics
        // (This is expected and normal)
    }

    #[test]
    fn test_inter_cluster_distances() {
        let (data, labels) = generate_test_data();
        let validator = ClusteringValidator::euclidean();

        let distances = validator
            .compute_inter_cluster_distances(&data, &labels)
            .unwrap();

        // Should have distances between all cluster pairs
        assert!(distances.contains_key(&(0, 1)));
        assert!(distances.contains_key(&(1, 0)));
        assert!(distances.contains_key(&(0, 2)));
        assert!(distances.contains_key(&(2, 0)));
        assert!(distances.contains_key(&(1, 2)));
        assert!(distances.contains_key(&(2, 1)));

        // Distances should be symmetric
        assert_eq!(distances[&(0, 1)], distances[&(1, 0)]);
        assert_eq!(distances[&(0, 2)], distances[&(2, 0)]);
        assert_eq!(distances[&(1, 2)], distances[&(2, 1)]);

        // All distances should be positive
        for (_, &distance) in &distances {
            assert!(distance > 0.0);
            assert!(distance.is_finite());
        }
    }

    #[test]
    fn test_intra_cluster_distances() {
        let (data, labels) = generate_test_data();
        let validator = ClusteringValidator::euclidean();

        let distances = validator
            .compute_intra_cluster_distances(&data, &labels)
            .unwrap();

        // Should have distances for each cluster
        assert_eq!(distances.len(), 3);
        assert!(distances.contains_key(&0));
        assert!(distances.contains_key(&1));
        assert!(distances.contains_key(&2));

        // Each cluster should have pairwise distances
        for (_, cluster_distances) in &distances {
            // For 3 points, should have 3 pairwise distances
            assert_eq!(cluster_distances.len(), 3);

            // All distances should be non-negative
            for &distance in cluster_distances {
                assert!(distance >= 0.0);
                assert!(distance.is_finite());
            }
        }
    }
}

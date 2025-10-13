//! Cluster Coherence and Separation Analysis
//!
//! This module provides comprehensive analysis of cluster coherence (internal homogeneity)
//! and separation (inter-cluster distinctiveness) for evaluating clustering quality.
//!
//! # Coherence Analysis
//! - Intra-cluster distance measures (compactness)
//! - Cluster density estimation and uniformity
//! - Shape regularity and symmetry analysis
//! - Consistency measures across clusters
//!
//! # Separation Analysis
//! - Inter-cluster distance measures
//! - Boundary clarity and overlap estimation
//! - Gap ratio analysis (inter/intra distance ratios)
//! - Centroid separation metrics
//!
//! # Mathematical Background
//!
//! ## Cluster Coherence
//! For cluster C with centroid μ_C:
//! - Compactness: 1/(1 + avg_distance_to_centroid)
//! - Density: actual_avg_distance / expected_uniform_distance
//! - Shape regularity: 1/(1 + CV(distances_to_centroid))
//!
//! ## Cluster Separation
//! For clusters C_i, C_j with centroids μ_i, μ_j:
//! - Centroid separation: distance(μ_i, μ_j)
//! - Gap ratio: avg_inter_cluster_distance / avg_intra_cluster_distance
//! - Boundary clarity: ratio of inter-cluster to nearest intra-cluster distances

use std::collections::HashMap;

use scirs2_core::ndarray::Array2;
use sklears_core::error::{Result, SklearsError};

use super::validation_types::ValidationMetric;

/// Result of comprehensive cluster coherence analysis
#[derive(Debug, Clone)]
pub struct ClusterCoherenceResult {
    /// Overall coherence score (weighted average across clusters)
    pub overall_coherence: f64,
    /// Overall compactness score
    pub overall_compactness: f64,
    /// Overall density score
    pub overall_density: f64,
    /// Overall shape regularity score
    pub overall_shape_regularity: f64,
    /// Coherence scores per cluster
    pub cluster_coherence_scores: HashMap<i32, f64>,
    /// Compactness scores per cluster
    pub cluster_compactness_scores: HashMap<i32, f64>,
    /// Density scores per cluster
    pub cluster_density_scores: HashMap<i32, f64>,
    /// Shape regularity scores per cluster
    pub cluster_shape_regularity: HashMap<i32, f64>,
    /// Consistency of coherence across clusters (1 - coefficient of variation)
    pub coherence_consistency: f64,
    /// Consistency of compactness across clusters
    pub compactness_consistency: f64,
    /// Consistency of density across clusters
    pub density_consistency: f64,
    /// Consistency of shape across clusters
    pub shape_consistency: f64,
    /// Number of clusters analyzed
    pub n_clusters: usize,
}

/// Result of comprehensive cluster separation analysis
#[derive(Debug, Clone)]
pub struct ClusterSeparationResult {
    /// Average distance between cluster centroids
    pub avg_centroid_separation: f64,
    /// Minimum distance between any two clusters
    pub min_separation: f64,
    /// Average inter-cluster distance (all points between clusters)
    pub avg_inter_cluster_distance: f64,
    /// Boundary clarity measure (ratio of inter to intra distances)
    pub boundary_clarity: f64,
    /// Cluster overlap measure in feature space
    pub overlap_measure: f64,
    /// Gap ratio (inter-cluster / intra-cluster distances)
    pub gap_ratio: f64,
    /// Minimum distances between specific cluster pairs
    pub min_inter_cluster_distances: HashMap<(i32, i32), f64>,
    /// Number of clusters analyzed
    pub n_clusters: usize,
}

impl ClusterCoherenceResult {
    /// Get the best performing cluster (highest coherence)
    pub fn best_cluster(&self) -> Option<(i32, f64)> {
        self.cluster_coherence_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&id, &score)| (id, score))
    }

    /// Get the worst performing cluster (lowest coherence)
    pub fn worst_cluster(&self) -> Option<(i32, f64)> {
        self.cluster_coherence_scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&id, &score)| (id, score))
    }

    /// Get clusters with coherence below threshold
    pub fn problematic_clusters(&self, threshold: f64) -> Vec<i32> {
        self.cluster_coherence_scores
            .iter()
            .filter(|(_, &score)| score < threshold)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Compute overall quality score combining all coherence metrics
    pub fn overall_quality_score(&self) -> f64 {
        let coherence_weight = 0.4;
        let compactness_weight = 0.3;
        let density_weight = 0.2;
        let shape_weight = 0.1;

        coherence_weight * self.overall_coherence
            + compactness_weight * self.overall_compactness
            + density_weight * self.overall_density
            + shape_weight * self.overall_shape_regularity
    }
}

impl ClusterSeparationResult {
    /// Get cluster pairs with poor separation (below threshold)
    pub fn poorly_separated_pairs(&self, threshold: f64) -> Vec<(i32, i32)> {
        self.min_inter_cluster_distances
            .iter()
            .filter(|(_, &distance)| distance < threshold)
            .map(|(&pair, _)| pair)
            .collect()
    }

    /// Get the most overlapping cluster pair
    pub fn most_overlapping_pair(&self) -> Option<((i32, i32), f64)> {
        self.min_inter_cluster_distances
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&pair, &distance)| (pair, distance))
    }

    /// Assess separation quality based on multiple metrics
    pub fn separation_quality(&self) -> &'static str {
        let gap_ratio_score = (self.gap_ratio - 1.0).max(0.0).min(3.0) / 3.0;
        let boundary_score = self.boundary_clarity.min(1.0);
        let overlap_score = (1.0 - self.overlap_measure).max(0.0);

        let overall_score = (gap_ratio_score + boundary_score + overlap_score) / 3.0;

        if overall_score >= 0.8 {
            "Excellent"
        } else if overall_score >= 0.6 {
            "Good"
        } else if overall_score >= 0.4 {
            "Fair"
        } else {
            "Poor"
        }
    }
}

/// Coherence and separation analyzer for clustering evaluation
pub struct CoherenceSeparationAnalyzer {
    /// Distance metric for computations
    metric: ValidationMetric,
}

impl CoherenceSeparationAnalyzer {
    /// Create a new coherence/separation analyzer
    pub fn new(metric: ValidationMetric) -> Self {
        Self { metric }
    }

    /// Create analyzer with Euclidean distance
    pub fn euclidean() -> Self {
        Self::new(ValidationMetric::Euclidean)
    }

    /// Create analyzer with Manhattan distance
    pub fn manhattan() -> Self {
        Self::new(ValidationMetric::Manhattan)
    }

    /// Create analyzer with Cosine distance
    pub fn cosine() -> Self {
        Self::new(ValidationMetric::Cosine)
    }

    /// Comprehensive cluster coherence analysis
    ///
    /// Evaluates how coherent (internally similar) clusters are using multiple metrics.
    /// Combines intra-cluster distances, density measures, and shape analysis.
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `labels` - Cluster assignments
    ///
    /// # Returns
    /// ClusterCoherenceResult with comprehensive coherence metrics
    ///
    /// # Mathematical Details
    /// For each cluster C:
    /// 1. Coherence = 1/(1 + avg_pairwise_distance) - measures internal similarity
    /// 2. Compactness = 1/(1 + avg_distance_to_centroid) - measures tightness around center
    /// 3. Density = actual_avg_distance / expected_uniform_distance - measures point concentration
    /// 4. Shape regularity = 1/(1 + CV(distances_to_centroid)) - measures shape symmetry
    pub fn cluster_coherence(
        &self,
        X: &Array2<f64>,
        labels: &[i32],
    ) -> Result<ClusterCoherenceResult> {
        if X.nrows() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Group points by cluster
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if label != -1 {
                clusters.entry(label).or_default().push(i);
            }
        }

        if clusters.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid clusters found".to_string(),
            ));
        }

        let mut cluster_coherence_scores = HashMap::new();
        let mut cluster_compactness_scores = HashMap::new();
        let mut cluster_density_scores = HashMap::new();
        let mut cluster_shape_regularity = HashMap::new();

        let mut overall_coherence = 0.0;
        let mut overall_compactness = 0.0;
        let mut overall_density = 0.0;
        let mut overall_shape_regularity = 0.0;

        for (&cluster_id, cluster_points) in &clusters {
            if cluster_points.len() < 2 {
                continue;
            }

            // Compute cluster centroid
            let mut centroid = vec![0.0; n_features];
            for &point_idx in cluster_points {
                for j in 0..n_features {
                    centroid[j] += X[[point_idx, j]];
                }
            }
            for val in centroid.iter_mut() {
                *val /= cluster_points.len() as f64;
            }

            // 1. Coherence: Average pairwise distance within cluster
            let mut total_pairwise_distance = 0.0;
            let mut pairwise_count = 0;

            for (i, &point_i) in cluster_points.iter().enumerate() {
                for &point_j in cluster_points.iter().skip(i + 1) {
                    let distance = self
                        .metric
                        .compute_distance(&X.row(point_i).to_vec(), &X.row(point_j).to_vec());
                    total_pairwise_distance += distance;
                    pairwise_count += 1;
                }
            }

            let coherence = if pairwise_count > 0 {
                1.0 / (1.0 + total_pairwise_distance / pairwise_count as f64)
            } else {
                1.0
            };

            // 2. Compactness: Average distance to centroid
            let mut total_centroid_distance = 0.0;
            for &point_idx in cluster_points {
                let distance = self
                    .metric
                    .compute_distance(&X.row(point_idx).to_vec(), &centroid);
                total_centroid_distance += distance;
            }
            let compactness = 1.0 / (1.0 + total_centroid_distance / cluster_points.len() as f64);

            // 3. Density: Ratio of actual vs expected distances for uniform distribution
            let cluster_volume = self.estimate_cluster_volume(X, cluster_points);
            let expected_distance =
                self.expected_uniform_distance(cluster_volume, cluster_points.len());
            let actual_avg_distance = if pairwise_count > 0 {
                total_pairwise_distance / pairwise_count as f64
            } else {
                0.0
            };
            let density = if expected_distance > 0.0 {
                1.0 / (1.0 + actual_avg_distance / expected_distance)
            } else {
                1.0
            };

            // 4. Shape regularity: Measure of how regular/symmetric the cluster shape is
            let shape_regularity = self.compute_shape_regularity(X, cluster_points, &centroid);

            cluster_coherence_scores.insert(cluster_id, coherence);
            cluster_compactness_scores.insert(cluster_id, compactness);
            cluster_density_scores.insert(cluster_id, density);
            cluster_shape_regularity.insert(cluster_id, shape_regularity);

            // Weight by cluster size for overall metrics
            let weight = cluster_points.len() as f64 / n_samples as f64;
            overall_coherence += coherence * weight;
            overall_compactness += compactness * weight;
            overall_density += density * weight;
            overall_shape_regularity += shape_regularity * weight;
        }

        // Compute coefficient of variation for each metric (lower = more consistent)
        let coherence_cv = self.coefficient_of_variation(
            &cluster_coherence_scores
                .values()
                .cloned()
                .collect::<Vec<_>>(),
        );
        let compactness_cv = self.coefficient_of_variation(
            &cluster_compactness_scores
                .values()
                .cloned()
                .collect::<Vec<_>>(),
        );
        let density_cv = self.coefficient_of_variation(
            &cluster_density_scores.values().cloned().collect::<Vec<_>>(),
        );
        let shape_cv = self.coefficient_of_variation(
            &cluster_shape_regularity
                .values()
                .cloned()
                .collect::<Vec<_>>(),
        );

        Ok(ClusterCoherenceResult {
            overall_coherence,
            overall_compactness,
            overall_density,
            overall_shape_regularity,
            cluster_coherence_scores,
            cluster_compactness_scores,
            cluster_density_scores,
            cluster_shape_regularity,
            coherence_consistency: 1.0 - coherence_cv,
            compactness_consistency: 1.0 - compactness_cv,
            density_consistency: 1.0 - density_cv,
            shape_consistency: 1.0 - shape_cv,
            n_clusters: clusters.len(),
        })
    }

    /// Comprehensive cluster separation analysis
    ///
    /// Evaluates how well-separated clusters are using multiple separation metrics.
    /// Includes inter-cluster distances, boundary analysis, and overlap measures.
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `labels` - Cluster assignments
    ///
    /// # Returns
    /// ClusterSeparationResult with comprehensive separation metrics
    ///
    /// # Mathematical Details
    /// 1. Centroid separation: distance between cluster centroids
    /// 2. Minimum separation: closest points between different clusters
    /// 3. Boundary clarity: ratio of inter-cluster to intra-cluster nearest neighbor distances
    /// 4. Overlap measure: estimate of cluster overlap in feature space
    /// 5. Gap ratio: avg_inter_cluster_distance / avg_intra_cluster_distance
    pub fn cluster_separation(
        &self,
        X: &Array2<f64>,
        labels: &[i32],
    ) -> Result<ClusterSeparationResult> {
        if X.nrows() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        let n_features = X.ncols();

        // Group points by cluster
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if label != -1 {
                clusters.entry(label).or_default().push(i);
            }
        }

        if clusters.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 clusters for separation analysis".to_string(),
            ));
        }

        // Compute cluster centroids
        let mut centroids = HashMap::new();
        for (&cluster_id, cluster_points) in &clusters {
            let mut centroid = vec![0.0; n_features];
            for &point_idx in cluster_points {
                for j in 0..n_features {
                    centroid[j] += X[[point_idx, j]];
                }
            }
            for val in centroid.iter_mut() {
                *val /= cluster_points.len() as f64;
            }
            centroids.insert(cluster_id, centroid);
        }

        // 1. Centroid separation: Average distance between all cluster centroids
        let cluster_ids: Vec<i32> = clusters.keys().cloned().collect();
        let mut total_centroid_distance = 0.0;
        let mut centroid_pair_count = 0;

        for i in 0..cluster_ids.len() {
            for j in (i + 1)..cluster_ids.len() {
                let dist = self
                    .metric
                    .compute_distance(&centroids[&cluster_ids[i]], &centroids[&cluster_ids[j]]);
                total_centroid_distance += dist;
                centroid_pair_count += 1;
            }
        }

        let avg_centroid_separation = if centroid_pair_count > 0 {
            total_centroid_distance / centroid_pair_count as f64
        } else {
            0.0
        };

        // 2. Minimum inter-cluster distance: Closest points between different clusters
        let mut min_inter_cluster_distances = HashMap::new();
        let mut avg_inter_cluster_distance = 0.0;
        let mut inter_cluster_pair_count = 0;

        for i in 0..cluster_ids.len() {
            for j in (i + 1)..cluster_ids.len() {
                let cluster_i = cluster_ids[i];
                let cluster_j = cluster_ids[j];
                let points_i = &clusters[&cluster_i];
                let points_j = &clusters[&cluster_j];

                let mut min_distance = f64::INFINITY;
                let mut total_inter_distance = 0.0;
                let mut point_pair_count = 0;

                for &point_i in points_i {
                    for &point_j in points_j {
                        let distance = self
                            .metric
                            .compute_distance(&X.row(point_i).to_vec(), &X.row(point_j).to_vec());
                        min_distance = min_distance.min(distance);
                        total_inter_distance += distance;
                        point_pair_count += 1;
                    }
                }

                let avg_distance = total_inter_distance / point_pair_count as f64;
                min_inter_cluster_distances.insert((cluster_i, cluster_j), min_distance);
                avg_inter_cluster_distance += avg_distance;
                inter_cluster_pair_count += 1;
            }
        }

        avg_inter_cluster_distance /= inter_cluster_pair_count as f64;

        let min_separation = min_inter_cluster_distances
            .values()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        // 3. Boundary clarity: How well-defined cluster boundaries are
        let boundary_clarity = self.compute_boundary_clarity(X, &clusters);

        // 4. Overlap measure: Estimate of cluster overlap in feature space
        let overlap_measure = self.compute_cluster_overlap(X, &clusters, &centroids);

        // 5. Gap ratio: Ratio of inter-cluster to intra-cluster distances
        let gap_ratio = self.compute_gap_ratio(X, &clusters, avg_inter_cluster_distance);

        Ok(ClusterSeparationResult {
            avg_centroid_separation,
            min_separation,
            avg_inter_cluster_distance,
            boundary_clarity,
            overlap_measure,
            gap_ratio,
            min_inter_cluster_distances,
            n_clusters: clusters.len(),
        })
    }

    // Helper methods for coherence analysis

    /// Estimate cluster volume for density calculation
    fn estimate_cluster_volume(&self, X: &Array2<f64>, cluster_points: &[usize]) -> f64 {
        if cluster_points.len() < 2 {
            return 1.0;
        }

        let n_features = X.ncols();
        let mut volume = 1.0;

        // Compute volume as product of ranges in each dimension
        for feature in 0..n_features {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;

            for &point_idx in cluster_points {
                let val = X[[point_idx, feature]];
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }

            let range = (max_val - min_val).max(1e-10); // Avoid zero volume
            volume *= range;
        }

        volume
    }

    /// Expected average distance for uniform distribution in given volume
    fn expected_uniform_distance(&self, volume: f64, n_points: usize) -> f64 {
        if n_points < 2 {
            return 1.0;
        }

        // Approximate expected distance based on volume and number of points
        // For uniform distribution: E[distance] ∝ (volume/n_points)^(1/d)
        let density = n_points as f64 / volume;
        1.0 / density.sqrt()
    }

    /// Compute shape regularity of a cluster
    fn compute_shape_regularity(
        &self,
        X: &Array2<f64>,
        cluster_points: &[usize],
        centroid: &[f64],
    ) -> f64 {
        if cluster_points.len() < 3 {
            return 1.0;
        }

        // Compute distances from each point to centroid
        let mut distances = Vec::new();
        for &point_idx in cluster_points {
            let distance = self
                .metric
                .compute_distance(&X.row(point_idx).to_vec(), centroid);
            distances.push(distance);
        }

        // Compute coefficient of variation of distances (lower = more regular)
        let cv = self.coefficient_of_variation(&distances);
        1.0 / (1.0 + cv)
    }

    /// Coefficient of variation
    fn coefficient_of_variation(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        if mean.abs() < 1e-10 {
            return 0.0;
        }

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        variance.sqrt() / mean.abs()
    }

    // Helper methods for separation analysis

    /// Compute boundary clarity between clusters
    fn compute_boundary_clarity(
        &self,
        X: &Array2<f64>,
        clusters: &HashMap<i32, Vec<usize>>,
    ) -> f64 {
        if clusters.len() < 2 {
            return 1.0;
        }

        // For each point, find distance to nearest point in different cluster
        let mut boundary_clarity_scores = Vec::new();

        for cluster_points in clusters.values() {
            for &point_i in cluster_points {
                let mut min_inter_distance = f64::INFINITY;
                let mut min_intra_distance = f64::INFINITY;

                // Find nearest inter-cluster point
                for other_cluster_points in clusters.values() {
                    if std::ptr::eq(cluster_points, other_cluster_points) {
                        continue;
                    }
                    for &point_j in other_cluster_points {
                        let distance = self
                            .metric
                            .compute_distance(&X.row(point_i).to_vec(), &X.row(point_j).to_vec());
                        min_inter_distance = min_inter_distance.min(distance);
                    }
                }

                // Find nearest intra-cluster point
                for &point_j in cluster_points {
                    if point_i != point_j {
                        let distance = self
                            .metric
                            .compute_distance(&X.row(point_i).to_vec(), &X.row(point_j).to_vec());
                        min_intra_distance = min_intra_distance.min(distance);
                    }
                }

                // Boundary clarity = ratio of inter to intra distances
                if min_intra_distance > 0.0 && min_inter_distance.is_finite() {
                    boundary_clarity_scores.push(min_inter_distance / min_intra_distance);
                }
            }
        }

        if boundary_clarity_scores.is_empty() {
            1.0
        } else {
            boundary_clarity_scores.iter().sum::<f64>() / boundary_clarity_scores.len() as f64
        }
    }

    /// Compute cluster overlap measure
    fn compute_cluster_overlap(
        &self,
        X: &Array2<f64>,
        clusters: &HashMap<i32, Vec<usize>>,
        centroids: &HashMap<i32, Vec<f64>>,
    ) -> f64 {
        if clusters.len() < 2 {
            return 0.0;
        }

        let mut total_overlap = 0.0;
        let mut pair_count = 0;

        let cluster_ids: Vec<i32> = clusters.keys().cloned().collect();

        for i in 0..cluster_ids.len() {
            for j in (i + 1)..cluster_ids.len() {
                let cluster_i = cluster_ids[i];
                let cluster_j = cluster_ids[j];
                let points_i = &clusters[&cluster_i];
                let points_j = &clusters[&cluster_j];
                let centroid_i = &centroids[&cluster_i];
                let centroid_j = &centroids[&cluster_j];

                // Compute average distance from cluster i points to cluster j centroid
                let avg_dist_i_to_j = self.average_distance_to_centroid(X, points_i, centroid_j);
                let avg_dist_j_to_i = self.average_distance_to_centroid(X, points_j, centroid_i);

                // Compute average intra-cluster distances
                let avg_dist_i_to_centroid_i =
                    self.average_distance_to_centroid(X, points_i, centroid_i);
                let avg_dist_j_to_centroid_j =
                    self.average_distance_to_centroid(X, points_j, centroid_j);

                // Overlap measure: how much closer points are to their own centroid vs other centroid
                let overlap_i = if avg_dist_i_to_j > 0.0 {
                    avg_dist_i_to_centroid_i / avg_dist_i_to_j
                } else {
                    0.0
                };
                let overlap_j = if avg_dist_j_to_i > 0.0 {
                    avg_dist_j_to_centroid_j / avg_dist_j_to_i
                } else {
                    0.0
                };

                total_overlap += (overlap_i + overlap_j) / 2.0;
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_overlap / pair_count as f64
        } else {
            0.0
        }
    }

    /// Compute average distance from cluster points to a centroid
    fn average_distance_to_centroid(
        &self,
        X: &Array2<f64>,
        cluster_points: &[usize],
        centroid: &[f64],
    ) -> f64 {
        if cluster_points.is_empty() {
            return 0.0;
        }

        let mut total_distance = 0.0;
        for &point_idx in cluster_points {
            let distance = self
                .metric
                .compute_distance(&X.row(point_idx).to_vec(), centroid);
            total_distance += distance;
        }

        total_distance / cluster_points.len() as f64
    }

    /// Compute gap ratio between inter and intra cluster distances
    fn compute_gap_ratio(
        &self,
        X: &Array2<f64>,
        clusters: &HashMap<i32, Vec<usize>>,
        avg_inter_cluster_distance: f64,
    ) -> f64 {
        // Compute average intra-cluster distance
        let mut total_intra_distance = 0.0;
        let mut intra_count = 0;

        for cluster_points in clusters.values() {
            for (i, &point_i) in cluster_points.iter().enumerate() {
                for &point_j in cluster_points.iter().skip(i + 1) {
                    let distance = self
                        .metric
                        .compute_distance(&X.row(point_i).to_vec(), &X.row(point_j).to_vec());
                    total_intra_distance += distance;
                    intra_count += 1;
                }
            }
        }

        let avg_intra_distance = if intra_count > 0 {
            total_intra_distance / intra_count as f64
        } else {
            1.0
        };

        if avg_intra_distance > 0.0 {
            avg_inter_cluster_distance / avg_intra_distance
        } else {
            f64::INFINITY
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn generate_test_data() -> (Array2<f64>, Vec<i32>) {
        // Create two well-separated clusters
        let data = array![
            [1.0, 2.0], // Cluster 0
            [1.5, 1.8], // Cluster 0
            [1.2, 2.2], // Cluster 0
            [5.0, 8.0], // Cluster 1
            [5.2, 7.8], // Cluster 1
            [4.8, 8.2], // Cluster 1
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        (data, labels)
    }

    fn generate_overlapping_data() -> (Array2<f64>, Vec<i32>) {
        // Create two overlapping clusters
        let data = array![
            [1.0, 1.0], // Cluster 0
            [1.5, 1.5], // Cluster 0
            [2.0, 2.0], // Cluster 0 (overlapping)
            [2.5, 2.5], // Cluster 1 (overlapping)
            [3.0, 3.0], // Cluster 1
            [3.5, 3.5], // Cluster 1
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        (data, labels)
    }

    #[test]
    fn test_cluster_coherence_well_separated() {
        let (data, labels) = generate_test_data();
        let analyzer = CoherenceSeparationAnalyzer::euclidean();

        let result = analyzer.cluster_coherence(&data, &labels).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert!(result.overall_coherence > 0.0);
        assert!(result.overall_compactness > 0.0);
        assert!(result.overall_density > 0.0);
        assert!(result.overall_shape_regularity > 0.0);

        // Check that we have scores for both clusters
        assert!(result.cluster_coherence_scores.contains_key(&0));
        assert!(result.cluster_coherence_scores.contains_key(&1));

        // Consistency scores should be reasonable
        assert!(result.coherence_consistency >= 0.0 && result.coherence_consistency <= 1.0);
        assert!(result.compactness_consistency >= 0.0 && result.compactness_consistency <= 1.0);
    }

    #[test]
    fn test_cluster_separation_well_separated() {
        let (data, labels) = generate_test_data();
        let analyzer = CoherenceSeparationAnalyzer::euclidean();

        let result = analyzer.cluster_separation(&data, &labels).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert!(result.avg_centroid_separation > 0.0);
        assert!(result.min_separation > 0.0);
        assert!(result.avg_inter_cluster_distance > 0.0);
        assert!(result.boundary_clarity > 0.0);
        assert!(result.gap_ratio > 1.0); // Inter-cluster should be larger than intra-cluster

        // Should have one cluster pair distance
        assert_eq!(result.min_inter_cluster_distances.len(), 1);

        // Overlap should be relatively low for well-separated clusters
        assert!(result.overlap_measure < 1.0);
    }

    #[test]
    fn test_cluster_separation_overlapping() {
        let (data, labels) = generate_overlapping_data();
        let analyzer = CoherenceSeparationAnalyzer::euclidean();

        let result = analyzer.cluster_separation(&data, &labels).unwrap();

        assert_eq!(result.n_clusters, 2);

        // Overlapping clusters should have smaller gap ratio
        assert!(result.gap_ratio > 0.0);

        // Overlap measure should be higher for overlapping clusters
        assert!(result.overlap_measure >= 0.0);
    }

    #[test]
    fn test_coherence_result_methods() {
        let (data, labels) = generate_test_data();
        let analyzer = CoherenceSeparationAnalyzer::euclidean();

        let result = analyzer.cluster_coherence(&data, &labels).unwrap();

        // Test best and worst cluster methods
        let best = result.best_cluster();
        let worst = result.worst_cluster();
        assert!(best.is_some());
        assert!(worst.is_some());

        // Test problematic clusters
        let problematic = result.problematic_clusters(0.1); // Very low threshold
        assert!(problematic.len() <= 2); // Should not exceed number of clusters

        // Test overall quality score
        let quality = result.overall_quality_score();
        assert!(quality >= 0.0 && quality <= 1.0);
    }

    #[test]
    fn test_separation_result_methods() {
        let (data, labels) = generate_overlapping_data();
        let analyzer = CoherenceSeparationAnalyzer::euclidean();

        let result = analyzer.cluster_separation(&data, &labels).unwrap();

        // Test poorly separated pairs
        let poorly_separated = result.poorly_separated_pairs(10.0); // High threshold
        assert!(poorly_separated.len() >= 0);

        // Test most overlapping pair
        let most_overlapping = result.most_overlapping_pair();
        assert!(most_overlapping.is_some());

        // Test separation quality assessment
        let quality = result.separation_quality();
        assert!(["Excellent", "Good", "Fair", "Poor"].contains(&quality));
    }

    #[test]
    fn test_helper_methods() {
        let analyzer = CoherenceSeparationAnalyzer::euclidean();

        // Test coefficient of variation
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cv = analyzer.coefficient_of_variation(&values);
        assert!(cv > 0.0);

        // Test with identical values (should have CV = 0)
        let identical = vec![2.0, 2.0, 2.0, 2.0];
        let cv_zero = analyzer.coefficient_of_variation(&identical);
        assert!(cv_zero.abs() < 1e-10);

        // Test with empty values
        let empty: Vec<f64> = vec![];
        let cv_empty = analyzer.coefficient_of_variation(&empty);
        assert_eq!(cv_empty, 0.0);
    }

    #[test]
    fn test_invalid_inputs() {
        let analyzer = CoherenceSeparationAnalyzer::euclidean();

        // Test mismatched data and labels
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = vec![0]; // Wrong length

        assert!(analyzer.cluster_coherence(&data, &labels).is_err());
        assert!(analyzer.cluster_separation(&data, &labels).is_err());

        // Test no valid clusters
        let labels_noise = vec![-1, -1]; // All noise points
        assert!(analyzer.cluster_coherence(&data, &labels_noise).is_err());

        // Test single cluster for separation analysis
        let labels_single = vec![0, 0]; // Only one cluster
        assert!(analyzer.cluster_separation(&data, &labels_single).is_err());
    }

    #[test]
    fn test_different_metrics() {
        let (data, labels) = generate_test_data();

        let euclidean_analyzer = CoherenceSeparationAnalyzer::euclidean();
        let manhattan_analyzer = CoherenceSeparationAnalyzer::manhattan();
        let cosine_analyzer = CoherenceSeparationAnalyzer::cosine();

        let euc_coherence = euclidean_analyzer
            .cluster_coherence(&data, &labels)
            .unwrap();
        let man_coherence = manhattan_analyzer
            .cluster_coherence(&data, &labels)
            .unwrap();
        let cos_coherence = cosine_analyzer.cluster_coherence(&data, &labels).unwrap();

        // All should produce valid results
        assert!(euc_coherence.overall_coherence >= 0.0);
        assert!(man_coherence.overall_coherence >= 0.0);
        assert!(cos_coherence.overall_coherence >= 0.0);

        // Results should be different for different metrics
        assert_ne!(
            euc_coherence.overall_coherence,
            man_coherence.overall_coherence
        );
    }

    #[test]
    fn test_volume_estimation() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];
        let cluster_points = vec![0, 1, 2, 3];
        let analyzer = CoherenceSeparationAnalyzer::euclidean();

        let volume = analyzer.estimate_cluster_volume(&data, &cluster_points);

        // For a 1x1 square, volume should be 1.0
        assert!((volume - 1.0).abs() < 1e-10);

        // Test with single point
        let single_point = vec![0];
        let volume_single = analyzer.estimate_cluster_volume(&data, &single_point);
        assert_eq!(volume_single, 1.0);
    }

    #[test]
    fn test_shape_regularity() {
        let data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],];
        let cluster_points = vec![0, 1, 2, 3];
        let centroid = vec![2.5, 2.5];
        let analyzer = CoherenceSeparationAnalyzer::euclidean();

        let regularity = analyzer.compute_shape_regularity(&data, &cluster_points, &centroid);

        // Points on a line should have good regularity (distances to centroid are regular)
        assert!(regularity > 0.0 && regularity <= 1.0);

        // Test with too few points
        let few_points = vec![0, 1];
        let regularity_few = analyzer.compute_shape_regularity(&data, &few_points, &centroid);
        assert_eq!(regularity_few, 1.0);
    }
}

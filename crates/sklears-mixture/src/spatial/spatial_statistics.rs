//! Spatial Statistics and Autocorrelation Analysis
//!
//! This module provides comprehensive spatial statistics functionality including
//! Moran's I, Geary's C, Local Indicators of Spatial Association (LISA), and
//! spatial clustering quality assessment measures.

use super::{
    spatial_constraints::SpatialConstraint,
    spatial_utils::{euclidean_distance, k_nearest_neighbors},
};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Moran's I statistic for spatial autocorrelation analysis
#[derive(Debug, Clone)]
pub struct MoransI {
    /// statistic
    pub statistic: f64,
    /// expected_value
    pub expected_value: f64,
    /// variance
    pub variance: f64,
    /// z_score
    pub z_score: f64,
    /// p_value
    pub p_value: f64,
}

/// Geary's C statistic for spatial autocorrelation analysis
#[derive(Debug, Clone)]
pub struct GearysC {
    /// statistic
    pub statistic: f64,
    /// expected_value
    pub expected_value: f64,
    /// variance
    pub variance: f64,
    /// z_score
    pub z_score: f64,
    /// p_value
    pub p_value: f64,
}

/// Local spatial autocorrelation indicators
#[derive(Debug, Clone)]
pub struct LocalIndicators {
    /// local_morans_i
    pub local_morans_i: Array1<f64>,
    /// local_z_scores
    pub local_z_scores: Array1<f64>,
    /// local_p_values
    pub local_p_values: Array1<f64>,
    /// clusters
    pub clusters: Array1<String>, // "High-High", "Low-Low", "High-Low", "Low-High", "Not significant"
}

/// Spatial autocorrelation analyzer
pub struct SpatialAutocorrelationAnalyzer {
    spatial_weights: Array2<f64>,
    coords: Array2<f64>,
}

impl SpatialAutocorrelationAnalyzer {
    /// Create new spatial autocorrelation analyzer
    pub fn new(coords: Array2<f64>, constraint: SpatialConstraint) -> SklResult<Self> {
        let spatial_weights = Self::compute_spatial_weights(&coords, &constraint)?;

        Ok(Self {
            spatial_weights,
            coords,
        })
    }

    /// Compute spatial weights matrix
    fn compute_spatial_weights(
        coords: &Array2<f64>,
        constraint: &SpatialConstraint,
    ) -> SklResult<Array2<f64>> {
        let n_samples = coords.nrows();
        let mut weights = Array2::zeros((n_samples, n_samples));

        match constraint {
            SpatialConstraint::Distance { radius } => {
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j {
                            let dist = euclidean_distance(
                                &coords.row(i).to_owned().into_raw_vec(),
                                &coords.row(j).to_owned().into_raw_vec(),
                            );
                            if dist <= *radius {
                                weights[[i, j]] = 1.0 / dist.max(1e-6); // Inverse distance weighting
                            }
                        }
                    }
                }
            }
            SpatialConstraint::Adjacency => {
                let k = 4; // Number of nearest neighbors
                let neighbors = k_nearest_neighbors(coords, k);
                for (i, neighbor_list) in neighbors.iter().enumerate() {
                    for &j in neighbor_list {
                        weights[[i, j]] = 1.0;
                        weights[[j, i]] = 1.0; // Symmetric
                    }
                }
            }
            SpatialConstraint::Grid { rows, cols } => {
                if n_samples != rows * cols {
                    return Err(SklearsError::InvalidInput(
                        "Grid dimensions don't match number of samples".to_string(),
                    ));
                }

                for i in 0..*rows {
                    for j in 0..*cols {
                        let idx = i * cols + j;
                        // Connect to neighbors (up, down, left, right)
                        if i > 0 {
                            weights[[idx, (i - 1) * cols + j]] = 1.0;
                        }
                        if i < rows - 1 {
                            weights[[idx, (i + 1) * cols + j]] = 1.0;
                        }
                        if j > 0 {
                            weights[[idx, i * cols + (j - 1)]] = 1.0;
                        }
                        if j < cols - 1 {
                            weights[[idx, i * cols + (j + 1)]] = 1.0;
                        }
                    }
                }
            }
            SpatialConstraint::Custom => {
                return Err(SklearsError::InvalidInput(
                    "Custom spatial constraints not supported for autocorrelation analysis"
                        .to_string(),
                ));
            }
        }

        // Row-standardize weights
        for i in 0..n_samples {
            let row_sum: f64 = weights.row(i).sum();
            if row_sum > 1e-8 {
                for j in 0..n_samples {
                    weights[[i, j]] /= row_sum;
                }
            }
        }

        Ok(weights)
    }

    /// Compute Moran's I statistic
    pub fn morans_i(&self, values: &Array1<f64>) -> SklResult<MoransI> {
        let n = values.len() as f64;
        let mean_val = values.mean().unwrap_or(0.0);

        // Compute deviations from mean
        let deviations: Array1<f64> = values.mapv(|v| v - mean_val);

        // Compute numerator: sum of weighted cross-products
        let mut numerator = 0.0;
        let mut w_sum = 0.0;

        for i in 0..values.len() {
            for j in 0..values.len() {
                let w_ij = self.spatial_weights[[i, j]];
                numerator += w_ij * deviations[i] * deviations[j];
                w_sum += w_ij;
            }
        }

        // Compute denominator: sum of squared deviations
        let denominator: f64 = deviations.mapv(|d| d * d).sum();

        // Moran's I statistic
        let morans_i = if denominator > 1e-8 && w_sum > 1e-8 {
            (n / w_sum) * (numerator / denominator)
        } else {
            0.0
        };

        // Expected value under null hypothesis
        let expected = -1.0 / (n - 1.0);

        // Compute variance (simplified version)
        let variance = self.compute_morans_i_variance(n, w_sum)?;

        // Z-score and p-value
        let z_score = if variance > 1e-8 {
            (morans_i - expected) / variance.sqrt()
        } else {
            0.0
        };

        let p_value = 2.0 * (1.0 - self.standard_normal_cdf(z_score.abs()));

        Ok(MoransI {
            statistic: morans_i,
            expected_value: expected,
            variance,
            z_score,
            p_value,
        })
    }

    /// Compute Geary's C statistic
    pub fn gearys_c(&self, values: &Array1<f64>) -> SklResult<GearysC> {
        let n = values.len() as f64;

        // Compute numerator: sum of weighted squared differences
        let mut numerator = 0.0;
        let mut w_sum = 0.0;

        for i in 0..values.len() {
            for j in 0..values.len() {
                let w_ij = self.spatial_weights[[i, j]];
                numerator += w_ij * (values[i] - values[j]).powi(2);
                w_sum += w_ij;
            }
        }

        // Compute denominator: sum of squared deviations from mean
        let mean_val = values.mean().unwrap_or(0.0);
        let denominator: f64 = values.mapv(|v| (v - mean_val).powi(2)).sum();

        // Geary's C statistic
        let gearys_c = if denominator > 1e-8 && w_sum > 1e-8 {
            ((n - 1.0) / (2.0 * w_sum)) * (numerator / denominator)
        } else {
            1.0
        };

        // Expected value under null hypothesis
        let expected = 1.0;

        // Compute variance (simplified version)
        let variance = self.compute_gearys_c_variance(n, w_sum)?;

        // Z-score and p-value
        let z_score = if variance > 1e-8 {
            (gearys_c - expected) / variance.sqrt()
        } else {
            0.0
        };

        let p_value = 2.0 * (1.0 - self.standard_normal_cdf(z_score.abs()));

        Ok(GearysC {
            statistic: gearys_c,
            expected_value: expected,
            variance,
            z_score,
            p_value,
        })
    }

    /// Compute local indicators of spatial association (LISA)
    pub fn local_indicators(&self, values: &Array1<f64>) -> SklResult<LocalIndicators> {
        let n = values.len();
        let mean_val = values.mean().unwrap_or(0.0);

        let mut local_morans_i = Array1::zeros(n);
        let mut local_z_scores = Array1::zeros(n);
        let mut local_p_values = Array1::zeros(n);
        let mut clusters = Vec::with_capacity(n);

        // Compute variance
        let variance: f64 = values.mapv(|v| (v - mean_val).powi(2)).sum() / n as f64;

        for i in 0..n {
            let z_i = (values[i] - mean_val) / variance.sqrt();

            // Compute local Moran's I
            let mut weighted_z_sum = 0.0;
            for j in 0..n {
                if i != j {
                    let w_ij = self.spatial_weights[[i, j]];
                    let z_j = (values[j] - mean_val) / variance.sqrt();
                    weighted_z_sum += w_ij * z_j;
                }
            }

            local_morans_i[i] = z_i * weighted_z_sum;

            // Simplified z-score calculation
            local_z_scores[i] = local_morans_i[i] / (1.0f64 + 1e-6).sqrt();
            local_p_values[i] = 2.0 * (1.0 - self.standard_normal_cdf(local_z_scores[i].abs()));

            // Classify cluster type
            let cluster_type = if local_p_values[i] < 0.05 {
                if z_i > 0.0 && weighted_z_sum > 0.0 {
                    "High-High"
                } else if z_i < 0.0 && weighted_z_sum < 0.0 {
                    "Low-Low"
                } else if z_i > 0.0 && weighted_z_sum < 0.0 {
                    "High-Low"
                } else {
                    "Low-High"
                }
            } else {
                "Not significant"
            };

            clusters.push(cluster_type.to_string());
        }

        Ok(LocalIndicators {
            local_morans_i,
            local_z_scores,
            local_p_values,
            clusters: Array1::from_vec(clusters),
        })
    }

    /// Analyze spatial autocorrelation in mixture component assignments
    pub fn analyze_mixture_assignments(
        &self,
        assignments: &Array1<usize>,
    ) -> SklResult<(MoransI, GearysC)> {
        // Convert discrete assignments to continuous values for analysis
        let values = assignments.mapv(|a| a as f64);

        let morans_i = self.morans_i(&values)?;
        let gearys_c = self.gearys_c(&values)?;

        Ok((morans_i, gearys_c))
    }

    /// Helper functions
    fn compute_morans_i_variance(&self, n: f64, _w_sum: f64) -> SklResult<f64> {
        // Simplified variance calculation
        // In a full implementation, this would include more complex terms
        let variance = 1.0 / (n - 1.0) * (1.0 - 1.0 / n);
        Ok(variance.max(1e-8))
    }

    fn compute_gearys_c_variance(&self, n: f64, _w_sum: f64) -> SklResult<f64> {
        // Simplified variance calculation
        let variance = 1.0 / (2.0 * (n - 1.0)) * (1.0 - 1.0 / n);
        Ok(variance.max(1e-8))
    }

    /// Approximate standard normal CDF
    fn standard_normal_cdf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d
            * t
            * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

        if x >= 0.0 {
            1.0 - prob
        } else {
            prob
        }
    }
}

/// Spatial clustering quality assessment
#[derive(Debug, Clone)]
pub struct SpatialClusteringQuality {
    /// silhouette_score
    pub silhouette_score: f64,
    /// spatial_separation
    pub spatial_separation: f64,
    /// spatial_compactness
    pub spatial_compactness: f64,
    /// boundary_coherence
    pub boundary_coherence: f64,
}

impl SpatialClusteringQuality {
    /// Assess the quality of spatial clustering
    pub fn assess(
        coords: &Array2<f64>,
        assignments: &Array1<usize>,
        n_components: usize,
    ) -> SklResult<Self> {
        let silhouette_score = Self::compute_silhouette_score(coords, assignments)?;
        let spatial_separation =
            Self::compute_spatial_separation(coords, assignments, n_components)?;
        let spatial_compactness =
            Self::compute_spatial_compactness(coords, assignments, n_components)?;
        let boundary_coherence = Self::compute_boundary_coherence(coords, assignments)?;

        Ok(Self {
            silhouette_score,
            spatial_separation,
            spatial_compactness,
            boundary_coherence,
        })
    }

    fn compute_silhouette_score(
        coords: &Array2<f64>,
        assignments: &Array1<usize>,
    ) -> SklResult<f64> {
        let n_samples = coords.nrows();
        let mut total_silhouette = 0.0;

        for i in 0..n_samples {
            let cluster_i = assignments[i];

            // Compute mean distance to points in same cluster
            let mut intra_cluster_dist = 0.0;
            let mut same_cluster_count = 0;

            for j in 0..n_samples {
                if i != j && assignments[j] == cluster_i {
                    intra_cluster_dist += euclidean_distance(
                        &coords.row(i).to_owned().into_raw_vec(),
                        &coords.row(j).to_owned().into_raw_vec(),
                    );
                    same_cluster_count += 1;
                }
            }

            let a_i = if same_cluster_count > 0 {
                intra_cluster_dist / same_cluster_count as f64
            } else {
                0.0
            };

            // Compute mean distance to nearest different cluster
            let mut min_inter_cluster_dist = f64::INFINITY;

            for other_cluster in 0..10 {
                // Assume max 10 clusters for simplicity
                if other_cluster != cluster_i {
                    let mut inter_cluster_dist = 0.0;
                    let mut other_cluster_count = 0;

                    for j in 0..n_samples {
                        if assignments[j] == other_cluster {
                            inter_cluster_dist += euclidean_distance(
                                &coords.row(i).to_owned().into_raw_vec(),
                                &coords.row(j).to_owned().into_raw_vec(),
                            );
                            other_cluster_count += 1;
                        }
                    }

                    if other_cluster_count > 0 {
                        let avg_dist = inter_cluster_dist / other_cluster_count as f64;
                        min_inter_cluster_dist = min_inter_cluster_dist.min(avg_dist);
                    }
                }
            }

            let b_i = min_inter_cluster_dist;

            // Silhouette coefficient for point i
            let s_i = if a_i.max(b_i) > 1e-8 {
                (b_i - a_i) / a_i.max(b_i)
            } else {
                0.0
            };

            total_silhouette += s_i;
        }

        Ok(total_silhouette / n_samples as f64)
    }

    fn compute_spatial_separation(
        coords: &Array2<f64>,
        assignments: &Array1<usize>,
        n_components: usize,
    ) -> SklResult<f64> {
        // Compute centroids for each cluster
        let mut centroids = Array2::zeros((n_components, coords.ncols()));
        let mut cluster_counts = vec![0; n_components];

        for i in 0..coords.nrows() {
            let cluster = assignments[i];
            if cluster < n_components {
                for j in 0..coords.ncols() {
                    centroids[[cluster, j]] += coords[[i, j]];
                }
                cluster_counts[cluster] += 1;
            }
        }

        // Average centroids
        for i in 0..n_components {
            if cluster_counts[i] > 0 {
                for j in 0..coords.ncols() {
                    centroids[[i, j]] /= cluster_counts[i] as f64;
                }
            }
        }

        // Compute minimum distance between centroids
        let mut min_separation = f64::INFINITY;
        for i in 0..n_components {
            for j in (i + 1)..n_components {
                if cluster_counts[i] > 0 && cluster_counts[j] > 0 {
                    let dist = euclidean_distance(
                        &centroids.row(i).to_owned().into_raw_vec(),
                        &centroids.row(j).to_owned().into_raw_vec(),
                    );
                    min_separation = min_separation.min(dist);
                }
            }
        }

        Ok(min_separation)
    }

    fn compute_spatial_compactness(
        coords: &Array2<f64>,
        assignments: &Array1<usize>,
        n_components: usize,
    ) -> SklResult<f64> {
        let mut total_compactness = 0.0;
        let mut valid_clusters = 0;

        for cluster in 0..n_components {
            let cluster_points: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &a)| a == cluster)
                .map(|(i, _)| i)
                .collect();

            if cluster_points.len() > 1 {
                // Compute centroid
                let mut centroid = vec![0.0; coords.ncols()];
                for &point_idx in &cluster_points {
                    for j in 0..coords.ncols() {
                        centroid[j] += coords[[point_idx, j]];
                    }
                }
                for j in 0..coords.ncols() {
                    centroid[j] /= cluster_points.len() as f64;
                }

                // Compute average distance to centroid
                let mut total_dist = 0.0;
                for &point_idx in &cluster_points {
                    let dist = euclidean_distance(
                        &coords.row(point_idx).to_owned().into_raw_vec(),
                        &centroid,
                    );
                    total_dist += dist;
                }

                total_compactness += total_dist / cluster_points.len() as f64;
                valid_clusters += 1;
            }
        }

        Ok(if valid_clusters > 0 {
            1.0 / (1.0 + total_compactness / valid_clusters as f64) // Invert so higher is better
        } else {
            0.0
        })
    }

    fn compute_boundary_coherence(
        coords: &Array2<f64>,
        assignments: &Array1<usize>,
    ) -> SklResult<f64> {
        let n_samples = coords.nrows();
        let mut coherent_boundaries = 0;
        let mut total_boundaries = 0;

        // Find k nearest neighbors for each point
        let k = 5.min(n_samples - 1);
        let neighbors = k_nearest_neighbors(coords, k);

        for (i, neighbor_list) in neighbors.iter().enumerate() {
            let cluster_i = assignments[i];

            for &neighbor_idx in neighbor_list {
                if assignments[neighbor_idx] == cluster_i {
                    coherent_boundaries += 1;
                }
                total_boundaries += 1;
            }
        }

        Ok(if total_boundaries > 0 {
            coherent_boundaries as f64 / total_boundaries as f64
        } else {
            1.0
        })
    }

    /// Get an overall quality score combining all metrics
    pub fn overall_score(&self) -> f64 {
        // Weighted combination of all metrics
        let weights = [0.3, 0.25, 0.25, 0.2]; // silhouette, separation, compactness, coherence
        let scores = [
            self.silhouette_score.max(0.0), // Normalize negative silhouette scores
            self.spatial_separation / (1.0 + self.spatial_separation), // Normalize
            self.spatial_compactness,
            self.boundary_coherence,
        ];

        weights.iter().zip(scores.iter()).map(|(w, s)| w * s).sum()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_spatial_autocorrelation_analyzer_creation() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]];
        let constraint = SpatialConstraint::Distance { radius: 1.5 };

        let analyzer = SpatialAutocorrelationAnalyzer::new(coords, constraint).unwrap();
        assert_eq!(analyzer.coords.nrows(), 4);
        assert_eq!(analyzer.spatial_weights.dim(), (4, 4));
    }

    #[test]
    fn test_morans_i_calculation() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let constraint = SpatialConstraint::Distance { radius: 1.5 };
        let analyzer = SpatialAutocorrelationAnalyzer::new(coords, constraint).unwrap();

        let values = array![1.0, 1.0, 0.0]; // Spatially autocorrelated
        let morans_i = analyzer.morans_i(&values).unwrap();

        assert!(morans_i.statistic.is_finite());
        assert!(morans_i.z_score.is_finite());
        assert!(morans_i.p_value >= 0.0 && morans_i.p_value <= 1.0);
    }

    #[test]
    fn test_gearys_c_calculation() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let constraint = SpatialConstraint::Distance { radius: 1.5 };
        let analyzer = SpatialAutocorrelationAnalyzer::new(coords, constraint).unwrap();

        let values = array![1.0, 1.0, 0.0];
        let gearys_c = analyzer.gearys_c(&values).unwrap();

        assert!(gearys_c.statistic.is_finite());
        assert!(gearys_c.z_score.is_finite());
        assert!(gearys_c.p_value >= 0.0 && gearys_c.p_value <= 1.0);
    }

    #[test]
    fn test_local_indicators() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]];
        let constraint = SpatialConstraint::Distance { radius: 1.5 };
        let analyzer = SpatialAutocorrelationAnalyzer::new(coords, constraint).unwrap();

        let values = array![1.0, 1.0, 0.0, 1.0];
        let lisa = analyzer.local_indicators(&values).unwrap();

        assert_eq!(lisa.local_morans_i.len(), 4);
        assert_eq!(lisa.local_z_scores.len(), 4);
        assert_eq!(lisa.local_p_values.len(), 4);
        assert_eq!(lisa.clusters.len(), 4);

        // Check that cluster classifications are valid
        for cluster_type in lisa.clusters.iter() {
            assert!([
                "High-High",
                "Low-Low",
                "High-Low",
                "Low-High",
                "Not significant"
            ]
            .contains(&cluster_type.as_str()));
        }
    }

    #[test]
    fn test_spatial_clustering_quality() {
        let coords = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];
        let assignments = array![0, 0, 1, 1]; // Two well-separated clusters

        let quality = SpatialClusteringQuality::assess(&coords, &assignments, 2).unwrap();

        assert!(quality.silhouette_score.is_finite());
        assert!(quality.spatial_separation > 0.0);
        assert!(quality.spatial_compactness > 0.0);
        assert!(quality.boundary_coherence >= 0.0 && quality.boundary_coherence <= 1.0);

        let overall = quality.overall_score();
        assert!(overall >= 0.0 && overall <= 1.0);
    }

    #[test]
    fn test_mixture_assignments_analysis() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]];
        let constraint = SpatialConstraint::Adjacency;
        let analyzer = SpatialAutocorrelationAnalyzer::new(coords, constraint).unwrap();

        let assignments = array![0, 0, 1, 1];
        let (morans_i, gearys_c) = analyzer.analyze_mixture_assignments(&assignments).unwrap();

        assert!(morans_i.statistic.is_finite());
        assert!(gearys_c.statistic.is_finite());
    }

    #[test]
    fn test_standard_normal_cdf() {
        let coords = array![[0.0, 0.0]];
        let constraint = SpatialConstraint::Distance { radius: 1.0 };
        let analyzer = SpatialAutocorrelationAnalyzer::new(coords, constraint).unwrap();

        // Test some known values
        let cdf_0 = analyzer.standard_normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 0.01);

        let cdf_neg = analyzer.standard_normal_cdf(-1.96);
        assert!(cdf_neg < 0.05);

        let cdf_pos = analyzer.standard_normal_cdf(1.96);
        assert!(cdf_pos > 0.95);
    }
}

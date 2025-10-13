//! Spatial cross-validation for geographic and spatial data
//!
//! This module provides cross-validation methods that account for spatial
//! autocorrelation and geographic dependencies in spatial datasets.

use numrs2::prelude::*;
use scirs2_core::random::Rng;
use sklears_core::error::{Result, SklearsError};
use std::collections::HashSet;

/// Spatial coordinates for geographic data
#[derive(Debug, Clone, Copy)]
pub struct SpatialCoordinate {
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>, // For 3D spatial data
}

impl SpatialCoordinate {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y, z: None }
    }

    pub fn new_3d(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z: Some(z) }
    }

    /// Calculate Euclidean distance between two coordinates
    pub fn distance(&self, other: &SpatialCoordinate) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = match (self.z, other.z) {
            (Some(z1), Some(z2)) => z1 - z2,
            _ => 0.0,
        };
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate Haversine distance for lat/lon coordinates (in kilometers)
    pub fn haversine_distance(&self, other: &SpatialCoordinate) -> f64 {
        const EARTH_RADIUS_KM: f64 = 6371.0;

        let lat1 = self.y.to_radians();
        let lat2 = other.y.to_radians();
        let delta_lat = (other.y - self.y).to_radians();
        let delta_lon = (other.x - self.x).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        EARTH_RADIUS_KM * c
    }
}

/// Configuration for spatial cross-validation
#[derive(Debug, Clone)]
pub struct SpatialValidationConfig {
    /// Number of folds for cross-validation
    pub n_splits: usize,
    /// Minimum distance between training and test samples
    pub buffer_distance: f64,
    /// Method for distance calculation
    pub distance_method: DistanceMethod,
    /// Clustering method for spatial grouping
    pub clustering_method: SpatialClusteringMethod,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Whether to use geographic coordinates (lat/lon)
    pub geographic: bool,
}

impl Default for SpatialValidationConfig {
    fn default() -> Self {
        Self {
            n_splits: 5,
            buffer_distance: 1000.0, // 1km default for geographic data
            distance_method: DistanceMethod::Euclidean,
            clustering_method: SpatialClusteringMethod::KMeans,
            random_state: None,
            geographic: false,
        }
    }
}

/// Distance calculation methods
#[derive(Debug, Clone)]
pub enum DistanceMethod {
    /// Euclidean
    Euclidean,
    /// Haversine
    Haversine, // For lat/lon coordinates
    /// Manhattan
    Manhattan,
    /// Chebyshev
    Chebyshev,
}

/// Spatial clustering methods for grouping
#[derive(Debug, Clone)]
pub enum SpatialClusteringMethod {
    /// KMeans
    KMeans,
    /// Grid
    Grid,
    /// Hierarchical
    Hierarchical,
    /// DBSCAN
    DBSCAN,
}

/// Spatial cross-validator that accounts for spatial autocorrelation
#[derive(Debug, Clone)]
pub struct SpatialCrossValidator {
    config: SpatialValidationConfig,
}

impl SpatialCrossValidator {
    pub fn new(config: SpatialValidationConfig) -> Self {
        Self { config }
    }

    /// Generate spatial cross-validation splits
    pub fn split(
        &self,
        n_samples: usize,
        coordinates: &[SpatialCoordinate],
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if coordinates.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of coordinates must match number of samples".to_string(),
            ));
        }

        // Create spatial clusters
        let clusters = self.create_spatial_clusters(coordinates)?;

        // Generate cross-validation splits based on clusters
        let splits = self.generate_cluster_splits(&clusters)?;

        // Apply buffer constraints
        let filtered_splits = self.apply_buffer_constraints(&splits, coordinates)?;

        Ok(filtered_splits)
    }

    /// Create spatial clusters using the specified method
    fn create_spatial_clusters(&self, coordinates: &[SpatialCoordinate]) -> Result<Vec<usize>> {
        match self.config.clustering_method {
            SpatialClusteringMethod::KMeans => self.kmeans_clustering(coordinates),
            SpatialClusteringMethod::Grid => self.grid_clustering(coordinates),
            SpatialClusteringMethod::Hierarchical => self.hierarchical_clustering(coordinates),
            SpatialClusteringMethod::DBSCAN => self.dbscan_clustering(coordinates),
        }
    }

    /// K-means clustering for spatial grouping
    fn kmeans_clustering(&self, coordinates: &[SpatialCoordinate]) -> Result<Vec<usize>> {
        let n_samples = coordinates.len();
        let mut clusters = vec![0; n_samples];
        let mut centroids = Vec::new();

        // Initialize centroids randomly
        let mut rng = self.get_rng();
        for i in 0..self.config.n_splits {
            let idx = rng.gen_range(0..n_samples);
            centroids.push(coordinates[idx]);
        }

        // K-means iterations
        for _ in 0..100 {
            // Max iterations
            let mut new_centroids = vec![SpatialCoordinate::new(0.0, 0.0); self.config.n_splits];
            let mut cluster_counts = vec![0; self.config.n_splits];
            let mut changed = false;

            // Assign points to nearest centroid
            for (i, coord) in coordinates.iter().enumerate() {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let distance = self.calculate_distance(coord, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = j;
                    }
                }

                if clusters[i] != best_cluster {
                    changed = true;
                    clusters[i] = best_cluster;
                }

                // Update centroid sum
                new_centroids[best_cluster].x += coord.x;
                new_centroids[best_cluster].y += coord.y;
                if let Some(z) = coord.z {
                    if new_centroids[best_cluster].z.is_none() {
                        new_centroids[best_cluster].z = Some(0.0);
                    }
                    new_centroids[best_cluster].z =
                        Some(new_centroids[best_cluster].z.unwrap() + z);
                }
                cluster_counts[best_cluster] += 1;
            }

            // Update centroids
            for (i, count) in cluster_counts.iter().enumerate() {
                if *count > 0 {
                    new_centroids[i].x /= *count as f64;
                    new_centroids[i].y /= *count as f64;
                    if let Some(z) = new_centroids[i].z {
                        new_centroids[i].z = Some(z / *count as f64);
                    }
                }
            }

            centroids = new_centroids;

            if !changed {
                break;
            }
        }

        Ok(clusters)
    }

    /// Grid-based clustering for regular spatial division
    fn grid_clustering(&self, coordinates: &[SpatialCoordinate]) -> Result<Vec<usize>> {
        // Find bounds
        let min_x = coordinates
            .iter()
            .map(|c| c.x)
            .fold(f64::INFINITY, f64::min);
        let max_x = coordinates
            .iter()
            .map(|c| c.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = coordinates
            .iter()
            .map(|c| c.y)
            .fold(f64::INFINITY, f64::min);
        let max_y = coordinates
            .iter()
            .map(|c| c.y)
            .fold(f64::NEG_INFINITY, f64::max);

        // Calculate grid dimensions
        let grid_size = (self.config.n_splits as f64).sqrt().ceil() as usize;
        let x_step = (max_x - min_x) / grid_size as f64;
        let y_step = (max_y - min_y) / grid_size as f64;

        let mut clusters = Vec::new();

        for coord in coordinates {
            let x_grid = ((coord.x - min_x) / x_step).floor() as usize;
            let y_grid = ((coord.y - min_y) / y_step).floor() as usize;

            let x_grid = x_grid.min(grid_size - 1);
            let y_grid = y_grid.min(grid_size - 1);

            let cluster_id = (y_grid * grid_size + x_grid) % self.config.n_splits;
            clusters.push(cluster_id);
        }

        Ok(clusters)
    }

    /// Hierarchical clustering for spatial grouping
    fn hierarchical_clustering(&self, coordinates: &[SpatialCoordinate]) -> Result<Vec<usize>> {
        let n_samples = coordinates.len();

        // Calculate distance matrix
        let mut distances = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = self.calculate_distance(&coordinates[i], &coordinates[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        // Simple hierarchical clustering
        let mut clusters = (0..n_samples).collect::<Vec<_>>();
        let mut cluster_map = (0..n_samples).collect::<Vec<_>>();

        // Merge closest clusters until we have n_splits clusters
        while clusters.len() > self.config.n_splits {
            let mut min_distance = f64::INFINITY;
            let mut merge_i = 0;
            let mut merge_j = 0;

            for i in 0..clusters.len() {
                for j in i + 1..clusters.len() {
                    let dist = distances[clusters[i]][clusters[j]];
                    if dist < min_distance {
                        min_distance = dist;
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }

            // Merge clusters
            let cluster_j = clusters.remove(merge_j);
            let cluster_i = clusters[merge_i];

            // Update cluster assignments
            for assignment in &mut cluster_map {
                if *assignment == cluster_j {
                    *assignment = cluster_i;
                }
            }
        }

        // Map cluster IDs to 0..n_splits
        let unique_clusters: Vec<_> = cluster_map
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let mut final_clusters = vec![0; n_samples];

        for (i, &cluster_id) in cluster_map.iter().enumerate() {
            final_clusters[i] = unique_clusters
                .iter()
                .position(|&x| x == cluster_id)
                .unwrap_or(0);
        }

        Ok(final_clusters)
    }

    /// DBSCAN clustering for density-based spatial grouping
    fn dbscan_clustering(&self, coordinates: &[SpatialCoordinate]) -> Result<Vec<usize>> {
        let n_samples = coordinates.len();
        let eps = self.config.buffer_distance / 2.0;
        let min_pts = (n_samples / self.config.n_splits).max(2);

        let mut clusters = vec![None; n_samples];
        let mut visited = vec![false; n_samples];
        let mut cluster_id = 0;

        for i in 0..n_samples {
            if visited[i] {
                continue;
            }

            visited[i] = true;
            let neighbors = self.find_neighbors(i, coordinates, eps);

            if neighbors.len() < min_pts {
                clusters[i] = Some(usize::MAX); // Noise point
            } else {
                self.expand_cluster(
                    i,
                    &neighbors,
                    cluster_id,
                    coordinates,
                    eps,
                    min_pts,
                    &mut clusters,
                    &mut visited,
                );
                cluster_id += 1;
            }
        }

        // Convert to 0..n_splits range
        let max_clusters = cluster_id.min(self.config.n_splits);
        let mut final_clusters = vec![0; n_samples];

        for (i, cluster) in clusters.iter().enumerate() {
            final_clusters[i] = match cluster {
                Some(id) if *id != usize::MAX => *id % max_clusters,
                _ => i % self.config.n_splits, // Assign noise points randomly
            };
        }

        Ok(final_clusters)
    }

    fn find_neighbors(
        &self,
        point: usize,
        coordinates: &[SpatialCoordinate],
        eps: f64,
    ) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for (i, coord) in coordinates.iter().enumerate() {
            if i != point && self.calculate_distance(&coordinates[point], coord) <= eps {
                neighbors.push(i);
            }
        }
        neighbors
    }

    fn expand_cluster(
        &self,
        point: usize,
        neighbors: &[usize],
        cluster_id: usize,
        coordinates: &[SpatialCoordinate],
        eps: f64,
        min_pts: usize,
        clusters: &mut [Option<usize>],
        visited: &mut [bool],
    ) {
        clusters[point] = Some(cluster_id);
        let mut seed_set = neighbors.to_vec();
        let mut i = 0;

        while i < seed_set.len() {
            let q = seed_set[i];

            if !visited[q] {
                visited[q] = true;
                let q_neighbors = self.find_neighbors(q, coordinates, eps);

                if q_neighbors.len() >= min_pts {
                    seed_set.extend(q_neighbors);
                }
            }

            if clusters[q].is_none() {
                clusters[q] = Some(cluster_id);
            }

            i += 1;
        }
    }

    /// Generate cross-validation splits based on spatial clusters
    fn generate_cluster_splits(&self, clusters: &[usize]) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();

        for test_cluster in 0..self.config.n_splits {
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for (i, &cluster) in clusters.iter().enumerate() {
                if cluster == test_cluster {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            if !train_indices.is_empty() && !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }

        Ok(splits)
    }

    /// Apply buffer constraints to prevent spatial leakage
    fn apply_buffer_constraints(
        &self,
        splits: &[(Vec<usize>, Vec<usize>)],
        coordinates: &[SpatialCoordinate],
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut filtered_splits = Vec::new();

        for (train_indices, test_indices) in splits {
            let mut filtered_train = Vec::new();

            for &train_idx in train_indices {
                let mut too_close = false;

                for &test_idx in test_indices {
                    let distance =
                        self.calculate_distance(&coordinates[train_idx], &coordinates[test_idx]);

                    if distance < self.config.buffer_distance {
                        too_close = true;
                        break;
                    }
                }

                if !too_close {
                    filtered_train.push(train_idx);
                }
            }

            if !filtered_train.is_empty() && !test_indices.is_empty() {
                filtered_splits.push((filtered_train, test_indices.clone()));
            }
        }

        Ok(filtered_splits)
    }

    /// Calculate distance between two coordinates
    fn calculate_distance(&self, coord1: &SpatialCoordinate, coord2: &SpatialCoordinate) -> f64 {
        match self.config.distance_method {
            DistanceMethod::Euclidean => coord1.distance(coord2),
            DistanceMethod::Haversine => coord1.haversine_distance(coord2),
            DistanceMethod::Manhattan => (coord1.x - coord2.x).abs() + (coord1.y - coord2.y).abs(),
            DistanceMethod::Chebyshev => {
                (coord1.x - coord2.x).abs().max((coord1.y - coord2.y).abs())
            }
        }
    }

    fn get_rng(&self) -> impl scirs2_core::random::Rng {
        use scirs2_core::random::{rngs::StdRng, SeedableRng};
        match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42),
        }
    }
}

/// Leave-one-region-out cross-validator for spatial data
#[derive(Debug, Clone)]
pub struct LeaveOneRegionOut {
    region_labels: Vec<usize>,
}

impl LeaveOneRegionOut {
    pub fn new(region_labels: Vec<usize>) -> Self {
        Self { region_labels }
    }

    /// Generate splits where each region is used as test set once
    pub fn split(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if self.region_labels.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Region labels length must match number of samples".to_string(),
            ));
        }

        let unique_regions: HashSet<usize> = self.region_labels.iter().cloned().collect();
        let mut splits = Vec::new();

        for test_region in unique_regions {
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for (i, &region) in self.region_labels.iter().enumerate() {
                if region == test_region {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            if !train_indices.is_empty() && !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }

        Ok(splits)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_coordinate_distance() {
        let coord1 = SpatialCoordinate::new(0.0, 0.0);
        let coord2 = SpatialCoordinate::new(3.0, 4.0);

        assert!((coord1.distance(&coord2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_spatial_cross_validator() {
        let config = SpatialValidationConfig {
            buffer_distance: 1.0, // Use a reasonable buffer distance for test data
            ..Default::default()
        };
        let cv = SpatialCrossValidator::new(config);

        // Create simple grid of coordinates
        let mut coordinates = Vec::new();
        for i in 0..25 {
            let x = (i % 5) as f64;
            let y = (i / 5) as f64;
            coordinates.push(SpatialCoordinate::new(x, y));
        }

        let splits = cv.split(25, &coordinates).unwrap();
        assert!(!splits.is_empty(), "Should generate at least one split");

        for (train_indices, test_indices) in &splits {
            assert!(
                !train_indices.is_empty(),
                "Training set should not be empty"
            );
            assert!(!test_indices.is_empty(), "Test set should not be empty");
        }
    }

    #[test]
    fn test_leave_one_region_out() {
        let region_labels = vec![0, 0, 1, 1, 2, 2];
        let cv = LeaveOneRegionOut::new(region_labels);

        let splits = cv.split(6).unwrap();
        assert_eq!(splits.len(), 3, "Should have 3 splits for 3 regions");

        for (train_indices, test_indices) in &splits {
            assert!(
                !train_indices.is_empty(),
                "Training set should not be empty"
            );
            assert!(!test_indices.is_empty(), "Test set should not be empty");
        }
    }
}

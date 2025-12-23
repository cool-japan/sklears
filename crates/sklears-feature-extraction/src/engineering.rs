//! Feature engineering utilities

use crate::{Float, SklResult};
use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::prelude::{Fit, SklearsError, Transform};
use sklears_core::traits::{Estimator, Untrained};

// Re-export basic feature engineering methods
pub use crate::basic_features::*;

// Re-export RBF and kernel approximation methods
pub use crate::rbf_methods::*;

// Re-export temporal feature extraction methods
pub use crate::temporal_features::*;

// Re-export statistical feature extraction methods
pub use crate::statistical_features::*;

// Re-export sketching and sampling methods
pub use crate::sketching_sampling::*;

// =============================================================================
// Re-exports from time_series_features for backward compatibility
// =============================================================================

// Time series wavelet transform components have been moved to time_series_features.rs
pub use crate::time_series_features::{
    TimeSeriesWaveletExtractor, TimeSeriesWaveletType, WaveletFeatureType,
};

// Time-frequency feature extractor has been moved to time_series_features.rs
pub use crate::time_series_features::TimeFrequencyExtractor;

// =============================================================================
// Re-exports from topological_features for backward compatibility
// =============================================================================

// Topological feature extractors have been moved to topological_features.rs
pub use crate::topological_features::{
    MapperExtractor, PersistentHomologyExtractor, SimplicialComplexExtractor,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod simplicial_complex_tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_simplicial_complex_extractor_basic() {
        // Create a simple 2D dataset forming a triangle
        let data = Array2::from_shape_vec(
            (3, 2),
            vec![
                0.0, 0.0, // Point 0
                1.0, 0.0, // Point 1
                0.5, 0.8, // Point 2
            ],
        )
        .unwrap();

        let extractor = SimplicialComplexExtractor::new()
            .max_dimension(2)
            .distance_threshold(1.5)
            .include_face_counts(true)
            .include_euler_characteristic(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should have features for face counts (3 dimensions: 0, 1, 2) + euler characteristic
        assert_eq!(features.len(), 4);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }

        // Should have 3 vertices
        assert_eq!(features[0], 3.0);

        // Should have 3 edges (triangle)
        assert_eq!(features[1], 3.0);

        // Should have 1 triangle
        assert_eq!(features[2], 1.0);

        // Euler characteristic for triangle should be 1 (3 - 3 + 1 = 1)
        assert_eq!(features[3], 1.0);
    }

    #[test]
    fn test_simplicial_complex_extractor_depth_measures() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, // Point 0
                1.0, 0.0, // Point 1
                0.5, 0.8, // Point 2
                0.5, 0.2, // Point 3
            ],
        )
        .unwrap();

        let extractor = SimplicialComplexExtractor::new()
            .max_dimension(2)
            .distance_threshold(1.0)
            .include_face_counts(false)
            .include_euler_characteristic(false)
            .include_depth_measures(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should have depth measures: max_dim, mean_depth, max_depth, min_depth, complexity
        assert_eq!(features.len(), 5);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }

        // Max dimension should be <= 2
        assert!(features[0] <= 2.0);

        // Mean, max, min depths should be positive
        assert!(features[1] > 0.0);
        assert!(features[2] > 0.0);
        assert!(features[3] > 0.0);

        // Complexity should be between 0 and 1
        assert!(features[4] >= 0.0 && features[4] <= 1.0);
    }

    #[test]
    fn test_simplicial_complex_extractor_boundary_properties() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, // Point 0
                1.0, 0.0, // Point 1
                0.0, 1.0, // Point 2
                1.0, 1.0, // Point 3
            ],
        )
        .unwrap();

        let extractor = SimplicialComplexExtractor::new()
            .max_dimension(2)
            .distance_threshold(1.5)
            .include_face_counts(false)
            .include_euler_characteristic(false)
            .include_boundary_properties(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should have boundary properties for dimensions 1 and 2
        // For each dimension: faces_dim, faces_dim_minus_1, density
        assert_eq!(features.len(), 6);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }

        // All counts should be non-negative
        for &feature in &features {
            assert!(
                feature >= 0.0,
                "Feature should be non-negative: {}",
                feature
            );
        }
    }

    #[test]
    fn test_simplicial_complex_extractor_normalization() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, // Point 0
                1.0, 0.0, // Point 1
                0.5, 0.8, // Point 2
                0.0, 1.0, // Point 3
                1.0, 1.0, // Point 4
            ],
        )
        .unwrap();

        let extractor = SimplicialComplexExtractor::new()
            .max_dimension(2)
            .distance_threshold(1.5)
            .include_face_counts(true)
            .include_euler_characteristic(false)
            .normalize_by_vertices(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should have face counts normalized by number of vertices
        assert_eq!(features.len(), 3);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }

        // Vertices count should be 1.0 (normalized)
        assert_eq!(features[0], 1.0);

        // Other counts should be between 0 and some reasonable upper bound
        for i in 1..features.len() {
            assert!(features[i] >= 0.0 && features[i] <= 10.0);
        }
    }

    #[test]
    fn test_simplicial_complex_extractor_manhattan_distance() {
        let data = Array2::from_shape_vec(
            (3, 2),
            vec![
                0.0, 0.0, // Point 0
                1.0, 0.0, // Point 1
                0.0, 1.0, // Point 2
            ],
        )
        .unwrap();

        let extractor = SimplicialComplexExtractor::new()
            .max_dimension(2)
            .distance_threshold(1.5)
            .distance_metric("manhattan")
            .include_face_counts(true)
            .include_euler_characteristic(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should have face counts + euler characteristic
        assert_eq!(features.len(), 4);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }

        // Should still form a valid simplicial complex
        assert_eq!(features[0], 3.0); // vertices
        assert!(features[1] > 0.0); // edges
        assert!(features[3].is_finite()); // euler characteristic
    }

    #[test]
    fn test_simplicial_complex_extractor_empty_data() {
        let data = Array2::zeros((0, 2));
        let extractor = SimplicialComplexExtractor::new();

        let result = extractor.extract_features(&data.view());
        assert!(result.is_err(), "Should return error for empty data");
    }

    #[test]
    fn test_simplicial_complex_extractor_no_features_selected() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.5, 0.8]).unwrap();

        let extractor = SimplicialComplexExtractor::new()
            .include_face_counts(false)
            .include_euler_characteristic(false)
            .include_boundary_properties(false)
            .include_depth_measures(false);

        let result = extractor.extract_features(&data.view());
        assert!(
            result.is_err(),
            "Should return error when no features are selected"
        );
    }

    #[test]
    fn test_simplicial_complex_extractor_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();

        let extractor = SimplicialComplexExtractor::new()
            .max_dimension(2)
            .distance_threshold(1.0)
            .include_face_counts(true)
            .include_euler_characteristic(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should have face counts + euler characteristic
        assert_eq!(features.len(), 4);

        // Should have 1 vertex, 0 edges, 0 triangles
        assert_eq!(features[0], 1.0);
        assert_eq!(features[1], 0.0);
        assert_eq!(features[2], 0.0);

        // Euler characteristic should be 1
        assert_eq!(features[3], 1.0);
    }

    #[test]
    fn test_simplicial_complex_extractor_invalid_distance_metric() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.5, 0.8]).unwrap();

        let extractor = SimplicialComplexExtractor::new().distance_metric("invalid");

        let result = extractor.extract_features(&data.view());
        assert!(
            result.is_err(),
            "Should return error for invalid distance metric"
        );
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod parallel_feature_tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_parallel_feature_extractor_basic() {
        let data = Array2::from_shape_vec((100, 5), (0..500).map(|x| x as f64).collect()).unwrap();

        let extractor = ParallelFeatureExtractor::new()
            .chunk_size(20)
            .include_statistical_features(true)
            .include_distance_matrix(false)
            .include_correlation_features(false);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should have statistical features: 9 features total
        assert_eq!(features.len(), 9);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }
    }

    #[test]
    fn test_parallel_feature_extractor_distance_matrix() {
        let data =
            Array2::from_shape_vec((50, 3), (0..150).map(|x| (x as f64) * 0.1).collect()).unwrap();

        let extractor = ParallelFeatureExtractor::new()
            .chunk_size(10)
            .include_statistical_features(false)
            .include_distance_matrix(true)
            .distance_metric("euclidean");

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should have distance matrix features: mean, std, min, max
        assert_eq!(features.len(), 4);

        // All features should be finite and non-negative
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
            assert!(
                feature >= 0.0,
                "Distance feature should be non-negative: {}",
                feature
            );
        }

        // Mean distance should be > 0 for non-identical points
        assert!(features[0] > 0.0, "Mean distance should be positive");
    }

    #[test]
    fn test_parallel_feature_extractor_correlation() {
        let data =
            Array2::from_shape_vec((80, 4), (0..320).map(|x| (x as f64) * 0.01).collect()).unwrap();

        let extractor = ParallelFeatureExtractor::new()
            .chunk_size(15)
            .include_statistical_features(false)
            .include_distance_matrix(false)
            .include_correlation_features(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should have correlation features: mean, std, max_abs
        assert_eq!(features.len(), 3);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }

        // Max absolute correlation should be between 0 and 1
        assert!(
            features[2] >= 0.0 && features[2] <= 1.0,
            "Max absolute correlation should be in [0,1]: {}",
            features[2]
        );
    }

    #[test]
    fn test_parallel_feature_extractor_all_features() {
        let data =
            Array2::from_shape_vec((60, 3), (0..180).map(|x| (x as f64) * 0.05).collect()).unwrap();

        let extractor = ParallelFeatureExtractor::new()
            .chunk_size(12)
            .include_statistical_features(true)
            .include_distance_matrix(true)
            .include_correlation_features(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should have 9 + 4 + 3 = 16 features
        assert_eq!(features.len(), 16);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }
    }

    #[test]
    fn test_parallel_feature_extractor_different_metrics() {
        let data = Array2::from_shape_vec((30, 2), (0..60).map(|x| x as f64).collect()).unwrap();

        // Test Manhattan distance
        let extractor_manhattan = ParallelFeatureExtractor::new()
            .include_statistical_features(false)
            .include_distance_matrix(true)
            .distance_metric("manhattan");

        let features_manhattan = extractor_manhattan.extract_features(&data.view()).unwrap();
        assert_eq!(features_manhattan.len(), 4);

        // Test Cosine distance
        let extractor_cosine = ParallelFeatureExtractor::new()
            .include_statistical_features(false)
            .include_distance_matrix(true)
            .distance_metric("cosine");

        let features_cosine = extractor_cosine.extract_features(&data.view()).unwrap();
        assert_eq!(features_cosine.len(), 4);

        // Features should be different for different metrics
        assert_ne!(features_manhattan[0], features_cosine[0]);
    }

    #[test]
    fn test_parallel_feature_extractor_custom_threads() {
        let data = Array2::from_shape_vec((40, 3), (0..120).map(|x| x as f64).collect()).unwrap();

        let extractor = ParallelFeatureExtractor::new()
            .chunk_size(8)
            .num_threads(2)
            .include_statistical_features(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should extract features successfully with custom thread count
        assert_eq!(features.len(), 9);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }
    }

    #[test]
    fn test_parallel_feature_extractor_small_data() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();

        let extractor = ParallelFeatureExtractor::new()
            .chunk_size(2)
            .include_statistical_features(true)
            .include_distance_matrix(true)
            .include_correlation_features(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should handle small datasets gracefully
        assert_eq!(features.len(), 16);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }
    }

    #[test]
    fn test_parallel_feature_extractor_single_sample() {
        let data = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();

        let extractor = ParallelFeatureExtractor::new()
            .include_statistical_features(true)
            .include_distance_matrix(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should handle single sample (distance matrix features will be zeros)
        assert_eq!(features.len(), 13); // 9 statistical + 4 distance (zeros)

        // Distance features should be zero for single sample
        for i in 9..13 {
            assert_eq!(
                features[i], 0.0,
                "Distance feature should be zero for single sample"
            );
        }
    }

    #[test]
    fn test_parallel_feature_extractor_empty_data() {
        let data = Array2::zeros((0, 3));
        let extractor = ParallelFeatureExtractor::new();

        let result = extractor.extract_features(&data.view());
        assert!(result.is_err(), "Should return error for empty data");
    }

    #[test]
    fn test_parallel_feature_extractor_no_features_selected() {
        let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();

        let extractor = ParallelFeatureExtractor::new()
            .include_statistical_features(false)
            .include_distance_matrix(false)
            .include_correlation_features(false);

        let result = extractor.extract_features(&data.view());
        assert!(
            result.is_err(),
            "Should return error when no features are selected"
        );
    }

    #[test]
    fn test_parallel_feature_extractor_invalid_distance_metric() {
        let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();

        let extractor = ParallelFeatureExtractor::new()
            .include_statistical_features(false)
            .include_distance_matrix(true)
            .include_correlation_features(false)
            .distance_metric("invalid");

        let result = extractor.extract_features(&data.view());
        assert!(
            result.is_err(),
            "Should return error for invalid distance metric"
        );
    }

    #[test]
    fn test_parallel_feature_extractor_large_chunk_size() {
        let data = Array2::from_shape_vec((20, 3), (0..60).map(|x| x as f64).collect()).unwrap();

        let extractor = ParallelFeatureExtractor::new()
            .chunk_size(100) // Larger than data size
            .include_statistical_features(true);

        let features = extractor.extract_features(&data.view()).unwrap();

        // Should handle large chunk sizes gracefully
        assert_eq!(features.len(), 9);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {}", feature);
        }
    }
}

/// Parallel Feature Extractor for High-Performance Computing
///
/// This extractor provides parallel computation of various feature types across data chunks,
/// optimized for large datasets and multi-core systems.
///
/// Features extracted include:
/// - Statistical features computed in parallel across columns and rows
/// - Distance matrix features with parallel distance computation
/// - Correlation features with parallel correlation computation
/// - Configurable chunk size and thread count for optimal performance
///
/// # Parameters
///
/// * `chunk_size` - Size of data chunks for parallel processing
/// * `num_threads` - Number of threads to use (None for automatic)
/// * `include_distance_matrix` - Whether to compute distance matrix features
/// * `include_statistical_features` - Whether to compute statistical features
/// * `include_correlation_features` - Whether to compute correlation features
/// * `distance_metric` - Distance metric for distance matrix computation
///
/// # Examples
///
/// ```
/// # use sklears_feature_extraction::engineering::ParallelFeatureExtractor;
/// # use scirs2_core::ndarray::Array2;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((1000, 50), (0..50000).map(|x| x as f64).collect())?;
///
/// let extractor = ParallelFeatureExtractor::new()
///     .chunk_size(100)
///     .num_threads(4)
///     .include_statistical_features(true)
///     .include_distance_matrix(true);
///
/// let features = extractor.extract_features(&data.view())?;
/// assert!(features.len() > 0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ParallelFeatureExtractor {
    chunk_size: usize,
    num_threads: Option<usize>,
    include_distance_matrix: bool,
    include_statistical_features: bool,
    include_correlation_features: bool,
    distance_metric: String,
}

impl ParallelFeatureExtractor {
    /// Create a new parallel feature extractor
    pub fn new() -> Self {
        Self {
            chunk_size: 1000,
            num_threads: None,
            include_distance_matrix: false,
            include_statistical_features: true,
            include_correlation_features: false,
            distance_metric: "euclidean".to_string(),
        }
    }

    /// Set the chunk size for parallel processing
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.max(1);
        self
    }

    /// Set the number of threads to use
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads.max(1));
        self
    }

    /// Set whether to compute distance matrix features
    pub fn include_distance_matrix(mut self, include: bool) -> Self {
        self.include_distance_matrix = include;
        self
    }

    /// Set whether to compute statistical features
    pub fn include_statistical_features(mut self, include: bool) -> Self {
        self.include_statistical_features = include;
        self
    }

    /// Set whether to compute correlation features
    pub fn include_correlation_features(mut self, include: bool) -> Self {
        self.include_correlation_features = include;
        self
    }

    /// Set the distance metric for distance matrix computation
    pub fn distance_metric(mut self, metric: &str) -> Self {
        self.distance_metric = metric.to_string();
        self
    }

    /// Extract features using parallel processing
    pub fn extract_features(&self, data: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        if data.nrows() == 0 || data.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        // Set up thread pool if specified
        let result = if let Some(threads) = self.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to create thread pool: {}", e))
                })?
                .install(|| self.extract_features_impl(data))
        } else {
            self.extract_features_impl(data)
        };

        result
    }

    /// Internal implementation of feature extraction
    fn extract_features_impl(&self, data: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        let mut features = Vec::new();

        if self.include_statistical_features {
            features.extend(self.compute_parallel_statistical_features(data)?);
        }

        if self.include_distance_matrix {
            features.extend(self.compute_parallel_distance_matrix_features(data)?);
        }

        if self.include_correlation_features {
            features.extend(self.compute_parallel_correlation_features(data)?);
        }

        if features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected for extraction".to_string(),
            ));
        }

        // Verify all features are finite
        for &feature in &features {
            if !feature.is_finite() {
                return Err(SklearsError::InvalidInput(
                    "Non-finite feature detected".to_string(),
                ));
            }
        }

        Ok(Array1::from_vec(features))
    }

    /// Compute statistical features in parallel across data chunks
    fn compute_parallel_statistical_features(
        &self,
        data: &ArrayView2<Float>,
    ) -> SklResult<Vec<Float>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Process columns in parallel
        let column_stats: Vec<Vec<Float>> = (0..n_features)
            .into_par_iter()
            .map(|col_idx| {
                let column = data.column(col_idx);
                self.compute_column_statistics(&column)
            })
            .collect();

        // Aggregate statistics across all columns
        let mut features = Vec::new();

        // Mean of column means
        let column_means: Vec<Float> = column_stats.iter().map(|stats| stats[0]).collect();
        features.push(column_means.iter().sum::<Float>() / n_features as Float);

        // Mean of column standard deviations
        let column_stds: Vec<Float> = column_stats.iter().map(|stats| stats[1]).collect();
        features.push(column_stds.iter().sum::<Float>() / n_features as Float);

        // Mean of column mins and maxs
        let column_mins: Vec<Float> = column_stats.iter().map(|stats| stats[2]).collect();
        let column_maxs: Vec<Float> = column_stats.iter().map(|stats| stats[3]).collect();
        features.push(column_mins.iter().sum::<Float>() / n_features as Float);
        features.push(column_maxs.iter().sum::<Float>() / n_features as Float);

        // Global statistics
        features.push(
            *column_mins
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        );
        features.push(
            *column_maxs
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        );

        // Parallel computation of row-wise statistics
        let row_stats: Vec<Vec<Float>> = (0..n_samples)
            .into_par_iter()
            .step_by(self.chunk_size)
            .map(|start_idx| {
                let end_idx = (start_idx + self.chunk_size).min(n_samples);
                let mut chunk_stats = Vec::new();

                for row_idx in start_idx..end_idx {
                    let row = data.row(row_idx);
                    let row_mean = row.iter().sum::<Float>() / n_features as Float;
                    let row_var = row.iter().map(|&x| (x - row_mean).powi(2)).sum::<Float>()
                        / n_features as Float;
                    chunk_stats.push(row_mean);
                    chunk_stats.push(row_var.sqrt());
                }

                chunk_stats
            })
            .collect();

        // Aggregate row statistics
        let all_row_stats: Vec<Float> = row_stats.into_iter().flatten().collect();
        if !all_row_stats.is_empty() {
            features.push(all_row_stats.iter().sum::<Float>() / all_row_stats.len() as Float);
            features.push(all_row_stats.iter().cloned().fold(0.0, Float::max));
            features.push(
                all_row_stats
                    .iter()
                    .cloned()
                    .fold(Float::INFINITY, Float::min),
            );
        }

        Ok(features)
    }

    /// Compute statistics for a single column
    fn compute_column_statistics(&self, column: &ArrayView1<Float>) -> Vec<Float> {
        let n = column.len() as Float;
        let mean = column.iter().sum::<Float>() / n;
        let variance = column.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / n;
        let std_dev = variance.sqrt();
        let min_val = column.iter().cloned().fold(Float::INFINITY, Float::min);
        let max_val = column.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

        vec![mean, std_dev, min_val, max_val]
    }

    /// Compute distance matrix features in parallel
    fn compute_parallel_distance_matrix_features(
        &self,
        data: &ArrayView2<Float>,
    ) -> SklResult<Vec<Float>> {
        let n_samples = data.nrows();

        if n_samples < 2 {
            return Ok(vec![0.0, 0.0, 0.0, 0.0]); // mean, std, min, max distances
        }

        // Compute distance matrix in parallel using chunks
        let distance_chunks: Result<Vec<Vec<Float>>, SklearsError> = (0..n_samples)
            .into_par_iter()
            .step_by(self.chunk_size)
            .map(|start_i| -> SklResult<Vec<Float>> {
                let end_i = (start_i + self.chunk_size).min(n_samples);
                let mut chunk_distances = Vec::new();

                for i in start_i..end_i {
                    for j in (i + 1)..n_samples {
                        let dist = self.compute_distance(&data.row(i), &data.row(j))?;
                        chunk_distances.push(dist);
                    }
                }

                Ok(chunk_distances)
            })
            .collect();

        let distance_chunks = distance_chunks?;

        // Flatten and aggregate distances
        let all_distances: Vec<Float> = distance_chunks.into_iter().flatten().collect();

        if all_distances.is_empty() {
            return Ok(vec![0.0, 0.0, 0.0, 0.0]);
        }

        let n_distances = all_distances.len() as Float;
        let mean_distance = all_distances.iter().sum::<Float>() / n_distances;
        let variance = all_distances
            .iter()
            .map(|&d| (d - mean_distance).powi(2))
            .sum::<Float>()
            / n_distances;
        let std_distance = variance.sqrt();
        let min_distance = all_distances
            .iter()
            .cloned()
            .fold(Float::INFINITY, Float::min);
        let max_distance = all_distances
            .iter()
            .cloned()
            .fold(Float::NEG_INFINITY, Float::max);

        Ok(vec![
            mean_distance,
            std_distance,
            min_distance,
            max_distance,
        ])
    }

    /// Compute correlation features in parallel
    fn compute_parallel_correlation_features(
        &self,
        data: &ArrayView2<Float>,
    ) -> SklResult<Vec<Float>> {
        let n_features = data.ncols();

        if n_features < 2 {
            return Ok(vec![0.0, 0.0, 0.0]); // mean, std, max absolute correlation
        }

        // Compute column means in parallel
        let column_means: Vec<Float> = (0..n_features)
            .into_par_iter()
            .map(|col_idx| {
                let column = data.column(col_idx);
                column.iter().sum::<Float>() / data.nrows() as Float
            })
            .collect();

        // Compute correlations in parallel
        let correlation_chunks: Vec<Vec<Float>> = (0..n_features)
            .into_par_iter()
            .step_by(self.chunk_size)
            .map(|start_i| {
                let end_i = (start_i + self.chunk_size).min(n_features);
                let mut chunk_correlations = Vec::new();

                for i in start_i..end_i {
                    for j in (i + 1)..n_features {
                        let corr = self.compute_correlation(
                            &data.column(i),
                            &data.column(j),
                            column_means[i],
                            column_means[j],
                        );
                        chunk_correlations.push(corr);
                    }
                }

                chunk_correlations
            })
            .collect();

        // Aggregate correlations
        let all_correlations: Vec<Float> = correlation_chunks.into_iter().flatten().collect();

        if all_correlations.is_empty() {
            return Ok(vec![0.0, 0.0, 0.0]);
        }

        let n_correlations = all_correlations.len() as Float;
        let mean_correlation = all_correlations.iter().sum::<Float>() / n_correlations;
        let variance = all_correlations
            .iter()
            .map(|&c| (c - mean_correlation).powi(2))
            .sum::<Float>()
            / n_correlations;
        let std_correlation = variance.sqrt();
        let max_abs_correlation = all_correlations
            .iter()
            .map(|&c| c.abs())
            .fold(0.0, Float::max);

        Ok(vec![mean_correlation, std_correlation, max_abs_correlation])
    }

    /// Compute distance between two points using the specified metric
    fn compute_distance(&self, p1: &ArrayView1<Float>, p2: &ArrayView1<Float>) -> SklResult<Float> {
        match self.distance_metric.as_str() {
            "euclidean" => {
                let dist = p1
                    .iter()
                    .zip(p2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>()
                    .sqrt();
                Ok(dist)
            }
            "manhattan" => {
                let dist = p1
                    .iter()
                    .zip(p2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<Float>();
                Ok(dist)
            }
            "cosine" => {
                let dot_product = p1.iter().zip(p2.iter()).map(|(a, b)| a * b).sum::<Float>();
                let norm1 = p1.iter().map(|&a| a.powi(2)).sum::<Float>().sqrt();
                let norm2 = p2.iter().map(|&a| a.powi(2)).sum::<Float>().sqrt();

                if norm1 == 0.0 || norm2 == 0.0 {
                    Ok(1.0) // Maximum cosine distance
                } else {
                    Ok(1.0 - dot_product / (norm1 * norm2))
                }
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unsupported distance metric: {}",
                self.distance_metric
            ))),
        }
    }

    /// Compute Pearson correlation between two columns
    fn compute_correlation(
        &self,
        col1: &ArrayView1<Float>,
        col2: &ArrayView1<Float>,
        mean1: Float,
        mean2: Float,
    ) -> Float {
        let n = col1.len() as Float;

        let covariance = col1
            .iter()
            .zip(col2.iter())
            .map(|(&x, &y)| (x - mean1) * (y - mean2))
            .sum::<Float>()
            / n;

        let var1 = col1.iter().map(|&x| (x - mean1).powi(2)).sum::<Float>() / n;
        let var2 = col2.iter().map(|&x| (x - mean2).powi(2)).sum::<Float>() / n;

        let std1 = var1.sqrt();
        let std2 = var2.sqrt();

        if std1 == 0.0 || std2 == 0.0 {
            0.0
        } else {
            covariance / (std1 * std2)
        }
    }
}

impl Default for ParallelFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Mixed-type feature extractor for handling heterogeneous data
///
/// This extractor can handle different types of features (numerical, categorical, binary, ordinal)
/// and apply appropriate transformations to each type, creating a unified feature matrix.
#[derive(Debug, Clone)]
pub struct MixedTypeFeatureExtractor {
    /// Types of features for each column
    pub feature_types: Vec<FeatureType>,
    /// Whether to normalize numerical features
    pub normalize_numerical: bool,
    /// Whether to handle missing values
    pub handle_missing_values: bool,
    /// Strategy for handling missing values
    pub missing_value_strategy: MissingValueStrategy,
    /// Categorical encoding strategy
    pub encoding_strategy: CategoricalEncoding,
    /// Whether to include interaction features
    pub include_interaction_features: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

/// Types of features that can be handled
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureType {
    /// Continuous numerical features
    Numerical,
    /// Categorical features with no inherent order
    Categorical,
    /// Binary features (0/1)
    Binary,
    /// Ordinal features with inherent order
    Ordinal,
    /// Date/time features
    DateTime,
}

/// Strategies for handling missing values
#[derive(Debug, Clone, Copy)]
pub enum MissingValueStrategy {
    /// Replace with mean (for numerical) or mode (for categorical)
    Mean,
    /// Replace with median (for numerical) or mode (for categorical)
    Median,
    /// Replace with mode (most frequent value)
    Mode,
    /// Replace with zero
    Zero,
    /// Forward fill
    ForwardFill,
    /// Backward fill
    BackwardFill,
}

/// Categorical encoding strategies
#[derive(Debug, Clone, Copy)]
pub enum CategoricalEncoding {
    /// One-hot encoding
    OneHot,
    /// Label encoding (ordinal integers)
    LabelEncoding,
    /// Target encoding (mean of target for each category)
    TargetEncoding,
    /// Binary encoding
    BinaryEncoding,
    /// Frequency encoding
    FrequencyEncoding,
}

impl MixedTypeFeatureExtractor {
    /// Create a new mixed-type feature extractor
    pub fn new() -> Self {
        Self {
            feature_types: Vec::new(),
            normalize_numerical: true,
            handle_missing_values: true,
            missing_value_strategy: MissingValueStrategy::Mean,
            encoding_strategy: CategoricalEncoding::OneHot,
            include_interaction_features: false,
            random_state: None,
        }
    }

    /// Set feature types for each column
    pub fn feature_types(mut self, types: Vec<FeatureType>) -> Self {
        self.feature_types = types;
        self
    }

    /// Set whether to normalize numerical features
    pub fn normalize_numerical(mut self, normalize: bool) -> Self {
        self.normalize_numerical = normalize;
        self
    }

    /// Set whether to handle missing values
    pub fn handle_missing_values(mut self, handle: bool) -> Self {
        self.handle_missing_values = handle;
        self
    }

    /// Set missing value strategy
    pub fn missing_value_strategy(mut self, strategy: MissingValueStrategy) -> Self {
        self.missing_value_strategy = strategy;
        self
    }

    /// Set categorical encoding strategy
    pub fn encoding_strategy(mut self, strategy: CategoricalEncoding) -> Self {
        self.encoding_strategy = strategy;
        self
    }

    /// Set whether to include interaction features
    pub fn include_interaction_features(mut self, include: bool) -> Self {
        self.include_interaction_features = include;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Handle missing values in data
    fn handle_missing(
        &self,
        data: &Array2<f64>,
        col_idx: usize,
        feature_type: FeatureType,
    ) -> Array1<f64> {
        let column = data.column(col_idx);
        let mut result = Array1::zeros(column.len());

        // Find non-missing values
        let valid_values: Vec<f64> = column.iter().filter(|&&x| x.is_finite()).copied().collect();

        if valid_values.is_empty() {
            return result; // All zeros if no valid values
        }

        let replacement_value = match (self.missing_value_strategy, feature_type) {
            (MissingValueStrategy::Mean, FeatureType::Numerical) => {
                valid_values.iter().sum::<f64>() / valid_values.len() as f64
            }
            (MissingValueStrategy::Median, FeatureType::Numerical) => {
                let mut sorted = valid_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                }
            }
            (MissingValueStrategy::Mode, _) => {
                // Find most frequent value
                let mut counts = std::collections::HashMap::new();
                for &val in &valid_values {
                    *counts.entry(val as i64).or_insert(0) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(val, _)| val as f64)
                    .unwrap_or(0.0)
            }
            (MissingValueStrategy::Zero, _) => 0.0,
            (MissingValueStrategy::ForwardFill, _) => {
                // Use first valid value
                valid_values[0]
            }
            (MissingValueStrategy::BackwardFill, _) => {
                // Use last valid value
                valid_values[valid_values.len() - 1]
            }
            _ => valid_values.iter().sum::<f64>() / valid_values.len() as f64, // Default to mean
        };

        // Replace missing values
        for (i, &value) in column.iter().enumerate() {
            result[i] = if value.is_finite() {
                value
            } else {
                replacement_value
            };
        }

        result
    }

    /// Normalize numerical features
    fn normalize_column(&self, column: &Array1<f64>) -> Array1<f64> {
        let mean = column.mean().unwrap_or(0.0);
        let std = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
        let std = std.sqrt();

        if std == 0.0 {
            Array1::zeros(column.len())
        } else {
            column.mapv(|x| (x - mean) / std)
        }
    }

    /// Encode categorical features
    fn encode_categorical(&self, column: &Array1<f64>) -> Array2<f64> {
        match self.encoding_strategy {
            CategoricalEncoding::OneHot => {
                // Find unique values
                let mut unique_values: Vec<i64> = column
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                unique_values.sort();

                let n_categories = unique_values.len();
                let mut encoded = Array2::zeros((column.len(), n_categories));

                for (row_idx, &value) in column.iter().enumerate() {
                    if let Some(cat_idx) = unique_values.iter().position(|&x| x == value as i64) {
                        encoded[[row_idx, cat_idx]] = 1.0;
                    }
                }

                encoded
            }
            CategoricalEncoding::LabelEncoding => {
                // Map each unique value to an integer
                let mut unique_values: Vec<i64> = column
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                unique_values.sort();

                let mut encoded = Array2::zeros((column.len(), 1));
                for (row_idx, &value) in column.iter().enumerate() {
                    if let Some(label) = unique_values.iter().position(|&x| x == value as i64) {
                        encoded[[row_idx, 0]] = label as f64;
                    }
                }

                encoded
            }
            CategoricalEncoding::TargetEncoding => {
                // For simplicity, use the value itself (would normally use target mean)
                let mut encoded = Array2::zeros((column.len(), 1));
                for (row_idx, &value) in column.iter().enumerate() {
                    encoded[[row_idx, 0]] = value;
                }
                encoded
            }
            _ => {
                // Default to label encoding
                let mut encoded = Array2::zeros((column.len(), 1));
                for (row_idx, &value) in column.iter().enumerate() {
                    encoded[[row_idx, 0]] = value;
                }
                encoded
            }
        }
    }

    /// Process a single feature column
    fn process_feature(
        &self,
        data: &Array2<f64>,
        col_idx: usize,
        feature_type: FeatureType,
    ) -> Array2<f64> {
        // Handle missing values
        let column = if self.handle_missing_values {
            self.handle_missing(data, col_idx, feature_type)
        } else {
            data.column(col_idx).to_owned()
        };

        match feature_type {
            FeatureType::Numerical => {
                let processed = if self.normalize_numerical {
                    self.normalize_column(&column)
                } else {
                    column
                };
                processed.insert_axis(Axis(1))
            }
            FeatureType::Categorical => self.encode_categorical(&column),
            FeatureType::Binary => {
                // Ensure binary values are 0 or 1
                let binary = column.mapv(|x| if x > 0.5 { 1.0 } else { 0.0 });
                binary.insert_axis(Axis(1))
            }
            FeatureType::Ordinal => {
                // Treat as numerical but preserve order
                let processed = if self.normalize_numerical {
                    self.normalize_column(&column)
                } else {
                    column
                };
                processed.insert_axis(Axis(1))
            }
            FeatureType::DateTime => {
                // For simplicity, treat as numerical
                let processed = if self.normalize_numerical {
                    self.normalize_column(&column)
                } else {
                    column
                };
                processed.insert_axis(Axis(1))
            }
        }
    }

    /// Transform the data according to feature types
    fn transform_data(&self, data: &Array2<f64>) -> SklResult<Array2<f64>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if self.feature_types.len() != data.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of feature types ({}) doesn't match number of columns ({})",
                self.feature_types.len(),
                data.ncols()
            )));
        }

        let mut processed_features = Vec::new();

        // Process each feature according to its type
        for (col_idx, &feature_type) in self.feature_types.iter().enumerate() {
            let processed = self.process_feature(data, col_idx, feature_type);
            processed_features.push(processed);
        }

        // Concatenate all processed features
        if processed_features.is_empty() {
            return Ok(Array2::zeros((data.nrows(), 0)));
        }

        let total_cols: usize = processed_features.iter().map(|f| f.ncols()).sum();
        let mut result = Array2::zeros((data.nrows(), total_cols));

        let mut col_offset = 0;
        for feature_matrix in processed_features {
            let n_cols = feature_matrix.ncols();
            result
                .slice_mut(s![.., col_offset..col_offset + n_cols])
                .assign(&feature_matrix);
            col_offset += n_cols;
        }

        Ok(result)
    }
}

/// Fitted mixed-type feature extractor
pub struct FittedMixedTypeFeatureExtractor {
    extractor: MixedTypeFeatureExtractor,
    transformation_stats: std::collections::HashMap<String, f64>,
}

impl FittedMixedTypeFeatureExtractor {
    /// Get transformation statistics
    pub fn transformation_statistics(&self) -> &std::collections::HashMap<String, f64> {
        &self.transformation_stats
    }
}

impl Estimator<Untrained> for MixedTypeFeatureExtractor {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for MixedTypeFeatureExtractor {
    type Fitted = FittedMixedTypeFeatureExtractor;

    fn fit(self, X: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        if X.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Set default feature types if not specified
        let feature_types = if self.feature_types.is_empty() {
            vec![FeatureType::Numerical; X.ncols()]
        } else {
            self.feature_types.clone()
        };

        // Calculate transformation statistics
        let mut stats = std::collections::HashMap::new();

        let n_numerical = feature_types
            .iter()
            .filter(|&&t| t == FeatureType::Numerical)
            .count();
        let n_categorical = feature_types
            .iter()
            .filter(|&&t| t == FeatureType::Categorical)
            .count();
        let n_binary = feature_types
            .iter()
            .filter(|&&t| t == FeatureType::Binary)
            .count();
        let n_ordinal = feature_types
            .iter()
            .filter(|&&t| t == FeatureType::Ordinal)
            .count();

        stats.insert("n_numerical_features".to_string(), n_numerical as f64);
        stats.insert("n_categorical_features".to_string(), n_categorical as f64);
        stats.insert("n_binary_features".to_string(), n_binary as f64);
        stats.insert("n_ordinal_features".to_string(), n_ordinal as f64);
        stats.insert("total_input_features".to_string(), X.ncols() as f64);

        // Estimate output feature count (simplified)
        let estimated_output_features = match self.encoding_strategy {
            CategoricalEncoding::OneHot => {
                // Estimate expanded size for one-hot encoding
                let categorical_expansion = n_categorical * 3; // Assume avg 3 categories
                n_numerical + n_binary + n_ordinal + categorical_expansion
            }
            _ => feature_types.len(),
        };

        stats.insert(
            "total_output_features".to_string(),
            estimated_output_features as f64,
        );

        let updated_extractor = MixedTypeFeatureExtractor {
            feature_types,
            ..self
        };

        Ok(FittedMixedTypeFeatureExtractor {
            extractor: updated_extractor,
            transformation_stats: stats,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedMixedTypeFeatureExtractor {
    fn transform(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.extractor.transform_data(X)
    }
}

impl Default for MixedTypeFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod sketching_tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_mixed_type_feature_extractor_basic() {
        let data = Array2::from_shape_vec(
            (4, 6),
            vec![
                1.0, 2.0, 3.0, 1.0, 0.0,
                5.0, // numerical: 1,2,3  categorical: 1,0  ordinal: 5
                4.0, 5.0, 6.0, 2.0, 1.0,
                3.0, // numerical: 4,5,6  categorical: 2,1  ordinal: 3
                7.0, 8.0, 9.0, 1.0, 0.0,
                4.0, // numerical: 7,8,9  categorical: 1,0  ordinal: 4
                2.0, 3.0, 4.0, 3.0, 1.0,
                2.0, // numerical: 2,3,4  categorical: 3,1  ordinal: 2
            ],
        )
        .unwrap();

        let feature_types = vec![
            FeatureType::Numerical,   // col 0
            FeatureType::Numerical,   // col 1
            FeatureType::Numerical,   // col 2
            FeatureType::Categorical, // col 3
            FeatureType::Binary,      // col 4
            FeatureType::Ordinal,     // col 5
        ];

        let extractor = MixedTypeFeatureExtractor::new()
            .feature_types(feature_types)
            .normalize_numerical(true)
            .encoding_strategy(CategoricalEncoding::OneHot);

        let fitted = extractor.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.nrows(), 4);
        assert!(features.ncols() > 6); // Should be expanded due to one-hot encoding

        // Check that all features are finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_mixed_type_feature_extractor_different_encodings() {
        let data = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 1.0, 0.0, 5.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 0.0, 4.0],
        )
        .unwrap();

        let feature_types = vec![
            FeatureType::Numerical,
            FeatureType::Categorical,
            FeatureType::Binary,
            FeatureType::Ordinal,
        ];

        let encodings = vec![
            CategoricalEncoding::OneHot,
            CategoricalEncoding::LabelEncoding,
            CategoricalEncoding::TargetEncoding,
        ];

        for encoding in encodings {
            let extractor = MixedTypeFeatureExtractor::new()
                .feature_types(feature_types.clone())
                .encoding_strategy(encoding);

            let fitted = extractor.fit(&data, &()).unwrap();
            let features = fitted.transform(&data).unwrap();

            assert_eq!(features.nrows(), 3);
            assert!(features.ncols() > 0);

            // Check that all features are finite
            for &value in features.iter() {
                assert!(value.is_finite());
            }
        }
    }

    #[test]
    fn test_mixed_type_feature_extractor_empty_data() {
        let data = Array2::zeros((0, 0));
        let extractor = MixedTypeFeatureExtractor::new();

        let result = extractor.fit(&data, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_mixed_type_feature_extractor_configuration() {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 3.0, 1.0, 0.0])
                .unwrap();

        let feature_types = vec![
            FeatureType::Numerical,
            FeatureType::Categorical,
            FeatureType::Binary,
        ];

        let extractor = MixedTypeFeatureExtractor::new()
            .feature_types(feature_types)
            .normalize_numerical(true)
            .handle_missing_values(true)
            .missing_value_strategy(MissingValueStrategy::Mean)
            .encoding_strategy(CategoricalEncoding::OneHot)
            .include_interaction_features(true)
            .random_state(42);

        let fitted = extractor.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.nrows(), 3);

        // Check that configuration was applied
        let stats = fitted.transformation_statistics();
        assert!(stats.contains_key("n_numerical_features"));
        assert!(stats.contains_key("n_categorical_features"));
        assert!(stats.contains_key("n_binary_features"));
        assert!(stats.contains_key("total_output_features"));
    }

    #[test]
    fn test_mixed_type_feature_extractor_missing_values() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0,
                1.0,
                f64::NAN,
                2.0,
                f64::NAN,
                1.0,
                f64::NAN,
                2.0,
                0.0,
                3.0,
                1.0,
                1.0,
            ],
        )
        .unwrap();

        let feature_types = vec![
            FeatureType::Numerical,
            FeatureType::Categorical,
            FeatureType::Binary,
        ];

        let strategies = vec![
            MissingValueStrategy::Mean,
            MissingValueStrategy::Median,
            MissingValueStrategy::Mode,
            MissingValueStrategy::Zero,
        ];

        for strategy in strategies {
            let extractor = MixedTypeFeatureExtractor::new()
                .feature_types(feature_types.clone())
                .handle_missing_values(true)
                .missing_value_strategy(strategy);

            let fitted = extractor.fit(&data, &()).unwrap();
            let features = fitted.transform(&data).unwrap();

            assert_eq!(features.nrows(), 4);

            // Check that no NaN values remain
            for &value in features.iter() {
                assert!(
                    value.is_finite(),
                    "All values should be finite after missing value handling"
                );
            }
        }
    }

    #[test]
    fn test_mixed_type_feature_extractor_consistency() {
        let data = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 1.0, 0.0, 5.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 0.0, 4.0],
        )
        .unwrap();

        let feature_types = vec![
            FeatureType::Numerical,
            FeatureType::Categorical,
            FeatureType::Binary,
            FeatureType::Ordinal,
        ];

        let extractor = MixedTypeFeatureExtractor::new()
            .feature_types(feature_types)
            .random_state(42);

        let fitted = extractor.fit(&data, &()).unwrap();
        let features1 = fitted.transform(&data).unwrap();
        let features2 = fitted.transform(&data).unwrap();

        // Results should be identical
        assert_eq!(features1.shape(), features2.shape());
        for (a, b) in features1.iter().zip(features2.iter()) {
            assert_eq!(a, b);
        }
    }
}

// Temporal-spatial feature extractor has been moved to time_series_features.rs
pub use crate::time_series_features::TemporalSpatialExtractor;

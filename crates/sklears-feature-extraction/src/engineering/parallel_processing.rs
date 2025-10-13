//! Parallel Feature Processing Module
//!
//! This module provides high-performance parallel feature extraction capabilities
//! optimized for large datasets and multi-core systems.

use crate::*;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use sklears_core::prelude::{SklearsError, Transform};
use std::collections::HashMap;

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
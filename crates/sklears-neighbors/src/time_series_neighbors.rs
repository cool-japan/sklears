//! Time series neighbor search methods
//!
//! This module provides specialized neighbor search algorithms for time series data,
//! including dynamic time warping (DTW) distance, time series shapelets, temporal
//! neighbor search, subsequence similarity search, and streaming time series neighbors.

use crate::distance::Distance;
use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::types::Float;
use std::collections::{HashMap, VecDeque};

/// Dynamic Time Warping (DTW) distance calculator
#[derive(Debug, Clone)]
pub struct DtwDistance {
    /// Warping window constraint (Sakoe-Chiba band)
    pub window: Option<usize>,
    /// Step pattern for DTW
    pub step_pattern: DtwStepPattern,
    /// Whether to use normalized DTW
    pub normalize: bool,
}

/// DTW step patterns
#[derive(Debug, Clone, Copy)]
pub enum DtwStepPattern {
    Symmetric,
    Asymmetric,
    QuasiSymmetric,
}

impl Default for DtwDistance {
    fn default() -> Self {
        Self {
            window: None,
            step_pattern: DtwStepPattern::Symmetric,
            normalize: true,
        }
    }
}

impl DtwDistance {
    /// Create a new DTW distance calculator
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the warping window constraint
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = Some(window);
        self
    }

    /// Set the step pattern
    pub fn with_step_pattern(mut self, step_pattern: DtwStepPattern) -> Self {
        self.step_pattern = step_pattern;
        self
    }

    /// Set normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Compute DTW distance between two time series
    pub fn distance(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        let n = x.len();
        let m = y.len();

        if n == 0 || m == 0 {
            return Float::INFINITY;
        }

        // Create cost matrix
        let mut cost_matrix = Array2::from_elem((n + 1, m + 1), Float::INFINITY);
        cost_matrix[[0, 0]] = 0.0;

        // Fill cost matrix with DTW computation
        for i in 1..=n {
            let window_start = if let Some(w) = self.window {
                std::cmp::max(1, i.saturating_sub(w))
            } else {
                1
            };
            let window_end = if let Some(w) = self.window {
                std::cmp::min(m, i + w)
            } else {
                m
            };

            for j in window_start..=window_end {
                let cost = (x[i - 1] - y[j - 1]).powi(2);

                let previous_cost = match self.step_pattern {
                    DtwStepPattern::Symmetric => {
                        [
                            cost_matrix[[i - 1, j]],     // insertion
                            cost_matrix[[i, j - 1]],     // deletion
                            cost_matrix[[i - 1, j - 1]], // match
                        ]
                        .iter()
                        .fold(Float::INFINITY, |a, &b| a.min(b))
                    }
                    DtwStepPattern::Asymmetric => {
                        [
                            cost_matrix[[i - 1, j]],     // insertion
                            cost_matrix[[i, j - 1]],     // deletion
                            cost_matrix[[i - 1, j - 1]], // match
                        ]
                        .iter()
                        .fold(Float::INFINITY, |a, &b| a.min(b))
                    }
                    DtwStepPattern::QuasiSymmetric => [
                        cost_matrix[[i - 1, j]] + cost,
                        cost_matrix[[i, j - 1]] + cost,
                        cost_matrix[[i - 1, j - 1]] + 2.0 * cost,
                    ]
                    .iter()
                    .fold(Float::INFINITY, |a, &b| a.min(b)),
                };

                cost_matrix[[i, j]] = cost + previous_cost;
            }
        }

        let distance = cost_matrix[[n, m]];

        if self.normalize {
            distance / (n + m) as Float
        } else {
            distance
        }
    }

    /// Compute DTW alignment path
    pub fn alignment_path(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
    ) -> Vec<(usize, usize)> {
        let n = x.len();
        let m = y.len();

        if n == 0 || m == 0 {
            return Vec::new();
        }

        // Compute cost matrix
        let mut cost_matrix = Array2::from_elem((n + 1, m + 1), Float::INFINITY);
        cost_matrix[[0, 0]] = 0.0;

        for i in 1..=n {
            let window_start = if let Some(w) = self.window {
                std::cmp::max(1, i.saturating_sub(w))
            } else {
                1
            };
            let window_end = if let Some(w) = self.window {
                std::cmp::min(m, i + w)
            } else {
                m
            };

            for j in window_start..=window_end {
                let cost = (x[i - 1] - y[j - 1]).powi(2);
                let previous_cost = [
                    cost_matrix[[i - 1, j]],
                    cost_matrix[[i, j - 1]],
                    cost_matrix[[i - 1, j - 1]],
                ]
                .iter()
                .fold(Float::INFINITY, |a, &b| a.min(b));

                cost_matrix[[i, j]] = cost + previous_cost;
            }
        }

        // Backtrack to find alignment path
        let mut path = Vec::new();
        let mut i = n;
        let mut j = m;

        while i > 0 && j > 0 {
            path.push((i - 1, j - 1));

            let costs = [
                (cost_matrix[[i - 1, j - 1]], (i - 1, j - 1)),
                (cost_matrix[[i - 1, j]], (i - 1, j)),
                (cost_matrix[[i, j - 1]], (i, j - 1)),
            ];

            let (_, (new_i, new_j)) = costs
                .iter()
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            i = *new_i;
            j = *new_j;
        }

        path.reverse();
        path
    }
}

/// Time series shapelet for subsequence matching
#[derive(Debug, Clone)]
pub struct Shapelet {
    /// The shapelet time series
    pub data: Array1<Float>,
    /// Starting position in the original time series
    pub start_pos: usize,
    /// Length of the shapelet
    pub length: usize,
    /// Quality score of the shapelet
    pub quality_score: Float,
    /// Class label associated with the shapelet
    pub class_label: Option<usize>,
}

impl Shapelet {
    /// Create a new shapelet
    pub fn new(data: Array1<Float>, start_pos: usize, quality_score: Float) -> Self {
        let length = data.len();
        Self {
            data,
            start_pos,
            length,
            quality_score,
            class_label: None,
        }
    }

    /// Set the class label
    pub fn with_class_label(mut self, class_label: usize) -> Self {
        self.class_label = Some(class_label);
        self
    }

    /// Compute distance to a time series subsequence
    pub fn distance_to_subsequence(&self, subsequence: &ArrayView1<Float>) -> Float {
        if subsequence.len() != self.length {
            return Float::INFINITY;
        }

        self.data
            .iter()
            .zip(subsequence.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<Float>()
            .sqrt()
    }

    /// Find best matching subsequence in a time series
    pub fn best_match(&self, time_series: &ArrayView1<Float>) -> Option<(usize, Float)> {
        if time_series.len() < self.length {
            return None;
        }

        let mut best_distance = Float::INFINITY;
        let mut best_position = 0;

        for i in 0..=(time_series.len() - self.length) {
            let subsequence = time_series.slice(scirs2_core::ndarray::s![i..i + self.length]);
            let distance = self.distance_to_subsequence(&subsequence);

            if distance < best_distance {
                best_distance = distance;
                best_position = i;
            }
        }

        Some((best_position, best_distance))
    }
}

/// Shapelet discovery algorithm
pub struct ShapeletDiscovery {
    min_length: usize,
    max_length: usize,
    max_shapelets: usize,
    quality_threshold: Float,
}

impl ShapeletDiscovery {
    /// Create a new shapelet discovery algorithm
    pub fn new(min_length: usize, max_length: usize) -> Self {
        Self {
            min_length,
            max_length,
            max_shapelets: 100,
            quality_threshold: 0.1,
        }
    }

    /// Set maximum number of shapelets
    pub fn with_max_shapelets(mut self, max_shapelets: usize) -> Self {
        self.max_shapelets = max_shapelets;
        self
    }

    /// Set quality threshold
    pub fn with_quality_threshold(mut self, quality_threshold: Float) -> Self {
        self.quality_threshold = quality_threshold;
        self
    }

    /// Discover shapelets from time series data
    pub fn discover(
        &self,
        data: &ArrayView2<Float>,
        labels: Option<&ArrayView1<usize>>,
    ) -> Vec<Shapelet> {
        let mut shapelets = Vec::new();

        for (series_idx, series) in data.axis_iter(Axis(0)).enumerate() {
            let series_length = series.len();

            // Extract candidate shapelets of different lengths
            for length in self.min_length..=std::cmp::min(self.max_length, series_length) {
                for start in 0..=(series_length - length) {
                    let subsequence = series.slice(scirs2_core::ndarray::s![start..start + length]);
                    let shapelet_data = subsequence.to_owned();

                    // Compute quality score (information gain or similar metric)
                    let quality_score = self.compute_quality_score(&shapelet_data, data, labels);

                    if quality_score > self.quality_threshold {
                        let mut shapelet = Shapelet::new(shapelet_data, start, quality_score);

                        if let Some(labels) = labels {
                            shapelet = shapelet.with_class_label(labels[series_idx]);
                        }

                        shapelets.push(shapelet);
                    }
                }
            }
        }

        // Sort by quality score and keep top shapelets
        shapelets.sort_by(|a, b| {
            b.quality_score
                .partial_cmp(&a.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        shapelets.truncate(self.max_shapelets);
        shapelets
    }

    /// Compute quality score for a shapelet candidate
    fn compute_quality_score(
        &self,
        shapelet: &Array1<Float>,
        data: &ArrayView2<Float>,
        labels: Option<&ArrayView1<usize>>,
    ) -> Float {
        // Compute distances from shapelet to all time series
        let mut distances = Vec::new();

        for series in data.axis_iter(Axis(0)) {
            let mut min_distance = Float::INFINITY;

            if series.len() >= shapelet.len() {
                for i in 0..=(series.len() - shapelet.len()) {
                    let subsequence = series.slice(scirs2_core::ndarray::s![i..i + shapelet.len()]);
                    let distance = shapelet
                        .iter()
                        .zip(subsequence.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<Float>()
                        .sqrt();

                    min_distance = min_distance.min(distance);
                }
            }

            distances.push(min_distance);
        }

        // If no labels provided, use variance as quality measure
        if labels.is_none() {
            let mean = distances.iter().sum::<Float>() / distances.len() as Float;
            let variance = distances.iter().map(|d| (d - mean).powi(2)).sum::<Float>()
                / distances.len() as Float;

            return variance;
        }

        // Compute information gain using class labels
        let labels = labels.unwrap();
        let mut class_distances: HashMap<usize, Vec<Float>> = HashMap::new();

        for (i, &label) in labels.iter().enumerate() {
            class_distances.entry(label).or_default().push(distances[i]);
        }

        // Compute entropy-based quality measure
        let total_count = distances.len() as Float;
        let mut quality = 0.0;

        for (_, class_dists) in class_distances.iter() {
            let class_count = class_dists.len() as Float;
            let class_mean = class_dists.iter().sum::<Float>() / class_count;
            let class_variance = class_dists
                .iter()
                .map(|d| (d - class_mean).powi(2))
                .sum::<Float>()
                / class_count;

            quality += (class_count / total_count) * class_variance;
        }

        1.0 / (1.0 + quality) // Higher quality for lower within-class variance
    }
}

/// Temporal neighbor search for time series
pub struct TemporalNeighborSearch {
    /// Window size for temporal context
    pub window_size: usize,
    /// Distance metric
    pub distance: Distance,
    /// DTW distance calculator
    pub dtw_distance: DtwDistance,
    /// Whether to use DTW or standard distance
    pub use_dtw: bool,
}

impl TemporalNeighborSearch {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            distance: Distance::Euclidean,
            dtw_distance: DtwDistance::new(),
            use_dtw: false,
        }
    }

    /// Set distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Enable DTW distance
    pub fn with_dtw(mut self, dtw_distance: DtwDistance) -> Self {
        self.dtw_distance = dtw_distance;
        self.use_dtw = true;
        self
    }

    /// Find temporal neighbors for a query time series
    pub fn find_neighbors(
        &self,
        query: &ArrayView1<Float>,
        time_series_db: &ArrayView2<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        if time_series_db.nrows() == 0 {
            return Err(NeighborsError::EmptyInput);
        }

        let mut distances = Vec::new();

        for (i, series) in time_series_db.axis_iter(Axis(0)).enumerate() {
            let distance = if self.use_dtw {
                self.dtw_distance.distance(query, &series)
            } else {
                self.distance.calculate(query, &series)
            };

            distances.push((i, distance));
        }

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take k nearest neighbors
        let k = std::cmp::min(k, distances.len());
        let indices: Vec<usize> = distances.iter().take(k).map(|&(idx, _)| idx).collect();
        let neighbor_distances: Vec<Float> =
            distances.iter().take(k).map(|&(_, dist)| dist).collect();

        Ok((indices, neighbor_distances))
    }

    /// Find temporal neighbors within a specific time window
    pub fn find_neighbors_in_window(
        &self,
        query: &ArrayView1<Float>,
        time_series_db: &ArrayView2<Float>,
        timestamps: &ArrayView1<Float>,
        query_time: Float,
        time_window: Float,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        if time_series_db.nrows() == 0 {
            return Err(NeighborsError::EmptyInput);
        }

        if timestamps.len() != time_series_db.nrows() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![time_series_db.nrows()],
                actual: vec![timestamps.len()],
            });
        }

        let mut distances = Vec::new();

        for (i, (series, &timestamp)) in time_series_db
            .axis_iter(Axis(0))
            .zip(timestamps.iter())
            .enumerate()
        {
            // Check if within time window
            if (timestamp - query_time).abs() <= time_window {
                let distance = if self.use_dtw {
                    self.dtw_distance.distance(query, &series)
                } else {
                    self.distance.calculate(query, &series)
                };

                distances.push((i, distance));
            }
        }

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let indices: Vec<usize> = distances.iter().map(|&(idx, _)| idx).collect();
        let neighbor_distances: Vec<Float> = distances.iter().map(|&(_, dist)| dist).collect();

        Ok((indices, neighbor_distances))
    }
}

/// Subsequence similarity search
pub struct SubsequenceSearch {
    /// Minimum subsequence length
    pub min_length: usize,
    /// Maximum subsequence length
    pub max_length: usize,
    /// Distance metric
    pub distance: Distance,
    /// DTW distance calculator
    pub dtw_distance: DtwDistance,
    /// Whether to use DTW
    pub use_dtw: bool,
}

impl SubsequenceSearch {
    pub fn new(min_length: usize, max_length: usize) -> Self {
        Self {
            min_length,
            max_length,
            distance: Distance::Euclidean,
            dtw_distance: DtwDistance::new(),
            use_dtw: false,
        }
    }

    /// Enable DTW distance
    pub fn with_dtw(mut self, dtw_distance: DtwDistance) -> Self {
        self.dtw_distance = dtw_distance;
        self.use_dtw = true;
        self
    }

    /// Find similar subsequences in a time series
    pub fn find_similar_subsequences(
        &self,
        query: &ArrayView1<Float>,
        time_series: &ArrayView1<Float>,
        max_results: usize,
    ) -> Vec<(usize, usize, Float)> {
        let mut results = Vec::new();
        let query_length = query.len();

        // For subsequence search, we compare with subsequences of the same length as the query
        // or use DTW which can handle different lengths
        if self.use_dtw {
            // For DTW, we can search within the specified range
            for length in self.min_length..=std::cmp::min(self.max_length, time_series.len()) {
                for start in 0..=(time_series.len() - length) {
                    let subsequence =
                        time_series.slice(scirs2_core::ndarray::s![start..start + length]);
                    let distance = self.dtw_distance.distance(query, &subsequence);
                    results.push((start, length, distance));
                }
            }
        } else {
            // For regular distance metrics, subsequence length must match query length
            if query_length >= self.min_length
                && query_length <= self.max_length
                && query_length <= time_series.len()
            {
                for start in 0..=(time_series.len() - query_length) {
                    let subsequence =
                        time_series.slice(scirs2_core::ndarray::s![start..start + query_length]);
                    let distance = self.distance.calculate(query, &subsequence);
                    results.push((start, query_length, distance));
                }
            }
        }

        // Sort by distance
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Take top results
        results.truncate(max_results);
        results
    }
}

/// Streaming time series neighbors
pub struct StreamingTimeSeriesNeighbors {
    /// Buffer for storing recent time series
    buffer: VecDeque<Array1<Float>>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Distance metric
    distance: Distance,
    /// DTW distance calculator
    dtw_distance: DtwDistance,
    /// Whether to use DTW
    use_dtw: bool,
    /// Timestamps for buffered series
    timestamps: VecDeque<Float>,
}

impl StreamingTimeSeriesNeighbors {
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            max_buffer_size,
            distance: Distance::Euclidean,
            dtw_distance: DtwDistance::new(),
            use_dtw: false,
            timestamps: VecDeque::new(),
        }
    }

    /// Enable DTW distance
    pub fn with_dtw(mut self, dtw_distance: DtwDistance) -> Self {
        self.dtw_distance = dtw_distance;
        self.use_dtw = true;
        self
    }

    /// Add a new time series to the buffer
    pub fn add_time_series(&mut self, time_series: Array1<Float>, timestamp: Float) {
        self.buffer.push_back(time_series);
        self.timestamps.push_back(timestamp);

        // Remove old entries if buffer is full
        while self.buffer.len() > self.max_buffer_size {
            self.buffer.pop_front();
            self.timestamps.pop_front();
        }
    }

    /// Find neighbors for a query time series
    pub fn find_neighbors(&self, query: &ArrayView1<Float>, k: usize) -> (Vec<usize>, Vec<Float>) {
        if self.buffer.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut distances = Vec::new();

        for (i, series) in self.buffer.iter().enumerate() {
            let distance = if self.use_dtw {
                self.dtw_distance.distance(query, &series.view())
            } else {
                self.distance.calculate(query, &series.view())
            };

            distances.push((i, distance));
        }

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take k nearest neighbors
        let k = std::cmp::min(k, distances.len());
        let indices: Vec<usize> = distances.iter().take(k).map(|&(idx, _)| idx).collect();
        let neighbor_distances: Vec<Float> =
            distances.iter().take(k).map(|&(_, dist)| dist).collect();

        (indices, neighbor_distances)
    }

    /// Find neighbors within a time window
    pub fn find_neighbors_in_window(
        &self,
        query: &ArrayView1<Float>,
        current_time: Float,
        time_window: Float,
    ) -> (Vec<usize>, Vec<Float>) {
        if self.buffer.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut distances = Vec::new();

        for (i, (series, &timestamp)) in self.buffer.iter().zip(self.timestamps.iter()).enumerate()
        {
            // Check if within time window
            if (timestamp - current_time).abs() <= time_window {
                let distance = if self.use_dtw {
                    self.dtw_distance.distance(query, &series.view())
                } else {
                    self.distance.calculate(query, &series.view())
                };

                distances.push((i, distance));
            }
        }

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let indices: Vec<usize> = distances.iter().map(|&(idx, _)| idx).collect();
        let neighbor_distances: Vec<Float> = distances.iter().map(|&(_, dist)| dist).collect();

        (indices, neighbor_distances)
    }

    /// Get buffer statistics
    pub fn get_stats(&self) -> (usize, usize, Option<Float>, Option<Float>) {
        let current_size = self.buffer.len();
        let max_size = self.max_buffer_size;
        let oldest_timestamp = self.timestamps.front().copied();
        let newest_timestamp = self.timestamps.back().copied();

        (current_size, max_size, oldest_timestamp, newest_timestamp)
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.timestamps.clear();
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn create_test_time_series() -> Array1<Float> {
        Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
    }

    fn create_test_time_series_2() -> Array1<Float> {
        Array1::from_vec(vec![2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0])
    }

    #[test]
    fn test_dtw_distance_basic() {
        let dtw = DtwDistance::new();
        let ts1 = create_test_time_series();
        let ts2 = create_test_time_series_2();

        let distance = dtw.distance(&ts1.view(), &ts2.view());
        assert!(distance > 0.0);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_dtw_distance_with_window() {
        let dtw = DtwDistance::new().with_window(2);
        let ts1 = create_test_time_series();
        let ts2 = create_test_time_series_2();

        let distance = dtw.distance(&ts1.view(), &ts2.view());
        assert!(distance > 0.0);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_dtw_alignment_path() {
        let dtw = DtwDistance::new();
        let ts1 = create_test_time_series();
        let ts2 = create_test_time_series_2();

        let path = dtw.alignment_path(&ts1.view(), &ts2.view());
        assert!(!path.is_empty());
        assert!(path.len() >= std::cmp::max(ts1.len(), ts2.len()));
    }

    #[test]
    fn test_shapelet_creation() {
        let data = create_test_time_series();
        let shapelet = Shapelet::new(data.clone(), 0, 0.5);

        assert_eq!(shapelet.data, data);
        assert_eq!(shapelet.start_pos, 0);
        assert_eq!(shapelet.length, data.len());
        assert_eq!(shapelet.quality_score, 0.5);
    }

    #[test]
    fn test_shapelet_best_match() {
        let shapelet_data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let shapelet = Shapelet::new(shapelet_data, 0, 0.5);

        let time_series = create_test_time_series();
        let (position, distance) = shapelet.best_match(&time_series.view()).unwrap();

        assert!(position < time_series.len());
        assert!(distance >= 0.0);
    }

    #[test]
    fn test_shapelet_discovery() {
        let data = Array2::from_shape_vec(
            (2, 10),
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0,
                4.0, 5.0, 4.0, 3.0,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0, 1]);
        let discovery = ShapeletDiscovery::new(3, 5);

        let shapelets = discovery.discover(&data.view(), Some(&labels.view()));
        assert!(!shapelets.is_empty());
    }

    #[test]
    fn test_temporal_neighbor_search() {
        let search = TemporalNeighborSearch::new(5);
        let query = create_test_time_series();
        let db = Array2::from_shape_vec(
            (3, 10),
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0,
                4.0, 5.0, 4.0, 3.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0,
            ],
        )
        .unwrap();

        let (indices, distances) = search.find_neighbors(&query.view(), &db.view(), 2).unwrap();
        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
    }

    #[test]
    fn test_temporal_neighbor_search_with_window() {
        let search = TemporalNeighborSearch::new(5);
        let query = create_test_time_series();
        let db = Array2::from_shape_vec(
            (3, 10),
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0,
                4.0, 5.0, 4.0, 3.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0,
            ],
        )
        .unwrap();
        let timestamps = Array1::from_vec(vec![1.0, 2.0, 10.0]);

        let (indices, distances) = search
            .find_neighbors_in_window(&query.view(), &db.view(), &timestamps.view(), 1.5, 1.0)
            .unwrap();

        assert_eq!(indices.len(), 2); // Only first two should be in window
        assert_eq!(distances.len(), 2);
    }

    #[test]
    fn test_subsequence_search() {
        let search = SubsequenceSearch::new(3, 5);
        let query = Array1::from_vec(vec![2.0, 3.0, 2.0]);
        let time_series = create_test_time_series();

        let results = search.find_similar_subsequences(&query.view(), &time_series.view(), 3);
        assert!(!results.is_empty());

        for (start, length, distance) in results {
            assert!(start + length <= time_series.len());
            assert!(distance >= 0.0);
        }
    }

    #[test]
    fn test_streaming_time_series_neighbors() {
        let mut streaming = StreamingTimeSeriesNeighbors::new(5);

        // Add some time series
        let ts1 = create_test_time_series();
        let ts2 = create_test_time_series_2();

        streaming.add_time_series(ts1.clone(), 1.0);
        streaming.add_time_series(ts2.clone(), 2.0);

        // Find neighbors
        let query = Array1::from_vec(vec![1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 4.5, 3.5, 2.5]);
        let (indices, distances) = streaming.find_neighbors(&query.view(), 2);

        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
    }

    #[test]
    fn test_streaming_neighbors_with_window() {
        let mut streaming = StreamingTimeSeriesNeighbors::new(5);

        let ts1 = create_test_time_series();
        let ts2 = create_test_time_series_2();

        streaming.add_time_series(ts1.clone(), 1.0);
        streaming.add_time_series(ts2.clone(), 5.0);

        let query = Array1::from_vec(vec![1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 4.5, 3.5, 2.5]);
        let (indices, distances) = streaming.find_neighbors_in_window(&query.view(), 2.0, 1.5);

        assert_eq!(indices.len(), 1); // Only first should be in window
        assert_eq!(distances.len(), 1);
    }

    #[test]
    fn test_streaming_buffer_management() {
        let mut streaming = StreamingTimeSeriesNeighbors::new(2);

        // Add more time series than buffer size
        for i in 0..5 {
            let ts = Array1::from_vec(vec![i as Float; 10]);
            streaming.add_time_series(ts, i as Float);
        }

        let (current_size, max_size, _, _) = streaming.get_stats();
        assert_eq!(current_size, 2);
        assert_eq!(max_size, 2);
    }

    #[test]
    fn test_dtw_empty_series() {
        let dtw = DtwDistance::new();
        let empty = Array1::from_vec(vec![]);
        let ts = create_test_time_series();

        let distance = dtw.distance(&empty.view(), &ts.view());
        assert!(distance.is_infinite());
    }

    #[test]
    fn test_dtw_step_patterns() {
        let ts1 = create_test_time_series();
        let ts2 = create_test_time_series_2();

        let symmetric = DtwDistance::new().with_step_pattern(DtwStepPattern::Symmetric);
        let asymmetric = DtwDistance::new().with_step_pattern(DtwStepPattern::Asymmetric);
        let quasi_symmetric = DtwDistance::new().with_step_pattern(DtwStepPattern::QuasiSymmetric);

        let dist1 = symmetric.distance(&ts1.view(), &ts2.view());
        let dist2 = asymmetric.distance(&ts1.view(), &ts2.view());
        let dist3 = quasi_symmetric.distance(&ts1.view(), &ts2.view());

        assert!(dist1.is_finite());
        assert!(dist2.is_finite());
        assert!(dist3.is_finite());
    }
}

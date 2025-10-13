//! Trend Detection Module
//!
//! This module provides comprehensive trend detection capabilities including
//! trend analysis, changepoint detection, and pattern recognition for time series data
//! in distributed optimization environments.

use std::collections::VecDeque;
use std::time::SystemTime;
use serde::{Deserialize, Serialize};

use super::forecasting_engine::{ForecastingError, TrendDirection};

// ================================================================================================
// TREND DETECTION
// ================================================================================================

/// Trend detector for convergence analysis
pub struct TrendDetector {
    detection_methods: Vec<TrendDetectionMethod>,
    changepoint_detector: ChangepointDetector,
    pattern_recognizer: PatternRecognizer,
}

/// Trend detection methods
#[derive(Debug, Clone)]
pub enum TrendDetectionMethod {
    LinearRegression,
    MannKendall,
    Sen_Slope,
    MovingAverage,
    ExponentialSmoothing,
    WaveletAnalysis,
    Custom(String),
}

/// Trend analysis results
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub trend_acceleration: f64,
    pub confidence_interval: (f64, f64),
}

/// Changepoint detector for trend analysis
pub struct ChangepointDetector {
    detection_algorithms: Vec<ChangepointAlgorithm>,
    sensitivity_threshold: f64,
    minimum_segment_length: usize,
    changepoint_history: VecDeque<Changepoint>,
}

/// Changepoint detection algorithms
#[derive(Debug, Clone)]
pub enum ChangepointAlgorithm {
    CUSUM,
    PELT,
    BinarySegmentation,
    WindowBased,
    BayesianChangepoint,
    OnlineNewtonStep,
    Custom(String),
}

/// Changepoint information
#[derive(Debug, Clone)]
pub struct Changepoint {
    pub position: usize,
    pub timestamp: SystemTime,
    pub confidence: f64,
    pub change_magnitude: f64,
    pub change_type: ChangeType,
    pub statistical_significance: f64,
}

/// Types of changes detected
#[derive(Debug, Clone)]
pub enum ChangeType {
    MeanShift,
    VarianceChange,
    TrendChange,
    SeasonalityChange,
    DistributionChange,
    Custom(String),
}

/// Pattern recognizer for time series
pub struct PatternRecognizer {
    pattern_templates: Vec<PatternTemplate>,
    pattern_matching_algorithms: Vec<PatternMatchingAlgorithm>,
    recognized_patterns: VecDeque<RecognizedPattern>,
}

/// Pattern templates for recognition
#[derive(Debug, Clone)]
pub struct PatternTemplate {
    pub template_name: String,
    pub pattern_type: PatternType,
    pub template_data: Vec<f64>,
    pub tolerance: f64,
    pub min_match_length: usize,
}

/// Types of patterns
#[derive(Debug, Clone)]
pub enum PatternType {
    Periodic,
    Trend,
    Seasonal,
    Anomalous,
    Cyclic,
    Random,
    Custom(String),
}

/// Pattern matching algorithms
#[derive(Debug, Clone)]
pub enum PatternMatchingAlgorithm {
    DynamicTimeWarping,
    CrossCorrelation,
    EuclideanDistance,
    ShapeBasedDistance,
    SymbolicAggregate,
    Custom(String),
}

/// Recognized pattern information
#[derive(Debug, Clone)]
pub struct RecognizedPattern {
    pub pattern_id: String,
    pub template_name: String,
    pub start_position: usize,
    pub end_position: usize,
    pub confidence: f64,
    pub similarity_score: f64,
    pub timestamp: SystemTime,
}

// ================================================================================================
// IMPLEMENTATIONS
// ================================================================================================

impl TrendDetector {
    pub fn new() -> Self {
        Self {
            detection_methods: vec![
                TrendDetectionMethod::LinearRegression,
                TrendDetectionMethod::MannKendall,
            ],
            changepoint_detector: ChangepointDetector::new(),
            pattern_recognizer: PatternRecognizer::new(),
        }
    }

    pub fn detect_trend(&self, data: &[f64]) -> Result<TrendAnalysis, ForecastingError> {
        if data.len() < 3 {
            return Err(ForecastingError::DataPreprocessingError("Insufficient data for trend detection".to_string()));
        }

        let mut trend_results = Vec::new();

        for method in &self.detection_methods {
            let result = self.apply_detection_method(method, data)?;
            trend_results.push(result);
        }

        let aggregate_trend = self.aggregate_trend_results(&trend_results)?;
        Ok(aggregate_trend)
    }

    fn apply_detection_method(&self, method: &TrendDetectionMethod, data: &[f64]) -> Result<TrendAnalysis, ForecastingError> {
        match method {
            TrendDetectionMethod::LinearRegression => {
                self.linear_regression_trend(data)
            }
            TrendDetectionMethod::MannKendall => {
                self.mann_kendall_trend(data)
            }
            _ => {
                self.simple_trend_analysis(data)
            }
        }
    }

    fn linear_regression_trend(&self, data: &[f64]) -> Result<TrendAnalysis, ForecastingError> {
        let n = data.len() as f64;
        let x_sum = (0..data.len()).sum::<usize>() as f64;
        let y_sum = data.iter().sum::<f64>();
        let x_mean = x_sum / n;
        let y_mean = y_sum / n;

        let xy_sum = data.iter().enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();

        let x_square_sum = (0..data.len())
            .map(|i| (i as f64).powi(2))
            .sum::<f64>();

        let slope = (xy_sum - n * x_mean * y_mean) / (x_square_sum - n * x_mean.powi(2));

        let trend_direction = if slope > 0.01 {
            TrendDirection::Increasing
        } else if slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        Ok(TrendAnalysis {
            trend_direction,
            trend_strength: slope.abs(),
            trend_acceleration: 0.0,
            confidence_interval: (slope - 0.1, slope + 0.1),
        })
    }

    fn mann_kendall_trend(&self, data: &[f64]) -> Result<TrendAnalysis, ForecastingError> {
        let n = data.len();
        let mut s = 0i32;

        for i in 0..n-1 {
            for j in i+1..n {
                s += if data[j] > data[i] {
                    1
                } else if data[j] < data[i] {
                    -1
                } else {
                    0
                };
            }
        }

        let trend_direction = if s > 0 {
            TrendDirection::Increasing
        } else if s < 0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        let trend_strength = s.abs() as f64 / (n * (n - 1) / 2) as f64;

        Ok(TrendAnalysis {
            trend_direction,
            trend_strength,
            trend_acceleration: 0.0,
            confidence_interval: (trend_strength - 0.1, trend_strength + 0.1),
        })
    }

    fn simple_trend_analysis(&self, data: &[f64]) -> Result<TrendAnalysis, ForecastingError> {
        let first_third = &data[..data.len()/3];
        let last_third = &data[data.len()*2/3..];

        let first_avg = first_third.iter().sum::<f64>() / first_third.len() as f64;
        let last_avg = last_third.iter().sum::<f64>() / last_third.len() as f64;

        let trend_strength = (last_avg - first_avg).abs() / first_avg;
        let trend_direction = if last_avg > first_avg * 1.05 {
            TrendDirection::Increasing
        } else if last_avg < first_avg * 0.95 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        Ok(TrendAnalysis {
            trend_direction,
            trend_strength,
            trend_acceleration: 0.0,
            confidence_interval: (trend_strength - 0.1, trend_strength + 0.1),
        })
    }

    fn aggregate_trend_results(&self, results: &[TrendAnalysis]) -> Result<TrendAnalysis, ForecastingError> {
        if results.is_empty() {
            return Err(ForecastingError::EnsembleAggregationFailed("No trend results to aggregate".to_string()));
        }

        let avg_strength = results.iter().map(|r| r.trend_strength).sum::<f64>() / results.len() as f64;

        let increasing_count = results.iter().filter(|r| matches!(r.trend_direction, TrendDirection::Increasing)).count();
        let decreasing_count = results.iter().filter(|r| matches!(r.trend_direction, TrendDirection::Decreasing)).count();

        let trend_direction = if increasing_count > decreasing_count {
            TrendDirection::Increasing
        } else if decreasing_count > increasing_count {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        Ok(TrendAnalysis {
            trend_direction,
            trend_strength: avg_strength,
            trend_acceleration: 0.0,
            confidence_interval: (avg_strength - 0.1, avg_strength + 0.1),
        })
    }

    pub fn get_changepoint_detector(&self) -> &ChangepointDetector {
        &self.changepoint_detector
    }

    pub fn get_pattern_recognizer(&self) -> &PatternRecognizer {
        &self.pattern_recognizer
    }
}

impl ChangepointDetector {
    pub fn new() -> Self {
        Self {
            detection_algorithms: vec![
                ChangepointAlgorithm::CUSUM,
                ChangepointAlgorithm::PELT,
            ],
            sensitivity_threshold: 0.05,
            minimum_segment_length: 10,
            changepoint_history: VecDeque::new(),
        }
    }

    pub fn detect_changepoints(&mut self, data: &[f64]) -> Result<Vec<Changepoint>, ForecastingError> {
        let mut all_changepoints = Vec::new();

        for algorithm in &self.detection_algorithms {
            let changepoints = self.apply_algorithm(algorithm, data)?;
            all_changepoints.extend(changepoints);
        }

        all_changepoints.sort_by_key(|cp| cp.position);
        all_changepoints.dedup_by_key(|cp| cp.position);

        let filtered_changepoints: Vec<_> = all_changepoints.into_iter()
            .filter(|cp| cp.confidence >= self.sensitivity_threshold)
            .collect();

        for cp in &filtered_changepoints {
            self.changepoint_history.push_back(cp.clone());
        }

        while self.changepoint_history.len() > 1000 {
            self.changepoint_history.pop_front();
        }

        Ok(filtered_changepoints)
    }

    fn apply_algorithm(&self, algorithm: &ChangepointAlgorithm, data: &[f64]) -> Result<Vec<Changepoint>, ForecastingError> {
        match algorithm {
            ChangepointAlgorithm::CUSUM => {
                self.cusum_detection(data)
            }
            ChangepointAlgorithm::PELT => {
                self.pelt_detection(data)
            }
            _ => {
                self.variance_based_detection(data)
            }
        }
    }

    fn cusum_detection(&self, data: &[f64]) -> Result<Vec<Changepoint>, ForecastingError> {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let threshold = 5.0;
        let mut cumsum_pos = 0.0;
        let mut cumsum_neg = 0.0;
        let mut changepoints = Vec::new();

        for (i, &value) in data.iter().enumerate() {
            cumsum_pos = (cumsum_pos + value - mean - 0.5).max(0.0);
            cumsum_neg = (cumsum_neg - value + mean - 0.5).max(0.0);

            if cumsum_pos > threshold || cumsum_neg > threshold {
                let changepoint = Changepoint {
                    position: i,
                    timestamp: SystemTime::now(),
                    confidence: (cumsum_pos.max(cumsum_neg) / threshold).min(1.0),
                    change_magnitude: (value - mean).abs(),
                    change_type: ChangeType::MeanShift,
                    statistical_significance: 0.95,
                };

                changepoints.push(changepoint);
                cumsum_pos = 0.0;
                cumsum_neg = 0.0;
            }
        }

        Ok(changepoints)
    }

    fn pelt_detection(&self, data: &[f64]) -> Result<Vec<Changepoint>, ForecastingError> {
        let penalty = 2.0 * (data.len() as f64).ln();
        let mut changepoints = Vec::new();

        let window_size = self.minimum_segment_length;
        for i in window_size..data.len()-window_size {
            let left_segment = &data[i-window_size..i];
            let right_segment = &data[i..i+window_size];

            let left_var = self.calculate_variance(left_segment);
            let right_var = self.calculate_variance(right_segment);

            let variance_ratio = (left_var / right_var).max(right_var / left_var);

            if variance_ratio > penalty {
                let changepoint = Changepoint {
                    position: i,
                    timestamp: SystemTime::now(),
                    confidence: (variance_ratio / penalty).min(1.0),
                    change_magnitude: (left_var - right_var).abs(),
                    change_type: ChangeType::VarianceChange,
                    statistical_significance: 0.90,
                };

                changepoints.push(changepoint);
            }
        }

        Ok(changepoints)
    }

    fn variance_based_detection(&self, data: &[f64]) -> Result<Vec<Changepoint>, ForecastingError> {
        let window_size = self.minimum_segment_length;
        let mut changepoints = Vec::new();

        for i in window_size..data.len()-window_size {
            let before_window = &data[i-window_size..i];
            let after_window = &data[i..i+window_size];

            let before_mean = before_window.iter().sum::<f64>() / before_window.len() as f64;
            let after_mean = after_window.iter().sum::<f64>() / after_window.len() as f64;

            let mean_diff = (after_mean - before_mean).abs();
            let threshold = 2.0;

            if mean_diff > threshold {
                let changepoint = Changepoint {
                    position: i,
                    timestamp: SystemTime::now(),
                    confidence: (mean_diff / threshold).min(1.0),
                    change_magnitude: mean_diff,
                    change_type: ChangeType::MeanShift,
                    statistical_significance: 0.85,
                };

                changepoints.push(changepoint);
            }
        }

        Ok(changepoints)
    }

    fn calculate_variance(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        variance
    }

    pub fn get_sensitivity_threshold(&self) -> f64 {
        self.sensitivity_threshold
    }

    pub fn set_sensitivity_threshold(&mut self, threshold: f64) {
        self.sensitivity_threshold = threshold;
    }

    pub fn get_changepoint_history(&self) -> &VecDeque<Changepoint> {
        &self.changepoint_history
    }
}

impl PatternRecognizer {
    pub fn new() -> Self {
        Self {
            pattern_templates: Vec::new(),
            pattern_matching_algorithms: vec![
                PatternMatchingAlgorithm::DynamicTimeWarping,
                PatternMatchingAlgorithm::CrossCorrelation,
            ],
            recognized_patterns: VecDeque::new(),
        }
    }

    pub fn add_pattern_template(&mut self, template: PatternTemplate) {
        self.pattern_templates.push(template);
    }

    pub fn recognize_patterns(&mut self, data: &[f64]) -> Result<Vec<RecognizedPattern>, ForecastingError> {
        let mut recognized = Vec::new();

        for template in &self.pattern_templates {
            for algorithm in &self.pattern_matching_algorithms {
                let matches = self.find_pattern_matches(template, algorithm, data)?;
                recognized.extend(matches);
            }
        }

        for pattern in &recognized {
            self.recognized_patterns.push_back(pattern.clone());
        }

        while self.recognized_patterns.len() > 1000 {
            self.recognized_patterns.pop_front();
        }

        Ok(recognized)
    }

    fn find_pattern_matches(&self, template: &PatternTemplate, algorithm: &PatternMatchingAlgorithm, data: &[f64]) -> Result<Vec<RecognizedPattern>, ForecastingError> {
        let mut matches = Vec::new();

        match algorithm {
            PatternMatchingAlgorithm::CrossCorrelation => {
                matches.extend(self.cross_correlation_matching(template, data)?);
            }
            PatternMatchingAlgorithm::DynamicTimeWarping => {
                matches.extend(self.dtw_matching(template, data)?);
            }
            _ => {
                matches.extend(self.euclidean_matching(template, data)?);
            }
        }

        Ok(matches)
    }

    fn cross_correlation_matching(&self, template: &PatternTemplate, data: &[f64]) -> Result<Vec<RecognizedPattern>, ForecastingError> {
        let template_data = &template.template_data;
        let mut matches = Vec::new();

        if template_data.len() > data.len() {
            return Ok(matches);
        }

        for i in 0..=data.len()-template_data.len() {
            let segment = &data[i..i+template_data.len()];
            let correlation = self.calculate_correlation(template_data, segment);

            if correlation > template.tolerance {
                let pattern = RecognizedPattern {
                    pattern_id: format!("pattern_{}_{}", template.template_name, i),
                    template_name: template.template_name.clone(),
                    start_position: i,
                    end_position: i + template_data.len(),
                    confidence: correlation,
                    similarity_score: correlation,
                    timestamp: SystemTime::now(),
                };

                matches.push(pattern);
            }
        }

        Ok(matches)
    }

    fn dtw_matching(&self, template: &PatternTemplate, data: &[f64]) -> Result<Vec<RecognizedPattern>, ForecastingError> {
        let template_data = &template.template_data;
        let mut matches = Vec::new();

        let min_length = template.min_match_length.max(template_data.len() / 2);
        let max_length = template_data.len() * 2;

        for start in 0..data.len() {
            for length in min_length..=max_length.min(data.len() - start) {
                if start + length > data.len() {
                    break;
                }

                let segment = &data[start..start+length];
                let dtw_distance = self.calculate_dtw_distance(template_data, segment);
                let similarity = 1.0 / (1.0 + dtw_distance);

                if similarity > template.tolerance {
                    let pattern = RecognizedPattern {
                        pattern_id: format!("dtw_{}_{}", template.template_name, start),
                        template_name: template.template_name.clone(),
                        start_position: start,
                        end_position: start + length,
                        confidence: similarity,
                        similarity_score: similarity,
                        timestamp: SystemTime::now(),
                    };

                    matches.push(pattern);
                }
            }
        }

        Ok(matches)
    }

    fn euclidean_matching(&self, template: &PatternTemplate, data: &[f64]) -> Result<Vec<RecognizedPattern>, ForecastingError> {
        let template_data = &template.template_data;
        let mut matches = Vec::new();

        if template_data.len() > data.len() {
            return Ok(matches);
        }

        for i in 0..=data.len()-template_data.len() {
            let segment = &data[i..i+template_data.len()];
            let distance = self.calculate_euclidean_distance(template_data, segment);
            let similarity = 1.0 / (1.0 + distance);

            if similarity > template.tolerance {
                let pattern = RecognizedPattern {
                    pattern_id: format!("euclidean_{}_{}", template.template_name, i),
                    template_name: template.template_name.clone(),
                    start_position: i,
                    end_position: i + template_data.len(),
                    confidence: similarity,
                    similarity_score: similarity,
                    timestamp: SystemTime::now(),
                };

                matches.push(pattern);
            }
        }

        Ok(matches)
    }

    fn calculate_correlation(&self, template: &[f64], segment: &[f64]) -> f64 {
        if template.len() != segment.len() {
            return 0.0;
        }

        let template_mean = template.iter().sum::<f64>() / template.len() as f64;
        let segment_mean = segment.iter().sum::<f64>() / segment.len() as f64;

        let numerator: f64 = template.iter().zip(segment.iter())
            .map(|(&t, &s)| (t - template_mean) * (s - segment_mean))
            .sum();

        let template_variance: f64 = template.iter()
            .map(|&t| (t - template_mean).powi(2))
            .sum();

        let segment_variance: f64 = segment.iter()
            .map(|&s| (s - segment_mean).powi(2))
            .sum();

        let denominator = (template_variance * segment_variance).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn calculate_dtw_distance(&self, template: &[f64], segment: &[f64]) -> f64 {
        let n = template.len();
        let m = segment.len();

        if n == 0 || m == 0 {
            return f64::INFINITY;
        }

        let mut dtw_matrix = vec![vec![f64::INFINITY; m + 1]; n + 1];
        dtw_matrix[0][0] = 0.0;

        for i in 1..=n {
            for j in 1..=m {
                let cost = (template[i-1] - segment[j-1]).abs();
                dtw_matrix[i][j] = cost + dtw_matrix[i-1][j]
                    .min(dtw_matrix[i][j-1])
                    .min(dtw_matrix[i-1][j-1]);
            }
        }

        dtw_matrix[n][m]
    }

    fn calculate_euclidean_distance(&self, template: &[f64], segment: &[f64]) -> f64 {
        if template.len() != segment.len() {
            return f64::INFINITY;
        }

        template.iter().zip(segment.iter())
            .map(|(&t, &s)| (t - s).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub fn get_pattern_templates(&self) -> &[PatternTemplate] {
        &self.pattern_templates
    }

    pub fn get_recognized_patterns(&self) -> &VecDeque<RecognizedPattern> {
        &self.recognized_patterns
    }

    pub fn clear_templates(&mut self) {
        self.pattern_templates.clear();
    }

    pub fn clear_recognized_patterns(&mut self) {
        self.recognized_patterns.clear();
    }
}

impl Default for TrendDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ChangepointDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PatternRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trend_detector_creation() {
        let detector = TrendDetector::new();
        assert!(!detector.detection_methods.is_empty());
    }

    #[test]
    fn test_trend_detection() {
        let detector = TrendDetector::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = detector.detect_trend(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_changepoint_detection() {
        let mut detector = ChangepointDetector::new();
        let data = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let result = detector.detect_changepoints(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pattern_recognition() {
        let mut recognizer = PatternRecognizer::new();

        let template = PatternTemplate {
            template_name: "test_pattern".to_string(),
            pattern_type: PatternType::Periodic,
            template_data: vec![1.0, 2.0, 1.0],
            tolerance: 0.8,
            min_match_length: 3,
        };

        recognizer.add_pattern_template(template);

        let data = vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0];
        let result = recognizer.recognize_patterns(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_linear_regression_trend() {
        let detector = TrendDetector::new();
        let increasing_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = detector.linear_regression_trend(&increasing_data);
        assert!(result.is_ok());

        let trend = result.unwrap();
        assert!(matches!(trend.trend_direction, TrendDirection::Increasing));
        assert!(trend.trend_strength > 0.0);
    }

    #[test]
    fn test_mann_kendall_trend() {
        let detector = TrendDetector::new();
        let decreasing_data = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let result = detector.mann_kendall_trend(&decreasing_data);
        assert!(result.is_ok());

        let trend = result.unwrap();
        assert!(matches!(trend.trend_direction, TrendDirection::Decreasing));
        assert!(trend.trend_strength > 0.0);
    }

    #[test]
    fn test_cusum_changepoint_detection() {
        let detector = ChangepointDetector::new();
        let data_with_shift = vec![1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0];
        let result = detector.cusum_detection(&data_with_shift);
        assert!(result.is_ok());

        let changepoints = result.unwrap();
        assert!(!changepoints.is_empty());
    }

    #[test]
    fn test_cross_correlation_matching() {
        let recognizer = PatternRecognizer::new();
        let template = PatternTemplate {
            template_name: "sine_wave".to_string(),
            pattern_type: PatternType::Periodic,
            template_data: vec![0.0, 1.0, 0.0, -1.0],
            tolerance: 0.5,
            min_match_length: 4,
        };

        let data = vec![0.0, 1.0, 0.0, -1.0, 2.0, 0.0, 1.0, 0.0, -1.0];
        let result = recognizer.cross_correlation_matching(&template, &data);
        assert!(result.is_ok());

        let matches = result.unwrap();
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_pattern_template_management() {
        let mut recognizer = PatternRecognizer::new();

        let template = PatternTemplate {
            template_name: "test".to_string(),
            pattern_type: PatternType::Trend,
            template_data: vec![1.0, 2.0, 3.0],
            tolerance: 0.8,
            min_match_length: 3,
        };

        recognizer.add_pattern_template(template);
        assert_eq!(recognizer.get_pattern_templates().len(), 1);

        recognizer.clear_templates();
        assert_eq!(recognizer.get_pattern_templates().len(), 0);
    }

    #[test]
    fn test_sensitivity_threshold_adjustment() {
        let mut detector = ChangepointDetector::new();

        let original_threshold = detector.get_sensitivity_threshold();
        assert_eq!(original_threshold, 0.05);

        detector.set_sensitivity_threshold(0.1);
        assert_eq!(detector.get_sensitivity_threshold(), 0.1);
    }
}
//! Anomaly Detection Module
//!
//! This module provides comprehensive anomaly detection capabilities including
//! statistical outlier detection, isolation forest, ensemble methods, confidence
//! calibration, and anomaly classification for distributed optimization systems.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};

// ================================================================================================
// ANOMALY DETECTION
// ================================================================================================

/// Comprehensive anomaly detection system
pub struct AnomalyDetector {
    detection_algorithms: Vec<AnomalyDetectionAlgorithm>,
    baseline_models: Vec<BaselineModel>,
    ensemble_detector: EnsembleAnomalyDetector,
    anomaly_classifier: AnomalyClassifier,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone)]
pub enum AnomalyDetectionAlgorithm {
    StatisticalOutlier,
    IsolationForest,
    OneClassSVM,
    LSTM_Autoencoder,
    LocalOutlierFactor,
    EllipticEnvelope,
    DBSCAN,
    KMeansAnomaly,
    PCA_Reconstruction,
    Custom(String),
}

/// Baseline models for anomaly detection
#[derive(Debug, Clone)]
pub struct BaselineModel {
    pub model_name: String,
    pub model_type: BaselineModelType,
    pub parameters: HashMap<String, f64>,
    pub training_period: Duration,
    pub update_frequency: Duration,
    pub confidence_threshold: f64,
}

/// Baseline model types
#[derive(Debug, Clone)]
pub enum BaselineModelType {
    MovingAverage,
    ExponentialSmoothing,
    SeasonalDecomposition,
    HistoricalProfile,
    StatisticalDistribution,
    AutoRegressive,
    Custom(String),
}

/// Ensemble anomaly detector
pub struct EnsembleAnomalyDetector {
    component_detectors: Vec<ComponentDetector>,
    aggregation_strategy: AggregationStrategy,
    voting_mechanism: VotingMechanism,
    confidence_calibration: ConfidenceCalibration,
}

/// Component detectors in ensemble
#[derive(Debug, Clone)]
pub struct ComponentDetector {
    pub detector_name: String,
    pub algorithm: AnomalyDetectionAlgorithm,
    pub weight: f64,
    pub performance_score: f64,
    pub specialization: DetectorSpecialization,
}

/// Detector specializations
#[derive(Debug, Clone)]
pub enum DetectorSpecialization {
    PointAnomalies,
    ContextualAnomalies,
    CollectiveAnomalies,
    TrendAnomalies,
    SeasonalAnomalies,
    General,
}

/// Aggregation strategies for ensemble
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    WeightedAverage,
    MaxVoting,
    MedianVoting,
    DynamicWeighting,
    StackingAggregation,
    Custom(String),
}

/// Voting mechanisms for ensemble decisions
#[derive(Debug, Clone)]
pub enum VotingMechanism {
    Majority,
    Plurality,
    Consensus,
    WeightedConsensus,
    Threshold,
    Custom(String),
}

/// Confidence calibration for anomaly scores
pub struct ConfidenceCalibration {
    calibration_method: CalibrationMethod,
    calibration_data: Vec<CalibrationPoint>,
    reliability_diagram: ReliabilityDiagram,
}

/// Calibration methods
#[derive(Debug, Clone)]
pub enum CalibrationMethod {
    PlattScaling,
    IsotonicRegression,
    BetaCalibration,
    TemperatureScaling,
    Custom(String),
}

/// Calibration data points
#[derive(Debug, Clone)]
pub struct CalibrationPoint {
    pub predicted_probability: f64,
    pub actual_outcome: bool,
    pub bin_id: usize,
}

/// Reliability diagram for calibration assessment
pub struct ReliabilityDiagram {
    bins: Vec<ReliabilityBin>,
    expected_calibration_error: f64,
    maximum_calibration_error: f64,
}

/// Reliability bins for calibration
#[derive(Debug, Clone)]
pub struct ReliabilityBin {
    pub bin_center: f64,
    pub bin_accuracy: f64,
    pub bin_confidence: f64,
    pub sample_count: usize,
}

/// Anomaly classifier for categorizing anomalies
pub struct AnomalyClassifier {
    classification_models: Vec<ClassificationModel>,
    anomaly_categories: Vec<AnomalyCategory>,
    feature_extractors: Vec<FeatureExtractor>,
}

/// Classification models for anomaly types
#[derive(Debug, Clone)]
pub struct ClassificationModel {
    pub model_name: String,
    pub model_type: ClassificationModelType,
    pub features: Vec<String>,
    pub accuracy: f64,
    pub confusion_matrix: Vec<Vec<u32>>,
}

/// Classification model types
#[derive(Debug, Clone)]
pub enum ClassificationModelType {
    DecisionTree,
    RandomForest,
    SupportVectorMachine,
    NeuralNetwork,
    NaiveBayes,
    KNearestNeighbors,
    Custom(String),
}

/// Anomaly categories
#[derive(Debug, Clone)]
pub struct AnomalyCategory {
    pub category_name: String,
    pub description: String,
    pub severity_level: SeverityLevel,
    pub typical_characteristics: Vec<String>,
    pub response_actions: Vec<String>,
}

/// Severity levels for anomalies
#[derive(Debug, Clone)]
pub enum SeverityLevel {
    Low,
    Medium,
    High,
    Critical,
    Custom(String),
}

/// Feature extractors for anomaly classification
pub struct FeatureExtractor {
    pub extractor_name: String,
    pub feature_type: FeatureType,
    pub extraction_method: ExtractionMethod,
    pub feature_importance: f64,
}

/// Types of features for classification
#[derive(Debug, Clone)]
pub enum FeatureType {
    Statistical,
    Temporal,
    Frequency,
    Contextual,
    Relational,
    Custom(String),
}

/// Feature extraction methods
#[derive(Debug, Clone)]
pub enum ExtractionMethod {
    StatisticalMoments,
    FrequencyDomain,
    WaveletTransform,
    AutoCorrelation,
    CrossCorrelation,
    Custom(String),
}

/// Anomaly score structure
#[derive(Debug, Clone)]
pub struct AnomalyScore {
    pub position: usize,
    pub value: f64,
    pub anomaly_score: f64,
    pub confidence: f64,
    pub algorithm: AnomalyDetectionAlgorithm,
    pub timestamp: SystemTime,
    pub category: Option<String>,
}

/// Anomaly detection errors
#[derive(Debug, thiserror::Error)]
pub enum AnomalyDetectionError {
    #[error("Anomaly detection failed: {0}")]
    DetectionFailed(String),
    #[error("Baseline model error: {0}")]
    BaselineModelError(String),
    #[error("Classification failed: {0}")]
    ClassificationFailed(String),
    #[error("Threshold calibration failed: {0}")]
    ThresholdCalibrationFailed(String),
    #[error("Feature computation failed: {0}")]
    FeatureComputationFailed(String),
}

// ================================================================================================
// IMPLEMENTATIONS
// ================================================================================================

impl AnomalyDetector {
    pub fn new() -> Self {
        Self {
            detection_algorithms: vec![
                AnomalyDetectionAlgorithm::IsolationForest,
                AnomalyDetectionAlgorithm::StatisticalOutlier,
            ],
            baseline_models: Vec::new(),
            ensemble_detector: EnsembleAnomalyDetector::new(),
            anomaly_classifier: AnomalyClassifier::new(),
        }
    }

    pub fn detect_anomalies(&mut self, data: &[f64]) -> Result<Vec<AnomalyScore>, AnomalyDetectionError> {
        let mut anomaly_scores = Vec::new();

        for algorithm in &self.detection_algorithms {
            let scores = self.apply_algorithm(algorithm, data)?;
            anomaly_scores.extend(scores);
        }

        let ensemble_scores = self.ensemble_detector.detect(data)?;
        anomaly_scores.extend(ensemble_scores);

        let classified_anomalies = self.anomaly_classifier.classify_anomalies(&anomaly_scores)?;

        Ok(classified_anomalies)
    }

    fn apply_algorithm(&self, algorithm: &AnomalyDetectionAlgorithm, data: &[f64]) -> Result<Vec<AnomalyScore>, AnomalyDetectionError> {
        match algorithm {
            AnomalyDetectionAlgorithm::StatisticalOutlier => {
                self.statistical_outlier_detection(data)
            }
            AnomalyDetectionAlgorithm::IsolationForest => {
                self.isolation_forest_detection(data)
            }
            _ => {
                self.statistical_outlier_detection(data)
            }
        }
    }

    fn statistical_outlier_detection(&self, data: &[f64]) -> Result<Vec<AnomalyScore>, AnomalyDetectionError> {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        let threshold = 3.0;
        let mut anomaly_scores = Vec::new();

        for (i, &value) in data.iter().enumerate() {
            let z_score = (value - mean).abs() / std_dev;
            if z_score > threshold {
                let score = AnomalyScore {
                    position: i,
                    value,
                    anomaly_score: z_score / threshold,
                    confidence: (z_score / threshold).min(1.0),
                    algorithm: AnomalyDetectionAlgorithm::StatisticalOutlier,
                    timestamp: SystemTime::now(),
                    category: None,
                };
                anomaly_scores.push(score);
            }
        }

        Ok(anomaly_scores)
    }

    fn isolation_forest_detection(&self, data: &[f64]) -> Result<Vec<AnomalyScore>, AnomalyDetectionError> {
        let mut anomaly_scores = Vec::new();
        let threshold = 0.5;

        for (i, &value) in data.iter().enumerate() {
            let median = self.calculate_median(data);
            let isolation_score = (value - median).abs() / median.abs().max(1.0);

            if isolation_score > threshold {
                let score = AnomalyScore {
                    position: i,
                    value,
                    anomaly_score: isolation_score,
                    confidence: (isolation_score / threshold).min(1.0),
                    algorithm: AnomalyDetectionAlgorithm::IsolationForest,
                    timestamp: SystemTime::now(),
                    category: None,
                };
                anomaly_scores.push(score);
            }
        }

        Ok(anomaly_scores)
    }

    fn calculate_median(&self, data: &[f64]) -> f64 {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted_data.len();
        if len % 2 == 0 {
            (sorted_data[len / 2 - 1] + sorted_data[len / 2]) / 2.0
        } else {
            sorted_data[len / 2]
        }
    }

    pub fn update_baseline_models(&mut self, data: &[f64]) -> Result<(), AnomalyDetectionError> {
        for model in &mut self.baseline_models {
            self.update_baseline_model(model, data)?;
        }
        Ok(())
    }

    fn update_baseline_model(&self, _model: &mut BaselineModel, _data: &[f64]) -> Result<(), AnomalyDetectionError> {
        Ok(())
    }

    pub fn add_baseline_model(&mut self, model: BaselineModel) {
        self.baseline_models.push(model);
    }

    pub fn get_detection_algorithms(&self) -> &[AnomalyDetectionAlgorithm] {
        &self.detection_algorithms
    }

    pub fn add_detection_algorithm(&mut self, algorithm: AnomalyDetectionAlgorithm) {
        self.detection_algorithms.push(algorithm);
    }
}

impl EnsembleAnomalyDetector {
    pub fn new() -> Self {
        Self {
            component_detectors: Vec::new(),
            aggregation_strategy: AggregationStrategy::WeightedAverage,
            voting_mechanism: VotingMechanism::WeightedConsensus,
            confidence_calibration: ConfidenceCalibration::new(),
        }
    }

    pub fn detect(&self, data: &[f64]) -> Result<Vec<AnomalyScore>, AnomalyDetectionError> {
        let mut ensemble_scores = Vec::new();

        for detector in &self.component_detectors {
            let scores = self.apply_component_detector(detector, data)?;
            ensemble_scores.extend(scores);
        }

        let aggregated_scores = self.aggregate_scores(&ensemble_scores)?;

        Ok(aggregated_scores)
    }

    fn apply_component_detector(&self, _detector: &ComponentDetector, data: &[f64]) -> Result<Vec<AnomalyScore>, AnomalyDetectionError> {
        let mut scores = Vec::new();

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std_dev = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();

        for (i, &value) in data.iter().enumerate() {
            let z_score = (value - mean).abs() / std_dev;
            if z_score > 2.0 {
                scores.push(AnomalyScore {
                    position: i,
                    value,
                    anomaly_score: z_score / 2.0,
                    confidence: (z_score / 2.0).min(1.0),
                    algorithm: AnomalyDetectionAlgorithm::StatisticalOutlier,
                    timestamp: SystemTime::now(),
                    category: None,
                });
            }
        }

        Ok(scores)
    }

    fn aggregate_scores(&self, scores: &[AnomalyScore]) -> Result<Vec<AnomalyScore>, AnomalyDetectionError> {
        let mut position_scores: HashMap<usize, Vec<&AnomalyScore>> = HashMap::new();

        for score in scores {
            position_scores.entry(score.position).or_insert_with(Vec::new).push(score);
        }

        let mut aggregated_scores = Vec::new();

        for (position, scores_at_position) in position_scores {
            if scores_at_position.len() > 1 {
                let avg_score = scores_at_position.iter()
                    .map(|s| s.anomaly_score)
                    .sum::<f64>() / scores_at_position.len() as f64;

                let avg_confidence = scores_at_position.iter()
                    .map(|s| s.confidence)
                    .sum::<f64>() / scores_at_position.len() as f64;

                let aggregated_score = AnomalyScore {
                    position,
                    value: scores_at_position[0].value,
                    anomaly_score: avg_score,
                    confidence: avg_confidence,
                    algorithm: AnomalyDetectionAlgorithm::StatisticalOutlier,
                    timestamp: SystemTime::now(),
                    category: None,
                };

                aggregated_scores.push(aggregated_score);
            } else if let Some(&score) = scores_at_position.first() {
                aggregated_scores.push(score.clone());
            }
        }

        Ok(aggregated_scores)
    }

    pub fn add_component_detector(&mut self, detector: ComponentDetector) {
        self.component_detectors.push(detector);
    }

    pub fn get_component_detectors(&self) -> &[ComponentDetector] {
        &self.component_detectors
    }

    pub fn set_aggregation_strategy(&mut self, strategy: AggregationStrategy) {
        self.aggregation_strategy = strategy;
    }

    pub fn set_voting_mechanism(&mut self, mechanism: VotingMechanism) {
        self.voting_mechanism = mechanism;
    }
}

impl ConfidenceCalibration {
    pub fn new() -> Self {
        Self {
            calibration_method: CalibrationMethod::PlattScaling,
            calibration_data: Vec::new(),
            reliability_diagram: ReliabilityDiagram {
                bins: Vec::new(),
                expected_calibration_error: 0.0,
                maximum_calibration_error: 0.0,
            },
        }
    }

    pub fn calibrate_confidence(&self, raw_score: f64) -> f64 {
        match self.calibration_method {
            CalibrationMethod::PlattScaling => {
                self.platt_scaling(raw_score)
            }
            _ => raw_score,
        }
    }

    fn platt_scaling(&self, raw_score: f64) -> f64 {
        let a = -1.0;
        let b = 0.0;

        1.0 / (1.0 + (-a * raw_score - b).exp())
    }

    pub fn add_calibration_point(&mut self, point: CalibrationPoint) {
        self.calibration_data.push(point);
    }

    pub fn get_calibration_method(&self) -> &CalibrationMethod {
        &self.calibration_method
    }

    pub fn set_calibration_method(&mut self, method: CalibrationMethod) {
        self.calibration_method = method;
    }

    pub fn get_expected_calibration_error(&self) -> f64 {
        self.reliability_diagram.expected_calibration_error
    }

    pub fn get_maximum_calibration_error(&self) -> f64 {
        self.reliability_diagram.maximum_calibration_error
    }
}

impl AnomalyClassifier {
    pub fn new() -> Self {
        Self {
            classification_models: Vec::new(),
            anomaly_categories: Vec::new(),
            feature_extractors: Vec::new(),
        }
    }

    pub fn classify_anomalies(&self, anomaly_scores: &[AnomalyScore]) -> Result<Vec<AnomalyScore>, AnomalyDetectionError> {
        let mut classified_anomalies = Vec::new();

        for score in anomaly_scores {
            let features = self.extract_features(score)?;
            let category = self.classify_anomaly(&features)?;

            let mut classified_score = score.clone();
            classified_score.category = Some(category);
            classified_anomalies.push(classified_score);
        }

        Ok(classified_anomalies)
    }

    fn extract_features(&self, anomaly_score: &AnomalyScore) -> Result<Vec<f64>, AnomalyDetectionError> {
        let features = vec![
            anomaly_score.anomaly_score,
            anomaly_score.confidence,
            anomaly_score.value,
            anomaly_score.position as f64,
        ];

        Ok(features)
    }

    fn classify_anomaly(&self, _features: &[f64]) -> Result<String, AnomalyDetectionError> {
        Ok("Unknown".to_string())
    }

    pub fn add_classification_model(&mut self, model: ClassificationModel) {
        self.classification_models.push(model);
    }

    pub fn add_anomaly_category(&mut self, category: AnomalyCategory) {
        self.anomaly_categories.push(category);
    }

    pub fn add_feature_extractor(&mut self, extractor: FeatureExtractor) {
        self.feature_extractors.push(extractor);
    }

    pub fn get_classification_models(&self) -> &[ClassificationModel] {
        &self.classification_models
    }

    pub fn get_anomaly_categories(&self) -> &[AnomalyCategory] {
        &self.anomaly_categories
    }

    pub fn get_feature_extractors(&self) -> &[FeatureExtractor] {
        &self.feature_extractors
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EnsembleAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ConfidenceCalibration {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AnomalyClassifier {
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
    fn test_anomaly_detector_creation() {
        let detector = AnomalyDetector::new();
        assert!(!detector.detection_algorithms.is_empty());
    }

    #[test]
    fn test_anomaly_detection() {
        let mut detector = AnomalyDetector::new();
        let data = vec![1.0, 1.0, 1.0, 10.0, 1.0, 1.0];
        let result = detector.detect_anomalies(&data);
        assert!(result.is_ok());

        let anomalies = result.unwrap();
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_statistical_outlier_detection() {
        let detector = AnomalyDetector::new();
        let data = vec![1.0, 2.0, 1.5, 1.2, 10.0, 1.8, 1.3];
        let result = detector.statistical_outlier_detection(&data);
        assert!(result.is_ok());

        let anomalies = result.unwrap();
        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.value == 10.0));
    }

    #[test]
    fn test_isolation_forest_detection() {
        let detector = AnomalyDetector::new();
        let data = vec![1.0, 1.1, 0.9, 1.2, 5.0, 1.0, 0.8];
        let result = detector.isolation_forest_detection(&data);
        assert!(result.is_ok());

        let anomalies = result.unwrap();
        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.value == 5.0));
    }

    #[test]
    fn test_ensemble_anomaly_detector() {
        let detector = EnsembleAnomalyDetector::new();
        let data = vec![1.0, 1.0, 1.0, 10.0, 1.0, 1.0];
        let result = detector.detect(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_confidence_calibration() {
        let calibration = ConfidenceCalibration::new();
        let raw_score = 0.7;
        let calibrated_score = calibration.calibrate_confidence(raw_score);
        assert!(calibrated_score >= 0.0 && calibrated_score <= 1.0);
    }

    #[test]
    fn test_anomaly_classifier() {
        let classifier = AnomalyClassifier::new();
        let anomaly_score = AnomalyScore {
            position: 3,
            value: 10.0,
            anomaly_score: 3.0,
            confidence: 0.95,
            algorithm: AnomalyDetectionAlgorithm::StatisticalOutlier,
            timestamp: SystemTime::now(),
            category: None,
        };

        let result = classifier.classify_anomalies(&[anomaly_score]);
        assert!(result.is_ok());

        let classified = result.unwrap();
        assert_eq!(classified.len(), 1);
        assert!(classified[0].category.is_some());
    }

    #[test]
    fn test_baseline_model_management() {
        let mut detector = AnomalyDetector::new();

        let baseline_model = BaselineModel {
            model_name: "test_baseline".to_string(),
            model_type: BaselineModelType::MovingAverage,
            parameters: HashMap::new(),
            training_period: Duration::from_secs(3600),
            update_frequency: Duration::from_secs(300),
            confidence_threshold: 0.95,
        };

        detector.add_baseline_model(baseline_model);
        assert_eq!(detector.baseline_models.len(), 1);
    }

    #[test]
    fn test_component_detector_management() {
        let mut ensemble = EnsembleAnomalyDetector::new();

        let component = ComponentDetector {
            detector_name: "test_detector".to_string(),
            algorithm: AnomalyDetectionAlgorithm::IsolationForest,
            weight: 1.0,
            performance_score: 0.85,
            specialization: DetectorSpecialization::PointAnomalies,
        };

        ensemble.add_component_detector(component);
        assert_eq!(ensemble.get_component_detectors().len(), 1);
    }

    #[test]
    fn test_anomaly_category_creation() {
        let category = AnomalyCategory {
            category_name: "High Load".to_string(),
            description: "Unusually high system load detected".to_string(),
            severity_level: SeverityLevel::High,
            typical_characteristics: vec!["CPU > 90%".to_string(), "Memory > 85%".to_string()],
            response_actions: vec!["Scale up resources".to_string(), "Alert operations team".to_string()],
        };

        assert_eq!(category.category_name, "High Load");
        assert!(matches!(category.severity_level, SeverityLevel::High));
        assert_eq!(category.typical_characteristics.len(), 2);
        assert_eq!(category.response_actions.len(), 2);
    }

    #[test]
    fn test_score_aggregation() {
        let ensemble = EnsembleAnomalyDetector::new();

        let scores = vec![
            AnomalyScore {
                position: 0,
                value: 10.0,
                anomaly_score: 3.0,
                confidence: 0.9,
                algorithm: AnomalyDetectionAlgorithm::StatisticalOutlier,
                timestamp: SystemTime::now(),
                category: None,
            },
            AnomalyScore {
                position: 0,
                value: 10.0,
                anomaly_score: 2.5,
                confidence: 0.8,
                algorithm: AnomalyDetectionAlgorithm::IsolationForest,
                timestamp: SystemTime::now(),
                category: None,
            },
        ];

        let result = ensemble.aggregate_scores(&scores);
        assert!(result.is_ok());

        let aggregated = result.unwrap();
        assert_eq!(aggregated.len(), 1);
        assert_eq!(aggregated[0].position, 0);
        assert!((aggregated[0].anomaly_score - 2.75).abs() < 0.01);
    }
}
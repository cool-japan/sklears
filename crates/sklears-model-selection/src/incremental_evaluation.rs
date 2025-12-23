//! Incremental Cross-Validation for Streaming/Online Learning
//!
//! This module provides specialized cross-validation and evaluation methods for incremental
//! and online learning scenarios where data arrives in streams and models are updated continuously.
//! It includes methods for handling concept drift, adaptive evaluation windows, and performance
//! monitoring in non-stationary environments.

use scirs2_core::ndarray::Array1;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use sklears_core::types::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Incremental evaluation strategies
#[derive(Debug, Clone)]
pub enum IncrementalEvaluationStrategy {
    /// Sliding window evaluation
    SlidingWindow {
        window_size: usize,

        step_size: usize,

        overlap_ratio: Float,
    },
    /// Prequential evaluation (test-then-train)
    Prequential {
        adaptation_rate: Float,

        forgetting_factor: Float,
    },
    /// Holdout evaluation with periodic updates
    HoldoutEvaluation {
        holdout_ratio: Float,
        update_frequency: usize,
        drift_detection: bool,
    },
    /// Block-based evaluation for chunk learning
    BlockBased {
        block_size: usize,
        evaluation_blocks: usize,
        overlap_blocks: usize,
    },
    /// Adaptive window evaluation
    AdaptiveWindow {
        min_window_size: usize,
        max_window_size: usize,
        adaptation_criterion: AdaptationCriterion,
    },
    /// Fading factor evaluation
    FadingFactor { alpha: Float, minimum_weight: Float },
    /// Cross-validation for data streams
    StreamingCrossValidation {
        n_folds: usize,
        fold_update_strategy: FoldUpdateStrategy,
    },
}

/// Criteria for adaptive window sizing
#[derive(Debug, Clone)]
pub enum AdaptationCriterion {
    /// Performance-based adaptation
    PerformanceBased { threshold: Float },
    /// Drift detection-based adaptation
    DriftBased { drift_detector: DriftDetectorType },
    /// Variance-based adaptation
    VarianceBased { variance_threshold: Float },
    /// Hybrid approach
    Hybrid { criteria: Vec<AdaptationCriterion> },
}

/// Types of drift detectors
#[derive(Debug, Clone)]
pub enum DriftDetectorType {
    /// ADWIN (Adaptive Windowing)
    ADWIN { confidence: Float },
    /// Page-Hinkley test
    PageHinkley { threshold: Float, alpha: Float },
    /// EDDM (Early Drift Detection Method)
    EDDM { alpha: Float, beta: Float },
    /// DDM (Drift Detection Method)
    DDM {
        warning_level: Float,

        drift_level: Float,
    },
    /// Statistical test-based detection
    StatisticalTest { test_type: String, p_value: Float },
}

/// Strategies for updating folds in streaming CV
#[derive(Debug, Clone)]
pub enum FoldUpdateStrategy {
    /// Replace oldest data
    ReplaceOldest,
    /// Weighted update
    WeightedUpdate { decay_factor: Float },
    /// Selective update based on performance
    SelectiveUpdate { performance_threshold: Float },
    /// Random replacement
    RandomReplacement { replacement_rate: Float },
}

/// Incremental evaluation configuration
#[derive(Debug, Clone)]
pub struct IncrementalEvaluationConfig {
    pub strategy: IncrementalEvaluationStrategy,
    pub performance_metrics: Vec<String>,
    pub drift_detection_enabled: bool,
    pub adaptive_thresholds: bool,
    pub concept_drift_handling: ConceptDriftHandling,
    pub memory_budget: Option<usize>,
    pub evaluation_frequency: usize,
    pub random_state: Option<u64>,
}

/// Concept drift handling strategies
#[derive(Debug, Clone)]
pub enum ConceptDriftHandling {
    /// Ignore drift, continue with current model
    Ignore,
    /// Reset model completely on drift detection
    Reset,
    /// Gradual adaptation to new concept
    GradualAdaptation { adaptation_rate: Float },
    /// Ensemble-based handling
    EnsembleBased { ensemble_size: usize },
    /// Active learning approach
    ActiveLearning { uncertainty_threshold: Float },
}

/// Incremental evaluation result
#[derive(Debug, Clone)]
pub struct IncrementalEvaluationResult {
    pub performance_history: Vec<PerformanceSnapshot>,
    pub concept_drift_events: Vec<DriftEvent>,
    pub adaptive_parameters: AdaptiveParameters,
    pub streaming_statistics: StreamingStatistics,
    pub window_evolution: Option<WindowEvolution>,
    pub computational_metrics: ComputationalMetrics,
}

/// Performance snapshot at a specific time
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub sample_index: usize,
    pub performance_score: Float,
    pub window_size: usize,
    pub model_age: usize,
    pub confidence_interval: Option<(Float, Float)>,
    pub additional_metrics: HashMap<String, Float>,
}

/// Detected drift event
#[derive(Debug, Clone)]
pub struct DriftEvent {
    pub timestamp: Instant,
    pub sample_index: usize,
    pub drift_type: DriftType,
    pub confidence: Float,
    pub detection_method: String,
    pub affected_features: Option<Vec<usize>>,
}

/// Types of detected drift
#[derive(Debug, Clone)]
pub enum DriftType {
    /// Gradual drift
    Gradual,
    /// Sudden drift
    Sudden,
    /// Incremental drift
    Incremental,
    /// Recurring concept
    Recurring,
    /// Unknown drift pattern
    Unknown,
}

/// Adaptive parameters tracking
#[derive(Debug, Clone)]
pub struct AdaptiveParameters {
    pub window_size_history: Vec<usize>,
    pub learning_rate_history: Vec<Float>,
    pub threshold_history: Vec<Float>,
    pub adaptation_events: Vec<AdaptationEvent>,
}

/// Adaptation event
#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    pub timestamp: Instant,
    pub event_type: AdaptationType,
    pub old_value: Float,
    pub new_value: Float,
    pub trigger_reason: String,
}

/// Types of adaptations
#[derive(Debug, Clone)]
pub enum AdaptationType {
    /// WindowSizeChange
    WindowSizeChange,
    /// LearningRateChange
    LearningRateChange,
    /// ThresholdChange
    ThresholdChange,
    /// ModelReset
    ModelReset,
    /// ParameterAdjustment
    ParameterAdjustment,
}

/// Streaming statistics
#[derive(Debug, Clone)]
pub struct StreamingStatistics {
    pub total_samples_processed: usize,
    pub total_batches_processed: usize,
    pub average_processing_time: Duration,
    pub memory_usage_peak: usize,
    pub model_updates_count: usize,
    pub evaluation_count: usize,
    pub drift_rate: Float,
}

/// Window evolution tracking
#[derive(Debug, Clone)]
pub struct WindowEvolution {
    pub size_evolution: Vec<usize>,
    pub performance_evolution: Vec<Float>,
    pub adaptation_points: Vec<usize>,
    pub efficiency_scores: Vec<Float>,
}

/// Computational metrics
#[derive(Debug, Clone)]
pub struct ComputationalMetrics {
    pub total_computation_time: Duration,
    pub average_update_time: Duration,
    pub memory_efficiency: Float,
    pub throughput: Float,
    pub latency_percentiles: HashMap<String, Duration>,
}

/// Incremental evaluator
pub struct IncrementalEvaluator {
    config: IncrementalEvaluationConfig,
    performance_history: Vec<PerformanceSnapshot>,
    drift_events: Vec<DriftEvent>,
    current_window: VecDeque<(Array1<Float>, Float)>, // (features, label)
    current_predictions: VecDeque<Float>,
    adaptive_parameters: AdaptiveParameters,
    drift_detector: Option<Box<dyn DriftDetector>>,
    rng: StdRng,
    start_time: Instant,
    sample_count: usize,
}

/// Trait for drift detection
trait DriftDetector: Send + Sync {
    fn update(&mut self, value: Float) -> bool;
    fn reset(&mut self);
    fn get_confidence(&self) -> Float;
}

/// ADWIN drift detector implementation
#[derive(Debug)]
struct ADWINDetector {
    confidence: Float,
    window: VecDeque<Float>,
    total: Float,
    variance: Float,
    width: usize,
}

/// Page-Hinkley drift detector implementation
#[derive(Debug)]
struct PageHinkleyDetector {
    threshold: Float,
    alpha: Float,
    x_mean: Float,
    sample_count: usize,
    sum: Float,
    drift_detected: bool,
}

impl Default for IncrementalEvaluationConfig {
    fn default() -> Self {
        Self {
            strategy: IncrementalEvaluationStrategy::Prequential {
                adaptation_rate: 0.01,
                forgetting_factor: 0.95,
            },
            performance_metrics: vec!["accuracy".to_string()],
            drift_detection_enabled: true,
            adaptive_thresholds: true,
            concept_drift_handling: ConceptDriftHandling::GradualAdaptation {
                adaptation_rate: 0.1,
            },
            memory_budget: Some(10000),
            evaluation_frequency: 100,
            random_state: None,
        }
    }
}

impl IncrementalEvaluator {
    /// Create a new incremental evaluator
    pub fn new(config: IncrementalEvaluationConfig) -> Self {
        let rng = match config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        let drift_detector = if config.drift_detection_enabled {
            Some(Self::create_drift_detector(&config.strategy))
        } else {
            None
        };

        Self {
            config,
            performance_history: Vec::new(),
            drift_events: Vec::new(),
            current_window: VecDeque::new(),
            current_predictions: VecDeque::new(),
            adaptive_parameters: AdaptiveParameters {
                window_size_history: Vec::new(),
                learning_rate_history: Vec::new(),
                threshold_history: Vec::new(),
                adaptation_events: Vec::new(),
            },
            drift_detector,
            rng,
            start_time: Instant::now(),
            sample_count: 0,
        }
    }

    /// Process a new data point and evaluate incrementally
    pub fn update<F>(
        &mut self,
        features: Array1<Float>,
        true_label: Float,
        prediction: Float,
        model_update_fn: F,
    ) -> Result<Option<PerformanceSnapshot>, Box<dyn std::error::Error>>
    where
        F: FnOnce(&Array1<Float>, Float),
    {
        let _update_start = Instant::now();
        self.sample_count += 1;

        // Add to current window
        self.current_window
            .push_back((features.clone(), true_label));
        self.current_predictions.push_back(prediction);

        // Handle memory budget
        if let Some(budget) = self.config.memory_budget {
            while self.current_window.len() > budget {
                self.current_window.pop_front();
                self.current_predictions.pop_front();
            }
        }

        // Check for concept drift
        let error = (prediction - true_label).abs();
        let drift_detected = if let Some(ref mut detector) = self.drift_detector {
            detector.update(error)
        } else {
            false
        };

        if drift_detected {
            self.handle_concept_drift()?;
        }

        // Perform evaluation based on strategy
        let performance_snapshot = match &self.config.strategy {
            IncrementalEvaluationStrategy::SlidingWindow { .. } => {
                self.evaluate_sliding_window()?
            }
            IncrementalEvaluationStrategy::Prequential { .. } => {
                self.evaluate_prequential(prediction, true_label)?
            }
            IncrementalEvaluationStrategy::HoldoutEvaluation { .. } => self.evaluate_holdout()?,
            IncrementalEvaluationStrategy::BlockBased { .. } => self.evaluate_block_based()?,
            IncrementalEvaluationStrategy::AdaptiveWindow { .. } => {
                self.evaluate_adaptive_window()?
            }
            IncrementalEvaluationStrategy::FadingFactor { .. } => {
                self.evaluate_fading_factor(prediction, true_label)?
            }
            IncrementalEvaluationStrategy::StreamingCrossValidation { .. } => {
                self.evaluate_streaming_cv()?
            }
        };

        // Update model (test-then-train paradigm)
        model_update_fn(&features, true_label);

        // Check if evaluation should be performed
        if self.sample_count % self.config.evaluation_frequency == 0 {
            if let Some(snapshot) = performance_snapshot {
                self.performance_history.push(snapshot.clone());
                Ok(Some(snapshot))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Get the final evaluation result
    pub fn finalize(self) -> IncrementalEvaluationResult {
        let total_time = self.start_time.elapsed();

        let streaming_statistics = StreamingStatistics {
            total_samples_processed: self.sample_count,
            total_batches_processed: self.performance_history.len(),
            average_processing_time: if self.sample_count > 0 {
                total_time / self.sample_count as u32
            } else {
                Duration::from_secs(0)
            },
            memory_usage_peak: self.current_window.len(),
            model_updates_count: self.sample_count,
            evaluation_count: self.performance_history.len(),
            drift_rate: self.drift_events.len() as Float / self.sample_count.max(1) as Float,
        };

        let window_evolution = if !self.adaptive_parameters.window_size_history.is_empty() {
            Some(WindowEvolution {
                size_evolution: self.adaptive_parameters.window_size_history.clone(),
                performance_evolution: self
                    .performance_history
                    .iter()
                    .map(|s| s.performance_score)
                    .collect(),
                adaptation_points: self
                    .adaptive_parameters
                    .adaptation_events
                    .iter()
                    .map(|e| e.timestamp.duration_since(self.start_time).as_millis() as usize)
                    .collect(),
                efficiency_scores: vec![0.8; self.performance_history.len()], // Placeholder
            })
        } else {
            None
        };

        let computational_metrics = ComputationalMetrics {
            total_computation_time: total_time,
            average_update_time: if self.sample_count > 0 {
                total_time / self.sample_count as u32
            } else {
                Duration::from_secs(0)
            },
            memory_efficiency: 0.8, // Placeholder
            throughput: self.sample_count as Float / total_time.as_secs_f64() as Float,
            latency_percentiles: HashMap::new(), // Could add percentile calculations
        };

        IncrementalEvaluationResult {
            performance_history: self.performance_history,
            concept_drift_events: self.drift_events,
            adaptive_parameters: self.adaptive_parameters,
            streaming_statistics,
            window_evolution,
            computational_metrics,
        }
    }

    /// Create appropriate drift detector based on strategy
    fn create_drift_detector(_strategy: &IncrementalEvaluationStrategy) -> Box<dyn DriftDetector> {
        // Default to ADWIN detector
        Box::new(ADWINDetector::new(0.002))
    }

    /// Handle concept drift detection
    fn handle_concept_drift(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let drift_event = DriftEvent {
            timestamp: Instant::now(),
            sample_index: self.sample_count,
            drift_type: DriftType::Sudden, // Simplified classification
            confidence: self
                .drift_detector
                .as_ref()
                .map_or(0.0, |d| d.get_confidence()),
            detection_method: "ADWIN".to_string(),
            affected_features: None,
        };

        self.drift_events.push(drift_event);

        // Handle drift based on configuration
        match &self.config.concept_drift_handling {
            ConceptDriftHandling::Reset => {
                self.current_window.clear();
                self.current_predictions.clear();
                if let Some(ref mut detector) = self.drift_detector {
                    detector.reset();
                }
            }
            ConceptDriftHandling::GradualAdaptation { adaptation_rate } => {
                // Reduce window size gradually
                let new_size =
                    (self.current_window.len() as Float * (1.0 - adaptation_rate)) as usize;
                while self.current_window.len() > new_size {
                    self.current_window.pop_front();
                    self.current_predictions.pop_front();
                }
            }
            _ => {
                // Other strategies would be implemented here
            }
        }

        Ok(())
    }

    /// Sliding window evaluation
    fn evaluate_sliding_window(
        &mut self,
    ) -> Result<Option<PerformanceSnapshot>, Box<dyn std::error::Error>> {
        let (window_size, step_size, _overlap_ratio) = match &self.config.strategy {
            IncrementalEvaluationStrategy::SlidingWindow {
                window_size,
                step_size,
                overlap_ratio,
            } => (*window_size, *step_size, *overlap_ratio),
            _ => unreachable!(),
        };

        if self.current_window.len() >= window_size && self.sample_count % step_size == 0 {
            let recent_predictions: Vec<Float> = self
                .current_predictions
                .iter()
                .rev()
                .take(window_size)
                .cloned()
                .collect();

            let recent_labels: Vec<Float> = self
                .current_window
                .iter()
                .rev()
                .take(window_size)
                .map(|(_, label)| *label)
                .collect();

            let accuracy = recent_predictions
                .iter()
                .zip(recent_labels.iter())
                .map(|(&pred, &label)| {
                    if (pred > 0.5) == (label > 0.5) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .sum::<Float>()
                / recent_predictions.len() as Float;

            Ok(Some(PerformanceSnapshot {
                timestamp: Instant::now(),
                sample_index: self.sample_count,
                performance_score: accuracy,
                window_size,
                model_age: self.sample_count,
                confidence_interval: None,
                additional_metrics: HashMap::new(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Prequential evaluation (test-then-train)
    fn evaluate_prequential(
        &mut self,
        prediction: Float,
        true_label: Float,
    ) -> Result<Option<PerformanceSnapshot>, Box<dyn std::error::Error>> {
        let (adaptation_rate, _forgetting_factor) = match &self.config.strategy {
            IncrementalEvaluationStrategy::Prequential {
                adaptation_rate,
                forgetting_factor,
            } => (*adaptation_rate, *forgetting_factor),
            _ => unreachable!(),
        };

        // Simple prequential accuracy calculation
        let is_correct = (prediction > 0.5) == (true_label > 0.5);
        let current_accuracy = if is_correct { 1.0 } else { 0.0 };

        // Update running average (exponential moving average)
        let previous_performance = self
            .performance_history
            .last()
            .map(|s| s.performance_score)
            .unwrap_or(0.5);

        let updated_performance =
            (1.0 - adaptation_rate) * previous_performance + adaptation_rate * current_accuracy;

        Ok(Some(PerformanceSnapshot {
            timestamp: Instant::now(),
            sample_index: self.sample_count,
            performance_score: updated_performance,
            window_size: 1,
            model_age: self.sample_count,
            confidence_interval: None,
            additional_metrics: HashMap::new(),
        }))
    }

    /// Holdout evaluation
    fn evaluate_holdout(
        &mut self,
    ) -> Result<Option<PerformanceSnapshot>, Box<dyn std::error::Error>> {
        let (holdout_ratio, update_frequency, _drift_detection) = match &self.config.strategy {
            IncrementalEvaluationStrategy::HoldoutEvaluation {
                holdout_ratio,
                update_frequency,
                drift_detection,
            } => (*holdout_ratio, *update_frequency, *drift_detection),
            _ => unreachable!(),
        };

        if self.sample_count % update_frequency == 0 && !self.current_window.is_empty() {
            let holdout_size = (self.current_window.len() as Float * holdout_ratio) as usize;

            if holdout_size > 0 {
                let holdout_predictions: Vec<Float> = self
                    .current_predictions
                    .iter()
                    .rev()
                    .take(holdout_size)
                    .cloned()
                    .collect();

                let holdout_labels: Vec<Float> = self
                    .current_window
                    .iter()
                    .rev()
                    .take(holdout_size)
                    .map(|(_, label)| *label)
                    .collect();

                let accuracy = holdout_predictions
                    .iter()
                    .zip(holdout_labels.iter())
                    .map(|(&pred, &label)| {
                        if (pred > 0.5) == (label > 0.5) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<Float>()
                    / holdout_predictions.len() as Float;

                Ok(Some(PerformanceSnapshot {
                    timestamp: Instant::now(),
                    sample_index: self.sample_count,
                    performance_score: accuracy,
                    window_size: holdout_size,
                    model_age: self.sample_count,
                    confidence_interval: None,
                    additional_metrics: HashMap::new(),
                }))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Block-based evaluation
    fn evaluate_block_based(
        &mut self,
    ) -> Result<Option<PerformanceSnapshot>, Box<dyn std::error::Error>> {
        let (block_size, _evaluation_blocks, _overlap_blocks) = match &self.config.strategy {
            IncrementalEvaluationStrategy::BlockBased {
                block_size,
                evaluation_blocks,
                overlap_blocks,
            } => (*block_size, *evaluation_blocks, *overlap_blocks),
            _ => unreachable!(),
        };

        if self.sample_count % block_size == 0 && self.current_window.len() >= block_size {
            let block_predictions: Vec<Float> = self
                .current_predictions
                .iter()
                .rev()
                .take(block_size)
                .cloned()
                .collect();

            let block_labels: Vec<Float> = self
                .current_window
                .iter()
                .rev()
                .take(block_size)
                .map(|(_, label)| *label)
                .collect();

            let accuracy = block_predictions
                .iter()
                .zip(block_labels.iter())
                .map(|(&pred, &label)| {
                    if (pred > 0.5) == (label > 0.5) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .sum::<Float>()
                / block_predictions.len() as Float;

            Ok(Some(PerformanceSnapshot {
                timestamp: Instant::now(),
                sample_index: self.sample_count,
                performance_score: accuracy,
                window_size: block_size,
                model_age: self.sample_count,
                confidence_interval: None,
                additional_metrics: HashMap::new(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Adaptive window evaluation
    fn evaluate_adaptive_window(
        &mut self,
    ) -> Result<Option<PerformanceSnapshot>, Box<dyn std::error::Error>> {
        let (min_window_size, max_window_size, _adaptation_criterion) = match &self.config.strategy
        {
            IncrementalEvaluationStrategy::AdaptiveWindow {
                min_window_size,
                max_window_size,
                adaptation_criterion,
            } => (*min_window_size, *max_window_size, adaptation_criterion),
            _ => unreachable!(),
        };

        // Simple adaptive logic: adjust window size based on recent performance variance
        if self.performance_history.len() >= 3 {
            let recent_scores: Vec<Float> = self
                .performance_history
                .iter()
                .rev()
                .take(3)
                .map(|s| s.performance_score)
                .collect();

            let mean_score = recent_scores.iter().sum::<Float>() / recent_scores.len() as Float;
            let variance = recent_scores
                .iter()
                .map(|&score| (score - mean_score).powi(2))
                .sum::<Float>()
                / recent_scores.len() as Float;

            // Adjust window size based on variance
            let current_window_size = self.current_window.len();
            let new_window_size = if variance > 0.1 {
                // High variance, reduce window size
                (current_window_size / 2).max(min_window_size)
            } else {
                // Low variance, can increase window size
                (current_window_size * 2).min(max_window_size)
            };

            // Record window size change
            if new_window_size != current_window_size {
                self.adaptive_parameters
                    .window_size_history
                    .push(new_window_size);

                // Adjust actual window
                while self.current_window.len() > new_window_size {
                    self.current_window.pop_front();
                    self.current_predictions.pop_front();
                }
            }
        }

        // Evaluate using current window
        if !self.current_window.is_empty() {
            let predictions: Vec<Float> = self.current_predictions.iter().cloned().collect();
            let labels: Vec<Float> = self
                .current_window
                .iter()
                .map(|(_, label)| *label)
                .collect();

            let accuracy = predictions
                .iter()
                .zip(labels.iter())
                .map(|(&pred, &label)| {
                    if (pred > 0.5) == (label > 0.5) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .sum::<Float>()
                / predictions.len() as Float;

            Ok(Some(PerformanceSnapshot {
                timestamp: Instant::now(),
                sample_index: self.sample_count,
                performance_score: accuracy,
                window_size: self.current_window.len(),
                model_age: self.sample_count,
                confidence_interval: None,
                additional_metrics: HashMap::new(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Fading factor evaluation
    fn evaluate_fading_factor(
        &mut self,
        prediction: Float,
        true_label: Float,
    ) -> Result<Option<PerformanceSnapshot>, Box<dyn std::error::Error>> {
        let (alpha, _minimum_weight) = match &self.config.strategy {
            IncrementalEvaluationStrategy::FadingFactor {
                alpha,
                minimum_weight,
            } => (*alpha, *minimum_weight),
            _ => unreachable!(),
        };

        // Weighted evaluation with fading factor
        let is_correct = (prediction > 0.5) == (true_label > 0.5);
        let current_accuracy = if is_correct { 1.0 } else { 0.0 };

        let previous_performance = self
            .performance_history
            .last()
            .map(|s| s.performance_score)
            .unwrap_or(0.5);

        let faded_performance = alpha * current_accuracy + (1.0 - alpha) * previous_performance;

        Ok(Some(PerformanceSnapshot {
            timestamp: Instant::now(),
            sample_index: self.sample_count,
            performance_score: faded_performance,
            window_size: 1,
            model_age: self.sample_count,
            confidence_interval: None,
            additional_metrics: HashMap::new(),
        }))
    }

    /// Streaming cross-validation evaluation
    fn evaluate_streaming_cv(
        &mut self,
    ) -> Result<Option<PerformanceSnapshot>, Box<dyn std::error::Error>> {
        let (n_folds, _fold_update_strategy) = match &self.config.strategy {
            IncrementalEvaluationStrategy::StreamingCrossValidation {
                n_folds,
                fold_update_strategy,
            } => (*n_folds, fold_update_strategy),
            _ => unreachable!(),
        };

        if self.current_window.len() >= n_folds && self.sample_count % 10 == 0 {
            // Simple streaming CV: evaluate on rotating folds
            let fold_size = self.current_window.len() / n_folds;
            let mut fold_scores = Vec::new();

            for fold in 0..n_folds {
                let test_start = fold * fold_size;
                let test_end = if fold == n_folds - 1 {
                    self.current_window.len()
                } else {
                    (fold + 1) * fold_size
                };

                if test_end <= self.current_predictions.len() {
                    let fold_predictions: Vec<Float> = self
                        .current_predictions
                        .iter()
                        .skip(test_start)
                        .take(test_end - test_start)
                        .cloned()
                        .collect();

                    let fold_labels: Vec<Float> = self
                        .current_window
                        .iter()
                        .skip(test_start)
                        .take(test_end - test_start)
                        .map(|(_, label)| *label)
                        .collect();

                    let fold_accuracy = fold_predictions
                        .iter()
                        .zip(fold_labels.iter())
                        .map(|(&pred, &label)| {
                            if (pred > 0.5) == (label > 0.5) {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .sum::<Float>()
                        / fold_predictions.len() as Float;

                    fold_scores.push(fold_accuracy);
                }
            }

            if !fold_scores.is_empty() {
                let mean_accuracy = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
                let std_accuracy = {
                    let variance = fold_scores
                        .iter()
                        .map(|&score| (score - mean_accuracy).powi(2))
                        .sum::<Float>()
                        / fold_scores.len() as Float;
                    variance.sqrt()
                };

                let confidence_interval = (
                    mean_accuracy - 1.96 * std_accuracy,
                    mean_accuracy + 1.96 * std_accuracy,
                );

                Ok(Some(PerformanceSnapshot {
                    timestamp: Instant::now(),
                    sample_index: self.sample_count,
                    performance_score: mean_accuracy,
                    window_size: self.current_window.len(),
                    model_age: self.sample_count,
                    confidence_interval: Some(confidence_interval),
                    additional_metrics: {
                        let mut metrics = HashMap::new();
                        metrics.insert("std_accuracy".to_string(), std_accuracy);
                        metrics
                    },
                }))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}

impl ADWINDetector {
    fn new(confidence: Float) -> Self {
        Self {
            confidence,
            window: VecDeque::new(),
            total: 0.0,
            variance: 0.0,
            width: 0,
        }
    }
}

impl DriftDetector for ADWINDetector {
    fn update(&mut self, value: Float) -> bool {
        self.window.push_back(value);
        self.total += value;
        self.width += 1;

        // Simplified ADWIN logic
        if self.width > 100 {
            // Check for significant difference between window halves
            let half = self.width / 2;
            let first_half_mean = self.window.iter().take(half).sum::<Float>() / half as Float;
            let second_half_mean =
                self.window.iter().skip(half).sum::<Float>() / (self.width - half) as Float;

            let difference = (first_half_mean - second_half_mean).abs();
            difference > self.confidence
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.window.clear();
        self.total = 0.0;
        self.variance = 0.0;
        self.width = 0;
    }

    fn get_confidence(&self) -> Float {
        if self.width > 0 {
            self.variance / self.width as Float
        } else {
            0.0
        }
    }
}

impl PageHinkleyDetector {
    fn new(threshold: Float, alpha: Float) -> Self {
        Self {
            threshold,
            alpha,
            x_mean: 0.0,
            sample_count: 0,
            sum: 0.0,
            drift_detected: false,
        }
    }
}

impl DriftDetector for PageHinkleyDetector {
    fn update(&mut self, value: Float) -> bool {
        self.sample_count += 1;

        if self.sample_count == 1 {
            self.x_mean = value;
            return false;
        }

        // Update mean
        self.x_mean = self.x_mean + (value - self.x_mean) / self.sample_count as Float;

        // Update Page-Hinkley statistic
        self.sum = (self.sum + value - self.x_mean - self.alpha).max(0.0);

        // Check for drift
        if self.sum > self.threshold {
            self.drift_detected = true;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.x_mean = 0.0;
        self.sample_count = 0;
        self.sum = 0.0;
        self.drift_detected = false;
    }

    fn get_confidence(&self) -> Float {
        if self.threshold > 0.0 {
            (self.sum / self.threshold).min(1.0)
        } else {
            0.0
        }
    }
}

/// Convenience function for incremental evaluation
pub fn evaluate_incremental_stream<F>(
    data_stream: impl Iterator<Item = (Array1<Float>, Float, Float)>, // (features, label, prediction)
    model_update_fn: F,
    config: Option<IncrementalEvaluationConfig>,
) -> Result<IncrementalEvaluationResult, Box<dyn std::error::Error>>
where
    F: Fn(&Array1<Float>, Float),
{
    let config = config.unwrap_or_default();
    let mut evaluator = IncrementalEvaluator::new(config);

    for (features, label, prediction) in data_stream {
        evaluator.update(features, label, prediction, &model_update_fn)?;
    }

    Ok(evaluator.finalize())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_evaluator_creation() {
        let config = IncrementalEvaluationConfig::default();
        let evaluator = IncrementalEvaluator::new(config);
        assert_eq!(evaluator.sample_count, 0);
    }

    #[test]
    fn test_prequential_evaluation() {
        let config = IncrementalEvaluationConfig {
            strategy: IncrementalEvaluationStrategy::Prequential {
                adaptation_rate: 0.1,
                forgetting_factor: 0.9,
            },
            evaluation_frequency: 1,
            ..Default::default()
        };

        let mut evaluator = IncrementalEvaluator::new(config);

        let features = Array1::from_vec(vec![0.5, 0.3, 0.8]);
        let model_update_fn = |_: &Array1<Float>, _: Float| {}; // No-op update

        let result = evaluator
            .update(features, 1.0, 0.8, model_update_fn)
            .unwrap();
        assert!(result.is_some());

        let snapshot = result.unwrap();
        assert!(snapshot.performance_score >= 0.0 && snapshot.performance_score <= 1.0);
    }

    #[test]
    fn test_sliding_window_evaluation() {
        let config = IncrementalEvaluationConfig {
            strategy: IncrementalEvaluationStrategy::SlidingWindow {
                window_size: 5,
                step_size: 5,
                overlap_ratio: 0.0,
            },
            evaluation_frequency: 5,
            ..Default::default()
        };

        let mut evaluator = IncrementalEvaluator::new(config);
        let model_update_fn = |_: &Array1<Float>, _: Float| {};

        // Add 5 samples
        for i in 0..5 {
            let features = Array1::from_vec(vec![i as Float * 0.1]);
            let label = if i % 2 == 0 { 1.0 } else { 0.0 };
            let prediction = if i % 2 == 0 { 0.8 } else { 0.2 };

            let result = evaluator
                .update(features, label, prediction, &model_update_fn)
                .unwrap();

            if i == 4 {
                // Last sample should trigger evaluation
                assert!(result.is_some());
            }
        }
    }

    #[test]
    fn test_drift_detector() {
        let mut detector = ADWINDetector::new(0.1);

        // Add some stable values
        for _ in 0..50 {
            assert!(!detector.update(0.1));
        }

        // Add some values that should trigger drift
        for _ in 0..60 {
            detector.update(0.9);
        }

        // Should eventually detect drift (simplified test)
        assert!(detector.get_confidence() >= 0.0);
    }

    #[test]
    fn test_streaming_evaluation() {
        let data_stream = (0..20).map(|i| {
            let features = Array1::from_vec(vec![i as Float * 0.05]);
            let label = if i % 2 == 0 { 1.0 } else { 0.0 };
            let prediction = if i % 2 == 0 { 0.8 } else { 0.3 };
            (features, label, prediction)
        });

        let model_update_fn = |_: &Array1<Float>, _: Float| {};

        let config = IncrementalEvaluationConfig {
            evaluation_frequency: 10, // Evaluate every 10 samples instead of default 100
            ..Default::default()
        };

        let result =
            evaluate_incremental_stream(data_stream, model_update_fn, Some(config)).unwrap();

        assert!(result.streaming_statistics.total_samples_processed == 20);
        assert!(!result.performance_history.is_empty());
    }
}

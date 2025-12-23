//! Adaptive Decomposition Classifiers
//!
//! This module provides adaptive decomposition strategies for multiclass classification.
//! Adaptive decomposition automatically adjusts the decomposition approach based on data
//! characteristics, providing dynamic class grouping, context-aware decomposition,
//! performance-based adaptation, and online adaptive learning.

use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained, Transform},
    types::Float,
};
use std::collections::HashMap;

use crate::{
    DynamicMulticlassTrainedData, ECOCClassifier, ECOCStrategy, OneVsOneClassifier,
    OneVsRestClassifier,
};
use sklears_core::traits::Fit;

/// Adaptive decomposition strategy configuration
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptiveStrategy {
    /// Dynamic class grouping based on similarity
    DynamicGrouping {
        /// Similarity threshold for grouping classes
        similarity_threshold: f64,
        /// Maximum group size
        max_group_size: usize,
    },
    /// Context-aware decomposition based on data characteristics
    ContextAware {
        /// Use class imbalance information
        consider_imbalance: bool,
        /// Use feature complexity
        consider_complexity: bool,
        /// Use class separability
        consider_separability: bool,
    },
    /// Performance-based adaptive selection
    PerformanceBased {
        /// Number of samples to use for initial evaluation
        sample_size: usize,
        /// Minimum accuracy improvement required to switch strategies
        improvement_threshold: f64,
    },
    /// Online decomposition that adapts as more data arrives
    OnlineAdaptive {
        window_size: usize,
        adaptation_frequency: usize,
    },
}

impl Default for AdaptiveStrategy {
    fn default() -> Self {
        AdaptiveStrategy::ContextAware {
            consider_imbalance: true,
            consider_complexity: true,
            consider_separability: true,
        }
    }
}

/// Configuration for adaptive decomposition
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// The adaptive strategy to use
    pub strategy: AdaptiveStrategy,
    /// Minimum number of classes required to enable adaptive behavior
    pub min_classes: usize,
    /// Maximum number of classes before defaulting to specific strategies
    pub max_classes: usize,
    /// StdRng state for reproducible results
    pub random_state: Option<u64>,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            strategy: AdaptiveStrategy::default(),
            min_classes: 3,
            max_classes: 1000,
            random_state: None,
        }
    }
}

/// Multiclass strategy selection
#[derive(Debug, Clone, PartialEq)]
pub enum MulticlassStrategy {
    /// One-vs-Rest strategy
    OneVsRest,
    /// One-vs-One strategy
    OneVsOne,
    /// Error-Correcting Output Codes strategy
    ECOC(ECOCStrategy),
    /// Adaptive decomposition strategy based on data characteristics
    Adaptive(AdaptiveStrategy),
    /// Automatically select the best strategy
    Auto,
}

impl Default for MulticlassStrategy {
    fn default() -> Self {
        MulticlassStrategy::Auto
    }
}

/// Strategy selection criteria
#[derive(Debug, Clone)]
pub enum SelectionCriteria {
    /// Select based on training time
    TrainingTime,
    /// Select based on prediction accuracy
    Accuracy,
    /// Select based on memory usage
    Memory,
    /// Combined criteria with weights
    Combined {

        accuracy_weight: f64,

        time_weight: f64,
        memory_weight: f64,
    },
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        SelectionCriteria::Combined {
            accuracy_weight: 0.6,
            time_weight: 0.3,
            memory_weight: 0.1,
        }
    }
}

/// Adaptive Decomposition Classifier
///
/// This classifier implements adaptive decomposition strategies that automatically
/// adjust their approach based on data characteristics. It can perform dynamic
/// class grouping, context-aware decomposition, performance-based adaptation,
/// and online adaptive learning.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```
/// use sklears_multiclass::{AdaptiveDecompositionClassifier, AdaptiveStrategy};
/// use scirs2_autograd::ndarray::array;
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let adaptive = AdaptiveDecompositionClassifier::new(base_classifier)
/// //     .strategy(AdaptiveStrategy::ContextAware {
/// //         consider_imbalance: true,
/// //         consider_complexity: true,
/// //         consider_separability: true,
/// //     });
/// ```
#[derive(Debug)]
pub struct AdaptiveDecompositionClassifier<C, S = Untrained> {
    base_estimator: C,
    config: AdaptiveConfig,
    state: PhantomData<S>,
}

impl<C> AdaptiveDecompositionClassifier<C, Untrained> {
    /// Create a new AdaptiveDecompositionClassifier instance with a base estimator
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: AdaptiveConfig::default(),
            state: PhantomData,
        }
    }

    /// Set the adaptive strategy
    pub fn strategy(mut self, strategy: AdaptiveStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the minimum number of classes for adaptive behavior
    pub fn min_classes(mut self, min_classes: usize) -> Self {
        self.config.min_classes = min_classes;
        self
    }

    /// Set the maximum number of classes
    pub fn max_classes(mut self, max_classes: usize) -> Self {
        self.config.max_classes = max_classes;
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Get a reference to the base estimator
    pub fn base_estimator(&self) -> &C {
        &self.base_estimator
    }
}

impl<C: Clone> Clone for AdaptiveDecompositionClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C> Estimator for AdaptiveDecompositionClassifier<C, Untrained> {
    type Config = AdaptiveConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Data for trained adaptive decomposition classifier
#[derive(Debug)]
pub struct AdaptiveDecompositionTrainedData<T> {
    /// The selected underlying strategy
    pub selected_strategy: MulticlassStrategy,
    /// The trained classifier for the selected strategy
    pub strategy_classifier: DynamicMulticlassTrainedData<T>,
    /// Classes found during training
    pub classes: Array1<i32>,
    /// Adaptation metadata
    pub adaptation_metadata: AdaptationMetadata,
}

impl<T: Clone> Clone for AdaptiveDecompositionTrainedData<T> {
    fn clone(&self) -> Self {
        Self {
            selected_strategy: self.selected_strategy.clone(),
            strategy_classifier: self.strategy_classifier.clone(),
            classes: self.classes.clone(),
            adaptation_metadata: self.adaptation_metadata.clone(),
        }
    }
}

/// Metadata about the adaptation process
#[derive(Debug, Clone)]
pub struct AdaptationMetadata {
    /// Detected data characteristics
    pub data_characteristics: DatasetCharacteristics,
    /// Strategy selection reasoning
    pub selection_reasoning: String,
    /// Adaptation decisions made
    pub adaptation_decisions: Vec<String>,
}

/// Characteristics of the dataset for adaptive strategy selection
#[derive(Debug, Clone)]
pub struct DatasetCharacteristics {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of classes
    pub n_classes: usize,
    /// Average samples per class
    pub samples_per_class: f64,
    /// Class imbalance level
    pub class_imbalance: ClassImbalanceLevel,
    /// Dimensionality level
    pub dimensionality: DimensionalityLevel,
    /// Complexity level
    pub complexity: ComplexityLevel,
}

/// Level of class imbalance in the dataset
#[derive(Debug, Clone, PartialEq)]
pub enum ClassImbalanceLevel {
    /// Low imbalance (ratio <= 2.0)
    Low,
    /// Medium imbalance (2.0 < ratio <= 5.0)
    Medium,
    /// High imbalance (ratio > 5.0)
    High,
}

/// Level of dimensionality in the dataset
#[derive(Debug, Clone, PartialEq)]
pub enum DimensionalityLevel {
    /// Low dimensionality
    Low,
    /// Medium dimensionality
    Medium,
    /// High dimensionality
    High,
}

/// Level of complexity in the dataset
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityLevel {
    /// Low complexity
    Low,
    /// Medium complexity
    Medium,
    /// High complexity
    High,
}


type TrainedAdaptiveDecomposition<T> =
    /// AdaptiveDecompositionClassifier

    AdaptiveDecompositionClassifier<AdaptiveDecompositionTrainedData<T>, Trained>;

impl<C> AdaptiveDecompositionClassifier<C, Untrained>
where
    C: Clone + Send + Sync + Fit<Array2<Float>, Array1<Float>>,
    C::Fitted: Predict<Array2<Float>, Array1<Float>> + Send,
{
    /// Fit the adaptive decomposition classifier
    pub fn fit(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> SklResult<TrainedAdaptiveDecomposition<C::Fitted>> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match".to_string(),
            ));
        }

        let unique_classes = {
            let mut classes: Vec<i32> = y.iter().cloned().collect();
            classes.sort_unstable();
            classes.dedup();
            Array1::from(classes)
        };

        let n_classes = unique_classes.len();

        // Check if adaptive behavior is appropriate
        if n_classes < self.config.min_classes {
            // Fall back to simple strategy for few classes
            return self.fit_fallback_strategy(
                x,
                y,
                &unique_classes,
                MulticlassStrategy::OneVsRest,
            );
        }

        if n_classes > self.config.max_classes {
            // Fall back to efficient strategy for many classes
            return self.fit_fallback_strategy(x, y, &unique_classes, MulticlassStrategy::OneVsOne);
        }

        // Analyze data characteristics
        let data_characteristics = self.analyze_data_characteristics(x, y, &unique_classes)?;

        // Select strategy based on adaptive approach
        let (selected_strategy, reasoning) =
            self.select_adaptive_strategy(&data_characteristics)?;

        // Train the selected strategy
        let strategy_classifier = self.train_selected_strategy(&selected_strategy, x, y)?;

        let adaptation_metadata = AdaptationMetadata {
            data_characteristics,
            selection_reasoning: reasoning,
            adaptation_decisions: vec![
                format!("Selected strategy: {:?}", selected_strategy),
                format!("Classes detected: {}", n_classes),
            ],
        };

        let trained_data = AdaptiveDecompositionTrainedData {
            selected_strategy,
            strategy_classifier,
            classes: unique_classes,
            adaptation_metadata,
        };

        Ok(AdaptiveDecompositionClassifier {
            base_estimator: trained_data,
            config: self.config.clone(),
            state: PhantomData,
        })
    }

    /// Analyze data characteristics for adaptive strategy selection
    fn analyze_data_characteristics(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> SklResult<DatasetCharacteristics> {
        let (n_samples, n_features) = x.dim();
        let n_classes = classes.len();

        // Calculate class distribution
        let mut class_counts = std::collections::HashMap::new();
        for &class in y.iter() {
            *class_counts.entry(class).or_insert(0) += 1;
        }

        let class_frequencies: Vec<f64> = classes
            .iter()
            .map(|&class| *class_counts.get(&class).unwrap_or(&0) as f64 / n_samples as f64)
            .collect();

        // Calculate imbalance ratio
        let max_freq = class_frequencies.iter().cloned().fold(0.0, f64::max);
        let min_freq = class_frequencies
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let imbalance_ratio = if min_freq > 0.0 {
            max_freq / min_freq
        } else {
            f64::INFINITY
        };

        // Estimate dimensionality (simplified)
        let dimensionality = if n_features > n_samples {
            DimensionalityLevel::High
        } else if n_features > n_samples / 2 {
            DimensionalityLevel::Medium
        } else {
            DimensionalityLevel::Low
        };

        // Estimate complexity (simplified - based on feature/sample ratio)
        let complexity = if n_features as f64 / n_samples as f64 > 0.8 {
            ComplexityLevel::High
        } else if n_features as f64 / n_samples as f64 > 0.4 {
            ComplexityLevel::Medium
        } else {
            ComplexityLevel::Low
        };

        // Determine class imbalance level
        let class_imbalance = if imbalance_ratio <= 2.0 {
            ClassImbalanceLevel::Low
        } else if imbalance_ratio <= 5.0 {
            ClassImbalanceLevel::Medium
        } else {
            ClassImbalanceLevel::High
        };

        Ok(DatasetCharacteristics {
            n_samples,
            n_features,
            n_classes,
            samples_per_class: n_samples as f64 / n_classes as f64,
            class_imbalance,
            dimensionality,
            complexity,
        })
    }

    /// Select strategy based on adaptive configuration
    fn select_adaptive_strategy(
        &self,
        characteristics: &DatasetCharacteristics,
    ) -> SklResult<(MulticlassStrategy, String)> {
        match &self.config.strategy {
            AdaptiveStrategy::ContextAware {
                consider_imbalance,
                consider_complexity,
                consider_separability: _,
            } => {
                let mut reasoning = Vec::new();
                let mut score_ovr = 1.0;
                let mut score_ovo = 1.0;
                let mut score_ecoc = 1.0;

                if *consider_imbalance
                    && matches!(characteristics.class_imbalance, ClassImbalanceLevel::High)
                {
                    // High imbalance favors OvR with cost-sensitive learning
                    score_ovr += 2.0;
                    reasoning
                        .push("High class imbalance detected, favoring One-vs-Rest".to_string());
                }

                if *consider_complexity {
                    match characteristics.complexity {
                        ComplexityLevel::High => {
                            // High complexity favors simpler strategies
                            score_ovr += 1.5;
                            reasoning
                                .push("High complexity detected, favoring One-vs-Rest".to_string());
                        }
                        ComplexityLevel::Medium => {
                            // Medium complexity is good for all strategies
                            score_ovo += 1.0;
                            score_ecoc += 0.5;
                        }
                        ComplexityLevel::Low => {
                            // Low complexity allows for more sophisticated strategies
                            score_ovo += 1.5;
                            score_ecoc += 2.0;
                            reasoning.push("Low complexity detected, considering ECOC".to_string());
                        }
                    }
                }

                if characteristics.n_classes > 10 {
                    // Many classes favor OvO or ECOC
                    score_ovo += 1.0;
                    score_ecoc += 1.5;
                    reasoning
                        .push("Many classes detected, favoring One-vs-One or ECOC".to_string());
                } else if characteristics.n_classes < 5 {
                    // Few classes work well with any strategy
                    score_ovr += 0.5;
                    score_ovo += 0.5;
                }

                let selected_strategy = if score_ecoc >= score_ovr && score_ecoc >= score_ovo {
                    MulticlassStrategy::ECOC(ECOCStrategy::StdRng)
                } else if score_ovo >= score_ovr {
                    MulticlassStrategy::OneVsOne
                } else {
                    MulticlassStrategy::OneVsRest
                };

                Ok((selected_strategy, reasoning.join("; ")))
            }

            AdaptiveStrategy::DynamicGrouping {
                similarity_threshold: _,
                max_group_size: _,
            } => {
                // For now, implement a simple grouping strategy
                // In a full implementation, this would analyze class similarities
                Ok((
                    MulticlassStrategy::OneVsOne,
                    "Dynamic grouping selected One-vs-One as base strategy".to_string(),
                ))
            }

            AdaptiveStrategy::PerformanceBased {
                sample_size: _,
                improvement_threshold: _,
            } => {
                // For now, default to balanced approach
                Ok((
                    MulticlassStrategy::ECOC(ECOCStrategy::StdRng),
                    "Performance-based selection chose ECOC for robustness".to_string(),
                ))
            }

            AdaptiveStrategy::OnlineAdaptive {
                window_size: _,
                adaptation_frequency: _,
            } => {
                // For now, start with efficient strategy suitable for online learning
                Ok((
                    MulticlassStrategy::OneVsRest,
                    "Online adaptive selected One-vs-Rest for efficiency".to_string(),
                ))
            }
        }
    }

    /// Train the selected strategy
    fn train_selected_strategy(
        &self,
        strategy: &MulticlassStrategy,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> SklResult<DynamicMulticlassTrainedData<C::Fitted>> {
        match strategy {
            MulticlassStrategy::OneVsRest => {
                let ovr = OneVsRestClassifier::new(self.base_estimator.clone());
                let fitted = ovr.fit(x, y)?;
                Ok(DynamicMulticlassTrainedData::OneVsRest(
                    fitted.base_estimator,
                ))
            }

            MulticlassStrategy::OneVsOne => {
                let ovo = OneVsOneClassifier::new(self.base_estimator.clone());
                let fitted = ovo.fit(x, y)?;
                Ok(DynamicMulticlassTrainedData::OneVsOne(
                    fitted.base_estimator,
                ))
            }

            MulticlassStrategy::ECOC(ecoc_strategy) => {
                let ecoc = ECOCClassifier::new(self.base_estimator.clone())
                    .strategy(ecoc_strategy.clone());
                let fitted = ecoc.fit(x, y)?;
                Ok(DynamicMulticlassTrainedData::ECOC(fitted.base_estimator))
            }

            _ => Err(SklearsError::InvalidInput(
                "Adaptive strategy cannot be nested".to_string(),
            )),
        }
    }

    /// Fallback strategy for edge cases
    fn fit_fallback_strategy(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        strategy: MulticlassStrategy,
    ) -> SklResult<TrainedAdaptiveDecomposition<C::Fitted>> {
        let strategy_classifier = self.train_selected_strategy(&strategy, x, y)?;

        let data_characteristics = DatasetCharacteristics {
            n_samples: x.nrows(),
            n_features: x.ncols(),
            n_classes: classes.len(),
            samples_per_class: x.nrows() as f64 / classes.len() as f64,
            class_imbalance: ClassImbalanceLevel::Low,
            dimensionality: DimensionalityLevel::Low,
            complexity: ComplexityLevel::Low,
        };

        let adaptation_metadata = AdaptationMetadata {
            data_characteristics,
            selection_reasoning: format!("Fallback to {:?} due to edge case", strategy),
            adaptation_decisions: vec!["Used fallback strategy".to_string()],
        };

        let trained_data = AdaptiveDecompositionTrainedData {
            selected_strategy: strategy,
            strategy_classifier,
            classes: classes.clone(),
            adaptation_metadata,
        };

        Ok(AdaptiveDecompositionClassifier {
            base_estimator: trained_data,
            config: self.config.clone(),
            state: PhantomData,
        })
    }
}

impl<T> TrainedAdaptiveDecomposition<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    /// Get the selected strategy
    pub fn selected_strategy(&self) -> &MulticlassStrategy {
        &self.base_estimator.selected_strategy
    }

    /// Get adaptation metadata
    pub fn adaptation_metadata(&self) -> &AdaptationMetadata {
        &self.base_estimator.adaptation_metadata
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.base_estimator.classes
    }
}

impl<T> Predict<Array2<Float>, Array1<i32>> for TrainedAdaptiveDecomposition<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        // Delegate to the underlying trained strategy
        match &self.base_estimator.strategy_classifier {
            DynamicMulticlassTrainedData::OneVsRest(data) => {
                // For OneVsRest, use the trained classifiers and classes
                let n_samples = x.nrows();
                let n_classes = data.classes.len();
                let mut votes = Array2::zeros((n_samples, n_classes));

                for (class_idx, &_class_label) in data.classes.iter().enumerate() {
                    let scores = data.estimators[class_idx].predict(x)?;
                    for sample_idx in 0..n_samples {
                        votes[[sample_idx, class_idx]] = scores[sample_idx] as f64;
                    }
                }

                // Find the class with the highest vote for each sample
                let mut predictions = Array1::zeros(n_samples);
                for sample_idx in 0..n_samples {
                    let best_class_idx = votes
                        .row(sample_idx)
                        .iter()
                        .enumerate()
                        .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| {
                            a.partial_cmp(b).unwrap()
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    predictions[sample_idx] = data.classes[best_class_idx];
                }

                Ok(predictions)
            }
            DynamicMulticlassTrainedData::OneVsOne(data) => {
                // For OneVsOne, use majority voting
                let n_samples = x.nrows();
                let n_classes = data.classes.len();
                let mut votes = Array2::zeros((n_samples, n_classes));

                for (pair_idx, (i, j)) in data.class_pairs.iter().enumerate() {
                    let scores = data.estimators[pair_idx].predict(x)?;
                    for sample_idx in 0..n_samples {
                        let class_i_idx = data.classes.iter().position(|&c| c == *i).unwrap();
                        let class_j_idx = data.classes.iter().position(|&c| c == *j).unwrap();

                        if scores[sample_idx] > 0.0 {
                            votes[[sample_idx, class_i_idx]] += 1.0;
                        } else {
                            votes[[sample_idx, class_j_idx]] += 1.0;
                        }
                    }
                }

                // Find the class with the most votes for each sample
                let mut predictions = Array1::zeros(n_samples);
                for sample_idx in 0..n_samples {
                    let best_class_idx = votes
                        .row(sample_idx)
                        .iter()
                        .enumerate()
                        .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| {
                            a.partial_cmp(b).unwrap()
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    predictions[sample_idx] = data.classes[best_class_idx];
                }

                Ok(predictions)
            }
            DynamicMulticlassTrainedData::ECOC(data) => {
                // For ECOC, use Hamming distance decoding
                let n_samples = x.nrows();
                let n_classes = data.classes.len();
                let n_codes = data.code_matrix.ncols();

                // Predict with each binary classifier
                let mut binary_predictions = Array2::zeros((n_samples, n_codes));
                for code_idx in 0..n_codes {
                    let scores = data.estimators[code_idx].predict(x)?;
                    for sample_idx in 0..n_samples {
                        binary_predictions[[sample_idx, code_idx]] =
                            if scores[sample_idx] > 0.0 { 1.0 } else { -1.0 };
                    }
                }

                // Find the class with minimum Hamming distance for each sample
                let mut predictions = Array1::zeros(n_samples);
                for sample_idx in 0..n_samples {
                    let mut min_distance = f64::INFINITY;
                    let mut best_class = data.classes[0];

                    for (class_idx, &class_label) in data.classes.iter().enumerate() {
                        let mut distance = 0.0;
                        for code_idx in 0..n_codes {
                            let code_bit = data.code_matrix[[class_idx, code_idx]] as f64;
                            let pred_bit = binary_predictions[[sample_idx, code_idx]];
                            if (code_bit - pred_bit).abs() > 0.5 {
                                distance += 1.0;
                            }
                        }

                        if distance < min_distance {
                            min_distance = distance;
                            best_class = class_label;
                        }
                    }

                    predictions[sample_idx] = best_class;
                }

                Ok(predictions)
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockNativeClassifier;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_adaptive_strategy_creation() {
        let classifier = MockNativeClassifier::new();
        let adaptive = AdaptiveDecompositionClassifier::new(classifier)
            .strategy(AdaptiveStrategy::ContextAware {
                consider_imbalance: true,
                consider_complexity: true,
                consider_separability: true,
            })
            .min_classes(3)
            .max_classes(100);

        assert_eq!(adaptive.config.min_classes, 3);
        assert_eq!(adaptive.config.max_classes, 100);
        assert!(matches!(
            adaptive.config.strategy,
            AdaptiveStrategy::ContextAware { .. }
        ));
    }

    #[test]
    fn test_dataset_characteristics() {
        let characteristics = DatasetCharacteristics {
            n_samples: 100,
            n_features: 10,
            n_classes: 5,
            samples_per_class: 20.0,
            class_imbalance: ClassImbalanceLevel::Low,
            dimensionality: DimensionalityLevel::Low,
            complexity: ComplexityLevel::Low,
        };

        assert_eq!(characteristics.n_samples, 100);
        assert_eq!(characteristics.n_features, 10);
        assert_eq!(characteristics.n_classes, 5);
        assert_eq!(characteristics.samples_per_class, 20.0);
        assert_eq!(characteristics.class_imbalance, ClassImbalanceLevel::Low);
        assert_eq!(characteristics.dimensionality, DimensionalityLevel::Low);
        assert_eq!(characteristics.complexity, ComplexityLevel::Low);
    }

    #[test]
    fn test_adaptive_config_defaults() {
        let config = AdaptiveConfig::default();
        assert_eq!(config.min_classes, 3);
        assert_eq!(config.max_classes, 1000);
        assert!(matches!(
            config.strategy,
            AdaptiveStrategy::ContextAware { .. }
        ));
    }

    #[test]
    fn test_multiclass_strategy_variants() {
        let strategy = MulticlassStrategy::OneVsRest;
        assert!(matches!(strategy, MulticlassStrategy::OneVsRest));

        let strategy = MulticlassStrategy::OneVsOne;
        assert!(matches!(strategy, MulticlassStrategy::OneVsOne));

        let strategy = MulticlassStrategy::ECOC(ECOCStrategy::StdRng);
        assert!(matches!(strategy, MulticlassStrategy::ECOC(_)));

        let strategy = MulticlassStrategy::Auto;
        assert!(matches!(strategy, MulticlassStrategy::Auto));
    }

    #[test]
    fn test_selection_criteria_defaults() {
        let criteria = SelectionCriteria::default();
        assert!(matches!(criteria, SelectionCriteria::Combined { .. }));

        if let SelectionCriteria::Combined {
            accuracy_weight,
            time_weight,
            memory_weight,
        } = criteria
        {
            assert_eq!(accuracy_weight, 0.6);
            assert_eq!(time_weight, 0.3);
            assert_eq!(memory_weight, 0.1);
        }
    }
}
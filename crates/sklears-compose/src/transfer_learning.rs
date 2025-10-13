//! Transfer learning pipeline components
//!
//! This module provides transfer learning capabilities including pre-trained model
//! integration, feature extraction, fine-tuning strategies, and knowledge distillation.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::{Float, FloatBounds},
};
use std::collections::HashMap;

use crate::{PipelinePredictor, PipelineStep};

/// Pre-trained model wrapper for transfer learning
#[derive(Debug)]
pub struct PretrainedModel {
    /// The pre-trained model
    pub model: Box<dyn PipelinePredictor>,
    /// Feature extraction layers (frozen)
    pub frozen_layers: Vec<String>,
    /// Fine-tuning layers (trainable)
    pub trainable_layers: Vec<String>,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

impl PretrainedModel {
    /// Create a new pre-trained model wrapper
    #[must_use]
    pub fn new(model: Box<dyn PipelinePredictor>) -> Self {
        Self {
            model,
            frozen_layers: Vec::new(),
            trainable_layers: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add frozen layers
    #[must_use]
    pub fn with_frozen_layers(mut self, layers: Vec<String>) -> Self {
        self.frozen_layers = layers;
        self
    }

    /// Add trainable layers
    #[must_use]
    pub fn with_trainable_layers(mut self, layers: Vec<String>) -> Self {
        self.trainable_layers = layers;
        self
    }

    /// Add metadata
    #[must_use]
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Extract features using frozen layers
    pub fn extract_features(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // In a real implementation, this would extract features from intermediate layers
        // For now, we'll use the full model prediction as feature extraction
        let features = self.model.predict(x)?;
        Array2::from_shape_vec(
            (x.nrows(), features.len() / x.nrows()),
            features.into_raw_vec(),
        )
        .map_err(|e| SklearsError::InvalidData {
            reason: format!("Feature extraction failed: {e}"),
        })
    }
}

/// Transfer learning strategy
#[derive(Debug, Clone)]
pub enum TransferStrategy {
    /// Feature extraction only (freeze all layers)
    FeatureExtraction {
        /// Whether to add new classifier head
        add_classifier: bool,
    },
    /// Fine-tune all layers
    FineTuning {
        /// Learning rate for fine-tuning
        learning_rate: f64,
        /// Number of training epochs
        epochs: usize,
    },
    /// Progressive unfreezing
    ProgressiveUnfreezing {
        /// Learning rate schedule
        learning_rates: Vec<f64>,
        /// Layers to unfreeze per step
        unfreeze_schedule: Vec<Vec<String>>,
    },
    /// Layer-wise adaptive rates
    LayerWiseAdaptive {
        /// Learning rates per layer group
        layer_rates: HashMap<String, f64>,
    },
    /// Knowledge distillation
    KnowledgeDistillation {
        /// Temperature for softmax distillation
        temperature: f64,
        /// Weight for distillation loss
        distillation_weight: f64,
        /// Weight for task loss
        task_weight: f64,
    },
}

/// Transfer learning pipeline
#[derive(Debug)]
pub struct TransferLearningPipeline<S = Untrained> {
    state: S,
    pretrained_model: Option<PretrainedModel>,
    target_estimator: Option<Box<dyn PipelinePredictor>>,
    transfer_strategy: TransferStrategy,
    adaptation_config: AdaptationConfig,
}

/// Trained state for `TransferLearningPipeline`
#[derive(Debug)]
pub struct TransferLearningPipelineTrained {
    adapted_model: Box<dyn PipelinePredictor>,
    feature_extractor: Option<PretrainedModel>,
    transfer_strategy: TransferStrategy,
    adaptation_metrics: HashMap<String, f64>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

/// Configuration for adaptation process
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// Maximum number of adaptation steps
    pub max_steps: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Validation split ratio
    pub validation_split: f64,
    /// Batch size for adaptation
    pub batch_size: usize,
    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            max_steps: 1000,
            patience: 10,
            min_improvement: 1e-4,
            validation_split: 0.2,
            batch_size: 32,
            lr_schedule: LearningRateSchedule::Constant { rate: 0.001 },
        }
    }
}

/// Learning rate schedule types
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant { rate: f64 },
    /// Exponential decay
    ExponentialDecay {
        initial_rate: f64,
        decay_rate: f64,
        decay_steps: usize,
    },
    /// Step decay
    StepDecay {
        initial_rate: f64,
        drop_rate: f64,
        epochs_drop: usize,
    },
    /// Cosine annealing
    CosineAnnealing {
        max_rate: f64,
        min_rate: f64,
        cycle_length: usize,
    },
}

impl LearningRateSchedule {
    /// Get learning rate for a given step
    #[must_use]
    pub fn get_rate(&self, step: usize) -> f64 {
        match self {
            LearningRateSchedule::Constant { rate } => *rate,
            LearningRateSchedule::ExponentialDecay {
                initial_rate,
                decay_rate,
                decay_steps,
            } => initial_rate * decay_rate.powf(step as f64 / *decay_steps as f64),
            LearningRateSchedule::StepDecay {
                initial_rate,
                drop_rate,
                epochs_drop,
            } => initial_rate * drop_rate.powf((step / epochs_drop) as f64),
            LearningRateSchedule::CosineAnnealing {
                max_rate,
                min_rate,
                cycle_length,
            } => {
                let cycle_position = (step % cycle_length) as f64 / *cycle_length as f64;
                min_rate
                    + (max_rate - min_rate) * (1.0 + (std::f64::consts::PI * cycle_position).cos())
                        / 2.0
            }
        }
    }
}

impl TransferLearningPipeline<Untrained> {
    /// Create a new transfer learning pipeline
    #[must_use]
    pub fn new(
        pretrained_model: PretrainedModel,
        target_estimator: Box<dyn PipelinePredictor>,
    ) -> Self {
        Self {
            state: Untrained,
            pretrained_model: Some(pretrained_model),
            target_estimator: Some(target_estimator),
            transfer_strategy: TransferStrategy::FineTuning {
                learning_rate: 0.001,
                epochs: 10,
            },
            adaptation_config: AdaptationConfig::default(),
        }
    }

    /// Set the transfer strategy
    #[must_use]
    pub fn transfer_strategy(mut self, strategy: TransferStrategy) -> Self {
        self.transfer_strategy = strategy;
        self
    }

    /// Set the adaptation configuration
    #[must_use]
    pub fn adaptation_config(mut self, config: AdaptationConfig) -> Self {
        self.adaptation_config = config;
        self
    }

    /// Create a feature extraction pipeline
    #[must_use]
    pub fn feature_extraction(pretrained_model: PretrainedModel) -> Self {
        let strategy = TransferStrategy::FeatureExtraction {
            add_classifier: true,
        };
        Self {
            state: Untrained,
            pretrained_model: Some(pretrained_model),
            target_estimator: None,
            transfer_strategy: strategy,
            adaptation_config: AdaptationConfig::default(),
        }
    }

    /// Create a fine-tuning pipeline
    #[must_use]
    pub fn fine_tuning(
        pretrained_model: PretrainedModel,
        target_estimator: Box<dyn PipelinePredictor>,
        learning_rate: f64,
        epochs: usize,
    ) -> Self {
        let strategy = TransferStrategy::FineTuning {
            learning_rate,
            epochs,
        };
        Self {
            state: Untrained,
            pretrained_model: Some(pretrained_model),
            target_estimator: Some(target_estimator),
            transfer_strategy: strategy,
            adaptation_config: AdaptationConfig::default(),
        }
    }

    /// Create a knowledge distillation pipeline
    #[must_use]
    pub fn knowledge_distillation(
        teacher_model: PretrainedModel,
        student_estimator: Box<dyn PipelinePredictor>,
        temperature: f64,
        distillation_weight: f64,
        task_weight: f64,
    ) -> Self {
        let strategy = TransferStrategy::KnowledgeDistillation {
            temperature,
            distillation_weight,
            task_weight,
        };
        Self {
            state: Untrained,
            pretrained_model: Some(teacher_model),
            target_estimator: Some(student_estimator),
            transfer_strategy: strategy,
            adaptation_config: AdaptationConfig::default(),
        }
    }
}

impl Estimator for TransferLearningPipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>>
    for TransferLearningPipeline<Untrained>
{
    type Fitted = TransferLearningPipeline<TransferLearningPipelineTrained>;

    fn fit(
        mut self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let pretrained_model = self.pretrained_model.take().ok_or_else(|| {
            SklearsError::InvalidInput("No pretrained model provided".to_string())
        })?;

        let transfer_strategy = self.transfer_strategy.clone();
        let adapted_model = match &transfer_strategy {
            TransferStrategy::FeatureExtraction { add_classifier } => {
                self.apply_feature_extraction(&pretrained_model, x, y, *add_classifier)?
            }
            TransferStrategy::FineTuning {
                learning_rate,
                epochs,
            } => self.apply_fine_tuning(&pretrained_model, x, y, *learning_rate, *epochs)?,
            TransferStrategy::ProgressiveUnfreezing {
                learning_rates,
                unfreeze_schedule,
            } => self.apply_progressive_unfreezing(
                &pretrained_model,
                x,
                y,
                learning_rates,
                unfreeze_schedule,
            )?,
            TransferStrategy::LayerWiseAdaptive { layer_rates } => {
                self.apply_layer_wise_adaptive(&pretrained_model, x, y, layer_rates)?
            }
            TransferStrategy::KnowledgeDistillation {
                temperature,
                distillation_weight,
                task_weight,
            } => self.apply_knowledge_distillation(
                &pretrained_model,
                x,
                y,
                *temperature,
                *distillation_weight,
                *task_weight,
            )?,
        };

        let mut adaptation_metrics = HashMap::new();
        adaptation_metrics.insert(
            "adaptation_steps".to_string(),
            self.adaptation_config.max_steps as f64,
        );

        Ok(TransferLearningPipeline {
            state: TransferLearningPipelineTrained {
                adapted_model,
                feature_extractor: Some(pretrained_model),
                transfer_strategy: self.transfer_strategy,
                adaptation_metrics,
                n_features_in: x.ncols(),
                feature_names_in: None,
            },
            pretrained_model: None,
            target_estimator: None,
            transfer_strategy: TransferStrategy::FeatureExtraction {
                add_classifier: false,
            },
            adaptation_config: AdaptationConfig::default(),
        })
    }
}

impl TransferLearningPipeline<Untrained> {
    /// Apply feature extraction strategy
    fn apply_feature_extraction(
        &mut self,
        pretrained_model: &PretrainedModel,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
        add_classifier: bool,
    ) -> SklResult<Box<dyn PipelinePredictor>> {
        if add_classifier {
            if let Some(mut target_estimator) = self.target_estimator.take() {
                // Extract features and train classifier
                let features = pretrained_model.extract_features(x)?;
                let y_ref = y.as_ref().ok_or_else(|| {
                    SklearsError::InvalidInput("No target values provided".to_string())
                })?;
                target_estimator.fit(&features.view(), y_ref)?;
                Ok(target_estimator)
            } else {
                // Use pretrained model as-is
                Ok(Box::new(FeatureExtractorWrapper::new(pretrained_model)))
            }
        } else {
            // Use pretrained model for feature extraction only
            Ok(Box::new(FeatureExtractorWrapper::new(pretrained_model)))
        }
    }

    /// Apply fine-tuning strategy
    fn apply_fine_tuning(
        &mut self,
        pretrained_model: &PretrainedModel,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
        learning_rate: f64,
        epochs: usize,
    ) -> SklResult<Box<dyn PipelinePredictor>> {
        if let Some(mut target_estimator) = self.target_estimator.take() {
            // Simulate fine-tuning by training the target estimator
            for epoch in 0..epochs {
                let current_lr = learning_rate * (0.95_f64).powi(epoch as i32); // Simple decay
                let y_ref = y.as_ref().ok_or_else(|| {
                    SklearsError::InvalidInput("No target values provided".to_string())
                })?;
                target_estimator.fit(x, y_ref)?;
            }
            Ok(target_estimator)
        } else {
            // Return the pretrained model
            Err(SklearsError::InvalidInput(
                "Target estimator required for fine-tuning".to_string(),
            ))
        }
    }

    /// Apply progressive unfreezing strategy
    fn apply_progressive_unfreezing(
        &mut self,
        pretrained_model: &PretrainedModel,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
        learning_rates: &[f64],
        unfreeze_schedule: &[Vec<String>],
    ) -> SklResult<Box<dyn PipelinePredictor>> {
        if let Some(mut target_estimator) = self.target_estimator.take() {
            // Simulate progressive unfreezing
            for (step, (lr, layers)) in learning_rates
                .iter()
                .zip(unfreeze_schedule.iter())
                .enumerate()
            {
                // In a real implementation, we would unfreeze specific layers
                // For now, we'll just train with different learning rates
                let y_ref = y.as_ref().ok_or_else(|| {
                    SklearsError::InvalidInput("No target values provided".to_string())
                })?;
                target_estimator.fit(x, y_ref)?;
            }
            Ok(target_estimator)
        } else {
            Err(SklearsError::InvalidInput(
                "Target estimator required for progressive unfreezing".to_string(),
            ))
        }
    }

    /// Apply layer-wise adaptive rates strategy
    fn apply_layer_wise_adaptive(
        &mut self,
        pretrained_model: &PretrainedModel,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
        layer_rates: &HashMap<String, f64>,
    ) -> SklResult<Box<dyn PipelinePredictor>> {
        if let Some(mut target_estimator) = self.target_estimator.take() {
            // Simulate layer-wise adaptive training
            if let Some(y_ref) = y.as_ref() {
                target_estimator.fit(x, y_ref)?;
            } else {
                return Err(SklearsError::InvalidInput(
                    "Target y is required for fitting".to_string(),
                ));
            }
            Ok(target_estimator)
        } else {
            Err(SklearsError::InvalidInput(
                "Target estimator required for layer-wise adaptive rates".to_string(),
            ))
        }
    }

    /// Apply knowledge distillation strategy
    fn apply_knowledge_distillation(
        &mut self,
        teacher_model: &PretrainedModel,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
        temperature: f64,
        distillation_weight: f64,
        task_weight: f64,
    ) -> SklResult<Box<dyn PipelinePredictor>> {
        if let Some(mut student_estimator) = self.target_estimator.take() {
            // Get teacher predictions
            let teacher_predictions = teacher_model.model.predict(x)?;

            // Apply temperature scaling (softmax with temperature)
            let soft_targets = self.apply_temperature_scaling(&teacher_predictions, temperature);

            // Train student with both hard and soft targets
            // In a real implementation, this would involve a custom loss function
            if let Some(y_ref) = y.as_ref() {
                student_estimator.fit(x, y_ref)?;
            } else {
                return Err(SklearsError::InvalidInput(
                    "Target y is required for fitting student model".to_string(),
                ));
            }

            Ok(student_estimator)
        } else {
            Err(SklearsError::InvalidInput(
                "Student estimator required for knowledge distillation".to_string(),
            ))
        }
    }

    /// Apply temperature scaling for knowledge distillation
    fn apply_temperature_scaling(
        &self,
        predictions: &Array1<f64>,
        temperature: f64,
    ) -> Array1<f64> {
        if temperature == 1.0 {
            return predictions.clone();
        }

        // Apply temperature scaling: softmax(logits / T)
        let scaled_logits = predictions.mapv(|x| x / temperature);
        let max_logit = scaled_logits.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits = scaled_logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();

        exp_logits.mapv(|x| x / sum_exp)
    }
}

impl TransferLearningPipeline<TransferLearningPipelineTrained> {
    /// Predict using the adapted model
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        self.state.adapted_model.predict(x)
    }

    /// Get the adaptation metrics
    #[must_use]
    pub fn adaptation_metrics(&self) -> &HashMap<String, f64> {
        &self.state.adaptation_metrics
    }

    /// Get the feature extractor
    #[must_use]
    pub fn feature_extractor(&self) -> Option<&PretrainedModel> {
        self.state.feature_extractor.as_ref()
    }

    /// Extract features using the pretrained model
    pub fn extract_features(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        if let Some(ref extractor) = self.state.feature_extractor {
            extractor.extract_features(x)
        } else {
            Err(SklearsError::InvalidInput(
                "No feature extractor available".to_string(),
            ))
        }
    }

    /// Fine-tune on new data
    pub fn fine_tune(
        &mut self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        learning_rate: f64,
        epochs: usize,
    ) -> SklResult<()> {
        // Simulate fine-tuning the adapted model
        for _ in 0..epochs {
            self.state.adapted_model.fit(x, y)?;
        }
        Ok(())
    }
}

/// Wrapper for feature extraction functionality
#[derive(Debug)]
pub struct FeatureExtractorWrapper {
    extractor: PretrainedModel,
}

impl FeatureExtractorWrapper {
    #[must_use]
    pub fn new(extractor: &PretrainedModel) -> Self {
        // Clone the essential parts of the PretrainedModel
        Self {
            extractor: PretrainedModel {
                model: Box::new(MockExtractor::new()), // Placeholder
                frozen_layers: extractor.frozen_layers.clone(),
                trainable_layers: extractor.trainable_layers.clone(),
                metadata: extractor.metadata.clone(),
            },
        }
    }
}

impl PipelinePredictor for FeatureExtractorWrapper {
    fn fit(&mut self, _x: &ArrayView2<'_, Float>, _y: &ArrayView1<'_, Float>) -> SklResult<()> {
        // Feature extractors don't need fitting
        Ok(())
    }

    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let features = self.extractor.extract_features(x)?;
        // Return first column as prediction (placeholder)
        if features.ncols() > 0 {
            Ok(features.column(0).to_owned())
        } else {
            Ok(Array1::zeros(x.nrows()))
        }
    }

    fn clone_predictor(&self) -> Box<dyn PipelinePredictor> {
        Box::new(FeatureExtractorWrapper::new(&self.extractor))
    }
}

/// Mock feature extractor for testing
#[derive(Debug)]
pub struct MockExtractor {}

impl Default for MockExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MockExtractor {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl PipelinePredictor for MockExtractor {
    fn fit(&mut self, _x: &ArrayView2<'_, Float>, _y: &ArrayView1<'_, Float>) -> SklResult<()> {
        Ok(())
    }

    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        Ok(Array1::zeros(x.nrows()))
    }

    fn clone_predictor(&self) -> Box<dyn PipelinePredictor> {
        Box::new(MockExtractor::new())
    }
}

/// Domain adaptation utilities
pub mod domain_adaptation {
    use super::{
        Array1, Array2, ArrayView1, ArrayView2, Axis, Estimator, Fit, Float, FloatBounds, HashMap,
        PipelinePredictor, PipelineStep, Predict, SklResult, SklearsError, Untrained,
    };

    /// Domain adaptation strategy
    #[derive(Debug, Clone)]
    pub enum DomainAdaptationStrategy {
        /// Maximum Mean Discrepancy (MMD) alignment
        MMD { bandwidth: f64, lambda: f64 },
        /// Adversarial domain adaptation
        Adversarial {
            discriminator_lr: f64,
            generator_lr: f64,
            adversarial_weight: f64,
        },
        /// Correlation alignment (CORAL)
        CORAL { lambda: f64 },
        /// Deep domain confusion
        DeepDomainConfusion {
            adaptation_factor: f64,
            confusion_weight: f64,
        },
    }

    /// Domain adaptation pipeline
    #[derive(Debug)]
    pub struct DomainAdaptationPipeline<S = Untrained> {
        state: S,
        source_data: Option<(Array2<f64>, Array1<f64>)>,
        adaptation_strategy: DomainAdaptationStrategy,
        base_estimator: Option<Box<dyn PipelinePredictor>>,
    }

    /// Trained state for `DomainAdaptationPipeline`
    #[derive(Debug)]
    pub struct DomainAdaptationPipelineTrained {
        adapted_estimator: Box<dyn PipelinePredictor>,
        domain_alignment_metrics: HashMap<String, f64>,
        adaptation_strategy: DomainAdaptationStrategy,
        n_features_in: usize,
        feature_names_in: Option<Vec<String>>,
    }

    impl DomainAdaptationPipeline<Untrained> {
        /// Create a new domain adaptation pipeline
        #[must_use]
        pub fn new(
            source_data: (Array2<f64>, Array1<f64>),
            adaptation_strategy: DomainAdaptationStrategy,
            base_estimator: Box<dyn PipelinePredictor>,
        ) -> Self {
            Self {
                state: Untrained,
                source_data: Some(source_data),
                adaptation_strategy,
                base_estimator: Some(base_estimator),
            }
        }

        /// Create MMD-based domain adaptation
        #[must_use]
        pub fn mmd(
            source_data: (Array2<f64>, Array1<f64>),
            base_estimator: Box<dyn PipelinePredictor>,
            bandwidth: f64,
            lambda: f64,
        ) -> Self {
            Self::new(
                source_data,
                DomainAdaptationStrategy::MMD { bandwidth, lambda },
                base_estimator,
            )
        }

        /// Create adversarial domain adaptation
        #[must_use]
        pub fn adversarial(
            source_data: (Array2<f64>, Array1<f64>),
            base_estimator: Box<dyn PipelinePredictor>,
            discriminator_lr: f64,
            generator_lr: f64,
            adversarial_weight: f64,
        ) -> Self {
            Self::new(
                source_data,
                DomainAdaptationStrategy::Adversarial {
                    discriminator_lr,
                    generator_lr,
                    adversarial_weight,
                },
                base_estimator,
            )
        }
    }

    impl Estimator for DomainAdaptationPipeline<Untrained> {
        type Config = ();
        type Error = SklearsError;
        type Float = Float;

        fn config(&self) -> &Self::Config {
            &()
        }
    }

    impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>>
        for DomainAdaptationPipeline<Untrained>
    {
        type Fitted = DomainAdaptationPipeline<DomainAdaptationPipelineTrained>;

        fn fit(
            mut self,
            target_x: &ArrayView2<'_, Float>,
            target_y: &Option<&ArrayView1<'_, Float>>,
        ) -> SklResult<Self::Fitted> {
            let (source_x, source_y) = self
                .source_data
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidInput("No source data provided".to_string()))?;

            let mut base_estimator = self.base_estimator.take().ok_or_else(|| {
                SklearsError::InvalidInput("No base estimator provided".to_string())
            })?;

            let target_x_f64 = target_x.mapv(|v| v);

            // Apply domain adaptation strategy
            let alignment_metrics = match &self.adaptation_strategy {
                DomainAdaptationStrategy::MMD { bandwidth, lambda } => {
                    self.apply_mmd_adaptation(source_x, &target_x_f64, *bandwidth, *lambda)?
                }
                DomainAdaptationStrategy::Adversarial {
                    discriminator_lr,
                    generator_lr,
                    adversarial_weight,
                } => self.apply_adversarial_adaptation(
                    source_x,
                    &target_x_f64,
                    *discriminator_lr,
                    *generator_lr,
                    *adversarial_weight,
                )?,
                DomainAdaptationStrategy::CORAL { lambda } => {
                    self.apply_coral_adaptation(source_x, &target_x_f64, *lambda)?
                }
                DomainAdaptationStrategy::DeepDomainConfusion {
                    adaptation_factor,
                    confusion_weight,
                } => self.apply_deep_domain_confusion(
                    source_x,
                    &target_x_f64,
                    *adaptation_factor,
                    *confusion_weight,
                )?,
            };

            // Train the base estimator on source data
            let source_x_float = source_x.mapv(|v| v as Float);
            let source_y_float = source_y.mapv(|v| v as Float);
            base_estimator.fit(&source_x_float.view(), &source_y_float.view())?;

            Ok(DomainAdaptationPipeline {
                state: DomainAdaptationPipelineTrained {
                    adapted_estimator: base_estimator,
                    domain_alignment_metrics: alignment_metrics,
                    adaptation_strategy: self.adaptation_strategy,
                    n_features_in: target_x.ncols(),
                    feature_names_in: None,
                },
                source_data: None,
                adaptation_strategy: DomainAdaptationStrategy::MMD {
                    bandwidth: 1.0,
                    lambda: 1.0,
                },
                base_estimator: None,
            })
        }
    }

    impl DomainAdaptationPipeline<Untrained> {
        /// Apply MMD-based domain adaptation
        fn apply_mmd_adaptation(
            &self,
            source_x: &Array2<f64>,
            target_x: &Array2<f64>,
            bandwidth: f64,
            lambda: f64,
        ) -> SklResult<HashMap<String, f64>> {
            let mmd_distance = self.compute_mmd_distance(source_x, target_x, bandwidth);

            let mut metrics = HashMap::new();
            metrics.insert("mmd_distance".to_string(), mmd_distance);
            metrics.insert("bandwidth".to_string(), bandwidth);
            metrics.insert("lambda".to_string(), lambda);

            Ok(metrics)
        }

        /// Apply adversarial domain adaptation
        fn apply_adversarial_adaptation(
            &self,
            source_x: &Array2<f64>,
            target_x: &Array2<f64>,
            discriminator_lr: f64,
            generator_lr: f64,
            adversarial_weight: f64,
        ) -> SklResult<HashMap<String, f64>> {
            // Simulate adversarial training metrics
            let mut metrics = HashMap::new();
            metrics.insert("discriminator_accuracy".to_string(), 0.6); // Placeholder
            metrics.insert("generator_loss".to_string(), 1.2); // Placeholder
            metrics.insert("adversarial_weight".to_string(), adversarial_weight);

            Ok(metrics)
        }

        /// Apply CORAL adaptation
        fn apply_coral_adaptation(
            &self,
            source_x: &Array2<f64>,
            target_x: &Array2<f64>,
            lambda: f64,
        ) -> SklResult<HashMap<String, f64>> {
            let coral_loss = self.compute_coral_loss(source_x, target_x);

            let mut metrics = HashMap::new();
            metrics.insert("coral_loss".to_string(), coral_loss);
            metrics.insert("lambda".to_string(), lambda);

            Ok(metrics)
        }

        /// Apply deep domain confusion
        fn apply_deep_domain_confusion(
            &self,
            source_x: &Array2<f64>,
            target_x: &Array2<f64>,
            adaptation_factor: f64,
            confusion_weight: f64,
        ) -> SklResult<HashMap<String, f64>> {
            let confusion_loss = self.compute_confusion_loss(source_x, target_x);

            let mut metrics = HashMap::new();
            metrics.insert("confusion_loss".to_string(), confusion_loss);
            metrics.insert("adaptation_factor".to_string(), adaptation_factor);
            metrics.insert("confusion_weight".to_string(), confusion_weight);

            Ok(metrics)
        }

        /// Compute MMD distance between domains
        fn compute_mmd_distance(
            &self,
            source_x: &Array2<f64>,
            target_x: &Array2<f64>,
            bandwidth: f64,
        ) -> f64 {
            // Simplified MMD computation using mean differences
            let source_mean = source_x.mean_axis(Axis(0)).unwrap();
            let target_mean = target_x.mean_axis(Axis(0)).unwrap();
            let diff = &source_mean - &target_mean;
            (diff.mapv(|x| x * x).sum() / bandwidth).sqrt()
        }

        /// Compute CORAL loss (correlation alignment)
        fn compute_coral_loss(&self, source_x: &Array2<f64>, target_x: &Array2<f64>) -> f64 {
            // Simplified CORAL loss using covariance differences
            if source_x.ncols() != target_x.ncols() {
                return f64::INFINITY;
            }

            // Compute covariance matrices (simplified)
            let source_mean = source_x.mean_axis(Axis(0)).unwrap();
            let target_mean = target_x.mean_axis(Axis(0)).unwrap();

            // For simplicity, just compute variance differences
            let source_var = source_x.var_axis(Axis(0), 1.0);
            let target_var = target_x.var_axis(Axis(0), 1.0);

            (&source_var - &target_var).mapv(|x| x * x).sum()
        }

        /// Compute confusion loss for deep domain confusion
        fn compute_confusion_loss(&self, source_x: &Array2<f64>, target_x: &Array2<f64>) -> f64 {
            // Simplified confusion loss using feature distribution differences
            let source_std = source_x.std_axis(Axis(0), 1.0);
            let target_std = target_x.std_axis(Axis(0), 1.0);
            (&source_std - &target_std).mapv(|x| x * x).sum()
        }
    }

    impl DomainAdaptationPipeline<DomainAdaptationPipelineTrained> {
        /// Predict on target domain data
        pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
            self.state.adapted_estimator.predict(x)
        }

        /// Get domain alignment metrics
        #[must_use]
        pub fn alignment_metrics(&self) -> &HashMap<String, f64> {
            &self.state.domain_alignment_metrics
        }

        /// Measure domain discrepancy
        pub fn measure_domain_discrepancy(
            &self,
            source_x: &ArrayView2<'_, Float>,
            target_x: &ArrayView2<'_, Float>,
        ) -> SklResult<f64> {
            let source_x_f64 = source_x.mapv(|v| v);
            let target_x_f64 = target_x.mapv(|v| v);

            match &self.state.adaptation_strategy {
                DomainAdaptationStrategy::MMD { bandwidth, .. } => {
                    Ok(self.compute_mmd_distance(&source_x_f64, &target_x_f64, *bandwidth))
                }
                DomainAdaptationStrategy::CORAL { .. } => {
                    Ok(self.compute_coral_loss(&source_x_f64, &target_x_f64))
                }
                _ => Ok(0.0), // Placeholder for other strategies
            }
        }

        /// Compute MMD distance between domains
        fn compute_mmd_distance(
            &self,
            source_x: &Array2<f64>,
            target_x: &Array2<f64>,
            bandwidth: f64,
        ) -> f64 {
            let source_mean = source_x.mean_axis(Axis(0)).unwrap();
            let target_mean = target_x.mean_axis(Axis(0)).unwrap();
            let diff = &source_mean - &target_mean;
            (diff.mapv(|x| x * x).sum() / bandwidth).sqrt()
        }

        /// Compute CORAL loss
        fn compute_coral_loss(&self, source_x: &Array2<f64>, target_x: &Array2<f64>) -> f64 {
            if source_x.ncols() != target_x.ncols() {
                return f64::INFINITY;
            }

            let source_var = source_x.var_axis(Axis(0), 1.0);
            let target_var = target_x.var_axis(Axis(0), 1.0);

            (&source_var - &target_var).mapv(|x| x * x).sum()
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockPredictor;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pretrained_model() {
        let base_model = Box::new(MockPredictor::new());
        let pretrained = PretrainedModel::new(base_model)
            .with_frozen_layers(vec!["layer1".to_string(), "layer2".to_string()])
            .with_trainable_layers(vec!["layer3".to_string()]);

        assert_eq!(pretrained.frozen_layers.len(), 2);
        assert_eq!(pretrained.trainable_layers.len(), 1);
    }

    #[test]
    fn test_learning_rate_schedule() {
        let schedule = LearningRateSchedule::ExponentialDecay {
            initial_rate: 0.1,
            decay_rate: 0.9,
            decay_steps: 10,
        };

        let rate_0 = schedule.get_rate(0);
        let rate_10 = schedule.get_rate(10);

        assert_eq!(rate_0, 0.1);
        assert!(rate_10 < rate_0);
    }

    #[test]
    fn test_transfer_learning_pipeline() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 0.0];

        let pretrained_model = PretrainedModel::new(Box::new(MockPredictor::new()));
        let target_estimator = Box::new(MockPredictor::new());

        let pipeline =
            TransferLearningPipeline::fine_tuning(pretrained_model, target_estimator, 0.001, 5);

        let fitted_pipeline = pipeline.fit(&x.view(), &Some(&y.view())).unwrap();
        let predictions = fitted_pipeline.predict(&x.view()).unwrap();

        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    fn test_domain_adaptation_pipeline() {
        use domain_adaptation::*;

        let source_x = array![[1.0, 2.0], [3.0, 4.0]];
        let source_y = array![1.0, 0.0];
        let target_x = array![[2.0, 3.0], [4.0, 5.0]];

        let base_estimator = Box::new(MockPredictor::new());
        let pipeline =
            DomainAdaptationPipeline::mmd((source_x, source_y), base_estimator, 1.0, 0.1);

        let fitted_pipeline = pipeline.fit(&target_x.view(), &None).unwrap();
        let predictions = fitted_pipeline.predict(&target_x.view()).unwrap();

        assert_eq!(predictions.len(), target_x.nrows());
        assert!(fitted_pipeline
            .alignment_metrics()
            .contains_key("mmd_distance"));
    }
}

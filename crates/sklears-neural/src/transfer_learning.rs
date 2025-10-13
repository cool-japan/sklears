//! Transfer Learning Utilities
//!
//! This module provides comprehensive transfer learning capabilities including
//! model freezing, layer replacement, fine-tuning strategies, and domain adaptation.

use crate::layers::{Layer, ParameterizedLayer};
use crate::models::{Functional, Sequential};
use crate::weight_init::{InitStrategy, WeightInitializer};
use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::RandomExt;
use scirs2_core::random::SeedableRng;
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

/// Configuration for transfer learning
#[derive(Debug, Clone)]
pub struct TransferConfig<T: FloatBounds> {
    /// Layers to freeze during training
    pub frozen_layers: HashSet<String>,
    /// Learning rate multipliers for different layers
    pub layer_learning_rates: HashMap<String, T>,
    /// Fine-tuning strategy
    pub fine_tuning_strategy: FineTuningStrategy,
    /// Gradual unfreezing schedule
    pub unfreeze_schedule: Option<UnfreezeSchedule>,
    /// Domain adaptation settings
    pub domain_adaptation: Option<DomainAdaptationConfig<T>>,
    /// Whether to use discriminative learning rates
    pub discriminative_lr: bool,
    /// Base learning rate for unfrozen layers
    pub base_learning_rate: T,
}

impl<T: FloatBounds> Default for TransferConfig<T> {
    fn default() -> Self {
        Self {
            frozen_layers: HashSet::new(),
            layer_learning_rates: HashMap::new(),
            fine_tuning_strategy: FineTuningStrategy::FineTuneAll,
            unfreeze_schedule: None,
            domain_adaptation: None,
            discriminative_lr: false,
            base_learning_rate: T::from(1e-3).unwrap(),
        }
    }
}

/// Fine-tuning strategies for transfer learning
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FineTuningStrategy {
    /// Freeze all layers, only train new layers
    FeatureExtraction,
    /// Fine-tune all layers
    FineTuneAll,
    /// Fine-tune only top layers
    FineTuneTop { num_layers: usize },
    /// Gradual unfreezing strategy
    GradualUnfreeze,
    /// Layer-wise adaptive fine-tuning
    LayerWiseAdaptive,
    /// Task-specific fine-tuning
    TaskSpecific,
}

/// Schedule for gradual unfreezing of layers
#[derive(Debug, Clone)]
pub struct UnfreezeSchedule {
    /// Number of epochs between unfreezing steps
    pub epochs_per_step: usize,
    /// Number of layers to unfreeze at each step
    pub layers_per_step: usize,
    /// Start from top (last) layers or bottom (first) layers
    pub unfreeze_direction: UnfreezeDirection,
}

/// Direction for gradual unfreezing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnfreezeDirection {
    /// Unfreeze from top (output) layers to bottom (input) layers
    TopToBottom,
    /// Unfreeze from bottom (input) layers to top (output) layers
    BottomToTop,
    /// Unfreeze middle layers first, then expand outward
    MiddleOut,
}

/// Domain adaptation configuration
#[derive(Debug, Clone)]
pub struct DomainAdaptationConfig<T: FloatBounds> {
    /// Domain adaptation technique
    pub technique: DomainAdaptationTechnique,
    /// Adaptation loss weight
    pub adaptation_weight: T,
    /// Number of adaptation iterations
    pub adaptation_iterations: usize,
    /// Whether to use adversarial training
    pub adversarial_training: bool,
}

/// Domain adaptation techniques
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DomainAdaptationTechnique {
    MMD,
    DANN,
    CORAL,
    AdaBN,
}

/// Transfer learning manager for handling model adaptation
#[derive(Debug, Clone)]
pub struct TransferLearningManager<T: FloatBounds> {
    /// Transfer learning configuration
    config: TransferConfig<T>,
    /// Current epoch for scheduling
    current_epoch: usize,
    /// Layer freeze status
    layer_freeze_status: HashMap<String, bool>,
    /// Layer learning rate multipliers
    layer_lr_multipliers: HashMap<String, T>,
    /// Training history for adaptation
    training_history: Vec<TransferMetrics<T>>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> TransferLearningManager<T> {
    /// Create a new transfer learning manager
    pub fn new(config: TransferConfig<T>) -> Self {
        let mut layer_freeze_status = HashMap::new();
        let mut layer_lr_multipliers = HashMap::new();

        // Initialize freeze status
        for layer_name in &config.frozen_layers {
            layer_freeze_status.insert(layer_name.clone(), true);
        }

        // Initialize learning rate multipliers
        for (layer_name, lr_mult) in &config.layer_learning_rates {
            layer_lr_multipliers.insert(layer_name.clone(), *lr_mult);
        }

        Self {
            config,
            current_epoch: 0,
            layer_freeze_status,
            layer_lr_multipliers,
            training_history: Vec::new(),
        }
    }

    /// Freeze specific layers in a model
    pub fn freeze_layers(&mut self, layer_names: &[String]) -> NeuralResult<()> {
        for layer_name in layer_names {
            self.layer_freeze_status.insert(layer_name.clone(), true);
        }
        Ok(())
    }

    /// Unfreeze specific layers in a model
    pub fn unfreeze_layers(&mut self, layer_names: &[String]) -> NeuralResult<()> {
        for layer_name in layer_names {
            self.layer_freeze_status.insert(layer_name.clone(), false);
        }
        Ok(())
    }

    /// Check if a layer is frozen
    pub fn is_layer_frozen(&self, layer_name: &str) -> bool {
        self.layer_freeze_status
            .get(layer_name)
            .copied()
            .unwrap_or(false)
    }

    /// Get learning rate multiplier for a layer
    pub fn get_layer_lr_multiplier(&self, layer_name: &str) -> T {
        self.layer_lr_multipliers
            .get(layer_name)
            .copied()
            .unwrap_or(T::one())
    }

    /// Update epoch and apply scheduling
    pub fn update_epoch(&mut self, epoch: usize) -> NeuralResult<()> {
        self.current_epoch = epoch;

        // Apply unfreezing schedule if configured
        if let Some(schedule) = self.config.unfreeze_schedule.clone() {
            self.apply_unfreeze_schedule(&schedule)?;
        }

        Ok(())
    }

    /// Apply gradual unfreezing schedule
    fn apply_unfreeze_schedule(&mut self, schedule: &UnfreezeSchedule) -> NeuralResult<()> {
        if self.current_epoch % schedule.epochs_per_step == 0 && self.current_epoch > 0 {
            let step = self.current_epoch / schedule.epochs_per_step;
            let layers_to_unfreeze = self.select_layers_for_unfreezing(schedule, step)?;
            self.unfreeze_layers(&layers_to_unfreeze)?;
        }
        Ok(())
    }

    /// Select layers for unfreezing based on schedule
    fn select_layers_for_unfreezing(
        &self,
        schedule: &UnfreezeSchedule,
        step: usize,
    ) -> NeuralResult<Vec<String>> {
        let frozen_layers: Vec<String> = self
            .layer_freeze_status
            .iter()
            .filter(|(_, &is_frozen)| is_frozen)
            .map(|(name, _)| name.clone())
            .collect();

        let start_idx = step * schedule.layers_per_step;
        let end_idx = ((step + 1) * schedule.layers_per_step).min(frozen_layers.len());

        if start_idx >= frozen_layers.len() {
            return Ok(Vec::new());
        }

        let selected_layers = match schedule.unfreeze_direction {
            UnfreezeDirection::TopToBottom => frozen_layers[start_idx..end_idx].to_vec(),
            UnfreezeDirection::BottomToTop => {
                let mut layers = frozen_layers.clone();
                layers.reverse();
                layers[start_idx..end_idx].to_vec()
            }
            UnfreezeDirection::MiddleOut => {
                // Unfreeze from middle outward
                let middle = frozen_layers.len() / 2;
                let mut selected = Vec::new();

                for i in 0..schedule.layers_per_step {
                    if step * schedule.layers_per_step + i >= frozen_layers.len() {
                        break;
                    }

                    let offset = i / 2;
                    if i % 2 == 0 {
                        // Go towards end
                        if middle + offset < frozen_layers.len() {
                            selected.push(frozen_layers[middle + offset].clone());
                        }
                    } else {
                        // Go towards beginning
                        if offset < middle {
                            selected.push(frozen_layers[middle - offset - 1].clone());
                        }
                    }
                }
                selected
            }
        };

        Ok(selected_layers)
    }

    /// Apply discriminative learning rates
    pub fn apply_discriminative_learning_rates(
        &mut self,
        layer_names: &[String],
    ) -> NeuralResult<()> {
        if !self.config.discriminative_lr {
            return Ok(());
        }

        // Apply decreasing learning rates for lower layers
        let num_layers = layer_names.len();
        for (i, layer_name) in layer_names.iter().enumerate() {
            let layer_depth = (num_layers - i - 1) as f64;
            let lr_multiplier = T::from(0.1_f64.powf(layer_depth / num_layers as f64)).unwrap();
            self.layer_lr_multipliers
                .insert(layer_name.clone(), lr_multiplier);
        }

        Ok(())
    }

    /// Record training metrics
    pub fn record_metrics(&mut self, metrics: TransferMetrics<T>) {
        self.training_history.push(metrics);
    }

    /// Get training history
    pub fn get_training_history(&self) -> &[TransferMetrics<T>] {
        &self.training_history
    }

    /// Calculate transfer learning effectiveness
    pub fn calculate_transfer_effectiveness(&self) -> Option<T> {
        if self.training_history.len() < 2 {
            return None;
        }

        let initial_loss = self.training_history[0].validation_loss;
        let final_loss = self.training_history.last().unwrap().validation_loss;

        Some((initial_loss - final_loss) / initial_loss)
    }
}

/// Metrics for transfer learning evaluation
#[derive(Debug, Clone)]
pub struct TransferMetrics<T: FloatBounds> {
    /// Current epoch
    pub epoch: usize,
    /// Training loss
    pub training_loss: T,
    /// Validation loss
    pub validation_loss: T,
    /// Number of frozen layers
    pub frozen_layer_count: usize,
    /// Learning rate statistics
    pub lr_stats: LearningRateStats<T>,
    /// Domain adaptation loss (if applicable)
    pub domain_adaptation_loss: Option<T>,
}

/// Learning rate statistics
#[derive(Debug, Clone)]
pub struct LearningRateStats<T: FloatBounds> {
    /// Mean learning rate across layers
    pub mean_lr: T,
    /// Maximum learning rate
    pub max_lr: T,
    /// Minimum learning rate
    pub min_lr: T,
    /// Learning rate variance
    pub lr_variance: T,
}

/// Model adapter for replacing layers during transfer learning
pub struct ModelAdapter<T: FloatBounds> {
    /// Layer replacement mapping
    layer_replacements: HashMap<String, Box<dyn Layer<T>>>,
    /// Initialization strategy for new layers
    init_strategy: InitStrategy,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> ModelAdapter<T> {
    /// Create a new model adapter
    pub fn new(init_strategy: InitStrategy) -> Self {
        Self {
            layer_replacements: HashMap::new(),
            init_strategy,
        }
    }

    /// Add a layer replacement
    pub fn replace_layer(&mut self, layer_name: String, new_layer: Box<dyn Layer<T>>) {
        self.layer_replacements.insert(layer_name, new_layer);
    }

    /// Remove the final classification layer and add a new one
    pub fn replace_classifier(
        &mut self,
        num_classes: usize,
        hidden_size: usize,
        layer_name: String,
    ) -> NeuralResult<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let initializer: WeightInitializer<T> = WeightInitializer::new(self.init_strategy.clone());

        // Create new classification layer (simple dense layer implementation)
        let weights = initializer.initialize_2d(&mut rng, (hidden_size, num_classes))?;
        let bias: Array1<T> = Array1::zeros(num_classes);

        // For now, we'll create a placeholder - in practice this would be a proper Dense layer
        // self.layer_replacements.insert(layer_name, Box::new(DenseLayer::new(weights, bias)));

        Ok(())
    }

    /// Apply layer replacements to a model
    pub fn apply_replacements<M>(&self, model: &mut M) -> NeuralResult<()>
    where
        M: HasReplaceableLayer<T>,
    {
        for (layer_name, replacement_layer) in &self.layer_replacements {
            model.replace_layer(layer_name, replacement_layer.as_ref())?;
        }
        Ok(())
    }
}

impl<T: FloatBounds> std::fmt::Debug for ModelAdapter<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelAdapter")
            .field(
                "layer_replacements",
                &format!("{} layers", self.layer_replacements.len()),
            )
            .field("init_strategy", &self.init_strategy)
            .finish()
    }
}

/// Trait for models that support layer replacement
pub trait HasReplaceableLayer<T: FloatBounds> {
    /// Replace a layer in the model
    fn replace_layer(&mut self, layer_name: &str, new_layer: &dyn Layer<T>) -> NeuralResult<()>;

    /// Get layer names
    fn get_layer_names(&self) -> Vec<String>;
}

/// Feature extractor for using pre-trained models as feature extractors
#[derive(Debug, Clone)]
pub struct FeatureExtractor<T: FloatBounds> {
    /// Extract features up to this layer
    extract_layer: String,
    /// Whether to apply global pooling
    global_pooling: bool,
    /// Pooling strategy
    pooling_strategy: PoolingStrategy,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T: FloatBounds> FeatureExtractor<T> {
    /// Create a new feature extractor
    pub fn new(
        extract_layer: String,
        global_pooling: bool,
        pooling_strategy: PoolingStrategy,
    ) -> Self {
        Self {
            extract_layer,
            global_pooling,
            pooling_strategy,
            _phantom: PhantomData,
        }
    }

    /// Extract features from input using the specified layer
    pub fn extract_features<M>(&self, model: &M, input: &Array3<T>) -> NeuralResult<Array2<T>>
    where
        M: HasFeatureExtraction<T>,
    {
        let features = model.extract_features_at_layer(&self.extract_layer, input)?;

        if self.global_pooling {
            self.apply_global_pooling(&features)
        } else {
            // Flatten features
            let (batch_size, _, _) = features.dim();
            let flattened_size = features.len() / batch_size;
            Ok(features.into_shape((batch_size, flattened_size)).unwrap())
        }
    }

    /// Apply global pooling to features
    fn apply_global_pooling(&self, features: &Array3<T>) -> NeuralResult<Array2<T>> {
        let (batch_size, _, num_features) = features.dim();
        let mut pooled = Array2::zeros((batch_size, num_features));

        for b in 0..batch_size {
            for f in 0..num_features {
                let feature_map = features.slice(scirs2_core::ndarray::s![b, .., f]);
                let pooled_value = match self.pooling_strategy {
                    PoolingStrategy::Mean => feature_map.mean().unwrap_or(T::zero()),
                    PoolingStrategy::Max => feature_map.fold(T::zero(), |acc, &x| acc.max(x)),
                    PoolingStrategy::Min => feature_map.fold(T::zero(), |acc, &x| acc.min(x)),
                };
                pooled[[b, f]] = pooled_value;
            }
        }

        Ok(pooled)
    }
}

/// Pooling strategies for feature extraction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolingStrategy {
    /// Mean pooling
    Mean,
    /// Max pooling
    Max,
    /// Min pooling
    Min,
}

/// Trait for models that support feature extraction
pub trait HasFeatureExtraction<T: FloatBounds> {
    /// Extract features at a specific layer
    fn extract_features_at_layer(
        &self,
        layer_name: &str,
        input: &Array3<T>,
    ) -> NeuralResult<Array3<T>>;
}

/// Domain adaptation utilities
#[derive(Debug, Clone)]
pub struct DomainAdapter<T: FloatBounds> {
    /// Domain adaptation configuration
    config: DomainAdaptationConfig<T>,
    /// Source domain statistics
    source_stats: Option<DomainStatistics<T>>,
    /// Target domain statistics
    target_stats: Option<DomainStatistics<T>>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> DomainAdapter<T> {
    /// Create a new domain adapter
    pub fn new(config: DomainAdaptationConfig<T>) -> Self {
        Self {
            config,
            source_stats: None,
            target_stats: None,
        }
    }

    /// Compute domain statistics
    pub fn compute_domain_statistics(
        &mut self,
        source_data: &Array3<T>,
        target_data: &Array3<T>,
    ) -> NeuralResult<()> {
        self.source_stats = Some(self.compute_statistics(source_data)?);
        self.target_stats = Some(self.compute_statistics(target_data)?);
        Ok(())
    }

    /// Apply domain adaptation
    pub fn adapt_features(&self, features: &Array3<T>, is_source: bool) -> NeuralResult<Array3<T>> {
        match self.config.technique {
            DomainAdaptationTechnique::CORAL => self.apply_coral_adaptation(features, is_source),
            DomainAdaptationTechnique::AdaBN => self.apply_adaptive_batch_norm(features, is_source),
            _ => {
                // Other techniques require more complex implementation
                Ok(features.clone())
            }
        }
    }

    /// Compute statistics for a domain
    fn compute_statistics(&self, data: &Array3<T>) -> NeuralResult<DomainStatistics<T>> {
        let (batch_size, seq_len, feature_dim) = data.dim();
        let total_samples = batch_size * seq_len;

        // Compute mean
        let mut mean = Array1::zeros(feature_dim);
        for b in 0..batch_size {
            for s in 0..seq_len {
                for f in 0..feature_dim {
                    mean[f] = mean[f] + data[[b, s, f]];
                }
            }
        }
        mean = mean / T::from(total_samples).unwrap();

        // Compute covariance
        let mut covariance = Array2::zeros((feature_dim, feature_dim));
        for b in 0..batch_size {
            for s in 0..seq_len {
                for i in 0..feature_dim {
                    for j in 0..feature_dim {
                        let diff_i = data[[b, s, i]] - mean[i];
                        let diff_j = data[[b, s, j]] - mean[j];
                        covariance[[i, j]] = covariance[[i, j]] + diff_i * diff_j;
                    }
                }
            }
        }
        covariance = covariance / T::from(total_samples - 1).unwrap();

        Ok(DomainStatistics { mean, covariance })
    }

    /// Apply CORAL (Correlation Alignment) adaptation
    fn apply_coral_adaptation(
        &self,
        features: &Array3<T>,
        is_source: bool,
    ) -> NeuralResult<Array3<T>> {
        let stats = if is_source {
            &self.source_stats
        } else {
            &self.target_stats
        };

        if let Some(domain_stats) = stats {
            // For CORAL, we would typically align second-order statistics
            // This is a simplified implementation
            let (batch_size, seq_len, feature_dim) = features.dim();
            let mut adapted = features.clone();

            // Center the features
            for b in 0..batch_size {
                for s in 0..seq_len {
                    for f in 0..feature_dim {
                        adapted[[b, s, f]] = adapted[[b, s, f]] - domain_stats.mean[f];
                    }
                }
            }

            Ok(adapted)
        } else {
            Ok(features.clone())
        }
    }

    /// Apply adaptive batch normalization
    fn apply_adaptive_batch_norm(
        &self,
        features: &Array3<T>,
        _is_source: bool,
    ) -> NeuralResult<Array3<T>> {
        // Simplified adaptive batch normalization
        let (batch_size, seq_len, feature_dim) = features.dim();
        let mut normalized = Array3::zeros((batch_size, seq_len, feature_dim));

        for f in 0..feature_dim {
            // Compute statistics for this feature across all samples
            let mut sum = T::zero();
            let mut count = 0;

            for b in 0..batch_size {
                for s in 0..seq_len {
                    sum = sum + features[[b, s, f]];
                    count += 1;
                }
            }

            let mean = sum / T::from(count).unwrap();

            let mut variance_sum = T::zero();
            for b in 0..batch_size {
                for s in 0..seq_len {
                    let diff = features[[b, s, f]] - mean;
                    variance_sum = variance_sum + diff * diff;
                }
            }

            let variance = variance_sum / T::from(count).unwrap();
            let std = (variance + T::from(1e-5).unwrap()).sqrt();

            // Normalize
            for b in 0..batch_size {
                for s in 0..seq_len {
                    normalized[[b, s, f]] = (features[[b, s, f]] - mean) / std;
                }
            }
        }

        Ok(normalized)
    }
}

/// Domain statistics for adaptation
#[derive(Debug, Clone)]
pub struct DomainStatistics<T: FloatBounds> {
    /// Feature means
    pub mean: Array1<T>,
    /// Feature covariance matrix
    pub covariance: Array2<T>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::essentials::Normal;
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_transfer_learning_manager_creation() {
        let config = TransferConfig::default();
        let manager = TransferLearningManager::<f64>::new(config);
        assert_eq!(manager.current_epoch, 0);
    }

    #[test]
    fn test_layer_freezing() {
        let mut config = TransferConfig::<f64>::default();
        config.frozen_layers.insert("layer1".to_string());

        let mut manager = TransferLearningManager::new(config);
        assert!(manager.is_layer_frozen("layer1"));
        assert!(!manager.is_layer_frozen("layer2"));

        manager.unfreeze_layers(&["layer1".to_string()]).unwrap();
        assert!(!manager.is_layer_frozen("layer1"));
    }

    #[test]
    fn test_learning_rate_multipliers() {
        let mut config = TransferConfig::<f64>::default();
        config
            .layer_learning_rates
            .insert("layer1".to_string(), 0.5);

        let manager = TransferLearningManager::new(config);
        assert_eq!(manager.get_layer_lr_multiplier("layer1"), 0.5);
        assert_eq!(manager.get_layer_lr_multiplier("layer2"), 1.0);
    }

    #[test]
    fn test_model_adapter_creation() {
        let adapter = ModelAdapter::<f64>::new(InitStrategy::XavierUniform);
        assert_eq!(adapter.layer_replacements.len(), 0);
    }

    #[test]
    fn test_feature_extractor() {
        let extractor =
            FeatureExtractor::<f64>::new("conv_layer".to_string(), true, PoolingStrategy::Mean);
        assert_eq!(extractor.extract_layer, "conv_layer");
        assert!(extractor.global_pooling);
    }

    #[test]
    fn test_domain_adapter() {
        let config = DomainAdaptationConfig {
            technique: DomainAdaptationTechnique::CORAL,
            adaptation_weight: 0.1,
            adaptation_iterations: 100,
            adversarial_training: false,
        };

        let adapter = DomainAdapter::<f64>::new(config);
        assert!(adapter.source_stats.is_none());
        assert!(adapter.target_stats.is_none());
    }

    #[test]
    fn test_domain_statistics_computation() {
        let config = DomainAdaptationConfig {
            technique: DomainAdaptationTechnique::CORAL,
            adaptation_weight: 0.1,
            adaptation_iterations: 100,
            adversarial_training: false,
        };

        let mut adapter = DomainAdapter::new(config);

        let source_data = Array3::from_shape_fn((10, 5, 8), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let target_data = Array3::from_shape_fn((10, 5, 8), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });

        let result = adapter.compute_domain_statistics(&source_data, &target_data);
        assert!(result.is_ok());
        assert!(adapter.source_stats.is_some());
        assert!(adapter.target_stats.is_some());
    }

    #[test]
    fn test_unfreeze_schedule() {
        let schedule = UnfreezeSchedule {
            epochs_per_step: 5,
            layers_per_step: 2,
            unfreeze_direction: UnfreezeDirection::TopToBottom,
        };

        let mut config = TransferConfig::<f64>::default();
        config.unfreeze_schedule = Some(schedule);

        // Add some frozen layers
        for i in 0..6 {
            config.frozen_layers.insert(format!("layer_{}", i));
        }

        let mut manager = TransferLearningManager::new(config);

        // Update to epoch 5 (should trigger unfreezing)
        manager.update_epoch(5).unwrap();

        // Some layers should have been unfrozen
        let frozen_count = manager.layer_freeze_status.values().filter(|&&v| v).count();
        assert!(frozen_count < 6);
    }

    #[test]
    fn test_transfer_effectiveness_calculation() {
        let config = TransferConfig::default();
        let mut manager = TransferLearningManager::new(config);

        // Add some training history
        manager.record_metrics(TransferMetrics {
            epoch: 0,
            training_loss: 1.0,
            validation_loss: 1.0,
            frozen_layer_count: 5,
            lr_stats: LearningRateStats {
                mean_lr: 0.001,
                max_lr: 0.001,
                min_lr: 0.001,
                lr_variance: 0.0,
            },
            domain_adaptation_loss: None,
        });

        manager.record_metrics(TransferMetrics {
            epoch: 10,
            training_loss: 0.5,
            validation_loss: 0.6,
            frozen_layer_count: 3,
            lr_stats: LearningRateStats {
                mean_lr: 0.001,
                max_lr: 0.001,
                min_lr: 0.001,
                lr_variance: 0.0,
            },
            domain_adaptation_loss: None,
        });

        let effectiveness = manager.calculate_transfer_effectiveness();
        assert!(effectiveness.is_some());
        assert!(effectiveness.unwrap() > 0.0); // Should show improvement
    }
}

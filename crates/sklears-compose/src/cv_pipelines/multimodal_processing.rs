//! Multi-modal processing and fusion strategies
//!
//! This module provides comprehensive multi-modal processing capabilities including
//! fusion strategies, cross-modal learning, temporal alignment, and synchronization
//! methods for computer vision pipelines.

use super::types_config::{
    CrossModalStrategy, FusionStrategy, InterpolationMethod, Modality, SyncMethod,
    TransformParameter,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Multi-modal processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Enable multi-modal processing
    pub enabled: bool,
    /// Supported modalities
    pub modalities: Vec<Modality>,
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy,
    /// Temporal alignment configuration
    pub temporal_alignment: TemporalAlignmentConfig,
    /// Cross-modal learning configuration
    pub cross_modal_learning: CrossModalLearningConfig,
    /// Modality-specific weights
    pub modality_weights: HashMap<Modality, f64>,
    /// Synchronization requirements
    pub sync_requirements: SynchronizationRequirements,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            modalities: vec![Modality::Visual],
            fusion_strategy: FusionStrategy::LateFusion,
            temporal_alignment: TemporalAlignmentConfig::default(),
            cross_modal_learning: CrossModalLearningConfig::default(),
            modality_weights: HashMap::new(),
            sync_requirements: SynchronizationRequirements::default(),
        }
    }
}

impl MultiModalConfig {
    /// Create configuration for vision-audio fusion
    #[must_use]
    pub fn vision_audio() -> Self {
        let mut weights = HashMap::new();
        weights.insert(Modality::Visual, 0.7);
        weights.insert(Modality::Audio, 0.3);

        Self {
            enabled: true,
            modalities: vec![Modality::Visual, Modality::Audio],
            fusion_strategy: FusionStrategy::EarlyFusion,
            temporal_alignment: TemporalAlignmentConfig::strict(),
            cross_modal_learning: CrossModalLearningConfig::contrastive(),
            modality_weights: weights,
            sync_requirements: SynchronizationRequirements::hardware(),
        }
    }

    /// Create configuration for vision-depth fusion
    #[must_use]
    pub fn vision_depth() -> Self {
        let mut weights = HashMap::new();
        weights.insert(Modality::Visual, 0.6);
        weights.insert(Modality::Depth, 0.4);

        Self {
            enabled: true,
            modalities: vec![Modality::Visual, Modality::Depth],
            fusion_strategy: FusionStrategy::HybridFusion,
            temporal_alignment: TemporalAlignmentConfig::relaxed(),
            cross_modal_learning: CrossModalLearningConfig::shared_representation(),
            modality_weights: weights,
            sync_requirements: SynchronizationRequirements::software(),
        }
    }

    /// Create configuration for multi-sensor fusion (vision, `LiDAR`, radar)
    #[must_use]
    pub fn multi_sensor() -> Self {
        let mut weights = HashMap::new();
        weights.insert(Modality::Visual, 0.4);
        weights.insert(Modality::LiDAR, 0.35);
        weights.insert(Modality::Radar, 0.25);

        Self {
            enabled: true,
            modalities: vec![Modality::Visual, Modality::LiDAR, Modality::Radar],
            fusion_strategy: FusionStrategy::AttentionFusion,
            temporal_alignment: TemporalAlignmentConfig::precise(),
            cross_modal_learning: CrossModalLearningConfig::alignment(),
            modality_weights: weights,
            sync_requirements: SynchronizationRequirements::gps(),
        }
    }

    /// Get weight for a specific modality
    #[must_use]
    pub fn get_modality_weight(&self, modality: &Modality) -> f64 {
        self.modality_weights.get(modality).copied().unwrap_or(1.0)
    }

    /// Set weight for a specific modality
    pub fn set_modality_weight(&mut self, modality: Modality, weight: f64) {
        self.modality_weights.insert(modality, weight);
    }

    /// Validate modality configuration
    pub fn validate(&self) -> Result<(), MultiModalError> {
        if self.enabled && self.modalities.is_empty() {
            return Err(MultiModalError::NoModalities);
        }

        // Check that weights sum to approximately 1.0
        let total_weight: f64 = self.modality_weights.values().sum();
        if (total_weight - 1.0).abs() > 0.1 {
            return Err(MultiModalError::InvalidWeights {
                total: total_weight,
            });
        }

        Ok(())
    }
}

/// Temporal alignment configuration for multi-modal data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAlignmentConfig {
    /// Enable temporal alignment
    pub enabled: bool,
    /// Synchronization method
    pub sync_method: SyncMethod,
    /// Maximum allowed time offset
    pub max_time_offset: Duration,
    /// Interpolation method for alignment
    pub interpolation: InterpolationMethod,
    /// Buffer size for temporal windowing
    pub buffer_size: usize,
    /// Alignment tolerance
    pub alignment_tolerance: Duration,
    /// Enable predictive alignment
    pub predictive_alignment: bool,
}

impl Default for TemporalAlignmentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_method: SyncMethod::Software,
            max_time_offset: Duration::from_millis(100),
            interpolation: InterpolationMethod::Linear,
            buffer_size: 10,
            alignment_tolerance: Duration::from_millis(50),
            predictive_alignment: false,
        }
    }
}

impl TemporalAlignmentConfig {
    /// Create strict temporal alignment configuration
    #[must_use]
    pub fn strict() -> Self {
        Self {
            enabled: true,
            sync_method: SyncMethod::Hardware,
            max_time_offset: Duration::from_millis(10),
            interpolation: InterpolationMethod::Cubic,
            buffer_size: 20,
            alignment_tolerance: Duration::from_millis(5),
            predictive_alignment: true,
        }
    }

    /// Create relaxed temporal alignment configuration
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            enabled: true,
            sync_method: SyncMethod::Software,
            max_time_offset: Duration::from_millis(500),
            interpolation: InterpolationMethod::Nearest,
            buffer_size: 5,
            alignment_tolerance: Duration::from_millis(200),
            predictive_alignment: false,
        }
    }

    /// Create precise temporal alignment configuration
    #[must_use]
    pub fn precise() -> Self {
        Self {
            enabled: true,
            sync_method: SyncMethod::GPS,
            max_time_offset: Duration::from_micros(100),
            interpolation: InterpolationMethod::Spline,
            buffer_size: 50,
            alignment_tolerance: Duration::from_micros(50),
            predictive_alignment: true,
        }
    }
}

/// Cross-modal learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalLearningConfig {
    /// Enable cross-modal learning
    pub enabled: bool,
    /// Cross-modal strategy
    pub strategy: CrossModalStrategy,
    /// Contrastive learning configuration
    pub contrastive_learning: ContrastiveLearningConfig,
    /// Knowledge distillation configuration
    pub distillation: DistillationConfig,
    /// Alignment learning configuration
    pub alignment_learning: AlignmentLearningConfig,
    /// Shared representation dimension
    pub shared_representation_dim: usize,
}

impl Default for CrossModalLearningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: CrossModalStrategy::SharedRepresentation,
            contrastive_learning: ContrastiveLearningConfig::default(),
            distillation: DistillationConfig::default(),
            alignment_learning: AlignmentLearningConfig::default(),
            shared_representation_dim: 512,
        }
    }
}

impl CrossModalLearningConfig {
    /// Create contrastive learning configuration
    #[must_use]
    pub fn contrastive() -> Self {
        Self {
            enabled: true,
            strategy: CrossModalStrategy::Contrastive,
            contrastive_learning: ContrastiveLearningConfig::strong(),
            distillation: DistillationConfig::default(),
            alignment_learning: AlignmentLearningConfig::default(),
            shared_representation_dim: 256,
        }
    }

    /// Create shared representation learning configuration
    #[must_use]
    pub fn shared_representation() -> Self {
        Self {
            enabled: true,
            strategy: CrossModalStrategy::SharedRepresentation,
            contrastive_learning: ContrastiveLearningConfig::default(),
            distillation: DistillationConfig::default(),
            alignment_learning: AlignmentLearningConfig::default(),
            shared_representation_dim: 1024,
        }
    }

    /// Create alignment learning configuration
    #[must_use]
    pub fn alignment() -> Self {
        Self {
            enabled: true,
            strategy: CrossModalStrategy::Alignment,
            contrastive_learning: ContrastiveLearningConfig::default(),
            distillation: DistillationConfig::default(),
            alignment_learning: AlignmentLearningConfig::canonical_correlation(),
            shared_representation_dim: 512,
        }
    }
}

/// Contrastive learning configuration for cross-modal learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveLearningConfig {
    /// Enable contrastive learning
    pub enabled: bool,
    /// Temperature parameter for contrastive loss
    pub temperature: f64,
    /// Number of negative samples
    pub negative_samples: usize,
    /// Enable hard negative mining
    pub hard_negative_mining: bool,
    /// Momentum coefficient for momentum contrastive learning
    pub momentum: f64,
    /// Queue size for momentum contrastive learning
    pub queue_size: usize,
    /// Projection head dimension
    pub projection_dim: usize,
}

impl Default for ContrastiveLearningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            temperature: 0.07,
            negative_samples: 64,
            hard_negative_mining: false,
            momentum: 0.999,
            queue_size: 4096,
            projection_dim: 128,
        }
    }
}

impl ContrastiveLearningConfig {
    /// Create strong contrastive learning configuration
    #[must_use]
    pub fn strong() -> Self {
        Self {
            enabled: true,
            temperature: 0.05,
            negative_samples: 128,
            hard_negative_mining: true,
            momentum: 0.9999,
            queue_size: 8192,
            projection_dim: 256,
        }
    }
}

/// Knowledge distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Enable knowledge distillation
    pub enabled: bool,
    /// Teacher model weight in loss function
    pub teacher_weight: f64,
    /// Student model weight in loss function
    pub student_weight: f64,
    /// Temperature for distillation softmax
    pub temperature: f64,
    /// Feature matching weight
    pub feature_matching_weight: f64,
    /// Attention transfer weight
    pub attention_transfer_weight: f64,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            teacher_weight: 0.7,
            student_weight: 0.3,
            temperature: 4.0,
            feature_matching_weight: 0.1,
            attention_transfer_weight: 0.1,
        }
    }
}

/// Alignment learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentLearningConfig {
    /// Enable alignment learning
    pub enabled: bool,
    /// Alignment loss weight
    pub alignment_weight: f64,
    /// Use canonical correlation analysis
    pub use_cca: bool,
    /// CCA regularization parameter
    pub cca_regularization: f64,
    /// Maximum canonical components
    pub max_canonical_components: usize,
    /// Use adversarial alignment
    pub adversarial_alignment: bool,
    /// Adversarial loss weight
    pub adversarial_weight: f64,
}

impl Default for AlignmentLearningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            alignment_weight: 1.0,
            use_cca: false,
            cca_regularization: 1e-5,
            max_canonical_components: 100,
            adversarial_alignment: false,
            adversarial_weight: 0.1,
        }
    }
}

impl AlignmentLearningConfig {
    /// Create canonical correlation analysis configuration
    #[must_use]
    pub fn canonical_correlation() -> Self {
        Self {
            enabled: true,
            alignment_weight: 1.0,
            use_cca: true,
            cca_regularization: 1e-4,
            max_canonical_components: 50,
            adversarial_alignment: false,
            adversarial_weight: 0.0,
        }
    }
}

/// Synchronization requirements for multi-modal data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationRequirements {
    /// Required synchronization accuracy
    pub sync_accuracy: Duration,
    /// Synchronization method
    pub sync_method: SyncMethod,
    /// Enable drift correction
    pub drift_correction: bool,
    /// Maximum allowed drift
    pub max_drift: Duration,
    /// Synchronization check interval
    pub sync_check_interval: Duration,
    /// Fallback synchronization method
    pub fallback_method: Option<SyncMethod>,
}

impl Default for SynchronizationRequirements {
    fn default() -> Self {
        Self {
            sync_accuracy: Duration::from_millis(100),
            sync_method: SyncMethod::Software,
            drift_correction: true,
            max_drift: Duration::from_millis(500),
            sync_check_interval: Duration::from_secs(10),
            fallback_method: Some(SyncMethod::Software),
        }
    }
}

impl SynchronizationRequirements {
    /// Create hardware synchronization requirements
    #[must_use]
    pub fn hardware() -> Self {
        Self {
            sync_accuracy: Duration::from_micros(100),
            sync_method: SyncMethod::Hardware,
            drift_correction: true,
            max_drift: Duration::from_millis(10),
            sync_check_interval: Duration::from_secs(1),
            fallback_method: Some(SyncMethod::Software),
        }
    }

    /// Create software synchronization requirements
    #[must_use]
    pub fn software() -> Self {
        Self {
            sync_accuracy: Duration::from_millis(50),
            sync_method: SyncMethod::Software,
            drift_correction: true,
            max_drift: Duration::from_millis(200),
            sync_check_interval: Duration::from_secs(5),
            fallback_method: None,
        }
    }

    /// Create GPS synchronization requirements
    #[must_use]
    pub fn gps() -> Self {
        Self {
            sync_accuracy: Duration::from_micros(10),
            sync_method: SyncMethod::GPS,
            drift_correction: true,
            max_drift: Duration::from_micros(100),
            sync_check_interval: Duration::from_millis(100),
            fallback_method: Some(SyncMethod::NTP),
        }
    }
}

/// Multi-modal data sample with temporal information
#[derive(Debug, Clone)]
pub struct MultiModalSample {
    /// Timestamp for synchronization
    pub timestamp: SystemTime,
    /// Modality-specific data
    pub modality_data: HashMap<Modality, ModalityData>,
    /// Sample metadata
    pub metadata: HashMap<String, String>,
    /// Synchronization status
    pub sync_status: SyncStatus,
}

impl MultiModalSample {
    /// Create a new multi-modal sample
    #[must_use]
    pub fn new(timestamp: SystemTime) -> Self {
        Self {
            timestamp,
            modality_data: HashMap::new(),
            metadata: HashMap::new(),
            sync_status: SyncStatus::Unknown,
        }
    }

    /// Add data for a specific modality
    pub fn add_modality_data(&mut self, modality: Modality, data: ModalityData) {
        self.modality_data.insert(modality, data);
    }

    /// Get data for a specific modality
    #[must_use]
    pub fn get_modality_data(&self, modality: &Modality) -> Option<&ModalityData> {
        self.modality_data.get(modality)
    }

    /// Check if sample contains all required modalities
    #[must_use]
    pub fn has_modalities(&self, required_modalities: &[Modality]) -> bool {
        required_modalities
            .iter()
            .all(|m| self.modality_data.contains_key(m))
    }

    /// Get age of the sample
    #[must_use]
    pub fn age(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.timestamp)
            .unwrap_or(Duration::from_secs(0))
    }
}

/// Modality-specific data container
#[derive(Debug, Clone)]
pub struct ModalityData {
    /// Raw data
    pub data: Vec<u8>,
    /// Data format/type
    pub format: String,
    /// Processing metadata
    pub metadata: HashMap<String, TransformParameter>,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
}

impl ModalityData {
    /// Create new modality data
    #[must_use]
    pub fn new(data: Vec<u8>, format: String) -> Self {
        Self {
            data,
            format,
            metadata: HashMap::new(),
            quality_metrics: HashMap::new(),
        }
    }

    /// Get data size in bytes
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Add quality metric
    pub fn add_quality_metric(&mut self, name: String, value: f64) {
        self.quality_metrics.insert(name, value);
    }
}

/// Synchronization status for multi-modal samples
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStatus {
    /// Synchronization status unknown
    Unknown,
    /// Sample is synchronized
    Synchronized,
    /// Sample has timing drift
    Drift,
    /// Sample is out of sync
    OutOfSync,
    /// Synchronization failed
    Failed,
}

/// Multi-modal processing errors
#[derive(Debug, Clone, PartialEq)]
pub enum MultiModalError {
    /// No modalities configured
    NoModalities,
    /// Invalid modality weights
    InvalidWeights { total: f64 },
    /// Missing required modality
    MissingModality(Modality),
    /// Synchronization failure
    SyncFailure(String),
    /// Temporal alignment failure
    AlignmentFailure(String),
    /// Cross-modal learning error
    CrossModalError(String),
    /// Configuration error
    ConfigurationError(String),
}

impl std::fmt::Display for MultiModalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoModalities => write!(f, "No modalities configured for multi-modal processing"),
            Self::InvalidWeights { total } => {
                write!(
                    f,
                    "Invalid modality weights: total weight is {total}, should be ~1.0"
                )
            }
            Self::MissingModality(modality) => {
                write!(f, "Missing required modality: {modality:?}")
            }
            Self::SyncFailure(msg) => write!(f, "Synchronization failure: {msg}"),
            Self::AlignmentFailure(msg) => write!(f, "Temporal alignment failure: {msg}"),
            Self::CrossModalError(msg) => write!(f, "Cross-modal learning error: {msg}"),
            Self::ConfigurationError(msg) => write!(f, "Configuration error: {msg}"),
        }
    }
}

impl std::error::Error for MultiModalError {}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_config_presets() {
        let vision_audio = MultiModalConfig::vision_audio();
        assert!(vision_audio.enabled);
        assert_eq!(vision_audio.modalities.len(), 2);
        assert!(vision_audio.modalities.contains(&Modality::Visual));
        assert!(vision_audio.modalities.contains(&Modality::Audio));

        let vision_depth = MultiModalConfig::vision_depth();
        assert_eq!(vision_depth.fusion_strategy, FusionStrategy::HybridFusion);

        let multi_sensor = MultiModalConfig::multi_sensor();
        assert_eq!(multi_sensor.modalities.len(), 3);
        assert_eq!(
            multi_sensor.fusion_strategy,
            FusionStrategy::AttentionFusion
        );
    }

    #[test]
    fn test_temporal_alignment_config() {
        let strict = TemporalAlignmentConfig::strict();
        assert_eq!(strict.sync_method, SyncMethod::Hardware);
        assert!(strict.predictive_alignment);

        let relaxed = TemporalAlignmentConfig::relaxed();
        assert_eq!(relaxed.sync_method, SyncMethod::Software);
        assert!(!relaxed.predictive_alignment);

        let precise = TemporalAlignmentConfig::precise();
        assert_eq!(precise.sync_method, SyncMethod::GPS);
        assert!(precise.predictive_alignment);
    }

    #[test]
    fn test_cross_modal_learning_config() {
        let contrastive = CrossModalLearningConfig::contrastive();
        assert_eq!(contrastive.strategy, CrossModalStrategy::Contrastive);
        assert!(contrastive.contrastive_learning.enabled);

        let shared_rep = CrossModalLearningConfig::shared_representation();
        assert_eq!(
            shared_rep.strategy,
            CrossModalStrategy::SharedRepresentation
        );
        assert_eq!(shared_rep.shared_representation_dim, 1024);

        let alignment = CrossModalLearningConfig::alignment();
        assert_eq!(alignment.strategy, CrossModalStrategy::Alignment);
        assert!(alignment.alignment_learning.use_cca);
    }

    #[test]
    fn test_synchronization_requirements() {
        let hardware = SynchronizationRequirements::hardware();
        assert_eq!(hardware.sync_method, SyncMethod::Hardware);
        assert_eq!(hardware.sync_accuracy, Duration::from_micros(100));

        let software = SynchronizationRequirements::software();
        assert_eq!(software.sync_method, SyncMethod::Software);
        assert_eq!(software.sync_accuracy, Duration::from_millis(50));

        let gps = SynchronizationRequirements::gps();
        assert_eq!(gps.sync_method, SyncMethod::GPS);
        assert_eq!(gps.sync_accuracy, Duration::from_micros(10));
    }

    #[test]
    fn test_multimodal_sample() {
        let mut sample = MultiModalSample::new(SystemTime::now());

        let visual_data = ModalityData::new(vec![1, 2, 3, 4], "jpeg".to_string());
        sample.add_modality_data(Modality::Visual, visual_data);

        let audio_data = ModalityData::new(vec![5, 6, 7, 8], "wav".to_string());
        sample.add_modality_data(Modality::Audio, audio_data);

        assert!(sample.has_modalities(&[Modality::Visual, Modality::Audio]));
        assert!(!sample.has_modalities(&[Modality::Visual, Modality::Audio, Modality::Depth]));

        assert!(sample.get_modality_data(&Modality::Visual).is_some());
        assert!(sample.get_modality_data(&Modality::Depth).is_none());
    }

    #[test]
    fn test_modality_weights() {
        let mut config = MultiModalConfig::default();
        config.set_modality_weight(Modality::Visual, 0.6);
        config.set_modality_weight(Modality::Audio, 0.4);

        assert_eq!(config.get_modality_weight(&Modality::Visual), 0.6);
        assert_eq!(config.get_modality_weight(&Modality::Audio), 0.4);
        assert_eq!(config.get_modality_weight(&Modality::Depth), 1.0); // Default
    }

    #[test]
    fn test_multimodal_error_display() {
        let error = MultiModalError::InvalidWeights { total: 1.5 };
        let error_str = error.to_string();
        assert!(error_str.contains("Invalid modality weights"));
        assert!(error_str.contains("1.5"));

        let error = MultiModalError::MissingModality(Modality::Audio);
        let error_str = error.to_string();
        assert!(error_str.contains("Missing required modality"));
        assert!(error_str.contains("Audio"));
    }
}

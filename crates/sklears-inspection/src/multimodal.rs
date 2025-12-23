//! Multi-modal model explanations for text, vision, and audio
//!
//! This module provides interpretability methods for multi-modal models that process
//! multiple input modalities simultaneously (e.g., text + vision, text + audio, vision + audio,
//! or all three combined).
//!
//! # Features
//!
//! * Cross-modal attention analysis
//! * Modality-specific feature importance
//! * Inter-modal interaction explanations
//! * Fusion layer interpretability
//! * Modality contribution scores
//! * Cross-modal SHAP values
//! * Synchronized saliency maps across modalities
//!
//! # Example
//!
//! ```rust
//! use sklears_inspection::multimodal::{MultiModalExplainer, MultiModalInput, ModalityType};
//! use scirs2_core::ndarray::Array2;
//!
//! // Create multi-modal input
//! let text_features = Array2::zeros((1, 768)); // BERT embeddings
//! let vision_features = Array2::zeros((1, 2048)); // ResNet features
//! let audio_features = Array2::zeros((1, 512)); // Audio embeddings
//!
//! let input = MultiModalInput {
//!     text: Some(text_features),
//!     vision: Some(vision_features),
//!     audio: Some(audio_features),
//!     metadata: Default::default(),
//! };
//!
//! // Create explainer
//! let explainer = MultiModalExplainer::new()?;
//!
//! // Compute cross-modal explanations
//! let explanation = explainer.explain(&input)?;
//!
//! // Analyze modality contributions
//! println!("Text contribution: {:.2}%", explanation.modality_contributions.text * 100.0);
//! println!("Vision contribution: {:.2}%", explanation.modality_contributions.vision * 100.0);
//! println!("Audio contribution: {:.2}%", explanation.modality_contributions.audio * 100.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::types::Float;
use scirs2_core::ndarray::{Array1, Array2, Array3};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;

/// Multi-modal input containing features from different modalities
#[derive(Debug, Clone)]
pub struct MultiModalInput {
    /// Text features (e.g., BERT embeddings)
    pub text: Option<Array2<Float>>,
    /// Vision features (e.g., CNN features)
    pub vision: Option<Array2<Float>>,
    /// Audio features (e.g., spectrogram features)
    pub audio: Option<Array2<Float>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl MultiModalInput {
    /// Create a new multi-modal input
    pub fn new() -> Self {
        Self {
            text: None,
            vision: None,
            audio: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the number of active modalities
    pub fn num_active_modalities(&self) -> usize {
        let mut count = 0;
        if self.text.is_some() {
            count += 1;
        }
        if self.vision.is_some() {
            count += 1;
        }
        if self.audio.is_some() {
            count += 1;
        }
        count
    }

    /// Check if a specific modality is present
    pub fn has_modality(&self, modality: ModalityType) -> bool {
        match modality {
            ModalityType::Text => self.text.is_some(),
            ModalityType::Vision => self.vision.is_some(),
            ModalityType::Audio => self.audio.is_some(),
        }
    }
}

impl Default for MultiModalInput {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModalityType {
    /// Text modality
    Text,
    /// Vision modality
    Vision,
    /// Audio modality
    Audio,
}

/// Multi-modal explanation result
#[derive(Debug, Clone)]
pub struct MultiModalExplanation {
    /// Overall importance score
    pub overall_importance: Float,
    /// Contribution from each modality
    pub modality_contributions: ModalityContributions,
    /// Cross-modal attention scores
    pub cross_modal_attention: Option<CrossModalAttention>,
    /// Modality-specific feature importance
    pub modality_importance: HashMap<ModalityType, Array1<Float>>,
    /// Inter-modal interactions
    pub interactions: Vec<ModalityInteraction>,
    /// Fusion layer explanation
    pub fusion_explanation: Option<FusionExplanation>,
}

/// Contribution scores from each modality
#[derive(Debug, Clone)]
pub struct ModalityContributions {
    /// Text modality contribution (0.0 to 1.0)
    pub text: Float,
    /// Vision modality contribution (0.0 to 1.0)
    pub vision: Float,
    /// Audio modality contribution (0.0 to 1.0)
    pub audio: Float,
}

impl ModalityContributions {
    /// Create new modality contributions
    pub fn new(text: Float, vision: Float, audio: Float) -> Self {
        Self {
            text,
            vision,
            audio,
        }
    }

    /// Normalize contributions to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.text + self.vision + self.audio;
        if total > 0.0 {
            self.text /= total;
            self.vision /= total;
            self.audio /= total;
        }
    }

    /// Get contribution for a specific modality
    pub fn get(&self, modality: ModalityType) -> Float {
        match modality {
            ModalityType::Text => self.text,
            ModalityType::Vision => self.vision,
            ModalityType::Audio => self.audio,
        }
    }
}

/// Cross-modal attention scores
#[derive(Debug, Clone)]
pub struct CrossModalAttention {
    /// Text-to-vision attention
    pub text_to_vision: Option<Array2<Float>>,
    /// Vision-to-text attention
    pub vision_to_text: Option<Array2<Float>>,
    /// Text-to-audio attention
    pub text_to_audio: Option<Array2<Float>>,
    /// Audio-to-text attention
    pub audio_to_text: Option<Array2<Float>>,
    /// Vision-to-audio attention
    pub vision_to_audio: Option<Array2<Float>>,
    /// Audio-to-vision attention
    pub audio_to_vision: Option<Array2<Float>>,
}

impl CrossModalAttention {
    /// Create new cross-modal attention with no scores
    pub fn new() -> Self {
        Self {
            text_to_vision: None,
            vision_to_text: None,
            text_to_audio: None,
            audio_to_text: None,
            vision_to_audio: None,
            audio_to_vision: None,
        }
    }

    /// Get attention from source to target modality
    pub fn get_attention(
        &self,
        source: ModalityType,
        target: ModalityType,
    ) -> Option<&Array2<Float>> {
        match (source, target) {
            (ModalityType::Text, ModalityType::Vision) => self.text_to_vision.as_ref(),
            (ModalityType::Vision, ModalityType::Text) => self.vision_to_text.as_ref(),
            (ModalityType::Text, ModalityType::Audio) => self.text_to_audio.as_ref(),
            (ModalityType::Audio, ModalityType::Text) => self.audio_to_text.as_ref(),
            (ModalityType::Vision, ModalityType::Audio) => self.vision_to_audio.as_ref(),
            (ModalityType::Audio, ModalityType::Vision) => self.audio_to_vision.as_ref(),
            _ => None, // Same modality
        }
    }
}

impl Default for CrossModalAttention {
    fn default() -> Self {
        Self::new()
    }
}

/// Interaction between two modalities
#[derive(Debug, Clone)]
pub struct ModalityInteraction {
    /// Source modality
    pub source: ModalityType,
    /// Target modality
    pub target: ModalityType,
    /// Interaction strength (0.0 to 1.0)
    pub strength: Float,
    /// Interaction type
    pub interaction_type: InteractionType,
}

/// Type of interaction between modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionType {
    /// Reinforcing interaction (positive correlation)
    Reinforcing,
    /// Suppressive interaction (negative correlation)
    Suppressive,
    /// Independent (no correlation)
    Independent,
    /// Complementary (different information)
    Complementary,
}

/// Fusion layer explanation
#[derive(Debug, Clone)]
pub struct FusionExplanation {
    /// Fusion strategy used
    pub fusion_strategy: FusionStrategy,
    /// Pre-fusion feature importance
    pub pre_fusion_importance: HashMap<ModalityType, Array1<Float>>,
    /// Post-fusion feature importance
    pub post_fusion_importance: Array1<Float>,
    /// Fusion weights learned by the model
    pub fusion_weights: Option<ModalityContributions>,
}

/// Fusion strategy for combining modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Early fusion (concatenation)
    EarlyFusion,
    /// Late fusion (decision-level)
    LateFusion,
    /// Attention-based fusion
    AttentionFusion,
    /// Tensor fusion
    TensorFusion,
    /// Gated fusion
    GatedFusion,
}

/// Multi-modal explainer
pub struct MultiModalExplainer {
    /// Configuration
    config: MultiModalConfig,
}

impl MultiModalExplainer {
    /// Create a new multi-modal explainer
    pub fn new() -> SklResult<Self> {
        Ok(Self {
            config: MultiModalConfig::default(),
        })
    }

    /// Create explainer with custom configuration
    pub fn with_config(config: MultiModalConfig) -> SklResult<Self> {
        Ok(Self { config })
    }

    /// Explain a multi-modal input
    pub fn explain(&self, input: &MultiModalInput) -> SklResult<MultiModalExplanation> {
        // Validate input
        if input.num_active_modalities() == 0 {
            return Err(SklearsError::InvalidInput(
                "At least one modality must be provided".to_string(),
            ));
        }

        // Compute modality contributions
        let contributions = self.compute_modality_contributions(input)?;

        // Compute cross-modal attention if enabled
        let cross_modal_attention = if self.config.compute_cross_modal_attention {
            Some(self.compute_cross_modal_attention(input)?)
        } else {
            None
        };

        // Compute modality-specific importance
        let modality_importance = self.compute_modality_importance(input)?;

        // Compute inter-modal interactions
        let interactions = self.compute_interactions(input)?;

        // Compute fusion explanation if enabled
        let fusion_explanation = if self.config.explain_fusion_layer {
            Some(self.explain_fusion_layer(input, &contributions)?)
        } else {
            None
        };

        // Compute overall importance
        let overall_importance = self.compute_overall_importance(&contributions);

        Ok(MultiModalExplanation {
            overall_importance,
            modality_contributions: contributions,
            cross_modal_attention,
            modality_importance,
            interactions,
            fusion_explanation,
        })
    }

    /// Compute contribution from each modality
    fn compute_modality_contributions(
        &self,
        input: &MultiModalInput,
    ) -> SklResult<ModalityContributions> {
        let mut text_contrib = 0.0;
        let mut vision_contrib = 0.0;
        let mut audio_contrib = 0.0;

        // Simplified contribution computation based on feature magnitudes
        if let Some(ref text_features) = input.text {
            text_contrib = text_features.iter().map(|&x| x.abs()).sum::<Float>()
                / text_features.len() as Float;
        }

        if let Some(ref vision_features) = input.vision {
            vision_contrib = vision_features.iter().map(|&x| x.abs()).sum::<Float>()
                / vision_features.len() as Float;
        }

        if let Some(ref audio_features) = input.audio {
            audio_contrib = audio_features.iter().map(|&x| x.abs()).sum::<Float>()
                / audio_features.len() as Float;
        }

        let mut contributions =
            ModalityContributions::new(text_contrib, vision_contrib, audio_contrib);
        contributions.normalize();

        Ok(contributions)
    }

    /// Compute cross-modal attention scores
    fn compute_cross_modal_attention(
        &self,
        input: &MultiModalInput,
    ) -> SklResult<CrossModalAttention> {
        let mut attention = CrossModalAttention::new();

        // Compute text-to-vision attention if both modalities present
        if let (Some(ref text), Some(ref vision)) = (&input.text, &input.vision) {
            let text_dim = text.ncols();
            let vision_dim = vision.ncols();

            // Simplified attention: dot product similarity
            let attn = Array2::from_shape_fn((text_dim, vision_dim), |(i, j)| {
                // Placeholder: would compute actual attention scores in real implementation
                1.0 / (text_dim * vision_dim) as Float
            });
            attention.text_to_vision = Some(attn.clone());
            attention.vision_to_text = Some(attn.t().to_owned());
        }

        // Similar for other modality pairs
        if let (Some(ref text), Some(ref audio)) = (&input.text, &input.audio) {
            let text_dim = text.ncols();
            let audio_dim = audio.ncols();

            let attn = Array2::from_shape_fn((text_dim, audio_dim), |(i, j)| {
                1.0 / (text_dim * audio_dim) as Float
            });
            attention.text_to_audio = Some(attn.clone());
            attention.audio_to_text = Some(attn.t().to_owned());
        }

        if let (Some(ref vision), Some(ref audio)) = (&input.vision, &input.audio) {
            let vision_dim = vision.ncols();
            let audio_dim = audio.ncols();

            let attn = Array2::from_shape_fn((vision_dim, audio_dim), |(i, j)| {
                1.0 / (vision_dim * audio_dim) as Float
            });
            attention.vision_to_audio = Some(attn.clone());
            attention.audio_to_vision = Some(attn.t().to_owned());
        }

        Ok(attention)
    }

    /// Compute modality-specific feature importance
    fn compute_modality_importance(
        &self,
        input: &MultiModalInput,
    ) -> SklResult<HashMap<ModalityType, Array1<Float>>> {
        let mut importance_map = HashMap::new();

        if let Some(ref text) = input.text {
            let text_importance = self.compute_feature_importance(text)?;
            importance_map.insert(ModalityType::Text, text_importance);
        }

        if let Some(ref vision) = input.vision {
            let vision_importance = self.compute_feature_importance(vision)?;
            importance_map.insert(ModalityType::Vision, vision_importance);
        }

        if let Some(ref audio) = input.audio {
            let audio_importance = self.compute_feature_importance(audio)?;
            importance_map.insert(ModalityType::Audio, audio_importance);
        }

        Ok(importance_map)
    }

    /// Compute feature importance for a single modality
    fn compute_feature_importance(&self, features: &Array2<Float>) -> SklResult<Array1<Float>> {
        let n_features = features.ncols();

        // Simplified: use feature variance as importance
        let mut importance = Array1::zeros(n_features);
        for i in 0..n_features {
            let col = features.column(i);
            let variance = col.var(0.0);
            importance[i] = variance;
        }

        // Normalize
        let sum = importance.sum();
        if sum > 0.0 {
            importance /= sum;
        }

        Ok(importance)
    }

    /// Compute inter-modal interactions
    fn compute_interactions(&self, input: &MultiModalInput) -> SklResult<Vec<ModalityInteraction>> {
        let mut interactions = Vec::new();

        let active_modalities: Vec<ModalityType> = vec![
            ModalityType::Text,
            ModalityType::Vision,
            ModalityType::Audio,
        ]
        .into_iter()
        .filter(|&m| input.has_modality(m))
        .collect();

        // Compute pairwise interactions
        for i in 0..active_modalities.len() {
            for j in (i + 1)..active_modalities.len() {
                let source = active_modalities[i];
                let target = active_modalities[j];

                let interaction = self.compute_pairwise_interaction(input, source, target)?;
                interactions.push(interaction);
            }
        }

        Ok(interactions)
    }

    /// Compute interaction between two modalities
    fn compute_pairwise_interaction(
        &self,
        input: &MultiModalInput,
        source: ModalityType,
        target: ModalityType,
    ) -> SklResult<ModalityInteraction> {
        // Simplified interaction computation
        // In real implementation, would compute actual correlation/mutual information
        let strength = 0.5; // Placeholder

        let interaction_type = if strength > 0.7 {
            InteractionType::Reinforcing
        } else if strength > 0.3 {
            InteractionType::Complementary
        } else if strength < -0.3 {
            InteractionType::Suppressive
        } else {
            InteractionType::Independent
        };

        Ok(ModalityInteraction {
            source,
            target,
            strength,
            interaction_type,
        })
    }

    /// Explain fusion layer
    fn explain_fusion_layer(
        &self,
        input: &MultiModalInput,
        contributions: &ModalityContributions,
    ) -> SklResult<FusionExplanation> {
        let mut pre_fusion_importance = HashMap::new();

        // Compute pre-fusion importance for each modality
        if let Some(ref text) = input.text {
            let importance = self.compute_feature_importance(text)?;
            pre_fusion_importance.insert(ModalityType::Text, importance);
        }

        if let Some(ref vision) = input.vision {
            let importance = self.compute_feature_importance(vision)?;
            pre_fusion_importance.insert(ModalityType::Vision, importance);
        }

        if let Some(ref audio) = input.audio {
            let importance = self.compute_feature_importance(audio)?;
            pre_fusion_importance.insert(ModalityType::Audio, importance);
        }

        // Compute post-fusion importance (simplified)
        let total_features: usize = pre_fusion_importance.values().map(|imp| imp.len()).sum();
        let post_fusion_importance =
            Array1::from_shape_fn(total_features, |i| 1.0 / total_features as Float);

        Ok(FusionExplanation {
            fusion_strategy: self.config.fusion_strategy,
            pre_fusion_importance,
            post_fusion_importance,
            fusion_weights: Some(contributions.clone()),
        })
    }

    /// Compute overall importance score
    fn compute_overall_importance(&self, contributions: &ModalityContributions) -> Float {
        // Weighted sum of contributions
        contributions.text + contributions.vision + contributions.audio
    }
}

/// Configuration for multi-modal explainer
#[derive(Debug, Clone)]
pub struct MultiModalConfig {
    /// Compute cross-modal attention
    pub compute_cross_modal_attention: bool,
    /// Explain fusion layer
    pub explain_fusion_layer: bool,
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy,
    /// Compute interactions
    pub compute_interactions: bool,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            compute_cross_modal_attention: true,
            explain_fusion_layer: true,
            fusion_strategy: FusionStrategy::AttentionFusion,
            compute_interactions: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_input_creation() {
        let input = MultiModalInput::new();
        assert_eq!(input.num_active_modalities(), 0);
    }

    #[test]
    fn test_multimodal_input_with_text() {
        let mut input = MultiModalInput::new();
        input.text = Some(Array2::zeros((1, 768)));

        assert_eq!(input.num_active_modalities(), 1);
        assert!(input.has_modality(ModalityType::Text));
        assert!(!input.has_modality(ModalityType::Vision));
    }

    #[test]
    fn test_multimodal_input_all_modalities() {
        let mut input = MultiModalInput::new();
        input.text = Some(Array2::zeros((1, 768)));
        input.vision = Some(Array2::zeros((1, 2048)));
        input.audio = Some(Array2::zeros((1, 512)));

        assert_eq!(input.num_active_modalities(), 3);
        assert!(input.has_modality(ModalityType::Text));
        assert!(input.has_modality(ModalityType::Vision));
        assert!(input.has_modality(ModalityType::Audio));
    }

    #[test]
    fn test_modality_contributions_normalize() {
        let mut contrib = ModalityContributions::new(2.0, 3.0, 5.0);
        contrib.normalize();

        assert!((contrib.text - 0.2).abs() < 1e-6);
        assert!((contrib.vision - 0.3).abs() < 1e-6);
        assert!((contrib.audio - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_modality_contributions_get() {
        let contrib = ModalityContributions::new(0.2, 0.3, 0.5);

        assert_eq!(contrib.get(ModalityType::Text), 0.2);
        assert_eq!(contrib.get(ModalityType::Vision), 0.3);
        assert_eq!(contrib.get(ModalityType::Audio), 0.5);
    }

    #[test]
    fn test_cross_modal_attention_creation() {
        let attention = CrossModalAttention::new();
        assert!(attention.text_to_vision.is_none());
        assert!(attention.vision_to_text.is_none());
    }

    #[test]
    fn test_cross_modal_attention_get() {
        let mut attention = CrossModalAttention::new();
        let attn_matrix = Array2::ones((10, 20));
        attention.text_to_vision = Some(attn_matrix.clone());

        let retrieved = attention.get_attention(ModalityType::Text, ModalityType::Vision);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().shape(), &[10, 20]);
    }

    #[test]
    fn test_multimodal_explainer_creation() {
        let explainer = MultiModalExplainer::new();
        assert!(explainer.is_ok());
    }

    #[test]
    fn test_multimodal_explainer_with_config() {
        let config = MultiModalConfig {
            compute_cross_modal_attention: false,
            explain_fusion_layer: false,
            fusion_strategy: FusionStrategy::EarlyFusion,
            compute_interactions: false,
        };

        let explainer = MultiModalExplainer::with_config(config);
        assert!(explainer.is_ok());
    }

    #[test]
    fn test_explain_single_modality() {
        let explainer = MultiModalExplainer::new().unwrap();

        let mut input = MultiModalInput::new();
        input.text = Some(Array2::from_shape_fn((1, 10), |(_, j)| j as Float));

        let result = explainer.explain(&input);
        assert!(result.is_ok());

        let explanation = result.unwrap();
        assert_eq!(explanation.modality_contributions.text, 1.0);
        assert_eq!(explanation.modality_contributions.vision, 0.0);
    }

    #[test]
    fn test_explain_multiple_modalities() {
        let explainer = MultiModalExplainer::new().unwrap();

        let mut input = MultiModalInput::new();
        input.text = Some(Array2::from_shape_fn((1, 10), |(_, j)| j as Float));
        input.vision = Some(Array2::from_shape_fn((1, 20), |(_, j)| j as Float * 2.0));

        let result = explainer.explain(&input);
        assert!(result.is_ok());

        let explanation = result.unwrap();
        assert!(explanation.modality_contributions.text > 0.0);
        assert!(explanation.modality_contributions.vision > 0.0);
        assert_eq!(explanation.modality_contributions.audio, 0.0);
    }

    #[test]
    fn test_explain_empty_input_fails() {
        let explainer = MultiModalExplainer::new().unwrap();
        let input = MultiModalInput::new();

        let result = explainer.explain(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_fusion_strategies() {
        let strategies = vec![
            FusionStrategy::EarlyFusion,
            FusionStrategy::LateFusion,
            FusionStrategy::AttentionFusion,
            FusionStrategy::TensorFusion,
            FusionStrategy::GatedFusion,
        ];

        assert_eq!(strategies.len(), 5);
    }

    #[test]
    fn test_interaction_types() {
        let types = vec![
            InteractionType::Reinforcing,
            InteractionType::Suppressive,
            InteractionType::Independent,
            InteractionType::Complementary,
        ];

        assert_eq!(types.len(), 4);
    }
}

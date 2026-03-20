//! Trait implementations for text feature extractors
//!
//! This module provides implementations of the feature extraction traits
//! for text-based extractors.

use crate::feature_traits::{
    ConfigurableExtractor, FeatureExtractor, FeatureMetadata, TextFeatureExtractor,
};
use crate::text::advanced_sentiment::{AdvancedSentimentAnalyzer, Language};
use crate::text::{AspectBasedSentimentAnalyzer, EmotionDetector, SentimentAnalyzer};
use scirs2_core::ndarray::Array2;
use sklears_core::{error::Result as SklResult, types::Float};

// ============================================================================
// EmotionDetector Trait Implementations
// ============================================================================

impl FeatureExtractor for EmotionDetector {
    type Input = String;
    type Output = Array2<Float>;

    fn extract_features(&self, input: &[Self::Input]) -> SklResult<Self::Output> {
        self.extract_features(input)
    }

    fn feature_names(&self) -> Option<Vec<String>> {
        Some(vec![
            "joy_score".to_string(),
            "sadness_score".to_string(),
            "anger_score".to_string(),
            "fear_score".to_string(),
            "surprise_score".to_string(),
            "disgust_score".to_string(),
            "joy_onehot".to_string(),
            "sadness_onehot".to_string(),
            "anger_onehot".to_string(),
            "fear_onehot".to_string(),
            "surprise_onehot".to_string(),
            "disgust_onehot".to_string(),
            "confidence".to_string(),
            "intensity".to_string(),
        ])
    }

    fn n_features(&self) -> Option<usize> {
        Some(14)
    }

    fn validate_input(&self, input: &[Self::Input]) -> SklResult<()> {
        if input.is_empty() {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(
                "Empty input document collection".to_string(),
            ));
        }
        Ok(())
    }
}

impl TextFeatureExtractor for EmotionDetector {
    fn vocabulary(&self) -> Option<Vec<String>> {
        // Collect all emotion words from all lexicons
        let mut vocab: Vec<String> = self
            .emotion_lexicons
            .values()
            .flat_map(|lexicon| lexicon.iter().cloned())
            .collect();
        vocab.sort();
        vocab.dedup();
        Some(vocab)
    }

    fn vocabulary_size(&self) -> Option<usize> {
        self.vocabulary().map(|v| v.len())
    }
}

/// Configuration for EmotionDetector
#[derive(Debug, Clone)]
pub struct EmotionDetectorConfig {
    pub case_sensitive: bool,
    pub min_confidence: f64,
    pub intensity_weight: f64,
}

impl Default for EmotionDetectorConfig {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            min_confidence: 0.1,
            intensity_weight: 0.5,
        }
    }
}

impl ConfigurableExtractor for EmotionDetector {
    type Config = EmotionDetectorConfig;

    fn config(&self) -> Self::Config {
        EmotionDetectorConfig {
            case_sensitive: self.case_sensitive,
            min_confidence: self.min_confidence,
            intensity_weight: self.intensity_weight,
        }
    }

    fn with_config(&self, config: Self::Config) -> Self {
        let mut detector = self.clone();
        detector.case_sensitive = config.case_sensitive;
        detector.min_confidence = config.min_confidence;
        detector.intensity_weight = config.intensity_weight;
        detector
    }
}

impl FeatureMetadata for EmotionDetector {
    fn feature_types(&self) -> Option<Vec<String>> {
        Some(vec![
            "continuous".to_string(), // joy_score
            "continuous".to_string(), // sadness_score
            "continuous".to_string(), // anger_score
            "continuous".to_string(), // fear_score
            "continuous".to_string(), // surprise_score
            "continuous".to_string(), // disgust_score
            "binary".to_string(),     // joy_onehot
            "binary".to_string(),     // sadness_onehot
            "binary".to_string(),     // anger_onehot
            "binary".to_string(),     // fear_onehot
            "binary".to_string(),     // surprise_onehot
            "binary".to_string(),     // disgust_onehot
            "continuous".to_string(), // confidence
            "continuous".to_string(), // intensity
        ])
    }

    fn feature_descriptions(&self) -> Option<Vec<String>> {
        Some(vec![
            "Joy emotion score".to_string(),
            "Sadness emotion score".to_string(),
            "Anger emotion score".to_string(),
            "Fear emotion score".to_string(),
            "Surprise emotion score".to_string(),
            "Disgust emotion score".to_string(),
            "Joy primary emotion indicator".to_string(),
            "Sadness primary emotion indicator".to_string(),
            "Anger primary emotion indicator".to_string(),
            "Fear primary emotion indicator".to_string(),
            "Surprise primary emotion indicator".to_string(),
            "Disgust primary emotion indicator".to_string(),
            "Confidence in primary emotion".to_string(),
            "Overall emotion intensity".to_string(),
        ])
    }
}

// ============================================================================
// AspectBasedSentimentAnalyzer Trait Implementations
// ============================================================================

impl FeatureExtractor for AspectBasedSentimentAnalyzer {
    type Input = String;
    type Output = Array2<Float>;

    fn extract_features(&self, input: &[Self::Input]) -> SklResult<Self::Output> {
        self.extract_features(input)
    }

    fn feature_names(&self) -> Option<Vec<String>> {
        Some(vec![
            "avg_aspect_score".to_string(),
            "positive_aspects".to_string(),
            "negative_aspects".to_string(),
            "neutral_aspects".to_string(),
            "aspect_diversity".to_string(),
            "avg_confidence".to_string(),
        ])
    }

    fn n_features(&self) -> Option<usize> {
        Some(6)
    }

    fn validate_input(&self, input: &[Self::Input]) -> SklResult<()> {
        if input.is_empty() {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(
                "Empty input document collection".to_string(),
            ));
        }
        Ok(())
    }
}

impl TextFeatureExtractor for AspectBasedSentimentAnalyzer {
    fn vocabulary(&self) -> Option<Vec<String>> {
        if !self.aspects.is_empty() {
            Some(self.aspects.iter().cloned().collect())
        } else {
            None
        }
    }

    fn vocabulary_size(&self) -> Option<usize> {
        if !self.aspects.is_empty() {
            Some(self.aspects.len())
        } else {
            None
        }
    }
}

/// Configuration for AspectBasedSentimentAnalyzer
#[derive(Debug, Clone)]
pub struct AspectSentimentConfig {
    pub context_window: usize,
    pub min_confidence: f64,
    pub case_sensitive: bool,
    pub auto_extract_aspects: bool,
}

impl Default for AspectSentimentConfig {
    fn default() -> Self {
        Self {
            context_window: 3,
            min_confidence: 0.1,
            case_sensitive: false,
            auto_extract_aspects: true,
        }
    }
}

impl ConfigurableExtractor for AspectBasedSentimentAnalyzer {
    type Config = AspectSentimentConfig;

    fn config(&self) -> Self::Config {
        AspectSentimentConfig {
            context_window: self.context_window,
            min_confidence: self.min_confidence,
            case_sensitive: self.case_sensitive,
            auto_extract_aspects: self.auto_extract_aspects,
        }
    }

    fn with_config(&self, config: Self::Config) -> Self {
        let mut analyzer = self.clone();
        analyzer.context_window = config.context_window;
        analyzer.min_confidence = config.min_confidence;
        analyzer.case_sensitive = config.case_sensitive;
        analyzer.auto_extract_aspects = config.auto_extract_aspects;
        analyzer
    }
}

impl FeatureMetadata for AspectBasedSentimentAnalyzer {
    fn feature_types(&self) -> Option<Vec<String>> {
        Some(vec![
            "continuous".to_string(), // avg_aspect_score
            "count".to_string(),      // positive_aspects
            "count".to_string(),      // negative_aspects
            "count".to_string(),      // neutral_aspects
            "continuous".to_string(), // aspect_diversity
            "continuous".to_string(), // avg_confidence
        ])
    }

    fn feature_descriptions(&self) -> Option<Vec<String>> {
        Some(vec![
            "Average sentiment score across all detected aspects".to_string(),
            "Number of aspects with positive sentiment".to_string(),
            "Number of aspects with negative sentiment".to_string(),
            "Number of aspects with neutral sentiment".to_string(),
            "Diversity of detected aspects (unique/total)".to_string(),
            "Average confidence across all aspect-sentiment pairs".to_string(),
        ])
    }
}

// ============================================================================
// SentimentAnalyzer Trait Implementations
// ============================================================================

impl FeatureExtractor for SentimentAnalyzer {
    type Input = String;
    type Output = Array2<Float>;

    fn extract_features(&self, input: &[Self::Input]) -> SklResult<Self::Output> {
        self.extract_features(input)
    }

    fn feature_names(&self) -> Option<Vec<String>> {
        Some(vec![
            "sentiment_score".to_string(),
            "positive_ratio".to_string(),
            "negative_ratio".to_string(),
            "sentiment_density".to_string(),
            "polarity_encoded".to_string(),
        ])
    }

    fn n_features(&self) -> Option<usize> {
        Some(5)
    }

    fn validate_input(&self, input: &[Self::Input]) -> SklResult<()> {
        if input.is_empty() {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(
                "Empty input document collection".to_string(),
            ));
        }
        Ok(())
    }
}

impl TextFeatureExtractor for SentimentAnalyzer {
    fn vocabulary(&self) -> Option<Vec<String>> {
        let mut vocab: Vec<String> = self
            .positive_words
            .iter()
            .chain(self.negative_words.iter())
            .cloned()
            .collect();
        vocab.sort();
        Some(vocab)
    }

    fn vocabulary_size(&self) -> Option<usize> {
        Some(self.positive_words.len() + self.negative_words.len())
    }
}

/// Configuration for SentimentAnalyzer
#[derive(Debug, Clone)]
pub struct SentimentAnalyzerConfig {
    pub neutral_threshold: f64,
    pub case_sensitive: bool,
}

impl Default for SentimentAnalyzerConfig {
    fn default() -> Self {
        Self {
            neutral_threshold: 0.1,
            case_sensitive: false,
        }
    }
}

impl ConfigurableExtractor for SentimentAnalyzer {
    type Config = SentimentAnalyzerConfig;

    fn config(&self) -> Self::Config {
        SentimentAnalyzerConfig {
            neutral_threshold: self.neutral_threshold,
            case_sensitive: self.case_sensitive,
        }
    }

    fn with_config(&self, config: Self::Config) -> Self {
        let mut analyzer = self.clone();
        analyzer.neutral_threshold = config.neutral_threshold;
        analyzer.case_sensitive = config.case_sensitive;
        analyzer
    }
}

impl FeatureMetadata for SentimentAnalyzer {
    fn feature_types(&self) -> Option<Vec<String>> {
        Some(vec![
            "continuous".to_string(),  // sentiment_score
            "continuous".to_string(),  // positive_ratio
            "continuous".to_string(),  // negative_ratio
            "continuous".to_string(),  // sentiment_density
            "categorical".to_string(), // polarity_encoded
        ])
    }

    fn feature_descriptions(&self) -> Option<Vec<String>> {
        Some(vec![
            "Overall sentiment score [-1.0, 1.0]".to_string(),
            "Ratio of positive words to total words".to_string(),
            "Ratio of negative words to total words".to_string(),
            "Density of sentiment words in text".to_string(),
            "Encoded polarity (-1: negative, 0: neutral, 1: positive)".to_string(),
        ])
    }
}

// ============================================================================
// AdvancedSentimentAnalyzer Trait Implementations
// ============================================================================

impl FeatureExtractor for AdvancedSentimentAnalyzer {
    type Input = String;
    type Output = Array2<Float>;

    fn extract_features(&self, input: &[Self::Input]) -> SklResult<Self::Output> {
        self.extract_features(input)
    }

    fn feature_names(&self) -> Option<Vec<String>> {
        Some(vec![
            "normalized_score".to_string(),
            "raw_score".to_string(),
            "intensity".to_string(),
            "positive_ratio".to_string(),
            "negative_ratio".to_string(),
            "sentiment_density".to_string(),
            "booster_ratio".to_string(),
            "dampener_ratio".to_string(),
            "negation_ratio".to_string(),
            "polarity_encoded".to_string(),
            "absolute_score".to_string(),
            "weighted_score".to_string(),
        ])
    }

    fn n_features(&self) -> Option<usize> {
        Some(12)
    }

    fn validate_input(&self, input: &[Self::Input]) -> SklResult<()> {
        if input.is_empty() {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(
                "Empty input document collection".to_string(),
            ));
        }
        Ok(())
    }
}

impl TextFeatureExtractor for AdvancedSentimentAnalyzer {
    fn vocabulary(&self) -> Option<Vec<String>> {
        // Collect all sentiment words from all languages
        let mut vocab: Vec<String> = Vec::new();

        for lang in &[
            Language::English,
            Language::Spanish,
            Language::French,
            Language::German,
            Language::Japanese,
            Language::Chinese,
        ] {
            if let Some(pos_words) = self.lexicon.get_positive_words(*lang) {
                vocab.extend(pos_words.iter().cloned());
            }
            if let Some(neg_words) = self.lexicon.get_negative_words(*lang) {
                vocab.extend(neg_words.iter().cloned());
            }
        }

        vocab.sort();
        vocab.dedup();
        Some(vocab)
    }

    fn vocabulary_size(&self) -> Option<usize> {
        self.vocabulary().map(|v| v.len())
    }
}

/// Configuration for AdvancedSentimentAnalyzer
#[derive(Debug, Clone)]
pub struct AdvancedSentimentConfig {
    pub auto_detect_language: bool,
    pub default_language: Language,
    pub neutral_threshold: f64,
    pub negation_window: usize,
    pub booster_dampener_window: usize,
    pub case_sensitive: bool,
}

impl Default for AdvancedSentimentConfig {
    fn default() -> Self {
        Self {
            auto_detect_language: true,
            default_language: Language::English,
            neutral_threshold: 0.1,
            negation_window: 3,
            booster_dampener_window: 1,
            case_sensitive: false,
        }
    }
}

impl ConfigurableExtractor for AdvancedSentimentAnalyzer {
    type Config = AdvancedSentimentConfig;

    fn config(&self) -> Self::Config {
        AdvancedSentimentConfig {
            auto_detect_language: self.auto_detect_language,
            default_language: self.default_language,
            neutral_threshold: self.neutral_threshold,
            negation_window: self.negation_window,
            booster_dampener_window: self.booster_dampener_window,
            case_sensitive: self.case_sensitive,
        }
    }

    fn with_config(&self, config: Self::Config) -> Self {
        let mut analyzer = self.clone();
        analyzer.auto_detect_language = config.auto_detect_language;
        analyzer.default_language = config.default_language;
        analyzer.neutral_threshold = config.neutral_threshold;
        analyzer.negation_window = config.negation_window;
        analyzer.booster_dampener_window = config.booster_dampener_window;
        analyzer.case_sensitive = config.case_sensitive;
        analyzer
    }
}

impl FeatureMetadata for AdvancedSentimentAnalyzer {
    fn feature_types(&self) -> Option<Vec<String>> {
        Some(vec![
            "continuous".to_string(),  // normalized_score
            "continuous".to_string(),  // raw_score
            "continuous".to_string(),  // intensity
            "continuous".to_string(),  // positive_ratio
            "continuous".to_string(),  // negative_ratio
            "continuous".to_string(),  // sentiment_density
            "continuous".to_string(),  // booster_ratio
            "continuous".to_string(),  // dampener_ratio
            "continuous".to_string(),  // negation_ratio
            "categorical".to_string(), // polarity_encoded
            "continuous".to_string(),  // absolute_score
            "continuous".to_string(),  // weighted_score
        ])
    }

    fn feature_descriptions(&self) -> Option<Vec<String>> {
        Some(vec![
            "Normalized sentiment score [-1.0, 1.0]".to_string(),
            "Raw sentiment score (sum of all sentiment values)".to_string(),
            "Sentiment intensity [0.0, 1.0]".to_string(),
            "Ratio of positive words to total words".to_string(),
            "Ratio of negative words to total words".to_string(),
            "Density of sentiment-bearing words".to_string(),
            "Ratio of intensity boosters to total words".to_string(),
            "Ratio of intensity dampeners to total words".to_string(),
            "Ratio of negations to total words".to_string(),
            "Encoded polarity (-1: negative, 0: neutral, 1: positive)".to_string(),
            "Absolute value of sentiment score".to_string(),
            "Weighted score (score * intensity)".to_string(),
        ])
    }
}

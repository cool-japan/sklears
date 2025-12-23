//! Tests for the trait-based feature extraction framework

use crate::feature_traits::{
    ConfigurableExtractor, FeatureExtractor, FeatureMetadata, TextFeatureExtractor,
};
use crate::text::trait_impls::{
    AspectSentimentConfig, EmotionDetectorConfig, SentimentAnalyzerConfig,
};
use crate::text::{AspectBasedSentimentAnalyzer, EmotionDetector, SentimentAnalyzer};

// ============================================================================
// EmotionDetector Trait Tests
// ============================================================================

#[test]
fn test_emotion_detector_feature_extractor() {
    let detector = EmotionDetector::new();

    let documents = vec![
        "I am so happy and excited!".to_string(),
        "This is sad and disappointing.".to_string(),
    ];

    // Test extract_features through trait
    let features = detector.extract_features(&documents).unwrap();

    assert_eq!(features.nrows(), 2);
    assert_eq!(features.ncols(), 14);
}

#[test]
fn test_emotion_detector_feature_names() {
    let detector = EmotionDetector::new();

    let names = detector.feature_names().unwrap();

    assert_eq!(names.len(), 14);
    assert_eq!(names[0], "joy_score");
    assert_eq!(names[12], "confidence");
    assert_eq!(names[13], "intensity");
}

#[test]
fn test_emotion_detector_n_features() {
    let detector = EmotionDetector::new();

    assert_eq!(detector.n_features(), Some(14));
}

#[test]
fn test_emotion_detector_validate_input() {
    let detector = EmotionDetector::new();

    // Valid input
    let valid = vec!["test".to_string()];
    assert!(detector.validate_input(&valid).is_ok());

    // Empty input
    let empty: Vec<String> = vec![];
    assert!(detector.validate_input(&empty).is_err());
}

#[test]
fn test_emotion_detector_text_feature_extractor() {
    let detector = EmotionDetector::new();

    // Test vocabulary
    let vocab = detector.vocabulary().unwrap();
    assert!(!vocab.is_empty());
    assert!(vocab.contains(&"happy".to_string()));
    assert!(vocab.contains(&"sad".to_string()));

    // Test vocabulary size
    let vocab_size = detector.vocabulary_size().unwrap();
    assert_eq!(vocab_size, vocab.len());
}

#[test]
fn test_emotion_detector_configurable() {
    let detector = EmotionDetector::new();

    // Get config
    let config = detector.config();
    assert!(!config.case_sensitive);
    assert_eq!(config.min_confidence, 0.1);

    // Create new detector with modified config
    let new_config = EmotionDetectorConfig {
        case_sensitive: true,
        min_confidence: 0.3,
        intensity_weight: 0.7,
    };

    let new_detector = detector.with_config(new_config.clone());
    let retrieved_config = new_detector.config();

    assert_eq!(retrieved_config.case_sensitive, new_config.case_sensitive);
    assert_eq!(retrieved_config.min_confidence, new_config.min_confidence);
}

#[test]
fn test_emotion_detector_metadata() {
    let detector = EmotionDetector::new();

    // Test feature types
    let types = detector.feature_types().unwrap();
    assert_eq!(types.len(), 14);
    assert_eq!(types[0], "continuous"); // joy_score
    assert_eq!(types[6], "binary"); // joy_onehot

    // Test feature descriptions
    let descriptions = detector.feature_descriptions().unwrap();
    assert_eq!(descriptions.len(), 14);
    assert!(descriptions[0].contains("Joy"));
}

// ============================================================================
// AspectBasedSentimentAnalyzer Trait Tests
// ============================================================================

#[test]
fn test_aspect_sentiment_feature_extractor() {
    let analyzer = AspectBasedSentimentAnalyzer::new()
        .add_aspects(vec!["food".to_string(), "service".to_string()]);

    let documents = vec![
        "The food was great!".to_string(),
        "Service was terrible.".to_string(),
    ];

    // Test extract_features through trait
    let features = analyzer.extract_features(&documents).unwrap();

    assert_eq!(features.nrows(), 2);
    assert_eq!(features.ncols(), 6);
}

#[test]
fn test_aspect_sentiment_feature_names() {
    let analyzer = AspectBasedSentimentAnalyzer::new();

    let names = analyzer.feature_names().unwrap();

    assert_eq!(names.len(), 6);
    assert_eq!(names[0], "avg_aspect_score");
    assert_eq!(names[4], "aspect_diversity");
}

#[test]
fn test_aspect_sentiment_n_features() {
    let analyzer = AspectBasedSentimentAnalyzer::new();

    assert_eq!(analyzer.n_features(), Some(6));
}

#[test]
fn test_aspect_sentiment_text_feature_extractor() {
    let analyzer = AspectBasedSentimentAnalyzer::new()
        .add_aspects(vec!["food".to_string(), "service".to_string()]);

    // Test vocabulary
    let vocab = analyzer.vocabulary().unwrap();
    assert_eq!(vocab.len(), 2);
    assert!(vocab.contains(&"food".to_string()));
    assert!(vocab.contains(&"service".to_string()));

    // Test vocabulary size
    let vocab_size = analyzer.vocabulary_size().unwrap();
    assert_eq!(vocab_size, 2);
}

#[test]
fn test_aspect_sentiment_configurable() {
    let analyzer = AspectBasedSentimentAnalyzer::new();

    // Get config
    let config = analyzer.config();
    assert_eq!(config.context_window, 3);

    // Create new analyzer with modified config
    let new_config = AspectSentimentConfig {
        context_window: 5,
        min_confidence: 0.3,
        case_sensitive: true,
        auto_extract_aspects: false,
    };

    let new_analyzer = analyzer.with_config(new_config.clone());
    let retrieved_config = new_analyzer.config();

    assert_eq!(retrieved_config.context_window, new_config.context_window);
    assert_eq!(retrieved_config.min_confidence, new_config.min_confidence);
}

#[test]
fn test_aspect_sentiment_metadata() {
    let analyzer = AspectBasedSentimentAnalyzer::new();

    // Test feature types
    let types = analyzer.feature_types().unwrap();
    assert_eq!(types.len(), 6);
    assert_eq!(types[0], "continuous"); // avg_aspect_score
    assert_eq!(types[1], "count"); // positive_aspects

    // Test feature descriptions
    let descriptions = analyzer.feature_descriptions().unwrap();
    assert_eq!(descriptions.len(), 6);
    assert!(descriptions[0].contains("sentiment score"));
}

// ============================================================================
// SentimentAnalyzer Trait Tests
// ============================================================================

#[test]
fn test_sentiment_analyzer_feature_extractor() {
    let analyzer = SentimentAnalyzer::new();

    let documents = vec![
        "This is great and wonderful!".to_string(),
        "This is terrible and awful.".to_string(),
    ];

    // Test extract_features through trait
    let features = analyzer.extract_features(&documents).unwrap();

    assert_eq!(features.nrows(), 2);
    assert_eq!(features.ncols(), 5);
}

#[test]
fn test_sentiment_analyzer_feature_names() {
    let analyzer = SentimentAnalyzer::new();

    let names = analyzer.feature_names().unwrap();

    assert_eq!(names.len(), 5);
    assert_eq!(names[0], "sentiment_score");
    assert_eq!(names[4], "polarity_encoded");
}

#[test]
fn test_sentiment_analyzer_n_features() {
    let analyzer = SentimentAnalyzer::new();

    assert_eq!(analyzer.n_features(), Some(5));
}

#[test]
fn test_sentiment_analyzer_text_feature_extractor() {
    let analyzer = SentimentAnalyzer::new();

    // Test vocabulary
    let vocab = analyzer.vocabulary().unwrap();
    assert!(!vocab.is_empty());
    assert!(vocab.contains(&"good".to_string()));
    assert!(vocab.contains(&"bad".to_string()));

    // Test vocabulary size
    let vocab_size = analyzer.vocabulary_size().unwrap();
    assert_eq!(vocab_size, vocab.len());
}

#[test]
fn test_sentiment_analyzer_configurable() {
    let analyzer = SentimentAnalyzer::new();

    // Get config
    let config = analyzer.config();
    assert_eq!(config.neutral_threshold, 0.1);

    // Create new analyzer with modified config
    let new_config = SentimentAnalyzerConfig {
        neutral_threshold: 0.2,
        case_sensitive: true,
    };

    let new_analyzer = analyzer.with_config(new_config.clone());
    let retrieved_config = new_analyzer.config();

    assert_eq!(
        retrieved_config.neutral_threshold,
        new_config.neutral_threshold
    );
    assert_eq!(retrieved_config.case_sensitive, new_config.case_sensitive);
}

#[test]
fn test_sentiment_analyzer_metadata() {
    let analyzer = SentimentAnalyzer::new();

    // Test feature types
    let types = analyzer.feature_types().unwrap();
    assert_eq!(types.len(), 5);
    assert_eq!(types[0], "continuous"); // sentiment_score
    assert_eq!(types[4], "categorical"); // polarity_encoded

    // Test feature descriptions
    let descriptions = analyzer.feature_descriptions().unwrap();
    assert_eq!(descriptions.len(), 5);
    assert!(descriptions[0].contains("sentiment score"));
}

// ============================================================================
// Generic Trait Tests
// ============================================================================

#[test]
fn test_trait_polymorphism() {
    // Test that we can use extractors polymorphically through trait objects
    let detector = EmotionDetector::new();
    let analyzer = SentimentAnalyzer::new();

    let documents = vec!["I am happy!".to_string()];

    // Both should work through FeatureExtractor trait
    let emotion_features = detector.extract_features(&documents).unwrap();
    let sentiment_features = analyzer.extract_features(&documents).unwrap();

    assert!(emotion_features.nrows() > 0);
    assert!(sentiment_features.nrows() > 0);
}

#[test]
fn test_feature_names_consistency() {
    let detector = EmotionDetector::new();

    let names = detector.feature_names().unwrap();
    let n_features = detector.n_features().unwrap();

    // Feature names count should match n_features
    assert_eq!(names.len(), n_features);
}

#[test]
fn test_config_immutability() {
    let detector = EmotionDetector::new();
    let original_config = detector.config();

    // Create new detector with different config
    let new_config = EmotionDetectorConfig {
        case_sensitive: true,
        min_confidence: 0.5,
        intensity_weight: 0.8,
    };

    let _new_detector = detector.with_config(new_config);

    // Original detector config should be unchanged
    let still_original_config = detector.config();
    assert_eq!(
        still_original_config.case_sensitive,
        original_config.case_sensitive
    );
}

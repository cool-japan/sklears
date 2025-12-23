//! Demonstration of the Trait-Based Feature Extraction Framework
//!
//! This example showcases the new trait-based feature extraction framework,
//! including polymorphic extractors, configurable extractors, and feature pipelines.

use sklears_feature_extraction::feature_traits::{
    ConfigurableExtractor, FeatureExtractor, FeatureMetadata, FeatureUnion, TextFeatureExtractor,
};
use sklears_feature_extraction::pipelines::{
    IndexFeatureSelector, ParallelFeatureUnion, WeightedFeatureUnion,
};
use sklears_feature_extraction::text::trait_impls::{
    EmotionDetectorConfig, SentimentAnalyzerConfig,
};
use sklears_feature_extraction::text::{
    AspectBasedSentimentAnalyzer, EmotionDetector, SentimentAnalyzer,
};

fn main() {
    println!("========================================");
    println!("Trait-Based Feature Extraction Framework Demo");
    println!("========================================\n");

    // ========================================================================
    // 1. Basic Feature Extraction Through Traits
    // ========================================================================
    println!("1. BASIC FEATURE EXTRACTION THROUGH TRAITS");
    println!("-------------------------------------------");

    let documents = vec![
        "I am so happy and excited!".to_string(),
        "This is sad and disappointing.".to_string(),
        "The food was excellent but the service was terrible.".to_string(),
    ];

    // Create extractors
    let emotion_detector = EmotionDetector::new();
    let sentiment_analyzer = SentimentAnalyzer::new();
    let aspect_analyzer = AspectBasedSentimentAnalyzer::new()
        .add_aspects(vec!["food".to_string(), "service".to_string()]);

    // Use polymorphically through trait
    println!("\n📊 Emotion Detection:");
    let emotion_features = emotion_detector.extract_features(&documents).unwrap();
    println!(
        "  Shape: {} x {}",
        emotion_features.nrows(),
        emotion_features.ncols()
    );
    println!(
        "  Feature names: {:?}",
        emotion_detector.feature_names().unwrap()[0..3].to_vec()
    );

    println!("\n📊 Sentiment Analysis:");
    let sentiment_features = sentiment_analyzer.extract_features(&documents).unwrap();
    println!(
        "  Shape: {} x {}",
        sentiment_features.nrows(),
        sentiment_features.ncols()
    );
    println!(
        "  Feature names: {:?}",
        sentiment_analyzer.feature_names().unwrap()
    );

    println!("\n📊 Aspect-Based Sentiment:");
    let aspect_features = aspect_analyzer.extract_features(&documents).unwrap();
    println!(
        "  Shape: {} x {}",
        aspect_features.nrows(),
        aspect_features.ncols()
    );
    println!(
        "  Feature names: {:?}",
        aspect_analyzer.feature_names().unwrap()
    );

    // ========================================================================
    // 2. Configurable Extractors
    // ========================================================================
    println!("\n\n2. CONFIGURABLE EXTRACTORS");
    println!("---------------------------");

    // Get current configuration
    let current_config = emotion_detector.config();
    println!("\nCurrent EmotionDetector config:");
    println!("  Case sensitive: {}", current_config.case_sensitive);
    println!("  Min confidence: {:.2}", current_config.min_confidence);
    println!("  Intensity weight: {:.2}", current_config.intensity_weight);

    // Create new detector with custom configuration
    let custom_config = EmotionDetectorConfig {
        case_sensitive: true,
        min_confidence: 0.3,
        intensity_weight: 0.7,
    };

    let custom_detector = emotion_detector.with_config(custom_config);
    println!("\nCustom EmotionDetector config:");
    let new_config = custom_detector.config();
    println!("  Case sensitive: {}", new_config.case_sensitive);
    println!("  Min confidence: {:.2}", new_config.min_confidence);
    println!("  Intensity weight: {:.2}", new_config.intensity_weight);

    // ========================================================================
    // 3. Feature Metadata
    // ========================================================================
    println!("\n\n3. FEATURE METADATA");
    println!("-------------------");

    println!("\nEmotionDetector metadata:");
    let feature_types = emotion_detector.feature_types().unwrap();
    println!("  Feature types (first 5): {:?}", &feature_types[0..5]);

    let descriptions = emotion_detector.feature_descriptions().unwrap();
    println!("  Descriptions (first 3):");
    for (i, desc) in descriptions.iter().take(3).enumerate() {
        println!("    [{}]: {}", i, desc);
    }

    // ========================================================================
    // 4. Text Feature Extractor Traits
    // ========================================================================
    println!("\n\n4. TEXT FEATURE EXTRACTOR TRAITS");
    println!("---------------------------------");

    println!("\nEmotionDetector vocabulary:");
    let vocab = emotion_detector.vocabulary().unwrap();
    println!("  Vocabulary size: {}", vocab.len());
    println!("  Sample words: {:?}", &vocab[0..5.min(vocab.len())]);

    println!("\nAspectAnalyzer vocabulary:");
    let aspect_vocab = aspect_analyzer.vocabulary().unwrap();
    println!("  Aspects: {:?}", aspect_vocab);

    // ========================================================================
    // 5. Parallel Feature Union
    // ========================================================================
    println!("\n\n5. PARALLEL FEATURE UNION");
    println!("--------------------------");

    // Create union with multiple emotion detectors with different configs
    let detector1 = EmotionDetector::new();
    let detector2 = EmotionDetector::new().min_confidence(0.2);

    let union = ParallelFeatureUnion::new()
        .add_extractor(detector1, "default_emotion".to_string())
        .add_extractor(detector2, "sensitive_emotion".to_string());

    println!("\nFeature Union:");
    println!("  Number of extractors: {}", union.n_extractors());
    println!("  Extractor names: {:?}", union.extractor_names());

    let combined_features = union.extract_union(&documents).unwrap();
    println!("\nCombined features:");
    println!(
        "  Shape: {} x {}",
        combined_features.nrows(),
        combined_features.ncols()
    );

    let splits = union.feature_splits();
    println!("  Feature splits: {:?}", splits);
    println!("  Extractor 1 features: columns 0-{}", splits[1] - 1);
    println!(
        "  Extractor 2 features: columns {}-{}",
        splits[1],
        splits[2] - 1
    );

    // ========================================================================
    // 6. Weighted Feature Union
    // ========================================================================
    println!("\n\n6. WEIGHTED FEATURE UNION");
    println!("-------------------------");

    let weighted_union = WeightedFeatureUnion::new()
        .add_weighted_extractor(EmotionDetector::new(), 2.0, "high_weight".to_string())
        .add_weighted_extractor(EmotionDetector::new(), 0.5, "low_weight".to_string());

    println!("\nWeighted Feature Union:");
    println!("  Number of extractors: {}", weighted_union.n_extractors());
    println!("  Weights: {:?}", weighted_union.weights());

    let weighted_features = weighted_union.extract_union(&documents).unwrap();
    println!("\nWeighted features:");
    println!(
        "  Shape: {} x {}",
        weighted_features.nrows(),
        weighted_features.ncols()
    );

    // Compare first few features to show weight effect
    println!("\n  First document features (first 3 from each extractor):");
    println!(
        "    High weight (2.0): {:.2} {:.2} {:.2}",
        weighted_features[[0, 0]],
        weighted_features[[0, 1]],
        weighted_features[[0, 2]]
    );
    println!(
        "    Low weight (0.5): {:.2} {:.2} {:.2}",
        weighted_features[[0, 14]],
        weighted_features[[0, 15]],
        weighted_features[[0, 16]]
    );

    // ========================================================================
    // 7. Feature Selection
    // ========================================================================
    println!("\n\n7. FEATURE SELECTION");
    println!("--------------------");

    // Extract features
    let all_features = emotion_detector.extract_features(&documents).unwrap();
    println!(
        "\nOriginal features: {} x {}",
        all_features.nrows(),
        all_features.ncols()
    );

    // Select specific features (e.g., only emotion scores, skip one-hot)
    let selector = IndexFeatureSelector::new(vec![0, 1, 2, 3, 4, 5, 12, 13]);
    let selected_features = selector.select_features(&all_features).unwrap();

    println!(
        "Selected features: {} x {}",
        selected_features.nrows(),
        selected_features.ncols()
    );
    println!("Selected indices: {:?}", selector.selected_indices());

    // ========================================================================
    // 8. Multiple Configurations
    // ========================================================================
    println!("\n\n8. CONFIGURATION COMPARISON");
    println!("---------------------------");

    let configs = vec![
        SentimentAnalyzerConfig {
            neutral_threshold: 0.1,
            case_sensitive: false,
        },
        SentimentAnalyzerConfig {
            neutral_threshold: 0.2,
            case_sensitive: false,
        },
        SentimentAnalyzerConfig {
            neutral_threshold: 0.3,
            case_sensitive: false,
        },
    ];

    let base_analyzer = SentimentAnalyzer::new();

    println!("\nSentiment scores with different neutral thresholds:");
    for (i, config) in configs.iter().enumerate() {
        let analyzer = base_analyzer.with_config(config.clone());
        let features = analyzer.extract_features(&documents).unwrap();

        println!(
            "\n  Config {}: threshold = {:.1}",
            i + 1,
            config.neutral_threshold
        );
        for (doc_idx, _doc) in documents.iter().enumerate() {
            println!(
                "    Doc {}: score = {:.2}, polarity = {:.0}",
                doc_idx + 1,
                features[[doc_idx, 0]],
                features[[doc_idx, 4]]
            );
        }
    }

    // ========================================================================
    // 9. Practical Example: Multi-Level Analysis
    // ========================================================================
    println!("\n\n9. PRACTICAL EXAMPLE: MULTI-LEVEL ANALYSIS");
    println!("------------------------------------------");

    let review = "The food was absolutely amazing and delicious! \
                  However, the service was slow and disappointing. \
                  Overall, I felt frustrated but happy with the meal.";

    println!("\nReview: \"{}\"", review);

    // Create analyzers
    let emotion = EmotionDetector::new();
    let sentiment = SentimentAnalyzer::new();
    let aspects = AspectBasedSentimentAnalyzer::new()
        .add_aspects(vec!["food".to_string(), "service".to_string()]);

    // Extract features
    let emotion_result = emotion.extract_features(&[review.to_string()]).unwrap();
    let sentiment_result = sentiment.extract_features(&[review.to_string()]).unwrap();
    let aspect_result = aspects.extract_features(&[review.to_string()]).unwrap();

    println!("\nMulti-level analysis:");
    println!("  📊 Emotion scores:");
    let emotion_names = emotion.feature_names().unwrap();
    for i in 0..6 {
        if emotion_result[[0, i]] > 0.0 {
            println!("    {}: {:.2}", emotion_names[i], emotion_result[[0, i]]);
        }
    }

    println!("\n  📊 Sentiment:");
    println!("    Score: {:.2}", sentiment_result[[0, 0]]);
    println!(
        "    Polarity: {}",
        match sentiment_result[[0, 4]] as i32 {
            -1 => "Negative",
            0 => "Neutral",
            1 => "Positive",
            _ => "Unknown",
        }
    );

    println!("\n  📊 Aspect-based:");
    println!("    Avg aspect score: {:.2}", aspect_result[[0, 0]]);
    println!("    Positive aspects: {:.0}", aspect_result[[0, 1]]);
    println!("    Negative aspects: {:.0}", aspect_result[[0, 2]]);
    println!("    Aspect diversity: {:.2}", aspect_result[[0, 4]]);

    println!("\n========================================");
    println!("Demo completed!");
    println!("========================================");
}

//! Demonstration of Multilingual Sentiment Analysis and Intensity Scoring
//!
//! This example showcases:
//! - Automatic language detection
//! - Multilingual sentiment analysis (English, Spanish, French, German, Japanese, Chinese)
//! - VADER-style intensity modifiers (boosters, dampeners, negations)
//! - Enhanced feature extraction with intensity metrics
//! - Comparison between basic and advanced sentiment analysis

use sklears_feature_extraction::feature_traits::{
    ConfigurableExtractor, FeatureExtractor, FeatureMetadata, TextFeatureExtractor,
};
use sklears_feature_extraction::text::advanced_sentiment::{AdvancedSentimentAnalyzer, Language};
use sklears_feature_extraction::text::trait_impls::AdvancedSentimentConfig;
use sklears_feature_extraction::text::SentimentAnalyzer;

fn main() {
    println!("========================================================================");
    println!("Multilingual Sentiment Analysis & Intensity Scoring Demo");
    println!("========================================================================\n");

    // ========================================================================
    // 1. Language Detection
    // ========================================================================
    println!("1. AUTOMATIC LANGUAGE DETECTION");
    println!("--------------------------------");

    let texts = vec![
        ("Hello, this is great!", Language::English),
        ("Hola, esto es excelente!", Language::Spanish),
        ("Bonjour, c'est magnifique!", Language::French),
        ("Hallo, das ist wunderbar!", Language::German),
        ("こんにちは、これは素晴らしい！", Language::Japanese),
        ("你好，这很棒！", Language::Chinese),
    ];

    println!("\nDetecting languages:");
    for (text, expected_lang) in &texts {
        let detected = Language::detect(text);
        let match_indicator = if detected == *expected_lang {
            "✓"
        } else {
            "✗"
        };
        println!(
            "  {} Text: \"{}\" → Detected: {:?}",
            match_indicator, text, detected
        );
    }

    // ========================================================================
    // 2. Basic vs Advanced Sentiment Analysis
    // ========================================================================
    println!("\n\n2. BASIC VS ADVANCED SENTIMENT ANALYSIS");
    println!("----------------------------------------");

    let test_sentences = vec![
        "This is good.",
        "This is very good.",
        "This is extremely good.",
        "This is not good.",
        "This is not bad.",
        "This is somewhat good.",
        "This is barely acceptable.",
    ];

    let basic_analyzer = SentimentAnalyzer::new();
    let advanced_analyzer = AdvancedSentimentAnalyzer::new();

    println!("\nComparing basic vs advanced sentiment analysis:\n");
    println!(
        "{:<30} | {:>8} | {:>8} | {:>10} | {:>8}",
        "Text", "Basic", "Advanced", "Intensity", "Details"
    );
    println!("{}", "-".repeat(85));

    for sentence in &test_sentences {
        let basic_result = basic_analyzer.analyze_sentiment(sentence);
        let advanced_result = advanced_analyzer.analyze(sentence);

        let details = if advanced_result.booster_count > 0 {
            format!("{}B", advanced_result.booster_count)
        } else if advanced_result.dampener_count > 0 {
            format!("{}D", advanced_result.dampener_count)
        } else if advanced_result.negation_count > 0 {
            format!("{}N", advanced_result.negation_count)
        } else {
            "-".to_string()
        };

        println!(
            "{:<30} | {:>8.2} | {:>8.2} | {:>10.2} | {:>8}",
            sentence, basic_result.score, advanced_result.score, advanced_result.intensity, details
        );
    }
    println!("\nLegend: B=Boosters, D=Dampeners, N=Negations\n");

    // ========================================================================
    // 3. Multilingual Sentiment Analysis
    // ========================================================================
    println!("\n3. MULTILINGUAL SENTIMENT ANALYSIS");
    println!("----------------------------------");

    let multilingual_texts = vec![
        ("This is very good and wonderful!", "English"),
        ("¡Esto es muy bueno y maravilloso!", "Spanish"),
        ("C'est très bon et magnifique!", "French"),
        ("Das ist sehr gut und wunderbar!", "German"),
        ("これは非常に良い素晴らしい！", "Japanese"),
        ("这非常好很棒！", "Chinese"),
    ];

    println!("\nSentiment analysis across languages:\n");
    println!(
        "{:<20} | {:>10} | {:>8} | {:>10} | {:>10}",
        "Language", "Detected", "Score", "Intensity", "Polarity"
    );
    println!("{}", "-".repeat(70));

    for (text, lang_name) in &multilingual_texts {
        let result = advanced_analyzer.analyze(text);
        let polarity_str = match result.polarity {
            sklears_feature_extraction::text::advanced_sentiment::SentimentPolarity::Positive => {
                "Positive"
            }
            sklears_feature_extraction::text::advanced_sentiment::SentimentPolarity::Negative => {
                "Negative"
            }
            sklears_feature_extraction::text::advanced_sentiment::SentimentPolarity::Neutral => {
                "Neutral"
            }
        };

        println!(
            "{:<20} | {:>10?} | {:>8.2} | {:>10.2} | {:>10}",
            lang_name, result.detected_language, result.score, result.intensity, polarity_str
        );
    }

    // ========================================================================
    // 4. Intensity Modifiers in Action
    // ========================================================================
    println!("\n\n4. INTENSITY MODIFIERS IN ACTION");
    println!("---------------------------------");

    let intensity_examples = vec![
        ("I like this", "Base sentiment"),
        ("I really like this", "With booster"),
        ("I extremely love this", "Strong booster"),
        ("I somewhat like this", "With dampener"),
        ("I barely like this", "Strong dampener"),
        ("I don't like this", "With negation"),
        ("I really don't like this", "Booster + negation"),
    ];

    println!("\nIntensity modifier effects:\n");
    println!(
        "{:<30} | {:>8} | {:>10} | {:<25}",
        "Text", "Score", "Intensity", "Modifiers"
    );
    println!("{}", "-".repeat(80));

    for (text, _description) in &intensity_examples {
        let result = advanced_analyzer.analyze(text);

        let modifiers = if result.booster_count > 0 && result.negation_count > 0 {
            format!(
                "Boosters: {}, Negations: {}",
                result.booster_count, result.negation_count
            )
        } else if result.booster_count > 0 {
            format!("Boosters: {}", result.booster_count)
        } else if result.dampener_count > 0 {
            format!("Dampeners: {}", result.dampener_count)
        } else if result.negation_count > 0 {
            format!("Negations: {}", result.negation_count)
        } else {
            "None".to_string()
        };

        println!(
            "{:<30} | {:>8.2} | {:>10.2} | {:<25}",
            text, result.score, result.intensity, modifiers
        );
    }

    // ========================================================================
    // 5. Feature Extraction
    // ========================================================================
    println!("\n\n5. FEATURE EXTRACTION");
    println!("---------------------");

    let documents = vec![
        "This is extremely good and wonderful!".to_string(),
        "This is not bad at all.".to_string(),
        "This is somewhat okay.".to_string(),
    ];

    let features = advanced_analyzer.extract_features(&documents).unwrap();
    let feature_names = advanced_analyzer.feature_names().unwrap();

    println!(
        "\nExtracted features ({} features per document):",
        features.ncols()
    );
    println!("\nFeature names:");
    for (i, name) in feature_names.iter().enumerate() {
        println!("  [{}]: {}", i, name);
    }

    println!("\nFeature matrix:");
    println!("{:<30} | {}", "Document", "Features (first 6)");
    println!("{}", "-".repeat(90));

    for (i, doc) in documents.iter().enumerate() {
        let doc_short = if doc.len() > 27 {
            format!("{}...", &doc[..27])
        } else {
            doc.clone()
        };

        print!("{:<30} | ", doc_short);
        for j in 0..6 {
            print!("{:>6.2} ", features[(i, j)]);
        }
        println!();
    }

    // ========================================================================
    // 6. Configuration Options
    // ========================================================================
    println!("\n\n6. CONFIGURATION OPTIONS");
    println!("------------------------");

    // Create analyzer with custom configuration
    let custom_config = AdvancedSentimentConfig {
        auto_detect_language: false,
        default_language: Language::Spanish,
        neutral_threshold: 0.2,
        negation_window: 5,
        booster_dampener_window: 2,
        case_sensitive: false,
    };

    let custom_analyzer = advanced_analyzer.with_config(custom_config);

    println!("\nCustom configuration:");
    let current_config = custom_analyzer.config();
    println!(
        "  Auto-detect language: {}",
        current_config.auto_detect_language
    );
    println!("  Default language: {:?}", current_config.default_language);
    println!("  Neutral threshold: {}", current_config.neutral_threshold);
    println!("  Negation window: {}", current_config.negation_window);
    println!(
        "  Booster/dampener window: {}",
        current_config.booster_dampener_window
    );

    // ========================================================================
    // 7. Trait-Based Feature Extraction
    // ========================================================================
    println!("\n\n7. TRAIT-BASED FEATURE EXTRACTION");
    println!("----------------------------------");

    // Use polymorphically through trait
    let analyzer: &dyn FeatureExtractor<
        Input = String,
        Output = scirs2_core::ndarray::Array2<f64>,
    > = &advanced_analyzer;

    println!("\nAnalyzer metadata:");
    println!("  Number of features: {:?}", analyzer.n_features());
    println!(
        "  Vocabulary size: {:?}",
        advanced_analyzer.vocabulary_size()
    );

    let feature_types = advanced_analyzer.feature_types().unwrap();
    println!("\nFeature types:");
    for (name, ftype) in feature_names.iter().zip(feature_types.iter()) {
        println!("  {:<25}: {}", name, ftype);
    }

    // ========================================================================
    // 8. Real-World Example: Product Reviews
    // ========================================================================
    println!("\n\n8. REAL-WORLD EXAMPLE: PRODUCT REVIEWS");
    println!("---------------------------------------");

    let reviews = vec![
        "This product is absolutely amazing! I love it so much.",
        "Not bad, but could be better. Somewhat disappointed.",
        "Terrible experience. I really hate this product.",
        "It's okay, nothing special.",
        "Very very good! Extremely satisfied with my purchase!",
    ];

    println!("\nAnalyzing product reviews:\n");
    println!(
        "{:<55} | {:>8} | {:>10} | {:>10}",
        "Review", "Score", "Intensity", "Sentiment"
    );
    println!("{}", "-".repeat(90));

    for review in &reviews {
        let result = advanced_analyzer.analyze(review);

        let review_short = if review.len() > 52 {
            format!("{}...", &review[..52])
        } else {
            review.to_string()
        };

        let sentiment = match result.polarity {
            sklears_feature_extraction::text::advanced_sentiment::SentimentPolarity::Positive => {
                "👍 Positive"
            }
            sklears_feature_extraction::text::advanced_sentiment::SentimentPolarity::Negative => {
                "👎 Negative"
            }
            sklears_feature_extraction::text::advanced_sentiment::SentimentPolarity::Neutral => {
                "😐 Neutral"
            }
        };

        println!(
            "{:<55} | {:>8.2} | {:>10.2} | {:>10}",
            review_short, result.score, result.intensity, sentiment
        );
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n\n========================================================================");
    println!("SUMMARY");
    println!("========================================================================");
    println!("\n✓ Demonstrated automatic language detection for 6 languages");
    println!("✓ Showed intensity modifiers (boosters, dampeners, negations)");
    println!("✓ Compared basic vs advanced sentiment analysis");
    println!("✓ Extracted 12-dimensional feature vectors");
    println!("✓ Demonstrated configurable analyzer options");
    println!("✓ Analyzed real-world product reviews");
    println!("\nKey Features:");
    println!("  - Multilingual support: English, Spanish, French, German, Japanese, Chinese");
    println!("  - VADER-style intensity scoring");
    println!("  - Context-aware negation handling");
    println!("  - Trait-based polymorphic interface");
    println!("  - Rich feature extraction (12 features)");
    println!("\n========================================================================");
}

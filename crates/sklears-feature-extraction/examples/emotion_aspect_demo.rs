//! Demonstration of Emotion Detection and Aspect-Based Sentiment Analysis
//!
//! This example showcases the new emotion detection and aspect-based sentiment
//! analysis capabilities for advanced text analytics.

use sklears_feature_extraction::text::{
    AspectBasedSentimentAnalyzer, EmotionDetector, SentimentAnalyzer,
};

fn main() {
    println!("========================================");
    println!("Emotion Detection and Aspect-Based Sentiment Analysis Demo");
    println!("========================================\n");

    // ========================================================================
    // 1. Emotion Detection
    // ========================================================================
    println!("1. EMOTION DETECTION");
    println!("--------------------");

    let emotion_detector = EmotionDetector::new();

    let test_texts = vec![
        "I am so happy and excited about this wonderful news!",
        "This makes me feel sad and depressed. I'm heartbroken.",
        "I'm absolutely furious and angry about this situation!",
        "I'm scared and terrified. This is really frightening.",
        "Wow! This is so surprising and amazing! I can't believe it!",
        "This is disgusting and revolting. It makes me sick.",
    ];

    for text in &test_texts {
        let result = emotion_detector.detect_emotion(text);
        println!("\nText: \"{}\"", text);
        println!("  Primary Emotion: {:?}", result.primary_emotion);
        println!("  Confidence: {:.2}", result.confidence);
        println!("  Intensity: {:.2}", result.intensity());

        if let Some((secondary, score)) = result.secondary_emotion() {
            println!("  Secondary Emotion: {:?} (score: {:.2})", secondary, score);
        }

        // Show top emotion scores
        let mut scores: Vec<_> = result.emotion_scores.iter().collect();
        scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        println!("  Top emotions:");
        for (emotion, score) in scores.iter().take(3) {
            if **score > 0.0 {
                println!("    {:?}: {:.2}", emotion, score);
            }
        }
    }

    // ========================================================================
    // 2. Emotion Feature Extraction
    // ========================================================================
    println!("\n\n2. EMOTION FEATURE EXTRACTION");
    println!("------------------------------");

    let documents = vec![
        "I love this! It's amazing and wonderful!".to_string(),
        "This is terrible and makes me sad.".to_string(),
        "I'm so angry and frustrated with this!".to_string(),
    ];

    let features = emotion_detector.extract_features(&documents).unwrap();
    println!(
        "\nFeature matrix shape: {} x {}",
        features.nrows(),
        features.ncols()
    );
    println!("\nFeature breakdown (14 features per document):");
    println!("  [0-5]: Emotion scores (joy, sadness, anger, fear, surprise, disgust)");
    println!("  [6-11]: Primary emotion one-hot encoding");
    println!("  [12]: Confidence score");
    println!("  [13]: Emotion intensity");

    for (i, doc) in documents.iter().enumerate() {
        println!("\nDocument {}: \"{}\"", i + 1, doc);
        let row = features.row(i);
        println!("  Joy score: {:.2}", row[0]);
        println!("  Sadness score: {:.2}", row[1]);
        println!("  Anger score: {:.2}", row[2]);
        println!("  Confidence: {:.2}", row[12]);
        println!("  Intensity: {:.2}", row[13]);
    }

    // ========================================================================
    // 3. Aspect-Based Sentiment Analysis
    // ========================================================================
    println!("\n\n3. ASPECT-BASED SENTIMENT ANALYSIS");
    println!("-----------------------------------");

    let aspect_analyzer = AspectBasedSentimentAnalyzer::new().add_aspects(vec![
        "food".to_string(),
        "service".to_string(),
        "price".to_string(),
        "atmosphere".to_string(),
    ]);

    let reviews = [
        "The food was absolutely delicious, but the service was terrible.",
        "Great atmosphere and excellent service, though a bit pricey.",
        "The price is reasonable and the food is amazing!",
        "Wonderful food and great atmosphere. Service could be better.",
    ];

    for (i, review) in reviews.iter().enumerate() {
        println!("\nReview {}: \"{}\"", i + 1, review);

        let aspects = aspect_analyzer.analyze(review);

        if aspects.is_empty() {
            println!("  No aspects detected");
            continue;
        }

        for aspect in &aspects {
            println!("\n  Aspect: {}", aspect.aspect);
            println!("    Sentiment: {:?}", aspect.sentiment);
            println!("    Score: {:.2}", aspect.score);
            println!("    Confidence: {:.2}", aspect.confidence);
            if !aspect.opinion_words.is_empty() {
                println!("    Opinion words: {}", aspect.opinion_words.join(", "));
            }
        }
    }

    // ========================================================================
    // 4. Aspect Aggregation and Summary
    // ========================================================================
    println!("\n\n4. ASPECT AGGREGATION AND SUMMARY");
    println!("----------------------------------");

    let review_docs: Vec<String> = reviews.iter().map(|s| s.to_string()).collect();

    let summary = aspect_analyzer.aspect_summary(&review_docs);

    println!("\nAspect Summary (sorted by frequency):");
    println!("{:<15} {:<15} {:<10}", "Aspect", "Avg Score", "Count");
    println!("{}", "-".repeat(40));

    for (aspect, avg_score, count) in summary {
        let sentiment_label = if avg_score > 0.2 {
            "Positive"
        } else if avg_score < -0.2 {
            "Negative"
        } else {
            "Neutral"
        };
        println!(
            "{:<15} {:<15} {:<10}",
            aspect,
            format!("{:.2} ({})", avg_score, sentiment_label),
            count
        );
    }

    // ========================================================================
    // 5. Combined Analysis
    // ========================================================================
    println!("\n\n5. COMBINED ANALYSIS");
    println!("--------------------");

    let sentiment_analyzer = SentimentAnalyzer::new();

    let test_review = "The food was absolutely amazing and the atmosphere was wonderful, but the service was disappointing and made me frustrated.";

    println!("\nReview: \"{}\"", test_review);

    // Sentiment analysis
    let sentiment = sentiment_analyzer.analyze_sentiment(test_review);
    println!("\nOverall Sentiment:");
    println!("  Polarity: {:?}", sentiment.polarity);
    println!("  Score: {:.2}", sentiment.score);

    // Emotion detection
    let emotion = emotion_detector.detect_emotion(test_review);
    println!("\nPrimary Emotion:");
    println!("  Emotion: {:?}", emotion.primary_emotion);
    println!("  Confidence: {:.2}", emotion.confidence);

    // Aspect-based analysis
    let aspects = aspect_analyzer.analyze(test_review);
    println!("\nAspect-Based Analysis:");
    for aspect in aspects {
        println!(
            "  {} -> {:?} (score: {:.2})",
            aspect.aspect, aspect.sentiment, aspect.score
        );
    }

    // ========================================================================
    // 6. Emotion Distribution Analysis
    // ========================================================================
    println!("\n\n6. EMOTION DISTRIBUTION ANALYSIS");
    println!("---------------------------------");

    let mixed_documents = vec![
        "I'm so happy!".to_string(),
        "This is wonderful!".to_string(),
        "I feel sad today.".to_string(),
        "This is frustrating!".to_string(),
        "Neutral statement here.".to_string(),
    ];

    let distribution = emotion_detector.analyze_distribution(&mixed_documents);

    println!(
        "\nEmotion distribution across {} documents:",
        mixed_documents.len()
    );
    let mut dist_vec: Vec<_> = distribution.iter().collect();
    dist_vec.sort_by_key(|(_emotion, count)| std::cmp::Reverse(*count));

    for (emotion, count) in dist_vec {
        if *count > 0 {
            println!("  {:?}: {} document(s)", emotion, count);
        }
    }

    println!("\n========================================");
    println!("Demo completed!");
    println!("========================================");
}

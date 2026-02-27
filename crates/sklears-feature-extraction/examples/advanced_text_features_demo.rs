//! Advanced Text Features Demonstration
//!
//! Shows sentiment analysis, memory-efficient processing, and comprehensive text vectorization

use sklears_feature_extraction::{
    CountVectorizer, SentimentAnalyzer, SentimentPolarity, StreamingTextProcessor, TfidfVectorizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Text Feature Extraction Demo");
    println!("========================================");

    // Sample documents with clear sentiment
    let documents = vec![
        "This movie is absolutely fantastic! I loved every minute of it.".to_string(),
        "What a terrible and boring film. Complete waste of time.".to_string(),
        "The weather today is nice and sunny.".to_string(),
        "I hate waiting in long lines. So frustrating and annoying!".to_string(),
        "The book was okay, nothing special but not bad either.".to_string(),
        "Amazing performance! Outstanding acting and brilliant direction.".to_string(),
    ];

    println!("📄 Sample Documents with Sentiment:");
    for (i, doc) in documents.iter().enumerate() {
        println!("   {}: \"{}\"", i + 1, doc);
    }
    println!();

    // 1. Sentiment Analysis Demo
    println!("😊 Sentiment Analysis Demonstration");
    println!("===================================");

    let sentiment_analyzer = SentimentAnalyzer::new()
        .neutral_threshold(0.15)
        .case_sensitive(false);

    println!("Individual Sentiment Analysis:");
    for (i, document) in documents.iter().enumerate() {
        let sentiment = sentiment_analyzer.analyze_sentiment(document);
        let polarity_str = match sentiment.polarity {
            SentimentPolarity::Positive => "😊 Positive",
            SentimentPolarity::Negative => "😞 Negative",
            SentimentPolarity::Neutral => "😐 Neutral",
        };

        println!(
            "   Doc {}: {} (score: {:.3}, +{} -{} words)",
            i + 1,
            polarity_str,
            sentiment.score,
            sentiment.positive_words,
            sentiment.negative_words
        );
    }
    println!();

    // Extract sentiment feature matrix
    match sentiment_analyzer.extract_features(&documents) {
        Ok(sentiment_features) => {
            let (n_docs, n_features) = sentiment_features.dim();
            println!("✅ Sentiment feature extraction successful!");
            println!(
                "   Feature matrix: {} documents × {} features",
                n_docs, n_features
            );
            println!(
                "   Features: [score, pos_ratio, neg_ratio, sentiment_density, polarity_encoded]"
            );

            // Show features for first document
            if n_docs > 0 {
                let first_doc_features: Vec<f64> =
                    sentiment_features.row(0).iter().cloned().collect();
                println!(
                    "   First doc features: {:?}",
                    first_doc_features
                        .iter()
                        .map(|x| format!("{:.3}", x))
                        .collect::<Vec<_>>()
                );
            }
        }
        Err(e) => {
            println!("❌ Sentiment feature extraction failed: {:?}", e);
        }
    }
    println!();

    // 2. Memory-Efficient Processing Demo
    println!("💾 Memory-Efficient Streaming Processing");
    println!("=======================================");

    // Create a large text document for streaming
    let large_text = documents.join(" ").repeat(50); // Simulate large document
    println!("📊 Large document stats:");
    println!("   Length: {} characters", large_text.len());
    println!("   Words: ~{}", large_text.split_whitespace().count());

    let streaming_processor = StreamingTextProcessor::new()
        .chunk_size(1000) // Small chunks for demo
        .overlap_size(100)
        .min_chunk_words(10);

    // Extract streaming statistical features
    match streaming_processor.extract_streaming_stats(&large_text) {
        Ok(stats) => {
            println!("✅ Streaming statistics extraction successful!");
            println!(
                "   Stats features: {:?}",
                stats
                    .iter()
                    .map(|x| format!("{:.1}", x))
                    .collect::<Vec<_>>()
            );
            println!("   [total_chars, total_words, total_sentences, avg_word_len, avg_words_per_sentence, avg_chars_per_word]");
        }
        Err(e) => {
            println!("❌ Streaming statistics failed: {:?}", e);
        }
    }

    // Stream processing with CountVectorizer
    let mut count_vectorizer = CountVectorizer::new()
        .ngram_range((1, 2))
        .min_df(2)
        .max_df(0.8);

    match streaming_processor
        .stream_process_with_count_vectorizer(&large_text, &mut count_vectorizer)
    {
        Ok(count_features) => {
            let (n_docs, n_features) = count_features.dim();
            println!("✅ Streaming count vectorization successful!");
            println!(
                "   Aggregated features: {} documents × {} features",
                n_docs, n_features
            );

            let feature_names = count_vectorizer.get_feature_names();
            if !feature_names.is_empty() {
                println!("   Total vocabulary size: {}", feature_names.len());
                println!(
                    "   Sample vocabulary: {:?}",
                    &feature_names[..5.min(feature_names.len())]
                );
            }
        }
        Err(e) => {
            println!("❌ Streaming count vectorization failed: {:?}", e);
        }
    }

    // Stream processing with TF-IDF
    let mut tfidf_vectorizer = TfidfVectorizer::new()
        .ngram_range((1, 2))
        .min_df(2)
        .use_idf(true)
        .norm(Some("l2".to_string()));

    match streaming_processor.stream_process_with_tfidf(&large_text, &mut tfidf_vectorizer) {
        Ok(tfidf_features) => {
            let (n_docs, n_features) = tfidf_features.dim();
            println!("✅ Streaming TF-IDF vectorization successful!");
            println!(
                "   Aggregated features: {} documents × {} features",
                n_docs, n_features
            );

            let idf_weights = tfidf_vectorizer.get_idf_weights();
            if !idf_weights.is_empty() {
                let mean_idf = idf_weights.iter().sum::<f64>() / idf_weights.len() as f64;
                let max_idf = idf_weights.iter().fold(0.0_f64, |acc, &x| f64::max(acc, x));
                println!(
                    "   IDF weights - Mean: {:.3}, Max: {:.3}",
                    mean_idf, max_idf
                );
            }
        }
        Err(e) => {
            println!("❌ Streaming TF-IDF vectorization failed: {:?}", e);
        }
    }
    println!();

    // 3. Advanced Vectorization Demo
    println!("🔬 Advanced Vectorization Features");
    println!("==================================");

    // Test with different configurations
    let configurations = vec![
        ("Basic unigrams", CountVectorizer::new().ngram_range((1, 1))),
        (
            "Unigrams + Bigrams",
            CountVectorizer::new().ngram_range((1, 2)),
        ),
        ("Binary mode", CountVectorizer::new().binary(true)),
        (
            "With stop words",
            CountVectorizer::new().stop_words(Some(vec![
                "is".to_string(),
                "a".to_string(),
                "the".to_string(),
            ])),
        ),
    ];

    for (name, mut vectorizer) in configurations {
        match vectorizer.fit_transform(&documents) {
            Ok(features) => {
                let (n_docs, n_features) = features.dim();
                let sparsity = calculate_sparsity(&features);
                println!(
                    "✅ {}: {} × {} features (sparsity: {:.1}%)",
                    name,
                    n_docs,
                    n_features,
                    sparsity * 100.0
                );
            }
            Err(e) => {
                println!("❌ {} failed: {:?}", name, e);
            }
        }
    }
    println!();

    // 4. Performance Comparison
    println!("⚡ Performance Features Summary");
    println!("==============================");

    let start_time = std::time::Instant::now();

    // Quick benchmark of different approaches
    let mut results = Vec::new();

    // Basic count vectorization
    let timer = std::time::Instant::now();
    let mut cv = CountVectorizer::new().min_df(1);
    if cv.fit_transform(&documents).is_ok() {
        results.push(("Count Vectorizer", timer.elapsed()));
    }

    // TF-IDF vectorization
    let timer = std::time::Instant::now();
    let mut tfidf = TfidfVectorizer::new().min_df(1);
    if tfidf.fit_transform(&documents).is_ok() {
        results.push(("TF-IDF Vectorizer", timer.elapsed()));
    }

    // Sentiment analysis
    let timer = std::time::Instant::now();
    let sentiment = SentimentAnalyzer::new();
    if sentiment.extract_features(&documents).is_ok() {
        results.push(("Sentiment Analysis", timer.elapsed()));
    }

    let total_time = start_time.elapsed();

    println!("Timing Results:");
    for (method, duration) in results {
        println!("   {}: {:.2?}", method, duration);
    }
    println!("   Total demo time: {:.2?}", total_time);
    println!();

    println!("🎉 Advanced Text Features Demo Complete!");
    println!("========================================");
    println!("✅ Implemented Features:");
    println!("   • Complete CountVectorizer with n-grams, filtering, stop words");
    println!("   • Full TF-IDF with sublinear scaling, normalization, IDF options");
    println!("   • Sentiment Analysis with configurable lexicon and thresholds");
    println!("   • Memory-efficient streaming processing for large texts");
    println!("   • Advanced configuration options and error handling");
    println!("   • Production-ready performance and scalability");

    Ok(())
}

fn calculate_sparsity(matrix: &scirs2_core::ndarray::Array2<f64>) -> f64 {
    let total_elements = matrix.len();
    let zero_elements = matrix.iter().filter(|&&x| x == 0.0).count();
    zero_elements as f64 / total_elements as f64
}

//! Benchmarks for text analysis features
//!
//! These benchmarks measure the performance of text feature extraction methods
//! including sentiment analysis, emotion detection, and aspect-based sentiment analysis.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sklears_core::traits::Fit;
use sklears_feature_extraction::text::{
    AspectBasedSentimentAnalyzer, CountVectorizer, EmotionDetector, SentimentAnalyzer,
    TfidfVectorizer,
};

/// Generate sample documents for benchmarking
fn generate_documents(count: usize, words_per_doc: usize) -> Vec<String> {
    let words = vec![
        "great",
        "terrible",
        "good",
        "bad",
        "excellent",
        "awful",
        "amazing",
        "horrible",
        "wonderful",
        "disgusting",
        "happy",
        "sad",
        "angry",
        "fearful",
        "surprised",
        "food",
        "service",
        "product",
        "quality",
        "price",
        "value",
        "experience",
        "the",
        "is",
        "was",
        "and",
        "but",
        "with",
        "for",
        "at",
        "to",
        "from",
    ];

    (0..count)
        .map(|_| {
            (0..words_per_doc)
                .map(|i| words[i % words.len()])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Benchmark sentiment analysis
fn bench_sentiment_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentiment_analysis");

    for size in [10, 50, 100, 500].iter() {
        let documents = generate_documents(*size, 20);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let analyzer = SentimentAnalyzer::new();
            b.iter(|| {
                for doc in &documents {
                    black_box(analyzer.analyze_sentiment(doc));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark sentiment feature extraction
fn bench_sentiment_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentiment_features");

    for size in [10, 50, 100, 500].iter() {
        let documents = generate_documents(*size, 20);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let analyzer = SentimentAnalyzer::new();
            b.iter(|| {
                black_box(analyzer.extract_features(&documents).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark emotion detection
fn bench_emotion_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("emotion_detection");

    for size in [10, 50, 100, 500].iter() {
        let documents = generate_documents(*size, 20);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let detector = EmotionDetector::new();
            b.iter(|| {
                for doc in &documents {
                    black_box(detector.detect_emotion(doc));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark emotion feature extraction
fn bench_emotion_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("emotion_features");

    for size in [10, 50, 100, 500].iter() {
        let documents = generate_documents(*size, 20);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let detector = EmotionDetector::new();
            b.iter(|| {
                black_box(detector.extract_features(&documents).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark aspect-based sentiment analysis
fn bench_aspect_sentiment(c: &mut Criterion) {
    let mut group = c.benchmark_group("aspect_sentiment");

    for size in [10, 50, 100, 500].iter() {
        let documents = generate_documents(*size, 20);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let analyzer = AspectBasedSentimentAnalyzer::new().add_aspects(vec![
                "food".to_string(),
                "service".to_string(),
                "product".to_string(),
            ]);
            b.iter(|| {
                for doc in &documents {
                    black_box(analyzer.analyze(doc));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark aspect feature extraction
fn bench_aspect_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("aspect_features");

    for size in [10, 50, 100, 500].iter() {
        let documents = generate_documents(*size, 20);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let analyzer = AspectBasedSentimentAnalyzer::new().add_aspects(vec![
                "food".to_string(),
                "service".to_string(),
                "product".to_string(),
            ]);
            b.iter(|| {
                black_box(analyzer.extract_features(&documents).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark CountVectorizer
fn bench_count_vectorizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_vectorizer");

    for size in [10, 50, 100, 500].iter() {
        let documents = generate_documents(*size, 20);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let vectorizer = CountVectorizer::new();
                let fitted = vectorizer.fit(&documents, &()).unwrap();
                black_box(fitted.transform(&documents).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark TfidfVectorizer
fn bench_tfidf_vectorizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("tfidf_vectorizer");

    for size in [10, 50, 100, 500].iter() {
        let documents = generate_documents(*size, 20);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let vectorizer = TfidfVectorizer::new();
                let fitted = vectorizer.fit(&documents, &()).unwrap();
                black_box(fitted.transform(&documents).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark combined pipeline
fn bench_combined_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_pipeline");

    for size in [10, 50, 100].iter() {
        let documents = generate_documents(*size, 20);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                // Combined analysis: sentiment + emotion + aspect
                let sentiment_analyzer = SentimentAnalyzer::new();
                let emotion_detector = EmotionDetector::new();
                let aspect_analyzer = AspectBasedSentimentAnalyzer::new()
                    .add_aspects(vec!["food".to_string(), "service".to_string()]);

                for doc in &documents {
                    black_box(sentiment_analyzer.analyze_sentiment(doc));
                    black_box(emotion_detector.detect_emotion(doc));
                    black_box(aspect_analyzer.analyze(doc));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    text_benches,
    bench_sentiment_analysis,
    bench_sentiment_features,
    bench_emotion_detection,
    bench_emotion_features,
    bench_aspect_sentiment,
    bench_aspect_features,
    bench_count_vectorizer,
    bench_tfidf_vectorizer,
    bench_combined_pipeline,
);

criterion_main!(text_benches);

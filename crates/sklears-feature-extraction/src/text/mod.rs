//! Text feature extraction utilities
//!
//! This module provides comprehensive text feature extraction implementations including
//! text preprocessing, embeddings, and various vectorization techniques.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Fit, SklearsError, Transform},
    types::Float,
};
use std::collections::HashMap;

// Export existing modules
pub mod advanced_sentiment;
pub mod embeddings;
pub mod preprocessing;
pub mod trait_impls;

/// Complete Count Vectorizer implementation for text feature extraction
///
/// Converts a collection of text documents to a matrix of token counts.
/// Supports n-grams, stop word filtering, and various preprocessing options.
#[derive(Debug, Clone)]
pub struct CountVectorizer {
    max_features: Option<usize>,
    min_df: Option<usize>,
    max_df: Option<f64>,
    ngram_range: (usize, usize),
    binary: bool,
    lowercase: bool,
    stop_words: Option<Vec<String>>,
    vocabulary: Option<HashMap<String, usize>>,
    feature_names: Vec<String>,
    document_frequencies: HashMap<String, usize>,
}

impl CountVectorizer {
    /// Create a new CountVectorizer with default parameters
    pub fn new() -> Self {
        Self {
            max_features: None,
            min_df: Some(1),
            max_df: Some(1.0),
            ngram_range: (1, 1),
            binary: false,
            lowercase: true,
            stop_words: None,
            vocabulary: None,
            feature_names: Vec::new(),
            document_frequencies: HashMap::new(),
        }
    }

    /// Set maximum number of features to select
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set minimum document frequency for feature inclusion
    pub fn min_df(mut self, min_df: usize) -> Self {
        self.min_df = Some(min_df);
        self
    }

    /// Set maximum document frequency for feature inclusion
    pub fn max_df(mut self, max_df: f64) -> Self {
        self.max_df = Some(max_df);
        self
    }

    /// Set n-gram range (min_n, max_n)
    pub fn ngram_range(mut self, ngram_range: (usize, usize)) -> Self {
        self.ngram_range = ngram_range;
        self
    }

    /// Set binary mode (presence/absence instead of counts)
    pub fn binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// Set whether to convert to lowercase
    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Set stop words to filter out
    pub fn stop_words(mut self, stop_words: Option<Vec<String>>) -> Self {
        self.stop_words = stop_words;
        self
    }

    /// Fit the vectorizer to a collection of documents
    pub fn fit_instance_method(&mut self, documents: &[String]) -> SklResult<()> {
        if documents.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty document collection".to_string(),
            ));
        }

        // Extract all n-grams and count document frequencies
        let mut term_doc_freq = HashMap::new();
        let total_docs = documents.len();

        for document in documents {
            let tokens = self.tokenize(document);
            let ngrams = self.extract_ngrams(&tokens);
            let unique_ngrams: std::collections::HashSet<_> = ngrams.into_iter().collect();

            for ngram in unique_ngrams {
                *term_doc_freq.entry(ngram).or_insert(0) += 1;
            }
        }

        // Filter terms based on document frequency constraints
        let mut filtered_terms = Vec::new();
        let min_df = self.min_df.unwrap_or(1);
        let max_df_count = if let Some(max_df) = self.max_df {
            if max_df <= 1.0 {
                (total_docs as f64 * max_df).ceil() as usize
            } else {
                max_df as usize
            }
        } else {
            total_docs
        };

        for (term, doc_freq) in &term_doc_freq {
            if *doc_freq >= min_df && *doc_freq <= max_df_count {
                if let Some(stop_words) = &self.stop_words {
                    if !stop_words.contains(term) {
                        filtered_terms.push(term.clone());
                    }
                } else {
                    filtered_terms.push(term.clone());
                }
            }
        }

        // Sort terms for consistent ordering
        filtered_terms.sort();

        // Limit features if specified
        if let Some(max_features) = self.max_features {
            if filtered_terms.len() > max_features {
                // Sort by document frequency and take most frequent terms
                filtered_terms.sort_by_key(|term| std::cmp::Reverse(term_doc_freq[term]));
                filtered_terms.truncate(max_features);
                filtered_terms.sort(); // Sort alphabetically again
            }
        }

        // Build vocabulary
        let mut vocabulary = HashMap::new();
        for (idx, term) in filtered_terms.iter().enumerate() {
            vocabulary.insert(term.clone(), idx);
        }

        self.vocabulary = Some(vocabulary);
        self.feature_names = filtered_terms;
        self.document_frequencies = term_doc_freq;

        Ok(())
    }

    /// Transform documents to feature vectors
    pub fn transform(&self, documents: &[String]) -> SklResult<Array2<Float>> {
        let vocabulary = self
            .vocabulary
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Vectorizer not fitted".to_string()))?;

        let n_docs = documents.len();
        let n_features = vocabulary.len();
        let mut feature_matrix = Array2::zeros((n_docs, n_features));

        for (doc_idx, document) in documents.iter().enumerate() {
            let tokens = self.tokenize(document);
            let ngrams = self.extract_ngrams(&tokens);

            // Count n-gram frequencies
            let mut term_counts = HashMap::new();
            for ngram in ngrams {
                if vocabulary.contains_key(&ngram) {
                    *term_counts.entry(ngram).or_insert(0) += 1;
                }
            }

            // Fill feature vector
            for (term, count) in term_counts {
                if let Some(&feature_idx) = vocabulary.get(&term) {
                    let value = if self.binary { 1.0 } else { count as Float };
                    feature_matrix[(doc_idx, feature_idx)] = value;
                }
            }
        }

        Ok(feature_matrix)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, documents: &[String]) -> SklResult<Array2<Float>> {
        self.fit_instance_method(documents)?;
        self.transform(documents)
    }

    /// Get feature names
    pub fn get_feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Get vocabulary
    pub fn get_vocabulary(&self) -> Option<&HashMap<String, usize>> {
        self.vocabulary.as_ref()
    }

    pub(crate) fn tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        // Simple whitespace and punctuation tokenization
        text.split_whitespace()
            .map(|token| {
                // Remove punctuation from ends
                token
                    .trim_matches(|c: char| c.is_ascii_punctuation())
                    .to_string()
            })
            .filter(|token| !token.is_empty())
            .collect()
    }

    pub(crate) fn extract_ngrams(&self, tokens: &[String]) -> Vec<String> {
        let mut ngrams = Vec::new();
        let (min_n, max_n) = self.ngram_range;

        for n in min_n..=max_n {
            if n <= tokens.len() {
                for i in 0..=tokens.len() - n {
                    let ngram = tokens[i..i + n].join(" ");
                    ngrams.push(ngram);
                }
            }
        }

        ngrams
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted CountVectorizer that can transform new documents
#[derive(Debug, Clone)]
pub struct FittedCountVectorizer {
    inner: CountVectorizer,
}

impl FittedCountVectorizer {
    /// Transform new documents using the fitted vocabulary
    pub fn transform(&self, documents: &[String]) -> SklResult<Array2<Float>> {
        self.inner.transform(documents)
    }

    /// Get feature names
    pub fn get_feature_names(&self) -> &[String] {
        self.inner.get_feature_names()
    }

    /// Get vocabulary
    pub fn get_vocabulary(&self) -> Option<&HashMap<String, usize>> {
        self.inner.get_vocabulary()
    }
}

impl Fit<Vec<String>, ()> for CountVectorizer {
    type Fitted = FittedCountVectorizer;

    fn fit(mut self, x: &Vec<String>, _y: &()) -> SklResult<Self::Fitted> {
        self.fit_instance_method(x)?;
        Ok(FittedCountVectorizer { inner: self })
    }
}

impl Transform<Vec<String>, Array2<Float>> for FittedCountVectorizer {
    fn transform(&self, x: &Vec<String>) -> SklResult<Array2<Float>> {
        self.transform(x)
    }
}

/// Complete TF-IDF Vectorizer implementation
///
/// Converts text documents to TF-IDF weighted feature vectors.
/// Combines term frequency (TF) with inverse document frequency (IDF).
#[derive(Debug, Clone)]
pub struct TfidfVectorizer {
    max_features: Option<usize>,
    min_df: Option<usize>,
    max_df: Option<f64>,
    ngram_range: (usize, usize),
    use_idf: bool,
    sublinear_tf: bool,
    smooth_idf: bool,
    norm: Option<String>,
    lowercase: bool,
    stop_words: Option<Vec<String>>,
    count_vectorizer: CountVectorizer,
    idf_weights: Vec<Float>,
}

impl TfidfVectorizer {
    /// Create a new TfidfVectorizer with default parameters
    pub fn new() -> Self {
        Self {
            max_features: None,
            min_df: Some(1),
            max_df: Some(1.0),
            ngram_range: (1, 1),
            use_idf: true,
            sublinear_tf: false,
            smooth_idf: true,
            norm: Some("l2".to_string()),
            lowercase: true,
            stop_words: None,
            count_vectorizer: CountVectorizer::new(),
            idf_weights: Vec::new(),
        }
    }

    /// Set maximum number of features
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self.count_vectorizer = self.count_vectorizer.max_features(max_features);
        self
    }

    /// Set minimum document frequency
    pub fn min_df(mut self, min_df: usize) -> Self {
        self.min_df = Some(min_df);
        self.count_vectorizer = self.count_vectorizer.min_df(min_df);
        self
    }

    /// Set maximum document frequency
    pub fn max_df(mut self, max_df: f64) -> Self {
        self.max_df = Some(max_df);
        self.count_vectorizer = self.count_vectorizer.max_df(max_df);
        self
    }

    /// Set n-gram range
    pub fn ngram_range(mut self, ngram_range: (usize, usize)) -> Self {
        self.ngram_range = ngram_range;
        self.count_vectorizer = self.count_vectorizer.ngram_range(ngram_range);
        self
    }

    /// Set whether to use IDF weighting
    pub fn use_idf(mut self, use_idf: bool) -> Self {
        self.use_idf = use_idf;
        self
    }

    /// Set whether to use sublinear TF scaling (1 + log(tf))
    pub fn sublinear_tf(mut self, sublinear_tf: bool) -> Self {
        self.sublinear_tf = sublinear_tf;
        self
    }

    /// Set whether to add one to IDF weights to prevent zero divisions
    pub fn smooth_idf(mut self, smooth_idf: bool) -> Self {
        self.smooth_idf = smooth_idf;
        self
    }

    /// Set normalization method ("l1", "l2", or None)
    pub fn norm(mut self, norm: Option<String>) -> Self {
        self.norm = norm;
        self
    }

    /// Set whether to convert to lowercase
    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self.count_vectorizer = self.count_vectorizer.lowercase(lowercase);
        self
    }

    /// Set stop words
    pub fn stop_words(mut self, stop_words: Option<Vec<String>>) -> Self {
        self.stop_words = stop_words.clone();
        self.count_vectorizer = self.count_vectorizer.stop_words(stop_words);
        self
    }

    /// Fit the vectorizer to documents
    pub fn fit_instance_method(&mut self, documents: &[String]) -> SklResult<()> {
        if documents.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty document collection".to_string(),
            ));
        }

        // Fit the count vectorizer
        self.count_vectorizer.fit_instance_method(documents)?;

        if self.use_idf {
            // Calculate IDF weights
            self.calculate_idf_weights(documents)?;
        }

        Ok(())
    }

    /// Transform documents to TF-IDF vectors
    pub fn transform(&self, documents: &[String]) -> SklResult<Array2<Float>> {
        // Get raw term frequencies
        let mut tf_matrix = self.count_vectorizer.transform(documents)?;

        // Apply sublinear TF scaling if enabled
        if self.sublinear_tf {
            tf_matrix.mapv_inplace(|x| if x > 0.0 { 1.0 + x.ln() } else { 0.0 });
        }

        // Apply IDF weighting if enabled
        if self.use_idf && !self.idf_weights.is_empty() {
            for mut row in tf_matrix.axis_iter_mut(Axis(0)) {
                for (i, &idf_weight) in self.idf_weights.iter().enumerate() {
                    row[i] *= idf_weight;
                }
            }
        }

        // Apply normalization if specified
        if let Some(norm_type) = &self.norm {
            self.normalize_rows(&mut tf_matrix, norm_type)?;
        }

        Ok(tf_matrix)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, documents: &[String]) -> SklResult<Array2<Float>> {
        self.fit_instance_method(documents)?;
        self.transform(documents)
    }

    /// Get feature names
    pub fn get_feature_names(&self) -> &[String] {
        self.count_vectorizer.get_feature_names()
    }

    /// Get vocabulary
    pub fn get_vocabulary(&self) -> Option<&HashMap<String, usize>> {
        self.count_vectorizer.get_vocabulary()
    }

    /// Get IDF weights
    pub fn get_idf_weights(&self) -> &[Float] {
        &self.idf_weights
    }

    fn calculate_idf_weights(&mut self, documents: &[String]) -> SklResult<()> {
        let vocabulary = self
            .count_vectorizer
            .get_vocabulary()
            .ok_or_else(|| SklearsError::InvalidInput("Count vectorizer not fitted".to_string()))?;

        let n_docs = documents.len() as Float;
        let mut idf_weights = vec![0.0; vocabulary.len()];

        // Count document frequencies for each term
        let mut doc_frequencies = vec![0; vocabulary.len()];

        for document in documents {
            let tokens = self.count_vectorizer.tokenize(document);
            let ngrams = self.count_vectorizer.extract_ngrams(&tokens);
            let unique_ngrams: std::collections::HashSet<_> = ngrams.into_iter().collect();

            for ngram in unique_ngrams {
                if let Some(&feature_idx) = vocabulary.get(&ngram) {
                    doc_frequencies[feature_idx] += 1;
                }
            }
        }

        // Calculate IDF weights
        for (i, &doc_freq) in doc_frequencies.iter().enumerate() {
            let df = doc_freq as Float;
            let idf = if self.smooth_idf {
                (n_docs / (1.0 + df)).ln() + 1.0
            } else {
                (n_docs / df).ln() + 1.0
            };
            idf_weights[i] = idf;
        }

        self.idf_weights = idf_weights;
        Ok(())
    }

    fn normalize_rows(&self, matrix: &mut Array2<Float>, norm_type: &str) -> SklResult<()> {
        for mut row in matrix.axis_iter_mut(Axis(0)) {
            let norm = match norm_type {
                "l1" => row.iter().map(|x| x.abs()).sum::<Float>(),
                "l2" => row.iter().map(|x| x * x).sum::<Float>().sqrt(),
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown norm type: {}",
                        norm_type
                    )))
                }
            };

            if norm > 0.0 {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }

        Ok(())
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted TfidfVectorizer that can transform new documents
#[derive(Debug, Clone)]
pub struct FittedTfidfVectorizer {
    inner: TfidfVectorizer,
}

impl FittedTfidfVectorizer {
    /// Transform new documents using the fitted vectorizer
    pub fn transform(&self, documents: &[String]) -> SklResult<Array2<Float>> {
        self.inner.transform(documents)
    }

    /// Get feature names
    pub fn get_feature_names(&self) -> &[String] {
        self.inner.get_feature_names()
    }

    /// Get vocabulary
    pub fn get_vocabulary(&self) -> Option<&HashMap<String, usize>> {
        self.inner.get_vocabulary()
    }

    /// Get IDF weights
    pub fn get_idf_weights(&self) -> &[Float] {
        self.inner.get_idf_weights()
    }
}

impl Fit<Vec<String>, ()> for TfidfVectorizer {
    type Fitted = FittedTfidfVectorizer;

    fn fit(mut self, x: &Vec<String>, _y: &()) -> SklResult<Self::Fitted> {
        self.fit_instance_method(x)?;
        Ok(FittedTfidfVectorizer { inner: self })
    }
}

impl Transform<Vec<String>, Array2<Float>> for FittedTfidfVectorizer {
    fn transform(&self, x: &Vec<String>) -> SklResult<Array2<Float>> {
        self.inner.transform(x)
    }
}

#[derive(Debug, Clone)]
pub struct HashingVectorizer {
    n_features: usize,
    ngram_range: (usize, usize),
}

impl HashingVectorizer {
    pub fn new() -> Self {
        Self {
            n_features: 1024,
            ngram_range: (1, 1),
        }
    }

    pub fn n_features(mut self, n_features: usize) -> Self {
        self.n_features = n_features;
        self
    }
}

impl Default for HashingVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Sentiment Analysis extractor for text
///
/// Provides simple rule-based sentiment analysis with lexicon-based scoring
/// and configurable sentiment categories.
#[derive(Debug, Clone)]
pub struct SentimentAnalyzer {
    positive_words: std::collections::HashSet<String>,
    negative_words: std::collections::HashSet<String>,
    neutral_threshold: f64,
    case_sensitive: bool,
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer with default lexicon
    pub fn new() -> Self {
        let mut analyzer = Self {
            positive_words: std::collections::HashSet::new(),
            negative_words: std::collections::HashSet::new(),
            neutral_threshold: 0.1,
            case_sensitive: false,
        };

        // Add basic positive words
        for word in &[
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "awesome",
            "love",
            "like",
            "enjoy",
            "happy",
            "pleased",
            "satisfied",
            "perfect",
            "best",
            "brilliant",
            "outstanding",
            "superb",
            "marvelous",
            "incredible",
        ] {
            analyzer.positive_words.insert(word.to_string());
        }

        // Add basic negative words
        for word in &[
            "bad",
            "terrible",
            "awful",
            "horrible",
            "disgusting",
            "hate",
            "dislike",
            "disappointed",
            "frustrated",
            "angry",
            "sad",
            "worst",
            "pathetic",
            "useless",
            "annoying",
            "boring",
            "stupid",
            "ridiculous",
            "poor",
        ] {
            analyzer.negative_words.insert(word.to_string());
        }

        analyzer
    }

    /// Set neutral threshold (sentiment scores within [-threshold, threshold] are neutral)
    pub fn neutral_threshold(mut self, threshold: f64) -> Self {
        self.neutral_threshold = threshold;
        self
    }

    /// Set case sensitivity
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Add custom positive words to the lexicon
    pub fn add_positive_words(mut self, words: Vec<String>) -> Self {
        for word in words {
            self.positive_words.insert(word);
        }
        self
    }

    /// Add custom negative words to the lexicon
    pub fn add_negative_words(mut self, words: Vec<String>) -> Self {
        for word in words {
            self.negative_words.insert(word);
        }
        self
    }

    /// Analyze sentiment of a single text
    pub fn analyze_sentiment(&self, text: &str) -> SentimentResult {
        let text = if self.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut positive_count = 0;
        let mut negative_count = 0;

        for word in &words {
            // Remove punctuation
            let clean_word = word.trim_matches(|c: char| c.is_ascii_punctuation());

            if self.positive_words.contains(clean_word) {
                positive_count += 1;
            } else if self.negative_words.contains(clean_word) {
                negative_count += 1;
            }
        }

        let total_sentiment_words = positive_count + negative_count;
        let score = if total_sentiment_words == 0 {
            0.0
        } else {
            (positive_count as f64 - negative_count as f64) / total_sentiment_words as f64
        };

        let polarity = if score.abs() <= self.neutral_threshold {
            SentimentPolarity::Neutral
        } else if score > 0.0 {
            SentimentPolarity::Positive
        } else {
            SentimentPolarity::Negative
        };

        // SentimentResult
        SentimentResult {
            polarity,
            score,
            positive_words: positive_count,
            negative_words: negative_count,
            total_words: words.len(),
        }
    }

    /// Extract sentiment features from multiple documents
    pub fn extract_features(&self, documents: &[String]) -> SklResult<Array2<Float>> {
        if documents.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty document collection".to_string(),
            ));
        }

        let mut features = Array2::zeros((documents.len(), 5));

        for (i, document) in documents.iter().enumerate() {
            let sentiment = self.analyze_sentiment(document);

            // Feature vector: [score, positive_ratio, negative_ratio, sentiment_density, polarity_encoded]
            features[(i, 0)] = sentiment.score;
            features[(i, 1)] =
                sentiment.positive_words as f64 / sentiment.total_words.max(1) as f64;
            features[(i, 2)] =
                sentiment.negative_words as f64 / sentiment.total_words.max(1) as f64;
            features[(i, 3)] = (sentiment.positive_words + sentiment.negative_words) as f64
                / sentiment.total_words.max(1) as f64;
            features[(i, 4)] = match sentiment.polarity {
                SentimentPolarity::Negative => -1.0,
                SentimentPolarity::Neutral => 0.0,
                SentimentPolarity::Positive => 1.0,
            };
        }

        Ok(features)
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Sentiment analysis result
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// polarity
    pub polarity: SentimentPolarity,
    /// score
    pub score: f64,
    /// positive_words
    pub positive_words: usize,
    /// negative_words
    pub negative_words: usize,
    /// total_words
    pub total_words: usize,
}

/// Sentiment polarity classification
#[derive(Debug, Clone, PartialEq)]
pub enum SentimentPolarity {
    /// Positive
    Positive,
    /// Negative
    Negative,
    /// Neutral
    Neutral,
}

/// Memory-efficient streaming text processor
///
/// Processes large text datasets in chunks to minimize memory usage
/// while maintaining feature extraction quality.
#[derive(Debug, Clone)]
pub struct StreamingTextProcessor {
    chunk_size: usize,
    overlap_size: usize,
    min_chunk_words: usize,
}

impl StreamingTextProcessor {
    /// Create a new streaming text processor
    pub fn new() -> Self {
        Self {
            chunk_size: 10000, // characters per chunk
            overlap_size: 1000,
            min_chunk_words: 50,
        }
    }

    /// Set chunk size in characters
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set overlap size between chunks
    pub fn overlap_size(mut self, overlap_size: usize) -> Self {
        self.overlap_size = overlap_size;
        self
    }

    /// Set minimum words per chunk
    pub fn min_chunk_words(mut self, min_words: usize) -> Self {
        self.min_chunk_words = min_words;
        self
    }

    /// Process large text with a streaming vectorizer
    pub fn stream_process_with_count_vectorizer(
        &self,
        large_text: &str,
        vectorizer: &mut CountVectorizer,
    ) -> SklResult<Array2<Float>> {
        let chunks = self.create_chunks(large_text);

        if chunks.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid chunks created".to_string(),
            ));
        }

        // First pass: fit vocabulary on a sample of chunks
        let sample_size = (chunks.len() / 4).max(1);
        let sample_chunks: Vec<String> = chunks
            .iter()
            .step_by(chunks.len() / sample_size)
            .take(sample_size)
            .cloned()
            .collect();

        vectorizer.fit_instance_method(&sample_chunks)?;

        // Second pass: transform all chunks
        let chunk_features = vectorizer.transform(&chunks)?;

        // Aggregate features (e.g., mean across chunks)
        let (n_chunks, n_features) = chunk_features.dim();
        let mut aggregated = Array2::zeros((1, n_features));

        for j in 0..n_features {
            let mut sum = 0.0;
            for i in 0..n_chunks {
                sum += chunk_features[(i, j)];
            }
            aggregated[(0, j)] = sum / n_chunks as f64;
        }

        Ok(aggregated)
    }

    /// Process large text with streaming TF-IDF
    pub fn stream_process_with_tfidf(
        &self,
        large_text: &str,
        vectorizer: &mut TfidfVectorizer,
    ) -> SklResult<Array2<Float>> {
        let chunks = self.create_chunks(large_text);

        if chunks.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid chunks created".to_string(),
            ));
        }

        // Fit and transform in streaming fashion
        vectorizer.fit_instance_method(&chunks)?;
        let chunk_features = vectorizer.transform(&chunks)?;

        // Aggregate using weighted average based on chunk size
        let (n_chunks, n_features) = chunk_features.dim();
        let mut aggregated = Array2::zeros((1, n_features));
        let chunk_weights: Vec<f64> = chunks
            .iter()
            .map(|chunk| chunk.split_whitespace().count() as f64)
            .collect();
        let total_weight: f64 = chunk_weights.iter().sum();

        for j in 0..n_features {
            let mut weighted_sum = 0.0;
            for i in 0..n_chunks {
                weighted_sum += chunk_features[(i, j)] * chunk_weights[i];
            }
            aggregated[(0, j)] = weighted_sum / total_weight;
        }

        Ok(aggregated)
    }

    /// Extract memory-efficient statistical features
    pub fn extract_streaming_stats(&self, large_text: &str) -> SklResult<Array1<Float>> {
        let chunks = self.create_chunks(large_text);

        if chunks.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid chunks created".to_string(),
            ));
        }

        let mut total_chars = 0;
        let mut total_words = 0;
        let mut total_sentences = 0;
        let mut word_lengths = Vec::new();

        for chunk in &chunks {
            total_chars += chunk.len();
            let words: Vec<&str> = chunk.split_whitespace().collect();
            total_words += words.len();

            // Count sentences (approximate)
            total_sentences += chunk.matches('.').count()
                + chunk.matches('!').count()
                + chunk.matches('?').count();

            // Collect word lengths for statistics
            for word in words {
                let clean_word = word.trim_matches(|c: char| c.is_ascii_punctuation());
                if !clean_word.is_empty() {
                    word_lengths.push(clean_word.len() as f64);
                }
            }
        }

        // Calculate statistics
        let avg_word_length = if !word_lengths.is_empty() {
            word_lengths.iter().sum::<f64>() / word_lengths.len() as f64
        } else {
            0.0
        };

        let avg_words_per_sentence = if total_sentences > 0 {
            total_words as f64 / total_sentences as f64
        } else {
            0.0
        };

        let features = vec![
            total_chars as f64,
            total_words as f64,
            total_sentences as f64,
            avg_word_length,
            avg_words_per_sentence,
            total_chars as f64 / total_words.max(1) as f64, // avg chars per word
        ];

        Ok(Array1::from_vec(features))
    }

    fn create_chunks(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = (start + self.chunk_size).min(text.len());
            let chunk = &text[start..end];

            // Try to break at word boundary
            let final_chunk = if end < text.len() {
                if let Some(last_space) = chunk.rfind(' ') {
                    &chunk[..last_space]
                } else {
                    chunk
                }
            } else {
                chunk
            };

            // Only add chunk if it has enough words
            if final_chunk.split_whitespace().count() >= self.min_chunk_words {
                chunks.push(final_chunk.to_string());
            }

            start += final_chunk.len().saturating_sub(self.overlap_size);
            if start >= text.len() {
                break;
            }
        }

        chunks
    }
}

impl Default for StreamingTextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// LSA (Latent Semantic Analysis) placeholder
#[derive(Debug, Clone)]
pub struct LSA {
    n_components: usize,
}

impl LSA {
    pub fn new() -> Self {
        Self { n_components: 100 }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn algorithm(self, _algorithm: &str) -> Self {
        // Placeholder - algorithm selection not implemented
        self
    }

    pub fn fit(&self, _X: &ArrayView2<f64>) -> SklResult<LSATrained> {
        // Placeholder implementation
        Ok(LSATrained {
            components: Array2::zeros((self.n_components, 10)),
        })
    }
}

#[derive(Debug, Clone)]
pub struct LSATrained {
    components: Array2<f64>,
}

impl Default for LSA {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Emotion Detection
// ============================================================================

/// Emotion types for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmotionType {
    /// Joy, happiness, pleasure
    Joy,
    /// Sadness, sorrow, grief
    Sadness,
    /// Anger, rage, frustration
    Anger,
    /// Fear, anxiety, worry
    Fear,
    /// Surprise, amazement
    Surprise,
    /// Disgust, revulsion
    Disgust,
    /// Neutral, no strong emotion
    Neutral,
}

impl EmotionType {
    /// Get all emotion types (excluding Neutral)
    pub fn all_emotions() -> Vec<EmotionType> {
        vec![
            EmotionType::Joy,
            EmotionType::Sadness,
            EmotionType::Anger,
            EmotionType::Fear,
            EmotionType::Surprise,
            EmotionType::Disgust,
        ]
    }

    /// Get emotion name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            EmotionType::Joy => "joy",
            EmotionType::Sadness => "sadness",
            EmotionType::Anger => "anger",
            EmotionType::Fear => "fear",
            EmotionType::Surprise => "surprise",
            EmotionType::Disgust => "disgust",
            EmotionType::Neutral => "neutral",
        }
    }
}

/// Result of emotion detection for a single text
#[derive(Debug, Clone)]
pub struct EmotionResult {
    /// Primary detected emotion
    pub primary_emotion: EmotionType,
    /// Confidence score for primary emotion [0.0, 1.0]
    pub confidence: f64,
    /// Scores for all emotion types
    pub emotion_scores: std::collections::HashMap<EmotionType, f64>,
    /// Total emotion words detected
    pub total_emotion_words: usize,
    /// Total words in text
    pub total_words: usize,
}

impl EmotionResult {
    /// Get emotion intensity (normalized score)
    pub fn intensity(&self) -> f64 {
        if self.total_words == 0 {
            0.0
        } else {
            self.total_emotion_words as f64 / self.total_words as f64
        }
    }

    /// Get secondary emotion (second highest score)
    pub fn secondary_emotion(&self) -> Option<(EmotionType, f64)> {
        let mut sorted: Vec<_> = self
            .emotion_scores
            .iter()
            .filter(|(e, _)| **e != self.primary_emotion)
            .map(|(e, s)| (*e, *s))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.first().copied()
    }
}

/// Emotion Detection extractor for text
///
/// Provides multi-class emotion detection using lexicon-based approach
/// with support for detecting joy, sadness, anger, fear, surprise, and disgust.
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::text::EmotionDetector;
///
/// let detector = EmotionDetector::new();
/// let result = detector.detect_emotion("I am so happy and excited!");
/// println!("Primary emotion: {:?}", result.primary_emotion);
/// ```
#[derive(Debug, Clone)]
pub struct EmotionDetector {
    emotion_lexicons: std::collections::HashMap<EmotionType, std::collections::HashSet<String>>,
    case_sensitive: bool,
    min_confidence: f64,
    intensity_weight: f64,
}

impl EmotionDetector {
    /// Create a new emotion detector with default lexicons
    pub fn new() -> Self {
        let mut detector = Self {
            emotion_lexicons: std::collections::HashMap::new(),
            case_sensitive: false,
            min_confidence: 0.1,
            intensity_weight: 0.5,
        };

        // Initialize emotion lexicons
        detector.initialize_lexicons();
        detector
    }

    /// Initialize default emotion word lexicons
    fn initialize_lexicons(&mut self) {
        // Joy words
        let joy_words = vec![
            "happy",
            "joy",
            "joyful",
            "cheerful",
            "delighted",
            "pleased",
            "glad",
            "excited",
            "thrilled",
            "ecstatic",
            "elated",
            "wonderful",
            "amazing",
            "fantastic",
            "great",
            "excellent",
            "love",
            "loving",
            "loved",
            "enjoy",
            "fun",
            "amusing",
            "hilarious",
            "laugh",
            "laughter",
            "smile",
            "smiling",
            "blessed",
            "grateful",
            "thankful",
            "celebration",
            "celebrate",
            "victory",
        ];
        self.add_emotion_words(EmotionType::Joy, joy_words);

        // Sadness words
        let sadness_words = vec![
            "sad",
            "unhappy",
            "depressed",
            "miserable",
            "sorrowful",
            "gloomy",
            "melancholy",
            "dejected",
            "downcast",
            "heartbroken",
            "crying",
            "cry",
            "tears",
            "weeping",
            "grief",
            "grieving",
            "mourning",
            "loss",
            "lost",
            "lonely",
            "loneliness",
            "alone",
            "isolated",
            "hopeless",
            "despair",
            "disappointed",
            "regret",
            "sorry",
            "hurt",
            "pain",
            "painful",
        ];
        self.add_emotion_words(EmotionType::Sadness, sadness_words);

        // Anger words
        let anger_words = vec![
            "angry",
            "mad",
            "furious",
            "enraged",
            "outraged",
            "livid",
            "irate",
            "annoyed",
            "irritated",
            "frustrated",
            "rage",
            "fury",
            "wrath",
            "hatred",
            "hate",
            "hating",
            "hostile",
            "aggravated",
            "infuriated",
            "resentful",
            "bitter",
            "indignant",
            "offensive",
            "insulting",
            "provoked",
            "incensed",
            "exasperated",
            "disgusted",
            "contempt",
        ];
        self.add_emotion_words(EmotionType::Anger, anger_words);

        // Fear words
        let fear_words = vec![
            "afraid",
            "scared",
            "fearful",
            "frightened",
            "terrified",
            "horrified",
            "panic",
            "panicked",
            "anxious",
            "anxiety",
            "worried",
            "nervous",
            "uneasy",
            "alarmed",
            "threatened",
            "intimidated",
            "dread",
            "dreading",
            "apprehensive",
            "tense",
            "stressed",
            "stress",
            "concern",
            "concerned",
            "phobia",
            "terror",
            "horror",
            "nightmare",
            "shock",
            "shocked",
        ];
        self.add_emotion_words(EmotionType::Fear, fear_words);

        // Surprise words
        let surprise_words = vec![
            "surprised",
            "amazed",
            "astonished",
            "astounded",
            "shocked",
            "stunned",
            "startled",
            "unexpected",
            "sudden",
            "wow",
            "incredible",
            "unbelievable",
            "remarkable",
            "extraordinary",
            "spectacular",
            "breathtaking",
            "dumbfounded",
            "flabbergasted",
            "bewildered",
            "confused",
            "puzzled",
            "wonder",
            "wondering",
            "curious",
            "odd",
            "strange",
            "unusual",
        ];
        self.add_emotion_words(EmotionType::Surprise, surprise_words);

        // Disgust words
        let disgust_words = vec![
            "disgusted",
            "disgusting",
            "revolting",
            "repulsive",
            "repugnant",
            "nauseating",
            "sickening",
            "gross",
            "yuck",
            "nasty",
            "foul",
            "vile",
            "loathsome",
            "offensive",
            "appalling",
            "abhorrent",
            "detestable",
            "horrible",
            "terrible",
            "awful",
            "unpleasant",
            "distasteful",
            "repellent",
            "obnoxious",
            "hideous",
            "ugly",
            "grotesque",
        ];
        self.add_emotion_words(EmotionType::Disgust, disgust_words);
    }

    /// Add words to an emotion lexicon
    fn add_emotion_words(&mut self, emotion: EmotionType, words: Vec<&str>) {
        let word_set = self
            .emotion_lexicons
            .entry(emotion)
            .or_insert_with(std::collections::HashSet::new);

        for word in words {
            word_set.insert(word.to_string());
        }
    }

    /// Set case sensitivity
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Set minimum confidence threshold for emotion detection
    pub fn min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence.max(0.0).min(1.0);
        self
    }

    /// Set intensity weight (how much to weight emotion word density vs raw counts)
    pub fn intensity_weight(mut self, weight: f64) -> Self {
        self.intensity_weight = weight.max(0.0).min(1.0);
        self
    }

    /// Add custom words to an emotion lexicon
    pub fn add_custom_emotion_words(mut self, emotion: EmotionType, words: Vec<String>) -> Self {
        let word_set = self
            .emotion_lexicons
            .entry(emotion)
            .or_insert_with(std::collections::HashSet::new);

        for word in words {
            word_set.insert(word);
        }
        self
    }

    /// Detect emotion in a single text
    pub fn detect_emotion(&self, text: &str) -> EmotionResult {
        let text = if self.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };

        let words: Vec<&str> = text.split_whitespace().collect();
        let total_words = words.len();

        // Count emotion words for each emotion type
        let mut emotion_counts: std::collections::HashMap<EmotionType, usize> =
            std::collections::HashMap::new();
        let mut total_emotion_words = 0;

        for word in &words {
            // Remove punctuation
            let clean_word = word.trim_matches(|c: char| c.is_ascii_punctuation());

            // Check each emotion lexicon
            for (emotion, lexicon) in &self.emotion_lexicons {
                if lexicon.contains(clean_word) {
                    *emotion_counts.entry(*emotion).or_insert(0) += 1;
                    total_emotion_words += 1;
                }
            }
        }

        // Calculate emotion scores
        let mut emotion_scores: std::collections::HashMap<EmotionType, f64> =
            std::collections::HashMap::new();

        for emotion in EmotionType::all_emotions() {
            let count = *emotion_counts.get(&emotion).unwrap_or(&0) as f64;
            let intensity = if total_words > 0 {
                count / total_words as f64
            } else {
                0.0
            };

            // Combine raw count and intensity based on intensity_weight
            let score = if total_emotion_words > 0 {
                (1.0 - self.intensity_weight) * (count / total_emotion_words as f64)
                    + self.intensity_weight * intensity
            } else {
                0.0
            };

            emotion_scores.insert(emotion, score);
        }

        // Determine primary emotion
        let (primary_emotion, confidence) = if total_emotion_words == 0 {
            (EmotionType::Neutral, 0.0)
        } else {
            let mut sorted: Vec<_> = emotion_scores.iter().map(|(e, s)| (*e, *s)).collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let (emotion, score) = sorted[0];
            if score >= self.min_confidence {
                (emotion, score)
            } else {
                (EmotionType::Neutral, score)
            }
        };

        EmotionResult {
            primary_emotion,
            confidence,
            emotion_scores,
            total_emotion_words,
            total_words,
        }
    }

    /// Extract emotion features from multiple documents
    ///
    /// Returns a feature matrix with shape (n_documents, 14) containing:
    /// - Scores for each emotion (6 features: joy, sadness, anger, fear, surprise, disgust)
    /// - Primary emotion one-hot encoding (6 features)
    /// - Confidence score (1 feature)
    /// - Emotion intensity (1 feature)
    pub fn extract_features(&self, documents: &[String]) -> SklResult<Array2<Float>> {
        if documents.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty document collection".to_string(),
            ));
        }

        // Feature vector: [joy_score, sadness_score, anger_score, fear_score, surprise_score, disgust_score,
        //                  joy_onehot, sadness_onehot, anger_onehot, fear_onehot, surprise_onehot, disgust_onehot,
        //                  confidence, intensity]
        let mut features = Array2::zeros((documents.len(), 14));

        for (i, document) in documents.iter().enumerate() {
            let result = self.detect_emotion(document);

            // Emotion scores (0-5)
            features[(i, 0)] = *result.emotion_scores.get(&EmotionType::Joy).unwrap_or(&0.0);
            features[(i, 1)] = *result
                .emotion_scores
                .get(&EmotionType::Sadness)
                .unwrap_or(&0.0);
            features[(i, 2)] = *result
                .emotion_scores
                .get(&EmotionType::Anger)
                .unwrap_or(&0.0);
            features[(i, 3)] = *result
                .emotion_scores
                .get(&EmotionType::Fear)
                .unwrap_or(&0.0);
            features[(i, 4)] = *result
                .emotion_scores
                .get(&EmotionType::Surprise)
                .unwrap_or(&0.0);
            features[(i, 5)] = *result
                .emotion_scores
                .get(&EmotionType::Disgust)
                .unwrap_or(&0.0);

            // Primary emotion one-hot encoding (6-11)
            let primary_idx = match result.primary_emotion {
                EmotionType::Joy => 6,
                EmotionType::Sadness => 7,
                EmotionType::Anger => 8,
                EmotionType::Fear => 9,
                EmotionType::Surprise => 10,
                EmotionType::Disgust => 11,
                EmotionType::Neutral => continue, // Don't encode neutral
            };
            features[(i, primary_idx)] = 1.0;

            // Confidence and intensity (12-13)
            features[(i, 12)] = result.confidence;
            features[(i, 13)] = result.intensity();
        }

        Ok(features)
    }

    /// Analyze emotion distribution across multiple documents
    pub fn analyze_distribution(
        &self,
        documents: &[String],
    ) -> std::collections::HashMap<EmotionType, usize> {
        let mut distribution: std::collections::HashMap<EmotionType, usize> =
            std::collections::HashMap::new();

        for document in documents {
            let result = self.detect_emotion(document);
            *distribution.entry(result.primary_emotion).or_insert(0) += 1;
        }

        distribution
    }
}

impl Default for EmotionDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Aspect-Based Sentiment Analysis
// ============================================================================

/// Aspect sentiment pair representing sentiment towards a specific aspect
#[derive(Debug, Clone, PartialEq)]
pub struct AspectSentiment {
    /// The aspect or feature being discussed
    pub aspect: String,
    /// Sentiment polarity for this aspect
    pub sentiment: SentimentPolarity,
    /// Sentiment score for this aspect [-1.0, 1.0]
    pub score: f64,
    /// Confidence in the aspect-sentiment pairing [0.0, 1.0]
    pub confidence: f64,
    /// Opinion words associated with this aspect
    pub opinion_words: Vec<String>,
}

/// Aspect-Based Sentiment Analysis extractor
///
/// Analyzes sentiment towards specific aspects or features mentioned in text.
/// This is particularly useful for product reviews, customer feedback, and opinion mining.
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::text::AspectBasedSentimentAnalyzer;
///
/// let analyzer = AspectBasedSentimentAnalyzer::new();
/// let result = analyzer.analyze("The food was great but the service was terrible.");
/// println!("Found {} aspects with sentiment", result.len());
/// ```
#[derive(Debug, Clone)]
pub struct AspectBasedSentimentAnalyzer {
    /// Predefined aspects to look for
    aspects: std::collections::HashSet<String>,
    /// Sentiment analyzer for sentiment detection
    sentiment_analyzer: SentimentAnalyzer,
    /// Context window size (words before and after aspect)
    context_window: usize,
    /// Minimum confidence threshold for aspect-sentiment pairs
    min_confidence: f64,
    /// Case sensitivity
    case_sensitive: bool,
    /// Use automatic aspect extraction
    auto_extract_aspects: bool,
}

impl AspectBasedSentimentAnalyzer {
    /// Create a new aspect-based sentiment analyzer with default settings
    pub fn new() -> Self {
        Self {
            aspects: std::collections::HashSet::new(),
            sentiment_analyzer: SentimentAnalyzer::new(),
            context_window: 3,
            min_confidence: 0.1,
            case_sensitive: false,
            auto_extract_aspects: true,
        }
    }

    /// Add predefined aspects to look for
    pub fn add_aspects(mut self, aspects: Vec<String>) -> Self {
        for aspect in aspects {
            self.aspects.insert(if self.case_sensitive {
                aspect
            } else {
                aspect.to_lowercase()
            });
        }
        self
    }

    /// Set context window size (number of words before and after aspect)
    pub fn context_window(mut self, window: usize) -> Self {
        self.context_window = window;
        self
    }

    /// Set minimum confidence threshold
    pub fn min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence.max(0.0).min(1.0);
        self
    }

    /// Set case sensitivity
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Enable/disable automatic aspect extraction
    pub fn auto_extract_aspects(mut self, auto_extract: bool) -> Self {
        self.auto_extract_aspects = auto_extract;
        self
    }

    /// Set custom sentiment analyzer
    pub fn sentiment_analyzer(mut self, analyzer: SentimentAnalyzer) -> Self {
        self.sentiment_analyzer = analyzer;
        self
    }

    /// Analyze aspect-based sentiment in text
    pub fn analyze(&self, text: &str) -> Vec<AspectSentiment> {
        let normalized_text = if self.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };

        let words: Vec<&str> = normalized_text.split_whitespace().collect();
        let mut aspect_sentiments = Vec::new();

        // Find all aspect occurrences
        let mut aspect_positions: Vec<(usize, String)> = Vec::new();

        // Look for predefined aspects
        if !self.aspects.is_empty() {
            for (i, word) in words.iter().enumerate() {
                let clean_word = word.trim_matches(|c: char| c.is_ascii_punctuation());

                // Check for single-word aspects
                if self.aspects.contains(clean_word) {
                    aspect_positions.push((i, clean_word.to_string()));
                }

                // Check for multi-word aspects (up to 3 words)
                for window_size in 2..=3 {
                    if i + window_size <= words.len() {
                        let multi_word = words[i..i + window_size]
                            .iter()
                            .map(|w| w.trim_matches(|c: char| c.is_ascii_punctuation()))
                            .collect::<Vec<_>>()
                            .join(" ");

                        if self.aspects.contains(&multi_word) {
                            aspect_positions.push((i, multi_word));
                        }
                    }
                }
            }
        }

        // Automatic aspect extraction using noun detection heuristics
        if self.auto_extract_aspects && self.aspects.is_empty() {
            aspect_positions.extend(self.extract_aspects_auto(&words));
        }

        // Analyze sentiment for each aspect
        for (position, aspect) in aspect_positions {
            let (context_words, opinion_words) = self.get_context(position, &words);

            // Analyze sentiment of context
            let context_text = context_words.join(" ");
            let sentiment_result = self.sentiment_analyzer.analyze_sentiment(&context_text);

            // Calculate confidence based on sentiment word density in context
            let confidence = if context_words.is_empty() {
                0.0
            } else {
                (sentiment_result.positive_words + sentiment_result.negative_words) as f64
                    / context_words.len() as f64
            };

            if confidence >= self.min_confidence {
                aspect_sentiments.push(AspectSentiment {
                    aspect,
                    sentiment: sentiment_result.polarity,
                    score: sentiment_result.score,
                    confidence,
                    opinion_words,
                });
            }
        }

        aspect_sentiments
    }

    /// Extract context words around an aspect
    fn get_context(&self, position: usize, words: &[&str]) -> (Vec<String>, Vec<String>) {
        let start = position.saturating_sub(self.context_window);
        let end = (position + self.context_window + 1).min(words.len());

        let context_words: Vec<String> = words[start..end]
            .iter()
            .map(|w| {
                w.trim_matches(|c: char| c.is_ascii_punctuation())
                    .to_string()
            })
            .collect();

        // Extract opinion words (words that express sentiment)
        let opinion_words: Vec<String> = context_words
            .iter()
            .filter(|w| {
                self.sentiment_analyzer.positive_words.contains(w.as_str())
                    || self.sentiment_analyzer.negative_words.contains(w.as_str())
            })
            .cloned()
            .collect();

        (context_words, opinion_words)
    }

    /// Automatically extract aspect candidates using heuristics
    fn extract_aspects_auto(&self, words: &[&str]) -> Vec<(usize, String)> {
        let mut aspects = Vec::new();

        // Common aspect indicators (nouns that often represent aspects)
        let aspect_indicators = [
            "food",
            "service",
            "price",
            "quality",
            "staff",
            "location",
            "room",
            "product",
            "delivery",
            "support",
            "performance",
            "design",
            "battery",
            "screen",
            "camera",
            "sound",
            "comfort",
            "size",
            "color",
            "taste",
            "menu",
            "atmosphere",
            "experience",
            "value",
            "selection",
        ];

        for (i, word) in words.iter().enumerate() {
            let clean_word = word.trim_matches(|c: char| c.is_ascii_punctuation());

            // Check if word matches aspect indicators
            if aspect_indicators.contains(&clean_word) {
                aspects.push((i, clean_word.to_string()));
            }
        }

        aspects
    }

    /// Extract features from multiple documents
    ///
    /// Returns a feature matrix with aggregated aspect sentiments.
    /// Features include: [avg_aspect_score, positive_aspects, negative_aspects,
    ///                    neutral_aspects, aspect_diversity, avg_confidence]
    pub fn extract_features(&self, documents: &[String]) -> SklResult<Array2<Float>> {
        if documents.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty document collection".to_string(),
            ));
        }

        let mut features = Array2::zeros((documents.len(), 6));

        for (i, document) in documents.iter().enumerate() {
            let aspect_sentiments = self.analyze(document);

            if aspect_sentiments.is_empty() {
                continue;
            }

            // Compute features
            let avg_score = aspect_sentiments.iter().map(|a| a.score).sum::<f64>()
                / aspect_sentiments.len() as f64;

            let positive_count = aspect_sentiments
                .iter()
                .filter(|a| a.sentiment == SentimentPolarity::Positive)
                .count() as f64;

            let negative_count = aspect_sentiments
                .iter()
                .filter(|a| a.sentiment == SentimentPolarity::Negative)
                .count() as f64;

            let neutral_count = aspect_sentiments
                .iter()
                .filter(|a| a.sentiment == SentimentPolarity::Neutral)
                .count() as f64;

            // Aspect diversity (unique aspects / total aspects)
            let unique_aspects: std::collections::HashSet<_> =
                aspect_sentiments.iter().map(|a| &a.aspect).collect();
            let diversity = unique_aspects.len() as f64 / aspect_sentiments.len() as f64;

            let avg_confidence = aspect_sentiments.iter().map(|a| a.confidence).sum::<f64>()
                / aspect_sentiments.len() as f64;

            features[(i, 0)] = avg_score;
            features[(i, 1)] = positive_count;
            features[(i, 2)] = negative_count;
            features[(i, 3)] = neutral_count;
            features[(i, 4)] = diversity;
            features[(i, 5)] = avg_confidence;
        }

        Ok(features)
    }

    /// Aggregate aspect sentiments across multiple documents
    pub fn aggregate_aspects(
        &self,
        documents: &[String],
    ) -> std::collections::HashMap<String, Vec<AspectSentiment>> {
        let mut aggregated: std::collections::HashMap<String, Vec<AspectSentiment>> =
            std::collections::HashMap::new();

        for document in documents {
            let sentiments = self.analyze(document);
            for sentiment in sentiments {
                aggregated
                    .entry(sentiment.aspect.clone())
                    .or_insert_with(Vec::new)
                    .push(sentiment);
            }
        }

        aggregated
    }

    /// Get summary statistics for each aspect
    pub fn aspect_summary(&self, documents: &[String]) -> Vec<(String, f64, usize)> {
        let aggregated = self.aggregate_aspects(documents);

        let mut summary: Vec<(String, f64, usize)> = aggregated
            .iter()
            .map(|(aspect, sentiments)| {
                let avg_score =
                    sentiments.iter().map(|s| s.score).sum::<f64>() / sentiments.len() as f64;
                let count = sentiments.len();
                (aspect.clone(), avg_score, count)
            })
            .collect();

        // Sort by frequency
        summary.sort_by(|a, b| b.2.cmp(&a.2));
        summary
    }
}

impl Default for AspectBasedSentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

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
pub mod embeddings;
pub mod preprocessing;

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

        /// SentimentResult
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

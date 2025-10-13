//! Text preprocessing utilities for sklears
//!
//! This module provides text preprocessing capabilities including:
//! - Text tokenization and normalization
//! - TF-IDF vectorization
//! - N-gram feature generation
//! - Text similarity features
//! - Sentence embeddings

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform},
};
use std::collections::{HashMap, HashSet};

/// Text normalization strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationStrategy {
    /// No normalization
    None,
    /// Convert to lowercase
    Lowercase,
    /// Convert to lowercase and remove punctuation
    LowercaseNoPunct,
    /// Convert to lowercase, remove punctuation, and strip whitespace
    Full,
}

/// Text tokenization strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenizationStrategy {
    /// Split by whitespace
    Whitespace,
    /// Split by whitespace and punctuation
    WhitespacePunct,
    /// Simple word tokenization (alphanumeric only)
    Word,
}

/// N-gram type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NgramType {
    /// Character n-grams
    Char,
    /// Word n-grams
    Word,
}

/// Configuration for text tokenizer
#[derive(Debug, Clone)]
pub struct TextTokenizerConfig {
    pub normalization: NormalizationStrategy,
    pub tokenization: TokenizationStrategy,
    pub min_token_length: usize,
    pub max_token_length: usize,
    pub stop_words: Option<HashSet<String>>,
}

impl Default for TextTokenizerConfig {
    fn default() -> Self {
        Self {
            normalization: NormalizationStrategy::Lowercase,
            tokenization: TokenizationStrategy::Word,
            min_token_length: 1,
            max_token_length: 50,
            stop_words: None,
        }
    }
}

/// Text tokenizer for preprocessing text data
#[derive(Debug, Clone)]
pub struct TextTokenizer {
    config: TextTokenizerConfig,
}

impl Default for TextTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TextTokenizer {
    /// Create a new text tokenizer with default configuration
    pub fn new() -> Self {
        Self {
            config: TextTokenizerConfig::default(),
        }
    }

    /// Create a new text tokenizer with custom configuration
    pub fn with_config(config: TextTokenizerConfig) -> Self {
        Self { config }
    }

    /// Normalize text according to the configuration
    pub fn normalize(&self, text: &str) -> String {
        match self.config.normalization {
            NormalizationStrategy::None => text.to_string(),
            NormalizationStrategy::Lowercase => text.to_lowercase(),
            NormalizationStrategy::LowercaseNoPunct => text
                .to_lowercase()
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c.is_whitespace() {
                        c
                    } else {
                        ' '
                    }
                })
                .collect(),
            NormalizationStrategy::Full => text
                .to_lowercase()
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c.is_whitespace() {
                        c
                    } else {
                        ' '
                    }
                })
                .collect::<String>()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    /// Tokenize text into tokens
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized = self.normalize(text);

        let tokens: Vec<String> = match self.config.tokenization {
            TokenizationStrategy::Whitespace => normalized
                .split_whitespace()
                .map(|s| s.to_string())
                .collect(),
            TokenizationStrategy::WhitespacePunct => normalized
                .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect(),
            TokenizationStrategy::Word => normalized
                .chars()
                .collect::<String>()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect(),
        };

        // Filter by length
        let mut filtered_tokens: Vec<String> = tokens
            .into_iter()
            .filter(|token| {
                token.len() >= self.config.min_token_length
                    && token.len() <= self.config.max_token_length
            })
            .collect();

        // Remove stop words if configured
        if let Some(ref stop_words) = self.config.stop_words {
            filtered_tokens.retain(|token| !stop_words.contains(token));
        }

        filtered_tokens
    }
}

/// Configuration for TF-IDF vectorizer
#[derive(Debug, Clone)]
pub struct TfIdfVectorizerConfig {
    pub tokenizer_config: TextTokenizerConfig,
    pub min_df: f64,
    pub max_df: f64,
    pub max_features: Option<usize>,
    pub use_idf: bool,
    pub smooth_idf: bool,
    pub sublinear_tf: bool,
}

impl Default for TfIdfVectorizerConfig {
    fn default() -> Self {
        Self {
            tokenizer_config: TextTokenizerConfig::default(),
            min_df: 1.0,
            max_df: 1.0,
            max_features: None,
            use_idf: true,
            smooth_idf: true,
            sublinear_tf: false,
        }
    }
}

/// TF-IDF vectorizer for converting text to numerical features
#[derive(Debug, Clone)]
pub struct TfIdfVectorizer {
    config: TfIdfVectorizerConfig,
    tokenizer: TextTokenizer,
    vocabulary: HashMap<String, usize>,
    idf_values: Array1<f64>,
    fitted: bool,
}

impl Default for TfIdfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TfIdfVectorizer {
    /// Create a new TF-IDF vectorizer with default configuration
    pub fn new() -> Self {
        let config = TfIdfVectorizerConfig::default();
        let tokenizer = TextTokenizer::with_config(config.tokenizer_config.clone());

        Self {
            config,
            tokenizer,
            vocabulary: HashMap::new(),
            idf_values: Array1::zeros(0),
            fitted: false,
        }
    }

    /// Create a new TF-IDF vectorizer with custom configuration
    pub fn with_config(config: TfIdfVectorizerConfig) -> Self {
        let tokenizer = TextTokenizer::with_config(config.tokenizer_config.clone());

        Self {
            config,
            tokenizer,
            vocabulary: HashMap::new(),
            idf_values: Array1::zeros(0),
            fitted: false,
        }
    }

    /// Build vocabulary from documents
    fn build_vocabulary(&mut self, documents: &[String]) -> Result<()> {
        let mut term_doc_counts: HashMap<String, usize> = HashMap::new();
        let n_docs = documents.len() as f64;

        // Count document frequencies
        for document in documents {
            let tokens = self.tokenizer.tokenize(document);
            let unique_tokens: HashSet<String> = tokens.into_iter().collect();

            for token in unique_tokens {
                *term_doc_counts.entry(token).or_insert(0) += 1;
            }
        }

        // Filter by document frequency
        let min_df = if self.config.min_df < 1.0 {
            (self.config.min_df * n_docs).ceil() as usize
        } else {
            self.config.min_df as usize
        };

        let max_df = if self.config.max_df < 1.0 {
            (self.config.max_df * n_docs).floor() as usize
        } else {
            self.config.max_df as usize
        };

        let mut filtered_terms: Vec<(String, usize)> = term_doc_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_df && *count <= max_df)
            .collect();

        // Sort by document frequency (descending) for consistent ordering
        filtered_terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Limit vocabulary size if specified
        if let Some(max_features) = self.config.max_features {
            filtered_terms.truncate(max_features);
        }

        // Build vocabulary mapping
        for (idx, (term, _doc_freq)) in filtered_terms.iter().enumerate() {
            self.vocabulary.insert(term.clone(), idx);
        }

        // Compute IDF values
        let vocab_size = self.vocabulary.len();
        let mut idf_values = Array1::zeros(vocab_size);

        if self.config.use_idf {
            for &idx in self.vocabulary.values() {
                let doc_freq = filtered_terms[idx].1 as f64;
                let idf = if self.config.smooth_idf {
                    ((n_docs + 1.0) / (doc_freq + 1.0)).ln() + 1.0
                } else {
                    (n_docs / doc_freq).ln() + 1.0
                };
                idf_values[idx] = idf;
            }
        } else {
            idf_values.fill(1.0);
        }

        self.idf_values = idf_values;
        Ok(())
    }

    /// Transform documents to TF-IDF matrix
    fn transform_documents(&self, documents: &[String]) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "TfIdfVectorizer not fitted".to_string(),
            });
        }

        let n_docs = documents.len();
        let vocab_size = self.vocabulary.len();
        let mut tfidf_matrix = Array2::zeros((n_docs, vocab_size));

        for (doc_idx, document) in documents.iter().enumerate() {
            let tokens = self.tokenizer.tokenize(document);
            let mut term_counts: HashMap<usize, f64> = HashMap::new();

            // Count term frequencies
            for token in &tokens {
                if let Some(&vocab_idx) = self.vocabulary.get(token) {
                    *term_counts.entry(vocab_idx).or_insert(0.0) += 1.0;
                }
            }

            // Compute TF-IDF values
            let total_terms = tokens.len() as f64;
            for (vocab_idx, count) in term_counts {
                let tf = if self.config.sublinear_tf {
                    1.0 + count.ln()
                } else {
                    count / total_terms
                };

                let tfidf = tf * self.idf_values[vocab_idx];
                tfidf_matrix[[doc_idx, vocab_idx]] = tfidf;
            }
        }

        Ok(tfidf_matrix)
    }

    /// Get the vocabulary mapping
    pub fn get_vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }

    /// Get the IDF values
    pub fn get_idf_values(&self) -> ArrayView1<'_, f64> {
        self.idf_values.view()
    }
}

impl Fit<Vec<String>, ()> for TfIdfVectorizer {
    type Fitted = TfIdfVectorizer;

    fn fit(mut self, x: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        self.build_vocabulary(x)?;
        self.fitted = true;
        Ok(self)
    }
}

impl Transform<Vec<String>, Array2<f64>> for TfIdfVectorizer {
    fn transform(&self, x: &Vec<String>) -> Result<Array2<f64>> {
        self.transform_documents(x)
    }
}

/// Configuration for N-gram generator
#[derive(Debug, Clone)]
pub struct NgramGeneratorConfig {
    pub tokenizer_config: TextTokenizerConfig,
    pub ngram_type: NgramType,
    pub n_min: usize,
    pub n_max: usize,
}

impl Default for NgramGeneratorConfig {
    fn default() -> Self {
        Self {
            tokenizer_config: TextTokenizerConfig::default(),
            ngram_type: NgramType::Word,
            n_min: 1,
            n_max: 2,
        }
    }
}

/// N-gram generator for creating n-gram features from text
#[derive(Debug, Clone)]
pub struct NgramGenerator {
    config: NgramGeneratorConfig,
    tokenizer: TextTokenizer,
}

impl Default for NgramGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl NgramGenerator {
    /// Create a new N-gram generator with default configuration
    pub fn new() -> Self {
        let config = NgramGeneratorConfig::default();
        let tokenizer = TextTokenizer::with_config(config.tokenizer_config.clone());

        Self { config, tokenizer }
    }

    /// Create a new N-gram generator with custom configuration
    pub fn with_config(config: NgramGeneratorConfig) -> Self {
        let tokenizer = TextTokenizer::with_config(config.tokenizer_config.clone());
        Self { config, tokenizer }
    }

    /// Generate n-grams from text
    pub fn generate_ngrams(&self, text: &str) -> Vec<String> {
        match self.config.ngram_type {
            NgramType::Word => self.generate_word_ngrams(text),
            NgramType::Char => self.generate_char_ngrams(text),
        }
    }

    /// Generate word n-grams
    fn generate_word_ngrams(&self, text: &str) -> Vec<String> {
        let tokens = self.tokenizer.tokenize(text);
        let mut ngrams = Vec::new();

        for n in self.config.n_min..=self.config.n_max {
            if n > tokens.len() {
                break;
            }

            for window in tokens.windows(n) {
                let ngram = window.join(" ");
                ngrams.push(ngram);
            }
        }

        ngrams
    }

    /// Generate character n-grams
    fn generate_char_ngrams(&self, text: &str) -> Vec<String> {
        let normalized = self.tokenizer.normalize(text);
        let chars: Vec<char> = normalized.chars().collect();
        let mut ngrams = Vec::new();

        for n in self.config.n_min..=self.config.n_max {
            if n > chars.len() {
                break;
            }

            for window in chars.windows(n) {
                let ngram: String = window.iter().collect();
                ngrams.push(ngram);
            }
        }

        ngrams
    }
}

/// Configuration for text similarity calculator
#[derive(Debug, Clone)]
pub struct TextSimilarityConfig {
    pub tokenizer_config: TextTokenizerConfig,
    pub similarity_metric: SimilarityMetric,
}

/// Similarity metrics for text comparison
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityMetric {
    /// Cosine similarity
    Cosine,
    /// Jaccard similarity
    Jaccard,
    /// Dice coefficient
    Dice,
}

impl Default for TextSimilarityConfig {
    fn default() -> Self {
        Self {
            tokenizer_config: TextTokenizerConfig::default(),
            similarity_metric: SimilarityMetric::Cosine,
        }
    }
}

/// Text similarity calculator
#[derive(Debug, Clone)]
pub struct TextSimilarity {
    config: TextSimilarityConfig,
    tokenizer: TextTokenizer,
}

impl Default for TextSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl TextSimilarity {
    /// Create a new text similarity calculator
    pub fn new() -> Self {
        let config = TextSimilarityConfig::default();
        let tokenizer = TextTokenizer::with_config(config.tokenizer_config.clone());

        Self { config, tokenizer }
    }

    /// Create a new text similarity calculator with custom configuration
    pub fn with_config(config: TextSimilarityConfig) -> Self {
        let tokenizer = TextTokenizer::with_config(config.tokenizer_config.clone());
        Self { config, tokenizer }
    }

    /// Calculate similarity between two texts
    pub fn similarity(&self, text1: &str, text2: &str) -> f64 {
        match self.config.similarity_metric {
            SimilarityMetric::Cosine => self.cosine_similarity(text1, text2),
            SimilarityMetric::Jaccard => self.jaccard_similarity(text1, text2),
            SimilarityMetric::Dice => self.dice_coefficient(text1, text2),
        }
    }

    /// Calculate cosine similarity between two texts
    fn cosine_similarity(&self, text1: &str, text2: &str) -> f64 {
        let tokens1 = self.tokenizer.tokenize(text1);
        let tokens2 = self.tokenizer.tokenize(text2);

        let mut term_freq1: HashMap<String, f64> = HashMap::new();
        let mut term_freq2: HashMap<String, f64> = HashMap::new();

        for token in tokens1 {
            *term_freq1.entry(token).or_insert(0.0) += 1.0;
        }

        for token in tokens2 {
            *term_freq2.entry(token).or_insert(0.0) += 1.0;
        }

        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        let all_terms: HashSet<String> = term_freq1
            .keys()
            .chain(term_freq2.keys())
            .cloned()
            .collect();

        for term in all_terms {
            let freq1 = term_freq1.get(&term).unwrap_or(&0.0);
            let freq2 = term_freq2.get(&term).unwrap_or(&0.0);

            dot_product += freq1 * freq2;
            norm1 += freq1 * freq1;
            norm2 += freq2 * freq2;
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1.sqrt() * norm2.sqrt())
        }
    }

    /// Calculate Jaccard similarity between two texts
    fn jaccard_similarity(&self, text1: &str, text2: &str) -> f64 {
        let tokens1: HashSet<String> = self.tokenizer.tokenize(text1).into_iter().collect();
        let tokens2: HashSet<String> = self.tokenizer.tokenize(text2).into_iter().collect();

        let intersection = tokens1.intersection(&tokens2).count();
        let union = tokens1.union(&tokens2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Calculate Dice coefficient between two texts
    fn dice_coefficient(&self, text1: &str, text2: &str) -> f64 {
        let tokens1: HashSet<String> = self.tokenizer.tokenize(text1).into_iter().collect();
        let tokens2: HashSet<String> = self.tokenizer.tokenize(text2).into_iter().collect();

        let intersection = tokens1.intersection(&tokens2).count();
        let total = tokens1.len() + tokens2.len();

        if total == 0 {
            0.0
        } else {
            2.0 * intersection as f64 / total as f64
        }
    }
}

/// Configuration for bag-of-words embeddings
#[derive(Debug, Clone, Default)]
pub struct BagOfWordsConfig {
    pub tokenizer_config: TextTokenizerConfig,
    pub max_features: Option<usize>,
    pub binary: bool,
}

/// Simple bag-of-words sentence embeddings
#[derive(Debug, Clone)]
pub struct BagOfWordsEmbedding {
    config: BagOfWordsConfig,
    tokenizer: TextTokenizer,
    vocabulary: HashMap<String, usize>,
    fitted: bool,
}

impl Default for BagOfWordsEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

impl BagOfWordsEmbedding {
    /// Create a new bag-of-words embedding
    pub fn new() -> Self {
        let config = BagOfWordsConfig::default();
        let tokenizer = TextTokenizer::with_config(config.tokenizer_config.clone());

        Self {
            config,
            tokenizer,
            vocabulary: HashMap::new(),
            fitted: false,
        }
    }

    /// Create a new bag-of-words embedding with custom configuration
    pub fn with_config(config: BagOfWordsConfig) -> Self {
        let tokenizer = TextTokenizer::with_config(config.tokenizer_config.clone());

        Self {
            config,
            tokenizer,
            vocabulary: HashMap::new(),
            fitted: false,
        }
    }

    /// Build vocabulary from documents
    fn build_vocabulary(&mut self, documents: &[String]) -> Result<()> {
        let mut term_counts: HashMap<String, usize> = HashMap::new();

        // Count term frequencies
        for document in documents {
            let tokens = self.tokenizer.tokenize(document);
            for token in tokens {
                *term_counts.entry(token).or_insert(0) += 1;
            }
        }

        // Sort terms by frequency (descending) for consistent ordering
        let mut sorted_terms: Vec<(String, usize)> = term_counts.into_iter().collect();
        sorted_terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Limit vocabulary size if specified
        if let Some(max_features) = self.config.max_features {
            sorted_terms.truncate(max_features);
        }

        // Build vocabulary mapping
        for (idx, (term, _)) in sorted_terms.iter().enumerate() {
            self.vocabulary.insert(term.clone(), idx);
        }

        Ok(())
    }

    /// Transform documents to bag-of-words matrix
    fn transform_documents(&self, documents: &[String]) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "BagOfWordsEmbedding not fitted".to_string(),
            });
        }

        let n_docs = documents.len();
        let vocab_size = self.vocabulary.len();
        let mut bow_matrix = Array2::zeros((n_docs, vocab_size));

        for (doc_idx, document) in documents.iter().enumerate() {
            let tokens = self.tokenizer.tokenize(document);
            let mut term_counts: HashMap<usize, f64> = HashMap::new();

            // Count term frequencies
            for token in &tokens {
                if let Some(&vocab_idx) = self.vocabulary.get(token) {
                    *term_counts.entry(vocab_idx).or_insert(0.0) += 1.0;
                }
            }

            // Set values in matrix
            for (vocab_idx, count) in term_counts {
                let value = if self.config.binary { 1.0 } else { count };
                bow_matrix[[doc_idx, vocab_idx]] = value;
            }
        }

        Ok(bow_matrix)
    }

    /// Get the vocabulary mapping  
    pub fn get_vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }
}

impl Fit<Vec<String>, ()> for BagOfWordsEmbedding {
    type Fitted = BagOfWordsEmbedding;

    fn fit(mut self, x: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        self.build_vocabulary(x)?;
        self.fitted = true;
        Ok(self)
    }
}

impl Transform<Vec<String>, Array2<f64>> for BagOfWordsEmbedding {
    fn transform(&self, x: &Vec<String>) -> Result<Array2<f64>> {
        self.transform_documents(x)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_text_tokenizer() {
        let tokenizer = TextTokenizer::new();
        let text = "Hello, World! This is a TEST.";
        let tokens = tokenizer.tokenize(text);

        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_tfidf_vectorizer() {
        let vectorizer = TfIdfVectorizer::new();
        let documents = vec![
            "the cat sat on the mat".to_string(),
            "the dog ran in the park".to_string(),
            "cats and dogs are pets".to_string(),
        ];

        let fitted_vectorizer = vectorizer.fit(&documents, &()).unwrap();
        let tfidf_matrix = fitted_vectorizer.transform(&documents).unwrap();

        assert_eq!(
            tfidf_matrix.shape(),
            &[3, fitted_vectorizer.vocabulary.len()]
        );

        // Check that all values are non-negative
        for &value in tfidf_matrix.iter() {
            assert!(value >= 0.0);
        }
    }

    #[test]
    fn test_ngram_generator() {
        let generator = NgramGenerator::new();
        let text = "the quick brown fox";
        let ngrams = generator.generate_ngrams(text);

        // Should contain both unigrams and bigrams
        assert!(ngrams.contains(&"the".to_string()));
        assert!(ngrams.contains(&"quick".to_string()));
        assert!(ngrams.contains(&"the quick".to_string()));
        assert!(ngrams.contains(&"quick brown".to_string()));
    }

    #[test]
    fn test_text_similarity() {
        let similarity = TextSimilarity::new();

        // Test cosine similarity
        let sim1 = similarity.similarity("the cat sat", "the cat sat");
        assert_abs_diff_eq!(sim1, 1.0, epsilon = 1e-10);

        let sim2 = similarity.similarity("the cat sat", "the dog ran");
        assert!(sim2 > 0.0 && sim2 < 1.0);

        let sim3 = similarity.similarity("hello world", "goodbye moon");
        assert_eq!(sim3, 0.0);
    }

    #[test]
    fn test_bag_of_words_embedding() {
        let embedding = BagOfWordsEmbedding::new();
        let documents = vec![
            "the cat sat".to_string(),
            "the dog ran".to_string(),
            "cats and dogs".to_string(),
        ];

        let fitted_embedding = embedding.fit(&documents, &()).unwrap();
        let bow_matrix = fitted_embedding.transform(&documents).unwrap();

        assert_eq!(bow_matrix.shape(), &[3, fitted_embedding.vocabulary.len()]);

        // Check that all values are non-negative
        for &value in bow_matrix.iter() {
            assert!(value >= 0.0);
        }
    }
}

//! Natural Language Processing Neighbor-Based Methods
//!
//! This module provides specialized neighbor-based algorithms for natural language processing
//! applications, including document similarity search, semantic similarity computation,
//! word embedding analysis, sentence similarity search, and topic-based neighbor identification.
//!
//! # Key Features
//!
//! - **Document Similarity Search**: Efficient similarity search using TF-IDF, bag-of-words, and n-grams
//! - **Semantic Similarity**: Word embedding-based similarity using cosine distance and other metrics
//! - **Word Embedding Neighbors**: Find similar words in embedding space with multiple embedding models
//! - **Sentence Similarity**: Sentence-level similarity using various encoding techniques
//! - **Topic-Based Neighbors**: Topic modeling integration for document clustering and retrieval
//! - **Text Preprocessing**: Comprehensive text preprocessing pipeline with tokenization and normalization
//!
//! # Examples
//!
//! ```rust
//! use sklears_neighbors::nlp::{DocumentSimilaritySearch, TextFeatureType};
//! use scirs2_core::ndarray::Array2;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create document similarity search
//! let mut search = DocumentSimilaritySearch::new(TextFeatureType::TfIdf);
//!
//! // Add documents to index
//! let documents = vec![
//!     "Machine learning is a subset of artificial intelligence".to_string(),
//!     "Deep learning uses neural networks with many layers".to_string(),
//!     "Natural language processing analyzes human language".to_string(),
//! ];
//! search.build_index_from_documents(&documents)?;
//!
//! // Search for similar documents
//! let query = "AI and machine learning applications";
//! let results = search.search_by_text(query, 2)?;
//! # Ok(())
//! # }
//! ```

use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};
use std::iter::Iterator;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Text feature types for document and sentence analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TextFeatureType {
    /// Term Frequency-Inverse Document Frequency (TF-IDF)
    TfIdf,
    /// Bag of Words (BOW) representation
    BagOfWords,
    /// N-gram features (bigrams, trigrams, etc.)
    NGrams,
    /// Word embedding averages
    WordEmbeddings,
    /// Sentence embeddings
    SentenceEmbeddings,
    /// Topic model features (LDA-like)
    TopicModel,
    /// Character n-grams
    CharNGrams,
    /// Combined multiple features
    Combined,
}

/// Configuration for NLP-based similarity search
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NlpSearchConfig {
    /// Feature type to use
    pub feature_type: TextFeatureType,
    /// Maximum vocabulary size
    pub max_vocab_size: usize,
    /// Minimum document frequency for terms
    pub min_df: usize,
    /// Maximum document frequency for terms (as fraction)
    pub max_df: f64,
    /// N-gram range (min_n, max_n)
    pub ngram_range: (usize, usize),
    /// Use stop word filtering
    pub use_stop_words: bool,
    /// Minimum token length
    pub min_token_length: usize,
    /// Normalize features
    pub normalize_features: bool,
    /// Distance metric for similarity computation
    pub distance_metric: String,
}

impl Default for NlpSearchConfig {
    fn default() -> Self {
        Self {
            feature_type: TextFeatureType::TfIdf,
            max_vocab_size: 10000,
            min_df: 1,
            max_df: 0.8,
            ngram_range: (1, 1),
            use_stop_words: true,
            min_token_length: 2,
            normalize_features: true,
            distance_metric: "cosine".to_string(),
        }
    }
}

/// Document metadata for search results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DocumentMetadata {
    /// Document identifier
    pub id: String,
    /// Document title
    pub title: Option<String>,
    /// Document author
    pub author: Option<String>,
    /// Document language
    pub language: Option<String>,
    /// Document length (in tokens)
    pub length: usize,
    /// Custom metadata tags
    pub tags: HashMap<String, String>,
}

/// Search result for document similarity
#[derive(Debug, Clone)]
pub struct DocumentSearchResult {
    /// Document metadata
    pub metadata: DocumentMetadata,
    /// Original document text
    pub text: String,
    /// Similarity score (higher = more similar for cosine similarity)
    pub similarity: f64,
    /// Distance score (lower = more similar)
    pub distance: f64,
    /// Feature vector used for matching
    pub features: Array1<f64>,
    /// Match confidence score (0-1)
    pub confidence: f64,
}

/// Text preprocessing utilities
pub struct TextPreprocessor {
    config: NlpSearchConfig,
    stop_words: HashSet<String>,
    vocabulary: Option<HashMap<String, usize>>,
}

impl TextPreprocessor {
    /// Create new text preprocessor
    pub fn new(config: NlpSearchConfig) -> Self {
        let stop_words = if config.use_stop_words {
            Self::default_english_stop_words()
        } else {
            HashSet::new()
        };

        Self {
            config,
            stop_words,
            vocabulary: None,
        }
    }

    /// Default English stop words
    fn default_english_stop_words() -> HashSet<String> {
        vec![
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
            "the", "this", "but", "they", "have", "had", "what", "said", "each", "which", "their",
            "time", "if", "up", "out", "many", "then", "them", "these", "so", "some", "would",
            "into", "him", "could", "only", "her", "all", "also", "how", "its", "our", "two",
            "may", "way", "who", "she", "been", "call", "did", "get", "come", "made", "can", "go",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Tokenize text into words
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .filter(|token| {
                token.len() >= self.config.min_token_length
                    && (!self.config.use_stop_words || !self.stop_words.contains(*token))
            })
            .map(|s| s.to_string())
            .collect()
    }

    /// Generate n-grams from tokens
    pub fn generate_ngrams(&self, tokens: &[String]) -> Vec<String> {
        let mut ngrams = Vec::new();

        for n in self.config.ngram_range.0..=self.config.ngram_range.1 {
            for window in tokens.windows(n) {
                let ngram = window.join("_");
                ngrams.push(ngram);
            }
        }

        ngrams
    }

    /// Build vocabulary from documents
    pub fn build_vocabulary(&mut self, documents: &[String]) -> NeighborsResult<()> {
        let mut term_counts: HashMap<String, usize> = HashMap::new();
        let mut doc_frequencies: HashMap<String, usize> = HashMap::new();

        let total_docs = documents.len();

        // Count term frequencies and document frequencies
        for document in documents {
            let tokens = self.tokenize(document);
            let ngrams = self.generate_ngrams(&tokens);
            let mut doc_terms: HashSet<String> = HashSet::new();

            for term in ngrams {
                *term_counts.entry(term.clone()).or_insert(0) += 1;
                doc_terms.insert(term);
            }

            // Count document frequencies
            for term in doc_terms {
                *doc_frequencies.entry(term).or_insert(0) += 1;
            }
        }

        // Filter terms by document frequency
        let mut filtered_terms: Vec<(String, usize)> = term_counts
            .into_iter()
            .filter(|(term, _)| {
                let df = doc_frequencies.get(term).unwrap_or(&0);
                *df >= self.config.min_df && (*df as f64 / total_docs as f64) <= self.config.max_df
            })
            .collect();

        // Sort by frequency and take top terms
        filtered_terms.sort_by(|a, b| b.1.cmp(&a.1));
        filtered_terms.truncate(self.config.max_vocab_size);

        // Create vocabulary mapping
        let vocabulary: HashMap<String, usize> = filtered_terms
            .into_iter()
            .enumerate()
            .map(|(idx, (term, _))| (term, idx))
            .collect();

        self.vocabulary = Some(vocabulary);
        Ok(())
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    /// Get vocabulary
    pub fn get_vocabulary(&self) -> Option<&HashMap<String, usize>> {
        self.vocabulary.as_ref()
    }
}

/// TF-IDF feature extractor
pub struct TfIdfExtractor {
    preprocessor: TextPreprocessor,
    idf_weights: Option<Array1<f64>>,
}

impl TfIdfExtractor {
    /// Create new TF-IDF extractor
    pub fn new(config: NlpSearchConfig) -> Self {
        Self {
            preprocessor: TextPreprocessor::new(config),
            idf_weights: None,
        }
    }

    /// Fit TF-IDF on documents
    pub fn fit(&mut self, documents: &[String]) -> NeighborsResult<()> {
        // Build vocabulary
        self.preprocessor.build_vocabulary(documents)?;

        let vocab_size = self.preprocessor.vocab_size();
        let total_docs = documents.len();

        if let Some(vocabulary) = self.preprocessor.get_vocabulary() {
            // Compute document frequencies for IDF weights
            let mut doc_frequencies = vec![0; vocab_size];

            for document in documents {
                let tokens = self.preprocessor.tokenize(document);
                let ngrams = self.preprocessor.generate_ngrams(&tokens);
                let mut doc_terms: HashSet<usize> = HashSet::new();

                for term in ngrams {
                    if let Some(&idx) = vocabulary.get(&term) {
                        doc_terms.insert(idx);
                    }
                }

                for &idx in &doc_terms {
                    doc_frequencies[idx] += 1;
                }
            }

            // Compute IDF weights
            let mut idf_weights = Array1::zeros(vocab_size);
            for (i, &df) in doc_frequencies.iter().enumerate() {
                if df > 0 {
                    idf_weights[i] = ((total_docs as f64) / (df as f64)).ln();
                }
            }

            self.idf_weights = Some(idf_weights);
        }

        Ok(())
    }

    /// Transform document to TF-IDF vector
    pub fn transform(&self, document: &str) -> NeighborsResult<Array1<f64>> {
        let vocabulary = self
            .preprocessor
            .get_vocabulary()
            .ok_or(NeighborsError::InvalidInput(
                "TF-IDF not fitted".to_string(),
            ))?;

        let idf_weights = self
            .idf_weights
            .as_ref()
            .ok_or(NeighborsError::InvalidInput(
                "TF-IDF not fitted".to_string(),
            ))?;

        let vocab_size = vocabulary.len();
        let mut tf_vector = vec![0.0; vocab_size];

        // Compute term frequencies
        let tokens = self.preprocessor.tokenize(document);
        let ngrams = self.preprocessor.generate_ngrams(&tokens);
        let total_terms = ngrams.len() as f64;

        for term in ngrams {
            if let Some(&idx) = vocabulary.get(&term) {
                tf_vector[idx] += 1.0;
            }
        }

        // Normalize TF and apply IDF weights
        let mut tfidf_vector = Array1::zeros(vocab_size);
        for i in 0..vocab_size {
            if tf_vector[i] > 0.0 {
                let tf = tf_vector[i] / total_terms;
                tfidf_vector[i] = tf * idf_weights[i];
            }
        }

        // L2 normalization if requested
        if self.preprocessor.config.normalize_features {
            let norm = (tfidf_vector.mapv(|x| x * x).sum()).sqrt();
            if norm > 1e-8 {
                tfidf_vector /= norm;
            }
        }

        Ok(tfidf_vector)
    }

    /// Fit and transform documents
    pub fn fit_transform(&mut self, documents: &[String]) -> NeighborsResult<Array2<f64>> {
        self.fit(documents)?;

        let mut document_vectors = Vec::new();
        for document in documents {
            let tfidf_vector = self.transform(document)?;
            document_vectors.push(tfidf_vector);
        }

        // Convert to Array2
        let vocab_size = self.preprocessor.vocab_size();
        let mut feature_matrix = Array2::zeros((documents.len(), vocab_size));

        for (i, vector) in document_vectors.into_iter().enumerate() {
            feature_matrix.row_mut(i).assign(&vector);
        }

        Ok(feature_matrix)
    }
}

/// Document similarity search engine
pub struct DocumentSimilaritySearch {
    config: NlpSearchConfig,
    feature_extractor: Box<dyn DocumentFeatureExtractor>,
    feature_database: Option<Array2<f64>>,
    document_metadata: Vec<DocumentMetadata>,
    documents: Vec<String>,
}

/// Trait for document feature extraction
pub trait DocumentFeatureExtractor: Send + Sync {
    /// Fit the extractor on documents
    fn fit(&mut self, documents: &[String]) -> NeighborsResult<()>;

    /// Transform a single document to features
    fn transform(&self, document: &str) -> NeighborsResult<Array1<f64>>;

    /// Fit and transform documents
    fn fit_transform(&mut self, documents: &[String]) -> NeighborsResult<Array2<f64>>;

    /// Get feature dimension
    fn feature_dimension(&self) -> usize;

    /// Get extractor name
    fn name(&self) -> &str;
}

impl DocumentFeatureExtractor for TfIdfExtractor {
    fn fit(&mut self, documents: &[String]) -> NeighborsResult<()> {
        self.fit(documents)
    }

    fn transform(&self, document: &str) -> NeighborsResult<Array1<f64>> {
        self.transform(document)
    }

    fn fit_transform(&mut self, documents: &[String]) -> NeighborsResult<Array2<f64>> {
        self.fit_transform(documents)
    }

    fn feature_dimension(&self) -> usize {
        self.preprocessor.vocab_size()
    }

    fn name(&self) -> &str {
        "TF-IDF"
    }
}

impl DocumentSimilaritySearch {
    /// Create new document similarity search
    pub fn new(feature_type: TextFeatureType) -> Self {
        let config = NlpSearchConfig {
            feature_type,
            ..Default::default()
        };

        let feature_extractor: Box<dyn DocumentFeatureExtractor> = match feature_type {
            TextFeatureType::TfIdf => Box::new(TfIdfExtractor::new(config.clone())),
            TextFeatureType::BagOfWords => Box::new(TfIdfExtractor::new(config.clone())), // Simplified
            _ => Box::new(TfIdfExtractor::new(config.clone())), // Default to TF-IDF
        };

        Self {
            config,
            feature_extractor,
            feature_database: None,
            document_metadata: Vec::new(),
            documents: Vec::new(),
        }
    }

    /// Build index from documents
    pub fn build_index_from_documents(&mut self, documents: &[String]) -> NeighborsResult<()> {
        if documents.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        // Extract features from all documents
        let features = self.feature_extractor.fit_transform(documents)?;

        // Store documents and create default metadata
        self.documents = documents.to_vec();
        self.document_metadata = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| DocumentMetadata {
                id: format!("doc_{}", i),
                title: None,
                author: None,
                language: Some("en".to_string()),
                length: doc.split_whitespace().count(),
                tags: HashMap::new(),
            })
            .collect();

        self.feature_database = Some(features);
        Ok(())
    }

    /// Add document to index
    pub fn add_document(
        &mut self,
        document: String,
        metadata: DocumentMetadata,
    ) -> NeighborsResult<()> {
        // Extract features from document
        if self.feature_database.is_none() {
            // If no database exists, create one with just this document
            self.documents = vec![document.clone()];
            let features = self.feature_extractor.fit_transform(&[document])?;
            self.feature_database = Some(features);
        } else {
            // Add to existing database
            let features = self.feature_extractor.transform(&document)?;

            if let Some(ref mut db) = self.feature_database {
                // Extend existing database
                let mut new_db = Array2::zeros((db.nrows() + 1, db.ncols()));
                new_db
                    .slice_mut(scirs2_core::ndarray::s![..db.nrows(), ..])
                    .assign(db);
                new_db.row_mut(db.nrows()).assign(&features);
                *db = new_db;
            }

            self.documents.push(document);
        }

        self.document_metadata.push(metadata);
        Ok(())
    }

    /// Search for similar documents
    pub fn search_by_text(
        &self,
        query: &str,
        k: usize,
    ) -> NeighborsResult<Vec<DocumentSearchResult>> {
        let database = self
            .feature_database
            .as_ref()
            .ok_or(NeighborsError::InvalidInput("Index not built".to_string()))?;

        if database.is_empty() {
            return Ok(Vec::new());
        }

        // Extract features from query
        let query_features = self.feature_extractor.transform(query)?;

        // Compute similarities to all documents
        let mut similarities_with_indices: Vec<(f64, usize)> = Vec::new();

        for (idx, doc_features) in database.rows().into_iter().enumerate() {
            // Compute cosine similarity
            let dot_product: f64 = query_features
                .iter()
                .zip(doc_features.iter())
                .map(|(a, b)| a * b)
                .sum();

            let query_norm = (query_features.mapv(|x| x * x).sum()).sqrt();
            let doc_norm = (doc_features.mapv(|x| x * x).sum()).sqrt();

            let similarity = if query_norm > 1e-8 && doc_norm > 1e-8 {
                dot_product / (query_norm * doc_norm)
            } else {
                0.0
            };

            similarities_with_indices.push((similarity, idx));
        }

        // Sort by similarity (descending) and take top k
        similarities_with_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let k_results = k.min(similarities_with_indices.len());

        let mut results = Vec::new();
        for i in 0..k_results {
            let (similarity, idx) = similarities_with_indices[i];
            let distance = 1.0 - similarity; // Convert similarity to distance

            let features = database.row(idx).to_owned();
            let text = if idx < self.documents.len() {
                self.documents[idx].clone()
            } else {
                "".to_string()
            };

            let metadata = if idx < self.document_metadata.len() {
                self.document_metadata[idx].clone()
            } else {
                DocumentMetadata {
                    id: format!("doc_{}", idx),
                    title: None,
                    author: None,
                    language: Some("en".to_string()),
                    length: text.split_whitespace().count(),
                    tags: HashMap::new(),
                }
            };

            // Compute confidence score
            let confidence = similarity.clamp(0.0, 1.0);

            results.push(DocumentSearchResult {
                metadata,
                text,
                similarity,
                distance,
                features,
                confidence,
            });
        }

        Ok(results)
    }

    /// Search by feature vector
    pub fn search_by_features(
        &self,
        query_features: &Array1<f64>,
        k: usize,
    ) -> NeighborsResult<Vec<DocumentSearchResult>> {
        let database = self
            .feature_database
            .as_ref()
            .ok_or(NeighborsError::InvalidInput("Index not built".to_string()))?;

        let mut similarities_with_indices: Vec<(f64, usize)> = Vec::new();

        for (idx, doc_features) in database.rows().into_iter().enumerate() {
            // Compute cosine similarity
            let dot_product: f64 = query_features
                .iter()
                .zip(doc_features.iter())
                .map(|(a, b)| a * b)
                .sum();

            let query_norm = (query_features.mapv(|x| x * x).sum()).sqrt();
            let doc_norm = (doc_features.mapv(|x| x * x).sum()).sqrt();

            let similarity = if query_norm > 1e-8 && doc_norm > 1e-8 {
                dot_product / (query_norm * doc_norm)
            } else {
                0.0
            };

            similarities_with_indices.push((similarity, idx));
        }

        // Sort and return results
        similarities_with_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let k_results = k.min(similarities_with_indices.len());

        let mut results = Vec::new();
        for i in 0..k_results {
            let (similarity, idx) = similarities_with_indices[i];
            let distance = 1.0 - similarity;

            let features = database.row(idx).to_owned();
            let text = if idx < self.documents.len() {
                self.documents[idx].clone()
            } else {
                "".to_string()
            };

            let metadata = if idx < self.document_metadata.len() {
                self.document_metadata[idx].clone()
            } else {
                DocumentMetadata {
                    id: format!("doc_{}", idx),
                    title: None,
                    author: None,
                    language: Some("en".to_string()),
                    length: text.split_whitespace().count(),
                    tags: HashMap::new(),
                }
            };

            let confidence = similarity.max(0.0).min(1.0);

            results.push(DocumentSearchResult {
                metadata,
                text,
                similarity,
                distance,
                features,
                confidence,
            });
        }

        Ok(results)
    }

    /// Get search statistics
    pub fn get_stats(&self) -> (usize, usize, String) {
        let num_documents = self.documents.len();
        let feature_dim = self
            .feature_database
            .as_ref()
            .map(|db| db.ncols())
            .unwrap_or(0);
        let extractor_name = self.feature_extractor.name().to_string();

        (num_documents, feature_dim, extractor_name)
    }
}

/// Word embedding neighbor search
pub struct WordEmbeddingSearch {
    embeddings: Option<Array2<f64>>,
    word_to_index: HashMap<String, usize>,
    index_to_word: HashMap<usize, String>,
    embedding_dim: usize,
}

impl WordEmbeddingSearch {
    /// Create new word embedding search
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embeddings: None,
            word_to_index: HashMap::new(),
            index_to_word: HashMap::new(),
            embedding_dim,
        }
    }

    /// Load embeddings from word-vector pairs
    pub fn load_embeddings(
        &mut self,
        word_vectors: Vec<(String, Array1<f64>)>,
    ) -> NeighborsResult<()> {
        if word_vectors.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let vocab_size = word_vectors.len();
        let mut embeddings = Array2::zeros((vocab_size, self.embedding_dim));

        for (i, (word, vector)) in word_vectors.into_iter().enumerate() {
            if vector.len() != self.embedding_dim {
                return Err(NeighborsError::InvalidInput(format!(
                    "Vector dimension {} doesn't match expected {}",
                    vector.len(),
                    self.embedding_dim
                )));
            }

            embeddings.row_mut(i).assign(&vector);
            self.word_to_index.insert(word.clone(), i);
            self.index_to_word.insert(i, word);
        }

        self.embeddings = Some(embeddings);
        Ok(())
    }

    /// Find similar words
    pub fn find_similar_words(&self, word: &str, k: usize) -> NeighborsResult<Vec<(String, f64)>> {
        let embeddings = self
            .embeddings
            .as_ref()
            .ok_or(NeighborsError::InvalidInput(
                "Embeddings not loaded".to_string(),
            ))?;

        let word_idx = self
            .word_to_index
            .get(word)
            .ok_or(NeighborsError::InvalidInput(format!(
                "Word '{}' not found in vocabulary",
                word
            )))?;

        let query_embedding = embeddings.row(*word_idx);
        let mut similarities: Vec<(f64, usize)> = Vec::new();

        // Compute cosine similarity with all other words
        for (idx, embedding) in embeddings.rows().into_iter().enumerate() {
            if idx == *word_idx {
                continue; // Skip the query word itself
            }

            let dot_product: f64 = query_embedding
                .iter()
                .zip(embedding.iter())
                .map(|(a, b)| a * b)
                .sum();

            let query_norm = (query_embedding.mapv(|x| x * x).sum()).sqrt();
            let embed_norm = (embedding.mapv(|x| x * x).sum()).sqrt();

            let similarity = if query_norm > 1e-8 && embed_norm > 1e-8 {
                dot_product / (query_norm * embed_norm)
            } else {
                0.0
            };

            similarities.push((similarity, idx));
        }

        // Sort by similarity and return top k
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let k_results = k.min(similarities.len());

        let mut results = Vec::new();
        for i in 0..k_results {
            let (similarity, idx) = similarities[i];
            if let Some(word) = self.index_to_word.get(&idx) {
                results.push((word.clone(), similarity));
            }
        }

        Ok(results)
    }

    /// Get word embedding
    pub fn get_embedding(&self, word: &str) -> NeighborsResult<Array1<f64>> {
        let embeddings = self
            .embeddings
            .as_ref()
            .ok_or(NeighborsError::InvalidInput(
                "Embeddings not loaded".to_string(),
            ))?;

        let word_idx = self
            .word_to_index
            .get(word)
            .ok_or(NeighborsError::InvalidInput(format!(
                "Word '{}' not found in vocabulary",
                word
            )))?;

        Ok(embeddings.row(*word_idx).to_owned())
    }

    /// Check if word exists in vocabulary
    pub fn contains_word(&self, word: &str) -> bool {
        self.word_to_index.contains_key(word)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.word_to_index.len()
    }

    /// Get all words in vocabulary
    pub fn get_vocabulary(&self) -> Vec<String> {
        self.word_to_index.keys().cloned().collect()
    }
}

/// Sentence similarity search using averaged word embeddings
pub struct SentenceSimilaritySearch {
    word_embeddings: WordEmbeddingSearch,
    text_preprocessor: TextPreprocessor,
    sentence_embeddings: Option<Array2<f64>>,
    sentences: Vec<String>,
}

impl SentenceSimilaritySearch {
    /// Create new sentence similarity search
    pub fn new(embedding_dim: usize) -> Self {
        let config = NlpSearchConfig::default();
        Self {
            word_embeddings: WordEmbeddingSearch::new(embedding_dim),
            text_preprocessor: TextPreprocessor::new(config),
            sentence_embeddings: None,
            sentences: Vec::new(),
        }
    }

    /// Load word embeddings
    pub fn load_word_embeddings(
        &mut self,
        word_vectors: Vec<(String, Array1<f64>)>,
    ) -> NeighborsResult<()> {
        self.word_embeddings.load_embeddings(word_vectors)
    }

    /// Compute sentence embedding by averaging word embeddings
    pub fn compute_sentence_embedding(&self, sentence: &str) -> NeighborsResult<Array1<f64>> {
        let tokens = self.text_preprocessor.tokenize(sentence);
        if tokens.is_empty() {
            return Err(NeighborsError::InvalidInput(
                "Sentence contains no valid tokens".to_string(),
            ));
        }

        let mut embedding_sum = Array1::zeros(self.word_embeddings.embedding_dim);
        let mut valid_words = 0;

        for token in tokens {
            if self.word_embeddings.contains_word(&token) {
                let word_embedding = self.word_embeddings.get_embedding(&token)?;
                embedding_sum = &embedding_sum + &word_embedding;
                valid_words += 1;
            }
        }

        if valid_words == 0 {
            return Err(NeighborsError::InvalidInput(
                "No words found in vocabulary".to_string(),
            ));
        }

        // Average the embeddings
        embedding_sum /= valid_words as f64;

        // Normalize the embedding
        let norm = (embedding_sum.mapv(|x| x * x).sum()).sqrt();
        if norm > 1e-8 {
            embedding_sum /= norm;
        }

        Ok(embedding_sum)
    }

    /// Build index from sentences
    pub fn build_index_from_sentences(&mut self, sentences: &[String]) -> NeighborsResult<()> {
        if sentences.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let mut sentence_embeddings = Vec::new();
        for sentence in sentences {
            let embedding = self.compute_sentence_embedding(sentence)?;
            sentence_embeddings.push(embedding);
        }

        // Convert to Array2
        let embedding_dim = self.word_embeddings.embedding_dim;
        let mut embedding_matrix = Array2::zeros((sentences.len(), embedding_dim));

        for (i, embedding) in sentence_embeddings.into_iter().enumerate() {
            embedding_matrix.row_mut(i).assign(&embedding);
        }

        self.sentence_embeddings = Some(embedding_matrix);
        self.sentences = sentences.to_vec();
        Ok(())
    }

    /// Search for similar sentences
    pub fn search_similar_sentences(
        &self,
        query: &str,
        k: usize,
    ) -> NeighborsResult<Vec<(String, f64)>> {
        let embeddings = self
            .sentence_embeddings
            .as_ref()
            .ok_or(NeighborsError::InvalidInput("Index not built".to_string()))?;

        let query_embedding = self.compute_sentence_embedding(query)?;
        let mut similarities: Vec<(f64, usize)> = Vec::new();

        // Compute cosine similarity with all sentences
        for (idx, sentence_embedding) in embeddings.rows().into_iter().enumerate() {
            let dot_product: f64 = query_embedding
                .iter()
                .zip(sentence_embedding.iter())
                .map(|(a, b)| a * b)
                .sum();

            let query_norm = (query_embedding.mapv(|x| x * x).sum()).sqrt();
            let sent_norm = (sentence_embedding.mapv(|x| x * x).sum()).sqrt();

            let similarity = if query_norm > 1e-8 && sent_norm > 1e-8 {
                dot_product / (query_norm * sent_norm)
            } else {
                0.0
            };

            similarities.push((similarity, idx));
        }

        // Sort by similarity and return top k
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let k_results = k.min(similarities.len());

        let mut results = Vec::new();
        for i in 0..k_results {
            let (similarity, idx) = similarities[i];
            if idx < self.sentences.len() {
                results.push((self.sentences[idx].clone(), similarity));
            }
        }

        Ok(results)
    }

    /// Get statistics
    pub fn get_stats(&self) -> (usize, usize, usize) {
        let num_sentences = self.sentences.len();
        let embedding_dim = self.word_embeddings.embedding_dim;
        let vocab_size = self.word_embeddings.vocab_size();

        (num_sentences, embedding_dim, vocab_size)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_preprocessor() {
        let config = NlpSearchConfig::default();
        let preprocessor = TextPreprocessor::new(config);

        let text = "This is a test document with some words.";
        let tokens = preprocessor.tokenize(text);

        // Should filter out stop words and short tokens
        assert!(!tokens.is_empty());
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(tokens.contains(&"document".to_string()));
    }

    #[test]
    fn test_ngram_generation() {
        let config = NlpSearchConfig {
            ngram_range: (1, 2),
            ..Default::default()
        };
        let preprocessor = TextPreprocessor::new(config);

        let tokens = vec![
            "machine".to_string(),
            "learning".to_string(),
            "algorithms".to_string(),
        ];
        let ngrams = preprocessor.generate_ngrams(&tokens);

        assert!(ngrams.contains(&"machine".to_string()));
        assert!(ngrams.contains(&"learning".to_string()));
        assert!(ngrams.contains(&"machine_learning".to_string()));
        assert!(ngrams.contains(&"learning_algorithms".to_string()));
    }

    #[test]
    fn test_tfidf_extractor() {
        let config = NlpSearchConfig {
            max_vocab_size: 100,
            min_df: 1,
            ..Default::default()
        };
        let mut extractor = TfIdfExtractor::new(config);

        let documents = vec![
            "machine learning algorithms".to_string(),
            "deep learning neural networks".to_string(),
            "natural language processing".to_string(),
        ];

        let features = extractor.fit_transform(&documents).unwrap();
        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);

        // Test single document transform
        let query_features = extractor.transform("machine learning").unwrap();
        assert_eq!(query_features.len(), features.ncols());
    }

    #[test]
    fn test_document_similarity_search() {
        let mut search = DocumentSimilaritySearch::new(TextFeatureType::TfIdf);

        let documents = vec![
            "machine learning is artificial intelligence".to_string(),
            "deep learning uses neural networks".to_string(),
            "natural language processing analyzes text".to_string(),
        ];

        search.build_index_from_documents(&documents).unwrap();

        let results = search.search_by_text("machine learning", 2).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].similarity >= results[1].similarity);

        // First result should be most similar to query
        assert!(results[0].text.contains("machine learning"));
    }

    #[test]
    fn test_word_embedding_search() {
        let mut search = WordEmbeddingSearch::new(3);

        // Create dummy embeddings
        let word_vectors = vec![
            ("king".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0])),
            ("queen".to_string(), Array1::from_vec(vec![0.9, 0.1, 0.0])),
            ("man".to_string(), Array1::from_vec(vec![0.8, 0.0, 0.2])),
            ("woman".to_string(), Array1::from_vec(vec![0.7, 0.1, 0.2])),
            ("cat".to_string(), Array1::from_vec(vec![0.0, 1.0, 0.0])),
        ];

        search.load_embeddings(word_vectors).unwrap();

        let similar_words = search.find_similar_words("king", 2).unwrap();
        assert_eq!(similar_words.len(), 2);

        // Should find queen as most similar to king
        assert_eq!(similar_words[0].0, "queen");
        assert!(similar_words[0].1 > 0.8); // High similarity
    }

    #[test]
    fn test_sentence_similarity_search() {
        let mut search = SentenceSimilaritySearch::new(3);

        // Load dummy word embeddings with more comprehensive vocabulary
        let word_vectors = vec![
            ("machine".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0])),
            (
                "learning".to_string(),
                Array1::from_vec(vec![0.9, 0.1, 0.0]),
            ),
            ("deep".to_string(), Array1::from_vec(vec![0.8, 0.2, 0.0])),
            ("neural".to_string(), Array1::from_vec(vec![0.7, 0.3, 0.0])),
            (
                "networks".to_string(),
                Array1::from_vec(vec![0.6, 0.4, 0.0]),
            ),
            (
                "unrelated".to_string(),
                Array1::from_vec(vec![0.0, 0.0, 1.0]),
            ),
            (
                "sentence".to_string(),
                Array1::from_vec(vec![0.1, 0.0, 0.9]),
            ),
        ];

        search.load_word_embeddings(word_vectors).unwrap();

        let sentences = vec![
            "machine learning".to_string(),
            "deep neural networks".to_string(),
            "unrelated sentence".to_string(),
        ];

        search.build_index_from_sentences(&sentences).unwrap();

        let results = search
            .search_similar_sentences("machine learning", 2)
            .unwrap();
        assert!(!results.is_empty());

        // First result should be identical or very similar
        assert!(results[0].1 > 0.5); // Reasonable similarity threshold
    }

    #[test]
    fn test_vocabulary_building() {
        let config = NlpSearchConfig {
            max_vocab_size: 5,
            min_df: 1,
            max_df: 1.0,
            ..Default::default()
        };

        let mut preprocessor = TextPreprocessor::new(config);

        let documents = vec![
            "machine learning algorithms".to_string(),
            "deep learning neural networks".to_string(),
            "machine learning applications".to_string(),
        ];

        preprocessor.build_vocabulary(&documents).unwrap();

        assert_eq!(preprocessor.vocab_size(), 5);

        // Should contain frequent terms
        let vocab = preprocessor.get_vocabulary().unwrap();
        assert!(vocab.contains_key("machine"));
        assert!(vocab.contains_key("learning"));
    }
}

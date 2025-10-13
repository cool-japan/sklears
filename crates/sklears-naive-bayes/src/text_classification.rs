//! Text classification specific preprocessing and Naive Bayes implementations
//!
//! This module provides specialized text classification functionality including
//! TF-IDF integration, n-gram support, and document preprocessing.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, FitTransform, Predict, PredictProba, Transform},
};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::MultinomialNB;
use sklears_core::traits::{Fit, Trained};
use std::cmp::Ordering;

/// TF-IDF (Term Frequency-Inverse Document Frequency) transformer
#[derive(Debug, Clone)]
pub struct TfIdfTransformer {
    /// Use IDF weights
    pub use_idf: bool,
    /// Smooth IDF weights by adding 1 to document frequencies
    pub smooth_idf: bool,
    /// Enable sublinear TF scaling (replace tf with 1 + log(tf))
    pub sublinear_tf: bool,
    /// L2 normalization
    pub norm: Option<String>,
    /// Document length normalization - normalize TF by document length
    pub doc_length_norm: bool,
    /// Fitted IDF weights
    idf_: Option<Array1<f64>>,
    /// Number of documents seen during fit
    n_docs_: Option<usize>,
}

impl Default for TfIdfTransformer {
    fn default() -> Self {
        Self {
            use_idf: true,
            smooth_idf: true,
            sublinear_tf: false,
            norm: Some("l2".to_string()),
            doc_length_norm: false,
            idf_: None,
            n_docs_: None,
        }
    }
}

impl TfIdfTransformer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn use_idf(mut self, use_idf: bool) -> Self {
        self.use_idf = use_idf;
        self
    }

    pub fn smooth_idf(mut self, smooth_idf: bool) -> Self {
        self.smooth_idf = smooth_idf;
        self
    }

    pub fn sublinear_tf(mut self, sublinear_tf: bool) -> Self {
        self.sublinear_tf = sublinear_tf;
        self
    }

    pub fn norm(mut self, norm: Option<String>) -> Self {
        self.norm = norm;
        self
    }

    pub fn doc_length_norm(mut self, doc_length_norm: bool) -> Self {
        self.doc_length_norm = doc_length_norm;
        self
    }

    /// Compute IDF weights from term frequency matrix
    fn compute_idf(&self, tf_matrix: &Array2<f64>) -> Array1<f64> {
        let n_docs = tf_matrix.nrows() as f64;
        let n_features = tf_matrix.ncols();
        let mut idf = Array1::zeros(n_features);

        for j in 0..n_features {
            let df = tf_matrix.column(j).iter().filter(|&&x| x > 0.0).count() as f64;
            let df_smooth = if self.smooth_idf { df + 1.0 } else { df };
            let n_docs_smooth = if self.smooth_idf {
                n_docs + 1.0
            } else {
                n_docs
            };
            idf[j] = (n_docs_smooth / df_smooth).ln() + 1.0;
        }

        idf
    }

    /// Apply TF transformations
    fn transform_tf(&self, tf_matrix: &mut Array2<f64>) {
        if self.sublinear_tf {
            tf_matrix.mapv_inplace(|x| if x > 0.0 { 1.0 + x.ln() } else { 0.0 });
        }
    }

    /// Apply document length normalization
    fn apply_doc_length_norm(&self, tf_matrix: &mut Array2<f64>) {
        if self.doc_length_norm {
            for mut row in tf_matrix.axis_iter_mut(Axis(0)) {
                let doc_length: f64 = row.sum();
                if doc_length > 0.0 {
                    row /= doc_length;
                }
            }
        }
    }

    /// Apply normalization
    fn normalize(&self, matrix: &mut Array2<f64>) {
        if let Some(ref norm_type) = self.norm {
            match norm_type.as_str() {
                "l2" => {
                    for mut row in matrix.axis_iter_mut(Axis(0)) {
                        let norm = row.dot(&row).sqrt();
                        if norm > 0.0 {
                            row /= norm;
                        }
                    }
                }
                "l1" => {
                    for mut row in matrix.axis_iter_mut(Axis(0)) {
                        let norm: f64 = row.iter().map(|x| x.abs()).sum();
                        if norm > 0.0 {
                            row /= norm;
                        }
                    }
                }
                _ => {} // No normalization
            }
        }
    }
}

impl FitTransform<Array2<f64>, (), Array2<f64>> for TfIdfTransformer {
    fn fit_transform(mut self, x: &Array2<f64>, _y: Option<&()>) -> Result<Array2<f64>> {
        self.n_docs_ = Some(x.nrows());

        if self.use_idf {
            self.idf_ = Some(self.compute_idf(x));
        }

        self.transform_internal(x)
    }
}

impl Transform<Array2<f64>, Array2<f64>> for TfIdfTransformer {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.transform_internal(x)
    }
}

impl TfIdfTransformer {
    /// Internal transform method
    fn transform_internal(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let mut result = x.clone();

        // Apply TF transformations
        self.transform_tf(&mut result);

        // Apply document length normalization
        self.apply_doc_length_norm(&mut result);

        // Apply IDF weights
        if self.use_idf {
            if let Some(ref idf) = self.idf_ {
                for mut row in result.axis_iter_mut(Axis(0)) {
                    row *= idf;
                }
            }
        }

        // Apply normalization
        self.normalize(&mut result);

        Ok(result)
    }

    /// Fit the transformer
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<&mut Self> {
        self.n_docs_ = Some(x.nrows());

        if self.use_idf {
            self.idf_ = Some(self.compute_idf(x));
        }

        Ok(self)
    }

    /// Transform using fitted transformer
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.transform_internal(x)
    }
}

/// N-gram feature extractor for text
#[derive(Debug, Clone)]
pub struct NGramExtractor {
    /// Range of n-gram sizes (min, max)
    pub ngram_range: (usize, usize),
    /// Maximum number of features
    pub max_features: Option<usize>,
    /// Minimum document frequency for a term
    pub min_df: f64,
    /// Maximum document frequency for a term
    pub max_df: f64,
    /// Whether to use character n-grams instead of word n-grams
    pub analyzer: String,
    /// Vocabulary mapping
    vocabulary_: Option<HashMap<String, usize>>,
}

impl Default for NGramExtractor {
    fn default() -> Self {
        Self {
            ngram_range: (1, 1),
            max_features: None,
            min_df: 1.0,
            max_df: 1.0,
            analyzer: "word".to_string(),
            vocabulary_: None,
        }
    }
}

impl NGramExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ngram_range(mut self, ngram_range: (usize, usize)) -> Self {
        self.ngram_range = ngram_range;
        self
    }

    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    pub fn min_df(mut self, min_df: f64) -> Self {
        self.min_df = min_df;
        self
    }

    pub fn max_df(mut self, max_df: f64) -> Self {
        self.max_df = max_df;
        self
    }

    /// Extract word n-grams from text
    fn extract_word_ngrams(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut ngrams = Vec::new();

        for n in self.ngram_range.0..=self.ngram_range.1 {
            for window in words.windows(n) {
                let ngram = window.join(" ");
                ngrams.push(ngram);
            }
        }

        ngrams
    }

    /// Extract character n-grams from text
    fn extract_char_ngrams(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut ngrams = Vec::new();

        for n in self.ngram_range.0..=self.ngram_range.1 {
            for window in chars.windows(n) {
                let ngram: String = window.iter().collect();
                ngrams.push(ngram);
            }
        }

        ngrams
    }

    /// Build vocabulary from documents
    fn build_vocabulary(&self, documents: &[String]) -> HashMap<String, usize> {
        let mut term_counts: HashMap<String, usize> = HashMap::new();
        let n_docs = documents.len() as f64;

        // Count term frequencies across documents
        for doc in documents {
            let ngrams = match self.analyzer.as_str() {
                "word" => self.extract_word_ngrams(doc),
                "char" => self.extract_char_ngrams(doc),
                _ => self.extract_word_ngrams(doc),
            };

            let unique_ngrams: HashSet<String> = ngrams.into_iter().collect();
            for ngram in unique_ngrams {
                *term_counts.entry(ngram).or_insert(0) += 1;
            }
        }

        // Filter terms by document frequency
        let mut vocabulary: Vec<(String, usize)> = term_counts
            .into_iter()
            .filter(|(_, count)| {
                let df = *count as f64 / n_docs;
                df >= self.min_df && df <= self.max_df
            })
            .collect();

        // Sort by frequency and limit vocabulary size
        vocabulary.sort_by(|a, b| b.1.cmp(&a.1));
        if let Some(max_features) = self.max_features {
            vocabulary.truncate(max_features);
        }

        // Create vocabulary mapping
        vocabulary
            .into_iter()
            .enumerate()
            .map(|(i, (term, _))| (term, i))
            .collect()
    }

    /// Transform documents to feature matrix
    pub fn fit_transform(&mut self, documents: &[String]) -> Result<Array2<f64>> {
        self.vocabulary_ = Some(self.build_vocabulary(documents));
        self.transform(documents)
    }

    /// Transform documents using fitted vocabulary
    pub fn transform(&self, documents: &[String]) -> Result<Array2<f64>> {
        let vocabulary = self
            .vocabulary_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let n_docs = documents.len();
        let n_features = vocabulary.len();
        let mut feature_matrix = Array2::zeros((n_docs, n_features));

        for (doc_idx, doc) in documents.iter().enumerate() {
            let ngrams = match self.analyzer.as_str() {
                "word" => self.extract_word_ngrams(doc),
                "char" => self.extract_char_ngrams(doc),
                _ => self.extract_word_ngrams(doc),
            };

            let mut term_counts: HashMap<String, f64> = HashMap::new();
            for ngram in ngrams {
                *term_counts.entry(ngram).or_insert(0.0) += 1.0;
            }

            for (term, count) in term_counts {
                if let Some(&feature_idx) = vocabulary.get(&term) {
                    feature_matrix[[doc_idx, feature_idx]] = count;
                }
            }
        }

        Ok(feature_matrix)
    }
}

/// Text preprocessing pipeline for Naive Bayes
#[derive(Debug)]
pub struct TextPreprocessor {
    /// N-gram extractor
    pub ngram_extractor: NGramExtractor,
    /// TF-IDF transformer
    pub tfidf_transformer: TfIdfTransformer,
    /// Whether to use TF-IDF
    pub use_tfidf: bool,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self {
            ngram_extractor: NGramExtractor::default(),
            tfidf_transformer: TfIdfTransformer::default(),
            use_tfidf: true,
        }
    }
}

impl TextPreprocessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ngram_range(mut self, ngram_range: (usize, usize)) -> Self {
        self.ngram_extractor.ngram_range = ngram_range;
        self
    }

    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.ngram_extractor.max_features = max_features;
        self
    }

    pub fn use_tfidf(mut self, use_tfidf: bool) -> Self {
        self.use_tfidf = use_tfidf;
        self
    }

    /// Fit and transform documents
    pub fn fit_transform(&mut self, documents: &[String]) -> Result<Array2<f64>> {
        let mut feature_matrix = self.ngram_extractor.fit_transform(documents)?;

        if self.use_tfidf {
            self.tfidf_transformer.fit(&feature_matrix)?;
            feature_matrix = self.tfidf_transformer.transform(&feature_matrix)?;
        }

        Ok(feature_matrix)
    }

    /// Transform documents using fitted preprocessor
    pub fn transform(&self, documents: &[String]) -> Result<Array2<f64>> {
        let mut feature_matrix = self.ngram_extractor.transform(documents)?;

        if self.use_tfidf {
            feature_matrix = self.tfidf_transformer.transform(&feature_matrix)?;
        }

        Ok(feature_matrix)
    }
}

/// Text-optimized Multinomial Naive Bayes
#[derive(Debug)]
pub struct TextMultinomialNB {
    /// Text preprocessor
    pub preprocessor: TextPreprocessor,
    /// Multinomial Naive Bayes classifier
    pub classifier: Option<MultinomialNB<Trained>>,
    /// Configuration for the classifier
    pub config: TextMultinomialNBConfig,
}

/// Configuration for TextMultinomialNB
#[derive(Debug, Clone)]
pub struct TextMultinomialNBConfig {
    pub alpha: f64,
    pub fit_prior: bool,
    pub ngram_range: (usize, usize),
    pub max_features: Option<usize>,
    pub use_tfidf: bool,
    pub doc_length_norm: bool,
}

impl Default for TextMultinomialNBConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_prior: true,
            ngram_range: (1, 1),
            max_features: None,
            use_tfidf: true,
            doc_length_norm: false,
        }
    }
}

impl Default for TextMultinomialNB {
    fn default() -> Self {
        let config = TextMultinomialNBConfig::default();
        let mut preprocessor = TextPreprocessor::default()
            .ngram_range(config.ngram_range)
            .max_features(config.max_features)
            .use_tfidf(config.use_tfidf);

        // Configure document length normalization
        preprocessor.tfidf_transformer.doc_length_norm = config.doc_length_norm;

        Self {
            preprocessor,
            classifier: None,
            config,
        }
    }
}

impl TextMultinomialNB {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn fit_prior(mut self, fit_prior: bool) -> Self {
        self.config.fit_prior = fit_prior;
        self
    }

    pub fn ngram_range(mut self, ngram_range: (usize, usize)) -> Self {
        self.config.ngram_range = ngram_range;
        self.preprocessor = self.preprocessor.ngram_range(ngram_range);
        self
    }

    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.config.max_features = max_features;
        self.preprocessor = self.preprocessor.max_features(max_features);
        self
    }

    pub fn use_tfidf(mut self, use_tfidf: bool) -> Self {
        self.config.use_tfidf = use_tfidf;
        self.preprocessor = self.preprocessor.use_tfidf(use_tfidf);
        self
    }

    pub fn doc_length_norm(mut self, doc_length_norm: bool) -> Self {
        self.config.doc_length_norm = doc_length_norm;
        self.preprocessor.tfidf_transformer.doc_length_norm = doc_length_norm;
        self
    }

    /// Fit the model on text documents
    pub fn fit(&mut self, documents: &[String], y: &Array1<i32>) -> Result<()> {
        let x = self.preprocessor.fit_transform(documents)?;
        let classifier = MultinomialNB::new()
            .alpha(self.config.alpha)
            .fit_prior(self.config.fit_prior);
        let trained_classifier = classifier.fit(&x, y)?;
        self.classifier = Some(trained_classifier);
        Ok(())
    }

    /// Predict classes for text documents
    pub fn predict(&self, documents: &[String]) -> Result<Array1<i32>> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let x = self.preprocessor.transform(documents)?;
        classifier.predict(&x)
    }

    /// Predict class probabilities for text documents
    pub fn predict_proba(&self, documents: &[String]) -> Result<Array2<f64>> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            })?;

        let x = self.preprocessor.transform(documents)?;
        classifier.predict_proba(&x)
    }
}

/// Simple Latent Dirichlet Allocation (LDA) topic model for feature extraction
#[derive(Debug, Clone)]
pub struct LDATopicModel {
    /// Number of topics
    pub n_topics: usize,
    /// Number of iterations for training
    pub max_iter: usize,
    /// Alpha parameter (document-topic prior)
    pub alpha: f64,
    /// Beta parameter (topic-word prior)
    pub beta: f64,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Fitted topic-word distributions
    topic_word_distributions: Option<Array2<f64>>,
    /// Vocabulary mapping
    vocabulary: Option<HashMap<String, usize>>,
    /// Document-topic distributions from training
    doc_topic_distributions: Option<Array2<f64>>,
}

impl Default for LDATopicModel {
    fn default() -> Self {
        Self {
            n_topics: 10,
            max_iter: 100,
            alpha: 1.0,
            beta: 0.1,
            random_state: None,
            topic_word_distributions: None,
            vocabulary: None,
            doc_topic_distributions: None,
        }
    }
}

impl LDATopicModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_topics(mut self, n_topics: usize) -> Self {
        self.n_topics = n_topics;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Build vocabulary from documents
    fn build_vocabulary(&self, documents: &[String]) -> HashMap<String, usize> {
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            for word in doc.split_whitespace() {
                let word = word.to_lowercase();
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        // Filter words that appear at least twice
        word_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .enumerate()
            .map(|(i, (word, _))| (word, i))
            .collect()
    }

    /// Convert documents to word count matrix
    fn documents_to_word_counts(&self, documents: &[String]) -> (Array2<f64>, Vec<Vec<String>>) {
        let vocabulary = self.vocabulary.as_ref().unwrap();
        let n_docs = documents.len();
        let vocab_size = vocabulary.len();
        let mut word_counts = Array2::zeros((n_docs, vocab_size));
        let mut tokenized_docs = Vec::new();

        for (doc_idx, doc) in documents.iter().enumerate() {
            let words: Vec<String> = doc
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .filter(|w| vocabulary.contains_key(w))
                .collect();

            tokenized_docs.push(words.clone());

            for word in &words {
                if let Some(&word_idx) = vocabulary.get(word) {
                    word_counts[[doc_idx, word_idx]] += 1.0;
                }
            }
        }

        (word_counts, tokenized_docs)
    }

    /// Simplified LDA training using variational inference approximation
    fn train_lda(&mut self, word_counts: &Array2<f64>) -> Result<()> {
        let n_docs = word_counts.nrows();
        let vocab_size = word_counts.ncols();

        // Initialize topic-word and document-topic distributions
        let mut topic_word = Array2::from_elem((self.n_topics, vocab_size), self.beta);
        let mut doc_topic = Array2::from_elem((n_docs, self.n_topics), self.alpha);

        // Simple initialization with normalized random values
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let seed = self.random_state.unwrap_or_else(|| {
            let mut hasher = DefaultHasher::new();
            std::time::SystemTime::now().hash(&mut hasher);
            hasher.finish()
        });

        // Initialize with pseudo-random values based on seed
        for i in 0..self.n_topics {
            for j in 0..vocab_size {
                let hash_input = (seed, i, j);
                let mut hasher = DefaultHasher::new();
                hash_input.hash(&mut hasher);
                let random_val = (hasher.finish() % 1000) as f64 / 1000.0;
                topic_word[[i, j]] += random_val;
            }
        }

        // Iterative updates (simplified variational inference)
        for _iter in 0..self.max_iter {
            // Update document-topic distributions
            for d in 0..n_docs {
                for k in 0..self.n_topics {
                    let mut prob = self.alpha;
                    for w in 0..vocab_size {
                        if word_counts[[d, w]] > 0.0 {
                            prob += word_counts[[d, w]] * topic_word[[k, w]];
                        }
                    }
                    doc_topic[[d, k]] = prob;
                }

                // Normalize document-topic distribution
                let doc_sum: f64 = doc_topic.row(d).sum();
                if doc_sum > 0.0 {
                    for k in 0..self.n_topics {
                        doc_topic[[d, k]] /= doc_sum;
                    }
                }
            }

            // Update topic-word distributions
            for k in 0..self.n_topics {
                for w in 0..vocab_size {
                    let mut prob = self.beta;
                    for d in 0..n_docs {
                        if word_counts[[d, w]] > 0.0 {
                            prob += word_counts[[d, w]] * doc_topic[[d, k]];
                        }
                    }
                    topic_word[[k, w]] = prob;
                }

                // Normalize topic-word distribution
                let topic_sum: f64 = topic_word.row(k).sum();
                if topic_sum > 0.0 {
                    for w in 0..vocab_size {
                        topic_word[[k, w]] /= topic_sum;
                    }
                }
            }
        }

        self.topic_word_distributions = Some(topic_word);
        self.doc_topic_distributions = Some(doc_topic);

        Ok(())
    }

    /// Fit the LDA model on documents
    pub fn fit(&mut self, documents: &[String]) -> Result<()> {
        self.vocabulary = Some(self.build_vocabulary(documents));
        let (word_counts, _) = self.documents_to_word_counts(documents);
        self.train_lda(&word_counts)
    }

    /// Transform documents to topic distributions
    pub fn transform(&self, documents: &[String]) -> Result<Array2<f64>> {
        let vocabulary = self
            .vocabulary
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let topic_word =
            self.topic_word_distributions
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let (word_counts, _) = self.documents_to_word_counts(documents);
        let n_docs = word_counts.nrows();
        let mut doc_topic = Array2::from_elem((n_docs, self.n_topics), self.alpha);

        // Infer topic distributions for new documents
        for _iter in 0..20 {
            // Fewer iterations for inference
            for d in 0..n_docs {
                for k in 0..self.n_topics {
                    let mut prob = self.alpha;
                    for w in 0..vocabulary.len() {
                        if word_counts[[d, w]] > 0.0 {
                            prob += word_counts[[d, w]] * topic_word[[k, w]];
                        }
                    }
                    doc_topic[[d, k]] = prob;
                }

                // Normalize
                let doc_sum: f64 = doc_topic.row(d).sum();
                if doc_sum > 0.0 {
                    for k in 0..self.n_topics {
                        doc_topic[[d, k]] /= doc_sum;
                    }
                }
            }
        }

        Ok(doc_topic)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, documents: &[String]) -> Result<Array2<f64>> {
        self.fit(documents)?;
        self.transform(documents)
    }

    /// Get the top words for each topic
    pub fn get_topic_words(&self, n_words: usize) -> Result<Vec<Vec<(String, f64)>>> {
        let topic_word =
            self.topic_word_distributions
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "get_topic_words".to_string(),
                })?;

        let vocabulary = self
            .vocabulary
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "get_topic_words".to_string(),
            })?;

        // Create reverse vocabulary mapping
        let mut word_to_idx: Vec<(String, usize)> = vocabulary
            .iter()
            .map(|(word, &idx)| (word.clone(), idx))
            .collect();
        word_to_idx.sort_by_key(|(_, idx)| *idx);

        let mut topics = Vec::new();

        for k in 0..self.n_topics {
            let mut word_probs: Vec<(String, f64)> = word_to_idx
                .iter()
                .map(|(word, idx)| (word.clone(), topic_word[[k, *idx]]))
                .collect();

            // Sort by probability (descending)
            word_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            word_probs.truncate(n_words);

            topics.push(word_probs);
        }

        Ok(topics)
    }
}

/// Text classifier with topic model features
#[derive(Debug)]
pub struct TopicAugmentedTextClassifier {
    /// Text preprocessor
    pub text_preprocessor: TextPreprocessor,
    /// Topic model
    pub topic_model: LDATopicModel,
    /// Multinomial Naive Bayes classifier
    pub classifier: Option<MultinomialNB<Trained>>,
    /// Configuration
    pub config: TopicAugmentedConfig,
}

/// Configuration for TopicAugmentedTextClassifier
#[derive(Debug, Clone)]
pub struct TopicAugmentedConfig {
    pub alpha: f64,
    pub fit_prior: bool,
    pub n_topics: usize,
    pub topic_weight: f64, // Weight for topic features vs text features
    pub use_text_features: bool,
    pub use_topic_features: bool,
}

impl Default for TopicAugmentedConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_prior: true,
            n_topics: 10,
            topic_weight: 0.5,
            use_text_features: true,
            use_topic_features: true,
        }
    }
}

impl Default for TopicAugmentedTextClassifier {
    fn default() -> Self {
        let config = TopicAugmentedConfig::default();
        Self {
            text_preprocessor: TextPreprocessor::default(),
            topic_model: LDATopicModel::new().n_topics(config.n_topics),
            classifier: None,
            config,
        }
    }
}

impl TopicAugmentedTextClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_topics(mut self, n_topics: usize) -> Self {
        self.config.n_topics = n_topics;
        self.topic_model = self.topic_model.n_topics(n_topics);
        self
    }

    pub fn topic_weight(mut self, weight: f64) -> Self {
        self.config.topic_weight = weight;
        self
    }

    pub fn use_text_features(mut self, use_text: bool) -> Self {
        self.config.use_text_features = use_text;
        self
    }

    pub fn use_topic_features(mut self, use_topics: bool) -> Self {
        self.config.use_topic_features = use_topics;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Combine text features and topic features
    fn combine_features(
        &self,
        text_features: &Array2<f64>,
        topic_features: &Array2<f64>,
    ) -> Array2<f64> {
        let n_docs = text_features.nrows();
        let text_cols = if self.config.use_text_features {
            text_features.ncols()
        } else {
            0
        };
        let topic_cols = if self.config.use_topic_features {
            topic_features.ncols()
        } else {
            0
        };

        let mut combined = Array2::zeros((n_docs, text_cols + topic_cols));

        let mut col_offset = 0;

        // Add text features
        if self.config.use_text_features {
            for i in 0..n_docs {
                for j in 0..text_features.ncols() {
                    combined[[i, col_offset + j]] = text_features[[i, j]];
                }
            }
            col_offset += text_features.ncols();
        }

        // Add weighted topic features
        if self.config.use_topic_features {
            for i in 0..n_docs {
                for j in 0..topic_features.ncols() {
                    combined[[i, col_offset + j]] =
                        topic_features[[i, j]] * self.config.topic_weight;
                }
            }
        }

        combined
    }

    /// Fit the model on text documents
    pub fn fit(&mut self, documents: &[String], y: &Array1<i32>) -> Result<()> {
        // Get text features
        let text_features = if self.config.use_text_features {
            self.text_preprocessor.fit_transform(documents)?
        } else {
            Array2::zeros((documents.len(), 0))
        };

        // Get topic features
        let topic_features = if self.config.use_topic_features {
            self.topic_model.fit_transform(documents)?
        } else {
            Array2::zeros((documents.len(), 0))
        };

        // Combine features
        let combined_features = self.combine_features(&text_features, &topic_features);

        // Train classifier
        let classifier = MultinomialNB::new()
            .alpha(self.config.alpha)
            .fit_prior(self.config.fit_prior);
        let trained_classifier = classifier.fit(&combined_features, y)?;
        self.classifier = Some(trained_classifier);

        Ok(())
    }

    /// Predict classes for text documents
    pub fn predict(&self, documents: &[String]) -> Result<Array1<i32>> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        // Get text features
        let text_features = if self.config.use_text_features {
            self.text_preprocessor.transform(documents)?
        } else {
            Array2::zeros((documents.len(), 0))
        };

        // Get topic features
        let topic_features = if self.config.use_topic_features {
            self.topic_model.transform(documents)?
        } else {
            Array2::zeros((documents.len(), 0))
        };

        // Combine features
        let combined_features = self.combine_features(&text_features, &topic_features);

        classifier.predict(&combined_features)
    }

    /// Predict class probabilities for text documents
    pub fn predict_proba(&self, documents: &[String]) -> Result<Array2<f64>> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            })?;

        // Get text features
        let text_features = if self.config.use_text_features {
            self.text_preprocessor.transform(documents)?
        } else {
            Array2::zeros((documents.len(), 0))
        };

        // Get topic features
        let topic_features = if self.config.use_topic_features {
            self.topic_model.transform(documents)?
        } else {
            Array2::zeros((documents.len(), 0))
        };

        // Combine features
        let combined_features = self.combine_features(&text_features, &topic_features);

        classifier.predict_proba(&combined_features)
    }

    /// Get the discovered topics
    pub fn get_topics(&self, n_words: usize) -> Result<Vec<Vec<(String, f64)>>> {
        self.topic_model.get_topic_words(n_words)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ngram_extractor() {
        let mut extractor = NGramExtractor::new()
            .ngram_range((1, 2))
            .min_df(0.0) // Allow all terms
            .max_df(1.0); // Allow all terms

        let documents = vec![
            "hello world".to_string(),
            "world peace".to_string(),
            "hello peace".to_string(),
        ];

        let features = extractor.fit_transform(&documents).unwrap();

        // Should have 3 documents and some number of features
        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);
    }

    #[test]
    fn test_tfidf_transformer() {
        let tf_matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 1.0, 0.0, 1.0]).unwrap();

        let mut transformer = TfIdfTransformer::new();
        transformer.fit(&tf_matrix).unwrap();
        let tfidf_matrix = transformer.transform(&tf_matrix).unwrap();

        // Check that transformation was applied
        assert_eq!(tfidf_matrix.dim(), tf_matrix.dim());
        // Values should be different due to IDF weighting
        assert!(tfidf_matrix != tf_matrix);
    }

    #[test]
    fn test_text_multinomial_nb() {
        let documents = vec![
            "this is good".to_string(),
            "this is bad".to_string(),
            "good news".to_string(),
            "bad news".to_string(),
        ];
        let y = Array1::from_vec(vec![1, 0, 1, 0]);

        let mut model = TextMultinomialNB::new()
            .ngram_range((1, 1))
            .use_tfidf(false);

        model.fit(&documents, &y).unwrap();

        let test_docs = vec!["this is good".to_string()];
        let predictions = model.predict(&test_docs).unwrap();

        assert_eq!(predictions.len(), 1);
    }

    #[test]
    fn test_document_length_normalization() {
        // Create a simple TF matrix where documents have different lengths
        let tf_matrix = Array2::from_shape_vec(
            (3, 4),
            vec![
                2.0, 4.0, 0.0, 0.0, // doc 1: length = 6
                1.0, 0.0, 3.0, 0.0, // doc 2: length = 4
                0.0, 2.0, 0.0, 8.0, // doc 3: length = 10
            ],
        )
        .unwrap();

        let transformer = TfIdfTransformer::new()
            .doc_length_norm(true)
            .use_idf(false)
            .norm(None);

        let result = transformer.transform(&tf_matrix).unwrap();

        // Check that each row sums to 1.0 (document length normalized)
        for i in 0..result.nrows() {
            let row_sum: f64 = result.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_text_multinomial_nb_with_doc_length_norm() {
        let documents = vec![
            "this is a very long document with many words".to_string(),
            "short".to_string(),
            "this has medium length".to_string(),
            "brief".to_string(),
        ];
        let y = Array1::from_vec(vec![1, 0, 1, 0]);

        let mut model = TextMultinomialNB::new()
            .ngram_range((1, 1))
            .use_tfidf(true)
            .doc_length_norm(true);

        model.fit(&documents, &y).unwrap();

        let test_docs = vec!["this is medium".to_string()];
        let predictions = model.predict(&test_docs).unwrap();
        let probabilities = model.predict_proba(&test_docs).unwrap();

        assert_eq!(predictions.len(), 1);
        assert_eq!(probabilities.nrows(), 1);
        assert_eq!(probabilities.ncols(), 2);

        // Check that probabilities sum to 1
        let prob_sum: f64 = probabilities.row(0).sum();
        assert_abs_diff_eq!(prob_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lda_topic_model() {
        let documents = vec![
            "machine learning algorithm".to_string(),
            "deep learning neural network".to_string(),
            "algorithm optimization machine".to_string(),
            "neural deep learning".to_string(),
        ];

        let mut model = LDATopicModel::new()
            .n_topics(2)
            .max_iter(10)
            .random_state(Some(42));

        model.fit(&documents).unwrap();
        let topic_distributions = model.transform(&documents).unwrap();

        assert_eq!(topic_distributions.nrows(), 4);
        assert_eq!(topic_distributions.ncols(), 2);

        // Check that each document's topic distribution sums to 1
        for i in 0..topic_distributions.nrows() {
            let row_sum: f64 = topic_distributions.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        // Test topic words extraction
        let topics = model.get_topic_words(3).unwrap();
        assert_eq!(topics.len(), 2);
        for topic in topics {
            assert!(topic.len() <= 3);
        }
    }

    #[test]
    fn test_topic_augmented_text_classifier() {
        let documents = vec![
            "machine learning is great for data analysis".to_string(),
            "I love programming and coding algorithms".to_string(),
            "data science uses statistics and machine learning".to_string(),
            "programming languages are tools for software development".to_string(),
            "statistical analysis helps understand data patterns".to_string(),
            "software development requires good programming skills".to_string(),
        ];
        let y = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]); // 0: data science, 1: programming

        let mut model = TopicAugmentedTextClassifier::new()
            .n_topics(2)
            .topic_weight(0.3)
            .alpha(0.1);

        model.fit(&documents, &y).unwrap();

        let test_docs = vec!["machine learning programming".to_string()];
        let predictions = model.predict(&test_docs).unwrap();
        let probabilities = model.predict_proba(&test_docs).unwrap();

        assert_eq!(predictions.len(), 1);
        assert_eq!(probabilities.nrows(), 1);
        assert_eq!(probabilities.ncols(), 2);

        // Check that probabilities sum to 1
        let prob_sum: f64 = probabilities.row(0).sum();
        assert_abs_diff_eq!(prob_sum, 1.0, epsilon = 1e-10);

        // Test topics extraction
        let topics = model.get_topics(3).unwrap();
        assert_eq!(topics.len(), 2);
    }

    #[test]
    fn test_topic_only_classifier() {
        let documents = vec![
            "apple banana fruit".to_string(),
            "car vehicle transport".to_string(),
            "fruit apple orange".to_string(),
            "transport car bus".to_string(),
        ];
        let y = Array1::from_vec(vec![0, 1, 0, 1]); // 0: fruit, 1: transport

        let mut model = TopicAugmentedTextClassifier::new()
            .n_topics(2)
            .use_text_features(false) // Only use topic features
            .use_topic_features(true);

        model.fit(&documents, &y).unwrap();

        let test_docs = vec!["apple car".to_string()];
        let predictions = model.predict(&test_docs).unwrap();

        assert_eq!(predictions.len(), 1);
    }
}

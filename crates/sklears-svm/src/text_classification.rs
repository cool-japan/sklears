//! Text classification specific kernels and utilities for SVM
//!
//! This module provides specialized kernels and preprocessing utilities for text classification
//! tasks using Support Vector Machines. It includes:
//! - N-gram kernels for text similarity
//! - String kernels for sequence comparison
//! - TF-IDF integration for text preprocessing
//! - Document similarity kernels
//! - Text preprocessing utilities

use std::collections::HashMap;

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;
use scirs2_core::ndarray::{Array2, ArrayView1};

use crate::kernels::Kernel;
use sklears_core::error::{Result, SklearsError};

/// N-gram kernel for text classification
///
/// The n-gram kernel computes similarity between documents based on the number
/// of shared n-grams (subsequences of n consecutive characters or words).
/// This kernel is particularly effective for text classification tasks.
///
/// K(x, y) = Σ_{g ∈ N-grams} φ_g(x) * φ_g(y)
///
/// where φ_g(x) is the count (or normalized count) of n-gram g in document x.
///
/// References:
/// - Lodhi, H. et al. (2002). Text classification using string kernels.
/// - Cancedda, N. et al. (2003). Kernel methods for document analysis.
#[derive(Debug, Clone)]
pub struct NGramKernel {
    /// N-gram size (e.g., 2 for bigrams, 3 for trigrams)
    pub n: usize,
    /// Whether to normalize by document length
    pub normalize: bool,
    /// Whether to use character-level n-grams (true) or word-level (false)
    pub char_level: bool,
    /// Case sensitivity
    pub case_sensitive: bool,
    /// Minimum frequency threshold for n-grams
    pub min_freq: usize,
    /// Maximum number of n-grams to consider
    pub max_features: Option<usize>,
}

impl Default for NGramKernel {
    fn default() -> Self {
        Self {
            n: 3,
            normalize: true,
            char_level: true,
            case_sensitive: false,
            min_freq: 1,
            max_features: Some(10000),
        }
    }
}

impl NGramKernel {
    /// Create a new n-gram kernel
    pub fn new(n: usize) -> Self {
        Self {
            n,
            ..Default::default()
        }
    }

    /// Set normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set character level
    pub fn with_char_level(mut self, char_level: bool) -> Self {
        self.char_level = char_level;
        self
    }

    /// Set case sensitivity
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Set minimum frequency
    pub fn with_min_freq(mut self, min_freq: usize) -> Self {
        self.min_freq = min_freq;
        self
    }

    /// Set maximum features
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Extract n-grams from text
    pub fn extract_ngrams(&self, text: &str) -> Result<HashMap<String, usize>> {
        let processed_text = if self.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };

        let mut ngrams = HashMap::new();

        if self.char_level {
            // Character-level n-grams
            let chars: Vec<char> = processed_text.chars().collect();
            for window in chars.windows(self.n) {
                let ngram: String = window.iter().collect();
                *ngrams.entry(ngram).or_insert(0) += 1;
            }
        } else {
            // Word-level n-grams
            let words: Vec<&str> = processed_text.split_whitespace().collect();
            for window in words.windows(self.n) {
                let ngram = window.join(" ");
                *ngrams.entry(ngram).or_insert(0) += 1;
            }
        }

        // Filter by minimum frequency
        ngrams.retain(|_, &mut count| count >= self.min_freq);

        Ok(ngrams)
    }

    /// Compute kernel value between two texts
    pub fn compute_text_similarity(&self, text1: &str, text2: &str) -> Result<f64> {
        let ngrams1 = self.extract_ngrams(text1)?;
        let ngrams2 = self.extract_ngrams(text2)?;

        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        // Compute dot product and norms
        for (ngram, &count1) in &ngrams1 {
            norm1 += (count1 as f64).powi(2);
            if let Some(&count2) = ngrams2.get(ngram) {
                dot_product += (count1 as f64) * (count2 as f64);
            }
        }

        for &count2 in ngrams2.values() {
            norm2 += (count2 as f64).powi(2);
        }

        if self.normalize {
            let norm_product = norm1.sqrt() * norm2.sqrt();
            if norm_product > 0.0 {
                Ok(dot_product / norm_product)
            } else {
                Ok(0.0)
            }
        } else {
            Ok(dot_product)
        }
    }

    /// Convert texts to feature vectors
    pub fn texts_to_features(&self, texts: &[String]) -> Result<(Array2<f64>, Vec<String>)> {
        // Extract all n-grams from all texts
        let mut all_ngrams = HashMap::new();
        for text in texts {
            let ngrams = self.extract_ngrams(text)?;
            for (ngram, count) in ngrams {
                *all_ngrams.entry(ngram).or_insert(0) += count;
            }
        }

        // Filter and sort n-grams
        let mut ngram_features: Vec<(String, usize)> = all_ngrams
            .into_iter()
            .filter(|(_, count)| *count >= self.min_freq)
            .collect();

        // Sort by frequency (descending) and limit features
        ngram_features.sort_by(|a, b| b.1.cmp(&a.1));
        if let Some(max_features) = self.max_features {
            ngram_features.truncate(max_features);
        }

        let feature_names: Vec<String> = ngram_features
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let num_features = feature_names.len();

        // Create feature matrix
        let mut feature_matrix = Array2::zeros((texts.len(), num_features));

        for (text_idx, text) in texts.iter().enumerate() {
            let ngrams = self.extract_ngrams(text)?;
            let mut text_norm = 0.0;

            // Fill feature vector
            for (feat_idx, feature_name) in feature_names.iter().enumerate() {
                let count = ngrams.get(feature_name).copied().unwrap_or(0) as f64;
                feature_matrix[[text_idx, feat_idx]] = count;
                text_norm += count * count;
            }

            // Normalize if requested
            if self.normalize && text_norm > 0.0 {
                let norm = text_norm.sqrt();
                for feat_idx in 0..num_features {
                    feature_matrix[[text_idx, feat_idx]] /= norm;
                }
            }
        }

        Ok((feature_matrix, feature_names))
    }
}

impl Kernel for NGramKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // For pre-computed feature vectors, just compute dot product
        // Manual dot product to avoid recursion limit issues
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n".to_string(), self.n as f64);
        params
    }
}

/// String kernel for sequence comparison
///
/// The string kernel computes similarity between strings based on the number
/// of common subsequences. This is useful for comparing DNA sequences, protein
/// sequences, or any other string data.
///
/// K_n(s, t) = Σ_{u∈Σ^n} φ_u(s) * φ_u(t)
///
/// where φ_u(s) is the number of occurrences of subsequence u in string s.
///
/// References:
/// - Lodhi, H. et al. (2002). Text classification using string kernels.
/// - Shawe-Taylor, J. & Cristianini, N. (2004). Kernel Methods for Pattern Analysis.
#[derive(Debug, Clone)]
pub struct StringKernel {
    /// Maximum subsequence length
    pub max_length: usize,
    /// Decay factor for distant matches
    pub lambda: f64,
    /// Whether to normalize
    pub normalize: bool,
}

impl Default for StringKernel {
    fn default() -> Self {
        Self {
            max_length: 5,
            lambda: 0.5,
            normalize: true,
        }
    }
}

impl StringKernel {
    /// Create a new string kernel
    pub fn new(max_length: usize, lambda: f64) -> Self {
        Self {
            max_length,
            lambda,
            normalize: true,
        }
    }

    /// Compute string kernel between two strings
    pub fn compute_string_similarity(&self, s1: &str, s2: &str) -> f64 {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        let mut kernel_value = 0.0;

        // Dynamic programming approach for computing string kernel
        for length in 1..=self.max_length {
            kernel_value += self.compute_subsequence_kernel(&chars1, &chars2, length);
        }

        if self.normalize {
            let norm1 = self.compute_string_norm(&chars1);
            let norm2 = self.compute_string_norm(&chars2);
            let norm_product = norm1 * norm2;
            if norm_product > 0.0 {
                kernel_value / norm_product.sqrt()
            } else {
                0.0
            }
        } else {
            kernel_value
        }
    }

    /// Compute subsequence kernel for a specific length
    fn compute_subsequence_kernel(&self, s1: &[char], s2: &[char], length: usize) -> f64 {
        if length == 0 {
            return 1.0;
        }

        let n1 = s1.len();
        let n2 = s2.len();

        if n1 < length || n2 < length {
            return 0.0;
        }

        // Dynamic programming table
        let mut dp = vec![vec![0.0; n2 + 1]; n1 + 1];

        // Base case
        for i in 0..=n1 {
            for j in 0..=n2 {
                if length == 1 {
                    dp[i][j] = if i > 0 && j > 0 && s1[i - 1] == s2[j - 1] {
                        self.lambda.powi(2)
                    } else {
                        0.0
                    };
                }
            }
        }

        // Fill DP table for longer subsequences
        if length > 1 {
            let _prev_kernel = self.compute_subsequence_kernel(s1, s2, length - 1);

            for i in 1..=n1 {
                for j in 1..=n2 {
                    if s1[i - 1] == s2[j - 1] {
                        // Characters match, consider all previous positions
                        let mut sum = 0.0;
                        for ii in 0..i {
                            for jj in 0..j {
                                let dist1 = i - ii - 1;
                                let dist2 = j - jj - 1;
                                let decay = self.lambda.powi((dist1 + dist2 + 2) as i32);
                                sum += decay
                                    * self.compute_subsequence_kernel(
                                        &s1[0..ii + 1],
                                        &s2[0..jj + 1],
                                        length - 1,
                                    );
                            }
                        }
                        dp[i][j] = sum;
                    }
                }
            }
        }

        dp[n1][n2]
    }

    /// Compute string norm for normalization
    fn compute_string_norm(&self, s: &[char]) -> f64 {
        let mut norm = 0.0;
        for length in 1..=self.max_length {
            norm += self.compute_subsequence_kernel(s, s, length);
        }
        norm
    }
}

/// TF-IDF (Term Frequency-Inverse Document Frequency) preprocessor
///
/// TF-IDF is a numerical statistic that reflects how important a word is to a document
/// in a collection of documents. It increases proportionally to the number of times
/// a word appears in the document but is offset by the frequency of the word in the corpus.
///
/// TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
///
/// where:
/// - TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in d)
/// - IDF(t,D) = log(Total number of documents / Number of documents containing term t)
#[derive(Debug, Clone)]
pub struct TfIdfVectorizer {
    /// Minimum document frequency (ignore terms that appear in fewer documents)
    pub min_df: usize,
    /// Maximum document frequency (ignore terms that appear in more documents)
    pub max_df: f64,
    /// Maximum number of features
    pub max_features: Option<usize>,
    /// N-gram range (min_n, max_n)
    pub ngram_range: (usize, usize),
    /// Whether to use character-level n-grams
    pub char_level: bool,
    /// Case sensitivity
    pub case_sensitive: bool,
    /// Learned vocabulary
    vocabulary: Option<HashMap<String, usize>>,
    /// Document frequencies
    doc_frequencies: Option<HashMap<String, usize>>,
    /// Number of documents
    n_docs: usize,
}

impl Default for TfIdfVectorizer {
    fn default() -> Self {
        Self {
            min_df: 1,
            max_df: 1.0,
            max_features: None,
            ngram_range: (1, 1),
            char_level: false,
            case_sensitive: false,
            vocabulary: None,
            doc_frequencies: None,
            n_docs: 0,
        }
    }
}

impl TfIdfVectorizer {
    /// Create a new TF-IDF vectorizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum document frequency
    pub fn with_min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set maximum document frequency
    pub fn with_max_df(mut self, max_df: f64) -> Self {
        self.max_df = max_df;
        self
    }

    /// Set maximum features
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set n-gram range
    pub fn with_ngram_range(mut self, min_n: usize, max_n: usize) -> Self {
        self.ngram_range = (min_n, max_n);
        self
    }

    /// Set character level
    pub fn with_char_level(mut self, char_level: bool) -> Self {
        self.char_level = char_level;
        self
    }

    /// Set case sensitivity
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Fit the vectorizer on a corpus
    pub fn fit(&mut self, documents: &[String]) -> Result<()> {
        self.n_docs = documents.len();

        // Extract all terms and their document frequencies
        let mut term_doc_freq = HashMap::new();
        let mut vocabulary = HashMap::new();

        for document in documents {
            let terms = self.extract_terms(document)?;
            let unique_terms: std::collections::HashSet<String> = terms.into_iter().collect();

            for term in unique_terms {
                *term_doc_freq.entry(term.clone()).or_insert(0) += 1;
                vocabulary.insert(term, 0); // Will set proper indices later
            }
        }

        // Filter terms by document frequency
        let max_df_count = (self.max_df * self.n_docs as f64) as usize;

        let filtered_terms: Vec<String> = term_doc_freq
            .iter()
            .filter(|(_, &df)| df >= self.min_df && df <= max_df_count)
            .map(|(term, _)| term.clone())
            .collect();

        // Sort terms and limit by max_features
        let mut terms_with_freq: Vec<(String, usize)> = filtered_terms
            .iter()
            .map(|term| (term.clone(), *term_doc_freq.get(term).unwrap()))
            .collect();

        terms_with_freq.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency (descending)

        if let Some(max_features) = self.max_features {
            terms_with_freq.truncate(max_features);
        }

        // Create final vocabulary
        let mut final_vocabulary = HashMap::new();
        let mut final_doc_freq = HashMap::new();

        for (idx, (term, freq)) in terms_with_freq.iter().enumerate() {
            final_vocabulary.insert(term.clone(), idx);
            final_doc_freq.insert(term.clone(), *freq);
        }

        self.vocabulary = Some(final_vocabulary);
        self.doc_frequencies = Some(final_doc_freq);

        Ok(())
    }

    /// Transform documents to TF-IDF matrix
    pub fn transform(&self, documents: &[String]) -> Result<Array2<f64>> {
        let vocabulary = self
            .vocabulary
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let doc_frequencies =
            self.doc_frequencies
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let n_features = vocabulary.len();
        let mut tfidf_matrix = Array2::zeros((documents.len(), n_features));

        for (doc_idx, document) in documents.iter().enumerate() {
            let terms = self.extract_terms(document)?;
            let mut term_counts = HashMap::new();

            // Count term frequencies
            for term in &terms {
                if vocabulary.contains_key(term) {
                    *term_counts.entry(term.clone()).or_insert(0) += 1;
                }
            }

            let total_terms = terms.len() as f64;

            // Compute TF-IDF for each term
            for (term, count) in term_counts {
                if let (Some(&term_idx), Some(&doc_freq)) =
                    (vocabulary.get(&term), doc_frequencies.get(&term))
                {
                    let tf = count as f64 / total_terms;
                    let idf = (self.n_docs as f64 / doc_freq as f64).ln();
                    let tfidf = tf * idf;

                    tfidf_matrix[[doc_idx, term_idx]] = tfidf;
                }
            }
        }

        Ok(tfidf_matrix)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, documents: &[String]) -> Result<Array2<f64>> {
        self.fit(documents)?;
        self.transform(documents)
    }

    /// Extract terms from a document
    fn extract_terms(&self, document: &str) -> Result<Vec<String>> {
        let processed_doc = if self.case_sensitive {
            document.to_string()
        } else {
            document.to_lowercase()
        };

        let mut terms = Vec::new();

        if self.char_level {
            // Character-level n-grams
            let chars: Vec<char> = processed_doc.chars().collect();
            for n in self.ngram_range.0..=self.ngram_range.1 {
                for window in chars.windows(n) {
                    let term: String = window.iter().collect();
                    terms.push(term);
                }
            }
        } else {
            // Word-level n-grams
            let words: Vec<&str> = processed_doc.split_whitespace().collect();
            for n in self.ngram_range.0..=self.ngram_range.1 {
                for window in words.windows(n) {
                    let term = window.join(" ");
                    terms.push(term);
                }
            }
        }

        Ok(terms)
    }

    /// Get feature names
    pub fn get_feature_names(&self) -> Result<Vec<String>> {
        let vocabulary = self
            .vocabulary
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "get_feature_names".to_string(),
            })?;

        let mut features = vec![String::new(); vocabulary.len()];
        for (term, &idx) in vocabulary {
            features[idx] = term.clone();
        }

        Ok(features)
    }
}

/// Document similarity kernel based on cosine similarity
///
/// This kernel computes the cosine similarity between document vectors,
/// which is commonly used in information retrieval and text classification.
///
/// K(d1, d2) = cos(θ) = (d1 · d2) / (||d1|| × ||d2||)
#[derive(Debug, Clone)]
pub struct DocumentSimilarityKernel {
    /// TF-IDF vectorizer
    pub vectorizer: TfIdfVectorizer,
    /// Precomputed document vectors
    document_vectors: Option<Array2<f64>>,
    /// Similarity threshold for sparse kernels
    similarity_threshold: f64,
    /// Whether to normalize vectors
    normalize_vectors: bool,
}

impl DocumentSimilarityKernel {
    /// Create a new document similarity kernel
    pub fn new(vectorizer: TfIdfVectorizer) -> Self {
        Self {
            vectorizer,
            document_vectors: None,
            similarity_threshold: 0.1,
            normalize_vectors: true,
        }
    }

    /// Fit the kernel on a corpus
    pub fn fit(&mut self, documents: &[String]) -> Result<()> {
        let vectors = self.vectorizer.fit_transform(documents)?;
        self.document_vectors = Some(vectors);
        Ok(())
    }

    /// Compute similarity between two document indices
    pub fn compute_document_similarity(&self, doc1_idx: usize, doc2_idx: usize) -> Result<f64> {
        let vectors = self
            .document_vectors
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "compute_similarity".to_string(),
            })?;

        if doc1_idx >= vectors.nrows() || doc2_idx >= vectors.nrows() {
            return Err(SklearsError::InvalidInput(
                "Document index out of bounds".to_string(),
            ));
        }

        let vec1 = vectors.row(doc1_idx);
        let vec2 = vectors.row(doc2_idx);

        let dot_product = vec1.dot(&vec2);
        let norm1 = vec1.dot(&vec1).sqrt();
        let norm2 = vec2.dot(&vec2).sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            Ok(dot_product / (norm1 * norm2))
        } else {
            Ok(0.0)
        }
    }

    /// Get the document vector matrix
    pub fn get_document_vectors(&self) -> Option<&Array2<f64>> {
        self.document_vectors.as_ref()
    }
}

impl Kernel for DocumentSimilarityKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // Manual dot products to avoid recursion limit issues
        let dot_product: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let norm_x: f64 = x.iter().map(|a| a * a).sum::<f64>().sqrt();
        let norm_y: f64 = y.iter().map(|b| b * b).sum::<f64>().sqrt();

        if norm_x > 0.0 && norm_y > 0.0 {
            dot_product / (norm_x * norm_y)
        } else {
            0.0
        }
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert(
            "similarity_threshold".to_string(),
            self.similarity_threshold,
        );
        params.insert(
            "normalize_vectors".to_string(),
            if self.normalize_vectors { 1.0 } else { 0.0 },
        );
        params
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_kernel_basic() {
        let kernel = NGramKernel::new(2).with_char_level(true);

        let text1 = "hello";
        let text2 = "hello";
        let similarity = kernel.compute_text_similarity(text1, text2).unwrap();
        assert_eq!(similarity, 1.0); // Identical texts should have similarity 1.0

        let text3 = "world";
        let similarity2 = kernel.compute_text_similarity(text1, text3).unwrap();
        assert!(similarity2 < 1.0); // Different texts should have lower similarity
    }

    #[test]
    fn test_ngram_extraction() {
        let kernel = NGramKernel::new(2).with_char_level(true);
        let ngrams = kernel.extract_ngrams("hello").unwrap();

        assert!(ngrams.contains_key("he"));
        assert!(ngrams.contains_key("el"));
        assert!(ngrams.contains_key("ll"));
        assert!(ngrams.contains_key("lo"));
        assert_eq!(ngrams.len(), 4);
    }

    #[test]
    fn test_word_level_ngrams() {
        let kernel = NGramKernel::new(2).with_char_level(false);
        let ngrams = kernel.extract_ngrams("hello world test").unwrap();

        assert!(ngrams.contains_key("hello world"));
        assert!(ngrams.contains_key("world test"));
        assert_eq!(ngrams.len(), 2);
    }

    #[test]
    fn test_string_kernel() {
        let kernel = StringKernel::new(3, 0.5);

        let s1 = "abc";
        let s2 = "abc";
        let similarity = kernel.compute_string_similarity(s1, s2);
        assert!(similarity > 0.0);

        let s3 = "xyz";
        let similarity2 = kernel.compute_string_similarity(s1, s3);
        assert!(similarity2 < similarity);
    }

    #[test]
    fn test_tfidf_vectorizer() {
        let mut vectorizer = TfIdfVectorizer::new().with_min_df(1).with_ngram_range(1, 1);

        let documents = vec![
            "hello world".to_string(),
            "world test".to_string(),
            "hello test".to_string(),
        ];

        let result = vectorizer.fit_transform(&documents);
        assert!(result.is_ok());

        let matrix = result.unwrap();
        assert_eq!(matrix.nrows(), 3);
        assert!(matrix.ncols() > 0);
    }

    #[test]
    fn test_tfidf_feature_names() {
        let mut vectorizer = TfIdfVectorizer::new().with_min_df(1).with_ngram_range(1, 1);

        let documents = vec!["hello world".to_string(), "world test".to_string()];

        vectorizer.fit(&documents).unwrap();
        let feature_names = vectorizer.get_feature_names().unwrap();

        assert!(feature_names.contains(&"hello".to_string()));
        assert!(feature_names.contains(&"world".to_string()));
        assert!(feature_names.contains(&"test".to_string()));
    }

    #[test]
    fn test_document_similarity_kernel() {
        let vectorizer = TfIdfVectorizer::new().with_min_df(1).with_ngram_range(1, 1);

        let mut kernel = DocumentSimilarityKernel::new(vectorizer);

        let documents = vec![
            "hello world".to_string(),
            "hello test".to_string(),
            "completely different text".to_string(),
        ];

        kernel.fit(&documents).unwrap();

        // Documents 0 and 1 should be more similar (both contain "hello")
        let sim_01 = kernel.compute_document_similarity(0, 1).unwrap();
        let sim_02 = kernel.compute_document_similarity(0, 2).unwrap();

        assert!(sim_01 > sim_02);
    }

    #[test]
    fn test_texts_to_features() {
        let kernel = NGramKernel::new(2)
            .with_char_level(true)
            .with_normalize(true);

        let texts = vec!["hello".to_string(), "world".to_string()];

        let result = kernel.texts_to_features(&texts);
        assert!(result.is_ok());

        let (features, feature_names) = result.unwrap();
        assert_eq!(features.nrows(), 2);
        assert!(features.ncols() > 0);
        assert_eq!(features.ncols(), feature_names.len());
    }

    #[test]
    fn test_case_sensitivity() {
        let kernel_sensitive = NGramKernel::new(2)
            .with_char_level(true)
            .with_case_sensitive(true);

        let kernel_insensitive = NGramKernel::new(2)
            .with_char_level(true)
            .with_case_sensitive(false);

        let text1 = "Hello";
        let text2 = "hello";

        let sim_sensitive = kernel_sensitive
            .compute_text_similarity(text1, text2)
            .unwrap();
        let sim_insensitive = kernel_insensitive
            .compute_text_similarity(text1, text2)
            .unwrap();

        assert!(sim_insensitive > sim_sensitive);
        assert_eq!(sim_insensitive, 1.0); // Should be identical when case-insensitive
    }
}

//! Natural Language Processing kernel approximations
//!
//! This module provides kernel approximation methods specifically designed for NLP tasks,
//! including text kernels, semantic features, syntactic approximations, and word embedding kernels.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::essentials::Normal as RandNormal;
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use sklears_core::error::Result;
use sklears_core::traits::{Fit, Transform};

/// Text kernel approximation using bag-of-words and n-gram features
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TextKernelApproximation
pub struct TextKernelApproximation {
    /// n_components
    pub n_components: usize,
    /// max_features
    pub max_features: usize,
    /// ngram_range
    pub ngram_range: (usize, usize),
    /// min_df
    pub min_df: usize,
    /// max_df
    pub max_df: f64,
    /// use_tf_idf
    pub use_tf_idf: bool,
    /// use_hashing
    pub use_hashing: bool,
    /// sublinear_tf
    pub sublinear_tf: bool,
}

impl TextKernelApproximation {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_features: 10000,
            ngram_range: (1, 1),
            min_df: 1,
            max_df: 1.0,
            use_tf_idf: true,
            use_hashing: false,
            sublinear_tf: false,
        }
    }

    pub fn max_features(mut self, max_features: usize) -> Self {
        self.max_features = max_features;
        self
    }

    pub fn ngram_range(mut self, ngram_range: (usize, usize)) -> Self {
        self.ngram_range = ngram_range;
        self
    }

    pub fn min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df;
        self
    }

    pub fn max_df(mut self, max_df: f64) -> Self {
        self.max_df = max_df;
        self
    }

    pub fn use_tf_idf(mut self, use_tf_idf: bool) -> Self {
        self.use_tf_idf = use_tf_idf;
        self
    }

    pub fn use_hashing(mut self, use_hashing: bool) -> Self {
        self.use_hashing = use_hashing;
        self
    }

    pub fn sublinear_tf(mut self, sublinear_tf: bool) -> Self {
        self.sublinear_tf = sublinear_tf;
        self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    fn extract_ngrams(&self, tokens: &[String]) -> Vec<String> {
        let mut ngrams = Vec::new();

        for n in self.ngram_range.0..=self.ngram_range.1 {
            for i in 0..=tokens.len().saturating_sub(n) {
                if i + n <= tokens.len() {
                    let ngram = tokens[i..i + n].join(" ");
                    ngrams.push(ngram);
                }
            }
        }

        ngrams
    }

    fn hash_feature(&self, feature: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        feature.hash(&mut hasher);
        hasher.finish() as usize % self.max_features
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// FittedTextKernelApproximation
pub struct FittedTextKernelApproximation {
    /// n_components
    pub n_components: usize,
    /// max_features
    pub max_features: usize,
    /// ngram_range
    pub ngram_range: (usize, usize),
    /// min_df
    pub min_df: usize,
    /// max_df
    pub max_df: f64,
    /// use_tf_idf
    pub use_tf_idf: bool,
    /// use_hashing
    pub use_hashing: bool,
    /// sublinear_tf
    pub sublinear_tf: bool,
    /// vocabulary
    pub vocabulary: HashMap<String, usize>,
    /// idf_values
    pub idf_values: Array1<f64>,
    /// random_weights
    pub random_weights: Array2<f64>,
}

impl Fit<Vec<String>, ()> for TextKernelApproximation {
    type Fitted = FittedTextKernelApproximation;

    fn fit(self, documents: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        let mut vocabulary = HashMap::new();
        let mut document_frequencies = HashMap::new();
        let n_documents = documents.len();

        // Build vocabulary and compute document frequencies
        for doc in documents {
            let tokens = self.tokenize(doc);
            let ngrams = self.extract_ngrams(&tokens);
            let unique_ngrams: HashSet<String> = ngrams.into_iter().collect();

            for ngram in unique_ngrams {
                *document_frequencies.entry(ngram.clone()).or_insert(0) += 1;
                if !vocabulary.contains_key(&ngram) {
                    vocabulary.insert(ngram, vocabulary.len());
                }
            }
        }

        // Filter vocabulary based on min_df and max_df
        let mut filtered_vocabulary = HashMap::new();
        for (term, &df) in &document_frequencies {
            let df_ratio = df as f64 / n_documents as f64;
            if df >= self.min_df && df_ratio <= self.max_df {
                filtered_vocabulary.insert(term.clone(), filtered_vocabulary.len());
            }
        }

        // Limit vocabulary size
        if filtered_vocabulary.len() > self.max_features {
            let mut sorted_vocab: Vec<_> = filtered_vocabulary.iter().collect();
            sorted_vocab.sort_by(|a, b| {
                document_frequencies[a.0]
                    .cmp(&document_frequencies[b.0])
                    .reverse()
            });

            let mut new_vocabulary = HashMap::new();
            for (term, _) in sorted_vocab.iter().take(self.max_features) {
                new_vocabulary.insert(term.to_string(), new_vocabulary.len());
            }
            filtered_vocabulary = new_vocabulary;
        }

        // Compute IDF values
        let vocab_size = filtered_vocabulary.len();
        let mut idf_values = Array1::zeros(vocab_size);

        if self.use_tf_idf {
            for (term, &idx) in &filtered_vocabulary {
                let df = document_frequencies.get(term).unwrap_or(&0);
                idf_values[idx] = (n_documents as f64 / (*df as f64 + 1.0)).ln() + 1.0;
            }
        } else {
            idf_values.fill(1.0);
        }

        // Generate random weights for kernel approximation
        let mut rng = RealStdRng::from_seed(thread_rng().gen());
        let normal = RandNormal::new(0.0, 1.0).unwrap();

        let mut random_weights = Array2::zeros((self.n_components, vocab_size));
        for i in 0..self.n_components {
            for j in 0..vocab_size {
                random_weights[[i, j]] = rng.sample(normal);
            }
        }

        Ok(FittedTextKernelApproximation {
            n_components: self.n_components,
            max_features: self.max_features,
            ngram_range: self.ngram_range,
            min_df: self.min_df,
            max_df: self.max_df,
            use_tf_idf: self.use_tf_idf,
            use_hashing: self.use_hashing,
            sublinear_tf: self.sublinear_tf,
            vocabulary: filtered_vocabulary,
            idf_values,
            random_weights,
        })
    }
}

impl FittedTextKernelApproximation {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    fn extract_ngrams(&self, tokens: &[String]) -> Vec<String> {
        let mut ngrams = Vec::new();

        for n in self.ngram_range.0..=self.ngram_range.1 {
            for i in 0..=tokens.len().saturating_sub(n) {
                if i + n <= tokens.len() {
                    let ngram = tokens[i..i + n].join(" ");
                    ngrams.push(ngram);
                }
            }
        }

        ngrams
    }
}

impl Transform<Vec<String>, Array2<f64>> for FittedTextKernelApproximation {
    fn transform(&self, documents: &Vec<String>) -> Result<Array2<f64>> {
        let n_documents = documents.len();
        let vocab_size = self.vocabulary.len();

        // Convert documents to TF-IDF vectors
        let mut tf_idf_matrix = Array2::zeros((n_documents, vocab_size));

        for (doc_idx, doc) in documents.iter().enumerate() {
            let tokens = self.tokenize(doc);
            let ngrams = self.extract_ngrams(&tokens);

            // Count term frequencies
            let mut term_counts = HashMap::new();
            for ngram in ngrams {
                *term_counts.entry(ngram).or_insert(0) += 1;
            }

            // Compute TF-IDF
            for (term, &count) in &term_counts {
                if let Some(&vocab_idx) = self.vocabulary.get(term) {
                    let tf = if self.sublinear_tf {
                        1.0 + (count as f64).ln()
                    } else {
                        count as f64
                    };

                    let tf_idf = tf * self.idf_values[vocab_idx];
                    tf_idf_matrix[[doc_idx, vocab_idx]] = tf_idf;
                }
            }
        }

        // Apply random projection for kernel approximation
        let mut result = Array2::zeros((n_documents, self.n_components));

        for i in 0..n_documents {
            for j in 0..self.n_components {
                let mut dot_product = 0.0;
                for k in 0..vocab_size {
                    dot_product += tf_idf_matrix[[i, k]] * self.random_weights[[j, k]];
                }
                result[[i, j]] = dot_product;
            }
        }

        Ok(result)
    }
}

/// Semantic kernel approximation using word embeddings and similarity measures
#[derive(Debug, Clone, Serialize, Deserialize)]
/// SemanticKernelApproximation
pub struct SemanticKernelApproximation {
    /// n_components
    pub n_components: usize,
    /// embedding_dim
    pub embedding_dim: usize,
    /// similarity_measure
    pub similarity_measure: SimilarityMeasure,
    /// aggregation_method
    pub aggregation_method: AggregationMethod,
    /// use_attention
    pub use_attention: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// SimilarityMeasure
pub enum SimilarityMeasure {
    /// Cosine
    Cosine,
    /// Euclidean
    Euclidean,
    /// Manhattan
    Manhattan,
    /// Dot
    Dot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// AggregationMethod
pub enum AggregationMethod {
    /// Mean
    Mean,
    /// Max
    Max,
    /// Sum
    Sum,
    /// AttentionWeighted
    AttentionWeighted,
}

impl SemanticKernelApproximation {
    pub fn new(n_components: usize, embedding_dim: usize) -> Self {
        Self {
            n_components,
            embedding_dim,
            similarity_measure: SimilarityMeasure::Cosine,
            aggregation_method: AggregationMethod::Mean,
            use_attention: false,
        }
    }

    pub fn similarity_measure(mut self, measure: SimilarityMeasure) -> Self {
        self.similarity_measure = measure;
        self
    }

    pub fn aggregation_method(mut self, method: AggregationMethod) -> Self {
        self.aggregation_method = method;
        self
    }

    pub fn use_attention(mut self, use_attention: bool) -> Self {
        self.use_attention = use_attention;
        self
    }

    fn compute_similarity(&self, vec1: &ArrayView1<f64>, vec2: &ArrayView1<f64>) -> f64 {
        match self.similarity_measure {
            SimilarityMeasure::Cosine => {
                let dot = vec1.dot(vec2);
                let norm1 = vec1.dot(vec1).sqrt();
                let norm2 = vec2.dot(vec2).sqrt();
                if norm1 > 0.0 && norm2 > 0.0 {
                    dot / (norm1 * norm2)
                } else {
                    0.0
                }
            }
            SimilarityMeasure::Euclidean => {
                let diff = vec1 - vec2;
                -diff.dot(&diff).sqrt()
            }
            SimilarityMeasure::Manhattan => {
                let diff = vec1 - vec2;
                -diff.mapv(|x| x.abs()).sum()
            }
            SimilarityMeasure::Dot => vec1.dot(vec2),
        }
    }

    fn aggregate_embeddings(&self, embeddings: &Array2<f64>) -> Array1<f64> {
        match self.aggregation_method {
            AggregationMethod::Mean => embeddings.mean_axis(Axis(0)).unwrap(),
            AggregationMethod::Max => {
                let mut result = Array1::zeros(embeddings.ncols());
                for i in 0..embeddings.ncols() {
                    let col = embeddings.column(i);
                    result[i] = col.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                }
                result
            }
            AggregationMethod::Sum => embeddings.sum_axis(Axis(0)),
            AggregationMethod::AttentionWeighted => {
                // Simplified attention mechanism
                let n_tokens = embeddings.nrows();
                let mut attention_weights = Array1::zeros(n_tokens);

                for i in 0..n_tokens {
                    let token_embedding = embeddings.row(i);
                    attention_weights[i] = token_embedding.dot(&token_embedding).sqrt();
                }

                // Softmax
                let max_weight = attention_weights.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                attention_weights.mapv_inplace(|x| (x - max_weight).exp());
                let sum_weights = attention_weights.sum();
                if sum_weights > 0.0 {
                    attention_weights /= sum_weights;
                }

                // Weighted average
                let mut result = Array1::zeros(embeddings.ncols());
                for i in 0..n_tokens {
                    let token_embedding = embeddings.row(i);
                    for j in 0..embeddings.ncols() {
                        result[j] += attention_weights[i] * token_embedding[j];
                    }
                }
                result
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// FittedSemanticKernelApproximation
pub struct FittedSemanticKernelApproximation {
    /// n_components
    pub n_components: usize,
    /// embedding_dim
    pub embedding_dim: usize,
    /// similarity_measure
    pub similarity_measure: SimilarityMeasure,
    /// aggregation_method
    pub aggregation_method: AggregationMethod,
    /// use_attention
    pub use_attention: bool,
    /// word_embeddings
    pub word_embeddings: HashMap<String, Array1<f64>>,
    /// projection_matrix
    pub projection_matrix: Array2<f64>,
}

impl Fit<Vec<String>, ()> for SemanticKernelApproximation {
    type Fitted = FittedSemanticKernelApproximation;

    fn fit(self, documents: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        let mut rng = RealStdRng::from_seed(thread_rng().gen());
        let normal = RandNormal::new(0.0, 1.0).unwrap();

        // Generate random word embeddings (in practice, these would be pre-trained)
        let mut word_embeddings = HashMap::new();
        let mut vocabulary = HashSet::new();

        for doc in documents {
            let tokens: Vec<String> = doc
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            for token in tokens {
                vocabulary.insert(token);
            }
        }

        for word in vocabulary {
            let embedding = Array1::from_vec(
                (0..self.embedding_dim)
                    .map(|_| rng.sample(normal))
                    .collect(),
            );
            word_embeddings.insert(word, embedding);
        }

        // Generate projection matrix for kernel approximation
        let mut projection_matrix = Array2::zeros((self.n_components, self.embedding_dim));
        for i in 0..self.n_components {
            for j in 0..self.embedding_dim {
                projection_matrix[[i, j]] = rng.sample(normal);
            }
        }

        Ok(FittedSemanticKernelApproximation {
            n_components: self.n_components,
            embedding_dim: self.embedding_dim,
            similarity_measure: self.similarity_measure,
            aggregation_method: self.aggregation_method,
            use_attention: self.use_attention,
            word_embeddings,
            projection_matrix,
        })
    }
}

impl FittedSemanticKernelApproximation {
    fn aggregate_embeddings(&self, embeddings: &Array2<f64>) -> Array1<f64> {
        match self.aggregation_method {
            AggregationMethod::Mean => embeddings.mean_axis(Axis(0)).unwrap(),
            AggregationMethod::Max => {
                let mut result = Array1::zeros(embeddings.ncols());
                for i in 0..embeddings.ncols() {
                    let col = embeddings.column(i);
                    result[i] = col.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                }
                result
            }
            AggregationMethod::Sum => embeddings.sum_axis(Axis(0)),
            AggregationMethod::AttentionWeighted => {
                // Simplified attention mechanism
                let n_tokens = embeddings.nrows();
                let mut attention_weights = Array1::zeros(n_tokens);

                for i in 0..n_tokens {
                    let token_embedding = embeddings.row(i);
                    attention_weights[i] = token_embedding.dot(&token_embedding).sqrt();
                }

                // Softmax
                let max_weight = attention_weights.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                attention_weights.mapv_inplace(|x| (x - max_weight).exp());
                let sum_weights = attention_weights.sum();
                if sum_weights > 0.0 {
                    attention_weights /= sum_weights;
                }

                // Weighted average
                let mut result = Array1::zeros(embeddings.ncols());
                for i in 0..n_tokens {
                    let token_embedding = embeddings.row(i);
                    for j in 0..embeddings.ncols() {
                        result[j] += attention_weights[i] * token_embedding[j];
                    }
                }
                result
            }
        }
    }
}

impl Transform<Vec<String>, Array2<f64>> for FittedSemanticKernelApproximation {
    fn transform(&self, documents: &Vec<String>) -> Result<Array2<f64>> {
        let n_documents = documents.len();
        let mut result = Array2::zeros((n_documents, self.n_components));

        for (doc_idx, doc) in documents.iter().enumerate() {
            let tokens: Vec<String> = doc
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            // Get embeddings for tokens
            let mut token_embeddings = Vec::new();
            for token in tokens {
                if let Some(embedding) = self.word_embeddings.get(&token) {
                    token_embeddings.push(embedding.clone());
                }
            }

            if !token_embeddings.is_empty() {
                // Convert to Array2
                let embeddings_matrix = Array2::from_shape_vec(
                    (token_embeddings.len(), self.embedding_dim),
                    token_embeddings
                        .iter()
                        .flat_map(|e| e.iter().cloned())
                        .collect(),
                )?;

                // Aggregate embeddings
                let doc_embedding = self.aggregate_embeddings(&embeddings_matrix);

                // Apply projection for kernel approximation
                for i in 0..self.n_components {
                    let projected = self.projection_matrix.row(i).dot(&doc_embedding);
                    result[[doc_idx, i]] = projected.tanh(); // Non-linear activation
                }
            }
        }

        Ok(result)
    }
}

/// Syntactic kernel approximation using parse trees and grammatical features
#[derive(Debug, Clone, Serialize, Deserialize)]
/// SyntacticKernelApproximation
pub struct SyntacticKernelApproximation {
    /// n_components
    pub n_components: usize,
    /// max_tree_depth
    pub max_tree_depth: usize,
    /// use_pos_tags
    pub use_pos_tags: bool,
    /// use_dependencies
    pub use_dependencies: bool,
    /// tree_kernel_type
    pub tree_kernel_type: TreeKernelType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TreeKernelType
pub enum TreeKernelType {
    /// Subset
    Subset,
    /// Subsequence
    Subsequence,
    /// Partial
    Partial,
}

impl SyntacticKernelApproximation {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_tree_depth: 10,
            use_pos_tags: true,
            use_dependencies: true,
            tree_kernel_type: TreeKernelType::Subset,
        }
    }

    pub fn max_tree_depth(mut self, depth: usize) -> Self {
        self.max_tree_depth = depth;
        self
    }

    pub fn use_pos_tags(mut self, use_pos: bool) -> Self {
        self.use_pos_tags = use_pos;
        self
    }

    pub fn use_dependencies(mut self, use_deps: bool) -> Self {
        self.use_dependencies = use_deps;
        self
    }

    pub fn tree_kernel_type(mut self, kernel_type: TreeKernelType) -> Self {
        self.tree_kernel_type = kernel_type;
        self
    }

    fn extract_syntactic_features(&self, text: &str) -> Vec<String> {
        let mut features = Vec::new();

        // Simplified syntactic feature extraction
        let tokens: Vec<&str> = text.split_whitespace().collect();

        // Add POS tag features (simplified)
        if self.use_pos_tags {
            for token in &tokens {
                let pos_tag = self.simple_pos_tag(token);
                features.push(format!("POS_{}", pos_tag));
            }
        }

        // Add dependency features (simplified)
        if self.use_dependencies {
            for i in 0..tokens.len() {
                if i > 0 {
                    features.push(format!("DEP_{}_{}", tokens[i - 1], tokens[i]));
                }
            }
        }

        // Add n-gram features
        for n in 1..=3 {
            for i in 0..=tokens.len().saturating_sub(n) {
                if i + n <= tokens.len() {
                    let ngram = tokens[i..i + n].join("_");
                    features.push(format!("NGRAM_{}", ngram));
                }
            }
        }

        features
    }

    fn simple_pos_tag(&self, token: &str) -> String {
        // Very simplified POS tagging
        let token_lower = token.to_lowercase();

        if token_lower.ends_with("ing") {
            "VBG".to_string()
        } else if token_lower.ends_with("ed") {
            "VBD".to_string()
        } else if token_lower.ends_with("ly") {
            "RB".to_string()
        } else if token_lower.ends_with("s") && !token_lower.ends_with("ss") {
            "NNS".to_string()
        } else if token.chars().all(|c| c.is_alphabetic() && c.is_uppercase()) {
            "NNP".to_string()
        } else if token.chars().all(|c| c.is_alphabetic()) {
            "NN".to_string()
        } else if token.chars().all(|c| c.is_numeric()) {
            "CD".to_string()
        } else {
            "UNK".to_string()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// FittedSyntacticKernelApproximation
pub struct FittedSyntacticKernelApproximation {
    /// n_components
    pub n_components: usize,
    /// max_tree_depth
    pub max_tree_depth: usize,
    /// use_pos_tags
    pub use_pos_tags: bool,
    /// use_dependencies
    pub use_dependencies: bool,
    /// tree_kernel_type
    pub tree_kernel_type: TreeKernelType,
    /// feature_vocabulary
    pub feature_vocabulary: HashMap<String, usize>,
    /// random_weights
    pub random_weights: Array2<f64>,
}

impl Fit<Vec<String>, ()> for SyntacticKernelApproximation {
    type Fitted = FittedSyntacticKernelApproximation;

    fn fit(self, documents: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        let mut feature_vocabulary = HashMap::new();

        // Extract syntactic features from all documents
        for doc in documents {
            let features = self.extract_syntactic_features(doc);
            for feature in features {
                if !feature_vocabulary.contains_key(&feature) {
                    feature_vocabulary.insert(feature, feature_vocabulary.len());
                }
            }
        }

        // Generate random weights
        let mut rng = RealStdRng::from_seed(thread_rng().gen());
        let normal = RandNormal::new(0.0, 1.0).unwrap();

        let vocab_size = feature_vocabulary.len();
        let mut random_weights = Array2::zeros((self.n_components, vocab_size));

        for i in 0..self.n_components {
            for j in 0..vocab_size {
                random_weights[[i, j]] = rng.sample(normal);
            }
        }

        Ok(FittedSyntacticKernelApproximation {
            n_components: self.n_components,
            max_tree_depth: self.max_tree_depth,
            use_pos_tags: self.use_pos_tags,
            use_dependencies: self.use_dependencies,
            tree_kernel_type: self.tree_kernel_type,
            feature_vocabulary,
            random_weights,
        })
    }
}

impl FittedSyntacticKernelApproximation {
    fn extract_syntactic_features(&self, text: &str) -> Vec<String> {
        let mut features = Vec::new();

        // Simplified syntactic feature extraction
        let tokens: Vec<&str> = text.split_whitespace().collect();

        // Add POS tag features (simplified)
        if self.use_pos_tags {
            for token in &tokens {
                let pos_tag = self.simple_pos_tag(token);
                features.push(format!("POS_{}", pos_tag));
            }
        }

        // Add dependency features (simplified)
        if self.use_dependencies {
            for i in 0..tokens.len() {
                if i > 0 {
                    features.push(format!("DEP_{}_{}", tokens[i - 1], tokens[i]));
                }
            }
        }

        // Add n-gram features
        for n in 1..=3 {
            for i in 0..=tokens.len().saturating_sub(n) {
                if i + n <= tokens.len() {
                    let ngram = tokens[i..i + n].join("_");
                    features.push(format!("NGRAM_{}", ngram));
                }
            }
        }

        features
    }

    fn simple_pos_tag(&self, token: &str) -> String {
        // Very simplified POS tagging
        let token_lower = token.to_lowercase();

        if token_lower.ends_with("ing") {
            "VBG".to_string()
        } else if token_lower.ends_with("ed") {
            "VBD".to_string()
        } else if token_lower.ends_with("ly") {
            "RB".to_string()
        } else if token_lower.ends_with("s") && !token_lower.ends_with("ss") {
            "NNS".to_string()
        } else if token.chars().all(|c| c.is_alphabetic() && c.is_uppercase()) {
            "NNP".to_string()
        } else if token.chars().all(|c| c.is_alphabetic()) {
            "NN".to_string()
        } else if token.chars().all(|c| c.is_numeric()) {
            "CD".to_string()
        } else {
            "UNK".to_string()
        }
    }
}

impl Transform<Vec<String>, Array2<f64>> for FittedSyntacticKernelApproximation {
    fn transform(&self, documents: &Vec<String>) -> Result<Array2<f64>> {
        let n_documents = documents.len();
        let vocab_size = self.feature_vocabulary.len();

        // Convert documents to feature vectors
        let mut feature_matrix = Array2::zeros((n_documents, vocab_size));

        for (doc_idx, doc) in documents.iter().enumerate() {
            let features = self.extract_syntactic_features(doc);
            let mut feature_counts = HashMap::new();

            for feature in features {
                *feature_counts.entry(feature).or_insert(0) += 1;
            }

            for (feature, count) in feature_counts {
                if let Some(&vocab_idx) = self.feature_vocabulary.get(&feature) {
                    feature_matrix[[doc_idx, vocab_idx]] = count as f64;
                }
            }
        }

        // Apply random projection
        let mut result = Array2::zeros((n_documents, self.n_components));

        for i in 0..n_documents {
            for j in 0..self.n_components {
                let mut dot_product = 0.0;
                for k in 0..vocab_size {
                    dot_product += feature_matrix[[i, k]] * self.random_weights[[j, k]];
                }
                result[[i, j]] = dot_product.tanh();
            }
        }

        Ok(result)
    }
}

/// Document kernel approximation for document-level features
#[derive(Debug, Clone, Serialize, Deserialize)]
/// DocumentKernelApproximation
pub struct DocumentKernelApproximation {
    /// n_components
    pub n_components: usize,
    /// use_topic_features
    pub use_topic_features: bool,
    /// use_readability_features
    pub use_readability_features: bool,
    /// use_stylometric_features
    pub use_stylometric_features: bool,
    /// n_topics
    pub n_topics: usize,
}

impl DocumentKernelApproximation {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            use_topic_features: true,
            use_readability_features: true,
            use_stylometric_features: true,
            n_topics: 10,
        }
    }

    pub fn use_topic_features(mut self, use_topics: bool) -> Self {
        self.use_topic_features = use_topics;
        self
    }

    pub fn use_readability_features(mut self, use_readability: bool) -> Self {
        self.use_readability_features = use_readability;
        self
    }

    pub fn use_stylometric_features(mut self, use_stylometric: bool) -> Self {
        self.use_stylometric_features = use_stylometric;
        self
    }

    pub fn n_topics(mut self, n_topics: usize) -> Self {
        self.n_topics = n_topics;
        self
    }

    fn extract_document_features(&self, text: &str) -> Vec<f64> {
        let mut features = Vec::new();

        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();
        let words: Vec<&str> = text.split_whitespace().collect();
        let characters: Vec<char> = text.chars().collect();

        if self.use_readability_features {
            // Readability features
            let avg_sentence_length = if !sentences.is_empty() {
                words.len() as f64 / sentences.len() as f64
            } else {
                0.0
            };

            let avg_word_length = if !words.is_empty() {
                characters.len() as f64 / words.len() as f64
            } else {
                0.0
            };

            features.push(avg_sentence_length);
            features.push(avg_word_length);
            features.push(sentences.len() as f64);
            features.push(words.len() as f64);
        }

        if self.use_stylometric_features {
            // Stylometric features
            let punctuation_count = characters
                .iter()
                .filter(|c| c.is_ascii_punctuation())
                .count();
            let uppercase_count = characters.iter().filter(|c| c.is_uppercase()).count();
            let digit_count = characters.iter().filter(|c| c.is_numeric()).count();

            features.push(punctuation_count as f64 / characters.len() as f64);
            features.push(uppercase_count as f64 / characters.len() as f64);
            features.push(digit_count as f64 / characters.len() as f64);

            // Type-token ratio
            let unique_words: HashSet<&str> = words.iter().cloned().collect();
            let ttr = if !words.is_empty() {
                unique_words.len() as f64 / words.len() as f64
            } else {
                0.0
            };
            features.push(ttr);
        }

        if self.use_topic_features {
            // Simplified topic features (in practice, use LDA or similar)
            let mut topic_features = vec![0.0; self.n_topics];
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            let hash = hasher.finish();

            for i in 0..self.n_topics {
                topic_features[i] = ((hash + i as u64) % 1000) as f64 / 1000.0;
            }

            features.extend(topic_features);
        }

        features
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// FittedDocumentKernelApproximation
pub struct FittedDocumentKernelApproximation {
    /// n_components
    pub n_components: usize,
    /// use_topic_features
    pub use_topic_features: bool,
    /// use_readability_features
    pub use_readability_features: bool,
    /// use_stylometric_features
    pub use_stylometric_features: bool,
    /// n_topics
    pub n_topics: usize,
    /// feature_dim
    pub feature_dim: usize,
    /// random_weights
    pub random_weights: Array2<f64>,
}

impl Fit<Vec<String>, ()> for DocumentKernelApproximation {
    type Fitted = FittedDocumentKernelApproximation;

    fn fit(self, documents: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        // Determine feature dimension
        let sample_features = self.extract_document_features(&documents[0]);
        let feature_dim = sample_features.len();

        // Generate random weights
        let mut rng = RealStdRng::from_seed(thread_rng().gen());
        let normal = RandNormal::new(0.0, 1.0).unwrap();

        let mut random_weights = Array2::zeros((self.n_components, feature_dim));
        for i in 0..self.n_components {
            for j in 0..feature_dim {
                random_weights[[i, j]] = rng.sample(normal);
            }
        }

        Ok(FittedDocumentKernelApproximation {
            n_components: self.n_components,
            use_topic_features: self.use_topic_features,
            use_readability_features: self.use_readability_features,
            use_stylometric_features: self.use_stylometric_features,
            n_topics: self.n_topics,
            feature_dim,
            random_weights,
        })
    }
}

impl FittedDocumentKernelApproximation {
    fn extract_document_features(&self, text: &str) -> Vec<f64> {
        let mut features = Vec::new();

        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();
        let words: Vec<&str> = text.split_whitespace().collect();
        let characters: Vec<char> = text.chars().collect();

        if self.use_readability_features {
            // Readability features
            let avg_sentence_length = if !sentences.is_empty() {
                words.len() as f64 / sentences.len() as f64
            } else {
                0.0
            };

            let avg_word_length = if !words.is_empty() {
                characters.len() as f64 / words.len() as f64
            } else {
                0.0
            };

            features.push(avg_sentence_length);
            features.push(avg_word_length);
            features.push(sentences.len() as f64);
            features.push(words.len() as f64);
        }

        if self.use_stylometric_features {
            // Stylometric features
            let punctuation_count = characters
                .iter()
                .filter(|c| c.is_ascii_punctuation())
                .count();
            let uppercase_count = characters.iter().filter(|c| c.is_uppercase()).count();
            let digit_count = characters.iter().filter(|c| c.is_numeric()).count();

            features.push(punctuation_count as f64 / characters.len() as f64);
            features.push(uppercase_count as f64 / characters.len() as f64);
            features.push(digit_count as f64 / characters.len() as f64);

            // Type-token ratio
            let unique_words: HashSet<&str> = words.iter().cloned().collect();
            let ttr = if !words.is_empty() {
                unique_words.len() as f64 / words.len() as f64
            } else {
                0.0
            };
            features.push(ttr);
        }

        if self.use_topic_features {
            // Simplified topic features (in practice, use LDA or similar)
            let mut topic_features = vec![0.0; self.n_topics];
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            let hash = hasher.finish();

            for i in 0..self.n_topics {
                topic_features[i] = ((hash + i as u64) % 1000) as f64 / 1000.0;
            }

            features.extend(topic_features);
        }

        features
    }
}

impl Transform<Vec<String>, Array2<f64>> for FittedDocumentKernelApproximation {
    fn transform(&self, documents: &Vec<String>) -> Result<Array2<f64>> {
        let n_documents = documents.len();
        let mut result = Array2::zeros((n_documents, self.n_components));

        for (doc_idx, doc) in documents.iter().enumerate() {
            let features = self.extract_document_features(doc);
            let feature_array = Array1::from_vec(features);

            for i in 0..self.n_components {
                let projected = self.random_weights.row(i).dot(&feature_array);
                result[[doc_idx, i]] = projected.tanh();
            }
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_kernel_approximation() {
        let docs = vec![
            "This is a test document".to_string(),
            "Another test document here".to_string(),
            "Third document for testing".to_string(),
        ];

        let text_kernel = TextKernelApproximation::new(50);
        let fitted = text_kernel.fit(&docs, &()).unwrap();
        let transformed = fitted.transform(&docs).unwrap();

        assert_eq!(transformed.shape()[0], 3);
        assert_eq!(transformed.shape()[1], 50);
    }

    #[test]
    fn test_semantic_kernel_approximation() {
        let docs = vec![
            "Semantic similarity test".to_string(),
            "Another semantic test".to_string(),
        ];

        let semantic_kernel = SemanticKernelApproximation::new(30, 100);
        let fitted = semantic_kernel.fit(&docs, &()).unwrap();
        let transformed = fitted.transform(&docs).unwrap();

        assert_eq!(transformed.shape()[0], 2);
        assert_eq!(transformed.shape()[1], 30);
    }

    #[test]
    fn test_syntactic_kernel_approximation() {
        let docs = vec![
            "The cat sat on the mat".to_string(),
            "Dogs are running quickly".to_string(),
        ];

        let syntactic_kernel = SyntacticKernelApproximation::new(40);
        let fitted = syntactic_kernel.fit(&docs, &()).unwrap();
        let transformed = fitted.transform(&docs).unwrap();

        assert_eq!(transformed.shape()[0], 2);
        assert_eq!(transformed.shape()[1], 40);
    }

    #[test]
    fn test_document_kernel_approximation() {
        let docs = vec![
            "This is a long document with multiple sentences. It contains various features."
                .to_string(),
            "Short doc.".to_string(),
        ];

        let doc_kernel = DocumentKernelApproximation::new(25);
        let fitted = doc_kernel.fit(&docs, &()).unwrap();
        let transformed = fitted.transform(&docs).unwrap();

        assert_eq!(transformed.shape()[0], 2);
        assert_eq!(transformed.shape()[1], 25);
    }
}

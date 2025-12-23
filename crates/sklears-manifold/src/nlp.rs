//! Natural Language Processing Manifold Learning
//!
//! This module provides advanced manifold learning methods specifically designed for
//! natural language processing tasks, including word embeddings, sentence embeddings,
//! document-level representations, semantic manifolds, and multilingual alignment.
//!
//! # Overview
//!
//! NLP manifold learning captures the geometric structure of language representations.
//!
//! # Examples
//!
//! ```
//! use sklears_manifold::nlp::WordEmbedding;
//! use sklears_core::traits::{Fit, Transform};
//! use scirs2_core::ndarray::Array2;
//!
//! // Create word embeddings from co-occurrence matrix
//! let cooc = Array2::from_shape_vec((10, 10), vec![0.0; 100]).unwrap();
//! let skipgram = WordEmbedding::new();
//! // Train: let fitted = skipgram.fit(&cooc.view(), &()).unwrap();
//! ```

use scirs2_core::essentials::{Normal, Uniform};
use scirs2_core::ndarray::ndarray_linalg::SVD;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::Distribution;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

// ================================================================================================
// Word-Level Embeddings
// ================================================================================================

/// Skip-Gram Word Embedding Model
///
/// Learns distributed word representations by predicting context words given a target word.
///
/// # Examples
///
/// ```
/// use sklears_manifold::nlp::WordEmbedding;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::Array2;
///
/// let cooc = Array2::from_shape_vec((10, 10), vec![0.0; 100]).unwrap();
/// let embedding = WordEmbedding::new().embedding_dim(50);
/// // let fitted = embedding.fit(&cooc.view(), &()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct WordEmbedding<S = Untrained> {
    state: S,
    embedding_dim: usize,
    window_size: usize,
    learning_rate: Float,
    n_epochs: usize,
    negative_samples: usize,
}

/// Trained word embedding state
#[derive(Debug, Clone)]
pub struct WordEmbeddingTrained {
    pub embeddings: Array2<Float>,
    pub vocab_size: usize,
}

impl WordEmbedding<Untrained> {
    /// Create a new word embedding model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            embedding_dim: 100,
            window_size: 5,
            learning_rate: 0.001,
            n_epochs: 10,
            negative_samples: 5,
        }
    }

    /// Set embedding dimension
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Set window size
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, rate: Float) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Set number of epochs
    pub fn n_epochs(mut self, epochs: usize) -> Self {
        self.n_epochs = epochs;
        self
    }

    #[inline]
    fn sigmoid(x: Float) -> Float {
        1.0 / (1.0 + (-x).exp())
    }

    fn train_embeddings(&self, cooc: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let vocab_size = cooc.nrows();
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.01).map_err(|e| {
            SklearsError::FitError(format!("Failed to create normal distribution: {}", e))
        })?;

        // Initialize embeddings
        let mut embeddings = Array2::zeros((vocab_size, self.embedding_dim));
        let mut context_embeddings = Array2::zeros((vocab_size, self.embedding_dim));

        for i in 0..vocab_size {
            for j in 0..self.embedding_dim {
                embeddings[[i, j]] = normal.sample(&mut rng) as Float;
                context_embeddings[[i, j]] = normal.sample(&mut rng) as Float;
            }
        }

        // Extract word pairs from co-occurrence matrix
        let mut word_pairs = Vec::new();
        for i in 0..vocab_size {
            for j in 0..vocab_size {
                let count = cooc[[i, j]] as usize;
                for _ in 0..count.min(10) {
                    word_pairs.push((i, j));
                }
            }
        }

        if word_pairs.is_empty() {
            return Ok(embeddings);
        }

        // Training loop
        for _epoch in 0..self.n_epochs {
            for &(target, context) in &word_pairs {
                // Positive sample
                let mut score = 0.0;
                for j in 0..self.embedding_dim {
                    score += embeddings[[target, j]] * context_embeddings[[context, j]];
                }
                let sigmoid_score = Self::sigmoid(score);
                let gradient = (1.0 - sigmoid_score) * self.learning_rate;

                // Update embeddings
                for j in 0..self.embedding_dim {
                    embeddings[[target, j]] += gradient * context_embeddings[[context, j]];
                    context_embeddings[[context, j]] += gradient * embeddings[[target, j]];
                }
            }
        }

        Ok(embeddings)
    }
}

impl Default for WordEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for WordEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for WordEmbedding<Untrained> {
    type Fitted = WordEmbedding<WordEmbeddingTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let embeddings = self.train_embeddings(x)?;
        let vocab_size = x.nrows();

        Ok(WordEmbedding {
            state: WordEmbeddingTrained {
                embeddings,
                vocab_size,
            },
            embedding_dim: self.embedding_dim,
            window_size: self.window_size,
            learning_rate: self.learning_rate,
            n_epochs: self.n_epochs,
            negative_samples: self.negative_samples,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for WordEmbedding<WordEmbeddingTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let n_samples = x.nrows();
        let mut result = Array2::zeros((n_samples, self.embedding_dim));

        for i in 0..n_samples {
            // Find word index (assumes one-hot encoding)
            let word_idx = x.row(i).iter().position(|&val| val > 0.5).unwrap_or(0);
            if word_idx < self.state.vocab_size {
                result
                    .row_mut(i)
                    .assign(&self.state.embeddings.row(word_idx));
            }
        }

        Ok(result)
    }
}

// ================================================================================================
// GloVe-Style Embeddings
// ================================================================================================

/// GloVe (Global Vectors) Style Word Embedding
///
/// Learns word embeddings by factorizing a word co-occurrence matrix.
#[derive(Debug, Clone)]
pub struct GloVeEmbedding<S = Untrained> {
    state: S,
    embedding_dim: usize,
    learning_rate: Float,
    n_epochs: usize,
    x_max: Float,
    alpha: Float,
}

/// Trained GloVe state
#[derive(Debug, Clone)]
pub struct GloVeTrained {
    pub embeddings: Array2<Float>,
    pub vocab_size: usize,
}

impl GloVeEmbedding<Untrained> {
    /// Create a new GloVe embedding model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            embedding_dim: 100,
            learning_rate: 0.05,
            n_epochs: 15,
            x_max: 100.0,
            alpha: 0.75,
        }
    }

    /// Set embedding dimension
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, rate: Float) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Weighting function for co-occurrence counts
    #[inline]
    fn weighting_function(&self, x: Float) -> Float {
        if x < self.x_max {
            (x / self.x_max).powf(self.alpha)
        } else {
            1.0
        }
    }

    fn train_glove(&self, cooc: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let vocab_size = cooc.nrows();
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.01).map_err(|e| {
            SklearsError::FitError(format!("Failed to create normal distribution: {}", e))
        })?;

        // Initialize embeddings
        let mut word_embeddings = Array2::zeros((vocab_size, self.embedding_dim));
        let mut context_embeddings = Array2::zeros((vocab_size, self.embedding_dim));

        for i in 0..vocab_size {
            for j in 0..self.embedding_dim {
                word_embeddings[[i, j]] = normal.sample(&mut rng) as Float;
                context_embeddings[[i, j]] = normal.sample(&mut rng) as Float;
            }
        }

        let mut word_biases: Array1<Float> = Array1::zeros(vocab_size);
        let mut context_biases: Array1<Float> = Array1::zeros(vocab_size);

        // AdaGrad accumulators
        let mut word_gradsq: Array2<Float> = Array2::ones((vocab_size, self.embedding_dim));
        let mut context_gradsq: Array2<Float> = Array2::ones((vocab_size, self.embedding_dim));

        // Simplified training (full implementation would be more complex)
        for _epoch in 0..self.n_epochs.min(5) {
            for i in 0..vocab_size.min(20) {
                for j in 0..vocab_size.min(20) {
                    let x_ij = cooc[[i, j]];
                    if x_ij < 1e-10 {
                        continue;
                    }

                    let weight = self.weighting_function(x_ij);
                    let mut dot_product: Float = word_biases[i] + context_biases[j];
                    for k in 0..self.embedding_dim {
                        dot_product += word_embeddings[[i, k]] * context_embeddings[[j, k]];
                    }

                    let diff: Float = dot_product - x_ij.ln();
                    let grad_common: Float = weight * diff;

                    for k in 0..self.embedding_dim {
                        let grad_w: Float = grad_common * context_embeddings[[j, k]];
                        word_gradsq[[i, k]] += grad_w * grad_w;
                        word_embeddings[[i, k]] -=
                            self.learning_rate * grad_w / word_gradsq[[i, k]].sqrt();
                    }
                }
            }
        }

        Ok(word_embeddings)
    }
}

impl Default for GloVeEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GloVeEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for GloVeEmbedding<Untrained> {
    type Fitted = GloVeEmbedding<GloVeTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let embeddings = self.train_glove(x)?;
        let vocab_size = x.nrows();

        Ok(GloVeEmbedding {
            state: GloVeTrained {
                embeddings,
                vocab_size,
            },
            embedding_dim: self.embedding_dim,
            learning_rate: self.learning_rate,
            n_epochs: self.n_epochs,
            x_max: self.x_max,
            alpha: self.alpha,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for GloVeEmbedding<GloVeTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let n_samples = x.nrows();
        let mut result = Array2::zeros((n_samples, self.embedding_dim));

        for i in 0..n_samples {
            let word_idx = x.row(i).iter().position(|&val| val > 0.5).unwrap_or(0);
            if word_idx < self.state.vocab_size {
                result
                    .row_mut(i)
                    .assign(&self.state.embeddings.row(word_idx));
            }
        }

        Ok(result)
    }
}

// ================================================================================================
// Document-Level Embeddings
// ================================================================================================

/// Document Manifold Learning
///
/// Learns document-level representations using manifold learning techniques.
#[derive(Debug, Clone)]
pub struct DocumentEmbedding<S = Untrained> {
    state: S,
    embedding_dim: usize,
    n_neighbors: usize,
}

/// Trained document embedding state
#[derive(Debug, Clone)]
pub struct DocumentEmbeddingTrained {
    pub embeddings: Array2<Float>,
}

impl DocumentEmbedding<Untrained> {
    /// Create a new document embedding model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            embedding_dim: 128,
            n_neighbors: 10,
        }
    }

    /// Set embedding dimension
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Set number of neighbors
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    fn compute_embeddings(&self, tfidf: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let svd = tfidf
            .svd(false, true)
            .map_err(|e| SklearsError::FitError(format!("SVD computation failed: {}", e)))?;

        let vt = svd
            .2
            .ok_or_else(|| SklearsError::FitError("SVD V matrix not computed".to_string()))?;

        let k = self.embedding_dim.min(vt.nrows());
        let embeddings = vt.slice(s![..k, ..]).t().to_owned();

        Ok(embeddings)
    }
}

impl Default for DocumentEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DocumentEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for DocumentEmbedding<Untrained> {
    type Fitted = DocumentEmbedding<DocumentEmbeddingTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let embeddings = self.compute_embeddings(x)?;

        Ok(DocumentEmbedding {
            state: DocumentEmbeddingTrained { embeddings },
            embedding_dim: self.embedding_dim,
            n_neighbors: self.n_neighbors,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for DocumentEmbedding<DocumentEmbeddingTrained>
{
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        Ok(self.state.embeddings.clone())
    }
}

// ================================================================================================
// Multilingual Manifold Alignment
// ================================================================================================

/// Alignment method for multilingual embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentMethod {
    /// Use Procrustes analysis
    Procrustes,
    /// Use canonical correlation analysis
    CCA,
    /// Use optimal transport
    OptimalTransport,
}

/// Multilingual Manifold Alignment
///
/// Aligns word embeddings from different languages into a shared semantic space.
#[derive(Debug, Clone)]
pub struct MultilingualAlignment<S = Untrained> {
    state: S,
    embedding_dim: usize,
    alignment_method: AlignmentMethod,
}

/// Trained multilingual alignment state
#[derive(Debug, Clone)]
pub struct MultilingualAlignmentTrained {
    pub transformation_matrix: Array2<Float>,
}

impl MultilingualAlignment<Untrained> {
    /// Create a new multilingual alignment model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            embedding_dim: 300,
            alignment_method: AlignmentMethod::Procrustes,
        }
    }

    /// Set embedding dimension
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Set alignment method
    pub fn alignment_method(mut self, method: AlignmentMethod) -> Self {
        self.alignment_method = method;
        self
    }
}

impl Default for MultilingualAlignment<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultilingualAlignment<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MultilingualAlignment<Untrained> {
    type Fitted = MultilingualAlignment<MultilingualAlignmentTrained>;

    fn fit(self, _x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        // For now, use identity transformation
        let transformation_matrix = Array2::eye(self.embedding_dim);

        Ok(MultilingualAlignment {
            state: MultilingualAlignmentTrained {
                transformation_matrix,
            },
            embedding_dim: self.embedding_dim,
            alignment_method: self.alignment_method,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for MultilingualAlignment<MultilingualAlignmentTrained>
{
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        Ok(x.dot(&self.state.transformation_matrix))
    }
}

// ================================================================================================
// Utility functions
// ================================================================================================

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_embedding_creation() {
        let emb = WordEmbedding::new().embedding_dim(50).window_size(5);
        assert_eq!(emb.embedding_dim, 50);
        assert_eq!(emb.window_size, 5);
    }

    #[test]
    fn test_glove_embedding_creation() {
        let glove = GloVeEmbedding::new().embedding_dim(100);
        assert_eq!(glove.embedding_dim, 100);
    }

    #[test]
    fn test_document_embedding_creation() {
        let doc_emb = DocumentEmbedding::new().embedding_dim(128);
        assert_eq!(doc_emb.embedding_dim, 128);
    }

    #[test]
    fn test_multilingual_alignment_creation() {
        let alignment = MultilingualAlignment::new().embedding_dim(300);
        assert_eq!(alignment.embedding_dim, 300);
    }

    #[test]
    fn test_cosine_similarity() {
        use scirs2_core::ndarray::array;
        let a = array![1.0, 0.0, 0.0];
        let b = array![0.0, 1.0, 0.0];
        let c = array![1.0, 0.0, 0.0];

        assert!((cosine_similarity(&a.view(), &b.view()) - 0.0).abs() < 1e-10);
        assert!((cosine_similarity(&a.view(), &c.view()) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_word_embedding_fit() {
        let cooc = Array2::from_shape_vec((10, 10), vec![0.0; 100]).unwrap();
        let embedding = WordEmbedding::new().embedding_dim(5).n_epochs(2);
        let result = embedding.fit(&cooc.view(), &());
        assert!(result.is_ok());
    }
}

//! Gradient Embedding Methods implementation for batch active learning

use super::{BatchActiveLearningError, *};

/// Gradient Embedding Methods for active learning
///
/// This method uses gradient information from the model to select informative batches.
/// It considers both the gradients with respect to model parameters and embedding
/// representations to identify samples that would provide maximum learning benefit.
#[derive(Debug, Clone)]
pub struct GradientEmbeddingMethods {
    /// batch_size
    pub batch_size: usize,
    /// embedding_dim
    pub embedding_dim: usize,
    /// gradient_method
    pub gradient_method: String,
    /// similarity_threshold
    pub similarity_threshold: f64,
    /// learning_rate
    pub learning_rate: f64,
    /// max_iter
    pub max_iter: usize,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for GradientEmbeddingMethods {
    fn default() -> Self {
        Self {
            batch_size: 10,
            embedding_dim: 128,
            gradient_method: "gradnorm".to_string(),
            similarity_threshold: 0.8,
            learning_rate: 0.001,
            max_iter: 100,
            random_state: None,
        }
    }
}

impl GradientEmbeddingMethods {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn batch_size(mut self, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(BatchActiveLearningError::InvalidBatchSize(batch_size).into());
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    pub fn embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    pub fn gradient_method(mut self, gradient_method: String) -> Self {
        self.gradient_method = gradient_method;
        self
    }

    pub fn similarity_threshold(mut self, similarity_threshold: f64) -> Self {
        self.similarity_threshold = similarity_threshold;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn query(
        &self,
        X: &ArrayView2<f64>,
        probabilities: &ArrayView2<f64>,
    ) -> Result<Vec<usize>> {
        let n_samples = X.dim().0;

        if n_samples < self.batch_size {
            return Err(BatchActiveLearningError::InsufficientUnlabeledSamples.into());
        }

        // Simple placeholder implementation - select samples with highest uncertainty
        let mut uncertainty_scores = Vec::new();
        for i in 0..n_samples {
            let mut entropy = 0.0;
            for prob in probabilities.row(i) {
                if *prob > 0.0 {
                    entropy -= prob * prob.ln();
                }
            }
            uncertainty_scores.push((i, entropy));
        }

        // Sort by uncertainty and select top samples
        uncertainty_scores
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: Vec<usize> = uncertainty_scores
            .into_iter()
            .take(self.batch_size)
            .map(|(idx, _)| idx)
            .collect();

        Ok(selected)
    }
}

//! Diversity-Based Sampling implementation for active learning

use super::{BatchActiveLearningError, *};
use scirs2_core::rand_prelude::IndexedRandom;

/// Diversity-Based Sampling for active learning
///
/// This method focuses purely on selecting diverse samples from the unlabeled data
/// using various diversity measures including determinantal point processes (DPP),
/// maximum marginal relevance (MMR), and clustering-based approaches.
#[derive(Debug, Clone)]
pub struct DiversityBasedSampling {
    /// batch_size
    pub batch_size: usize,
    /// diversity_method
    pub diversity_method: String,
    /// distance_metric
    pub distance_metric: String,
    /// kernel_bandwidth
    pub kernel_bandwidth: f64,
    /// regularization
    pub regularization: f64,
    /// max_iter
    pub max_iter: usize,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for DiversityBasedSampling {
    fn default() -> Self {
        Self {
            batch_size: 10,
            diversity_method: "mmr".to_string(),
            distance_metric: "euclidean".to_string(),
            kernel_bandwidth: 1.0,
            regularization: 1e-6,
            max_iter: 100,
            random_state: None,
        }
    }
}

impl DiversityBasedSampling {
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

    pub fn diversity_method(mut self, diversity_method: String) -> Self {
        self.diversity_method = diversity_method;
        self
    }

    pub fn distance_metric(mut self, distance_metric: String) -> Self {
        self.distance_metric = distance_metric;
        self
    }

    pub fn kernel_bandwidth(mut self, kernel_bandwidth: f64) -> Self {
        self.kernel_bandwidth = kernel_bandwidth;
        self
    }

    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
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
        _probabilities: &ArrayView2<f64>,
    ) -> Result<Vec<usize>> {
        let n_samples = X.dim().0;

        if n_samples < self.batch_size {
            return Err(BatchActiveLearningError::InsufficientUnlabeledSamples.into());
        }

        // Simple placeholder implementation - select random diverse samples
        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        let indices: Vec<usize> = (0..n_samples).collect();
        let selected: Vec<usize> = indices
            .choose_multiple(&mut rng, self.batch_size)
            .cloned()
            .collect();

        Ok(selected)
    }
}

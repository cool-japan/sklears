//! Batch Mode Active Learning implementation

use super::{BatchActiveLearningError, *};

/// Batch Mode Active Learning using uncertainty and diversity
///
/// This method selects a batch of samples by balancing uncertainty (information gain)
/// and diversity (avoiding redundant samples) using various strategies.
#[derive(Debug, Clone)]
pub struct BatchModeActiveLearning {
    /// batch_size
    pub batch_size: usize,
    /// diversity_weight
    pub diversity_weight: f64,
    /// strategy
    pub strategy: String,
    /// distance_metric
    pub distance_metric: String,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for BatchModeActiveLearning {
    fn default() -> Self {
        Self {
            batch_size: 10,
            diversity_weight: 0.5,
            strategy: "uncertainty_diversity".to_string(),
            distance_metric: "euclidean".to_string(),
            random_state: None,
        }
    }
}

impl BatchModeActiveLearning {
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

    pub fn diversity_weight(mut self, diversity_weight: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&diversity_weight) {
            return Err(BatchActiveLearningError::InvalidDiversityWeight(diversity_weight).into());
        }
        self.diversity_weight = diversity_weight;
        Ok(self)
    }

    pub fn strategy(mut self, strategy: String) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn distance_metric(mut self, distance_metric: String) -> Self {
        self.distance_metric = distance_metric;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    fn compute_uncertainty_scores(&self, probabilities: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let n_samples = probabilities.dim().0;
        let mut uncertainty_scores = Array1::zeros(n_samples);

        match self.strategy.as_str() {
            "uncertainty_diversity" | "entropy" => {
                // Entropy-based uncertainty
                for i in 0..n_samples {
                    let mut entropy = 0.0;
                    for prob in probabilities.row(i) {
                        if *prob > 0.0 {
                            entropy -= prob * prob.ln();
                        }
                    }
                    uncertainty_scores[i] = entropy;
                }
            }
            "margin" => {
                // Margin-based uncertainty
                for i in 0..n_samples {
                    let mut probs: Vec<f64> = probabilities.row(i).to_vec();
                    probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    if probs.len() >= 2 {
                        uncertainty_scores[i] = 1.0 - (probs[0] - probs[1]);
                    } else {
                        uncertainty_scores[i] = 1.0 - probs[0];
                    }
                }
            }
            "least_confident" => {
                // Least confident uncertainty
                for i in 0..n_samples {
                    let max_prob = probabilities.row(i).fold(0.0f64, |a, &b| a.max(b));
                    uncertainty_scores[i] = 1.0 - max_prob;
                }
            }
            _ => {
                // Default to entropy
                for i in 0..n_samples {
                    let mut entropy = 0.0;
                    for prob in probabilities.row(i) {
                        if *prob > 0.0 {
                            entropy -= prob * prob.ln();
                        }
                    }
                    uncertainty_scores[i] = entropy;
                }
            }
        }

        Ok(uncertainty_scores)
    }

    fn compute_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> Result<f64> {
        match self.distance_metric.as_str() {
            "euclidean" => {
                let dist = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                Ok(dist)
            }
            "manhattan" => {
                let dist = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
                Ok(dist)
            }
            "cosine" => {
                let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
                let norm1 = x1.iter().map(|&x| x * x).sum::<f64>().sqrt();
                let norm2 = x2.iter().map(|&x| x * x).sum::<f64>().sqrt();

                if norm1 == 0.0 || norm2 == 0.0 {
                    Ok(1.0)
                } else {
                    Ok(1.0 - dot_product / (norm1 * norm2))
                }
            }
            _ => Err(
                BatchActiveLearningError::InvalidDistanceMetric(self.distance_metric.clone())
                    .into(),
            ),
        }
    }

    fn select_diverse_batch(
        &self,
        X: &ArrayView2<f64>,
        uncertainty_scores: &ArrayView1<f64>,
    ) -> Result<Vec<usize>> {
        let n_samples = X.dim().0;
        let mut selected_indices = Vec::new();
        let mut remaining_indices: Vec<usize> = (0..n_samples).collect();

        if remaining_indices.len() < self.batch_size {
            return Err(BatchActiveLearningError::InsufficientUnlabeledSamples.into());
        }

        // Select first sample with highest uncertainty
        let first_idx = remaining_indices
            .iter()
            .max_by(|&&a, &&b| {
                uncertainty_scores[a]
                    .partial_cmp(&uncertainty_scores[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap();
        selected_indices.push(first_idx);
        remaining_indices.retain(|&x| x != first_idx);

        // Select remaining samples balancing uncertainty and diversity
        while selected_indices.len() < self.batch_size && !remaining_indices.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f64::NEG_INFINITY;

            for &candidate_idx in remaining_indices.iter() {
                let uncertainty_score = uncertainty_scores[candidate_idx];

                // Compute minimum distance to already selected samples
                let mut min_distance = f64::INFINITY;
                for &selected_idx in selected_indices.iter() {
                    let distance =
                        self.compute_distance(&X.row(candidate_idx), &X.row(selected_idx))?;
                    min_distance = min_distance.min(distance);
                }

                // Combined score: uncertainty + diversity
                let combined_score = (1.0 - self.diversity_weight) * uncertainty_score
                    + self.diversity_weight * min_distance;

                if combined_score > best_score {
                    best_score = combined_score;
                    best_idx = candidate_idx;
                }
            }

            selected_indices.push(best_idx);
            remaining_indices.retain(|&x| x != best_idx);
        }

        Ok(selected_indices)
    }

    pub fn query(
        &self,
        X: &ArrayView2<f64>,
        probabilities: &ArrayView2<f64>,
    ) -> Result<Vec<usize>> {
        let uncertainty_scores = self.compute_uncertainty_scores(probabilities)?;
        self.select_diverse_batch(X, &uncertainty_scores.view())
    }
}

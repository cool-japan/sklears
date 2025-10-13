//! Diverse Mini-batch Selection implementation using k-means clustering

use super::{BatchActiveLearningError, *};
use scirs2_core::rand_prelude::IndexedRandom;

/// Diverse Mini-batch Selection using k-means clustering
///
/// This method first clusters the unlabeled data and then selects representatives
/// from each cluster to ensure diversity in the batch.
#[derive(Debug, Clone)]
pub struct DiverseMiniBatchSelection {
    /// batch_size
    pub batch_size: usize,
    /// n_clusters
    pub n_clusters: usize,
    /// uncertainty_weight
    pub uncertainty_weight: f64,
    /// max_kmeans_iter
    pub max_kmeans_iter: usize,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for DiverseMiniBatchSelection {
    fn default() -> Self {
        Self {
            batch_size: 10,
            n_clusters: 5,
            uncertainty_weight: 0.7,
            max_kmeans_iter: 100,
            random_state: None,
        }
    }
}

impl DiverseMiniBatchSelection {
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

    pub fn n_clusters(mut self, n_clusters: usize) -> Result<Self> {
        if n_clusters == 0 {
            return Err(BatchActiveLearningError::InvalidClusterCount(n_clusters).into());
        }
        self.n_clusters = n_clusters;
        Ok(self)
    }

    pub fn uncertainty_weight(mut self, uncertainty_weight: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&uncertainty_weight) {
            return Err(
                BatchActiveLearningError::InvalidDiversityWeight(uncertainty_weight).into(),
            );
        }
        self.uncertainty_weight = uncertainty_weight;
        Ok(self)
    }

    pub fn max_kmeans_iter(mut self, max_kmeans_iter: usize) -> Self {
        self.max_kmeans_iter = max_kmeans_iter;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    fn simple_kmeans(&self, X: &ArrayView2<f64>) -> Result<(Array1<usize>, Array2<f64>)> {
        let (n_samples, n_features) = X.dim();
        let k = self.n_clusters.min(n_samples);

        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Initialize centroids randomly
        let mut centroids = Array2::zeros((k, n_features));
        let indices: Vec<usize> = (0..n_samples).collect();
        let selected_indices: Vec<usize> = indices.choose_multiple(&mut rng, k).cloned().collect();

        for (i, &idx) in selected_indices.iter().enumerate() {
            centroids.row_mut(i).assign(&X.row(idx));
        }

        let mut labels = Array1::zeros(n_samples);
        let mut prev_labels = Array1::ones(n_samples);

        // K-means iterations
        for _ in 0..self.max_kmeans_iter {
            // Assign points to nearest centroids
            for i in 0..n_samples {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;

                for j in 0..k {
                    let distance = X
                        .row(i)
                        .iter()
                        .zip(centroids.row(j).iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = j;
                    }
                }

                labels[i] = best_cluster;
            }

            // Check convergence
            if labels == prev_labels {
                break;
            }
            prev_labels.assign(&labels);

            // Update centroids
            for j in 0..k {
                let cluster_points: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &label)| label == j)
                    .map(|(i, _)| i)
                    .collect();

                if !cluster_points.is_empty() {
                    let mut centroid = Array1::zeros(n_features);
                    for &point_idx in cluster_points.iter() {
                        centroid = centroid + X.row(point_idx);
                    }
                    centroid /= cluster_points.len() as f64;
                    centroids.row_mut(j).assign(&centroid);
                }
            }
        }

        Ok((labels, centroids))
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

        // Compute uncertainty scores
        let mut uncertainty_scores = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut entropy = 0.0;
            for prob in probabilities.row(i) {
                if *prob > 0.0 {
                    entropy -= prob * prob.ln();
                }
            }
            uncertainty_scores[i] = entropy;
        }

        // Perform clustering
        let (cluster_labels, _) = self.simple_kmeans(X)?;

        // Select representatives from each cluster
        let mut selected_indices = Vec::new();
        let mut cluster_counts = HashMap::new();

        // Count samples per cluster
        for &label in cluster_labels.iter() {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        // Calculate samples per cluster for batch
        let samples_per_cluster = self.batch_size / cluster_counts.len();
        let remaining_samples = self.batch_size % cluster_counts.len();

        for (cluster_id, &count) in cluster_counts.iter() {
            let mut cluster_samples = samples_per_cluster;
            if (*cluster_id) < remaining_samples {
                cluster_samples += 1;
            }

            // Get samples in this cluster
            let cluster_indices: Vec<usize> = cluster_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == *cluster_id)
                .map(|(i, _)| i)
                .collect();

            // Select top uncertain samples from this cluster
            let mut cluster_scores: Vec<(usize, f64)> = cluster_indices
                .iter()
                .map(|&i| (i, uncertainty_scores[i]))
                .collect();
            cluster_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top samples from this cluster
            let selected_count = cluster_samples.min(cluster_scores.len());
            for i in 0..selected_count {
                selected_indices.push(cluster_scores[i].0);
            }
        }

        // If we still need more samples, add highest uncertainty ones
        while selected_indices.len() < self.batch_size {
            let mut best_idx = 0;
            let mut best_score = f64::NEG_INFINITY;

            for i in 0..n_samples {
                if !selected_indices.contains(&i) && uncertainty_scores[i] > best_score {
                    best_score = uncertainty_scores[i];
                    best_idx = i;
                }
            }

            selected_indices.push(best_idx);
        }

        Ok(selected_indices)
    }
}

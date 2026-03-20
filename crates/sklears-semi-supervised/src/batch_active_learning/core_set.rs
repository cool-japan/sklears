//! Core-set approach implementation for batch active learning

use super::{BatchActiveLearningError, *};

/// Core-set approach for batch active learning
///
/// This method selects a batch of samples that best represents the unlabeled data
/// distribution, acting as a "core-set" or summary of the data.
#[derive(Debug, Clone)]
pub struct CoreSetApproach {
    /// batch_size
    pub batch_size: usize,
    /// distance_metric
    pub distance_metric: String,
    /// initialization
    pub initialization: String,
    /// max_iter
    pub max_iter: usize,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for CoreSetApproach {
    fn default() -> Self {
        Self {
            batch_size: 10,
            distance_metric: "euclidean".to_string(),
            initialization: "farthest_first".to_string(),
            max_iter: 100,
            random_state: None,
        }
    }
}

impl CoreSetApproach {
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

    pub fn distance_metric(mut self, distance_metric: String) -> Self {
        self.distance_metric = distance_metric;
        self
    }

    pub fn initialization(mut self, initialization: String) -> Self {
        self.initialization = initialization;
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
            _ => Err(
                BatchActiveLearningError::InvalidDistanceMetric(self.distance_metric.clone())
                    .into(),
            ),
        }
    }

    fn farthest_first_initialization(&self, X: &ArrayView2<f64>) -> Result<Vec<usize>> {
        let n_samples = X.dim().0;
        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        if n_samples < self.batch_size {
            return Err(BatchActiveLearningError::InsufficientUnlabeledSamples.into());
        }

        let mut selected_indices = Vec::new();
        let mut distances = vec![f64::INFINITY; n_samples];

        // Select first point randomly
        let first_idx = rng.gen_range(0..n_samples);
        selected_indices.push(first_idx);

        // Update distances to first point
        for (i, dist) in distances.iter_mut().enumerate() {
            if i != first_idx {
                *dist = self.compute_distance(&X.row(i), &X.row(first_idx))?;
            }
        }

        // Select remaining points
        for _ in 1..self.batch_size {
            // Find point with maximum distance to nearest selected point
            let mut max_distance = 0.0;
            let mut best_idx = 0;

            for (i, &dist) in distances.iter().enumerate() {
                if !selected_indices.contains(&i) && dist > max_distance {
                    max_distance = dist;
                    best_idx = i;
                }
            }

            selected_indices.push(best_idx);

            // Update distances
            for (i, dist) in distances.iter_mut().enumerate() {
                if !selected_indices.contains(&i) {
                    let new_distance = self.compute_distance(&X.row(i), &X.row(best_idx))?;
                    *dist = (*dist).min(new_distance);
                }
            }
        }

        Ok(selected_indices)
    }

    fn k_center_greedy(&self, X: &ArrayView2<f64>) -> Result<Vec<usize>> {
        let n_samples = X.dim().0;
        let mut selected_indices = Vec::new();
        let mut distances = vec![f64::INFINITY; n_samples];

        if n_samples < self.batch_size {
            return Err(BatchActiveLearningError::InsufficientUnlabeledSamples.into());
        }

        // Select first point (center of data)
        let mut centroid = Array1::zeros(X.dim().1);
        for i in 0..n_samples {
            centroid = centroid + X.row(i);
        }
        centroid /= n_samples as f64;

        // Find closest point to centroid
        let mut min_distance = f64::INFINITY;
        let mut first_idx = 0;
        for i in 0..n_samples {
            let distance = self.compute_distance(&X.row(i), &centroid.view())?;
            if distance < min_distance {
                min_distance = distance;
                first_idx = i;
            }
        }

        selected_indices.push(first_idx);

        // Update distances to first point
        for (i, dist) in distances.iter_mut().enumerate() {
            if i != first_idx {
                *dist = self.compute_distance(&X.row(i), &X.row(first_idx))?;
            }
        }

        // Greedily select remaining points
        for _ in 1..self.batch_size {
            let mut max_distance = 0.0;
            let mut best_idx = 0;

            for (i, &dist) in distances.iter().enumerate() {
                if !selected_indices.contains(&i) && dist > max_distance {
                    max_distance = dist;
                    best_idx = i;
                }
            }

            selected_indices.push(best_idx);

            // Update distances
            for (i, dist) in distances.iter_mut().enumerate() {
                if !selected_indices.contains(&i) {
                    let new_distance = self.compute_distance(&X.row(i), &X.row(best_idx))?;
                    *dist = (*dist).min(new_distance);
                }
            }
        }

        Ok(selected_indices)
    }

    pub fn query(
        &self,
        X: &ArrayView2<f64>,
        _probabilities: &ArrayView2<f64>,
    ) -> Result<Vec<usize>> {
        match self.initialization.as_str() {
            "farthest_first" => self.farthest_first_initialization(X),
            "k_center_greedy" => self.k_center_greedy(X),
            _ => self.farthest_first_initialization(X), // default
        }
    }
}

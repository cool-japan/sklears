//! Instance-based learning algorithms for multi-label classification

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Instance-Based Label Ranking (IBLR)
///
/// IBLR performs multi-label classification using k-nearest neighbors
/// with label ranking for prediction.
#[derive(Debug, Clone)]
pub struct IBLR<S = Untrained> {
    state: S,
    k_neighbors: usize,
    weights: WeightFunction,
}

/// Weight function for IBLR
#[derive(Debug, Clone)]
pub enum WeightFunction {
    /// Uniform weighting (all neighbors weighted equally)
    Uniform,
    /// Distance-based weighting (closer neighbors weighted more)
    Distance,
    /// Inverse distance weighting
    InverseDistance,
}

/// Trained state for IBLR
#[derive(Debug, Clone)]
pub struct IBLRTrained {
    /// Training data
    training_data: Array2<Float>,
    /// Training labels
    training_labels: Array2<i32>,
    /// Number of labels
    n_labels: usize,
}

impl IBLR<Untrained> {
    /// Create a new IBLR instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            k_neighbors: 5,
            weights: WeightFunction::Uniform,
        }
    }

    /// Set the number of neighbors
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the weight function
    pub fn weights(mut self, weights: WeightFunction) -> Self {
        self.weights = weights;
        self
    }
}

impl Default for IBLR<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for IBLR<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, i32>> for IBLR<Untrained> {
    type Fitted = IBLR<IBLRTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView2<'_, i32>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let (y_samples, n_labels) = y.dim();

        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if self.k_neighbors == 0 {
            return Err(SklearsError::InvalidInput(
                "k_neighbors must be greater than zero".to_string(),
            ));
        }

        if self.k_neighbors >= n_samples {
            return Err(SklearsError::InvalidInput(
                "k_neighbors must be less than number of samples".to_string(),
            ));
        }

        Ok(IBLR {
            state: IBLRTrained {
                training_data: x.to_owned(),
                training_labels: y.to_owned(),
                n_labels,
            },
            k_neighbors: self.k_neighbors,
            weights: self.weights,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>> for IBLR<IBLRTrained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = x.dim();
        let training_features = self.state.training_data.ncols();

        if n_features != training_features {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features in X ({}) does not match training data ({})",
                n_features, training_features
            )));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            let query = x.row(sample_idx);

            // Find k nearest neighbors
            let neighbors = self.find_k_nearest_neighbors(&query)?;

            // Predict labels based on neighbors
            for label_idx in 0..self.state.n_labels {
                let label_votes = self.compute_label_votes(&neighbors, label_idx);
                predictions[[sample_idx, label_idx]] = if label_votes > 0.5 { 1 } else { 0 };
            }
        }

        Ok(predictions)
    }
}

impl IBLR<IBLRTrained> {
    fn find_k_nearest_neighbors(
        &self,
        query: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<(usize, Float)>> {
        let n_training = self.state.training_data.nrows();
        let mut distances = Vec::new();

        for i in 0..n_training {
            let training_sample = self.state.training_data.row(i);
            let distance = self.compute_distance(query, &training_sample);
            distances.push((i, distance));
        }

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok(distances.into_iter().take(self.k_neighbors).collect())
    }

    fn compute_distance(&self, a: &ArrayView1<'_, Float>, b: &ArrayView1<'_, Float>) -> Float {
        let diff = a.to_owned() - b.to_owned();
        diff.mapv(|x| x * x).sum().sqrt()
    }

    fn compute_label_votes(&self, neighbors: &[(usize, Float)], label_idx: usize) -> Float {
        let mut total_weight = 0.0;
        let mut positive_weight = 0.0;

        for &(neighbor_idx, distance) in neighbors {
            let weight = match self.weights {
                WeightFunction::Uniform => 1.0,
                WeightFunction::Distance => {
                    if distance > 1e-10 {
                        1.0 / distance
                    } else {
                        1e10
                    }
                }
                WeightFunction::InverseDistance => {
                    if distance > 1e-10 {
                        1.0 / (distance * distance)
                    } else {
                        1e10
                    }
                }
            };

            total_weight += weight;

            if self.state.training_labels[[neighbor_idx, label_idx]] == 1 {
                positive_weight += weight;
            }
        }

        if total_weight > 0.0 {
            positive_weight / total_weight
        } else {
            0.0
        }
    }
}

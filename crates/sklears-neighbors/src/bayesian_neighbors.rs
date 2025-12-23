//! Bayesian Nearest Neighbors with Uncertainty Quantification
//!
//! This module implements Bayesian approaches to nearest neighbors that provide
//! uncertainty estimates along with predictions. These methods are particularly
//! useful when understanding prediction confidence is important.

use crate::distance::Distance;
use crate::NeighborsResult;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::thread_rng;
use sklears_core::traits::{Fit, Predict, PredictProba};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Bayesian K-Nearest Neighbors Classifier with uncertainty quantification
#[derive(Debug, Clone)]
pub struct BayesianKNeighborsClassifier<S> {
    k: usize,
    distance: Distance,
    uncertainty_method: UncertaintyMethod,
    prior_strength: Float,
    bootstrap_samples: usize,
    state: S,
}

/// Methods for quantifying prediction uncertainty
#[derive(Debug, Clone)]
pub enum UncertaintyMethod {
    /// Posterior variance based on neighbor label distribution
    PosteriorVariance,
    /// Bootstrap sampling of neighbor sets
    Bootstrap,
    /// Credible intervals using Dirichlet-multinomial model
    CredibleIntervals { confidence: Float },
    /// Entropy-based uncertainty
    Entropy,
}

/// Bayesian prediction result with uncertainty quantification
#[derive(Debug, Clone)]
pub struct BayesianPrediction {
    /// Point prediction (mode of posterior)
    pub prediction: i32,
    /// Posterior probability distribution over classes
    pub probabilities: Array1<Float>,
    /// Uncertainty estimate (method-dependent)
    pub uncertainty: Float,
    /// Credible interval bounds (if applicable)
    pub credible_interval: Option<(Float, Float)>,
    /// Entropy of posterior distribution
    pub entropy: Float,
}

/// Credible set of neighbors with uncertainty quantification
#[derive(Debug, Clone)]
pub struct CredibleNeighborSet {
    /// Indices of neighbors in the credible set
    pub neighbor_indices: Vec<usize>,
    /// Distances to credible neighbors
    pub distances: Vec<Float>,
    /// Probability each neighbor is in true k-NN set
    pub inclusion_probabilities: Vec<Float>,
    /// Confidence level used to construct set
    pub confidence_level: Float,
}

/// Untrained state for type safety
#[derive(Debug, Clone)]
pub struct Untrained;

/// Trained state for classification
#[derive(Debug, Clone)]
pub struct Trained {
    pub x_train: Array2<Float>,
    pub y_train: Array1<i32>,
    pub classes: Array1<i32>,
    pub class_priors: Array1<Float>,
}

/// Trained state for regression
#[derive(Debug, Clone)]
pub struct RegressorTrained {
    pub x_train: Array2<Float>,
    pub y_train: Array1<Float>,
}

impl BayesianKNeighborsClassifier<Untrained> {
    /// Create a new Bayesian KNN classifier
    pub fn new(k: usize) -> Self {
        Self {
            k,
            distance: Distance::Euclidean,
            uncertainty_method: UncertaintyMethod::PosteriorVariance,
            prior_strength: 1.0,
            bootstrap_samples: 100,
            state: Untrained,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set the uncertainty quantification method
    pub fn with_uncertainty_method(mut self, method: UncertaintyMethod) -> Self {
        self.uncertainty_method = method;
        self
    }

    /// Set the prior strength (pseudo-count for Dirichlet prior)
    pub fn with_prior_strength(mut self, strength: Float) -> Self {
        self.prior_strength = strength;
        self
    }

    /// Set the number of bootstrap samples for uncertainty estimation
    pub fn with_bootstrap_samples(mut self, samples: usize) -> Self {
        self.bootstrap_samples = samples;
        self
    }
}

impl Fit<Array2<Float>, Array1<i32>, BayesianKNeighborsClassifier<Trained>>
    for BayesianKNeighborsClassifier<Untrained>
{
    type Fitted = BayesianKNeighborsClassifier<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> sklears_core::error::Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(crate::NeighborsError::ShapeMismatch {
                expected: vec![y.len()],
                actual: vec![x.nrows(), x.ncols()],
            }
            .into());
        }

        if self.k >= x.nrows() {
            return Err(crate::NeighborsError::InvalidInput(format!(
                "k={} should be less than n_samples={}",
                self.k,
                x.nrows()
            ))
            .into());
        }

        // Extract unique classes and compute priors
        let mut unique_classes: Vec<i32> = y.iter().cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);

        // Compute class priors (with Laplace smoothing)
        let mut class_counts = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let total_count = y.len() as Float + self.prior_strength * classes.len() as Float;
        let class_priors = classes
            .iter()
            .map(|&class| {
                let count = class_counts.get(&class).unwrap_or(&0);
                (*count as Float + self.prior_strength) / total_count
            })
            .collect::<Vec<_>>();

        Ok(BayesianKNeighborsClassifier {
            k: self.k,
            distance: self.distance,
            uncertainty_method: self.uncertainty_method,
            prior_strength: self.prior_strength,
            bootstrap_samples: self.bootstrap_samples,
            state: Trained {
                x_train: x.clone(),
                y_train: y.clone(),
                classes,
                class_priors: Array1::from_vec(class_priors),
            },
        })
    }
}

impl BayesianKNeighborsClassifier<Trained> {
    /// Predict with uncertainty quantification
    pub fn predict_with_uncertainty(
        &self,
        x: &Array2<Float>,
    ) -> NeighborsResult<Vec<BayesianPrediction>> {
        let mut predictions = Vec::with_capacity(x.nrows());

        for row in x.rows() {
            let prediction = self.predict_single_with_uncertainty(&row)?;
            predictions.push(prediction);
        }

        Ok(predictions)
    }

    /// Predict a single sample with uncertainty
    fn predict_single_with_uncertainty(
        &self,
        query: &ArrayView1<Float>,
    ) -> NeighborsResult<BayesianPrediction> {
        // Find k nearest neighbors
        let mut distances_indices: Vec<(Float, usize)> = self
            .state
            .x_train
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| (self.distance.calculate(query, &row), i))
            .collect();

        distances_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbor_indices: Vec<usize> = distances_indices
            .into_iter()
            .take(self.k)
            .map(|(_, idx)| idx)
            .collect();

        // Extract neighbor labels
        let neighbor_labels: Vec<i32> = neighbor_indices
            .iter()
            .map(|&idx| self.state.y_train[idx])
            .collect();

        // Compute posterior probabilities using Dirichlet-multinomial model
        let probabilities = self.compute_posterior_probabilities(&neighbor_labels)?;

        // Compute uncertainty based on selected method
        let uncertainty = match &self.uncertainty_method {
            UncertaintyMethod::PosteriorVariance => self.compute_posterior_variance(&probabilities),
            UncertaintyMethod::Bootstrap => {
                self.compute_bootstrap_uncertainty(query, &neighbor_indices)?
            }
            UncertaintyMethod::CredibleIntervals { confidence } => {
                self.compute_credible_interval_uncertainty(&probabilities, *confidence)
            }
            UncertaintyMethod::Entropy => self.compute_entropy_uncertainty(&probabilities),
        };

        // Compute credible interval if using that method
        let credible_interval = match &self.uncertainty_method {
            UncertaintyMethod::CredibleIntervals { confidence } => {
                Some(self.compute_credible_interval(&probabilities, *confidence))
            }
            _ => None,
        };

        // Find prediction (mode of posterior)
        let prediction_idx = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let prediction = self.state.classes[prediction_idx];

        // Compute entropy
        let entropy = self.compute_entropy_uncertainty(&probabilities);

        Ok(BayesianPrediction {
            prediction,
            probabilities,
            uncertainty,
            credible_interval,
            entropy,
        })
    }

    /// Compute posterior probabilities using Dirichlet-multinomial model
    fn compute_posterior_probabilities(
        &self,
        neighbor_labels: &[i32],
    ) -> NeighborsResult<Array1<Float>> {
        let mut posterior_counts = vec![self.prior_strength; self.state.classes.len()];

        // Count neighbor labels
        for &label in neighbor_labels {
            if let Some(class_idx) = self.state.classes.iter().position(|&c| c == label) {
                posterior_counts[class_idx] += 1.0;
            }
        }

        // Normalize to get probabilities
        let total: Float = posterior_counts.iter().sum();
        let probabilities: Vec<Float> = posterior_counts
            .into_iter()
            .map(|count| count / total)
            .collect();

        Ok(Array1::from_vec(probabilities))
    }

    /// Compute uncertainty as posterior variance
    fn compute_posterior_variance(&self, probabilities: &Array1<Float>) -> Float {
        let mean = probabilities.mean().unwrap_or(0.0);
        probabilities
            .iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<Float>()
            / (probabilities.len() as Float - 1.0).max(1.0)
    }

    /// Compute uncertainty using bootstrap sampling
    fn compute_bootstrap_uncertainty(
        &self,
        _query: &ArrayView1<Float>,
        neighbor_indices: &[usize],
    ) -> NeighborsResult<Float> {
        let mut bootstrap_predictions = Vec::new();
        let mut rng = thread_rng();

        for _ in 0..self.bootstrap_samples {
            // Bootstrap sample of neighbors
            let mut bootstrap_neighbors = Vec::new();
            for _ in 0..self.k {
                let idx = rng.gen_range(0..neighbor_indices.len());
                bootstrap_neighbors.push(self.state.y_train[neighbor_indices[idx]]);
            }

            // Compute prediction for this bootstrap sample
            let probs = self.compute_posterior_probabilities(&bootstrap_neighbors)?;
            let pred_idx = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            bootstrap_predictions.push(pred_idx);
        }

        // Compute variance of bootstrap predictions
        let mean_pred =
            bootstrap_predictions.iter().sum::<usize>() as Float / self.bootstrap_samples as Float;
        let variance = bootstrap_predictions
            .iter()
            .map(|&pred| (pred as Float - mean_pred).powi(2))
            .sum::<Float>()
            / (self.bootstrap_samples as Float - 1.0).max(1.0);

        Ok(variance.sqrt())
    }

    /// Compute uncertainty based on credible interval width
    fn compute_credible_interval_uncertainty(
        &self,
        probabilities: &Array1<Float>,
        confidence: Float,
    ) -> Float {
        let (lower, upper) = self.compute_credible_interval(probabilities, confidence);
        upper - lower
    }

    /// Compute credible interval for the most probable class
    fn compute_credible_interval(
        &self,
        probabilities: &Array1<Float>,
        confidence: Float,
    ) -> (Float, Float) {
        // For Dirichlet posterior, compute credible interval
        // This is a simplified implementation using normal approximation
        let max_prob = probabilities.iter().cloned().fold(0.0, Float::max);
        let _alpha = (1.0 - confidence) / 2.0;

        // Normal approximation variance
        let n = self.k as Float + self.state.classes.len() as Float * self.prior_strength;
        let variance = max_prob * (1.0 - max_prob) / n;
        let std_dev = variance.sqrt();

        // Approximate credible interval (using normal quantiles)
        let z_score = 1.96; // Approximate 95% confidence
        let margin = z_score * std_dev;

        ((max_prob - margin).max(0.0), (max_prob + margin).min(1.0))
    }

    /// Compute uncertainty as entropy of posterior distribution
    fn compute_entropy_uncertainty(&self, probabilities: &Array1<Float>) -> Float {
        -probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<Float>()
    }

    /// Compute credible neighbor set for a query point
    ///
    /// Returns a set of neighbors with high posterior probability of being
    /// among the true k-nearest neighbors, accounting for uncertainty.
    pub fn credible_neighbor_set(
        &self,
        query: &ArrayView1<Float>,
        confidence: Float,
    ) -> NeighborsResult<CredibleNeighborSet> {
        // Find more neighbors than k to build credible set
        let candidate_k = (self.k * 3).min(self.state.x_train.nrows());

        let mut distances_indices: Vec<(Float, usize)> = self
            .state
            .x_train
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| (self.distance.calculate(query, &row), i))
            .collect();

        distances_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Take candidate neighbors
        let candidates: Vec<(Float, usize)> =
            distances_indices.into_iter().take(candidate_k).collect();

        // Bootstrap to estimate probability each candidate is in true k-NN set
        let mut neighbor_inclusion_counts = vec![0; candidates.len()];
        let mut rng = thread_rng();

        for _ in 0..self.bootstrap_samples {
            // Add noise to distances to simulate uncertainty
            let mut noisy_distances: Vec<(Float, usize)> = candidates
                .iter()
                .enumerate()
                .map(|(i, &(dist, _idx))| {
                    let noise = rng.gen_range(-0.1..0.1) * dist.abs();
                    (dist + noise, i)
                })
                .collect();

            noisy_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Mark which candidates are in top k
            for (_, candidate_idx) in noisy_distances.iter().take(self.k) {
                neighbor_inclusion_counts[*candidate_idx] += 1;
            }
        }

        // Compute inclusion probabilities
        let inclusion_probabilities: Vec<Float> = neighbor_inclusion_counts
            .iter()
            .map(|&count| count as Float / self.bootstrap_samples as Float)
            .collect();

        // Determine threshold for credible set
        // We want neighbors with inclusion probability > (1 - confidence)
        let threshold = 1.0 - confidence;

        let credible_neighbors: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| inclusion_probabilities[*i] > threshold)
            .map(|(_, &(_, idx))| idx)
            .collect();

        let credible_distances: Vec<Float> = candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| inclusion_probabilities[*i] > threshold)
            .map(|(_, &(dist, _))| dist)
            .collect();

        let credible_probabilities: Vec<Float> = inclusion_probabilities
            .into_iter()
            .enumerate()
            .filter(|(_i, p)| *p > threshold)
            .map(|(_, p)| p)
            .collect();

        Ok(CredibleNeighborSet {
            neighbor_indices: credible_neighbors,
            distances: credible_distances,
            inclusion_probabilities: credible_probabilities,
            confidence_level: confidence,
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for BayesianKNeighborsClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> sklears_core::error::Result<Array1<i32>> {
        let predictions = self.predict_with_uncertainty(x)?;
        let predictions_vec: Vec<i32> = predictions.into_iter().map(|p| p.prediction).collect();
        Ok(Array1::from_vec(predictions_vec))
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for BayesianKNeighborsClassifier<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> sklears_core::error::Result<Array2<Float>> {
        let predictions = self.predict_with_uncertainty(x)?;
        let n_samples = predictions.len();
        let n_classes = self.state.classes.len();

        let mut proba_matrix = Array2::zeros((n_samples, n_classes));
        for (i, prediction) in predictions.into_iter().enumerate() {
            for (j, &prob) in prediction.probabilities.iter().enumerate() {
                proba_matrix[[i, j]] = prob;
            }
        }

        Ok(proba_matrix)
    }
}

/// Bayesian K-Nearest Neighbors Regressor with uncertainty quantification
#[derive(Debug, Clone)]
pub struct BayesianKNeighborsRegressor<S> {
    k: usize,
    distance: Distance,
    prior_mean: Float,
    prior_variance: Float,
    noise_variance: Float,
    state: S,
}

/// Regression prediction with uncertainty
#[derive(Debug, Clone)]
pub struct BayesianRegressionPrediction {
    /// Point prediction (posterior mean)
    pub prediction: Float,
    /// Prediction variance
    pub variance: Float,
    /// Standard deviation
    pub std_dev: Float,
    /// Credible interval
    pub credible_interval: (Float, Float),
}

impl BayesianKNeighborsRegressor<Untrained> {
    /// Create a new Bayesian KNN regressor
    pub fn new(k: usize) -> Self {
        Self {
            k,
            distance: Distance::Euclidean,
            prior_mean: 0.0,
            prior_variance: 1.0,
            noise_variance: 1.0,
            state: Untrained,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set prior parameters
    pub fn with_prior(mut self, mean: Float, variance: Float) -> Self {
        self.prior_mean = mean;
        self.prior_variance = variance;
        self
    }

    /// Set noise variance estimate
    pub fn with_noise_variance(mut self, variance: Float) -> Self {
        self.noise_variance = variance;
        self
    }
}

impl Fit<Array2<Float>, Array1<Float>, BayesianKNeighborsRegressor<RegressorTrained>>
    for BayesianKNeighborsRegressor<Untrained>
{
    type Fitted = BayesianKNeighborsRegressor<RegressorTrained>;

    fn fit(
        self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> sklears_core::error::Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(crate::NeighborsError::ShapeMismatch {
                expected: vec![y.len()],
                actual: vec![x.nrows(), x.ncols()],
            }
            .into());
        }

        if self.k >= x.nrows() {
            return Err(crate::NeighborsError::InvalidInput(format!(
                "k={} should be less than n_samples={}",
                self.k,
                x.nrows()
            ))
            .into());
        }

        Ok(BayesianKNeighborsRegressor {
            k: self.k,
            distance: self.distance,
            prior_mean: self.prior_mean,
            prior_variance: self.prior_variance,
            noise_variance: self.noise_variance,
            state: RegressorTrained {
                x_train: x.clone(),
                y_train: y.clone(),
            },
        })
    }
}

impl BayesianKNeighborsRegressor<RegressorTrained> {
    /// Predict with uncertainty quantification
    pub fn predict_with_uncertainty(
        &self,
        x: &Array2<Float>,
    ) -> NeighborsResult<Vec<BayesianRegressionPrediction>> {
        let mut predictions = Vec::with_capacity(x.nrows());

        for row in x.rows() {
            let prediction = self.predict_single_with_uncertainty(&row)?;
            predictions.push(prediction);
        }

        Ok(predictions)
    }

    /// Predict a single sample with uncertainty
    fn predict_single_with_uncertainty(
        &self,
        query: &ArrayView1<Float>,
    ) -> NeighborsResult<BayesianRegressionPrediction> {
        // Find k nearest neighbors
        let mut distances_indices: Vec<(Float, usize)> = self
            .state
            .x_train
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| (self.distance.calculate(query, &row), i))
            .collect();

        distances_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbor_values: Vec<Float> = distances_indices
            .into_iter()
            .take(self.k)
            .map(|(_, idx)| self.state.y_train[idx])
            .collect();

        // Bayesian update using conjugate normal-normal model
        let n = neighbor_values.len() as Float;
        let sample_mean = neighbor_values.iter().sum::<Float>() / n;

        // Prior precision and updated precision
        let prior_precision = 1.0 / self.prior_variance;
        let noise_precision = 1.0 / self.noise_variance;
        let posterior_precision = prior_precision + n * noise_precision;
        let posterior_variance = 1.0 / posterior_precision;

        // Posterior mean
        let posterior_mean = posterior_variance
            * (prior_precision * self.prior_mean + n * noise_precision * sample_mean);

        // Predictive variance (includes both posterior uncertainty and noise)
        let predictive_variance = posterior_variance + self.noise_variance;
        let std_dev = predictive_variance.sqrt();

        // 95% credible interval
        let margin = 1.96 * std_dev;
        let credible_interval = (posterior_mean - margin, posterior_mean + margin);

        Ok(BayesianRegressionPrediction {
            prediction: posterior_mean,
            variance: predictive_variance,
            std_dev,
            credible_interval,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for BayesianKNeighborsRegressor<RegressorTrained> {
    fn predict(&self, x: &Array2<Float>) -> sklears_core::error::Result<Array1<Float>> {
        let predictions = self.predict_with_uncertainty(x)?;
        let predictions_vec: Vec<Float> = predictions.into_iter().map(|p| p.prediction).collect();
        Ok(Array1::from_vec(predictions_vec))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_bayesian_knn_classifier_basic() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                -1.0, -1.0, // Class 0
                -1.0, -0.5, // Class 0
                -0.5, -1.0, // Class 0
                1.0, 1.0, // Class 1
                1.0, 0.5, // Class 1
                0.5, 1.0, // Class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let classifier = BayesianKNeighborsClassifier::new(3);
        let fitted = classifier.fit(&x, &y).unwrap();
        let predictions = fitted.predict_with_uncertainty(&x).unwrap();

        assert_eq!(predictions.len(), 6);

        // Check that predictions have uncertainty information
        for pred in &predictions {
            assert!(pred.uncertainty >= 0.0);
            assert!(pred.entropy >= 0.0);
            assert!(pred.probabilities.iter().all(|&p| p >= 0.0 && p <= 1.0));
            assert_abs_diff_eq!(pred.probabilities.sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bayesian_knn_regressor_basic() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0, // y = 2
                2.0, 1.0, // y = 3
                1.0, 2.0, // y = 3
                2.0, 2.0, // y = 4
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![2.0, 3.0, 3.0, 4.0]);

        let regressor = BayesianKNeighborsRegressor::new(2);
        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict_with_uncertainty(&x).unwrap();

        assert_eq!(predictions.len(), 4);

        // Check that predictions have uncertainty information
        for pred in &predictions {
            assert!(pred.variance > 0.0);
            assert!(pred.std_dev > 0.0);
            assert!(pred.credible_interval.0 <= pred.prediction);
            assert!(pred.credible_interval.1 >= pred.prediction);
        }
    }

    #[test]
    fn test_uncertainty_methods() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                -1.0, -1.0, -0.5, -1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 1.5, 1.5,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1, 1, 1]);

        let methods = vec![
            UncertaintyMethod::PosteriorVariance,
            UncertaintyMethod::Bootstrap,
            UncertaintyMethod::CredibleIntervals { confidence: 0.95 },
            UncertaintyMethod::Entropy,
        ];

        for method in methods {
            let classifier = BayesianKNeighborsClassifier::new(3).with_uncertainty_method(method);
            let fitted = classifier.fit(&x, &y).unwrap();
            let predictions = fitted.predict_with_uncertainty(&x).unwrap();

            // All methods should produce reasonable uncertainty estimates
            for pred in &predictions {
                assert!(pred.uncertainty.is_finite());
                assert!(pred.uncertainty >= 0.0);
            }
        }
    }

    #[test]
    fn test_bayesian_knn_error_cases() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![0, 1]);

        // k too large
        let classifier = BayesianKNeighborsClassifier::new(5);
        assert!(classifier.fit(&x, &y).is_err());

        // Shape mismatch
        let y_wrong = Array1::from_vec(vec![0, 1, 2]);
        let classifier = BayesianKNeighborsClassifier::new(1);
        assert!(classifier.fit(&x, &y_wrong).is_err());
    }

    #[test]
    fn test_credible_neighbor_set() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                -1.0, -1.0, -0.9, -1.0, -1.0, -0.9, -0.8, -0.8, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 0.8,
                0.8, -0.5, -0.5, 0.5, 0.5,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 1]);

        let classifier = BayesianKNeighborsClassifier::new(3);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Test query point close to class 0
        let query = Array1::from_vec(vec![-0.95, -0.95]);
        let credible_set = fitted.credible_neighbor_set(&query.view(), 0.95).unwrap();

        // Check that we got a credible set
        assert!(!credible_set.neighbor_indices.is_empty());
        assert_eq!(
            credible_set.neighbor_indices.len(),
            credible_set.distances.len()
        );
        assert_eq!(
            credible_set.neighbor_indices.len(),
            credible_set.inclusion_probabilities.len()
        );
        assert_eq!(credible_set.confidence_level, 0.95);

        // Check that probabilities are valid
        for &prob in &credible_set.inclusion_probabilities {
            assert!(prob >= 0.0 && prob <= 1.0);
        }

        // Check that distances are sorted
        for i in 1..credible_set.distances.len() {
            assert!(credible_set.distances[i - 1] <= credible_set.distances[i]);
        }
    }

    #[test]
    fn test_credible_neighbor_set_different_confidence() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);

        let classifier = BayesianKNeighborsClassifier::new(3);
        let fitted = classifier.fit(&x, &y).unwrap();

        let query = Array1::from_vec(vec![1.5, 1.5]);

        // Higher confidence should give larger credible set
        let set_90 = fitted.credible_neighbor_set(&query.view(), 0.90).unwrap();
        let set_99 = fitted.credible_neighbor_set(&query.view(), 0.99).unwrap();

        assert!(set_99.neighbor_indices.len() >= set_90.neighbor_indices.len());
    }
}

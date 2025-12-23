//! Fuzzy C-Means clustering implementation
//!
//! Fuzzy C-Means (FCM) is a clustering algorithm where each data point
//! can belong to multiple clusters with varying degrees of membership.
//! Unlike hard clustering (like K-means), FCM assigns membership values
//! between 0 and 1 to each point for each cluster.

use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::{Float, FloatBounds},
    validation::{ConfigValidation, Validate, ValidationRule, ValidationRules},
};

/// Configuration for Fuzzy C-Means clustering
#[derive(Debug, Clone)]
pub struct FuzzyCMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Fuzzification parameter (m > 1)
    pub fuzziness: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance for membership matrix
    pub tol: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Error threshold for stopping criterion
    pub error_threshold: Float,
}

impl Default for FuzzyCMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            fuzziness: 2.0,
            max_iter: 150,
            tol: 1e-4,
            random_state: None,
            error_threshold: 1e-5,
        }
    }
}

impl Validate for FuzzyCMeansConfig {
    fn validate(&self) -> Result<()> {
        ValidationRules::new("n_clusters")
            .add_rule(ValidationRule::Positive)
            .validate_usize(&self.n_clusters)?;

        ValidationRules::new("fuzziness")
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.fuzziness)?;

        if self.fuzziness <= 1.0 {
            return Err(SklearsError::InvalidInput(
                "Fuzziness parameter must be > 1.0".to_string(),
            ));
        }

        ValidationRules::new("max_iter")
            .add_rule(ValidationRule::Positive)
            .validate_usize(&self.max_iter)?;

        ValidationRules::new("tol")
            .add_rule(ValidationRule::NonNegative)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.tol)?;

        ValidationRules::new("error_threshold")
            .add_rule(ValidationRule::NonNegative)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.error_threshold)?;

        Ok(())
    }
}

impl ConfigValidation for FuzzyCMeansConfig {
    fn validate_config(&self) -> Result<()> {
        self.validate()?;

        if self.fuzziness > 10.0 {
            log::warn!(
                "Very high fuzziness parameter {} may lead to poor clustering",
                self.fuzziness
            );
        }

        if self.fuzziness < 1.1 {
            log::warn!(
                "Low fuzziness parameter {} may behave like hard clustering",
                self.fuzziness
            );
        }

        if self.max_iter < 50 {
            log::warn!(
                "Low max_iter {} may prevent convergence for fuzzy clustering",
                self.max_iter
            );
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.n_clusters > 50 {
            warnings
                .push("Very large number of clusters may be computationally expensive".to_string());
        }

        if self.tol > 1e-2 {
            warnings.push("Large tolerance may lead to premature convergence".to_string());
        }

        warnings
    }
}

/// Fuzzy C-Means clustering algorithm
#[derive(Debug, Clone)]
pub struct FuzzyCMeans<State = Untrained> {
    config: FuzzyCMeansConfig,
    state: PhantomData<State>,
    // Trained state
    centroids_: Option<Array2<Float>>,
    membership_matrix_: Option<Array2<Float>>,
    labels_: Option<Array1<usize>>,
    objective_function_history_: Option<Vec<Float>>,
    n_iter_: Option<usize>,
}

impl FuzzyCMeans<Untrained> {
    /// Create a new Fuzzy C-Means model
    pub fn new(n_clusters: usize) -> Self {
        Self {
            config: FuzzyCMeansConfig {
                n_clusters,
                ..Default::default()
            },
            state: PhantomData,
            centroids_: None,
            membership_matrix_: None,
            labels_: None,
            objective_function_history_: None,
            n_iter_: None,
        }
    }

    /// Set the fuzziness parameter
    pub fn fuzziness(mut self, fuzziness: Float) -> Self {
        self.config.fuzziness = fuzziness;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the error threshold
    pub fn error_threshold(mut self, threshold: Float) -> Self {
        self.config.error_threshold = threshold;
        self
    }
}

impl Default for FuzzyCMeans<Untrained> {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Estimator for FuzzyCMeans<Untrained> {
    type Config = FuzzyCMeansConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for FuzzyCMeans<Untrained> {
    type Fitted = FuzzyCMeans<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        self.config.validate_config()?;

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_clusters = self.config.n_clusters;

        if n_samples < n_clusters {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples ({}) must be >= n_clusters ({})",
                n_samples, n_clusters
            )));
        }

        // Initialize random number generator
        let mut rng = match self.config.random_state {
            Some(seed) => Random::default(),
            None => Random::default(),
        };

        // Initialize membership matrix randomly
        let mut membership_matrix = initialize_membership_matrix(n_samples, n_clusters, &mut rng)?;

        // Initialize centroids and objective function history
        let mut centroids = Array2::zeros((n_clusters, n_features));
        let mut objective_history = Vec::new();
        let mut n_iter = 0;

        // Main FCM iteration loop
        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // Update centroids based on current membership matrix
            update_centroids(&mut centroids, x, &membership_matrix, self.config.fuzziness)?;

            // Calculate objective function value
            let objective_value = calculate_objective_function(
                x,
                &centroids,
                &membership_matrix,
                self.config.fuzziness,
            );
            objective_history.push(objective_value);

            // Update membership matrix based on new centroids
            let new_membership_matrix =
                update_membership_matrix(x, &centroids, self.config.fuzziness)?;

            // Check for convergence
            let change = calculate_membership_change(&membership_matrix, &new_membership_matrix);
            membership_matrix = new_membership_matrix;

            if change < self.config.tol {
                break;
            }

            // Check if objective function is not improving
            if iter > 0 {
                let improvement = objective_history[iter - 1] - objective_value;
                if improvement.abs() < self.config.error_threshold {
                    break;
                }
            }
        }

        // Generate hard cluster assignments from membership matrix
        let labels = membership_matrix_to_labels(&membership_matrix);

        Ok(FuzzyCMeans {
            config: self.config,
            state: PhantomData,
            centroids_: Some(centroids),
            membership_matrix_: Some(membership_matrix),
            labels_: Some(labels),
            objective_function_history_: Some(objective_history),
            n_iter_: Some(n_iter),
        })
    }
}

impl FuzzyCMeans<Trained> {
    /// Get the cluster centroids
    pub fn centroids(&self) -> &Array2<Float> {
        self.centroids_.as_ref().unwrap()
    }

    /// Get the membership matrix (samples × clusters)
    pub fn membership_matrix(&self) -> &Array2<Float> {
        self.membership_matrix_.as_ref().unwrap()
    }

    /// Get the hard cluster labels
    pub fn labels(&self) -> &Array1<usize> {
        self.labels_.as_ref().unwrap()
    }

    /// Get the objective function history
    pub fn objective_function_history(&self) -> &[Float] {
        self.objective_function_history_.as_ref().unwrap()
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Get the membership degree for a specific sample and cluster
    pub fn membership_degree(&self, sample_idx: usize, cluster_idx: usize) -> Option<Float> {
        let membership = self.membership_matrix();
        if sample_idx < membership.nrows() && cluster_idx < membership.ncols() {
            Some(membership[[sample_idx, cluster_idx]])
        } else {
            None
        }
    }

    /// Get the membership degrees for a specific sample across all clusters
    pub fn sample_memberships(&self, sample_idx: usize) -> Option<Array1<Float>> {
        let membership = self.membership_matrix();
        if sample_idx < membership.nrows() {
            Some(membership.row(sample_idx).to_owned())
        } else {
            None
        }
    }
}

impl Predict<Array2<Float>, Array1<usize>> for FuzzyCMeans<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<usize>> {
        let centroids = self.centroids();
        let n_samples = x.nrows();

        // Calculate membership matrix for new data
        let membership_matrix = update_membership_matrix(x, centroids, self.config.fuzziness)?;

        // Convert to hard labels
        Ok(membership_matrix_to_labels(&membership_matrix))
    }
}

/// Predict membership probabilities for new data
pub trait PredictMembership<X> {
    /// Predict membership degrees for input data
    fn predict_membership(&self, x: &X) -> Result<Array2<Float>>;
}

impl PredictMembership<Array2<Float>> for FuzzyCMeans<Trained> {
    fn predict_membership(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let centroids = self.centroids();
        update_membership_matrix(x, centroids, self.config.fuzziness)
    }
}

/// Initialize membership matrix randomly
fn initialize_membership_matrix(
    n_samples: usize,
    n_clusters: usize,
    rng: &mut impl Rng,
) -> Result<Array2<Float>> {
    let mut membership = Array2::zeros((n_samples, n_clusters));

    for i in 0..n_samples {
        // Generate random values for each cluster
        let mut row_sum = 0.0;
        for j in 0..n_clusters {
            let value: Float = rng.random_range(0.0..1.0);
            membership[[i, j]] = value;
            row_sum += value;
        }

        // Normalize to ensure sum equals 1
        if row_sum > 0.0 {
            for j in 0..n_clusters {
                membership[[i, j]] /= row_sum;
            }
        } else {
            // If all values are zero, assign equal membership
            for j in 0..n_clusters {
                membership[[i, j]] = 1.0 / n_clusters as Float;
            }
        }
    }

    Ok(membership)
}

/// Update cluster centroids based on membership matrix
fn update_centroids(
    centroids: &mut Array2<Float>,
    data: &Array2<Float>,
    membership: &Array2<Float>,
    fuzziness: Float,
) -> Result<()> {
    let n_clusters = centroids.nrows();
    let n_features = centroids.ncols();

    for k in 0..n_clusters {
        let mut numerator = Array1::<Float>::zeros(n_features);
        let mut denominator = 0.0;

        for i in 0..data.nrows() {
            let membership_fuzzy = membership[[i, k]].powf(fuzziness);
            numerator += &(data.row(i).to_owned() * membership_fuzzy);
            denominator += membership_fuzzy;
        }

        if denominator > Float::EPSILON {
            for j in 0..n_features {
                centroids[[k, j]] = numerator[j] / denominator;
            }
        } else {
            return Err(SklearsError::InvalidInput(
                "Zero denominator in centroid update".to_string(),
            ));
        }
    }

    Ok(())
}

/// Update membership matrix based on current centroids
fn update_membership_matrix(
    data: &Array2<Float>,
    centroids: &Array2<Float>,
    fuzziness: Float,
) -> Result<Array2<Float>> {
    let n_samples = data.nrows();
    let n_clusters = centroids.nrows();
    let mut membership = Array2::zeros((n_samples, n_clusters));

    let fuzziness_factor = 2.0 / (fuzziness - 1.0);

    for i in 0..n_samples {
        let sample = data.row(i);

        // Calculate distances to all centroids
        let mut distances = Vec::with_capacity(n_clusters);
        for k in 0..n_clusters {
            let centroid = centroids.row(k);
            let dist = euclidean_distance(&sample, &centroid);
            distances.push(dist);
        }

        // Update membership values
        for j in 0..n_clusters {
            if distances[j] < Float::EPSILON {
                // Sample is very close to centroid j
                for k in 0..n_clusters {
                    membership[[i, k]] = if k == j { 1.0 } else { 0.0 };
                }
                break;
            } else {
                let mut sum = 0.0;
                for k in 0..n_clusters {
                    if distances[k] < Float::EPSILON {
                        sum = Float::INFINITY;
                        break;
                    }
                    sum += (distances[j] / distances[k]).powf(fuzziness_factor);
                }

                if sum.is_finite() && sum > 0.0 {
                    membership[[i, j]] = 1.0 / sum;
                } else {
                    membership[[i, j]] = 1.0 / n_clusters as Float;
                }
            }
        }
    }

    Ok(membership)
}

/// Calculate objective function value
fn calculate_objective_function(
    data: &Array2<Float>,
    centroids: &Array2<Float>,
    membership: &Array2<Float>,
    fuzziness: Float,
) -> Float {
    let mut objective = 0.0;

    for i in 0..data.nrows() {
        let sample = data.row(i);
        for k in 0..centroids.nrows() {
            let centroid = centroids.row(k);
            let distance = euclidean_distance(&sample, &centroid);
            let membership_fuzzy = membership[[i, k]].powf(fuzziness);
            objective += membership_fuzzy * distance * distance;
        }
    }

    objective
}

/// Calculate change in membership matrix
fn calculate_membership_change(
    old_membership: &Array2<Float>,
    new_membership: &Array2<Float>,
) -> Float {
    let mut max_change = 0.0;

    for i in 0..old_membership.nrows() {
        for j in 0..old_membership.ncols() {
            let change = (old_membership[[i, j]] - new_membership[[i, j]]).abs();
            if change > max_change {
                max_change = change;
            }
        }
    }

    max_change
}

/// Convert membership matrix to hard cluster labels
fn membership_matrix_to_labels(membership: &Array2<Float>) -> Array1<usize> {
    let n_samples = membership.nrows();
    let mut labels = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let row = membership.row(i);
        let mut max_membership = row[0];
        let mut best_cluster = 0;

        for (j, &value) in row.iter().enumerate() {
            if value > max_membership {
                max_membership = value;
                best_cluster = j;
            }
        }

        labels[i] = best_cluster;
    }

    labels
}

/// Calculate Euclidean distance between two points
fn euclidean_distance(
    point1: &scirs2_core::ndarray::ArrayView1<Float>,
    point2: &scirs2_core::ndarray::ArrayView1<Float>,
) -> Float {
    let diff = point1 - point2;
    diff.mapv(|x| x * x).sum().sqrt()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_fuzzy_cmeans_basic() {
        let data = array![
            [1.0, 1.0],
            [1.5, 2.0],
            [3.0, 4.0],
            [5.0, 7.0],
            [3.5, 5.0],
            [4.5, 5.0],
            [3.5, 4.5],
        ];

        let model = FuzzyCMeans::new(2)
            .fuzziness(2.0)
            .max_iter(100)
            .random_state(42)
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();
        assert_eq!(labels.len(), data.nrows());

        let centroids = model.centroids();
        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), data.ncols());

        let membership = model.membership_matrix();
        assert_eq!(membership.nrows(), data.nrows());
        assert_eq!(membership.ncols(), 2);

        // Check that membership values sum to 1 for each sample
        for i in 0..membership.nrows() {
            let row_sum: Float = membership.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fuzzy_cmeans_predict() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0],];

        let model = FuzzyCMeans::new(2)
            .random_state(42)
            .fit(&data, &())
            .unwrap();

        let test_data = array![[0.5, 0.5], [10.5, 10.5]];
        let predictions = model.predict(&test_data).unwrap();

        assert_eq!(predictions.len(), 2);
        // Points should likely be assigned to different clusters
        assert_ne!(predictions[0], predictions[1]);
    }

    #[test]
    fn test_fuzzy_cmeans_membership_prediction() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0],];

        let model = FuzzyCMeans::new(2)
            .random_state(42)
            .fit(&data, &())
            .unwrap();

        let test_data = array![[0.5, 0.5], [10.5, 10.5]];
        let membership = model.predict_membership(&test_data).unwrap();

        assert_eq!(membership.nrows(), 2);
        assert_eq!(membership.ncols(), 2);

        // Check that membership values sum to 1 for each test sample
        for i in 0..membership.nrows() {
            let row_sum: Float = membership.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fuzzy_cmeans_validation() {
        // Test invalid fuzziness <= 1.0
        let config = FuzzyCMeansConfig {
            fuzziness: 0.8,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Test valid configuration
        let config = FuzzyCMeansConfig {
            fuzziness: 2.0,
            n_clusters: 3,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_membership_matrix_properties() {
        let data = array![[1.0, 1.0], [2.0, 2.0], [10.0, 10.0],];

        let model = FuzzyCMeans::new(2)
            .random_state(42)
            .fit(&data, &())
            .unwrap();

        let membership = model.membership_matrix();

        // Test membership degree access
        assert!(model.membership_degree(0, 0).is_some());
        assert!(model.membership_degree(0, 1).is_some());
        assert!(model.membership_degree(10, 0).is_none()); // Invalid sample index

        // Test sample memberships
        let sample_0_memberships = model.sample_memberships(0).unwrap();
        assert_eq!(sample_0_memberships.len(), 2);

        assert!(model.sample_memberships(10).is_none()); // Invalid sample index
    }
}

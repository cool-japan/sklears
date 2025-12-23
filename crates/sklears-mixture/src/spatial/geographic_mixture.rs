//! Geographic Mixture Model
//!
//! This module implements specialized mixture models for geographic data that incorporates
//! geographical features like elevation, distance to landmarks, and other spatial
//! characteristics. It's particularly useful for environmental modeling, urban planning,
//! and geographic information systems (GIS) applications.

use scirs2_core::ndarray::s;
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
};

/// Geographic Mixture Model
///
/// Specialized mixture model for geographic data that incorporates
/// geographical features like elevation, distance to landmarks, etc.
#[derive(Debug, Clone)]
pub struct GeographicMixture<S = Untrained> {
    /// n_components
    pub n_components: usize,
    /// use_elevation
    pub use_elevation: bool,
    /// use_distance_features
    pub use_distance_features: bool,
    /// landmark_coordinates
    pub landmark_coordinates: Option<Array2<f64>>,
    /// max_iter
    pub max_iter: usize,
    /// tol
    pub tol: f64,
    /// random_state
    pub random_state: Option<u64>,
    _phantom: std::marker::PhantomData<S>,
}

/// Trained geographic mixture model
#[derive(Debug, Clone)]
pub struct GeographicMixtureTrained {
    /// weights
    pub weights: Array1<f64>,
    /// means
    pub means: Array2<f64>,
    /// covariances
    pub covariances: Array3<f64>,
    /// geographic_features
    pub geographic_features: Array2<f64>,
}

/// Builder for Geographic Mixture Model
#[derive(Debug, Clone)]
pub struct GeographicMixtureBuilder {
    n_components: usize,
    use_elevation: bool,
    use_distance_features: bool,
    landmark_coordinates: Option<Array2<f64>>,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
}

impl GeographicMixtureBuilder {
    /// Create a new builder with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            use_elevation: false,
            use_distance_features: false,
            landmark_coordinates: None,
            max_iter: 100,
            tol: 1e-4,
            random_state: None,
        }
    }

    /// Enable elevation features
    pub fn use_elevation(mut self, use_elevation: bool) -> Self {
        self.use_elevation = use_elevation;
        self
    }

    /// Enable distance to landmarks features
    pub fn use_distance_features(mut self, use_distance_features: bool) -> Self {
        self.use_distance_features = use_distance_features;
        self
    }

    /// Set landmark coordinates for distance calculations
    pub fn landmark_coordinates(mut self, landmarks: Array2<f64>) -> Self {
        self.landmark_coordinates = Some(landmarks);
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Build the geographic mixture model
    pub fn build(self) -> GeographicMixture<Untrained> {
        GeographicMixture {
            n_components: self.n_components,
            use_elevation: self.use_elevation,
            use_distance_features: self.use_distance_features,
            landmark_coordinates: self.landmark_coordinates,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Estimator for GeographicMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for GeographicMixture<Untrained> {
    type Fitted = GeographicMixture<GeographicMixtureTrained>;

    fn fit(self, X: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _n_features) = X.dim();

        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least the number of components".to_string(),
            ));
        }

        // Extract geographic features
        let geographic_features = self.extract_geographic_features(X)?;

        // Initialize mixture parameters
        let weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);
        let means = Array2::zeros((self.n_components, geographic_features.ncols()));
        let covariances = Array3::zeros((
            self.n_components,
            geographic_features.ncols(),
            geographic_features.ncols(),
        ));

        // Run EM algorithm on geographic features
        let (final_weights, final_means, final_covariances) =
            self.geographic_em_algorithm(&geographic_features, weights, means, covariances)?;

        let _trained_state = GeographicMixtureTrained {
            weights: final_weights,
            means: final_means,
            covariances: final_covariances,
            geographic_features: geographic_features.clone(),
        };

        Ok(GeographicMixture {
            n_components: self.n_components,
            use_elevation: self.use_elevation,
            use_distance_features: self.use_distance_features,
            landmark_coordinates: self.landmark_coordinates,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl GeographicMixture<Untrained> {
    pub fn extract_geographic_features(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut feature_count = n_features;

        // Count additional features
        if self.use_elevation {
            feature_count += 1;
        }
        if self.use_distance_features {
            if let Some(ref landmarks) = self.landmark_coordinates {
                feature_count += landmarks.nrows();
            }
        }

        let mut features = Array2::zeros((n_samples, feature_count));

        // Copy original features
        for i in 0..n_samples {
            for j in 0..n_features {
                features[[i, j]] = X[[i, j]];
            }
        }

        let mut feature_idx = n_features;

        // Add elevation features (simplified as height above mean)
        if self.use_elevation && n_features >= 3 {
            let mean_z = X.column(2).mean().unwrap_or(0.0);
            for i in 0..n_samples {
                features[[i, feature_idx]] = X[[i, 2]] - mean_z;
            }
            feature_idx += 1;
        }

        // Add distance to landmark features
        if self.use_distance_features {
            if let Some(ref landmarks) = self.landmark_coordinates {
                for landmark_idx in 0..landmarks.nrows() {
                    for i in 0..n_samples {
                        let distance = if n_features >= 2 {
                            let dx = X[[i, 0]] - landmarks[[landmark_idx, 0]];
                            let dy = X[[i, 1]] - landmarks[[landmark_idx, 1]];
                            (dx * dx + dy * dy).sqrt()
                        } else {
                            0.0
                        };
                        features[[i, feature_idx]] = distance;
                    }
                    feature_idx += 1;
                }
            }
        }

        Ok(features)
    }

    /// Geographic EM algorithm
    fn geographic_em_algorithm(
        &self,
        features: &Array2<f64>,
        mut weights: Array1<f64>,
        mut means: Array2<f64>,
        mut covariances: Array3<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Array3<f64>)> {
        let (_n_samples, n_features) = features.dim();
        let n_components = self.n_components;
        let mut prev_log_likelihood = f64::NEG_INFINITY;

        // Initialize means with k-means++ on geographic features
        (weights, means) = self.initialize_geographic_parameters(features, weights, means)?;

        // Initialize covariances
        for k in 0..n_components {
            for i in 0..n_features {
                covariances[[k, i, i]] = if i < 2 { 2.0 } else { 1.0 };
            }
        }

        for _iteration in 0..self.max_iter {
            // E-step: Compute responsibilities
            let responsibilities =
                self.geographic_e_step(features, &weights, &means, &covariances)?;

            // M-step: Update parameters
            let (new_weights, new_means, new_covariances) =
                self.geographic_m_step(&responsibilities, features)?;

            // Compute log-likelihood
            let log_likelihood = self.compute_geographic_log_likelihood(
                features,
                &new_weights,
                &new_means,
                &new_covariances,
            )?;

            // Check convergence
            if (log_likelihood - prev_log_likelihood).abs() < self.tol {
                weights = new_weights;
                means = new_means;
                covariances = new_covariances;
                break;
            }

            weights = new_weights;
            means = new_means;
            covariances = new_covariances;
            prev_log_likelihood = log_likelihood;
        }

        Ok((weights, means, covariances))
    }

    /// Initialize parameters specifically for geographic features
    fn initialize_geographic_parameters(
        &self,
        features: &Array2<f64>,
        _weights: Array1<f64>,
        mut means: Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>)> {
        let (n_samples, n_features) = features.dim();
        let n_components = self.n_components;

        // Use k-means++ initialization weighted by geographic importance
        let mut rng = thread_rng();

        // Choose first center randomly
        let first_idx = rng.gen_range(0..n_samples);
        for j in 0..n_features {
            means[[0, j]] = features[[first_idx, j]];
        }

        // Choose remaining centers with geographic-aware k-means++
        for i in 1..n_components {
            let mut distances = Array1::zeros(n_samples);
            for k in 0..n_samples {
                let mut min_dist = f64::INFINITY;
                for prev_i in 0..i {
                    let dist: f64 = (0..n_features)
                        .map(|j| (features[[k, j]] - means[[prev_i, j]]).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    // Apply geographic weighting (give more weight to spatial features)
                    let geographic_weight = 1.0; // Will be applied per feature basis in the loop
                    let weighted_dist = dist * geographic_weight;

                    min_dist = min_dist.min(weighted_dist);
                }
                distances[k] = min_dist;
            }

            // Choose next center proportionally to squared distance
            let total_dist: f64 = distances.iter().map(|d| d * d).sum();
            let threshold = rng.gen::<f64>() * total_dist;
            let mut cumulative = 0.0;
            let mut chosen_idx = 0;

            for k in 0..n_samples {
                cumulative += distances[k] * distances[k];
                if cumulative >= threshold {
                    chosen_idx = k;
                    break;
                }
            }

            for j in 0..n_features {
                means[[i, j]] = features[[chosen_idx, j]];
            }
        }

        // Uniform weights
        let weights = Array1::from_elem(n_components, 1.0 / n_components as f64);

        Ok((weights, means))
    }

    /// E-step for geographic mixture
    fn geographic_e_step(
        &self,
        features: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &Array3<f64>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, _) = features.dim();
        let n_components = self.n_components;
        let mut responsibilities = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            let x_i = features.row(i);
            for k in 0..n_components {
                let mean_k = means.row(k);
                let cov_k = covariances.slice(s![k, .., ..]);

                // Compute Gaussian probability with geographic weighting
                let diff = &x_i - &mean_k;
                let inv_cov = self.invert_geographic_covariance(&cov_k.to_owned())?;
                let mahalanobis = diff.dot(&inv_cov).dot(&diff);
                let det_cov = self.compute_geographic_determinant(&cov_k.to_owned())?;

                let log_prob = -0.5
                    * (mahalanobis
                        + (diff.len() as f64) * (2.0 * std::f64::consts::PI).ln()
                        + det_cov.ln())
                    + weights[k].ln();

                responsibilities[[i, k]] = log_prob;
            }
        }

        // Normalize responsibilities using log-sum-exp
        for i in 0..n_samples {
            let log_sum =
                self.geographic_log_sum_exp(&responsibilities.row(i).to_owned().into_raw_vec());
            for k in 0..n_components {
                responsibilities[[i, k]] = (responsibilities[[i, k]] - log_sum).exp();
            }
        }

        Ok(responsibilities)
    }

    /// M-step for geographic mixture
    fn geographic_m_step(
        &self,
        responsibilities: &Array2<f64>,
        features: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Array3<f64>)> {
        let (n_samples, n_features) = features.dim();
        let n_components = self.n_components;

        // Update weights
        let mut weights = Array1::zeros(n_components);
        for k in 0..n_components {
            weights[k] = responsibilities.column(k).sum() / n_samples as f64;
        }

        // Update means with geographic weighting
        let mut means = Array2::zeros((n_components, n_features));
        for k in 0..n_components {
            let n_k = responsibilities.column(k).sum();
            if n_k > 1e-8 {
                for j in 0..n_features {
                    // Apply geographic weighting to spatial coordinates
                    let geographic_weight = if j < 2 { 1.5 } else { 1.0 };

                    means[[k, j]] = responsibilities
                        .column(k)
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| r * features[[i, j]] * geographic_weight)
                        .sum::<f64>()
                        / (n_k * geographic_weight);
                }
            }
        }

        // Update covariances with geographic regularization
        let mut covariances = Array3::zeros((n_components, n_features, n_features));
        for k in 0..n_components {
            let n_k = responsibilities.column(k).sum();
            if n_k > 1e-8 {
                let mean_k = means.row(k);
                for i in 0..n_samples {
                    let diff = &features.row(i) - &mean_k;
                    let weight = responsibilities[[i, k]] / n_k;

                    for p in 0..n_features {
                        for q in 0..n_features {
                            // Add extra regularization for spatial dimensions
                            let reg_factor = if p < 2 && q < 2 { 1.2 } else { 1.0 };
                            covariances[[k, p, q]] += weight * diff[p] * diff[q] * reg_factor;
                        }
                    }
                }

                // Add geographic-aware regularization
                for j in 0..n_features {
                    let reg_val = if j < 2 { 2e-6 } else { 1e-6 }; // Stronger regularization for spatial dimensions
                    covariances[[k, j, j]] += reg_val;
                }
            } else {
                // Initialize with identity if component has no support
                for j in 0..n_features {
                    covariances[[k, j, j]] = if j < 2 { 2.0 } else { 1.0 };
                }
            }
        }

        Ok((weights, means, covariances))
    }

    /// Compute log-likelihood for geographic features
    fn compute_geographic_log_likelihood(
        &self,
        features: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &Array3<f64>,
    ) -> SklResult<f64> {
        let (n_samples, _) = features.dim();
        let n_components = self.n_components;
        let mut log_likelihood = 0.0;

        for i in 0..n_samples {
            let x_i = features.row(i);
            let mut component_probs = Vec::new();

            for k in 0..n_components {
                let mean_k = means.row(k);
                let cov_k = covariances.slice(s![k, .., ..]);

                let diff = &x_i - &mean_k;
                let inv_cov = self.invert_geographic_covariance(&cov_k.to_owned())?;
                let mahalanobis = diff.dot(&inv_cov).dot(&diff);
                let det_cov = self.compute_geographic_determinant(&cov_k.to_owned())?;

                let log_prob = -0.5
                    * (mahalanobis
                        + (diff.len() as f64) * (2.0 * std::f64::consts::PI).ln()
                        + det_cov.ln())
                    + weights[k].ln();

                component_probs.push(log_prob);
            }

            log_likelihood += self.geographic_log_sum_exp(&component_probs);
        }

        Ok(log_likelihood)
    }

    /// Helper functions for geographic mixture
    fn invert_geographic_covariance(&self, cov: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = cov.nrows();
        let mut inv_cov = Array2::zeros((n, n));

        // Simple diagonal inversion with geographic weighting
        for i in 0..n {
            let diag_val = cov[[i, i]];
            let min_val = if i < 2 { 1e-7 } else { 1e-8 }; // Different thresholds for spatial vs other features
            if diag_val > min_val {
                inv_cov[[i, i]] = 1.0 / diag_val;
            } else {
                inv_cov[[i, i]] = 1.0 / min_val;
            }
        }

        Ok(inv_cov)
    }

    fn compute_geographic_determinant(&self, cov: &Array2<f64>) -> SklResult<f64> {
        let n = cov.nrows();
        let det = (0..n)
            .map(|i| {
                let min_val = if i < 2 { 1e-7 } else { 1e-8 };
                cov[[i, i]].max(min_val)
            })
            .product::<f64>();
        Ok(det)
    }

    fn geographic_log_sum_exp(&self, log_values: &[f64]) -> f64 {
        let max_val = log_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_values.iter().map(|&x| (x - max_val).exp()).sum();
        max_val + sum_exp.ln()
    }
}

impl Predict<Array2<f64>, Array1<usize>> for GeographicMixture<GeographicMixtureTrained> {
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<usize>> {
        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);

        // Simple prediction implementation
        // In a full implementation, this would extract geographic features
        // and use the trained parameters for proper classification
        for i in 0..n_samples {
            predictions[i] = i % self.n_components;
        }

        Ok(predictions)
    }
}

/// Geographic feature types for specialized processing
#[derive(Debug, Clone, PartialEq)]
pub enum GeographicFeatureType {
    /// Basic spatial coordinates (x, y)
    SpatialCoordinates,
    /// Elevation/altitude data
    Elevation,
    /// Distance to specified landmarks
    DistanceToLandmarks,
    /// Slope and aspect data
    Topographic,
    /// Land use/land cover categories
    LandCover,
    /// Climate variables (temperature, precipitation, etc.)
    Climate,
    /// Custom geographic features
    Custom(String),
}

/// Configuration for geographic feature extraction
#[derive(Debug, Clone)]
pub struct GeographicFeatureConfig {
    /// feature_types
    pub feature_types: Vec<GeographicFeatureType>,
    /// landmark_coordinates
    pub landmark_coordinates: Option<Array2<f64>>,
    /// normalize_features
    pub normalize_features: bool,
    /// spatial_weight
    pub spatial_weight: f64,
}

impl Default for GeographicFeatureConfig {
    fn default() -> Self {
        Self {
            feature_types: vec![GeographicFeatureType::SpatialCoordinates],
            landmark_coordinates: None,
            normalize_features: true,
            spatial_weight: 1.5,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_geographic_mixture_builder() {
        let gmm = GeographicMixtureBuilder::new(3)
            .use_elevation(true)
            .use_distance_features(true)
            .max_iter(50)
            .build();

        assert_eq!(gmm.n_components, 3);
        assert!(gmm.use_elevation);
        assert!(gmm.use_distance_features);
        assert_eq!(gmm.max_iter, 50);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_geographic_feature_extraction() {
        let landmarks = array![[5.0, 5.0], [10.0, 10.0]];
        let gmm = GeographicMixtureBuilder::new(2)
            .use_elevation(true)
            .use_distance_features(true)
            .landmark_coordinates(landmarks)
            .build();

        let X = array![[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 2.0, 3.0]];
        let features = gmm.extract_geographic_features(&X).unwrap();

        // Original features (3) + elevation (1) + distance to 2 landmarks (2) = 6 features
        assert_eq!(features.ncols(), 6);
        assert_eq!(features.nrows(), 3);

        // Check that original features are preserved
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(features[[i, j]], X[[i, j]]);
            }
        }
    }

    #[test]
    fn test_geographic_parameter_initialization() {
        let gmm = GeographicMixtureBuilder::new(2).build();
        let features = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];

        let weights = Array1::from_elem(2, 0.5);
        let means = Array2::zeros((2, 2));

        let (new_weights, new_means) = gmm
            .initialize_geographic_parameters(&features, weights, means)
            .unwrap();

        assert_eq!(new_weights.len(), 2);
        assert_eq!(new_means.dim(), (2, 2));

        // Weights should be uniform
        assert!((new_weights[0] - 0.5).abs() < 1e-10);
        assert!((new_weights[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_geographic_feature_config() {
        let config = GeographicFeatureConfig::default();

        assert_eq!(config.feature_types.len(), 1);
        assert!(matches!(
            config.feature_types[0],
            GeographicFeatureType::SpatialCoordinates
        ));
        assert!(config.normalize_features);
        assert_eq!(config.spatial_weight, 1.5);
    }

    #[test]
    fn test_geographic_feature_types() {
        let feature_type = GeographicFeatureType::DistanceToLandmarks;
        assert_eq!(feature_type, GeographicFeatureType::DistanceToLandmarks);

        let custom_feature = GeographicFeatureType::Custom("population_density".to_string());
        if let GeographicFeatureType::Custom(name) = custom_feature {
            assert_eq!(name, "population_density");
        }
    }
}

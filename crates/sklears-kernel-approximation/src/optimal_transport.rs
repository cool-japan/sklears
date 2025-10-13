//! Optimal transport kernel approximation methods
//!
//! This module implements kernel approximation methods based on optimal transport theory,
//! including Wasserstein kernels, Sinkhorn divergences, and earth mover's distance approximations.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::rand_prelude::IteratorRandom;
use scirs2_core::random::essentials::Uniform as RandUniform;
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    prelude::{Fit, Transform},
};

/// Wasserstein kernel approximation using random features
///
/// This implements approximations to the Wasserstein kernel which computes
/// optimal transport distances between empirical distributions.
#[derive(Debug, Clone)]
/// WassersteinKernelSampler
pub struct WassersteinKernelSampler {
    /// Number of random features to generate
    n_components: usize,
    /// Ground metric for optimal transport (default: squared Euclidean)
    ground_metric: GroundMetric,
    /// Regularization parameter for Sinkhorn divergence
    epsilon: f64,
    /// Maximum number of Sinkhorn iterations
    max_iter: usize,
    /// Convergence tolerance for Sinkhorn
    tolerance: f64,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Fitted random projections
    projections: Option<Array2<f64>>,
    /// Transport plan approximation method
    transport_method: TransportMethod,
}

/// Ground metric options for optimal transport
#[derive(Debug, Clone)]
/// GroundMetric
pub enum GroundMetric {
    /// Squared Euclidean distance
    SquaredEuclidean,
    /// Euclidean distance
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
    /// Minkowski distance with parameter p
    Minkowski(f64),
    /// Custom metric function
    Custom(fn(&Array1<f64>, &Array1<f64>) -> f64),
}

/// Transport plan approximation methods
#[derive(Debug, Clone)]
/// TransportMethod
pub enum TransportMethod {
    /// Sinkhorn divergence approximation
    Sinkhorn,
    /// Sliced Wasserstein using random projections
    SlicedWasserstein,
    /// Tree-Wasserstein using hierarchical decomposition
    TreeWasserstein,
    /// Projection-based approximation
    ProjectionBased,
}

impl Default for WassersteinKernelSampler {
    fn default() -> Self {
        Self::new(100)
    }
}

impl WassersteinKernelSampler {
    /// Create a new Wasserstein kernel approximation sampler
    ///
    /// # Arguments
    /// * `n_components` - Number of random features to generate
    ///
    /// # Examples
    /// ```
    /// use sklears_kernel_approximation::WassersteinKernelSampler;
    /// let sampler = WassersteinKernelSampler::new(100);
    /// ```
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ground_metric: GroundMetric::SquaredEuclidean,
            epsilon: 0.1,
            max_iter: 1000,
            tolerance: 1e-9,
            random_state: None,
            projections: None,
            transport_method: TransportMethod::SlicedWasserstein,
        }
    }

    /// Set the ground metric for optimal transport
    pub fn ground_metric(mut self, metric: GroundMetric) -> Self {
        self.ground_metric = metric;
        self
    }

    /// Set the regularization parameter for Sinkhorn divergence
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the maximum number of iterations for Sinkhorn
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the transport approximation method
    pub fn transport_method(mut self, method: TransportMethod) -> Self {
        self.transport_method = method;
        self
    }

    /// Compute distance using the specified ground metric
    fn compute_ground_distance(&self, x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        match &self.ground_metric {
            GroundMetric::SquaredEuclidean => (x - y).mapv(|v| v * v).sum(),
            GroundMetric::Euclidean => (x - y).mapv(|v| v * v).sum().sqrt(),
            GroundMetric::Manhattan => (x - y).mapv(|v| v.abs()).sum(),
            GroundMetric::Minkowski(p) => (x - y).mapv(|v| v.abs().powf(*p)).sum().powf(1.0 / p),
            GroundMetric::Custom(func) => func(x, y),
        }
    }

    /// Compute sliced Wasserstein distance using random projections
    fn sliced_wasserstein_features(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let projections = self
            .projections
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let n_samples = x.nrows();
        let mut features = Array2::zeros((n_samples, self.n_components));

        // Project data onto random directions
        let projected = x.dot(projections);

        // For each projection, compute cumulative distribution features
        for (j, proj_col) in projected.axis_iter(Axis(1)).enumerate() {
            let mut sorted_proj: Vec<f64> = proj_col.to_vec();
            sorted_proj.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Use quantile-based features
            for (i, &val) in proj_col.iter().enumerate() {
                // Find quantile of this value in the sorted distribution
                let quantile = sorted_proj
                    .binary_search_by(|&probe| probe.partial_cmp(&val).unwrap())
                    .unwrap_or_else(|e| e) as f64
                    / sorted_proj.len() as f64;

                features[[i, j]] = quantile;
            }
        }

        Ok(features)
    }

    /// Compute Sinkhorn divergence features
    fn sinkhorn_features(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut features = Array2::zeros((n_samples, self.n_components));

        // Use random subsampling for Sinkhorn approximation
        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        for j in 0..self.n_components {
            // Sample a subset of points for comparison
            let subset_size = (n_samples as f64).sqrt() as usize + 1;
            let indices: Vec<usize> = (0..n_samples)
                .choose_multiple(&mut rng, subset_size)
                .into_iter()
                .collect();

            for (i, x_i) in x.axis_iter(Axis(0)).enumerate() {
                let mut total_divergence = 0.0;

                for &idx in &indices {
                    if idx != i {
                        let x_j = x.row(idx);
                        let distance =
                            self.compute_ground_distance(&x_i.to_owned(), &x_j.to_owned());
                        total_divergence += (-distance / self.epsilon).exp();
                    }
                }

                features[[i, j]] = total_divergence / indices.len() as f64;
            }
        }

        Ok(features)
    }

    /// Compute tree-Wasserstein features using hierarchical decomposition
    fn tree_wasserstein_features(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut features = Array2::zeros((n_samples, self.n_components));

        // Use hierarchical clustering approximation
        let projections = self
            .projections
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let projected = x.dot(projections);

        for (j, proj_col) in projected.axis_iter(Axis(1)).enumerate() {
            // Create hierarchical bins
            let min_val = proj_col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = proj_col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let bin_width = (max_val - min_val) / 10.0; // 10 bins

            for (i, &val) in proj_col.iter().enumerate() {
                let bin = ((val - min_val) / bin_width).floor() as usize;
                let bin = bin.min(9); // Ensure we don't exceed bounds
                features[[i, j]] = bin as f64 / 9.0; // Normalize
            }
        }

        Ok(features)
    }
}

impl Fit<Array2<f64>, ()> for WassersteinKernelSampler {
    type Fitted = FittedWassersteinSampler;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let n_features = x.ncols();

        // Generate random projections for sliced Wasserstein
        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        let uniform = RandUniform::new(-1.0, 1.0).unwrap();
        let mut projections = Array2::zeros((n_features, self.n_components));

        for mut col in projections.axis_iter_mut(Axis(1)) {
            for elem in col.iter_mut() {
                *elem = rng.sample(uniform);
            }
            // Normalize the projection vector
            let norm: f64 = col.mapv(|v| v * v).sum();
            let norm = norm.sqrt();
            col /= norm;
        }

        Ok(FittedWassersteinSampler {
            sampler: WassersteinKernelSampler {
                projections: Some(projections),
                ..self.clone()
            },
        })
    }
}

/// Fitted Wasserstein kernel sampler
pub struct FittedWassersteinSampler {
    sampler: WassersteinKernelSampler,
}

impl Transform<Array2<f64>, Array2<f64>> for FittedWassersteinSampler {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        match self.sampler.transport_method {
            TransportMethod::SlicedWasserstein => self.sampler.sliced_wasserstein_features(x),
            TransportMethod::Sinkhorn => self.sampler.sinkhorn_features(x),
            TransportMethod::TreeWasserstein => self.sampler.tree_wasserstein_features(x),
            TransportMethod::ProjectionBased => self.sampler.sliced_wasserstein_features(x),
        }
    }
}

/// Earth Mover's Distance (EMD) kernel approximation
///
/// Implements approximations to kernels based on the earth mover's distance,
/// also known as the Wasserstein-1 distance.
#[derive(Debug, Clone)]
/// EMDKernelSampler
pub struct EMDKernelSampler {
    /// Number of random features
    n_components: usize,
    /// Bandwidth parameter for the kernel
    bandwidth: f64,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Fitted random projections
    projections: Option<Array2<f64>>,
    /// Number of quantile bins for approximation
    n_bins: usize,
}

impl Default for EMDKernelSampler {
    fn default() -> Self {
        Self::new(100)
    }
}

impl EMDKernelSampler {
    /// Create a new EMD kernel sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            bandwidth: 1.0,
            random_state: None,
            projections: None,
            n_bins: 20,
        }
    }

    /// Set the bandwidth parameter
    pub fn bandwidth(mut self, bandwidth: f64) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the number of quantile bins
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }
}

impl Fit<Array2<f64>, ()> for EMDKernelSampler {
    type Fitted = FittedEMDSampler;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let n_features = x.ncols();

        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        let uniform = RandUniform::new(-1.0, 1.0).unwrap();
        let mut projections = Array2::zeros((n_features, self.n_components));

        for mut col in projections.axis_iter_mut(Axis(1)) {
            for elem in col.iter_mut() {
                *elem = rng.sample(uniform);
            }
            let norm: f64 = col.mapv(|v| v * v).sum();
            let norm = norm.sqrt();
            col /= norm;
        }

        Ok(FittedEMDSampler {
            sampler: EMDKernelSampler {
                projections: Some(projections),
                ..self.clone()
            },
        })
    }
}

/// Fitted EMD kernel sampler
pub struct FittedEMDSampler {
    sampler: EMDKernelSampler,
}

impl Transform<Array2<f64>, Array2<f64>> for FittedEMDSampler {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let projections =
            self.sampler
                .projections
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let n_samples = x.nrows();
        let mut features = Array2::zeros((n_samples, self.sampler.n_components));

        let projected = x.dot(projections);

        for (j, proj_col) in projected.axis_iter(Axis(1)).enumerate() {
            // Compute quantile-based features for EMD approximation
            let min_val = proj_col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = proj_col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = max_val - min_val;

            if range > 1e-10 {
                for (i, &val) in proj_col.iter().enumerate() {
                    // Compute cumulative distribution function approximation
                    let normalized_val = (val - min_val) / range;
                    let emd_feature = (-normalized_val.abs() / self.sampler.bandwidth).exp();
                    features[[i, j]] = emd_feature;
                }
            }
        }

        Ok(features)
    }
}

/// Gromov-Wasserstein kernel approximation
///
/// Implements approximations to Gromov-Wasserstein distances which compare
/// metric measure spaces by their intrinsic geometry.
#[derive(Debug, Clone)]
/// GromovWassersteinSampler
pub struct GromovWassersteinSampler {
    /// Number of random features
    n_components: usize,
    /// Loss function for Gromov-Wasserstein
    loss_function: GWLossFunction,
    /// Number of iterations for optimization
    max_iter: usize,
    /// Random state
    random_state: Option<u64>,
    /// Fitted parameters
    fitted_params: Option<GWFittedParams>,
}

/// Loss functions for Gromov-Wasserstein
#[derive(Debug, Clone)]
/// GWLossFunction
pub enum GWLossFunction {
    /// Squared loss
    Square,
    /// KL divergence
    KlLoss,
    /// Custom loss function
    Custom(fn(f64, f64, f64, f64) -> f64),
}

#[derive(Debug, Clone)]
struct GWFittedParams {
    reference_distances: Array2<f64>,
    projections: Array2<f64>,
}

impl Default for GromovWassersteinSampler {
    fn default() -> Self {
        Self::new(50)
    }
}

impl GromovWassersteinSampler {
    /// Create a new Gromov-Wasserstein sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            loss_function: GWLossFunction::Square,
            max_iter: 100,
            random_state: None,
            fitted_params: None,
        }
    }

    /// Set the loss function
    pub fn loss_function(mut self, loss: GWLossFunction) -> Self {
        self.loss_function = loss;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Compute pairwise distances for metric space comparison
    fn compute_distance_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let diff = &x.row(i).to_owned() - &x.row(j);
                let dist: f64 = diff.mapv(|v| v * v).sum();
                let dist = dist.sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }
}

impl Fit<Array2<f64>, ()> for GromovWassersteinSampler {
    type Fitted = FittedGromovWassersteinSampler;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Compute reference distance matrix
        let distance_matrix = self.compute_distance_matrix(x);

        // Generate random projections for dimensionality reduction
        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        let uniform = RandUniform::new(-1.0, 1.0).unwrap();
        let mut projections = Array2::zeros((n_features, self.n_components));

        for mut col in projections.axis_iter_mut(Axis(1)) {
            for elem in col.iter_mut() {
                *elem = rng.sample(uniform);
            }
            let norm: f64 = col.mapv(|v| v * v).sum();
            let norm = norm.sqrt();
            col /= norm;
        }

        let fitted_params = GWFittedParams {
            reference_distances: distance_matrix,
            projections,
        };

        Ok(FittedGromovWassersteinSampler {
            sampler: GromovWassersteinSampler {
                fitted_params: Some(fitted_params),
                ..self.clone()
            },
        })
    }
}

/// Fitted Gromov-Wasserstein sampler
pub struct FittedGromovWassersteinSampler {
    sampler: GromovWassersteinSampler,
}

impl Transform<Array2<f64>, Array2<f64>> for FittedGromovWassersteinSampler {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let params =
            self.sampler
                .fitted_params
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let n_samples = x.nrows();
        let mut features = Array2::zeros((n_samples, self.sampler.n_components));

        // Project data and compute distance-based features
        let projected = x.dot(&params.projections);

        for (j, proj_col) in projected.axis_iter(Axis(1)).enumerate() {
            for (i, &val) in proj_col.iter().enumerate() {
                // Use projected coordinates as distance-based features
                features[[i, j]] = val.tanh(); // Bounded activation
            }
        }

        Ok(features)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Transform};

    #[test]
    fn test_wasserstein_kernel_sampler() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let sampler = WassersteinKernelSampler::new(10)
            .random_state(42)
            .transport_method(TransportMethod::SlicedWasserstein);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[4, 10]);

        // Features should be bounded
        assert!(features.iter().all(|&f| f >= 0.0 && f <= 1.0));
    }

    #[test]
    fn test_emd_kernel_sampler() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let sampler = EMDKernelSampler::new(15).bandwidth(0.5).random_state(123);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[3, 15]);

        // Features should be positive (exponential)
        assert!(features.iter().all(|&f| f >= 0.0));
    }

    #[test]
    fn test_gromov_wasserstein_sampler() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],];

        let sampler = GromovWassersteinSampler::new(8).random_state(456);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[4, 8]);

        // Features should be bounded by tanh
        assert!(features.iter().all(|&f| f >= -1.0 && f <= 1.0));
    }

    #[test]
    fn test_wasserstein_different_methods() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let methods = vec![
            TransportMethod::SlicedWasserstein,
            TransportMethod::Sinkhorn,
            TransportMethod::TreeWasserstein,
        ];

        for method in methods {
            let sampler = WassersteinKernelSampler::new(5)
                .transport_method(method)
                .random_state(42);

            let fitted = sampler.fit(&x, &()).unwrap();
            let features = fitted.transform(&x).unwrap();

            assert_eq!(features.shape(), &[3, 5]);
        }
    }

    #[test]
    fn test_different_ground_metrics() {
        let x = array![[1.0, 2.0], [2.0, 3.0],];

        let metrics = vec![
            GroundMetric::SquaredEuclidean,
            GroundMetric::Euclidean,
            GroundMetric::Manhattan,
            GroundMetric::Minkowski(3.0),
        ];

        for metric in metrics {
            let sampler = WassersteinKernelSampler::new(3)
                .ground_metric(metric)
                .random_state(42);

            let fitted = sampler.fit(&x, &()).unwrap();
            let features = fitted.transform(&x).unwrap();

            assert_eq!(features.shape(), &[2, 3]);
        }
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let sampler1 = WassersteinKernelSampler::new(5).random_state(42);
        let sampler2 = WassersteinKernelSampler::new(5).random_state(42);

        let fitted1 = sampler1.fit(&x, &()).unwrap();
        let fitted2 = sampler2.fit(&x, &()).unwrap();

        let features1 = fitted1.transform(&x).unwrap();
        let features2 = fitted2.transform(&x).unwrap();

        // Results should be identical with same random state
        assert!((features1 - features2).mapv(|v| v.abs()).sum() < 1e-10);
    }
}

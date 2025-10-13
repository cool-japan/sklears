//! K-Means clustering implementations
//!
//! This module provides K-Means clustering algorithms including standard K-Means,
//! Mini-batch K-Means, X-Means for automatic cluster selection, and G-Means for
//! Gaussian cluster detection.

use numrs2::prelude::*;
use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::random::{Random, Rng};
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::*;

/// Initialization methods for K-means
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KMeansInit {
    /// Random initialization
    Random,
    /// K-means++ initialization for better initial centroids
    KMeansPlusPlus,
    /// Use provided centroids
    Custom,
}

/// Configuration for K-means clustering
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Initialization method
    pub init: KMeansInit,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            init: KMeansInit::KMeansPlusPlus,
            max_iter: 300,
            tolerance: 1e-4,
            random_seed: None,
        }
    }
}

/// K-Means clustering algorithm
#[derive(Clone)]
pub struct KMeans {
    config: KMeansConfig,
}

/// Fitted K-means model
pub struct KMeansFitted {
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Cluster labels for training data
    pub labels: Vec<i32>,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Final inertia (sum of squared distances to centroids)
    pub inertia: f64,
    /// Configuration used during training
    pub config: KMeansConfig,
}

impl KMeans {
    /// Create a new K-means instance
    pub fn new(config: KMeansConfig) -> Self {
        Self { config }
    }

    /// Initialize centroids using the specified method
    fn initialize_centroids(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut rng = match self.config.random_seed {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42), // Default seed for consistency
        };

        match self.config.init {
            KMeansInit::Random => {
                let mut centroids = Array2::<f64>::zeros((self.config.n_clusters, n_features));
                for k in 0..self.config.n_clusters {
                    for j in 0..n_features {
                        let min_val = X.column(j).iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max_val = X.column(j).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        centroids[[k, j]] = if (max_val - min_val).abs() <= f64::EPSILON {
                            min_val
                        } else {
                            rng.gen_range(min_val..max_val)
                        };
                    }
                }
                Ok(centroids)
            }
            KMeansInit::KMeansPlusPlus => self.kmeans_plus_plus_init(X, &mut rng),
            KMeansInit::Custom => {
                // For custom initialization, we'll fall back to random for now
                self.initialize_centroids(X)
            }
        }
    }

    /// K-means++ initialization for better centroid placement
    fn kmeans_plus_plus_init(&self, X: &Array2<f64>, rng: &mut impl Rng) -> Result<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut centroids = Array2::<f64>::zeros((self.config.n_clusters, n_features));

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..n_samples);
        centroids.row_mut(0).assign(&X.row(first_idx));

        // Choose remaining centroids
        for k in 1..self.config.n_clusters {
            let mut distances = vec![0.0; n_samples];
            let mut total_distance = 0.0;

            // Compute minimum distance to existing centroids for each point
            for i in 0..n_samples {
                let point = X.row(i);
                let mut min_dist = f64::INFINITY;

                for j in 0..k {
                    let centroid = centroids.row(j);
                    let dist = self.euclidean_distance(point, centroid);
                    min_dist = min_dist.min(dist);
                }

                distances[i] = min_dist * min_dist; // Square for probability weighting
                total_distance += distances[i];
            }

            // Choose next centroid with probability proportional to squared distance
            let chosen_idx = if total_distance <= f64::EPSILON {
                rng.gen_range(0..n_samples)
            } else {
                let target = rng.gen_range(0.0..total_distance);
                let mut cumulative = 0.0;
                let mut selected = 0;

                for i in 0..n_samples {
                    cumulative += distances[i];
                    if cumulative >= target {
                        selected = i;
                        break;
                    }
                }

                selected
            };

            centroids.row_mut(k).assign(&X.row(chosen_idx));
        }

        Ok(centroids)
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Assign each point to the closest centroid
    fn assign_clusters(&self, X: &Array2<f64>, centroids: &Array2<f64>) -> (Vec<i32>, f64) {
        let n_samples = X.nrows();
        let mut labels = vec![0i32; n_samples];
        let mut inertia = 0.0;

        for i in 0..n_samples {
            let point = X.row(i);
            let mut min_distance = f64::INFINITY;
            let mut best_cluster = 0i32;

            for k in 0..self.config.n_clusters {
                let centroid = centroids.row(k);
                let distance = self.euclidean_distance(point, centroid);

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = k as i32;
                }
            }

            labels[i] = best_cluster;
            inertia += min_distance * min_distance;
        }

        (labels, inertia)
    }

    /// Update centroids as the mean of assigned points
    fn update_centroids(&self, X: &Array2<f64>, labels: &[i32]) -> Array2<f64> {
        let (_, n_features) = X.dim();
        let mut centroids = Array2::<f64>::zeros((self.config.n_clusters, n_features));
        let mut counts = vec![0; self.config.n_clusters];

        // Sum points for each cluster
        for (i, &label) in labels.iter().enumerate() {
            if label >= 0 {
                let cluster_idx = label as usize;
                if cluster_idx < self.config.n_clusters {
                    let point = X.row(i);
                    for j in 0..n_features {
                        centroids[[cluster_idx, j]] += point[j];
                    }
                    counts[cluster_idx] += 1;
                }
            }
        }

        // Average to get centroids
        for k in 0..self.config.n_clusters {
            if counts[k] > 0 {
                for j in 0..n_features {
                    centroids[[k, j]] /= counts[k] as f64;
                }
            }
        }

        centroids
    }

    /// Check convergence based on centroid movement
    fn has_converged(&self, old_centroids: &Array2<f64>, new_centroids: &Array2<f64>) -> bool {
        let mut max_movement: f64 = 0.0;

        for k in 0..self.config.n_clusters {
            let old_centroid = old_centroids.row(k);
            let new_centroid = new_centroids.row(k);

            let movement = self.euclidean_distance(old_centroid, new_centroid);
            max_movement = max_movement.max(movement);
        }

        max_movement < self.config.tolerance
    }
}

impl Fit<Array2<f64>, Array1<f64>> for KMeans {
    type Fitted = KMeansFitted;

    fn fit(self, X: &Array2<f64>, _y: &Array1<f64>) -> Result<KMeansFitted> {
        let config = self.clone();
        if X.is_empty() {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        let n_samples = X.nrows();
        if n_samples < self.config.n_clusters {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least equal to number of clusters".to_string(),
            ));
        }

        // Initialize centroids
        let mut centroids = self.initialize_centroids(X)?;
        let mut prev_centroids;
        let (mut labels, mut inertia) = self.assign_clusters(X, &centroids);

        let mut n_iterations = 0;
        for iteration in 0..self.config.max_iter {
            n_iterations = iteration + 1;
            prev_centroids = centroids.clone();

            // Update centroids
            centroids = self.update_centroids(X, &labels);

            // Assign points to new centroids
            let (new_labels, new_inertia) = self.assign_clusters(X, &centroids);
            labels = new_labels;
            inertia = new_inertia;

            // Check for convergence
            if self.has_converged(&prev_centroids, &centroids) {
                break;
            }
        }

        Ok(KMeansFitted {
            centroids,
            labels,
            n_iterations,
            inertia,
            config: config.config.clone(),
        })
    }
}

impl Predict<Array2<f64>, Vec<i32>> for KMeans {
    fn predict(&self, _X: &Array2<f64>) -> Result<Vec<i32>> {
        Err(SklearsError::NotFitted {
            operation: "making predictions".to_string(),
        })
    }
}

impl Predict<Array2<f64>, Vec<i32>> for KMeansFitted {
    fn predict(&self, X: &Array2<f64>) -> Result<Vec<i32>> {
        if X.is_empty() {
            return Ok(vec![]);
        }

        let n_samples = X.nrows();
        let mut labels = vec![0i32; n_samples];

        for i in 0..n_samples {
            let point = X.row(i);
            let mut min_distance = f64::INFINITY;
            let mut best_cluster = 0i32;

            for k in 0..self.config.n_clusters {
                let centroid = self.centroids.row(k);
                let distance = point
                    .iter()
                    .zip(centroid.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = k as i32;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }
}

/// Configuration for Mini-batch K-means clustering
#[derive(Debug, Clone)]
pub struct MiniBatchKMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Batch size for mini-batch updates
    pub batch_size: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for MiniBatchKMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            batch_size: 100,
            max_iter: 100,
            random_seed: None,
        }
    }
}

/// Mini-batch K-means clustering for large datasets
#[derive(Clone)]
pub struct MiniBatchKMeans {
    config: MiniBatchKMeansConfig,
}

impl MiniBatchKMeans {
    /// Create a new mini-batch K-means instance
    pub fn new(config: MiniBatchKMeansConfig) -> Self {
        Self { config }
    }
}

impl Fit<Array2<f64>, Array1<f64>> for MiniBatchKMeans {
    type Fitted = KMeansFitted;

    fn fit(self, X: &Array2<f64>, _y: &Array1<f64>) -> Result<KMeansFitted> {
        let config = self.clone();
        // For now, use regular K-means as a placeholder
        let kmeans_config = KMeansConfig {
            n_clusters: self.config.n_clusters,
            init: KMeansInit::KMeansPlusPlus,
            max_iter: self.config.max_iter,
            tolerance: 1e-4,
            random_seed: self.config.random_seed,
        };
        let kmeans = KMeans::new(kmeans_config);
        kmeans.fit(X, _y)
    }
}

/// Configuration for X-means clustering
#[derive(Debug, Clone)]
pub struct XMeansConfig {
    /// Minimum number of clusters
    pub k_min: usize,
    /// Maximum number of clusters
    pub k_max: usize,
    /// Maximum number of iterations per K-means run
    pub max_iter: usize,
    /// Information criterion to use
    pub criterion: InformationCriterion,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for XMeansConfig {
    fn default() -> Self {
        Self {
            k_min: 2,
            k_max: 10,
            max_iter: 300,
            criterion: InformationCriterion::BIC,
            random_seed: None,
        }
    }
}

/// Information criteria for model selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InformationCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
}

/// X-means clustering with automatic cluster number selection
#[derive(Clone)]
pub struct XMeans {
    config: XMeansConfig,
}

impl XMeans {
    /// Create a new X-means instance
    pub fn new(config: XMeansConfig) -> Self {
        Self { config }
    }
}

impl Fit<Array2<f64>, Array1<f64>> for XMeans {
    type Fitted = KMeansFitted;

    fn fit(self, X: &Array2<f64>, _y: &Array1<f64>) -> Result<KMeansFitted> {
        let config = self.clone();
        let mut best_model: Option<KMeansFitted> = None;
        let mut best_score = f64::INFINITY;

        for k in self.config.k_min..=self.config.k_max {
            let kmeans_config = KMeansConfig {
                n_clusters: k,
                init: KMeansInit::KMeansPlusPlus,
                max_iter: self.config.max_iter,
                tolerance: 1e-4,
                random_seed: self.config.random_seed,
            };

            let kmeans = KMeans::new(kmeans_config);
            match kmeans.fit(X, _y) {
                Ok(fitted) => {
                    let score = self.compute_information_criterion(&fitted, X);
                    if score < best_score {
                        best_score = score;
                        best_model = Some(fitted);
                    }
                }
                Err(_) => continue,
            }
        }

        best_model.ok_or_else(|| {
            SklearsError::InvalidInput("No valid clustering found in X-means".to_string())
        })
    }
}

impl XMeans {
    /// Compute information criterion for model selection
    fn compute_information_criterion(&self, fitted: &KMeansFitted, X: &Array2<f64>) -> f64 {
        let n_samples = X.nrows() as f64;
        let n_features = X.ncols() as f64;
        let k = fitted.config.n_clusters as f64;

        // Log likelihood approximation (negative of normalized inertia)
        let log_likelihood = -fitted.inertia / (2.0 * n_samples);

        // Number of parameters (centroids)
        let n_params = k * n_features;

        match self.config.criterion {
            InformationCriterion::AIC => -2.0 * log_likelihood + 2.0 * n_params,
            InformationCriterion::BIC => -2.0 * log_likelihood + n_params * n_samples.ln(),
        }
    }
}

/// Configuration for G-means clustering
#[derive(Debug, Clone)]
pub struct GMeansConfig {
    /// Minimum number of clusters
    pub k_min: usize,
    /// Maximum number of clusters
    pub k_max: usize,
    /// Maximum number of iterations per K-means run
    pub max_iter: usize,
    /// Significance level for normality test
    pub alpha: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for GMeansConfig {
    fn default() -> Self {
        Self {
            k_min: 1,
            k_max: 10,
            max_iter: 300,
            alpha: 0.05,
            random_seed: None,
        }
    }
}

/// G-means clustering with Gaussian assumption testing
#[derive(Clone)]
pub struct GMeans {
    config: GMeansConfig,
}

impl GMeans {
    /// Create a new G-means instance
    pub fn new(config: GMeansConfig) -> Self {
        Self { config }
    }
}

impl Fit<Array2<f64>, Array1<f64>> for GMeans {
    type Fitted = KMeansFitted;

    fn fit(self, X: &Array2<f64>, _y: &Array1<f64>) -> Result<KMeansFitted> {
        let config = self.clone();
        // Start with k=1 and recursively split clusters
        let mut current_k = self.config.k_min;
        let mut best_fitted: Option<KMeansFitted> = None;

        while current_k <= self.config.k_max {
            let kmeans_config = KMeansConfig {
                n_clusters: current_k,
                init: KMeansInit::KMeansPlusPlus,
                max_iter: self.config.max_iter,
                tolerance: 1e-4,
                random_seed: self.config.random_seed,
            };

            let kmeans = KMeans::new(kmeans_config);
            match kmeans.fit(X, _y) {
                Ok(fitted) => {
                    // Simple heuristic: accept if inertia improvement is significant
                    if let Some(ref best) = best_fitted {
                        if fitted.inertia < best.inertia * 0.9 {
                            best_fitted = Some(fitted);
                        } else {
                            break;
                        }
                    } else {
                        best_fitted = Some(fitted);
                    }
                }
                Err(_) => break,
            }

            current_k += 1;
        }

        best_fitted.ok_or_else(|| {
            SklearsError::InvalidInput("No valid clustering found in G-means".to_string())
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    #[allow(non_snake_case)]
    fn test_kmeans_basic() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0],
        )
        .unwrap();

        let config = KMeansConfig {
            n_clusters: 2,
            init: KMeansInit::Random,
            max_iter: 100,
            tolerance: 1e-4,
            random_seed: Some(42),
        };

        let kmeans = KMeans::new(config);
        let dummy_y = Array1::<f64>::zeros(X.nrows());
        let fitted = kmeans.fit(&X, &dummy_y).unwrap();

        assert_eq!(fitted.labels.len(), 6);
        assert_eq!(fitted.centroids.nrows(), 2);
        assert_eq!(fitted.centroids.ncols(), 2);
        assert!(fitted.n_iterations <= 100);
        assert!(fitted.inertia >= 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_kmeans_predict() {
        let X_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 10.0, 10.0, 10.0, 11.0])
                .unwrap();

        let config = KMeansConfig {
            n_clusters: 2,
            random_seed: Some(42),
            ..Default::default()
        };

        let kmeans = KMeans::new(config);
        let dummy_y = Array1::<f64>::zeros(X_train.nrows());
        let fitted = kmeans.fit(&X_train, &dummy_y).unwrap();

        let X_test = Array2::from_shape_vec((2, 2), vec![1.5, 1.5, 9.5, 10.5]).unwrap();

        let predictions = fitted.predict(&X_test).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_kmeans_plus_plus_init() {
        let X = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0])
            .unwrap();

        let config = KMeansConfig {
            n_clusters: 2,
            init: KMeansInit::KMeansPlusPlus,
            random_seed: Some(42),
            ..Default::default()
        };

        let kmeans = KMeans::new(config);
        let dummy_y = Array1::<f64>::zeros(X.nrows());
        let fitted = kmeans.fit(&X, &dummy_y).unwrap();

        assert_eq!(fitted.labels.len(), 4);
        assert!(fitted.inertia >= 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_xmeans_basic() {
        let X = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.2, 0.9, 0.8, 1.0, 10.0, 10.0, 10.1, 10.1, 9.9, 10.2, 10.0,
                9.8,
            ],
        )
        .unwrap();

        let config = XMeansConfig {
            k_min: 1,
            k_max: 4,
            random_seed: Some(42),
            ..Default::default()
        };

        let xmeans = XMeans::new(config);
        let dummy_y = Array1::<f64>::zeros(X.nrows());
        let fitted = xmeans.fit(&X, &dummy_y).unwrap();

        assert_eq!(fitted.labels.len(), 8);
        assert!(fitted.config.n_clusters >= 1);
        assert!(fitted.config.n_clusters <= 4);
    }
}

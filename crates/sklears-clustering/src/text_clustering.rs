//! Text and High-Dimensional Clustering Algorithms
//!
//! This module provides specialized clustering algorithms designed for text data
//! and other high-dimensional sparse datasets commonly found in natural language
//! processing and information retrieval.
//!
//! # Algorithms Provided
//! - **Spherical K-Means**: K-Means clustering on the unit sphere using cosine distance
//! - **Document Clustering**: Specialized clustering for document collections
//! - **Concept Clustering**: Semantic clustering using vector embeddings
//!
//! # Mathematical Background
//!
//! ## Spherical K-Means
//! Spherical K-Means operates on normalized vectors on the unit sphere, making it
//! particularly suitable for text data where the magnitude of feature vectors is
//! less important than their direction. The algorithm uses cosine similarity/distance:
//!
//! cos(θ) = (u · v) / (||u|| ||v||) for normalized vectors ||u|| = ||v|| = 1
//!
//! ## Benefits for Text Data
//! - Invariant to document length
//! - Focuses on term frequency ratios rather than absolute counts
//! - Naturally handles high-dimensional sparse features
//! - Robust to scaling differences between documents

use std::collections::HashMap;

use numrs2::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::Distribution;
// Normal distribution via scirs2_core::random::RandNormal
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::*;

/// Configuration for spherical K-means clustering
#[derive(Debug, Clone)]
pub struct SphericalKMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance for centroid movement
    pub tolerance: f64,
    /// Initialization method
    pub init: SphericalInit,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Whether to normalize input data to unit vectors
    pub normalize_input: bool,
}

/// Initialization methods for spherical K-means
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SphericalInit {
    /// Random initialization on the unit sphere
    Random,
    /// K-means++ initialization adapted for spherical distance
    SphericalKMeansPlusPlus,
    /// Use random samples from input data (normalized)
    RandomSamples,
}

impl Default for SphericalKMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            max_iter: 300,
            tolerance: 1e-4,
            init: SphericalInit::SphericalKMeansPlusPlus,
            random_seed: None,
            normalize_input: true,
        }
    }
}

/// Spherical K-Means clustering algorithm
///
/// Clusters data on the unit sphere using cosine similarity as the distance metric.
/// This is particularly effective for text data and other high-dimensional sparse datasets.
#[derive(Clone)]
pub struct SphericalKMeans {
    config: SphericalKMeansConfig,
}

/// Fitted spherical K-means model
pub struct SphericalKMeansFitted {
    /// Cluster centroids (normalized to unit vectors)
    pub centroids: Array2<f64>,
    /// Cluster labels for training data
    pub labels: Vec<i32>,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Final inertia (sum of squared cosine distances)
    pub inertia: f64,
    /// Configuration used during training
    pub config: SphericalKMeansConfig,
}

impl SphericalKMeans {
    /// Create a new spherical K-means instance
    pub fn new(config: SphericalKMeansConfig) -> Self {
        Self { config }
    }

    /// Normalize data to unit vectors (L2 normalization)
    fn normalize_data(&self, X: &Array2<f64>) -> Array2<f64> {
        let mut normalized = X.clone();
        for mut row in normalized.rows_mut() {
            let norm = row.mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                row /= norm;
            }
        }
        normalized
    }

    /// Compute cosine similarity between two normalized vectors
    fn cosine_similarity(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        // For normalized vectors, cosine similarity is just the dot product
        a.dot(&b)
    }

    /// Compute cosine distance between two normalized vectors
    fn cosine_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        1.0 - self.cosine_similarity(a, b)
    }

    /// Initialize centroids using the specified method
    fn initialize_centroids(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut rng = thread_rng();

        match self.config.init {
            SphericalInit::Random => {
                let mut centroids = Array2::<f64>::zeros((self.config.n_clusters, n_features));
                let normal = scirs2_core::random::RandNormal::new(0.0, 1.0).unwrap();
                for mut centroid in centroids.rows_mut() {
                    // Generate random vector and normalize
                    for j in 0..n_features {
                        centroid[j] = normal.sample(&mut rng);
                    }
                    let norm = centroid.mapv(|x| x * x).sum().sqrt();
                    if norm > 1e-10 {
                        centroid /= norm;
                    }
                }
                Ok(centroids)
            }
            SphericalInit::SphericalKMeansPlusPlus => self.kmeans_plus_plus_spherical(X, &mut rng),
            SphericalInit::RandomSamples => {
                let mut centroids = Array2::<f64>::zeros((self.config.n_clusters, n_features));
                let mut selected_indices = Vec::new();
                for _ in 0..self.config.n_clusters {
                    selected_indices.push(rng.gen_range(0..n_samples));
                }

                for (i, &idx) in selected_indices.iter().enumerate() {
                    centroids.row_mut(i).assign(&X.row(idx));
                }
                Ok(centroids)
            }
        }
    }

    /// Spherical K-means++ initialization
    fn kmeans_plus_plus_spherical(
        &self,
        X: &Array2<f64>,
        rng: &mut impl Rng,
    ) -> Result<Array2<f64>> {
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
                    let dist = self.cosine_distance(point, centroid);
                    min_dist = min_dist.min(dist);
                }

                distances[i] = min_dist * min_dist; // Square for probability weighting
                total_distance += distances[i];
            }

            // Choose next centroid with probability proportional to squared distance
            let target = rng.gen_range(0.0..total_distance);
            let mut cumulative = 0.0;
            let mut chosen_idx = 0;

            for i in 0..n_samples {
                cumulative += distances[i];
                if cumulative >= target {
                    chosen_idx = i;
                    break;
                }
            }

            centroids.row_mut(k).assign(&X.row(chosen_idx));
        }

        Ok(centroids)
    }

    /// Assign each point to the closest centroid
    fn assign_clusters(&self, X: &Array2<f64>, centroids: &Array2<f64>) -> (Vec<i32>, f64) {
        let n_samples = X.nrows();
        let mut labels = vec![0i32; n_samples];
        let mut inertia = 0.0;

        for i in 0..n_samples {
            let point = X.row(i);
            let mut best_similarity = -1.0; // Start with worst possible cosine similarity
            let mut best_cluster = 0i32;

            for k in 0..self.config.n_clusters {
                let centroid = centroids.row(k);
                let similarity = self.cosine_similarity(point, centroid);

                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_cluster = k as i32;
                }
            }

            labels[i] = best_cluster;
            // Inertia is based on cosine distance
            inertia += self.cosine_distance(point, centroids.row(best_cluster as usize));
        }

        (labels, inertia)
    }

    /// Update centroids as the mean of assigned points (then normalize)
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

        // Normalize centroids to unit vectors
        for k in 0..self.config.n_clusters {
            if counts[k] > 0 {
                // Average the vectors
                for j in 0..n_features {
                    centroids[[k, j]] /= counts[k] as f64;
                }

                // Normalize to unit vector
                let norm = centroids.row(k).mapv(|x| x * x).sum().sqrt();
                if norm > 1e-10 {
                    for j in 0..n_features {
                        centroids[[k, j]] /= norm;
                    }
                }
            } else {
                // Reinitialize empty cluster randomly
                let mut rng = thread_rng(); // Use thread-local RNG
                let normal = scirs2_core::random::RandNormal::new(0.0, 1.0).unwrap();
                for j in 0..n_features {
                    centroids[[k, j]] = normal.sample(&mut rng);
                }
                let norm = centroids.row(k).mapv(|x| x * x).sum().sqrt();
                if norm > 1e-10 {
                    for j in 0..n_features {
                        centroids[[k, j]] /= norm;
                    }
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

            // Use cosine distance to measure centroid movement
            let movement = self.cosine_distance(old_centroid, new_centroid);
            max_movement = max_movement.max(movement);
        }

        max_movement < self.config.tolerance
    }
}

impl Fit<Array2<f64>, Array1<f64>> for SphericalKMeans {
    type Fitted = SphericalKMeansFitted;

    fn fit(self, X: &Array2<f64>, _y: &Array1<f64>) -> Result<SphericalKMeansFitted> {
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

        // Normalize input data if requested
        let data = if self.config.normalize_input {
            self.normalize_data(X)
        } else {
            X.clone()
        };

        // Initialize centroids
        let mut centroids = self.initialize_centroids(&data)?;
        let mut prev_centroids;
        let (mut labels, mut inertia) = self.assign_clusters(&data, &centroids);

        let mut n_iterations = 0;
        for iteration in 0..self.config.max_iter {
            n_iterations = iteration + 1;
            prev_centroids = centroids.clone();

            // Update centroids
            centroids = self.update_centroids(&data, &labels);

            // Assign points to new centroids
            let (new_labels, new_inertia) = self.assign_clusters(&data, &centroids);
            labels = new_labels;
            inertia = new_inertia;

            // Check for convergence
            if self.has_converged(&prev_centroids, &centroids) {
                break;
            }
        }

        Ok(SphericalKMeansFitted {
            centroids,
            labels,
            n_iterations,
            inertia,
            config: config.config.clone(),
        })
    }
}

impl Predict<Array2<f64>, Vec<i32>> for SphericalKMeansFitted {
    fn predict(&self, X: &Array2<f64>) -> Result<Vec<i32>> {
        if X.is_empty() {
            return Ok(vec![]);
        }

        // Normalize input data if requested
        let data = if self.config.normalize_input {
            let mut normalized = X.clone();
            for mut row in normalized.rows_mut() {
                let norm = row.mapv(|x| x * x).sum().sqrt();
                if norm > 1e-10 {
                    row /= norm;
                }
            }
            normalized
        } else {
            X.clone()
        };

        let n_samples = data.nrows();
        let mut labels = vec![0i32; n_samples];

        for i in 0..n_samples {
            let point = data.row(i);
            let mut best_similarity = -1.0;
            let mut best_cluster = 0i32;

            for k in 0..self.config.n_clusters {
                let centroid = self.centroids.row(k);
                let similarity = point.dot(&centroid); // Cosine similarity for normalized vectors

                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_cluster = k as i32;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }
}

/// Document Clustering using spherical K-means with TF-IDF preprocessing
#[derive(Debug, Clone)]
pub struct DocumentClustering {
    pub config: DocumentClusteringConfig,
}

/// Configuration for document clustering
#[derive(Debug, Clone)]
pub struct DocumentClusteringConfig {
    /// Spherical K-means configuration
    pub spherical_config: SphericalKMeansConfig,
    /// Minimum document frequency for terms
    pub min_df: f64,
    /// Maximum document frequency for terms
    pub max_df: f64,
    /// Maximum number of features (vocabulary size)
    pub max_features: Option<usize>,
    /// Whether to use TF-IDF weighting
    pub use_tfidf: bool,
}

impl Default for DocumentClusteringConfig {
    fn default() -> Self {
        Self {
            spherical_config: SphericalKMeansConfig::default(),
            min_df: 0.01,
            max_df: 0.95,
            max_features: Some(10000),
            use_tfidf: true,
        }
    }
}

/// Result of document clustering
#[derive(Debug, Clone)]
pub struct DocumentClusteringResult {
    /// Cluster labels for documents
    pub labels: Vec<i32>,
    /// Cluster centroids in feature space
    pub centroids: Array2<f64>,
    /// Number of iterations
    pub n_iterations: usize,
    /// Final inertia
    pub inertia: f64,
    /// Vocabulary mapping (if available)
    pub vocabulary: HashMap<String, usize>,
}

impl DocumentClustering {
    /// Create a new document clustering instance
    pub fn new(config: DocumentClusteringConfig) -> Self {
        Self { config }
    }

    /// Apply TF-IDF weighting to term frequency matrix
    fn apply_tfidf(&self, tf_matrix: &Array2<f64>) -> Array2<f64> {
        let (n_docs, n_terms) = tf_matrix.dim();
        let mut tfidf_matrix = tf_matrix.clone();

        // Compute document frequencies
        let mut df = vec![0.0; n_terms];
        for j in 0..n_terms {
            for i in 0..n_docs {
                if tf_matrix[[i, j]] > 0.0 {
                    df[j] += 1.0;
                }
            }
        }

        // Apply TF-IDF weighting
        for i in 0..n_docs {
            for j in 0..n_terms {
                if tf_matrix[[i, j]] > 0.0 && df[j] > 0.0 {
                    let tf = tf_matrix[[i, j]];
                    let idf = (n_docs as f64 / df[j]).ln();
                    tfidf_matrix[[i, j]] = tf * idf;
                }
            }
        }

        tfidf_matrix
    }

    /// Cluster documents represented as TF or TF-IDF matrix
    pub fn fit_predict(&self, tf_matrix: &Array2<f64>) -> Result<DocumentClusteringResult> {
        if tf_matrix.is_empty() {
            return Err(SklearsError::InvalidInput("Empty TF matrix".to_string()));
        }

        // Apply TF-IDF weighting if requested
        let feature_matrix = if self.config.use_tfidf {
            self.apply_tfidf(tf_matrix)
        } else {
            tf_matrix.clone()
        };

        // Filter features by document frequency if needed
        let filtered_matrix = self.filter_features(&feature_matrix)?;

        // Apply spherical K-means clustering
        let spherical_kmeans = SphericalKMeans::new(self.config.spherical_config.clone());
        // For clustering, provide dummy Y parameter
        let dummy_y = Array1::zeros(filtered_matrix.nrows());
        let fitted = spherical_kmeans.fit(&filtered_matrix, &dummy_y)?;

        Ok(DocumentClusteringResult {
            labels: fitted.labels,
            centroids: fitted.centroids,
            n_iterations: fitted.n_iterations,
            inertia: fitted.inertia,
            vocabulary: HashMap::new(), // Would be populated if vocabulary is provided
        })
    }

    /// Filter features based on document frequency thresholds
    fn filter_features(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_docs, n_terms) = matrix.dim();
        let mut keep_features = vec![true; n_terms];

        // Compute document frequencies
        for j in 0..n_terms {
            let mut df = 0;
            for i in 0..n_docs {
                if matrix[[i, j]] > 0.0 {
                    df += 1;
                }
            }

            let df_ratio = df as f64 / n_docs as f64;
            if df_ratio < self.config.min_df || df_ratio > self.config.max_df {
                keep_features[j] = false;
            }
        }

        // Count features to keep
        let n_keep = keep_features.iter().filter(|&&x| x).count();
        if n_keep == 0 {
            return Err(SklearsError::InvalidInput(
                "No features remain after filtering".to_string(),
            ));
        }

        // Create filtered matrix
        let final_n_features = if let Some(max_features) = self.config.max_features {
            n_keep.min(max_features)
        } else {
            n_keep
        };

        let mut filtered = Array2::<f64>::zeros((n_docs, final_n_features));
        let mut new_j = 0;

        for j in 0..n_terms {
            if keep_features[j] && new_j < final_n_features {
                for i in 0..n_docs {
                    filtered[[i, new_j]] = matrix[[i, j]];
                }
                new_j += 1;
            }
        }

        Ok(filtered)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    #[allow(non_snake_case)]
    fn test_spherical_kmeans_basic() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 0.0, // First cluster
                0.9, 0.1, 0.8, 0.2, 0.0, 1.0, // Second cluster
                0.1, 0.9, 0.2, 0.8,
            ],
        )
        .unwrap();

        let config = SphericalKMeansConfig {
            n_clusters: 2,
            max_iter: 100,
            tolerance: 1e-4,
            init: SphericalInit::RandomSamples,
            random_seed: Some(42),
            normalize_input: true,
        };

        let clusterer = SphericalKMeans::new(config);
        let dummy_y = Array1::<f64>::zeros(X.nrows());
        let fitted = clusterer.fit(&X, &dummy_y).unwrap();

        assert_eq!(fitted.labels.len(), 6);
        assert!(fitted.n_iterations <= 100);
        assert!(fitted.inertia >= 0.0);

        // Check that centroids are normalized
        for k in 0..2 {
            let centroid = fitted.centroids.row(k);
            let norm = centroid.mapv(|x| x * x).sum().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_spherical_kmeans_predict() {
        let X_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]).unwrap();

        let config = SphericalKMeansConfig {
            n_clusters: 2,
            random_seed: Some(42),
            ..Default::default()
        };

        let clusterer = SphericalKMeans::new(config);
        let dummy_y = Array1::<f64>::zeros(X_train.nrows());
        let fitted = clusterer.fit(&X_train, &dummy_y).unwrap();

        let X_test = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, -0.1, 0.9]).unwrap();

        let predictions = fitted.predict(&X_test).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_document_clustering_basic() {
        // Simple TF matrix: 4 documents, 3 terms
        let tf_matrix = Array2::from_shape_vec(
            (4, 3),
            vec![
                2.0, 1.0, 0.0, // Doc 1: focused on terms 0,1
                3.0, 2.0, 0.0, // Doc 2: focused on terms 0,1
                0.0, 0.0, 2.0, // Doc 3: focused on term 2
                0.0, 1.0, 3.0, // Doc 4: focused on terms 1,2
            ],
        )
        .unwrap();

        let config = DocumentClusteringConfig {
            spherical_config: SphericalKMeansConfig {
                n_clusters: 2,
                random_seed: Some(42),
                ..Default::default()
            },
            min_df: 0.0,
            max_df: 1.0,
            max_features: None,
            use_tfidf: false,
        };

        let doc_clusterer = DocumentClustering::new(config);
        let result = doc_clusterer.fit_predict(&tf_matrix).unwrap();

        assert_eq!(result.labels.len(), 4);
        assert!(result.inertia >= 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_normalize_data() {
        let X = Array2::from_shape_vec(
            (2, 3),
            vec![
                3.0, 4.0, 0.0, // Should normalize to [0.6, 0.8, 0.0]
                0.0, 5.0, 12.0, // Should normalize to [0.0, 5/13, 12/13]
            ],
        )
        .unwrap();

        let config = SphericalKMeansConfig::default();
        let clusterer = SphericalKMeans::new(config);
        let normalized = clusterer.normalize_data(&X);

        // Check first row normalization
        let row1 = normalized.row(0);
        assert_abs_diff_eq!(row1[0], 0.6, epsilon = 1e-6);
        assert_abs_diff_eq!(row1[1], 0.8, epsilon = 1e-6);
        assert_abs_diff_eq!(row1[2], 0.0, epsilon = 1e-6);

        // Check second row normalization
        let row2 = normalized.row(1);
        let expected_norm = (5.0_f64.powi(2) + 12.0_f64.powi(2)).sqrt(); // 13.0
        assert_abs_diff_eq!(row2[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(row2[1], 5.0 / expected_norm, epsilon = 1e-6);
        assert_abs_diff_eq!(row2[2], 12.0 / expected_norm, epsilon = 1e-6);

        // Verify both rows are unit vectors
        for i in 0..2 {
            let norm = normalized.row(i).mapv(|x| x * x).sum().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
        }
    }
}

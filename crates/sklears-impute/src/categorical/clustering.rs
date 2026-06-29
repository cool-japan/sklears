//! Categorical Clustering Imputer
#![allow(non_snake_case)]
#![allow(dead_code)]
//!
//! Imputation based on clustering categorical data into homogeneous groups
//! and using cluster centroids or most frequent values within clusters.
//!
//! # Note
//!
//! Not implemented in v0.1.0. `fit` and `transform` return
//! `Err(SklearsError::NotImplemented)`. Planned for v0.2.0.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Categorical Clustering Imputer
///
/// Imputation based on clustering categorical data into homogeneous groups
/// and using cluster centroids or most frequent values within clusters.
///
/// # Parameters
///
/// * `n_clusters` - Number of clusters to form
/// * `distance_metric` - Distance metric for clustering
/// * `max_iter` - Maximum number of iterations for clustering
/// * `missing_values` - The placeholder for missing values
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_impute::CategoricalClusteringImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 2.0, 1.0], [2.0, f64::NAN, 3.0]];
///
/// let imputer = CategoricalClusteringImputer::new()
///     .n_clusters(2)
///     .max_iter(100);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CategoricalClusteringImputer<S = Untrained> {
    state: S,
    n_clusters: usize,
    distance_metric: String,
    max_iter: usize,
    missing_values: f64,
    random_state: Option<u64>,
}

/// Trained state for CategoricalClusteringImputer
#[derive(Debug, Clone)]
pub struct CategoricalClusteringImputerTrained {
    cluster_centers_: Array2<f64>,
    cluster_labels_: Array1<usize>,
    n_features_in_: usize,
}

impl CategoricalClusteringImputer<Untrained> {
    /// Create a new CategoricalClusteringImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_clusters: 3,
            distance_metric: "hamming".to_string(),
            max_iter: 100,
            missing_values: f64::NAN,
            random_state: None,
        }
    }

    /// Set the number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Set the distance metric
    pub fn distance_metric(mut self, distance_metric: String) -> Self {
        self.distance_metric = distance_metric;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for CategoricalClusteringImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CategoricalClusteringImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for CategoricalClusteringImputer<Untrained> {
    type Fitted = CategoricalClusteringImputer<CategoricalClusteringImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array must have at least one sample and one feature".to_string(),
            ));
        }

        // Step 1: collect indices of complete rows (no missing values)
        let complete_row_indices: Vec<usize> = (0..n_samples)
            .filter(|&i| (0..n_features).all(|j| !self.is_missing(X[[i, j]])))
            .collect();

        if complete_row_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No complete rows found: k-means clustering requires at least one row with no missing values".to_string(),
            ));
        }

        let n_complete = complete_row_indices.len();
        // Clamp n_clusters to available complete rows to avoid empty assignments
        let k = self.n_clusters.min(n_complete).max(1);

        // Step 2: initialize cluster centers by picking k evenly-spaced rows from complete data
        let mut centers = Array2::<f64>::zeros((k, n_features));
        for c in 0..k {
            // evenly spaced index into complete rows
            let row_idx = if k == 1 {
                0
            } else {
                (c * (n_complete - 1)) / (k - 1)
            };
            let src_row = complete_row_indices[row_idx];
            for j in 0..n_features {
                centers[[c, j]] = X[[src_row, j]];
            }
        }

        // Step 3: Lloyd's k-means iterations on complete rows only
        let mut assignments = vec![0usize; n_complete];
        for _iter in 0..self.max_iter {
            // Assignment step: find nearest center for each complete row
            let mut changed = false;
            for (ci, &row_idx) in complete_row_indices.iter().enumerate() {
                let mut best_cluster = 0usize;
                let mut best_dist = f64::INFINITY;
                for c in 0..k {
                    let mut dist_sq = 0.0_f64;
                    for j in 0..n_features {
                        let diff = X[[row_idx, j]] - centers[[c, j]];
                        dist_sq += diff * diff;
                    }
                    if dist_sq < best_dist {
                        best_dist = dist_sq;
                        best_cluster = c;
                    }
                }
                if assignments[ci] != best_cluster {
                    assignments[ci] = best_cluster;
                    changed = true;
                }
            }

            // Update step: recompute centers as mean of assigned points
            let mut new_centers = Array2::<f64>::zeros((k, n_features));
            let mut counts = vec![0usize; k];
            for (ci, &row_idx) in complete_row_indices.iter().enumerate() {
                let c = assignments[ci];
                counts[c] += 1;
                for j in 0..n_features {
                    new_centers[[c, j]] += X[[row_idx, j]];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..n_features {
                        new_centers[[c, j]] /= counts[c] as f64;
                    }
                } else {
                    // empty cluster: keep old center
                    for j in 0..n_features {
                        new_centers[[c, j]] = centers[[c, j]];
                    }
                }
            }

            // Check convergence: max shift across all center coordinates
            let mut max_shift = 0.0_f64;
            for c in 0..k {
                for j in 0..n_features {
                    let shift = (new_centers[[c, j]] - centers[[c, j]]).abs();
                    if shift > max_shift {
                        max_shift = shift;
                    }
                }
            }
            centers = new_centers;

            if !changed || max_shift <= 1e-8 {
                break;
            }
        }

        // Build cluster_labels_ for all n_samples rows (assign each row to nearest center,
        // using only non-missing features for rows with missing values)
        let mut cluster_labels = vec![0usize; n_samples];
        for i in 0..n_samples {
            let mut best_cluster = 0usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..k {
                let mut dist_sq = 0.0_f64;
                let mut count = 0usize;
                for j in 0..n_features {
                    if !self.is_missing(X[[i, j]]) {
                        let diff = X[[i, j]] - centers[[c, j]];
                        dist_sq += diff * diff;
                        count += 1;
                    }
                }
                // normalize by count of non-missing dims to make distances comparable
                let normalized_dist = if count > 0 {
                    dist_sq / count as f64
                } else {
                    0.0
                };
                if normalized_dist < best_dist {
                    best_dist = normalized_dist;
                    best_cluster = c;
                }
            }
            cluster_labels[i] = best_cluster;
        }

        Ok(CategoricalClusteringImputer {
            state: CategoricalClusteringImputerTrained {
                cluster_centers_: centers,
                cluster_labels_: Array1::from(cluster_labels),
                n_features_in_: n_features,
            },
            n_clusters: k,
            distance_metric: self.distance_metric,
            max_iter: self.max_iter,
            missing_values: self.missing_values,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for CategoricalClusteringImputer<CategoricalClusteringImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let k = self.state.cluster_centers_.nrows();

        // Build output as owned copy, then fill in missing values
        let mut X_out = X.mapv(|x| x);

        for i in 0..n_samples {
            // Check whether this row has any missing values
            let has_missing = (0..n_features).any(|j| self.is_missing(X_out[[i, j]]));
            if !has_missing {
                continue;
            }

            // Find nearest cluster center using only non-missing features,
            // normalized by the count of non-missing dimensions
            let mut best_cluster = 0usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..k {
                let mut dist_sq = 0.0_f64;
                let mut count = 0usize;
                for j in 0..n_features {
                    if !self.is_missing(X_out[[i, j]]) {
                        let diff = X_out[[i, j]] - self.state.cluster_centers_[[c, j]];
                        dist_sq += diff * diff;
                        count += 1;
                    }
                }
                let normalized_dist = if count > 0 {
                    dist_sq / count as f64
                } else {
                    // all features missing: treat as distance 0 (all clusters equally close)
                    0.0
                };
                if normalized_dist < best_dist {
                    best_dist = normalized_dist;
                    best_cluster = c;
                }
            }

            // Fill each missing feature with the nearest cluster center's value
            for j in 0..n_features {
                if self.is_missing(X_out[[i, j]]) {
                    X_out[[i, j]] = self.state.cluster_centers_[[best_cluster, j]];
                }
            }
        }

        Ok(X_out)
    }
}

impl CategoricalClusteringImputer<CategoricalClusteringImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

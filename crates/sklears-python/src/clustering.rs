//! Python bindings for clustering algorithms
//!
//! This module provides Python bindings for sklears clustering algorithms,
//! offering scikit-learn compatible interfaces with performance improvements.

use crate::linear::common::{core_array2_to_py, pyarray_to_core_array2, PyValueError};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_clustering::kmeans::KMeansFitted;
use sklears_clustering::{KMeans, KMeansConfig, KMeansInit, DBSCAN};
use sklears_core::traits::{Fit, Predict, Trained};

/// K-Means clustering.
///
/// Partitions data into `n_clusters` clusters by iteratively assigning
/// points to the nearest centroid and recomputing centroids, using
/// K-means++ initialization for the starting centroids.
#[pyclass(name = "KMeans")]
pub struct PyKMeans {
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
    fitted: Option<KMeansFitted>,
}

impl PyKMeans {
    fn config(&self) -> KMeansConfig {
        KMeansConfig {
            n_clusters: self.n_clusters,
            init: KMeansInit::KMeansPlusPlus,
            max_iter: self.max_iter,
            tolerance: self.tol,
            random_seed: self.random_state,
        }
    }

    /// Core fit logic operating on plain `ndarray` types (no PyO3
    /// dependency), so it is directly unit-testable without a live Python
    /// interpreter. This crate builds with pyo3's `extension-module`
    /// feature (required so the compiled `cdylib` can be imported from
    /// Python), which means `Python::with_gil` cannot be used from a
    /// standalone `cargo test` binary -- so all `#[cfg(test)]` coverage in
    /// this file goes through `*_core` helpers like this one instead of
    /// exercising the `#[pymethods]` directly.
    fn fit_core(&mut self, x: &Array2<f64>) -> PyResult<()> {
        let dummy_y = Array1::<f64>::zeros(x.nrows());
        let fitted = KMeans::new(self.config())
            .fit(x, &dummy_y)
            .map_err(|e| PyValueError::new_err(format!("Failed to fit KMeans: {e}")))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    /// Core predict logic; see `fit_core` for why this is split out.
    fn predict_core(&self, x: &Array2<f64>) -> PyResult<Vec<i32>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;
        fitted
            .predict(x)
            .map_err(|e| PyValueError::new_err(format!("Prediction failed: {e}")))
    }
}

#[pymethods]
impl PyKMeans {
    #[new]
    #[pyo3(signature = (n_clusters=8, max_iter=300, tol=1e-4, random_state=None))]
    fn new(n_clusters: usize, max_iter: usize, tol: f64, random_state: Option<u64>) -> Self {
        Self {
            n_clusters,
            max_iter,
            tol,
            random_state,
            fitted: None,
        }
    }

    /// Compute K-means clustering from training data.
    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_arr = pyarray_to_core_array2(x)?;
        self.fit_core(&x_arr)
    }

    /// Predict the closest cluster each sample in `x` belongs to.
    fn predict(&self, py: Python<'_>, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        let x_arr = pyarray_to_core_array2(x)?;
        let labels = self.predict_core(&x_arr)?;
        Ok(PyArray1::from_vec(py, labels).unbind())
    }

    /// Fit the model to `x`, then return the cluster label assigned to
    /// each training sample.
    fn fit_predict(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray1<i32>>> {
        let x_arr = pyarray_to_core_array2(x)?;
        self.fit_core(&x_arr)?;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;
        Ok(PyArray1::from_vec(py, fitted.labels.clone()).unbind())
    }

    /// Cluster labels for the training data (available after `fit`).
    #[getter]
    fn labels_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<i32>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;
        Ok(PyArray1::from_vec(py, fitted.labels.clone()).unbind())
    }

    /// Coordinates of cluster centers (available after `fit`).
    #[getter]
    fn cluster_centers_(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;
        core_array2_to_py(py, &fitted.centroids)
    }

    /// Sum of squared distances of samples to their closest cluster center
    /// (available after `fit`).
    #[getter]
    fn inertia_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;
        Ok(fitted.inertia)
    }

    /// Number of iterations run before convergence (available after
    /// `fit`).
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;
        Ok(fitted.n_iterations)
    }

    fn __repr__(&self) -> String {
        format!(
            "KMeans(n_clusters={}, fitted={})",
            self.n_clusters,
            self.fitted.is_some()
        )
    }
}

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
///
/// Like scikit-learn's implementation, DBSCAN is transductive: it has no
/// `predict()` for unseen data. Use `fit_predict()` to cluster the data it
/// is fit on, and the `labels_` attribute to retrieve the result again
/// afterwards (`-1` marks noise points).
#[pyclass(name = "DBSCAN")]
pub struct PyDBSCAN {
    eps: f64,
    min_samples: usize,
    fitted: Option<DBSCAN<Trained>>,
}

impl PyDBSCAN {
    /// Core fit_predict logic; see `PyKMeans::fit_core` for why this is
    /// split out from the `#[pymethods]` wrapper.
    fn fit_predict_core(&mut self, x: &Array2<f64>) -> PyResult<Vec<i32>> {
        let model = DBSCAN::new().eps(self.eps).min_samples(self.min_samples);
        let fitted = model
            .fit(x, &())
            .map_err(|e| PyValueError::new_err(format!("Failed to fit DBSCAN: {e}")))?;
        let labels = fitted.labels().to_vec();
        self.fitted = Some(fitted);
        Ok(labels)
    }

    /// Core labels_ logic; see `PyKMeans::fit_core` for why this is split
    /// out from the `#[pymethods]` wrapper.
    fn labels_core(&self) -> PyResult<Vec<i32>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit_predict() first."))?;
        Ok(fitted.labels().to_vec())
    }
}

#[pymethods]
impl PyDBSCAN {
    #[new]
    #[pyo3(signature = (eps=0.5, min_samples=5))]
    fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            fitted: None,
        }
    }

    /// Fit DBSCAN to `x` and return the cluster label assigned to each
    /// sample (`-1` for noise points).
    fn fit_predict(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray1<i32>>> {
        let x_arr = pyarray_to_core_array2(x)?;
        let labels = self.fit_predict_core(&x_arr)?;
        Ok(PyArray1::from_vec(py, labels).unbind())
    }

    /// Cluster labels for the training data (available after
    /// `fit_predict`).
    #[getter]
    fn labels_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<i32>>> {
        let labels = self.labels_core()?;
        Ok(PyArray1::from_vec(py, labels).unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "DBSCAN(eps={}, min_samples={}, fitted={})",
            self.eps,
            self.min_samples,
            self.fitted.is_some()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated blobs: 3 points near (0, 0) and 3 points near
    /// (10, 10). Any working clustering algorithm run with 2 clusters
    /// should separate these into 2 groups.
    fn two_blob_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, -0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.2,
            ],
        )
        .expect("shape matches data length")
    }

    #[test]
    fn kmeans_fit_predict_round_trip_finds_two_clusters() {
        let mut model = PyKMeans::new(2, 300, 1e-4, Some(42));
        let x = two_blob_data();

        model
            .fit_core(&x)
            .expect("fit should succeed on well-separated data");
        let labels = model
            .predict_core(&x)
            .expect("predict should succeed after fit");

        assert_eq!(labels.len(), 6);

        let distinct: std::collections::HashSet<i32> = labels.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            2,
            "expected exactly 2 distinct cluster labels, got {distinct:?}"
        );

        // The first 3 points must share one label and the last 3 must
        // share a different label -- the old stub (always predicting [0])
        // would fail this.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn kmeans_fitted_attributes_work_post_fit() {
        let mut model = PyKMeans::new(2, 300, 1e-4, Some(42));
        let x = two_blob_data();
        model.fit_core(&x).expect("fit should succeed");

        let fitted = model.fitted.as_ref().expect("model was just fitted");
        assert_eq!(fitted.labels.len(), 6);
        assert_eq!(fitted.centroids.nrows(), 2);
        assert_eq!(fitted.centroids.ncols(), 2);
        assert!(fitted.inertia >= 0.0);
        assert!(fitted.n_iterations >= 1);
    }

    #[test]
    fn kmeans_predict_before_fit_errors_instead_of_panicking() {
        let model = PyKMeans::new(2, 300, 1e-4, Some(42));
        let x = two_blob_data();
        assert!(model.predict_core(&x).is_err());
    }

    #[test]
    fn dbscan_fit_predict_finds_more_than_one_group() {
        let mut model = PyDBSCAN::new(1.0, 2);
        let x = two_blob_data();

        let labels = model
            .fit_predict_core(&x)
            .expect("fit_predict should succeed on well-separated data");

        assert_eq!(labels.len(), 6);
        assert!(
            !labels.iter().all(|&l| l == 0),
            "labels must not all be 0 (this is what the old stub always returned): {labels:?}"
        );

        let distinct: std::collections::HashSet<i32> = labels.iter().copied().collect();
        assert!(
            distinct.len() >= 2,
            "expected at least 2 distinct labels/groups, got {distinct:?}"
        );
    }

    #[test]
    fn dbscan_labels_getter_errors_before_fit_and_works_after() {
        let mut model = PyDBSCAN::new(1.0, 2);
        assert!(
            model.labels_core().is_err(),
            "labels_ must error before fit_predict, not panic"
        );

        let x = two_blob_data();
        model
            .fit_predict_core(&x)
            .expect("fit_predict should succeed");

        let labels = model.labels_core().expect("labels_ should work after fit");
        assert_eq!(labels.len(), 6);
    }
}

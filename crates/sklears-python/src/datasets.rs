//! Python bindings for datasets and data generation
//!
//! This module provides PyO3-based Python bindings for sklears dataset loading
//! and synthetic data generation functions.

use numpy::IntoPyArray;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use sklears_datasets::{
    make_blobs, make_circles, make_classification, make_moons, make_regression,
};

/// Generate isotropic Gaussian blobs for clustering
#[pyfunction]
#[pyo3(signature = (
    n_samples=100,
    n_features=2,
    centers=None,
    cluster_std=1.0,
    _center_box=None,
    _shuffle=true,
    random_state=None,
    _return_centers=false
))]
fn make_blobs_py<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_features: usize,
    centers: Option<usize>,
    cluster_std: f64,
    _center_box: Option<(f64, f64)>,
    _shuffle: bool,
    random_state: Option<u64>,
    _return_centers: bool,
) -> PyResult<Py<PyAny>> {
    let n_centers = centers.unwrap_or(3);

    let (data, labels) = make_blobs(n_samples, n_features, n_centers, cluster_std, random_state)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate blobs: {}", e)))?;

    let data_py = data.into_pyarray(py).into_any();
    let labels_py = labels.into_pyarray(py).into_any();

    Ok(PyTuple::new(py, &[data_py, labels_py])?.into_any().unbind())
}

/// Generate a random classification problem
#[pyfunction]
#[pyo3(signature = (
    n_samples=100,
    n_features=20,
    n_informative=None,
    n_redundant=None,
    _n_repeated=0,
    n_classes=2,
    _n_clusters_per_class=2,
    _weights=None,
    _flip_y=0.01,
    _class_sep=1.0,
    _hypercube=true,
    _shift=0.0,
    _scale=1.0,
    _shuffle=true,
    random_state=None
))]
fn make_classification_py<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_features: usize,
    n_informative: Option<usize>,
    n_redundant: Option<usize>,
    _n_repeated: usize,
    n_classes: usize,
    _n_clusters_per_class: usize,
    _weights: Option<Vec<f64>>,
    _flip_y: f64,
    _class_sep: f64,
    _hypercube: bool,
    _shift: f64,
    _scale: f64,
    _shuffle: bool,
    random_state: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let informative = n_informative.unwrap_or(n_features);
    let redundant = n_redundant.unwrap_or(0);

    let (data, labels) = make_classification(
        n_samples,
        n_features,
        informative,
        redundant,
        n_classes,
        random_state,
    )
    .map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to generate classification dataset: {}", e))
    })?;

    let data_py = data.into_pyarray(py).into_any();
    let labels_py = labels.into_pyarray(py).into_any();

    Ok(PyTuple::new(py, &[data_py, labels_py])?.into_any().unbind())
}

/// Generate a random regression problem
#[pyfunction]
#[pyo3(signature = (
    n_samples=100,
    n_features=100,
    n_informative=10,
    _n_targets=1,
    _bias=0.0,
    _effective_rank=None,
    _tail_strength=0.5,
    noise=0.0,
    _shuffle=true,
    _coef=false,
    random_state=None
))]
fn make_regression_py<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    _n_targets: usize,
    _bias: f64,
    _effective_rank: Option<usize>,
    _tail_strength: f64,
    noise: f64,
    _shuffle: bool,
    _coef: bool,
    random_state: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let (data, target) = make_regression(n_samples, n_features, n_informative, noise, random_state)
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to generate regression dataset: {}", e))
        })?;

    let data_py = data.into_pyarray(py).into_any();
    let target_py = target.into_pyarray(py).into_any();

    Ok(PyTuple::new(py, &[data_py, target_py])?.into_any().unbind())
}

/// Generate 2d classification dataset with two moon shapes
#[pyfunction]
#[pyo3(signature = (n_samples=100, _shuffle=true, noise=None, random_state=None))]
fn make_moons_py<'py>(
    py: Python<'py>,
    n_samples: usize,
    _shuffle: bool,
    noise: Option<f64>,
    random_state: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let (data, labels) = make_moons(n_samples, noise, random_state)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate moons dataset: {}", e)))?;

    let data_py = data.into_pyarray(py).into_any();
    let labels_py = labels.into_pyarray(py).into_any();

    Ok(PyTuple::new(py, &[data_py, labels_py])?.into_any().unbind())
}

/// Generate 2d classification dataset with two circles
#[pyfunction]
#[pyo3(signature = (n_samples=100, _shuffle=true, noise=None, random_state=None, factor=0.8))]
fn make_circles_py<'py>(
    py: Python<'py>,
    n_samples: usize,
    _shuffle: bool,
    noise: Option<f64>,
    random_state: Option<u64>,
    factor: f64,
) -> PyResult<Py<PyAny>> {
    let (data, labels) = make_circles(n_samples, noise, factor, random_state).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to generate circles dataset: {}", e))
    })?;

    let data_py = data.into_pyarray(py).into_any();
    let labels_py = labels.into_pyarray(py).into_any();

    Ok(PyTuple::new(py, &[data_py, labels_py])?.into_any().unbind())
}

/// Unified function to register all dataset functions
pub(crate) fn register_dataset_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(make_blobs_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_classification_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_regression_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_moons_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_circles_py, m)?)?;

    Ok(())
}

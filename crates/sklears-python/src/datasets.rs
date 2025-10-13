//! Python bindings for datasets and data generation
//!
//! This module provides PyO3-based Python bindings for sklears dataset loading
//! and synthetic data generation functions.

use crate::utils::{ndarray_to_numpy, numpy_to_ndarray2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use sklears_core::error::{Result as SklearsResult, SklearsError};
use sklears_datasets::{
    load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine, make_blobs, make_circles,
    make_classification, make_moons, make_regression, Dataset,
};

/// Load the Iris dataset
#[pyfunction]
#[pyo3(signature = (return_X_y=false, as_frame=false))]
fn load_iris_py(py: Python, return_X_y: bool, as_frame: bool) -> PyResult<PyObject> {
    let dataset = load_iris()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load iris dataset: {}", e)))?;

    if return_X_y {
        let data = dataset.data.into_pyarray(py);
        let target = dataset.target.into_pyarray(py);
        Ok((data, target).to_object(py))
    } else {
        let dict = PyDict::new(py);
        dict.set_item("data", dataset.data.into_pyarray(py))?;
        dict.set_item("target", dataset.target.into_pyarray(py))?;
        dict.set_item("feature_names", dataset.feature_names)?;
        dict.set_item("target_names", dataset.target_names)?;
        dict.set_item("DESCR", dataset.description)?;
        dict.set_item("filename", dataset.filename)?;
        Ok(dict.to_object(py))
    }
}

/// Load the breast cancer dataset
#[pyfunction]
#[pyo3(signature = (return_X_y=false, as_frame=false))]
fn load_breast_cancer_py(py: Python, return_X_y: bool, as_frame: bool) -> PyResult<PyObject> {
    let dataset = load_breast_cancer().map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to load breast cancer dataset: {}", e))
    })?;

    if return_X_y {
        let data = dataset.data.into_pyarray(py);
        let target = dataset.target.into_pyarray(py);
        Ok((data, target).to_object(py))
    } else {
        let dict = PyDict::new(py);
        dict.set_item("data", dataset.data.into_pyarray(py))?;
        dict.set_item("target", dataset.target.into_pyarray(py))?;
        dict.set_item("feature_names", dataset.feature_names)?;
        dict.set_item("target_names", dataset.target_names)?;
        dict.set_item("DESCR", dataset.description)?;
        dict.set_item("filename", dataset.filename)?;
        Ok(dict.to_object(py))
    }
}

/// Load the diabetes dataset
#[pyfunction]
#[pyo3(signature = (return_X_y=false, as_frame=false, scaled=true))]
fn load_diabetes_py(
    py: Python,
    return_X_y: bool,
    as_frame: bool,
    scaled: bool,
) -> PyResult<PyObject> {
    let dataset = load_diabetes()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load diabetes dataset: {}", e)))?;

    if return_X_y {
        let data = dataset.data.into_pyarray(py);
        let target = dataset.target.into_pyarray(py);
        Ok((data, target).to_object(py))
    } else {
        let dict = PyDict::new(py);
        dict.set_item("data", dataset.data.into_pyarray(py))?;
        dict.set_item("target", dataset.target.into_pyarray(py))?;
        dict.set_item("feature_names", dataset.feature_names)?;
        dict.set_item("DESCR", dataset.description)?;
        dict.set_item("filename", dataset.filename)?;
        Ok(dict.to_object(py))
    }
}

/// Load the wine dataset
#[pyfunction]
#[pyo3(signature = (return_X_y=false, as_frame=false))]
fn load_wine_py(py: Python, return_X_y: bool, as_frame: bool) -> PyResult<PyObject> {
    let dataset = load_wine()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load wine dataset: {}", e)))?;

    if return_X_y {
        let data = dataset.data.into_pyarray(py);
        let target = dataset.target.into_pyarray(py);
        Ok((data, target).to_object(py))
    } else {
        let dict = PyDict::new(py);
        dict.set_item("data", dataset.data.into_pyarray(py))?;
        dict.set_item("target", dataset.target.into_pyarray(py))?;
        dict.set_item("feature_names", dataset.feature_names)?;
        dict.set_item("target_names", dataset.target_names)?;
        dict.set_item("DESCR", dataset.description)?;
        dict.set_item("filename", dataset.filename)?;
        Ok(dict.to_object(py))
    }
}

/// Load the digits dataset
#[pyfunction]
#[pyo3(signature = (n_class=10, return_X_y=false, as_frame=false))]
fn load_digits_py(
    py: Python,
    n_class: usize,
    return_X_y: bool,
    as_frame: bool,
) -> PyResult<PyObject> {
    let dataset = load_digits()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load digits dataset: {}", e)))?;

    if return_X_y {
        let data = dataset.data.into_pyarray(py);
        let target = dataset.target.into_pyarray(py);
        Ok((data, target).to_object(py))
    } else {
        let dict = PyDict::new(py);
        dict.set_item("data", dataset.data.into_pyarray(py))?;
        dict.set_item("target", dataset.target.into_pyarray(py))?;
        dict.set_item("target_names", dataset.target_names)?;
        dict.set_item("images", dataset.data.into_pyarray(py))?; // For digits, data and images are the same
        dict.set_item("DESCR", dataset.description)?;
        Ok(dict.to_object(py))
    }
}

/// Generate isotropic Gaussian blobs for clustering
#[pyfunction]
#[pyo3(signature = (
    n_samples=100,
    n_features=2,
    centers=None,
    cluster_std=1.0,
    center_box=None,
    shuffle=true,
    random_state=None,
    return_centers=false
))]
fn make_blobs_py(
    py: Python,
    n_samples: usize,
    n_features: usize,
    centers: Option<usize>,
    cluster_std: f64,
    center_box: Option<(f64, f64)>,
    shuffle: bool,
    random_state: Option<u64>,
    return_centers: bool,
) -> PyResult<PyObject> {
    let n_centers = centers.unwrap_or(3);

    let (data, labels, centers) =
        make_blobs(n_samples, n_features, n_centers, cluster_std, random_state)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate blobs: {}", e)))?;

    let data_py = data.into_pyarray(py);
    let labels_py = labels.into_pyarray(py);

    if return_centers {
        let centers_py = centers.into_pyarray(py);
        Ok((data_py, labels_py, centers_py).to_object(py))
    } else {
        Ok((data_py, labels_py).to_object(py))
    }
}

/// Generate a random classification problem
#[pyfunction]
#[pyo3(signature = (
    n_samples=100,
    n_features=20,
    n_informative=None,
    n_redundant=None,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=true,
    shift=0.0,
    scale=1.0,
    shuffle=true,
    random_state=None
))]
fn make_classification_py(
    py: Python,
    n_samples: usize,
    n_features: usize,
    n_informative: Option<usize>,
    n_redundant: Option<usize>,
    n_repeated: usize,
    n_classes: usize,
    n_clusters_per_class: usize,
    weights: Option<Vec<f64>>,
    flip_y: f64,
    class_sep: f64,
    hypercube: bool,
    shift: f64,
    scale: f64,
    shuffle: bool,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
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

    let data_py = data.into_pyarray(py);
    let labels_py = labels.into_pyarray(py);

    Ok((data_py, labels_py).to_object(py))
}

/// Generate a random regression problem
#[pyfunction]
#[pyo3(signature = (
    n_samples=100,
    n_features=100,
    n_informative=10,
    n_targets=1,
    bias=0.0,
    effective_rank=None,
    tail_strength=0.5,
    noise=0.0,
    shuffle=true,
    coef=false,
    random_state=None
))]
fn make_regression_py(
    py: Python,
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    n_targets: usize,
    bias: f64,
    effective_rank: Option<usize>,
    tail_strength: f64,
    noise: f64,
    shuffle: bool,
    coef: bool,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let (data, target, coefficients) =
        make_regression(n_samples, n_features, n_informative, noise, random_state).map_err(
            |e| PyRuntimeError::new_err(format!("Failed to generate regression dataset: {}", e)),
        )?;

    let data_py = data.into_pyarray(py);
    let target_py = target.into_pyarray(py);

    if coef {
        let coef_py = coefficients.into_pyarray(py);
        Ok((data_py, target_py, coef_py).to_object(py))
    } else {
        Ok((data_py, target_py).to_object(py))
    }
}

/// Generate 2d classification dataset with two moon shapes
#[pyfunction]
#[pyo3(signature = (n_samples=100, shuffle=true, noise=None, random_state=None))]
fn make_moons_py(
    py: Python,
    n_samples: usize,
    shuffle: bool,
    noise: Option<f64>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let noise_val = noise.unwrap_or(0.0);

    let (data, labels) = make_moons(n_samples, noise_val, random_state)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate moons dataset: {}", e)))?;

    let data_py = data.into_pyarray(py);
    let labels_py = labels.into_pyarray(py);

    Ok((data_py, labels_py).to_object(py))
}

/// Generate 2d classification dataset with two circles
#[pyfunction]
#[pyo3(signature = (n_samples=100, shuffle=true, noise=None, random_state=None, factor=0.8))]
fn make_circles_py(
    py: Python,
    n_samples: usize,
    shuffle: bool,
    noise: Option<f64>,
    random_state: Option<u64>,
    factor: f64,
) -> PyResult<PyObject> {
    let noise_val = noise.unwrap_or(0.0);

    let (data, labels) = make_circles(n_samples, noise_val, factor, random_state).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to generate circles dataset: {}", e))
    })?;

    let data_py = data.into_pyarray(py);
    let labels_py = labels.into_pyarray(py);

    Ok((data_py, labels_py).to_object(py))
}

/// Unified function to register all dataset functions
pub(crate) fn register_dataset_functions(py: Python, m: &PyModule) -> PyResult<()> {
    // Dataset loaders
    m.add_function(wrap_pyfunction!(load_iris_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_breast_cancer_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_diabetes_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_wine_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_digits_py, m)?)?;

    // Data generators
    m.add_function(wrap_pyfunction!(make_blobs_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_classification_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_regression_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_moons_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_circles_py, m)?)?;

    Ok(())
}

//! Python bindings for utility functions
//!
//! This module provides Python bindings for sklears utilities,
//! including version information and build details.

// Use SciRS2-Core for array operations instead of direct ndarray
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::Bound;
use scirs2_autograd::ndarray::{Array1, Array2};
// Use SciRS2-Core for random number generation instead of direct rand
use scirs2_core::random::{thread_rng, Rng};
use std::collections::HashMap;

use crate::linear::{
    core_array1_to_py, core_array2_to_py, pyarray_to_core_array1, pyarray_to_core_array2,
};

/// Get the version of sklears
#[pyfunction]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get build information about sklears
#[pyfunction]
pub fn get_build_info() -> HashMap<String, String> {
    let mut info = HashMap::new();

    info.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
    info.insert("authors".to_string(), env!("CARGO_PKG_AUTHORS").to_string());
    info.insert(
        "description".to_string(),
        env!("CARGO_PKG_DESCRIPTION").to_string(),
    );
    info.insert(
        "homepage".to_string(),
        env!("CARGO_PKG_HOMEPAGE").to_string(),
    );
    info.insert(
        "repository".to_string(),
        env!("CARGO_PKG_REPOSITORY").to_string(),
    );
    info.insert("license".to_string(), env!("CARGO_PKG_LICENSE").to_string());
    info.insert(
        "rust_version".to_string(),
        env!("CARGO_PKG_RUST_VERSION").to_string(),
    );

    // Build-time information
    info.insert(
        "target_triple".to_string(),
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
    );
    info.insert(
        "build_profile".to_string(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .to_string(),
    );

    // Feature information
    let features = [
        #[cfg(feature = "pandas-integration")]
        "pandas-integration",
    ];

    info.insert("features".to_string(), features.join(", "));

    // Dependency versions
    info.insert("scirs2_core_version".to_string(), "workspace".to_string());
    info.insert("pyo3_version".to_string(), "0.26".to_string());
    info.insert("numpy_version".to_string(), "0.26".to_string());

    info
}

/// Check if specific features are enabled
#[pyfunction]
pub fn has_feature(feature_name: &str) -> bool {
    match feature_name {
        "pandas-integration" => {
            #[cfg(feature = "pandas-integration")]
            return true;
            #[cfg(not(feature = "pandas-integration"))]
            return false;
        }
        _ => false,
    }
}

/// Get hardware acceleration capabilities
#[pyfunction]
pub fn get_hardware_info() -> HashMap<String, bool> {
    let mut info = HashMap::new();

    // CPU features
    #[cfg(target_arch = "x86_64")]
    {
        info.insert("x86_64".to_string(), true);
        info.insert("avx2".to_string(), is_x86_feature_detected!("avx2"));
        info.insert("fma".to_string(), is_x86_feature_detected!("fma"));
        info.insert("sse4_1".to_string(), is_x86_feature_detected!("sse4.1"));
        info.insert("sse4_2".to_string(), is_x86_feature_detected!("sse4.2"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        info.insert("aarch64".to_string(), true);
        info.insert("neon".to_string(), cfg!(target_feature = "neon"));
    }

    // Other architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        info.insert("simd_support".to_string(), false);
    }

    // GPU support (placeholder - would need actual detection)
    info.insert("cuda_available".to_string(), false);
    info.insert("opencl_available".to_string(), false);

    // Thread support
    info.insert("parallel_support".to_string(), true);
    info.insert("num_cpus".to_string(), num_cpus::get() > 1);

    info
}

/// Get memory usage information
#[pyfunction]
pub fn get_memory_info() -> HashMap<String, u64> {
    let mut info = HashMap::new();

    // Get number of CPUs
    info.insert("num_cpus".to_string(), num_cpus::get() as u64);

    // Physical memory would require additional dependencies
    // This is a placeholder
    info.insert("available_memory_mb".to_string(), 0);
    info.insert("used_memory_mb".to_string(), 0);

    info
}

/// Set global configuration options
#[pyfunction]
pub fn set_config(option: &str, _value: &str) -> PyResult<()> {
    match option {
        "n_jobs" => {
            // Set global parallelism configuration
            // This would require implementing global state management
            Ok(())
        }
        "assume_finite" => {
            // Set validation configuration
            Ok(())
        }
        "working_memory" => {
            // Set memory limit for operations
            Ok(())
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown configuration option: {}",
            option
        ))),
    }
}

/// Get current configuration
#[pyfunction]
pub fn get_config() -> HashMap<String, String> {
    let mut config = HashMap::new();

    // Default configuration values
    config.insert("n_jobs".to_string(), "1".to_string());
    config.insert("assume_finite".to_string(), "false".to_string());
    config.insert("working_memory".to_string(), "1024".to_string());
    config.insert("print_changed_only".to_string(), "true".to_string());
    config.insert("display".to_string(), "text".to_string());

    config
}

/// Print system information
#[pyfunction]
pub fn show_versions() -> String {
    let mut output = String::new();

    output.push_str("sklears information:\n");
    output.push_str("=====================\n");

    let build_info = get_build_info();
    for (key, value) in &build_info {
        output.push_str(&format!("{}: {}\n", key, value));
    }

    output.push_str("\nHardware information:\n");
    output.push_str("====================\n");

    let hardware_info = get_hardware_info();
    for (key, value) in &hardware_info {
        output.push_str(&format!("{}: {}\n", key, value));
    }

    output.push_str("\nMemory information:\n");
    output.push_str("==================\n");

    let memory_info = get_memory_info();
    for (key, value) in &memory_info {
        output.push_str(&format!("{}: {}\n", key, value));
    }

    output
}

/// Performance testing utility
#[pyfunction]
pub fn benchmark_basic_operations() -> HashMap<String, f64> {
    use std::time::Instant;

    let mut results = HashMap::new();
    let mut rng = thread_rng();

    // Matrix multiplication benchmark
    let start = Instant::now();
    let a = Array2::from_shape_fn((100, 100), |_| rng.gen::<f64>());
    let b = Array2::from_shape_fn((100, 100), |_| rng.gen::<f64>());
    let _c = a.dot(&b);
    let matrix_mul_time = start.elapsed().as_nanos() as f64 / 1_000_000.0; // Convert to milliseconds
    results.insert(
        "matrix_multiplication_100x100_ms".to_string(),
        matrix_mul_time,
    );

    // Vector operations benchmark
    let start = Instant::now();
    let v1 = Array1::from_shape_fn(10000, |_| rng.gen::<f64>());
    let v2 = Array1::from_shape_fn(10000, |_| rng.gen::<f64>());
    let _dot_product = v1.dot(&v2);
    let vector_ops_time = start.elapsed().as_nanos() as f64 / 1_000_000.0;
    results.insert("vector_dot_product_10k_ms".to_string(), vector_ops_time);

    // Memory allocation benchmark
    let start = Instant::now();
    let _large_array = Array2::<f64>::zeros((1000, 1000));
    let allocation_time = start.elapsed().as_nanos() as f64 / 1_000_000.0;
    results.insert(
        "memory_allocation_1M_elements_ms".to_string(),
        allocation_time,
    );

    results
}

/// Convert NumPy array to ndarray Array2`<f64>`
pub fn numpy_to_ndarray2(py_array: &PyArray2<f64>) -> PyResult<Array2<f64>> {
    Python::with_gil(|py| {
        let ptr = py_array as *const PyArray2<f64> as *mut ffi::PyObject;
        let bound_any = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, ptr) };
        let bound_array = bound_any.downcast::<PyArray2<f64>>()?;
        let readonly = bound_array.try_readonly().map_err(|err| {
            PyValueError::new_err(format!(
                "Failed to borrow NumPy array as read-only view: {err}"
            ))
        })?;
        pyarray_to_core_array2(readonly)
    })
}

/// Convert NumPy array to ndarray Array1`<f64>`
pub fn numpy_to_ndarray1(py_array: &PyArray1<f64>) -> PyResult<Array1<f64>> {
    Python::with_gil(|py| {
        let ptr = py_array as *const PyArray1<f64> as *mut ffi::PyObject;
        let bound_any = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, ptr) };
        let bound_array = bound_any.downcast::<PyArray1<f64>>()?;
        let readonly = bound_array.try_readonly().map_err(|err| {
            PyValueError::new_err(format!(
                "Failed to borrow NumPy array as read-only view: {err}"
            ))
        })?;
        pyarray_to_core_array1(readonly)
    })
}

/// Convert ndarray Array2`<f64>` to NumPy array
pub fn ndarray_to_numpy<'py>(py: Python<'py>, array: Array2<f64>) -> Py<PyArray2<f64>> {
    core_array2_to_py(py, &array).expect("Failed to convert ndarray to NumPy array")
}

/// Convert ndarray Array1`<f64>` to NumPy array
pub fn ndarray1_to_numpy<'py>(py: Python<'py>, array: Array1<f64>) -> Py<PyArray1<f64>> {
    core_array1_to_py(py, &array)
}

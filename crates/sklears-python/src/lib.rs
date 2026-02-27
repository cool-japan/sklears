#![allow(missing_docs)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
//! Python bindings for the sklears machine learning library
//!
//! This crate provides PyO3-based Python bindings for sklears, enabling
//! seamless integration with the Python ecosystem while maintaining
//! Rust's performance advantages.
//!
//! # Features
//!
//! - Drop-in replacement for scikit-learn's most common algorithms
//! - Pure Rust implementation with ongoing performance optimization
//! - Full NumPy array compatibility
//! - Comprehensive error handling with Python exceptions
//! - Memory-safe operations with automatic reference counting
//!
//! # Example
//!
//! ```python
//! import sklears_python as skl
//! import numpy as np
//!
//! # Create sample data
//! X = np.random.randn(100, 4)
//! y = np.random.randn(100)
//!
//! # Train a linear regression model
//! model = skl.LinearRegression()
//! model.fit(X, y)
//! predictions = model.predict(X)
//! ```

#[allow(unused_imports)]
use pyo3::prelude::*;

// Import modules
mod clustering;
mod datasets;
mod ensemble;
mod linear;
// mod metrics; // TODO: Needs refactoring to use sklears-metrics directly
mod model_selection;
mod naive_bayes;
mod neural_network;
// mod preprocessing; // Temporarily disabled to test ensemble
mod tree;
mod utils;

// Re-export main classes
pub use clustering::*;
pub use ensemble::*;
pub use linear::*;
// pub use metrics::*; // TODO: Needs refactoring
pub use model_selection::*;
pub use naive_bayes::*;
pub use neural_network::*;
// pub use preprocessing::*; // Temporarily disabled to test ensemble
pub use tree::*;
pub use utils::*;

/// Python module for sklears machine learning library
#[pymodule]
fn _sklears(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Set module metadata
    m.add("__version__", "0.1.0-rc.1")?;
    m.add(
        "__doc__",
        "High-performance machine learning library with scikit-learn compatibility",
    )?;

    // Linear models
    m.add_class::<linear::PyLinearRegression>()?;
    m.add_class::<linear::PyRidge>()?;
    m.add_class::<linear::PyLasso>()?;
    m.add_class::<linear::PyElasticNet>()?;
    m.add_class::<linear::PyBayesianRidge>()?;
    m.add_class::<linear::PyARDRegression>()?;
    m.add_class::<linear::PyLogisticRegression>()?;

    // Ensemble methods
    m.add_class::<ensemble::PyGradientBoostingClassifier>()?;
    m.add_class::<ensemble::PyGradientBoostingRegressor>()?;
    m.add_class::<ensemble::PyAdaBoostClassifier>()?;
    m.add_class::<ensemble::PyVotingClassifier>()?;
    m.add_class::<ensemble::PyBaggingClassifier>()?;

    // Neural networks
    m.add_class::<neural_network::PyMLPClassifier>()?;
    m.add_class::<neural_network::PyMLPRegressor>()?;

    // Tree-based models - Temporarily disabled to test ensemble
    // m.add_class::<tree::PyDecisionTreeClassifier>()?;
    // m.add_class::<tree::PyDecisionTreeRegressor>()?;
    // m.add_class::<tree::PyRandomForestClassifier>()?;
    // m.add_class::<tree::PyRandomForestRegressor>()?;

    // Naive Bayes
    m.add_class::<naive_bayes::PyGaussianNB>()?;
    m.add_class::<naive_bayes::PyMultinomialNB>()?;
    m.add_class::<naive_bayes::PyBernoulliNB>()?;
    m.add_class::<naive_bayes::PyComplementNB>()?;

    // Clustering
    m.add_class::<clustering::PyKMeans>()?;
    m.add_class::<clustering::PyDBSCAN>()?;

    // Preprocessing - Temporarily disabled to test ensemble
    // m.add_class::<preprocessing::PyStandardScaler>()?;
    // m.add_class::<preprocessing::PyMinMaxScaler>()?;
    // m.add_class::<preprocessing::PyLabelEncoder>()?;

    // TODO: Re-enable metrics after refactoring to use sklears-metrics directly
    // Metrics - Regression
    // m.add_function(wrap_pyfunction!(metrics::mean_squared_error, m)?)?;
    // m.add_function(wrap_pyfunction!(metrics::mean_absolute_error, m)?)?;
    // m.add_function(wrap_pyfunction!(metrics::r2_score, m)?)?;
    // m.add_function(wrap_pyfunction!(metrics::mean_squared_log_error, m)?)?;
    // m.add_function(wrap_pyfunction!(metrics::median_absolute_error, m)?)?;

    // Metrics - Classification
    // m.add_function(wrap_pyfunction!(metrics::accuracy_score, m)?)?;
    // m.add_function(wrap_pyfunction!(metrics::precision_score, m)?)?;
    // m.add_function(wrap_pyfunction!(metrics::recall_score, m)?)?;
    // m.add_function(wrap_pyfunction!(metrics::f1_score, m)?)?;
    // m.add_function(wrap_pyfunction!(metrics::confusion_matrix, m)?)?;
    // m.add_function(wrap_pyfunction!(metrics::classification_report, m)?)?;

    // Model selection
    m.add_function(wrap_pyfunction!(model_selection::train_test_split, m)?)?;
    m.add_class::<model_selection::PyKFold>()?;

    // Dataset functions
    datasets::register_dataset_functions(m)?;

    // Utility functions
    m.add_function(wrap_pyfunction!(utils::get_version, m)?)?;
    m.add_function(wrap_pyfunction!(utils::get_build_info, m)?)?;

    Ok(())
}

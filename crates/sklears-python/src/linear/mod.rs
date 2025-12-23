//! Python bindings for linear models
//!
//! This module provides Python bindings for linear models,
//! offering scikit-learn compatible interfaces with significant
//! performance improvements. Each model type is implemented in
//! its own submodule for better code organization.

// Common functionality shared across linear models
mod common;
pub use common::*;

// Individual model implementations
mod ard_regression;
mod bayesian_ridge;
mod elastic_net;
mod lasso;
mod linear_regression;
mod logistic_regression;
mod ridge;

// Re-export working model classes for PyO3
pub use ard_regression::PyARDRegression;
pub use bayesian_ridge::PyBayesianRidge;
pub use elastic_net::PyElasticNet;
pub use lasso::PyLasso;
pub use linear_regression::PyLinearRegression;
pub use logistic_regression::PyLogisticRegression;
pub use ridge::PyRidge;

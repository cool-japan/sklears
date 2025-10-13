//! Missing value imputation strategies
//!
//! This module provides comprehensive missing value imputation functionality
//! through a modular architecture supporting multiple imputation paradigms.
//!
//! ## Architecture
//!
//! The imputation system is organized into focused modules:
//! - **Simple Imputer**: Basic strategies (mean, median, mode, forward/backward fill)
//! - **KNN Imputer**: K-Nearest Neighbors based imputation
//! - **EM Imputer**: Expectation Maximization imputation
//! - **Iterative Imputer**: MICE (Multiple Imputation by Chained Equations)
//! - **Regression Imputers**: Linear and logistic regression based methods
//! - **Multiple Imputer**: Multiple imputation framework
//! - **Traits**: Common interfaces and abstractions
//! - **Validation**: Input validation and error handling
//! - **Utils**: Shared utilities and helper functions
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use sklears_impute::SimpleImputer;
//! use sklears_core::traits::{Transform, Fit};
//! use scirs2_core::ndarray::array;
//!
//! // Basic mean imputation
//! let X = array![[1.0, 2.0], [f64::NAN, 3.0], [7.0, 6.0]];
//! let imputer = SimpleImputer::new().strategy("mean".to_string());
//! let fitted = imputer.fit(&X.view(), &()).unwrap();
//! let X_imputed = fitted.transform(&X.view()).unwrap();
//!
//! // Advanced MICE imputation
//! let mice = IterativeImputer::new().max_iter(10).random_state(42);
//! let fitted_mice = mice.fit(&X.view(), &()).unwrap();
//! let X_mice = fitted_mice.transform(&X.view()).unwrap();
//! ```

mod simple_imputer;
mod knn_imputer;
mod em_imputer;
mod iterative_imputer;
mod regression_imputers;
mod multiple_imputer;
mod imputer_traits;
mod validation;
mod utils;

pub use simple_imputer::*;
pub use knn_imputer::*;
pub use em_imputer::*;
pub use iterative_imputer::*;
pub use regression_imputers::*;
pub use multiple_imputer::*;
pub use imputer_traits::*;
pub use validation::*;
pub use utils::*;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests;
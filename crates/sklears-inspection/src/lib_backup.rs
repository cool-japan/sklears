//! Model inspection and interpretation tools
//!
//! This module provides comprehensive tools for understanding and interpreting machine learning models
//! through a modular architecture supporting multiple inspection paradigms.
//!
//! ## Architecture
//!
//! The model inspection system is organized into focused modules:
//! - **Permutation Importance**: Feature importance through permutation testing
//! - **Partial Dependence**: Marginal effects of features on predictions
//! - **SHAP Analysis**: SHapley Additive exPlanations for feature contributions
//! - **Learning Curves**: Model performance vs training set size
//! - **ICE Plots**: Individual Conditional Expectation plots
//! - **Feature Interactions**: Detection and analysis of feature interactions
//! - **LIME Explanations**: Local Interpretable Model-agnostic Explanations
//! - **Residual Analysis**: Analysis of model residuals and diagnostics
//! - **Bias-Variance**: Bias-variance decomposition analysis
//! - **ALE Plots**: Accumulated Local Effects plots
//! - **Inspection Types**: Common type definitions and enumerations
//! - **Inspection Utils**: Shared utilities and helper functions
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use sklears_inspection::*;
//! // ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::Array2;
//!
//! // Permutation importance analysis
//! let perm_result = permutation_importance(
//!     &predict_fn,
//!     &X.view(),
//!     &y.view(),
//!     ScoreFunction::R2,
//!     5,
//!     Some(42),
//! )?;
//!
//! // Partial dependence analysis
//! let pd_result = partial_dependence(
//!     &predict_fn,
//!     &X.view(),
//!     &[0, 1],
//!     &[grid_0, grid_1],
//!     PartialDependenceKind::Average,
//! )?;
//!
//! // SHAP value computation
//! let shap_result = shap_values(
//!     &predict_fn,
//!     &X.view(),
//!     &baseline.view(),
//!     100,
//!     Some(42),
//! )?;
//! ```

#![warn(missing_docs)]

mod permutation_importance;
mod partial_dependence;
mod shap_analysis;
mod learning_curves;
mod ice_plots;
mod feature_interactions;
mod lime_explanations;
mod residual_analysis;
mod bias_variance;
mod ale_plots;
mod inspection_types;
mod inspection_utils;

pub use permutation_importance::*;
pub use partial_dependence::*;
pub use shap_analysis::*;
pub use learning_curves::*;
pub use ice_plots::*;
pub use feature_interactions::*;
pub use lime_explanations::*;
pub use residual_analysis::*;
pub use bias_variance::*;
pub use ale_plots::*;
pub use inspection_types::*;
pub use inspection_utils::*;


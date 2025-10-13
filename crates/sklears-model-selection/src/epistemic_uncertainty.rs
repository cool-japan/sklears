//! Epistemic Uncertainty Quantification
//!
//! This module provides comprehensive methods for quantifying both epistemic and aleatoric
//! uncertainty in machine learning models through a modular architecture.

mod aleatoric_quantifier;
mod bayesian_methods;
mod calibration;
mod ensemble_methods;
mod epistemic_quantifier;
mod monte_carlo_methods;
mod reliability_analysis;
mod uncertainty_config;
mod uncertainty_decomposition;
mod uncertainty_methods;
mod uncertainty_quantifier;
mod uncertainty_results;
mod uncertainty_types;
mod uncertainty_utilities;
mod variance_estimation;

pub use aleatoric_quantifier::*;
pub use calibration::*;
pub use epistemic_quantifier::*;
pub use uncertainty_config::*;
pub use uncertainty_decomposition::*;
pub use uncertainty_methods::*;
pub use uncertainty_quantifier::*;
pub use uncertainty_results::*;
pub use uncertainty_types::*;
pub use uncertainty_utilities::*;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests;

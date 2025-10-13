//! Entropy-based semi-supervised learning methods
//!
//! This module provides entropy-based semi-supervised learning algorithms
//! including entropy regularization, confident learning, and active learning methods.

pub mod confident_learning;
pub mod decision_boundary_semi_supervised;
pub mod entropy_active_learning;
pub mod entropy_regularization;
pub mod minimum_entropy_discrimination;

pub use confident_learning::*;
pub use decision_boundary_semi_supervised::*;
pub use entropy_active_learning::*;
pub use entropy_regularization::*;
pub use minimum_entropy_discrimination::*;

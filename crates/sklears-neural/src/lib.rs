#![allow(missing_docs)]
//! Neural network implementations for the sklears machine learning library.
//!
//! This crate provides implementations of neural network algorithms compatible with
//! the scikit-learn API, including Multi-Layer Perceptron (MLP) for classification
//! and regression tasks.
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_neural::{MLPClassifier, Activation};
//! use sklears_core::traits::{Predict};
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let x = Array2::from_shape_vec((4, 2), vec![
//!     0.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 0.0,
//!     1.0, 1.0,
//! ])?;
//! let y = vec![0, 1, 1, 0]; // XOR problem
//!
//! let mlp = MLPClassifier::new()
//!     .hidden_layer_sizes(&[10, 5])
//!     .activation(Activation::Relu)
//!     .max_iter(1000)
//!     .learning_rate_init(0.01)
//!     .random_state(42);
//!
//! let trained_mlp = mlp.fit(&x, &y)?;
//! let predictions = trained_mlp.predict(&x)?;
//! # Ok(())
//! # }
//! ```

// Allow common patterns in machine learning code and research environments
#![allow(non_snake_case)] // Allow X, Y variable names common in ML
#![allow(unused_imports)] // Allow unused imports for conditional compilation
#![allow(unused_variables)] // Allow unused variables in development/research code
#![allow(dead_code)] // Allow dead code during development
#![allow(deprecated)] // Allow deprecated items during transitions
#![allow(unexpected_cfgs)] // Allow GPU and other conditional compilation features
#![allow(ambiguous_glob_reexports)]
// Allow re-export conflicts in ML modules

// Clippy lints that are overly strict for ML/research code
#![allow(clippy::too_many_arguments)] // Neural networks often have many parameters
#![allow(clippy::type_complexity)] // Complex types are common in ML
#![allow(clippy::multiple_bound_locations)] // Common in generic ML code
#![allow(clippy::assign_op_pattern)] // Allow manual assignment patterns for clarity
#![allow(clippy::nonminimal_bool)] // Allow explicit boolean expressions
#![allow(clippy::derivable_impls)] // Allow manual Default implementations for clarity
#![allow(clippy::redundant_field_names)] // Allow redundant field names for clarity
#![allow(clippy::needless_update)] // Allow needless struct updates for consistency
#![allow(clippy::match_same_arms)] // Allow same arms in match for completeness
#![allow(clippy::single_match)] // Allow single match expressions
#![allow(clippy::large_enum_variant)] // Allow large enum variants in ML structures
#![allow(clippy::module_inception)] // Allow module inception patterns
#![allow(clippy::new_without_default)] // Allow new without Default for ML types
#![allow(clippy::empty_line_after_doc_comments)] // Allow formatting flexibility
#![allow(clippy::should_implement_trait)] // Allow custom implementations for ML types
#![allow(clippy::clone_on_copy)] // Allow explicit cloning for code clarity
#![allow(clippy::collapsible_else_if)] // Allow explicit conditional structure
#![allow(clippy::if_same_then_else)] // Allow explicit branching in ML code
#![allow(clippy::ptr_arg)] // Allow Vec parameters for ML data structures
#![allow(clippy::option_as_ref_deref)] // Allow explicit Option handling
#![allow(clippy::manual_is_multiple_of)] // Allow manual modulo operations
#![allow(clippy::enum_variant_names)] // Allow descriptive enum variant names
#![allow(clippy::field_reassign_with_default)] // Allow explicit field assignment
#![allow(unused_mut)] // Allow unused mutable variables in research code
#![allow(unused_assignments)] // Allow unused assignments for completeness
#![allow(clippy::unwrap_or_default)] // Allow explicit unwrap_or patterns
#![allow(clippy::wrong_self_convention)] // Allow custom method naming
#![allow(clippy::needless_borrows_for_generic_args)] // Allow explicit borrows
#![allow(clippy::useless_asref)] // Allow explicit as_ref usage
#![allow(clippy::borrow_deref_ref)] // Allow explicit reference patterns
#![allow(clippy::op_ref)] // Allow reference operations for clarity
#![allow(clippy::needless_borrow)] // Allow explicit borrowing for clarity

pub mod activation;
pub mod attention_rnn;
pub mod autoencoder;
pub mod benchmarking;
pub mod checkpointing;
pub mod config;
pub mod conv_layers;
pub mod curriculum;
pub mod data_augmentation;
pub mod diffusion;
pub mod distributed;
pub mod ebm;
pub mod evolutionary_nas;
pub mod experiment_tracking;
pub mod gan;
pub mod gnn;
pub mod gpu;
pub mod gradient_checking;
pub mod interpretation;
pub mod knowledge_distillation;
pub mod layers;
pub mod memory_leak_tests;
pub mod mlp_classifier;
pub mod mlp_regressor;
pub mod model_selection;
pub mod models;
pub mod multi_task;
pub mod nas;
pub mod neural_metrics;
pub mod normalizing_flows;
pub mod performance_testing;
pub mod quantization;
pub mod rbm;
pub mod regularization;
pub mod reinforcement_learning;
pub mod self_supervised;
pub mod seq2seq;
pub mod solvers;
pub mod transfer_learning;
pub mod transformer;
pub mod utils;
pub mod vae;
pub mod validation;
pub mod versioning;
pub mod visualization;
pub mod weight_init;

pub use activation::*;
pub use attention_rnn::*;
pub use autoencoder::*;
pub use benchmarking::*;
pub use checkpointing::*;
pub use config::*;
pub use conv_layers::*;
pub use curriculum::*;
pub use data_augmentation::*;
pub use diffusion::*;
pub use distributed::*;
pub use ebm::*;
pub use evolutionary_nas::*;
pub use experiment_tracking::*;
pub use gan::*;
pub use gnn::*;
pub use gpu::*;
pub use gradient_checking::*;
pub use interpretation::*;
pub use knowledge_distillation::*;
pub use layers::*;
pub use memory_leak_tests::*;
pub use mlp_classifier::*;
pub use mlp_regressor::*;
pub use model_selection::*;
pub use models::*;
pub use multi_task::*;
pub use nas::*;
pub use neural_metrics::*;
pub use normalizing_flows::*;
pub use performance_testing::*;
pub use quantization::*;
pub use rbm::*;
pub use regularization::*;
pub use reinforcement_learning::*;
pub use self_supervised::*;
pub use seq2seq::*;
pub use solvers::*;
pub use transfer_learning::*;
pub use transformer::*;
pub use utils::*;
pub use vae::*;
pub use validation::*;
pub use versioning::*;
pub use visualization::*;
pub use weight_init::*;

#[allow(non_snake_case)]
#[cfg(test)]
mod test_simple;

#[allow(non_snake_case)]
#[cfg(test)]
mod advanced_property_tests;

use sklears_core::error::SklearsError;

/// Result type for neural network operations
pub type NeuralResult<T> = Result<T, SklearsError>;

//! Test modules for Linear and Quadratic Discriminant Analysis
//!
//! This module organizes comprehensive tests for all discriminant analysis algorithms
//! implemented in the sklears-discriminant-analysis crate into focused test modules.

// Core algorithm tests
pub mod basic_lda_qda_tests;
pub mod regularization_tests;
// pub mod robust_methods_tests; // Temporarily disabled

// Parameter optimization tests
// pub mod grid_search_tests; // Temporarily disabled

// Advanced algorithm tests
pub mod discriminant_locality_alignment_tests;
pub mod heteroscedastic_tests;
pub mod locality_preserving_tests;
pub mod marginal_fisher_tests;

// Feature selection and error correction
pub mod error_correcting_tests;
// pub mod feature_selection_tests; // Temporarily disabled

// Specialized methods
pub mod bayesian_tests;
pub mod canonical_tests;
pub mod hierarchical_tests;
pub mod kernel_methods_tests;
pub mod mixture_tests;
pub mod multi_task_tests;
pub mod online_learning_tests;
pub mod stochastic_tests;

// Common test utilities and helpers
pub mod test_utils;

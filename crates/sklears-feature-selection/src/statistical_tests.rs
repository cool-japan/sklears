//! Statistical tests for feature selection
//!
//! This module provides comprehensive statistical testing methods for feature selection including
//! parametric tests, non-parametric tests, mutual information methods, and correlation analysis.
//! All algorithms have been refactored into focused modules for better maintainability
//! and comply with SciRS2 Policy.

// Statistical distribution utilities
mod distributions;
pub use distributions::{
    beta_inc, chi2_cdf, f_cdf, gamma, gamma_inc_lower, gamma_inc_upper, normal_cdf, t_cdf,
};

// Chi-squared tests
mod chi_squared_tests;
pub use chi_squared_tests::{chi2, chi2_contingency};

// F-tests for classification and regression
mod f_tests;
pub use f_tests::{f_classif, f_oneway, f_regression};

// Correlation-based tests
mod correlation_tests;
pub use correlation_tests::{pearson_correlation, r_regression};

// Mutual information methods
mod mutual_information;
pub use mutual_information::{
    estimate_mi_cc, estimate_mi_dc, mutual_info_classif, mutual_info_regression,
};

// Non-parametric tests
mod nonparametric_tests;
pub use nonparametric_tests::{
    friedman_test, kolmogorov_smirnov, kruskal_wallis, mann_whitney_u, wilcoxon_signed_rank,
};

// ANOVA tests
mod anova_tests;
pub use anova_tests::{one_way_anova, repeated_measures_anova, two_way_anova};

// Multivariate tests
pub mod multivariate_tests;
pub use multivariate_tests::{canonical_correlation_test, hotelling_t2, manova};

// Permutation tests
pub mod permutation_tests;
pub use permutation_tests::{bootstrap_test, permutation_test, randomization_test};

// Bayesian tests
mod bayesian_tests;
pub use bayesian_tests::{bayes_factor_test, bayesian_correlation, bayesian_t_test};

// Time series tests
mod time_series_tests;
pub use time_series_tests::{adf_test, granger_causality, ljung_box_test};

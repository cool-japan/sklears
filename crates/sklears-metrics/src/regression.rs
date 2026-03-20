//! Regression metrics for model evaluation
//!
//! This module provides comprehensive regression evaluation metrics including basic error measures,
//! coefficient of determination variants, robust metrics, probabilistic scoring rules, deviance
//! functions, loss functions, and specialized evaluation techniques. All algorithms have been
//! refactored into focused modules for better maintainability and comply with SciRS2 Policy.

// Basic error metrics
mod basic_error_metrics;
pub use basic_error_metrics::{
    max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
    median_absolute_error, median_absolute_percentage_error, root_mean_squared_error,
};

// Coefficient of determination (R²) metrics
mod r2_metrics;
pub use r2_metrics::{
    explained_variance_score, huber_r2_score, mad_r2_score, median_r2_score, r2_score,
    robust_r2_score, trimmed_r2_score,
};

// Logarithmic error metrics
mod logarithmic_metrics;
pub use logarithmic_metrics::{
    log_absolute_error, log_cosh_error, mean_squared_log_error, root_mean_squared_log_error,
};

// Deviance metrics
mod deviance_metrics;
pub use deviance_metrics::{
    kullback_leibler_divergence, mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance,
    negative_log_likelihood,
};

// Loss functions
mod loss_functions;
pub use loss_functions::{
    epsilon_insensitive_loss, hinge_loss, huber_loss, mean_pinball_loss, quantile_loss,
    squared_hinge_loss,
};

// D² score metrics (coefficient of determination variants)
mod d2_score_metrics;
pub use d2_score_metrics::{
    d2_absolute_error_score, d2_gamma_score, d2_pinball_score, d2_poisson_score, d2_tweedie_score,
};

// Probabilistic scoring rules
mod probabilistic_scoring;
pub use probabilistic_scoring::{
    continuous_ranked_probability_score, crps_ensemble, crps_gaussian, dawid_sebastiani_score,
    energy_score, logarithmic_score,
};

// Robust regression metrics
mod robust_metrics;
pub use robust_metrics::{
    biweight_midvariance, kendall_tau_distance, theil_sen_slope, trimmed_mean_error,
    winsorized_mean_error,
};

// Time series specific metrics
mod time_series_metrics;
pub use time_series_metrics::{
    mean_absolute_scaled_error, mean_directional_accuracy,
    symmetric_mean_absolute_percentage_error, weighted_absolute_percentage_error,
};

// Distribution-based metrics
mod distribution_metrics;
pub use distribution_metrics::{
    anderson_darling_statistic, jensen_shannon_divergence, kolmogorov_smirnov_distance,
    wasserstein_distance,
};

// Information-theoretic metrics
mod information_theoretic_metrics;
pub use information_theoretic_metrics::{
    adjusted_mutual_information, mutual_information_score, normalized_mutual_information,
    variation_of_information,
};

// Specialized evaluation metrics
mod specialized_metrics;
pub use specialized_metrics::{
    concordance_correlation_coefficient, lin_concordance_coefficient, mean_absolute_bias,
    mean_bias_error, relative_error_metrics,
};

// Metric utilities and aggregation
mod metric_utilities;
pub use metric_utilities::{
    BootstrapMetrics, CrossValidationMetrics, MetricAggregator, MetricConfidenceIntervals,
    WeightedMetrics,
};

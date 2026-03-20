//! Configuration Presets for Common Use Cases
//!
//! This module provides pre-configured estimators for common covariance estimation scenarios,
//! making it easy to get started with appropriate defaults for different use cases.

use crate::{
    adaptive_lasso::AdaptiveLassoCovariance,
    // bayesian_covariance::{BayesianCovariance, BayesianMethod}, // temporarily commented out
    bigquic::BigQUIC,
    elastic_net::ElasticNetCovariance,
    empirical::EmpiricalCovariance,
    graphical_lasso::GraphicalLasso,
    group_lasso::GroupLassoCovariance,
    huber::HuberCovariance,
    ledoit_wolf::LedoitWolf,
    min_cov_det::MinCovDet,
    oas::OAS,
    ridge::RidgeCovariance,
    shrunk::ShrunkCovariance,
    time_varying_covariance::{TimeVaryingCovariance, TimeVaryingMethod},
};

/// Configuration presets for common covariance estimation scenarios
pub struct CovariancePresets;

impl CovariancePresets {
    /// Standard empirical covariance for clean, well-conditioned data
    pub fn empirical() -> EmpiricalCovariance {
        EmpiricalCovariance::new().assume_centered(false)
    }

    /// Conservative shrinkage for high-dimensional data (p > n)
    pub fn high_dimensional() -> LedoitWolf {
        LedoitWolf::new()
            .assume_centered(false)
            .store_precision(true)
    }

    /// Robust estimation for data with outliers
    pub fn robust() -> MinCovDet {
        MinCovDet::new()
            .support_fraction(0.5) // Default value
            .assume_centered(false)
            .store_precision(true)
    }

    /// Highly robust estimation for heavily contaminated data
    pub fn very_robust() -> HuberCovariance {
        HuberCovariance::new().max_iter(1000).tol(1e-6)
    }

    /// Sparse precision matrix estimation (Gaussian Graphical Models)
    pub fn sparse_precision() -> GraphicalLasso {
        GraphicalLasso::new()
            .alpha(0.01) // Moderate sparsity
            .mode("cd".to_string())
            .tol(1e-4)
            .max_iter(100)
            .assume_centered(false)
    }

    /// Large-scale sparse precision matrix estimation
    pub fn large_scale_sparse() -> BigQUIC {
        BigQUIC::new()
            .lambda(0.1)
            .tol(1e-3)
            .max_iter(100)
            .mode(crate::bigquic::BigQUICMode::Standard)
    }

    /// Adaptive sparsity with data-dependent regularization
    pub fn adaptive_sparse() -> AdaptiveLassoCovariance {
        AdaptiveLassoCovariance::new()
            .gamma(1.0) // Standard adaptive weights
            .max_iter(1000)
            .tol(1e-6)
    }

    /// Elastic net regularization (balanced L1/L2)
    pub fn elastic_net() -> ElasticNetCovariance {
        ElasticNetCovariance::new()
            .alpha(0.1)
            .l1_ratio(0.5) // Balanced L1/L2
            .max_iter(1000)
            .tol(1e-6)
    }

    /// Ridge regularization for numerical stability
    pub fn ridge_stable() -> RidgeCovariance {
        RidgeCovariance::new()
            .alpha(1e-3) // Small regularization for stability
            .assume_centered(false)
            .store_precision(true)
    }

    /// Group sparse estimation for structured features
    pub fn group_sparse(groups: Vec<Vec<usize>>) -> GroupLassoCovariance {
        GroupLassoCovariance::new()
            .alpha(0.1)
            .max_iter(1000)
            .tol(1e-6)
    }

    /// Oracle Approximating Shrinkage for optimal bias-variance tradeoff
    pub fn optimal_shrinkage() -> OAS {
        OAS::new().assume_centered(false).store_precision(false)
    }

    /// Classical shrinkage with fixed shrinkage intensity
    pub fn classical_shrinkage(shrinkage: Option<f64>) -> ShrunkCovariance {
        ShrunkCovariance::new()
            .shrinkage(shrinkage.unwrap_or(0.1))
            .assume_centered(false)
            .store_precision(false)
    }

    /// Financial time series with GARCH-type volatility modeling
    pub fn financial_timeseries() -> TimeVaryingCovariance<f64> {
        TimeVaryingCovariance::builder()
            .method(TimeVaryingMethod::DCC)
            .rolling_window_size(250) // One year of daily data
            .build()
    }

    // Temporarily commented out due to compilation issues
    // /// Bayesian estimation with informative priors
    // pub fn bayesian_informative() -> BayesianCovariance<f64> {
    //     BayesianCovariance::builder()
    //         .method(BayesianMethod::InverseWishart)
    //         .mcmc_samples(1000)
    //         .mcmc_burn_in(200)
    //         .build()
    // }

    // /// Bayesian estimation with vague priors
    // pub fn bayesian_vague() -> BayesianCovariance<f64> {
    //     BayesianCovariance::builder()
    //         .method(BayesianMethod::VariationalBayes)
    //         .mcmc_samples(2000)
    //         .mcmc_burn_in(500)
    //         .build()
    // }
}

/// Financial domain presets
pub struct Financial;

impl Financial {
    /// Risk factor modeling
    pub fn risk_factors() -> LedoitWolf {
        LedoitWolf::new()
            .assume_centered(true) // Risk factors often centered
            .store_precision(true)
    }

    /// Portfolio optimization
    pub fn portfolio() -> GraphicalLasso {
        GraphicalLasso::new()
            .alpha(0.05) // Moderate sparsity for diversification
            .mode("cd".to_string())
            .max_iter(200)
            .assume_centered(false)
    }

    /// High-frequency trading data
    pub fn high_frequency() -> TimeVaryingCovariance<f64> {
        TimeVaryingCovariance::builder()
            .method(TimeVaryingMethod::EWMA)
            .rolling_window_size(100) // Short memory for HFT
            .build()
    }
}

/// Genomics domain presets
pub struct Genomics;

impl Genomics {
    /// Gene expression analysis
    pub fn expression() -> BigQUIC {
        BigQUIC::new()
            .lambda(0.01) // High sparsity for gene networks
            .tol(1e-4)
            .max_iter(500)
            .mode(crate::bigquic::BigQUICMode::Standard)
    }

    /// Pathway analysis
    pub fn pathways(pathway_groups: Vec<Vec<usize>>) -> GroupLassoCovariance {
        GroupLassoCovariance::new()
            .alpha(0.05)
            .max_iter(2000)
            .tol(1e-8)
    }
}

/// Signal processing domain presets
pub struct SignalProcessing;

impl SignalProcessing {
    /// Spatial covariance for antenna arrays
    pub fn spatial() -> HuberCovariance {
        HuberCovariance::new().max_iter(500).tol(1e-8)
    }

    /// Adaptive filtering
    pub fn adaptive() -> TimeVaryingCovariance<f64> {
        TimeVaryingCovariance::builder()
            .method(TimeVaryingMethod::RLS)
            .rolling_window_size(50) // Short adaptation window
            .build()
    }
}

/// Domain-specific preset collections
pub struct DomainPresets;

impl DomainPresets {
    /// Get financial domain presets
    pub fn financial() -> &'static Financial {
        &Financial
    }

    /// Get genomics domain presets
    pub fn genomics() -> &'static Genomics {
        &Genomics
    }

    /// Get signal processing domain presets
    pub fn signal_processing() -> &'static SignalProcessing {
        &SignalProcessing
    }
}

/// Preset validation and recommendations
pub struct PresetRecommendations;

impl PresetRecommendations {
    /// Recommend preset based on data characteristics
    pub fn recommend_for_data(
        n_samples: usize,
        n_features: usize,
        has_outliers: bool,
        needs_sparsity: bool,
        is_time_series: bool,
    ) -> &'static str {
        match (
            n_samples < n_features,
            has_outliers,
            needs_sparsity,
            is_time_series,
        ) {
            (true, false, false, false) => "high_dimensional",
            (true, false, true, false) => "sparse_precision",
            (false, true, false, false) => "robust",
            (false, true, true, false) => "adaptive_sparse",
            (false, false, true, false) => "sparse_precision", // Sparsity needed regardless of sample ratio
            (_, _, _, true) => "financial_timeseries",
            (false, false, false, false) => "empirical",
            (true, true, false, false) => "very_robust",
            _ => "empirical", // Default fallback
        }
    }

    /// Get preset description
    pub fn describe(preset_name: &str) -> &'static str {
        match preset_name {
            "empirical" => "Standard empirical covariance for clean, well-conditioned data",
            "high_dimensional" => "Ledoit-Wolf shrinkage for high-dimensional data (p > n)",
            "robust" => "MinCovDet robust estimation for data with outliers",
            "very_robust" => "Huber robust estimation for heavily contaminated data",
            "sparse_precision" => "GraphicalLasso for sparse precision matrix estimation",
            "large_scale_sparse" => "BigQUIC for large-scale sparse problems",
            "adaptive_sparse" => "AdaptiveLasso with data-dependent regularization",
            "elastic_net" => "ElasticNet with balanced L1/L2 regularization",
            "ridge_stable" => "Ridge regularization for numerical stability",
            "optimal_shrinkage" => "OAS for optimal bias-variance tradeoff",
            "financial_timeseries" => "Time-varying covariance for financial data",
            "bayesian_informative" => "Bayesian estimation with informative priors",
            "bayesian_vague" => "Bayesian estimation with vague priors",
            _ => "Unknown preset",
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_preset_creation() {
        // Test that all presets can be created without errors
        let _emp = CovariancePresets::empirical();
        let _hd = CovariancePresets::high_dimensional();
        let _rob = CovariancePresets::robust();
        let _sparse = CovariancePresets::sparse_precision();
        let _ridge = CovariancePresets::ridge_stable();
    }

    #[test]
    fn test_domain_presets() {
        // Test domain-specific presets
        let _risk = Financial::risk_factors();
        let _portfolio = Financial::portfolio();
        let _hft = Financial::high_frequency();

        let _expression = Genomics::expression();
        let _spatial = SignalProcessing::spatial();
    }

    #[test]
    fn test_recommendations() {
        // Test preset recommendations
        assert_eq!(
            PresetRecommendations::recommend_for_data(100, 50, false, false, false),
            "empirical"
        );
        assert_eq!(
            PresetRecommendations::recommend_for_data(50, 100, false, false, false),
            "high_dimensional"
        );
        assert_eq!(
            PresetRecommendations::recommend_for_data(100, 50, true, false, false),
            "robust"
        );
        assert_eq!(
            PresetRecommendations::recommend_for_data(100, 50, false, true, false),
            "sparse_precision"
        );
        assert_eq!(
            PresetRecommendations::recommend_for_data(100, 50, false, false, true),
            "financial_timeseries"
        );
    }

    #[test]
    fn test_preset_descriptions() {
        assert!(!PresetRecommendations::describe("empirical").is_empty());
        assert!(!PresetRecommendations::describe("robust").is_empty());
        assert!(!PresetRecommendations::describe("sparse_precision").is_empty());
    }
}

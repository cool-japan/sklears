//! Statistical distribution functions for feature selection tests
//!
//! This module provides statistical distribution functions used in various feature selection tests.
//! All implementations follow SciRS2 Policy using scirs2-core for numerical operations.

use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, Normal, StudentsT};
use statrs::function::{
    beta::beta_reg,
    gamma::{self as statrs_gamma},
};

/// Gamma function using `statrs`
pub fn gamma(x: f64) -> f64 {
    statrs_gamma::gamma(x)
}

/// Lower incomplete gamma function using `statrs`
pub fn gamma_inc_lower(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    statrs_gamma::gamma_li(a, x)
}

/// Upper incomplete gamma function using `statrs`
pub fn gamma_inc_upper(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return statrs_gamma::gamma(a);
    }
    statrs_gamma::gamma_ui(a, x)
}

/// Regularized incomplete beta function
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    beta_reg(a, b, x)
}

/// Chi-squared cumulative distribution function
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    match ChiSquared::new(df) {
        Ok(dist) => dist.cdf(x).clamp(0.0, 1.0),
        Err(_) => f64::NAN,
    }
}

/// F-distribution cumulative distribution function
pub fn f_cdf(x: f64, df1: f64, df2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    match FisherSnedecor::new(df1, df2) {
        Ok(dist) => dist.cdf(x).clamp(0.0, 1.0),
        Err(_) => f64::NAN,
    }
}

/// Student's t-distribution cumulative distribution function
pub fn t_cdf(x: f64, df: f64) -> f64 {
    match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => dist.cdf(x).clamp(0.0, 1.0),
        Err(_) => f64::NAN,
    }
}

/// Standard normal cumulative distribution function
pub fn normal_cdf(x: f64) -> f64 {
    match Normal::new(0.0, 1.0) {
        Ok(dist) => dist.cdf(x).clamp(0.0, 1.0),
        Err(_) => f64::NAN,
    }
}

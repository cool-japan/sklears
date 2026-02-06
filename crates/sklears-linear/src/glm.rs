//! Generalized Linear Models (GLM) implementation
//!
//! This module provides a flexible framework for fitting generalized linear models
//! with various distributions and link functions.

use std::marker::PhantomData;

use scirs2_core::ndarray::{stack, Array, Axis};
use scirs2_linalg::compat::ArrayLinalgExt;
use sklears_core::{
    error::{validate, Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::{Array1, Array2, Float},
};

/// Distribution families for GLM
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    /// Normal distribution with identity link (ordinary linear regression)
    Gaussian,
    /// Poisson distribution with log link
    Poisson,
    /// Binomial distribution with logit link
    Binomial,
    /// Gamma distribution with inverse link
    Gamma,
    /// Inverse Gaussian distribution
    InverseGaussian,
}

/// Link functions for GLM
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Link {
    /// Identity link: g(μ) = μ
    Identity,
    /// Log link: g(μ) = log(μ)
    Log,
    /// Logit link: g(μ) = log(μ/(1-μ))
    Logit,
    /// Inverse link: g(μ) = 1/μ
    Inverse,
    /// Square root link: g(μ) = sqrt(μ)
    Sqrt,
    /// Probit link: g(μ) = Φ^(-1)(μ) where Φ is the normal CDF
    Probit,
}

impl Family {
    /// Get the canonical link function for this family
    pub fn canonical_link(&self) -> Link {
        match self {
            Family::Gaussian => Link::Identity,
            Family::Poisson => Link::Log,
            Family::Binomial => Link::Logit,
            Family::Gamma => Link::Inverse,
            Family::InverseGaussian => Link::Inverse,
        }
    }

    /// Variance function for the family
    pub fn variance(&self, mu: Float) -> Float {
        match self {
            Family::Gaussian => 1.0,
            Family::Poisson => mu.max(1e-10),
            Family::Binomial => mu * (1.0 - mu).max(1e-10),
            Family::Gamma => mu * mu,
            Family::InverseGaussian => mu.powi(3),
        }
    }

    /// Deviance function
    pub fn deviance(&self, y: Float, mu: Float) -> Float {
        let eps = 1e-10;
        match self {
            Family::Gaussian => (y - mu).powi(2),
            Family::Poisson => {
                let mu_safe = mu.max(eps);
                let y_safe = y.max(eps);
                2.0 * (y * (y_safe / mu_safe).ln() - (y - mu))
            }
            Family::Binomial => {
                let mu_safe = mu.clamp(eps, 1.0 - eps);
                let _y_safe = y.clamp(eps, 1.0 - eps);
                -2.0 * (y * (mu_safe).ln() + (1.0 - y) * (1.0 - mu_safe).ln())
            }
            Family::Gamma => {
                let mu_safe = mu.max(eps);
                let y_safe = y.max(eps);
                2.0 * ((y - mu) / mu_safe - (y_safe / mu_safe).ln())
            }
            Family::InverseGaussian => {
                let mu_safe = mu.max(eps);
                let y_safe = y.max(eps);
                (y - mu).powi(2) / (mu_safe.powi(2) * y_safe)
            }
        }
    }
}

impl Link {
    /// Apply the link function
    pub fn link(&self, mu: Float) -> Float {
        let eps = 1e-10;
        match self {
            Link::Identity => mu,
            Link::Log => mu.max(eps).ln(),
            Link::Logit => {
                let mu_safe = mu.clamp(eps, 1.0 - eps);
                (mu_safe / (1.0 - mu_safe)).ln()
            }
            Link::Inverse => 1.0 / mu.max(eps),
            Link::Sqrt => mu.max(0.0).sqrt(),
            Link::Probit => {
                // Approximate inverse normal CDF
                let mu_safe = mu.clamp(eps, 1.0 - eps);
                // Using a simple approximation for now
                (2.0 * mu_safe - 1.0).clamp(-0.99, 0.99) * 2.5066
            }
        }
    }

    /// Apply the inverse link function
    pub fn inverse_link(&self, eta: Float) -> Float {
        match self {
            Link::Identity => eta,
            Link::Log => eta.exp(),
            Link::Logit => 1.0 / (1.0 + (-eta).exp()),
            Link::Inverse => 1.0 / eta.max(1e-10),
            Link::Sqrt => eta.max(0.0).powi(2),
            Link::Probit => {
                // Approximate normal CDF
                // Using a simple approximation for now
                0.5 * (1.0 + (eta / 2.5066).tanh())
            }
        }
    }

    /// Derivative of the inverse link function
    pub fn inverse_link_deriv(&self, eta: Float) -> Float {
        let eps = 1e-10;
        match self {
            Link::Identity => 1.0,
            Link::Log => eta.exp(),
            Link::Logit => {
                let exp_eta = eta.exp();
                exp_eta / (1.0 + exp_eta).powi(2)
            }
            Link::Inverse => -1.0 / eta.max(eps).powi(2),
            Link::Sqrt => 2.0 * eta.max(0.0),
            Link::Probit => {
                // Derivative of normal CDF is normal PDF
                let x = eta / 2.5066;
                0.3989 * (-0.5 * x * x).exp()
            }
        }
    }
}

/// Configuration for Generalized Linear Model
#[derive(Debug, Clone)]
pub struct GLMConfig {
    /// Distribution family
    pub family: Family,
    /// Link function (None = use canonical link)
    pub link: Option<Link>,
    /// Maximum iterations for IRLS
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Regularization parameter (L2 penalty)
    pub alpha: f64,
}

impl Default for GLMConfig {
    fn default() -> Self {
        Self {
            family: Family::Gaussian,
            link: None,
            max_iter: 100,
            tol: 1e-4,
            fit_intercept: true,
            alpha: 0.0,
        }
    }
}

/// Generalized Linear Model
#[derive(Debug, Clone)]
pub struct GeneralizedLinearModel<State = Untrained> {
    config: GLMConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_: Option<usize>,
    link_: Option<Link>,
    deviance_: Option<Float>,
    n_iter_: Option<usize>,
}

impl GeneralizedLinearModel<Untrained> {
    /// Create a new GLM
    pub fn new(family: Family) -> Self {
        Self {
            config: GLMConfig {
                family,
                ..Default::default()
            },
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
            link_: None,
            deviance_: None,
            n_iter_: None,
        }
    }

    /// Set custom link function
    pub fn link(mut self, link: Link) -> Self {
        self.config.link = Some(link);
        self
    }

    /// Set regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }
}

impl Estimator for GeneralizedLinearModel<Untrained> {
    type Config = GLMConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for GeneralizedLinearModel<Untrained> {
    type Fitted = GeneralizedLinearModel<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let fit_intercept = self.config.fit_intercept;

        // Get link function
        let link = self
            .config
            .link
            .unwrap_or_else(|| self.config.family.canonical_link());

        // Initialize parameters
        let mut coef = Array::zeros(n_features);
        let mut intercept = if self.config.fit_intercept {
            // Initialize intercept based on the mean of y through the link
            let y_mean = y.mean().unwrap();
            match self.config.family {
                Family::Binomial => {
                    // For binomial, ensure y_mean is in (0, 1)
                    let p = y_mean.clamp(0.01, 0.99);
                    link.link(p)
                }
                _ => link.link(y_mean.max(1e-10)),
            }
        } else {
            0.0
        };

        // IRLS (Iteratively Reweighted Least Squares) algorithm
        let mut converged = false;
        let mut n_iter = 0;
        let mut deviance_old = Float::INFINITY;

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // Compute linear predictor
            let mut eta = x.dot(&coef);
            if self.config.fit_intercept {
                eta += intercept;
            }

            // Compute predicted mean
            let mu: Array1<Float> = eta.mapv(|e| link.inverse_link(e));

            // Compute weights and working response
            let mut weights = Array::zeros(n_samples);
            let mut z = Array::zeros(n_samples);

            for i in 0..n_samples {
                let mu_i = mu[i];
                let eta_i = eta[i];
                let y_i = y[i];

                // Variance and derivative
                let variance = self.config.family.variance(mu_i);
                let g_prime = link.inverse_link_deriv(eta_i);

                // Weight for IRLS
                weights[i] = (g_prime * g_prime / variance).max(1e-10);

                // Working response
                z[i] = eta_i + (y_i - mu_i) / g_prime;
            }

            // Weighted least squares update
            let sqrt_w = weights.mapv(|w| w.sqrt());

            // Build weighted design matrix
            let x_weighted_views: Vec<_> = x
                .outer_iter()
                .zip(sqrt_w.iter())
                .map(|(row, &w)| row.mapv(|x| x * w))
                .collect();
            let x_weighted_refs: Vec<_> = x_weighted_views.iter().map(|row| row.view()).collect();
            let x_weighted = stack(Axis(0), &x_weighted_refs).map_err(|_| {
                SklearsError::NumericalError("Failed to create weighted matrix".to_string())
            })?;

            let z_weighted = &z * &sqrt_w;

            // Solve weighted least squares
            // (X^T W X) β = X^T W z
            let xtw = x_weighted.t();
            let xtwx = xtw.dot(&x_weighted);
            let xtwz = xtw.dot(&z_weighted);

            let mut params = &xtwx.solve(&xtwz).map_err(|e| {
                SklearsError::NumericalError(format!(
                    "Failed to solve weighted least squares: {}",
                    e
                ))
            })?;

            // Add L2 regularization if needed
            let params = if self.config.alpha > 0.0 {
                let reg_factor = 1.0 / (1.0 + self.config.alpha);
                params * reg_factor
            } else {
                params.to_owned()
            };

            // Update parameters
            if self.config.fit_intercept {
                intercept = params
                    .iter()
                    .zip(z.iter())
                    .zip(weights.iter())
                    .map(|((&p, &z), &w)| w * (z - p))
                    .sum::<Float>()
                    / weights.sum();
                coef.assign(&params);
            } else {
                coef.assign(&params);
            }

            // Compute deviance
            let mut deviance = 0.0;
            for i in 0..n_samples {
                deviance += self.config.family.deviance(y[i], mu[i]);
            }

            // Check convergence
            if (deviance_old - deviance).abs() < self.config.tol * deviance.abs().max(1.0) {
                converged = true;
                deviance_old = deviance;
                break;
            }
            deviance_old = deviance;
        }

        if !converged {
            eprintln!(
                "Warning: GLM did not converge within {} iterations",
                self.config.max_iter
            );
        }

        Ok(GeneralizedLinearModel {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: if fit_intercept { Some(intercept) } else { None },
            n_features_: Some(n_features),
            link_: Some(link),
            deviance_: Some(deviance_old),
            n_iter_: Some(n_iter),
        })
    }
}

impl GeneralizedLinearModel<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array1<Float> {
        self.coef_.as_ref().expect("Model is trained")
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<Float> {
        self.intercept_
    }

    /// Get the deviance
    pub fn deviance(&self) -> Float {
        self.deviance_.expect("Model is trained")
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.expect("Model is trained")
    }
}

impl Predict<Array2<Float>, Array1<Float>> for GeneralizedLinearModel<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        validate::check_n_features(x, n_features)?;

        let coef = self.coef_.as_ref().expect("Model is trained");
        let link = self.link_.expect("Model is trained");

        // Compute linear predictor
        let mut eta = x.dot(coef);
        if let Some(intercept) = self.intercept_ {
            eta += intercept;
        }

        // Apply inverse link to get predictions
        Ok(eta.mapv(|e| link.inverse_link(e)))
    }
}

impl Score<Array2<Float>, Array1<Float>> for GeneralizedLinearModel<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Compute deviance-based R²
        let mut total_deviance = 0.0;
        let mut null_deviance = 0.0;

        // Null model prediction (just the mean)
        let y_mean = y.mean().unwrap();
        let link = self.link_.expect("Model is trained");
        let null_pred = link.inverse_link(link.link(y_mean.max(1e-10)));

        for i in 0..y.len() {
            total_deviance += self.config.family.deviance(y[i], predictions[i]);
            null_deviance += self.config.family.deviance(y[i], null_pred);
        }

        // Pseudo R² (McFadden's R²)
        Ok(1.0 - total_deviance / null_deviance.max(1e-10))
    }
}

/// Convenience constructors for common GLM types
impl GeneralizedLinearModel<Untrained> {
    /// Create a Poisson regression model
    pub fn poisson() -> Self {
        Self::new(Family::Poisson)
    }

    /// Create a Gamma regression model
    pub fn gamma() -> Self {
        Self::new(Family::Gamma)
    }

    /// Create a Binomial regression model (alternative to LogisticRegression)
    pub fn binomial() -> Self {
        Self::new(Family::Binomial)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_link_functions() {
        // Test identity link
        let link = Link::Identity;
        assert_eq!(link.link(2.0), 2.0);
        assert_eq!(link.inverse_link(2.0), 2.0);

        // Test log link
        let link = Link::Log;
        assert_abs_diff_eq!(link.link(std::f64::consts::E), 1.0, epsilon = 0.001);
        assert_abs_diff_eq!(link.inverse_link(1.0), std::f64::consts::E, epsilon = 0.001);

        // Test logit link
        let link = Link::Logit;
        assert_abs_diff_eq!(link.link(0.5), 0.0, epsilon = 0.001);
        assert_abs_diff_eq!(link.inverse_link(0.0), 0.5, epsilon = 0.001);
    }

    #[test]
    fn test_glm_gaussian() {
        // Gaussian GLM should be equivalent to linear regression
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![3.0, 5.0, 7.0, 9.0]; // y = x1 + x2

        let model = GeneralizedLinearModel::new(Family::Gaussian)
            .fit_intercept(false) // No intercept since y = x1 + x2
            .fit(&x, &y)
            .unwrap();

        let coef = model.coef();
        assert_abs_diff_eq!(coef[0], 1.0, epsilon = 0.1);
        assert_abs_diff_eq!(coef[1], 1.0, epsilon = 0.1);

        let score = model.score(&x, &y).unwrap();
        println!("GLM Gaussian score = {}", score);
        assert!(score > 0.95); // Should be perfect fit
    }

    #[test]
    fn test_glm_poisson() {
        // Simple Poisson regression
        let x = array![[0.0], [1.0], [2.0], [3.0],];
        let y = array![1.0, 2.0, 4.0, 8.0]; // Roughly exponential

        let model = GeneralizedLinearModel::poisson().fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();

        // Predictions should be positive
        for &pred in predictions.iter() {
            assert!(pred > 0.0);
        }
    }
}

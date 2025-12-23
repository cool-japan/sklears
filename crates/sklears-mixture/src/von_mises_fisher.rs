//! Von Mises-Fisher Mixture Models
//!
//! This module provides mixture models for von Mises-Fisher distributions,
//! which are ideal for clustering directional data on the unit hypersphere.
//! Common applications include text document clustering, gene expression analysis,
//! and geographical direction clustering.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::f64::consts::PI;

use crate::common::ModelSelection;

/// Utility function for log-sum-exp
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max_val = a.max(b);
    if max_val.is_finite() {
        max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
    } else {
        max_val
    }
}

/// Approximate gamma function using Stirling's approximation for simplicity
fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        return PI / ((PI * x).sin() * gamma(1.0 - x));
    }

    if x < 1.5 {
        return gamma(x + 1.0) / x;
    }

    // Stirling's approximation
    (2.0 * PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
}

/// von Mises-Fisher distribution for directional data on the unit sphere
///
/// The von Mises-Fisher (vMF) distribution is the analog of the multivariate
/// normal distribution for directional data on the unit hypersphere.
/// It is parameterized by a mean direction μ (unit vector) and a concentration
/// parameter κ ≥ 0.
#[derive(Debug, Clone)]
pub struct VonMisesFisher {
    /// Mean direction (unit vector)
    pub mu: Array1<f64>,
    /// Concentration parameter (κ ≥ 0)
    pub kappa: f64,
    /// Cached normalization constant
    normalization_constant: f64,
}

impl VonMisesFisher {
    /// Create a new von Mises-Fisher distribution
    pub fn new(mu: Array1<f64>, kappa: f64) -> Result<Self, SklearsError> {
        if kappa < 0.0 {
            return Err(SklearsError::InvalidInput(
                "Concentration parameter kappa must be non-negative".to_string(),
            ));
        }

        // Normalize the mean direction
        let norm = mu.dot(&mu).sqrt();
        if norm < 1e-10 {
            return Err(SklearsError::InvalidInput(
                "Mean direction vector cannot be zero".to_string(),
            ));
        }
        let mu_normalized = &mu / norm;

        let d = mu.len() as f64;
        let normalization_constant = Self::compute_normalization_constant(kappa, d);

        Ok(VonMisesFisher {
            mu: mu_normalized,
            kappa,
            normalization_constant,
        })
    }

    /// Compute the normalization constant C_d(κ)
    fn compute_normalization_constant(kappa: f64, d: f64) -> f64 {
        if kappa < 1e-10 {
            // For small κ, use approximation
            return 1.0 / (2.0 * PI).powf(d / 2.0);
        }

        let nu = d / 2.0 - 1.0;

        // For numerical stability, use the following approximation for the modified Bessel function
        // I_ν(κ) ≈ exp(κ) / sqrt(2πκ) for large κ
        if kappa > 10.0 {
            let bessel_approx = (kappa.exp()) / (2.0 * PI * kappa).sqrt();
            return kappa.powf(nu) / ((2.0 * PI).powf(d / 2.0) * bessel_approx);
        }

        // For moderate κ, use series expansion for modified Bessel function
        let bessel_i_nu = Self::modified_bessel_i(nu, kappa);
        kappa.powf(nu) / ((2.0 * PI).powf(d / 2.0) * bessel_i_nu)
    }

    /// Approximate modified Bessel function of the first kind I_ν(x)
    fn modified_bessel_i(nu: f64, x: f64) -> f64 {
        if x < 1e-10 {
            if nu.abs() < 1e-10 {
                return 1.0;
            } else {
                return 0.0;
            }
        }

        // Use series expansion for small to moderate x
        if x < 20.0 {
            let mut sum = 0.0;
            let mut term = (x / 2.0).powf(nu) / gamma(nu + 1.0);
            sum += term;

            for k in 1..50 {
                term *= (x * x) / (4.0 * k as f64 * (nu + k as f64));
                sum += term;
                if term.abs() < sum.abs() * 1e-15 {
                    break;
                }
            }
            sum
        } else {
            // Asymptotic expansion for large x
            x.exp() / (2.0 * PI * x).sqrt()
        }
    }

    /// Compute log probability density
    pub fn log_pdf(&self, x: &ArrayView2<f64>) -> Array1<f64> {
        let n_samples = x.nrows();
        let mut log_probs = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let x_i = x.row(i);

            // Normalize x_i to unit vector
            let norm = x_i.dot(&x_i).sqrt();
            if norm < 1e-10 {
                log_probs[i] = f64::NEG_INFINITY;
                continue;
            }

            let x_normalized = &x_i / norm;
            let dot_product = self.mu.dot(&x_normalized);

            // Clamp dot product to [-1, 1] for numerical stability
            let dot_product = dot_product.clamp(-1.0, 1.0);

            log_probs[i] = self.normalization_constant.ln() + self.kappa * dot_product;
        }

        log_probs
    }

    /// Compute probability density
    pub fn pdf(&self, x: &ArrayView2<f64>) -> Array1<f64> {
        self.log_pdf(x).mapv(f64::exp)
    }

    /// Generate random samples from the distribution
    pub fn sample(&self, n_samples: usize, rng: &mut impl scirs2_core::random::Rng) -> Array2<f64> {
        let d = self.mu.len();
        let mut samples = Array2::zeros((n_samples, d));

        for i in 0..n_samples {
            let sample = self.sample_one(rng);
            samples.row_mut(i).assign(&sample);
        }

        samples
    }

    /// Generate a single random sample
    pub fn sample_one(&self, rng: &mut impl scirs2_core::random::Rng) -> Array1<f64> {
        let d = self.mu.len() as f64;

        if self.kappa < 1e-10 {
            // For κ ≈ 0, sample uniformly on the sphere
            return self.sample_uniform_sphere(rng);
        }

        // Use Ulrich's algorithm for sampling from vMF distribution
        let b = (-2.0 * self.kappa
            + (4.0 * self.kappa * self.kappa + (d - 1.0) * (d - 1.0)).sqrt())
            / (d - 1.0);
        let x0 = (1.0 - b) / (1.0 + b);
        let c = self.kappa * x0 + (d - 1.0) * (1.0 - x0 * x0).ln();

        loop {
            let z: f64 = rng.gen(); // uniform [0,1]
            let u: f64 = rng.gen(); // uniform [0,1]
            let w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z);

            if self.kappa * w + (d - 1.0) * (1.0 - x0 * w).ln() - c >= (u * 2.0 - 1.0).ln() {
                // Generate random unit vector orthogonal to mu
                let v = self.sample_uniform_sphere_orthogonal(rng);

                // Return w * mu + sqrt(1 - w²) * v
                let sqrt_term = (1.0 - w * w).sqrt();
                return &self.mu * w + &v * sqrt_term;
            }
        }
    }

    /// Sample uniformly from the unit sphere
    fn sample_uniform_sphere(&self, rng: &mut impl scirs2_core::random::Rng) -> Array1<f64> {
        let d = self.mu.len();
        let mut v = Array1::zeros(d);

        // Generate d independent standard normal random variables
        for i in 0..d {
            v[i] = self.sample_standard_normal(rng);
        }

        // Normalize to unit length
        let norm = v.dot(&v).sqrt();
        if norm > 1e-10 {
            v /= norm;
        }

        v
    }

    /// Sample uniformly from the unit sphere orthogonal to mu
    fn sample_uniform_sphere_orthogonal(
        &self,
        rng: &mut impl scirs2_core::random::Rng,
    ) -> Array1<f64> {
        let d = self.mu.len();
        let mut v = Array1::zeros(d);

        // Generate d independent standard normal random variables
        for i in 0..d {
            v[i] = self.sample_standard_normal(rng);
        }

        // Remove component parallel to mu
        let proj = v.dot(&self.mu);
        v = &v - &(&self.mu * proj);

        // Normalize to unit length
        let norm = v.dot(&v).sqrt();
        if norm > 1e-10 {
            v /= norm;
        }

        v
    }

    /// Sample from standard normal distribution using Box-Muller transform
    fn sample_standard_normal(&self, rng: &mut impl scirs2_core::random::Rng) -> f64 {
        let u1 = rng.gen::<f64>();
        let u2 = rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// von Mises-Fisher Mixture Model
///
/// A mixture of von Mises-Fisher distributions for clustering directional data.
/// This is particularly useful for data that lies on the unit hypersphere,
/// such as text document vectors, gene expression data, or geographical directions.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use sklears_mixture::VonMisesFisherMixture;
/// use sklears_core::traits::{Predict, Fit};
///
/// // Create some directional data (unit vectors)
/// let X = array![
///     [1.0, 0.0],    // East
///     [0.9, 0.1],    // Nearly east
///     [0.0, 1.0],    // North  
///     [0.1, 0.9],    // Nearly north
/// ];
///
/// let vmf_mixture = VonMisesFisherMixture::new()
///     .n_components(2)
///     .max_iter(100);
///
/// let fitted = vmf_mixture.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct VonMisesFisherMixture<S = Untrained> {
    state: S,
    /// Number of mixture components
    pub n_components: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Number of random initializations
    pub n_init: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Regularization parameter for concentration
    pub reg_kappa: f64,
}

impl VonMisesFisherMixture<Untrained> {
    /// Create a new von Mises-Fisher mixture model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 1,
            tol: 1e-3,
            max_iter: 100,
            n_init: 1,
            random_state: None,
            reg_kappa: 1e-6,
        }
    }

    /// Set the number of mixture components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the number of random initializations
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the regularization parameter for concentration
    pub fn reg_kappa(mut self, reg_kappa: f64) -> Self {
        self.reg_kappa = reg_kappa;
        self
    }

    /// Normalize data to unit vectors
    fn normalize_data(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut X_normalized = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let row = X.row(i);
            let norm = row.dot(&row).sqrt();

            if norm < 1e-10 {
                return Err(SklearsError::InvalidInput(format!(
                    "Sample {} has zero norm and cannot be normalized",
                    i
                )));
            }

            X_normalized.row_mut(i).assign(&(&row / norm));
        }

        Ok(X_normalized)
    }

    /// Fit a single model with random initialization
    fn fit_single(
        &self,
        X: &ArrayView2<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<VonMisesFisherMixtureFitted> {
        let (n_samples, n_features) = X.dim();

        // Initialize parameters
        let mut weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);
        let mut mean_directions = self.initialize_mean_directions(X, rng);
        let mut concentrations = Array1::from_elem(self.n_components, 1.0);

        let mut prev_log_likelihood = f64::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // E-step: compute responsibilities
            let responsibilities =
                self.compute_responsibilities(X, &weights, &mean_directions, &concentrations)?;

            // M-step: update parameters
            self.update_parameters(
                X,
                &responsibilities,
                &mut weights,
                &mut mean_directions,
                &mut concentrations,
            )?;

            // Compute log-likelihood
            let log_likelihood =
                self.compute_log_likelihood(X, &weights, &mean_directions, &concentrations)?;

            // Check convergence
            if (log_likelihood - prev_log_likelihood).abs() < self.tol {
                converged = true;
                break;
            }

            prev_log_likelihood = log_likelihood;
        }

        let log_likelihood =
            self.compute_log_likelihood(X, &weights, &mean_directions, &concentrations)?;

        // Compute information criteria
        let n_params = self.n_components - 1 + // weights (sum to 1)
                       self.n_components * n_features + // mean directions
                       self.n_components; // concentrations

        let bic = ModelSelection::bic(log_likelihood, n_params, n_samples);
        let aic = ModelSelection::aic(log_likelihood, n_params);

        Ok(VonMisesFisherMixtureFitted {
            weights: weights.clone(),
            mean_directions: mean_directions.clone(),
            concentrations: concentrations.clone(),
            n_iter,
            converged,
            log_likelihood,
            bic,
            aic,
            n_components: self.n_components,
            n_features,
        })
    }

    /// Initialize mean directions using k-means++ style initialization
    fn initialize_mean_directions(
        &self,
        X: &ArrayView2<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> Array2<f64> {
        let (n_samples, n_features) = X.dim();
        let mut mean_directions = Array2::zeros((self.n_components, n_features));

        // Choose first center randomly
        let first_idx = rng.gen_range(0..n_samples);
        mean_directions.row_mut(0).assign(&X.row(first_idx));

        // Choose remaining centers using k-means++ style selection
        for k in 1..self.n_components {
            let mut distances = Array1::zeros(n_samples);

            for i in 0..n_samples {
                let mut min_dist = f64::INFINITY;
                for j in 0..k {
                    // Use 1 - cosine similarity as distance for directional data
                    let cosine_sim = X.row(i).dot(&mean_directions.row(j));
                    let dist = 1.0 - cosine_sim;
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                distances[i] = min_dist;
            }

            // Choose next center with probability proportional to squared distance
            let total_dist: f64 = distances.iter().map(|&d| d * d).sum();
            let mut cumulative = 0.0;
            let target = rng.gen::<f64>() * total_dist;

            for i in 0..n_samples {
                cumulative += distances[i] * distances[i];
                if cumulative >= target {
                    mean_directions.row_mut(k).assign(&X.row(i));
                    break;
                }
            }
        }

        mean_directions
    }

    /// Compute responsibilities (posterior probabilities)
    fn compute_responsibilities(
        &self,
        X: &ArrayView2<f64>,
        weights: &Array1<f64>,
        mean_directions: &Array2<f64>,
        concentrations: &Array1<f64>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, _) = X.dim();
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let x_i = X.row(i);
            let mut log_probs = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                let mu_k = mean_directions.row(k).to_owned();
                let vmf = VonMisesFisher::new(mu_k, concentrations[k])?;
                let x_reshaped = x_i.to_owned().insert_axis(Axis(0));
                log_probs[k] = weights[k].ln() + vmf.log_pdf(&x_reshaped.view())[0];
            }

            // Use log-sum-exp trick for numerical stability
            let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = max_log_prob
                + log_probs
                    .iter()
                    .map(|&x| (x - max_log_prob).exp())
                    .sum::<f64>()
                    .ln();

            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_sum).exp();
            }
        }

        Ok(responsibilities)
    }

    /// Update parameters in M-step
    fn update_parameters(
        &self,
        X: &ArrayView2<f64>,
        responsibilities: &Array2<f64>,
        weights: &mut Array1<f64>,
        mean_directions: &mut Array2<f64>,
        concentrations: &mut Array1<f64>,
    ) -> SklResult<()> {
        let (n_samples, n_features) = X.dim();

        for k in 0..self.n_components {
            let r_k = responsibilities.column(k);
            let n_k: f64 = r_k.sum();

            // Update weight
            weights[k] = n_k / n_samples as f64;

            // Update mean direction
            let mut mean_direction = Array1::zeros(n_features);
            for i in 0..n_samples {
                mean_direction += &(X.row(i).to_owned() * r_k[i]);
            }
            mean_direction /= n_k;

            // Normalize mean direction
            let norm = mean_direction.dot(&mean_direction).sqrt();
            if norm > 1e-10 {
                mean_direction /= norm;
            }
            mean_directions.row_mut(k).assign(&mean_direction);

            // Update concentration parameter using approximation
            let r_bar = norm / n_k; // Mean resultant length
            concentrations[k] = self.estimate_concentration(r_bar, n_features) + self.reg_kappa;
        }

        Ok(())
    }

    /// Estimate concentration parameter from mean resultant length
    fn estimate_concentration(&self, r_bar: f64, d: usize) -> f64 {
        if r_bar < 1e-10 {
            return 0.0;
        }

        // Use Banerjee et al. approximation for concentration parameter
        let d_f = d as f64;
        let numerator = r_bar * (d_f - r_bar * r_bar);
        let denominator = 1.0 - r_bar * r_bar;

        if denominator < 1e-10 {
            return 100.0; // Large concentration for very concentrated data
        }

        (numerator / denominator).clamp(0.0, 100.0) // Clamp to reasonable range
    }

    /// Compute total log-likelihood
    fn compute_log_likelihood(
        &self,
        X: &ArrayView2<f64>,
        weights: &Array1<f64>,
        mean_directions: &Array2<f64>,
        concentrations: &Array1<f64>,
    ) -> SklResult<f64> {
        let n_samples = X.nrows();
        let mut total_log_likelihood = 0.0;

        for i in 0..n_samples {
            let x_i = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;

            for k in 0..self.n_components {
                let mu_k = mean_directions.row(k).to_owned();
                let vmf = VonMisesFisher::new(mu_k, concentrations[k])?;
                let x_reshaped = x_i.to_owned().insert_axis(Axis(0));
                let log_prob = weights[k].ln() + vmf.log_pdf(&x_reshaped.view())[0];

                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            total_log_likelihood += log_prob_sum;
        }

        Ok(total_log_likelihood)
    }
}

impl Default for VonMisesFisherMixture<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for VonMisesFisherMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for VonMisesFisherMixture<Untrained> {
    type Fitted = VonMisesFisherMixture<VonMisesFisherMixtureFitted>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples ({}) must be at least the number of components ({})",
                n_samples, self.n_components
            )));
        }

        if n_features < 2 {
            return Err(SklearsError::InvalidInput(
                "von Mises-Fisher distribution requires at least 2 dimensions".to_string(),
            ));
        }

        // Normalize input data to unit vectors
        let X_normalized = self.normalize_data(&X.view())?;

        let mut best_model = None;
        let mut best_log_likelihood = f64::NEG_INFINITY;

        let mut rng = match self.random_state {
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
        };

        for _init in 0..self.n_init {
            let result = self.fit_single(&X_normalized.view(), &mut rng)?;
            if result.log_likelihood > best_log_likelihood {
                best_log_likelihood = result.log_likelihood;
                best_model = Some(result);
            }
        }

        let fitted_state = best_model.unwrap();

        Ok(VonMisesFisherMixture {
            state: fitted_state,
            n_components: self.n_components,
            tol: self.tol,
            max_iter: self.max_iter,
            n_init: self.n_init,
            random_state: self.random_state,
            reg_kappa: self.reg_kappa,
        })
    }
}

/// Fitted von Mises-Fisher mixture model
#[derive(Debug, Clone)]
pub struct VonMisesFisherMixtureFitted {
    /// Mixture weights
    weights: Array1<f64>,
    /// Mean directions for each component
    mean_directions: Array2<f64>,
    /// Concentration parameters for each component
    concentrations: Array1<f64>,
    /// Number of iterations performed
    n_iter: usize,
    /// Whether the algorithm converged
    converged: bool,
    /// Final log-likelihood
    log_likelihood: f64,
    /// Bayesian Information Criterion
    bic: f64,
    /// Akaike Information Criterion
    aic: f64,
    /// Number of components
    n_components: usize,
    /// Number of features
    n_features: usize,
}

impl VonMisesFisherMixtureFitted {
    /// Get the mixture weights
    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    /// Get the mean directions
    pub fn mean_directions(&self) -> &Array2<f64> {
        &self.mean_directions
    }

    /// Get the concentration parameters
    pub fn concentrations(&self) -> &Array1<f64> {
        &self.concentrations
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Check if the algorithm converged
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Get the log-likelihood
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Get the BIC score
    pub fn bic(&self) -> f64 {
        self.bic
    }

    /// Get the AIC score
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// Normalize data to unit vectors
    fn normalize_data(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        let mut X_normalized = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let row = X.row(i);
            let norm = row.dot(&row).sqrt();

            if norm < 1e-10 {
                return Err(SklearsError::InvalidInput(format!(
                    "Sample {} has zero norm and cannot be normalized",
                    i
                )));
            }

            X_normalized.row_mut(i).assign(&(&row / norm));
        }

        Ok(X_normalized)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for VonMisesFisherMixture<VonMisesFisherMixtureFitted>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let proba = self.predict_proba(&X.view())?;
        let mut labels = Array1::zeros(X.nrows());

        for i in 0..X.nrows() {
            let mut max_prob = 0.0;
            let mut best_component = 0;

            for k in 0..self.n_components {
                if proba[[i, k]] > max_prob {
                    max_prob = proba[[i, k]];
                    best_component = k;
                }
            }

            labels[i] = best_component as i32;
        }

        Ok(labels)
    }
}

impl VonMisesFisherMixture<VonMisesFisherMixtureFitted> {
    /// Predict class probabilities
    #[allow(non_snake_case)]
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        // Normalize input data
        let X_normalized = self.state.normalize_data(&X.view())?;

        let n_samples = X_normalized.nrows();
        let mut probabilities = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let x_i = X_normalized.row(i);
            let mut log_probs = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                let mu_k = self.state.mean_directions.row(k).to_owned();
                let vmf = VonMisesFisher::new(mu_k, self.state.concentrations[k])?;
                let x_reshaped = x_i.to_owned().insert_axis(Axis(0));
                log_probs[k] = self.state.weights[k].ln() + vmf.log_pdf(&x_reshaped.view())[0];
            }

            // Use log-sum-exp trick for numerical stability
            let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = max_log_prob
                + log_probs
                    .iter()
                    .map(|&x| (x - max_log_prob).exp())
                    .sum::<f64>()
                    .ln();

            for k in 0..self.n_components {
                probabilities[[i, k]] = (log_probs[k] - log_sum).exp();
            }
        }

        Ok(probabilities)
    }

    /// Compute log-likelihood for samples
    #[allow(non_snake_case)]
    pub fn score(&self, X: &ArrayView2<'_, Float>) -> SklResult<f64> {
        let X = X.to_owned();
        let X_normalized = self.state.normalize_data(&X.view())?;
        let n_samples = X_normalized.nrows();
        let mut total_log_likelihood = 0.0;

        for i in 0..n_samples {
            let x_i = X_normalized.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;

            for k in 0..self.n_components {
                let mu_k = self.state.mean_directions.row(k).to_owned();
                let vmf = VonMisesFisher::new(mu_k, self.state.concentrations[k])?;
                let x_reshaped = x_i.to_owned().insert_axis(Axis(0));
                let log_prob = self.state.weights[k].ln() + vmf.log_pdf(&x_reshaped.view())[0];

                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            total_log_likelihood += log_prob_sum;
        }

        Ok(total_log_likelihood)
    }

    /// Sample from the fitted mixture model
    pub fn sample(&self, n_samples: usize, random_state: Option<u64>) -> SklResult<Array2<f64>> {
        let mut rng = match random_state {
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
        };

        let mut samples = Array2::zeros((n_samples, self.state.n_features));

        for i in 0..n_samples {
            // Choose component based on weights
            let u: f64 = rng.gen();
            let mut cumulative = 0.0;
            let mut chosen_component = 0;

            for k in 0..self.n_components {
                cumulative += self.state.weights[k];
                if u <= cumulative {
                    chosen_component = k;
                    break;
                }
            }

            // Sample from chosen component
            let mu_k = self.state.mean_directions.row(chosen_component).to_owned();
            let vmf = VonMisesFisher::new(mu_k, self.state.concentrations[chosen_component])?;
            let sample = vmf.sample_one(&mut rng);
            samples.row_mut(i).assign(&sample);
        }

        Ok(samples)
    }
}

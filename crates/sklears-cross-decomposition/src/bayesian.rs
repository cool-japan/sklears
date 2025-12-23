//! Bayesian approaches for cross-decomposition
//!
//! This module implements Bayesian versions of canonical correlation analysis
//! and partial least squares methods that provide uncertainty quantification
//! and automatic hyperparameter selection.

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{
    thread_rng, Distribution, RandGamma as Gamma, RandNormal as Normal, Random, Rng, SeedableRng,
};
use sklears_core::error::SklearsError;
use std::collections::HashMap;

/// Bayesian Canonical Correlation Analysis
///
/// A Bayesian approach to CCA that treats canonical weights as random variables
/// with prior distributions and performs posterior inference using MCMC sampling.
/// Provides uncertainty quantification for canonical correlations and weights.
///
/// # Mathematical Background
///
/// The Bayesian CCA model assumes:
/// - Canonical weights W_x, W_y follow multivariate normal priors
/// - Noise precision follows Gamma prior
/// - Posterior inference via Gibbs sampling
///
/// # Examples
///
/// ```rust
/// use sklears_cross_decomposition::BayesianCCA;
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];
///
/// let mut bcca = BayesianCCA::new(1)
///     .n_samples(1000)
///     .burn_in(200);
///
/// let result = bcca.fit(&x, &y).unwrap();
/// let correlations = result.canonical_correlations();
/// let uncertainty = result.correlation_credible_intervals(0.95);
/// ```
#[derive(Debug, Clone)]
pub struct BayesianCCA {
    n_components: usize,
    n_samples: usize,
    burn_in: usize,
    thin: usize,
    prior_precision: f64,
    noise_shape: f64,
    noise_rate: f64,
    random_state: Option<u64>,
}

impl BayesianCCA {
    /// Create a new BayesianCCA instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            n_samples: 2000,
            burn_in: 500,
            thin: 1,
            prior_precision: 1.0,
            noise_shape: 1.0,
            noise_rate: 1.0,
            random_state: None,
        }
    }

    /// Set the number of MCMC samples
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set the burn-in period
    pub fn burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Set the thinning interval
    pub fn thin(mut self, thin: usize) -> Self {
        self.thin = thin;
        self
    }

    /// Set the prior precision for canonical weights
    pub fn prior_precision(mut self, precision: f64) -> Self {
        self.prior_precision = precision;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Fit the Bayesian CCA model
    pub fn fit(self, x: &Array2<f64>, y: &Array2<f64>) -> Result<BayesianCCAResults, SklearsError> {
        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if x.nrows() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for Bayesian CCA".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            let mut entropy_rng = thread_rng();
            StdRng::from_rng(&mut entropy_rng)
        };

        let n_samples = x.nrows();
        let p_x = x.ncols();
        let p_y = y.ncols();

        // Center the data
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();
        let x_centered = x - &x_mean.clone().insert_axis(Axis(0));
        let y_centered = y - &y_mean.clone().insert_axis(Axis(0));

        // Initialize parameters
        let mut wx_samples = Vec::new();
        let mut wy_samples = Vec::new();
        let mut correlations = Vec::new();
        let mut noise_precision = 1.0;

        // Initialize canonical weights randomly
        let normal = Normal::new(0.0, 1.0 / self.prior_precision.sqrt()).unwrap();
        let mut wx = Array2::zeros((p_x, self.n_components));
        let mut wy = Array2::zeros((p_y, self.n_components));

        for i in 0..p_x {
            for j in 0..self.n_components {
                wx[[i, j]] = normal.sample(&mut rng);
            }
        }

        for i in 0..p_y {
            for j in 0..self.n_components {
                wy[[i, j]] = normal.sample(&mut rng);
            }
        }

        // MCMC sampling
        for iter in 0..self.n_samples + self.burn_in {
            // Sample canonical weights Wx
            self.sample_canonical_weights(
                &mut wx,
                &x_centered,
                &y_centered,
                &wy,
                noise_precision,
                &mut rng,
            )?;

            // Sample canonical weights Wy
            self.sample_canonical_weights(
                &mut wy,
                &y_centered,
                &x_centered,
                &wx,
                noise_precision,
                &mut rng,
            )?;

            // Sample noise precision
            noise_precision =
                self.sample_noise_precision(&x_centered, &y_centered, &wx, &wy, &mut rng)?;

            // Store samples after burn-in
            if iter >= self.burn_in && (iter - self.burn_in) % self.thin == 0 {
                wx_samples.push(wx.clone());
                wy_samples.push(wy.clone());

                // Compute canonical correlations for this sample
                let x_canonical = x_centered.dot(&wx);
                let y_canonical = y_centered.dot(&wy);
                let mut sample_correlations = Vec::new();

                for k in 0..self.n_components {
                    let x_k = x_canonical.column(k);
                    let y_k = y_canonical.column(k);
                    let corr = self.compute_correlation(&x_k, &y_k);
                    sample_correlations.push(corr);
                }

                correlations.push(sample_correlations);
            }
        }

        Ok(BayesianCCAResults {
            wx_samples,
            wy_samples,
            correlations: correlations.clone(),
            x_mean,
            y_mean,
            n_components: self.n_components,
            n_mcmc_samples: correlations.len(),
        })
    }

    fn sample_canonical_weights(
        &self,
        weights: &mut Array2<f64>,
        x: &Array2<f64>,
        y: &Array2<f64>,
        other_weights: &Array2<f64>,
        noise_precision: f64,
        rng: &mut impl Rng,
    ) -> Result<(), SklearsError> {
        let n = x.nrows();
        let p = x.ncols();

        // Compute posterior precision matrix
        let xtx = x.t().dot(x);
        let posterior_precision = self.prior_precision * Array2::eye(p) + noise_precision * xtx;

        // Compute posterior mean
        let y_canonical = y.dot(other_weights);
        let posterior_mean_numerator = noise_precision * x.t().dot(&y_canonical);

        // Sample from multivariate normal (simplified approach)
        for k in 0..self.n_components {
            let mean_k = posterior_mean_numerator.column(k).to_owned();
            let normal = Normal::new(0.0, 1.0).unwrap();

            for i in 0..p {
                let noise_sample = normal.sample(rng);
                let precision_inv = 1.0 / (posterior_precision[[i, i]] + 1e-8);
                weights[[i, k]] = mean_k[i] * precision_inv + noise_sample * precision_inv.sqrt();
            }
        }

        Ok(())
    }

    fn sample_noise_precision(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        wx: &Array2<f64>,
        wy: &Array2<f64>,
        rng: &mut impl Rng,
    ) -> Result<f64, SklearsError> {
        let n = x.nrows() as f64;
        let x_canonical = x.dot(wx);
        let y_canonical = y.dot(wy);

        let mut sse = 0.0;
        for k in 0..self.n_components {
            let residuals = &x_canonical.column(k) - &y_canonical.column(k);
            sse += residuals.iter().map(|&r| r * r).sum::<f64>();
        }

        let shape = self.noise_shape + n * self.n_components as f64 / 2.0;
        let rate = self.noise_rate + sse / 2.0;

        let gamma_dist = Gamma::new(shape, 1.0 / rate).unwrap();
        Ok(gamma_dist.sample(rng))
    }

    fn compute_correlation(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        if den_x == 0.0 || den_y == 0.0 {
            0.0
        } else {
            num / (den_x * den_y).sqrt()
        }
    }
}

/// Results from Bayesian CCA analysis
#[derive(Debug, Clone)]
pub struct BayesianCCAResults {
    wx_samples: Vec<Array2<f64>>,
    wy_samples: Vec<Array2<f64>>,
    correlations: Vec<Vec<f64>>,
    x_mean: Array1<f64>,
    y_mean: Array1<f64>,
    n_components: usize,
    n_mcmc_samples: usize,
}

impl BayesianCCAResults {
    /// Get posterior mean of canonical correlations
    pub fn canonical_correlations(&self) -> Array1<f64> {
        let mut means = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            let mut sum = 0.0;
            for sample in &self.correlations {
                sum += sample[k];
            }
            means[k] = sum / self.n_mcmc_samples as f64;
        }

        means
    }

    /// Get credible intervals for canonical correlations
    pub fn correlation_credible_intervals(&self, confidence: f64) -> Array2<f64> {
        let alpha = (1.0 - confidence) / 2.0;
        let lower_idx = (alpha * self.n_mcmc_samples as f64) as usize;
        let upper_idx = ((1.0 - alpha) * self.n_mcmc_samples as f64) as usize;

        let mut intervals = Array2::zeros((self.n_components, 2));

        for k in 0..self.n_components {
            let mut values: Vec<f64> = self.correlations.iter().map(|sample| sample[k]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            intervals[[k, 0]] = values[lower_idx];
            intervals[[k, 1]] = values[upper_idx];
        }

        intervals
    }

    /// Get posterior mean of X canonical weights
    pub fn x_weights(&self) -> Array2<f64> {
        let p = self.wx_samples[0].nrows();
        let mut mean_weights = Array2::zeros((p, self.n_components));

        for sample in &self.wx_samples {
            mean_weights = mean_weights + sample;
        }

        mean_weights / self.n_mcmc_samples as f64
    }

    /// Get posterior mean of Y canonical weights
    pub fn y_weights(&self) -> Array2<f64> {
        let p = self.wy_samples[0].nrows();
        let mut mean_weights = Array2::zeros((p, self.n_components));

        for sample in &self.wy_samples {
            mean_weights = mean_weights + sample;
        }

        mean_weights / self.n_mcmc_samples as f64
    }

    /// Transform X data using posterior mean weights
    pub fn transform_x(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let x_centered = x - &self.x_mean.clone().insert_axis(Axis(0));
        let weights = self.x_weights();
        Ok(x_centered.dot(&weights))
    }

    /// Transform Y data using posterior mean weights
    pub fn transform_y(&self, y: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let y_centered = y - &self.y_mean.clone().insert_axis(Axis(0));
        let weights = self.y_weights();
        Ok(y_centered.dot(&weights))
    }

    /// Get effective sample size for convergence diagnostics
    pub fn effective_sample_size(&self) -> Array1<f64> {
        let mut ess = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            let values: Vec<f64> = self.correlations.iter().map(|sample| sample[k]).collect();

            // Simple ESS estimation using autocorrelation at lag 1
            let mut autocorr = 0.0;
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let mut var = 0.0;

            for i in 0..values.len() {
                var += (values[i] - mean).powi(2);
            }
            var /= values.len() as f64;

            if var > 0.0 {
                for i in 1..values.len() {
                    autocorr += (values[i] - mean) * (values[i - 1] - mean);
                }
                autocorr /= (values.len() - 1) as f64 * var;

                // ESS approximation
                ess[k] = values.len() as f64 / (1.0 + 2.0 * autocorr.max(0.0));
            } else {
                ess[k] = values.len() as f64;
            }
        }

        ess
    }

    /// Get MCMC diagnostics
    pub fn mcmc_diagnostics(&self) -> HashMap<String, f64> {
        let mut diagnostics = HashMap::new();

        let ess = self.effective_sample_size();
        diagnostics.insert(
            "min_ess".to_string(),
            ess.iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .clone(),
        );
        diagnostics.insert("mean_ess".to_string(), ess.mean().unwrap());
        diagnostics.insert("n_samples".to_string(), self.n_mcmc_samples as f64);

        diagnostics
    }
}

/// Variational Bayesian Partial Least Squares
///
/// A variational Bayesian approach to PLS that uses mean-field variational inference
/// to approximate the posterior distribution. More computationally efficient than
/// MCMC-based approaches while still providing uncertainty quantification.
///
/// # Mathematical Background
///
/// The Variational PLS model assumes:
/// - Loading weights follow multivariate normal distributions with ARD priors
/// - Noise precision follows Gamma distribution
/// - Mean-field variational approximation for posterior inference
/// - ELBO optimization for parameter updates
///
/// # Examples
///
/// ```rust
/// use sklears_cross_decomposition::VariationalPLS;
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// let y = array![[1.5], [2.5], [3.5]];
///
/// let mut vpls = VariationalPLS::new(1)
///     .max_iter(100)
///     .tolerance(1e-6)
///     .ard_alpha(1e-6);
///
/// let result = vpls.fit(&x, &y).unwrap();
/// let loadings = result.x_loadings();
/// let uncertainty = result.loading_standard_deviations();
/// ```
#[derive(Debug, Clone)]
pub struct VariationalPLS {
    n_components: usize,
    max_iter: usize,
    tolerance: f64,
    ard_alpha: f64,
    ard_beta: f64,
    noise_alpha: f64,
    noise_beta: f64,
    random_state: Option<u64>,
}

impl VariationalPLS {
    /// Create a new VariationalPLS instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 300,
            tolerance: 1e-6,
            ard_alpha: 1e-6,
            ard_beta: 1e-6,
            noise_alpha: 1e-6,
            noise_beta: 1e-6,
            random_state: None,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set ARD alpha hyperparameter
    pub fn ard_alpha(mut self, alpha: f64) -> Self {
        self.ard_alpha = alpha;
        self
    }

    /// Set ARD beta hyperparameter
    pub fn ard_beta(mut self, beta: f64) -> Self {
        self.ard_beta = beta;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Fit the Variational PLS model
    pub fn fit(
        self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<VariationalPLSResults, SklearsError> {
        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if x.nrows() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for Variational PLS".to_string(),
            ));
        }

        let n_samples = x.nrows() as f64;
        let n_features_x = x.ncols();
        let n_features_y = y.ncols();

        // Center the data
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();
        let x_centered = x - &x_mean.clone().insert_axis(Axis(0));
        let y_centered = y - &y_mean.clone().insert_axis(Axis(0));

        // Initialize variational parameters
        let mut mean_wx = Array2::zeros((n_features_x, self.n_components));
        let mut cov_wx = Array2::eye(n_features_x) * 1.0;
        let mut mean_wy = Array2::zeros((n_features_y, self.n_components));
        let mut cov_wy = Array2::eye(n_features_y) * 1.0;

        // ARD precision parameters
        let mut alpha_mean = Array1::ones(n_features_x);
        let mut alpha_shape = self.ard_alpha;
        let mut alpha_rate = Array1::ones(n_features_x) * self.ard_beta;

        // Noise precision parameters
        let mut tau_shape = self.noise_alpha + n_samples / 2.0;
        let mut tau_rate = self.noise_beta;

        let mut prev_elbo = f64::NEG_INFINITY;
        let mut elbo_history = Vec::new();

        // Variational EM iterations
        for iter in 0..self.max_iter {
            // E-step: Update posterior distributions

            // Update X weights to maximize correlation with Y projections
            self.update_pls_weight_posterior(
                &x_centered,
                &y_centered,
                &mean_wy,
                &mut mean_wx,
                &mut cov_wx,
                &alpha_mean,
                tau_shape / tau_rate,
            )?;

            // Update Y weights to maximize correlation with X projections
            self.update_pls_weight_posterior(
                &y_centered,
                &x_centered,
                &mean_wx,
                &mut mean_wy,
                &mut cov_wy,
                &Array1::ones(n_features_y),
                tau_shape / tau_rate,
            )?;

            // M-step: Update hyperparameters
            self.update_ard_parameters(
                &mean_wx,
                &cov_wx,
                &mut alpha_mean,
                &mut alpha_shape,
                &mut alpha_rate,
            );

            self.update_noise_parameters(
                &x_centered,
                &y_centered,
                &mean_wx,
                &cov_wx,
                &mean_wy,
                &cov_wy,
                &mut tau_shape,
                &mut tau_rate,
            )?;

            // Compute ELBO
            let elbo = self.compute_elbo(
                &x_centered,
                &y_centered,
                &mean_wx,
                &cov_wx,
                &mean_wy,
                &cov_wy,
                &alpha_mean,
                alpha_shape,
                &alpha_rate,
                tau_shape,
                tau_rate,
            );

            elbo_history.push(elbo);

            // Check convergence
            if iter > 0 && (elbo - prev_elbo).abs() < self.tolerance {
                break;
            }

            prev_elbo = elbo;
        }

        Ok(VariationalPLSResults {
            mean_wx,
            cov_wx,
            mean_wy,
            cov_wy,
            alpha_mean,
            tau_mean: tau_shape / tau_rate,
            x_mean,
            y_mean,
            n_components: self.n_components,
            elbo_history,
        })
    }

    fn update_pls_weight_posterior(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        other_weights: &Array2<f64>,
        mean_w: &mut Array2<f64>,
        cov_w: &mut Array2<f64>,
        precision_prior: &Array1<f64>,
        noise_precision: f64,
    ) -> Result<(), SklearsError> {
        let n_features = x.ncols();

        // Compute Y projections using current other weights
        let y_projected = y.dot(other_weights);

        // For each component, update weights independently
        for k in 0..self.n_components {
            let y_k = y_projected.column(k);

            // Compute posterior precision matrix for this component
            let xtx = x.t().dot(x);
            let prior_precision = Array2::from_diag(precision_prior);
            let posterior_precision = prior_precision + xtx * noise_precision;

            // Compute posterior covariance (simplified diagonal approximation)
            let mut posterior_cov = Array2::zeros((n_features, n_features));
            for i in 0..n_features {
                posterior_cov[[i, i]] = 1.0 / (posterior_precision[[i, i]] + 1e-8);
            }

            // Compute posterior mean for this component
            let xty_k = x.t().dot(&y_k.insert_axis(Axis(1)));
            let mean_k = posterior_cov.dot(&(xty_k * noise_precision));

            // Update the k-th column of mean_w
            for i in 0..n_features {
                mean_w[[i, k]] = mean_k[[i, 0]];
                cov_w[[i, i]] = posterior_cov[[i, i]];
            }
        }

        Ok(())
    }

    fn update_ard_parameters(
        &self,
        mean_w: &Array2<f64>,
        cov_w: &Array2<f64>,
        alpha_mean: &mut Array1<f64>,
        alpha_shape: &mut f64,
        alpha_rate: &mut Array1<f64>,
    ) {
        *alpha_shape = self.ard_alpha + self.n_components as f64 / 2.0;

        for i in 0..mean_w.nrows() {
            let mut expected_w_squared = 0.0;
            for k in 0..self.n_components {
                expected_w_squared += mean_w[[i, k]].powi(2) + cov_w[[i, i]];
            }

            alpha_rate[i] = self.ard_beta + expected_w_squared / 2.0;
            alpha_mean[i] = *alpha_shape / alpha_rate[i];
        }
    }

    fn update_noise_parameters(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        mean_wx: &Array2<f64>,
        cov_wx: &Array2<f64>,
        mean_wy: &Array2<f64>,
        cov_wy: &Array2<f64>,
        tau_shape: &mut f64,
        tau_rate: &mut f64,
    ) -> Result<(), SklearsError> {
        let n_samples = x.nrows() as f64;

        // Compute expected squared residuals
        let x_proj = x.dot(mean_wx);
        let y_proj = y.dot(mean_wy);

        let mut expected_sse = 0.0;
        for i in 0..x.nrows() {
            for k in 0..self.n_components {
                let residual = x_proj[[i, k]] - y_proj[[i, k]];
                expected_sse += residual.powi(2);
            }
        }

        // Add uncertainty terms
        for k in 0..self.n_components {
            expected_sse += n_samples * (cov_wx[[k, k]] + cov_wy[[k, k]]);
        }

        *tau_shape = self.noise_alpha + n_samples * self.n_components as f64 / 2.0;
        *tau_rate = self.noise_beta + expected_sse / 2.0;

        Ok(())
    }

    fn compute_elbo(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        mean_wx: &Array2<f64>,
        cov_wx: &Array2<f64>,
        mean_wy: &Array2<f64>,
        cov_wy: &Array2<f64>,
        alpha_mean: &Array1<f64>,
        alpha_shape: f64,
        alpha_rate: &Array1<f64>,
        tau_shape: f64,
        tau_rate: f64,
    ) -> f64 {
        let mut elbo = 0.0;

        // Log likelihood term
        let x_proj = x.dot(mean_wx);
        let y_proj = y.dot(mean_wy);
        let mut sse = 0.0;

        for i in 0..x.nrows() {
            for k in 0..self.n_components {
                let residual = x_proj[[i, k]] - y_proj[[i, k]];
                sse += residual.powi(2);
            }
        }

        let n_samples = x.nrows() as f64;
        let tau_mean = tau_shape / tau_rate;
        elbo += 0.5
            * n_samples
            * self.n_components as f64
            * (tau_mean.ln() - (2.0 * std::f64::consts::PI).ln());
        elbo -= 0.5 * tau_mean * sse;

        // Prior terms (simplified)
        for i in 0..mean_wx.nrows() {
            for k in 0..self.n_components {
                elbo -= 0.5 * alpha_mean[i] * mean_wx[[i, k]].powi(2);
            }
        }

        // Entropy terms (simplified approximation)
        elbo += 0.5 * cov_wx.diag().iter().map(|&x| x.ln()).sum::<f64>();
        elbo += 0.5 * cov_wy.diag().iter().map(|&x| x.ln()).sum::<f64>();

        elbo
    }

    fn matrix_inverse(&self, matrix: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        // Simple diagonal approximation for efficiency
        let mut result = Array2::zeros(matrix.raw_dim());
        for i in 0..matrix.nrows() {
            if matrix[[i, i]].abs() > 1e-12 {
                result[[i, i]] = 1.0 / matrix[[i, i]];
            } else {
                result[[i, i]] = 1e12; // Large value for near-singular case
            }
        }
        Ok(result)
    }
}

/// Results from Variational PLS analysis
#[derive(Debug, Clone)]
pub struct VariationalPLSResults {
    mean_wx: Array2<f64>,
    cov_wx: Array2<f64>,
    mean_wy: Array2<f64>,
    cov_wy: Array2<f64>,
    alpha_mean: Array1<f64>,
    tau_mean: f64,
    x_mean: Array1<f64>,
    y_mean: Array1<f64>,
    n_components: usize,
    elbo_history: Vec<f64>,
}

impl VariationalPLSResults {
    /// Get posterior mean of X loadings
    pub fn x_loadings(&self) -> Array2<f64> {
        self.mean_wx.clone()
    }

    /// Get posterior mean of Y loadings
    pub fn y_loadings(&self) -> Array2<f64> {
        self.mean_wy.clone()
    }

    /// Get standard deviations of loadings (uncertainty quantification)
    pub fn loading_standard_deviations(&self) -> (Array2<f64>, Array2<f64>) {
        let std_wx = self.cov_wx.diag().mapv(|x| x.sqrt()).insert_axis(Axis(1));
        let std_wy = self.cov_wy.diag().mapv(|x| x.sqrt()).insert_axis(Axis(1));
        (std_wx, std_wy)
    }

    /// Get ARD precision parameters (feature relevance)
    pub fn feature_relevance(&self) -> Array1<f64> {
        self.alpha_mean.clone()
    }

    /// Get noise precision
    pub fn noise_precision(&self) -> f64 {
        self.tau_mean
    }

    /// Get ELBO convergence history
    pub fn elbo_history(&self) -> &Vec<f64> {
        &self.elbo_history
    }

    /// Transform X data using posterior mean loadings
    pub fn transform_x(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let x_centered = x - &self.x_mean.clone().insert_axis(Axis(0));
        Ok(x_centered.dot(&self.mean_wx))
    }

    /// Transform Y data using posterior mean loadings  
    pub fn transform_y(&self, y: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let y_centered = y - &self.y_mean.clone().insert_axis(Axis(0));
        Ok(y_centered.dot(&self.mean_wy))
    }

    /// Predict Y from X using the learned model
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let x_scores = self.transform_x(x)?;

        // Simple linear prediction from X scores to Y scores
        let y_scores = x_scores.clone(); // Simplified - in practice would learn this mapping

        // Transform back to Y space (inverse transform)
        let y_pred = y_scores.dot(&self.mean_wy.t()) + &self.y_mean.clone().insert_axis(Axis(0));
        Ok(y_pred)
    }

    /// Get model convergence status
    pub fn has_converged(&self) -> bool {
        if self.elbo_history.len() < 2 {
            return false;
        }

        let last_two = &self.elbo_history[self.elbo_history.len() - 2..];
        (last_two[1] - last_two[0]).abs() < 1e-6
    }
}

/// Hierarchical Bayesian Canonical Correlation Analysis
///
/// A hierarchical extension of Bayesian CCA that can handle grouped data structures
/// where observations are nested within groups (e.g., subjects, batches, conditions).
/// Uses hierarchical priors to share information across groups while allowing
/// group-specific effects.
///
/// # Mathematical Background
///
/// The Hierarchical Bayesian CCA model assumes:
/// - Group-level canonical weights follow hierarchical normal priors
/// - Hyperpriors on the group-level parameters  
/// - Within-group variation and between-group variation are modeled separately
/// - MCMC sampling for posterior inference across the hierarchy
///
/// # Examples
///
/// ```rust
/// use sklears_cross_decomposition::HierarchicalBayesianCCA;
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
/// let groups = vec![0, 0, 1, 1]; // Two groups with 2 observations each
///
/// let mut hbcca = HierarchicalBayesianCCA::new(1)
///     .n_samples(1000)
///     .burn_in(200);
///
/// let result = hbcca.fit(&x, &y, &groups).unwrap();
/// let group_effects = result.group_effects();
/// let population_effects = result.population_effects();
/// ```
#[derive(Debug, Clone)]
pub struct HierarchicalBayesianCCA {
    n_components: usize,
    n_samples: usize,
    burn_in: usize,
    thin: usize,
    population_precision: f64,
    group_precision: f64,
    noise_shape: f64,
    noise_rate: f64,
    random_state: Option<u64>,
}

impl HierarchicalBayesianCCA {
    /// Create a new HierarchicalBayesianCCA instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            n_samples: 2000,
            burn_in: 500,
            thin: 1,
            population_precision: 1.0,
            group_precision: 1.0,
            noise_shape: 1.0,
            noise_rate: 1.0,
            random_state: None,
        }
    }

    /// Set the number of MCMC samples
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set the burn-in period
    pub fn burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Set the thinning interval
    pub fn thin(mut self, thin: usize) -> Self {
        self.thin = thin;
        self
    }

    /// Set the population-level precision
    pub fn population_precision(mut self, precision: f64) -> Self {
        self.population_precision = precision;
        self
    }

    /// Set the group-level precision
    pub fn group_precision(mut self, precision: f64) -> Self {
        self.group_precision = precision;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Fit the Hierarchical Bayesian CCA model
    pub fn fit(
        self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        groups: &[usize],
    ) -> Result<HierarchicalBayesianCCAResults, SklearsError> {
        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if x.nrows() != groups.len() {
            return Err(SklearsError::InvalidInput(
                "Groups must have same length as number of samples".to_string(),
            ));
        }

        if x.nrows() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for Hierarchical Bayesian CCA".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            let mut entropy_rng = thread_rng();
            StdRng::from_rng(&mut entropy_rng)
        };

        let n_samples = x.nrows();
        let p_x = x.ncols();
        let p_y = y.ncols();
        let n_groups = groups.iter().max().unwrap() + 1;

        // Center the data
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();
        let x_centered = x - &x_mean.clone().insert_axis(Axis(0));
        let y_centered = y - &y_mean.clone().insert_axis(Axis(0));

        // Initialize hierarchical parameters
        let mut population_wx = Array2::zeros((p_x, self.n_components));
        let mut population_wy = Array2::zeros((p_y, self.n_components));
        let mut group_wx = Array3::zeros((n_groups, p_x, self.n_components));
        let mut group_wy = Array3::zeros((n_groups, p_y, self.n_components));

        // Storage for samples
        let mut population_wx_samples = Vec::new();
        let mut population_wy_samples = Vec::new();
        let mut group_wx_samples = Vec::new();
        let mut group_wy_samples = Vec::new();
        let mut correlations = Vec::new();
        let mut noise_precision = 1.0;

        // Initialize parameters randomly
        let normal = Normal::new(0.0, 1.0 / self.population_precision.sqrt()).unwrap();

        // Initialize population effects
        for i in 0..p_x {
            for j in 0..self.n_components {
                population_wx[[i, j]] = normal.sample(&mut rng);
            }
        }
        for i in 0..p_y {
            for j in 0..self.n_components {
                population_wy[[i, j]] = normal.sample(&mut rng);
            }
        }

        // Initialize group effects around population effects
        let group_normal = Normal::new(0.0, 1.0 / self.group_precision.sqrt()).unwrap();
        for g in 0..n_groups {
            for i in 0..p_x {
                for j in 0..self.n_components {
                    group_wx[[g, i, j]] = population_wx[[i, j]] + group_normal.sample(&mut rng);
                }
            }
            for i in 0..p_y {
                for j in 0..self.n_components {
                    group_wy[[g, i, j]] = population_wy[[i, j]] + group_normal.sample(&mut rng);
                }
            }
        }

        // MCMC sampling
        for iter in 0..self.n_samples + self.burn_in {
            // Sample population-level parameters
            self.sample_population_parameters(
                &mut population_wx,
                &mut population_wy,
                &group_wx,
                &group_wy,
                &mut rng,
            )?;

            // Sample group-level parameters
            self.sample_group_parameters(
                &mut group_wx,
                &mut group_wy,
                &population_wx,
                &population_wy,
                &x_centered,
                &y_centered,
                groups,
                noise_precision,
                &mut rng,
            )?;

            // Sample noise precision
            noise_precision = self.sample_hierarchical_noise_precision(
                &x_centered,
                &y_centered,
                &group_wx,
                &group_wy,
                groups,
                &mut rng,
            )?;

            // Store samples after burn-in
            if iter >= self.burn_in && (iter - self.burn_in) % self.thin == 0 {
                population_wx_samples.push(population_wx.clone());
                population_wy_samples.push(population_wy.clone());
                group_wx_samples.push(group_wx.clone());
                group_wy_samples.push(group_wy.clone());

                // Compute canonical correlations using group-specific weights
                let mut sample_correlations = Vec::new();
                for k in 0..self.n_components {
                    let mut total_corr = 0.0;
                    let mut total_weight = 0.0;

                    for g in 0..n_groups {
                        let group_indices: Vec<usize> = groups
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &group)| if group == g { Some(i) } else { None })
                            .collect();

                        if !group_indices.is_empty() {
                            let group_x = group_indices
                                .iter()
                                .map(|&i| x_centered.row(i))
                                .collect::<Vec<_>>();
                            let group_y = group_indices
                                .iter()
                                .map(|&i| y_centered.row(i))
                                .collect::<Vec<_>>();

                            if group_x.len() > 1 {
                                let group_x_mat =
                                    scirs2_core::ndarray::stack(Axis(0), &group_x).unwrap();
                                let group_y_mat =
                                    scirs2_core::ndarray::stack(Axis(0), &group_y).unwrap();

                                let wx_k = group_wx.slice(scirs2_core::ndarray::s![g, .., k]);
                                let wy_k = group_wy.slice(scirs2_core::ndarray::s![g, .., k]);

                                let x_canonical = group_x_mat.dot(&wx_k);
                                let y_canonical = group_y_mat.dot(&wy_k);

                                let corr = self
                                    .compute_correlation(&x_canonical.view(), &y_canonical.view());
                                total_corr += corr * group_x.len() as f64;
                                total_weight += group_x.len() as f64;
                            }
                        }
                    }

                    if total_weight > 0.0 {
                        sample_correlations.push(total_corr / total_weight);
                    } else {
                        sample_correlations.push(0.0);
                    }
                }

                correlations.push(sample_correlations);
            }
        }

        Ok(HierarchicalBayesianCCAResults {
            population_wx_samples,
            population_wy_samples,
            group_wx_samples,
            group_wy_samples,
            correlations: correlations.clone(),
            x_mean,
            y_mean,
            n_components: self.n_components,
            n_groups,
            n_mcmc_samples: correlations.len(),
        })
    }

    fn sample_population_parameters(
        &self,
        population_wx: &mut Array2<f64>,
        population_wy: &mut Array2<f64>,
        group_wx: &Array3<f64>,
        group_wy: &Array3<f64>,
        rng: &mut impl Rng,
    ) -> Result<(), SklearsError> {
        let n_groups = group_wx.shape()[0];

        // Update population X weights
        for i in 0..population_wx.nrows() {
            for k in 0..self.n_components {
                let mut sum = 0.0;
                for g in 0..n_groups {
                    sum += group_wx[[g, i, k]];
                }
                let mean = sum / n_groups as f64;
                let precision = self.population_precision + n_groups as f64 * self.group_precision;
                let variance = 1.0 / precision;

                let normal =
                    Normal::new(mean * self.group_precision / precision, variance.sqrt()).unwrap();
                population_wx[[i, k]] = normal.sample(rng);
            }
        }

        // Update population Y weights
        for i in 0..population_wy.nrows() {
            for k in 0..self.n_components {
                let mut sum = 0.0;
                for g in 0..n_groups {
                    sum += group_wy[[g, i, k]];
                }
                let mean = sum / n_groups as f64;
                let precision = self.population_precision + n_groups as f64 * self.group_precision;
                let variance = 1.0 / precision;

                let normal =
                    Normal::new(mean * self.group_precision / precision, variance.sqrt()).unwrap();
                population_wy[[i, k]] = normal.sample(rng);
            }
        }

        Ok(())
    }

    fn sample_group_parameters(
        &self,
        group_wx: &mut Array3<f64>,
        group_wy: &mut Array3<f64>,
        population_wx: &Array2<f64>,
        population_wy: &Array2<f64>,
        x: &Array2<f64>,
        y: &Array2<f64>,
        groups: &[usize],
        noise_precision: f64,
        rng: &mut impl Rng,
    ) -> Result<(), SklearsError> {
        let n_groups = group_wx.shape()[0];

        for g in 0..n_groups {
            let group_indices: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter_map(|(i, &group)| if group == g { Some(i) } else { None })
                .collect();

            if group_indices.is_empty() {
                continue;
            }

            // Extract group data
            let group_x = group_indices.iter().map(|&i| x.row(i)).collect::<Vec<_>>();
            let group_y = group_indices.iter().map(|&i| y.row(i)).collect::<Vec<_>>();
            let group_x_mat = scirs2_core::ndarray::stack(Axis(0), &group_x).unwrap();
            let group_y_mat = scirs2_core::ndarray::stack(Axis(0), &group_y).unwrap();

            // Sample group X weights
            self.sample_group_weight(
                group_wx.slice_mut(scirs2_core::ndarray::s![g, .., ..]),
                &group_x_mat,
                &group_y_mat,
                population_wx,
                noise_precision,
                rng,
            )?;

            // Sample group Y weights
            self.sample_group_weight(
                group_wy.slice_mut(scirs2_core::ndarray::s![g, .., ..]),
                &group_y_mat,
                &group_x_mat,
                population_wy,
                noise_precision,
                rng,
            )?;
        }

        Ok(())
    }

    fn sample_group_weight(
        &self,
        group_weight: scirs2_core::ndarray::ArrayViewMut2<f64>,
        x: &Array2<f64>,
        y: &Array2<f64>,
        population_weight: &Array2<f64>,
        noise_precision: f64,
        rng: &mut impl Rng,
    ) -> Result<(), SklearsError> {
        let mut group_weight = group_weight;
        let n_obs = x.nrows() as f64;
        let p = x.ncols();

        for k in 0..self.n_components {
            for i in 0..p {
                // Posterior precision
                let precision = self.group_precision + noise_precision * n_obs;
                let variance = 1.0 / precision;

                // Posterior mean components
                let prior_contribution = self.group_precision * population_weight[[i, k]];
                let data_contribution = if n_obs > 0.0 {
                    let x_i = x.column(i);
                    let y_k = y.column(k);
                    noise_precision * x_i.dot(&y_k)
                } else {
                    0.0
                };

                let posterior_mean = (prior_contribution + data_contribution) / precision;

                let normal = Normal::new(posterior_mean, variance.sqrt()).unwrap();
                group_weight[[i, k]] = normal.sample(rng);
            }
        }

        Ok(())
    }

    fn sample_hierarchical_noise_precision(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        group_wx: &Array3<f64>,
        group_wy: &Array3<f64>,
        groups: &[usize],
        rng: &mut impl Rng,
    ) -> Result<f64, SklearsError> {
        let mut total_sse = 0.0;
        let mut total_obs = 0;

        let n_groups = group_wx.shape()[0];

        for g in 0..n_groups {
            let group_indices: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter_map(|(i, &group)| if group == g { Some(i) } else { None })
                .collect();

            if group_indices.is_empty() {
                continue;
            }

            let group_x = group_indices.iter().map(|&i| x.row(i)).collect::<Vec<_>>();
            let group_y = group_indices.iter().map(|&i| y.row(i)).collect::<Vec<_>>();
            let group_x_mat = scirs2_core::ndarray::stack(Axis(0), &group_x).unwrap();
            let group_y_mat = scirs2_core::ndarray::stack(Axis(0), &group_y).unwrap();

            let wx_g = group_wx.slice(scirs2_core::ndarray::s![g, .., ..]);
            let wy_g = group_wy.slice(scirs2_core::ndarray::s![g, .., ..]);

            let x_canonical = group_x_mat.dot(&wx_g);
            let y_canonical = group_y_mat.dot(&wy_g);

            for k in 0..self.n_components {
                let residuals = &x_canonical.column(k) - &y_canonical.column(k);
                total_sse += residuals.iter().map(|&r| r * r).sum::<f64>();
            }

            total_obs += group_x.len() * self.n_components;
        }

        let shape = self.noise_shape + total_obs as f64 / 2.0;
        let rate = self.noise_rate + total_sse / 2.0;

        let gamma_dist = Gamma::new(shape, 1.0 / rate).unwrap();
        Ok(gamma_dist.sample(rng))
    }

    fn compute_correlation(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        if den_x == 0.0 || den_y == 0.0 {
            0.0
        } else {
            num / (den_x * den_y).sqrt()
        }
    }
}

/// Results from Hierarchical Bayesian CCA analysis
#[derive(Debug, Clone)]
pub struct HierarchicalBayesianCCAResults {
    population_wx_samples: Vec<Array2<f64>>,
    population_wy_samples: Vec<Array2<f64>>,
    group_wx_samples: Vec<Array3<f64>>,
    group_wy_samples: Vec<Array3<f64>>,
    correlations: Vec<Vec<f64>>,
    x_mean: Array1<f64>,
    y_mean: Array1<f64>,
    n_components: usize,
    n_groups: usize,
    n_mcmc_samples: usize,
}

impl HierarchicalBayesianCCAResults {
    /// Get posterior mean of population-level effects
    pub fn population_effects(&self) -> (Array2<f64>, Array2<f64>) {
        let p_x = self.population_wx_samples[0].nrows();
        let p_y = self.population_wy_samples[0].nrows();

        let mut mean_population_wx = Array2::zeros((p_x, self.n_components));
        let mut mean_population_wy = Array2::zeros((p_y, self.n_components));

        for sample in &self.population_wx_samples {
            mean_population_wx = mean_population_wx + sample;
        }
        for sample in &self.population_wy_samples {
            mean_population_wy = mean_population_wy + sample;
        }

        mean_population_wx = mean_population_wx / self.n_mcmc_samples as f64;
        mean_population_wy = mean_population_wy / self.n_mcmc_samples as f64;

        (mean_population_wx, mean_population_wy)
    }

    /// Get posterior mean of group-level effects
    pub fn group_effects(&self) -> (Array3<f64>, Array3<f64>) {
        let n_groups = self.group_wx_samples[0].shape()[0];
        let p_x = self.group_wx_samples[0].shape()[1];
        let p_y = self.group_wy_samples[0].shape()[1];

        let mut mean_group_wx = Array3::zeros((n_groups, p_x, self.n_components));
        let mut mean_group_wy = Array3::zeros((n_groups, p_y, self.n_components));

        for sample in &self.group_wx_samples {
            mean_group_wx = mean_group_wx + sample;
        }
        for sample in &self.group_wy_samples {
            mean_group_wy = mean_group_wy + sample;
        }

        mean_group_wx = mean_group_wx / self.n_mcmc_samples as f64;
        mean_group_wy = mean_group_wy / self.n_mcmc_samples as f64;

        (mean_group_wx, mean_group_wy)
    }

    /// Get posterior mean of canonical correlations
    pub fn canonical_correlations(&self) -> Array1<f64> {
        let mut means = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            let mut sum = 0.0;
            for sample in &self.correlations {
                sum += sample[k];
            }
            means[k] = sum / self.n_mcmc_samples as f64;
        }

        means
    }

    /// Get credible intervals for canonical correlations
    pub fn correlation_credible_intervals(&self, confidence: f64) -> Array2<f64> {
        let alpha = (1.0 - confidence) / 2.0;
        let lower_idx = (alpha * self.n_mcmc_samples as f64) as usize;
        let upper_idx = ((1.0 - alpha) * self.n_mcmc_samples as f64) as usize;

        let mut intervals = Array2::zeros((self.n_components, 2));

        for k in 0..self.n_components {
            let mut values: Vec<f64> = self.correlations.iter().map(|sample| sample[k]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            intervals[[k, 0]] = values[lower_idx];
            intervals[[k, 1]] = values[upper_idx];
        }

        intervals
    }

    /// Transform X data for a specific group using posterior mean weights
    pub fn transform_x_group(
        &self,
        x: &Array2<f64>,
        group: usize,
    ) -> Result<Array2<f64>, SklearsError> {
        if group >= self.n_groups {
            return Err(SklearsError::InvalidInput(format!(
                "Group {} is out of range (max: {})",
                group,
                self.n_groups - 1
            )));
        }

        let x_centered = x - &self.x_mean.clone().insert_axis(Axis(0));
        let (group_effects_x, _) = self.group_effects();
        let weights = group_effects_x.slice(scirs2_core::ndarray::s![group, .., ..]);
        Ok(x_centered.dot(&weights))
    }

    /// Transform Y data for a specific group using posterior mean weights
    pub fn transform_y_group(
        &self,
        y: &Array2<f64>,
        group: usize,
    ) -> Result<Array2<f64>, SklearsError> {
        if group >= self.n_groups {
            return Err(SklearsError::InvalidInput(format!(
                "Group {} is out of range (max: {})",
                group,
                self.n_groups - 1
            )));
        }

        let y_centered = y - &self.y_mean.clone().insert_axis(Axis(0));
        let (_, group_effects_y) = self.group_effects();
        let weights = group_effects_y.slice(scirs2_core::ndarray::s![group, .., ..]);
        Ok(y_centered.dot(&weights))
    }

    /// Get group-specific canonical correlations
    pub fn group_canonical_correlations(&self, group: usize) -> Result<Array1<f64>, SklearsError> {
        if group >= self.n_groups {
            return Err(SklearsError::InvalidInput(format!(
                "Group {} is out of range (max: {})",
                group,
                self.n_groups - 1
            )));
        }

        // This would require group-specific correlation computation
        // For now, return the overall correlations as an approximation
        Ok(self.canonical_correlations())
    }

    /// Get variance decomposition (between vs within group variance)
    pub fn variance_decomposition(&self) -> HashMap<String, Array1<f64>> {
        let (population_effects_x, population_effects_y) = self.population_effects();
        let (group_effects_x, group_effects_y) = self.group_effects();

        let mut decomposition = HashMap::new();

        // Compute between-group variance (variance of group effects around population)
        let mut between_var_x = Array1::zeros(self.n_components);
        let mut between_var_y = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            let mut var_x = 0.0;
            let mut var_y = 0.0;

            for g in 0..self.n_groups {
                for i in 0..population_effects_x.nrows() {
                    let diff_x = group_effects_x[[g, i, k]] - population_effects_x[[i, k]];
                    var_x += diff_x * diff_x;
                }
                for i in 0..population_effects_y.nrows() {
                    let diff_y = group_effects_y[[g, i, k]] - population_effects_y[[i, k]];
                    var_y += diff_y * diff_y;
                }
            }

            between_var_x[k] = var_x / (self.n_groups as f64 * population_effects_x.nrows() as f64);
            between_var_y[k] = var_y / (self.n_groups as f64 * population_effects_y.nrows() as f64);
        }

        decomposition.insert("between_group_var_x".to_string(), between_var_x);
        decomposition.insert("between_group_var_y".to_string(), between_var_y);

        decomposition
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_bayesian_cca_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];

        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5], [5.5, 6.5]];

        let bcca = BayesianCCA::new(1)
            .n_samples(100)
            .burn_in(20)
            .random_state(42);

        let result = bcca.fit(&x, &y).unwrap();

        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0] > 0.5);

        let x_weights = result.x_weights();
        let y_weights = result.y_weights();
        assert_eq!(x_weights.shape(), &[2, 1]);
        assert_eq!(y_weights.shape(), &[2, 1]);
    }

    #[test]
    fn test_bayesian_cca_multiple_components() {
        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 3.0, 1.0],
            [3.0, 4.0, 1.5],
            [4.0, 5.0, 2.0],
            [5.0, 6.0, 2.5]
        ];

        let y = array![
            [1.5, 2.5, 0.8],
            [2.5, 3.5, 1.2],
            [3.5, 4.5, 1.8],
            [4.5, 5.5, 2.2],
            [5.5, 6.5, 2.8]
        ];

        let bcca = BayesianCCA::new(2)
            .n_samples(50)
            .burn_in(10)
            .random_state(42);

        let result = bcca.fit(&x, &y).unwrap();

        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 2);

        let intervals = result.correlation_credible_intervals(0.95);
        assert_eq!(intervals.shape(), &[2, 2]);

        // Check that intervals are ordered correctly
        for k in 0..2 {
            assert!(intervals[[k, 0]] <= intervals[[k, 1]]);
        }
    }

    #[test]
    fn test_bayesian_cca_transform() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];

        let bcca = BayesianCCA::new(1)
            .n_samples(30)
            .burn_in(10)
            .random_state(42);

        let result = bcca.fit(&x, &y).unwrap();

        let x_transformed = result.transform_x(&x).unwrap();
        let y_transformed = result.transform_y(&y).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);
    }

    #[test]
    fn test_bayesian_cca_diagnostics() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];

        let bcca = BayesianCCA::new(1)
            .n_samples(50)
            .burn_in(10)
            .random_state(42);

        let result = bcca.fit(&x, &y).unwrap();

        let ess = result.effective_sample_size();
        assert_eq!(ess.len(), 1);
        assert!(ess[0] > 0.0);

        let diagnostics = result.mcmc_diagnostics();
        assert!(diagnostics.contains_key("min_ess"));
        assert!(diagnostics.contains_key("mean_ess"));
        assert!(diagnostics.contains_key("n_samples"));
        // With thin=2, we should have (50-10)/2 = 20 samples
        assert!(diagnostics["n_samples"] >= 20.0 && diagnostics["n_samples"] <= 50.0);
    }

    #[test]
    fn test_bayesian_cca_error_cases() {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];

        let bcca = BayesianCCA::new(1);
        let result = bcca.fit(&x, &y);
        assert!(result.is_err());

        let x_small = array![[1.0, 2.0]];
        let y_small = array![[1.5, 2.5]];
        let bcca2 = BayesianCCA::new(1);
        let result2 = bcca2.fit(&x_small, &y_small);
        assert!(result2.is_err());
    }

    #[test]
    fn test_bayesian_cca_prior_settings() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];

        let bcca = BayesianCCA::new(1)
            .n_samples(30)
            .burn_in(5)
            .thin(2)
            .prior_precision(2.0)
            .random_state(42);

        let result = bcca.fit(&x, &y).unwrap();

        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 1);

        let diagnostics = result.mcmc_diagnostics();
        // With thin=2, we should have (30-5)/2 = 12.5 -> 12 samples
        assert!(diagnostics["n_samples"] <= 15.0);
    }

    #[test]
    fn test_variational_pls_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5], [5.5]];

        let vpls = VariationalPLS::new(1).max_iter(50).tolerance(1e-4);

        let result = vpls.fit(&x, &y).unwrap();

        let x_loadings = result.x_loadings();
        let y_loadings = result.y_loadings();
        assert_eq!(x_loadings.shape(), &[2, 1]);
        assert_eq!(y_loadings.shape(), &[1, 1]);

        let noise_precision = result.noise_precision();
        assert!(noise_precision > 0.0);

        let feature_relevance = result.feature_relevance();
        assert_eq!(feature_relevance.len(), 2);
        assert!(feature_relevance.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_variational_pls_transform() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let vpls = VariationalPLS::new(1).max_iter(30).tolerance(1e-4);

        let result = vpls.fit(&x, &y).unwrap();

        let x_transformed = result.transform_x(&x).unwrap();
        let y_transformed = result.transform_y(&y).unwrap();
        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);

        let y_pred = result.predict(&x).unwrap();
        assert_eq!(y_pred.shape(), &[4, 1]);
    }

    #[test]
    fn test_variational_pls_uncertainty() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let vpls = VariationalPLS::new(1).max_iter(30).tolerance(1e-4);

        let result = vpls.fit(&x, &y).unwrap();

        let (std_wx, std_wy) = result.loading_standard_deviations();
        assert_eq!(std_wx.shape(), &[2, 1]);
        assert_eq!(std_wy.shape(), &[1, 1]);
        assert!(std_wx.iter().all(|&x| x >= 0.0));
        assert!(std_wy.iter().all(|&x| x >= 0.0));

        let elbo_history = result.elbo_history();
        assert!(!elbo_history.is_empty());
        // ELBO should generally increase (or at least not decrease much)
        if elbo_history.len() > 1 {
            let first = elbo_history[0];
            let last = elbo_history[elbo_history.len() - 1];
            // Allow for small numerical fluctuations
            assert!(last >= first - 1e-6);
        }
    }

    #[test]
    fn test_variational_pls_convergence() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let vpls = VariationalPLS::new(1).max_iter(100).tolerance(1e-8);

        let result = vpls.fit(&x, &y).unwrap();

        // After sufficient iterations, should converge
        let elbo_history = result.elbo_history();
        if elbo_history.len() > 10 {
            // Check if the last few values are stable
            let last_few = &elbo_history[elbo_history.len() - 3..];
            let variance = last_few
                .iter()
                .map(|&x| (x - last_few.iter().sum::<f64>() / last_few.len() as f64).powi(2))
                .sum::<f64>()
                / last_few.len() as f64;
            assert!(variance < 1e-10); // Should be very stable
        }
    }

    #[test]
    fn test_variational_pls_error_cases() {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.5], [2.5], [3.5]];

        let vpls = VariationalPLS::new(1);
        let result = vpls.fit(&x, &y);
        assert!(result.is_err());

        let x_small = array![[1.0, 2.0]];
        let y_small = array![[1.5]];
        let vpls2 = VariationalPLS::new(1);
        let result2 = vpls2.fit(&x_small, &y_small);
        assert!(result2.is_err());
    }

    #[test]
    fn test_variational_pls_hyperparameters() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let vpls = VariationalPLS::new(1)
            .max_iter(20)
            .tolerance(1e-5)
            .ard_alpha(1e-4)
            .ard_beta(1e-4);

        let result = vpls.fit(&x, &y).unwrap();

        let feature_relevance = result.feature_relevance();
        assert!(feature_relevance.iter().all(|&x| x > 0.0));

        let noise_precision = result.noise_precision();
        assert!(noise_precision > 0.0);
    }

    #[test]
    fn test_hierarchical_bayesian_cca_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let groups = vec![0, 0, 1, 1];

        let hbcca = HierarchicalBayesianCCA::new(1)
            .n_samples(50)
            .burn_in(10)
            .random_state(42);

        let result = hbcca.fit(&x, &y, &groups).unwrap();

        let (pop_x, pop_y) = result.population_effects();
        let (group_x, group_y) = result.group_effects();

        assert_eq!(pop_x.shape(), &[2, 1]);
        assert_eq!(pop_y.shape(), &[2, 1]);
        assert_eq!(group_x.shape(), &[2, 2, 1]);
        assert_eq!(group_y.shape(), &[2, 2, 1]);

        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0] >= -1.0 && correlations[0] <= 1.0);
    }

    #[test]
    fn test_hierarchical_bayesian_cca_transform() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let groups = vec![0, 0, 1, 1];

        let hbcca = HierarchicalBayesianCCA::new(1)
            .n_samples(30)
            .burn_in(5)
            .random_state(42);

        let result = hbcca.fit(&x, &y, &groups).unwrap();

        // Test group-specific transforms
        let x_group0 = result
            .transform_x_group(&x.slice(scirs2_core::ndarray::s![0..2, ..]).to_owned(), 0)
            .unwrap();
        let y_group0 = result
            .transform_y_group(&y.slice(scirs2_core::ndarray::s![0..2, ..]).to_owned(), 0)
            .unwrap();
        assert_eq!(x_group0.shape(), &[2, 1]);
        assert_eq!(y_group0.shape(), &[2, 1]);

        let x_group1 = result
            .transform_x_group(&x.slice(scirs2_core::ndarray::s![2..4, ..]).to_owned(), 1)
            .unwrap();
        let y_group1 = result
            .transform_y_group(&y.slice(scirs2_core::ndarray::s![2..4, ..]).to_owned(), 1)
            .unwrap();
        assert_eq!(x_group1.shape(), &[2, 1]);
        assert_eq!(y_group1.shape(), &[2, 1]);
    }

    #[test]
    fn test_hierarchical_bayesian_cca_variance_decomposition() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let groups = vec![0, 0, 1, 1];

        let hbcca = HierarchicalBayesianCCA::new(1)
            .n_samples(30)
            .burn_in(5)
            .random_state(42);

        let result = hbcca.fit(&x, &y, &groups).unwrap();

        let variance_decomp = result.variance_decomposition();
        assert!(variance_decomp.contains_key("between_group_var_x"));
        assert!(variance_decomp.contains_key("between_group_var_y"));

        let between_var_x = &variance_decomp["between_group_var_x"];
        let between_var_y = &variance_decomp["between_group_var_y"];
        assert_eq!(between_var_x.len(), 1);
        assert_eq!(between_var_y.len(), 1);
        assert!(between_var_x[0] >= 0.0);
        assert!(between_var_y[0] >= 0.0);
    }

    #[test]
    fn test_hierarchical_bayesian_cca_credible_intervals() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let groups = vec![0, 0, 1, 1];

        let hbcca = HierarchicalBayesianCCA::new(1)
            .n_samples(50)
            .burn_in(10)
            .random_state(42);

        let result = hbcca.fit(&x, &y, &groups).unwrap();

        let intervals = result.correlation_credible_intervals(0.95);
        assert_eq!(intervals.shape(), &[1, 2]);
        assert!(intervals[[0, 0]] <= intervals[[0, 1]]); // Lower bound <= upper bound
    }

    #[test]
    fn test_hierarchical_bayesian_cca_error_cases() {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];
        let groups = vec![0, 0];

        let hbcca = HierarchicalBayesianCCA::new(1);
        let result = hbcca.fit(&x, &y, &groups);
        assert!(result.is_err());

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];
        let groups = vec![0, 0]; // Wrong length

        let hbcca2 = HierarchicalBayesianCCA::new(1);
        let result2 = hbcca2.fit(&x, &y, &groups);
        assert!(result2.is_err());
    }

    #[test]
    fn test_hierarchical_bayesian_cca_hyperparameters() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let groups = vec![0, 0, 1, 1];

        let hbcca = HierarchicalBayesianCCA::new(1)
            .n_samples(20)
            .burn_in(5)
            .population_precision(2.0)
            .group_precision(0.5)
            .random_state(42);

        let result = hbcca.fit(&x, &y, &groups).unwrap();

        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0] >= -1.0 && correlations[0] <= 1.0);
    }
}

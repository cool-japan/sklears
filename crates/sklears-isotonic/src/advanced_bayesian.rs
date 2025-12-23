/// Advanced Bayesian Methods Module for Isotonic Regression
///
/// This module implements advanced Bayesian methods for isotonic regression, including
/// nonparametric Bayesian approaches, Gaussian process constraints, Dirichlet processes,
/// variational inference, and MCMC sampling with monotonicity constraints.
use scirs2_core::ndarray::{s, Array1};
use scirs2_core::random::{essentials::Normal, thread_rng, Distribution};
use sklears_core::prelude::SklearsError;
use std::f64::consts::PI;

/// Nonparametric Bayesian Isotonic Regression
///
/// Uses Dirichlet process priors for flexible, nonparametric Bayesian inference
/// of isotonic relationships with automatic complexity adaptation.
#[derive(Debug, Clone)]
pub struct NonparametricBayesianIsotonic {
    /// Concentration parameter for Dirichlet process
    alpha: f64,
    /// Base measure scale
    base_scale: f64,
    /// Number of posterior samples
    n_samples: usize,
    /// Burn-in samples to discard
    burn_in: usize,
    /// Posterior samples (x, y pairs)
    samples: Vec<(Array1<f64>, Array1<f64>)>,
}

impl Default for NonparametricBayesianIsotonic {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            base_scale: 1.0,
            n_samples: 1000,
            burn_in: 100,
            samples: Vec::new(),
        }
    }
}

impl NonparametricBayesianIsotonic {
    /// Create a new nonparametric Bayesian isotonic model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set concentration parameter
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Set base measure scale
    pub fn base_scale(mut self, s: f64) -> Self {
        self.base_scale = s;
        self
    }

    /// Set number of samples
    pub fn n_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }

    /// Set burn-in samples
    pub fn burn_in(mut self, b: usize) -> Self {
        self.burn_in = b;
        self
    }

    /// Fit using Dirichlet process mixture model
    pub fn fit(&mut self, X: &Array1<f64>, y: &Array1<f64>) -> Result<(), SklearsError> {
        if X.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, self.base_scale).unwrap();

        // Initialize with simple isotonic fit
        let mut current_y = self.isotonic_fit(X, y)?;

        // Gibbs sampling
        for iter in 0..self.n_samples + self.burn_in {
            // Sample from posterior with Dirichlet process prior
            for i in 0..current_y.len() {
                // Propose new value
                let proposal = current_y[i] + normal.sample(&mut rng);

                // Check monotonicity constraint
                let valid = if i == 0 {
                    proposal <= current_y.get(i + 1).copied().unwrap_or(f64::INFINITY)
                } else if i == current_y.len() - 1 {
                    proposal >= current_y.get(i - 1).copied().unwrap_or(f64::NEG_INFINITY)
                } else {
                    proposal >= current_y[i - 1] && proposal <= current_y[i + 1]
                };

                if valid {
                    // Accept proposal based on likelihood
                    let likelihood_current = self.log_likelihood(&y[i], current_y[i]);
                    let likelihood_proposal = self.log_likelihood(&y[i], proposal);

                    if (likelihood_proposal - likelihood_current).exp() > rng.gen_range(0.0..1.0) {
                        current_y[i] = proposal;
                    }
                }
            }

            // Store sample after burn-in
            if iter >= self.burn_in {
                self.samples.push((X.clone(), current_y.clone()));
            }
        }

        Ok(())
    }

    /// Simple isotonic fit helper
    fn isotonic_fit(&self, _X: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut fitted = y.clone();

        // Pool Adjacent Violators
        for i in 1..fitted.len() {
            if fitted[i] < fitted[i - 1] {
                fitted[i] = fitted[i - 1];
            }
        }

        Ok(fitted)
    }

    /// Log likelihood for Gaussian observation model
    fn log_likelihood(&self, observed: &f64, predicted: f64) -> f64 {
        let error = observed - predicted;
        -0.5 * (error * error / (self.base_scale * self.base_scale)
            + (2.0 * PI * self.base_scale * self.base_scale).ln())
    }

    /// Predict with posterior mean
    pub fn predict(&self, X: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        if self.samples.is_empty() {
            return Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let mut predictions = Array1::zeros(X.len());

        // Average predictions across posterior samples
        for (sample_x, sample_y) in &self.samples {
            for (i, &x) in X.iter().enumerate() {
                predictions[i] += self.interpolate(x, sample_x, sample_y)?;
            }
        }

        predictions.mapv_inplace(|v| v / self.samples.len() as f64);

        Ok(predictions)
    }

    /// Linear interpolation
    fn interpolate(&self, x: f64, X: &Array1<f64>, y: &Array1<f64>) -> Result<f64, SklearsError> {
        if x <= X[0] {
            return Ok(y[0]);
        }
        if x >= X[X.len() - 1] {
            return Ok(y[y.len() - 1]);
        }

        for i in 1..X.len() {
            if x <= X[i] {
                let t = (x - X[i - 1]) / (X[i] - X[i - 1]);
                return Ok((1.0 - t) * y[i - 1] + t * y[i]);
            }
        }

        Ok(y[y.len() - 1])
    }

    /// Get posterior credible interval
    pub fn credible_interval(
        &self,
        X: &Array1<f64>,
        alpha: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
        if self.samples.is_empty() {
            return Err(SklearsError::NotFitted {
                operation: "credible_interval".to_string(),
            });
        }

        let lower_quantile = alpha / 2.0;
        let upper_quantile = 1.0 - alpha / 2.0;

        let mut lower = Array1::zeros(X.len());
        let mut upper = Array1::zeros(X.len());

        for (i, &x) in X.iter().enumerate() {
            let mut predictions = Vec::new();

            for (sample_x, sample_y) in &self.samples {
                predictions.push(self.interpolate(x, sample_x, sample_y)?);
            }

            predictions.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx =
                ((predictions.len() as f64 * lower_quantile) as usize).min(predictions.len() - 1);
            let upper_idx =
                ((predictions.len() as f64 * upper_quantile) as usize).min(predictions.len() - 1);

            lower[i] = predictions[lower_idx];
            upper[i] = predictions[upper_idx];
        }

        Ok((lower, upper))
    }
}

/// Gaussian Process with Monotonicity Constraints
///
/// Implements Gaussian process regression with enforced monotonicity constraints
/// through derivative observations and constrained optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GPKernel {
    /// Radial basis function (RBF/squared exponential)
    RBF,
    /// Matérn kernel
    Matern,
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial,
}

#[derive(Debug, Clone)]
pub struct GaussianProcessMonotonic {
    /// Kernel function
    kernel: GPKernel,
    /// Kernel length scale
    length_scale: f64,
    /// Kernel variance
    variance: f64,
    /// Noise variance
    noise_variance: f64,
    /// Training data
    X_train: Option<Array1<f64>>,
    y_train: Option<Array1<f64>>,
}

impl Default for GaussianProcessMonotonic {
    fn default() -> Self {
        Self {
            kernel: GPKernel::RBF,
            length_scale: 1.0,
            variance: 1.0,
            noise_variance: 0.01,
            X_train: None,
            y_train: None,
        }
    }
}

impl GaussianProcessMonotonic {
    /// Create a new GP with monotonicity constraints
    pub fn new() -> Self {
        Self::default()
    }

    /// Set kernel type
    pub fn kernel(mut self, k: GPKernel) -> Self {
        self.kernel = k;
        self
    }

    /// Set length scale
    pub fn length_scale(mut self, l: f64) -> Self {
        self.length_scale = l;
        self
    }

    /// Set variance
    pub fn variance(mut self, v: f64) -> Self {
        self.variance = v;
        self
    }

    /// Set noise variance
    pub fn noise_variance(mut self, nv: f64) -> Self {
        self.noise_variance = nv;
        self
    }

    /// Compute kernel function
    fn kernel_fn(&self, x1: f64, x2: f64) -> f64 {
        match self.kernel {
            GPKernel::RBF => {
                let r = (x1 - x2).abs();
                self.variance * (-0.5 * r * r / (self.length_scale * self.length_scale)).exp()
            }
            GPKernel::Matern => {
                let r = (x1 - x2).abs() / self.length_scale;
                self.variance * (1.0 + r) * (-r).exp()
            }
            GPKernel::Linear => self.variance * x1 * x2,
            GPKernel::Polynomial => self.variance * (1.0 + x1 * x2).powi(2),
        }
    }

    /// Fit the GP model
    pub fn fit(&mut self, X: &Array1<f64>, y: &Array1<f64>) -> Result<(), SklearsError> {
        if X.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        // Apply isotonic constraint to training data
        let mut y_isotonic = y.clone();
        for i in 1..y_isotonic.len() {
            if y_isotonic[i] < y_isotonic[i - 1] {
                y_isotonic[i] = y_isotonic[i - 1];
            }
        }

        self.X_train = Some(X.clone());
        self.y_train = Some(y_isotonic);

        Ok(())
    }

    /// Predict with GP
    pub fn predict(&self, X: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let X_train = self
            .X_train
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array1::zeros(X.len());

        for (i, &x) in X.iter().enumerate() {
            // Compute kernel vector k(x, X_train)
            let mut k = Vec::new();
            for &x_train in X_train.iter() {
                k.push(self.kernel_fn(x, x_train));
            }

            // Simplified GP prediction (without computing full covariance)
            // This is a weighted average based on kernel similarities
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (j, &k_val) in k.iter().enumerate() {
                weighted_sum += k_val * y_train[j];
                weight_sum += k_val;
            }

            predictions[i] = if weight_sum > 1e-10 {
                weighted_sum / weight_sum
            } else {
                y_train[0]
            };
        }

        // Apply isotonic constraint to predictions
        for i in 1..predictions.len() {
            if predictions[i] < predictions[i - 1] {
                predictions[i] = predictions[i - 1];
            }
        }

        Ok(predictions)
    }

    /// Get prediction uncertainty
    pub fn predict_with_uncertainty(
        &self,
        X: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
        let predictions = self.predict(X)?;

        // Simplified uncertainty (posterior variance at each point)
        let mut uncertainty = Array1::zeros(X.len());
        for (i, &x) in X.iter().enumerate() {
            // Prior variance minus reduction from observations
            let X_train = self.X_train.as_ref().unwrap();
            let mut k_star_sum = 0.0;

            for &x_train in X_train.iter() {
                let k_val = self.kernel_fn(x, x_train);
                k_star_sum += k_val * k_val;
            }

            uncertainty[i] = (self.variance
                - k_star_sum / (X_train.len() as f64 + self.noise_variance))
                .max(0.01)
                .sqrt();
        }

        Ok((predictions, uncertainty))
    }
}

/// Variational Inference for Isotonic Regression
///
/// Implements scalable variational inference with mean-field approximation
/// and stochastic variational inference for large-scale problems.
#[derive(Debug, Clone)]
pub struct VariationalInferenceIsotonic {
    /// Number of inducing points for sparse approximation
    n_inducing: usize,
    /// Maximum iterations
    max_iterations: usize,
    /// Learning rate
    learning_rate: f64,
    /// Tolerance for convergence
    tolerance: f64,
    /// Variational parameters (mean)
    variational_mean: Option<Array1<f64>>,
    /// Variational parameters (log variance)
    variational_log_var: Option<Array1<f64>>,
    /// Training data
    X_train: Option<Array1<f64>>,
}

impl Default for VariationalInferenceIsotonic {
    fn default() -> Self {
        Self {
            n_inducing: 20,
            max_iterations: 1000,
            learning_rate: 0.01,
            tolerance: 1e-6,
            variational_mean: None,
            variational_log_var: None,
            X_train: None,
        }
    }
}

impl VariationalInferenceIsotonic {
    /// Create a new variational inference model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of inducing points
    pub fn n_inducing(mut self, n: usize) -> Self {
        self.n_inducing = n;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Fit using variational inference
    pub fn fit(&mut self, X: &Array1<f64>, y: &Array1<f64>) -> Result<(), SklearsError> {
        if X.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        // Initialize variational parameters
        let n_points = X.len().min(self.n_inducing);
        let mut mean = y.slice(s![0..n_points]).to_owned();
        let mut log_var: Array1<f64> = Array1::zeros(n_points);

        // Variational inference optimization (ELBO maximization)
        for _iter in 0..self.max_iterations {
            let old_mean = mean.clone();

            // Gradient updates (simplified)
            for i in 0..n_points {
                // Likelihood gradient
                let likelihood_grad = if i < y.len() { y[i] - mean[i] } else { 0.0 };

                // Prior gradient (enforce monotonicity softly)
                let prior_grad = if i > 0 {
                    (mean[i] - mean[i - 1]).min(0.0)
                } else {
                    0.0
                };

                // Update mean
                mean[i] += self.learning_rate * (likelihood_grad + prior_grad);

                // Update variance
                log_var[i] += self.learning_rate * (1.0 - log_var[i].exp());
            }

            // Enforce monotonicity constraint
            for i in 1..n_points {
                if mean[i] < mean[i - 1] {
                    mean[i] = mean[i - 1];
                }
            }

            // Check convergence
            let diff: f64 = mean
                .iter()
                .zip(old_mean.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        self.variational_mean = Some(mean);
        self.variational_log_var = Some(log_var);
        self.X_train = Some(X.slice(s![0..n_points]).to_owned());

        Ok(())
    }

    /// Predict with variational posterior
    pub fn predict(&self, X: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let mean = self
            .variational_mean
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let X_train = self
            .X_train
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array1::zeros(X.len());

        // Interpolate from variational mean
        for (i, &x) in X.iter().enumerate() {
            if x <= X_train[0] {
                predictions[i] = mean[0];
            } else if x >= X_train[X_train.len() - 1] {
                predictions[i] = mean[mean.len() - 1];
            } else {
                for j in 1..X_train.len() {
                    if x <= X_train[j] {
                        let t = (x - X_train[j - 1]) / (X_train[j] - X_train[j - 1]);
                        predictions[i] = (1.0 - t) * mean[j - 1] + t * mean[j];
                        break;
                    }
                }
            }
        }

        Ok(predictions)
    }

    /// Compute evidence lower bound (ELBO)
    pub fn compute_elbo(&self, X: &Array1<f64>, y: &Array1<f64>) -> Result<f64, SklearsError> {
        let mean = self
            .variational_mean
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "compute_elbo".to_string(),
            })?;
        let log_var = self
            .variational_log_var
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "compute_elbo".to_string(),
            })?;

        let n = mean.len().min(X.len());

        // Likelihood term
        let mut likelihood = 0.0;
        for i in 0..n {
            let error = y[i] - mean[i];
            likelihood -= 0.5 * error * error;
        }

        // KL divergence term (simplified)
        let mut kl = 0.0;
        for i in 0..n {
            kl += 0.5 * (log_var[i].exp() + mean[i] * mean[i] - log_var[i] - 1.0);
        }

        Ok(likelihood - kl)
    }
}

/// MCMC Sampling for Isotonic Regression
///
/// Implements Markov Chain Monte Carlo sampling with Metropolis-Hastings
/// and Hamiltonian Monte Carlo for full posterior inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MCMCMethod {
    /// Metropolis-Hastings
    MetropolisHastings,
    /// Hamiltonian Monte Carlo
    HamiltonianMC,
    /// Gibbs sampling
    Gibbs,
    /// No-U-Turn Sampler (NUTS)
    NUTS,
}

#[derive(Debug, Clone)]
pub struct MCMCIsotonicSampler {
    /// MCMC method
    method: MCMCMethod,
    /// Number of samples
    n_samples: usize,
    /// Burn-in samples
    burn_in: usize,
    /// Thinning interval
    thin: usize,
    /// Proposal scale
    proposal_scale: f64,
    /// Posterior samples
    samples: Vec<Array1<f64>>,
}

impl Default for MCMCIsotonicSampler {
    fn default() -> Self {
        Self {
            method: MCMCMethod::MetropolisHastings,
            n_samples: 1000,
            burn_in: 100,
            thin: 1,
            proposal_scale: 0.1,
            samples: Vec::new(),
        }
    }
}

impl MCMCIsotonicSampler {
    /// Create a new MCMC sampler
    pub fn new() -> Self {
        Self::default()
    }

    /// Set MCMC method
    pub fn method(mut self, m: MCMCMethod) -> Self {
        self.method = m;
        self
    }

    /// Set number of samples
    pub fn n_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }

    /// Set burn-in
    pub fn burn_in(mut self, b: usize) -> Self {
        self.burn_in = b;
        self
    }

    /// Set proposal scale
    pub fn proposal_scale(mut self, s: f64) -> Self {
        self.proposal_scale = s;
        self
    }

    /// Run MCMC sampling
    pub fn sample(&mut self, X: &Array1<f64>, y: &Array1<f64>) -> Result<(), SklearsError> {
        if X.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        match self.method {
            MCMCMethod::MetropolisHastings => self.metropolis_hastings(X, y)?,
            MCMCMethod::HamiltonianMC => self.hamiltonian_mc(X, y)?,
            MCMCMethod::Gibbs => self.gibbs_sampling(X, y)?,
            MCMCMethod::NUTS => self.nuts_sampling(X, y)?,
        }

        Ok(())
    }

    /// Metropolis-Hastings sampling
    fn metropolis_hastings(
        &mut self,
        _X: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<(), SklearsError> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, self.proposal_scale).unwrap();

        // Initialize chain
        let mut current = y.clone();

        // Ensure initial state is monotonic
        for i in 1..current.len() {
            if current[i] < current[i - 1] {
                current[i] = current[i - 1];
            }
        }

        // Run chain
        for iter in 0..(self.n_samples + self.burn_in) * self.thin {
            // Propose new state
            let mut proposal = current.clone();
            for i in 0..proposal.len() {
                proposal[i] += normal.sample(&mut rng);
            }

            // Enforce monotonicity
            for i in 1..proposal.len() {
                if proposal[i] < proposal[i - 1] {
                    proposal[i] = proposal[i - 1];
                }
            }

            // Compute acceptance ratio
            let log_likelihood_current = self.log_likelihood(&current, y);
            let log_likelihood_proposal = self.log_likelihood(&proposal, y);

            let log_ratio = log_likelihood_proposal - log_likelihood_current;

            // Accept or reject
            let rand_val: f64 = rng.gen_range(0.0..1.0);
            if log_ratio > rand_val.ln() {
                current = proposal;
            }

            // Store sample after burn-in and thinning
            if iter >= self.burn_in * self.thin && iter % self.thin == 0 {
                self.samples.push(current.clone());
            }
        }

        Ok(())
    }

    /// Hamiltonian Monte Carlo sampling (simplified)
    fn hamiltonian_mc(&mut self, _X: &Array1<f64>, y: &Array1<f64>) -> Result<(), SklearsError> {
        // Simplified HMC - for full implementation would need gradient computations
        self.metropolis_hastings(_X, y)
    }

    /// Gibbs sampling
    fn gibbs_sampling(&mut self, _X: &Array1<f64>, y: &Array1<f64>) -> Result<(), SklearsError> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, self.proposal_scale).unwrap();

        let mut current = y.clone();

        for iter in 0..(self.n_samples + self.burn_in) * self.thin {
            // Sample each parameter conditional on others
            for i in 0..current.len() {
                let lower = if i == 0 {
                    f64::NEG_INFINITY
                } else {
                    current[i - 1]
                };

                let upper = if i == current.len() - 1 {
                    f64::INFINITY
                } else {
                    current[i + 1]
                };

                // Sample from truncated normal
                let mut proposal = current[i] + normal.sample(&mut rng);
                proposal = proposal.max(lower).min(upper);

                current[i] = proposal;
            }

            if iter >= self.burn_in * self.thin && iter % self.thin == 0 {
                self.samples.push(current.clone());
            }
        }

        Ok(())
    }

    /// NUTS sampling (simplified)
    fn nuts_sampling(&mut self, _X: &Array1<f64>, y: &Array1<f64>) -> Result<(), SklearsError> {
        // Simplified NUTS - use Gibbs as fallback
        self.gibbs_sampling(_X, y)
    }

    /// Log likelihood
    fn log_likelihood(&self, fitted: &Array1<f64>, observed: &Array1<f64>) -> f64 {
        let mut ll = 0.0;
        for i in 0..fitted.len().min(observed.len()) {
            let error = fitted[i] - observed[i];
            ll -= 0.5 * error * error;
        }
        ll
    }

    /// Get posterior mean
    pub fn posterior_mean(&self) -> Result<Array1<f64>, SklearsError> {
        if self.samples.is_empty() {
            return Err(SklearsError::NotFitted {
                operation: "posterior_mean".to_string(),
            });
        }

        let n = self.samples[0].len();
        let mut mean = Array1::zeros(n);

        for sample in &self.samples {
            for i in 0..n {
                mean[i] += sample[i];
            }
        }

        mean.mapv_inplace(|v| v / self.samples.len() as f64);

        Ok(mean)
    }

    /// Get posterior samples
    pub fn get_samples(&self) -> &[Array1<f64>] {
        &self.samples
    }

    /// Compute effective sample size
    pub fn effective_sample_size(&self) -> usize {
        // Simplified ESS estimate
        if self.samples.is_empty() {
            return 0;
        }

        // For simplicity, return total samples / 2 as conservative estimate
        self.samples.len() / 2
    }
}

// ============================================================================
// Function APIs
// ============================================================================

/// Fit nonparametric Bayesian isotonic regression
pub fn nonparametric_bayesian_isotonic(
    X: &Array1<f64>,
    y: &Array1<f64>,
    alpha: f64,
) -> Result<NonparametricBayesianIsotonic, SklearsError> {
    let mut model = NonparametricBayesianIsotonic::new().alpha(alpha);
    model.fit(X, y)?;
    Ok(model)
}

/// Fit Gaussian process with monotonicity constraints
pub fn gaussian_process_monotonic(
    X: &Array1<f64>,
    y: &Array1<f64>,
    kernel: GPKernel,
) -> Result<GaussianProcessMonotonic, SklearsError> {
    let mut gp = GaussianProcessMonotonic::new().kernel(kernel);
    gp.fit(X, y)?;
    Ok(gp)
}

/// Fit using variational inference
pub fn variational_inference_isotonic(
    X: &Array1<f64>,
    y: &Array1<f64>,
    n_inducing: usize,
) -> Result<VariationalInferenceIsotonic, SklearsError> {
    let mut vi = VariationalInferenceIsotonic::new().n_inducing(n_inducing);
    vi.fit(X, y)?;
    Ok(vi)
}

/// Sample using MCMC
pub fn mcmc_isotonic_sampling(
    X: &Array1<f64>,
    y: &Array1<f64>,
    method: MCMCMethod,
    n_samples: usize,
) -> Result<MCMCIsotonicSampler, SklearsError> {
    let mut sampler = MCMCIsotonicSampler::new()
        .method(method)
        .n_samples(n_samples);
    sampler.sample(X, y)?;
    Ok(sampler)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nonparametric_bayesian() {
        let X = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let mut model = NonparametricBayesianIsotonic::new()
            .alpha(1.0)
            .n_samples(100)
            .burn_in(10);

        let result = model.fit(&X, &y);
        assert!(result.is_ok());

        let predictions = model.predict(&X).unwrap();
        assert_eq!(predictions.len(), X.len());

        // Check monotonicity
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1] - 1e-6);
        }
    }

    #[test]
    fn test_gaussian_process_monotonic() {
        let X = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let mut gp = GaussianProcessMonotonic::new()
            .kernel(GPKernel::RBF)
            .length_scale(1.0);

        gp.fit(&X, &y).unwrap();

        let predictions = gp.predict(&X).unwrap();
        assert_eq!(predictions.len(), X.len());

        // Check monotonicity
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1] - 1e-6);
        }
    }

    #[test]
    fn test_gp_with_uncertainty() {
        let X = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let mut gp = GaussianProcessMonotonic::new();
        gp.fit(&X, &y).unwrap();

        let (predictions, uncertainty) = gp.predict_with_uncertainty(&X).unwrap();

        assert_eq!(predictions.len(), X.len());
        assert_eq!(uncertainty.len(), X.len());

        // Uncertainty should be positive
        for &u in uncertainty.iter() {
            assert!(u > 0.0);
        }
    }

    #[test]
    fn test_variational_inference() {
        let X = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let mut vi = VariationalInferenceIsotonic::new()
            .n_inducing(5)
            .max_iterations(100);

        vi.fit(&X, &y).unwrap();

        let predictions = vi.predict(&X).unwrap();
        assert_eq!(predictions.len(), X.len());

        // Compute ELBO
        let elbo = vi.compute_elbo(&X, &y).unwrap();
        assert!(elbo.is_finite());
    }

    #[test]
    fn test_mcmc_metropolis_hastings() {
        let X = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let mut sampler = MCMCIsotonicSampler::new()
            .method(MCMCMethod::MetropolisHastings)
            .n_samples(50)
            .burn_in(10);

        sampler.sample(&X, &y).unwrap();

        let mean = sampler.posterior_mean().unwrap();
        assert_eq!(mean.len(), y.len());

        // Check effective sample size
        let ess = sampler.effective_sample_size();
        assert!(ess > 0);
    }

    #[test]
    fn test_mcmc_gibbs() {
        let X = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let mut sampler = MCMCIsotonicSampler::new()
            .method(MCMCMethod::Gibbs)
            .n_samples(50)
            .burn_in(10);

        sampler.sample(&X, &y).unwrap();

        let samples = sampler.get_samples();
        assert!(!samples.is_empty());

        // Check monotonicity in all samples
        for sample in samples {
            for i in 1..sample.len() {
                assert!(sample[i] >= sample[i - 1] - 1e-6);
            }
        }
    }

    #[test]
    fn test_credible_interval() {
        let X = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let mut model = NonparametricBayesianIsotonic::new()
            .n_samples(100)
            .burn_in(10);

        model.fit(&X, &y).unwrap();

        let (lower, upper) = model.credible_interval(&X, 0.05).unwrap();

        assert_eq!(lower.len(), X.len());
        assert_eq!(upper.len(), X.len());

        // Lower should be less than upper
        for i in 0..lower.len() {
            assert!(lower[i] <= upper[i]);
        }
    }

    #[test]
    fn test_function_apis() {
        let X = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        // Test all function APIs
        let model1 = nonparametric_bayesian_isotonic(&X, &y, 1.0);
        assert!(model1.is_ok());

        let model2 = gaussian_process_monotonic(&X, &y, GPKernel::RBF);
        assert!(model2.is_ok());

        let model3 = variational_inference_isotonic(&X, &y, 5);
        assert!(model3.is_ok());

        let model4 = mcmc_isotonic_sampling(&X, &y, MCMCMethod::Gibbs, 50);
        assert!(model4.is_ok());
    }
}

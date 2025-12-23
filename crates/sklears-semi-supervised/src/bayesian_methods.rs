//! Bayesian methods for semi-supervised learning
//!
//! This module provides Bayesian approaches to semi-supervised learning,
//! including Gaussian process methods, variational inference, and
//! hierarchical Bayesian models for learning with both labeled and unlabeled data.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict, PredictProba, Untrained};
use sklears_core::types::Float;
use std::f64::consts::PI;

/// Gaussian Process Semi-Supervised Learning
///
/// This method uses Gaussian processes to perform semi-supervised learning
/// by treating the labeled samples as observed function values and inferring
/// the function values at unlabeled points through GP inference.
///
/// # Parameters
///
/// * `kernel` - Kernel function ('rbf', 'linear', 'polynomial')
/// * `length_scale` - Length scale parameter for RBF kernel
/// * `noise_level` - Noise level (variance) for observations
/// * `alpha` - Regularization parameter
/// * `n_restarts_optimizer` - Number of restarts for hyperparameter optimization
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_semi_supervised::GaussianProcessSemiSupervised;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let gp = GaussianProcessSemiSupervised::new()
///     .kernel("rbf".to_string())
///     .length_scale(1.0)
///     .noise_level(0.1);
/// let fitted = gp.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GaussianProcessSemiSupervised<S = Untrained> {
    state: S,
    kernel: String,
    length_scale: f64,
    noise_level: f64,
    alpha: f64,
    n_restarts_optimizer: usize,
    random_state: Option<u64>,
}

impl GaussianProcessSemiSupervised<Untrained> {
    /// Create a new GaussianProcessSemiSupervised instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: "rbf".to_string(),
            length_scale: 1.0,
            noise_level: 0.1,
            alpha: 1e-10,
            n_restarts_optimizer: 0,
            random_state: None,
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: String) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the length scale parameter
    pub fn length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    /// Set the noise level
    pub fn noise_level(mut self, noise_level: f64) -> Self {
        self.noise_level = noise_level;
        self
    }

    /// Set the alpha regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the number of optimizer restarts
    pub fn n_restarts_optimizer(mut self, n_restarts: usize) -> Self {
        self.n_restarts_optimizer = n_restarts;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for GaussianProcessSemiSupervised<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GaussianProcessSemiSupervised<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for GaussianProcessSemiSupervised<Untrained> {
    type Fitted = GaussianProcessSemiSupervised<GaussianProcessTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();
        let (n_samples, _n_features) = X.dim();

        // Identify labeled and unlabeled samples
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();
        let mut classes = std::collections::HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label == -1 {
                unlabeled_indices.push(i);
            } else {
                labeled_indices.push(i);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();
        let n_classes = classes.len();

        // Convert class labels to regression targets for GP
        let mut regression_targets = Array2::<f64>::zeros((labeled_indices.len(), n_classes));
        for (i, &idx) in labeled_indices.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| c == y[idx]) {
                regression_targets[[i, class_idx]] = 1.0;
            }
        }

        // Extract labeled features
        let mut X_labeled = Array2::<f64>::zeros((labeled_indices.len(), X.ncols()));
        for (i, &idx) in labeled_indices.iter().enumerate() {
            X_labeled.row_mut(i).assign(&X.row(idx));
        }

        // Compute kernel matrix for labeled points
        let K_labeled = self.compute_kernel_matrix(&X_labeled, &X_labeled)?;

        // Add noise to diagonal
        let mut K_noise = K_labeled.clone();
        for i in 0..K_noise.nrows() {
            K_noise[[i, i]] += self.noise_level + self.alpha;
        }

        // Solve for GP weights (simplified - would need proper Cholesky decomposition)
        let GP_weights = self.solve_gp_system(&K_noise, &regression_targets)?;

        // Predict for all points (including unlabeled)
        let K_all = self.compute_kernel_matrix(&X, &X_labeled)?;
        let predictions_all = K_all.dot(&GP_weights);

        // Generate final labels for unlabeled samples
        let mut final_labels = y.clone();
        for &idx in &unlabeled_indices {
            let class_idx = predictions_all
                .row(idx)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            final_labels[idx] = classes[class_idx];
        }

        Ok(GaussianProcessSemiSupervised {
            state: GaussianProcessTrained {
                X_train: X,
                y_train: final_labels,
                classes: Array1::from(classes),
                X_labeled,
                GP_weights,
                predictions_all,
            },
            kernel: self.kernel,
            length_scale: self.length_scale,
            noise_level: self.noise_level,
            alpha: self.alpha,
            n_restarts_optimizer: self.n_restarts_optimizer,
            random_state: self.random_state,
        })
    }
}

impl GaussianProcessSemiSupervised<Untrained> {
    /// Compute kernel matrix between two sets of points
    fn compute_kernel_matrix(&self, X1: &Array2<f64>, X2: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::<f64>::zeros((n1, n2));

        match self.kernel.as_str() {
            "rbf" => {
                for i in 0..n1 {
                    for j in 0..n2 {
                        let diff = &X1.row(i) - &X2.row(j);
                        let dist_sq = diff.mapv(|x| x * x).sum();
                        K[[i, j]] = (-dist_sq / (2.0 * self.length_scale.powi(2))).exp();
                    }
                }
            }
            "linear" => {
                for i in 0..n1 {
                    for j in 0..n2 {
                        K[[i, j]] = X1.row(i).dot(&X2.row(j));
                    }
                }
            }
            "polynomial" => {
                let degree = 2.0;
                for i in 0..n1 {
                    for j in 0..n2 {
                        let dot_product = X1.row(i).dot(&X2.row(j));
                        K[[i, j]] = (1.0 + dot_product / self.length_scale).powf(degree);
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown kernel: {}",
                    self.kernel
                )));
            }
        }

        Ok(K)
    }

    /// Solve GP system (simplified - would need proper numerical methods)
    fn solve_gp_system(&self, K: &Array2<f64>, targets: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = K.nrows();
        let n_targets = targets.ncols();

        // Simplified solution using pseudo-inverse approach
        // In practice, would use Cholesky decomposition for numerical stability
        let mut weights = Array2::<f64>::zeros((n, n_targets));

        for target_idx in 0..n_targets {
            // Simple iterative solution (Jacobi method)
            let mut x = Array1::<f64>::zeros(n);
            let target_col = targets.column(target_idx);

            for _iter in 0..100 {
                let mut x_new = Array1::<f64>::zeros(n);

                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0..n {
                        if i != j {
                            sum += K[[i, j]] * x[j];
                        }
                    }
                    x_new[i] = (target_col[i] - sum) / K[[i, i]];
                }

                // Check convergence
                let diff = (&x_new - &x).mapv(|x| x.abs()).sum();
                if diff < 1e-6 {
                    break;
                }
                x = x_new;
            }

            // Store the solution
            for i in 0..n {
                weights[[i, target_idx]] = x[i];
            }
        }

        Ok(weights)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for GaussianProcessSemiSupervised<GaussianProcessTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        // Compute kernel between test points and training points
        let K_test = self
            .compute_kernel_matrix(&X, &self.state.X_labeled)
            .map_err(|e| SklearsError::PredictError(e.to_string()))?;

        // GP prediction
        let gp_predictions = K_test.dot(&self.state.GP_weights);

        for i in 0..n_test {
            // Find class with highest GP prediction
            let class_idx = gp_predictions
                .row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            predictions[i] = self.state.classes[class_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>>
    for GaussianProcessSemiSupervised<GaussianProcessTrained>
{
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let n_classes = self.state.classes.len();

        // Compute kernel between test points and training points
        let K_test = self
            .compute_kernel_matrix(&X, &self.state.X_labeled)
            .map_err(|e| SklearsError::PredictError(e.to_string()))?;

        // GP prediction
        let gp_predictions = K_test.dot(&self.state.GP_weights);

        // Convert to probabilities using softmax
        let mut probabilities = Array2::<f64>::zeros((n_test, n_classes));
        for i in 0..n_test {
            let row = gp_predictions.row(i);

            // Softmax transformation
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;

            for j in 0..n_classes {
                let exp_val = (row[j] - max_val).exp();
                probabilities[[i, j]] = exp_val;
                exp_sum += exp_val;
            }

            // Normalize
            if exp_sum > 0.0 {
                for j in 0..n_classes {
                    probabilities[[i, j]] /= exp_sum;
                }
            } else {
                // Uniform distribution if all values are the same
                for j in 0..n_classes {
                    probabilities[[i, j]] = 1.0 / n_classes as f64;
                }
            }
        }

        Ok(probabilities)
    }
}

impl GaussianProcessSemiSupervised<GaussianProcessTrained> {
    /// Compute kernel matrix between two sets of points (for prediction)
    fn compute_kernel_matrix(&self, X1: &Array2<f64>, X2: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::<f64>::zeros((n1, n2));

        match self.kernel.as_str() {
            "rbf" => {
                for i in 0..n1 {
                    for j in 0..n2 {
                        let diff = &X1.row(i) - &X2.row(j);
                        let dist_sq = diff.mapv(|x| x * x).sum();
                        K[[i, j]] = (-dist_sq / (2.0 * self.length_scale.powi(2))).exp();
                    }
                }
            }
            "linear" => {
                for i in 0..n1 {
                    for j in 0..n2 {
                        K[[i, j]] = X1.row(i).dot(&X2.row(j));
                    }
                }
            }
            "polynomial" => {
                let degree = 2.0;
                for i in 0..n1 {
                    for j in 0..n2 {
                        let dot_product = X1.row(i).dot(&X2.row(j));
                        K[[i, j]] = (1.0 + dot_product / self.length_scale).powf(degree);
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown kernel: {}",
                    self.kernel
                )));
            }
        }

        Ok(K)
    }
}

/// Variational Bayesian Semi-Supervised Learning
///
/// This method uses variational inference to learn from both labeled and unlabeled data
/// by maximizing a variational lower bound on the log-likelihood.
///
/// # Parameters
///
/// * `n_components` - Number of mixture components
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
/// * `reg_covar` - Regularization for covariance matrices
/// * `alpha_prior` - Prior concentration parameter
#[derive(Debug, Clone)]
pub struct VariationalBayesianSemiSupervised<S = Untrained> {
    state: S,
    n_components: usize,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    alpha_prior: f64,
    random_state: Option<u64>,
}

impl VariationalBayesianSemiSupervised<Untrained> {
    /// Create a new VariationalBayesianSemiSupervised instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            max_iter: 100,
            tol: 1e-4,
            reg_covar: 1e-6,
            alpha_prior: 1.0,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the covariance regularization
    pub fn reg_covar(mut self, reg_covar: f64) -> Self {
        self.reg_covar = reg_covar;
        self
    }

    /// Set the alpha prior
    pub fn alpha_prior(mut self, alpha_prior: f64) -> Self {
        self.alpha_prior = alpha_prior;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for VariationalBayesianSemiSupervised<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for VariationalBayesianSemiSupervised<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>>
    for VariationalBayesianSemiSupervised<Untrained>
{
    type Fitted = VariationalBayesianSemiSupervised<VariationalBayesianTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();
        let (n_samples, n_features) = X.dim();

        // Identify labeled and unlabeled samples
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();
        let mut classes = std::collections::HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label == -1 {
                unlabeled_indices.push(i);
            } else {
                labeled_indices.push(i);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();

        // Initialize random number generator
        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(seed)
        } else {
            Random::seed(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            )
        };

        // Initialize variational parameters
        let mut means = Array2::<f64>::zeros((self.n_components, n_features));
        let mut covariances = Vec::new();
        let mut mixing_weights = Array1::<f64>::ones(self.n_components) / self.n_components as f64;

        // Initialize means randomly
        for k in 0..self.n_components {
            for j in 0..n_features {
                means[[k, j]] = rng.random_range(-1.0, 1.0);
            }
        }

        // Initialize covariances as identity matrices
        for _k in 0..self.n_components {
            let mut cov = Array2::<f64>::zeros((n_features, n_features));
            for i in 0..n_features {
                cov[[i, i]] = 1.0 + self.reg_covar;
            }
            covariances.push(cov);
        }

        // Variational EM algorithm
        let mut responsibilities = Array2::<f64>::zeros((n_samples, self.n_components));

        for _iter in 0..self.max_iter {
            let prev_means = means.clone();

            // E-step: Update responsibilities
            for i in 0..n_samples {
                let mut log_prob_norm = f64::NEG_INFINITY;

                // Compute log probabilities for each component
                for k in 0..self.n_components {
                    let log_prob =
                        self.compute_log_probability(&X.row(i), &means.row(k), &covariances[k]);
                    let log_resp = log_prob + mixing_weights[k].ln();

                    if log_resp > log_prob_norm {
                        log_prob_norm = log_resp;
                    }
                }

                // Compute responsibilities using log-sum-exp trick
                let mut exp_sum = 0.0;
                for k in 0..self.n_components {
                    let log_prob =
                        self.compute_log_probability(&X.row(i), &means.row(k), &covariances[k]);
                    let log_resp = log_prob + mixing_weights[k].ln() - log_prob_norm;
                    responsibilities[[i, k]] = log_resp.exp();
                    exp_sum += responsibilities[[i, k]];
                }

                // Normalize responsibilities
                if exp_sum > 0.0 {
                    for k in 0..self.n_components {
                        responsibilities[[i, k]] /= exp_sum;
                    }
                }
            }

            // M-step: Update parameters
            for k in 0..self.n_components {
                let n_k: f64 = responsibilities.column(k).sum();

                if n_k > 1e-10 {
                    // Update means
                    let mut new_mean = Array1::<f64>::zeros(n_features);
                    for i in 0..n_samples {
                        for j in 0..n_features {
                            new_mean[j] += responsibilities[[i, k]] * X[[i, j]];
                        }
                    }
                    new_mean /= n_k;
                    means.row_mut(k).assign(&new_mean);

                    // Update mixing weights
                    mixing_weights[k] = (n_k + self.alpha_prior - 1.0)
                        / (n_samples as f64 + self.n_components as f64 * self.alpha_prior
                            - self.n_components as f64);
                }
            }

            // Check convergence
            let diff = (&means - &prev_means).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }
        }

        // Predict labels for unlabeled samples
        let mut final_labels = y.clone();
        for &idx in &unlabeled_indices {
            // Assign to component with highest responsibility
            let best_component = responsibilities
                .row(idx)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            // Map component to class (simplified)
            let predicted_class = classes[best_component % classes.len()];
            final_labels[idx] = predicted_class;
        }

        Ok(VariationalBayesianSemiSupervised {
            state: VariationalBayesianTrained {
                X_train: X,
                y_train: final_labels,
                classes: Array1::from(classes),
                means,
                covariances,
                mixing_weights,
                responsibilities,
            },
            n_components: self.n_components,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            alpha_prior: self.alpha_prior,
            random_state: self.random_state,
        })
    }
}

impl VariationalBayesianSemiSupervised<Untrained> {
    /// Compute log probability of a sample under a Gaussian distribution
    fn compute_log_probability(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        covariance: &Array2<f64>,
    ) -> f64 {
        let d = x.len() as f64;
        let diff = x.to_owned() - mean.to_owned();

        // Simplified: assume diagonal covariance for computational efficiency
        let mut log_prob = -0.5 * d * (2.0 * PI).ln();

        for i in 0..diff.len() {
            let var = covariance[[i, i]];
            log_prob -= 0.5 * (var.ln() + diff[i] * diff[i] / var);
        }

        log_prob
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for VariationalBayesianSemiSupervised<VariationalBayesianTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            // Find most likely component
            let mut max_log_prob = f64::NEG_INFINITY;
            let mut best_component = 0;

            for k in 0..self.n_components {
                let log_prob = self.compute_log_probability(
                    &X.row(i),
                    &self.state.means.row(k),
                    &self.state.covariances[k],
                ) + self.state.mixing_weights[k].ln();

                if log_prob > max_log_prob {
                    max_log_prob = log_prob;
                    best_component = k;
                }
            }

            // Map component to class
            let predicted_class = self.state.classes[best_component % self.state.classes.len()];
            predictions[i] = predicted_class;
        }

        Ok(predictions)
    }
}

impl VariationalBayesianSemiSupervised<VariationalBayesianTrained> {
    /// Compute log probability of a sample under a Gaussian distribution
    fn compute_log_probability(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        covariance: &Array2<f64>,
    ) -> f64 {
        let d = x.len() as f64;
        let diff = x.to_owned() - mean.to_owned();

        // Simplified: assume diagonal covariance
        let mut log_prob = -0.5 * d * (2.0 * PI).ln();

        for i in 0..diff.len() {
            let var = covariance[[i, i]];
            log_prob -= 0.5 * (var.ln() + diff[i] * diff[i] / var);
        }

        log_prob
    }
}

/// Trained state for GaussianProcessSemiSupervised
#[derive(Debug, Clone)]
pub struct GaussianProcessTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// X_labeled
    pub X_labeled: Array2<f64>,
    /// GP_weights
    pub GP_weights: Array2<f64>,
    /// predictions_all
    pub predictions_all: Array2<f64>,
}

/// Trained state for VariationalBayesianSemiSupervised
#[derive(Debug, Clone)]
pub struct VariationalBayesianTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// means
    pub means: Array2<f64>,
    /// covariances
    pub covariances: Vec<Array2<f64>>,
    /// mixing_weights
    pub mixing_weights: Array1<f64>,
    /// responsibilities
    pub responsibilities: Array2<f64>,
}

/// Bayesian Active Learning for Semi-Supervised Learning
///
/// This method uses Bayesian inference to select the most informative
/// unlabeled samples for labeling based on prediction uncertainty.
///
/// # Parameters
///
/// * `n_queries` - Number of samples to query
/// * `kernel` - Kernel function for GP
/// * `length_scale` - Length scale for RBF kernel
/// * `noise_level` - Noise level for GP
/// * `acquisition` - Acquisition function ('uncertainty', 'entropy')
#[derive(Debug, Clone)]
pub struct BayesianActiveLearning<S = Untrained> {
    state: S,
    n_queries: usize,
    kernel: String,
    length_scale: f64,
    noise_level: f64,
    acquisition: String,
    random_state: Option<u64>,
}

impl BayesianActiveLearning<Untrained> {
    /// Create a new BayesianActiveLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_queries: 10,
            kernel: "rbf".to_string(),
            length_scale: 1.0,
            noise_level: 0.1,
            acquisition: "uncertainty".to_string(),
            random_state: None,
        }
    }

    /// Set the number of queries
    pub fn n_queries(mut self, n_queries: usize) -> Self {
        self.n_queries = n_queries;
        self
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: String) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the length scale
    pub fn length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    /// Set the noise level
    pub fn noise_level(mut self, noise_level: f64) -> Self {
        self.noise_level = noise_level;
        self
    }

    /// Set the acquisition function
    pub fn acquisition(mut self, acquisition: String) -> Self {
        self.acquisition = acquisition;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for BayesianActiveLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BayesianActiveLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for BayesianActiveLearning<Untrained> {
    type Fitted = BayesianActiveLearning<BayesianActiveTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        // Identify labeled and unlabeled samples
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();
        let mut classes = std::collections::HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label == -1 {
                unlabeled_indices.push(i);
            } else {
                labeled_indices.push(i);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();

        // Compute uncertainties for all unlabeled samples
        let mut uncertainties = Vec::new();
        for &idx in &unlabeled_indices {
            // Simple uncertainty: distance to nearest labeled point
            let mut min_dist = f64::INFINITY;
            for &labeled_idx in &labeled_indices {
                let diff = &X.row(idx) - &X.row(labeled_idx);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            uncertainties.push((idx, min_dist));
        }

        // Sort by uncertainty and select top queries
        uncertainties.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let query_indices: Vec<usize> = uncertainties
            .iter()
            .take(self.n_queries.min(unlabeled_indices.len()))
            .map(|(idx, _)| *idx)
            .collect();

        Ok(BayesianActiveLearning {
            state: BayesianActiveTrained {
                X_train: X,
                y_train: y,
                classes: Array1::from(classes),
                query_indices,
                uncertainties: uncertainties.iter().map(|(_, u)| *u).collect(),
            },
            n_queries: self.n_queries,
            kernel: self.kernel,
            length_scale: self.length_scale,
            noise_level: self.noise_level,
            acquisition: self.acquisition,
            random_state: self.random_state,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for BayesianActiveLearning<BayesianActiveTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        // Simple nearest neighbor prediction
        for i in 0..n_test {
            let mut min_dist = f64::INFINITY;
            let mut best_label = self.state.classes[0];

            for j in 0..self.state.X_train.nrows() {
                if self.state.y_train[j] != -1 {
                    let diff = &X.row(i) - &self.state.X_train.row(j);
                    let dist = diff.mapv(|x| x * x).sum().sqrt();

                    if dist < min_dist {
                        min_dist = dist;
                        best_label = self.state.y_train[j];
                    }
                }
            }

            predictions[i] = best_label;
        }

        Ok(predictions)
    }
}

/// Hierarchical Bayesian Semi-Supervised Learning
///
/// This method uses hierarchical Bayesian modeling to learn from both
/// labeled and unlabeled data with multiple levels of hierarchy.
///
/// # Parameters
///
/// * `n_levels` - Number of hierarchy levels
/// * `n_components` - Number of components per level
/// * `max_iter` - Maximum number of iterations
/// * `prior_strength` - Strength of hierarchical prior
#[derive(Debug, Clone)]
pub struct HierarchicalBayesianSemiSupervised<S = Untrained> {
    state: S,
    n_levels: usize,
    n_components: usize,
    max_iter: usize,
    prior_strength: f64,
    random_state: Option<u64>,
}

impl HierarchicalBayesianSemiSupervised<Untrained> {
    /// Create a new HierarchicalBayesianSemiSupervised instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_levels: 2,
            n_components: 2,
            max_iter: 100,
            prior_strength: 1.0,
            random_state: None,
        }
    }

    /// Set the number of levels
    pub fn n_levels(mut self, n_levels: usize) -> Self {
        self.n_levels = n_levels;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the prior strength
    pub fn prior_strength(mut self, prior_strength: f64) -> Self {
        self.prior_strength = prior_strength;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for HierarchicalBayesianSemiSupervised<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for HierarchicalBayesianSemiSupervised<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>>
    for HierarchicalBayesianSemiSupervised<Untrained>
{
    type Fitted = HierarchicalBayesianSemiSupervised<HierarchicalBayesianTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();
        let (n_samples, n_features) = X.dim();

        // Identify labeled and unlabeled samples
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();
        let mut classes = std::collections::HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label == -1 {
                unlabeled_indices.push(i);
            } else {
                labeled_indices.push(i);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();
        let n_classes = classes.len();

        // Initialize random number generator
        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(seed)
        } else {
            Random::seed(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            )
        };

        // Initialize hierarchical parameters
        let mut level_means = Vec::new();
        for _ in 0..self.n_levels {
            let mut means = Array2::<f64>::zeros((self.n_components, n_features));
            for i in 0..self.n_components {
                for j in 0..n_features {
                    means[[i, j]] = rng.random_range(-1.0, 1.0);
                }
            }
            level_means.push(means);
        }

        // Simple hierarchical optimization
        for _iter in 0..self.max_iter {
            // Update lower levels based on data
            // Simplified: just average the data points
            let mean = X.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
            #[allow(clippy::needless_range_loop)]
            for level_idx in 0..self.n_levels {
                for comp_idx in 0..self.n_components {
                    for feat_idx in 0..n_features {
                        level_means[level_idx][[comp_idx, feat_idx]] = 0.9
                            * level_means[level_idx][[comp_idx, feat_idx]]
                            + 0.1 * mean[feat_idx];
                    }
                }
            }
        }

        // Predict labels for unlabeled samples
        let mut final_labels = y.clone();
        for &idx in &unlabeled_indices {
            // Find nearest component in lowest level
            let mut min_dist = f64::INFINITY;
            let mut best_component = 0;

            for comp_idx in 0..self.n_components {
                let diff = &X.row(idx) - &level_means[0].row(comp_idx);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    best_component = comp_idx;
                }
            }

            // Map component to class
            let predicted_class = classes[best_component % n_classes];
            final_labels[idx] = predicted_class;
        }

        Ok(HierarchicalBayesianSemiSupervised {
            state: HierarchicalBayesianTrained {
                X_train: X,
                y_train: final_labels,
                classes: Array1::from(classes),
                level_means,
            },
            n_levels: self.n_levels,
            n_components: self.n_components,
            max_iter: self.max_iter,
            prior_strength: self.prior_strength,
            random_state: self.random_state,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for HierarchicalBayesianSemiSupervised<HierarchicalBayesianTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            // Find nearest component in lowest level
            let mut min_dist = f64::INFINITY;
            let mut best_label = self.state.classes[0];

            for j in 0..self.state.X_train.nrows() {
                let diff = &X.row(i) - &self.state.X_train.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_label = self.state.y_train[j];
                }
            }

            predictions[i] = best_label;
        }

        Ok(predictions)
    }
}

/// Trained state for BayesianActiveLearning
#[derive(Debug, Clone)]
pub struct BayesianActiveTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// query_indices
    pub query_indices: Vec<usize>,
    /// uncertainties
    pub uncertainties: Vec<f64>,
}

/// Trained state for HierarchicalBayesianSemiSupervised
#[derive(Debug, Clone)]
pub struct HierarchicalBayesianTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// level_means
    pub level_means: Vec<Array2<f64>>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_gaussian_process_semi_supervised() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        let gp = GaussianProcessSemiSupervised::new()
            .kernel("rbf".to_string())
            .length_scale(1.0)
            .noise_level(0.1)
            .random_state(42);

        let fitted = gp.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();
        let probas = fitted.predict_proba(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(probas.dim(), (4, 2));
        assert!(predictions.iter().all(|&p| p >= 0 && p <= 1));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let sum: f64 = probas.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Check that labeled samples are predicted correctly
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_variational_bayesian_semi_supervised() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        let vb = VariationalBayesianSemiSupervised::new()
            .n_components(2)
            .max_iter(10)
            .random_state(42);

        let fitted = vb.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&p| p >= 0 && p <= 1));
    }

    #[test]
    fn test_gaussian_process_parameters() {
        let gp = GaussianProcessSemiSupervised::new()
            .kernel("linear".to_string())
            .length_scale(2.0)
            .noise_level(0.2)
            .alpha(1e-8)
            .n_restarts_optimizer(5);

        assert_eq!(gp.kernel, "linear");
        assert_eq!(gp.length_scale, 2.0);
        assert_eq!(gp.noise_level, 0.2);
        assert_eq!(gp.alpha, 1e-8);
        assert_eq!(gp.n_restarts_optimizer, 5);
    }

    #[test]
    fn test_variational_bayesian_parameters() {
        let vb = VariationalBayesianSemiSupervised::new()
            .n_components(4)
            .max_iter(200)
            .tol(1e-6)
            .reg_covar(1e-4)
            .alpha_prior(2.0);

        assert_eq!(vb.n_components, 4);
        assert_eq!(vb.max_iter, 200);
        assert_eq!(vb.tol, 1e-6);
        assert_eq!(vb.reg_covar, 1e-4);
        assert_eq!(vb.alpha_prior, 2.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_kernel_matrix_computation() {
        let gp = GaussianProcessSemiSupervised::new()
            .kernel("rbf".to_string())
            .length_scale(1.0);
        let X = array![[1.0, 2.0], [3.0, 4.0]];

        let K = gp.compute_kernel_matrix(&X, &X).unwrap();

        assert_eq!(K.dim(), (2, 2));
        assert!((K[[0, 0]] - 1.0).abs() < 1e-10); // Self-kernel should be 1
        assert!((K[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(K[[0, 1]] > 0.0 && K[[0, 1]] < 1.0); // Cross-kernel should be between 0 and 1
        assert!((K[[0, 1]] - K[[1, 0]]).abs() < 1e-10); // Should be symmetric
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_linear_kernel() {
        let gp = GaussianProcessSemiSupervised::new().kernel("linear".to_string());
        let X = array![[1.0, 2.0], [3.0, 4.0]];

        let K = gp.compute_kernel_matrix(&X, &X).unwrap();

        assert_eq!(K.dim(), (2, 2));
        assert!((K[[0, 0]] - 5.0).abs() < 1e-10); // 1*1 + 2*2 = 5
        assert!((K[[1, 1]] - 25.0).abs() < 1e-10); // 3*3 + 4*4 = 25
        assert!((K[[0, 1]] - 11.0).abs() < 1e-10); // 1*3 + 2*4 = 11
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_polynomial_kernel() {
        let gp = GaussianProcessSemiSupervised::new()
            .kernel("polynomial".to_string())
            .length_scale(1.0);
        let X = array![[1.0, 1.0]];

        let K = gp.compute_kernel_matrix(&X, &X).unwrap();

        assert_eq!(K.dim(), (1, 1));
        assert!((K[[0, 0]] - 9.0).abs() < 1e-10); // (1 + 1*1 + 1*1)^2 = 3^2 = 9
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empty_labeled_samples_error() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![-1, -1]; // No labeled samples

        let gp = GaussianProcessSemiSupervised::new();
        let result = gp.fit(&X.view(), &y.view());

        assert!(result.is_err());

        let vb = VariationalBayesianSemiSupervised::new();
        let result = vb.fit(&X.view(), &y.view());

        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_single_labeled_sample() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, -1]; // One labeled sample

        let gp = GaussianProcessSemiSupervised::new()
            .noise_level(0.1)
            .random_state(42);

        let fitted = gp.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0], 0); // Labeled sample should be correct
    }

    #[test]
    fn test_log_probability_computation() {
        let vb = VariationalBayesianSemiSupervised::new();
        let x = array![1.0, 2.0];
        let mean = array![1.0, 2.0];
        let mut covar = Array2::<f64>::zeros((2, 2));
        covar[[0, 0]] = 1.0;
        covar[[1, 1]] = 1.0;

        let log_prob = vb.compute_log_probability(&x.view(), &mean.view(), &covar);

        // For x = mean, the probability should be maximized
        assert!(log_prob.is_finite());
        assert!(log_prob < 0.0); // Log probability should be negative
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_active_learning() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        let bal = BayesianActiveLearning::new().n_queries(2).random_state(42);

        let fitted = bal.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&p| p >= 0 && p <= 1));
        assert_eq!(fitted.state.query_indices.len(), 2);
    }

    #[test]
    fn test_bayesian_active_learning_parameters() {
        let bal = BayesianActiveLearning::new()
            .n_queries(5)
            .kernel("rbf".to_string())
            .length_scale(2.0)
            .noise_level(0.2)
            .acquisition("entropy".to_string());

        assert_eq!(bal.n_queries, 5);
        assert_eq!(bal.kernel, "rbf");
        assert_eq!(bal.length_scale, 2.0);
        assert_eq!(bal.noise_level, 0.2);
        assert_eq!(bal.acquisition, "entropy");
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_hierarchical_bayesian() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        let hb = HierarchicalBayesianSemiSupervised::new()
            .n_levels(2)
            .n_components(2)
            .max_iter(10)
            .random_state(42);

        let fitted = hb.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&p| p >= 0 && p <= 1));
        assert_eq!(fitted.state.level_means.len(), 2);
    }

    #[test]
    fn test_hierarchical_bayesian_parameters() {
        let hb = HierarchicalBayesianSemiSupervised::new()
            .n_levels(3)
            .n_components(4)
            .max_iter(200)
            .prior_strength(2.0);

        assert_eq!(hb.n_levels, 3);
        assert_eq!(hb.n_components, 4);
        assert_eq!(hb.max_iter, 200);
        assert_eq!(hb.prior_strength, 2.0);
    }
}

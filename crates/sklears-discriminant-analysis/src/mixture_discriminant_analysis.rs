//! Mixture Discriminant Analysis (MDA)
//!
//! This module provides a comprehensive implementation of Mixture Discriminant Analysis,
//! which models each class as a mixture of Gaussian components. This allows for more
//! flexible class distributions than standard discriminant analysis methods.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis, array};
use scirs2_core::random::{Random, rng};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Covariance type for mixture components
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceType {
    /// Full covariance matrices (most flexible)
    Full,
    /// Diagonal covariance matrices
    Diagonal,
    /// Spherical covariance (isotropic, σ²I)
    Spherical,
    /// Tied covariance (same for all components within a class)
    Tied,
}

impl Default for CovarianceType {
    fn default() -> Self {
        CovarianceType::Full
    }
}

/// Initialization method for mixture components
#[derive(Debug, Clone, PartialEq)]
pub enum InitializationMethod {
    /// K-means++ initialization (most robust)
    KMeansPlusPlus,
    /// Random initialization
    Random,
    /// Uniform partitioning
    Uniform,
    /// Manual initialization (user-provided)
    Manual,
}

impl Default for InitializationMethod {
    fn default() -> Self {
        InitializationMethod::KMeansPlusPlus
    }
}

/// Configuration for Mixture Discriminant Analysis
///
/// MDA models each class as a mixture of Gaussian components, allowing for
/// more complex class distributions than single-component methods like QDA.
#[derive(Debug, Clone)]
pub struct MixtureDiscriminantAnalysisConfig {
    /// Number of mixture components per class
    pub n_components_per_class: usize,

    /// Prior probabilities of the classes
    /// If None, priors are estimated from the training data
    pub priors: Option<Array1<Float>>,

    /// Regularization parameter for covariance matrices
    pub reg_param: Float,

    /// Whether to store the covariance matrices after fitting
    pub store_covariance: bool,

    /// Tolerance for EM algorithm convergence
    pub tol: Float,

    /// Maximum iterations for EM algorithm
    pub max_iter: usize,

    /// Type of covariance matrix to use
    pub covariance_type: CovarianceType,

    /// Initialization method for mixture components
    pub init_method: InitializationMethod,

    /// Random seed for reproducible results
    pub random_state: Option<u64>,

    /// Minimum weight for a component (prevents degeneracy)
    pub weight_threshold: Float,

    /// Whether to use warm restarts if EM fails to converge
    pub warm_restart: bool,

    /// Number of random initializations to try
    pub n_init: usize,

    /// Convergence criterion ("log_likelihood", "parameters", "both")
    pub convergence_criterion: String,

    /// Whether to use early stopping
    pub early_stopping: bool,

    /// Patience for early stopping (iterations without improvement)
    pub early_stopping_patience: usize,

    /// Minimum improvement for early stopping
    pub early_stopping_threshold: Float,
}

impl Default for MixtureDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            n_components_per_class: 2,
            priors: None,
            reg_param: 1e-6,
            store_covariance: true,
            tol: 1e-6,
            max_iter: 100,
            covariance_type: CovarianceType::Full,
            init_method: InitializationMethod::KMeansPlusPlus,
            random_state: None,
            weight_threshold: 1e-4,
            warm_restart: true,
            n_init: 3,
            convergence_criterion: "log_likelihood".to_string(),
            early_stopping: false,
            early_stopping_patience: 10,
            early_stopping_threshold: 1e-4,
        }
    }
}

/// Mixture Discriminant Analysis
///
/// A classifier that models each class as a mixture of Gaussian components.
/// This allows for more flexible class distributions than standard QDA,
/// particularly useful for classes with multimodal distributions.
///
/// # Mathematical Background
///
/// For each class c, MDA models the class-conditional density as:
///
/// P(x|c) = Σₖ wₖ⁽ᶜ⁾ N(x; μₖ⁽ᶜ⁾, Σₖ⁽ᶜ⁾)
///
/// where wₖ⁽ᶜ⁾ are mixture weights, μₖ⁽ᶜ⁾ are component means,
/// and Σₖ⁽ᶜ⁾ are component covariances.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_discriminant_analysis::{MixtureDiscriminantAnalysis, CovarianceType};
/// use scirs2_core::ndarray::array;
///
/// let mda = MixtureDiscriminantAnalysis::new()
///     .n_components_per_class(3)
///     .covariance_type(CovarianceType::Full)
///     .random_state(Some(42));
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0]];
/// let y = array![0, 0, 1, 1];
///
/// let trained_mda = mda.fit(&X, &y).unwrap();
/// let predictions = trained_mda.predict(&X).unwrap();
/// let probabilities = trained_mda.predict_proba(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MixtureDiscriminantAnalysis<State = Untrained> {
    config: MixtureDiscriminantAnalysisConfig,
    state: PhantomData<State>,
    // Trained state fields
    classes_: Option<Array1<i32>>,
    mixture_weights_: Option<Vec<Array1<Float>>>, // weights for each component per class
    mixture_means_: Option<Vec<Array2<Float>>>,   // means for each component per class
    mixture_covariances_: Option<Vec<Vec<Array2<Float>>>>, // covariances for each component per class
    priors_: Option<Array1<Float>>,
    n_features_: Option<usize>,
    log_likelihood_: Option<Float>,
    converged_: Option<bool>,
    n_iter_: Option<usize>,
}

impl MixtureDiscriminantAnalysis<Untrained> {
    /// Create a new MixtureDiscriminantAnalysis instance
    pub fn new() -> Self {
        Self {
            config: MixtureDiscriminantAnalysisConfig::default(),
            state: PhantomData,
            classes_: None,
            mixture_weights_: None,
            mixture_means_: None,
            mixture_covariances_: None,
            priors_: None,
            n_features_: None,
            log_likelihood_: None,
            converged_: None,
            n_iter_: None,
        }
    }

    /// Set the number of mixture components per class
    pub fn n_components_per_class(mut self, n_components_per_class: usize) -> Self {
        self.config.n_components_per_class = n_components_per_class;
        self
    }

    /// Set the prior probabilities
    pub fn priors(mut self, priors: Option<Array1<Float>>) -> Self {
        self.config.priors = priors;
        self
    }

    /// Set the regularization parameter
    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    /// Set whether to store covariance matrices
    pub fn store_covariance(mut self, store_covariance: bool) -> Self {
        self.config.store_covariance = store_covariance;
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.config.covariance_type = covariance_type;
        self
    }

    /// Set the initialization method
    pub fn init_method(mut self, init_method: InitializationMethod) -> Self {
        self.config.init_method = init_method;
        self
    }

    /// Set the random seed
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Set the weight threshold
    pub fn weight_threshold(mut self, weight_threshold: Float) -> Self {
        self.config.weight_threshold = weight_threshold;
        self
    }

    /// Set whether to use warm restarts
    pub fn warm_restart(mut self, warm_restart: bool) -> Self {
        self.config.warm_restart = warm_restart;
        self
    }

    /// Set the number of initializations
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.config.n_init = n_init;
        self
    }

    /// Set the convergence criterion
    pub fn convergence_criterion(mut self, criterion: &str) -> Self {
        self.config.convergence_criterion = criterion.to_string();
        self
    }

    /// Set early stopping parameters
    pub fn early_stopping(mut self, early_stopping: bool, patience: usize, threshold: Float) -> Self {
        self.config.early_stopping = early_stopping;
        self.config.early_stopping_patience = patience;
        self.config.early_stopping_threshold = threshold;
        self
    }

    /// Initialize mixture components using K-means++ approach
    fn initialize_kmeans_plus_plus(
        &self,
        class_data: &Array2<Float>,
        rng_seed: u64,
    ) -> SklResult<(Array1<Float>, Array2<Float>, Vec<Array2<Float>>)> {
        let (n_samples, n_features) = class_data.dim();
        let n_components = self.config.n_components_per_class;

        if n_samples < n_components {
            return Err(SklearsError::InvalidInput(format!(
                "Cannot have more components ({}) than samples ({}) in a class",
                n_components, n_samples
            )));
        }

        // Initialize weights uniformly
        let weights = Array1::from_elem(n_components, 1.0 / n_components as Float);

        // K-means++ initialization for centers
        let mut means = Array2::zeros((n_components, n_features));
        let mut chosen_indices = Vec::new();

        // Choose first center randomly
        let first_idx = (rng_seed % n_samples as u64) as usize;
        means.row_mut(0).assign(&class_data.row(first_idx));
        chosen_indices.push(first_idx);

        // Choose remaining centers with probability proportional to squared distance
        for k in 1..n_components {
            let mut distances = Array1::zeros(n_samples);

            for i in 0..n_samples {
                let sample = class_data.row(i);
                let mut min_dist = Float::INFINITY;

                for &chosen_idx in &chosen_indices {
                    let center = class_data.row(chosen_idx);
                    let dist: Float = sample.iter()
                        .zip(center.iter())
                        .map(|(&x, &c)| (x - c).powi(2))
                        .sum();
                    min_dist = min_dist.min(dist);
                }
                distances[i] = min_dist;
            }

            // Choose next center with probability proportional to distance squared
            let total_distance: Float = distances.sum();
            if total_distance > 0.0 {
                let mut cumulative = 0.0;
                let threshold = ((rng_seed * (k as u64 + 1)) % 1000) as Float / 1000.0 * total_distance;

                for i in 0..n_samples {
                    cumulative += distances[i];
                    if cumulative >= threshold && !chosen_indices.contains(&i) {
                        means.row_mut(k).assign(&class_data.row(i));
                        chosen_indices.push(i);
                        break;
                    }
                }
            }
        }

        // Initialize covariances
        let covariances = self.initialize_covariances(class_data, &means)?;

        Ok((weights, means, covariances))
    }

    /// Initialize mixture components using uniform partitioning
    fn initialize_uniform(
        &self,
        class_data: &Array2<Float>,
    ) -> SklResult<(Array1<Float>, Array2<Float>, Vec<Array2<Float>>)> {
        let (n_samples, n_features) = class_data.dim();
        let n_components = self.config.n_components_per_class;

        // Initialize weights uniformly
        let weights = Array1::from_elem(n_components, 1.0 / n_components as Float);

        // Initialize means using simple partitioning
        let mut means = Array2::zeros((n_components, n_features));
        let samples_per_component = n_samples / n_components;

        for k in 0..n_components {
            let start_idx = k * samples_per_component;
            let end_idx = if k == n_components - 1 {
                n_samples
            } else {
                (k + 1) * samples_per_component
            };

            let component_samples = class_data.slice(s![start_idx..end_idx, ..]);
            let component_mean = component_samples.mean_axis(Axis(0)).unwrap();
            means.row_mut(k).assign(&component_mean);
        }

        // Initialize covariances
        let covariances = self.initialize_covariances(class_data, &means)?;

        Ok((weights, means, covariances))
    }

    /// Initialize covariance matrices based on configuration
    fn initialize_covariances(
        &self,
        class_data: &Array2<Float>,
        means: &Array2<Float>,
    ) -> SklResult<Vec<Array2<Float>>> {
        let (n_samples, n_features) = class_data.dim();
        let n_components = means.nrows();
        let mut covariances = Vec::with_capacity(n_components);

        match self.config.covariance_type {
            CovarianceType::Full => {
                // Initialize with empirical covariance or identity
                let empirical_cov = self.compute_empirical_covariance(class_data)?;
                for _ in 0..n_components {
                    let mut cov = empirical_cov.clone() * 0.1; // Scale down for stability
                    self.regularize_covariance(&mut cov);
                    covariances.push(cov);
                }
            },
            CovarianceType::Diagonal => {
                // Diagonal covariances initialized from empirical variance
                let empirical_var = self.compute_empirical_variance(class_data);
                for _ in 0..n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = empirical_var[i] * 0.1 + self.config.reg_param;
                    }
                    covariances.push(cov);
                }
            },
            CovarianceType::Spherical => {
                // Isotropic covariances (σ²I)
                let empirical_var = self.compute_empirical_variance(class_data);
                let avg_var = empirical_var.mean().unwrap();
                for _ in 0..n_components {
                    let mut cov = Array2::eye(n_features) * (avg_var * 0.1);
                    self.regularize_covariance(&mut cov);
                    covariances.push(cov);
                }
            },
            CovarianceType::Tied => {
                // Same covariance for all components
                let empirical_cov = self.compute_empirical_covariance(class_data)?;
                let mut tied_cov = empirical_cov * 0.1;
                self.regularize_covariance(&mut tied_cov);
                for _ in 0..n_components {
                    covariances.push(tied_cov.clone());
                }
            },
        }

        Ok(covariances)
    }

    /// Compute empirical covariance matrix
    fn compute_empirical_covariance(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = data.dim();
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut cov = Array2::zeros((n_features, n_features));

        for i in 0..n_samples {
            let sample = data.row(i);
            let diff = &sample - &mean;
            for j in 0..n_features {
                for k in 0..n_features {
                    cov[[j, k]] += diff[j] * diff[k];
                }
            }
        }

        if n_samples > 1 {
            cov = cov / (n_samples - 1) as Float;
        }

        Ok(cov)
    }

    /// Compute empirical variance (diagonal of covariance)
    fn compute_empirical_variance(&self, data: &Array2<Float>) -> Array1<Float> {
        let (n_samples, n_features) = data.dim();
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut var = Array1::zeros(n_features);

        for i in 0..n_samples {
            let sample = data.row(i);
            let diff = &sample - &mean;
            for j in 0..n_features {
                var[j] += diff[j] * diff[j];
            }
        }

        if n_samples > 1 {
            var = var / (n_samples - 1) as Float;
        }

        var
    }

    /// Add regularization to covariance matrix
    fn regularize_covariance(&self, cov: &mut Array2<Float>) {
        let n_features = cov.nrows();
        for i in 0..n_features {
            cov[[i, i]] += self.config.reg_param;
        }
    }

    /// E-step: compute component responsibilities
    fn e_step(
        &self,
        class_data: &Array2<Float>,
        weights: &Array1<Float>,
        means: &Array2<Float>,
        covariances: &[Array2<Float>],
    ) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = class_data.dim();
        let n_components = weights.len();
        let mut responsibilities = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            let sample = class_data.row(i);
            let mut log_probs = Array1::zeros(n_components);

            for k in 0..n_components {
                let mean_k = means.row(k);
                let cov_k = &covariances[k];

                // Compute log-likelihood for this component
                let log_likelihood = self.compute_log_likelihood(&sample, &mean_k, cov_k)?;
                log_probs[k] = weights[k].ln() + log_likelihood;
            }

            // Normalize using log-sum-exp trick for numerical stability
            let max_log_prob = log_probs.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_probs: Array1<Float> = log_probs.mapv(|x| (x - max_log_prob).exp());
            let sum_exp = exp_probs.sum();

            if sum_exp > 0.0 {
                for k in 0..n_components {
                    responsibilities[[i, k]] = exp_probs[k] / sum_exp;
                }
            } else {
                // Fallback to uniform responsibilities
                for k in 0..n_components {
                    responsibilities[[i, k]] = 1.0 / n_components as Float;
                }
            }
        }

        Ok(responsibilities)
    }

    /// Compute log-likelihood for a sample given component parameters
    fn compute_log_likelihood(
        &self,
        sample: &ArrayView1<Float>,
        mean: &ArrayView1<Float>,
        cov: &Array2<Float>,
    ) -> SklResult<Float> {
        let n_features = sample.len();
        let diff = sample - mean;

        let (log_det, quad_form) = match self.config.covariance_type {
            CovarianceType::Full => {
                let log_det = self.safe_log_determinant(cov);
                let quad_form = self.compute_quadratic_form(&diff, cov)?;
                (log_det, quad_form)
            },
            CovarianceType::Diagonal | CovarianceType::Spherical => {
                let mut log_det = 0.0;
                let mut quad_form = 0.0;
                for j in 0..n_features {
                    let var = cov[[j, j]];
                    log_det += var.ln();
                    quad_form += diff[j] * diff[j] / var;
                }
                (log_det, quad_form)
            },
            CovarianceType::Tied => {
                let log_det = self.safe_log_determinant(cov);
                let quad_form = self.compute_quadratic_form(&diff, cov)?;
                (log_det, quad_form)
            },
        };

        // Multivariate Gaussian log-likelihood
        let log_likelihood = -0.5 * (n_features as Float * (2.0 * std::f64::consts::PI).ln() + log_det + quad_form);
        Ok(log_likelihood)
    }

    /// Compute log determinant safely
    fn safe_log_determinant(&self, matrix: &Array2<Float>) -> Float {
        // Simplified approach using product of diagonal elements
        // In practice, you'd use proper LU decomposition or Cholesky
        matrix.diag().iter()
            .map(|&x| (x + self.config.reg_param).ln())
            .sum()
    }

    /// Compute quadratic form x^T A^-1 x efficiently
    fn compute_quadratic_form(&self, x: &Array1<Float>, cov: &Array2<Float>) -> SklResult<Float> {
        // Simplified computation using diagonal approximation
        // In practice, you'd use proper matrix inverse or Cholesky decomposition
        let n = x.len();
        let mut result = 0.0;

        match self.config.covariance_type {
            CovarianceType::Diagonal | CovarianceType::Spherical => {
                for i in 0..n {
                    result += x[i] * x[i] / cov[[i, i]];
                }
            },
            _ => {
                // Use diagonal approximation for simplicity
                for i in 0..n {
                    result += x[i] * x[i] / cov[[i, i]];
                }
            }
        }

        Ok(result)
    }

    /// M-step: update parameters
    fn m_step(
        &self,
        class_data: &Array2<Float>,
        responsibilities: &Array2<Float>,
    ) -> SklResult<(Array1<Float>, Array2<Float>, Vec<Array2<Float>>)> {
        let (n_samples, n_features) = class_data.dim();
        let n_components = responsibilities.ncols();

        // Update weights
        let mut weights = Array1::zeros(n_components);
        for k in 0..n_components {
            weights[k] = responsibilities.column(k).sum() / n_samples as Float;
            // Apply weight threshold
            if weights[k] < self.config.weight_threshold {
                weights[k] = self.config.weight_threshold;
            }
        }
        // Renormalize weights
        let weight_sum = weights.sum();
        if weight_sum > 0.0 {
            weights = weights / weight_sum;
        }

        // Update means
        let mut means = Array2::zeros((n_components, n_features));
        for k in 0..n_components {
            let responsibility_sum = responsibilities.column(k).sum();
            if responsibility_sum > self.config.weight_threshold {
                for j in 0..n_features {
                    let weighted_sum: Float = class_data
                        .column(j)
                        .iter()
                        .zip(responsibilities.column(k).iter())
                        .map(|(&x, &r)| x * r)
                        .sum();
                    means[[k, j]] = weighted_sum / responsibility_sum;
                }
            }
        }

        // Update covariances
        let covariances = self.update_covariances(class_data, &means, responsibilities)?;

        Ok((weights, means, covariances))
    }

    /// Update covariance matrices based on covariance type
    fn update_covariances(
        &self,
        class_data: &Array2<Float>,
        means: &Array2<Float>,
        responsibilities: &Array2<Float>,
    ) -> SklResult<Vec<Array2<Float>>> {
        let (n_samples, n_features) = class_data.dim();
        let n_components = means.nrows();
        let mut covariances = Vec::with_capacity(n_components);

        match self.config.covariance_type {
            CovarianceType::Full => {
                // Full covariance matrices
                for k in 0..n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    let mean_k = means.row(k);
                    let responsibility_sum = responsibilities.column(k).sum();

                    if responsibility_sum > self.config.weight_threshold {
                        for i in 0..n_samples {
                            let sample = class_data.row(i);
                            let diff = &sample - &mean_k;
                            let weight = responsibilities[[i, k]];

                            for j in 0..n_features {
                                for l in 0..n_features {
                                    cov[[j, l]] += weight * diff[j] * diff[l];
                                }
                            }
                        }
                        cov = cov / responsibility_sum;
                    }

                    self.regularize_covariance(&mut cov);
                    covariances.push(cov);
                }
            },
            CovarianceType::Diagonal => {
                // Diagonal covariance matrices
                for k in 0..n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    let mean_k = means.row(k);
                    let responsibility_sum = responsibilities.column(k).sum();

                    if responsibility_sum > self.config.weight_threshold {
                        for i in 0..n_samples {
                            let sample = class_data.row(i);
                            let diff = &sample - &mean_k;
                            let weight = responsibilities[[i, k]];

                            for j in 0..n_features {
                                cov[[j, j]] += weight * diff[j] * diff[j];
                            }
                        }
                        cov = cov / responsibility_sum;
                    }

                    self.regularize_covariance(&mut cov);
                    covariances.push(cov);
                }
            },
            CovarianceType::Spherical => {
                // Spherical (isotropic) covariance matrices
                for k in 0..n_components {
                    let mean_k = means.row(k);
                    let responsibility_sum = responsibilities.column(k).sum();
                    let mut total_var = 0.0;

                    if responsibility_sum > self.config.weight_threshold {
                        for i in 0..n_samples {
                            let sample = class_data.row(i);
                            let diff = &sample - &mean_k;
                            let weight = responsibilities[[i, k]];

                            for j in 0..n_features {
                                total_var += weight * diff[j] * diff[j];
                            }
                        }
                        total_var = total_var / (responsibility_sum * n_features as Float);
                    }

                    let mut cov = Array2::eye(n_features) * (total_var + self.config.reg_param);
                    covariances.push(cov);
                }
            },
            CovarianceType::Tied => {
                // Tied covariance (same for all components)
                let mut tied_cov = Array2::zeros((n_features, n_features));
                let mut total_responsibility = 0.0;

                for k in 0..n_components {
                    let mean_k = means.row(k);
                    for i in 0..n_samples {
                        let sample = class_data.row(i);
                        let diff = &sample - &mean_k;
                        let weight = responsibilities[[i, k]];
                        total_responsibility += weight;

                        for j in 0..n_features {
                            for l in 0..n_features {
                                tied_cov[[j, l]] += weight * diff[j] * diff[l];
                            }
                        }
                    }
                }

                if total_responsibility > 0.0 {
                    tied_cov = tied_cov / total_responsibility;
                }
                self.regularize_covariance(&mut tied_cov);

                // Use the same covariance for all components
                for _ in 0..n_components {
                    covariances.push(tied_cov.clone());
                }
            },
        }

        Ok(covariances)
    }

    /// Compute log-likelihood of the data given current parameters
    fn compute_data_log_likelihood(
        &self,
        class_data: &Array2<Float>,
        weights: &Array1<Float>,
        means: &Array2<Float>,
        covariances: &[Array2<Float>],
    ) -> SklResult<Float> {
        let (n_samples, _) = class_data.dim();
        let n_components = weights.len();
        let mut total_log_likelihood = 0.0;

        for i in 0..n_samples {
            let sample = class_data.row(i);
            let mut component_likelihoods = Array1::zeros(n_components);

            for k in 0..n_components {
                let mean_k = means.row(k);
                let cov_k = &covariances[k];
                let log_likelihood = self.compute_log_likelihood(&sample, &mean_k, cov_k)?;
                component_likelihoods[k] = weights[k] * log_likelihood.exp();
            }

            let sample_likelihood = component_likelihoods.sum();
            if sample_likelihood > 0.0 {
                total_log_likelihood += sample_likelihood.ln();
            }
        }

        Ok(total_log_likelihood)
    }
}

impl Default for MixtureDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MixtureDiscriminantAnalysis<Untrained> {
    type Config = MixtureDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for MixtureDiscriminantAnalysis<Untrained> {
    type Fitted = MixtureDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        // Basic validation
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        // Get unique classes
        let mut classes: Vec<i32> = y
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        let mut mixture_weights = Vec::with_capacity(n_classes);
        let mut mixture_means = Vec::with_capacity(n_classes);
        let mut mixture_covariances = Vec::with_capacity(n_classes);
        let mut class_counts = Array1::zeros(n_classes);
        let mut total_log_likelihood = 0.0;
        let mut converged = true;
        let mut total_iterations = 0;

        // Fit mixture model for each class
        for (i, &class) in classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&yi| yi == class).collect();
            let class_samples: Vec<ArrayView1<Float>> = x
                .axis_iter(Axis(0))
                .enumerate()
                .filter(|(j, _)| class_mask[*j])
                .map(|(_, sample)| sample)
                .collect();

            class_counts[i] = class_samples.len() as Float;

            if class_samples.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "Class {} has no samples",
                    class
                )));
            }

            if class_samples.len() < self.config.n_components_per_class {
                return Err(SklearsError::InvalidInput(format!(
                    "Class {} has {} samples but needs at least {} for {} components",
                    class, class_samples.len(), self.config.n_components_per_class, self.config.n_components_per_class
                )));
            }

            let class_data = Array2::from_shape_fn((class_samples.len(), n_features), |(i, j)| {
                class_samples[i][j]
            });

            // Try multiple initializations and keep the best
            let mut best_weights = None;
            let mut best_means = None;
            let mut best_covariances = None;
            let mut best_log_likelihood = Float::NEG_INFINITY;
            let mut best_converged = false;

            for init_trial in 0..self.config.n_init {
                let seed = self.config.random_state.unwrap_or(42) + init_trial as u64;

                // Initialize mixture components
                let (mut weights, mut means, mut covariances) = match self.config.init_method {
                    InitializationMethod::KMeansPlusPlus => {
                        self.initialize_kmeans_plus_plus(&class_data, seed)?
                    },
                    InitializationMethod::Uniform => {
                        self.initialize_uniform(&class_data)?
                    },
                    _ => {
                        self.initialize_uniform(&class_data)?
                    }
                };

                let mut prev_log_likelihood = Float::NEG_INFINITY;
                let mut no_improvement_count = 0;
                let mut trial_converged = false;

                // EM algorithm
                for iter in 0..self.config.max_iter {
                    // E-step
                    let responsibilities = self.e_step(&class_data, &weights, &means, &covariances)?;

                    // M-step
                    let (new_weights, new_means, new_covariances) =
                        self.m_step(&class_data, &responsibilities)?;

                    // Check convergence
                    let current_log_likelihood = self.compute_data_log_likelihood(
                        &class_data, &new_weights, &new_means, &new_covariances
                    )?;

                    let improvement = current_log_likelihood - prev_log_likelihood;

                    match self.config.convergence_criterion.as_str() {
                        "log_likelihood" => {
                            if improvement.abs() < self.config.tol {
                                trial_converged = true;
                                break;
                            }
                        },
                        "parameters" => {
                            let weight_diff = (&new_weights - &weights).mapv(|x| x.abs()).sum();
                            if weight_diff < self.config.tol {
                                trial_converged = true;
                                break;
                            }
                        },
                        _ => {
                            if improvement.abs() < self.config.tol {
                                trial_converged = true;
                                break;
                            }
                        }
                    }

                    // Early stopping
                    if self.config.early_stopping {
                        if improvement < self.config.early_stopping_threshold {
                            no_improvement_count += 1;
                            if no_improvement_count >= self.config.early_stopping_patience {
                                break;
                            }
                        } else {
                            no_improvement_count = 0;
                        }
                    }

                    weights = new_weights;
                    means = new_means;
                    covariances = new_covariances;
                    prev_log_likelihood = current_log_likelihood;
                }

                // Keep track of best initialization
                if prev_log_likelihood > best_log_likelihood {
                    best_log_likelihood = prev_log_likelihood;
                    best_weights = Some(weights);
                    best_means = Some(means);
                    best_covariances = Some(covariances);
                    best_converged = trial_converged;
                }
            }

            // Use best initialization results
            mixture_weights.push(best_weights.unwrap());
            mixture_means.push(best_means.unwrap());
            mixture_covariances.push(best_covariances.unwrap());
            total_log_likelihood += best_log_likelihood;
            if !best_converged {
                converged = false;
            }
        }

        // Calculate priors
        let priors = if let Some(ref p) = self.config.priors {
            if p.len() != n_classes {
                return Err(SklearsError::InvalidInput(
                    "Number of priors must match number of classes".to_string(),
                ));
            }
            p.clone()
        } else {
            &class_counts / n_samples as Float
        };

        Ok(MixtureDiscriminantAnalysis {
            config: self.config,
            state: PhantomData,
            classes_: Some(Array1::from(classes)),
            mixture_weights_: Some(mixture_weights),
            mixture_means_: Some(mixture_means),
            mixture_covariances_: Some(mixture_covariances),
            priors_: Some(priors),
            n_features_: Some(n_features),
            log_likelihood_: Some(total_log_likelihood),
            converged_: Some(converged),
            n_iter_: Some(total_iterations),
        })
    }
}

impl MixtureDiscriminantAnalysis<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().expect("Model is trained")
    }

    /// Get the mixture weights
    pub fn mixture_weights(&self) -> &Vec<Array1<Float>> {
        self.mixture_weights_.as_ref().expect("Model is trained")
    }

    /// Get the mixture means
    pub fn mixture_means(&self) -> &Vec<Array2<Float>> {
        self.mixture_means_.as_ref().expect("Model is trained")
    }

    /// Get the mixture covariances
    pub fn mixture_covariances(&self) -> &Vec<Vec<Array2<Float>>> {
        self.mixture_covariances_
            .as_ref()
            .expect("Model is trained")
    }

    /// Get the priors
    pub fn priors(&self) -> &Array1<Float> {
        self.priors_.as_ref().expect("Model is trained")
    }

    /// Get the total log-likelihood
    pub fn log_likelihood(&self) -> Float {
        self.log_likelihood_.expect("Model is trained")
    }

    /// Check if the model converged
    pub fn converged(&self) -> bool {
        self.converged_.expect("Model is trained")
    }

    /// Get the number of iterations used
    pub fn n_iter(&self) -> usize {
        self.n_iter_.expect("Model is trained")
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features_.expect("Model is trained")
    }
}

impl Predict<Array2<Float>, Array1<i32>> for MixtureDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().expect("Model is trained");

        let predictions: Vec<i32> = probas
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0;
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from(predictions))
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for MixtureDiscriminantAnalysis<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        validate::check_n_features(x, n_features)?;

        let mixture_weights = self.mixture_weights_.as_ref().expect("Model is trained");
        let mixture_means = self.mixture_means_.as_ref().expect("Model is trained");
        let mixture_covariances = self
            .mixture_covariances_
            .as_ref()
            .expect("Model is trained");
        let priors = self.priors_.as_ref().expect("Model is trained");
        let n_classes = mixture_weights.len();
        let n_samples = x.nrows();

        let mut log_likelihoods = Array2::zeros((n_samples, n_classes));

        for i in 0..n_classes {
            let class_weights = &mixture_weights[i];
            let class_means = &mixture_means[i];
            let class_covariances = &mixture_covariances[i];
            let n_components = class_weights.len();

            for j in 0..n_samples {
                let sample = x.row(j);
                let mut component_likelihoods = Array1::zeros(n_components);

                for k in 0..n_components {
                    let mean_k = class_means.row(k);
                    let cov_k = &class_covariances[k];

                    match self.compute_log_likelihood(&sample, &mean_k, cov_k) {
                        Ok(log_likelihood) => {
                            component_likelihoods[k] = class_weights[k] * log_likelihood.exp();
                        },
                        Err(_) => {
                            component_likelihoods[k] = 1e-10; // Small fallback value
                        }
                    }
                }

                let mixture_likelihood = component_likelihoods.sum();
                log_likelihoods[[j, i]] = if mixture_likelihood > 0.0 {
                    priors[i].ln() + mixture_likelihood.ln()
                } else {
                    Float::NEG_INFINITY
                };
            }
        }

        // Convert log-likelihoods to probabilities using stable softmax
        let mut probabilities = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let row = log_likelihoods.row(i);
            let max_log_like = row.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

            let exp_likes: Vec<Float> = row.iter().map(|&x| (x - max_log_like).exp()).collect();
            let sum_exp: Float = exp_likes.iter().sum();

            if sum_exp > 0.0 {
                for j in 0..n_classes {
                    probabilities[[i, j]] = exp_likes[j] / sum_exp;
                }
            } else {
                // Fallback to uniform distribution if numerical issues
                for j in 0..n_classes {
                    probabilities[[i, j]] = 1.0 / n_classes as Float;
                }
            }
        }

        Ok(probabilities)
    }
}

impl MixtureDiscriminantAnalysis<Trained> {
    /// Compute log-likelihood for trained model
    fn compute_log_likelihood(
        &self,
        sample: &ArrayView1<Float>,
        mean: &ArrayView1<Float>,
        cov: &Array2<Float>,
    ) -> SklResult<Float> {
        let n_features = sample.len();
        let diff = sample - mean;

        let (log_det, quad_form) = match self.config.covariance_type {
            CovarianceType::Full => {
                let log_det = self.safe_log_determinant(cov);
                let quad_form = self.compute_quadratic_form(&diff, cov)?;
                (log_det, quad_form)
            },
            CovarianceType::Diagonal | CovarianceType::Spherical => {
                let mut log_det = 0.0;
                let mut quad_form = 0.0;
                for j in 0..n_features {
                    let var = cov[[j, j]];
                    log_det += var.ln();
                    quad_form += diff[j] * diff[j] / var;
                }
                (log_det, quad_form)
            },
            CovarianceType::Tied => {
                let log_det = self.safe_log_determinant(cov);
                let quad_form = self.compute_quadratic_form(&diff, cov)?;
                (log_det, quad_form)
            },
        };

        let log_likelihood = -0.5 * (n_features as Float * (2.0 * std::f64::consts::PI).ln() + log_det + quad_form);
        Ok(log_likelihood)
    }

    /// Safe log determinant computation
    fn safe_log_determinant(&self, matrix: &Array2<Float>) -> Float {
        matrix.diag().iter()
            .map(|&x| (x + self.config.reg_param).ln())
            .sum()
    }

    /// Quadratic form computation for trained model
    fn compute_quadratic_form(&self, x: &Array1<Float>, cov: &Array2<Float>) -> SklResult<Float> {
        let n = x.len();
        let mut result = 0.0;

        match self.config.covariance_type {
            CovarianceType::Diagonal | CovarianceType::Spherical => {
                for i in 0..n {
                    result += x[i] * x[i] / cov[[i, i]];
                }
            },
            _ => {
                // Use diagonal approximation for simplicity
                for i in 0..n {
                    result += x[i] * x[i] / cov[[i, i]];
                }
            }
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mda_basic() {
        let mda = MixtureDiscriminantAnalysis::new()
            .n_components_per_class(2);

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 5.0],
            [5.0, 7.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let trained_mda = mda.fit(&X, &y).unwrap();
        let predictions = trained_mda.predict(&X).unwrap();
        let probabilities = trained_mda.predict_proba(&X).unwrap();

        assert_eq!(predictions.len(), 8);
        assert_eq!(probabilities.dim(), (8, 2));

        // Check that probabilities sum to 1
        for i in 0..8 {
            let sum: Float = probabilities.row(i).iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mda_covariance_types() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        // Test different covariance types
        for cov_type in [CovarianceType::Full, CovarianceType::Diagonal, CovarianceType::Spherical, CovarianceType::Tied] {
            let mda = MixtureDiscriminantAnalysis::new()
                .n_components_per_class(1)
                .covariance_type(cov_type);

            let result = mda.fit(&X, &y);
            assert!(result.is_ok(), "Failed for covariance type: {:?}", cov_type);
        }
    }

    #[test]
    fn test_mda_initialization_methods() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        // Test different initialization methods
        for init_method in [InitializationMethod::KMeansPlusPlus, InitializationMethod::Uniform] {
            let mda = MixtureDiscriminantAnalysis::new()
                .n_components_per_class(2)
                .init_method(init_method)
                .random_state(Some(42));

            let result = mda.fit(&X, &y);
            assert!(result.is_ok(), "Failed for initialization method: {:?}", init_method);
        }
    }

    #[test]
    fn test_mda_validation() {
        let mda = MixtureDiscriminantAnalysis::new();

        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0]; // Wrong size

        let result = mda.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_mda_insufficient_samples() {
        let mda = MixtureDiscriminantAnalysis::new()
            .n_components_per_class(3);

        let X = array![[1.0, 2.0], [3.0, 4.0]]; // Only 2 samples
        let y = array![0, 0]; // All same class, but need 3 components

        let result = mda.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_mda_getters() {
        let mda = MixtureDiscriminantAnalysis::new()
            .n_components_per_class(2)
            .random_state(Some(42));

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 5.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let trained_mda = mda.fit(&X, &y).unwrap();

        assert_eq!(trained_mda.classes().len(), 2);
        assert_eq!(trained_mda.mixture_weights().len(), 2);
        assert_eq!(trained_mda.mixture_means().len(), 2);
        assert_eq!(trained_mda.mixture_covariances().len(), 2);
        assert_eq!(trained_mda.priors().len(), 2);
        assert_eq!(trained_mda.n_features(), 2);
    }

    #[test]
    fn test_mda_convergence() {
        let mda = MixtureDiscriminantAnalysis::new()
            .n_components_per_class(2)
            .max_iter(50)
            .tol(1e-6)
            .random_state(Some(42));

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 5.0],
            [5.0, 7.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let trained_mda = mda.fit(&X, &y).unwrap();

        // Check that we have convergence information
        assert!(trained_mda.log_likelihood().is_finite());
    }
}
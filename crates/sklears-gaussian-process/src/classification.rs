//! Gaussian Process Classification Models
//!
//! This module provides different implementations of Gaussian Process classifiers:
//! - `GaussianProcessClassifier`: Binary classification using Laplace approximation
//! - `MultiClassGaussianProcessClassifier`: Multi-class classification using One-vs-Rest strategy
//! - `ExpectationPropagationGaussianProcessClassifier`: EP-based classification
//!
//! All classifiers support different kernel functions and provide probability estimates.

use std::collections::HashSet;
use std::f64::consts::PI;

// Use ndarray imports (following existing pattern in the codebase)
// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
};

use crate::kernels::Kernel;
use crate::utils;

///
/// let X = array![[1.0], [2.0], [3.0], [4.0]];
/// let y = array![0, 0, 1, 1];
///
/// let kernel = RBF::new(1.0);
/// let gpc = GaussianProcessClassifier::new().kernel(Box::new(kernel));
/// let fitted = gpc.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GaussianProcessClassifier<S = Untrained> {
    state: S,
    kernel: Option<Box<dyn Kernel>>,
    optimizer: Option<String>,
    n_restarts_optimizer: usize,
    max_iter_predict: usize,
    warm_start: bool,
    copy_x_train: bool,
    random_state: Option<u64>,
    config: GpcConfig,
}

/// Trained state for Gaussian Process Classifier
#[derive(Debug, Clone)]
pub struct GpcTrained {
    /// X_train
    pub X_train: Option<Array2<f64>>, // Training inputs
    /// y_train
    pub y_train: Array1<i32>, // Training labels
    /// classes
    pub classes: Array1<i32>, // Unique classes
    /// pi
    pub pi: Array1<f64>, // Approximate posterior probabilities
    /// W_sr
    pub W_sr: Array1<f64>, // sqrt(W) where W is the diagonal Hessian
    /// L
    pub L: Array2<f64>, // Cholesky decomposition
    /// K
    pub K: Array2<f64>, // Kernel matrix
    /// f
    pub f: Array1<f64>, // Latent function values
    /// kernel
    pub kernel: Box<dyn Kernel>, // Kernel function
    /// log_marginal_likelihood_value
    pub log_marginal_likelihood_value: f64, // Log marginal likelihood
}

impl GaussianProcessClassifier<Untrained> {
    /// Create a new GaussianProcessClassifier instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: None,
            optimizer: Some("fmin_l_bfgs_b".to_string()),
            n_restarts_optimizer: 0,
            max_iter_predict: 100,
            warm_start: false,
            copy_x_train: true,
            random_state: None,
            config: GpcConfig::default(),
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the optimizer
    pub fn optimizer(mut self, optimizer: Option<String>) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set the number of optimizer restarts
    pub fn n_restarts_optimizer(mut self, n_restarts: usize) -> Self {
        self.n_restarts_optimizer = n_restarts;
        self
    }

    /// Set the maximum number of iterations for prediction
    pub fn max_iter_predict(mut self, max_iter: usize) -> Self {
        self.max_iter_predict = max_iter;
        self
    }

    /// Set whether to use warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Set whether to copy X during training
    pub fn copy_x_train(mut self, copy_x_train: bool) -> Self {
        self.copy_x_train = copy_x_train;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

/// Configuration for Gaussian Process Classifier
#[derive(Debug, Clone)]
pub struct GpcConfig {
    /// kernel_name
    pub kernel_name: String,
    /// optimizer
    pub optimizer: Option<String>,
    /// n_restarts_optimizer
    pub n_restarts_optimizer: usize,
    /// max_iter_predict
    pub max_iter_predict: usize,
    /// warm_start
    pub warm_start: bool,
    /// copy_x_train
    pub copy_x_train: bool,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for GpcConfig {
    fn default() -> Self {
        Self {
            kernel_name: "RBF".to_string(),
            optimizer: Some("fmin_l_bfgs_b".to_string()),
            n_restarts_optimizer: 0,
            max_iter_predict: 100,
            warm_start: false,
            copy_x_train: true,
            random_state: None,
        }
    }
}

impl Estimator for GaussianProcessClassifier<Untrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for GaussianProcessClassifier<GpcTrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for GaussianProcessClassifier<Untrained> {
    type Fitted = GaussianProcessClassifier<GpcTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<f64>, y: &ArrayView1<i32>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let kernel = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?
            .clone();

        // Get unique classes
        let mut classes_set: HashSet<i32> = HashSet::new();
        for &label in y.iter() {
            classes_set.insert(label);
        }

        if classes_set.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Binary classification requires exactly 2 classes".to_string(),
            ));
        }

        let mut classes: Vec<i32> = classes_set.into_iter().collect();
        classes.sort();
        let classes = Array1::from(classes);

        // Convert labels to {-1, 1}
        let y_binary = y.mapv(|label| if label == classes[0] { -1.0 } else { 1.0 });

        // Compute kernel matrix
        let X_owned = X.to_owned();
        let K = kernel.compute_kernel_matrix(&X_owned, None)?;

        // Laplace approximation
        let (f, pi, W_sr, L, log_marginal_likelihood_value) =
            laplace_approximation(&K, &y_binary, self.max_iter_predict)?;

        let X_train = if self.copy_x_train {
            Some(X.to_owned())
        } else {
            None
        };

        Ok(GaussianProcessClassifier {
            state: GpcTrained {
                X_train,
                y_train: y.to_owned(),
                classes,
                pi,
                W_sr,
                L,
                K,
                f,
                kernel,
                log_marginal_likelihood_value,
            },
            kernel: None,
            optimizer: self.optimizer,
            n_restarts_optimizer: self.n_restarts_optimizer,
            max_iter_predict: self.max_iter_predict,
            warm_start: self.warm_start,
            copy_x_train: self.copy_x_train,
            random_state: self.random_state,
            config: self.config,
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>> for GaussianProcessClassifier<GpcTrained> {
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let predictions: Vec<i32> = probabilities
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                self.state.classes[max_idx]
            })
            .collect();
        Ok(Array1::from(predictions))
    }
}

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>> for GaussianProcessClassifier<GpcTrained> {
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let X_train =
            self.state.X_train.as_ref().ok_or_else(|| {
                SklearsError::InvalidInput("Training data not available".to_string())
            })?;

        // Compute kernel between test and training points
        let X_test_owned = X.to_owned();
        let K_star = self
            .state
            .kernel
            .compute_kernel_matrix(X_train, Some(&X_test_owned))?;

        // Predict latent function values
        let f_star =
            predict_latent_function(&K_star, &self.state.f, &self.state.W_sr, &self.state.L)?;

        // Convert to probabilities using sigmoid
        let mut probabilities = Array2::<f64>::zeros((X.nrows(), 2));
        for (i, &f_val) in f_star.iter().enumerate() {
            let prob_positive = sigmoid(f_val);
            probabilities[[i, 0]] = 1.0 - prob_positive; // Probability of class 0
            probabilities[[i, 1]] = prob_positive; // Probability of class 1
        }

        Ok(probabilities)
    }
}

impl GaussianProcessClassifier<GpcTrained> {
    /// Get the log marginal likelihood
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.state.log_marginal_likelihood_value
    }

    /// Get the unique classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.state.classes
    }
}

/// Sigmoid function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of sigmoid function
pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Laplace approximation for binary classification
#[allow(non_snake_case)]
fn laplace_approximation(
    K: &Array2<f64>,
    y: &Array1<f64>,
    max_iter: usize,
) -> SklResult<(Array1<f64>, Array1<f64>, Array1<f64>, Array2<f64>, f64)> {
    let n = K.nrows();
    let mut f = Array1::<f64>::zeros(n);
    let tol = 1e-6;

    for _iter in 0..max_iter {
        // Compute probabilities and their derivatives
        let pi = f.mapv(sigmoid);
        let W = f.mapv(sigmoid_derivative);
        let _W_sr = W.mapv(|w| w.sqrt());

        // Compute gradients
        let grad = &pi - y;

        // Newton's method update
        // (K^{-1} + W)^{-1} * grad
        let mut K_W = K.clone();
        for i in 0..n {
            K_W[[i, i]] += W[i];
        }

        let L = utils::robust_cholesky(&K_W)?;
        let delta_f = utils::triangular_solve(&L, &grad)?;

        let f_new = &f - &delta_f;

        // Check convergence
        let diff = (&f_new - &f).mapv(|x| x.abs()).sum();
        f = f_new;

        if diff < tol {
            break;
        }
    }

    // Final computation
    let pi = f.mapv(sigmoid);
    let W = f.mapv(sigmoid_derivative);
    let W_sr = W.mapv(|w| w.sqrt());

    // Compute final Cholesky decomposition
    let mut K_W = K.clone();
    for i in 0..n {
        K_W[[i, i]] += W[i];
    }
    let L = utils::robust_cholesky(&K_W)?;

    // Compute log marginal likelihood
    let log_marginal_likelihood = {
        let log_det = 2.0 * L.diag().mapv(|x| x.ln()).sum();
        let quadratic: f64 = f
            .iter()
            .zip(y.iter())
            .map(|(&f_i, &y_i)| y_i * f_i - (1.0 + f_i.exp()).ln())
            .sum();
        quadratic - 0.5 * log_det
    };

    Ok((f, pi, W_sr, L, log_marginal_likelihood))
}

/// Predict latent function values
fn predict_latent_function(
    K_star: &Array2<f64>,
    f_train: &Array1<f64>,
    _W_sr: &Array1<f64>,
    _L: &Array2<f64>,
) -> SklResult<Array1<f64>> {
    // Simplified prediction (should be more sophisticated)
    let f_star_values = K_star.dot(f_train);
    Ok(f_star_values)
}

impl Default for GaussianProcessClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-class Gaussian Process Classifier using One-vs-Rest strategy
///
/// This implementation extends binary Gaussian Process Classification to handle
/// multi-class problems by training one binary classifier per class against all others.
/// Each binary classifier uses the Laplace approximation for inference.
///
/// # Examples
///
/// ```
/// use sklears_gaussian_process::{MultiClassGaussianProcessClassifier, kernels::RBF};
/// use sklears_core::traits::{Fit, Predict};
/// // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
/// let y = array![0, 0, 1, 1, 2, 2];
///
/// let kernel = RBF::new(1.0);
/// let mc_gpc = MultiClassGaussianProcessClassifier::new()
///     .kernel(Box::new(kernel));
/// let fitted = mc_gpc.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiClassGaussianProcessClassifier<S = Untrained> {
    state: S,
    kernel: Option<Box<dyn Kernel>>,
    optimizer: Option<String>,
    n_restarts_optimizer: usize,
    max_iter_predict: usize,
    warm_start: bool,
    copy_x_train: bool,
    random_state: Option<u64>,
    config: GpcConfig,
}

/// Trained state for Multi-class Gaussian Process Classifier
#[derive(Debug, Clone)]
pub struct McGpcTrained {
    /// X_train
    pub X_train: Option<Array2<f64>>, // Training inputs
    /// y_train
    pub y_train: Array1<i32>, // Training labels
    /// classes
    pub classes: Array1<i32>, // Unique classes
    /// binary_classifiers
    pub binary_classifiers: Vec<GaussianProcessClassifier<GpcTrained>>, // One-vs-rest classifiers
    /// n_classes
    pub n_classes: usize, // Number of classes
}

impl MultiClassGaussianProcessClassifier<Untrained> {
    /// Create a new MultiClassGaussianProcessClassifier instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: None,
            optimizer: Some("fmin_l_bfgs_b".to_string()),
            n_restarts_optimizer: 0,
            max_iter_predict: 100,
            warm_start: false,
            copy_x_train: true,
            random_state: None,
            config: GpcConfig::default(),
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the optimizer
    pub fn optimizer(mut self, optimizer: Option<String>) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set the number of optimizer restarts
    pub fn n_restarts_optimizer(mut self, n_restarts: usize) -> Self {
        self.n_restarts_optimizer = n_restarts;
        self
    }

    /// Set the maximum number of iterations for prediction
    pub fn max_iter_predict(mut self, max_iter: usize) -> Self {
        self.max_iter_predict = max_iter;
        self
    }

    /// Set whether to use warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Set whether to copy X during training
    pub fn copy_x_train(mut self, copy_x_train: bool) -> Self {
        self.copy_x_train = copy_x_train;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Estimator for MultiClassGaussianProcessClassifier<Untrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for MultiClassGaussianProcessClassifier<McGpcTrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>>
    for MultiClassGaussianProcessClassifier<Untrained>
{
    type Fitted = MultiClassGaussianProcessClassifier<McGpcTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<f64>, y: &ArrayView1<i32>) -> SklResult<Self::Fitted> {
        let kernel = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?
            .clone();

        // Get unique classes
        let mut classes_set: HashSet<i32> = HashSet::new();
        for &label in y.iter() {
            classes_set.insert(label);
        }

        if classes_set.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Multi-class classification requires at least 2 classes".to_string(),
            ));
        }

        let mut classes: Vec<i32> = classes_set.into_iter().collect();
        classes.sort();
        let classes = Array1::from(classes);
        let n_classes = classes.len();

        // Handle binary case by delegating to binary classifier
        if n_classes == 2 {
            let binary_gpc = GaussianProcessClassifier::new()
                .kernel(kernel)
                .optimizer(self.optimizer.clone())
                .n_restarts_optimizer(self.n_restarts_optimizer)
                .max_iter_predict(self.max_iter_predict)
                .warm_start(self.warm_start)
                .copy_x_train(self.copy_x_train)
                .random_state(self.random_state);

            let fitted_binary = binary_gpc.fit(X, y)?;

            let X_train = if self.copy_x_train {
                Some(X.to_owned())
            } else {
                None
            };

            return Ok(MultiClassGaussianProcessClassifier {
                state: McGpcTrained {
                    X_train,
                    y_train: y.to_owned(),
                    classes,
                    binary_classifiers: vec![fitted_binary],
                    n_classes,
                },
                kernel: None,
                optimizer: self.optimizer.clone(),
                n_restarts_optimizer: self.n_restarts_optimizer,
                max_iter_predict: self.max_iter_predict,
                warm_start: self.warm_start,
                copy_x_train: self.copy_x_train,
                random_state: self.random_state,
                config: self.config.clone(),
            });
        }

        // Multi-class case: One-vs-Rest strategy
        let mut binary_classifiers = Vec::with_capacity(n_classes);

        for (class_idx, &current_class) in classes.iter().enumerate() {
            // Create binary labels: +1 for current class, -1 for all others
            let y_binary: Array1<i32> = y.mapv(|label| if label == current_class { 1 } else { 0 });

            // Create and train binary classifier for this class
            let binary_gpc = GaussianProcessClassifier::new()
                .kernel(kernel.clone())
                .optimizer(self.optimizer.clone())
                .n_restarts_optimizer(self.n_restarts_optimizer)
                .max_iter_predict(self.max_iter_predict)
                .warm_start(self.warm_start)
                .copy_x_train(self.copy_x_train)
                .random_state(self.random_state.map(|s| s + class_idx as u64));

            let fitted_binary = binary_gpc.fit(X, &y_binary.view())?;
            binary_classifiers.push(fitted_binary);
        }

        let X_train = if self.copy_x_train {
            Some(X.to_owned())
        } else {
            None
        };

        Ok(MultiClassGaussianProcessClassifier {
            state: McGpcTrained {
                X_train,
                y_train: y.to_owned(),
                classes,
                binary_classifiers,
                n_classes,
            },
            kernel: None,
            optimizer: self.optimizer,
            n_restarts_optimizer: self.n_restarts_optimizer,
            max_iter_predict: self.max_iter_predict,
            warm_start: self.warm_start,
            copy_x_train: self.copy_x_train,
            random_state: self.random_state,
            config: self.config.clone(),
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>>
    for MultiClassGaussianProcessClassifier<McGpcTrained>
{
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let predictions: Vec<i32> = probabilities
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                self.state.classes[max_idx]
            })
            .collect();
        Ok(Array1::from(predictions))
    }
}

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>>
    for MultiClassGaussianProcessClassifier<McGpcTrained>
{
    fn predict_proba(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        let n_classes = self.state.n_classes;

        // Handle binary case
        if n_classes == 2 {
            return self.state.binary_classifiers[0].predict_proba(X);
        }

        // Multi-class case: collect probabilities from all binary classifiers
        let mut all_probabilities = Array2::<f64>::zeros((n_samples, n_classes));

        for (class_idx, binary_classifier) in self.state.binary_classifiers.iter().enumerate() {
            let binary_proba = binary_classifier.predict_proba(X)?;
            // Take the positive class probability (column 1)
            for i in 0..n_samples {
                all_probabilities[[i, class_idx]] = binary_proba[[i, 1]];
            }
        }

        // Normalize probabilities to sum to 1 for each sample
        for i in 0..n_samples {
            let row_sum: f64 = all_probabilities.row(i).sum();
            if row_sum > 1e-12 {
                for j in 0..n_classes {
                    all_probabilities[[i, j]] /= row_sum;
                }
            } else {
                // If all probabilities are near zero, assign uniform probabilities
                for j in 0..n_classes {
                    all_probabilities[[i, j]] = 1.0 / n_classes as f64;
                }
            }
        }

        Ok(all_probabilities)
    }
}

impl MultiClassGaussianProcessClassifier<McGpcTrained> {
    /// Get the unique classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.state.classes
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.state.n_classes
    }

    /// Get the binary classifiers
    pub fn binary_classifiers(&self) -> &[GaussianProcessClassifier<GpcTrained>] {
        &self.state.binary_classifiers
    }

    /// Get the log marginal likelihood for a specific class
    pub fn log_marginal_likelihood(&self, class_idx: usize) -> Option<f64> {
        if class_idx < self.state.binary_classifiers.len() {
            Some(self.state.binary_classifiers[class_idx].log_marginal_likelihood())
        } else {
            None
        }
    }

    /// Get the average log marginal likelihood across all classes
    pub fn average_log_marginal_likelihood(&self) -> f64 {
        let sum: f64 = self
            .state
            .binary_classifiers
            .iter()
            .map(|classifier| classifier.log_marginal_likelihood())
            .sum();
        sum / self.state.binary_classifiers.len() as f64
    }
}

impl Default for MultiClassGaussianProcessClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Expectation Propagation Gaussian Process Classifier
///
/// This implementation uses Expectation Propagation (EP) for approximate inference
/// in Gaussian Process Classification. EP provides a better approximation than the
/// Laplace method by iteratively refining local approximations to the likelihood.
///
/// EP maintains a multivariate Gaussian approximation to the posterior by minimizing
/// the KL divergence between the true posterior and the approximation at each data point.
///
/// # Examples
///
/// ```
/// use sklears_gaussian_process::{ExpectationPropagationGaussianProcessClassifier, kernels::RBF};
/// use sklears_core::traits::{Fit, Predict};
/// // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0], [2.0], [3.0], [4.0]];
/// let y = array![0, 0, 1, 1];
///
/// let kernel = RBF::new(1.0);
/// let ep_gpc = ExpectationPropagationGaussianProcessClassifier::new()
///     .kernel(Box::new(kernel));
/// let fitted = ep_gpc.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ExpectationPropagationGaussianProcessClassifier<S = Untrained> {
    state: S,
    kernel: Option<Box<dyn Kernel>>,
    max_iter: usize,
    tol: f64,
    damping: f64,
    min_variance: f64,
    verbose: bool,
    random_state: Option<u64>,
    config: GpcConfig,
}

/// Trained state for Expectation Propagation Gaussian Process Classifier
#[derive(Debug, Clone)]
pub struct EpGpcTrained {
    /// X_train
    pub X_train: Option<Array2<f64>>, // Training inputs
    /// y_train
    pub y_train: Array1<i32>, // Training labels
    /// classes
    pub classes: Array1<i32>, // Unique classes
    /// mu
    pub mu: Array1<f64>, // Posterior mean
    /// Sigma
    pub Sigma: Array2<f64>, // Posterior covariance
    /// tau
    pub tau: Array1<f64>, // Site precisions
    /// nu
    pub nu: Array1<f64>, // Site means * precisions
    /// kernel
    pub kernel: Box<dyn Kernel>, // Kernel function
    /// log_marginal_likelihood_value
    pub log_marginal_likelihood_value: f64, // Log marginal likelihood
    /// n_iterations
    pub n_iterations: usize, // Number of EP iterations
}

impl ExpectationPropagationGaussianProcessClassifier<Untrained> {
    /// Create a new ExpectationPropagationGaussianProcessClassifier instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: None,
            max_iter: 100,
            tol: 1e-4,
            damping: 0.5,
            min_variance: 1e-10,
            verbose: false,
            random_state: None,
            config: GpcConfig::default(),
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set damping factor for updates
    pub fn damping(mut self, damping: f64) -> Self {
        self.damping = damping.max(0.0).min(1.0);
        self
    }

    /// Set minimum variance threshold
    pub fn min_variance(mut self, min_variance: f64) -> Self {
        self.min_variance = min_variance;
        self
    }

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Estimator for ExpectationPropagationGaussianProcessClassifier<Untrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for ExpectationPropagationGaussianProcessClassifier<EpGpcTrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>>
    for ExpectationPropagationGaussianProcessClassifier<Untrained>
{
    type Fitted = ExpectationPropagationGaussianProcessClassifier<EpGpcTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<f64>, y: &ArrayView1<i32>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let kernel = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?
            .clone();

        // Get unique classes
        let mut classes_set: HashSet<i32> = HashSet::new();
        for &label in y.iter() {
            classes_set.insert(label);
        }

        if classes_set.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Binary classification requires exactly 2 classes".to_string(),
            ));
        }

        let mut classes: Vec<i32> = classes_set.into_iter().collect();
        classes.sort();
        let classes = Array1::from(classes);

        // Convert labels to {-1, 1}
        let y_binary = y.mapv(|label| if label == classes[0] { -1.0 } else { 1.0 });

        // Compute kernel matrix
        let X_owned = X.to_owned();
        let K = kernel.compute_kernel_matrix(&X_owned, None)?;

        // Run Expectation Propagation
        let (mu, Sigma, tau, nu, log_marginal_likelihood_value, n_iterations) =
            expectation_propagation(
                &K,
                &y_binary,
                self.max_iter,
                self.tol,
                self.damping,
                self.min_variance,
                self.verbose,
            )?;

        let X_train = Some(X.to_owned());

        Ok(ExpectationPropagationGaussianProcessClassifier {
            state: EpGpcTrained {
                X_train,
                y_train: y.to_owned(),
                classes,
                mu,
                Sigma,
                tau,
                nu,
                kernel,
                log_marginal_likelihood_value,
                n_iterations,
            },
            kernel: None,
            max_iter: self.max_iter,
            tol: self.tol,
            damping: self.damping,
            min_variance: self.min_variance,
            verbose: self.verbose,
            random_state: self.random_state,
            config: self.config.clone(),
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>>
    for ExpectationPropagationGaussianProcessClassifier<EpGpcTrained>
{
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let predictions: Vec<i32> = probabilities
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                self.state.classes[max_idx]
            })
            .collect();
        Ok(Array1::from(predictions))
    }
}

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>>
    for ExpectationPropagationGaussianProcessClassifier<EpGpcTrained>
{
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let X_train =
            self.state.X_train.as_ref().ok_or_else(|| {
                SklearsError::InvalidInput("Training data not available".to_string())
            })?;

        // Compute kernel between test and training points
        let X_test_owned = X.to_owned();
        let K_star = self
            .state
            .kernel
            .compute_kernel_matrix(X_train, Some(&X_test_owned))?;

        // Predict using EP posterior
        let (f_star_mean, f_star_var) = ep_predict(&K_star, &self.state.mu, &self.state.Sigma)?;

        // Convert to probabilities using probit approximation
        let mut probabilities = Array2::<f64>::zeros((X.nrows(), 2));
        for (i, (&mean, &var)) in f_star_mean.iter().zip(f_star_var.iter()).enumerate() {
            let std_dev = var.sqrt().max(1e-10);
            let z = mean / std_dev;
            let prob_positive = normal_cdf(z);
            probabilities[[i, 0]] = 1.0 - prob_positive; // Probability of class 0
            probabilities[[i, 1]] = prob_positive; // Probability of class 1
        }

        Ok(probabilities)
    }
}

impl ExpectationPropagationGaussianProcessClassifier<EpGpcTrained> {
    /// Get the log marginal likelihood
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.state.log_marginal_likelihood_value
    }

    /// Get the unique classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.state.classes
    }

    /// Get the posterior mean
    pub fn posterior_mean(&self) -> &Array1<f64> {
        &self.state.mu
    }

    /// Get the posterior covariance
    pub fn posterior_covariance(&self) -> &Array2<f64> {
        &self.state.Sigma
    }

    /// Get the number of EP iterations used
    pub fn n_iterations(&self) -> usize {
        self.state.n_iterations
    }
}

impl Default for ExpectationPropagationGaussianProcessClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for Expectation Propagation

/// Run Expectation Propagation algorithm
fn expectation_propagation(
    K: &Array2<f64>,
    y: &Array1<f64>,
    max_iter: usize,
    tol: f64,
    damping: f64,
    min_variance: f64,
    verbose: bool,
) -> SklResult<(
    Array1<f64>,
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    f64,
    usize,
)> {
    let n = K.nrows();

    // Initialize site parameters
    let mut tau = Array1::<f64>::zeros(n); // Site precisions
    let mut nu = Array1::<f64>::zeros(n); // Site means * precisions

    // Initialize posterior
    let mut mu = Array1::<f64>::zeros(n); // Posterior mean
    let mut Sigma = K.clone(); // Posterior covariance

    let mut converged = false;
    let mut iteration = 0;

    for iter in 0..max_iter {
        iteration = iter + 1;
        let mut max_change: f64 = 0.0;

        // EP updates for each site
        for i in 0..n {
            // Remove site i from cavity distribution
            let tau_cavity = 1.0 / Sigma[[i, i]] - tau[i];
            let mu_cavity = if tau_cavity > 1e-12 {
                mu[i] / (tau_cavity * Sigma[[i, i]]) - nu[i] / tau_cavity
            } else {
                0.0
            };

            // Compute marginal moments
            let sigma_cavity = if tau_cavity > 1e-12 {
                1.0 / tau_cavity
            } else {
                1e6
            };
            let (z0, z1, z2) = marginal_moments(y[i], mu_cavity, sigma_cavity);

            if z0 > 1e-12 {
                // Update site parameters
                let delta_tau = z1 / sigma_cavity - z2 / (sigma_cavity * sigma_cavity) - tau_cavity;
                let delta_nu = z1 / sigma_cavity - mu_cavity * tau_cavity;

                // Apply damping
                let tau_new = tau[i] + damping * delta_tau;
                let nu_new = nu[i] + damping * delta_nu;

                // Track convergence
                let change = (tau_new - tau[i]).abs() + (nu_new - nu[i]).abs();
                max_change = max_change.max(change);

                // Update site parameters
                tau[i] = tau_new.max(min_variance);
                nu[i] = nu_new;

                // Update posterior
                let tau_diff = tau[i] - tau_cavity;
                let nu_diff = nu[i] - mu_cavity * tau_cavity;

                if tau_diff.abs() > 1e-12 {
                    // Rank-1 update of Sigma
                    let si = Sigma.column(i).to_owned();
                    let denom = 1.0 + tau_diff * Sigma[[i, i]];

                    if denom.abs() > 1e-12 {
                        for j in 0..n {
                            for k in 0..n {
                                Sigma[[j, k]] -= tau_diff * si[j] * si[k] / denom;
                            }
                        }

                        // Update mean
                        mu = &mu + (nu_diff / denom) * &si;
                    }
                }
            }
        }

        if verbose && iter % 10 == 0 {
            println!("EP iteration {}: max change = {:.6}", iter, max_change);
        }

        // Check convergence
        if max_change < tol {
            converged = true;
            if verbose {
                println!("EP converged at iteration {}", iter);
            }
            break;
        }
    }

    if !converged && verbose {
        println!("EP did not converge after {} iterations", max_iter);
    }

    // Compute log marginal likelihood
    let log_marginal_likelihood = compute_ep_log_marginal_likelihood(K, &tau, &nu, &mu, &Sigma, y)?;

    Ok((mu, Sigma, tau, nu, log_marginal_likelihood, iteration))
}

/// Compute marginal moments for probit likelihood
fn marginal_moments(y: f64, mu: f64, sigma2: f64) -> (f64, f64, f64) {
    let sigma = sigma2.sqrt();
    let z = y * mu / sigma;

    // Compute PDF and CDF of standard normal
    let pdf = (-0.5 * z * z).exp() / (2.0 * PI).sqrt();
    let cdf = normal_cdf(z);

    // Avoid numerical issues
    let z0 = cdf.max(1e-12);
    let ratio = if z0 > 1e-12 { pdf / z0 } else { 0.0 };

    let z1 = mu + y * sigma * ratio;
    let z2 = mu * mu + sigma2 * (1.0 - z * ratio - ratio * ratio);

    (z0, z1, z2.max(min_variance_global()))
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Global minimum variance constant
fn min_variance_global() -> f64 {
    1e-10
}

/// Predict with EP posterior
fn ep_predict(
    K_star: &Array2<f64>,
    mu: &Array1<f64>,
    Sigma: &Array2<f64>,
) -> SklResult<(Array1<f64>, Array1<f64>)> {
    // Predictive mean: K_star^T * Sigma^{-1} * mu
    let Sigma_inv = utils::matrix_inverse(Sigma)?;
    let mean = K_star.t().dot(&Sigma_inv.dot(mu));

    // Predictive variance: K_star_star - K_star^T * Sigma^{-1} * K_star
    let temp = K_star.t().dot(&Sigma_inv);
    let var_reduction = temp.dot(K_star);

    // Diagonal of var_reduction gives the variance reduction for each test point
    let variance = Array1::from_iter((0..K_star.ncols()).map(|i| {
        1.0 - var_reduction[[i, i]] // Assuming K_star_star diagonal is 1
    }));

    Ok((mean, variance))
}

/// Compute log marginal likelihood for EP
fn compute_ep_log_marginal_likelihood(
    K: &Array2<f64>,
    tau: &Array1<f64>,
    nu: &Array1<f64>,
    mu: &Array1<f64>,
    Sigma: &Array2<f64>,
    _y: &Array1<f64>,
) -> SklResult<f64> {
    let n = K.nrows();

    // Log determinant of posterior covariance
    let L_Sigma = utils::robust_cholesky(Sigma)?;
    let log_det_Sigma = 2.0 * L_Sigma.diag().mapv(|x| x.ln()).sum();

    // Log determinant of prior covariance
    let L_K = utils::robust_cholesky(K)?;
    let log_det_K = 2.0 * L_K.diag().mapv(|x| x.ln()).sum();

    // Quadratic terms
    let quad_prior = 0.5 * mu.dot(&utils::triangular_solve(&L_K, mu)?);
    let quad_posterior = 0.5 * mu.dot(&utils::triangular_solve(&L_Sigma, mu)?);

    // Site contributions
    let mut site_contrib = 0.0;
    for i in 0..n {
        if tau[i] > 1e-12 {
            let mu_i = nu[i] / tau[i];
            site_contrib += 0.5 * (tau[i].ln() - (2.0 * PI).ln()) - 0.5 * tau[i] * mu_i * mu_i;
        }
    }

    // Approximate marginal likelihood
    let log_ml =
        -0.5 * log_det_Sigma + 0.5 * log_det_K + site_contrib + quad_prior - quad_posterior;

    Ok(log_ml)
}

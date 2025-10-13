//! Semi-supervised Gaussian Mixture Model
//!
//! This module implements a semi-supervised Gaussian Mixture Model that can utilize
//! both labeled and unlabeled data for training. It uses the Expectation-Maximization
//! algorithm with partial supervision from labeled data.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};
use std::collections::HashSet;

/// Semi-supervised Gaussian Mixture Model
///
/// This classifier extends the standard Gaussian Mixture Model to handle
/// semi-supervised learning scenarios where only a subset of training data
/// is labeled. It uses the EM algorithm with constraints from labeled data.
///
/// # Parameters
///
/// * `n_components` - Number of mixture components
/// * `max_iter` - Maximum number of EM iterations
/// * `tol` - Convergence tolerance
/// * `covariance_type` - Type of covariance matrix ('full', 'diag', 'spherical')
/// * `reg_covar` - Regularization parameter for covariance matrices
/// * `labeled_weight` - Weight for labeled data constraints
/// * `random_seed` - Random seed for initialization
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::SemiSupervisedGMM;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let gmm = SemiSupervisedGMM::new()
///     .n_components(2)
///     .max_iter(100)
///     .labeled_weight(10.0);
/// let fitted = gmm.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SemiSupervisedGMM<S = Untrained> {
    state: S,
    n_components: usize,
    max_iter: usize,
    tol: f64,
    covariance_type: String,
    reg_covar: f64,
    labeled_weight: f64,
    random_seed: Option<u64>,
}

impl SemiSupervisedGMM<Untrained> {
    /// Create a new SemiSupervisedGMM instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            max_iter: 100,
            tol: 1e-6,
            covariance_type: "full".to_string(),
            reg_covar: 1e-6,
            labeled_weight: 1.0,
            random_seed: None,
        }
    }

    /// Set the number of mixture components
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

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: String) -> Self {
        self.covariance_type = covariance_type;
        self
    }

    /// Set the covariance regularization parameter
    pub fn reg_covar(mut self, reg_covar: f64) -> Self {
        self.reg_covar = reg_covar;
        self
    }

    /// Set the weight for labeled data constraints
    pub fn labeled_weight(mut self, labeled_weight: f64) -> Self {
        self.labeled_weight = labeled_weight;
        self
    }

    /// Set the random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    fn initialize_parameters(
        &self,
        X: &Array2<f64>,
        n_classes: usize,
    ) -> (Array1<f64>, Array2<f64>, Vec<Array2<f64>>) {
        let (n_samples, n_features) = X.dim();

        // Initialize mixture weights
        let weights = Array1::from_elem(n_classes, 1.0 / n_classes as f64);

        // Initialize means using simple clustering
        let mut means = Array2::zeros((n_classes, n_features));
        for k in 0..n_classes {
            let start_idx = (k * n_samples) / n_classes;
            let end_idx = ((k + 1) * n_samples) / n_classes;

            for j in 0..n_features {
                let mut sum = 0.0;
                let mut count = 0;
                for i in start_idx..end_idx.min(n_samples) {
                    sum += X[[i, j]];
                    count += 1;
                }
                means[[k, j]] = if count > 0 { sum / count as f64 } else { 0.0 };
            }
        }

        // Initialize covariances
        let mut covariances = Vec::new();
        for _k in 0..n_classes {
            let mut cov = Array2::eye(n_features);
            // Add regularization
            for i in 0..n_features {
                cov[[i, i]] += self.reg_covar;
            }
            covariances.push(cov);
        }

        (weights, means, covariances)
    }

    fn multivariate_normal_pdf(
        &self,
        x: &Array1<f64>,
        mean: &Array1<f64>,
        cov: &Array2<f64>,
    ) -> f64 {
        let n = x.len();
        let diff = x - mean;

        // Compute determinant (simplified for diagonal/spherical case)
        let det = match self.covariance_type.as_str() {
            "diag" | "spherical" => cov.diag().iter().product::<f64>(),
            _ => {
                // For full covariance, use a simple approximation
                cov.diag().iter().product::<f64>()
            }
        };

        if det <= 0.0 {
            return 1e-10; // Small non-zero value to avoid numerical issues
        }

        // Compute inverse (simplified)
        let inv_cov = match self.covariance_type.as_str() {
            "diag" | "spherical" => {
                let mut inv = Array2::zeros(cov.dim());
                for i in 0..n {
                    inv[[i, i]] = 1.0 / cov[[i, i]];
                }
                inv
            }
            _ => {
                // Simplified inverse for numerical stability
                let mut inv = Array2::zeros(cov.dim());
                for i in 0..n {
                    inv[[i, i]] = 1.0 / (cov[[i, i]] + self.reg_covar);
                }
                inv
            }
        };

        let mahalanobis = diff.dot(&inv_cov.dot(&diff));
        let norm_factor = 1.0 / ((2.0 * std::f64::consts::PI).powi(n as i32) * det).sqrt();

        norm_factor * (-0.5 * mahalanobis).exp()
    }

    fn expectation_step(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
        labeled_indices: &[usize],
        y_labeled: &Array1<i32>,
        classes: &[i32],
    ) -> Array2<f64> {
        let n_samples = X.nrows();
        let n_classes = classes.len();
        let mut responsibilities = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let x = X.row(i).to_owned();
            let mut total_likelihood = 0.0;
            let mut likelihoods = vec![0.0; n_classes];

            // Compute likelihoods for each component
            for k in 0..n_classes {
                let mean = means.row(k).to_owned();
                let likelihood = self.multivariate_normal_pdf(&x, &mean, &covariances[k]);
                likelihoods[k] = weights[k] * likelihood;
                total_likelihood += likelihoods[k];
            }

            // Check if this sample is labeled
            if let Some(labeled_pos) = labeled_indices.iter().position(|&idx| idx == i) {
                let true_label = y_labeled[labeled_pos];
                if let Some(class_idx) = classes.iter().position(|&c| c == true_label) {
                    // For labeled samples, set high probability for true class
                    for k in 0..n_classes {
                        responsibilities[[i, k]] = if k == class_idx {
                            self.labeled_weight
                        } else {
                            (1.0 - self.labeled_weight) / (n_classes - 1) as f64
                        };
                    }
                } else {
                    // Fallback: use standard posterior probabilities
                    for k in 0..n_classes {
                        responsibilities[[i, k]] = if total_likelihood > 0.0 {
                            likelihoods[k] / total_likelihood
                        } else {
                            1.0 / n_classes as f64
                        };
                    }
                }
            } else {
                // For unlabeled samples, use standard posterior probabilities
                for k in 0..n_classes {
                    responsibilities[[i, k]] = if total_likelihood > 0.0 {
                        likelihoods[k] / total_likelihood
                    } else {
                        1.0 / n_classes as f64
                    };
                }
            }
        }

        responsibilities
    }

    fn maximization_step(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Vec<Array2<f64>>) {
        let (n_samples, n_features) = X.dim();
        let n_classes = responsibilities.ncols();

        // Update weights
        let n_k = responsibilities.sum_axis(Axis(0));
        let weights = &n_k / n_samples as f64;

        // Update means
        let mut means = Array2::zeros((n_classes, n_features));
        for k in 0..n_classes {
            if n_k[k] > 0.0 {
                for j in 0..n_features {
                    let mut weighted_sum = 0.0;
                    for i in 0..n_samples {
                        weighted_sum += responsibilities[[i, k]] * X[[i, j]];
                    }
                    means[[k, j]] = weighted_sum / n_k[k];
                }
            }
        }

        // Update covariances (simplified for diagonal case)
        let mut covariances = Vec::new();
        for k in 0..n_classes {
            let mut cov = Array2::zeros((n_features, n_features));

            if n_k[k] > 0.0 {
                let mean_k = means.row(k).to_owned();

                for i in 0..n_samples {
                    let diff = &X.row(i).to_owned() - &mean_k;
                    let weight = responsibilities[[i, k]];

                    match self.covariance_type.as_str() {
                        "diag" | "spherical" => {
                            // Diagonal covariance
                            for j in 0..n_features {
                                cov[[j, j]] += weight * diff[j] * diff[j];
                            }
                        }
                        _ => {
                            // Full covariance (simplified to diagonal for numerical stability)
                            for j in 0..n_features {
                                cov[[j, j]] += weight * diff[j] * diff[j];
                            }
                        }
                    }
                }

                // Normalize and add regularization
                for j in 0..n_features {
                    cov[[j, j]] = (cov[[j, j]] / n_k[k]) + self.reg_covar;
                }
            } else {
                // Fallback for empty components
                for j in 0..n_features {
                    cov[[j, j]] = 1.0 + self.reg_covar;
                }
            }

            covariances.push(cov);
        }

        (weights, means, covariances)
    }

    fn compute_log_likelihood(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
    ) -> f64 {
        let n_samples = X.nrows();
        let n_classes = weights.len();
        let mut log_likelihood = 0.0;

        for i in 0..n_samples {
            let x = X.row(i).to_owned();
            let mut sample_likelihood = 0.0;

            for k in 0..n_classes {
                let mean = means.row(k).to_owned();
                let likelihood = self.multivariate_normal_pdf(&x, &mean, &covariances[k]);
                sample_likelihood += weights[k] * likelihood;
            }

            if sample_likelihood > 0.0 {
                log_likelihood += sample_likelihood.ln();
            }
        }

        log_likelihood
    }
}

impl Default for SemiSupervisedGMM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SemiSupervisedGMM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for SemiSupervisedGMM<Untrained> {
    type Fitted = SemiSupervisedGMM<SemiSupervisedGMMTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        // Identify labeled samples
        let mut labeled_indices = Vec::new();
        let mut y_labeled = Vec::new();
        let mut classes = HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label != -1 {
                labeled_indices.push(i);
                y_labeled.push(label);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();
        let y_labeled = Array1::from(y_labeled);
        let n_classes = classes.len();

        if n_classes != self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of components must equal number of classes for supervised learning"
                    .to_string(),
            ));
        }

        // Initialize parameters
        let (mut weights, mut means, mut covariances) = self.initialize_parameters(&X, n_classes);
        let mut prev_log_likelihood = f64::NEG_INFINITY;

        // EM algorithm
        for _iter in 0..self.max_iter {
            // E-step
            let responsibilities = self.expectation_step(
                &X,
                &weights,
                &means,
                &covariances,
                &labeled_indices,
                &y_labeled,
                &classes,
            );

            // M-step
            let (new_weights, new_means, new_covariances) =
                self.maximization_step(&X, &responsibilities);
            weights = new_weights;
            means = new_means;
            covariances = new_covariances;

            // Check convergence
            let log_likelihood = self.compute_log_likelihood(&X, &weights, &means, &covariances);
            if (log_likelihood - prev_log_likelihood).abs() < self.tol {
                break;
            }
            prev_log_likelihood = log_likelihood;
        }

        Ok(SemiSupervisedGMM {
            state: SemiSupervisedGMMTrained {
                X_train: X.clone(),
                y_train: y,
                classes: Array1::from(classes),
                weights,
                means,
                covariances,
            },
            n_components: self.n_components,
            max_iter: self.max_iter,
            tol: self.tol,
            covariance_type: self.covariance_type,
            reg_covar: self.reg_covar,
            labeled_weight: self.labeled_weight,
            random_seed: self.random_seed,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for SemiSupervisedGMM<SemiSupervisedGMMTrained> {
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let probas = self.predict_proba(X)?;
        let n_test = probas.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            let max_idx = probas
                .row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            predictions[i] = self.state.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>>
    for SemiSupervisedGMM<SemiSupervisedGMMTrained>
{
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let n_classes = self.state.classes.len();
        let mut probas = Array2::zeros((n_test, n_classes));

        for i in 0..n_test {
            let x = X.row(i).to_owned();
            let mut total_likelihood = 0.0;
            let mut likelihoods = vec![0.0; n_classes];

            // Compute likelihoods for each component
            for k in 0..n_classes {
                let mean = self.state.means.row(k).to_owned();
                let likelihood =
                    self.multivariate_normal_pdf(&x, &mean, &self.state.covariances[k]);
                likelihoods[k] = self.state.weights[k] * likelihood;
                total_likelihood += likelihoods[k];
            }

            // Normalize to get probabilities
            for k in 0..n_classes {
                probas[[i, k]] = if total_likelihood > 0.0 {
                    likelihoods[k] / total_likelihood
                } else {
                    1.0 / n_classes as f64
                };
            }
        }

        Ok(probas)
    }
}

impl SemiSupervisedGMM<SemiSupervisedGMMTrained> {
    fn multivariate_normal_pdf(
        &self,
        x: &Array1<f64>,
        mean: &Array1<f64>,
        cov: &Array2<f64>,
    ) -> f64 {
        let n = x.len();
        let diff = x - mean;

        // Compute determinant (simplified for diagonal/spherical case)
        let det = match self.covariance_type.as_str() {
            "diag" | "spherical" => cov.diag().iter().product::<f64>(),
            _ => {
                // For full covariance, use a simple approximation
                cov.diag().iter().product::<f64>()
            }
        };

        if det <= 0.0 {
            return 1e-10; // Small non-zero value to avoid numerical issues
        }

        // Compute inverse (simplified)
        let inv_cov = match self.covariance_type.as_str() {
            "diag" | "spherical" => {
                let mut inv = Array2::zeros(cov.dim());
                for i in 0..n {
                    inv[[i, i]] = 1.0 / cov[[i, i]];
                }
                inv
            }
            _ => {
                // Simplified inverse for numerical stability
                let mut inv = Array2::zeros(cov.dim());
                for i in 0..n {
                    inv[[i, i]] = 1.0 / (cov[[i, i]] + self.reg_covar);
                }
                inv
            }
        };

        let mahalanobis = diff.dot(&inv_cov.dot(&diff));
        let norm_factor = 1.0 / ((2.0 * std::f64::consts::PI).powi(n as i32) * det).sqrt();

        norm_factor * (-0.5 * mahalanobis).exp()
    }
}

/// Trained state for SemiSupervisedGMM
#[derive(Debug, Clone)]
pub struct SemiSupervisedGMMTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// weights
    pub weights: Array1<f64>,
    /// means
    pub means: Array2<f64>,
    /// covariances
    pub covariances: Vec<Array2<f64>>,
}

//! Semi-supervised Naive Bayes implementation

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};

/// Semi-supervised Naive Bayes classifier
///
/// This classifier extends the traditional Naive Bayes algorithm to work with
/// unlabeled data by using an expectation-maximization (EM) approach. It iteratively
/// updates class priors and feature likelihoods using both labeled and unlabeled data.
///
/// # Parameters
///
/// * `alpha` - Additive smoothing parameter (Laplace smoothing)
/// * `class_weight` - Weight given to labeled samples vs unlabeled samples
/// * `max_iter` - Maximum number of EM iterations
/// * `tol` - Convergence tolerance for EM algorithm
/// * `verbose` - Whether to print progress information
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::SemiSupervisedNaiveBayes;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let nb = SemiSupervisedNaiveBayes::new()
///     .alpha(1.0)
///     .max_iter(50)
///     .class_weight(1.0);
/// let fitted = nb.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SemiSupervisedNaiveBayes<S = Untrained> {
    state: S,
    alpha: f64,
    class_weight: f64,
    max_iter: usize,
    tol: f64,
    verbose: bool,
}

impl SemiSupervisedNaiveBayes<Untrained> {
    /// Create a new SemiSupervisedNaiveBayes instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            alpha: 1.0,
            class_weight: 1.0,
            max_iter: 100,
            tol: 1e-4,
            verbose: false,
        }
    }

    /// Set the additive smoothing parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the weight for labeled samples
    pub fn class_weight(mut self, weight: f64) -> Self {
        self.class_weight = weight;
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

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn compute_class_log_priors(&self, class_counts: &HashMap<i32, f64>) -> HashMap<i32, f64> {
        let total_count: f64 = class_counts.values().sum();
        let mut log_priors = HashMap::new();

        for (&class, &count) in class_counts {
            log_priors.insert(class, (count / total_count).ln());
        }

        log_priors
    }

    fn compute_feature_log_likelihoods(
        &self,
        X: &Array2<f64>,
        class_posteriors: &Array2<f64>,
        classes: &[i32],
    ) -> HashMap<i32, Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let n_classes = classes.len();
        let mut feature_likelihoods = HashMap::new();

        for (class_idx, &class) in classes.iter().enumerate() {
            let mut class_feature_sums = Array1::<f64>::zeros(n_features);
            let mut class_total_weight = 0.0;

            // Accumulate weighted feature sums for this class
            for i in 0..n_samples {
                let posterior = class_posteriors[[i, class_idx]];
                class_total_weight += posterior;
                for j in 0..n_features {
                    class_feature_sums[j] += X[[i, j]] * posterior;
                }
            }

            // Compute mean and variance for each feature
            let mut means = Array1::<f64>::zeros(n_features);
            let mut variances = Array1::<f64>::zeros(n_features);

            if class_total_weight > 0.0 {
                // Compute means
                for j in 0..n_features {
                    means[j] = class_feature_sums[j] / class_total_weight;
                }

                // Compute variances
                for i in 0..n_samples {
                    let posterior = class_posteriors[[i, class_idx]];
                    for j in 0..n_features {
                        let diff = X[[i, j]] - means[j];
                        variances[j] += posterior * diff * diff;
                    }
                }

                for j in 0..n_features {
                    variances[j] = (variances[j] / class_total_weight) + self.alpha;
                    // Add smoothing
                }
            } else {
                // No samples for this class, use uniform distribution
                for j in 0..n_features {
                    means[j] = 0.0;
                    variances[j] = 1.0;
                }
            }

            // Store parameters as a 2xn_features matrix [means; variances]
            let mut params = Array2::<f64>::zeros((2, n_features));
            for j in 0..n_features {
                params[[0, j]] = means[j];
                params[[1, j]] = variances[j].max(1e-9); // Prevent zero variance
            }

            feature_likelihoods.insert(class, params);
        }

        feature_likelihoods
    }

    fn compute_log_likelihood(
        &self,
        X: &Array2<f64>,
        log_priors: &HashMap<i32, f64>,
        feature_params: &HashMap<i32, Array2<f64>>,
        classes: &[i32],
    ) -> Array2<f64> {
        let (n_samples, n_features) = X.dim();
        let n_classes = classes.len();
        let mut log_likelihood = Array2::<f64>::zeros((n_samples, n_classes));

        for (class_idx, &class) in classes.iter().enumerate() {
            let prior = log_priors[&class];
            let params = &feature_params[&class];

            for i in 0..n_samples {
                let mut log_prob = prior;

                // Gaussian likelihood for each feature
                for j in 0..n_features {
                    let mean = params[[0, j]];
                    let variance = params[[1, j]];
                    let x = X[[i, j]];

                    // Log of Gaussian PDF: -0.5 * log(2π * σ²) - (x - μ)² / (2 * σ²)
                    let log_gaussian = -0.5 * (2.0 * std::f64::consts::PI * variance).ln()
                        - (x - mean).powi(2) / (2.0 * variance);
                    log_prob += log_gaussian;
                }

                log_likelihood[[i, class_idx]] = log_prob;
            }
        }

        log_likelihood
    }

    fn normalize_log_probabilities(&self, log_probs: &Array2<f64>) -> Array2<f64> {
        let (n_samples, n_classes) = log_probs.dim();
        let mut probabilities = Array2::<f64>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = log_probs.row(i);
            let max_log_prob = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            // Subtract max for numerical stability
            let mut sum_exp = 0.0;
            for j in 0..n_classes {
                let exp_val = (log_probs[[i, j]] - max_log_prob).exp();
                probabilities[[i, j]] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            if sum_exp > 0.0 {
                for j in 0..n_classes {
                    probabilities[[i, j]] /= sum_exp;
                }
            } else {
                // Uniform distribution if all probabilities are zero
                for j in 0..n_classes {
                    probabilities[[i, j]] = 1.0 / n_classes as f64;
                }
            }
        }

        probabilities
    }
}

impl Default for SemiSupervisedNaiveBayes<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SemiSupervisedNaiveBayes<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for SemiSupervisedNaiveBayes<Untrained> {
    type Fitted = SemiSupervisedNaiveBayes<SemiSupervisedNaiveBayesTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        let (n_samples, _n_features) = X.dim();

        // Identify labeled and unlabeled samples
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();
        let mut classes = HashSet::new();

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

        // Initialize class posteriors (responsibilities)
        let mut class_posteriors = Array2::<f64>::zeros((n_samples, n_classes));

        // Set labeled samples to have posterior = 1 for their class
        for &idx in &labeled_indices {
            if let Some(class_idx) = classes.iter().position(|&c| c == y[idx]) {
                class_posteriors[[idx, class_idx]] = self.class_weight;
            }
        }

        // Initialize unlabeled samples with uniform distribution
        let uniform_prob = 1.0 / n_classes as f64;
        for &idx in &unlabeled_indices {
            for j in 0..n_classes {
                class_posteriors[[idx, j]] = uniform_prob;
            }
        }

        let mut prev_log_likelihood = f64::NEG_INFINITY;

        // EM algorithm
        for iter in 0..self.max_iter {
            // M-step: Update parameters based on current posteriors

            // Update class priors
            let mut class_counts = HashMap::new();
            for (class_idx, &class) in classes.iter().enumerate() {
                let count: f64 = class_posteriors.column(class_idx).sum();
                class_counts.insert(class, count);
            }

            let log_priors = self.compute_class_log_priors(&class_counts);
            let feature_params =
                self.compute_feature_log_likelihoods(&X, &class_posteriors, &classes);

            // E-step: Update posteriors for unlabeled samples
            let log_likelihood =
                self.compute_log_likelihood(&X, &log_priors, &feature_params, &classes);
            let new_posteriors = self.normalize_log_probabilities(&log_likelihood);

            // Update posteriors for unlabeled samples only
            for &idx in &unlabeled_indices {
                for j in 0..n_classes {
                    class_posteriors[[idx, j]] = new_posteriors[[idx, j]];
                }
            }

            // Check convergence
            let current_log_likelihood: f64 = log_likelihood.sum();
            if (current_log_likelihood - prev_log_likelihood).abs() < self.tol {
                if self.verbose {
                    println!("Converged at iteration {}", iter + 1);
                }
                break;
            }
            prev_log_likelihood = current_log_likelihood;

            if self.verbose && iter % 10 == 0 {
                println!(
                    "Iteration {}: log-likelihood = {:.6}",
                    iter + 1,
                    current_log_likelihood
                );
            }
        }

        // Final parameter computation
        let mut class_counts = HashMap::new();
        for (class_idx, &class) in classes.iter().enumerate() {
            let count: f64 = class_posteriors.column(class_idx).sum();
            class_counts.insert(class, count);
        }

        let log_priors = self.compute_class_log_priors(&class_counts);
        let feature_params = self.compute_feature_log_likelihoods(&X, &class_posteriors, &classes);

        Ok(SemiSupervisedNaiveBayes {
            state: SemiSupervisedNaiveBayesTrained {
                X_train: X.clone(),
                y_train: y,
                classes: Array1::from(classes.clone()),
                log_priors,
                feature_params,
            },
            alpha: self.alpha,
            class_weight: self.class_weight,
            max_iter: self.max_iter,
            tol: self.tol,
            verbose: self.verbose,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for SemiSupervisedNaiveBayes<SemiSupervisedNaiveBayesTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let probas = self.predict_proba(X)?;
        let n_test = probas.nrows();
        let mut predictions = Array1::<i32>::zeros(n_test);

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
    for SemiSupervisedNaiveBayes<SemiSupervisedNaiveBayesTrained>
{
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let classes: Vec<i32> = self.state.classes.to_vec();

        let log_likelihood = self.compute_log_likelihood(
            &X,
            &self.state.log_priors,
            &self.state.feature_params,
            &classes,
        );
        let probabilities = self.normalize_log_probabilities(&log_likelihood);

        Ok(probabilities)
    }
}

impl SemiSupervisedNaiveBayes<SemiSupervisedNaiveBayesTrained> {
    fn compute_log_likelihood(
        &self,
        X: &Array2<f64>,
        log_priors: &HashMap<i32, f64>,
        feature_params: &HashMap<i32, Array2<f64>>,
        classes: &[i32],
    ) -> Array2<f64> {
        let (n_samples, n_features) = X.dim();
        let n_classes = classes.len();
        let mut log_likelihood = Array2::<f64>::zeros((n_samples, n_classes));

        for (class_idx, &class) in classes.iter().enumerate() {
            let prior = log_priors[&class];
            let params = &feature_params[&class];

            for i in 0..n_samples {
                let mut log_prob = prior;

                // Gaussian likelihood for each feature
                for j in 0..n_features {
                    let mean = params[[0, j]];
                    let variance = params[[1, j]];
                    let x = X[[i, j]];

                    // Log of Gaussian PDF
                    let log_gaussian = -0.5 * (2.0 * std::f64::consts::PI * variance).ln()
                        - (x - mean).powi(2) / (2.0 * variance);
                    log_prob += log_gaussian;
                }

                log_likelihood[[i, class_idx]] = log_prob;
            }
        }

        log_likelihood
    }

    fn normalize_log_probabilities(&self, log_probs: &Array2<f64>) -> Array2<f64> {
        let (n_samples, n_classes) = log_probs.dim();
        let mut probabilities = Array2::<f64>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = log_probs.row(i);
            let max_log_prob = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            // Subtract max for numerical stability
            let mut sum_exp = 0.0;
            for j in 0..n_classes {
                let exp_val = (log_probs[[i, j]] - max_log_prob).exp();
                probabilities[[i, j]] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            if sum_exp > 0.0 {
                for j in 0..n_classes {
                    probabilities[[i, j]] /= sum_exp;
                }
            } else {
                // Uniform distribution if all probabilities are zero
                for j in 0..n_classes {
                    probabilities[[i, j]] = 1.0 / n_classes as f64;
                }
            }
        }

        probabilities
    }
}

/// Trained state for SemiSupervisedNaiveBayes
#[derive(Debug, Clone)]
pub struct SemiSupervisedNaiveBayesTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// log_priors
    pub log_priors: HashMap<i32, f64>,
    /// feature_params
    pub feature_params: HashMap<i32, Array2<f64>>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_semi_supervised_naive_bayes_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let nb = SemiSupervisedNaiveBayes::new()
            .alpha(1.0)
            .max_iter(50)
            .class_weight(1.0);
        let fitted = nb.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let sum: f64 = probas.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_semi_supervised_naive_bayes_parameters() {
        let nb = SemiSupervisedNaiveBayes::new()
            .alpha(0.5)
            .class_weight(2.0)
            .max_iter(200)
            .tol(1e-6)
            .verbose(true);

        assert_eq!(nb.alpha, 0.5);
        assert_eq!(nb.class_weight, 2.0);
        assert_eq!(nb.max_iter, 200);
        assert_eq!(nb.tol, 1e-6);
        assert_eq!(nb.verbose, true);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_semi_supervised_naive_bayes_all_labeled() {
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0, 1]; // All labeled

        let nb = SemiSupervisedNaiveBayes::new().max_iter(10);
        let fitted = nb.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_semi_supervised_naive_bayes_error_cases() {
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![-1, -1]; // No labeled samples

        let nb = SemiSupervisedNaiveBayes::new();
        let result = nb.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }
}

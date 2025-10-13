//! Gaussian Naive Bayes classifier implementation

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result},
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, PredictProba, Score, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::{compute_class_prior, safe_log, validation::ProbabilisticModel, NaiveBayesMixin};

/// Configuration for Gaussian Naive Bayes
#[derive(Debug, Clone)]
pub struct GaussianNBConfig {
    /// Variance smoothing parameter
    pub var_smoothing: f64,
    /// Prior probabilities of the classes
    pub priors: Option<Array1<f64>>,
}

impl Default for GaussianNBConfig {
    fn default() -> Self {
        Self {
            var_smoothing: 1e-9,
            priors: None,
        }
    }
}

/// Gaussian Naive Bayes classifier
///
/// For each class, the likelihood of the features is assumed to be Gaussian.
#[derive(Debug, Clone)]
pub struct GaussianNB<State = Untrained> {
    config: GaussianNBConfig,
    state: PhantomData<State>,
    // Trained state fields
    theta_: Option<Array2<f64>>, // Mean of each feature per class
    var_: Option<Array2<f64>>,   // Variance of each feature per class
    class_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
}

impl GaussianNB<Untrained> {
    /// Create a new Gaussian Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: GaussianNBConfig::default(),
            state: PhantomData,
            theta_: None,
            var_: None,
            class_prior_: None,
            classes_: None,
        }
    }

    /// Set variance smoothing parameter
    pub fn var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.config.var_smoothing = var_smoothing;
        self
    }

    /// Set prior probabilities
    pub fn priors(mut self, priors: Array1<f64>) -> Self {
        self.config.priors = Some(priors);
        self
    }
}

impl Default for GaussianNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GaussianNB<Untrained> {
    type Float = Float;
    type Config = GaussianNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for GaussianNB<Untrained> {
    type Fitted = GaussianNB<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();
        let n_features = x.ncols();

        // Initialize parameters
        let mut theta = Array2::zeros((n_classes, n_features));
        let mut var = Array2::zeros((n_classes, n_features));

        // Compute class counts and priors
        let (_class_count, class_prior) = if let Some(ref priors) = self.config.priors {
            if priors.len() != n_classes {
                return Err(SklearsError::InvalidInput(format!(
                    "Number of priors ({}) doesn't match number of classes ({})",
                    priors.len(),
                    n_classes
                )));
            }
            let sum = priors.sum();
            if (sum - 1.0).abs() > 1e-10 {
                return Err(SklearsError::InvalidInput(
                    "The sum of the priors should be 1.0".to_string(),
                ));
            }
            let class_count = Array1::zeros(n_classes); // Not used when priors are given
            (class_count, priors.clone())
        } else {
            compute_class_prior(y, &classes)
        };

        // Compute mean and variance for each class
        for (class_idx, &class_label) in classes.iter().enumerate() {
            // Get samples belonging to this class
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            if mask.is_empty() {
                continue;
            }

            // Select rows belonging to this class
            let x_class = x.select(Axis(0), &mask);

            // Compute mean and variance for each feature
            for feature_idx in 0..n_features {
                let feature_values = x_class.column(feature_idx);
                let mean = feature_values.mean().unwrap_or(0.0);
                let variance = if feature_values.len() > 1 {
                    let var = feature_values.mapv(|v| (v - mean).powi(2)).sum()
                        / (feature_values.len() as f64);
                    var + self.config.var_smoothing
                } else {
                    self.config.var_smoothing
                };

                theta[[class_idx, feature_idx]] = mean;
                var[[class_idx, feature_idx]] = variance;
            }
        }

        Ok(GaussianNB {
            config: self.config,
            state: PhantomData,
            theta_: Some(theta),
            var_: Some(var),
            class_prior_: Some(class_prior),
            classes_: Some(classes),
        })
    }
}

impl GaussianNB<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let theta = self.theta_.as_ref().unwrap();
        let var = self.var_.as_ref().unwrap();
        let class_prior = self.class_prior_.as_ref().unwrap();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let n_samples = x.nrows();

        let mut joint_log_likelihood = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            let class_theta = theta.row(class_idx);
            let class_var = var.row(class_idx);

            // Compute log likelihood for each sample
            for (sample_idx, x_sample) in x.axis_iter(Axis(0)).enumerate() {
                let mut log_prob = 0.0;

                // Gaussian log likelihood
                for (feature_idx, &x_val) in x_sample.iter().enumerate() {
                    let mean = class_theta[feature_idx];
                    let variance = class_var[feature_idx];
                    let diff = x_val - mean;

                    // log(1/sqrt(2*pi*var)) - (x-mu)^2/(2*var)
                    log_prob += -0.5 * (2.0 * std::f64::consts::PI * variance).ln()
                        - 0.5 * diff * diff / variance;
                }

                joint_log_likelihood[[sample_idx, class_idx]] =
                    log_prob + safe_log(class_prior[class_idx]);
            }
        }

        Ok(joint_log_likelihood)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for GaussianNB<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let classes = self.classes_.as_ref().unwrap();

        // Find the class with maximum log probability for each sample
        Ok(log_prob.map_axis(Axis(1), |row| {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            classes[max_idx]
        }))
    }
}

impl PredictProba<Array2<Float>, Array2<f64>> for GaussianNB<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let n_samples = x.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let mut proba = Array2::zeros((n_samples, n_classes));

        // Normalize to get probabilities
        for i in 0..n_samples {
            let row = log_prob.row(i);
            let max_log_prob = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Compute exp(log_prob - max_log_prob) for numerical stability
            let mut exp_sum = 0.0;
            for j in 0..n_classes {
                let exp_val = (log_prob[[i, j]] - max_log_prob).exp();
                proba[[i, j]] = exp_val;
                exp_sum += exp_val;
            }

            // Normalize
            for j in 0..n_classes {
                proba[[i, j]] /= exp_sum;
            }
        }

        Ok(proba)
    }
}

impl Score<Array2<Float>, Array1<i32>> for GaussianNB<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<f64> {
        let predictions = Predict::predict(self, x)?;
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_val)| pred == true_val)
            .count();

        Ok(correct as f64 / y.len() as f64)
    }
}

impl NaiveBayesMixin for GaussianNB<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        // For Gaussian NB, this would be the log of the Gaussian PDF parameters
        // Not typically used directly, but we can return theta_ as a proxy
        self.theta_.as_ref().unwrap()
    }

    fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }
}

impl ProbabilisticModel for GaussianNB<Trained> {
    type Error = SklearsError;

    fn fit(&mut self, _x: &Array2<f64>, _y: &Array1<i32>) -> std::result::Result<(), Self::Error> {
        // Already fitted, no-op for trained models
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> std::result::Result<Array1<i32>, Self::Error> {
        Predict::predict(self, x)
    }

    fn predict_proba(&self, x: &Array2<f64>) -> std::result::Result<Array2<f64>, Self::Error> {
        PredictProba::predict_proba(self, x)
    }

    fn log_likelihood(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> std::result::Result<f64, Self::Error> {
        let joint_log_likelihood = self.joint_log_likelihood(x)?;
        let mut total_log_likelihood = 0.0;

        for (i, &true_class) in y.iter().enumerate() {
            // Find class index
            let class_idx = self
                .classes_
                .as_ref()
                .unwrap()
                .iter()
                .position(|&c| c == true_class)
                .ok_or_else(|| {
                    SklearsError::InvalidInput(format!("Unknown class: {}", true_class))
                })?;

            total_log_likelihood += joint_log_likelihood[[i, class_idx]];
        }

        Ok(total_log_likelihood)
    }

    fn get_n_parameters(&self) -> usize {
        let n_classes = self.classes_.as_ref().unwrap().len();
        let n_features = self.theta_.as_ref().unwrap().ncols();

        // Parameters: class priors + means + variances for each class-feature combination
        (n_classes - 1) + n_classes * n_features * 2
    }

    fn sample(&self, x: &Array2<f64>) -> std::result::Result<Array1<i32>, Self::Error> {
        // For simplicity, return predictions rather than actual sampling
        // A proper implementation would sample from the posterior distribution
        Predict::predict(self, x)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gaussian_nb_basic() {
        // Simple 2D data with two classes
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
            [-4.0, -5.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = GaussianNB::new().var_smoothing(1e-9).fit(&x, &y).unwrap();

        // Test predictions
        let predictions = Predict::predict(&model, &x).unwrap();
        assert_eq!(predictions, y);

        // Test score
        let score = model.score(&x, &y).unwrap();
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_gaussian_nb_predict_proba() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]];
        let y = array![0, 0, 1, 1];

        let model = GaussianNB::new().fit(&x, &y).unwrap();
        let proba = PredictProba::predict_proba(&model, &x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        // Check that class 0 has higher probability for first two samples
        assert!(proba[[0, 0]] > proba[[0, 1]]);
        assert!(proba[[1, 0]] > proba[[1, 1]]);

        // Check that class 1 has higher probability for last two samples
        assert!(proba[[2, 1]] > proba[[2, 0]]);
        assert!(proba[[3, 1]] > proba[[3, 0]]);
    }

    #[test]
    fn test_gaussian_nb_with_priors() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]];
        let y = array![0, 0, 1, 1];

        // Set custom priors
        let priors = array![0.3, 0.7];
        let model = GaussianNB::new().priors(priors).fit(&x, &y).unwrap();

        // The model should still work with custom priors
        let predictions = Predict::predict(&model, &x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_gaussian_nb_information_criteria() {
        use crate::validation::ProbabilisticValidator;

        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
            [-4.0, -5.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let mut model = GaussianNB::new().fit(&x, &y).unwrap();

        // Test log likelihood calculation
        let log_likelihood = model.log_likelihood(&x, &y).unwrap();
        assert!(log_likelihood < 0.0); // Log likelihood should be negative

        // Test number of parameters calculation
        let n_params = model.get_n_parameters();
        // For 2 classes, 2 features: (2-1) class priors + 2*2*2 (means + variances) = 1 + 8 = 9
        assert_eq!(n_params, 9);

        // Test model criticism with information criteria
        let validator = ProbabilisticValidator::new(crate::validation::CVStrategy::KFold(3));
        let criticism_results = validator.model_criticism(&x, &y, &mut model).unwrap();

        // Check that AIC and BIC are computed and reasonable
        assert!(criticism_results.aic > 0.0);
        assert!(criticism_results.bic > 0.0);
        assert!(criticism_results.bic > criticism_results.aic); // BIC should penalize more for small samples
        assert_abs_diff_eq!(
            criticism_results.deviance,
            -2.0 * log_likelihood,
            epsilon = 1e-10
        );
    }
}

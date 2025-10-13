//! Categorical Naive Bayes classifier implementation

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result},
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, PredictProba, Score, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::{safe_log, NaiveBayesMixin};

/// Configuration for Categorical Naive Bayes
#[derive(Debug, Clone)]
pub struct CategoricalNBConfig {
    /// Additive (Laplace/Lidstone) smoothing parameter
    pub alpha: f64,
    /// Whether to learn class prior probabilities
    pub fit_prior: bool,
    /// Prior probabilities of the classes
    pub class_prior: Option<Array1<f64>>,
}

impl Default for CategoricalNBConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_prior: true,
            class_prior: None,
        }
    }
}

/// Categorical Naive Bayes classifier
///
/// For categorical features. Each feature is assumed to have a finite set of possible
/// values, and the probability of each category is computed independently.
#[derive(Debug, Clone)]
pub struct CategoricalNB<State = Untrained> {
    config: CategoricalNBConfig,
    state: PhantomData<State>,
    // Trained state fields
    feature_log_prob_: Option<Vec<Array2<f64>>>, // List of log prob arrays, one per feature
    class_log_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
    n_features_: Option<usize>,
    n_categories_: Option<Array1<usize>>, // Number of categories per feature
}

impl CategoricalNB<Untrained> {
    /// Create a new Categorical Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: CategoricalNBConfig::default(),
            state: PhantomData,
            feature_log_prob_: None,
            class_log_prior_: None,
            classes_: None,
            n_features_: None,
            n_categories_: None,
        }
    }

    /// Set smoothing parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        if alpha < 0.0 {
            panic!("alpha must be >= 0");
        }
        self.config.alpha = alpha;
        self
    }

    /// Set whether to learn class prior probabilities
    pub fn fit_prior(mut self, fit_prior: bool) -> Self {
        self.config.fit_prior = fit_prior;
        self
    }

    /// Set class prior probabilities
    pub fn class_prior(mut self, class_prior: Array1<f64>) -> Self {
        self.config.class_prior = Some(class_prior);
        self
    }
}

impl Default for CategoricalNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CategoricalNB<Untrained> {
    type Float = Float;
    type Config = CategoricalNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for CategoricalNB<Untrained> {
    type Fitted = CategoricalNB<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Check that features are non-negative integers
        for &val in x.iter() {
            if val < 0.0 || val.fract() != 0.0 {
                return Err(SklearsError::InvalidInput(
                    "CategoricalNB requires non-negative integer features".to_string(),
                ));
            }
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();
        let n_features = x.ncols();

        // Determine number of categories per feature
        let mut n_categories = Array1::zeros(n_features);
        for j in 0..n_features {
            let max_val = x.column(j).iter().cloned().fold(0.0, f64::max) as usize;
            n_categories[j] = max_val + 1; // Categories are 0-indexed
        }

        // Initialize category counts for each feature
        let mut category_counts: Vec<Array2<f64>> = Vec::with_capacity(n_features);
        for j in 0..n_features {
            category_counts.push(Array2::zeros((n_classes, n_categories[j])));
        }

        // Count occurrences
        let mut class_count: Array1<f64> = Array1::zeros(n_classes);
        for (i, &label) in y.iter().enumerate() {
            let class_idx = classes.iter().position(|&c| c == label).unwrap();
            class_count[class_idx] += 1.0;

            let sample = x.row(i);
            for (j, &val) in sample.iter().enumerate() {
                let cat_idx = val as usize;
                if cat_idx < n_categories[j] {
                    category_counts[j][[class_idx, cat_idx]] += 1.0;
                }
            }
        }

        // Compute log probabilities with smoothing
        let mut feature_log_prob = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let smoothed_cat_count = &category_counts[j] + self.config.alpha;
            let smoothed_class_count = smoothed_cat_count.sum_axis(Axis(1));
            let mut log_prob = Array2::zeros((n_classes, n_categories[j]));

            for i in 0..n_classes {
                for k in 0..n_categories[j] {
                    log_prob[[i, k]] =
                        safe_log(smoothed_cat_count[[i, k]] / smoothed_class_count[i]);
                }
            }
            feature_log_prob.push(log_prob);
        }

        // Compute class log priors
        let class_log_prior = if self.config.fit_prior {
            if let Some(ref priors) = self.config.class_prior {
                if priors.len() != n_classes {
                    return Err(SklearsError::InvalidInput(format!(
                        "Number of priors ({}) doesn't match number of classes ({})",
                        priors.len(),
                        n_classes
                    )));
                }
                priors.mapv(safe_log)
            } else {
                let n_samples = y.len() as f64;
                class_count.mapv(|c| safe_log(c / n_samples))
            }
        } else {
            Array1::from_elem(n_classes, -(n_classes as f64).ln())
        };

        Ok(CategoricalNB {
            config: self.config,
            state: PhantomData,
            feature_log_prob_: Some(feature_log_prob),
            class_log_prior_: Some(class_log_prior),
            classes_: Some(classes),
            n_features_: Some(n_features),
            n_categories_: Some(n_categories),
        })
    }
}

impl CategoricalNB<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let feature_log_prob = self.feature_log_prob_.as_ref().unwrap();
        let class_log_prior = self.class_log_prior_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let n_categories = self.n_categories_.as_ref().unwrap();

        // Check that features are valid
        for j in 0..self.n_features_.unwrap() {
            for &val in x.column(j).iter() {
                if val < 0.0 || val.fract() != 0.0 || val as usize >= n_categories[j] {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature {} has invalid value {}. Expected integer in range [0, {})",
                        j, val, n_categories[j]
                    )));
                }
            }
        }

        let mut joint_log_likelihood = Array2::zeros((n_samples, n_classes));

        // Add log prior
        for i in 0..n_samples {
            for j in 0..n_classes {
                joint_log_likelihood[[i, j]] = class_log_prior[j];
            }
        }

        // Add feature log probabilities
        for i in 0..n_samples {
            let sample = x.row(i);
            for (j, &val) in sample.iter().enumerate() {
                let cat_idx = val as usize;
                for k in 0..n_classes {
                    joint_log_likelihood[[i, k]] += feature_log_prob[j][[k, cat_idx]];
                }
            }
        }

        Ok(joint_log_likelihood)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for CategoricalNB<Trained> {
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

impl PredictProba<Array2<Float>, Array2<f64>> for CategoricalNB<Trained> {
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

impl Score<Array2<Float>, Array1<i32>> for CategoricalNB<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<f64> {
        let predictions = self.predict(x)?;
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_val)| pred == true_val)
            .count();

        Ok(correct as f64 / y.len() as f64)
    }
}

impl NaiveBayesMixin for CategoricalNB<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_log_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        // For CategoricalNB, we can't easily return a single 2D array
        // as we have a list of arrays. For compatibility, we'll return
        // the first feature's log probabilities.
        &self.feature_log_prob_.as_ref().unwrap()[0]
    }

    fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
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
    fn test_categorical_nb_creation() {
        let nb = CategoricalNB::new();
        assert_eq!(nb.config.alpha, 1.0);
        assert_eq!(nb.config.fit_prior, true);
    }

    #[test]
    fn test_categorical_nb_basic() {
        // Categorical features (e.g., colors: 0=red, 1=green, 2=blue)
        let x = array![
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let model = CategoricalNB::new().alpha(1.0).fit(&x, &y).unwrap();

        // Test predictions
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);

        // Test score
        let score = model.score(&x, &y).unwrap();
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_categorical_nb_with_smoothing() {
        // Test with different alpha values
        let x = array![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
        let y = array![0, 1, 0, 1];

        let model = CategoricalNB::new()
            .alpha(0.5) // Different smoothing
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);
    }

    #[test]
    fn test_categorical_nb_predict_proba() {
        let x = array![[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]];
        let y = array![0, 1, 0, 1];

        let model = CategoricalNB::new().fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        // Check that predictions match highest probability
        let predictions = model.predict(&x).unwrap();
        for i in 0..x.nrows() {
            let max_idx = if proba[[i, 0]] > proba[[i, 1]] { 0 } else { 1 };
            assert_eq!(predictions[i], model.classes_.as_ref().unwrap()[max_idx]);
        }
    }

    #[test]
    fn test_categorical_nb_with_priors() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![0, 1];

        // Set custom priors
        let priors = array![0.3, 0.7];
        let model = CategoricalNB::new()
            .class_prior(priors)
            .fit(&x, &y)
            .unwrap();

        // The model should work with custom priors
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_categorical_nb_invalid_features() {
        // Non-integer features
        let x = array![
            [0.5, 1.0], // 0.5 is not an integer
            [1.0, 0.0]
        ];
        let y = array![0, 1];

        let result = CategoricalNB::new().fit(&x, &y);
        assert!(result.is_err());

        // Negative features
        let x = array![
            [-1.0, 1.0], // -1 is negative
            [1.0, 0.0]
        ];
        let result = CategoricalNB::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_categorical_nb_larger_categories() {
        // Test with more categories
        let x = array![
            [0.0, 3.0, 1.0],
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 2.0],
            [3.0, 0.0, 3.0],
            [0.0, 3.0, 1.0],
            [1.0, 2.0, 0.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let model = CategoricalNB::new().fit(&x, &y).unwrap();
        let score = model.score(&x, &y).unwrap();
        assert!(score >= 0.5); // Should perform reasonably well
    }

    #[test]
    #[should_panic(expected = "alpha must be >= 0")]
    fn test_categorical_nb_negative_alpha() {
        CategoricalNB::new().alpha(-1.0);
    }
}

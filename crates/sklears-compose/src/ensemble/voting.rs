//! Voting ensemble implementations
//!
//! This module provides `VotingClassifier` and `VotingRegressor` implementations
//! for combining multiple estimators through majority voting or averaging.

use super::common::simd_fallback;
use crate::PipelinePredictor;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use sklears_core::{
    error::Result as SklResult,
    prelude::SklearsError,
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// `VotingClassifier`
///
/// Ensemble classifier that combines multiple classifiers using majority voting
/// or probability averaging.
///
/// # Type Parameters
///
/// * `S` - State type (Untrained or `VotingClassifierTrained`)
///
/// # Examples
///
/// ```ignore
/// use sklears_compose::{VotingClassifier, MockPredictor};
/// use scirs2_core::ndarray::array;
///
/// let voting_clf = VotingClassifier::builder()
///     .estimator("clf1", Box::new(MockPredictor::new()))
///     .estimator("clf2", Box::new(MockPredictor::new()))
///     .voting("hard")
///     .build();
/// ```
pub struct VotingClassifier<S = Untrained> {
    state: S,
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    voting: String, // "hard" or "soft"
    weights: Option<Vec<f64>>,
    n_jobs: Option<i32>,
    flatten_transform: bool,
}

/// `VotingRegressor`
///
/// Ensemble regressor that combines multiple regressors by averaging their predictions.
///
/// # Type Parameters
///
/// * `S` - State type (Untrained or `VotingRegressorTrained`)
///
/// # Examples
///
/// ```ignore
/// use sklears_compose::{VotingRegressor, MockPredictor};
/// use scirs2_core::ndarray::array;
///
/// let voting_reg = VotingRegressor::builder()
///     .estimator("reg1", Box::new(MockPredictor::new()))
///     .estimator("reg2", Box::new(MockPredictor::new()))
///     .build();
/// ```
pub struct VotingRegressor<S = Untrained> {
    state: S,
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    weights: Option<Vec<f64>>,
    n_jobs: Option<i32>,
}

/// Trained state for `VotingClassifier`
pub struct VotingClassifierTrained {
    fitted_estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    classes: Array1<f64>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

/// Trained state for `VotingRegressor`
pub struct VotingRegressorTrained {
    fitted_estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl VotingClassifier<Untrained> {
    /// Create a new `VotingClassifier` instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            estimators: Vec::new(),
            voting: "hard".to_string(),
            weights: None,
            n_jobs: None,
            flatten_transform: true,
        }
    }

    /// Create a voting classifier builder
    #[must_use]
    pub fn builder() -> VotingClassifierBuilder {
        VotingClassifierBuilder::new()
    }

    /// Add an estimator
    pub fn add_estimator(&mut self, name: String, estimator: Box<dyn PipelinePredictor>) {
        self.estimators.push((name, estimator));
    }

    /// Set voting strategy
    #[must_use]
    pub fn voting(mut self, voting: &str) -> Self {
        self.voting = voting.to_string();
        self
    }

    /// Set weights
    #[must_use]
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set number of jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set flatten transform
    #[must_use]
    pub fn flatten_transform(mut self, flatten: bool) -> Self {
        self.flatten_transform = flatten;
        self
    }
}

impl Default for VotingClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for VotingClassifier<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for VotingClassifier<Untrained> {
    type Fitted = VotingClassifier<VotingClassifierTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        if let Some(y_values) = y.as_ref() {
            let mut fitted_estimators = Vec::new();

            for (name, mut estimator) in self.estimators {
                estimator.fit(x, y_values)?;
                fitted_estimators.push((name, estimator));
            }

            // Extract unique classes
            let mut classes: Vec<f64> = y_values.to_vec();
            classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            classes.dedup();
            let classes = Array1::from(classes);

            Ok(VotingClassifier {
                state: VotingClassifierTrained {
                    fitted_estimators,
                    classes,
                    n_features_in: x.ncols(),
                    feature_names_in: None,
                },
                estimators: Vec::new(),
                voting: self.voting,
                weights: self.weights,
                n_jobs: self.n_jobs,
                flatten_transform: self.flatten_transform,
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Target values required for fitting".to_string(),
            ))
        }
    }
}

impl VotingRegressor<Untrained> {
    /// Create a new `VotingRegressor` instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            estimators: Vec::new(),
            weights: None,
            n_jobs: None,
        }
    }

    /// Create a voting regressor builder
    #[must_use]
    pub fn builder() -> VotingRegressorBuilder {
        VotingRegressorBuilder::new()
    }

    /// Add an estimator
    pub fn add_estimator(&mut self, name: String, estimator: Box<dyn PipelinePredictor>) {
        self.estimators.push((name, estimator));
    }

    /// Set weights
    #[must_use]
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set number of jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for VotingRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for VotingRegressor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for VotingRegressor<Untrained> {
    type Fitted = VotingRegressor<VotingRegressorTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        if let Some(y_values) = y.as_ref() {
            let mut fitted_estimators = Vec::new();

            for (name, mut estimator) in self.estimators {
                estimator.fit(x, y_values)?;
                fitted_estimators.push((name, estimator));
            }

            Ok(VotingRegressor {
                state: VotingRegressorTrained {
                    fitted_estimators,
                    n_features_in: x.ncols(),
                    feature_names_in: None,
                },
                estimators: Vec::new(),
                weights: self.weights,
                n_jobs: self.n_jobs,
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Target values required for fitting".to_string(),
            ))
        }
    }
}

/// `VotingClassifier` builder for fluent construction
pub struct VotingClassifierBuilder {
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    voting: String,
    weights: Option<Vec<f64>>,
    n_jobs: Option<i32>,
    flatten_transform: bool,
}

impl VotingClassifierBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
            voting: "hard".to_string(),
            weights: None,
            n_jobs: None,
            flatten_transform: true,
        }
    }

    /// Add an estimator
    #[must_use]
    pub fn estimator(mut self, name: &str, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.estimators.push((name.to_string(), estimator));
        self
    }

    /// Set voting strategy
    #[must_use]
    pub fn voting(mut self, voting: &str) -> Self {
        self.voting = voting.to_string();
        self
    }

    /// Set weights
    #[must_use]
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set number of jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set flatten transform
    #[must_use]
    pub fn flatten_transform(mut self, flatten: bool) -> Self {
        self.flatten_transform = flatten;
        self
    }

    /// Build the `VotingClassifier`
    #[must_use]
    pub fn build(self) -> VotingClassifier<Untrained> {
        VotingClassifier {
            state: Untrained,
            estimators: self.estimators,
            voting: self.voting,
            weights: self.weights,
            n_jobs: self.n_jobs,
            flatten_transform: self.flatten_transform,
        }
    }
}

/// `VotingRegressor` builder for fluent construction
pub struct VotingRegressorBuilder {
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    weights: Option<Vec<f64>>,
    n_jobs: Option<i32>,
}

impl VotingRegressorBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
            weights: None,
            n_jobs: None,
        }
    }

    /// Add an estimator
    #[must_use]
    pub fn estimator(mut self, name: &str, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.estimators.push((name.to_string(), estimator));
        self
    }

    /// Set weights
    #[must_use]
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set number of jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Build the `VotingRegressor`
    #[must_use]
    pub fn build(self) -> VotingRegressor<Untrained> {
        VotingRegressor {
            state: Untrained,
            estimators: self.estimators,
            weights: self.weights,
            n_jobs: self.n_jobs,
        }
    }
}

impl Default for VotingClassifierBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for VotingRegressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD-accelerated ensemble aggregation functions for voting ensembles
pub mod simd_voting {
    use super::{simd_fallback, Array1};

    /// SIMD-accelerated weighted averaging of predictions
    #[must_use]
    pub fn simd_weighted_average_predictions(
        all_predictions: &[Array1<f64>],
        weights: &[f64],
    ) -> Array1<f64> {
        if all_predictions.is_empty() || weights.is_empty() {
            return Array1::zeros(0);
        }

        let n_samples = all_predictions[0].len();
        let n_estimators = all_predictions.len().min(weights.len());

        // Convert to f32 for SIMD processing
        let predictions_f32: Vec<Vec<f32>> = all_predictions[..n_estimators]
            .iter()
            .map(|pred| pred.iter().map(|&x| x as f32).collect())
            .collect();

        let weights_f32: Vec<f32> = weights[..n_estimators].iter().map(|&x| x as f32).collect();

        // Initialize result vector
        let mut result_f32 = vec![0.0f32; n_samples];

        // SIMD-accelerated weighted sum across all estimators
        for (i, (pred, &weight)) in predictions_f32.iter().zip(weights_f32.iter()).enumerate() {
            if i == 0 {
                // First prediction: multiply by weight and store
                simd_fallback::scale_vec(pred, weight, &mut result_f32);
            } else {
                // Subsequent predictions: multiply by weight and accumulate
                let mut weighted_pred = vec![0.0f32; n_samples];
                simd_fallback::scale_vec(pred, weight, &mut weighted_pred);
                for i in 0..result_f32.len() {
                    result_f32[i] += weighted_pred[i];
                }
            }
        }

        // Convert back to f64 and return as Array1
        let result_f64: Vec<f64> = result_f32.iter().map(|&x| f64::from(x)).collect();
        Array1::from_vec(result_f64)
    }
}

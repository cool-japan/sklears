//! One-vs-Rest Discriminant Analysis implementation
//!
//! This module implements One-vs-Rest (OvR) classification strategy for
//! discriminant analysis, extending binary classifiers to multi-class problems.

use crate::{LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis};
// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{validate, Result},
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for One-vs-Rest Discriminant Analysis
#[derive(Debug, Clone)]
pub struct OneVsRestDiscriminantAnalysisConfig {
    /// Base classifier type ("lda" or "qda")
    pub classifier_type: String,
    /// Shrinkage parameter for LDA (if using LDA)
    pub shrinkage: Option<Float>,
    /// Regularization parameter for QDA (if using QDA)
    pub reg_param: Float,
    /// Number of components for dimensionality reduction
    pub n_components: Option<usize>,
    /// Prior probabilities of the classes
    pub priors: Option<Array1<Float>>,
    /// Whether to store the covariance matrix
    pub store_covariance: bool,
    /// Tolerance for convergence
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to use class balancing
    pub balance_classes: bool,
}

impl Default for OneVsRestDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            classifier_type: "lda".to_string(),
            shrinkage: None,
            reg_param: 1e-6,
            n_components: None,
            priors: None,
            store_covariance: false,
            tol: 1e-8,
            max_iter: 100,
            balance_classes: false,
        }
    }
}

/// Classifier wrapper for binary classification in OvR
#[derive(Debug, Clone)]
pub enum BinaryClassifier<State = Untrained> {
    /// LDA
    LDA(LinearDiscriminantAnalysis<State>),
    /// QDA
    QDA(QuadraticDiscriminantAnalysis<State>),
}

impl BinaryClassifier<Untrained> {
    pub fn new_lda() -> Self {
        /// BinaryClassifier
        BinaryClassifier::LDA(LinearDiscriminantAnalysis::new())
    }

    pub fn new_qda() -> Self {
        BinaryClassifier::QDA(QuadraticDiscriminantAnalysis::new())
    }

    pub fn fit_binary(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<BinaryClassifier<Trained>> {
        match self {
            BinaryClassifier::LDA(lda) => {
                let fitted = lda.clone().fit(x, y)?;
                Ok(BinaryClassifier::LDA(fitted))
            }
            BinaryClassifier::QDA(qda) => {
                let fitted = qda.clone().fit(x, y)?;
                Ok(BinaryClassifier::QDA(fitted))
            }
        }
    }
}

impl BinaryClassifier<Trained> {
    pub fn predict_binary(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        match self {
            BinaryClassifier::LDA(lda) => lda.predict(x),
            BinaryClassifier::QDA(qda) => qda.predict(x),
        }
    }

    pub fn predict_proba_binary(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        match self {
            BinaryClassifier::LDA(lda) => lda.predict_proba(x),
            BinaryClassifier::QDA(qda) => qda.predict_proba(x),
        }
    }

    pub fn classes_binary(&self) -> &Array1<i32> {
        match self {
            BinaryClassifier::LDA(lda) => lda.classes(),
            BinaryClassifier::QDA(qda) => qda.classes(),
        }
    }
}

/// One-vs-Rest Discriminant Analysis
///
/// A multi-class classification strategy that trains one binary classifier per class.
/// Each classifier distinguishes one class from all others (rest).
///
/// During prediction:
/// 1. Each binary classifier predicts the probability that a sample belongs to its positive class
/// 2. The final prediction is the class with the highest probability
#[derive(Debug, Clone)]
pub struct OneVsRestDiscriminantAnalysis<State = Untrained> {
    config: OneVsRestDiscriminantAnalysisConfig,
    state: PhantomData<State>,
    // Trained state fields
    classes_: Option<Array1<i32>>,
    classifiers_: Option<Vec<BinaryClassifier<Trained>>>,
    n_features_: Option<usize>,
}

impl OneVsRestDiscriminantAnalysis<Untrained> {
    /// Create a new OneVsRestDiscriminantAnalysis instance
    pub fn new() -> Self {
        Self {
            config: OneVsRestDiscriminantAnalysisConfig::default(),
            state: PhantomData,
            classes_: None,
            classifiers_: None,
            n_features_: None,
        }
    }

    /// Set the base classifier type
    pub fn classifier_type(mut self, classifier_type: &str) -> Self {
        self.config.classifier_type = classifier_type.to_string();
        self
    }

    /// Set the shrinkage parameter for LDA
    pub fn shrinkage(mut self, shrinkage: Option<Float>) -> Self {
        self.config.shrinkage = shrinkage;
        self
    }

    /// Set the regularization parameter for QDA
    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set the prior probabilities
    pub fn priors(mut self, priors: Option<Array1<Float>>) -> Self {
        self.config.priors = priors;
        self
    }

    /// Set whether to store covariance matrix
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

    /// Set whether to use class balancing
    pub fn balance_classes(mut self, balance_classes: bool) -> Self {
        self.config.balance_classes = balance_classes;
        self
    }

    /// Extract unique classes from labels
    fn unique_classes(&self, y: &Array1<i32>) -> Result<Array1<i32>> {
        let mut classes = Vec::new();
        for &label in y.iter() {
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        classes.sort_unstable();
        Ok(Array1::from_vec(classes))
    }

    /// Create binary labels for one-vs-rest
    fn create_binary_labels(&self, y: &Array1<i32>, positive_class: i32) -> Array1<i32> {
        y.mapv(|label| if label == positive_class { 1 } else { 0 })
    }

    /// Balance class weights if requested
    fn balance_sample_weights(&self, y: &Array1<i32>, positive_class: i32) -> Array1<Float> {
        let n_samples = y.len();
        let n_positive = y.iter().filter(|&&label| label == positive_class).count();
        let n_negative = n_samples - n_positive;

        if self.config.balance_classes && n_positive > 0 && n_negative > 0 {
            let pos_weight = n_samples as Float / (2.0 * n_positive as Float);
            let neg_weight = n_samples as Float / (2.0 * n_negative as Float);

            y.mapv(|label| {
                if label == positive_class {
                    pos_weight
                } else {
                    neg_weight
                }
            })
        } else {
            Array1::ones(n_samples)
        }
    }

    /// Create base classifier instance
    fn create_base_classifier(&self) -> Result<BinaryClassifier<Untrained>> {
        match self.config.classifier_type.as_str() {
            "lda" => {
                let mut lda = LinearDiscriminantAnalysis::new()
                    .store_covariance(self.config.store_covariance)
                    .tol(self.config.tol);

                if let Some(shrinkage) = self.config.shrinkage {
                    lda = lda.shrinkage(Some(shrinkage));
                }

                if let Some(n_components) = self.config.n_components {
                    lda = lda.n_components(Some(n_components));
                }

                Ok(BinaryClassifier::LDA(lda))
            }
            "qda" => {
                let qda = QuadraticDiscriminantAnalysis::new()
                    .reg_param(self.config.reg_param)
                    .store_covariance(self.config.store_covariance)
                    .tol(self.config.tol);

                Ok(BinaryClassifier::QDA(qda))
            }
            _ => Err(SklearsError::InvalidParameter {
                name: "classifier_type".to_string(),
                reason: format!("Unknown classifier type: {}", self.config.classifier_type),
            }),
        }
    }
}

impl Estimator for OneVsRestDiscriminantAnalysis<Untrained> {
    type Config = OneVsRestDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for OneVsRestDiscriminantAnalysis<Untrained> {
    type Fitted = OneVsRestDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Get unique classes
        let classes = self.unique_classes(y)?;
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Train one binary classifier per class
        let mut classifiers = Vec::with_capacity(n_classes);

        for &class in classes.iter() {
            // Create binary labels (current class vs rest)
            let binary_labels = self.create_binary_labels(y, class);

            // Create and configure base classifier
            let base_classifier = self.create_base_classifier()?;

            // Train binary classifier
            let trained_classifier = base_classifier.fit_binary(x, &binary_labels)?;
            classifiers.push(trained_classifier);
        }

        Ok(OneVsRestDiscriminantAnalysis {
            config: self.config,
            state: PhantomData,
            classes_: Some(classes),
            classifiers_: Some(classifiers),
            n_features_: Some(n_features),
        })
    }
}

impl OneVsRestDiscriminantAnalysis<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the trained classifiers
    pub fn classifiers(&self) -> &Vec<BinaryClassifier<Trained>> {
        self.classifiers_.as_ref().unwrap()
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features_.unwrap()
    }
}

impl Predict<Array2<Float>, Array1<i32>> for OneVsRestDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes();

        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut best_class_idx = 0;
            let mut best_proba = probas[[i, 0]];

            for j in 1..classes.len() {
                if probas[[i, j]] > best_proba {
                    best_proba = probas[[i, j]];
                    best_class_idx = j;
                }
            }

            predictions[i] = classes[best_class_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for OneVsRestDiscriminantAnalysis<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if self.classifiers_.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            });
        }

        if x.ncols() != self.n_features() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                x.ncols()
            )));
        }

        let classes = self.classes();
        let classifiers = self.classifiers();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        let mut decision_scores = Array2::zeros((n_samples, n_classes));

        // Get decision scores from each binary classifier
        for (class_idx, classifier) in classifiers.iter().enumerate() {
            let binary_probas = classifier.predict_proba_binary(x)?;

            // Take the probability of positive class (class 1)
            for i in 0..n_samples {
                // binary_probas has shape (n_samples, 2) where column 1 is positive class probability
                decision_scores[[i, class_idx]] = binary_probas[[i, 1]];
            }
        }

        // Normalize decision scores to probabilities using softmax
        let mut probas = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let max_score = decision_scores
                .row(i)
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;

            // Compute softmax
            for j in 0..n_classes {
                let exp_score = (decision_scores[[i, j]] - max_score).exp();
                probas[[i, j]] = exp_score;
                sum_exp += exp_score;
            }

            // Normalize
            for j in 0..n_classes {
                probas[[i, j]] /= sum_exp;
            }
        }

        Ok(probas)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for OneVsRestDiscriminantAnalysis<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // For OvR, transformation can mean getting decision scores from all classifiers
        if self.classifiers_.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        if x.ncols() != self.n_features() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                x.ncols()
            )));
        }

        let classifiers = self.classifiers();
        let n_classes = classifiers.len();
        let n_samples = x.nrows();

        let mut decision_scores = Array2::zeros((n_samples, n_classes));

        // Get decision scores from each binary classifier
        for (class_idx, classifier) in classifiers.iter().enumerate() {
            let binary_probas = classifier.predict_proba_binary(x)?;

            // Take the log probability of positive class
            for i in 0..n_samples {
                decision_scores[[i, class_idx]] = binary_probas[[i, 1]].ln();
            }
        }

        Ok(decision_scores)
    }
}

impl Default for OneVsRestDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

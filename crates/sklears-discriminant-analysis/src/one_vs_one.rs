//! One-vs-One Discriminant Analysis implementation
//!
//! This module implements One-vs-One (OvO) classification strategy for
//! discriminant analysis, training binary classifiers for each pair of classes.

use crate::one_vs_rest::BinaryClassifier;
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

/// Configuration for One-vs-One Discriminant Analysis
#[derive(Debug, Clone)]
pub struct OneVsOneDiscriminantAnalysisConfig {
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
    /// Voting strategy ("hard" or "soft")
    pub voting: String,
}

impl Default for OneVsOneDiscriminantAnalysisConfig {
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
            voting: "hard".to_string(),
        }
    }
}

/// Pairwise classifier information
#[derive(Debug, Clone)]
pub struct PairwiseClassifier {
    /// First class in the pair
    pub class_i: i32,
    /// Second class in the pair
    pub class_j: i32,
    /// Trained binary classifier
    pub classifier: BinaryClassifier<Trained>,
}

/// One-vs-One Discriminant Analysis
///
/// A multi-class classification strategy that trains one binary classifier for each pair of classes.
/// For k classes, this results in k*(k-1)/2 binary classifiers.
///
/// During prediction:
/// 1. Each pairwise classifier votes for one of its two classes
/// 2. The final prediction is determined by majority vote (hard voting) or
///    probability aggregation (soft voting)
#[derive(Debug, Clone)]
pub struct OneVsOneDiscriminantAnalysis<State = Untrained> {
    config: OneVsOneDiscriminantAnalysisConfig,
    state: PhantomData<State>,
    // Trained state fields
    classes_: Option<Array1<i32>>,
    pairwise_classifiers_: Option<Vec<PairwiseClassifier>>,
    n_features_: Option<usize>,
}

impl OneVsOneDiscriminantAnalysis<Untrained> {
    /// Create a new OneVsOneDiscriminantAnalysis instance
    pub fn new() -> Self {
        Self {
            config: OneVsOneDiscriminantAnalysisConfig::default(),
            state: PhantomData,
            classes_: None,
            pairwise_classifiers_: None,
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

    /// Set the voting strategy
    pub fn voting(mut self, voting: &str) -> Self {
        self.config.voting = voting.to_string();
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

    /// Extract samples and labels for a specific pair of classes
    fn extract_pairwise_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        class_i: i32,
        class_j: i32,
    ) -> (Array2<Float>, Array1<i32>) {
        // Find indices of samples belonging to class_i or class_j
        let indices: Vec<usize> = y
            .iter()
            .enumerate()
            .filter_map(|(idx, &label)| {
                if label == class_i || label == class_j {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        let n_pairwise_samples = indices.len();
        let n_features = x.ncols();

        // Extract data
        let mut pairwise_x = Array2::zeros((n_pairwise_samples, n_features));
        let mut pairwise_y = Array1::zeros(n_pairwise_samples);

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            for j in 0..n_features {
                pairwise_x[[new_idx, j]] = x[[old_idx, j]];
            }
            // Map class labels to binary: class_i -> 0, class_j -> 1
            pairwise_y[new_idx] = if y[old_idx] == class_i { 0 } else { 1 };
        }

        (pairwise_x, pairwise_y)
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

impl Estimator for OneVsOneDiscriminantAnalysis<Untrained> {
    type Config = OneVsOneDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for OneVsOneDiscriminantAnalysis<Untrained> {
    type Fitted = OneVsOneDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let _n_samples = x.nrows();
        let n_features = x.ncols();

        // Get unique classes
        let classes = self.unique_classes(y)?;
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Train one binary classifier for each pair of classes
        let mut pairwise_classifiers = Vec::new();

        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let class_i = classes[i];
                let class_j = classes[j];

                // Extract data for this pair
                let (pairwise_x, pairwise_y) = self.extract_pairwise_data(x, y, class_i, class_j);

                if pairwise_x.nrows() == 0 {
                    continue; // Skip if no samples for this pair
                }

                // Create and train base classifier
                let base_classifier = self.create_base_classifier()?;
                let trained_classifier = base_classifier.fit_binary(&pairwise_x, &pairwise_y)?;

                pairwise_classifiers.push(PairwiseClassifier {
                    class_i,
                    class_j,
                    classifier: trained_classifier,
                });
            }
        }

        Ok(OneVsOneDiscriminantAnalysis {
            config: self.config,
            state: PhantomData,
            classes_: Some(classes),
            pairwise_classifiers_: Some(pairwise_classifiers),
            n_features_: Some(n_features),
        })
    }
}

impl OneVsOneDiscriminantAnalysis<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the pairwise classifiers
    pub fn pairwise_classifiers(&self) -> &Vec<PairwiseClassifier> {
        self.pairwise_classifiers_.as_ref().unwrap()
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features_.unwrap()
    }

    /// Perform hard voting: each classifier votes for one class
    fn hard_voting(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let classes = self.classes();
        let pairwise_classifiers = self.pairwise_classifiers();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        let mut votes: Array2<Float> = Array2::zeros((n_samples, n_classes));

        // Collect votes from all pairwise classifiers
        for pairwise in pairwise_classifiers {
            let predictions = pairwise.classifier.predict_binary(x)?;

            // Find indices of class_i and class_j in the classes array
            let class_i_idx = classes.iter().position(|&c| c == pairwise.class_i).unwrap();
            let class_j_idx = classes.iter().position(|&c| c == pairwise.class_j).unwrap();

            for (sample_idx, &prediction) in predictions.iter().enumerate() {
                if prediction == 0 {
                    votes[[sample_idx, class_i_idx]] += 1.0;
                } else {
                    votes[[sample_idx, class_j_idx]] += 1.0;
                }
            }
        }

        // Find class with most votes for each sample
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut best_class_idx = 0;
            let mut best_votes: Float = votes[[i, 0]];

            for j in 1..n_classes {
                if votes[[i, j]] > best_votes {
                    best_votes = votes[[i, j]];
                    best_class_idx = j;
                }
            }

            predictions[i] = classes[best_class_idx];
        }

        Ok(predictions)
    }

    /// Perform soft voting: aggregate probabilities from all classifiers
    fn soft_voting(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let classes = self.classes();
        let pairwise_classifiers = self.pairwise_classifiers();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        let mut probability_sums: Array2<Float> = Array2::zeros((n_samples, n_classes));
        let mut vote_counts: Array1<Float> = Array1::zeros(n_classes);

        // Collect probability votes from all pairwise classifiers
        for pairwise in pairwise_classifiers {
            let probas = pairwise.classifier.predict_proba_binary(x)?;

            // Find indices of class_i and class_j in the classes array
            let class_i_idx = classes.iter().position(|&c| c == pairwise.class_i).unwrap();
            let class_j_idx = classes.iter().position(|&c| c == pairwise.class_j).unwrap();

            for sample_idx in 0..n_samples {
                // probas has shape (n_samples, 2) where column 0 is class_i, column 1 is class_j
                probability_sums[[sample_idx, class_i_idx]] += probas[[sample_idx, 0]];
                probability_sums[[sample_idx, class_j_idx]] += probas[[sample_idx, 1]];
            }

            vote_counts[class_i_idx] += 1.0;
            vote_counts[class_j_idx] += 1.0;
        }

        // Normalize by number of votes each class received
        let mut normalized_probas = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            for j in 0..n_classes {
                if vote_counts[j] > 0.0 {
                    normalized_probas[[i, j]] = probability_sums[[i, j]] / vote_counts[j];
                }
            }

            // Normalize to sum to 1
            let row_sum: Float = normalized_probas.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_classes {
                    normalized_probas[[i, j]] /= row_sum;
                }
            } else {
                // If all probabilities are zero, use uniform distribution
                for j in 0..n_classes {
                    normalized_probas[[i, j]] = 1.0 / n_classes as Float;
                }
            }
        }

        Ok(normalized_probas)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for OneVsOneDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        if self.pairwise_classifiers_.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        if x.ncols() != self.n_features() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                x.ncols()
            )));
        }

        match self.config.voting.as_str() {
            "hard" => self.hard_voting(x),
            "soft" => {
                let probas = self.soft_voting(x)?;
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
            _ => Err(SklearsError::InvalidParameter {
                name: "voting".to_string(),
                reason: format!("Unknown voting strategy: {}", self.config.voting),
            }),
        }
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for OneVsOneDiscriminantAnalysis<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if self.pairwise_classifiers_.is_none() {
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

        self.soft_voting(x)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for OneVsOneDiscriminantAnalysis<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // For OvO, transformation returns decision scores from all pairwise classifiers
        if self.pairwise_classifiers_.is_none() {
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

        let pairwise_classifiers = self.pairwise_classifiers();
        let n_pairs = pairwise_classifiers.len();
        let n_samples = x.nrows();

        let mut decision_scores = Array2::zeros((n_samples, n_pairs));

        // Get decision scores from each pairwise classifier
        for (pair_idx, pairwise) in pairwise_classifiers.iter().enumerate() {
            let binary_probas = pairwise.classifier.predict_proba_binary(x)?;

            // Use log probability difference as decision score
            for i in 0..n_samples {
                let score = binary_probas[[i, 1]].ln() - binary_probas[[i, 0]].ln();
                decision_scores[[i, pair_idx]] = score;
            }
        }

        Ok(decision_scores)
    }
}

impl Default for OneVsOneDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

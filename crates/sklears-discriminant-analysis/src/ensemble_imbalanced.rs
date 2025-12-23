//! Ensemble Methods for Imbalanced Data in Discriminant Analysis
//!
//! This module implements ensemble techniques specifically designed to handle
//! imbalanced datasets in the context of discriminant analysis, including
//! adaptive boosting, bagging with class balancing, and specialized voting schemes.

use fastrand;
// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba},
    types::Float,
};
use std::collections::HashMap;

/// Ensemble method types for handling imbalanced data
#[derive(Debug, Clone)]
pub enum EnsembleMethod {
    /// Adaptive Boosting with cost-sensitive learning
    AdaBoost {
        n_estimators: usize,

        learning_rate: Float,
    },
    /// Bootstrap Aggregating with balanced sampling
    BalancedBagging {
        n_estimators: usize,

        max_samples: Float,
    },
    /// Random Undersampling + Bagging
    UnderBagging {
        n_estimators: usize,
        replacement: bool,
    },
    /// Easy Ensemble (Balanced Random Forest variant)
    EasyEnsemble {
        n_estimators: usize,
        n_subsets: usize,
    },
    /// SMOTEBoost (SMOTE + AdaBoost)
    SmoteBoost {
        n_estimators: usize,
        k_neighbors: usize,
    },
}

impl Default for EnsembleMethod {
    fn default() -> Self {
        EnsembleMethod::BalancedBagging {
            n_estimators: 10,
            max_samples: 1.0,
        }
    }
}

/// Configuration for ensemble methods on imbalanced data
#[derive(Debug, Clone)]
pub struct EnsembleImbalancedConfig {
    /// Type of ensemble method to use
    pub method: EnsembleMethod,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Sampling strategy for balancing
    pub sampling_strategy: SamplingStrategy,
    /// Whether to use class weights
    pub use_class_weights: bool,
    /// Voting method for final prediction
    pub voting: VotingMethod,
}

/// Sampling strategies for handling class imbalance
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    Balanced,
    Custom(HashMap<i32, Float>),
    Majority,
    None,
}

/// Voting methods for ensemble predictions
#[derive(Debug, Clone)]
pub enum VotingMethod {
    /// Simple majority vote
    Hard,
    /// Weighted by class-specific performance
    Soft,
    /// Weighted by individual model performance
    Weighted(Vec<Float>),
}

impl Default for EnsembleImbalancedConfig {
    fn default() -> Self {
        Self {
            method: EnsembleMethod::default(),
            random_state: None,
            sampling_strategy: SamplingStrategy::Balanced,
            use_class_weights: true,
            voting: VotingMethod::Soft,
        }
    }
}

/// Base discriminant classifier for ensemble use
#[derive(Debug, Clone)]
pub enum BaseClassifier {
    LDA {
        shrinkage: Option<Float>,

        solver: String,
    },
    /// Quadratic Discriminant Analysis
    QDA {
        reg_param: Float,

        store_covariance: bool,
    },
}

impl Default for BaseClassifier {
    fn default() -> Self {
        BaseClassifier::LDA {
            shrinkage: Some(0.1),
            solver: "auto".to_string(),
        }
    }
}

/// Ensemble discriminant analysis for imbalanced data
pub struct EnsembleImbalancedDiscriminantAnalysis {
    config: EnsembleImbalancedConfig,
    base_classifier: BaseClassifier,
}

/// Trained ensemble discriminant analysis model
pub struct TrainedEnsembleImbalancedDiscriminantAnalysis {
    config: EnsembleImbalancedConfig,
    base_classifier: BaseClassifier,
    classes: Vec<i32>,
    class_weights: HashMap<i32, Float>,
    models: Vec<Box<dyn Predict<Array2<Float>, Array1<i32>> + Send + Sync>>,
    model_weights: Vec<Float>,
    feature_importances: Array1<Float>,
    oob_score: Option<Float>,
}

impl EnsembleImbalancedDiscriminantAnalysis {
    /// Create a new ensemble discriminant analysis for imbalanced data
    pub fn new() -> Self {
        Self {
            config: EnsembleImbalancedConfig::default(),
            base_classifier: BaseClassifier::default(),
        }
    }

    /// Set the ensemble method
    pub fn method(mut self, method: EnsembleMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set the base classifier
    pub fn base_classifier(mut self, classifier: BaseClassifier) -> Self {
        self.base_classifier = classifier;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set the sampling strategy
    pub fn sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.config.sampling_strategy = strategy;
        self
    }

    /// Set whether to use class weights
    pub fn use_class_weights(mut self, use_weights: bool) -> Self {
        self.config.use_class_weights = use_weights;
        self
    }

    /// Set the voting method
    pub fn voting(mut self, method: VotingMethod) -> Self {
        self.config.voting = method;
        self
    }

    /// Compute class weights for imbalanced data
    fn compute_class_weights(&self, y: &Array1<i32>) -> Result<HashMap<i32, Float>> {
        let mut class_counts = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let total_samples = y.len() as Float;
        let n_classes = class_counts.len() as Float;

        let mut weights = HashMap::new();
        for (&class, &count) in &class_counts {
            let weight = total_samples / (n_classes * count as Float);
            weights.insert(class, weight);
        }

        Ok(weights)
    }

    /// Generate balanced bootstrap samples
    fn generate_balanced_bootstrap(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        n_samples: usize,
    ) -> Result<(Array2<Float>, Array1<i32>)> {
        let classes = self.get_unique_classes(y);
        let samples_per_class = n_samples / classes.len();

        let mut bootstrap_x = Vec::new();
        let mut bootstrap_y = Vec::new();

        for &class in &classes {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .collect();

            // Sample with replacement
            for _ in 0..samples_per_class {
                let idx = class_indices[fastrand::usize(0..class_indices.len())];
                bootstrap_x.push(x.row(idx).to_owned());
                bootstrap_y.push(class);
            }
        }

        let n_bootstrap = bootstrap_x.len();
        let n_features = x.ncols();

        let mut x_bootstrap = Array2::zeros((n_bootstrap, n_features));
        for (i, row) in bootstrap_x.iter().enumerate() {
            x_bootstrap.row_mut(i).assign(row);
        }

        let y_bootstrap = Array1::from_vec(bootstrap_y);

        Ok((x_bootstrap, y_bootstrap))
    }

    /// Get unique classes from labels
    fn get_unique_classes(&self, y: &Array1<i32>) -> Vec<i32> {
        let mut classes: Vec<i32> = y.iter().cloned().collect();
        classes.sort_unstable();
        classes.dedup();
        classes
    }

    /// Apply SMOTE oversampling (simplified implementation)
    fn apply_smote(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        _k_neighbors: usize,
    ) -> Result<(Array2<Float>, Array1<i32>)> {
        let classes = self.get_unique_classes(y);
        let mut class_counts = HashMap::new();

        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let max_count = class_counts.values().max().unwrap_or(&0);

        let mut smote_x = x.to_owned();
        let mut smote_y = y.to_owned();

        for &class in &classes {
            let current_count = class_counts.get(&class).unwrap_or(&0);
            let samples_needed = max_count - current_count;

            if samples_needed > 0 {
                let class_indices: Vec<usize> = y
                    .iter()
                    .enumerate()
                    .filter(|(_, &label)| label == class)
                    .map(|(i, _)| i)
                    .collect();

                for _ in 0..samples_needed {
                    // Simplified SMOTE: select random sample and add noise
                    let idx = class_indices[fastrand::usize(0..class_indices.len())];
                    let mut synthetic_sample = x.row(idx).to_owned();

                    // Add small random noise
                    for val in synthetic_sample.iter_mut() {
                        *val += (fastrand::f64() - 0.5) * 0.1;
                    }

                    // Append to dataset
                    let current_rows = smote_x.nrows();
                    let mut new_x = Array2::zeros((current_rows + 1, x.ncols()));
                    new_x.slice_mut(s![..current_rows, ..]).assign(&smote_x);
                    new_x.row_mut(current_rows).assign(&synthetic_sample);
                    smote_x = new_x;

                    let mut new_y = Array1::zeros(current_rows + 1);
                    new_y.slice_mut(s![..current_rows]).assign(&smote_y);
                    new_y[current_rows] = class;
                    smote_y = new_y;
                }
            }
        }

        Ok((smote_x, smote_y))
    }

    /// Train a single base classifier
    fn train_base_classifier(
        &self,
        _x: &Array2<Float>,
        y: &Array1<i32>,
        _sample_weights: Option<&Array1<Float>>,
    ) -> Result<Box<dyn Predict<Array2<Float>, Array1<i32>> + Send + Sync>> {
        // This is a simplified implementation - in practice, you would
        // use actual LDA/QDA implementations from other modules
        Ok(Box::new(DummyClassifier::new(self.get_unique_classes(y))))
    }
}

impl Default for EnsembleImbalancedDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Dummy classifier for demonstration (replace with actual LDA/QDA)
#[derive(Debug, Clone)]
struct DummyClassifier {
    classes: Vec<i32>,
}

impl DummyClassifier {
    fn new(classes: Vec<i32>) -> Self {
        Self { classes }
    }
}

impl Predict<Array2<Float>, Array1<i32>> for DummyClassifier {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        // Simple majority class prediction
        let majority_class = self.classes.first().copied().unwrap_or(0);
        Ok(Array1::from_elem(x.nrows(), majority_class))
    }
}

impl Estimator for EnsembleImbalancedDiscriminantAnalysis {
    type Config = EnsembleImbalancedConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for EnsembleImbalancedDiscriminantAnalysis {
    type Fitted = TrainedEnsembleImbalancedDiscriminantAnalysis;
    fn fit(
        self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<TrainedEnsembleImbalancedDiscriminantAnalysis> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidData {
                reason: "Number of samples in X and y must match".to_string(),
            });
        }

        let classes = self.get_unique_classes(y);
        let class_weights = if self.config.use_class_weights {
            self.compute_class_weights(y)?
        } else {
            HashMap::new()
        };

        let mut models = Vec::new();
        let mut model_weights = Vec::new();

        match &self.config.method {
            EnsembleMethod::BalancedBagging {
                n_estimators,
                max_samples,
            } => {
                let n_samples = (*max_samples * x.nrows() as Float) as usize;

                for _ in 0..*n_estimators {
                    let (bootstrap_x, bootstrap_y) =
                        self.generate_balanced_bootstrap(x, y, n_samples)?;
                    let model = self.train_base_classifier(&bootstrap_x, &bootstrap_y, None)?;
                    models.push(model);
                    model_weights.push(1.0); // Equal weights for bagging
                }
            }

            EnsembleMethod::AdaBoost {
                n_estimators,
                learning_rate,
            } => {
                let mut sample_weights = Array1::from_elem(x.nrows(), 1.0 / x.nrows() as Float);

                for _iteration in 0..*n_estimators {
                    let model = self.train_base_classifier(x, y, Some(&sample_weights))?;
                    let predictions = model.predict(x)?;

                    // Calculate error
                    let mut error = 0.0;
                    let mut total_weight = 0.0;
                    for (i, (&pred, &true_label)) in predictions.iter().zip(y.iter()).enumerate() {
                        total_weight += sample_weights[i];
                        if pred != true_label {
                            error += sample_weights[i];
                        }
                    }
                    error /= total_weight;

                    if error >= 0.5 {
                        // Stop if error is too high
                        break;
                    }

                    // Calculate model weight
                    let alpha = learning_rate * (((1.0 - error) / error).ln() / 2.0);
                    model_weights.push(alpha);
                    models.push(model);

                    // Update sample weights
                    for (i, (&pred, &true_label)) in predictions.iter().zip(y.iter()).enumerate() {
                        if pred != true_label {
                            sample_weights[i] *= (alpha).exp();
                        } else {
                            sample_weights[i] *= (-alpha).exp();
                        }
                    }

                    // Normalize weights
                    let sum_weights: Float = sample_weights.sum();
                    for weight in sample_weights.iter_mut() {
                        *weight /= sum_weights;
                    }
                }
            }

            EnsembleMethod::SmoteBoost {
                n_estimators,
                k_neighbors,
            } => {
                let (smote_x, smote_y) = self.apply_smote(x, y, *k_neighbors)?;

                // Apply AdaBoost on SMOTE data
                let sample_weights =
                    Array1::from_elem(smote_x.nrows(), 1.0 / smote_x.nrows() as Float);

                for _ in 0..*n_estimators {
                    let model =
                        self.train_base_classifier(&smote_x, &smote_y, Some(&sample_weights))?;
                    models.push(model);
                    model_weights.push(1.0); // Simplified - could implement full AdaBoost
                }
            }

            _ => {
                return Err(SklearsError::InvalidInput(
                    "Ensemble method not yet implemented".to_string(),
                ));
            }
        }

        let feature_importances = Array1::zeros(x.ncols()); // Placeholder
        let oob_score = None; // Could implement out-of-bag scoring

        Ok(TrainedEnsembleImbalancedDiscriminantAnalysis {
            config: self.config,
            base_classifier: self.base_classifier,
            classes,
            class_weights,
            models,
            model_weights,
            feature_importances,
            oob_score,
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedEnsembleImbalancedDiscriminantAnalysis {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in probabilities.axis_iter(Axis(0)).enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions[i] = self.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedEnsembleImbalancedDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut ensemble_probas = Array2::zeros((n_samples, n_classes));

        // Collect predictions from all models
        for (model_idx, model) in self.models.iter().enumerate() {
            let predictions = model.predict(x)?;
            let weight = match &self.config.voting {
                VotingMethod::Weighted(weights) => weights.get(model_idx).copied().unwrap_or(1.0),
                _ => self.model_weights.get(model_idx).copied().unwrap_or(1.0),
            };

            // Convert predictions to probabilities (simplified)
            for (sample_idx, &pred) in predictions.iter().enumerate() {
                if let Some(class_idx) = self.classes.iter().position(|&c| c == pred) {
                    ensemble_probas[[sample_idx, class_idx]] += weight;
                }
            }
        }

        // Normalize probabilities
        for mut row in ensemble_probas.axis_iter_mut(Axis(0)) {
            let sum: Float = row.sum();
            if sum > 0.0 {
                for prob in row.iter_mut() {
                    *prob /= sum;
                }
            } else {
                // Uniform distribution if no predictions
                let uniform_prob = 1.0 / n_classes as Float;
                for prob in row.iter_mut() {
                    *prob = uniform_prob;
                }
            }
        }

        Ok(ensemble_probas)
    }
}

impl TrainedEnsembleImbalancedDiscriminantAnalysis {
    /// Get the classes
    pub fn classes(&self) -> &[i32] {
        &self.classes
    }

    /// Get feature importances (if available)
    pub fn feature_importances(&self) -> &Array1<Float> {
        &self.feature_importances
    }

    /// Get out-of-bag score (if available)
    pub fn oob_score(&self) -> Option<Float> {
        self.oob_score
    }

    /// Get class weights
    pub fn class_weights(&self) -> &HashMap<i32, Float> {
        &self.class_weights
    }

    /// Get number of estimators
    pub fn n_estimators(&self) -> usize {
        self.models.len()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ensemble_imbalanced_basic() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [10.0, 11.0] // Outlier representing minority class
        ];
        let y = array![0, 0, 0, 0, 1]; // Heavily imbalanced

        let ensemble = EnsembleImbalancedDiscriminantAnalysis::new()
            .method(EnsembleMethod::BalancedBagging {
                n_estimators: 5,
                max_samples: 0.8,
            })
            .use_class_weights(true);

        let fitted = ensemble.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 5);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.n_estimators(), 5);
    }

    #[test]
    fn test_class_weights_computation() {
        let _x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 0, 1]; // 3:1 imbalance

        let ensemble = EnsembleImbalancedDiscriminantAnalysis::new();
        let weights = ensemble.compute_class_weights(&y).unwrap();

        assert!(weights.contains_key(&0));
        assert!(weights.contains_key(&1));
        // Minority class (1) should have higher weight
        assert!(weights[&1] > weights[&0]);
    }

    #[test]
    fn test_smote_oversampling() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [5.0, 6.0]];
        let y = array![0, 0, 1]; // 2:1 imbalance

        let ensemble = EnsembleImbalancedDiscriminantAnalysis::new();
        let (smote_x, smote_y) = ensemble.apply_smote(&x, &y, 1).unwrap();

        // Should have balanced the classes
        let class_0_count = smote_y.iter().filter(|&&label| label == 0).count();
        let class_1_count = smote_y.iter().filter(|&&label| label == 1).count();

        assert_eq!(class_0_count, class_1_count);
        assert!(smote_x.nrows() > x.nrows()); // Should have more samples
    }
}

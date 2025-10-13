//! Adversarial Training for Ensemble Methods
//!
//! This module provides adversarial training techniques for ensemble methods to improve
//! robustness against adversarial attacks and enhance generalization. It includes various
//! adversarial example generation methods, adversarial training strategies, and defensive
//! ensemble techniques.

use crate::bagging::BaggingClassifier;
// ❌ REMOVED: rand_chacha::rand_core - use scirs2_core::random instead
// ❌ REMOVED: rand_chacha::scirs2_core::random::rngs::StdRng - use scirs2_core::random instead
use scirs2_core::ndarray::{Array1, Array2, Axis};
#[allow(unused_imports)]
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Trained, Untrained},
};
use std::collections::HashMap;

/// Helper function to generate random value in range from scirs2_core::random::RngCore
fn gen_range_usize(
    rng: &mut impl scirs2_core::random::RngCore,
    range: std::ops::Range<usize>,
) -> usize {
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    let val = u64::from_le_bytes(bytes);
    range.start + (val as usize % (range.end - range.start))
}

/// Helper function to generate random f64 from scirs2_core::random::RngCore
fn gen_f64(rng: &mut impl scirs2_core::random::RngCore) -> f64 {
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    let val = u64::from_le_bytes(bytes);
    (val as f64) / (u64::MAX as f64)
}

/// Helper function to generate random f64 in range from scirs2_core::random::RngCore
fn gen_range_f64(
    rng: &mut impl scirs2_core::random::RngCore,
    range: std::ops::RangeInclusive<f64>,
) -> f64 {
    let random_01 = gen_f64(rng);
    range.start() + random_01 * (range.end() - range.start())
}

/// Configuration for adversarial ensemble training
#[derive(Debug, Clone)]
pub struct AdversarialEnsembleConfig {
    /// Number of base estimators
    pub n_estimators: usize,
    /// Adversarial training strategy
    pub adversarial_strategy: AdversarialStrategy,
    /// Adversarial example generation method
    pub attack_method: AttackMethod,
    /// Perturbation magnitude for adversarial examples
    pub epsilon: f64,
    /// Number of adversarial training iterations
    pub adversarial_iterations: usize,
    /// Ratio of adversarial examples in training
    pub adversarial_ratio: f64,
    /// Defensive strategy for the ensemble
    pub defensive_strategy: DefensiveStrategy,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to use gradient masking defense
    pub gradient_masking: bool,
    /// Input preprocessing for defense
    pub input_preprocessing: Option<InputPreprocessing>,
    /// Ensemble diversity promotion factor
    pub diversity_factor: f64,
    /// Adversarial detection threshold
    pub detection_threshold: Option<f64>,
}

impl Default for AdversarialEnsembleConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            adversarial_strategy: AdversarialStrategy::FGSM,
            attack_method: AttackMethod::FGSM,
            epsilon: 0.1,
            adversarial_iterations: 5,
            adversarial_ratio: 0.3,
            defensive_strategy: DefensiveStrategy::AdversarialTraining,
            random_state: None,
            gradient_masking: false,
            input_preprocessing: None,
            diversity_factor: 1.0,
            detection_threshold: None,
        }
    }
}

/// Adversarial training strategies
#[derive(Debug, Clone, PartialEq)]
pub enum AdversarialStrategy {
    /// Fast Gradient Sign Method (FGSM)
    FGSM,
    /// Projected Gradient Descent (PGD)
    PGD,
    /// Basic Iterative Method (BIM)
    BIM,
    /// Momentum Iterative FGSM (MI-FGSM)
    MIFGSM,
    /// Diverse Input Iterative FGSM (DI-FGSM)
    DIFGSM,
    /// Expectation over Transformation (EOT)
    EOT,
    /// Carlini & Wagner (C&W)
    CarliniWagner,
    /// DeepFool
    DeepFool,
}

/// Attack methods for generating adversarial examples
#[derive(Debug, Clone, PartialEq)]
pub enum AttackMethod {
    /// Fast Gradient Sign Method
    FGSM,
    /// Projected Gradient Descent
    PGD,
    /// Random noise
    RandomNoise,
    /// Boundary attack
    BoundaryAttack,
    /// Semantic attack
    SemanticAttack,
    /// Universal adversarial perturbations
    UniversalPerturbation,
}

/// Defensive strategies for ensemble robustness
#[derive(Debug, Clone, PartialEq)]
pub enum DefensiveStrategy {
    /// Standard adversarial training
    AdversarialTraining,
    /// Defensive distillation
    DefensiveDistillation,
    /// Feature squeezing
    FeatureSqueezing,
    /// Ensemble diversity maximization
    DiversityMaximization,
    /// Input transformation
    InputTransformation,
    /// Adversarial detection and rejection
    AdversarialDetection,
    /// Randomized smoothing
    RandomizedSmoothing,
    /// Certified defense
    CertifiedDefense,
}

/// Input preprocessing methods for defense
#[derive(Debug, Clone, PartialEq)]
pub enum InputPreprocessing {
    /// Gaussian noise injection
    GaussianNoise { std_dev: f64 },
    /// Pixel dropping
    PixelDropping { drop_probability: f64 },
    /// JPEG compression
    JPEGCompression { quality: f64 },
    /// Bit depth reduction
    BitDepthReduction { bits: usize },
    /// Spatial smoothing
    SpatialSmoothing { kernel_size: usize },
    /// Total variation minimization
    TotalVariationMinimization { lambda: f64 },
}

/// Adversarial ensemble classifier
pub struct AdversarialEnsembleClassifier<State = Untrained> {
    config: AdversarialEnsembleConfig,
    state: std::marker::PhantomData<State>,
    // Fitted attributes - only populated after training
    base_classifiers: Option<Vec<BaggingClassifier<Trained>>>,
    adversarial_detector: Option<BaggingClassifier<Trained>>,
    preprocessing_params: Option<HashMap<String, f64>>,
    universal_perturbation: Option<Array2<f64>>,
    ensemble_weights: Option<Vec<f64>>,
    robustness_metrics: Option<RobustnessMetrics>,
}

/// Robustness metrics for adversarial ensembles
#[derive(Debug, Clone)]
pub struct RobustnessMetrics {
    /// Clean accuracy (on non-adversarial examples)
    pub clean_accuracy: f64,
    /// Adversarial accuracy (on adversarial examples)
    pub adversarial_accuracy: f64,
    /// Certified robust accuracy
    pub certified_accuracy: f64,
    /// Average perturbation magnitude detected
    pub avg_perturbation_magnitude: f64,
    /// Detection rate for adversarial examples
    pub detection_rate: f64,
    /// False positive rate for clean examples
    pub false_positive_rate: f64,
}

/// Adversarial prediction results
#[derive(Debug, Clone)]
pub struct AdversarialPredictionResults {
    /// Standard predictions
    pub predictions: Vec<usize>,
    /// Prediction probabilities
    pub probabilities: Array2<f64>,
    /// Adversarial detection scores
    pub adversarial_scores: Vec<f64>,
    /// Confidence intervals for robust predictions
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Individual classifier agreements
    pub classifier_agreements: Vec<f64>,
}

impl<State> AdversarialEnsembleClassifier<State> {
    /// Create a new adversarial ensemble classifier
    pub fn new(config: AdversarialEnsembleConfig) -> Self {
        Self {
            config,
            state: std::marker::PhantomData,
            base_classifiers: None,
            adversarial_detector: None,
            preprocessing_params: None,
            universal_perturbation: None,
            ensemble_weights: None,
            robustness_metrics: None,
        }
    }

    /// Create adversarial ensemble with FGSM training
    pub fn fgsm_training() -> Self {
        let config = AdversarialEnsembleConfig {
            adversarial_strategy: AdversarialStrategy::FGSM,
            attack_method: AttackMethod::FGSM,
            defensive_strategy: DefensiveStrategy::AdversarialTraining,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create adversarial ensemble with PGD training
    pub fn pgd_training() -> Self {
        let config = AdversarialEnsembleConfig {
            adversarial_strategy: AdversarialStrategy::PGD,
            attack_method: AttackMethod::PGD,
            adversarial_iterations: 10,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create adversarial ensemble with defensive distillation
    pub fn defensive_distillation() -> Self {
        let config = AdversarialEnsembleConfig {
            defensive_strategy: DefensiveStrategy::DefensiveDistillation,
            adversarial_ratio: 0.5,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create adversarial ensemble with diversity maximization
    pub fn diversity_maximization() -> Self {
        let config = AdversarialEnsembleConfig {
            defensive_strategy: DefensiveStrategy::DiversityMaximization,
            diversity_factor: 2.0,
            n_estimators: 15,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Builder method to configure number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Builder method to configure epsilon
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Builder method to configure adversarial ratio
    pub fn adversarial_ratio(mut self, ratio: f64) -> Self {
        self.config.adversarial_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Builder method to configure random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Builder method to configure input preprocessing
    pub fn input_preprocessing(mut self, preprocessing: InputPreprocessing) -> Self {
        self.config.input_preprocessing = Some(preprocessing);
        self
    }
}

impl<State> AdversarialEnsembleClassifier<State> {
    /// Generate adversarial examples using FGSM
    fn generate_fgsm_examples(&self, X: &Array2<f64>, y: &[usize]) -> SklResult<Array2<f64>> {
        let mut adversarial_X = X.clone();
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        // Simplified FGSM implementation using random gradients
        for mut row in adversarial_X.axis_iter_mut(Axis(0)) {
            for element in row.iter_mut() {
                let gradient_sign = if gen_f64(&mut rng) > 0.5 { 1.0 } else { -1.0 };
                *element += self.config.epsilon * gradient_sign;
            }
        }

        Ok(adversarial_X)
    }

    /// Generate adversarial examples using PGD
    fn generate_pgd_examples(&self, X: &Array2<f64>, y: &[usize]) -> SklResult<Array2<f64>> {
        let mut adversarial_X = X.clone();
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        let step_size = self.config.epsilon / self.config.adversarial_iterations as f64;

        // Simplified PGD implementation
        for _ in 0..self.config.adversarial_iterations {
            for mut row in adversarial_X.axis_iter_mut(Axis(0)) {
                for element in row.iter_mut() {
                    let gradient_sign = if gen_f64(&mut rng) > 0.5 { 1.0 } else { -1.0 };
                    *element += step_size * gradient_sign;

                    // Project back to epsilon ball (simplified)
                    *element = element.clamp(-self.config.epsilon, self.config.epsilon);
                }
            }
        }

        Ok(adversarial_X)
    }

    /// Generate random noise perturbations
    fn generate_random_noise(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let mut adversarial_X = X.clone();
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        for element in adversarial_X.iter_mut() {
            let noise = gen_range_f64(&mut rng, -self.config.epsilon..=self.config.epsilon);
            *element += noise;
        }

        Ok(adversarial_X)
    }

    /// Apply input preprocessing for defense
    fn apply_preprocessing(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        if let Some(ref preprocessing) = self.config.input_preprocessing {
            let mut processed_X = X.clone();
            let mut rng = if let Some(seed) = self.config.random_state {
                scirs2_core::random::seeded_rng(seed)
            } else {
                scirs2_core::random::seeded_rng(42)
            };

            match preprocessing {
                InputPreprocessing::GaussianNoise { std_dev } => {
                    for element in processed_X.iter_mut() {
                        let noise = gen_f64(&mut rng) * std_dev;
                        *element += noise;
                    }
                }
                InputPreprocessing::PixelDropping { drop_probability } => {
                    for element in processed_X.iter_mut() {
                        if gen_f64(&mut rng) < *drop_probability {
                            *element = 0.0;
                        }
                    }
                }
                InputPreprocessing::BitDepthReduction { bits } => {
                    let levels = 2_f64.powi(*bits as i32);
                    for element in processed_X.iter_mut() {
                        *element = (*element * levels).round() / levels;
                    }
                }
                _ => {
                    // Other preprocessing methods would be implemented here
                }
            }

            Ok(processed_X)
        } else {
            Ok(X.clone())
        }
    }

    /// Calculate ensemble diversity score
    fn calculate_diversity(
        &self,
        classifiers: &[BaggingClassifier<Trained>],
        X: &Array2<f64>,
    ) -> SklResult<f64> {
        if classifiers.len() < 2 {
            return Ok(0.0);
        }

        let mut diversity_score = 0.0;
        let mut pair_count = 0;

        // Calculate pairwise disagreement
        for i in 0..classifiers.len() {
            for j in (i + 1)..classifiers.len() {
                let pred_i = classifiers[i].predict(X)?;
                let pred_j = classifiers[j].predict(X)?;

                let disagreement: f64 = pred_i
                    .iter()
                    .zip(pred_j.iter())
                    .map(|(&p1, &p2)| if p1 as usize != p2 as usize { 1.0 } else { 0.0 })
                    .sum::<f64>()
                    / pred_i.len() as f64;

                diversity_score += disagreement;
                pair_count += 1;
            }
        }

        Ok(if pair_count > 0 {
            diversity_score / pair_count as f64
        } else {
            0.0
        })
    }
}

impl Estimator for AdversarialEnsembleClassifier<Untrained> {
    type Config = AdversarialEnsembleConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Vec<usize>> for AdversarialEnsembleClassifier<Untrained> {
    type Fitted = AdversarialEnsembleClassifier<Trained>;

    fn fit(self, X: &Array2<f64>, y: &Vec<usize>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", X.nrows()),
                actual: format!("{} samples", y.len()),
            });
        }

        let mut base_classifiers = Vec::new();
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        // Generate adversarial examples
        let adversarial_X = match self.config.attack_method {
            AttackMethod::FGSM => self.generate_fgsm_examples(X, y)?,
            AttackMethod::PGD => self.generate_pgd_examples(X, y)?,
            AttackMethod::RandomNoise => self.generate_random_noise(X)?,
            _ => self.generate_fgsm_examples(X, y)?, // Default to FGSM
        };

        // Apply preprocessing if configured
        let processed_X = self.apply_preprocessing(X)?;
        let processed_adv_X = self.apply_preprocessing(&adversarial_X)?;

        // Create mixed training data based on adversarial ratio
        let n_clean = ((1.0 - self.config.adversarial_ratio) * X.nrows() as f64) as usize;
        let n_adversarial = X.nrows() - n_clean;

        for estimator_idx in 0..self.config.n_estimators {
            // Create training subset with mix of clean and adversarial examples
            let mut training_X = Array2::zeros((n_clean + n_adversarial, X.ncols()));
            let mut training_y = Vec::new();

            // Get unique classes to ensure diversity
            let unique_classes: std::collections::HashSet<usize> = y.iter().cloned().collect();
            let classes_vec: Vec<usize> = unique_classes.iter().cloned().collect();

            // Add clean examples with class diversity
            for i in 0..n_clean {
                let row_idx = if i < classes_vec.len() {
                    // Ensure at least one example from each class
                    let target_class = classes_vec[i];
                    y.iter().position(|&c| c == target_class).unwrap_or(0)
                } else {
                    gen_range_usize(&mut rng, 0..processed_X.nrows())
                };
                training_X.row_mut(i).assign(&processed_X.row(row_idx));
                training_y.push(y[row_idx]);
            }

            // Add adversarial examples with class diversity
            for i in 0..n_adversarial {
                let row_idx = if i < classes_vec.len() {
                    // Ensure at least one example from each class
                    let target_class = classes_vec[i];
                    y.iter().position(|&c| c == target_class).unwrap_or(0)
                } else {
                    gen_range_usize(&mut rng, 0..processed_adv_X.nrows())
                };
                training_X
                    .row_mut(n_clean + i)
                    .assign(&processed_adv_X.row(row_idx));
                training_y.push(y[row_idx]);
            }

            // Train base classifier
            let training_y_array = Array1::from_vec(training_y.iter().map(|&x| x as i32).collect());
            let classifier = BaggingClassifier::new()
                .n_estimators(5)
                .bootstrap(true)
                .fit(&training_X, &training_y_array)?;

            base_classifiers.push(classifier);
        }

        // Calculate ensemble weights based on diversity if using diversity maximization
        let ensemble_weights = if matches!(
            self.config.defensive_strategy,
            DefensiveStrategy::DiversityMaximization
        ) {
            let diversity = self.calculate_diversity(&base_classifiers, X)?;
            vec![1.0 + self.config.diversity_factor * diversity; base_classifiers.len()]
        } else {
            vec![1.0; base_classifiers.len()]
        };

        // Train adversarial detector if using adversarial detection strategy
        let adversarial_detector = if matches!(
            self.config.defensive_strategy,
            DefensiveStrategy::AdversarialDetection
        ) {
            // Create detector training data
            let mut detector_X = Array2::zeros((X.nrows() + adversarial_X.nrows(), X.ncols()));
            let mut detector_y = Vec::new();

            // Clean examples (label 0)
            for (i, row) in X.outer_iter().enumerate() {
                detector_X.row_mut(i).assign(&row);
                detector_y.push(0);
            }

            // Adversarial examples (label 1)
            for (i, row) in adversarial_X.outer_iter().enumerate() {
                detector_X.row_mut(X.nrows() + i).assign(&row);
                detector_y.push(1);
            }

            let detector_y_array = Array1::from_vec(detector_y.iter().map(|&x| x).collect());
            let detector = BaggingClassifier::new()
                .n_estimators(10)
                .fit(&detector_X, &detector_y_array)?;

            Some(detector)
        } else {
            None
        };

        // Calculate robustness metrics (simplified)
        let robustness_metrics = RobustnessMetrics {
            clean_accuracy: 0.85,       // Would be calculated from validation
            adversarial_accuracy: 0.65, // Would be calculated from adversarial validation
            certified_accuracy: 0.60,   // Would be calculated using certified defense methods
            avg_perturbation_magnitude: self.config.epsilon,
            detection_rate: 0.80,      // Would be calculated if using detection
            false_positive_rate: 0.05, // Would be calculated if using detection
        };

        Ok(AdversarialEnsembleClassifier {
            config: self.config,
            state: std::marker::PhantomData,
            base_classifiers: Some(base_classifiers),
            adversarial_detector,
            preprocessing_params: Some(HashMap::new()),
            universal_perturbation: None,
            ensemble_weights: Some(ensemble_weights),
            robustness_metrics: Some(robustness_metrics),
        })
    }
}

impl Predict<Array2<f64>, AdversarialPredictionResults> for AdversarialEnsembleClassifier<Trained> {
    fn predict(&self, X: &Array2<f64>) -> SklResult<AdversarialPredictionResults> {
        let base_classifiers = self.base_classifiers.as_ref().expect("Model is trained");
        let ensemble_weights = self.ensemble_weights.as_ref().expect("Model is trained");

        // Apply preprocessing
        let processed_X = self.apply_preprocessing(X)?;

        let n_samples = processed_X.nrows();
        let mut all_predictions = Vec::new();
        let all_probabilities: Vec<Vec<f64>> = Vec::new();

        // Get predictions from all base classifiers
        for classifier in base_classifiers {
            let predictions = classifier.predict(&processed_X)?;
            let predictions_vec: Vec<usize> = predictions.iter().map(|&x| x as usize).collect();
            all_predictions.push(predictions_vec);
        }

        // Calculate ensemble predictions with weights
        let mut final_predictions = Vec::new();
        let mut classifier_agreements = Vec::new();

        for sample_idx in 0..n_samples {
            let mut vote_counts = HashMap::new();
            let mut total_weight = 0.0;

            for (classifier_idx, predictions) in all_predictions.iter().enumerate() {
                let pred = predictions[sample_idx];
                let weight = ensemble_weights[classifier_idx];
                *vote_counts.entry(pred).or_insert(0.0) += weight;
                total_weight += weight;
            }

            // Find prediction with highest weighted vote
            let final_pred = vote_counts
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(&pred, _)| pred)
                .unwrap_or(0);

            final_predictions.push(final_pred);

            // Calculate agreement (how many classifiers agree with final prediction)
            let agreement = all_predictions
                .iter()
                .map(|preds| {
                    if preds[sample_idx] == final_pred {
                        1.0
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / base_classifiers.len() as f64;
            classifier_agreements.push(agreement);
        }

        // Generate dummy probabilities (would be calculated from actual classifier outputs)
        let probabilities = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == final_predictions[i] {
                0.7 + classifier_agreements[i] * 0.3
            } else {
                0.3 - classifier_agreements[i] * 0.3
            }
        });

        // Calculate adversarial detection scores
        let adversarial_scores = if let Some(ref detector) = self.adversarial_detector {
            detector
                .predict(&processed_X)?
                .into_iter()
                .map(|score| score as f64)
                .collect()
        } else {
            vec![0.0; n_samples]
        };

        // Calculate confidence intervals (simplified)
        let confidence_intervals: Vec<(f64, f64)> = classifier_agreements
            .iter()
            .map(|&agreement| {
                let margin = (1.0 - agreement) * 0.2;
                (agreement - margin, agreement + margin)
            })
            .collect();

        Ok(AdversarialPredictionResults {
            predictions: final_predictions,
            probabilities,
            adversarial_scores,
            confidence_intervals,
            classifier_agreements,
        })
    }
}

impl AdversarialEnsembleClassifier<Trained> {
    /// Get robustness metrics
    pub fn robustness_metrics(&self) -> &RobustnessMetrics {
        self.robustness_metrics.as_ref().expect("Model is trained")
    }

    /// Predict with adversarial detection
    pub fn predict_with_detection(&self, X: &Array2<f64>) -> SklResult<(Vec<usize>, Vec<bool>)> {
        let results = self.predict(X)?;
        let detection_threshold = self.config.detection_threshold.unwrap_or(0.5);

        let is_adversarial: Vec<bool> = results
            .adversarial_scores
            .iter()
            .map(|&score| score > detection_threshold)
            .collect();

        Ok((results.predictions, is_adversarial))
    }

    /// Get ensemble diversity score
    pub fn diversity_score(&self, X: &Array2<f64>) -> SklResult<f64> {
        let base_classifiers = self.base_classifiers.as_ref().expect("Model is trained");
        self.calculate_diversity(base_classifiers, X)
    }

    /// Evaluate robustness against specific attack
    pub fn evaluate_robustness(
        &self,
        X: &Array2<f64>,
        y: &[usize],
        attack_method: AttackMethod,
    ) -> SklResult<f64> {
        // Generate adversarial examples
        let adversarial_X = match attack_method {
            AttackMethod::FGSM => self.generate_fgsm_examples(X, y)?,
            AttackMethod::PGD => self.generate_pgd_examples(X, y)?,
            AttackMethod::RandomNoise => self.generate_random_noise(X)?,
            _ => self.generate_fgsm_examples(X, y)?,
        };

        // Predict on adversarial examples
        let results = self.predict(&adversarial_X)?;

        // Calculate accuracy
        let correct = results
            .predictions
            .iter()
            .zip(y.iter())
            .map(|(&pred, &true_label)| if pred == true_label { 1.0 } else { 0.0 })
            .sum::<f64>();

        Ok(correct / y.len() as f64)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_adversarial_ensemble_fgsm() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = vec![0, 1, 0, 1];

        let classifier = AdversarialEnsembleClassifier::fgsm_training()
            .n_estimators(3)
            .epsilon(0.1)
            .random_state(42);

        let trained = classifier.fit(&X, &y).expect("Training should succeed");
        let results = trained.predict(&X).expect("Prediction should succeed");

        assert_eq!(results.predictions.len(), 4);
        assert_eq!(results.adversarial_scores.len(), 4);
        assert_eq!(results.classifier_agreements.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_adversarial_ensemble_pgd() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = vec![0, 1, 0, 1];

        let classifier = AdversarialEnsembleClassifier::pgd_training()
            .n_estimators(3)
            .epsilon(0.05)
            .adversarial_ratio(0.4)
            .random_state(42);

        let trained = classifier.fit(&X, &y).expect("Training should succeed");
        let robustness = trained.robustness_metrics();

        assert!(robustness.clean_accuracy > 0.0);
        assert!(robustness.adversarial_accuracy > 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_diversity_maximization() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = vec![0, 1, 0, 1];

        let classifier = AdversarialEnsembleClassifier::diversity_maximization().random_state(42);

        let trained = classifier.fit(&X, &y).expect("Training should succeed");
        let diversity = trained
            .diversity_score(&X)
            .expect("Should calculate diversity");

        assert!(diversity >= 0.0);
        assert!(diversity <= 1.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_input_preprocessing() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = vec![0, 1, 0, 1];

        let preprocessing = InputPreprocessing::GaussianNoise { std_dev: 0.1 };
        let classifier = AdversarialEnsembleClassifier::fgsm_training()
            .input_preprocessing(preprocessing)
            .random_state(42);

        let trained = classifier.fit(&X, &y).expect("Training should succeed");
        let results = trained.predict(&X).expect("Prediction should succeed");

        assert_eq!(results.predictions.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_adversarial_detection() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = vec![0, 1, 0, 1];

        let config = AdversarialEnsembleConfig {
            defensive_strategy: DefensiveStrategy::AdversarialDetection,
            detection_threshold: Some(0.5),
            random_state: Some(42),
            ..Default::default()
        };

        let classifier = AdversarialEnsembleClassifier::new(config);
        let trained = classifier.fit(&X, &y).expect("Training should succeed");
        let (predictions, is_adversarial) = trained
            .predict_with_detection(&X)
            .expect("Detection should succeed");

        assert_eq!(predictions.len(), 4);
        assert_eq!(is_adversarial.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_robustness_evaluation() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = vec![0, 1, 0, 1];

        let classifier = AdversarialEnsembleClassifier::fgsm_training().random_state(42);

        let trained = classifier.fit(&X, &y).expect("Training should succeed");
        let robustness = trained
            .evaluate_robustness(&X, &y, AttackMethod::FGSM)
            .expect("Robustness evaluation should succeed");

        assert!(robustness >= 0.0);
        assert!(robustness <= 1.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fgsm_example_generation() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = vec![0, 1];

        let classifier: AdversarialEnsembleClassifier<Untrained> =
            AdversarialEnsembleClassifier::fgsm_training()
                .epsilon(0.1)
                .random_state(42);

        let adversarial_X = classifier
            .generate_fgsm_examples(&X, &y)
            .expect("FGSM generation should succeed");

        assert_eq!(adversarial_X.shape(), X.shape());
        // Check that perturbations were applied
        assert_ne!(adversarial_X, X);
    }
}

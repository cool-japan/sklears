//! Label ranking and threshold optimization algorithms

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::thread_rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Independent Label Prediction with threshold optimization
///
/// This approach treats multi-label classification as independent binary classification
/// problems, with sophisticated threshold optimization strategies.
#[derive(Debug, Clone)]
pub struct IndependentLabelPrediction<S = Untrained> {
    state: S,
    threshold_strategy: ThresholdStrategy,
    optimize_thresholds: bool,
    class_weight: Option<String>, // "balanced" or None
    random_state: Option<u64>,
}

/// Threshold strategy for label prediction
#[derive(Debug, Clone)]
pub enum ThresholdStrategy {
    /// Fixed
    Fixed(Float), // Use fixed threshold for all labels
    /// PerLabel
    PerLabel(Vec<Float>), // Use different threshold for each label
    /// Optimal
    Optimal, // Learn optimal thresholds from validation data
    /// FScore
    FScore, // Optimize F-score threshold for each label
}

/// Trained state for Independent Label Prediction
#[derive(Debug, Clone)]
pub struct IndependentLabelPredictionTrained {
    binary_classifiers: Vec<BinaryClassifierModel>,
    thresholds: Vec<Float>,
    n_labels: usize,
}

/// Simple binary classifier model
#[derive(Debug, Clone)]
pub struct BinaryClassifierModel {
    weights: Array1<Float>,
    bias: Float,
    feature_means: Array1<Float>,
    feature_stds: Array1<Float>,
}

impl IndependentLabelPrediction<Untrained> {
    /// Create a new IndependentLabelPrediction instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            threshold_strategy: ThresholdStrategy::Fixed(0.5),
            optimize_thresholds: false,
            class_weight: None,
            random_state: None,
        }
    }

    /// Set the threshold strategy
    pub fn threshold_strategy(mut self, strategy: ThresholdStrategy) -> Self {
        self.threshold_strategy = strategy;
        self
    }

    /// Set whether to optimize thresholds
    pub fn optimize_thresholds(mut self, optimize: bool) -> Self {
        self.optimize_thresholds = optimize;
        self
    }

    /// Set class weight strategy
    pub fn class_weight(mut self, weight: Option<String>) -> Self {
        self.class_weight = weight;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Default for IndependentLabelPrediction<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for IndependentLabelPrediction<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, i32>> for IndependentLabelPrediction<Untrained> {
    type Fitted = IndependentLabelPrediction<IndependentLabelPredictionTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView2<'_, i32>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let (y_samples, n_labels) = y.dim();

        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for training".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = thread_rng();

        // Train binary classifiers for each label
        let mut binary_classifiers = Vec::new();
        for label_idx in 0..n_labels {
            let label_column = y.column(label_idx);
            let classifier = self.train_binary_classifier(x, &label_column, &mut rng)?;
            binary_classifiers.push(classifier);
        }

        // Determine thresholds
        let thresholds = match &self.threshold_strategy {
            ThresholdStrategy::Fixed(threshold) => vec![*threshold; n_labels],
            ThresholdStrategy::PerLabel(thresholds) => {
                if thresholds.len() != n_labels {
                    return Err(SklearsError::InvalidInput(
                        "Number of thresholds must match number of labels".to_string(),
                    ));
                }
                thresholds.clone()
            }
            ThresholdStrategy::Optimal => {
                self.optimize_thresholds_for_accuracy(x, y, &binary_classifiers)?
            }
            ThresholdStrategy::FScore => {
                self.optimize_thresholds_for_fscore(x, y, &binary_classifiers)?
            }
        };

        Ok(IndependentLabelPrediction {
            state: IndependentLabelPredictionTrained {
                binary_classifiers,
                thresholds,
                n_labels,
            },
            threshold_strategy: self.threshold_strategy,
            optimize_thresholds: self.optimize_thresholds,
            class_weight: self.class_weight,
            random_state: self.random_state,
        })
    }
}

impl IndependentLabelPrediction<Untrained> {
    fn train_binary_classifier(
        &self,
        x: &ArrayView2<'_, Float>,
        y_label: &ArrayView1<'_, i32>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<BinaryClassifierModel> {
        let (n_samples, n_features) = x.dim();

        // Compute feature statistics for normalization
        let feature_means = x.mean_axis(Axis(0)).unwrap();
        let feature_stds = x.mapv(|val| val * val).mean_axis(Axis(0)).unwrap()
            - &feature_means.mapv(|mean| mean * mean);
        let feature_stds = feature_stds.mapv(|var| (var.max(1e-10)).sqrt());

        // Normalize features
        let mut x_normalized = x.to_owned();
        for (i, mut row) in x_normalized.rows_mut().into_iter().enumerate() {
            row -= &feature_means;
            row /= &feature_stds;
        }

        // Compute class weights if requested
        let class_weights = if self.class_weight.as_deref() == Some("balanced") {
            let pos_count = y_label.iter().filter(|&&y| y == 1).count();
            let neg_count = n_samples - pos_count;

            if pos_count == 0 || neg_count == 0 {
                (1.0, 1.0)
            } else {
                let pos_weight = n_samples as Float / (2.0 * pos_count as Float);
                let neg_weight = n_samples as Float / (2.0 * neg_count as Float);
                (neg_weight, pos_weight)
            }
        } else {
            (1.0, 1.0)
        };

        // Simple logistic regression using gradient descent
        let mut weights = Array1::<Float>::zeros(n_features);
        let mut bias = 0.0;

        let learning_rate = 0.01;
        let max_iter = 1000;
        let tolerance = 1e-6;

        for iteration in 0..max_iter {
            let mut weight_gradient = Array1::<Float>::zeros(n_features);
            let mut bias_gradient = 0.0;
            let mut total_loss = 0.0;

            for sample_idx in 0..n_samples {
                let x_sample = x_normalized.row(sample_idx);
                let y_true = y_label[sample_idx] as Float;

                // Forward pass
                let logits = x_sample.dot(&weights) + bias;
                let prediction = 1.0 / (1.0 + (-logits).exp());

                // Compute loss with class weights
                let sample_weight = if y_true > 0.5 {
                    class_weights.1
                } else {
                    class_weights.0
                };
                let loss = -sample_weight
                    * (y_true * prediction.ln() + (1.0 - y_true) * (1.0 - prediction).ln());
                total_loss += loss;

                // Backward pass
                let error = sample_weight * (prediction - y_true);
                weight_gradient += &(x_sample.to_owned() * error);
                bias_gradient += error;
            }

            // Update parameters
            weights -= &(weight_gradient * (learning_rate / n_samples as Float));
            bias -= bias_gradient * (learning_rate / n_samples as Float);

            // Check convergence
            if iteration > 10 {
                let avg_loss = total_loss / n_samples as Float;
                if avg_loss < tolerance {
                    break;
                }
            }
        }

        Ok(BinaryClassifierModel {
            weights,
            bias,
            feature_means,
            feature_stds,
        })
    }

    fn optimize_thresholds_for_accuracy(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, i32>,
        classifiers: &[BinaryClassifierModel],
    ) -> SklResult<Vec<Float>> {
        let n_labels = y.ncols();
        let mut thresholds = Vec::new();

        for label_idx in 0..n_labels {
            let y_true = y.column(label_idx);
            let y_scores = self.predict_probabilities_single_label(x, &classifiers[label_idx])?;

            let mut best_threshold = 0.5;
            let mut best_accuracy = 0.0;

            // Grid search for best threshold
            for threshold_int in 1..100 {
                let threshold = threshold_int as Float / 100.0;

                let mut correct = 0;
                for sample_idx in 0..x.nrows() {
                    let predicted = if y_scores[sample_idx] >= threshold {
                        1
                    } else {
                        0
                    };
                    if predicted == y_true[sample_idx] {
                        correct += 1;
                    }
                }

                let accuracy = correct as Float / x.nrows() as Float;
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                    best_threshold = threshold;
                }
            }

            thresholds.push(best_threshold);
        }

        Ok(thresholds)
    }

    fn optimize_thresholds_for_fscore(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, i32>,
        classifiers: &[BinaryClassifierModel],
    ) -> SklResult<Vec<Float>> {
        let n_labels = y.ncols();
        let mut thresholds = Vec::new();

        for label_idx in 0..n_labels {
            let y_true = y.column(label_idx);
            let y_scores = self.predict_probabilities_single_label(x, &classifiers[label_idx])?;

            let mut best_threshold = 0.5;
            let mut best_fscore = 0.0;

            // Grid search for best F-score threshold
            for threshold_int in 1..100 {
                let threshold = threshold_int as Float / 100.0;

                let mut tp = 0;
                let mut fp = 0;
                let mut fn_count = 0;

                for sample_idx in 0..x.nrows() {
                    let predicted = if y_scores[sample_idx] >= threshold {
                        1
                    } else {
                        0
                    };
                    let actual = y_true[sample_idx];

                    match (actual, predicted) {
                        (1, 1) => tp += 1,
                        (0, 1) => fp += 1,
                        (1, 0) => fn_count += 1,
                        _ => {}
                    }
                }

                let precision = if tp + fp > 0 {
                    tp as Float / (tp + fp) as Float
                } else {
                    0.0
                };
                let recall = if tp + fn_count > 0 {
                    tp as Float / (tp + fn_count) as Float
                } else {
                    0.0
                };
                let fscore = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };

                if fscore > best_fscore {
                    best_fscore = fscore;
                    best_threshold = threshold;
                }
            }

            thresholds.push(best_threshold);
        }

        Ok(thresholds)
    }

    fn predict_probabilities_single_label(
        &self,
        x: &ArrayView2<'_, Float>,
        classifier: &BinaryClassifierModel,
    ) -> SklResult<Array1<Float>> {
        let n_samples = x.nrows();
        let mut probabilities = Array1::<Float>::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let x_sample = x.row(sample_idx);

            // Normalize features
            let x_normalized =
                (&x_sample.to_owned() - &classifier.feature_means) / &classifier.feature_stds;

            // Compute logits and probability
            let logits = x_normalized.dot(&classifier.weights) + classifier.bias;
            let probability = 1.0 / (1.0 + (-logits).exp());

            probabilities[sample_idx] = probability;
        }

        Ok(probabilities)
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for IndependentLabelPrediction<IndependentLabelPredictionTrained>
{
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = x.dim();
        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        for label_idx in 0..self.state.n_labels {
            let classifier = &self.state.binary_classifiers[label_idx];
            let threshold = self.state.thresholds[label_idx];

            for sample_idx in 0..n_samples {
                let x_sample = x.row(sample_idx);

                // Normalize features
                let x_normalized =
                    (&x_sample.to_owned() - &classifier.feature_means) / &classifier.feature_stds;

                // Compute probability
                let logits = x_normalized.dot(&classifier.weights) + classifier.bias;
                let probability = 1.0 / (1.0 + (-logits).exp());

                // Apply threshold
                predictions[[sample_idx, label_idx]] = if probability >= threshold { 1 } else { 0 };
            }
        }

        Ok(predictions)
    }
}

impl IndependentLabelPrediction<IndependentLabelPredictionTrained> {
    /// Predict probabilities for each label
    pub fn predict_proba(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, _) = x.dim();
        let mut probabilities = Array2::<Float>::zeros((n_samples, self.state.n_labels));

        for label_idx in 0..self.state.n_labels {
            let classifier = &self.state.binary_classifiers[label_idx];

            for sample_idx in 0..n_samples {
                let x_sample = x.row(sample_idx);

                // Normalize features
                let x_normalized =
                    (&x_sample.to_owned() - &classifier.feature_means) / &classifier.feature_stds;

                // Compute probability
                let logits = x_normalized.dot(&classifier.weights) + classifier.bias;
                let probability = 1.0 / (1.0 + (-logits).exp());

                probabilities[[sample_idx, label_idx]] = probability;
            }
        }

        Ok(probabilities)
    }

    /// Get the learned thresholds for each label
    pub fn thresholds(&self) -> &[Float] {
        &self.state.thresholds
    }

    /// Get the feature importance scores for each label
    pub fn feature_importances(&self) -> Vec<Array1<Float>> {
        self.state
            .binary_classifiers
            .iter()
            .map(|classifier| classifier.weights.mapv(|w| w.abs()))
            .collect()
    }
}

//! Ensemble methods for multi-output learning

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Gradient Boosting Multi-Output
///
/// A gradient boosting implementation that can handle multiple output variables simultaneously.
/// This implementation builds an additive model of weak learners (multi-target regression trees)
/// in a forward stage-wise fashion, optimizing a loss function for all targets jointly.
///
/// # Mathematical Foundation
///
/// For multi-target regression, gradient boosting minimizes:
/// - L(y, F(x)) = Σ_i Σ_j (y_ij - F_j(x_i))^2 for all samples i and targets j
/// - F_j(x) = F_0j + Σ_m ρ_m * h_mj(x) where h_mj are weak learners for target j at stage m
/// - At each stage, fit weak learners to negative gradients: -∂L/∂F_j
///
/// # Examples
///
/// ```
/// use sklears_multioutput::GradientBoostingMultiOutput;
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 1.5], [4.5, 4.5]];
///
/// let gbm = GradientBoostingMultiOutput::new()
///     .n_estimators(50)
///     .learning_rate(0.1)
///     .max_depth(3);
///
/// let fitted = gbm.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GradientBoostingMultiOutput<S = Untrained> {
    state: S,
    /// Number of boosting stages to perform
    n_estimators: usize,
    /// Learning rate shrinks the contribution of each tree
    learning_rate: Float,
    /// Maximum depth of the individual regression estimators
    max_depth: usize,
    /// Minimum number of samples required to split an internal node
    min_samples_split: usize,
    /// Random state for reproducibility
    random_state: Option<u64>,
}

/// Trained state for Gradient Boosting Multi-Output
#[derive(Debug, Clone)]
pub struct GradientBoostingMultiOutputTrained {
    /// Initial predictions (mean of targets)
    pub initial_predictions: Array1<Float>,
    /// Weak learners for each stage and target
    pub estimators: Vec<Vec<WeakLearner>>,
    /// Number of features
    pub n_features: usize,
    /// Number of targets
    pub n_targets: usize,
}

/// A simple weak learner (decision stump) for gradient boosting
#[derive(Debug, Clone)]
pub struct WeakLearner {
    /// Feature index to split on
    feature_idx: usize,
    /// Threshold value for splitting
    threshold: Float,
    /// Prediction for left branch (feature <= threshold)
    left_value: Float,
    /// Prediction for right branch (feature > threshold)
    right_value: Float,
}

impl GradientBoostingMultiOutput<Untrained> {
    /// Create a new GradientBoostingMultiOutput instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            random_state: None,
        }
    }

    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum depth
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum samples to split
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Default for GradientBoostingMultiOutput<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GradientBoostingMultiOutput<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>> for GradientBoostingMultiOutput<Untrained> {
    type Fitted = GradientBoostingMultiOutput<GradientBoostingMultiOutputTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView2<'_, Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let (y_samples, n_targets) = y.dim();

        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if n_samples < self.min_samples_split {
            return Err(SklearsError::InvalidInput(
                "Not enough samples to perform gradient boosting".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = thread_rng();

        // Initialize predictions with the mean of each target
        let mut initial_predictions = Array1::<Float>::zeros(n_targets);
        for target_idx in 0..n_targets {
            let target_sum: Float = y.column(target_idx).sum();
            initial_predictions[target_idx] = target_sum / n_samples as Float;
        }

        // Initialize current predictions
        let mut current_predictions = Array2::<Float>::zeros((n_samples, n_targets));
        for target_idx in 0..n_targets {
            for sample_idx in 0..n_samples {
                current_predictions[[sample_idx, target_idx]] = initial_predictions[target_idx];
            }
        }

        let mut estimators = Vec::new();

        // Boosting iterations
        for _stage in 0..self.n_estimators {
            let mut stage_estimators = Vec::new();

            // Train one weak learner for each target
            for target_idx in 0..n_targets {
                // Compute negative gradients (residuals for squared loss)
                let mut residuals = Array1::<Float>::zeros(n_samples);
                for sample_idx in 0..n_samples {
                    residuals[sample_idx] =
                        y[[sample_idx, target_idx]] - current_predictions[[sample_idx, target_idx]];
                }

                // Train weak learner on residuals
                let weak_learner = self.train_weak_learner(x, &residuals, &mut rng)?;

                // Make predictions with this weak learner
                let weak_predictions = self.predict_weak_learner(&weak_learner, x);

                // Update current predictions
                for sample_idx in 0..n_samples {
                    current_predictions[[sample_idx, target_idx]] +=
                        self.learning_rate * weak_predictions[sample_idx];
                }

                stage_estimators.push(weak_learner);
            }

            estimators.push(stage_estimators);
        }

        Ok(GradientBoostingMultiOutput {
            state: GradientBoostingMultiOutputTrained {
                initial_predictions,
                estimators,
                n_features,
                n_targets,
            },
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            random_state: self.random_state,
        })
    }
}

impl GradientBoostingMultiOutput<Untrained> {
    fn train_weak_learner(
        &self,
        x: &ArrayView2<'_, Float>,
        residuals: &Array1<Float>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<WeakLearner> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples to train weak learner".to_string(),
            ));
        }

        let mut best_loss = Float::INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_left_value = 0.0;
        let mut best_right_value = 0.0;

        // Try random features and thresholds
        let n_trials = (n_features * 10).min(100);
        for _ in 0..n_trials {
            let feature_idx = rng.gen_range(0..n_features);

            // Get feature values and find min/max
            let feature_values: Vec<Float> = (0..n_samples).map(|i| x[[i, feature_idx]]).collect();

            let min_val = feature_values
                .iter()
                .cloned()
                .fold(Float::INFINITY, Float::min);
            let max_val = feature_values
                .iter()
                .cloned()
                .fold(Float::NEG_INFINITY, Float::max);

            if (max_val - min_val).abs() < 1e-10 {
                continue; // Skip if all values are the same
            }

            // Try random thresholds
            for _ in 0..10 {
                let threshold = min_val + rng.gen::<Float>() * (max_val - min_val);

                // Split samples based on threshold
                let mut left_residuals = Vec::new();
                let mut right_residuals = Vec::new();

                for sample_idx in 0..n_samples {
                    if x[[sample_idx, feature_idx]] <= threshold {
                        left_residuals.push(residuals[sample_idx]);
                    } else {
                        right_residuals.push(residuals[sample_idx]);
                    }
                }

                if left_residuals.is_empty() || right_residuals.is_empty() {
                    continue;
                }

                // Compute predictions for each split
                let left_value =
                    left_residuals.iter().sum::<Float>() / left_residuals.len() as Float;
                let right_value =
                    right_residuals.iter().sum::<Float>() / right_residuals.len() as Float;

                // Compute loss (mean squared error)
                let left_loss: Float = left_residuals
                    .iter()
                    .map(|&r| (r - left_value).powi(2))
                    .sum();
                let right_loss: Float = right_residuals
                    .iter()
                    .map(|&r| (r - right_value).powi(2))
                    .sum();
                let total_loss = left_loss + right_loss;

                if total_loss < best_loss {
                    best_loss = total_loss;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                    best_left_value = left_value;
                    best_right_value = right_value;
                }
            }
        }

        Ok(WeakLearner {
            feature_idx: best_feature,
            threshold: best_threshold,
            left_value: best_left_value,
            right_value: best_right_value,
        })
    }

    fn predict_weak_learner(
        &self,
        learner: &WeakLearner,
        x: &ArrayView2<'_, Float>,
    ) -> Array1<Float> {
        let n_samples = x.nrows();
        let mut predictions = Array1::<Float>::zeros(n_samples);

        for sample_idx in 0..n_samples {
            if x[[sample_idx, learner.feature_idx]] <= learner.threshold {
                predictions[sample_idx] = learner.left_value;
            } else {
                predictions[sample_idx] = learner.right_value;
            }
        }

        predictions
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>>
    for GradientBoostingMultiOutput<GradientBoostingMultiOutputTrained>
{
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features, n_features
            )));
        }

        let mut predictions = Array2::<Float>::zeros((n_samples, self.state.n_targets));

        // Initialize with base predictions
        for target_idx in 0..self.state.n_targets {
            for sample_idx in 0..n_samples {
                predictions[[sample_idx, target_idx]] = self.state.initial_predictions[target_idx];
            }
        }

        // Add contributions from all stages
        for stage_estimators in &self.state.estimators {
            for (target_idx, weak_learner) in stage_estimators.iter().enumerate() {
                for sample_idx in 0..n_samples {
                    let prediction =
                        if x[[sample_idx, weak_learner.feature_idx]] <= weak_learner.threshold {
                            weak_learner.left_value
                        } else {
                            weak_learner.right_value
                        };

                    predictions[[sample_idx, target_idx]] += self.learning_rate * prediction;
                }
            }
        }

        Ok(predictions)
    }
}

impl GradientBoostingMultiOutput<GradientBoostingMultiOutputTrained> {
    /// Get the feature importance scores
    pub fn feature_importances(&self) -> Array1<Float> {
        let mut importances = Array1::<Float>::zeros(self.state.n_features);

        for stage_estimators in &self.state.estimators {
            for weak_learner in stage_estimators {
                // Simple importance: frequency of feature usage
                importances[weak_learner.feature_idx] += 1.0;
            }
        }

        // Normalize by total number of estimators
        let total = importances.sum();
        if total > 0.0 {
            importances /= total;
        }

        importances
    }

    /// Get the training loss history
    pub fn training_loss_history(&self) -> Vec<Float> {
        // This would require storing training history during fit
        // For now, return empty vec as placeholder
        Vec::new()
    }
}

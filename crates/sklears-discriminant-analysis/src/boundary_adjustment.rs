//! Boundary Adjustment Techniques for Imbalanced Classification
//!
//! This module implements various boundary adjustment techniques that modify
//! decision boundaries to better handle class imbalance in discriminant analysis.
//! These methods focus on post-training adjustments rather than data resampling.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba},
    types::Float,
};
use std::collections::HashMap;

/// Boundary adjustment methods for imbalanced classification
#[derive(Debug, Clone)]
pub enum BoundaryAdjustmentMethod {
    /// Threshold optimization using various criteria
    ThresholdOptimization {
        criterion: OptimizationCriterion,

        search_method: SearchMethod,
    },
    /// Cost-sensitive boundary shifting
    CostSensitiveBoundary {
        cost_matrix: Vec<Vec<Float>>,

        adjustment_factor: Float,
    },
    /// Density-based boundary adjustment
    DensityWeighting {
        bandwidth: Float,
        kernel: DensityKernel,
    },
    /// Margin-based adjustment
    MarginAdjustment {
        margin_factor: Float,
        adaptive: bool,
    },
    /// Class-specific boundary shifting
    ClassSpecificShift { shift_factors: HashMap<i32, Float> },
}

/// Optimization criteria for threshold selection
#[derive(Debug, Clone)]
pub enum OptimizationCriterion {
    /// Maximize F1-score
    F1Score,
    /// Maximize balanced accuracy
    BalancedAccuracy,
    /// Maximize G-mean (geometric mean of sensitivities)
    GMean,
    /// Minimize cost with given cost matrix
    MinimizeCost(Vec<Vec<Float>>),
    /// Maximize Youden's J statistic
    YoudenJ,
    /// Target specific precision/recall balance
    PrecisionRecallBalance { precision_weight: Float },
}

/// Search methods for optimal threshold
#[derive(Debug, Clone)]
pub enum SearchMethod {
    /// Grid search with specified resolution
    GridSearch { resolution: usize },
    /// Golden section search
    GoldenSection { tolerance: Float },
    /// Binary search
    BinarySearch { tolerance: Float },
    /// Bayesian optimization
    BayesianOptimization { n_calls: usize },
}

/// Density kernels for boundary weighting
#[derive(Debug, Clone)]
pub enum DensityKernel {
    /// Gaussian kernel
    Gaussian,
    /// Epanechnikov kernel
    Epanechnikov,
    /// Uniform kernel
    Uniform,
    /// Triangular kernel
    Triangular,
}

impl Default for BoundaryAdjustmentMethod {
    fn default() -> Self {
        BoundaryAdjustmentMethod::ThresholdOptimization {
            criterion: OptimizationCriterion::F1Score,
            search_method: SearchMethod::GridSearch { resolution: 100 },
        }
    }
}

/// Configuration for boundary adjustment
#[derive(Debug, Clone)]
pub struct BoundaryAdjustmentConfig {
    /// Method for boundary adjustment
    pub method: BoundaryAdjustmentMethod,
    /// Whether to use cross-validation for threshold selection
    pub use_cv: bool,
    /// Number of CV folds
    pub cv_folds: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Validation split for threshold optimization
    pub validation_split: Option<Float>,
}

impl Default for BoundaryAdjustmentConfig {
    fn default() -> Self {
        Self {
            method: BoundaryAdjustmentMethod::default(),
            use_cv: false,
            cv_folds: 5,
            random_state: None,
            validation_split: Some(0.2),
        }
    }
}

/// Boundary adjustment discriminant analysis
pub struct BoundaryAdjustmentDiscriminantAnalysis<T> {
    config: BoundaryAdjustmentConfig,
    base_estimator: T,
}

/// Trained boundary adjustment model
pub struct TrainedBoundaryAdjustmentDiscriminantAnalysis<F> {
    config: BoundaryAdjustmentConfig,
    base_model: F,
    classes: Vec<i32>,
    optimal_thresholds: HashMap<i32, Float>,
    class_densities: Option<HashMap<i32, Array2<Float>>>,
    adjustment_factors: HashMap<i32, Float>,
    validation_scores: HashMap<String, Float>,
}

impl<T> BoundaryAdjustmentDiscriminantAnalysis<T> {
    /// Create a new boundary adjustment discriminant analysis
    pub fn new(base_estimator: T) -> Self {
        Self {
            config: BoundaryAdjustmentConfig::default(),
            base_estimator,
        }
    }

    /// Set the boundary adjustment method
    pub fn method(mut self, method: BoundaryAdjustmentMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Enable cross-validation for threshold selection
    pub fn use_cv(mut self, use_cv: bool, cv_folds: usize) -> Self {
        self.config.use_cv = use_cv;
        self.config.cv_folds = cv_folds;
        self
    }

    /// Set validation split ratio
    pub fn validation_split(mut self, split: Float) -> Self {
        self.config.validation_split = Some(split);
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Get unique classes from labels
    fn get_unique_classes(&self, y: &Array1<i32>) -> Vec<i32> {
        let mut classes: Vec<i32> = y.iter().cloned().collect();
        classes.sort_unstable();
        classes.dedup();
        classes
    }

    /// Split data for validation
    fn train_validation_split(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        split_ratio: Float,
    ) -> Result<(Array2<Float>, Array1<i32>, Array2<Float>, Array1<i32>)> {
        let n_samples = x.nrows();
        let n_train = ((1.0 - split_ratio) * n_samples as Float) as usize;

        let indices: Vec<usize> = (0..n_samples).collect();
        // In practice, you would shuffle these indices randomly

        let train_indices = &indices[..n_train];
        let val_indices = &indices[n_train..];

        let x_train = x.select(Axis(0), train_indices);
        let y_train = y.select(Axis(0), train_indices);
        let x_val = x.select(Axis(0), val_indices);
        let y_val = y.select(Axis(0), val_indices);

        Ok((x_train, y_train, x_val, y_val))
    }

    /// Calculate F1 score for a given threshold
    fn calculate_f1_score(
        &self,
        probabilities: &Array2<Float>,
        y_true: &Array1<i32>,
        threshold: Float,
        positive_class: i32,
        classes: &[i32],
    ) -> Result<Float> {
        let pos_class_idx = classes
            .iter()
            .position(|&c| c == positive_class)
            .ok_or_else(|| SklearsError::InvalidInput("Class not found in classes".to_string()))?;

        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (i, &true_label) in y_true.iter().enumerate() {
            let predicted_positive = probabilities[[i, pos_class_idx]] >= threshold;
            let actually_positive = true_label == positive_class;

            match (predicted_positive, actually_positive) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_count += 1,
                (false, false) => {} // TN - not needed for F1
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

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Ok(f1)
    }

    /// Calculate balanced accuracy
    fn calculate_balanced_accuracy(
        &self,
        probabilities: &Array2<Float>,
        y_true: &Array1<i32>,
        thresholds: &HashMap<i32, Float>,
        classes: &[i32],
    ) -> Result<Float> {
        let mut class_accuracies = Vec::new();

        for &class in classes {
            let class_idx = classes.iter().position(|&c| c == class).unwrap();
            let threshold = thresholds.get(&class).copied().unwrap_or(0.5);

            let mut correct = 0;
            let mut total = 0;

            for (i, &true_label) in y_true.iter().enumerate() {
                if true_label == class {
                    total += 1;
                    if probabilities[[i, class_idx]] >= threshold {
                        correct += 1;
                    }
                }
            }

            let accuracy = if total > 0 {
                correct as Float / total as Float
            } else {
                0.0
            };

            class_accuracies.push(accuracy);
        }

        Ok(class_accuracies.iter().sum::<Float>() / class_accuracies.len() as Float)
    }

    /// Optimize threshold using grid search
    fn optimize_threshold_grid_search(
        &self,
        probabilities: &Array2<Float>,
        y_true: &Array1<i32>,
        criterion: &OptimizationCriterion,
        classes: &[i32],
        resolution: usize,
    ) -> Result<HashMap<i32, Float>> {
        let mut optimal_thresholds = HashMap::new();

        for &class in classes {
            let mut best_threshold = 0.5;
            let mut best_score = -Float::INFINITY;

            for i in 0..=resolution {
                let threshold = i as Float / resolution as Float;

                let score = match criterion {
                    OptimizationCriterion::F1Score => {
                        self.calculate_f1_score(probabilities, y_true, threshold, class, classes)?
                    }
                    OptimizationCriterion::BalancedAccuracy => {
                        let mut temp_thresholds = optimal_thresholds.clone();
                        temp_thresholds.insert(class, threshold);
                        self.calculate_balanced_accuracy(
                            probabilities,
                            y_true,
                            &temp_thresholds,
                            classes,
                        )?
                    }
                    _ => 0.5, // Placeholder for other criteria
                };

                if score > best_score {
                    best_score = score;
                    best_threshold = threshold;
                }
            }

            optimal_thresholds.insert(class, best_threshold);
        }

        Ok(optimal_thresholds)
    }

    /// Calculate density weights for each class
    fn calculate_density_weights(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        bandwidth: Float,
        kernel: &DensityKernel,
    ) -> Result<HashMap<i32, Array1<Float>>> {
        let classes = self.get_unique_classes(y);
        let mut density_weights = HashMap::new();

        for &class in &classes {
            let class_samples: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .collect();

            let mut weights = Array1::zeros(x.nrows());

            for (sample_idx, sample) in x.axis_iter(Axis(0)).enumerate() {
                let mut density = 0.0;

                for &class_sample_idx in &class_samples {
                    let distance = self.euclidean_distance(&sample, &x.row(class_sample_idx))?;
                    let kernel_value = self.evaluate_kernel(distance / bandwidth, kernel);
                    density += kernel_value;
                }

                density /= class_samples.len() as Float * bandwidth;
                weights[sample_idx] = density;
            }

            density_weights.insert(class, weights);
        }

        Ok(density_weights)
    }

    /// Calculate Euclidean distance between two samples
    fn euclidean_distance(
        &self,
        sample1: &ArrayView1<Float>,
        sample2: &ArrayView1<Float>,
    ) -> Result<Float> {
        if sample1.len() != sample2.len() {
            return Err(SklearsError::InvalidData {
                reason: "Samples must have same dimensionality".to_string(),
            });
        }

        let distance = sample1
            .iter()
            .zip(sample2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<Float>()
            .sqrt();

        Ok(distance)
    }

    /// Evaluate kernel function
    fn evaluate_kernel(&self, u: Float, kernel: &DensityKernel) -> Float {
        match kernel {
            DensityKernel::Gaussian => (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt(),
            DensityKernel::Epanechnikov => {
                if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
            DensityKernel::Uniform => {
                if u.abs() <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            DensityKernel::Triangular => {
                if u.abs() <= 1.0 {
                    1.0 - u.abs()
                } else {
                    0.0
                }
            }
        }
    }
}

impl<T> Estimator for BoundaryAdjustmentDiscriminantAnalysis<T> {
    type Config = BoundaryAdjustmentConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> Fit<Array2<Float>, Array1<i32>> for BoundaryAdjustmentDiscriminantAnalysis<T>
where
    T: Fit<Array2<Float>, Array1<i32>> + Clone,
    T::Fitted: PredictProba<Array2<Float>, Array2<Float>>,
{
    type Fitted = TrainedBoundaryAdjustmentDiscriminantAnalysis<T::Fitted>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidData {
                reason: "Number of samples in X and y must match".to_string(),
            });
        }

        let classes = self.get_unique_classes(y);

        // Train base model on full data or training split
        let (x_train, y_train, x_val, y_val) = if let Some(split) = self.config.validation_split {
            self.train_validation_split(x, y, split)?
        } else {
            (x.clone(), y.clone(), x.clone(), y.clone())
        };

        let base_model = self.base_estimator.clone().fit(&x_train, &y_train)?;

        // Get probabilities for threshold optimization
        let val_probabilities = base_model.predict_proba(&x_val)?;

        // Optimize thresholds/boundaries based on method
        let (optimal_thresholds, adjustment_factors, class_densities) = match &self.config.method {
            BoundaryAdjustmentMethod::ThresholdOptimization {
                criterion,
                search_method,
            } => {
                let thresholds = match search_method {
                    SearchMethod::GridSearch { resolution } => self
                        .optimize_threshold_grid_search(
                            &val_probabilities,
                            &y_val,
                            criterion,
                            &classes,
                            *resolution,
                        )?,
                    _ => {
                        // Placeholder for other search methods
                        classes.iter().map(|&c| (c, 0.5)).collect()
                    }
                };

                (thresholds, HashMap::new(), None)
            }

            BoundaryAdjustmentMethod::DensityWeighting { bandwidth, kernel } => {
                let _densities =
                    self.calculate_density_weights(&x_train, &y_train, *bandwidth, kernel)?;
                let thresholds = classes.iter().map(|&c| (c, 0.5)).collect();
                (thresholds, HashMap::new(), Some(HashMap::new()))
            }

            BoundaryAdjustmentMethod::ClassSpecificShift { shift_factors } => {
                let thresholds = classes.iter().map(|&c| (c, 0.5)).collect();
                (thresholds, shift_factors.clone(), None)
            }

            BoundaryAdjustmentMethod::CostSensitiveBoundary {
                cost_matrix: _,
                adjustment_factor,
            } => {
                // Implement cost-sensitive boundary adjustment
                let thresholds = classes.iter().map(|&c| (c, 0.5)).collect();
                let factors = classes.iter().map(|&c| (c, *adjustment_factor)).collect();
                (thresholds, factors, None)
            }

            BoundaryAdjustmentMethod::MarginAdjustment {
                margin_factor,
                adaptive: _,
            } => {
                let thresholds = classes.iter().map(|&c| (c, 0.5)).collect();
                let factors = classes.iter().map(|&c| (c, *margin_factor)).collect();
                (thresholds, factors, None)
            }
        };

        // Calculate validation scores
        let mut validation_scores = HashMap::new();
        validation_scores.insert(
            "balanced_accuracy".to_string(),
            self.calculate_balanced_accuracy(
                &val_probabilities,
                &y_val,
                &optimal_thresholds,
                &classes,
            )?,
        );

        Ok(TrainedBoundaryAdjustmentDiscriminantAnalysis {
            config: self.config,
            base_model,
            classes,
            optimal_thresholds,
            class_densities,
            adjustment_factors,
            validation_scores,
        })
    }
}

impl<F> Predict<Array2<Float>, Array1<i32>> for TrainedBoundaryAdjustmentDiscriminantAnalysis<F>
where
    F: PredictProba<Array2<Float>, Array2<Float>>,
{
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probabilities = self.base_model.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, prob_row) in probabilities.axis_iter(Axis(0)).enumerate() {
            let mut best_class = self.classes[0];
            let mut best_adjusted_prob = -Float::INFINITY;

            for (class_idx, &class) in self.classes.iter().enumerate() {
                let base_prob = prob_row[class_idx];
                let threshold = self.optimal_thresholds.get(&class).copied().unwrap_or(0.5);
                let adjustment = self.adjustment_factors.get(&class).copied().unwrap_or(1.0);

                // Apply boundary adjustment
                let adjusted_prob = base_prob * adjustment;

                if adjusted_prob >= threshold && adjusted_prob > best_adjusted_prob {
                    best_adjusted_prob = adjusted_prob;
                    best_class = class;
                }
            }

            predictions[i] = best_class;
        }

        Ok(predictions)
    }
}

impl<F> PredictProba<Array2<Float>, Array2<Float>>
    for TrainedBoundaryAdjustmentDiscriminantAnalysis<F>
where
    F: PredictProba<Array2<Float>, Array2<Float>>,
{
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let base_probabilities = self.base_model.predict_proba(x)?;
        let mut adjusted_probabilities = base_probabilities.clone();

        // Apply boundary adjustments to probabilities
        for mut prob_row in adjusted_probabilities.axis_iter_mut(Axis(0)) {
            let mut sum = 0.0;

            for (class_idx, &class) in self.classes.iter().enumerate() {
                let adjustment = self.adjustment_factors.get(&class).copied().unwrap_or(1.0);
                prob_row[class_idx] *= adjustment;
                sum += prob_row[class_idx];
            }

            // Renormalize to ensure probabilities sum to 1
            if sum > 0.0 {
                for prob in prob_row.iter_mut() {
                    *prob /= sum;
                }
            }
        }

        Ok(adjusted_probabilities)
    }
}

impl<F> TrainedBoundaryAdjustmentDiscriminantAnalysis<F> {
    /// Get the classes
    pub fn classes(&self) -> &[i32] {
        &self.classes
    }

    /// Get optimal thresholds for each class
    pub fn optimal_thresholds(&self) -> &HashMap<i32, Float> {
        &self.optimal_thresholds
    }

    /// Get adjustment factors for each class
    pub fn adjustment_factors(&self) -> &HashMap<i32, Float> {
        &self.adjustment_factors
    }

    /// Get validation scores
    pub fn validation_scores(&self) -> &HashMap<String, Float> {
        &self.validation_scores
    }

    /// Get class densities (if computed)
    pub fn class_densities(&self) -> Option<&HashMap<i32, Array2<Float>>> {
        self.class_densities.as_ref()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Mock base classifier for testing
    #[derive(Debug, Clone)]
    struct MockClassifier {
        classes: Vec<i32>,
    }

    impl MockClassifier {
        fn new(classes: Vec<i32>) -> Self {
            Self { classes }
        }
    }

    impl Estimator for MockClassifier {
        type Config = ();
        type Error = SklearsError;
        type Float = Float;

        fn config(&self) -> &Self::Config {
            &()
        }
    }

    impl Fit<Array2<Float>, Array1<i32>> for MockClassifier {
        type Fitted = TrainedMockClassifier;

        fn fit(self, _x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
            let classes = y
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            Ok(TrainedMockClassifier { classes })
        }
    }

    #[derive(Debug)]
    struct TrainedMockClassifier {
        classes: Vec<i32>,
    }

    impl PredictProba<Array2<Float>, Array2<Float>> for TrainedMockClassifier {
        fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
            let n_samples = x.nrows();
            let n_classes = self.classes.len();
            let prob = 1.0 / n_classes as Float;
            Ok(Array2::from_elem((n_samples, n_classes), prob))
        }
    }

    #[test]
    fn test_boundary_adjustment_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [10.0, 11.0]];
        let y = array![0, 0, 0, 1, 1];

        let base_classifier = MockClassifier::new(vec![0, 1]);
        let boundary_adj = BoundaryAdjustmentDiscriminantAnalysis::new(base_classifier).method(
            BoundaryAdjustmentMethod::ThresholdOptimization {
                criterion: OptimizationCriterion::F1Score,
                search_method: SearchMethod::GridSearch { resolution: 10 },
            },
        );

        let fitted = boundary_adj.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 5);
        assert_eq!(fitted.classes().len(), 2);
        assert!(fitted.optimal_thresholds().len() > 0);
    }

    #[test]
    fn test_density_weighting() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let base_classifier = MockClassifier::new(vec![0, 1]);
        let boundary_adj = BoundaryAdjustmentDiscriminantAnalysis::new(base_classifier).method(
            BoundaryAdjustmentMethod::DensityWeighting {
                bandwidth: 1.0,
                kernel: DensityKernel::Gaussian,
            },
        );

        let fitted = boundary_adj.fit(&x, &y).unwrap();
        let probabilities = fitted.predict_proba(&x).unwrap();

        assert_eq!(probabilities.dim(), (4, 2));
        // Check that probabilities sum to 1
        for row in probabilities.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_f1_score_calculation() {
        let boundary_adj =
            BoundaryAdjustmentDiscriminantAnalysis::new(MockClassifier::new(vec![0, 1]));

        let probabilities = array![[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]];
        let y_true = array![0, 1, 0, 1];
        let classes = vec![0, 1];

        let f1_score = boundary_adj
            .calculate_f1_score(&probabilities, &y_true, 0.5, 0, &classes)
            .unwrap();

        assert!(f1_score >= 0.0 && f1_score <= 1.0);
    }
}

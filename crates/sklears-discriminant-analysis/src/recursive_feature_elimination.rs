//! Recursive Feature Elimination (RFE) for Discriminant Analysis
//!
//! This module implements Recursive Feature Elimination, a feature selection technique
//! that recursively removes features and builds the model on the remaining attributes
//! which are ranked by the model's importance.

use crate::lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig};
// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, Trained, Transform},
    types::Float,
};
use std::collections::HashSet;

/// Configuration for Recursive Feature Elimination
#[derive(Debug, Clone)]
pub struct RecursiveFeatureEliminationConfig {
    /// Number of features to select
    pub n_features_to_select: Option<usize>,
    /// Step size for feature elimination (number of features to remove at each iteration)
    pub step: usize,
    /// Base estimator configuration
    pub estimator_config: LinearDiscriminantAnalysisConfig,
    /// Importance scoring method
    pub importance_method: String,
    /// Whether to use cross-validation for scoring
    pub cv: Option<usize>,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for RecursiveFeatureEliminationConfig {
    fn default() -> Self {
        Self {
            n_features_to_select: None,
            step: 1,
            estimator_config: LinearDiscriminantAnalysisConfig::default(),
            importance_method: "coef".to_string(),
            cv: None,
            random_state: None,
            verbose: false,
        }
    }
}

/// Recursive Feature Elimination for discriminant analysis
#[derive(Debug, Clone)]
pub struct RecursiveFeatureElimination {
    config: RecursiveFeatureEliminationConfig,
}

/// Trained Recursive Feature Elimination model
#[derive(Debug, Clone)]
pub struct TrainedRecursiveFeatureElimination {
    /// Selected feature indices
    support: Array1<bool>,
    /// Feature rankings (1 = best)
    ranking: Array1<usize>,
    /// Final trained estimator on selected features
    estimator: LinearDiscriminantAnalysis<Trained>,
    /// Number of features originally present
    n_features_in: usize,
    /// Configuration used for training
    config: RecursiveFeatureEliminationConfig,
    /// Feature elimination history
    elimination_history: Vec<EliminationStep>,
}

/// Information about each elimination step
#[derive(Debug, Clone)]
pub struct EliminationStep {
    /// Features remaining at this step
    pub features_remaining: usize,
    /// Features eliminated in this step
    pub eliminated_features: Vec<usize>,
    /// Importance scores at this step
    pub importance_scores: Array1<Float>,
    /// Cross-validation score (if cv enabled)
    pub cv_score: Option<Float>,
}

impl Default for RecursiveFeatureElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl RecursiveFeatureElimination {
    /// Create a new Recursive Feature Elimination model
    pub fn new() -> Self {
        Self {
            config: RecursiveFeatureEliminationConfig::default(),
        }
    }

    /// Set the number of features to select
    pub fn n_features_to_select(mut self, n_features: usize) -> Self {
        self.config.n_features_to_select = Some(n_features);
        self
    }

    /// Set the step size for elimination
    pub fn step(mut self, step: usize) -> Self {
        self.config.step = step.max(1);
        self
    }

    /// Set the base estimator configuration
    pub fn estimator_config(mut self, config: LinearDiscriminantAnalysisConfig) -> Self {
        self.config.estimator_config = config;
        self
    }

    /// Set the importance scoring method
    pub fn importance_method(mut self, method: &str) -> Self {
        self.config.importance_method = method.to_string();
        self
    }

    /// Set cross-validation folds
    pub fn cv(mut self, cv_folds: usize) -> Self {
        self.config.cv = Some(cv_folds);
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Extract feature importance from a trained estimator
    fn extract_importance(
        &self,
        estimator: &LinearDiscriminantAnalysis<Trained>,
        n_features: usize,
    ) -> Result<Array1<Float>> {
        match self.config.importance_method.as_str() {
            "coef" => {
                // Use LDA coefficients as importance scores
                let coef = estimator.coef();
                let importance = coef.map(|x| x.abs());
                // Determine which axis to sum over to get n_features values
                if coef.nrows() == n_features {
                    // Shape is (n_features, n_classes), sum across classes
                    Ok(importance.sum_axis(Axis(1)))
                } else if coef.ncols() == n_features {
                    // Shape is (n_classes, n_features), sum across classes
                    Ok(importance.sum_axis(Axis(0)))
                } else {
                    // If neither dimension matches, create uniform importance
                    Ok(Array1::ones(n_features))
                }
            }
            "scalings" => {
                // Use LDA scalings as importance scores
                let scalings = estimator.scalings();
                // Ensure we get the right number of importance values
                if scalings.nrows() == n_features {
                    // Shape is (n_features, n_components), use L2 norm across components
                    let mut importance = Array1::zeros(n_features);
                    for (i, row) in scalings.axis_iter(Axis(0)).enumerate() {
                        importance[i] = row.dot(&row).sqrt(); // L2 norm
                    }
                    Ok(importance)
                } else if scalings.ncols() == n_features {
                    // Shape is (n_components, n_features), use L2 norm across components
                    let mut importance = Array1::zeros(n_features);
                    for (i, col) in scalings.axis_iter(Axis(1)).enumerate() {
                        importance[i] = col.dot(&col).sqrt(); // L2 norm
                    }
                    Ok(importance)
                } else {
                    // If neither dimension matches, create uniform importance
                    Ok(Array1::ones(n_features))
                }
            }
            "fisher_score" => {
                // Calculate Fisher score for each feature
                // This would require access to the original data and class information
                // For now, fall back to coefficients with proper dimension handling
                let coef = estimator.coef();
                let importance = coef.map(|x| x.abs());
                if coef.nrows() == n_features {
                    Ok(importance.sum_axis(Axis(1)))
                } else if coef.ncols() == n_features {
                    Ok(importance.sum_axis(Axis(0)))
                } else {
                    Ok(Array1::ones(n_features))
                }
            }
            _ => Err(SklearsError::InvalidParameter {
                name: "importance_method".to_string(),
                reason: format!(
                    "Unknown importance method: {}. Available methods: coef, scalings, fisher_score",
                    self.config.importance_method
                ),
            }),
        }
    }

    /// Perform cross-validation scoring
    fn cross_validate(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        feature_mask: &Array1<bool>,
    ) -> Result<Float> {
        let cv_folds = self.config.cv.unwrap_or(5);
        let n_samples = x.nrows();

        // Create CV folds
        let fold_size = n_samples / cv_folds;
        let mut scores = Vec::new();

        for fold in 0..cv_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test splits
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for i in 0..n_samples {
                if i >= test_start && i < test_end {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            if train_indices.is_empty() || test_indices.is_empty() {
                continue;
            }

            // Extract features for this fold
            let selected_features: Vec<usize> = feature_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
                .collect();

            if selected_features.is_empty() {
                continue;
            }

            // Create train data
            let x_train = x.select(Axis(0), &train_indices);
            let x_train_selected = x_train.select(Axis(1), &selected_features);
            let y_train = y.select(Axis(0), &train_indices);

            // Create test data
            let x_test = x.select(Axis(0), &test_indices);
            let x_test_selected = x_test.select(Axis(1), &selected_features);
            let y_test = y.select(Axis(0), &test_indices);

            // Train and evaluate
            let estimator = LinearDiscriminantAnalysis::new();

            if let Ok(fitted) = estimator.fit(&x_train_selected, &y_train) {
                if let Ok(predictions) = fitted.predict(&x_test_selected) {
                    // Calculate accuracy
                    let correct = predictions
                        .iter()
                        .zip(y_test.iter())
                        .filter(|(&pred, &true_val)| pred == true_val)
                        .count();
                    let accuracy = correct as Float / y_test.len() as Float;
                    scores.push(accuracy);
                }
            }
        }

        if scores.is_empty() {
            return Ok(0.0);
        }

        // Return mean score
        let mean_score = scores.iter().sum::<Float>() / scores.len() as Float;
        Ok(mean_score)
    }
}

impl Estimator for RecursiveFeatureElimination {
    type Config = RecursiveFeatureEliminationConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for RecursiveFeatureElimination {
    type Fitted = TrainedRecursiveFeatureElimination;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<TrainedRecursiveFeatureElimination> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.len()".to_string(),
                actual: format!("X.shape[0]={}, y.len()={}", x.nrows(), y.len()),
            });
        }

        let n_features = x.ncols();
        let n_features_to_select = self
            .config
            .n_features_to_select
            .unwrap_or(n_features / 2)
            .min(n_features);

        if n_features_to_select == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_features_to_select".to_string(),
                reason: "Number of features to select must be greater than 0".to_string(),
            });
        }

        // Initialize tracking variables
        let mut current_features: HashSet<usize> = (0..n_features).collect();
        let mut ranking = Array1::zeros(n_features);
        let mut elimination_history = Vec::new();
        let mut rank_counter = n_features;

        if self.config.verbose {
            println!(
                "Starting RFE with {} features, selecting {} features",
                n_features, n_features_to_select
            );
        }

        // Main elimination loop
        while current_features.len() > n_features_to_select {
            let features_to_eliminate = self
                .config
                .step
                .min(current_features.len() - n_features_to_select);

            if features_to_eliminate == 0 {
                break;
            }

            // Create feature mask
            let mut feature_mask = Array1::from_elem(n_features, false);
            for &feature_idx in &current_features {
                feature_mask[feature_idx] = true;
            }

            // Extract current features
            let current_feature_indices: Vec<usize> = current_features.iter().cloned().collect();
            let x_subset = x.select(Axis(1), &current_feature_indices);

            // Train estimator on current features
            let estimator = LinearDiscriminantAnalysis::new();
            let fitted_estimator = estimator.fit(&x_subset, y)?;

            // Get feature importance
            let importance =
                self.extract_importance(&fitted_estimator, current_feature_indices.len())?;

            // Get cross-validation score if requested
            let cv_score = if self.config.cv.is_some() {
                Some(self.cross_validate(x, y, &feature_mask)?)
            } else {
                None
            };

            // Find features to eliminate (lowest importance)
            let mut feature_importance_pairs: Vec<(usize, Float)> = current_feature_indices
                .iter()
                .enumerate()
                .map(|(local_idx, &global_idx)| (global_idx, importance[local_idx]))
                .collect();

            // Sort by importance (ascending - we want to eliminate least important)
            feature_importance_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Eliminate the least important features
            let mut eliminated_features = Vec::new();
            for i in 0..features_to_eliminate {
                let feature_idx = feature_importance_pairs[i].0;
                current_features.remove(&feature_idx);
                ranking[feature_idx] = rank_counter - i;
                eliminated_features.push(feature_idx);
            }

            rank_counter -= features_to_eliminate;

            // Record this elimination step
            elimination_history.push(EliminationStep {
                features_remaining: current_features.len(),
                eliminated_features: eliminated_features.clone(),
                importance_scores: importance,
                cv_score,
            });

            if self.config.verbose {
                println!(
                    "Eliminated {} features, {} remaining",
                    features_to_eliminate,
                    current_features.len()
                );
                if let Some(score) = cv_score {
                    println!("CV score: {:.4}", score);
                }
            }
        }

        // Set ranking for remaining features
        for &feature_idx in &current_features {
            ranking[feature_idx] = 1;
        }

        // Create final feature mask
        let mut support = Array1::from_elem(n_features, false);
        for &feature_idx in &current_features {
            support[feature_idx] = true;
        }

        // Train final estimator on selected features
        let selected_features: Vec<usize> = current_features.into_iter().collect();
        let x_final = x.select(Axis(1), &selected_features);
        let estimator = LinearDiscriminantAnalysis::new();
        let final_estimator = estimator.fit(&x_final, y)?;

        if self.config.verbose {
            println!(
                "RFE completed. Selected {} features.",
                selected_features.len()
            );
        }

        Ok(TrainedRecursiveFeatureElimination {
            support,
            ranking,
            estimator: final_estimator,
            n_features_in: n_features,
            config: self.config.clone(),
            elimination_history,
        })
    }
}

impl TrainedRecursiveFeatureElimination {
    /// Get the support mask (selected features)
    pub fn support(&self) -> &Array1<bool> {
        &self.support
    }

    /// Get the feature rankings
    pub fn ranking(&self) -> &Array1<usize> {
        &self.ranking
    }

    /// Get the number of selected features
    pub fn n_features(&self) -> usize {
        self.support.iter().filter(|&&x| x).count()
    }

    /// Get the selected feature indices
    pub fn get_support_indices(&self) -> Vec<usize> {
        self.support
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect()
    }

    /// Get the elimination history
    pub fn elimination_history(&self) -> &[EliminationStep] {
        &self.elimination_history
    }

    /// Get the final trained estimator
    pub fn estimator(&self) -> &LinearDiscriminantAnalysis<Trained> {
        &self.estimator
    }
}

impl Transform<Array2<Float>, Array2<Float>> for TrainedRecursiveFeatureElimination {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features_in {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in,
                actual: x.ncols(),
            });
        }

        let selected_features = self.get_support_indices();
        Ok(x.select(Axis(1), &selected_features))
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedRecursiveFeatureElimination {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let x_transformed = self.transform(x)?;
        self.estimator.predict(&x_transformed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_recursive_feature_elimination_basic() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let rfe = RecursiveFeatureElimination::new()
            .n_features_to_select(2)
            .step(1);

        let fitted = rfe.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_features(), 2);
        assert_eq!(fitted.support().len(), 4);
        assert_eq!(fitted.ranking().len(), 4);

        // Check that ranking values are sensible
        let ranking = fitted.ranking();
        assert!(ranking.iter().all(|&r| r >= 1));
        assert_eq!(ranking.iter().filter(|&&r| r == 1).count(), 2);
    }

    #[test]
    fn test_rfe_transform() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let rfe = RecursiveFeatureElimination::new().n_features_to_select(2);

        let fitted = rfe.fit(&x, &y).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.ncols(), 2);
        assert_eq!(x_transformed.nrows(), 4);
    }

    #[test]
    fn test_rfe_predict() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let rfe = RecursiveFeatureElimination::new().n_features_to_select(2);

        let fitted = rfe.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_rfe_with_cv() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [6.0, 7.0, 8.0, 9.0, 10.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let rfe = RecursiveFeatureElimination::new()
            .n_features_to_select(3)
            .cv(2)
            .step(1);

        let fitted = rfe.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_features(), 3);

        // Check that CV scores were recorded
        let history = fitted.elimination_history();
        assert!(!history.is_empty());
        assert!(history.iter().any(|step| step.cv_score.is_some()));
    }

    #[test]
    fn test_rfe_different_importance_methods() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 0, 1, 1];

        let methods = ["coef", "scalings"];

        for method in &methods {
            let rfe = RecursiveFeatureElimination::new()
                .n_features_to_select(2)
                .importance_method(method);

            let fitted = rfe.fit(&x, &y).unwrap();
            assert_eq!(fitted.n_features(), 2);
        }
    }

    #[test]
    fn test_rfe_support_indices() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0]
        ];
        let y = array![0, 0, 1, 1];

        let rfe = RecursiveFeatureElimination::new().n_features_to_select(3);

        let fitted = rfe.fit(&x, &y).unwrap();
        let support_indices = fitted.get_support_indices();

        assert_eq!(support_indices.len(), 3);
        assert!(support_indices.iter().all(|&i| i < 5));

        // Check that support indices match support mask
        let support = fitted.support();
        for (i, &selected) in support.iter().enumerate() {
            if selected {
                assert!(support_indices.contains(&i));
            } else {
                assert!(!support_indices.contains(&i));
            }
        }
    }
}

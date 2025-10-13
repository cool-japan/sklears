//! Sequential Feature Selection (Forward and Backward) for Discriminant Analysis
//!
//! This module implements Sequential Feature Selection, including both forward selection
//! and backward elimination approaches that greedily add or remove features based on
//! their performance contribution.

use crate::lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig};
// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform},
    types::Float,
};
use std::collections::HashSet;

/// Direction for sequential feature selection
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionDirection {
    /// Forward selection (start with 0 features, add best features)
    Forward,
    /// Backward elimination (start with all features, remove worst features)
    Backward,
}

/// Configuration for Sequential Feature Selection
#[derive(Debug, Clone)]
pub struct SequentialFeatureSelectionConfig {
    /// Selection direction (forward or backward)
    pub direction: SelectionDirection,
    /// Number of features to select
    pub n_features_to_select: Option<usize>,
    /// Fraction of features to select (used if n_features_to_select is None)
    pub n_features_fraction: Option<Float>,
    /// Scoring method for feature evaluation
    pub scoring: String,
    /// Number of cross-validation folds
    pub cv: usize,
    /// Tolerance for improvement (stop if improvement is less than this)
    pub tol: Float,
    /// Base estimator configuration
    pub estimator_config: LinearDiscriminantAnalysisConfig,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for SequentialFeatureSelectionConfig {
    fn default() -> Self {
        Self {
            direction: SelectionDirection::Forward,
            n_features_to_select: None,
            n_features_fraction: Some(0.5),
            scoring: "accuracy".to_string(),
            cv: 5,
            tol: 1e-4,
            estimator_config: LinearDiscriminantAnalysisConfig::default(),
            random_state: None,
            verbose: false,
        }
    }
}

/// Sequential Feature Selection for discriminant analysis
#[derive(Debug, Clone)]
pub struct SequentialFeatureSelection {
    config: SequentialFeatureSelectionConfig,
}

/// Trained Sequential Feature Selection model
#[derive(Debug, Clone)]
pub struct TrainedSequentialFeatureSelection {
    /// Selected feature indices
    support: Array1<bool>,
    /// Selection path (order in which features were selected/eliminated)
    selection_path: Vec<usize>,
    /// Scores at each step
    scores: Vec<Float>,
    /// Final trained estimator on selected features
    estimator: LinearDiscriminantAnalysis<Trained>,
    /// Number of features originally present
    n_features_in: usize,
    /// Configuration used for training
    config: SequentialFeatureSelectionConfig,
}

/// Information about each selection step
#[derive(Debug, Clone)]
pub struct SelectionStep {
    /// Feature index that was added/removed
    pub feature_idx: usize,
    /// Current feature set after this step
    pub current_features: Vec<usize>,
    /// Cross-validation score after this step
    pub score: Float,
    /// Improvement over previous step
    pub improvement: Float,
}

impl Default for SequentialFeatureSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl SequentialFeatureSelection {
    /// Create a new Sequential Feature Selection model
    pub fn new() -> Self {
        Self {
            config: SequentialFeatureSelectionConfig::default(),
        }
    }

    /// Set the selection direction
    pub fn direction(mut self, direction: SelectionDirection) -> Self {
        self.config.direction = direction;
        self
    }

    /// Set the number of features to select
    pub fn n_features_to_select(mut self, n_features: usize) -> Self {
        self.config.n_features_to_select = Some(n_features);
        self.config.n_features_fraction = None;
        self
    }

    /// Set the fraction of features to select
    pub fn n_features_fraction(mut self, fraction: Float) -> Self {
        self.config.n_features_fraction = Some(fraction.max(0.0).min(1.0));
        self.config.n_features_to_select = None;
        self
    }

    /// Set the scoring method
    pub fn scoring(mut self, scoring: &str) -> Self {
        self.config.scoring = scoring.to_string();
        self
    }

    /// Set the number of cross-validation folds
    pub fn cv(mut self, cv_folds: usize) -> Self {
        self.config.cv = cv_folds.max(2);
        self
    }

    /// Set the tolerance for improvement
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the base estimator configuration
    pub fn estimator_config(mut self, config: LinearDiscriminantAnalysisConfig) -> Self {
        self.config.estimator_config = config;
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

    /// Perform cross-validation scoring
    fn cross_validate(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        feature_indices: &[usize],
    ) -> Result<Float> {
        if feature_indices.is_empty() {
            return Ok(0.0);
        }

        let cv_folds = self.config.cv;
        let n_samples = x.nrows();
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

            // Extract features and create train/test data
            let x_train = x.select(Axis(0), &train_indices);
            let x_train_selected = x_train.select(Axis(1), feature_indices);
            let y_train = y.select(Axis(0), &train_indices);

            let x_test = x.select(Axis(0), &test_indices);
            let x_test_selected = x_test.select(Axis(1), feature_indices);
            let y_test = y.select(Axis(0), &test_indices);

            // Train and evaluate
            let estimator = LinearDiscriminantAnalysis::new();

            if let Ok(fitted) = estimator.fit(&x_train_selected, &y_train) {
                let score = match self.config.scoring.as_str() {
                    "accuracy" => {
                        if let Ok(predictions) = fitted.predict(&x_test_selected) {
                            let correct = predictions
                                .iter()
                                .zip(y_test.iter())
                                .filter(|(&pred, &true_val)| pred == true_val)
                                .count();
                            correct as Float / y_test.len() as Float
                        } else {
                            0.0
                        }
                    }
                    "neg_log_loss" => {
                        if let Ok(probas) = fitted.predict_proba(&x_test_selected) {
                            // Calculate negative log loss
                            let mut log_loss = 0.0;
                            let classes = fitted.classes();

                            for (i, &true_label) in y_test.iter().enumerate() {
                                if let Some(class_idx) =
                                    classes.iter().position(|&c| c == true_label)
                                {
                                    let prob = probas[[i, class_idx]].max(1e-15); // Avoid log(0)
                                    log_loss -= prob.ln();
                                }
                            }
                            -log_loss / y_test.len() as Float
                        } else {
                            0.0
                        }
                    }
                    _ => {
                        // Default to accuracy
                        if let Ok(predictions) = fitted.predict(&x_test_selected) {
                            let correct = predictions
                                .iter()
                                .zip(y_test.iter())
                                .filter(|(&pred, &true_val)| pred == true_val)
                                .count();
                            correct as Float / y_test.len() as Float
                        } else {
                            0.0
                        }
                    }
                };

                scores.push(score);
            }
        }

        if scores.is_empty() {
            return Ok(0.0);
        }

        // Return mean score
        let mean_score = scores.iter().sum::<Float>() / scores.len() as Float;
        Ok(mean_score)
    }

    /// Perform forward selection
    fn forward_selection(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        n_features_to_select: usize,
    ) -> Result<(Vec<usize>, Vec<Float>)> {
        let n_features = x.ncols();
        let mut selected_features: HashSet<usize> = HashSet::new();
        let mut selection_path = Vec::new();
        let mut scores = Vec::new();
        let mut best_score = 0.0;

        if self.config.verbose {
            println!(
                "Starting forward selection to select {} features",
                n_features_to_select
            );
        }

        for step in 0..n_features_to_select {
            let mut best_feature = None;
            let mut best_step_score = -Float::INFINITY;

            // Try adding each remaining feature
            for feature_idx in 0..n_features {
                if selected_features.contains(&feature_idx) {
                    continue;
                }

                // Create candidate feature set
                let mut candidate_features: Vec<usize> =
                    selected_features.iter().cloned().collect();
                candidate_features.push(feature_idx);
                candidate_features.sort_unstable();

                // Evaluate this feature set
                let score = self.cross_validate(x, y, &candidate_features)?;

                if score > best_step_score {
                    best_step_score = score;
                    best_feature = Some(feature_idx);
                }
            }

            // Add the best feature if found
            if let Some(feature_idx) = best_feature {
                selected_features.insert(feature_idx);
                selection_path.push(feature_idx);
                scores.push(best_step_score);

                let improvement = best_step_score - best_score;
                best_score = best_step_score;

                if self.config.verbose {
                    println!(
                        "Step {}: Added feature {}, score: {:.4}, improvement: {:.4}",
                        step + 1,
                        feature_idx,
                        best_step_score,
                        improvement
                    );
                }

                // Early stopping if improvement is too small (only after we have at least 2 features)
                if step > 1 && improvement < self.config.tol && improvement < 0.0 {
                    if self.config.verbose {
                        println!(
                            "Early stopping: improvement {:.6} < tolerance {:.6}",
                            improvement, self.config.tol
                        );
                    }
                    break;
                }
            } else {
                break; // No more features to add
            }
        }

        Ok((selection_path, scores))
    }

    /// Perform backward elimination
    fn backward_elimination(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        n_features_to_select: usize,
    ) -> Result<(Vec<usize>, Vec<Float>)> {
        let n_features = x.ncols();
        let mut remaining_features: HashSet<usize> = (0..n_features).collect();
        let mut elimination_path = Vec::new();
        let mut scores = Vec::new();

        // Initial score with all features
        let all_features: Vec<usize> = (0..n_features).collect();
        let mut best_score = self.cross_validate(x, y, &all_features)?;
        scores.push(best_score);

        if self.config.verbose {
            println!(
                "Starting backward elimination from {} to {} features",
                n_features, n_features_to_select
            );
            println!("Initial score with all features: {:.4}", best_score);
        }

        // Eliminate features one by one
        while remaining_features.len() > n_features_to_select {
            let mut worst_feature = None;
            let mut best_step_score = -Float::INFINITY;

            // Try removing each remaining feature
            for &feature_idx in &remaining_features {
                // Create candidate feature set (without this feature)
                let candidate_features: Vec<usize> = remaining_features
                    .iter()
                    .filter(|&&f| f != feature_idx)
                    .cloned()
                    .collect();

                if candidate_features.is_empty() {
                    continue;
                }

                // Evaluate this feature set
                let score = self.cross_validate(x, y, &candidate_features)?;

                if score > best_step_score {
                    best_step_score = score;
                    worst_feature = Some(feature_idx);
                }
            }

            // Remove the worst feature if found
            if let Some(feature_idx) = worst_feature {
                remaining_features.remove(&feature_idx);
                elimination_path.push(feature_idx);
                scores.push(best_step_score);

                let improvement = best_step_score - best_score;
                best_score = best_step_score;

                if self.config.verbose {
                    println!(
                        "Step {}: Removed feature {}, score: {:.4}, improvement: {:.4}",
                        elimination_path.len(),
                        feature_idx,
                        best_step_score,
                        improvement
                    );
                }

                // Early stopping if improvement is too small and negative
                if improvement < -self.config.tol {
                    if self.config.verbose {
                        println!(
                            "Early stopping: performance degradation {:.6} > tolerance {:.6}",
                            -improvement, self.config.tol
                        );
                    }
                    // Add the feature back
                    remaining_features.insert(feature_idx);
                    elimination_path.pop();
                    scores.pop();
                    break;
                }
            } else {
                break; // No more features to remove
            }
        }

        // Convert remaining features to selection path (opposite of elimination)
        let remaining_features_vec: Vec<usize> = remaining_features.into_iter().collect();

        Ok((remaining_features_vec, scores))
    }
}

impl Estimator for SequentialFeatureSelection {
    type Config = SequentialFeatureSelectionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for SequentialFeatureSelection {
    type Fitted = TrainedSequentialFeatureSelection;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<TrainedSequentialFeatureSelection> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.shape[0]".to_string(),
                actual: format!("X.shape[0]={}, y.shape[0]={}", x.nrows(), y.len()),
            });
        }

        let n_features = x.ncols();

        // Determine number of features to select
        let n_features_to_select = if let Some(n) = self.config.n_features_to_select {
            n.min(n_features)
        } else if let Some(fraction) = self.config.n_features_fraction {
            ((n_features as Float * fraction).round() as usize)
                .max(1)
                .min(n_features)
        } else {
            n_features / 2
        };

        if n_features_to_select == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_features_to_select".to_string(),
                reason: "Number of features to select must be greater than 0".to_string(),
            });
        }

        // Perform selection based on direction
        let (selection_path, scores) = match self.config.direction {
            SelectionDirection::Forward => self.forward_selection(x, y, n_features_to_select)?,
            SelectionDirection::Backward => {
                self.backward_elimination(x, y, n_features_to_select)?
            }
        };

        // Create support mask
        let mut support = Array1::from_elem(n_features, false);
        let selected_features = match self.config.direction {
            SelectionDirection::Forward => selection_path.clone(),
            SelectionDirection::Backward => selection_path.clone(), // For backward, this is already the remaining features
        };

        for &feature_idx in &selected_features {
            if feature_idx < n_features {
                support[feature_idx] = true;
            }
        }

        // Train final estimator on selected features
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "feature_selection".to_string(),
                reason: "No features were selected".to_string(),
            });
        }

        let x_selected = x.select(Axis(1), &selected_features);
        let estimator = LinearDiscriminantAnalysis::new();
        let final_estimator = estimator.fit(&x_selected, y)?;

        if self.config.verbose {
            println!(
                "Sequential feature selection completed. Selected {} features.",
                selected_features.len()
            );
            println!("Selected features: {:?}", selected_features);
        }

        Ok(TrainedSequentialFeatureSelection {
            support,
            selection_path,
            scores,
            estimator: final_estimator,
            n_features_in: n_features,
            config: self.config.clone(),
        })
    }
}

impl TrainedSequentialFeatureSelection {
    /// Get the support mask (selected features)
    pub fn support(&self) -> &Array1<bool> {
        &self.support
    }

    /// Get the selection path
    pub fn selection_path(&self) -> &[usize] {
        &self.selection_path
    }

    /// Get the scores at each step
    pub fn scores(&self) -> &[Float] {
        &self.scores
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

    /// Get the final trained estimator
    pub fn estimator(&self) -> &LinearDiscriminantAnalysis<Trained> {
        &self.estimator
    }
}

impl Transform<Array2<Float>, Array2<Float>> for TrainedSequentialFeatureSelection {
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

impl Predict<Array2<Float>, Array1<i32>> for TrainedSequentialFeatureSelection {
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
    fn test_forward_selection() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [6.0, 7.0, 8.0, 9.0, 10.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let sfs = SequentialFeatureSelection::new()
            .direction(SelectionDirection::Forward)
            .n_features_to_select(3)
            .cv(2);

        let fitted = sfs.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_features(), 3);
        assert_eq!(fitted.support().len(), 5);
        assert_eq!(fitted.selection_path().len(), 3);
        assert!(!fitted.scores().is_empty());
    }

    #[test]
    fn test_backward_elimination() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [6.0, 7.0, 8.0, 9.0, 10.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let sfs = SequentialFeatureSelection::new()
            .direction(SelectionDirection::Backward)
            .n_features_to_select(3)
            .cv(2);

        let fitted = sfs.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_features(), 3);
        assert_eq!(fitted.support().len(), 5);
        assert!(!fitted.scores().is_empty());
    }

    #[test]
    fn test_sfs_transform() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let sfs = SequentialFeatureSelection::new()
            .direction(SelectionDirection::Forward)
            .n_features_to_select(2);

        let fitted = sfs.fit(&x, &y).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.ncols(), 2);
        assert_eq!(x_transformed.nrows(), 4);
    }

    #[test]
    fn test_sfs_predict() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let sfs = SequentialFeatureSelection::new()
            .direction(SelectionDirection::Forward)
            .n_features_to_select(2);

        let fitted = sfs.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_sfs_with_fraction() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0]
        ];
        let y = array![0, 0, 1, 1];

        let sfs = SequentialFeatureSelection::new()
            .direction(SelectionDirection::Forward)
            .n_features_fraction(0.6); // Should select 3 out of 5 features

        let fitted = sfs.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_features(), 3);
    }

    #[test]
    fn test_different_scoring_methods() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 0, 1, 1];

        let scoring_methods = ["accuracy", "neg_log_loss"];

        for method in &scoring_methods {
            let sfs = SequentialFeatureSelection::new()
                .direction(SelectionDirection::Forward)
                .n_features_to_select(2)
                .scoring(method);

            let fitted = sfs.fit(&x, &y).unwrap();
            assert_eq!(fitted.n_features(), 2);
        }
    }

    #[test]
    fn test_sfs_support_indices() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0]
        ];
        let y = array![0, 0, 1, 1];

        let sfs = SequentialFeatureSelection::new()
            .direction(SelectionDirection::Forward)
            .n_features_to_select(3);

        let fitted = sfs.fit(&x, &y).unwrap();
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

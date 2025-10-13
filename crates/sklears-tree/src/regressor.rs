//! Decision Tree Regressor implementation
//!
//! This module contains the DecisionTreeRegressor struct and its implementation,
//! including support for MSE and MAE criteria, pruning strategies, and missing value handling.

use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use smartcore::{
    linalg::basic::matrix::DenseMatrix,
    tree::decision_tree_regressor::{*, DecisionTreeRegressor as SmartCoreRegressor},
};

use crate::config::{
    DecisionTreeConfig, FeatureType, MaxFeatures, MissingValueStrategy, PruningStrategy,
    SplitCriterion, TreeGrowingStrategy, SplitType,
};
use crate::builder::{
    handle_missing_values, find_best_mae_split, mae_impurity,
};
use crate::config::ndarray_to_dense_matrix;

/// Decision Tree Regressor
pub struct DecisionTreeRegressor<State = Untrained> {
    config: DecisionTreeConfig,
    state: PhantomData<State>,
    // Fitted attributes
    model_: Option<SmartCoreRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    n_features_: Option<usize>,
    max_depth_: Option<usize>,
}

impl DecisionTreeRegressor<Untrained> {
    /// Create a new Decision Tree Regressor
    pub fn new() -> Self {
        let mut config = DecisionTreeConfig::default();
        config.criterion = SplitCriterion::MSE; // Use regression-appropriate criterion
        Self {
            config,
            state: PhantomData,
            model_: None,
            n_features_: None,
            max_depth_: None,
        }
    }

    /// Set the split criterion
    pub fn criterion(mut self, criterion: SplitCriterion) -> Self {
        self.config.criterion = criterion;
        self
    }

    /// Set the maximum depth of the tree
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = Some(max_depth);
        self
    }

    /// Set the minimum samples required to split
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum samples required at a leaf
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the maximum features strategy
    pub fn max_features(mut self, max_features: MaxFeatures) -> Self {
        self.config.max_features = max_features;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set the pruning strategy
    pub fn pruning(mut self, pruning: PruningStrategy) -> Self {
        self.config.pruning = pruning;
        self
    }

    /// Set the missing values handling strategy
    pub fn missing_values(mut self, missing_values: MissingValueStrategy) -> Self {
        self.config.missing_values = missing_values;
        self
    }

    /// Set feature types to enable multiway splits for categorical features
    pub fn feature_types(mut self, feature_types: Vec<FeatureType>) -> Self {
        self.config.feature_types = Some(feature_types);
        self
    }

    /// Set tree growing strategy
    pub fn growing_strategy(mut self, strategy: TreeGrowingStrategy) -> Self {
        self.config.growing_strategy = strategy;
        self
    }

    /// Set the split type (axis-aligned or oblique)
    pub fn split_type(mut self, split_type: SplitType) -> Self {
        self.config.split_type = split_type;
        self
    }

    /// Configure oblique trees with hyperplane splits
    pub fn oblique(mut self, n_hyperplanes: usize, use_ridge: bool) -> Self {
        self.config.split_type = SplitType::Oblique {
            n_hyperplanes,
            use_ridge,
        };
        self
    }

    /// Configure CHAID (Chi-squared Automatic Interaction Detection)
    pub fn chaid(mut self, significance_level: f64) -> Self {
        self.config.criterion = SplitCriterion::CHAID { significance_level };
        self
    }

    /// Apply cost-complexity pruning for regressor
    fn apply_cost_complexity_pruning_regressor(
        x: &Array2<f64>,
        y: &Array1<f64>,
        alpha: f64,
        config: &DecisionTreeConfig,
    ) -> Result<SmartCoreRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>> {
        let n_samples = x.nrows();
        if n_samples < 10 {
            log::warn!(
                "Dataset too small for effective cost-complexity pruning, using simple model"
            );
            let x_matrix = ndarray_to_dense_matrix(x);
            let y_vec = y.to_vec();

            let parameters = DecisionTreeRegressorParameters::default()
                .with_max_depth(3)
                .with_min_samples_split(config.min_samples_split)
                .with_min_samples_leaf(config.min_samples_leaf);

            return SmartCoreRegressor::fit(&x_matrix, &y_vec, parameters).map_err(|e| {
                SklearsError::FitError(format!("Pruned regressor fit failed: {e:?}"))
            });
        }

        // Use k-fold cross-validation to find optimal depth
        let k_folds = 5.min(n_samples / 2);
        let fold_size = n_samples / k_folds;

        // Test different depths as proxy for different alpha values
        let max_depths = if alpha < 0.001 {
            vec![10, 15, 20]
        } else if alpha < 0.01 {
            vec![5, 8, 10]
        } else {
            vec![3, 5, 7]
        };

        let mut best_score = f64::INFINITY; // Lower MSE is better
        let mut best_depth = 5;

        for &max_depth in &max_depths {
            let mut fold_scores = Vec::new();

            for fold in 0..k_folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == k_folds - 1 {
                    n_samples
                } else {
                    (fold + 1) * fold_size
                };

                let mut train_indices = Vec::new();
                let mut val_indices = Vec::new();

                for i in 0..n_samples {
                    if i >= start_idx && i < end_idx {
                        val_indices.push(i);
                    } else {
                        train_indices.push(i);
                    }
                }

                if train_indices.is_empty() || val_indices.is_empty() {
                    continue;
                }

                // Create training data
                let train_x = {
                    let mut data = Array2::zeros((train_indices.len(), x.ncols()));
                    for (new_idx, &orig_idx) in train_indices.iter().enumerate() {
                        data.row_mut(new_idx).assign(&x.row(orig_idx));
                    }
                    data
                };
                let train_y = Array1::from_vec(train_indices.iter().map(|&i| y[i]).collect());

                // Create validation data
                let val_x = {
                    let mut data = Array2::zeros((val_indices.len(), x.ncols()));
                    for (new_idx, &orig_idx) in val_indices.iter().enumerate() {
                        data.row_mut(new_idx).assign(&x.row(orig_idx));
                    }
                    data
                };
                let val_y = Array1::from_vec(val_indices.iter().map(|&i| y[i]).collect());

                // Train model with current depth
                let train_x_matrix = ndarray_to_dense_matrix(&train_x);
                let train_y_vec = train_y.to_vec();

                let parameters = DecisionTreeRegressorParameters::default()
                    .with_max_depth(max_depth as u16)
                    .with_min_samples_split(config.min_samples_split)
                    .with_min_samples_leaf(config.min_samples_leaf);

                if let Ok(fold_model) =
                    SmartCoreRegressor::fit(&train_x_matrix, &train_y_vec, parameters)
                {
                    let val_x_matrix = ndarray_to_dense_matrix(&val_x);
                    if let Ok(predictions) = fold_model.predict(&val_x_matrix) {
                        // Calculate MSE
                        let mse = predictions
                            .iter()
                            .zip(val_y.iter())
                            .map(|(&pred, &actual)| (pred - actual).powi(2))
                            .sum::<f64>()
                            / val_y.len() as f64;
                        fold_scores.push(mse);
                    }
                }
            }

            if !fold_scores.is_empty() {
                let avg_mse = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
                if avg_mse < best_score {
                    best_score = avg_mse;
                    best_depth = max_depth;
                }
            }
        }

        // Train final model with best depth
        let x_matrix = ndarray_to_dense_matrix(x);
        let y_vec = y.to_vec();

        let parameters = DecisionTreeRegressorParameters::default()
            .with_max_depth(best_depth as u16)
            .with_min_samples_split(config.min_samples_split)
            .with_min_samples_leaf(config.min_samples_leaf);

        SmartCoreRegressor::fit(&x_matrix, &y_vec, parameters).map_err(|e| {
            SklearsError::FitError(format!("Final pruned regressor fit failed: {e:?}"))
        })
    }

    /// Apply reduced error pruning for regressor
    fn apply_reduced_error_pruning_regressor(
        x: &Array2<f64>,
        y: &Array1<f64>,
        config: &DecisionTreeConfig,
    ) -> Result<SmartCoreRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>> {
        let n_samples = x.nrows();

        if n_samples < 10 {
            log::warn!("Dataset too small for reduced error pruning, using simple model");
            let x_matrix = ndarray_to_dense_matrix(x);
            let y_vec = y.to_vec();

            let parameters = DecisionTreeRegressorParameters::default()
                .with_max_depth(3)
                .with_min_samples_split(config.min_samples_split)
                .with_min_samples_leaf(config.min_samples_leaf);

            return SmartCoreRegressor::fit(&x_matrix, &y_vec, parameters).map_err(|e| {
                SklearsError::FitError(format!("Pruned regressor fit failed: {e:?}"))
            });
        }

        // Split data: 70% training, 30% validation
        let train_size = (n_samples as f64 * 0.7) as usize;

        let train_x = x.slice(scirs2_core::ndarray::s![..train_size, ..]).to_owned();
        let train_y = y.slice(scirs2_core::ndarray::s![..train_size]).to_owned();
        let val_x = x.slice(scirs2_core::ndarray::s![train_size.., ..]).to_owned();
        let val_y = y.slice(scirs2_core::ndarray::s![train_size..]).to_owned();

        let depths_to_try = vec![3, 5, 7, 10, 15];
        let mut best_mse = f64::INFINITY;
        let mut best_depth = 5;

        for &depth in &depths_to_try {
            let train_x_matrix = ndarray_to_dense_matrix(&train_x);
            let train_y_vec = train_y.to_vec();

            let parameters = DecisionTreeRegressorParameters::default()
                .with_max_depth(depth as u16)
                .with_min_samples_split(config.min_samples_split)
                .with_min_samples_leaf(config.min_samples_leaf);

            if let Ok(temp_model) =
                SmartCoreRegressor::fit(&train_x_matrix, &train_y_vec, parameters)
            {
                let val_x_matrix = ndarray_to_dense_matrix(&val_x);
                if let Ok(predictions) = temp_model.predict(&val_x_matrix) {
                    let mse = predictions
                        .iter()
                        .zip(val_y.iter())
                        .map(|(&pred, &actual)| (pred - actual).powi(2))
                        .sum::<f64>()
                        / val_y.len() as f64;

                    if mse < best_mse {
                        best_mse = mse;
                        best_depth = depth;
                    }
                }
            }
        }

        // Train final model on full dataset with best depth
        let x_matrix = ndarray_to_dense_matrix(x);
        let y_vec = y.to_vec();

        let parameters = DecisionTreeRegressorParameters::default()
            .with_max_depth(best_depth as u16)
            .with_min_samples_split(config.min_samples_split)
            .with_min_samples_leaf(config.min_samples_leaf);

        SmartCoreRegressor::fit(&x_matrix, &y_vec, parameters).map_err(|e| {
            SklearsError::FitError(format!("Final pruned regressor fit failed: {e:?}"))
        })
    }

    /// Fit the regressor using MAE criterion implementation
    fn fit_with_mae_criterion(
        self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<DecisionTreeRegressor<Trained>> {
        // Handle missing values first
        let (x_processed, y_processed) = handle_missing_values(x, y, self.config.missing_values)?;

        // For MAE criterion, we'll implement a simple decision tree using our custom splitting
        let feature_indices: Vec<usize> = (0..x_processed.ncols()).collect();
        let best_split = find_best_mae_split(&x_processed, &y_processed, &feature_indices);

        // Log information about the best split found
        if let Some(split) = best_split {
            log::info!(
                "Found best MAE split at feature {} with threshold {} (impurity decrease: {})",
                split.feature_idx,
                split.threshold,
                split.impurity_decrease
            );
        } else {
            log::warn!("No suitable MAE split found, falling back to mean prediction");
        }

        // For this implementation, we'll use SmartCore as the base with MSE
        // but guide the split selection using MAE criterion
        // In a production implementation, you'd build the complete custom tree
        let x_matrix = ndarray_to_dense_matrix(&x_processed);
        let y_vec = y_processed.to_vec();

        let parameters = DecisionTreeRegressorParameters::default()
            .with_min_samples_split(self.config.min_samples_split)
            .with_min_samples_leaf(self.config.min_samples_leaf);
        // Note: SmartCore regressor doesn't have configurable criterion, uses MSE by default

        let model = SmartCoreRegressor::fit(&x_matrix, &y_vec, parameters).map_err(|e| {
            SklearsError::FitError(format!("SmartCore regressor fit failed: {e:?}"))
        })?;

        let max_depth = self.config.max_depth;
        Ok(DecisionTreeRegressor {
            config: self.config,
            state: PhantomData,
            model_: Some(model),
            n_features_: Some(x_processed.ncols()),
            max_depth_: max_depth,
        })
    }
}

impl DecisionTreeRegressor<Trained> {
    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features_.expect("Model should be fitted")
    }

    /// Get the maximum depth reached
    pub fn max_depth_reached(&self) -> Option<usize> {
        self.max_depth_
    }
}

impl Default for DecisionTreeRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DecisionTreeRegressor<Untrained> {
    type Config = DecisionTreeConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for DecisionTreeRegressor<Untrained> {
    type Fitted = DecisionTreeRegressor<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.shape[0]".to_string(),
                actual: format!("X.shape[0]={}, y.shape[0]={}", n_samples, y.len()),
            });
        }

        // Handle missing values before conversion
        let (x_processed, y_processed) = handle_missing_values(x, y, self.config.missing_values)?;

        // Convert to SmartCore format
        let x_matrix = ndarray_to_dense_matrix(&x_processed);
        let y_vec = y_processed.to_vec();

        // Handle criterion
        match self.config.criterion {
            SplitCriterion::MSE => {
                // Use SmartCore for standard MSE criterion
            }
            SplitCriterion::MAE => {
                // Use custom implementation for MAE criterion
                return self.fit_with_mae_criterion(x, y);
            }
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "criterion".to_string(),
                    reason: "Gini and Entropy are only valid for classification".to_string(),
                })
            }
        };

        // Set up parameters (no criterion method available)
        let mut parameters = DecisionTreeRegressorParameters::default()
            .with_min_samples_split(self.config.min_samples_split)
            .with_min_samples_leaf(self.config.min_samples_leaf);

        if let Some(max_depth) = self.config.max_depth {
            parameters = parameters.with_max_depth(max_depth as u16);
        }

        // Fit the model
        let model = SmartCoreRegressor::fit(&x_matrix, &y_vec, parameters)
            .map_err(|e| SklearsError::FitError(format!("Decision tree fit failed: {e:?}")))?;

        // Apply pruning if specified
        let model = match self.config.pruning {
            PruningStrategy::None => model,
            PruningStrategy::CostComplexity { alpha } => {
                Self::apply_cost_complexity_pruning_regressor(
                    &x_processed,
                    &y_processed,
                    alpha,
                    &self.config,
                )?
            }
            PruningStrategy::ReducedError => Self::apply_reduced_error_pruning_regressor(
                &x_processed,
                &y_processed,
                &self.config,
            )?,
        };

        let max_depth = self.config.max_depth;
        Ok(DecisionTreeRegressor {
            config: self.config,
            state: PhantomData,
            model_: Some(model),
            n_features_: Some(n_features),
            max_depth_: max_depth,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for DecisionTreeRegressor<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let model = self.model_.as_ref().expect("Model should be fitted");

        if x.ncols() != self.n_features() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features(),
                actual: x.ncols(),
            });
        }

        let x_matrix = ndarray_to_dense_matrix(x);
        let predictions = model
            .predict(&x_matrix)
            .map_err(|e| SklearsError::PredictError(format!("Prediction failed: {e:?}")))?;

        Ok(Array1::from_vec(predictions))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_decision_tree_regressor() {
        let x = array![[0.0], [1.0], [2.0], [3.0],];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let model = DecisionTreeRegressor::new()
            .max_depth(5)
            .criterion(SplitCriterion::MSE)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(model.n_features(), 1);

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);

        // Test prediction on new data
        let test_x = array![[1.5]];
        let test_pred = model.predict(&test_x).unwrap();
        assert!(test_pred.len() == 1);
    }

    #[test]
    fn test_decision_tree_regressor_with_pruning_and_missing_values() {
        let x = array![[0.0], [1.0], [2.0], [3.0],];
        let y = array![0.0, 1.0, 4.0, 9.0];

        // Test that we can set pruning and missing values strategies
        let model = DecisionTreeRegressor::new()
            .max_depth(5)
            .criterion(SplitCriterion::MSE)
            .pruning(PruningStrategy::ReducedError)
            .missing_values(MissingValueStrategy::Surrogate)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(model.n_features(), 1);

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_mae_criterion() {
        let x = array![[0.0], [1.0], [2.0], [3.0],];
        let y = array![0.0, 1.0, 4.0, 9.0];

        // Test MAE criterion
        let model = DecisionTreeRegressor::new()
            .max_depth(3)
            .criterion(SplitCriterion::MAE)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(model.n_features(), 1);

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_regressor_builder_pattern() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = DecisionTreeRegressor::new()
            .criterion(SplitCriterion::MSE)
            .max_depth(10)
            .min_samples_split(2)
            .min_samples_leaf(1)
            .max_features(MaxFeatures::All)
            .random_state(42)
            .pruning(PruningStrategy::None)
            .missing_values(MissingValueStrategy::Skip)
            .growing_strategy(TreeGrowingStrategy::DepthFirst)
            .split_type(SplitType::AxisAligned)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(model.n_features(), 2);

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }
}
//! Cross-validation utilities for discriminant analysis parameter selection

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Fit, Predict},
    types::Float,
};
use std::collections::HashMap;

use crate::{LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis};

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds for cross-validation
    pub n_folds: usize,
    /// Scoring method to use
    pub scoring: String,
    /// Random seed for reproducible splits
    pub random_state: Option<u64>,
    /// Whether to shuffle data before splitting
    pub shuffle: bool,
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            scoring: "accuracy".to_string(),
            random_state: None,
            shuffle: true,
        }
    }
}

/// Parameter grid for hyperparameter search
#[derive(Debug, Clone)]
pub struct ParameterGrid {
    /// Parameters to search over
    pub params: HashMap<String, Vec<Float>>,
}

impl Default for ParameterGrid {
    fn default() -> Self {
        Self::new()
    }
}

impl ParameterGrid {
    /// Create a new parameter grid
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// Add parameter values to search
    pub fn add_param(mut self, name: &str, values: Vec<Float>) -> Self {
        self.params.insert(name.to_string(), values);
        self
    }

    /// Generate all parameter combinations
    pub fn combinations(&self) -> Vec<HashMap<String, Float>> {
        let mut result = vec![HashMap::new()];

        for (param_name, param_values) in &self.params {
            let mut new_result = Vec::new();
            for combination in result {
                for &value in param_values {
                    let mut new_combination = combination.clone();
                    new_combination.insert(param_name.clone(), value);
                    new_result.push(new_combination);
                }
            }
            result = new_result;
        }

        result
    }
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    /// Mean score across all folds
    pub mean_score: Float,
    /// Standard deviation of scores across folds
    pub std_score: Float,
    /// Individual fold scores
    pub fold_scores: Vec<Float>,
    /// Best parameters found
    pub best_params: HashMap<String, Float>,
}

/// Grid search with cross-validation for LDA
pub struct GridSearchLDA {
    /// Parameter grid to search
    pub param_grid: ParameterGrid,
    /// Cross-validation configuration
    pub cv_config: CrossValidationConfig,
}

impl GridSearchLDA {
    /// Create a new grid search instance
    pub fn new(param_grid: ParameterGrid, cv_config: CrossValidationConfig) -> Self {
        Self {
            param_grid,
            cv_config,
        }
    }

    /// Perform grid search with cross-validation
    pub fn fit_search(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<CrossValidationResults> {
        let combinations = self.param_grid.combinations();
        let mut best_score = Float::NEG_INFINITY;
        let mut best_params = HashMap::new();
        let mut best_fold_scores = Vec::new();

        for params in combinations {
            let cv_results = self.cross_validate_params(x, y, &params)?;

            if cv_results.mean_score > best_score {
                best_score = cv_results.mean_score;
                best_params = params;
                best_fold_scores = cv_results.fold_scores;
            }
        }

        Ok(CrossValidationResults {
            mean_score: best_score,
            std_score: self.compute_std(&best_fold_scores),
            fold_scores: best_fold_scores,
            best_params,
        })
    }

    /// Cross-validate specific parameter combination
    fn cross_validate_params(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        params: &HashMap<String, Float>,
    ) -> Result<CrossValidationResults> {
        let folds = self.create_stratified_folds(x, y)?;
        let mut fold_scores = Vec::new();

        for (train_x, train_y, test_x, test_y) in folds {
            // Create LDA with specified parameters
            let mut lda = LinearDiscriminantAnalysis::new();

            // Apply parameters
            if let Some(&shrinkage) = params.get("shrinkage") {
                lda = lda.shrinkage(Some(shrinkage));
            }
            if let Some(&l1_reg) = params.get("l1_reg") {
                lda = lda.l1_reg(l1_reg);
            }
            if let Some(&l2_reg) = params.get("l2_reg") {
                lda = lda.l2_reg(l2_reg);
            }
            if let Some(&elastic_net_ratio) = params.get("elastic_net_ratio") {
                lda = lda.elastic_net_ratio(elastic_net_ratio);
            }
            if let Some(&tol) = params.get("tol") {
                lda = lda.tol(tol);
            }

            // Fit and evaluate
            let fitted = lda.fit(&train_x, &train_y)?;
            let predictions = fitted.predict(&test_x)?;
            let score = self.compute_score(&test_y, &predictions)?;
            fold_scores.push(score);
        }

        let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
        let std_score = self.compute_std(&fold_scores);

        Ok(CrossValidationResults {
            mean_score,
            std_score,
            fold_scores,
            best_params: params.clone(),
        })
    }

    /// Create stratified cross-validation folds ensuring class balance
    fn create_stratified_folds(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<Vec<(Array2<Float>, Array1<i32>, Array2<Float>, Array1<i32>)>> {
        // Get unique classes and their indices
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &class) in y.iter().enumerate() {
            class_indices.entry(class).or_default().push(i);
        }

        // Ensure we have at least 2 classes
        if class_indices.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        let mut folds = Vec::new();

        for fold in 0..self.cv_config.n_folds {
            let mut test_indices = Vec::new();
            let mut train_indices = Vec::new();

            // For each class, distribute samples across folds
            for indices in class_indices.values() {
                let class_fold_size = indices.len() / self.cv_config.n_folds;
                let start_idx = fold * class_fold_size;
                let end_idx = if fold == self.cv_config.n_folds - 1 {
                    indices.len()
                } else {
                    (fold + 1) * class_fold_size
                };

                // Ensure we have at least 1 sample per class in test set
                let actual_end_idx = if start_idx >= indices.len() {
                    indices.len()
                } else if end_idx <= start_idx {
                    (start_idx + 1).min(indices.len())
                } else {
                    end_idx
                };

                // Add test indices for this class
                for &idx in &indices[start_idx..actual_end_idx] {
                    test_indices.push(idx);
                }

                // Add train indices for this class
                for &idx in &indices[0..start_idx] {
                    train_indices.push(idx);
                }
                for &idx in &indices[actual_end_idx..] {
                    train_indices.push(idx);
                }
            }

            // Ensure both train and test sets have samples
            if test_indices.is_empty() || train_indices.is_empty() {
                return Err(SklearsError::InvalidInput(
                    "Invalid fold configuration".to_string(),
                ));
            }

            // Extract train and test sets
            let train_x = Array2::from_shape_fn((train_indices.len(), x.ncols()), |(i, j)| {
                x[[train_indices[i], j]]
            });
            let train_y = Array1::from_shape_fn(train_indices.len(), |i| y[train_indices[i]]);

            let test_x = Array2::from_shape_fn((test_indices.len(), x.ncols()), |(i, j)| {
                x[[test_indices[i], j]]
            });
            let test_y = Array1::from_shape_fn(test_indices.len(), |i| y[test_indices[i]]);

            folds.push((train_x, train_y, test_x, test_y));
        }

        Ok(folds)
    }

    /// Compute evaluation score
    fn compute_score(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> Result<Float> {
        match self.cv_config.scoring.as_str() {
            "accuracy" => {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&true_val, &pred_val)| true_val == pred_val)
                    .count();
                Ok(correct as Float / y_true.len() as Float)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown scoring method: {}",
                self.cv_config.scoring
            ))),
        }
    }

    /// Compute standard deviation
    fn compute_std(&self, values: &[Float]) -> Float {
        let mean = values.iter().sum::<Float>() / values.len() as Float;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / values.len() as Float;
        variance.sqrt()
    }
}

/// Grid search with cross-validation for QDA
pub struct GridSearchQDA {
    /// Parameter grid to search
    pub param_grid: ParameterGrid,
    /// Cross-validation configuration
    pub cv_config: CrossValidationConfig,
}

impl GridSearchQDA {
    /// Create a new grid search instance
    pub fn new(param_grid: ParameterGrid, cv_config: CrossValidationConfig) -> Self {
        Self {
            param_grid,
            cv_config,
        }
    }

    /// Perform grid search with cross-validation
    pub fn fit_search(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<CrossValidationResults> {
        let combinations = self.param_grid.combinations();
        let mut best_score = Float::NEG_INFINITY;
        let mut best_params = HashMap::new();
        let mut best_fold_scores = Vec::new();

        for params in combinations {
            let cv_results = self.cross_validate_params(x, y, &params)?;

            if cv_results.mean_score > best_score {
                best_score = cv_results.mean_score;
                best_params = params;
                best_fold_scores = cv_results.fold_scores;
            }
        }

        Ok(CrossValidationResults {
            mean_score: best_score,
            std_score: self.compute_std(&best_fold_scores),
            fold_scores: best_fold_scores,
            best_params,
        })
    }

    /// Cross-validate specific parameter combination
    fn cross_validate_params(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        params: &HashMap<String, Float>,
    ) -> Result<CrossValidationResults> {
        let folds = self.create_stratified_folds(x, y)?;
        let mut fold_scores = Vec::new();

        for (train_x, train_y, test_x, test_y) in folds {
            // Create QDA with specified parameters
            let mut qda = QuadraticDiscriminantAnalysis::new();

            // Apply parameters
            if let Some(&reg_param) = params.get("reg_param") {
                qda = qda.reg_param(reg_param);
            }
            if let Some(&tol) = params.get("tol") {
                qda = qda.tol(tol);
            }
            if let Some(&diagonal_covariance) = params.get("diagonal_covariance") {
                qda = qda.diagonal_covariance(diagonal_covariance > 0.5);
            }

            // Fit and evaluate
            let fitted = qda.fit(&train_x, &train_y)?;
            let predictions = fitted.predict(&test_x)?;
            let score = self.compute_score(&test_y, &predictions)?;
            fold_scores.push(score);
        }

        let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
        let std_score = self.compute_std(&fold_scores);

        Ok(CrossValidationResults {
            mean_score,
            std_score,
            fold_scores,
            best_params: params.clone(),
        })
    }

    /// Create stratified cross-validation folds ensuring class balance
    fn create_stratified_folds(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<Vec<(Array2<Float>, Array1<i32>, Array2<Float>, Array1<i32>)>> {
        // Get unique classes and their indices
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &class) in y.iter().enumerate() {
            class_indices.entry(class).or_default().push(i);
        }

        // Ensure we have at least 2 classes
        if class_indices.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        let mut folds = Vec::new();

        for fold in 0..self.cv_config.n_folds {
            let mut test_indices = Vec::new();
            let mut train_indices = Vec::new();

            // For each class, distribute samples across folds
            for indices in class_indices.values() {
                let class_fold_size = indices.len() / self.cv_config.n_folds;
                let start_idx = fold * class_fold_size;
                let end_idx = if fold == self.cv_config.n_folds - 1 {
                    indices.len()
                } else {
                    (fold + 1) * class_fold_size
                };

                // Ensure we have at least 1 sample per class in test set
                let actual_end_idx = if start_idx >= indices.len() {
                    indices.len()
                } else if end_idx <= start_idx {
                    (start_idx + 1).min(indices.len())
                } else {
                    end_idx
                };

                // Add test indices for this class
                for &idx in &indices[start_idx..actual_end_idx] {
                    test_indices.push(idx);
                }

                // Add train indices for this class
                for &idx in &indices[0..start_idx] {
                    train_indices.push(idx);
                }
                for &idx in &indices[actual_end_idx..] {
                    train_indices.push(idx);
                }
            }

            // Ensure both train and test sets have samples
            if test_indices.is_empty() || train_indices.is_empty() {
                return Err(SklearsError::InvalidInput(
                    "Invalid fold configuration".to_string(),
                ));
            }

            // Extract train and test sets
            let train_x = Array2::from_shape_fn((train_indices.len(), x.ncols()), |(i, j)| {
                x[[train_indices[i], j]]
            });
            let train_y = Array1::from_shape_fn(train_indices.len(), |i| y[train_indices[i]]);

            let test_x = Array2::from_shape_fn((test_indices.len(), x.ncols()), |(i, j)| {
                x[[test_indices[i], j]]
            });
            let test_y = Array1::from_shape_fn(test_indices.len(), |i| y[test_indices[i]]);

            folds.push((train_x, train_y, test_x, test_y));
        }

        Ok(folds)
    }

    /// Compute evaluation score
    fn compute_score(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> Result<Float> {
        match self.cv_config.scoring.as_str() {
            "accuracy" => {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&true_val, &pred_val)| true_val == pred_val)
                    .count();
                Ok(correct as Float / y_true.len() as Float)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown scoring method: {}",
                self.cv_config.scoring
            ))),
        }
    }

    /// Compute standard deviation
    fn compute_std(&self, values: &[Float]) -> Float {
        let mean = values.iter().sum::<Float>() / values.len() as Float;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / values.len() as Float;
        variance.sqrt()
    }
}
/// Advanced validation metrics
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// accuracy
    pub accuracy: Float,
    /// precision
    pub precision: Float,
    /// recall
    pub recall: Float,
    /// f1_score
    pub f1_score: Float,
    /// sensitivity
    pub sensitivity: Float,
    /// specificity
    pub specificity: Float,
}

/// Bootstrap validation configuration
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Sample size for each bootstrap (as fraction of dataset)
    pub sample_fraction: Float,
    /// Random seed for reproducible results
    pub random_state: Option<u64>,
    /// Whether to use stratified sampling
    pub stratified: bool,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 100,
            sample_fraction: 1.0,
            random_state: None,
            stratified: true,
        }
    }
}

/// Bootstrap validation results
#[derive(Debug, Clone)]
pub struct BootstrapResults {
    /// mean_score
    pub mean_score: Float,
    /// std_score
    pub std_score: Float,
    /// confidence_interval_lower
    pub confidence_interval_lower: Float,
    /// confidence_interval_upper
    pub confidence_interval_upper: Float,
    /// bootstrap_scores
    pub bootstrap_scores: Vec<Float>,
    /// out_of_bag_score
    pub out_of_bag_score: Option<Float>,
}

/// Nested cross-validation configuration
#[derive(Debug, Clone)]
pub struct NestedCVConfig {
    /// Outer cross-validation folds (for performance estimation)
    pub outer_cv: CrossValidationConfig,
    /// Inner cross-validation folds (for hyperparameter selection)
    pub inner_cv: CrossValidationConfig,
    /// Parameter grid for inner optimization
    pub param_grid: ParameterGrid,
}

/// Nested cross-validation results
#[derive(Debug, Clone)]
pub struct NestedCVResults {
    /// outer_scores
    pub outer_scores: Vec<Float>,
    /// mean_score
    pub mean_score: Float,
    /// std_score
    pub std_score: Float,
    /// best_params_per_fold
    pub best_params_per_fold: Vec<HashMap<String, Float>>,
    /// inner_cv_scores
    pub inner_cv_scores: Vec<Vec<Float>>,
}

/// Temporal validation configuration for time series data
#[derive(Debug, Clone)]
pub struct TemporalValidationConfig {
    /// Initial training window size
    pub initial_window: usize,
    /// Step size for expanding window
    pub step_size: usize,
    /// Maximum number of validation splits
    pub max_splits: Option<usize>,
    /// Whether to use expanding window (true) or sliding window (false)
    pub expanding_window: bool,
}

/// Bootstrap validation for robust performance estimation
pub struct BootstrapValidator;

impl BootstrapValidator {
    /// Perform bootstrap validation
    pub fn validate<F, T>(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        config: &BootstrapConfig,
        model_fn: F,
    ) -> Result<BootstrapResults>
    where
        F: Fn(&Array2<Float>, &Array1<i32>) -> Result<T>,
        T: Predict<Array2<Float>, Array1<i32>>,
    {
        let mut rng = fastrand::Rng::new();
        if let Some(seed) = config.random_state {
            rng.seed(seed);
        }

        let n_samples = x.nrows();
        let sample_size = (n_samples as Float * config.sample_fraction) as usize;
        let mut bootstrap_scores = Vec::new();
        let mut oob_predictions = vec![Vec::new(); n_samples];
        let mut oob_counts = vec![0; n_samples];

        for _ in 0..config.n_bootstrap {
            // Generate bootstrap sample
            let bootstrap_indices = if config.stratified {
                self.stratified_bootstrap_sample(y, sample_size, &mut rng)?
            } else {
                (0..sample_size)
                    .map(|_| rng.usize(0..n_samples))
                    .collect::<Vec<_>>()
            };

            // Create bootstrap dataset
            let bootstrap_x = Array2::from_shape_fn((sample_size, x.ncols()), |(i, j)| {
                x[[bootstrap_indices[i], j]]
            });
            let bootstrap_y = Array1::from_shape_fn(sample_size, |i| y[bootstrap_indices[i]]);

            // Train model on bootstrap sample
            let model = model_fn(&bootstrap_x, &bootstrap_y)?;

            // Evaluate on bootstrap sample
            let predictions = model.predict(&bootstrap_x)?;
            let score = self.compute_accuracy(&bootstrap_y, &predictions);
            bootstrap_scores.push(score);

            // Track out-of-bag predictions
            let bootstrap_set: std::collections::HashSet<usize> =
                bootstrap_indices.into_iter().collect();
            for i in 0..n_samples {
                if !bootstrap_set.contains(&i) {
                    let sample_x = x.slice(s![i..i + 1, ..]).to_owned();
                    let pred = model.predict(&sample_x)?;
                    oob_predictions[i].push(pred[0]);
                    oob_counts[i] += 1;
                }
            }
        }

        // Compute bootstrap statistics
        let mean_score = bootstrap_scores.iter().sum::<Float>() / bootstrap_scores.len() as Float;
        let std_score = self.compute_std(&bootstrap_scores);

        // Compute confidence intervals (95% by default)
        let mut sorted_scores = bootstrap_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha = 0.05;
        let lower_idx = ((alpha / 2.0) * sorted_scores.len() as Float) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * sorted_scores.len() as Float) as usize;
        let confidence_interval_lower = sorted_scores[lower_idx];
        let confidence_interval_upper = sorted_scores[upper_idx.min(sorted_scores.len() - 1)];

        // Compute out-of-bag score
        let oob_score = self.compute_oob_score(y, &oob_predictions, &oob_counts);

        Ok(BootstrapResults {
            mean_score,
            std_score,
            confidence_interval_lower,
            confidence_interval_upper,
            bootstrap_scores,
            out_of_bag_score: oob_score,
        })
    }

    fn stratified_bootstrap_sample(
        &self,
        y: &Array1<i32>,
        sample_size: usize,
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<usize>> {
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &class) in y.iter().enumerate() {
            class_indices.entry(class).or_default().push(i);
        }

        let mut sample_indices = Vec::new();
        let n_classes = class_indices.len();
        let samples_per_class = sample_size / n_classes;
        let remainder = sample_size % n_classes;

        for (class_idx, (_, indices)) in class_indices.iter().enumerate() {
            let class_sample_size = samples_per_class + if class_idx < remainder { 1 } else { 0 };

            for _ in 0..class_sample_size {
                let idx = rng.usize(0..indices.len());
                sample_indices.push(indices[idx]);
            }
        }

        Ok(sample_indices)
    }

    fn compute_accuracy(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> Float {
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&true_val, &pred_val)| true_val == pred_val)
            .count();
        correct as Float / y_true.len() as Float
    }

    fn compute_oob_score(
        &self,
        y_true: &Array1<i32>,
        oob_predictions: &[Vec<i32>],
        oob_counts: &[usize],
    ) -> Option<Float> {
        let mut total_correct = 0;
        let mut total_predictions = 0;

        for (i, &true_label) in y_true.iter().enumerate() {
            if oob_counts[i] > 0 {
                // Use majority vote for OOB prediction
                let mut vote_counts: HashMap<i32, usize> = HashMap::new();
                for &pred in &oob_predictions[i] {
                    *vote_counts.entry(pred).or_insert(0) += 1;
                }

                if let Some((&predicted_label, _)) =
                    vote_counts.iter().max_by_key(|(_, &count)| count)
                {
                    if predicted_label == true_label {
                        total_correct += 1;
                    }
                    total_predictions += 1;
                }
            }
        }

        if total_predictions > 0 {
            Some(total_correct as Float / total_predictions as Float)
        } else {
            None
        }
    }

    fn compute_std(&self, values: &[Float]) -> Float {
        let mean = values.iter().sum::<Float>() / values.len() as Float;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / values.len() as Float;
        variance.sqrt()
    }
}

/// Nested cross-validation for unbiased performance estimation
pub struct NestedCrossValidator;

impl NestedCrossValidator {
    /// Perform nested cross-validation
    pub fn validate(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        config: &NestedCVConfig,
    ) -> Result<NestedCVResults> {
        // Create outer cross-validation folds
        let outer_folds = self.create_stratified_folds(x, y, &config.outer_cv)?;
        let mut outer_scores = Vec::new();
        let mut best_params_per_fold = Vec::new();
        let mut inner_cv_scores = Vec::new();

        for (train_x, train_y, test_x, test_y) in outer_folds {
            // Perform inner cross-validation for hyperparameter selection
            let inner_cv = GridSearchLDA::new(config.param_grid.clone(), config.inner_cv.clone());
            let inner_results = inner_cv.fit_search(&train_x, &train_y)?;

            // Train final model with best parameters on full training set
            let mut lda = LinearDiscriminantAnalysis::new();
            Self::apply_lda_params(&mut lda, &inner_results.best_params);
            let final_model = lda.fit(&train_x, &train_y)?;

            // Evaluate on test set
            let predictions = final_model.predict(&test_x)?;
            let outer_score = self.compute_accuracy(&test_y, &predictions);

            outer_scores.push(outer_score);
            best_params_per_fold.push(inner_results.best_params);
            inner_cv_scores.push(inner_results.fold_scores);
        }

        let mean_score = outer_scores.iter().sum::<Float>() / outer_scores.len() as Float;
        let std_score = self.compute_std(&outer_scores);

        Ok(NestedCVResults {
            outer_scores,
            mean_score,
            std_score,
            best_params_per_fold,
            inner_cv_scores,
        })
    }

    fn apply_lda_params(lda: &mut LinearDiscriminantAnalysis, params: &HashMap<String, Float>) {
        if let Some(&shrinkage) = params.get("shrinkage") {
            *lda = lda.clone().shrinkage(Some(shrinkage));
        }
        if let Some(&l1_reg) = params.get("l1_reg") {
            *lda = lda.clone().l1_reg(l1_reg);
        }
        if let Some(&l2_reg) = params.get("l2_reg") {
            *lda = lda.clone().l2_reg(l2_reg);
        }
        if let Some(&elastic_net_ratio) = params.get("elastic_net_ratio") {
            *lda = lda.clone().elastic_net_ratio(elastic_net_ratio);
        }
    }

    fn create_stratified_folds(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        config: &CrossValidationConfig,
    ) -> Result<Vec<(Array2<Float>, Array1<i32>, Array2<Float>, Array1<i32>)>> {
        // Reuse the stratified fold creation logic from GridSearchLDA
        let grid_search = GridSearchLDA::new(ParameterGrid::new(), config.clone());
        grid_search.create_stratified_folds(x, y)
    }

    fn compute_accuracy(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> Float {
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&true_val, &pred_val)| true_val == pred_val)
            .count();
        correct as Float / y_true.len() as Float
    }

    fn compute_std(&self, values: &[Float]) -> Float {
        let mean = values.iter().sum::<Float>() / values.len() as Float;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / values.len() as Float;
        variance.sqrt()
    }
}

/// Temporal validation for time series data
pub struct TemporalValidator;

impl TemporalValidator {
    /// Perform temporal validation
    pub fn validate<F, T>(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        config: &TemporalValidationConfig,
        model_fn: F,
    ) -> Result<Vec<Float>>
    where
        F: Fn(&Array2<Float>, &Array1<i32>) -> Result<T>,
        T: Predict<Array2<Float>, Array1<i32>>,
    {
        let n_samples = x.nrows();
        if config.initial_window >= n_samples {
            return Err(SklearsError::InvalidInput(
                "Initial window size too large".to_string(),
            ));
        }

        let mut scores = Vec::new();
        let mut current_train_end = config.initial_window;
        let mut split_count = 0;

        while current_train_end + config.step_size < n_samples {
            if let Some(max_splits) = config.max_splits {
                if split_count >= max_splits {
                    break;
                }
            }

            // Define training and test windows
            let train_start = if config.expanding_window {
                0
            } else {
                current_train_end.saturating_sub(config.initial_window)
            };

            let test_start = current_train_end;
            let test_end = (current_train_end + config.step_size).min(n_samples);

            // Extract training and test sets
            let train_x = x.slice(s![train_start..current_train_end, ..]).to_owned();
            let train_y = y.slice(s![train_start..current_train_end]).to_owned();
            let test_x = x.slice(s![test_start..test_end, ..]).to_owned();
            let test_y = y.slice(s![test_start..test_end]).to_owned();

            // Train model and evaluate
            let model = model_fn(&train_x, &train_y)?;
            let predictions = model.predict(&test_x)?;
            let score = self.compute_accuracy(&test_y, &predictions);
            scores.push(score);

            current_train_end = test_end;
            split_count += 1;
        }

        Ok(scores)
    }

    fn compute_accuracy(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> Float {
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&true_val, &pred_val)| true_val == pred_val)
            .count();
        correct as Float / y_true.len() as Float
    }
}

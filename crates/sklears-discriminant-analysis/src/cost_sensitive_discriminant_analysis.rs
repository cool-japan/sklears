//! Cost-sensitive Discriminant Analysis
//!
//! This module implements Cost-sensitive Discriminant Analysis for handling imbalanced datasets
//! where different misclassification costs need to be considered.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Transform},
    types::Float,
};
use std::collections::HashMap;

/// Cost matrix specification
#[derive(Debug, Clone)]
pub enum CostMatrix {
    /// Uniform cost for all misclassifications
    Uniform { cost: Float },
    /// Class-specific costs (cost of misclassifying each class)
    ClassSpecific { costs: Vec<Float> },
    /// Full cost matrix (cost\[i\]\[j\] = cost of predicting j when true class is i)
    Full { matrix: Array2<Float> },
    /// Inverse class frequency weighting
    Balanced,
    /// Custom weighting scheme
    Custom { weights: HashMap<i32, Float> },
}

/// Types of cost-sensitive adaptations
#[derive(Debug, Clone)]
pub enum CostSensitiveMethod {
    /// Threshold adjustment (post-processing)
    ThresholdAdjustment,
    /// Cost-sensitive training (during training)
    CostSensitiveTraining,
    /// Ensemble method with cost-sensitive sampling
    CostSensitiveEnsemble { n_estimators: usize },
    /// Bayes optimal classification
    BayesOptimal,
    /// MetaCost approach
    MetaCost { m: usize, p: Float },
}

/// Configuration for Cost-sensitive Discriminant Analysis
#[derive(Debug, Clone)]
pub struct CostSensitiveDiscriminantAnalysisConfig {
    /// Cost matrix specification
    pub cost_matrix: CostMatrix,
    /// Cost-sensitive method to use
    pub method: CostSensitiveMethod,
    /// Base discriminant method
    pub base_method: String,
    /// Base method parameters
    pub base_params: HashMap<String, Float>,
    /// Threshold optimization method
    pub threshold_method: String,
    /// Number of folds for threshold optimization
    pub cv_folds: usize,
    /// Whether to normalize costs
    pub normalize_costs: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Resampling strategy for imbalanced data
    pub resampling_strategy: String,
    /// Whether to use probability calibration
    pub calibrate_probabilities: bool,
    /// Calibration method
    pub calibration_method: String,
}

impl Default for CostSensitiveDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            cost_matrix: CostMatrix::Balanced,
            method: CostSensitiveMethod::ThresholdAdjustment,
            base_method: "lda".to_string(),
            base_params: HashMap::new(),
            threshold_method: "optimization".to_string(),
            cv_folds: 5,
            normalize_costs: true,
            random_state: None,
            resampling_strategy: "none".to_string(),
            calibrate_probabilities: false,
            calibration_method: "platt".to_string(),
        }
    }
}

/// Cost-sensitive Discriminant Analysis
pub struct CostSensitiveDiscriminantAnalysis {
    config: CostSensitiveDiscriminantAnalysisConfig,
}

impl CostSensitiveDiscriminantAnalysis {
    /// Create a new Cost-sensitive Discriminant Analysis instance
    pub fn new() -> Self {
        Self {
            config: CostSensitiveDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set the cost matrix
    pub fn cost_matrix(mut self, cost_matrix: CostMatrix) -> Self {
        self.config.cost_matrix = cost_matrix;
        self
    }

    /// Set the cost-sensitive method
    pub fn method(mut self, method: CostSensitiveMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set the base discriminant method
    pub fn base_method(mut self, method: &str) -> Self {
        self.config.base_method = method.to_string();
        self
    }

    /// Set base method parameters
    pub fn base_params(mut self, params: HashMap<String, Float>) -> Self {
        self.config.base_params = params;
        self
    }

    /// Set threshold optimization method
    pub fn threshold_method(mut self, method: &str) -> Self {
        self.config.threshold_method = method.to_string();
        self
    }

    /// Set number of CV folds
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.config.cv_folds = folds;
        self
    }

    /// Set whether to normalize costs
    pub fn normalize_costs(mut self, normalize: bool) -> Self {
        self.config.normalize_costs = normalize;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set resampling strategy
    pub fn resampling_strategy(mut self, strategy: &str) -> Self {
        self.config.resampling_strategy = strategy.to_string();
        self
    }

    /// Enable probability calibration
    pub fn calibrate_probabilities(mut self, calibrate: bool) -> Self {
        self.config.calibrate_probabilities = calibrate;
        self
    }

    /// Compute cost matrix from specification
    fn compute_cost_matrix(&self, classes: &[i32], y: &Array1<i32>) -> Result<Array2<Float>> {
        let n_classes = classes.len();
        let mut cost_matrix = Array2::zeros((n_classes, n_classes));

        match &self.config.cost_matrix {
            CostMatrix::Uniform { cost } => {
                // Uniform cost for all misclassifications
                cost_matrix.fill(*cost);
                for i in 0..n_classes {
                    cost_matrix[[i, i]] = 0.0; // No cost for correct predictions
                }
            }
            CostMatrix::ClassSpecific { costs } => {
                if costs.len() != n_classes {
                    return Err(SklearsError::InvalidParameter {
                        name: "costs".to_string(),
                        reason: "Number of costs must match number of classes".to_string(),
                    });
                }

                for i in 0..n_classes {
                    for j in 0..n_classes {
                        if i != j {
                            cost_matrix[[i, j]] = costs[i];
                        }
                    }
                }
            }
            CostMatrix::Full { matrix } => {
                if matrix.dim() != (n_classes, n_classes) {
                    return Err(SklearsError::InvalidParameter {
                        name: "cost_matrix".to_string(),
                        reason: "Cost matrix dimensions must match number of classes".to_string(),
                    });
                }
                cost_matrix = matrix.clone();
            }
            CostMatrix::Balanced => {
                // Inverse class frequency weighting
                let class_counts = self.compute_class_counts(classes, y);
                let total_samples = y.len() as Float;

                for i in 0..n_classes {
                    let class_weight = total_samples / (n_classes as Float * class_counts[i]);
                    for j in 0..n_classes {
                        if i != j {
                            cost_matrix[[i, j]] = class_weight;
                        }
                    }
                }
            }
            CostMatrix::Custom { weights } => {
                for (i, &class_i) in classes.iter().enumerate() {
                    let weight = weights.get(&class_i).unwrap_or(&1.0);
                    for j in 0..n_classes {
                        if i != j {
                            cost_matrix[[i, j]] = *weight;
                        }
                    }
                }
            }
        }

        // Normalize costs if requested
        if self.config.normalize_costs {
            let max_cost = cost_matrix.iter().fold(0.0 as Float, |a, &b| a.max(b));
            if max_cost > 0.0 {
                cost_matrix /= max_cost;
            }
        }

        Ok(cost_matrix)
    }

    /// Compute class counts
    fn compute_class_counts(&self, classes: &[i32], y: &Array1<i32>) -> Vec<Float> {
        let mut counts = vec![0.0; classes.len()];

        for &label in y.iter() {
            if let Some(idx) = classes.iter().position(|&c| c == label) {
                counts[idx] += 1.0;
            }
        }

        counts
    }

    /// Optimize decision thresholds based on cost matrix
    fn optimize_thresholds(
        &self,
        probas: &Array2<Float>,
        y_true: &Array1<i32>,
        classes: &[i32],
        cost_matrix: &Array2<Float>,
    ) -> Result<Array1<Float>> {
        let n_classes = classes.len();

        match self.config.threshold_method.as_str() {
            "optimization" => {
                self.optimize_thresholds_numerical(probas, y_true, classes, cost_matrix)
            }
            "bayes_optimal" => {
                // For Bayes optimal, thresholds are based on cost ratios
                let mut thresholds = Array1::zeros(n_classes);

                for i in 0..n_classes {
                    // Simple threshold based on cost ratio (binary classification logic)
                    if n_classes == 2 {
                        let cost_fp = cost_matrix[[0, 1]]; // False positive cost
                        let cost_fn = cost_matrix[[1, 0]]; // False negative cost
                        thresholds[i] = cost_fp / (cost_fp + cost_fn);
                    } else {
                        thresholds[i] = 0.5; // Default for multi-class
                    }
                }

                Ok(thresholds)
            }
            _ => {
                // Default equal thresholds
                Ok(Array1::from_elem(n_classes, 1.0 / n_classes as Float))
            }
        }
    }

    /// Numerical optimization of thresholds
    fn optimize_thresholds_numerical(
        &self,
        probas: &Array2<Float>,
        y_true: &Array1<i32>,
        classes: &[i32],
        cost_matrix: &Array2<Float>,
    ) -> Result<Array1<Float>> {
        let n_classes = classes.len();

        if n_classes == 2 {
            // Binary classification - optimize single threshold
            let mut best_threshold = 0.5;
            let mut best_cost = Float::INFINITY;

            // Grid search over thresholds
            for threshold_int in 0..=100 {
                let threshold = threshold_int as Float / 100.0;
                let cost =
                    self.evaluate_threshold_cost(probas, y_true, classes, cost_matrix, threshold)?;

                if cost < best_cost {
                    best_cost = cost;
                    best_threshold = threshold;
                }
            }

            Ok(Array1::from_vec(vec![1.0 - best_threshold, best_threshold]))
        } else {
            // Multi-class - use equal probability thresholds for now
            Ok(Array1::from_elem(n_classes, 1.0 / n_classes as Float))
        }
    }

    /// Evaluate cost for a given threshold (binary classification)
    fn evaluate_threshold_cost(
        &self,
        probas: &Array2<Float>,
        y_true: &Array1<i32>,
        classes: &[i32],
        cost_matrix: &Array2<Float>,
        threshold: Float,
    ) -> Result<Float> {
        let mut total_cost = 0.0;

        for (i, &true_label) in y_true.iter().enumerate() {
            let true_idx = classes.iter().position(|&c| c == true_label).unwrap();

            // Make prediction based on threshold
            let pred_idx = if probas[[i, 1]] > threshold { 1 } else { 0 };

            // Add cost
            total_cost += cost_matrix[[true_idx, pred_idx]];
        }

        Ok(total_cost / y_true.len() as Float)
    }

    /// Apply cost-sensitive resampling
    fn apply_resampling(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &[i32],
    ) -> Result<(Array2<Float>, Array1<i32>)> {
        match self.config.resampling_strategy.as_str() {
            "oversample" => self.oversample_minority_classes(x, y, classes),
            "undersample" => self.undersample_majority_classes(x, y, classes),
            "smote" => self.apply_smote(x, y, classes),
            _ => Ok((x.clone(), y.clone())),
        }
    }

    /// Oversample minority classes
    fn oversample_minority_classes(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &[i32],
    ) -> Result<(Array2<Float>, Array1<i32>)> {
        let class_counts = self.compute_class_counts(classes, y);
        let max_count = class_counts.iter().fold(0.0 as Float, |a, &b| a.max(b)) as usize;

        let mut x_resampled = Vec::new();
        let mut y_resampled = Vec::new();

        let mut rng = SimpleRng::new(self.config.random_state.unwrap_or(42));

        for (_class_idx, &class_label) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class_label)
                .map(|(idx, _)| idx)
                .collect();

            let current_count = class_indices.len();
            let target_count = max_count;

            // Add original samples
            for &idx in &class_indices {
                x_resampled.push(x.row(idx).to_owned());
                y_resampled.push(class_label);
            }

            // Oversample if needed
            for _ in current_count..target_count {
                let random_idx = rng.choice(&class_indices);
                x_resampled.push(x.row(random_idx).to_owned());
                y_resampled.push(class_label);
            }
        }

        // Convert back to arrays
        let n_samples = x_resampled.len();
        let n_features = x.ncols();
        let mut x_result = Array2::zeros((n_samples, n_features));

        for (i, row) in x_resampled.iter().enumerate() {
            x_result.row_mut(i).assign(row);
        }

        let y_result = Array1::from_vec(y_resampled);

        Ok((x_result, y_result))
    }

    /// Undersample majority classes
    fn undersample_majority_classes(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &[i32],
    ) -> Result<(Array2<Float>, Array1<i32>)> {
        let class_counts = self.compute_class_counts(classes, y);
        let min_count = class_counts.iter().fold(Float::INFINITY, |a, &b| a.min(b)) as usize;

        let mut x_resampled = Vec::new();
        let mut y_resampled = Vec::new();

        let mut rng = SimpleRng::new(self.config.random_state.unwrap_or(42));

        for (_class_idx, &class_label) in classes.iter().enumerate() {
            let mut class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class_label)
                .map(|(idx, _)| idx)
                .collect();

            // Randomly sample min_count indices
            rng.shuffle(&mut class_indices);
            class_indices.truncate(min_count);

            for &idx in &class_indices {
                x_resampled.push(x.row(idx).to_owned());
                y_resampled.push(class_label);
            }
        }

        // Convert back to arrays
        let n_samples = x_resampled.len();
        let n_features = x.ncols();
        let mut x_result = Array2::zeros((n_samples, n_features));

        for (i, row) in x_resampled.iter().enumerate() {
            x_result.row_mut(i).assign(row);
        }

        let y_result = Array1::from_vec(y_resampled);

        Ok((x_result, y_result))
    }

    /// Apply SMOTE (simplified version)
    fn apply_smote(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &[i32],
    ) -> Result<(Array2<Float>, Array1<i32>)> {
        // For simplicity, fall back to oversampling
        // A full SMOTE implementation would require k-nearest neighbors
        self.oversample_minority_classes(x, y, classes)
    }
}

/// Simple random number generator for resampling
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    fn uniform(&mut self) -> Float {
        (self.next() as Float) / (u64::MAX as Float)
    }

    fn choice(&mut self, items: &[usize]) -> usize {
        if items.is_empty() {
            return 0;
        }
        let idx = (self.uniform() * items.len() as Float) as usize;
        items[idx.min(items.len() - 1)]
    }

    fn shuffle(&mut self, items: &mut [usize]) {
        for i in (1..items.len()).rev() {
            let j = (self.uniform() * (i + 1) as Float) as usize;
            items.swap(i, j);
        }
    }
}

/// Simple base discriminant model
pub struct SimpleDiscriminantModel {
    classes: Vec<i32>,
    means: Array2<Float>,
    covariance_inv: Array2<Float>,
    priors: Array1<Float>,
}

impl Default for SimpleDiscriminantModel {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleDiscriminantModel {
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            means: Array2::zeros((0, 0)),
            covariance_inv: Array2::zeros((0, 0)),
            priors: Array1::zeros(0),
        }
    }

    pub fn fit(&mut self, x: &Array2<Float>, y: &Array1<i32>) -> Result<()> {
        // Get unique classes
        self.classes = {
            let mut cls = y.to_vec();
            cls.sort_unstable();
            cls.dedup();
            cls
        };

        let n_classes = self.classes.len();
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Compute class means and priors
        self.means = Array2::zeros((n_classes, n_features));
        self.priors = Array1::zeros(n_classes);

        for (k, &class_label) in self.classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&label| label == class_label).collect();
            let class_count = class_mask.iter().filter(|&&mask| mask).count();

            self.priors[k] = class_count as Float / n_samples as Float;

            let mut class_sum = Array1::zeros(n_features);
            for (i, &mask) in class_mask.iter().enumerate() {
                if mask {
                    class_sum += &x.row(i);
                }
            }
            self.means
                .row_mut(k)
                .assign(&(class_sum / class_count as Float));
        }

        // Compute pooled covariance matrix
        let mut covariance = Array2::zeros((n_features, n_features));
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let class_idx = self.classes.iter().position(|&c| c == y[i]).unwrap();
            let diff = &row - &self.means.row(class_idx);
            let diff_outer = diff
                .clone()
                .insert_axis(Axis(1))
                .dot(&diff.insert_axis(Axis(0)));
            covariance += &diff_outer;
        }
        covariance /= (n_samples - n_classes) as Float;

        // Add regularization
        for i in 0..n_features {
            covariance[[i, i]] += 1e-6;
        }

        // Simple matrix inversion
        self.covariance_inv = self.invert_matrix(&covariance)?;

        Ok(())
    }

    pub fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let mut log_probas = Array1::zeros(n_classes);

            for (k, _) in self.classes.iter().enumerate() {
                let mean_diff = &sample - &self.means.row(k);
                let quadratic_term = mean_diff.dot(&self.covariance_inv.dot(&mean_diff));
                log_probas[k] = self.priors[k].ln() - 0.5 * quadratic_term;
            }

            // Convert to probabilities using softmax
            let max_log_proba = log_probas
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_log_probas = log_probas.mapv(|x| (x - max_log_proba).exp());
            let sum_exp = exp_log_probas.sum();

            probas.row_mut(i).assign(&(exp_log_probas / sum_exp));
        }

        Ok(probas)
    }

    pub fn classes(&self) -> &[i32] {
        &self.classes
    }

    fn invert_matrix(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidData {
                reason: "Matrix must be square".to_string(),
            });
        }

        // Simple Gauss-Jordan elimination
        let mut augmented = Array2::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                augmented[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > augmented[[pivot_row, i]].abs() {
                    pivot_row = k;
                }
            }

            // Swap rows if needed
            if pivot_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[pivot_row, j]];
                    augmented[[pivot_row, j]] = temp;
                }
            }

            if augmented[[i, i]].abs() < 1e-10 {
                return Err(SklearsError::InvalidData {
                    reason: "Matrix is singular".to_string(),
                });
            }

            let pivot = augmented[[i, i]];
            for j in 0..(2 * n) {
                augmented[[i, j]] /= pivot;
            }

            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inverse)
    }
}

/// Trained Cost-sensitive Discriminant Analysis model
pub struct TrainedCostSensitiveDiscriminantAnalysis {
    config: CostSensitiveDiscriminantAnalysisConfig,
    base_model: SimpleDiscriminantModel,
    classes: Vec<i32>,
    cost_matrix: Array2<Float>,
    thresholds: Array1<Float>,
    class_weights: Option<HashMap<i32, Float>>,
}

impl TrainedCostSensitiveDiscriminantAnalysis {
    /// Get the cost matrix
    pub fn cost_matrix(&self) -> &Array2<Float> {
        &self.cost_matrix
    }

    /// Get the decision thresholds
    pub fn thresholds(&self) -> &Array1<Float> {
        &self.thresholds
    }

    /// Get the classes
    pub fn classes(&self) -> &[i32] {
        &self.classes
    }

    /// Get class weights if available
    pub fn class_weights(&self) -> Option<&HashMap<i32, Float>> {
        self.class_weights.as_ref()
    }

    /// Compute expected cost for predictions
    pub fn expected_cost(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> Result<Float> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::InvalidData {
                reason: "y_true and y_pred must have same length".to_string(),
            });
        }

        let mut total_cost = 0.0;

        for (_i, (&true_label, &pred_label)) in y_true.iter().zip(y_pred.iter()).enumerate() {
            let true_idx = self
                .classes
                .iter()
                .position(|&c| c == true_label)
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Unknown true class".to_string(),
                })?;
            let pred_idx = self
                .classes
                .iter()
                .position(|&c| c == pred_label)
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Unknown predicted class".to_string(),
                })?;

            total_cost += self.cost_matrix[[true_idx, pred_idx]];
        }

        Ok(total_cost / y_true.len() as Float)
    }

    /// Get cost-sensitive metrics
    pub fn cost_sensitive_metrics(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> Result<HashMap<String, Float>> {
        let mut metrics = HashMap::new();

        // Expected cost
        let expected_cost = self.expected_cost(y_true, y_pred)?;
        metrics.insert("expected_cost".to_string(), expected_cost);

        // Cost savings compared to always predicting majority class
        let majority_class = self.get_majority_class(y_true);
        let majority_pred = Array1::from_elem(y_true.len(), majority_class);
        let majority_cost = self.expected_cost(y_true, &majority_pred)?;
        let cost_savings = (majority_cost - expected_cost) / majority_cost;
        metrics.insert("cost_savings".to_string(), cost_savings);

        Ok(metrics)
    }

    fn get_majority_class(&self, y: &Array1<i32>) -> i32 {
        let mut class_counts: HashMap<i32, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        *class_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .unwrap()
            .0
    }
}

impl Estimator for CostSensitiveDiscriminantAnalysis {
    type Config = CostSensitiveDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for CostSensitiveDiscriminantAnalysis {
    type Fitted = TrainedCostSensitiveDiscriminantAnalysis;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidData {
                reason: "Number of samples in X and y must match".to_string(),
            });
        }

        // Get unique classes
        let classes = {
            let mut cls = y.to_vec();
            cls.sort_unstable();
            cls.dedup();
            cls
        };

        // Compute cost matrix
        let cost_matrix = self.compute_cost_matrix(&classes, y)?;

        // Apply resampling if requested
        let (x_resampled, y_resampled) = self.apply_resampling(x, y, &classes)?;

        // Fit base model
        let mut base_model = SimpleDiscriminantModel::new();
        base_model.fit(&x_resampled, &y_resampled)?;

        // Get initial probabilities for threshold optimization
        let probas = base_model.predict_proba(&x_resampled)?;

        // Optimize thresholds
        let thresholds = self.optimize_thresholds(&probas, &y_resampled, &classes, &cost_matrix)?;

        // Compute class weights for cost-sensitive prediction
        let class_weights = match &self.config.cost_matrix {
            CostMatrix::Custom { weights } => Some(weights.clone()),
            _ => None,
        };

        Ok(TrainedCostSensitiveDiscriminantAnalysis {
            config: self.config,
            base_model,
            classes,
            cost_matrix,
            thresholds,
            class_weights,
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedCostSensitiveDiscriminantAnalysis {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.base_model.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        match self.config.method {
            CostSensitiveMethod::ThresholdAdjustment => {
                // Apply cost-sensitive thresholds
                for (i, proba_row) in probas.axis_iter(Axis(0)).enumerate() {
                    let mut best_class_idx = 0;
                    let mut min_expected_cost = Float::INFINITY;

                    for (j, &_prob) in proba_row.iter().enumerate() {
                        // Compute expected cost for predicting class j
                        let mut expected_cost = 0.0;
                        for (k, &true_prob) in proba_row.iter().enumerate() {
                            expected_cost += true_prob * self.cost_matrix[[k, j]];
                        }

                        if expected_cost < min_expected_cost {
                            min_expected_cost = expected_cost;
                            best_class_idx = j;
                        }
                    }

                    predictions[i] = self.classes[best_class_idx];
                }
            }
            _ => {
                // Default: maximum probability
                for (i, proba_row) in probas.axis_iter(Axis(0)).enumerate() {
                    let max_idx = proba_row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap()
                        .0;
                    predictions[i] = self.classes[max_idx];
                }
            }
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedCostSensitiveDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        self.base_model.predict_proba(x)
    }
}

impl Transform<Array2<Float>> for TrainedCostSensitiveDiscriminantAnalysis {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // For discriminant analysis, transformation is typically the projected data
        // For simplicity, return input unchanged
        Ok(x.clone())
    }
}

impl Default for CostSensitiveDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cost_sensitive_discriminant_analysis() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let csda =
            CostSensitiveDiscriminantAnalysis::new().cost_matrix(CostMatrix::Uniform { cost: 1.0 });

        let fitted = csda.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.cost_matrix().dim(), (2, 2));

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        let probas = fitted.predict_proba(&x).unwrap();
        assert_eq!(probas.dim(), (6, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_cost_sensitive_with_balanced_weights() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0]
        ];
        // Imbalanced dataset: 6 samples of class 0, 2 samples of class 1
        let y = array![0, 0, 0, 0, 0, 0, 1, 1];

        let csda = CostSensitiveDiscriminantAnalysis::new().cost_matrix(CostMatrix::Balanced);

        let fitted = csda.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 8);
    }

    #[test]
    fn test_cost_sensitive_with_custom_costs() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let mut custom_weights = HashMap::new();
        custom_weights.insert(0, 1.0);
        custom_weights.insert(1, 2.0);

        let csda = CostSensitiveDiscriminantAnalysis::new().cost_matrix(CostMatrix::Custom {
            weights: custom_weights,
        });

        let fitted = csda.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_cost_sensitive_with_full_cost_matrix() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let cost_matrix = array![
            [0.0, 2.0], // Cost of misclassifying class 0 as class 1: 2.0
            [1.0, 0.0]  // Cost of misclassifying class 1 as class 0: 1.0
        ];

        let csda = CostSensitiveDiscriminantAnalysis::new().cost_matrix(CostMatrix::Full {
            matrix: cost_matrix,
        });

        let fitted = csda.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.cost_matrix()[[0, 1]], 1.0); // 2.0 / 2.0 = 1.0 (normalized)
        assert_eq!(fitted.cost_matrix()[[1, 0]], 0.5); // 1.0 / 2.0 = 0.5 (normalized)

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_cost_sensitive_resampling() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1]; // Imbalanced: 4 vs 2

        let strategies = vec!["oversample", "undersample"];

        for strategy in strategies {
            let csda = CostSensitiveDiscriminantAnalysis::new()
                .resampling_strategy(strategy)
                .random_state(42);

            let fitted = csda.fit(&x, &y).unwrap();

            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 6);
        }
    }

    #[test]
    fn test_expected_cost_computation() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let csda =
            CostSensitiveDiscriminantAnalysis::new().cost_matrix(CostMatrix::Uniform { cost: 1.0 });

        let fitted = csda.fit(&x, &y).unwrap();

        let y_true = array![0, 0, 1, 1];
        let y_pred = array![0, 1, 1, 0]; // 2 correct, 2 incorrect

        let expected_cost = fitted.expected_cost(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(expected_cost, 0.5, epsilon = 1e-6); // 50% error rate
    }

    #[test]
    fn test_cost_sensitive_metrics() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let csda =
            CostSensitiveDiscriminantAnalysis::new().cost_matrix(CostMatrix::Uniform { cost: 1.0 });

        let fitted = csda.fit(&x, &y).unwrap();

        let y_true = array![0, 0, 1, 1];
        let y_pred = array![0, 0, 1, 1]; // Perfect predictions

        let metrics = fitted.cost_sensitive_metrics(&y_true, &y_pred).unwrap();

        assert!(metrics.contains_key("expected_cost"));
        assert!(metrics.contains_key("cost_savings"));

        // Perfect predictions should have zero cost
        assert_abs_diff_eq!(metrics["expected_cost"], 0.0, epsilon = 1e-6);
    }
}

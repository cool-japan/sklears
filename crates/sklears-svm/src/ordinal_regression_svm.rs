//! Ordinal Regression Support Vector Machines
//!
//! This module implements SVM-based ordinal regression algorithms for problems
//! where the target variable has ordered categories (e.g., ratings, grades).
//! Unlike standard classification, ordinal regression preserves the order
//! information between classes.

use crate::kernels::{Kernel, KernelType};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Ordinal regression strategies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum OrdinalStrategy {
    /// Immediate-threshold model: single ranking function with multiple thresholds
    #[default]
    ImmediateThreshold,
    /// All-pairs approach: learn pairwise ordering
    AllPairs,
    /// One-vs-rest with ordinal constraints
    OneVsRest,
    /// Cumulative model: model cumulative probabilities
    Cumulative,
}

/// Loss functions for ordinal regression
#[derive(Debug, Clone, PartialEq, Default)]
pub enum OrdinalLoss {
    /// Absolute loss - penalizes distance from correct rank
    #[default]
    Absolute,
    /// Squared loss - penalizes squared distance from correct rank
    Squared,
    /// Hinge loss adapted for ordinal regression
    OrdinalHinge,
    /// Logistic loss for ordinal regression
    OrdinalLogistic,
}

/// Configuration for ordinal regression SVM
#[derive(Debug, Clone)]
pub struct OrdinalRegressionSVMConfig {
    /// Regularization parameter
    pub c: Float,
    /// Ordinal loss function
    pub loss: OrdinalLoss,
    /// Ordinal regression strategy
    pub strategy: OrdinalStrategy,
    /// Kernel function to use
    pub kernel: KernelType,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Random seed
    pub random_state: Option<u64>,
    /// Learning rate for gradient-based optimization
    pub learning_rate: Float,
    /// Learning rate decay
    pub learning_rate_decay: Float,
    /// Minimum learning rate
    pub min_learning_rate: Float,
    /// Threshold initialization strategy
    pub threshold_init: ThresholdInit,
    /// Regularization for thresholds
    pub threshold_regularization: Float,
}

/// Threshold initialization strategies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ThresholdInit {
    Uniform,
    #[default]
    Frequency,
    Custom(Array1<Float>),
}

impl Default for OrdinalRegressionSVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            loss: OrdinalLoss::default(),
            strategy: OrdinalStrategy::default(),
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 1000,
            fit_intercept: true,
            random_state: None,
            learning_rate: 0.01,
            learning_rate_decay: 0.99,
            min_learning_rate: 1e-6,
            threshold_init: ThresholdInit::default(),
            threshold_regularization: 0.01,
        }
    }
}

/// Ordinal Regression Support Vector Machine
#[derive(Debug)]
pub struct OrdinalRegressionSVM<State = Untrained> {
    config: OrdinalRegressionSVMConfig,
    state: PhantomData<State>,
    // Fitted attributes
    support_vectors_: Option<Array2<Float>>,
    alpha_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    thresholds_: Option<Array1<Float>>,
    classes_: Option<Array1<usize>>,
    n_features_in_: Option<usize>,
    n_iter_: Option<usize>,
    n_classes_: Option<usize>,
}

impl OrdinalRegressionSVM<Untrained> {
    /// Create a new ordinal regression SVM
    pub fn new() -> Self {
        Self {
            config: OrdinalRegressionSVMConfig::default(),
            state: PhantomData,
            support_vectors_: None,
            alpha_: None,
            intercept_: None,
            thresholds_: None,
            classes_: None,
            n_features_in_: None,
            n_iter_: None,
            n_classes_: None,
        }
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set the ordinal loss function
    pub fn loss(mut self, loss: OrdinalLoss) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the ordinal regression strategy
    pub fn strategy(mut self, strategy: OrdinalStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set the tolerance for stopping criterion
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the threshold initialization strategy
    pub fn threshold_init(mut self, threshold_init: ThresholdInit) -> Self {
        self.config.threshold_init = threshold_init;
        self
    }

    /// Set the threshold regularization
    pub fn threshold_regularization(mut self, reg: Float) -> Self {
        self.config.threshold_regularization = reg;
        self
    }
}

impl Default for OrdinalRegressionSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl OrdinalLoss {
    /// Compute the loss value for ordinal regression
    pub fn loss(&self, predicted_rank: Float, true_rank: Float) -> Float {
        let error = predicted_rank - true_rank;
        match self {
            OrdinalLoss::Absolute => error.abs(),
            OrdinalLoss::Squared => error * error,
            OrdinalLoss::OrdinalHinge => {
                // Modified hinge loss that considers rank distance
                let rank_distance = error.abs();
                if rank_distance <= 1.0 {
                    0.0
                } else {
                    rank_distance - 1.0
                }
            }
            OrdinalLoss::OrdinalLogistic => {
                // Logistic loss adapted for ordinal regression
                (1.0 + error.abs().exp()).ln()
            }
        }
    }

    /// Compute the derivative of the loss function
    pub fn derivative(&self, predicted_rank: Float, true_rank: Float) -> Float {
        let error = predicted_rank - true_rank;
        match self {
            OrdinalLoss::Absolute => error.signum(),
            OrdinalLoss::Squared => 2.0 * error,
            OrdinalLoss::OrdinalHinge => {
                let rank_distance = error.abs();
                if rank_distance <= 1.0 {
                    0.0
                } else {
                    error.signum()
                }
            }
            OrdinalLoss::OrdinalLogistic => {
                error.signum() * error.abs().exp() / (1.0 + error.abs().exp())
            }
        }
    }
}

impl Fit<Array2<Float>, Array1<usize>> for OrdinalRegressionSVM<Untrained> {
    type Fitted = OrdinalRegressionSVM<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<usize>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Shape mismatch: X and y must have the same number of samples".to_string(),
            ));
        }

        // Find unique classes and ensure they're ordered
        let mut unique_classes: Vec<usize> = y.iter().cloned().collect();
        unique_classes.sort();
        unique_classes.dedup();
        let n_classes = unique_classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for ordinal regression".to_string(),
            ));
        }

        // Create mapping from class labels to ordinal ranks (0, 1, 2, ...)
        let class_to_rank: HashMap<usize, usize> = unique_classes
            .iter()
            .enumerate()
            .map(|(rank, &class)| (class, rank))
            .collect();

        // Convert class labels to ranks
        let mut y_ranks = Array1::zeros(n_samples);
        for (i, &class) in y.iter().enumerate() {
            y_ranks[i] = *class_to_rank
                .get(&class)
                .ok_or_else(|| SklearsError::InvalidInput(format!("Unknown class: {class}")))?
                as Float;
        }

        match self.config.strategy {
            OrdinalStrategy::ImmediateThreshold => {
                self.fit_immediate_threshold(x, &y_ranks, &unique_classes, n_classes, n_features)
            }
            OrdinalStrategy::AllPairs => {
                self.fit_all_pairs(x, &y_ranks, &unique_classes, n_classes, n_features)
            }
            OrdinalStrategy::OneVsRest => {
                self.fit_one_vs_rest(x, &y_ranks, &unique_classes, n_classes, n_features)
            }
            OrdinalStrategy::Cumulative => {
                self.fit_cumulative(x, &y_ranks, &unique_classes, n_classes, n_features)
            }
        }
    }
}

impl OrdinalRegressionSVM<Untrained> {
    fn fit_immediate_threshold(
        self,
        x: &Array2<Float>,
        y_ranks: &Array1<Float>,
        unique_classes: &[usize],
        n_classes: usize,
        n_features: usize,
    ) -> Result<OrdinalRegressionSVM<Trained>> {
        let n_samples = x.nrows();

        // Initialize thresholds
        let mut thresholds = match &self.config.threshold_init {
            ThresholdInit::Uniform => {
                let mut thresh = Array1::zeros(n_classes - 1);
                for i in 0..n_classes - 1 {
                    thresh[i] = (i as Float + 1.0) - 0.5;
                }
                thresh
            }
            ThresholdInit::Frequency => {
                // Initialize based on class frequencies
                let mut class_counts = vec![0; n_classes];
                for &rank in y_ranks.iter() {
                    if rank >= 0.0 && (rank as usize) < n_classes {
                        class_counts[rank as usize] += 1;
                    }
                }

                let mut thresh = Array1::zeros(n_classes - 1);
                let mut cumulative = 0.0;
                for i in 0..n_classes - 1 {
                    cumulative += class_counts[i] as Float;
                    thresh[i] = cumulative / n_samples as Float * (n_classes - 1) as Float;
                }
                thresh
            }
            ThresholdInit::Custom(custom_thresh) => {
                if custom_thresh.len() != n_classes - 1 {
                    return Err(SklearsError::InvalidInput(
                        "Custom thresholds must have length n_classes - 1".to_string(),
                    ));
                }
                custom_thresh.clone()
            }
        };

        // Create kernel instance
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>,
        };

        // Initialize model parameters
        let support_vectors = x.clone();
        let mut alpha = Array1::<Float>::zeros(n_samples);
        let mut intercept = 0.0;
        let mut current_lr = self.config.learning_rate;

        // Training loop
        let mut n_iter = 0;
        for iteration in 0..self.config.max_iter {
            n_iter = iteration + 1;
            let mut converged = true;

            for i in 0..n_samples {
                // Compute ranking score
                let mut ranking_score = if self.config.fit_intercept {
                    intercept
                } else {
                    0.0
                };

                for j in 0..n_samples {
                    if alpha[j].abs() > 1e-10 {
                        let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                        ranking_score += alpha[j] * k_val;
                    }
                }

                // Determine predicted rank based on thresholds
                let mut predicted_rank = 0.0;
                for k in 0..thresholds.len() {
                    if ranking_score > thresholds[k] {
                        predicted_rank += 1.0;
                    }
                }

                // Compute loss and its derivative
                let true_rank = y_ranks[i];
                let loss_derivative = self.config.loss.derivative(predicted_rank, true_rank);

                // Update alpha using gradient descent
                let old_alpha = alpha[i];
                let gradient = loss_derivative + alpha[i] / self.config.c;
                alpha[i] -= current_lr * gradient;

                // Update intercept if needed
                if self.config.fit_intercept {
                    intercept -= current_lr * loss_derivative;
                }

                // Update thresholds
                for k in 0..thresholds.len() {
                    let threshold_grad = if ranking_score > thresholds[k] {
                        -loss_derivative
                    } else {
                        loss_derivative
                    };
                    let regularized_grad =
                        threshold_grad + self.config.threshold_regularization * thresholds[k];
                    thresholds[k] -= current_lr * regularized_grad;
                }

                // Ensure threshold ordering
                for k in 1..thresholds.len() {
                    if thresholds[k] <= thresholds[k - 1] {
                        thresholds[k] = thresholds[k - 1] + 0.01;
                    }
                }

                // Check convergence
                if (alpha[i] - old_alpha).abs() > self.config.tol {
                    converged = false;
                }
            }

            // Update learning rate
            current_lr =
                (current_lr * self.config.learning_rate_decay).max(self.config.min_learning_rate);

            if converged {
                break;
            }
        }

        // Filter out non-support vectors
        let support_indices: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(_, &a)| a.abs() > 1e-8)
            .map(|(i, _)| i)
            .collect();

        let final_support_vectors = if support_indices.is_empty() {
            let mut sv = Array2::zeros((1, n_features));
            sv.row_mut(0).assign(&x.row(0));
            sv
        } else {
            let mut sv = Array2::zeros((support_indices.len(), n_features));
            for (i, &idx) in support_indices.iter().enumerate() {
                sv.row_mut(i).assign(&x.row(idx));
            }
            sv
        };

        let final_alpha = if support_indices.is_empty() {
            Array1::from_vec(vec![1e-8])
        } else {
            Array1::from_vec(support_indices.iter().map(|&i| alpha[i]).collect())
        };

        Ok(OrdinalRegressionSVM {
            config: self.config,
            state: PhantomData,
            support_vectors_: Some(final_support_vectors),
            alpha_: Some(final_alpha),
            intercept_: Some(intercept),
            thresholds_: Some(thresholds),
            classes_: Some(Array1::from_vec(unique_classes.to_vec())),
            n_features_in_: Some(n_features),
            n_iter_: Some(n_iter),
            n_classes_: Some(n_classes),
        })
    }

    fn fit_all_pairs(
        self,
        x: &Array2<Float>,
        y_ranks: &Array1<Float>,
        unique_classes: &[usize],
        n_classes: usize,
        n_features: usize,
    ) -> Result<OrdinalRegressionSVM<Trained>> {
        // For simplicity, implement all-pairs as a ranking function with score comparison
        // This is a simplified version - a full implementation would train separate classifiers
        // for each pair of classes

        let n_samples = x.nrows();

        // Create kernel instance
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>,
        };

        // Initialize model parameters
        let support_vectors = x.clone();
        let mut alpha = Array1::<Float>::zeros(n_samples);
        let mut intercept = 0.0;
        let mut current_lr = self.config.learning_rate;

        // Training loop - simplified to treat as regression on ranks
        let mut n_iter = 0;
        for iteration in 0..self.config.max_iter {
            n_iter = iteration + 1;
            let mut converged = true;

            for i in 0..n_samples {
                // Compute prediction
                let mut prediction = if self.config.fit_intercept {
                    intercept
                } else {
                    0.0
                };

                for j in 0..n_samples {
                    if alpha[j].abs() > 1e-10 {
                        let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                        prediction += alpha[j] * k_val;
                    }
                }

                // Compute loss derivative
                let loss_derivative = self.config.loss.derivative(prediction, y_ranks[i]);

                // Update alpha using gradient descent
                let old_alpha = alpha[i];
                let gradient = loss_derivative + alpha[i] / self.config.c;
                alpha[i] -= current_lr * gradient;

                // Update intercept if needed
                if self.config.fit_intercept {
                    intercept -= current_lr * loss_derivative;
                }

                // Check convergence
                if (alpha[i] - old_alpha).abs() > self.config.tol {
                    converged = false;
                }
            }

            // Update learning rate
            current_lr =
                (current_lr * self.config.learning_rate_decay).max(self.config.min_learning_rate);

            if converged {
                break;
            }
        }

        // Create dummy thresholds for all-pairs approach
        let thresholds = Array1::from_vec((0..n_classes - 1).map(|i| i as Float + 0.5).collect());

        // Filter support vectors
        let support_indices: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(_, &a)| a.abs() > 1e-8)
            .map(|(i, _)| i)
            .collect();

        let final_support_vectors = if support_indices.is_empty() {
            let mut sv = Array2::zeros((1, n_features));
            sv.row_mut(0).assign(&x.row(0));
            sv
        } else {
            let mut sv = Array2::zeros((support_indices.len(), n_features));
            for (i, &idx) in support_indices.iter().enumerate() {
                sv.row_mut(i).assign(&x.row(idx));
            }
            sv
        };

        let final_alpha = if support_indices.is_empty() {
            Array1::from_vec(vec![1e-8])
        } else {
            Array1::from_vec(support_indices.iter().map(|&i| alpha[i]).collect())
        };

        Ok(OrdinalRegressionSVM {
            config: self.config,
            state: PhantomData,
            support_vectors_: Some(final_support_vectors),
            alpha_: Some(final_alpha),
            intercept_: Some(intercept),
            thresholds_: Some(thresholds),
            classes_: Some(Array1::from_vec(unique_classes.to_vec())),
            n_features_in_: Some(n_features),
            n_iter_: Some(n_iter),
            n_classes_: Some(n_classes),
        })
    }

    fn fit_one_vs_rest(
        self,
        x: &Array2<Float>,
        y_ranks: &Array1<Float>,
        unique_classes: &[usize],
        n_classes: usize,
        n_features: usize,
    ) -> Result<OrdinalRegressionSVM<Trained>> {
        // Simplified one-vs-rest implementation
        // In practice, this would train multiple binary classifiers
        self.fit_all_pairs(x, y_ranks, unique_classes, n_classes, n_features)
    }

    fn fit_cumulative(
        self,
        x: &Array2<Float>,
        y_ranks: &Array1<Float>,
        unique_classes: &[usize],
        n_classes: usize,
        n_features: usize,
    ) -> Result<OrdinalRegressionSVM<Trained>> {
        // Simplified cumulative implementation
        // In practice, this would model cumulative probabilities
        self.fit_immediate_threshold(x, y_ranks, unique_classes, n_classes, n_features)
    }
}

impl Predict<Array2<Float>, Array1<usize>> for OrdinalRegressionSVM<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<usize>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(
                "Feature mismatch: X has different number of features than training data"
                    .to_string(),
            ));
        }

        let support_vectors = self.support_vectors_.as_ref().unwrap();
        let alpha = self.alpha_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap();
        let thresholds = self.thresholds_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>,
        };

        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            // Compute ranking score
            let mut ranking_score = if self.config.fit_intercept {
                intercept
            } else {
                0.0
            };

            for j in 0..support_vectors.nrows() {
                let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                ranking_score += alpha[j] * k_val;
            }

            // Determine predicted rank based on thresholds
            let mut predicted_rank = 0;
            for k in 0..thresholds.len() {
                if ranking_score > thresholds[k] {
                    predicted_rank += 1;
                }
            }

            // Convert rank back to original class label
            predicted_rank = predicted_rank.min(classes.len() - 1);
            predictions[i] = classes[predicted_rank];
        }

        Ok(predictions)
    }
}

impl OrdinalRegressionSVM<Trained> {
    /// Get the ranking scores for given data
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(
                "Feature mismatch: X has different number of features than training data"
                    .to_string(),
            ));
        }

        let support_vectors = self.support_vectors_.as_ref().unwrap();
        let alpha = self.alpha_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap();

        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>,
        };

        let mut scores = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut score = if self.config.fit_intercept {
                intercept
            } else {
                0.0
            };

            for j in 0..support_vectors.nrows() {
                let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                score += alpha[j] * k_val;
            }

            scores[i] = score;
        }

        Ok(scores)
    }

    /// Get the learned thresholds
    pub fn thresholds(&self) -> &Array1<Float> {
        self.thresholds_.as_ref().unwrap()
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<usize> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the support vectors
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.support_vectors_.as_ref().unwrap()
    }

    /// Get the support vector coefficients (alpha values)
    pub fn alpha(&self) -> &Array1<Float> {
        self.alpha_.as_ref().unwrap()
    }

    /// Get the intercept term
    pub fn intercept(&self) -> Float {
        self.intercept_.unwrap()
    }

    /// Get the number of features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.n_classes_.unwrap()
    }

    /// Get the number of iterations performed during training
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Compute mean absolute error on ordinal scale
    pub fn mean_absolute_error(&self, x: &Array2<Float>, y_true: &Array1<usize>) -> Result<Float> {
        let y_pred = self.predict(x)?;
        let classes = self.classes();

        // Create mapping from class to ordinal position
        let mut class_to_position = HashMap::new();
        for (pos, &class) in classes.iter().enumerate() {
            class_to_position.insert(class, pos);
        }

        let mut total_error = 0.0;
        let mut count = 0;

        for (pred, true_val) in y_pred.iter().zip(y_true.iter()) {
            if let (Some(&pred_pos), Some(&true_pos)) =
                (class_to_position.get(pred), class_to_position.get(true_val))
            {
                total_error += (pred_pos as Float - true_pos as Float).abs();
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_error / count as Float)
        } else {
            Err(SklearsError::InvalidInput(
                "No valid predictions".to_string(),
            ))
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ordinal_regression_svm_creation() {
        let orsvm = OrdinalRegressionSVM::new()
            .c(2.0)
            .loss(OrdinalLoss::Absolute)
            .strategy(OrdinalStrategy::ImmediateThreshold)
            .kernel(KernelType::Linear)
            .threshold_regularization(0.05);

        assert_eq!(orsvm.config.c, 2.0);
        assert_eq!(orsvm.config.loss, OrdinalLoss::Absolute);
        assert_eq!(orsvm.config.strategy, OrdinalStrategy::ImmediateThreshold);
        assert_eq!(orsvm.config.threshold_regularization, 0.05);
    }

    #[test]
    fn test_ordinal_loss_functions() {
        let absolute = OrdinalLoss::Absolute;
        let squared = OrdinalLoss::Squared;
        let ordinal_hinge = OrdinalLoss::OrdinalHinge;

        // Test absolute loss
        assert!((absolute.loss(3.0, 1.0) - 2.0).abs() < 1e-6);
        assert!((absolute.derivative(3.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((absolute.derivative(1.0, 3.0) - (-1.0)).abs() < 1e-6);

        // Test squared loss
        assert!((squared.loss(3.0, 1.0) - 4.0).abs() < 1e-6); // (3-1)^2
        assert!((squared.derivative(3.0, 1.0) - 4.0).abs() < 1e-6); // 2*(3-1)

        // Test ordinal hinge loss
        assert!((ordinal_hinge.loss(3.0, 1.0) - 1.0).abs() < 1e-6); // |3-1| - 1 = 1
        assert!((ordinal_hinge.loss(1.5, 1.0) - 0.0).abs() < 1e-6); // |1.5-1| <= 1, so 0
    }

    #[test]
    fn test_threshold_initialization() {
        // Test uniform initialization
        let uniform_init = ThresholdInit::Uniform;
        assert_eq!(uniform_init, ThresholdInit::Uniform);

        // Test custom initialization
        let custom_thresholds = array![0.5, 1.5, 2.5];
        let custom_init = ThresholdInit::Custom(custom_thresholds.clone());
        if let ThresholdInit::Custom(ref thresholds) = custom_init {
            assert_eq!(thresholds.len(), 3);
            assert!((thresholds[0] - 0.5).abs() < 1e-6);
        } else {
            panic!("Expected Custom threshold initialization");
        }
    }

    #[test]
    #[ignore = "Slow test: trains ordinal regression SVM. Run with --ignored flag"]
    fn test_ordinal_regression_svm_training() {
        // Create ordinal data: features and ordinal classes (0, 1, 2)
        let x = array![
            [1.0, 1.0], // Class 0
            [1.5, 1.2], // Class 0
            [3.0, 3.0], // Class 1
            [3.2, 3.1], // Class 1
            [5.0, 5.0], // Class 2
            [5.1, 4.9], // Class 2
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let orsvm = OrdinalRegressionSVM::new()
            .c(1.0)
            .loss(OrdinalLoss::Absolute)
            .strategy(OrdinalStrategy::ImmediateThreshold)
            .kernel(KernelType::Linear)
            .max_iter(100)
            .learning_rate(0.01)
            .random_state(42);

        let fitted_model = orsvm.fit(&x, &y).unwrap();

        assert_eq!(fitted_model.n_features_in(), 2);
        assert_eq!(fitted_model.n_classes(), 3);
        assert!(fitted_model.n_iter() > 0);

        // Test predictions
        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // Test decision function
        let scores = fitted_model.decision_function(&x).unwrap();
        assert_eq!(scores.len(), 6);

        // Test thresholds
        let thresholds = fitted_model.thresholds();
        assert_eq!(thresholds.len(), 2); // n_classes - 1

        // Test classes
        let classes = fitted_model.classes();
        assert_eq!(classes.len(), 3);
        assert_eq!(classes[0], 0);
        assert_eq!(classes[1], 1);
        assert_eq!(classes[2], 2);

        // Test mean absolute error
        let mae = fitted_model.mean_absolute_error(&x, &y).unwrap();
        assert!(mae >= 0.0);
        assert!(mae.is_finite());

        // Predictions should be finite and within expected range
        for &pred in predictions.iter() {
            assert!(pred <= 2);
        }
        for &score in scores.iter() {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_ordinal_regression_svm_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1]; // Wrong length

        let orsvm = OrdinalRegressionSVM::new();
        let result = orsvm.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_ordinal_regression_svm_single_class() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1, 1]; // Only one class

        let orsvm = OrdinalRegressionSVM::new();
        let result = orsvm.fit(&x, &y);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Need at least 2 classes"));
    }

    #[test]
    #[ignore = "Slow test: trains ordinal regression SVM with different strategies"]
    fn test_different_ordinal_strategies() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],];
        let y = array![0, 1, 2, 3];

        // Test AllPairs strategy
        let all_pairs_svm = OrdinalRegressionSVM::new()
            .strategy(OrdinalStrategy::AllPairs)
            .max_iter(50)
            .learning_rate(0.01);

        let fitted_all_pairs = all_pairs_svm.fit(&x, &y).unwrap();
        let predictions = fitted_all_pairs.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);

        // Test OneVsRest strategy
        let ovr_svm = OrdinalRegressionSVM::new()
            .strategy(OrdinalStrategy::OneVsRest)
            .max_iter(50)
            .learning_rate(0.01);

        let fitted_ovr = ovr_svm.fit(&x, &y).unwrap();
        let predictions = fitted_ovr.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);

        // Test Cumulative strategy
        let cumulative_svm = OrdinalRegressionSVM::new()
            .strategy(OrdinalStrategy::Cumulative)
            .max_iter(50)
            .learning_rate(0.01);

        let fitted_cumulative = cumulative_svm.fit(&x, &y).unwrap();
        let predictions = fitted_cumulative.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }
}

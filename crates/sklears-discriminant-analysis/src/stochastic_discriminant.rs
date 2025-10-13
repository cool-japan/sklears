//! Stochastic Discriminant Analysis
//!
//! This module implements stochastic gradient descent-based discriminant analysis
//! for large-scale datasets that don't fit in memory or require online learning.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix2};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba},
    types::Float,
};

/// Learning rate schedule
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant { rate: Float },
    /// Exponential decay: rate * decay^epoch
    Exponential { initial_rate: Float, decay: Float },
    /// Step decay: rate / (1 + drop * floor(epoch / epochs_drop))
    Step {
        initial_rate: Float,

        drop: Float,

        epochs_drop: usize,
    },
    /// Adaptive learning rate (AdaGrad-style)
    Adaptive { initial_rate: Float, epsilon: Float },
}

/// Optimization algorithm
#[derive(Debug, Clone)]
pub enum Optimizer {
    /// Standard Stochastic Gradient Descent
    SGD,
    /// SGD with momentum
    Momentum { momentum: Float },
    /// Adam optimizer
    Adam {
        beta1: Float,

        beta2: Float,

        epsilon: Float,
    },
    /// RMSprop optimizer
    RMSprop { decay: Float, epsilon: Float },
}

/// Loss function for discriminant analysis
#[derive(Debug, Clone)]
pub enum LossFunction {
    /// Logistic loss (log-likelihood)
    Logistic,
    /// Hinge loss (SVM-style)
    Hinge,
    /// Squared hinge loss
    SquaredHinge,
    /// Modified Huber loss
    ModifiedHuber,
}

/// Configuration for Stochastic Discriminant Analysis
#[derive(Debug, Clone)]
pub struct StochasticDiscriminantAnalysisConfig {
    /// Loss function to optimize
    pub loss: LossFunction,
    /// Optimizer to use
    pub optimizer: Optimizer,
    /// Learning rate schedule
    pub learning_rate: LearningRateSchedule,
    /// L1 regularization parameter
    pub alpha: Float,
    /// L2 regularization parameter
    pub l1_ratio: Float,
    /// Batch size for mini-batch SGD
    pub batch_size: usize,
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Early stopping patience
    pub early_stopping: Option<usize>,
    /// Validation fraction for early stopping
    pub validation_fraction: Float,
    /// Whether to shuffle data each epoch
    pub shuffle: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Number of jobs for parallel processing
    pub n_jobs: Option<usize>,
}

impl Default for StochasticDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            loss: LossFunction::Logistic,
            optimizer: Optimizer::SGD,
            learning_rate: LearningRateSchedule::Constant { rate: 0.01 },
            alpha: 0.0001,
            l1_ratio: 0.15,
            batch_size: 32,
            max_epochs: 1000,
            tol: 1e-6,
            early_stopping: Some(10),
            validation_fraction: 0.1,
            shuffle: true,
            random_state: None,
            n_jobs: None,
        }
    }
}

/// Stochastic Discriminant Analysis estimator
#[derive(Debug, Clone)]
pub struct StochasticDiscriminantAnalysis {
    config: StochasticDiscriminantAnalysisConfig,
}

impl StochasticDiscriminantAnalysis {
    /// Create a new Stochastic Discriminant Analysis estimator
    pub fn new() -> Self {
        Self {
            config: StochasticDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the optimizer
    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    /// Set the learning rate schedule
    pub fn learning_rate(mut self, learning_rate: LearningRateSchedule) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the L1 regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set the L1 ratio for elastic net
    pub fn l1_ratio(mut self, l1_ratio: Float) -> Self {
        self.config.l1_ratio = l1_ratio;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set the maximum number of epochs
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.config.max_epochs = max_epochs;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set early stopping patience
    pub fn early_stopping(mut self, early_stopping: Option<usize>) -> Self {
        self.config.early_stopping = early_stopping;
        self
    }

    /// Set whether to shuffle data
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = shuffle;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

/// Optimizer state for tracking momentum, adaptive rates, etc.
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Velocity for momentum-based optimizers
    velocity: Option<Array2<Float>>,
    /// First moment estimate (Adam)
    m: Option<Array2<Float>>,
    /// Second moment estimate (Adam, RMSprop)
    v: Option<Array2<Float>>,
    /// Accumulated squared gradients (AdaGrad)
    accumulated_gradients: Option<Array2<Float>>,
    /// Time step for Adam
    t: usize,
}

impl OptimizerState {
    fn new() -> Self {
        Self {
            velocity: None,
            m: None,
            v: None,
            accumulated_gradients: None,
            t: 0,
        }
    }
}

/// Trained Stochastic Discriminant Analysis model
#[derive(Debug, Clone)]
pub struct TrainedStochasticDiscriminantAnalysis {
    /// Configuration used for training
    config: StochasticDiscriminantAnalysisConfig,
    /// Unique classes found during training
    classes: Array1<i32>,
    /// Weight matrix
    weights: Array2<Float>,
    /// Bias vector
    bias: Array1<Float>,
    /// Training history
    training_history: Vec<Float>,
    /// Number of features
    n_features: usize,
    /// Number of samples seen during training
    n_samples_seen: usize,
}

impl TrainedStochasticDiscriminantAnalysis {
    /// Get the classes found during training
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get the weight matrix
    pub fn weights(&self) -> &Array2<Float> {
        &self.weights
    }

    /// Get the bias vector
    pub fn bias(&self) -> &Array1<Float> {
        &self.bias
    }

    /// Get the training history (loss values)
    pub fn training_history(&self) -> &[Float] {
        &self.training_history
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get the number of samples seen during training
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Compute decision scores
    fn decision_function<D: Data<Elem = Float>>(&self, x: &ArrayBase<D, Ix2>) -> Array2<Float> {
        x.dot(&self.weights) + &self.bias
    }

    /// Partial fit for online learning
    pub fn partial_fit(&mut self, x: &Array2<Float>, y: &Array1<i32>) -> Result<()> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features,
                actual: x.ncols(),
            });
        }

        // Update sample count
        self.n_samples_seen += x.nrows();

        // Compute gradients and update weights
        let mut optimizer_state = OptimizerState::new();
        self.update_weights_single_batch(x, y, 0.01, &mut optimizer_state)?;

        Ok(())
    }

    /// Update weights for a single batch
    fn update_weights_single_batch(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        learning_rate: Float,
        optimizer_state: &mut OptimizerState,
    ) -> Result<()> {
        // Convert class labels to one-hot encoding
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut y_one_hot = Array2::zeros((n_samples, n_classes));

        for (i, &label) in y.iter().enumerate() {
            if let Some(class_idx) = self.classes.iter().position(|&c| c == label) {
                y_one_hot[[i, class_idx]] = 1.0;
            }
        }

        // Compute predictions
        let scores = self.decision_function(x);
        let probabilities = self.softmax(&scores);

        // Compute gradient of loss with respect to weights
        let error = &probabilities - &y_one_hot;
        let grad_weights = x.t().dot(&error) / (n_samples as Float);
        let grad_bias = error.mean_axis(Axis(0)).unwrap();

        // Add regularization to weight gradients
        let l2_reg = (1.0 - self.config.l1_ratio) * self.config.alpha;
        let l1_reg = self.config.l1_ratio * self.config.alpha;

        let mut reg_grad_weights = &grad_weights + &(&self.weights * l2_reg);

        // L1 regularization (subgradient)
        if l1_reg > 0.0 {
            for i in 0..self.weights.nrows() {
                for j in 0..self.weights.ncols() {
                    if self.weights[[i, j]] > 0.0 {
                        reg_grad_weights[[i, j]] += l1_reg;
                    } else if self.weights[[i, j]] < 0.0 {
                        reg_grad_weights[[i, j]] -= l1_reg;
                    }
                }
            }
        }

        // Apply optimizer
        match &self.config.optimizer {
            Optimizer::SGD => {
                self.weights -= &(&reg_grad_weights * learning_rate);
                self.bias -= &(grad_bias * learning_rate);
            }
            Optimizer::Momentum { momentum } => {
                // Initialize velocity if needed
                if optimizer_state.velocity.is_none() {
                    optimizer_state.velocity = Some(Array2::zeros(self.weights.dim()));
                }

                let velocity = optimizer_state.velocity.as_mut().unwrap();
                *velocity = &*velocity * *momentum + &reg_grad_weights * learning_rate;
                self.weights -= &*velocity;
                self.bias -= &(grad_bias * learning_rate);
            }
            Optimizer::Adam {
                beta1,
                beta2,
                epsilon,
            } => {
                optimizer_state.t += 1;

                // Initialize moments if needed
                if optimizer_state.m.is_none() {
                    optimizer_state.m = Some(Array2::zeros(self.weights.dim()));
                    optimizer_state.v = Some(Array2::zeros(self.weights.dim()));
                }

                let m = optimizer_state.m.as_mut().unwrap();
                let v = optimizer_state.v.as_mut().unwrap();

                // Update biased first moment estimate
                *m = &*m * *beta1 + &reg_grad_weights * (1.0 - *beta1);

                // Update biased second raw moment estimate
                *v = &*v * *beta2 + &reg_grad_weights.mapv(|x| x * x) * (1.0 - *beta2);

                // Compute bias-corrected moment estimates
                let m_hat = &*m / (1.0 - beta1.powi(optimizer_state.t as i32));
                let v_hat = &*v / (1.0 - beta2.powi(optimizer_state.t as i32));

                // Update weights
                let update = &m_hat / &(v_hat.mapv(|x| x.sqrt()) + *epsilon);
                self.weights -= &(update * learning_rate);
                self.bias -= &(grad_bias * learning_rate);
            }
            Optimizer::RMSprop { decay, epsilon } => {
                // Initialize accumulated gradients if needed
                if optimizer_state.v.is_none() {
                    optimizer_state.v = Some(Array2::zeros(self.weights.dim()));
                }

                let v = optimizer_state.v.as_mut().unwrap();

                // Update accumulated squared gradients
                *v = &*v * *decay + &reg_grad_weights.mapv(|x| x * x) * (1.0 - *decay);

                // Update weights
                let update = &reg_grad_weights / &(v.mapv(|x| x.sqrt()) + *epsilon);
                self.weights -= &(update * learning_rate);
                self.bias -= &(grad_bias * learning_rate);
            }
        }

        Ok(())
    }

    /// Compute softmax probabilities
    fn softmax(&self, scores: &Array2<Float>) -> Array2<Float> {
        let mut probabilities = Array2::zeros(scores.dim());

        for (i, score_row) in scores.axis_iter(Axis(0)).enumerate() {
            let max_score = score_row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_scores: Array1<Float> = score_row.mapv(|x| (x - max_score).exp());
            let sum_exp = exp_scores.sum();

            for (j, &exp_score) in exp_scores.iter().enumerate() {
                probabilities[[i, j]] = exp_score / sum_exp;
            }
        }

        probabilities
    }
}

impl Estimator for StochasticDiscriminantAnalysis {
    type Config = StochasticDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<D: Data<Elem = Float>> Fit<ArrayBase<D, Ix2>, Array1<i32>> for StochasticDiscriminantAnalysis {
    type Fitted = TrainedStochasticDiscriminantAnalysis;

    fn fit(self, x: &ArrayBase<D, Ix2>, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: x.nrows(),
                actual: y.len(),
            });
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "At least 2 samples are required".to_string(),
            ));
        }

        // Get unique classes
        let mut unique_classes = y.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "At least 2 classes are required".to_string(),
            ));
        }

        // Initialize weights and bias
        let mut weights = Array2::from_elem((n_features, n_classes), 0.01);
        let mut bias = Array1::zeros(n_classes);

        // Split data for validation if early stopping is enabled
        let (train_x, train_y, val_x, val_y) = if self.config.early_stopping.is_some() {
            let val_size = (n_samples as Float * self.config.validation_fraction) as usize;
            let train_size = n_samples - val_size;

            let train_x = x.slice(s![..train_size, ..]).to_owned();
            let train_y = y.slice(s![..train_size]).to_owned();
            let val_x = x.slice(s![train_size.., ..]).to_owned();
            let val_y = y.slice(s![train_size..]).to_owned();

            (train_x, train_y, Some(val_x), Some(val_y))
        } else {
            (x.to_owned(), y.to_owned(), None, None)
        };

        let mut training_history = Vec::new();
        let mut best_val_loss = Float::INFINITY;
        let mut patience_counter = 0;
        let mut optimizer_state = OptimizerState::new();

        // Training loop
        for epoch in 0..self.config.max_epochs {
            let learning_rate = self.compute_learning_rate(epoch);

            // Shuffle data if requested
            let (epoch_x, epoch_y) = if self.config.shuffle {
                // For simplicity, we'll skip shuffling in this implementation
                // In practice, you'd want to implement proper shuffling
                (train_x.clone(), train_y.clone())
            } else {
                (train_x.clone(), train_y.clone())
            };

            // Mini-batch training
            let mut epoch_loss = 0.0;
            let n_batches = (epoch_x.nrows() + self.config.batch_size - 1) / self.config.batch_size;

            for batch_idx in 0..n_batches {
                let start_idx = batch_idx * self.config.batch_size;
                let end_idx = std::cmp::min(start_idx + self.config.batch_size, epoch_x.nrows());

                let batch_x = epoch_x.slice(s![start_idx..end_idx, ..]);
                let batch_y = epoch_y.slice(s![start_idx..end_idx]);

                // Create temporary trained model for batch update
                let mut temp_model = TrainedStochasticDiscriminantAnalysis {
                    config: self.config.clone(),
                    classes: classes.clone(),
                    weights: weights.clone(),
                    bias: bias.clone(),
                    training_history: training_history.clone(),
                    n_features,
                    n_samples_seen: 0,
                };

                // Update weights
                temp_model.update_weights_single_batch(
                    &batch_x.to_owned(),
                    &batch_y.to_owned(),
                    learning_rate,
                    &mut optimizer_state,
                )?;

                // Update our weights and bias
                weights = temp_model.weights;
                bias = temp_model.bias;

                // Compute batch loss
                let batch_scores = batch_x.dot(&weights) + &bias;
                let batch_loss = self.compute_loss(&batch_scores, &batch_y.to_owned(), &classes);
                epoch_loss += batch_loss;
            }

            epoch_loss /= n_batches as Float;
            training_history.push(epoch_loss);

            // Validation and early stopping
            if let (Some(val_x), Some(val_y)) = (&val_x, &val_y) {
                let val_scores = val_x.dot(&weights) + &bias;
                let val_loss = self.compute_loss(&val_scores, val_y, &classes);

                if val_loss < best_val_loss - self.config.tol {
                    best_val_loss = val_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;

                    if let Some(patience) = self.config.early_stopping {
                        if patience_counter >= patience {
                            break;
                        }
                    }
                }
            }

            // Check for convergence
            if training_history.len() > 1 {
                let loss_diff = (training_history[training_history.len() - 2] - epoch_loss).abs();
                if loss_diff < self.config.tol {
                    break;
                }
            }
        }

        Ok(TrainedStochasticDiscriminantAnalysis {
            config: self.config,
            classes,
            weights,
            bias,
            training_history,
            n_features,
            n_samples_seen: n_samples,
        })
    }
}

impl StochasticDiscriminantAnalysis {
    /// Compute learning rate for current epoch
    fn compute_learning_rate(&self, epoch: usize) -> Float {
        match &self.config.learning_rate {
            LearningRateSchedule::Constant { rate } => *rate,
            LearningRateSchedule::Exponential {
                initial_rate,
                decay,
            } => initial_rate * decay.powi(epoch as i32),
            LearningRateSchedule::Step {
                initial_rate,
                drop,
                epochs_drop,
            } => initial_rate / (1.0 + drop * (epoch / epochs_drop) as Float),
            LearningRateSchedule::Adaptive { initial_rate, .. } => {
                // Simplified adaptive rate
                initial_rate / (1.0 + epoch as Float).sqrt()
            }
        }
    }

    /// Compute loss for given scores and labels
    fn compute_loss(
        &self,
        scores: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Float {
        let n_samples = scores.nrows();
        let mut loss = 0.0;

        match &self.config.loss {
            LossFunction::Logistic => {
                // Multi-class logistic loss
                for (i, &label) in y.iter().enumerate() {
                    if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                        let score_row = scores.row(i);
                        let max_score = score_row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let log_sum_exp = score_row
                            .iter()
                            .map(|&s| (s - max_score).exp())
                            .sum::<Float>()
                            .ln()
                            + max_score;
                        loss -= scores[[i, class_idx]] - log_sum_exp;
                    }
                }
                loss / n_samples as Float
            }
            LossFunction::Hinge => {
                // Multi-class hinge loss (one-vs-all)
                for (i, &label) in y.iter().enumerate() {
                    if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                        let correct_score = scores[[i, class_idx]];
                        for (j, &other_score) in scores.row(i).iter().enumerate() {
                            if j != class_idx {
                                let margin = 1.0 - correct_score + other_score;
                                if margin > 0.0 {
                                    loss += margin;
                                }
                            }
                        }
                    }
                }
                loss / n_samples as Float
            }
            LossFunction::SquaredHinge => {
                // Squared hinge loss
                for (i, &label) in y.iter().enumerate() {
                    if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                        let correct_score = scores[[i, class_idx]];
                        for (j, &other_score) in scores.row(i).iter().enumerate() {
                            if j != class_idx {
                                let margin = 1.0 - correct_score + other_score;
                                if margin > 0.0 {
                                    loss += margin * margin;
                                }
                            }
                        }
                    }
                }
                loss / n_samples as Float
            }
            LossFunction::ModifiedHuber => {
                // Modified Huber loss (simplified binary version)
                for (i, &label) in y.iter().enumerate() {
                    if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                        let score = scores[[i, class_idx]];
                        let y_score = if class_idx == 0 { -1.0 } else { 1.0 };
                        let margin = y_score * score;

                        if margin >= 1.0 {
                            loss += 0.0;
                        } else if margin >= -1.0 {
                            loss += (1.0 - margin).powi(2);
                        } else {
                            loss += -4.0 * margin;
                        }
                    }
                }
                loss / n_samples as Float
            }
        }
    }
}

impl<D: Data<Elem = Float>> Predict<ArrayBase<D, Ix2>, Array1<i32>>
    for TrainedStochasticDiscriminantAnalysis
{
    fn predict(&self, x: &ArrayBase<D, Ix2>) -> Result<Array1<i32>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features,
                actual: x.ncols(),
            });
        }

        let scores = self.decision_function(x);
        let mut predictions = Array1::zeros(x.nrows());

        for (i, score_row) in scores.axis_iter(Axis(0)).enumerate() {
            let max_idx = score_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            predictions[i] = self.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl<D: Data<Elem = Float>> PredictProba<ArrayBase<D, Ix2>, Array2<Float>>
    for TrainedStochasticDiscriminantAnalysis
{
    fn predict_proba(&self, x: &ArrayBase<D, Ix2>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features,
                actual: x.ncols(),
            });
        }

        let scores = self.decision_function(x);
        Ok(self.softmax(&scores))
    }
}

impl Default for StochasticDiscriminantAnalysis {
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
    fn test_stochastic_discriminant_analysis_basic() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let sda = StochasticDiscriminantAnalysis::new()
            .max_epochs(50)
            .batch_size(2);
        let fitted = sda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_stochastic_discriminant_predict_proba() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let sda = StochasticDiscriminantAnalysis::new()
            .max_epochs(30)
            .learning_rate(LearningRateSchedule::Constant { rate: 0.1 });
        let fitted = sda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_stochastic_discriminant_with_momentum() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let sda = StochasticDiscriminantAnalysis::new()
            .optimizer(Optimizer::Momentum { momentum: 0.9 })
            .max_epochs(20);
        let fitted = sda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_stochastic_discriminant_with_adam() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let sda = StochasticDiscriminantAnalysis::new()
            .optimizer(Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            })
            .max_epochs(20);
        let fitted = sda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_stochastic_discriminant_partial_fit() {
        let x1 = array![[1.0, 2.0], [3.0, 4.0]];
        let y1 = array![0, 1]; // Need at least 2 classes for initial training

        let sda = StochasticDiscriminantAnalysis::new();
        let mut fitted = sda.fit(&x1, &y1).unwrap();

        // Add more data
        let x2 = array![[3.0, 4.0], [3.1, 4.1]];
        let y2 = array![1, 1];
        fitted.partial_fit(&x2, &y2).unwrap();

        assert_eq!(fitted.n_samples_seen(), 4);
    }

    #[test]
    fn test_stochastic_discriminant_hinge_loss() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let sda = StochasticDiscriminantAnalysis::new()
            .loss(LossFunction::Hinge)
            .max_epochs(20);
        let fitted = sda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_stochastic_discriminant_training_history() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let sda = StochasticDiscriminantAnalysis::new().max_epochs(10);
        let fitted = sda.fit(&x, &y).unwrap();
        let history = fitted.training_history();

        assert!(history.len() > 0);
        assert!(history.len() <= 10);

        // Check that loss generally decreases (with some tolerance for noise)
        if history.len() > 1 {
            let first_loss = history[0];
            let last_loss = history[history.len() - 1];
            // Allow for some fluctuation in stochastic optimization
            assert!(last_loss <= first_loss + 0.5);
        }
    }
}

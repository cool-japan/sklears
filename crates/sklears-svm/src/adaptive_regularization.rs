//! Adaptive Regularization Methods for SVMs
//!
//! This module implements adaptive regularization techniques that automatically
//! adjust regularization parameters during training for improved performance.
//!
//! Methods included:
//! - Adaptive C selection based on cross-validation
//! - Learning curve based regularization
//! - Bayesian regularization parameter optimization
//! - Early stopping with validation loss

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
};
use std::fmt;
use std::marker::PhantomData;

/// Errors that can occur during adaptive regularization
#[derive(Debug, Clone)]
pub enum AdaptiveRegularizationError {
    InvalidInput(String),
    OptimizationError(String),
    ConvergenceError(String),
}

impl fmt::Display for AdaptiveRegularizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdaptiveRegularizationError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            AdaptiveRegularizationError::OptimizationError(msg) => {
                write!(f, "Optimization error: {msg}")
            }
            AdaptiveRegularizationError::ConvergenceError(msg) => {
                write!(f, "Convergence error: {msg}")
            }
        }
    }
}

impl std::error::Error for AdaptiveRegularizationError {}

/// Adaptive regularization strategy
#[derive(Debug, Clone)]
pub enum AdaptiveStrategy {
    /// Cross-validation based C selection
    CrossValidation {
        folds: usize,
        c_range: (f64, f64),
        n_candidates: usize,
    },
    /// Learning curve based adaptation
    LearningCurve {
        patience: usize,
        min_improvement: f64,
        c_factor: f64,
    },
    /// Bayesian optimization of regularization
    BayesianOptimization {
        n_iterations: usize,
        acquisition_function: AcquisitionFunction,
    },
    /// Early stopping with validation
    EarlyStopping {
        patience: usize,
        validation_fraction: f64,
        min_delta: f64,
    },
}

/// Acquisition function for Bayesian optimization
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    UpperConfidenceBound { beta: f64 },
    ProbabilityOfImprovement,
}

/// Configuration for adaptive regularization
#[derive(Debug, Clone)]
pub struct AdaptiveRegularizationConfig {
    /// Adaptive strategy to use
    pub strategy: AdaptiveStrategy,
    /// Initial regularization parameter
    pub initial_c: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for AdaptiveRegularizationConfig {
    fn default() -> Self {
        Self {
            strategy: AdaptiveStrategy::CrossValidation {
                folds: 5,
                c_range: (0.001, 1000.0),
                n_candidates: 20,
            },
            initial_c: 1.0,
            max_iter: 100,
            tol: 1e-6,
            verbose: false,
        }
    }
}

/// Adaptive regularization SVM wrapper
#[derive(Debug, Clone)]
pub struct AdaptiveSVM<State = Untrained> {
    config: AdaptiveRegularizationConfig,
    state: PhantomData<State>,
    // Final regularization parameter
    optimal_c: Option<f64>,
    // Base SVM configuration
    base_svm_config: Option<crate::svc::SvcConfig>,
    // Trained base SVM
    trained_svm: Option<crate::svc::SVC<Trained>>,
    // Optimization history
    optimization_history: Option<Vec<(f64, f64)>>, // (C, validation_score)
}

impl Default for AdaptiveSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveSVM<Untrained> {
    /// Create a new adaptive SVM
    pub fn new() -> Self {
        Self {
            config: AdaptiveRegularizationConfig::default(),
            state: PhantomData,
            optimal_c: None,
            base_svm_config: None,
            trained_svm: None,
            optimization_history: None,
        }
    }

    /// Set the adaptive strategy
    pub fn with_strategy(mut self, strategy: AdaptiveStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set initial C parameter
    pub fn with_initial_c(mut self, c: f64) -> Self {
        self.config.initial_c = c;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set base SVM configuration
    pub fn with_svm_config(mut self, config: crate::svc::SvcConfig) -> Self {
        self.base_svm_config = Some(config);
        self
    }
}

impl Fit<Array2<f64>, Array1<f64>> for AdaptiveSVM<Untrained> {
    type Fitted = AdaptiveSVM<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match number of labels".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string(),
            ));
        }

        // Find optimal regularization parameter
        let (optimal_c, optimization_history) = self.optimize_regularization(x, y)?;

        // Train final SVM with optimal C
        let mut base_config = self.base_svm_config.unwrap_or_default();
        base_config.c = optimal_c;

        let base_svm = crate::svc::SVC::new()
            .c(optimal_c)
            .tol(base_config.tol)
            .max_iter(base_config.max_iter);

        //         use sklears_core::traits::Fit;
        let trained_svm = base_svm.fit(x, y)?;

        if self.config.verbose {
            println!("Adaptive regularization found optimal C = {:.6}", optimal_c);
        }

        Ok(AdaptiveSVM {
            config: self.config,
            state: PhantomData,
            optimal_c: Some(optimal_c),
            base_svm_config: Some(base_config),
            trained_svm: Some(trained_svm),
            optimization_history: Some(optimization_history),
        })
    }
}

impl AdaptiveSVM<Untrained> {
    /// Optimize regularization parameter based on strategy
    fn optimize_regularization(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(f64, Vec<(f64, f64)>)> {
        match &self.config.strategy {
            AdaptiveStrategy::CrossValidation {
                folds,
                c_range,
                n_candidates,
            } => self.cross_validation_optimization(x, y, *folds, *c_range, *n_candidates),
            AdaptiveStrategy::LearningCurve {
                patience,
                min_improvement,
                c_factor,
            } => self.learning_curve_optimization(x, y, *patience, *min_improvement, *c_factor),
            AdaptiveStrategy::BayesianOptimization {
                n_iterations,
                acquisition_function,
            } => self.bayesian_optimization(x, y, *n_iterations, acquisition_function),
            AdaptiveStrategy::EarlyStopping {
                patience,
                validation_fraction,
                min_delta,
            } => {
                self.early_stopping_optimization(x, y, *patience, *validation_fraction, *min_delta)
            }
        }
    }

    /// Cross-validation based C optimization
    fn cross_validation_optimization(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        folds: usize,
        c_range: (f64, f64),
        n_candidates: usize,
    ) -> Result<(f64, Vec<(f64, f64)>)> {
        let (c_min, c_max) = c_range;
        let mut best_c = self.config.initial_c;
        let mut best_score = f64::NEG_INFINITY;
        let mut history = Vec::new();

        // Generate C candidates (log scale)
        let log_c_min = c_min.ln();
        let log_c_max = c_max.ln();
        let step = (log_c_max - log_c_min) / (n_candidates - 1) as f64;

        for i in 0..n_candidates {
            let log_c = log_c_min + i as f64 * step;
            let c = log_c.exp();

            // Perform k-fold cross-validation
            let cv_score = self.cross_validate(x, y, c, folds)?;
            history.push((c, cv_score));

            if self.config.verbose {
                println!("C = {:.6}: CV Score = {:.6}", c, cv_score);
            }

            if cv_score > best_score {
                best_score = cv_score;
                best_c = c;
            }
        }

        Ok((best_c, history))
    }

    /// Learning curve based optimization
    fn learning_curve_optimization(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        patience: usize,
        min_improvement: f64,
        c_factor: f64,
    ) -> Result<(f64, Vec<(f64, f64)>)> {
        let mut current_c = self.config.initial_c;
        let mut best_c = current_c;
        let mut best_score = f64::NEG_INFINITY;
        let mut history = Vec::new();
        let mut patience_counter = 0;

        for iteration in 0..self.config.max_iter {
            // Evaluate current C
            let score = self.cross_validate(x, y, current_c, 3)?; // 3-fold CV for speed
            history.push((current_c, score));

            if self.config.verbose {
                println!(
                    "Iteration {}: C = {:.6}, Score = {:.6}",
                    iteration, current_c, score
                );
            }

            // Check for improvement
            if score > best_score + min_improvement {
                best_score = score;
                best_c = current_c;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            // Early stopping
            if patience_counter >= patience {
                if self.config.verbose {
                    println!("Early stopping: no improvement for {patience} iterations");
                }
                break;
            }

            // Adaptive C adjustment
            if score < best_score {
                current_c *= c_factor; // Increase regularization
            } else {
                current_c /= c_factor; // Decrease regularization
            }

            // Keep C within reasonable bounds
            current_c = current_c.clamp(1e-6, 1e6);
        }

        Ok((best_c, history))
    }

    /// Simplified Bayesian optimization
    fn bayesian_optimization(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        n_iterations: usize,
        _acquisition_function: &AcquisitionFunction,
    ) -> Result<(f64, Vec<(f64, f64)>)> {
        let mut history = Vec::new();
        let mut best_c = self.config.initial_c;

        // Initial evaluation
        let initial_score = self.cross_validate(x, y, self.config.initial_c, 5)?;
        history.push((self.config.initial_c, initial_score));
        let mut best_score = initial_score;

        // Simple grid search (simplified Bayesian optimization)
        let c_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];

        for (iteration, &c) in c_candidates.iter().enumerate().take(n_iterations) {
            if !history.iter().any(|(prev_c, _)| (prev_c - c).abs() < 1e-8) {
                let score = self.cross_validate(x, y, c, 5)?;
                history.push((c, score));

                if self.config.verbose {
                    println!(
                        "Bayesian Opt Iteration {}: C = {:.6}, Score = {:.6}",
                        iteration, c, score
                    );
                }

                if score > best_score {
                    best_score = score;
                    best_c = c;
                }
            }
        }

        Ok((best_c, history))
    }

    /// Early stopping optimization with validation split
    fn early_stopping_optimization(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        patience: usize,
        validation_fraction: f64,
        min_delta: f64,
    ) -> Result<(f64, Vec<(f64, f64)>)> {
        let n_samples = x.nrows();
        let val_size = (n_samples as f64 * validation_fraction) as usize;
        let train_size = n_samples - val_size;

        // Simple train/validation split
        let x_train = x
            .slice(scirs2_core::ndarray::s![..train_size, ..])
            .to_owned();
        let y_train = y.slice(scirs2_core::ndarray::s![..train_size]).to_owned();
        let x_val = x
            .slice(scirs2_core::ndarray::s![train_size.., ..])
            .to_owned();
        let y_val = y.slice(scirs2_core::ndarray::s![train_size..]).to_owned();

        let mut current_c = self.config.initial_c;
        let mut best_c = current_c;
        let mut best_score = f64::NEG_INFINITY;
        let mut history = Vec::new();
        let mut patience_counter = 0;

        for iteration in 0..self.config.max_iter {
            // Train on training set, evaluate on validation set
            let base_svm = crate::svc::SVC::new().c(current_c).max_iter(100);

            use sklears_core::traits::Predict;
            let trained_svm = base_svm.fit(&x_train, &y_train)?;
            let predictions = trained_svm.predict(&x_val)?;

            // Calculate accuracy
            let correct = predictions
                .iter()
                .zip(y_val.iter())
                .filter(|(&pred, &true_label)| (pred - true_label).abs() < 1e-6)
                .count();
            let accuracy = correct as f64 / y_val.len() as f64;

            history.push((current_c, accuracy));

            if self.config.verbose {
                println!(
                    "Early Stop Iteration {}: C = {:.6}, Val Accuracy = {:.6}",
                    iteration, current_c, accuracy
                );
            }

            // Check for improvement
            if accuracy > best_score + min_delta {
                best_score = accuracy;
                best_c = current_c;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            // Early stopping
            if patience_counter >= patience {
                if self.config.verbose {
                    println!("Early stopping: no improvement for {patience} iterations");
                }
                break;
            }

            // Simple search pattern
            if iteration % 2 == 0 {
                current_c *= 2.0; // Try higher regularization
            } else {
                current_c /= 4.0; // Try lower regularization
            }

            current_c = current_c.clamp(1e-6, 1e6);
        }

        Ok((best_c, history))
    }

    /// Perform k-fold cross-validation
    fn cross_validate(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        c: f64,
        folds: usize,
    ) -> Result<f64> {
        let n_samples = x.nrows();
        let fold_size = n_samples / folds;
        let mut scores = Vec::new();

        for fold in 0..folds {
            let val_start = fold * fold_size;
            let val_end = if fold == folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/validation split
            let mut train_indices = Vec::new();
            let mut val_indices = Vec::new();

            for i in 0..n_samples {
                if i >= val_start && i < val_end {
                    val_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            // Extract training and validation data
            let x_train = self.extract_rows(x, &train_indices);
            let y_train = self.extract_elements(y, &train_indices);
            let x_val = self.extract_rows(x, &val_indices);
            let y_val = self.extract_elements(y, &val_indices);

            // Train SVM
            let base_svm = crate::svc::SVC::new().c(c).max_iter(100);

            use sklears_core::traits::Predict;
            let trained_svm = base_svm.fit(&x_train, &y_train)?;
            let predictions = trained_svm.predict(&x_val)?;

            // Calculate accuracy
            let correct = predictions
                .iter()
                .zip(y_val.iter())
                .filter(|(&pred, &true_label)| (pred - true_label).abs() < 1e-6)
                .count();
            let accuracy = correct as f64 / y_val.len() as f64;
            scores.push(accuracy);
        }

        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Extract rows from array
    fn extract_rows(&self, array: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let mut result = Array2::zeros((indices.len(), array.ncols()));
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            result.row_mut(new_idx).assign(&array.row(old_idx));
        }
        result
    }

    /// Extract elements from array
    fn extract_elements(&self, array: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
        Array1::from_vec(indices.iter().map(|&i| array[i]).collect())
    }
}

impl Predict<Array2<f64>, Array1<f64>> for AdaptiveSVM<Trained> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let trained_svm = self.trained_svm.as_ref().unwrap();
        //         use sklears_core::traits::Predict;
        trained_svm.predict(x)
    }
}

impl AdaptiveSVM<Trained> {
    /// Get the optimal regularization parameter
    pub fn optimal_c(&self) -> f64 {
        *self.optimal_c.as_ref().unwrap()
    }

    /// Get the optimization history
    pub fn optimization_history(&self) -> &[(f64, f64)] {
        self.optimization_history.as_ref().unwrap()
    }

    /// Get the underlying trained SVM
    pub fn base_svm(&self) -> &crate::svc::SVC<Trained> {
        self.trained_svm.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn create_test_data() -> (Array2<f64>, Array1<f64>) {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 5.0],
            [1.5, 2.5],
            [2.5, 3.5],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        (x, y)
    }

    #[test]
    fn test_adaptive_svm_creation() {
        let adaptive_svm = AdaptiveSVM::new()
            .with_strategy(AdaptiveStrategy::CrossValidation {
                folds: 3,
                c_range: (0.1, 10.0),
                n_candidates: 5,
            })
            .with_initial_c(1.0)
            .verbose(false);

        assert!(matches!(
            adaptive_svm.config.strategy,
            AdaptiveStrategy::CrossValidation { .. }
        ));
        assert_eq!(adaptive_svm.config.initial_c, 1.0);
    }

    #[test]
    #[ignore]
    fn test_cross_validation_strategy() {
        let (x, y) = create_test_data();

        let adaptive_svm = AdaptiveSVM::new()
            .with_strategy(AdaptiveStrategy::CrossValidation {
                folds: 3,
                c_range: (0.1, 10.0),
                n_candidates: 5,
            })
            .verbose(false);

        use sklears_core::traits::Fit;
        let result = adaptive_svm.fit(&x, &y);
        assert!(result.is_ok());

        let trained_model = result.unwrap();
        assert!(trained_model.optimal_c.is_some());
        assert!(trained_model.optimization_history.is_some());
    }

    #[test]
    #[ignore]
    fn test_early_stopping_strategy() {
        let (x, y) = create_test_data();

        let adaptive_svm = AdaptiveSVM::new()
            .with_strategy(AdaptiveStrategy::EarlyStopping {
                patience: 3,
                validation_fraction: 0.3,
                min_delta: 0.01,
            })
            .verbose(false);

        use sklears_core::traits::Fit;
        let result = adaptive_svm.fit(&x, &y);
        assert!(result.is_ok());

        let trained_model = result.unwrap();
        assert!(trained_model.optimal_c.is_some());
    }

    #[test]
    #[ignore]
    fn test_adaptive_svm_predict() {
        let (x, y) = create_test_data();

        let adaptive_svm = AdaptiveSVM::new()
            .with_strategy(AdaptiveStrategy::CrossValidation {
                folds: 2,
                c_range: (0.1, 10.0),
                n_candidates: 3,
            })
            .verbose(false);

        use sklears_core::traits::Predict;
        let trained_model = adaptive_svm.fit(&x, &y).unwrap();

        let predictions = trained_model.predict(&x);
        assert!(predictions.is_ok());

        let pred_labels = predictions.unwrap();
        assert_eq!(pred_labels.len(), x.nrows());
    }

    #[test]
    fn test_invalid_input() {
        let x = Array2::zeros((5, 2));
        let y = Array1::zeros(6); // Wrong number of samples

        let adaptive_svm = AdaptiveSVM::new();
        use sklears_core::traits::Fit;
        let result = adaptive_svm.fit(&x, &y);
        assert!(result.is_err());
    }
}

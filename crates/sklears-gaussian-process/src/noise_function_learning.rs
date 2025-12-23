//! Advanced Noise Function Learning for Heteroscedastic Gaussian Processes
//!
//! This module implements sophisticated noise function learning techniques including:
//! - Automatic noise function selection based on data characteristics
//! - Hyperparameter optimization for noise function kernels
//! - Ensemble noise functions that combine multiple noise models
//! - Adaptive regularization and model selection
//!
//! # Mathematical Background
//!
//! Advanced noise function learning goes beyond simple parameter updates to include:
//! 1. **Model Selection**: Choosing the best noise function type using information criteria
//! 2. **Hyperparameter Optimization**: Optimizing kernel parameters for GP-based noise functions
//! 3. **Ensemble Methods**: Combining multiple noise functions with learned weights
//! 4. **Adaptive Regularization**: Adjusting regularization based on data characteristics
//!
//! # Example
//!
//! ```rust
//! use sklears_gaussian_process::noise_function_learning::{
//!     AutomaticNoiseFunctionSelector, EnsembleNoiseFunction
//! };
//! use sklears_gaussian_process::kernels::RBF;
//! use scirs2_core::ndarray::array;
//!
//! let x_train = array![[0.0], [1.0], [2.0], [3.0]];
//! let y_train = array![0.0, 1.0, 2.5, 4.2];
//!
//! // Automatic noise function selection
//! let selector = AutomaticNoiseFunctionSelector::new()
//!     .add_candidate_constant()
//!     .add_candidate_linear()
//!     .add_candidate_polynomial(2)
//!     .add_candidate_gaussian_process(Box::new(RBF::new(1.0)), 5);
//!
//! let best_noise_fn = selector.select_best(&x_train, &y_train).unwrap();
//! ```

use crate::heteroscedastic::{LearnableNoiseFunction, NoiseFunction};
use crate::kernels::Kernel;
use scirs2_core::ndarray::{Array1, Array2, Axis};
// SciRS2 Policy
use sklears_core::error::{Result as SklResult, SklearsError};

/// Information criteria for model selection
#[derive(Debug, Clone, Copy)]
pub enum InformationCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Hannan-Quinn Information Criterion
    HQC,
}

/// Result of noise function evaluation
#[derive(Debug, Clone)]
pub struct NoiseFunctionEvaluation {
    pub noise_function: LearnableNoiseFunction,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub hqc: f64,
    pub rmse: f64,
    pub n_parameters: usize,
}

/// Automatic noise function selector that evaluates multiple candidates
#[derive(Debug, Clone)]
pub struct AutomaticNoiseFunctionSelector {
    candidates: Vec<LearnableNoiseFunction>,
    criterion: InformationCriterion,
    max_iter: usize,
    learning_rate: f64,
    cross_validation_folds: Option<usize>,
    regularization_strength: f64,
}

impl AutomaticNoiseFunctionSelector {
    /// Create a new automatic noise function selector
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            criterion: InformationCriterion::BIC,
            max_iter: 100,
            learning_rate: 0.01,
            cross_validation_folds: None,
            regularization_strength: 1e-6,
        }
    }

    /// Set the information criterion for model selection
    pub fn criterion(mut self, criterion: InformationCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set maximum iterations for noise function training
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set learning rate for noise function training
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Enable cross-validation for model selection
    pub fn cross_validation_folds(mut self, folds: usize) -> Self {
        self.cross_validation_folds = Some(folds);
        self
    }

    /// Set regularization strength
    pub fn regularization_strength(mut self, strength: f64) -> Self {
        self.regularization_strength = strength;
        self
    }

    /// Add a constant noise function candidate
    pub fn add_candidate_constant(mut self) -> Self {
        self.candidates.push(LearnableNoiseFunction::constant(0.1));
        self
    }

    /// Add a linear noise function candidate
    pub fn add_candidate_linear(mut self) -> Self {
        self.candidates
            .push(LearnableNoiseFunction::linear(0.1, 0.01));
        self
    }

    /// Add a polynomial noise function candidate
    pub fn add_candidate_polynomial(mut self, degree: usize) -> Self {
        let mut coeffs = vec![0.1]; // Constant term
        for _ in 1..=degree {
            coeffs.push(0.01); // Higher order terms
        }
        self.candidates
            .push(LearnableNoiseFunction::polynomial(coeffs));
        self
    }

    /// Add a Gaussian process noise function candidate
    pub fn add_candidate_gaussian_process(
        mut self,
        kernel: Box<dyn Kernel>,
        n_inducing: usize,
    ) -> Self {
        self.candidates
            .push(LearnableNoiseFunction::gaussian_process(kernel, n_inducing));
        self
    }

    /// Add a neural network noise function candidate
    pub fn add_candidate_neural_network(
        mut self,
        input_dim: usize,
        hidden_sizes: Vec<usize>,
    ) -> Self {
        self.candidates.push(LearnableNoiseFunction::neural_network(
            input_dim,
            hidden_sizes,
        ));
        self
    }

    /// Select the best noise function based on the specified criterion
    pub fn select_best(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> SklResult<LearnableNoiseFunction> {
        if self.candidates.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No candidate noise functions specified".to_string(),
            ));
        }

        let mut best_evaluation: Option<NoiseFunctionEvaluation> = None;

        for candidate in &self.candidates {
            let evaluation = if let Some(folds) = self.cross_validation_folds {
                self.evaluate_with_cross_validation(candidate.clone(), x, y, folds)?
            } else {
                self.evaluate_noise_function(candidate.clone(), x, y)?
            };

            if let Some(ref best) = best_evaluation {
                let is_better = match self.criterion {
                    InformationCriterion::AIC => evaluation.aic < best.aic,
                    InformationCriterion::BIC => evaluation.bic < best.bic,
                    InformationCriterion::HQC => evaluation.hqc < best.hqc,
                };

                if is_better {
                    best_evaluation = Some(evaluation);
                }
            } else {
                best_evaluation = Some(evaluation);
            }
        }

        Ok(best_evaluation.unwrap().noise_function)
    }

    /// Evaluate a noise function on given data
    fn evaluate_noise_function(
        &self,
        mut noise_fn: LearnableNoiseFunction,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> SklResult<NoiseFunctionEvaluation> {
        // Train the noise function
        for _iter in 0..self.max_iter {
            // Compute residuals (simplified - assume zero mean for this evaluation)
            let residuals = y.clone();
            noise_fn.update_parameters(x, &residuals, self.learning_rate)?;
        }

        // Compute noise predictions
        let noise_predictions = noise_fn.compute_noise(x)?;

        // Compute residuals and log likelihood
        let residuals: Array1<f64> = y.iter().map(|&yi| yi * yi).collect(); // Squared residuals
        let mut log_likelihood = 0.0;

        for i in 0..y.len() {
            let noise_var = noise_predictions[i].max(1e-12);
            log_likelihood += -0.5
                * (residuals[i] / noise_var + noise_var.ln() + (2.0 * std::f64::consts::PI).ln());
        }

        // Add regularization penalty
        let params = noise_fn.get_parameters();
        let regularization_penalty =
            self.regularization_strength * params.iter().map(|&p| p * p).sum::<f64>();
        log_likelihood -= regularization_penalty;

        let n_data = y.len() as f64;
        let n_params = params.len() as f64;

        // Compute information criteria
        let aic = -2.0 * log_likelihood + 2.0 * n_params;
        let bic = -2.0 * log_likelihood + n_params * n_data.ln();
        let hqc = -2.0 * log_likelihood + 2.0 * n_params * n_data.ln().ln();

        // Compute RMSE
        let mut rmse = 0.0;
        for i in 0..y.len() {
            let predicted_noise = noise_predictions[i].sqrt();
            let actual_abs_residual = y[i].abs();
            rmse += (predicted_noise - actual_abs_residual).powi(2);
        }
        rmse = (rmse / n_data).sqrt();

        Ok(NoiseFunctionEvaluation {
            noise_function: noise_fn,
            log_likelihood,
            aic,
            bic,
            hqc,
            rmse,
            n_parameters: params.len(),
        })
    }

    /// Evaluate a noise function using cross-validation
    fn evaluate_with_cross_validation(
        &self,
        noise_fn: LearnableNoiseFunction,
        x: &Array2<f64>,
        y: &Array1<f64>,
        folds: usize,
    ) -> SklResult<NoiseFunctionEvaluation> {
        let n_data = x.nrows();
        let fold_size = n_data / folds;
        let mut total_log_likelihood = 0.0;
        let mut total_rmse = 0.0;

        for fold in 0..folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == folds - 1 {
                n_data
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test splits
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for i in 0..n_data {
                if i >= start_idx && i < end_idx {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            // Extract train and test data
            let x_train = x.select(Axis(0), &train_indices);
            let y_train = y.select(Axis(0), &train_indices);
            let x_test = x.select(Axis(0), &test_indices);
            let y_test = y.select(Axis(0), &test_indices);

            // Train noise function on training data
            let mut fold_noise_fn = noise_fn.clone();
            for _iter in 0..self.max_iter {
                let residuals = y_train.clone();
                fold_noise_fn.update_parameters(&x_train, &residuals, self.learning_rate)?;
            }

            // Evaluate on test data
            let test_noise_predictions = fold_noise_fn.compute_noise(&x_test)?;

            for i in 0..y_test.len() {
                let noise_var = test_noise_predictions[i].max(1e-12);
                let residual_sq = y_test[i] * y_test[i];
                total_log_likelihood += -0.5
                    * (residual_sq / noise_var
                        + noise_var.ln()
                        + (2.0 * std::f64::consts::PI).ln());

                let predicted_noise = noise_var.sqrt();
                let actual_abs_residual = y_test[i].abs();
                total_rmse += (predicted_noise - actual_abs_residual).powi(2);
            }
        }

        // Average across folds
        total_log_likelihood /= n_data as f64;
        total_rmse = (total_rmse / n_data as f64).sqrt();

        // Train final model on full data
        let mut final_noise_fn = noise_fn;
        for _iter in 0..self.max_iter {
            let residuals = y.clone();
            final_noise_fn.update_parameters(x, &residuals, self.learning_rate)?;
        }

        let params = final_noise_fn.get_parameters();
        let n_data = y.len() as f64;
        let n_params = params.len() as f64;

        // Compute information criteria using CV log likelihood
        let aic = -2.0 * total_log_likelihood + 2.0 * n_params;
        let bic = -2.0 * total_log_likelihood + n_params * n_data.ln();
        let hqc = -2.0 * total_log_likelihood + 2.0 * n_params * n_data.ln().ln();

        Ok(NoiseFunctionEvaluation {
            noise_function: final_noise_fn,
            log_likelihood: total_log_likelihood,
            aic,
            bic,
            hqc,
            rmse: total_rmse,
            n_parameters: params.len(),
        })
    }

    /// Get all evaluations for analysis
    pub fn evaluate_all(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> SklResult<Vec<NoiseFunctionEvaluation>> {
        let mut evaluations = Vec::new();

        for candidate in &self.candidates {
            let evaluation = if let Some(folds) = self.cross_validation_folds {
                self.evaluate_with_cross_validation(candidate.clone(), x, y, folds)?
            } else {
                self.evaluate_noise_function(candidate.clone(), x, y)?
            };
            evaluations.push(evaluation);
        }

        Ok(evaluations)
    }
}

/// Ensemble noise function that combines multiple noise functions
#[derive(Debug, Clone)]
pub struct EnsembleNoiseFunction {
    noise_functions: Vec<LearnableNoiseFunction>,
    weights: Array1<f64>,
    combination_method: CombinationMethod,
}

/// Method for combining ensemble predictions
#[derive(Debug, Clone, Copy)]
pub enum CombinationMethod {
    /// Weighted average
    WeightedAverage,
    /// Weighted geometric mean
    WeightedGeometricMean,
    /// Minimum variance combination
    MinimumVariance,
}

impl EnsembleNoiseFunction {
    /// Create a new ensemble noise function
    pub fn new(noise_functions: Vec<LearnableNoiseFunction>) -> Self {
        let n_functions = noise_functions.len();
        let weights = Array1::from_elem(n_functions, 1.0 / n_functions as f64);

        Self {
            noise_functions,
            weights,
            combination_method: CombinationMethod::WeightedAverage,
        }
    }

    /// Set the combination method
    pub fn combination_method(mut self, method: CombinationMethod) -> Self {
        self.combination_method = method;
        self
    }

    /// Set custom weights
    pub fn weights(mut self, weights: Array1<f64>) -> Self {
        // Normalize weights
        let sum = weights.sum();
        self.weights = weights / sum;
        self
    }

    /// Learn optimal weights based on data
    pub fn learn_weights(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        max_iter: usize,
    ) -> SklResult<()> {
        let n_functions = self.noise_functions.len();

        // Initialize weights uniformly
        self.weights = Array1::from_elem(n_functions, 1.0 / n_functions as f64);

        let learning_rate = 0.01;

        for _iter in 0..max_iter {
            // Compute predictions from each noise function
            let mut predictions = Vec::new();
            for noise_fn in &self.noise_functions {
                predictions.push(noise_fn.compute_noise(x)?);
            }

            // Compute ensemble prediction
            let ensemble_pred = self.combine_predictions(&predictions)?;

            // Compute residuals (simplified)
            let target_noise: Array1<f64> = y.iter().map(|&yi| yi * yi).collect();

            // Update weights using gradient descent
            let mut weight_gradients = Array1::zeros(n_functions);

            for i in 0..x.nrows() {
                let error = ensemble_pred[i] - target_noise[i];

                for j in 0..n_functions {
                    let pred_diff = predictions[j][i] - ensemble_pred[i];
                    weight_gradients[j] += error * pred_diff;
                }
            }

            // Normalize gradients
            weight_gradients /= x.nrows() as f64;

            // Update weights
            for j in 0..n_functions {
                self.weights[j] -= learning_rate * weight_gradients[j];
            }

            // Ensure weights are positive and normalized
            for weight in self.weights.iter_mut() {
                *weight = weight.max(1e-8);
            }
            let sum = self.weights.sum();
            self.weights /= sum;
        }

        Ok(())
    }

    /// Combine predictions from multiple noise functions
    fn combine_predictions(&self, predictions: &[Array1<f64>]) -> SklResult<Array1<f64>> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions to combine".to_string(),
            ));
        }

        let n_points = predictions[0].len();
        let mut combined = Array1::zeros(n_points);

        match self.combination_method {
            CombinationMethod::WeightedAverage => {
                for i in 0..n_points {
                    for (j, pred) in predictions.iter().enumerate() {
                        combined[i] += self.weights[j] * pred[i];
                    }
                }
            }
            CombinationMethod::WeightedGeometricMean => {
                for i in 0..n_points {
                    let mut log_sum = 0.0;
                    for (j, pred) in predictions.iter().enumerate() {
                        log_sum += self.weights[j] * pred[i].max(1e-12).ln();
                    }
                    combined[i] = log_sum.exp();
                }
            }
            CombinationMethod::MinimumVariance => {
                // Simplified minimum variance combination
                for i in 0..n_points {
                    let mut weighted_sum = 0.0;
                    let mut weight_sum = 0.0;

                    for (_j, pred) in predictions.iter().enumerate() {
                        let variance = pred[i].max(1e-12);
                        let inv_var_weight = 1.0 / variance;
                        weighted_sum += inv_var_weight * pred[i];
                        weight_sum += inv_var_weight;
                    }

                    combined[i] = weighted_sum / weight_sum;
                }
            }
        }

        Ok(combined)
    }
}

impl NoiseFunction for EnsembleNoiseFunction {
    fn compute_noise(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        let mut predictions = Vec::new();
        for noise_fn in &self.noise_functions {
            predictions.push(noise_fn.compute_noise(x)?);
        }
        self.combine_predictions(&predictions)
    }

    fn update_parameters(
        &mut self,
        x: &Array2<f64>,
        residuals: &Array1<f64>,
        learning_rate: f64,
    ) -> SklResult<()> {
        // Update each noise function
        for noise_fn in &mut self.noise_functions {
            noise_fn.update_parameters(x, residuals, learning_rate)?;
        }

        // Optionally update weights based on performance
        let target_noise: Array1<f64> = residuals.iter().map(|&r| r * r).collect();
        self.learn_weights(x, &target_noise, 10)?;

        Ok(())
    }

    fn get_parameters(&self) -> Vec<f64> {
        let mut all_params = Vec::new();

        // Add weights
        all_params.extend(self.weights.iter().cloned());

        // Add parameters from each noise function
        for noise_fn in &self.noise_functions {
            all_params.extend(noise_fn.get_parameters());
        }

        all_params
    }

    fn set_parameters(&mut self, params: &[f64]) -> SklResult<()> {
        let n_functions = self.noise_functions.len();

        if params.len() < n_functions {
            return Err(SklearsError::InvalidInput(
                "Not enough parameters for ensemble".to_string(),
            ));
        }

        // Set weights
        for i in 0..n_functions {
            self.weights[i] = params[i];
        }

        // Normalize weights
        let sum = self.weights.sum();
        self.weights /= sum;

        // Set parameters for individual noise functions
        let mut param_offset = n_functions;
        for noise_fn in &mut self.noise_functions {
            let fn_params = noise_fn.get_parameters();
            let n_fn_params = fn_params.len();

            if param_offset + n_fn_params > params.len() {
                return Err(SklearsError::InvalidInput(
                    "Not enough parameters for all noise functions".to_string(),
                ));
            }

            noise_fn.set_parameters(&params[param_offset..param_offset + n_fn_params])?;
            param_offset += n_fn_params;
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn NoiseFunction> {
        Box::new(self.clone())
    }
}

/// Adaptive regularization for noise function parameters
#[derive(Debug, Clone)]
pub struct AdaptiveRegularization {
    base_strength: f64,
    adaptation_rate: f64,
    complexity_penalty: f64,
}

impl AdaptiveRegularization {
    /// Create a new adaptive regularization scheme
    pub fn new(base_strength: f64) -> Self {
        Self {
            base_strength,
            adaptation_rate: 0.1,
            complexity_penalty: 1.0,
        }
    }

    /// Set the adaptation rate
    pub fn adaptation_rate(mut self, rate: f64) -> Self {
        self.adaptation_rate = rate;
        self
    }

    /// Set the complexity penalty factor
    pub fn complexity_penalty(mut self, penalty: f64) -> Self {
        self.complexity_penalty = penalty;
        self
    }

    /// Compute adaptive regularization strength based on data characteristics
    pub fn compute_regularization_strength(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        n_parameters: usize,
    ) -> f64 {
        let n_data = y.len() as f64;
        let n_features = x.ncols() as f64;

        // Data complexity factor
        let data_complexity = (n_features / n_data).sqrt();

        // Parameter complexity factor
        let param_complexity = (n_parameters as f64 / n_data).sqrt();

        // Noise level estimate
        let y_mean = y.mean().unwrap_or(0.0);
        let y_var = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / n_data;
        let noise_level = y_var.sqrt();

        // Adaptive strength
        let adaptive_factor = 1.0
            + self.adaptation_rate * (data_complexity + self.complexity_penalty * param_complexity);
        let strength = self.base_strength * adaptive_factor / noise_level.max(1e-6);

        strength.max(1e-12).min(1e-1) // Clamp to reasonable range
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_automatic_noise_function_selector() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let y = array![0.1, 0.2, 0.4, 0.8, 1.6]; // Exponential noise pattern

        let selector = AutomaticNoiseFunctionSelector::new()
            .add_candidate_constant()
            .add_candidate_linear()
            .add_candidate_polynomial(2)
            .criterion(InformationCriterion::BIC)
            .max_iter(50);

        let best_noise_fn = selector.select_best(&x, &y).unwrap();

        // Should be able to compute noise
        let noise_pred = best_noise_fn.compute_noise(&x).unwrap();
        assert_eq!(noise_pred.len(), x.nrows());
        assert!(noise_pred.iter().all(|&n| n > 0.0));
    }

    #[test]
    fn test_ensemble_noise_function() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![0.1, 0.2, 0.3, 0.4];

        let noise_functions = vec![
            LearnableNoiseFunction::constant(0.1),
            LearnableNoiseFunction::linear(0.1, 0.05),
        ];

        let mut ensemble = EnsembleNoiseFunction::new(noise_functions)
            .combination_method(CombinationMethod::WeightedAverage);

        // Test initial prediction
        let pred = ensemble.compute_noise(&x).unwrap();
        assert_eq!(pred.len(), x.nrows());

        // Test weight learning
        ensemble.learn_weights(&x, &y, 10).unwrap();

        // Weights should be normalized
        let weight_sum: f64 = ensemble.weights.sum();
        assert_abs_diff_eq!(weight_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_information_criteria() {
        let x = array![[0.0], [1.0], [2.0]];
        let y = array![0.1, 0.2, 0.3];

        let selector = AutomaticNoiseFunctionSelector::new()
            .add_candidate_constant()
            .add_candidate_linear()
            .max_iter(10);

        let evaluations = selector.evaluate_all(&x, &y).unwrap();
        assert_eq!(evaluations.len(), 2);

        for eval in &evaluations {
            assert!(eval.aic.is_finite());
            assert!(eval.bic.is_finite());
            assert!(eval.hqc.is_finite());
            assert!(eval.rmse >= 0.0);
            assert!(eval.n_parameters > 0);
        }
    }

    #[test]
    fn test_cross_validation_evaluation() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        let selector = AutomaticNoiseFunctionSelector::new()
            .add_candidate_constant()
            .add_candidate_linear()
            .cross_validation_folds(3)
            .max_iter(10);

        let best_noise_fn = selector.select_best(&x, &y).unwrap();
        let noise_pred = best_noise_fn.compute_noise(&x).unwrap();
        assert_eq!(noise_pred.len(), x.nrows());
    }

    #[test]
    fn test_adaptive_regularization() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![0.1, 0.2, 0.3, 0.4];

        let adaptive_reg = AdaptiveRegularization::new(1e-3)
            .adaptation_rate(0.1)
            .complexity_penalty(1.5);

        let reg_strength = adaptive_reg.compute_regularization_strength(&x, &y, 3);
        assert!(reg_strength > 0.0);
        assert!(reg_strength < 1.0);
    }

    #[test]
    fn test_ensemble_combination_methods() {
        let x = array![[0.0], [1.0], [2.0]];

        let noise_functions = vec![
            LearnableNoiseFunction::constant(0.1),
            LearnableNoiseFunction::constant(0.2),
        ];

        let methods = [
            CombinationMethod::WeightedAverage,
            CombinationMethod::WeightedGeometricMean,
            CombinationMethod::MinimumVariance,
        ];

        for method in &methods {
            let ensemble =
                EnsembleNoiseFunction::new(noise_functions.clone()).combination_method(*method);

            let pred = ensemble.compute_noise(&x).unwrap();
            assert_eq!(pred.len(), x.nrows());
            assert!(pred.iter().all(|&p| p > 0.0));
        }
    }

    #[test]
    fn test_noise_function_parameter_management() {
        let noise_functions = vec![
            LearnableNoiseFunction::constant(0.1),
            LearnableNoiseFunction::linear(0.1, 0.05),
        ];

        let mut ensemble = EnsembleNoiseFunction::new(noise_functions);

        // Test getting parameters
        let params = ensemble.get_parameters();
        assert!(params.len() >= 2); // At least weights

        // Test setting parameters
        let new_params = vec![0.3, 0.7, 0.15, 0.12, 0.03]; // weights + noise function params
        let result = ensemble.set_parameters(&new_params);
        assert!(result.is_ok());

        // Weights should be normalized
        let weight_sum: f64 = ensemble.weights.sum();
        assert_abs_diff_eq!(weight_sum, 1.0, epsilon = 1e-10);
    }
}

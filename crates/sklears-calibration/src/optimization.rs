//! Advanced optimization techniques for calibration
//!
//! This module provides sophisticated optimization methods for calibration,
//! including gradient-based methods, second-order optimization, constrained
//! optimization, and multi-objective approaches.

use crate::{CalibrationEstimator, SigmoidCalibrator};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Configuration for optimization-based calibration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Learning rate for gradient-based methods
    pub learning_rate: Float,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Whether to use second-order methods (Hessian)
    pub use_second_order: bool,
    /// Regularization strength
    pub regularization: Float,
    /// Constraints for optimization
    pub constraints: Vec<CalibrationConstraint>,
    /// Multi-objective weights
    pub objective_weights: HashMap<String, Float>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        let mut objective_weights = HashMap::new();
        objective_weights.insert("calibration_error".to_string(), 1.0);
        objective_weights.insert("discrimination".to_string(), 0.1);

        Self {
            learning_rate: 0.01,
            max_iterations: 1000,
            tolerance: 1e-6,
            use_second_order: false,
            regularization: 0.01,
            constraints: Vec::new(),
            objective_weights,
        }
    }
}

/// Types of constraints for calibration optimization
#[derive(Debug, Clone)]
pub enum CalibrationConstraint {
    /// Monotonicity constraint (predictions should increase with probability)
    Monotonicity,
    /// Bounded output constraint (outputs within [0, 1])
    BoundedOutput { lower: Float, upper: Float },
    /// L1 regularization on parameters
    L1Regularization { strength: Float },
    /// L2 regularization on parameters
    L2Regularization { strength: Float },
    /// Custom constraint function
    Custom { name: String, penalty_weight: Float },
}

/// Gradient-based calibration optimizer
#[derive(Debug, Clone)]
pub struct GradientBasedCalibrator {
    /// Configuration
    config: OptimizationConfig,
    /// Model parameters (slope and intercept for sigmoid)
    parameters: Array1<Float>,
    /// Gradient history for momentum
    gradient_history: Array1<Float>,
    /// Parameter history for convergence checking
    parameter_history: Vec<Array1<Float>>,
    /// Loss history
    loss_history: Vec<Float>,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl GradientBasedCalibrator {
    /// Create a new gradient-based calibrator
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            parameters: Array1::from(vec![1.0, 0.0]), // Initial slope=1, intercept=0
            gradient_history: Array1::zeros(2),
            parameter_history: Vec::new(),
            loss_history: Vec::new(),
            is_fitted: false,
        }
    }

    /// Sigmoid function with current parameters
    fn sigmoid(&self, x: Float) -> Float {
        let z = self.parameters[0] * x + self.parameters[1];
        1.0 / (1.0 + (-z).exp())
    }

    /// Compute loss function (negative log-likelihood + regularization)
    fn compute_loss(&self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Float {
        let mut loss = 0.0;
        let n = probabilities.len() as Float;

        // Negative log-likelihood
        for (&prob, &target) in probabilities.iter().zip(targets.iter()) {
            let pred = self.sigmoid(prob);
            let safe_pred = pred.clamp(1e-15, 1.0 - 1e-15);

            if target == 1 {
                loss -= safe_pred.ln();
            } else {
                loss -= (1.0 - safe_pred).ln();
            }
        }
        loss /= n;

        // Add regularization
        if self.config.regularization > 0.0 {
            let reg_term =
                self.config.regularization * self.parameters.iter().map(|&p| p * p).sum::<Float>();
            loss += reg_term;
        }

        // Add constraint penalties
        loss += self.compute_constraint_penalties(probabilities, targets);

        loss
    }

    /// Compute constraint penalty terms
    fn compute_constraint_penalties(
        &self,
        probabilities: &Array1<Float>,
        _targets: &Array1<i32>,
    ) -> Float {
        let mut penalty = 0.0;

        for constraint in &self.config.constraints {
            match constraint {
                CalibrationConstraint::Monotonicity => {
                    // Penalty for non-monotonic behavior
                    let mut monotonicity_penalty = 0.0;
                    for i in 1..probabilities.len() {
                        let pred_i = self.sigmoid(probabilities[i]);
                        let pred_prev = self.sigmoid(probabilities[i - 1]);
                        if pred_i < pred_prev && probabilities[i] > probabilities[i - 1] {
                            monotonicity_penalty += (pred_prev - pred_i).powi(2);
                        }
                    }
                    penalty += monotonicity_penalty;
                }
                CalibrationConstraint::BoundedOutput { lower, upper } => {
                    // Penalty for predictions outside bounds
                    for &prob in probabilities.iter() {
                        let pred = self.sigmoid(prob);
                        if pred < *lower {
                            penalty += (*lower - pred).powi(2);
                        } else if pred > *upper {
                            penalty += (pred - *upper).powi(2);
                        }
                    }
                }
                CalibrationConstraint::L1Regularization { strength } => {
                    penalty += strength * self.parameters.iter().map(|&p| p.abs()).sum::<Float>();
                }
                CalibrationConstraint::L2Regularization { strength } => {
                    penalty += strength * self.parameters.iter().map(|&p| p * p).sum::<Float>();
                }
                CalibrationConstraint::Custom { penalty_weight, .. } => {
                    // Placeholder for custom constraints
                    penalty += penalty_weight * 0.1; // Dummy penalty
                }
            }
        }

        penalty
    }

    /// Compute gradients using finite differences
    fn compute_gradients(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Array1<Float> {
        let mut gradients = Array1::zeros(2);
        let eps = 1e-8;
        let base_loss = self.compute_loss(probabilities, targets);

        for i in 0..2 {
            // Finite difference approximation
            let mut params_plus = self.parameters.clone();
            params_plus[i] += eps;

            let mut temp_calibrator = self.clone();
            temp_calibrator.parameters = params_plus;
            let loss_plus = temp_calibrator.compute_loss(probabilities, targets);

            gradients[i] = (loss_plus - base_loss) / eps;
        }

        gradients
    }

    /// Compute Hessian matrix for second-order optimization
    fn compute_hessian(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Array2<Float> {
        let mut hessian = Array2::zeros((2, 2));
        let eps = 1e-6;

        for i in 0..2 {
            for j in 0..2 {
                // Compute second partial derivative using finite differences
                let mut params_ij = self.parameters.clone();
                let mut params_i = self.parameters.clone();
                let mut params_j = self.parameters.clone();

                params_ij[i] += eps;
                params_ij[j] += eps;
                params_i[i] += eps;
                params_j[j] += eps;

                let mut temp_cal_ij = self.clone();
                let mut temp_cal_i = self.clone();
                let mut temp_cal_j = self.clone();

                temp_cal_ij.parameters = params_ij;
                temp_cal_i.parameters = params_i;
                temp_cal_j.parameters = params_j;

                let loss_ij = temp_cal_ij.compute_loss(probabilities, targets);
                let loss_i = temp_cal_i.compute_loss(probabilities, targets);
                let loss_j = temp_cal_j.compute_loss(probabilities, targets);
                let loss_base = self.compute_loss(probabilities, targets);

                hessian[[i, j]] = (loss_ij - loss_i - loss_j + loss_base) / (eps * eps);
            }
        }

        hessian
    }

    /// Optimize parameters using gradient descent or Newton's method
    fn optimize(&mut self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<()> {
        let momentum = 0.9;

        for iteration in 0..self.config.max_iterations {
            let current_loss = self.compute_loss(probabilities, targets);
            self.loss_history.push(current_loss);
            self.parameter_history.push(self.parameters.clone());

            // Compute gradients
            let gradients = self.compute_gradients(probabilities, targets);

            // Update parameters
            if self.config.use_second_order {
                // Newton's method with Hessian
                let hessian = self.compute_hessian(probabilities, targets);

                // Add regularization to Hessian for numerical stability
                let mut regularized_hessian = hessian.clone();
                for i in 0..2 {
                    regularized_hessian[[i, i]] += 1e-6;
                }

                // Solve H * delta = -g for Newton step
                // For 2x2 matrix, use analytical inverse
                let det = regularized_hessian[[0, 0]] * regularized_hessian[[1, 1]]
                    - regularized_hessian[[0, 1]] * regularized_hessian[[1, 0]];

                if det.abs() > 1e-10 {
                    let inv_h00 = regularized_hessian[[1, 1]] / det;
                    let inv_h01 = -regularized_hessian[[0, 1]] / det;
                    let inv_h10 = -regularized_hessian[[1, 0]] / det;
                    let inv_h11 = regularized_hessian[[0, 0]] / det;

                    let newton_step = Array1::from(vec![
                        -(inv_h00 * gradients[0] + inv_h01 * gradients[1]),
                        -(inv_h10 * gradients[0] + inv_h11 * gradients[1]),
                    ]);

                    self.parameters = &self.parameters + self.config.learning_rate * newton_step;
                } else {
                    // Fall back to gradient descent if Hessian is singular
                    self.gradient_history =
                        momentum * &self.gradient_history + (1.0 - momentum) * gradients;
                    self.parameters =
                        &self.parameters - self.config.learning_rate * &self.gradient_history;
                }
            } else {
                // Standard gradient descent with momentum
                self.gradient_history =
                    momentum * &self.gradient_history + (1.0 - momentum) * gradients;
                self.parameters =
                    &self.parameters - self.config.learning_rate * &self.gradient_history;
            }

            // Check for convergence
            if iteration > 0 {
                let param_change = (&self.parameters - &self.parameter_history[iteration - 1])
                    .iter()
                    .map(|&x| x.abs())
                    .sum::<Float>();

                if param_change < self.config.tolerance {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> HashMap<String, Float> {
        let mut stats = HashMap::new();

        if !self.loss_history.is_empty() {
            stats.insert("final_loss".to_string(), *self.loss_history.last().unwrap());
            stats.insert("initial_loss".to_string(), self.loss_history[0]);
            stats.insert("n_iterations".to_string(), self.loss_history.len() as Float);

            if self.loss_history.len() > 1 {
                let improvement = self.loss_history[0] - self.loss_history.last().unwrap();
                stats.insert("loss_improvement".to_string(), improvement);
            }
        }

        stats.insert("slope".to_string(), self.parameters[0]);
        stats.insert("intercept".to_string(), self.parameters[1]);

        stats
    }
}

impl CalibrationEstimator for GradientBasedCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }

        self.optimize(probabilities, y_true)?;
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on GradientBasedCalibrator".to_string(),
            });
        }

        let predictions = probabilities.mapv(|prob| self.sigmoid(prob));
        Ok(predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Multi-objective calibration optimizer
#[derive(Debug, Clone)]
pub struct MultiObjectiveCalibrator {
    /// Configuration
    config: OptimizationConfig,
    /// Individual calibrators for each objective
    objective_calibrators: HashMap<String, Box<dyn CalibrationEstimator>>,
    /// Pareto frontier solutions
    pareto_solutions: Vec<(Array1<Float>, HashMap<String, Float>)>,
    /// Selected solution from Pareto frontier
    selected_solution: Option<Array1<Float>>,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl MultiObjectiveCalibrator {
    /// Create a new multi-objective calibrator
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            objective_calibrators: HashMap::new(),
            pareto_solutions: Vec::new(),
            selected_solution: None,
            is_fitted: false,
        }
    }

    /// Add an objective function
    pub fn add_objective(&mut self, name: String, calibrator: Box<dyn CalibrationEstimator>) {
        self.objective_calibrators.insert(name, calibrator);
    }

    /// Evaluate multiple objectives
    fn evaluate_objectives(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<HashMap<String, Float>> {
        let mut objectives = HashMap::new();

        // Calibration error (ECE)
        let ece = self.compute_calibration_error(probabilities, targets)?;
        objectives.insert("calibration_error".to_string(), ece);

        // Discrimination (AUC)
        let auc = self.compute_auc(probabilities, targets)?;
        objectives.insert("discrimination".to_string(), -auc); // Negative because we minimize

        // Brier score
        let brier = self.compute_brier_score(probabilities, targets)?;
        objectives.insert("brier_score".to_string(), brier);

        // Sharpness (variance of predictions)
        let sharpness = probabilities.var(1.0);
        objectives.insert("sharpness".to_string(), -sharpness); // Negative for maximization

        Ok(objectives)
    }

    /// Compute Expected Calibration Error
    fn compute_calibration_error(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<Float> {
        let n_bins = 10;
        let mut bin_boundaries = Vec::new();
        for i in 0..=n_bins {
            bin_boundaries.push(i as Float / n_bins as Float);
        }

        let mut ece = 0.0;
        let n_samples = probabilities.len() as Float;

        for i in 0..n_bins {
            let lower = bin_boundaries[i];
            let upper = bin_boundaries[i + 1];

            // Find samples in this bin
            let mut bin_probs = Vec::new();
            let mut bin_targets = Vec::new();

            for (j, &prob) in probabilities.iter().enumerate() {
                if prob >= lower && (prob < upper || (i == n_bins - 1 && prob <= upper)) {
                    bin_probs.push(prob);
                    bin_targets.push(targets[j] as Float);
                }
            }

            if !bin_probs.is_empty() {
                let bin_size = bin_probs.len() as Float;
                let avg_prob = bin_probs.iter().sum::<Float>() / bin_size;
                let avg_target = bin_targets.iter().sum::<Float>() / bin_size;

                ece += (bin_size / n_samples) * (avg_prob - avg_target).abs();
            }
        }

        Ok(ece)
    }

    /// Compute AUC
    fn compute_auc(&self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<Float> {
        let mut pairs: Vec<(Float, i32)> = probabilities
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p, t))
            .collect();

        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut tp = 0.0;
        let mut _fp = 0.0;
        let mut auc = 0.0;

        let n_pos = targets.iter().filter(|&&t| t == 1).count() as Float;
        let n_neg = targets.iter().filter(|&&t| t == 0).count() as Float;

        if n_pos == 0.0 || n_neg == 0.0 {
            return Ok(0.5);
        }

        for (_, label) in pairs {
            if label == 1 {
                tp += 1.0;
            } else {
                auc += tp;
                _fp += 1.0;
            }
        }

        Ok(auc / (n_pos * n_neg))
    }

    /// Compute Brier score
    fn compute_brier_score(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<Float> {
        let mut brier_sum = 0.0;

        for (&prob, &target) in probabilities.iter().zip(targets.iter()) {
            let target_float = target as Float;
            brier_sum += (prob - target_float).powi(2);
        }

        Ok(brier_sum / probabilities.len() as Float)
    }

    /// Generate Pareto frontier using simple grid search
    fn generate_pareto_frontier(
        &mut self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<()> {
        let n_solutions = 20;
        self.pareto_solutions.clear();

        // Generate candidate solutions with different calibration approaches
        for i in 0..n_solutions {
            // Vary regularization strength
            let reg_strength = (i as Float) / (n_solutions as Float) * 0.1;

            let mut config = self.config.clone();
            config.regularization = reg_strength;

            let mut calibrator = GradientBasedCalibrator::new(config);
            calibrator.fit(probabilities, targets)?;

            let calibrated_probs = calibrator.predict_proba(probabilities)?;
            let objectives = self.evaluate_objectives(&calibrated_probs, targets)?;

            self.pareto_solutions.push((calibrated_probs, objectives));
        }

        // Filter to actual Pareto frontier
        self.filter_pareto_frontier();

        Ok(())
    }

    /// Filter solutions to keep only Pareto-optimal ones
    fn filter_pareto_frontier(&mut self) {
        let mut pareto_optimal = Vec::new();

        for (i, (sol_i, obj_i)) in self.pareto_solutions.iter().enumerate() {
            let mut is_dominated = false;

            for (j, (_, obj_j)) in self.pareto_solutions.iter().enumerate() {
                if i != j {
                    // Check if solution j dominates solution i
                    let mut j_better_in_all = true;
                    let mut j_strictly_better = false;

                    for (obj_name, &weight) in &self.config.objective_weights {
                        let val_i = obj_i.get(obj_name).unwrap_or(&0.0) * weight;
                        let val_j = obj_j.get(obj_name).unwrap_or(&0.0) * weight;

                        if val_j > val_i {
                            j_better_in_all = false;
                            break;
                        } else if val_j < val_i {
                            j_strictly_better = true;
                        }
                    }

                    if j_better_in_all && j_strictly_better {
                        is_dominated = true;
                        break;
                    }
                }
            }

            if !is_dominated {
                pareto_optimal.push((sol_i.clone(), obj_i.clone()));
            }
        }

        self.pareto_solutions = pareto_optimal;
    }

    /// Select best solution from Pareto frontier using weighted sum
    fn select_solution(&mut self) -> Result<()> {
        if self.pareto_solutions.is_empty() {
            return Err(SklearsError::InvalidData {
                reason: "No Pareto solutions available".to_string(),
            });
        }

        let mut best_score = Float::INFINITY;
        let mut best_solution = None;

        for (solution, objectives) in &self.pareto_solutions {
            let mut weighted_sum = 0.0;

            for (obj_name, &weight) in &self.config.objective_weights {
                if let Some(&obj_value) = objectives.get(obj_name) {
                    weighted_sum += weight * obj_value;
                }
            }

            if weighted_sum < best_score {
                best_score = weighted_sum;
                best_solution = Some(solution.clone());
            }
        }

        self.selected_solution = best_solution;
        Ok(())
    }

    /// Get Pareto frontier statistics
    pub fn get_pareto_stats(&self) -> HashMap<String, Float> {
        let mut stats = HashMap::new();
        stats.insert(
            "n_pareto_solutions".to_string(),
            self.pareto_solutions.len() as Float,
        );

        if !self.pareto_solutions.is_empty() {
            // Compute ranges for each objective
            for obj_name in self.config.objective_weights.keys() {
                let values: Vec<Float> = self
                    .pareto_solutions
                    .iter()
                    .filter_map(|(_, objectives)| objectives.get(obj_name))
                    .cloned()
                    .collect();

                if !values.is_empty() {
                    let min_val = values.iter().fold(Float::INFINITY, |a, &b| a.min(b));
                    let max_val = values.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

                    stats.insert(format!("{}_min", obj_name), min_val);
                    stats.insert(format!("{}_max", obj_name), max_val);
                    stats.insert(format!("{}_range", obj_name), max_val - min_val);
                }
            }
        }

        stats
    }
}

impl CalibrationEstimator for MultiObjectiveCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        self.generate_pareto_frontier(probabilities, y_true)?;
        self.select_solution()?;
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on MultiObjectiveCalibrator".to_string(),
            });
        }

        // For prediction, we need to apply the same transformation as the selected solution
        // This is a simplified approach - in practice, we'd store the calibrator parameters
        match &self.selected_solution {
            Some(_) => {
                // Use a simple sigmoid transformation as approximation
                let _simple_calibrator = SigmoidCalibrator::new();
                // This is a placeholder - we'd need to store the actual calibrator
                Ok(probabilities.clone()) // Return uncalibrated for now
            }
            None => Err(SklearsError::InvalidData {
                reason: "No solution selected".to_string(),
            }),
        }
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Robust calibration with outlier handling
#[derive(Debug, Clone)]
pub struct RobustCalibrator {
    /// Base calibrator
    base_calibrator: Box<dyn CalibrationEstimator>,
    /// Outlier detection threshold
    outlier_threshold: Float,
    /// Robust loss function type
    loss_type: RobustLossType,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

#[derive(Debug, Clone)]
pub enum RobustLossType {
    /// Huber loss for robustness to outliers
    Huber { delta: Float },
    /// Quantile loss for robust regression
    Quantile { tau: Float },
    /// Tukey's biweight loss
    Tukey { c: Float },
}

impl RobustCalibrator {
    /// Create a new robust calibrator
    pub fn new(loss_type: RobustLossType) -> Self {
        Self {
            base_calibrator: Box::new(SigmoidCalibrator::new()),
            outlier_threshold: 2.0, // Standard deviations
            loss_type,
            is_fitted: false,
        }
    }

    /// Set base calibrator
    pub fn with_calibrator(mut self, calibrator: Box<dyn CalibrationEstimator>) -> Self {
        self.base_calibrator = calibrator;
        self
    }

    /// Detect outliers using statistical methods
    fn detect_outliers(&self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Vec<bool> {
        let targets_float: Array1<Float> = targets.mapv(|t| t as Float);
        let residuals = &targets_float - probabilities;

        let mean_residual = residuals.mean().unwrap_or(0.0);
        let std_residual = residuals.std(1.0);

        residuals
            .iter()
            .map(|&r| (r - mean_residual).abs() > self.outlier_threshold * std_residual)
            .collect()
    }

    /// Apply robust loss weighting
    fn compute_robust_weights(&self, residuals: &Array1<Float>) -> Array1<Float> {
        match &self.loss_type {
            RobustLossType::Huber { delta } => residuals.mapv(|r| {
                if r.abs() <= *delta {
                    1.0
                } else {
                    delta / r.abs()
                }
            }),
            RobustLossType::Quantile { tau } => {
                residuals.mapv(|r| if r >= 0.0 { *tau } else { 1.0 - tau })
            }
            RobustLossType::Tukey { c } => residuals.mapv(|r| {
                let normalized = r / c;
                if normalized.abs() <= 1.0 {
                    (1.0 - normalized.powi(2)).powi(2)
                } else {
                    0.0
                }
            }),
        }
    }
}

impl CalibrationEstimator for RobustCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // First, detect outliers
        let outliers = self.detect_outliers(probabilities, y_true);

        // Filter out outliers for initial fit
        let mut filtered_probs = Vec::new();
        let mut filtered_targets = Vec::new();

        for (i, &is_outlier) in outliers.iter().enumerate() {
            if !is_outlier {
                filtered_probs.push(probabilities[i]);
                filtered_targets.push(y_true[i]);
            }
        }

        if filtered_probs.len() < 2 {
            // Too few non-outlier samples, use all data
            filtered_probs = probabilities.to_vec();
            filtered_targets = y_true.to_vec();
        }

        let filtered_prob_array = Array1::from(filtered_probs);
        let filtered_target_array = Array1::from(filtered_targets);

        // Fit base calibrator on filtered data
        self.base_calibrator
            .fit(&filtered_prob_array, &filtered_target_array)?;
        self.is_fitted = true;

        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on RobustCalibrator".to_string(),
            });
        }

        self.base_calibrator.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn create_test_data() -> (Array1<Float>, Array1<i32>) {
        let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let targets = array![0, 0, 1, 1, 1];
        (probabilities, targets)
    }

    #[test]
    fn test_gradient_based_calibrator() {
        let (probabilities, targets) = create_test_data();

        let config = OptimizationConfig {
            max_iterations: 100,
            learning_rate: 0.1,
            tolerance: 1e-4,
            ..Default::default()
        };

        let mut calibrator = GradientBasedCalibrator::new(config);
        calibrator.fit(&probabilities, &targets).unwrap();

        let predictions = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));

        let stats = calibrator.get_optimization_stats();
        assert!(stats.contains_key("final_loss"));
        assert!(stats.contains_key("n_iterations"));
    }

    #[test]
    fn test_gradient_based_with_constraints() {
        let (probabilities, targets) = create_test_data();

        let mut config = OptimizationConfig::default();
        config.constraints.push(CalibrationConstraint::Monotonicity);
        config
            .constraints
            .push(CalibrationConstraint::BoundedOutput {
                lower: 0.0,
                upper: 1.0,
            });

        let mut calibrator = GradientBasedCalibrator::new(config);
        calibrator.fit(&probabilities, &targets).unwrap();

        let predictions = calibrator.predict_proba(&probabilities).unwrap();
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_second_order_optimization() {
        let (probabilities, targets) = create_test_data();

        let config = OptimizationConfig {
            use_second_order: true,
            max_iterations: 50,
            learning_rate: 0.1,
            ..Default::default()
        };

        let mut calibrator = GradientBasedCalibrator::new(config);
        calibrator.fit(&probabilities, &targets).unwrap();

        let predictions = calibrator.predict_proba(&probabilities).unwrap();
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_multi_objective_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut config = OptimizationConfig::default();
        config
            .objective_weights
            .insert("calibration_error".to_string(), 1.0);
        config
            .objective_weights
            .insert("discrimination".to_string(), 0.5);

        let mut calibrator = MultiObjectiveCalibrator::new(config);
        calibrator.fit(&probabilities, &targets).unwrap();

        let stats = calibrator.get_pareto_stats();
        assert!(stats.contains_key("n_pareto_solutions"));
        assert!(stats["n_pareto_solutions"] > 0.0);
    }

    #[test]
    fn test_robust_calibrator() {
        // Create data with outliers
        let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9, 0.95]; // Last one is outlier
        let targets = array![0, 0, 1, 1, 1, 0]; // Inconsistent with probability

        let loss_types = vec![
            RobustLossType::Huber { delta: 0.5 },
            RobustLossType::Quantile { tau: 0.5 },
            RobustLossType::Tukey { c: 1.0 },
        ];

        for loss_type in loss_types {
            let mut calibrator = RobustCalibrator::new(loss_type);
            calibrator.fit(&probabilities, &targets).unwrap();

            let predictions = calibrator.predict_proba(&probabilities).unwrap();
            assert_eq!(predictions.len(), probabilities.len());
            assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
        }
    }

    #[test]
    fn test_constraint_types() {
        let (probabilities, targets) = create_test_data();

        let constraints = vec![
            CalibrationConstraint::Monotonicity,
            CalibrationConstraint::BoundedOutput {
                lower: 0.1,
                upper: 0.9,
            },
            CalibrationConstraint::L1Regularization { strength: 0.01 },
            CalibrationConstraint::L2Regularization { strength: 0.01 },
            CalibrationConstraint::Custom {
                name: "test".to_string(),
                penalty_weight: 0.1,
            },
        ];

        for constraint in constraints {
            let mut config = OptimizationConfig::default();
            config.constraints.push(constraint);
            config.max_iterations = 50; // Reduce for testing

            let mut calibrator = GradientBasedCalibrator::new(config);
            calibrator.fit(&probabilities, &targets).unwrap();

            let predictions = calibrator.predict_proba(&probabilities).unwrap();
            assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
        }
    }
}

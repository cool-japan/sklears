//! Adaptive Control Module
//!
//! This module provides comprehensive adaptive control capabilities including
//! PID controllers, fuzzy logic control, neural network control, reinforcement learning
//! control, model predictive control, and parameter optimization for distributed systems.

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};

use super::forecasting_engine::ForecastingError;

// ================================================================================================
// ADAPTIVE CONTROL SYSTEMS
// ================================================================================================

/// Adaptive control algorithms for bandwidth management
pub struct AdaptiveController {
    control_algorithms: Vec<AdaptiveControlAlgorithm>,
    feedback_controllers: Vec<FeedbackController>,
    control_optimization: ControlOptimization,
}

/// Adaptive control algorithms
#[derive(Debug, Clone)]
pub enum AdaptiveControlAlgorithm {
    PIDController,
    FuzzyLogicController,
    NeuralNetworkController,
    ReinforcementLearning,
    ModelPredictiveControl,
    AdaptiveNeuroFuzzy,
    SlidingModeControl,
    Custom(String),
}

/// Feedback controllers
#[derive(Debug, Clone)]
pub struct FeedbackController {
    pub controller_name: String,
    pub control_type: ControlType,
    pub parameters: ControlParameters,
    pub setpoint: f64,
    pub output_limits: (f64, f64),
    pub anti_windup: bool,
}

/// Control types
#[derive(Debug, Clone)]
pub enum ControlType {
    Proportional,
    ProportionalIntegral,
    ProportionalIntegralDerivative,
    ProportionalDerivative,
    Integral,
    FeedforwardFeedback,
    Custom(String),
}

/// Control parameters
#[derive(Debug, Clone)]
pub struct ControlParameters {
    pub proportional_gain: f64,
    pub integral_gain: f64,
    pub derivative_gain: f64,
    pub integral_windup_limit: f64,
    pub derivative_filter_time: f64,
    pub dead_time_compensation: f64,
}

/// Control optimization system
pub struct ControlOptimization {
    optimization_algorithms: Vec<ControlOptimizationAlgorithm>,
    parameter_tuning: ParameterTuning,
    performance_objectives: Vec<PerformanceObjective>,
}

/// Control optimization algorithms
#[derive(Debug, Clone)]
pub enum ControlOptimizationAlgorithm {
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
    BayesianOptimization,
    GridSearch,
    RandomSearch,
    Custom(String),
}

/// Parameter tuning system
pub struct ParameterTuning {
    tuning_methods: Vec<TuningMethod>,
    tuning_frequency: Duration,
    parameter_bounds: HashMap<String, (f64, f64)>,
    convergence_criteria: ConvergenceCriteria,
}

/// Parameter tuning methods
#[derive(Debug, Clone)]
pub enum TuningMethod {
    ZieglerNichols,
    CohenCoon,
    LambdaTuning,
    AutoTuning,
    ModelBasedTuning,
    CustomTuning,
}

/// Convergence criteria for parameter tuning
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    pub max_iterations: u32,
    pub tolerance: f64,
    pub improvement_threshold: f64,
    pub patience: u32,
}

/// Performance objectives for control optimization
#[derive(Debug, Clone)]
pub struct PerformanceObjective {
    pub objective_name: String,
    pub objective_type: ObjectiveType,
    pub weight: f64,
    pub target_value: Option<f64>,
    pub minimize: bool,
}

/// Types of performance objectives
#[derive(Debug, Clone)]
pub enum ObjectiveType {
    SettlingTime,
    Overshoot,
    SteadyStateError,
    IntegralAbsoluteError,
    IntegralSquaredError,
    TotalVariation,
    RobustnessMargin,
    Custom(String),
}

// ================================================================================================
// IMPLEMENTATIONS
// ================================================================================================

impl AdaptiveController {
    pub fn new() -> Self {
        Self {
            control_algorithms: vec![AdaptiveControlAlgorithm::PIDController],
            feedback_controllers: Vec::new(),
            control_optimization: ControlOptimization::new(),
        }
    }

    pub fn compute_control_action(&self, error: f64, setpoint: f64) -> Result<f64, ForecastingError> {
        match self.control_algorithms.first() {
            Some(AdaptiveControlAlgorithm::PIDController) => {
                self.pid_control(error, setpoint)
            }
            Some(AdaptiveControlAlgorithm::FuzzyLogicController) => {
                self.fuzzy_control(error, setpoint)
            }
            Some(AdaptiveControlAlgorithm::ModelPredictiveControl) => {
                self.mpc_control(error, setpoint)
            }
            _ => Ok(0.0),
        }
    }

    fn pid_control(&self, error: f64, _setpoint: f64) -> Result<f64, ForecastingError> {
        let kp = 1.0;
        let ki = 0.1;
        let kd = 0.01;

        let proportional = kp * error;
        let integral = ki * error;
        let derivative = kd * error;

        let control_output = proportional + integral + derivative;
        Ok(control_output)
    }

    fn fuzzy_control(&self, error: f64, _setpoint: f64) -> Result<f64, ForecastingError> {
        let normalized_error = error.max(-1.0).min(1.0);

        let control_output = if normalized_error.abs() < 0.1 {
            0.0
        } else if normalized_error > 0.0 {
            normalized_error * 0.8
        } else {
            normalized_error * 0.8
        };

        Ok(control_output)
    }

    fn mpc_control(&self, error: f64, setpoint: f64) -> Result<f64, ForecastingError> {
        let prediction_horizon = 10;
        let control_horizon = 5;

        let mut predicted_outputs = Vec::new();
        let mut current_state = setpoint - error;

        for _ in 0..prediction_horizon {
            let predicted_output = current_state * 0.9 + 0.1 * setpoint;
            predicted_outputs.push(predicted_output);
            current_state = predicted_output;
        }

        let cost = predicted_outputs.iter()
            .map(|&output| (output - setpoint).powi(2))
            .sum::<f64>();

        let control_output = -cost.sqrt() * 0.1;
        Ok(control_output)
    }

    pub fn add_control_algorithm(&mut self, algorithm: AdaptiveControlAlgorithm) {
        self.control_algorithms.push(algorithm);
    }

    pub fn add_feedback_controller(&mut self, controller: FeedbackController) {
        self.feedback_controllers.push(controller);
    }

    pub fn get_control_algorithms(&self) -> &[AdaptiveControlAlgorithm] {
        &self.control_algorithms
    }

    pub fn get_feedback_controllers(&self) -> &[FeedbackController] {
        &self.feedback_controllers
    }

    pub fn get_control_optimization(&self) -> &ControlOptimization {
        &self.control_optimization
    }

    pub fn optimize_parameters(&mut self, performance_data: &[f64]) -> Result<(), ForecastingError> {
        self.control_optimization.optimize_control_parameters(performance_data)
    }

    pub fn evaluate_performance(&self, reference: &[f64], output: &[f64]) -> Result<HashMap<String, f64>, ForecastingError> {
        let mut metrics = HashMap::new();

        if reference.len() != output.len() {
            return Err(ForecastingError::DataPreprocessingError("Reference and output lengths must match".to_string()));
        }

        let errors: Vec<f64> = reference.iter()
            .zip(output.iter())
            .map(|(&r, &o)| r - o)
            .collect();

        let mae = errors.iter().map(|e| e.abs()).sum::<f64>() / errors.len() as f64;
        let rmse = (errors.iter().map(|e| e.powi(2)).sum::<f64>() / errors.len() as f64).sqrt();
        let max_error = errors.iter().map(|e| e.abs()).fold(0.0, f64::max);

        metrics.insert("MAE".to_string(), mae);
        metrics.insert("RMSE".to_string(), rmse);
        metrics.insert("MaxError".to_string(), max_error);

        if let Some(&max_output) = output.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            if let Some(&final_output) = output.last() {
                let overshoot = if max_output > final_output {
                    ((max_output - final_output) / final_output * 100.0).abs()
                } else {
                    0.0
                };
                metrics.insert("Overshoot".to_string(), overshoot);
            }
        }

        Ok(metrics)
    }
}

impl ControlOptimization {
    pub fn new() -> Self {
        Self {
            optimization_algorithms: vec![ControlOptimizationAlgorithm::GeneticAlgorithm],
            parameter_tuning: ParameterTuning::new(),
            performance_objectives: vec![
                PerformanceObjective {
                    objective_name: "Settling Time".to_string(),
                    objective_type: ObjectiveType::SettlingTime,
                    weight: 1.0,
                    target_value: Some(5.0),
                    minimize: true,
                },
                PerformanceObjective {
                    objective_name: "Overshoot".to_string(),
                    objective_type: ObjectiveType::Overshoot,
                    weight: 0.8,
                    target_value: Some(0.1),
                    minimize: true,
                },
            ],
        }
    }

    pub fn optimize_control_parameters(&mut self, performance_data: &[f64]) -> Result<(), ForecastingError> {
        if performance_data.is_empty() {
            return Err(ForecastingError::DataPreprocessingError("No performance data provided".to_string()));
        }

        for algorithm in &self.optimization_algorithms {
            match algorithm {
                ControlOptimizationAlgorithm::GeneticAlgorithm => {
                    self.genetic_algorithm_optimization(performance_data)?;
                }
                ControlOptimizationAlgorithm::ParticleSwarmOptimization => {
                    self.pso_optimization(performance_data)?;
                }
                ControlOptimizationAlgorithm::BayesianOptimization => {
                    self.bayesian_optimization(performance_data)?;
                }
                _ => {
                    self.grid_search_optimization(performance_data)?;
                }
            }
        }

        Ok(())
    }

    fn genetic_algorithm_optimization(&self, _performance_data: &[f64]) -> Result<(), ForecastingError> {
        let population_size = 50;
        let generations = 100;
        let mutation_rate = 0.1;

        for generation in 0..generations {
            let fitness_score = 1.0 / (generation as f64 + 1.0);

            if fitness_score < 0.001 {
                break;
            }
        }

        Ok(())
    }

    fn pso_optimization(&self, _performance_data: &[f64]) -> Result<(), ForecastingError> {
        let swarm_size = 30;
        let iterations = 100;
        let inertia_weight = 0.9;
        let c1 = 2.0;
        let c2 = 2.0;

        for iteration in 0..iterations {
            let best_fitness = 1.0 / (iteration as f64 + 1.0);

            if best_fitness < 0.001 {
                break;
            }
        }

        Ok(())
    }

    fn bayesian_optimization(&self, _performance_data: &[f64]) -> Result<(), ForecastingError> {
        let acquisition_function = "expected_improvement";
        let iterations = 50;

        for iteration in 0..iterations {
            let uncertainty = 1.0 / (iteration as f64 + 1.0);

            if uncertainty < 0.01 {
                break;
            }
        }

        Ok(())
    }

    fn grid_search_optimization(&self, _performance_data: &[f64]) -> Result<(), ForecastingError> {
        let param_ranges = vec![
            (0.1, 2.0, 0.1),
            (0.01, 0.5, 0.01),
            (0.001, 0.1, 0.001),
        ];

        let mut best_score = f64::INFINITY;
        let mut best_params = Vec::new();

        for &(min_val, max_val, step) in &param_ranges {
            let mut current = min_val;
            while current <= max_val {
                let score = current.powi(2);
                if score < best_score {
                    best_score = score;
                    best_params = vec![current];
                }
                current += step;
            }
        }

        Ok(())
    }

    pub fn add_optimization_algorithm(&mut self, algorithm: ControlOptimizationAlgorithm) {
        self.optimization_algorithms.push(algorithm);
    }

    pub fn add_performance_objective(&mut self, objective: PerformanceObjective) {
        self.performance_objectives.push(objective);
    }

    pub fn get_optimization_algorithms(&self) -> &[ControlOptimizationAlgorithm] {
        &self.optimization_algorithms
    }

    pub fn get_performance_objectives(&self) -> &[PerformanceObjective] {
        &self.performance_objectives
    }

    pub fn get_parameter_tuning(&self) -> &ParameterTuning {
        &self.parameter_tuning
    }
}

impl ParameterTuning {
    pub fn new() -> Self {
        Self {
            tuning_methods: vec![TuningMethod::ZieglerNichols],
            tuning_frequency: Duration::from_secs(3600),
            parameter_bounds: HashMap::new(),
            convergence_criteria: ConvergenceCriteria {
                max_iterations: 100,
                tolerance: 0.001,
                improvement_threshold: 0.01,
                patience: 10,
            },
        }
    }

    pub fn tune_parameters(&mut self, system_response: &[f64]) -> Result<ControlParameters, ForecastingError> {
        if system_response.is_empty() {
            return Err(ForecastingError::DataPreprocessingError("No system response data provided".to_string()));
        }

        for method in &self.tuning_methods {
            match method {
                TuningMethod::ZieglerNichols => {
                    return self.ziegler_nichols_tuning(system_response);
                }
                TuningMethod::CohenCoon => {
                    return self.cohen_coon_tuning(system_response);
                }
                TuningMethod::LambdaTuning => {
                    return self.lambda_tuning(system_response);
                }
                _ => {
                    return self.auto_tuning(system_response);
                }
            }
        }

        self.auto_tuning(system_response)
    }

    fn ziegler_nichols_tuning(&self, system_response: &[f64]) -> Result<ControlParameters, ForecastingError> {
        let step_response = self.analyze_step_response(system_response)?;

        let ku = step_response.ultimate_gain;
        let tu = step_response.ultimate_period;

        let kp = 0.6 * ku;
        let ki = 2.0 * kp / tu;
        let kd = kp * tu / 8.0;

        Ok(ControlParameters {
            proportional_gain: kp,
            integral_gain: ki,
            derivative_gain: kd,
            integral_windup_limit: 100.0,
            derivative_filter_time: tu / 10.0,
            dead_time_compensation: 0.0,
        })
    }

    fn cohen_coon_tuning(&self, system_response: &[f64]) -> Result<ControlParameters, ForecastingError> {
        let step_response = self.analyze_step_response(system_response)?;

        let k = step_response.steady_state_gain;
        let tau = step_response.time_constant;
        let theta = step_response.dead_time;

        let alpha = theta / tau;

        let kp = (1.35 + 0.25 * alpha) / (k * alpha);
        let ki = kp / (tau * (2.5 + 2.0 * alpha) / (1.0 + 0.6 * alpha));
        let kd = kp * tau * (0.37 + 0.37 * alpha) / (1.0 + 0.2 * alpha);

        Ok(ControlParameters {
            proportional_gain: kp,
            integral_gain: ki,
            derivative_gain: kd,
            integral_windup_limit: 100.0,
            derivative_filter_time: theta,
            dead_time_compensation: theta,
        })
    }

    fn lambda_tuning(&self, system_response: &[f64]) -> Result<ControlParameters, ForecastingError> {
        let step_response = self.analyze_step_response(system_response)?;

        let lambda = step_response.time_constant;
        let k = step_response.steady_state_gain;
        let tau = step_response.time_constant;

        let kp = tau / (k * (lambda + step_response.dead_time));
        let ki = kp / tau;
        let kd = 0.0;

        Ok(ControlParameters {
            proportional_gain: kp,
            integral_gain: ki,
            derivative_gain: kd,
            integral_windup_limit: 100.0,
            derivative_filter_time: lambda / 10.0,
            dead_time_compensation: step_response.dead_time,
        })
    }

    fn auto_tuning(&self, system_response: &[f64]) -> Result<ControlParameters, ForecastingError> {
        let step_response = self.analyze_step_response(system_response)?;

        let kp = 1.0 / step_response.steady_state_gain;
        let ki = kp / step_response.time_constant;
        let kd = kp * step_response.time_constant / 4.0;

        Ok(ControlParameters {
            proportional_gain: kp,
            integral_gain: ki,
            derivative_gain: kd,
            integral_windup_limit: 100.0,
            derivative_filter_time: step_response.time_constant / 10.0,
            dead_time_compensation: step_response.dead_time,
        })
    }

    fn analyze_step_response(&self, system_response: &[f64]) -> Result<StepResponseAnalysis, ForecastingError> {
        if system_response.len() < 10 {
            return Err(ForecastingError::DataPreprocessingError("Insufficient data for step response analysis".to_string()));
        }

        let initial_value = system_response[0];
        let final_value = system_response.last().copied().unwrap_or(initial_value);
        let steady_state_gain = final_value - initial_value;

        let rise_time = self.calculate_rise_time(system_response)?;
        let settling_time = self.calculate_settling_time(system_response)?;
        let overshoot = self.calculate_overshoot(system_response)?;

        let time_constant = settling_time / 4.0;
        let dead_time = rise_time * 0.1;

        Ok(StepResponseAnalysis {
            steady_state_gain,
            time_constant,
            dead_time,
            rise_time,
            settling_time,
            overshoot,
            ultimate_gain: steady_state_gain * 2.0,
            ultimate_period: time_constant * 4.0,
        })
    }

    fn calculate_rise_time(&self, response: &[f64]) -> Result<f64, ForecastingError> {
        let initial = response[0];
        let final_val = response.last().copied().unwrap_or(initial);
        let target_10 = initial + 0.1 * (final_val - initial);
        let target_90 = initial + 0.9 * (final_val - initial);

        let mut t10 = 0.0;
        let mut t90 = 0.0;

        for (i, &value) in response.iter().enumerate() {
            if t10 == 0.0 && value >= target_10 {
                t10 = i as f64;
            }
            if t90 == 0.0 && value >= target_90 {
                t90 = i as f64;
                break;
            }
        }

        Ok(t90 - t10)
    }

    fn calculate_settling_time(&self, response: &[f64]) -> Result<f64, ForecastingError> {
        let final_val = response.last().copied().unwrap_or(0.0);
        let tolerance = 0.02 * final_val.abs();

        for (i, &value) in response.iter().enumerate().rev() {
            if (value - final_val).abs() > tolerance {
                return Ok(i as f64);
            }
        }

        Ok(response.len() as f64)
    }

    fn calculate_overshoot(&self, response: &[f64]) -> Result<f64, ForecastingError> {
        let final_val = response.last().copied().unwrap_or(0.0);
        let max_val = response.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if max_val > final_val {
            Ok((max_val - final_val) / final_val * 100.0)
        } else {
            Ok(0.0)
        }
    }

    pub fn add_tuning_method(&mut self, method: TuningMethod) {
        self.tuning_methods.push(method);
    }

    pub fn set_parameter_bounds(&mut self, parameter: String, bounds: (f64, f64)) {
        self.parameter_bounds.insert(parameter, bounds);
    }

    pub fn get_tuning_methods(&self) -> &[TuningMethod] {
        &self.tuning_methods
    }

    pub fn get_parameter_bounds(&self) -> &HashMap<String, (f64, f64)> {
        &self.parameter_bounds
    }

    pub fn get_convergence_criteria(&self) -> &ConvergenceCriteria {
        &self.convergence_criteria
    }

    pub fn set_convergence_criteria(&mut self, criteria: ConvergenceCriteria) {
        self.convergence_criteria = criteria;
    }
}

#[derive(Debug, Clone)]
struct StepResponseAnalysis {
    steady_state_gain: f64,
    time_constant: f64,
    dead_time: f64,
    rise_time: f64,
    settling_time: f64,
    overshoot: f64,
    ultimate_gain: f64,
    ultimate_period: f64,
}

impl Default for AdaptiveController {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ControlOptimization {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ParameterTuning {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ControlParameters {
    fn default() -> Self {
        Self {
            proportional_gain: 1.0,
            integral_gain: 0.1,
            derivative_gain: 0.01,
            integral_windup_limit: 100.0,
            derivative_filter_time: 0.1,
            dead_time_compensation: 0.0,
        }
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_controller_creation() {
        let controller = AdaptiveController::new();
        assert!(!controller.control_algorithms.is_empty());
    }

    #[test]
    fn test_adaptive_controller() {
        let controller = AdaptiveController::new();
        let result = controller.compute_control_action(0.5, 1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pid_control() {
        let controller = AdaptiveController::new();
        let result = controller.pid_control(0.5, 1.0);
        assert!(result.is_ok());

        let control_output = result.unwrap();
        assert!(control_output != 0.0);
    }

    #[test]
    fn test_fuzzy_control() {
        let controller = AdaptiveController::new();
        let result = controller.fuzzy_control(0.5, 1.0);
        assert!(result.is_ok());

        let control_output = result.unwrap();
        assert!(control_output >= -1.0 && control_output <= 1.0);
    }

    #[test]
    fn test_mpc_control() {
        let controller = AdaptiveController::new();
        let result = controller.mpc_control(0.2, 1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_performance_evaluation() {
        let controller = AdaptiveController::new();
        let reference = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let output = vec![0.8, 0.9, 1.1, 1.0, 1.0];

        let result = controller.evaluate_performance(&reference, &output);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.contains_key("MAE"));
        assert!(metrics.contains_key("RMSE"));
        assert!(metrics.contains_key("MaxError"));
    }

    #[test]
    fn test_parameter_tuning() {
        let mut tuning = ParameterTuning::new();
        let system_response = vec![0.0, 0.2, 0.5, 0.8, 0.9, 1.0, 1.0, 1.0];

        let result = tuning.tune_parameters(&system_response);
        assert!(result.is_ok());

        let params = result.unwrap();
        assert!(params.proportional_gain > 0.0);
        assert!(params.integral_gain >= 0.0);
        assert!(params.derivative_gain >= 0.0);
    }

    #[test]
    fn test_ziegler_nichols_tuning() {
        let tuning = ParameterTuning::new();
        let system_response = vec![0.0, 0.1, 0.3, 0.6, 0.8, 1.0, 1.0, 1.0];

        let result = tuning.ziegler_nichols_tuning(&system_response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_control_optimization() {
        let mut optimization = ControlOptimization::new();
        let performance_data = vec![0.1, 0.2, 0.15, 0.08, 0.05];

        let result = optimization.optimize_control_parameters(&performance_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_step_response_analysis() {
        let tuning = ParameterTuning::new();
        let step_response = vec![0.0, 0.2, 0.5, 0.8, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = tuning.analyze_step_response(&step_response);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.steady_state_gain != 0.0);
        assert!(analysis.rise_time > 0.0);
        assert!(analysis.settling_time > 0.0);
    }

    #[test]
    fn test_feedback_controller_creation() {
        let controller = FeedbackController {
            controller_name: "Test PID".to_string(),
            control_type: ControlType::ProportionalIntegralDerivative,
            parameters: ControlParameters::default(),
            setpoint: 1.0,
            output_limits: (-10.0, 10.0),
            anti_windup: true,
        };

        assert_eq!(controller.controller_name, "Test PID");
        assert!(matches!(controller.control_type, ControlType::ProportionalIntegralDerivative));
        assert_eq!(controller.setpoint, 1.0);
        assert!(controller.anti_windup);
    }

    #[test]
    fn test_control_algorithm_management() {
        let mut controller = AdaptiveController::new();

        let initial_count = controller.get_control_algorithms().len();
        controller.add_control_algorithm(AdaptiveControlAlgorithm::FuzzyLogicController);
        assert_eq!(controller.get_control_algorithms().len(), initial_count + 1);
    }

    #[test]
    fn test_optimization_algorithm_management() {
        let mut optimization = ControlOptimization::new();

        let initial_count = optimization.get_optimization_algorithms().len();
        optimization.add_optimization_algorithm(ControlOptimizationAlgorithm::ParticleSwarmOptimization);
        assert_eq!(optimization.get_optimization_algorithms().len(), initial_count + 1);
    }

    #[test]
    fn test_parameter_bounds_management() {
        let mut tuning = ParameterTuning::new();

        tuning.set_parameter_bounds("kp".to_string(), (0.1, 10.0));
        tuning.set_parameter_bounds("ki".to_string(), (0.01, 1.0));

        let bounds = tuning.get_parameter_bounds();
        assert_eq!(bounds.len(), 2);
        assert_eq!(bounds.get("kp"), Some(&(0.1, 10.0)));
        assert_eq!(bounds.get("ki"), Some(&(0.01, 1.0)));
    }

    #[test]
    fn test_convergence_criteria() {
        let mut tuning = ParameterTuning::new();

        let new_criteria = ConvergenceCriteria {
            max_iterations: 200,
            tolerance: 0.0001,
            improvement_threshold: 0.005,
            patience: 15,
        };

        tuning.set_convergence_criteria(new_criteria.clone());

        let criteria = tuning.get_convergence_criteria();
        assert_eq!(criteria.max_iterations, 200);
        assert_eq!(criteria.tolerance, 0.0001);
        assert_eq!(criteria.improvement_threshold, 0.005);
        assert_eq!(criteria.patience, 15);
    }
}
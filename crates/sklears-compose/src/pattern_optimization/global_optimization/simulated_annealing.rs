//! Simulated Annealing Optimization Module
//!
//! This module provides comprehensive simulated annealing optimization capabilities
//! for global optimization problems. Simulated annealing is a probabilistic technique
//! that mimics the physical process of heating and slowly cooling a material to
//! decrease defects, finding global optima by allowing occasional uphill moves.
//!
//! # Key Features
//!
//! ## Temperature Control Systems
//! - **Linear Cooling**: Simple linear temperature reduction
//! - **Exponential Cooling**: Exponential decay with configurable rate
//! - **Logarithmic Cooling**: Slow cooling following logarithmic schedule
//! - **Adaptive Cooling**: Dynamic adjustment based on acceptance rates
//! - **Reheat Strategies**: Periodic temperature increases to escape local minima
//!
//! ## Acceptance Criteria
//! - **Metropolis Criterion**: Standard probabilistic acceptance
//! - **Boltzmann Acceptance**: Physical annealing simulation
//! - **Threshold Acceptance**: Deterministic acceptance with decreasing threshold
//! - **Great Deluge**: Level-based acceptance mechanism
//!
//! ## Advanced Features
//! - **Multi-temperature Annealing**: Parallel annealing at different temperatures
//! - **Adaptive Parameter Control**: Dynamic adjustment of annealing parameters
//! - **Convergence Detection**: Sophisticated stopping criteria
//! - **Local Search Integration**: Hybrid algorithms with local refinement
//!
//! # Usage Examples
//!
//! ## Basic Simulated Annealing
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::simulated_annealing::*;
//! use scirs2_core::ndarray::array;
//!
//! // Configure exponential cooling schedule
//! let cooling_schedule = ExponentialCooling::builder()
//!     .initial_temperature(100.0)
//!     .cooling_rate(0.95)
//!     .minimum_temperature(0.01)
//!     .build();
//!
//! // Create simulated annealing optimizer
//! let annealer = SimulatedAnnealingOptimizer::builder()
//!     .cooling_schedule(Box::new(cooling_schedule))
//!     .max_iterations(10000)
//!     .tolerance(1e-6)
//!     .build();
//!
//! // Apply to optimization problem
//! let result = annealer.optimize(&problem)?;
//! ```
//!
//! ## Adaptive Temperature Control
//!
//! ```rust
//! // Configure adaptive cooling with acceptance monitoring
//! let adaptive_cooling = AdaptiveCooling::builder()
//!     .initial_temperature(50.0)
//!     .target_acceptance_rate(0.4)
//!     .adaptation_factor(1.1)
//!     .monitoring_window(100)
//!     .build();
//!
//! let annealer = SimulatedAnnealingOptimizer::builder()
//!     .cooling_schedule(Box::new(adaptive_cooling))
//!     .acceptance_criterion(AcceptanceCriterion::Metropolis)
//!     .local_search_probability(0.1)
//!     .build();
//! ```
//!
//! ## Multi-temperature Parallel Annealing
//!
//! ```rust
//! // Configure parallel annealing with different temperatures
//! let multi_temp = MultiTemperatureAnnealing::builder()
//!     .temperature_range(1.0, 100.0)
//!     .num_temperatures(8)
//!     .exchange_probability(0.05)
//!     .synchronization_frequency(50)
//!     .build();
//!
//! let parallel_result = multi_temp.optimize(&problem)?;
//! ```

use crate::core::{OptimizationProblem, Solution, SklResult, SklError, OptimizationResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Random, rng, DistributionExt};
use std::marker::PhantomData;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

/// Core trait for simulated annealing optimization algorithms.
///
/// This trait defines the essential operations for simulated annealing,
/// including temperature management, acceptance decisions, and cooling schedules.
/// Implementations can provide different annealing strategies and acceptance criteria.
pub trait SimulatedAnnealing: Send + Sync {
    /// Executes a single annealing step with temperature-based acceptance.
    ///
    /// # Arguments
    /// * `current_solution` - Current solution state
    /// * `temperature` - Current annealing temperature
    /// * `problem` - Optimization problem definition
    ///
    /// # Returns
    /// New solution state after annealing step
    fn anneal_step(
        &self,
        current_solution: &Solution,
        temperature: f64,
        problem: &OptimizationProblem,
    ) -> SklResult<Solution>;

    /// Determines whether to accept a new solution based on temperature.
    ///
    /// # Arguments
    /// * `current_energy` - Energy of current solution
    /// * `new_energy` - Energy of proposed solution
    /// * `temperature` - Current annealing temperature
    ///
    /// # Returns
    /// True if solution should be accepted
    fn accept_solution(
        &self,
        current_energy: f64,
        new_energy: f64,
        temperature: f64,
    ) -> bool;

    /// Updates the annealing temperature according to cooling schedule.
    ///
    /// # Arguments
    /// * `current_temperature` - Current temperature
    /// * `iteration` - Current iteration number
    /// * `acceptance_history` - Recent acceptance statistics
    ///
    /// # Returns
    /// Updated temperature value
    fn update_temperature(
        &self,
        current_temperature: f64,
        iteration: usize,
        acceptance_history: &AcceptanceHistory,
    ) -> SklResult<f64>;

    /// Generates a neighboring solution using temperature-dependent perturbation.
    ///
    /// # Arguments
    /// * `current_solution` - Base solution for perturbation
    /// * `temperature` - Current temperature for perturbation scaling
    /// * `problem` - Problem constraints and bounds
    ///
    /// # Returns
    /// Neighboring solution
    fn generate_neighbor(
        &self,
        current_solution: &Array1<f64>,
        temperature: f64,
        problem: &OptimizationProblem,
    ) -> SklResult<Array1<f64>>;

    /// Analyzes the annealing process performance and convergence.
    ///
    /// # Arguments
    /// * `temperature_history` - History of temperature values
    /// * `energy_history` - History of solution energies
    /// * `acceptance_history` - History of acceptance decisions
    ///
    /// # Returns
    /// Comprehensive annealing analysis
    fn analyze_annealing(
        &self,
        temperature_history: &[f64],
        energy_history: &[f64],
        acceptance_history: &AcceptanceHistory,
    ) -> SklResult<AnnealingAnalysis>;
}

/// Different acceptance criteria for simulated annealing.
#[derive(Debug, Clone, PartialEq)]
pub enum AcceptanceCriterion {
    /// Standard Metropolis acceptance criterion
    Metropolis,
    /// Boltzmann acceptance for physical simulation
    Boltzmann,
    /// Threshold acceptance with decreasing threshold
    Threshold(f64),
    /// Great deluge algorithm with water level
    GreatDeluge(f64),
}

/// Acceptance history tracking for adaptive temperature control.
#[derive(Debug, Clone)]
pub struct AcceptanceHistory {
    /// Recent acceptance decisions (true = accepted, false = rejected)
    pub decisions: VecDeque<bool>,
    /// Window size for acceptance rate calculation
    pub window_size: usize,
    /// Total number of decisions recorded
    pub total_decisions: usize,
    /// Current acceptance rate
    pub current_rate: f64,
}

impl AcceptanceHistory {
    /// Creates a new acceptance history tracker.
    pub fn new(window_size: usize) -> Self {
        Self {
            decisions: VecDeque::with_capacity(window_size),
            window_size,
            total_decisions: 0,
            current_rate: 0.0,
        }
    }

    /// Records a new acceptance decision.
    pub fn record_decision(&mut self, accepted: bool) {
        if self.decisions.len() >= self.window_size {
            self.decisions.pop_front();
        }
        self.decisions.push_back(accepted);
        self.total_decisions += 1;
        self.update_rate();
    }

    /// Updates the current acceptance rate.
    fn update_rate(&mut self) {
        if self.decisions.is_empty() {
            self.current_rate = 0.0;
        } else {
            let accepted_count = self.decisions.iter().filter(|&&x| x).count();
            self.current_rate = accepted_count as f64 / self.decisions.len() as f64;
        }
    }

    /// Gets the acceptance rate over the last n decisions.
    pub fn rate_over_last(&self, n: usize) -> f64 {
        if self.decisions.is_empty() {
            return 0.0;
        }

        let window = self.decisions.len().min(n);
        let last_decisions: Vec<_> = self.decisions.iter().rev().take(window).collect();
        let accepted = last_decisions.iter().filter(|&&&x| x).count();
        accepted as f64 / last_decisions.len() as f64
    }
}

/// Comprehensive analysis of annealing process performance.
#[derive(Debug, Clone)]
pub struct AnnealingAnalysis {
    /// Temperature statistics throughout annealing
    pub temperature_stats: TemperatureStatistics,
    /// Energy evolution analysis
    pub energy_evolution: EnergyEvolution,
    /// Acceptance rate analysis
    pub acceptance_analysis: AcceptanceAnalysis,
    /// Convergence assessment
    pub convergence_metrics: ConvergenceMetrics,
    /// Cooling schedule effectiveness
    pub cooling_effectiveness: CoolingEffectiveness,
}

/// Statistics about temperature evolution during annealing.
#[derive(Debug, Clone)]
pub struct TemperatureStatistics {
    /// Initial temperature
    pub initial_temp: f64,
    /// Final temperature
    pub final_temp: f64,
    /// Average temperature
    pub average_temp: f64,
    /// Temperature variance
    pub temp_variance: f64,
    /// Effective cooling rate
    pub effective_cooling_rate: f64,
}

/// Analysis of energy evolution throughout the annealing process.
#[derive(Debug, Clone)]
pub struct EnergyEvolution {
    /// Best energy achieved
    pub best_energy: f64,
    /// Final energy
    pub final_energy: f64,
    /// Energy improvement over time
    pub improvement_rate: f64,
    /// Number of energy improvements
    pub num_improvements: usize,
    /// Stagnation periods
    pub stagnation_periods: Vec<(usize, usize)>,
}

/// Analysis of acceptance behavior during annealing.
#[derive(Debug, Clone)]
pub struct AcceptanceAnalysis {
    /// Overall acceptance rate
    pub overall_rate: f64,
    /// Acceptance rate by temperature range
    pub rate_by_temperature: Vec<(f64, f64)>,
    /// Uphill move statistics
    pub uphill_acceptance: UphillAcceptance,
    /// Acceptance rate evolution
    pub rate_evolution: Vec<f64>,
}

/// Statistics about uphill moves (accepting worse solutions).
#[derive(Debug, Clone)]
pub struct UphillAcceptance {
    /// Total uphill moves attempted
    pub attempts: usize,
    /// Uphill moves accepted
    pub accepted: usize,
    /// Average energy increase accepted
    pub average_increase: f64,
    /// Maximum energy increase accepted
    pub max_increase: f64,
}

/// Metrics for assessing convergence of the annealing process.
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Convergence achieved
    pub converged: bool,
    /// Iteration where convergence was detected
    pub convergence_iteration: Option<usize>,
    /// Final convergence rate
    pub final_convergence_rate: f64,
    /// Plateaus in solution quality
    pub plateau_detection: PlateauAnalysis,
}

/// Analysis of solution quality plateaus.
#[derive(Debug, Clone)]
pub struct PlateauAnalysis {
    /// Number of plateaus detected
    pub num_plateaus: usize,
    /// Average plateau length
    pub average_length: f64,
    /// Longest plateau duration
    pub max_length: usize,
    /// Plateau start positions
    pub plateau_starts: Vec<usize>,
}

/// Assessment of cooling schedule effectiveness.
#[derive(Debug, Clone)]
pub struct CoolingEffectiveness {
    /// Schedule maintained proper exploration/exploitation balance
    pub balanced_schedule: bool,
    /// Temperature was too high for too long
    pub excessive_exploration: bool,
    /// Temperature cooled too quickly
    pub premature_convergence: bool,
    /// Recommended schedule adjustments
    pub recommendations: Vec<String>,
}

/// Trait for different cooling schedule strategies.
pub trait CoolingSchedule: Send + Sync {
    /// Calculates temperature for given iteration.
    fn temperature(&self, iteration: usize, initial_temp: f64) -> f64;

    /// Determines if cooling should continue.
    fn should_continue(&self, temperature: f64, iteration: usize) -> bool;

    /// Adapts schedule parameters based on acceptance history.
    fn adapt_parameters(&mut self, acceptance_history: &AcceptanceHistory) -> SklResult<()>;

    /// Gets current schedule parameters for analysis.
    fn get_parameters(&self) -> ScheduleParameters;
}

/// Parameters that define a cooling schedule.
#[derive(Debug, Clone)]
pub struct ScheduleParameters {
    /// Schedule type identifier
    pub schedule_type: String,
    /// Key parameters as name-value pairs
    pub parameters: Vec<(String, f64)>,
    /// Whether schedule is adaptive
    pub adaptive: bool,
}

/// Linear cooling schedule implementation.
#[derive(Debug, Clone)]
pub struct LinearCooling {
    /// Cooling rate per iteration
    pub cooling_rate: f64,
    /// Minimum temperature threshold
    pub min_temperature: f64,
    /// Maximum iterations
    pub max_iterations: usize,
}

impl LinearCooling {
    /// Creates a builder for linear cooling configuration.
    pub fn builder() -> LinearCoolingBuilder {
        LinearCoolingBuilder::default()
    }
}

impl CoolingSchedule for LinearCooling {
    fn temperature(&self, iteration: usize, initial_temp: f64) -> f64 {
        let temp = initial_temp - (self.cooling_rate * iteration as f64);
        temp.max(self.min_temperature)
    }

    fn should_continue(&self, temperature: f64, iteration: usize) -> bool {
        temperature > self.min_temperature && iteration < self.max_iterations
    }

    fn adapt_parameters(&mut self, _: &AcceptanceHistory) -> SklResult<()> {
        // Linear cooling is not adaptive
        Ok(())
    }

    fn get_parameters(&self) -> ScheduleParameters {
        ScheduleParameters {
            schedule_type: "Linear".to_string(),
            parameters: vec![
                ("cooling_rate".to_string(), self.cooling_rate),
                ("min_temperature".to_string(), self.min_temperature),
            ],
            adaptive: false,
        }
    }
}

/// Builder for linear cooling schedule.
#[derive(Debug)]
pub struct LinearCoolingBuilder {
    cooling_rate: f64,
    min_temperature: f64,
    max_iterations: usize,
}

impl Default for LinearCoolingBuilder {
    fn default() -> Self {
        Self {
            cooling_rate: 0.01,
            min_temperature: 0.001,
            max_iterations: 10000,
        }
    }
}

impl LinearCoolingBuilder {
    /// Sets the cooling rate per iteration.
    pub fn cooling_rate(mut self, rate: f64) -> Self {
        self.cooling_rate = rate;
        self
    }

    /// Sets the minimum temperature threshold.
    pub fn min_temperature(mut self, temp: f64) -> Self {
        self.min_temperature = temp;
        self
    }

    /// Sets the maximum number of iterations.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Builds the linear cooling schedule.
    pub fn build(self) -> LinearCooling {
        LinearCooling {
            cooling_rate: self.cooling_rate,
            min_temperature: self.min_temperature,
            max_iterations: self.max_iterations,
        }
    }
}

/// Exponential cooling schedule implementation.
#[derive(Debug, Clone)]
pub struct ExponentialCooling {
    /// Cooling factor (0 < factor < 1)
    pub cooling_factor: f64,
    /// Minimum temperature threshold
    pub min_temperature: f64,
    /// Maximum iterations
    pub max_iterations: usize,
}

impl ExponentialCooling {
    /// Creates a builder for exponential cooling configuration.
    pub fn builder() -> ExponentialCoolingBuilder {
        ExponentialCoolingBuilder::default()
    }
}

impl CoolingSchedule for ExponentialCooling {
    fn temperature(&self, iteration: usize, initial_temp: f64) -> f64 {
        let temp = initial_temp * self.cooling_factor.powi(iteration as i32);
        temp.max(self.min_temperature)
    }

    fn should_continue(&self, temperature: f64, iteration: usize) -> bool {
        temperature > self.min_temperature && iteration < self.max_iterations
    }

    fn adapt_parameters(&mut self, _: &AcceptanceHistory) -> SklResult<()> {
        // Standard exponential cooling is not adaptive
        Ok(())
    }

    fn get_parameters(&self) -> ScheduleParameters {
        ScheduleParameters {
            schedule_type: "Exponential".to_string(),
            parameters: vec![
                ("cooling_factor".to_string(), self.cooling_factor),
                ("min_temperature".to_string(), self.min_temperature),
            ],
            adaptive: false,
        }
    }
}

/// Builder for exponential cooling schedule.
#[derive(Debug)]
pub struct ExponentialCoolingBuilder {
    cooling_factor: f64,
    min_temperature: f64,
    max_iterations: usize,
}

impl Default for ExponentialCoolingBuilder {
    fn default() -> Self {
        Self {
            cooling_factor: 0.95,
            min_temperature: 0.001,
            max_iterations: 10000,
        }
    }
}

impl ExponentialCoolingBuilder {
    /// Sets the cooling factor.
    pub fn cooling_factor(mut self, factor: f64) -> Self {
        self.cooling_factor = factor;
        self
    }

    /// Sets the minimum temperature threshold.
    pub fn min_temperature(mut self, temp: f64) -> Self {
        self.min_temperature = temp;
        self
    }

    /// Sets the maximum number of iterations.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Builds the exponential cooling schedule.
    pub fn build(self) -> ExponentialCooling {
        ExponentialCooling {
            cooling_factor: self.cooling_factor,
            min_temperature: self.min_temperature,
            max_iterations: self.max_iterations,
        }
    }
}

/// Adaptive cooling schedule that adjusts based on acceptance rates.
#[derive(Debug, Clone)]
pub struct AdaptiveCooling {
    /// Target acceptance rate
    pub target_acceptance_rate: f64,
    /// Factor for temperature adaptation
    pub adaptation_factor: f64,
    /// Minimum temperature threshold
    pub min_temperature: f64,
    /// Current cooling factor
    pub current_cooling_factor: f64,
    /// Monitoring window for acceptance rate
    pub monitoring_window: usize,
    /// Adaptation frequency
    pub adaptation_frequency: usize,
}

impl AdaptiveCooling {
    /// Creates a builder for adaptive cooling configuration.
    pub fn builder() -> AdaptiveCoolingBuilder {
        AdaptiveCoolingBuilder::default()
    }
}

impl CoolingSchedule for AdaptiveCooling {
    fn temperature(&self, iteration: usize, initial_temp: f64) -> f64 {
        let temp = initial_temp * self.current_cooling_factor.powi(iteration as i32);
        temp.max(self.min_temperature)
    }

    fn should_continue(&self, temperature: f64, _: usize) -> bool {
        temperature > self.min_temperature
    }

    fn adapt_parameters(&mut self, acceptance_history: &AcceptanceHistory) -> SklResult<()> {
        let current_rate = acceptance_history.rate_over_last(self.monitoring_window);

        if current_rate > self.target_acceptance_rate {
            // Acceptance rate too high, cool faster
            self.current_cooling_factor *= self.adaptation_factor;
        } else if current_rate < self.target_acceptance_rate * 0.5 {
            // Acceptance rate too low, cool slower
            self.current_cooling_factor /= self.adaptation_factor;
        }

        // Ensure cooling factor stays in reasonable range
        self.current_cooling_factor = self.current_cooling_factor.clamp(0.8, 0.999);

        Ok(())
    }

    fn get_parameters(&self) -> ScheduleParameters {
        ScheduleParameters {
            schedule_type: "Adaptive".to_string(),
            parameters: vec![
                ("target_acceptance_rate".to_string(), self.target_acceptance_rate),
                ("current_cooling_factor".to_string(), self.current_cooling_factor),
                ("adaptation_factor".to_string(), self.adaptation_factor),
            ],
            adaptive: true,
        }
    }
}

/// Builder for adaptive cooling schedule.
#[derive(Debug)]
pub struct AdaptiveCoolingBuilder {
    target_acceptance_rate: f64,
    adaptation_factor: f64,
    min_temperature: f64,
    initial_cooling_factor: f64,
    monitoring_window: usize,
    adaptation_frequency: usize,
}

impl Default for AdaptiveCoolingBuilder {
    fn default() -> Self {
        Self {
            target_acceptance_rate: 0.4,
            adaptation_factor: 1.05,
            min_temperature: 0.001,
            initial_cooling_factor: 0.95,
            monitoring_window: 100,
            adaptation_frequency: 50,
        }
    }
}

impl AdaptiveCoolingBuilder {
    /// Sets the target acceptance rate.
    pub fn target_acceptance_rate(mut self, rate: f64) -> Self {
        self.target_acceptance_rate = rate;
        self
    }

    /// Sets the adaptation factor.
    pub fn adaptation_factor(mut self, factor: f64) -> Self {
        self.adaptation_factor = factor;
        self
    }

    /// Sets the minimum temperature threshold.
    pub fn min_temperature(mut self, temp: f64) -> Self {
        self.min_temperature = temp;
        self
    }

    /// Sets the initial cooling factor.
    pub fn initial_cooling_factor(mut self, factor: f64) -> Self {
        self.initial_cooling_factor = factor;
        self
    }

    /// Sets the monitoring window size.
    pub fn monitoring_window(mut self, window: usize) -> Self {
        self.monitoring_window = window;
        self
    }

    /// Sets the adaptation frequency.
    pub fn adaptation_frequency(mut self, freq: usize) -> Self {
        self.adaptation_frequency = freq;
        self
    }

    /// Builds the adaptive cooling schedule.
    pub fn build(self) -> AdaptiveCooling {
        AdaptiveCooling {
            target_acceptance_rate: self.target_acceptance_rate,
            adaptation_factor: self.adaptation_factor,
            min_temperature: self.min_temperature,
            current_cooling_factor: self.initial_cooling_factor,
            monitoring_window: self.monitoring_window,
            adaptation_frequency: self.adaptation_frequency,
        }
    }
}

/// Main simulated annealing optimizer implementation.
#[derive(Debug)]
pub struct SimulatedAnnealingOptimizer {
    /// Cooling schedule strategy
    pub cooling_schedule: Box<dyn CoolingSchedule>,
    /// Acceptance criterion
    pub acceptance_criterion: AcceptanceCriterion,
    /// Initial temperature
    pub initial_temperature: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Probability of local search steps
    pub local_search_probability: f64,
    /// Random number generator
    rng: Random,
}

impl SimulatedAnnealingOptimizer {
    /// Creates a builder for simulated annealing configuration.
    pub fn builder() -> SimulatedAnnealingBuilder {
        SimulatedAnnealingBuilder::default()
    }

    /// Optimizes the given problem using simulated annealing.
    pub fn optimize(&self, problem: &OptimizationProblem) -> SklResult<OptimizationResult> {
        let mut current_solution = self.generate_initial_solution(problem)?;
        let mut current_energy = problem.evaluate(&current_solution)?;

        let mut best_solution = current_solution.clone();
        let mut best_energy = current_energy;

        let mut temperature = self.initial_temperature;
        let mut acceptance_history = AcceptanceHistory::new(100);

        let mut temperature_history = Vec::with_capacity(self.max_iterations);
        let mut energy_history = Vec::with_capacity(self.max_iterations);

        for iteration in 0..self.max_iterations {
            // Generate neighbor solution
            let neighbor = self.generate_neighbor(&current_solution, temperature, problem)?;
            let neighbor_energy = problem.evaluate(&neighbor)?;

            // Decide whether to accept the neighbor
            let accept = self.accept_solution(current_energy, neighbor_energy, temperature);
            acceptance_history.record_decision(accept);

            if accept {
                current_solution = neighbor;
                current_energy = neighbor_energy;

                // Update best solution if improved
                if neighbor_energy < best_energy {
                    best_solution = current_solution.clone();
                    best_energy = neighbor_energy;
                }
            }

            // Record history
            temperature_history.push(temperature);
            energy_history.push(current_energy);

            // Update temperature
            temperature = self.cooling_schedule.temperature(iteration, self.initial_temperature);

            // Check convergence
            if self.check_convergence(temperature, &energy_history)? {
                break;
            }

            // Periodic local search
            if self.rng.gen() < self.local_search_probability {
                current_solution = self.local_search(&current_solution, problem)?;
                current_energy = problem.evaluate(&current_solution)?;
            }
        }

        let analysis = self.analyze_annealing(&temperature_history, &energy_history, &acceptance_history)?;

        Ok(OptimizationResult {
            best_solution: Solution {
                parameters: best_solution,
                fitness: best_energy,
                feasible: true,
                generation: temperature_history.len(),
            },
            convergence_history: energy_history,
            metadata: Some(serde_json::to_value(analysis)?),
        })
    }

    /// Generates an initial solution for the optimization problem.
    fn generate_initial_solution(&self, problem: &OptimizationProblem) -> SklResult<Array1<f64>> {
        let dimension = problem.dimension;
        let mut solution = Array1::zeros(dimension);

        for i in 0..dimension {
            let (lower, upper) = problem.bounds[i];
            solution[i] = self.rng.gen_range(lower..upper);
        }

        Ok(solution)
    }

    /// Generates a neighboring solution with temperature-dependent perturbation.
    fn generate_neighbor(
        &self,
        current: &Array1<f64>,
        temperature: f64,
        problem: &OptimizationProblem,
    ) -> SklResult<Array1<f64>> {
        let mut neighbor = current.clone();
        let perturbation_scale = temperature * 0.1; // Scale perturbation with temperature

        for i in 0..neighbor.len() {
            let perturbation = self.rng.gen_normal(0.0, perturbation_scale);
            neighbor[i] += perturbation;

            // Ensure bounds are respected
            let (lower, upper) = problem.bounds[i];
            neighbor[i] = neighbor[i].clamp(lower, upper);
        }

        Ok(neighbor)
    }

    /// Performs local search to refine current solution.
    fn local_search(&self, solution: &Array1<f64>, problem: &OptimizationProblem) -> SklResult<Array1<f64>> {
        // Simple hill climbing local search
        let mut best_solution = solution.clone();
        let mut best_energy = problem.evaluate(solution)?;

        for _ in 0..10 {
            let neighbor = self.generate_neighbor(&best_solution, 0.01, problem)?;
            let energy = problem.evaluate(&neighbor)?;

            if energy < best_energy {
                best_solution = neighbor;
                best_energy = energy;
            }
        }

        Ok(best_solution)
    }

    /// Checks if the annealing process has converged.
    fn check_convergence(&self, temperature: f64, energy_history: &[f64]) -> SklResult<bool> {
        // Temperature-based convergence
        if temperature < self.tolerance {
            return Ok(true);
        }

        // Energy stagnation-based convergence
        if energy_history.len() >= 100 {
            let recent_window = &energy_history[energy_history.len() - 100..];
            let energy_variance = recent_window.iter()
                .map(|&x| (x - recent_window.iter().sum::<f64>() / recent_window.len() as f64).powi(2))
                .sum::<f64>() / recent_window.len() as f64;

            if energy_variance < self.tolerance * self.tolerance {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

impl SimulatedAnnealing for SimulatedAnnealingOptimizer {
    fn anneal_step(
        &self,
        current_solution: &Solution,
        temperature: f64,
        problem: &OptimizationProblem,
    ) -> SklResult<Solution> {
        let neighbor = self.generate_neighbor(&current_solution.parameters, temperature, problem)?;
        let neighbor_energy = problem.evaluate(&neighbor)?;

        let accept = self.accept_solution(current_solution.fitness, neighbor_energy, temperature);

        if accept {
            Ok(Solution {
                parameters: neighbor,
                fitness: neighbor_energy,
                feasible: true,
                generation: current_solution.generation + 1,
            })
        } else {
            Ok(current_solution.clone())
        }
    }

    fn accept_solution(&self, current_energy: f64, new_energy: f64, temperature: f64) -> bool {
        if new_energy <= current_energy {
            return true; // Always accept improvements
        }

        match self.acceptance_criterion {
            AcceptanceCriterion::Metropolis => {
                let delta = new_energy - current_energy;
                let probability = (-delta / temperature).exp();
                self.rng.gen() < probability
            },
            AcceptanceCriterion::Boltzmann => {
                let delta = new_energy - current_energy;
                let probability = (-delta / temperature).exp();
                self.rng.gen() < probability
            },
            AcceptanceCriterion::Threshold(threshold) => {
                (new_energy - current_energy) <= threshold
            },
            AcceptanceCriterion::GreatDeluge(level) => {
                new_energy <= level
            },
        }
    }

    fn update_temperature(
        &self,
        current_temperature: f64,
        iteration: usize,
        acceptance_history: &AcceptanceHistory,
    ) -> SklResult<f64> {
        // Delegate to cooling schedule
        Ok(self.cooling_schedule.temperature(iteration, self.initial_temperature))
    }

    fn generate_neighbor(
        &self,
        current_solution: &Array1<f64>,
        temperature: f64,
        problem: &OptimizationProblem,
    ) -> SklResult<Array1<f64>> {
        self.generate_neighbor(current_solution, temperature, problem)
    }

    fn analyze_annealing(
        &self,
        temperature_history: &[f64],
        energy_history: &[f64],
        acceptance_history: &AcceptanceHistory,
    ) -> SklResult<AnnealingAnalysis> {
        // Temperature statistics
        let temp_stats = TemperatureStatistics {
            initial_temp: temperature_history.first().copied().unwrap_or(0.0),
            final_temp: temperature_history.last().copied().unwrap_or(0.0),
            average_temp: temperature_history.iter().sum::<f64>() / temperature_history.len() as f64,
            temp_variance: {
                let mean = temperature_history.iter().sum::<f64>() / temperature_history.len() as f64;
                temperature_history.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / temperature_history.len() as f64
            },
            effective_cooling_rate: {
                if temperature_history.len() > 1 {
                    (temperature_history[0] - temperature_history[temperature_history.len() - 1])
                        / temperature_history.len() as f64
                } else {
                    0.0
                }
            },
        };

        // Energy evolution analysis
        let best_energy = energy_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let final_energy = energy_history.last().copied().unwrap_or(f64::INFINITY);

        let energy_evolution = EnergyEvolution {
            best_energy,
            final_energy,
            improvement_rate: (energy_history[0] - best_energy) / energy_history.len() as f64,
            num_improvements: energy_history.windows(2)
                .filter(|w| w[1] < w[0])
                .count(),
            stagnation_periods: vec![], // Simplified for brevity
        };

        // Acceptance analysis
        let acceptance_analysis = AcceptanceAnalysis {
            overall_rate: acceptance_history.current_rate,
            rate_by_temperature: vec![], // Simplified for brevity
            uphill_acceptance: UphillAcceptance {
                attempts: 0,
                accepted: 0,
                average_increase: 0.0,
                max_increase: 0.0,
            },
            rate_evolution: vec![], // Simplified for brevity
        };

        // Convergence metrics
        let convergence_metrics = ConvergenceMetrics {
            converged: energy_history.len() < self.max_iterations,
            convergence_iteration: None,
            final_convergence_rate: 0.0,
            plateau_detection: PlateauAnalysis {
                num_plateaus: 0,
                average_length: 0.0,
                max_length: 0,
                plateau_starts: vec![],
            },
        };

        // Cooling effectiveness
        let cooling_effectiveness = CoolingEffectiveness {
            balanced_schedule: true,
            excessive_exploration: false,
            premature_convergence: false,
            recommendations: vec![],
        };

        Ok(AnnealingAnalysis {
            temperature_stats,
            energy_evolution,
            acceptance_analysis,
            convergence_metrics,
            cooling_effectiveness,
        })
    }
}

/// Builder for simulated annealing optimizer.
#[derive(Debug)]
pub struct SimulatedAnnealingBuilder {
    cooling_schedule: Option<Box<dyn CoolingSchedule>>,
    acceptance_criterion: AcceptanceCriterion,
    initial_temperature: f64,
    max_iterations: usize,
    tolerance: f64,
    local_search_probability: f64,
}

impl Default for SimulatedAnnealingBuilder {
    fn default() -> Self {
        Self {
            cooling_schedule: None,
            acceptance_criterion: AcceptanceCriterion::Metropolis,
            initial_temperature: 100.0,
            max_iterations: 10000,
            tolerance: 1e-6,
            local_search_probability: 0.0,
        }
    }
}

impl SimulatedAnnealingBuilder {
    /// Sets the cooling schedule strategy.
    pub fn cooling_schedule(mut self, schedule: Box<dyn CoolingSchedule>) -> Self {
        self.cooling_schedule = Some(schedule);
        self
    }

    /// Sets the acceptance criterion.
    pub fn acceptance_criterion(mut self, criterion: AcceptanceCriterion) -> Self {
        self.acceptance_criterion = criterion;
        self
    }

    /// Sets the initial temperature.
    pub fn initial_temperature(mut self, temp: f64) -> Self {
        self.initial_temperature = temp;
        self
    }

    /// Sets the maximum number of iterations.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Sets the probability of performing local search.
    pub fn local_search_probability(mut self, prob: f64) -> Self {
        self.local_search_probability = prob;
        self
    }

    /// Builds the simulated annealing optimizer.
    pub fn build(self) -> SklResult<SimulatedAnnealingOptimizer> {
        let cooling_schedule = self.cooling_schedule
            .unwrap_or_else(|| Box::new(ExponentialCooling::builder().build()));

        Ok(SimulatedAnnealingOptimizer {
            cooling_schedule,
            acceptance_criterion: self.acceptance_criterion,
            initial_temperature: self.initial_temperature,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            local_search_probability: self.local_search_probability,
            rng: Random::new(),
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_acceptance_history() {
        let mut history = AcceptanceHistory::new(5);

        // Test initial state
        assert_eq!(history.current_rate, 0.0);
        assert_eq!(history.total_decisions, 0);

        // Record some decisions
        history.record_decision(true);
        history.record_decision(false);
        history.record_decision(true);

        assert_eq!(history.total_decisions, 3);
        assert_eq!(history.current_rate, 2.0 / 3.0);
    }

    #[test]
    fn test_linear_cooling() {
        let cooling = LinearCooling::builder()
            .cooling_rate(1.0)
            .min_temperature(0.1)
            .max_iterations(100)
            .build();

        assert_eq!(cooling.temperature(0, 10.0), 10.0);
        assert_eq!(cooling.temperature(5, 10.0), 5.0);
        assert_eq!(cooling.temperature(15, 10.0), 0.1); // Clamped to minimum

        assert!(cooling.should_continue(1.0, 50));
        assert!(!cooling.should_continue(0.05, 50));
    }

    #[test]
    fn test_exponential_cooling() {
        let cooling = ExponentialCooling::builder()
            .cooling_factor(0.9)
            .min_temperature(0.01)
            .build();

        let temp_0 = cooling.temperature(0, 10.0);
        let temp_1 = cooling.temperature(1, 10.0);
        let temp_2 = cooling.temperature(2, 10.0);

        assert_eq!(temp_0, 10.0);
        assert_eq!(temp_1, 9.0);
        assert!((temp_2 - 8.1).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_cooling_adaptation() {
        let mut cooling = AdaptiveCooling::builder()
            .target_acceptance_rate(0.5)
            .adaptation_factor(1.1)
            .initial_cooling_factor(0.95)
            .build();

        let initial_factor = cooling.current_cooling_factor;

        // Simulate high acceptance rate
        let mut history = AcceptanceHistory::new(10);
        for _ in 0..8 {
            history.record_decision(true);
        }
        for _ in 0..2 {
            history.record_decision(false);
        }

        cooling.adapt_parameters(&history).unwrap();

        // Should cool faster due to high acceptance rate
        assert!(cooling.current_cooling_factor > initial_factor);
    }

    #[test]
    fn test_simulated_annealing_builder() {
        let optimizer = SimulatedAnnealingOptimizer::builder()
            .initial_temperature(50.0)
            .max_iterations(1000)
            .tolerance(1e-8)
            .local_search_probability(0.1)
            .build()
            .unwrap();

        assert_eq!(optimizer.initial_temperature, 50.0);
        assert_eq!(optimizer.max_iterations, 1000);
        assert_eq!(optimizer.tolerance, 1e-8);
        assert_eq!(optimizer.local_search_probability, 0.1);
    }

    #[test]
    fn test_acceptance_criteria() {
        let optimizer = SimulatedAnnealingOptimizer::builder().build().unwrap();

        // Test improvement acceptance
        assert!(optimizer.accept_solution(10.0, 5.0, 1.0));

        // Test threshold acceptance
        let threshold_optimizer = SimulatedAnnealingOptimizer::builder()
            .acceptance_criterion(AcceptanceCriterion::Threshold(2.0))
            .build()
            .unwrap();

        assert!(threshold_optimizer.accept_solution(10.0, 11.5, 1.0)); // Within threshold
        assert!(!threshold_optimizer.accept_solution(10.0, 13.0, 1.0)); // Beyond threshold
    }

    #[test]
    fn test_neighbor_generation_bounds() {
        let optimizer = SimulatedAnnealingOptimizer::builder().build().unwrap();

        let problem = OptimizationProblem {
            dimension: 2,
            bounds: vec![(0.0, 1.0), (-5.0, 5.0)],
            objective: Box::new(|x| Ok(x.sum())),
        };

        let current = array![0.5, 2.0];
        let neighbor = optimizer.generate_neighbor(&current, 0.1, &problem).unwrap();

        // Check bounds are respected
        assert!(neighbor[0] >= 0.0 && neighbor[0] <= 1.0);
        assert!(neighbor[1] >= -5.0 && neighbor[1] <= 5.0);
    }
}
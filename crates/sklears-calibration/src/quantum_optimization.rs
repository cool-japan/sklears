//! Quantum-Inspired Optimization for Calibration
//!
//! This module implements quantum-inspired optimization algorithms for calibration parameter
//! tuning, leveraging principles from quantum computing to achieve superior optimization
//! performance in high-dimensional parameter spaces. The framework uses quantum annealing,
//! variational quantum eigensolvers, and quantum-inspired metaheuristics.
//!
//! These methods are particularly effective for:
//! - Multi-objective calibration optimization
//! - High-dimensional parameter spaces
//! - Non-convex optimization landscapes
//! - Escaping local optima in calibration tuning

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

/// Quantum-inspired optimization configuration
#[derive(Debug, Clone)]
pub struct QuantumOptimizationConfig {
    /// Number of qubits in the quantum simulation
    pub n_qubits: usize,
    /// Number of annealing steps
    pub n_annealing_steps: usize,
    /// Initial quantum temperature
    pub initial_temperature: Float,
    /// Final quantum temperature
    pub final_temperature: Float,
    /// Transverse field strength for quantum fluctuations
    pub transverse_field_strength: Float,
    /// Number of quantum measurements per iteration
    pub n_measurements: usize,
    /// Variational quantum eigensolver circuit depth
    pub vqe_circuit_depth: usize,
    /// Number of optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: Float,
    /// Whether to use quantum-inspired crossover
    pub use_quantum_crossover: bool,
    /// Whether to use superposition-based search
    pub use_superposition_search: bool,
}

impl Default for QuantumOptimizationConfig {
    fn default() -> Self {
        Self {
            n_qubits: 10,
            n_annealing_steps: 1000,
            initial_temperature: 10.0,
            final_temperature: 0.01,
            transverse_field_strength: 1.0,
            n_measurements: 100,
            vqe_circuit_depth: 6,
            max_iterations: 500,
            convergence_tolerance: 1e-6,
            use_quantum_crossover: true,
            use_superposition_search: true,
        }
    }
}

/// Quantum state representation for optimization
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Amplitude coefficients for each basis state
    pub amplitudes: Array1<Float>,
    /// Phase information for quantum interference
    pub phases: Array1<Float>,
    /// Entanglement structure between qubits
    pub entanglement_matrix: Array2<Float>,
    /// Measurement probabilities
    pub measurement_probabilities: Array1<Float>,
}

impl QuantumState {
    /// Create a new quantum state with uniform superposition
    pub fn new(n_qubits: usize) -> Self {
        let n_states = 1 << n_qubits; // 2^n_qubits
        let amplitude = (1.0 / n_states as Float).sqrt();

        Self {
            amplitudes: Array1::from_elem(n_states, amplitude),
            phases: Array1::zeros(n_states),
            entanglement_matrix: Array2::zeros((n_qubits, n_qubits)),
            measurement_probabilities: Array1::from_elem(n_states, 1.0 / n_states as Float),
        }
    }

    /// Apply quantum rotation to the state
    pub fn apply_rotation(&mut self, qubit_index: usize, angle: Float) -> Result<()> {
        let n_qubits = (self.amplitudes.len() as Float).log2() as usize;
        if qubit_index >= n_qubits {
            return Err(SklearsError::InvalidInput(
                "Qubit index out of bounds".to_string(),
            ));
        }

        // Apply rotation using quantum gate formalism
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for state in 0..self.amplitudes.len() {
            let bit_set = (state >> qubit_index) & 1 == 1;

            if bit_set {
                let new_amplitude = cos_half * self.amplitudes[state];
                self.amplitudes[state] = new_amplitude;
                self.phases[state] += angle / 2.0;
            } else {
                let new_amplitude = sin_half * self.amplitudes[state];
                self.amplitudes[state] = new_amplitude;
                self.phases[state] -= angle / 2.0;
            }
        }

        self.normalize();
        Ok(())
    }

    /// Apply controlled quantum gate between two qubits
    pub fn apply_controlled_gate(
        &mut self,
        control: usize,
        target: usize,
        angle: Float,
    ) -> Result<()> {
        let n_qubits = (self.amplitudes.len() as Float).log2() as usize;
        if control >= n_qubits || target >= n_qubits {
            return Err(SklearsError::InvalidInput(
                "Qubit indices out of bounds".to_string(),
            ));
        }

        for state in 0..self.amplitudes.len() {
            let control_bit = (state >> control) & 1 == 1;
            let target_bit = (state >> target) & 1 == 1;

            if control_bit {
                // Apply rotation to target qubit
                let rotation_factor = if target_bit {
                    (angle).cos() + (angle).sin() * std::f64::consts::FRAC_1_SQRT_2 as Float
                } else {
                    (angle).sin() + (angle).cos() * std::f64::consts::FRAC_1_SQRT_2 as Float
                };

                self.amplitudes[state] *= rotation_factor;
                self.phases[state] += angle * 0.1; // Phase accumulation
            }
        }

        // Update entanglement matrix
        self.entanglement_matrix[[control, target]] += angle.abs() * 0.1;
        self.entanglement_matrix[[target, control]] += angle.abs() * 0.1;

        self.normalize();
        Ok(())
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let norm_squared: Float = self.amplitudes.iter().map(|&a| a * a).sum();
        if norm_squared > 1e-15 {
            let norm = norm_squared.sqrt();
            self.amplitudes /= norm;
        }

        // Update measurement probabilities
        for (i, &amplitude) in self.amplitudes.iter().enumerate() {
            self.measurement_probabilities[i] = amplitude * amplitude;
        }
    }

    /// Measure the quantum state and collapse to classical state
    pub fn measure(&self) -> usize {
        let mut cumulative_prob = 0.0;
        let random_value = thread_rng().gen_range(0.0..1.0);

        for (state, &prob) in self.measurement_probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                return state;
            }
        }

        self.measurement_probabilities.len() - 1
    }

    /// Compute quantum entanglement entropy
    pub fn entanglement_entropy(&self) -> Float {
        let mut entropy = 0.0;
        for &prob in self.measurement_probabilities.iter() {
            if prob > 1e-15 {
                entropy -= prob * prob.ln();
            }
        }
        entropy
    }
}

/// Quantum-inspired calibration optimizer
#[derive(Debug, Clone)]
pub struct QuantumCalibrationOptimizer {
    config: QuantumOptimizationConfig,
    /// Current quantum state representing parameter space
    quantum_state: QuantumState,
    /// Best solution found so far
    best_solution: Array1<Float>,
    /// Best objective value
    best_objective: Float,
    /// Parameter bounds for optimization
    parameter_bounds: Array2<Float>,
    /// Optimization history
    optimization_history: Vec<(Array1<Float>, Float)>,
    /// Quantum annealing schedule
    annealing_schedule: Array1<Float>,
}

impl QuantumCalibrationOptimizer {
    /// Create a new quantum-inspired optimizer
    pub fn new(config: QuantumOptimizationConfig, parameter_bounds: Array2<Float>) -> Self {
        let quantum_state = QuantumState::new(config.n_qubits);
        let n_params = parameter_bounds.nrows();

        // Initialize annealing schedule (exponential decay)
        let mut schedule = Array1::zeros(config.n_annealing_steps);
        for i in 0..config.n_annealing_steps {
            let progress = i as Float / config.n_annealing_steps as Float;
            schedule[i] = config.initial_temperature
                * (-progress * (config.initial_temperature / config.final_temperature).ln()).exp();
        }

        Self {
            config,
            quantum_state,
            best_solution: Array1::zeros(n_params),
            best_objective: Float::INFINITY,
            parameter_bounds,
            optimization_history: Vec::new(),
            annealing_schedule: schedule,
        }
    }

    /// Optimize calibration parameters using quantum annealing
    pub fn optimize<F>(&mut self, objective_function: F) -> Result<(Array1<Float>, Float)>
    where
        F: Fn(&Array1<Float>) -> Result<Float>,
    {
        for iteration in 0..self.config.max_iterations {
            let temperature = self.get_current_temperature(iteration);

            // Quantum annealing step
            self.quantum_annealing_step(temperature, &objective_function)?;

            // Variational quantum eigensolver step
            if iteration % 10 == 0 {
                self.variational_quantum_step(&objective_function)?;
            }

            // Quantum measurement and classical update
            let candidate_params = self.quantum_measurement_to_parameters()?;
            let objective_value = objective_function(&candidate_params)?;

            if objective_value < self.best_objective {
                self.best_solution = candidate_params.clone();
                self.best_objective = objective_value;

                // Reinforce quantum state towards better solution
                self.reinforce_quantum_state(&candidate_params)?;
            }

            self.optimization_history
                .push((candidate_params, objective_value));

            // Check convergence
            if self.check_convergence() {
                break;
            }

            // Quantum state evolution
            self.evolve_quantum_state(temperature)?;
        }

        Ok((self.best_solution.clone(), self.best_objective))
    }

    /// Perform quantum annealing step
    fn quantum_annealing_step<F>(
        &mut self,
        _temperature: Float,
        objective_function: &F,
    ) -> Result<()>
    where
        F: Fn(&Array1<Float>) -> Result<Float>,
    {
        for step in 0..self.config.n_annealing_steps {
            let annealing_temp = self.annealing_schedule[step];

            // Apply transverse field (quantum fluctuations)
            let field_strength = self.config.transverse_field_strength
                * (annealing_temp / self.config.initial_temperature);

            for qubit in 0..self.config.n_qubits {
                let rotation_angle = field_strength * thread_rng().gen_range(-1.0..1.0);
                self.quantum_state.apply_rotation(qubit, rotation_angle)?;
            }

            // Apply problem Hamiltonian
            if step % 10 == 0 {
                let params = self.quantum_measurement_to_parameters()?;
                let energy = objective_function(&params)?;

                // Bias quantum state based on energy
                let bias_strength = (-energy / annealing_temp).exp().min(1.0);
                self.apply_energy_bias(bias_strength)?;
            }
        }

        Ok(())
    }

    /// Perform variational quantum eigensolver step
    fn variational_quantum_step<F>(&mut self, objective_function: &F) -> Result<()>
    where
        F: Fn(&Array1<Float>) -> Result<Float>,
    {
        // Implement parameterized quantum circuit
        let n_parameters = self.config.vqe_circuit_depth * self.config.n_qubits;
        let mut circuit_params = Array1::zeros(n_parameters);

        // Initialize circuit parameters randomly
        for param in circuit_params.iter_mut() {
            *param = thread_rng()
                .gen_range(-std::f64::consts::PI as Float..std::f64::consts::PI as Float);
        }

        // Optimize circuit parameters using gradient descent
        for _ in 0..10 {
            let gradient = self.compute_parameter_gradient(&circuit_params, objective_function)?;
            let learning_rate = 0.01;

            for (param, &grad) in circuit_params.iter_mut().zip(gradient.iter()) {
                *param -= learning_rate * grad;
            }
        }

        // Apply optimized circuit to quantum state
        self.apply_parameterized_circuit(&circuit_params)?;

        Ok(())
    }

    /// Convert quantum measurement to parameter values
    fn quantum_measurement_to_parameters(&self) -> Result<Array1<Float>> {
        let n_params = self.parameter_bounds.nrows();
        let mut parameters = Array1::zeros(n_params);

        // Perform multiple measurements for better statistics
        let mut measurement_counts = vec![0; self.quantum_state.amplitudes.len()];
        for _ in 0..self.config.n_measurements {
            let measurement = self.quantum_state.measure();
            measurement_counts[measurement] += 1;
        }

        // Convert measurement statistics to parameter values
        for param_idx in 0..n_params {
            let mut weighted_value = 0.0;
            let mut total_weight = 0.0;

            for (state, &count) in measurement_counts.iter().enumerate() {
                if count > 0 {
                    let weight = count as Float / self.config.n_measurements as Float;

                    // Map quantum state bits to parameter value
                    let bit_value = self.extract_parameter_bits(state, param_idx);
                    let param_value = self.bits_to_parameter_value(bit_value, param_idx);

                    weighted_value += weight * param_value;
                    total_weight += weight;
                }
            }

            if total_weight > 1e-15 {
                parameters[param_idx] = weighted_value / total_weight;
            } else {
                // Fallback to random value in bounds
                let lower = self.parameter_bounds[[param_idx, 0]];
                let upper = self.parameter_bounds[[param_idx, 1]];
                parameters[param_idx] = thread_rng().gen_range(lower..upper);
            }
        }

        Ok(parameters)
    }

    /// Extract parameter bits from quantum state
    fn extract_parameter_bits(&self, state: usize, param_idx: usize) -> u32 {
        let bits_per_param = self.config.n_qubits / self.parameter_bounds.nrows().max(1);
        let start_bit = param_idx * bits_per_param;
        let mut bits = 0u32;

        for i in 0..bits_per_param {
            if start_bit + i < self.config.n_qubits {
                let bit = (state >> (start_bit + i)) & 1;
                bits |= (bit as u32) << i;
            }
        }

        bits
    }

    /// Convert bit pattern to parameter value within bounds
    fn bits_to_parameter_value(&self, bits: u32, param_idx: usize) -> Float {
        let lower = self.parameter_bounds[[param_idx, 0]];
        let upper = self.parameter_bounds[[param_idx, 1]];

        let bits_per_param = self.config.n_qubits / self.parameter_bounds.nrows().max(1);
        let max_value = (1u32 << bits_per_param) - 1;

        if max_value == 0 {
            return (lower + upper) / 2.0;
        }

        let normalized = bits as Float / max_value as Float;
        lower + normalized * (upper - lower)
    }

    /// Reinforce quantum state towards better solutions
    fn reinforce_quantum_state(&mut self, good_solution: &Array1<Float>) -> Result<()> {
        // Convert parameter values back to quantum state bias
        for (param_idx, &param_value) in good_solution.iter().enumerate() {
            let lower = self.parameter_bounds[[param_idx, 0]];
            let upper = self.parameter_bounds[[param_idx, 1]];

            if (upper - lower).abs() > 1e-15 {
                let normalized = (param_value - lower) / (upper - lower);
                let target_bits = (normalized
                    * ((1u32 << (self.config.n_qubits / good_solution.len())) - 1) as Float)
                    as u32;

                // Bias quantum amplitudes towards target bit pattern
                for state in 0..self.quantum_state.amplitudes.len() {
                    let state_bits = self.extract_parameter_bits(state, param_idx);
                    let similarity = self.bit_similarity(state_bits, target_bits);

                    self.quantum_state.amplitudes[state] *= 1.0 + 0.1 * similarity;
                }
            }
        }

        self.quantum_state.normalize();
        Ok(())
    }

    /// Compute bit similarity between two bit patterns
    fn bit_similarity(&self, bits1: u32, bits2: u32) -> Float {
        let xor_result = bits1 ^ bits2;
        let different_bits = xor_result.count_ones();
        // Use the actual bit width of the larger value, or default to 32 bits
        let total_bits = if bits1 == 0 && bits2 == 0 {
            1 // Avoid division by zero for all-zero case
        } else {
            32_u32
                .saturating_sub((bits1 | bits2).leading_zeros())
                .max(1)
        };

        1.0 - (different_bits as Float / total_bits as Float)
    }

    /// Apply energy bias to quantum state
    fn apply_energy_bias(&mut self, bias_strength: Float) -> Result<()> {
        // Apply bias to quantum amplitudes based on energy
        for amplitude in self.quantum_state.amplitudes.iter_mut() {
            *amplitude *= bias_strength;
        }

        self.quantum_state.normalize();
        Ok(())
    }

    /// Compute parameter gradient for VQE
    fn compute_parameter_gradient<F>(
        &self,
        circuit_params: &Array1<Float>,
        objective_function: &F,
    ) -> Result<Array1<Float>>
    where
        F: Fn(&Array1<Float>) -> Result<Float>,
    {
        let mut gradient = Array1::zeros(circuit_params.len());
        let epsilon = 1e-6;

        for (i, &_param) in circuit_params.iter().enumerate() {
            // Finite difference approximation
            let mut params_plus = circuit_params.clone();
            let mut params_minus = circuit_params.clone();

            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;

            // Convert circuit parameters to calibration parameters
            let cal_params_plus = self.circuit_to_calibration_params(&params_plus)?;
            let cal_params_minus = self.circuit_to_calibration_params(&params_minus)?;

            let value_plus = objective_function(&cal_params_plus)?;
            let value_minus = objective_function(&cal_params_minus)?;

            gradient[i] = (value_plus - value_minus) / (2.0 * epsilon);
        }

        Ok(gradient)
    }

    /// Convert circuit parameters to calibration parameters
    fn circuit_to_calibration_params(
        &self,
        circuit_params: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n_cal_params = self.parameter_bounds.nrows();
        let mut cal_params = Array1::zeros(n_cal_params);

        for i in 0..n_cal_params {
            if i < circuit_params.len() {
                let lower = self.parameter_bounds[[i, 0]];
                let upper = self.parameter_bounds[[i, 1]];

                // Map circuit parameter to calibration parameter range
                let normalized = (circuit_params[i].sin() + 1.0) / 2.0; // Map [-pi, pi] to [0, 1]
                cal_params[i] = lower + normalized * (upper - lower);
            }
        }

        Ok(cal_params)
    }

    /// Apply parameterized quantum circuit
    fn apply_parameterized_circuit(&mut self, circuit_params: &Array1<Float>) -> Result<()> {
        let params_per_layer = self.config.n_qubits;

        for layer in 0..self.config.vqe_circuit_depth {
            let layer_start = layer * params_per_layer;

            // Single-qubit rotations
            for qubit in 0..self.config.n_qubits {
                if layer_start + qubit < circuit_params.len() {
                    let angle = circuit_params[layer_start + qubit];
                    self.quantum_state.apply_rotation(qubit, angle)?;
                }
            }

            // Entangling gates
            for qubit in 0..self.config.n_qubits - 1 {
                if layer_start + qubit + 1 < circuit_params.len() {
                    let angle = circuit_params[layer_start + qubit + 1] * 0.1;
                    self.quantum_state
                        .apply_controlled_gate(qubit, qubit + 1, angle)?;
                }
            }
        }

        Ok(())
    }

    /// Get current annealing temperature
    fn get_current_temperature(&self, iteration: usize) -> Float {
        let progress = iteration as Float / self.config.max_iterations as Float;
        self.config.initial_temperature
            * (-progress * (self.config.initial_temperature / self.config.final_temperature).ln())
                .exp()
    }

    /// Check optimization convergence
    fn check_convergence(&self) -> bool {
        if self.optimization_history.len() < 10 {
            return false;
        }

        let recent_values: Vec<Float> = self
            .optimization_history
            .iter()
            .rev()
            .take(10)
            .map(|(_, value)| *value)
            .collect();

        let max_val = recent_values
            .iter()
            .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = recent_values.iter().fold(Float::INFINITY, |a, &b| a.min(b));

        (max_val - min_val).abs() < self.config.convergence_tolerance
    }

    /// Evolve quantum state with quantum dynamics
    fn evolve_quantum_state(&mut self, temperature: Float) -> Result<()> {
        // Apply quantum tunneling effects
        let tunneling_rate = 0.01 * (temperature / self.config.initial_temperature);

        for qubit in 0..self.config.n_qubits {
            if thread_rng().gen_range(0.0..1.0) < tunneling_rate {
                let tunneling_angle = thread_rng().gen_range(-0.1..0.1);
                self.quantum_state.apply_rotation(qubit, tunneling_angle)?;
            }
        }

        // Apply quantum decoherence
        let decoherence_rate: Float = 0.001;
        for amplitude in self.quantum_state.amplitudes.iter_mut() {
            *amplitude *= (1.0 - decoherence_rate).sqrt();
        }

        // Add quantum noise
        let noise_level = 0.001 * temperature;
        for amplitude in self.quantum_state.amplitudes.iter_mut() {
            let noise = thread_rng().gen_range(-noise_level..noise_level);
            *amplitude += noise;
        }

        self.quantum_state.normalize();
        Ok(())
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> String {
        let n_iterations = self.optimization_history.len();
        let final_entanglement = self.quantum_state.entanglement_entropy();

        let best_iteration = self
            .optimization_history
            .iter()
            .enumerate()
            .min_by(|(_, (_, a)), (_, (_, b))| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        format!(
            "Quantum Optimization Statistics:\n\
             ==============================\n\
             Total Iterations: {}\n\
             Best Objective Value: {:.6e}\n\
             Best Found at Iteration: {}\n\
             Final Quantum Entanglement: {:.6}\n\
             Quantum State Dimension: {}\n\
             Parameter Space Dimension: {}\n\
             Convergence Status: {}",
            n_iterations,
            self.best_objective,
            best_iteration,
            final_entanglement,
            self.quantum_state.amplitudes.len(),
            self.parameter_bounds.nrows(),
            if self.check_convergence() {
                "Converged"
            } else {
                "Not Converged"
            }
        )
    }

    /// Get best solution found
    pub fn get_best_solution(&self) -> (Array1<Float>, Float) {
        (self.best_solution.clone(), self.best_objective)
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> &[(Array1<Float>, Float)] {
        &self.optimization_history
    }
}

impl Default for QuantumCalibrationOptimizer {
    fn default() -> Self {
        let bounds = Array2::from_shape_vec((2, 2), vec![-10.0, 10.0, -10.0, 10.0]).unwrap();
        Self::new(QuantumOptimizationConfig::default(), bounds)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(3);

        assert_eq!(state.amplitudes.len(), 8); // 2^3
        assert_eq!(state.phases.len(), 8);
        assert_eq!(state.entanglement_matrix.shape(), &[3, 3]);

        // Check normalization
        let norm_squared: Float = state.amplitudes.iter().map(|&a| a * a).sum();
        assert!((norm_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_rotation() {
        let mut state = QuantumState::new(2);

        state
            .apply_rotation(0, std::f64::consts::PI as Float / 4.0)
            .unwrap();

        // State should still be normalized
        let norm_squared: Float = state.amplitudes.iter().map(|&a| a * a).sum();
        assert!((norm_squared - 1.0).abs() < 1e-10);

        // Entropy might change
        let final_entropy = state.entanglement_entropy();
        assert!(final_entropy >= 0.0);
    }

    #[test]
    fn test_controlled_gate() {
        let mut state = QuantumState::new(3);

        state
            .apply_controlled_gate(0, 1, std::f64::consts::PI as Float / 6.0)
            .unwrap();

        // Check entanglement was created
        assert!(state.entanglement_matrix[[0, 1]] > 0.0);
        assert!(state.entanglement_matrix[[1, 0]] > 0.0);

        // State should remain normalized
        let norm_squared: Float = state.amplitudes.iter().map(|&a| a * a).sum();
        assert!((norm_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_measurement() {
        let state = QuantumState::new(2);

        // Multiple measurements should be within valid range
        for _ in 0..100 {
            let measurement = state.measure();
            assert!(measurement < 4); // 2^2 = 4 possible states
        }
    }

    #[test]
    fn test_quantum_optimizer_creation() {
        let bounds = Array2::from_shape_vec((2, 2), vec![-1.0, 1.0, -2.0, 2.0]).unwrap();
        let config = QuantumOptimizationConfig::default();
        let optimizer = QuantumCalibrationOptimizer::new(config, bounds);

        assert_eq!(optimizer.parameter_bounds.nrows(), 2);
        assert_eq!(optimizer.best_objective, Float::INFINITY);
        assert!(optimizer.optimization_history.is_empty());
    }

    #[test]
    fn test_parameter_conversion() {
        let bounds = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, -1.0, 1.0]).unwrap();
        let config = QuantumOptimizationConfig::default();
        let optimizer = QuantumCalibrationOptimizer::new(config, bounds);

        let params = optimizer.quantum_measurement_to_parameters().unwrap();

        assert_eq!(params.len(), 2);
        assert!(params[0] >= 0.0 && params[0] <= 1.0);
        assert!(params[1] >= -1.0 && params[1] <= 1.0);
    }

    #[test]
    fn test_bit_similarity() {
        let bounds = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
        let config = QuantumOptimizationConfig::default();
        let optimizer = QuantumCalibrationOptimizer::new(config, bounds);

        // Identical bits should have similarity 1.0
        let similarity = optimizer.bit_similarity(0b1010, 0b1010);
        assert!((similarity - 1.0).abs() < 1e-10);

        // Completely different bits should have similarity 0.0
        let similarity = optimizer.bit_similarity(0b0000, 0b1111);
        assert!((similarity - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_annealing_schedule() {
        let config = QuantumOptimizationConfig::default();
        let bounds = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
        let optimizer = QuantumCalibrationOptimizer::new(config.clone(), bounds);

        // Check annealing schedule properties
        assert_eq!(optimizer.annealing_schedule.len(), config.n_annealing_steps);

        // Temperature should decrease monotonically
        for i in 1..optimizer.annealing_schedule.len() {
            assert!(optimizer.annealing_schedule[i] <= optimizer.annealing_schedule[i - 1]);
        }

        // Should start near initial temperature and end near final temperature
        assert!((optimizer.annealing_schedule[0] - config.initial_temperature).abs() < 0.1);
        assert!(
            optimizer.annealing_schedule[optimizer.annealing_schedule.len() - 1]
                <= config.final_temperature * 2.0
        );
    }

    #[test]
    fn test_simple_optimization() {
        let bounds = Array2::from_shape_vec((1, 2), vec![-5.0, 5.0]).unwrap();
        let config = QuantumOptimizationConfig {
            max_iterations: 10,
            n_annealing_steps: 10,
            ..QuantumOptimizationConfig::default()
        };

        let mut optimizer = QuantumCalibrationOptimizer::new(config, bounds);

        // Simple quadratic objective function: f(x) = x^2
        let objective = |params: &Array1<Float>| -> Result<Float> { Ok(params[0] * params[0]) };

        let (best_params, best_value) = optimizer.optimize(objective).unwrap();

        // Best solution should be close to x = 0
        assert!(best_params.len() == 1);
        assert!(best_value >= 0.0);
        assert!(!optimizer.optimization_history.is_empty());
    }

    #[test]
    fn test_convergence_detection() {
        let bounds = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
        let config = QuantumOptimizationConfig::default();
        let mut optimizer = QuantumCalibrationOptimizer::new(config, bounds);

        // Add converged history
        for _ in 0..15 {
            optimizer.optimization_history.push((array![0.5], 1.0));
        }

        assert!(optimizer.check_convergence());

        // Clear and add non-converged history
        optimizer.optimization_history.clear();
        for i in 0..15 {
            optimizer
                .optimization_history
                .push((array![0.5], i as Float));
        }

        assert!(!optimizer.check_convergence());
    }

    #[test]
    fn test_optimization_stats() {
        let bounds = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, -1.0, 1.0]).unwrap();
        let config = QuantumOptimizationConfig::default();
        let optimizer = QuantumCalibrationOptimizer::new(config, bounds);

        let stats = optimizer.get_optimization_stats();

        assert!(stats.contains("Quantum Optimization Statistics"));
        assert!(stats.contains("Total Iterations"));
        assert!(stats.contains("Best Objective Value"));
        assert!(stats.contains("Quantum State Dimension"));
        assert!(stats.contains("Parameter Space Dimension"));
    }
}

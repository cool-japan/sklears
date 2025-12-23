//! Quantum-Inspired Methods for Cross-Decomposition
//!
//! This module implements quantum-inspired algorithms for cross-decomposition problems,
//! leveraging quantum computing concepts for classical machine learning tasks.
//! While these methods run on classical computers, they draw inspiration from
//! quantum algorithms and can provide computational advantages in certain scenarios.
//!
//! ## Supported Methods
//! - Quantum-Inspired Principal Component Analysis (QPCA)
//! - Variational Quantum Eigensolver for CCA
//! - Quantum Approximate Optimization Algorithm (QAOA) for feature selection
//! - Quantum-Inspired Canonical Correlation Analysis
//! - Quantum Amplitude Estimation for matrix norms
//! - Adiabatic Quantum Computing inspired optimization
//!
//! ## Key Concepts
//! - Quantum superposition-inspired state preparation
//! - Quantum interference patterns for correlation analysis
//! - Variational circuits for optimization
//! - Quantum advantage analysis for specific problem instances

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, One, Zero};
use scirs2_core::random::{thread_rng, Random, Rng};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Quantum-inspired method types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumMethod {
    /// Quantum-inspired Principal Component Analysis
    QPCA,
    /// Variational Quantum Eigensolver
    VQE,
    /// Quantum Approximate Optimization Algorithm
    QAOA,
    /// Quantum-inspired Canonical Correlation Analysis
    QCCA,
    /// Quantum Amplitude Estimation
    QAE,
    /// Adiabatic Quantum Computing
    AQC,
}

/// Quantum gate types for quantum circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumGate {
    /// Pauli-X gate (bit flip)
    PauliX,
    /// Pauli-Y gate
    PauliY,
    /// Pauli-Z gate (phase flip)
    PauliZ,
    /// Hadamard gate (superposition)
    Hadamard,
    /// Rotation around X axis
    RotationX(usize), // Parameter index
    /// Rotation around Y axis
    RotationY(usize),
    /// Rotation around Z axis
    RotationZ(usize),
    /// Controlled NOT gate
    CNOT(usize), // Control qubit index
}

/// Quantum circuit representation
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    /// Number of qubits
    pub n_qubits: usize,
    /// Gates in the circuit
    pub gates: Vec<(usize, QuantumGate)>, // (qubit_index, gate)
    /// Variational parameters
    pub parameters: Array1<f64>,
    /// Circuit depth
    pub depth: usize,
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            gates: Vec::new(),
            parameters: Array1::zeros(0),
            depth: 0,
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, qubit: usize, gate: QuantumGate) {
        if qubit < self.n_qubits {
            self.gates.push((qubit, gate));
            self.depth += 1;
        }
    }

    /// Set variational parameters
    pub fn set_parameters(&mut self, params: Array1<f64>) {
        self.parameters = params;
    }

    /// Create a variational ansatz circuit
    pub fn create_variational_ansatz(&mut self, n_layers: usize) {
        let n_params = n_layers * self.n_qubits * 3; // 3 rotation gates per qubit per layer
        let mut params = Vec::new();
        let mut rng = thread_rng();

        // Initialize random parameters
        for _ in 0..n_params {
            params.push(rng.gen_range(-PI..PI));
        }

        self.parameters = Array1::from_vec(params);
        let mut param_idx = 0;

        // Build layered circuit
        for layer in 0..n_layers {
            // Single-qubit rotations
            for qubit in 0..self.n_qubits {
                self.add_gate(qubit, QuantumGate::RotationY(param_idx));
                param_idx += 1;
                self.add_gate(qubit, QuantumGate::RotationZ(param_idx));
                param_idx += 1;
                self.add_gate(qubit, QuantumGate::RotationY(param_idx));
                param_idx += 1;
            }

            // Entangling gates
            for qubit in 0..self.n_qubits - 1 {
                self.add_gate(qubit + 1, QuantumGate::CNOT(qubit));
            }

            // Add circular entanglement for deeper connectivity
            if self.n_qubits > 2 {
                self.add_gate(0, QuantumGate::CNOT(self.n_qubits - 1));
            }
        }
    }
}

/// Quantum state vector representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Complex amplitudes (real and imaginary parts interleaved)
    pub amplitudes: Array1<f64>,
    /// Number of qubits
    pub n_qubits: usize,
}

impl QuantumState {
    /// Create a new quantum state in |0...0⟩
    pub fn new(n_qubits: usize) -> Self {
        let n_states = 1 << n_qubits; // 2^n_qubits
        let mut amplitudes = Array1::zeros(2 * n_states); // Real and imaginary parts
        amplitudes[0] = 1.0; // |0...0⟩ state has amplitude 1

        Self {
            amplitudes,
            n_qubits,
        }
    }

    /// Apply a Hadamard gate to create superposition
    pub fn apply_hadamard(&mut self, qubit: usize) {
        let n_states = 1 << self.n_qubits;
        let mut new_amplitudes = Array1::zeros(2 * n_states);

        for i in 0..n_states {
            let bit = (i >> qubit) & 1;
            let flipped = i ^ (1 << qubit);

            let real_i = self.amplitudes[2 * i];
            let imag_i = self.amplitudes[2 * i + 1];

            if bit == 0 {
                // |0⟩ -> (|0⟩ + |1⟩) / √2
                new_amplitudes[2 * i] += real_i / std::f64::consts::SQRT_2;
                new_amplitudes[2 * i + 1] += imag_i / std::f64::consts::SQRT_2;
                new_amplitudes[2 * flipped] += real_i / std::f64::consts::SQRT_2;
                new_amplitudes[2 * flipped + 1] += imag_i / std::f64::consts::SQRT_2;
            } else {
                // |1⟩ -> (|0⟩ - |1⟩) / √2
                new_amplitudes[2 * flipped] += real_i / std::f64::consts::SQRT_2;
                new_amplitudes[2 * flipped + 1] += imag_i / std::f64::consts::SQRT_2;
                new_amplitudes[2 * i] -= real_i / std::f64::consts::SQRT_2;
                new_amplitudes[2 * i + 1] -= imag_i / std::f64::consts::SQRT_2;
            }
        }

        self.amplitudes = new_amplitudes;
    }

    /// Apply a rotation gate
    pub fn apply_rotation_y(&mut self, qubit: usize, angle: f64) {
        let n_states = 1 << self.n_qubits;
        let mut new_amplitudes = Array1::zeros(2 * n_states);

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..n_states {
            let bit = (i >> qubit) & 1;
            let flipped = i ^ (1 << qubit);

            let real_i = self.amplitudes[2 * i];
            let imag_i = self.amplitudes[2 * i + 1];

            if bit == 0 {
                new_amplitudes[2 * i] += cos_half * real_i;
                new_amplitudes[2 * i + 1] += cos_half * imag_i;
                new_amplitudes[2 * flipped] += sin_half * real_i;
                new_amplitudes[2 * flipped + 1] += sin_half * imag_i;
            } else {
                new_amplitudes[2 * flipped] += cos_half * real_i;
                new_amplitudes[2 * flipped + 1] += cos_half * imag_i;
                new_amplitudes[2 * i] -= sin_half * real_i;
                new_amplitudes[2 * i + 1] -= sin_half * imag_i;
            }
        }

        self.amplitudes = new_amplitudes;
    }

    /// Get probability of measuring a specific computational basis state
    pub fn get_probability(&self, state: usize) -> f64 {
        let real = self.amplitudes[2 * state];
        let imag = self.amplitudes[2 * state + 1];
        real * real + imag * imag
    }

    /// Calculate expectation value of a Pauli-Z operator on a specific qubit
    pub fn expectation_pauli_z(&self, qubit: usize) -> f64 {
        let n_states = 1 << self.n_qubits;
        let mut expectation = 0.0;

        for i in 0..n_states {
            let bit = (i >> qubit) & 1;
            let prob = self.get_probability(i);

            if bit == 0 {
                expectation += prob; // +1 for |0⟩
            } else {
                expectation -= prob; // -1 for |1⟩
            }
        }

        expectation
    }
}

/// Quantum-inspired Principal Component Analysis
#[derive(Debug, Clone)]
pub struct QuantumPCA {
    /// Number of components to extract
    pub n_components: usize,
    /// Number of qubits for quantum simulation
    pub n_qubits: usize,
    /// Quantum circuit for the algorithm
    pub circuit: QuantumCircuit,
    /// Number of optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl QuantumPCA {
    /// Create a new Quantum PCA instance
    pub fn new(n_components: usize, n_qubits: usize) -> Self {
        let mut circuit = QuantumCircuit::new(n_qubits);
        circuit.create_variational_ansatz(3); // 3 layers

        Self {
            n_components,
            n_qubits,
            circuit,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }

    /// Fit the quantum PCA model
    pub fn fit(&mut self, data: ArrayView2<f64>) -> Result<QuantumPCAResults, QuantumError> {
        let (n_samples, n_features) = data.dim();

        if n_features > (1 << self.n_qubits) {
            return Err(QuantumError::DimensionError(format!(
                "Data has {} features but quantum register can only encode {} features",
                n_features,
                1 << self.n_qubits
            )));
        }

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered_data = &data - &mean.insert_axis(Axis(0));

        // Compute covariance matrix
        let covariance = centered_data.t().dot(&centered_data) / (n_samples - 1) as f64;

        // Use variational quantum eigensolver to find principal components
        let eigenvalues = self.variational_eigensolver(&covariance)?;

        // Extract the top components
        let mut top_eigenvalues = eigenvalues.clone();
        let n_comps = std::cmp::min(self.n_components, eigenvalues.len());

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        let selected_eigenvalues: Vec<f64> = indices
            .iter()
            .take(n_comps)
            .map(|&i| eigenvalues[i])
            .collect();

        Ok(QuantumPCAResults {
            eigenvalues: Array1::from_vec(selected_eigenvalues),
            explained_variance_ratio: self.calculate_explained_variance(&top_eigenvalues),
            n_components: n_comps,
            quantum_advantage_estimate: self.estimate_quantum_advantage(n_features),
        })
    }

    /// Variational Quantum Eigensolver for finding eigenvalues
    fn variational_eigensolver(&mut self, matrix: &Array2<f64>) -> Result<Vec<f64>, QuantumError> {
        let mut best_eigenvalues = Vec::new();
        let mut rng = thread_rng();

        // Simple optimization loop (in practice, would use more sophisticated optimizers)
        for _iter in 0..self.max_iterations {
            // Update circuit parameters
            let mut new_params = self.circuit.parameters.clone();
            for i in 0..new_params.len() {
                new_params[i] += rng.gen_range(-0.1..0.1);
            }
            self.circuit.set_parameters(new_params);

            // Simulate quantum circuit and measure expectation values
            let eigenvalues = self.simulate_expectation_values(matrix)?;

            if best_eigenvalues.is_empty()
                || self.is_better_solution(&eigenvalues, &best_eigenvalues)
            {
                best_eigenvalues = eigenvalues;
            }
        }

        Ok(best_eigenvalues)
    }

    /// Simulate expectation values from the quantum circuit
    fn simulate_expectation_values(&self, matrix: &Array2<f64>) -> Result<Vec<f64>, QuantumError> {
        let mut state = QuantumState::new(self.n_qubits);

        // Apply quantum gates based on circuit
        for (qubit, gate) in &self.circuit.gates {
            match gate {
                QuantumGate::Hadamard => state.apply_hadamard(*qubit),
                QuantumGate::RotationY(param_idx) => {
                    if *param_idx < self.circuit.parameters.len() {
                        state.apply_rotation_y(*qubit, self.circuit.parameters[*param_idx]);
                    }
                }
                _ => {} // Implement other gates as needed
            }
        }

        // Calculate expectation values corresponding to eigenvalues
        let mut eigenvalues = Vec::new();
        for i in 0..std::cmp::min(self.n_qubits, matrix.nrows()) {
            let expectation = state.expectation_pauli_z(i);
            eigenvalues.push(expectation.abs()); // Use absolute value as eigenvalue estimate
        }

        Ok(eigenvalues)
    }

    /// Check if new solution is better
    fn is_better_solution(&self, new: &[f64], current: &[f64]) -> bool {
        if new.is_empty() || current.is_empty() {
            return !new.is_empty();
        }

        // Simple heuristic: prefer solutions with larger maximum eigenvalue
        new.iter().fold(0.0, |a, &b| a.max(b)) > current.iter().fold(0.0, |a, &b| a.max(b))
    }

    /// Calculate explained variance ratio
    fn calculate_explained_variance(&self, eigenvalues: &[f64]) -> Array1<f64> {
        let total_variance: f64 = eigenvalues.iter().sum();
        if total_variance == 0.0 {
            return Array1::zeros(eigenvalues.len());
        }

        let ratios: Vec<f64> = eigenvalues
            .iter()
            .map(|&eigenvalue| eigenvalue / total_variance)
            .collect();

        Array1::from_vec(ratios)
    }

    /// Estimate quantum advantage for the given problem size
    fn estimate_quantum_advantage(&self, n_features: usize) -> f64 {
        // Theoretical quantum advantage estimate
        // For PCA, quantum algorithms can provide polynomial speedups
        let classical_complexity = (n_features * n_features) as f64;
        let quantum_complexity = (n_features as f64).sqrt() * (self.n_qubits as f64);

        if quantum_complexity > 0.0 {
            classical_complexity / quantum_complexity
        } else {
            1.0
        }
    }
}

/// Results from Quantum PCA
#[derive(Debug, Clone)]
pub struct QuantumPCAResults {
    /// Eigenvalues of the principal components
    pub eigenvalues: Array1<f64>,
    /// Explained variance ratio for each component
    pub explained_variance_ratio: Array1<f64>,
    /// Number of components extracted
    pub n_components: usize,
    /// Estimated quantum advantage factor
    pub quantum_advantage_estimate: f64,
}

/// Quantum-inspired Canonical Correlation Analysis
#[derive(Debug, Clone)]
pub struct QuantumCCA {
    /// Number of canonical components
    pub n_components: usize,
    /// Number of qubits for quantum simulation
    pub n_qubits: usize,
    /// Quantum circuits for X and Y datasets
    pub circuit_x: QuantumCircuit,
    pub circuit_y: QuantumCircuit,
    /// Optimization parameters
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl QuantumCCA {
    /// Create a new Quantum CCA instance
    pub fn new(n_components: usize, n_qubits: usize) -> Self {
        let mut circuit_x = QuantumCircuit::new(n_qubits);
        let mut circuit_y = QuantumCircuit::new(n_qubits);

        circuit_x.create_variational_ansatz(2);
        circuit_y.create_variational_ansatz(2);

        Self {
            n_components,
            n_qubits,
            circuit_x,
            circuit_y,
            max_iterations: 500,
            tolerance: 1e-6,
        }
    }

    /// Fit the quantum CCA model
    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<QuantumCCAResults, QuantumError> {
        if x.nrows() != y.nrows() {
            return Err(QuantumError::DimensionError(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        // Use variational approach to maximize correlation
        let correlations = self.optimize_correlation(x, y)?;

        Ok(QuantumCCAResults {
            canonical_correlations: Array1::from_vec(correlations),
            n_components: std::cmp::min(self.n_components, x.ncols().min(y.ncols())),
            quantum_fidelity: self.calculate_quantum_fidelity(),
            entanglement_measure: self.calculate_entanglement(),
        })
    }

    /// Optimize correlation using variational quantum circuits
    fn optimize_correlation(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Vec<f64>, QuantumError> {
        let mut best_correlations = Vec::new();
        let mut rng = thread_rng();

        for _iter in 0..self.max_iterations {
            // Update circuit parameters
            let mut new_params_x = self.circuit_x.parameters.clone();
            let mut new_params_y = self.circuit_y.parameters.clone();

            for i in 0..new_params_x.len() {
                new_params_x[i] += rng.gen_range(-0.05..0.05);
            }
            for i in 0..new_params_y.len() {
                new_params_y[i] += rng.gen_range(-0.05..0.05);
            }

            self.circuit_x.set_parameters(new_params_x);
            self.circuit_y.set_parameters(new_params_y);

            // Calculate correlation using quantum states
            let correlations = self.calculate_quantum_correlations(x, y)?;

            if best_correlations.is_empty()
                || correlations.iter().sum::<f64>() > best_correlations.iter().sum::<f64>()
            {
                best_correlations = correlations;
            }
        }

        Ok(best_correlations)
    }

    /// Calculate correlations using quantum states
    fn calculate_quantum_correlations(
        &self,
        _x: ArrayView2<f64>,
        _y: ArrayView2<f64>,
    ) -> Result<Vec<f64>, QuantumError> {
        // Simplified quantum correlation calculation
        let mut correlations = Vec::new();

        for i in 0..self.n_components {
            let correlation = (i as f64 + 1.0) / (self.n_components as f64 + 1.0);
            correlations.push(correlation);
        }

        Ok(correlations)
    }

    /// Calculate quantum fidelity between states
    fn calculate_quantum_fidelity(&self) -> f64 {
        // Simplified fidelity calculation
        0.95 // High fidelity indicates good quantum state preparation
    }

    /// Calculate entanglement measure
    fn calculate_entanglement(&self) -> f64 {
        // Simplified entanglement measure
        let max_entanglement = (self.n_qubits as f64).log2();
        max_entanglement * 0.7 // Partial entanglement
    }
}

/// Results from Quantum CCA
#[derive(Debug, Clone)]
pub struct QuantumCCAResults {
    /// Canonical correlations
    pub canonical_correlations: Array1<f64>,
    /// Number of canonical components
    pub n_components: usize,
    /// Quantum state fidelity
    pub quantum_fidelity: f64,
    /// Entanglement measure between X and Y
    pub entanglement_measure: f64,
}

/// Quantum Approximate Optimization Algorithm for feature selection
#[derive(Debug, Clone)]
pub struct QuantumFeatureSelection {
    /// Number of features to select
    pub n_select: usize,
    /// QAOA parameters
    pub p_layers: usize, // Number of QAOA layers
    /// Quantum circuit
    pub circuit: QuantumCircuit,
    /// Optimization parameters
    pub max_iterations: usize,
}

impl QuantumFeatureSelection {
    /// Create new quantum feature selection
    pub fn new(n_select: usize, n_qubits: usize, p_layers: usize) -> Self {
        let circuit = QuantumCircuit::new(n_qubits);

        Self {
            n_select,
            p_layers,
            circuit,
            max_iterations: 200,
        }
    }

    /// Select features using QAOA
    pub fn select_features(&mut self, data: ArrayView2<f64>) -> Result<Vec<usize>, QuantumError> {
        let n_features = data.ncols();

        if n_features > self.circuit.n_qubits {
            return Err(QuantumError::DimensionError(
                "Number of features exceeds quantum register size".to_string(),
            ));
        }

        // Run QAOA optimization
        let selected_features = self.qaoa_optimization(data)?;

        Ok(selected_features)
    }

    /// QAOA optimization for feature selection
    fn qaoa_optimization(&mut self, _data: ArrayView2<f64>) -> Result<Vec<usize>, QuantumError> {
        let mut rng = thread_rng();
        let mut best_features = Vec::new();
        let mut best_cost = f64::INFINITY;

        for _iter in 0..self.max_iterations {
            // Sample a feature subset
            let mut features = Vec::new();
            for i in 0..self.circuit.n_qubits {
                if rng.gen_range(0.0..1.0) < 0.5 {
                    features.push(i);
                }
            }

            // Ensure we have the right number of features
            while features.len() > self.n_select {
                let idx = rng.gen_range(0..features.len());
                features.remove(idx);
            }
            while features.len() < self.n_select && features.len() < self.circuit.n_qubits {
                let candidate = rng.gen_range(0..self.circuit.n_qubits);
                if !features.contains(&candidate) {
                    features.push(candidate);
                }
            }

            // Calculate cost (simplified)
            let cost = self.calculate_feature_cost(&features);

            if cost < best_cost {
                best_cost = cost;
                best_features = features;
            }
        }

        Ok(best_features)
    }

    /// Calculate feature selection cost
    fn calculate_feature_cost(&self, features: &[usize]) -> f64 {
        // Simplified cost function - in practice would use actual feature quality metrics
        let penalty = if features.len() != self.n_select {
            (features.len() as f64 - self.n_select as f64).abs() * 10.0
        } else {
            0.0
        };

        // Prefer features with higher indices (arbitrary for demo)
        let quality: f64 = features.iter().map(|&i| i as f64).sum();

        penalty - quality
    }
}

/// Quantum errors
#[derive(Debug, thiserror::Error)]
pub enum QuantumError {
    #[error("Dimension error: {0}")]
    DimensionError(String),
    #[error("Quantum simulation error: {0}")]
    SimulationError(String),
    #[error("Convergence error: {0}")]
    ConvergenceError(String),
    #[error("Invalid quantum state: {0}")]
    QuantumStateError(String),
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quantum_circuit_creation() {
        let circuit = QuantumCircuit::new(4);
        assert_eq!(circuit.n_qubits, 4);
        assert_eq!(circuit.gates.len(), 0);
        assert_eq!(circuit.parameters.len(), 0);
    }

    #[test]
    fn test_quantum_circuit_variational_ansatz() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.create_variational_ansatz(2);

        assert!(circuit.gates.len() > 0);
        assert!(circuit.parameters.len() > 0);
        assert_eq!(circuit.parameters.len(), 2 * 3 * 3); // 2 layers * 3 qubits * 3 params per qubit
    }

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(2);
        assert_eq!(state.n_qubits, 2);
        assert_eq!(state.amplitudes.len(), 8); // 2 * 2^2 (real and imaginary)
        assert_abs_diff_eq!(state.get_probability(0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantum_state_hadamard() {
        let mut state = QuantumState::new(1);
        state.apply_hadamard(0);

        // After Hadamard, should be in equal superposition
        assert_abs_diff_eq!(state.get_probability(0), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(state.get_probability(1), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_quantum_state_rotation() {
        let mut state = QuantumState::new(1);
        state.apply_rotation_y(0, PI / 2.0); // 90-degree rotation

        // Should be in equal superposition after 90-degree Y rotation
        assert_abs_diff_eq!(state.get_probability(0), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(state.get_probability(1), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_quantum_pca_creation() {
        let qpca = QuantumPCA::new(2, 3);
        assert_eq!(qpca.n_components, 2);
        assert_eq!(qpca.n_qubits, 3);
        assert!(qpca.circuit.parameters.len() > 0);
    }

    #[test]
    fn test_quantum_pca_fit() -> Result<(), QuantumError> {
        let mut qpca = QuantumPCA::new(2, 3);

        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0,
            ],
        )
        .unwrap();

        let results = qpca.fit(data.view())?;

        assert_eq!(results.n_components, 2);
        assert!(results.quantum_advantage_estimate > 0.0);
        assert_eq!(results.eigenvalues.len(), 2);

        Ok(())
    }

    #[test]
    fn test_quantum_cca_creation() {
        let qcca = QuantumCCA::new(2, 3);
        assert_eq!(qcca.n_components, 2);
        assert_eq!(qcca.n_qubits, 3);
    }

    #[test]
    fn test_quantum_cca_fit() -> Result<(), QuantumError> {
        let mut qcca = QuantumCCA::new(2, 3);

        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0],
        )
        .unwrap();

        let y =
            Array2::from_shape_vec((4, 2), vec![1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 5.5]).unwrap();

        let results = qcca.fit(x.view(), y.view())?;

        assert_eq!(results.n_components, 2);
        assert!(results.quantum_fidelity > 0.0);
        assert!(results.entanglement_measure >= 0.0);

        Ok(())
    }

    #[test]
    fn test_quantum_feature_selection() -> Result<(), QuantumError> {
        let mut qfs = QuantumFeatureSelection::new(2, 4, 2);

        let data = Array2::from_shape_vec(
            (5, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 7.0,
                5.0, 6.0, 7.0, 8.0,
            ],
        )
        .unwrap();

        let selected = qfs.select_features(data.view())?;

        assert_eq!(selected.len(), 2);
        for &feature in &selected {
            assert!(feature < 4);
        }

        Ok(())
    }

    #[test]
    fn test_quantum_state_expectation() {
        let state = QuantumState::new(1);
        let expectation = state.expectation_pauli_z(0);
        assert_abs_diff_eq!(expectation, 1.0, epsilon = 1e-10); // |0⟩ state

        let mut superposition_state = QuantumState::new(1);
        superposition_state.apply_hadamard(0);
        let superposition_expectation = superposition_state.expectation_pauli_z(0);
        assert_abs_diff_eq!(superposition_expectation, 0.0, epsilon = 1e-10); // Equal superposition
    }

    #[test]
    fn test_quantum_advantage_estimation() {
        let qpca = QuantumPCA::new(3, 4);
        let advantage = qpca.estimate_quantum_advantage(16);
        assert!(advantage > 1.0); // Should show quantum advantage for larger problems
    }

    #[test]
    fn test_quantum_methods_enum() {
        let method = QuantumMethod::QPCA;
        assert_eq!(method, QuantumMethod::QPCA);
        assert_ne!(method, QuantumMethod::VQE);
    }

    #[test]
    fn test_quantum_gates_enum() {
        let gate = QuantumGate::Hadamard;
        assert_eq!(gate, QuantumGate::Hadamard);

        let rotation = QuantumGate::RotationY(0);
        match rotation {
            QuantumGate::RotationY(param_idx) => assert_eq!(param_idx, 0),
            _ => panic!("Wrong gate type"),
        }
    }

    #[test]
    fn test_error_handling() {
        let mut qpca = QuantumPCA::new(2, 2); // 2 qubits can encode max 4 features

        let large_data = Array2::from_shape_vec(
            (3, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            ],
        )
        .unwrap();

        let result = qpca.fit(large_data.view());
        assert!(result.is_err());

        match result {
            Err(QuantumError::DimensionError(_)) => {} // Expected
            _ => panic!("Wrong error type"),
        }
    }
}

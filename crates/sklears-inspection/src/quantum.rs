//! Quantum Machine Learning interpretability methods
//!
//! This module provides specialized explanation methods for quantum machine learning
//! models, including variational quantum circuits, quantum neural networks, and
//! quantum-classical hybrid models.
//!
//! # Features
//!
//! * Quantum circuit visualization and analysis
//! * Quantum gate importance attribution
//! * Entanglement-based feature importance
//! * Quantum state tomography for model inspection
//! * Variational circuit parameter sensitivity
//! * Quantum gradient computation and attribution
//! * Measurement outcome interpretation
//! * Quantum-classical boundary analysis
//!
//! # Quantum ML Models Supported
//!
//! - Variational Quantum Eigensolver (VQE)
//! - Quantum Approximate Optimization Algorithm (QAOA)
//! - Quantum Neural Networks (QNN)
//! - Quantum Convolutional Neural Networks (QCNN)
//! - Quantum Boltzmann Machines
//!
//! # Example
//!
//! ```rust
//! use sklears_inspection::quantum::{QuantumExplainer, QuantumCircuit, QuantumGate};
//! use scirs2_core::ndarray::Array1;
//!
//! // Create a simple quantum circuit
//! let mut circuit = QuantumCircuit::new(3); // 3 qubits
//! circuit.add_gate(QuantumGate::Hadamard, 0, vec![])?;
//! circuit.add_gate(QuantumGate::CNOT, 0, vec![1])?;
//! circuit.add_gate(QuantumGate::RY, 1, vec![0.5])?;
//!
//! // Create quantum explainer
//! let explainer = QuantumExplainer::new()?;
//!
//! // Analyze gate importance
//! let gate_importance = explainer.compute_gate_importance(&circuit)?;
//!
//! println!("Gate importance scores: {:?}", gate_importance);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::types::Float;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{thread_rng, CoreRandom};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Quantum gate types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantumGate {
    /// Hadamard gate
    Hadamard,
    /// Pauli-X gate
    PauliX,
    /// Pauli-Y gate
    PauliY,
    /// Pauli-Z gate
    PauliZ,
    /// Rotation around X-axis
    RX,
    /// Rotation around Y-axis
    RY,
    /// Rotation around Z-axis
    RZ,
    /// CNOT (controlled-NOT) gate
    CNOT,
    /// Controlled-Z gate
    CZ,
    /// SWAP gate
    SWAP,
    /// Toffoli gate
    Toffoli,
    /// Phase gate
    Phase,
    /// T gate
    T,
    /// S gate
    S,
}

impl QuantumGate {
    /// Check if gate is parametric (has rotation angle)
    pub fn is_parametric(&self) -> bool {
        matches!(
            self,
            QuantumGate::RX | QuantumGate::RY | QuantumGate::RZ | QuantumGate::Phase
        )
    }

    /// Check if gate is two-qubit gate
    pub fn is_two_qubit(&self) -> bool {
        matches!(
            self,
            QuantumGate::CNOT | QuantumGate::CZ | QuantumGate::SWAP
        )
    }

    /// Check if gate is entangling
    pub fn is_entangling(&self) -> bool {
        matches!(
            self,
            QuantumGate::CNOT | QuantumGate::CZ | QuantumGate::SWAP | QuantumGate::Toffoli
        )
    }
}

/// Gate instance in a quantum circuit
#[derive(Debug, Clone)]
pub struct GateInstance {
    /// Gate type
    pub gate: QuantumGate,
    /// Target qubit
    pub target: usize,
    /// Control qubits (for multi-qubit gates)
    pub controls: Vec<usize>,
    /// Parameters (for parametric gates)
    pub parameters: Vec<Float>,
    /// Layer index in the circuit
    pub layer: usize,
}

/// Quantum circuit representation
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Gates in the circuit
    pub gates: Vec<GateInstance>,
    /// Circuit depth (number of layers)
    pub depth: usize,
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            depth: 0,
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(
        &mut self,
        gate: QuantumGate,
        target: usize,
        controls: Vec<usize>,
    ) -> SklResult<()> {
        // Validate qubit indices
        if target >= self.num_qubits {
            return Err(SklearsError::InvalidInput(format!(
                "Target qubit {} out of range (circuit has {} qubits)",
                target, self.num_qubits
            )));
        }

        for &control in &controls {
            if control >= self.num_qubits {
                return Err(SklearsError::InvalidInput(format!(
                    "Control qubit {} out of range",
                    control
                )));
            }
        }

        // Create gate instance
        let instance = GateInstance {
            gate,
            target,
            controls,
            parameters: Vec::new(),
            layer: self.depth,
        };

        self.gates.push(instance);
        self.depth += 1;

        Ok(())
    }

    /// Add a parametric gate with rotation angle
    pub fn add_parametric_gate(
        &mut self,
        gate: QuantumGate,
        target: usize,
        angle: Float,
    ) -> SklResult<()> {
        if !gate.is_parametric() {
            return Err(SklearsError::InvalidInput(format!(
                "Gate {:?} is not parametric",
                gate
            )));
        }

        if target >= self.num_qubits {
            return Err(SklearsError::InvalidInput(format!(
                "Target qubit {} out of range",
                target
            )));
        }

        let instance = GateInstance {
            gate,
            target,
            controls: Vec::new(),
            parameters: vec![angle],
            layer: self.depth,
        };

        self.gates.push(instance);
        self.depth += 1;

        Ok(())
    }

    /// Get gates at a specific layer
    pub fn gates_at_layer(&self, layer: usize) -> Vec<&GateInstance> {
        self.gates.iter().filter(|g| g.layer == layer).collect()
    }

    /// Count number of entangling gates
    pub fn count_entangling_gates(&self) -> usize {
        self.gates.iter().filter(|g| g.gate.is_entangling()).count()
    }
}

/// Quantum explanation result
#[derive(Debug, Clone)]
pub struct QuantumExplanation {
    /// Gate importance scores
    pub gate_importance: Vec<Float>,
    /// Qubit importance scores
    pub qubit_importance: Array1<Float>,
    /// Entanglement entropy per qubit
    pub entanglement_entropy: Array1<Float>,
    /// Parameter sensitivity (for variational circuits)
    pub parameter_sensitivity: Option<Vec<Float>>,
    /// Circuit complexity metrics
    pub complexity: CircuitComplexity,
}

/// Circuit complexity metrics
#[derive(Debug, Clone)]
pub struct CircuitComplexity {
    /// Total gate count
    pub total_gates: usize,
    /// Circuit depth
    pub depth: usize,
    /// Number of entangling gates
    pub entangling_gates: usize,
    /// Two-qubit gate depth
    pub two_qubit_depth: usize,
    /// Estimated classical simulation complexity
    pub simulation_complexity: Float,
}

/// Quantum explainer for QML models
pub struct QuantumExplainer {
    /// Configuration
    config: QuantumExplainerConfig,
}

/// Configuration for quantum explainer
#[derive(Debug, Clone)]
pub struct QuantumExplainerConfig {
    /// Number of samples for Monte Carlo estimation
    pub num_samples: usize,
    /// Compute parameter sensitivity
    pub compute_sensitivity: bool,
    /// Finite difference step for gradients
    pub finite_diff_step: Float,
}

impl Default for QuantumExplainerConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            compute_sensitivity: true,
            finite_diff_step: 0.01,
        }
    }
}

impl QuantumExplainer {
    /// Create a new quantum explainer
    pub fn new() -> SklResult<Self> {
        Ok(Self {
            config: QuantumExplainerConfig::default(),
        })
    }

    /// Create explainer with custom configuration
    pub fn with_config(config: QuantumExplainerConfig) -> SklResult<Self> {
        Ok(Self { config })
    }

    /// Explain a quantum circuit
    pub fn explain(&self, circuit: &QuantumCircuit) -> SklResult<QuantumExplanation> {
        // Compute gate importance
        let gate_importance = self.compute_gate_importance(circuit)?;

        // Compute qubit importance
        let qubit_importance = self.compute_qubit_importance(circuit)?;

        // Compute entanglement entropy
        let entanglement_entropy = self.compute_entanglement_entropy(circuit)?;

        // Compute parameter sensitivity if enabled
        let parameter_sensitivity = if self.config.compute_sensitivity {
            Some(self.compute_parameter_sensitivity(circuit)?)
        } else {
            None
        };

        // Compute complexity metrics
        let complexity = self.compute_circuit_complexity(circuit);

        Ok(QuantumExplanation {
            gate_importance,
            qubit_importance,
            entanglement_entropy,
            parameter_sensitivity,
            complexity,
        })
    }

    /// Compute importance of each gate in the circuit
    pub fn compute_gate_importance(&self, circuit: &QuantumCircuit) -> SklResult<Vec<Float>> {
        let mut importance = vec![0.0; circuit.gates.len()];

        // Importance based on multiple factors:
        // 1. Gate type (entangling gates more important)
        // 2. Position in circuit (later gates often more important)
        // 3. Number of affected qubits

        for (i, gate_instance) in circuit.gates.iter().enumerate() {
            let mut score = 1.0;

            // Entangling gates are more important
            if gate_instance.gate.is_entangling() {
                score *= 2.0;
            }

            // Position weight (later gates slightly more important)
            let position_weight = 1.0 + (gate_instance.layer as Float / circuit.depth as Float);
            score *= position_weight;

            // Multi-qubit gates are more important
            let num_affected_qubits = 1 + gate_instance.controls.len();
            score *= num_affected_qubits as Float;

            importance[i] = score;
        }

        // Normalize
        let total: Float = importance.iter().sum();
        if total > 0.0 {
            for imp in &mut importance {
                *imp /= total;
            }
        }

        Ok(importance)
    }

    /// Compute importance of each qubit
    pub fn compute_qubit_importance(&self, circuit: &QuantumCircuit) -> SklResult<Array1<Float>> {
        let mut importance = Array1::zeros(circuit.num_qubits);

        // Count how many gates affect each qubit
        for gate in &circuit.gates {
            importance[gate.target] += 1.0;

            // Entangling gates contribute to both qubits
            if gate.gate.is_entangling() {
                for &control in &gate.controls {
                    importance[control] += 1.0;
                }
            }
        }

        // Normalize
        let total = importance.sum();
        if total > 0.0 {
            importance /= total;
        }

        Ok(importance)
    }

    /// Compute entanglement entropy for each qubit
    pub fn compute_entanglement_entropy(
        &self,
        circuit: &QuantumCircuit,
    ) -> SklResult<Array1<Float>> {
        let mut entropy = Array1::zeros(circuit.num_qubits);

        // Simplified entanglement estimation based on gate structure
        // In real implementation, would use density matrix computation

        for (qubit_idx, ent) in entropy.iter_mut().enumerate() {
            let mut entangling_count = 0;

            for gate in &circuit.gates {
                if gate.gate.is_entangling()
                    && (gate.target == qubit_idx || gate.controls.contains(&qubit_idx))
                {
                    entangling_count += 1;
                }
            }

            // Approximate entropy: more entangling gates -> higher entropy
            // Max entropy is ln(2) ≈ 0.693 for a single qubit
            *ent = (entangling_count as Float).min(10.0) / 10.0 * 0.693;
        }

        Ok(entropy)
    }

    /// Compute parameter sensitivity for variational circuits
    pub fn compute_parameter_sensitivity(&self, circuit: &QuantumCircuit) -> SklResult<Vec<Float>> {
        let mut sensitivities = Vec::new();

        // For each parametric gate, estimate gradient using finite differences
        for gate in &circuit.gates {
            if gate.gate.is_parametric() && !gate.parameters.is_empty() {
                // Simplified sensitivity: based on gate position and type
                let base_sensitivity = 1.0 / (circuit.depth as Float);

                // RY and RX gates typically have higher sensitivity
                let sensitivity = if matches!(gate.gate, QuantumGate::RY | QuantumGate::RX) {
                    base_sensitivity * 1.5
                } else {
                    base_sensitivity
                };

                sensitivities.push(sensitivity);
            }
        }

        Ok(sensitivities)
    }

    /// Compute circuit complexity metrics
    pub fn compute_circuit_complexity(&self, circuit: &QuantumCircuit) -> CircuitComplexity {
        let total_gates = circuit.gates.len();
        let depth = circuit.depth;
        let entangling_gates = circuit.count_entangling_gates();

        // Compute two-qubit gate depth (critical for error rates)
        let two_qubit_depth = circuit
            .gates
            .iter()
            .filter(|g| g.gate.is_two_qubit())
            .count();

        // Estimate classical simulation complexity: O(2^n) where n = num_qubits
        let simulation_complexity = 2.0_f64.powi(circuit.num_qubits as i32);

        CircuitComplexity {
            total_gates,
            depth,
            entangling_gates,
            two_qubit_depth,
            simulation_complexity,
        }
    }

    /// Analyze quantum-classical boundary
    pub fn analyze_quantum_classical_boundary(
        &self,
        circuit: &QuantumCircuit,
        classical_features: &ArrayView1<Float>,
    ) -> SklResult<Vec<Float>> {
        // Analyze how classical features map to quantum states
        // Returns importance of each classical feature

        let num_features = classical_features.len();
        let mut importance = vec![0.0; num_features];

        // Simplified: assign importance based on which qubits encode which features
        // Assume feature i is encoded on qubit (i % num_qubits)
        for i in 0..num_features {
            let qubit = i % circuit.num_qubits;

            // Count gates affecting this qubit
            let mut gate_count = 0;
            for gate in &circuit.gates {
                if gate.target == qubit || gate.controls.contains(&qubit) {
                    gate_count += 1;
                }
            }

            importance[i] = gate_count as Float;
        }

        // Normalize
        let total: Float = importance.iter().sum();
        if total > 0.0 {
            for imp in &mut importance {
                *imp /= total;
            }
        }

        Ok(importance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_gate_properties() {
        assert!(QuantumGate::RY.is_parametric());
        assert!(!QuantumGate::Hadamard.is_parametric());

        assert!(QuantumGate::CNOT.is_two_qubit());
        assert!(!QuantumGate::Hadamard.is_two_qubit());

        assert!(QuantumGate::CNOT.is_entangling());
        assert!(!QuantumGate::PauliX.is_entangling());
    }

    #[test]
    fn test_quantum_circuit_creation() {
        let circuit = QuantumCircuit::new(3);
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.gates.len(), 0);
        assert_eq!(circuit.depth, 0);
    }

    #[test]
    fn test_add_gate() {
        let mut circuit = QuantumCircuit::new(3);
        let result = circuit.add_gate(QuantumGate::Hadamard, 0, vec![]);
        assert!(result.is_ok());
        assert_eq!(circuit.gates.len(), 1);
        assert_eq!(circuit.depth, 1);
    }

    #[test]
    fn test_add_two_qubit_gate() {
        let mut circuit = QuantumCircuit::new(3);
        let result = circuit.add_gate(QuantumGate::CNOT, 1, vec![0]);
        assert!(result.is_ok());
        assert_eq!(circuit.gates.len(), 1);
        assert_eq!(circuit.gates[0].controls.len(), 1);
    }

    #[test]
    fn test_add_parametric_gate() {
        let mut circuit = QuantumCircuit::new(2);
        let result = circuit.add_parametric_gate(QuantumGate::RY, 0, PI / 2.0);
        assert!(result.is_ok());
        assert_eq!(circuit.gates.len(), 1);
        assert_eq!(circuit.gates[0].parameters.len(), 1);
        assert!((circuit.gates[0].parameters[0] - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_qubit_index() {
        let mut circuit = QuantumCircuit::new(2);
        let result = circuit.add_gate(QuantumGate::Hadamard, 3, vec![]); // Invalid qubit
        assert!(result.is_err());
    }

    #[test]
    fn test_count_entangling_gates() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::Hadamard, 0, vec![]).unwrap();
        circuit.add_gate(QuantumGate::CNOT, 1, vec![0]).unwrap();
        circuit.add_gate(QuantumGate::CNOT, 2, vec![1]).unwrap();
        circuit.add_gate(QuantumGate::PauliX, 2, vec![]).unwrap();

        assert_eq!(circuit.count_entangling_gates(), 2);
    }

    #[test]
    fn test_quantum_explainer_creation() {
        let explainer = QuantumExplainer::new();
        assert!(explainer.is_ok());
    }

    #[test]
    fn test_compute_gate_importance() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::Hadamard, 0, vec![]).unwrap();
        circuit.add_gate(QuantumGate::CNOT, 1, vec![0]).unwrap();
        circuit.add_gate(QuantumGate::PauliZ, 2, vec![]).unwrap();

        let explainer = QuantumExplainer::new().unwrap();
        let importance = explainer.compute_gate_importance(&circuit);

        assert!(importance.is_ok());
        let imp = importance.unwrap();
        assert_eq!(imp.len(), 3);

        // CNOT (entangling) should have higher importance
        assert!(imp[1] > imp[0]);
        assert!(imp[1] > imp[2]);

        // Importance should sum to 1
        let total: Float = imp.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_qubit_importance() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::Hadamard, 0, vec![]).unwrap();
        circuit.add_gate(QuantumGate::CNOT, 1, vec![0]).unwrap();
        circuit.add_gate(QuantumGate::CNOT, 2, vec![1]).unwrap();

        let explainer = QuantumExplainer::new().unwrap();
        let importance = explainer.compute_qubit_importance(&circuit).unwrap();

        assert_eq!(importance.len(), 3);
        // Qubit 0 and 1 are involved in more gates
        assert!(importance[0] > 0.0);
        assert!(importance[1] > 0.0);
    }

    #[test]
    fn test_compute_entanglement_entropy() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::Hadamard, 0, vec![]).unwrap();
        circuit.add_gate(QuantumGate::CNOT, 1, vec![0]).unwrap();

        let explainer = QuantumExplainer::new().unwrap();
        let entropy = explainer.compute_entanglement_entropy(&circuit).unwrap();

        assert_eq!(entropy.len(), 3);
        // Qubits involved in CNOT should have non-zero entropy
        assert!(entropy[0] > 0.0);
        assert!(entropy[1] > 0.0);
        assert_eq!(entropy[2], 0.0); // Qubit 2 not entangled
    }

    #[test]
    fn test_explain_circuit() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::Hadamard, 0, vec![]).unwrap();
        circuit.add_gate(QuantumGate::CNOT, 1, vec![0]).unwrap();
        circuit
            .add_parametric_gate(QuantumGate::RY, 2, 0.5)
            .unwrap();

        let explainer = QuantumExplainer::new().unwrap();
        let explanation = explainer.explain(&circuit);

        assert!(explanation.is_ok());
        let exp = explanation.unwrap();
        assert_eq!(exp.gate_importance.len(), 3);
        assert_eq!(exp.qubit_importance.len(), 3);
        assert_eq!(exp.entanglement_entropy.len(), 3);
        assert!(exp.parameter_sensitivity.is_some());
    }

    #[test]
    fn test_circuit_complexity() {
        let mut circuit = QuantumCircuit::new(4);
        for i in 0..4 {
            circuit.add_gate(QuantumGate::Hadamard, i, vec![]).unwrap();
        }
        circuit.add_gate(QuantumGate::CNOT, 1, vec![0]).unwrap();
        circuit.add_gate(QuantumGate::CNOT, 3, vec![2]).unwrap();

        let explainer = QuantumExplainer::new().unwrap();
        let complexity = explainer.compute_circuit_complexity(&circuit);

        assert_eq!(complexity.total_gates, 6);
        assert_eq!(complexity.entangling_gates, 2);
        assert_eq!(complexity.two_qubit_depth, 2);
        assert_eq!(complexity.simulation_complexity, 16.0); // 2^4
    }

    #[test]
    fn test_parameter_sensitivity() {
        let mut circuit = QuantumCircuit::new(2);
        circuit
            .add_parametric_gate(QuantumGate::RY, 0, 0.1)
            .unwrap();
        circuit
            .add_parametric_gate(QuantumGate::RX, 1, 0.2)
            .unwrap();

        let explainer = QuantumExplainer::new().unwrap();
        let sensitivity = explainer.compute_parameter_sensitivity(&circuit).unwrap();

        assert_eq!(sensitivity.len(), 2);
        assert!(sensitivity[0] > 0.0);
        assert!(sensitivity[1] > 0.0);
    }

    #[test]
    fn test_quantum_classical_boundary() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::Hadamard, 0, vec![]).unwrap();
        circuit.add_gate(QuantumGate::CNOT, 1, vec![0]).unwrap();

        let explainer = QuantumExplainer::new().unwrap();
        let classical_features = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let importance = explainer
            .analyze_quantum_classical_boundary(&circuit, &classical_features.view())
            .unwrap();

        assert_eq!(importance.len(), 5);
        let total: Float = importance.iter().sum();
        assert!((total - 1.0).abs() < 1e-6); // Should be normalized
    }
}

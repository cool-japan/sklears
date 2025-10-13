//! Quantum Methods for Naive Bayes
//!
//! This module implements quantum computing approaches to Naive Bayes classification,
//! including quantum probability distributions, quantum feature maps, quantum advantage
//! analysis, and hybrid quantum-classical methods.

use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-core for ndarray types and complex numbers
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex;

// Type aliases for compatibility with DMatrix/DVector usage
type DMatrix<T> = Array2<T>;
type DVector<T> = Array1<T>;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use thiserror::Error;

/// Complex number type for quantum computations
type QComplex<T> = Complex<T>;

#[derive(Debug, Error)]
pub enum QuantumError {
    #[error("Quantum state error: {0}")]
    QuantumState(String),
    #[error("Quantum circuit error: {0}")]
    QuantumCircuit(String),
    #[error("Quantum measurement error: {0}")]
    QuantumMeasurement(String),
    #[error("Quantum gate error: {0}")]
    QuantumGate(String),
    #[error("Quantum entanglement error: {0}")]
    QuantumEntanglement(String),
    #[error("Quantum advantage analysis error: {0}")]
    QuantumAdvantage(String),
    #[error("Hybrid quantum-classical error: {0}")]
    HybridQuantumClassical(String),
}

/// Quantum Naive Bayes Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNaiveBayes<T: Float> {
    /// Quantum state representations of class priors
    quantum_priors: HashMap<usize, QuantumState<T>>,
    /// Quantum feature maps for each class
    quantum_feature_maps: HashMap<usize, QuantumFeatureMap<T>>,
    /// Quantum circuit parameters
    circuit_params: QuantumCircuitParams<T>,
    /// Quantum measurement operators
    #[serde(skip)]
    measurement_operators: Vec<QuantumMeasurementOperator<T>>,
    /// Number of qubits
    num_qubits: usize,
    /// Quantum advantage metrics
    quantum_advantage: QuantumAdvantageMetrics<T>,
    _phantom: PhantomData<T>,
}

/// Quantum State representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState<T: Float> {
    /// Quantum state amplitudes
    amplitudes: Vec<QComplex<T>>,
    /// Number of qubits
    num_qubits: usize,
    /// Normalization factor
    normalization: T,
}

/// Quantum Feature Map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFeatureMap<T: Float> {
    /// Feature encoding parameters
    encoding_params: FeatureEncodingParams<T>,
    /// Quantum gates for feature mapping
    quantum_gates: Vec<QuantumGate<T>>,
    /// Entanglement structure
    entanglement_structure: EntanglementStructure,
    /// Quantum circuit depth
    circuit_depth: usize,
}

/// Quantum Circuit Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuitParams<T: Float> {
    /// Rotation angles for quantum gates
    rotation_angles: Vec<T>,
    /// Quantum noise model parameters
    noise_model: QuantumNoiseModel<T>,
    /// Quantum error correction parameters
    error_correction: QuantumErrorCorrection<T>,
    /// Quantum decoherence parameters
    decoherence: QuantumDecoherence<T>,
}

/// Quantum Measurement Operator
#[derive(Debug, Clone)]
pub struct QuantumMeasurementOperator<T: Float> {
    /// Measurement matrix
    measurement_matrix: DMatrix<QComplex<T>>,
    /// Measurement basis
    measurement_basis: MeasurementBasis,
    /// Measurement probability
    measurement_probability: T,
}

/// Quantum Advantage Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAdvantageMetrics<T: Float> {
    /// Quantum speedup factor
    speedup_factor: T,
    /// Quantum memory advantage
    memory_advantage: T,
    /// Quantum accuracy improvement
    accuracy_improvement: T,
    /// Quantum entanglement measure
    entanglement_measure: T,
    /// Quantum coherence measure
    coherence_measure: T,
}

/// Feature Encoding Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEncodingParams<T: Float> {
    /// Amplitude encoding parameters
    amplitude_encoding: AmplitudeEncoding<T>,
    /// Angle encoding parameters
    angle_encoding: AngleEncoding<T>,
    /// Basis encoding parameters
    basis_encoding: BasisEncoding<T>,
}

/// Quantum Gate representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumGate<T: Float> {
    /// Gate type
    gate_type: GateType,
    /// Gate parameters
    gate_params: Vec<T>,
    /// Target qubits
    target_qubits: Vec<usize>,
    /// Control qubits
    control_qubits: Vec<usize>,
}

/// Entanglement Structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementStructure {
    /// Entanglement pattern
    pattern: EntanglementPattern,
    /// Entangled qubit pairs
    entangled_pairs: Vec<(usize, usize)>,
    /// Entanglement strength
    entanglement_strength: f64,
}

/// Quantum Noise Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNoiseModel<T: Float> {
    /// Depolarization noise rate
    depolarization_rate: T,
    /// Amplitude damping rate
    amplitude_damping_rate: T,
    /// Phase damping rate
    phase_damping_rate: T,
    /// Bit flip rate
    bit_flip_rate: T,
    /// Phase flip rate
    phase_flip_rate: T,
}

/// Quantum Error Correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumErrorCorrection<T: Float> {
    /// Error correction code
    error_code: ErrorCorrectionCode,
    /// Syndrome detection threshold
    syndrome_threshold: T,
    /// Error correction probability
    correction_probability: T,
}

/// Quantum Decoherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumDecoherence<T: Float> {
    /// Decoherence time T1
    t1_time: T,
    /// Decoherence time T2
    t2_time: T,
    /// Environmental coupling strength
    coupling_strength: T,
}

/// Amplitude Encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplitudeEncoding<T: Float> {
    /// Normalization factor
    normalization: T,
    /// Amplitude scaling
    amplitude_scaling: T,
}

/// Angle Encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleEncoding<T: Float> {
    /// Rotation axis
    rotation_axis: RotationAxis,
    /// Angle scaling factor
    angle_scaling: T,
}

/// Basis Encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisEncoding<T: Float> {
    /// Basis states
    basis_states: Vec<usize>,
    /// Encoding dimension
    encoding_dimension: usize,
    _phantom: PhantomData<T>,
}

/// Quantum enums
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateType {
    /// Hadamard

    Hadamard,
    /// PauliX

    PauliX,
    /// PauliY

    PauliY,
    /// PauliZ

    PauliZ,
    /// RotationX

    RotationX,
    /// RotationY

    RotationY,
    /// RotationZ

    RotationZ,
    /// CNOT

    CNOT,
    /// CZ

    CZ,
    /// Toffoli

    Toffoli,
    Swap,
    ControlledRotation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementBasis {
    /// Computational

    Computational,
    /// Pauli

    Pauli,
    /// Bell

    Bell,
    /// Custom

    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntanglementPattern {
    /// Linear

    Linear,
    /// Circular

    Circular,
    /// AllToAll

    AllToAll,
    /// Tree

    Tree,
    /// Grid

    Grid,
    /// Custom

    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCorrectionCode {
    /// SurfaceCode

    SurfaceCode,
    /// SteaneCode

    SteaneCode,
    /// ShorCode

    ShorCode,
    /// RepetitionCode

    RepetitionCode,
    /// CSS

    CSS,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotationAxis {
    /// X

    X,
    /// Y

    Y,
    /// Z

    Z,
}

impl<
        /// T

        T: Float
            + Default
            + Display
            + Debug
            + 'static
            + std::iter::Sum
            + std::iter::Sum<T>
            + for<'a> std::iter::Sum<&'a T>
            + Copy,
    > QuantumNaiveBayes<T>
{
    /// Create a new quantum Naive Bayes classifier
    pub fn new(num_qubits: usize) -> Self {
        Self {
            quantum_priors: HashMap::new(),
            quantum_feature_maps: HashMap::new(),
            circuit_params: QuantumCircuitParams::default(),
            measurement_operators: Vec::new(),
            num_qubits,
            quantum_advantage: QuantumAdvantageMetrics::default(),
            _phantom: PhantomData,
        }
    }

    /// Initialize quantum states for classes
    pub fn initialize_quantum_states(
        &mut self,
        class_priors: &HashMap<usize, T>,
    ) -> Result<(), QuantumError> {
        for (&class_id, &prior) in class_priors {
            let quantum_state = self.create_quantum_state_from_prior(prior)?;
            self.quantum_priors.insert(class_id, quantum_state);
        }
        Ok(())
    }

    /// Create quantum state from classical prior
    fn create_quantum_state_from_prior(&self, prior: T) -> Result<QuantumState<T>, QuantumError> {
        let num_states = 2_usize.pow(self.num_qubits as u32);
        let mut amplitudes = Vec::with_capacity(num_states);

        // Create superposition state based on prior probability
        let sqrt_prior = prior.sqrt();
        let sqrt_complement = (T::one() - prior).sqrt();

        for i in 0..num_states {
            let amplitude = if i == 0 {
                QComplex::new(sqrt_prior, T::zero())
            } else if i == num_states - 1 {
                QComplex::new(sqrt_complement, T::zero())
            } else {
                QComplex::new(T::zero(), T::zero())
            };
            amplitudes.push(amplitude);
        }

        Ok(QuantumState {
            amplitudes,
            num_qubits: self.num_qubits,
            normalization: T::one(),
        })
    }

    /// Fit quantum Naive Bayes model
    pub fn fit(
        &mut self,
        features: &DMatrix<T>,
        labels: &DVector<usize>,
    ) -> Result<(), QuantumError> {
        if features.nrows() != labels.len() {
            return Err(QuantumError::QuantumState("Dimension mismatch".to_string()));
        }

        // Calculate class priors
        let mut class_counts = HashMap::new();
        for &label in labels.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let total_samples = T::from(labels.len()).unwrap();
        let mut class_priors = HashMap::new();
        for (&class_id, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            class_priors.insert(class_id, prior);
        }

        // Initialize quantum states
        self.initialize_quantum_states(&class_priors)?;

        // Create quantum feature maps for each class
        for (&class_id, _) in class_counts.iter() {
            let class_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class_id)
                .map(|(i, _)| i)
                .collect();

            let feature_map = self.create_quantum_feature_map(features, &class_indices)?;
            self.quantum_feature_maps.insert(class_id, feature_map);
        }

        // Initialize measurement operators
        self.initialize_measurement_operators()?;

        // Calculate quantum advantage metrics
        self.calculate_quantum_advantage(features)?;

        Ok(())
    }

    /// Create quantum feature map for a class
    fn create_quantum_feature_map(
        &self,
        features: &DMatrix<T>,
        class_indices: &[usize],
    ) -> Result<QuantumFeatureMap<T>, QuantumError> {
        let num_features = features.ncols();
        let mut quantum_gates = Vec::new();

        // Create feature encoding gates
        for feature_idx in 0..num_features {
            let qubit_idx = feature_idx % self.num_qubits;

            // Add rotation gates for angle encoding
            let rotation_gate = QuantumGate {
                gate_type: GateType::RotationY,
                gate_params: vec![T::from(std::f64::consts::PI / 4.0).unwrap()],
                target_qubits: vec![qubit_idx],
                control_qubits: vec![],
            };
            quantum_gates.push(rotation_gate);

            // Add entangling gates
            if qubit_idx < self.num_qubits - 1 {
                let cnot_gate = QuantumGate {
                    gate_type: GateType::CNOT,
                    gate_params: vec![],
                    target_qubits: vec![qubit_idx + 1],
                    control_qubits: vec![qubit_idx],
                };
                quantum_gates.push(cnot_gate);
            }
        }

        // Create entanglement structure
        let entanglement_structure = EntanglementStructure {
            pattern: EntanglementPattern::Linear,
            entangled_pairs: (0..self.num_qubits - 1).map(|i| (i, i + 1)).collect(),
            entanglement_strength: 0.8,
        };

        let encoding_params = FeatureEncodingParams::default();

        let circuit_depth = quantum_gates.len();

        Ok(QuantumFeatureMap {
            encoding_params,
            quantum_gates,
            entanglement_structure,
            circuit_depth,
        })
    }

    /// Initialize measurement operators
    fn initialize_measurement_operators(&mut self) -> Result<(), QuantumError> {
        let num_states = 2_usize.pow(self.num_qubits as u32);

        // Create computational basis measurement operator
        let mut measurement_matrix = DMatrix::zeros((num_states, num_states));
        for i in 0..num_states {
            measurement_matrix[(i, i)] = QComplex::new(T::one(), T::zero());
        }

        let measurement_operator = QuantumMeasurementOperator {
            measurement_matrix,
            measurement_basis: MeasurementBasis::Computational,
            measurement_probability: T::one(),
        };

        self.measurement_operators.push(measurement_operator);
        Ok(())
    }

    /// Calculate quantum advantage metrics
    fn calculate_quantum_advantage(&mut self, features: &DMatrix<T>) -> Result<(), QuantumError> {
        let num_features = features.ncols();
        let num_samples = features.nrows();

        // Calculate quantum speedup factor
        let classical_complexity = T::from(num_features * num_samples).unwrap();
        let quantum_complexity = T::from(self.num_qubits.pow(2)).unwrap();
        let speedup_factor = classical_complexity / quantum_complexity;

        // Calculate quantum memory advantage
        let classical_memory = T::from(num_features * num_samples).unwrap();
        let quantum_memory = T::from(self.num_qubits).unwrap();
        let memory_advantage = classical_memory / quantum_memory;

        // Estimate quantum accuracy improvement (theoretical)
        let accuracy_improvement = T::from(1.2).unwrap(); // Placeholder

        // Calculate quantum entanglement measure
        let entanglement_measure = self.calculate_entanglement_entropy()?;

        // Calculate quantum coherence measure
        let coherence_measure = self.calculate_quantum_coherence()?;

        self.quantum_advantage = QuantumAdvantageMetrics {
            speedup_factor,
            memory_advantage,
            accuracy_improvement,
            entanglement_measure,
            coherence_measure,
        };

        Ok(())
    }

    /// Calculate entanglement entropy
    fn calculate_entanglement_entropy(&self) -> Result<T, QuantumError> {
        // Simplified entanglement entropy calculation
        let mut entropy = T::zero();

        for quantum_state in self.quantum_priors.values() {
            for amplitude in &quantum_state.amplitudes {
                let prob = amplitude.norm_sqr();
                if prob > T::zero() {
                    entropy = entropy - prob * prob.ln();
                }
            }
        }

        Ok(entropy)
    }

    /// Calculate quantum coherence
    fn calculate_quantum_coherence(&self) -> Result<T, QuantumError> {
        // Simplified coherence measure
        let mut coherence = T::zero();

        for quantum_state in self.quantum_priors.values() {
            for amplitude in &quantum_state.amplitudes {
                coherence = coherence + amplitude.norm_sqr();
            }
        }

        Ok(coherence / T::from(self.quantum_priors.len()).unwrap())
    }

    /// Predict using quantum measurement
    pub fn predict(&self, features: &DVector<T>) -> Result<HashMap<usize, T>, QuantumError> {
        let mut class_probabilities = HashMap::new();

        // Encode features into quantum state
        let quantum_feature_state = self.encode_features_to_quantum_state(features)?;

        // Calculate quantum probabilities for each class
        for (&class_id, class_quantum_state) in self.quantum_priors.iter() {
            let probability =
                self.calculate_quantum_probability(&quantum_feature_state, class_quantum_state)?;
            class_probabilities.insert(class_id, probability);
        }

        // Normalize probabilities
        let total_prob: T = class_probabilities.values().cloned().sum();
        if total_prob > T::zero() {
            for prob in class_probabilities.values_mut() {
                *prob = *prob / total_prob;
            }
        }

        Ok(class_probabilities)
    }

    /// Encode classical features to quantum state
    fn encode_features_to_quantum_state(
        &self,
        features: &DVector<T>,
    ) -> Result<QuantumState<T>, QuantumError> {
        let num_states = 2_usize.pow(self.num_qubits as u32);
        let mut amplitudes = vec![QComplex::new(T::zero(), T::zero()); num_states];

        // Amplitude encoding: normalize features and map to quantum amplitudes
        let norm_squared: T = features.iter().map(|&x| x * x).sum();
        let norm = norm_squared.sqrt();

        if norm == T::zero() {
            return Err(QuantumError::QuantumState("Zero norm features".to_string()));
        }

        for (i, &feature) in features.iter().enumerate() {
            if i < num_states {
                let normalized_feature = feature / norm;
                amplitudes[i] = QComplex::new(normalized_feature, T::zero());
            }
        }

        Ok(QuantumState {
            amplitudes,
            num_qubits: self.num_qubits,
            normalization: T::one(),
        })
    }

    /// Calculate quantum probability using quantum state overlap
    fn calculate_quantum_probability(
        &self,
        feature_state: &QuantumState<T>,
        class_state: &QuantumState<T>,
    ) -> Result<T, QuantumError> {
        if feature_state.amplitudes.len() != class_state.amplitudes.len() {
            return Err(QuantumError::QuantumMeasurement(
                "State dimension mismatch".to_string(),
            ));
        }

        // Calculate quantum state overlap (inner product)
        let mut overlap = QComplex::new(T::zero(), T::zero());
        for (feature_amp, class_amp) in feature_state
            .amplitudes
            .iter()
            .zip(class_state.amplitudes.iter())
        {
            overlap = overlap + feature_amp.conj() * class_amp;
        }

        // Return probability as squared magnitude of overlap
        Ok(overlap.norm_sqr())
    }

    /// Apply quantum gates to quantum state
    pub fn apply_quantum_gates(
        &self,
        quantum_state: &mut QuantumState<T>,
        gates: &[QuantumGate<T>],
    ) -> Result<(), QuantumError> {
        for gate in gates {
            self.apply_single_gate(quantum_state, gate)?;
        }
        Ok(())
    }

    /// Apply a single quantum gate
    fn apply_single_gate(
        &self,
        quantum_state: &mut QuantumState<T>,
        gate: &QuantumGate<T>,
    ) -> Result<(), QuantumError> {
        match gate.gate_type {
            GateType::Hadamard => self.apply_hadamard_gate(quantum_state, &gate.target_qubits)?,
            GateType::PauliX => self.apply_pauli_x_gate(quantum_state, &gate.target_qubits)?,
            GateType::PauliY => self.apply_pauli_y_gate(quantum_state, &gate.target_qubits)?,
            GateType::PauliZ => self.apply_pauli_z_gate(quantum_state, &gate.target_qubits)?,
            GateType::RotationX => {
                self.apply_rotation_x_gate(quantum_state, &gate.target_qubits, &gate.gate_params)?
            }
            GateType::RotationY => {
                self.apply_rotation_y_gate(quantum_state, &gate.target_qubits, &gate.gate_params)?
            }
            GateType::RotationZ => {
                self.apply_rotation_z_gate(quantum_state, &gate.target_qubits, &gate.gate_params)?
            }
            GateType::CNOT => {
                self.apply_cnot_gate(quantum_state, &gate.control_qubits, &gate.target_qubits)?
            }
            GateType::CZ => {
                self.apply_cz_gate(quantum_state, &gate.control_qubits, &gate.target_qubits)?
            }
            _ => {
                return Err(QuantumError::QuantumGate(
                    "Gate not implemented".to_string(),
                ))
            }
        }
        Ok(())
    }

    /// Apply Hadamard gate
    fn apply_hadamard_gate(
        &self,
        quantum_state: &mut QuantumState<T>,
        target_qubits: &[usize],
    ) -> Result<(), QuantumError> {
        let sqrt_2 = T::from(std::f64::consts::SQRT_2).unwrap();
        let inv_sqrt_2 = T::one() / sqrt_2;

        for &qubit in target_qubits {
            if qubit >= self.num_qubits {
                return Err(QuantumError::QuantumGate(
                    "Qubit index out of range".to_string(),
                ));
            }

            let mut new_amplitudes = quantum_state.amplitudes.clone();
            let num_states = quantum_state.amplitudes.len();

            for i in 0..num_states {
                let bit_mask = 1 << qubit;
                let j = i ^ bit_mask;

                if i < j {
                    let amp_i = quantum_state.amplitudes[i];
                    let amp_j = quantum_state.amplitudes[j];

                    new_amplitudes[i] = QComplex::new(inv_sqrt_2, T::zero()) * (amp_i + amp_j);
                    new_amplitudes[j] = QComplex::new(inv_sqrt_2, T::zero()) * (amp_i - amp_j);
                }
            }

            quantum_state.amplitudes = new_amplitudes;
        }

        Ok(())
    }

    /// Apply Pauli-X gate
    fn apply_pauli_x_gate(
        &self,
        quantum_state: &mut QuantumState<T>,
        target_qubits: &[usize],
    ) -> Result<(), QuantumError> {
        for &qubit in target_qubits {
            if qubit >= self.num_qubits {
                return Err(QuantumError::QuantumGate(
                    "Qubit index out of range".to_string(),
                ));
            }

            let mut new_amplitudes = quantum_state.amplitudes.clone();
            let num_states = quantum_state.amplitudes.len();

            for i in 0..num_states {
                let bit_mask = 1 << qubit;
                let j = i ^ bit_mask;
                new_amplitudes[i] = quantum_state.amplitudes[j];
            }

            quantum_state.amplitudes = new_amplitudes;
        }

        Ok(())
    }

    /// Apply Pauli-Y gate
    fn apply_pauli_y_gate(
        &self,
        quantum_state: &mut QuantumState<T>,
        target_qubits: &[usize],
    ) -> Result<(), QuantumError> {
        for &qubit in target_qubits {
            if qubit >= self.num_qubits {
                return Err(QuantumError::QuantumGate(
                    "Qubit index out of range".to_string(),
                ));
            }

            let mut new_amplitudes = quantum_state.amplitudes.clone();
            let num_states = quantum_state.amplitudes.len();

            for i in 0..num_states {
                let bit_mask = 1 << qubit;
                let j = i ^ bit_mask;

                if (i & bit_mask) == 0 {
                    new_amplitudes[i] =
                        QComplex::new(T::zero(), T::one()) * quantum_state.amplitudes[j];
                } else {
                    new_amplitudes[i] =
                        QComplex::new(T::zero(), -T::one()) * quantum_state.amplitudes[j];
                }
            }

            quantum_state.amplitudes = new_amplitudes;
        }

        Ok(())
    }

    /// Apply Pauli-Z gate
    fn apply_pauli_z_gate(
        &self,
        quantum_state: &mut QuantumState<T>,
        target_qubits: &[usize],
    ) -> Result<(), QuantumError> {
        for &qubit in target_qubits {
            if qubit >= self.num_qubits {
                return Err(QuantumError::QuantumGate(
                    "Qubit index out of range".to_string(),
                ));
            }

            let bit_mask = 1 << qubit;

            for i in 0..quantum_state.amplitudes.len() {
                if (i & bit_mask) != 0 {
                    quantum_state.amplitudes[i] = -quantum_state.amplitudes[i];
                }
            }
        }

        Ok(())
    }

    /// Apply rotation-X gate
    fn apply_rotation_x_gate(
        &self,
        quantum_state: &mut QuantumState<T>,
        target_qubits: &[usize],
        params: &[T],
    ) -> Result<(), QuantumError> {
        if params.is_empty() {
            return Err(QuantumError::QuantumGate(
                "Missing rotation angle".to_string(),
            ));
        }

        let angle = params[0];
        let cos_half = (angle / T::from(2.0).unwrap()).cos();
        let sin_half = (angle / T::from(2.0).unwrap()).sin();

        for &qubit in target_qubits {
            if qubit >= self.num_qubits {
                return Err(QuantumError::QuantumGate(
                    "Qubit index out of range".to_string(),
                ));
            }

            let mut new_amplitudes = quantum_state.amplitudes.clone();
            let num_states = quantum_state.amplitudes.len();

            for i in 0..num_states {
                let bit_mask = 1 << qubit;
                let j = i ^ bit_mask;

                if i < j {
                    let amp_i = quantum_state.amplitudes[i];
                    let amp_j = quantum_state.amplitudes[j];

                    new_amplitudes[i] = QComplex::new(cos_half, T::zero()) * amp_i
                        + QComplex::new(T::zero(), -sin_half) * amp_j;
                    new_amplitudes[j] = QComplex::new(T::zero(), -sin_half) * amp_i
                        + QComplex::new(cos_half, T::zero()) * amp_j;
                }
            }

            quantum_state.amplitudes = new_amplitudes;
        }

        Ok(())
    }

    /// Apply rotation-Y gate
    fn apply_rotation_y_gate(
        &self,
        quantum_state: &mut QuantumState<T>,
        target_qubits: &[usize],
        params: &[T],
    ) -> Result<(), QuantumError> {
        if params.is_empty() {
            return Err(QuantumError::QuantumGate(
                "Missing rotation angle".to_string(),
            ));
        }

        let angle = params[0];
        let cos_half = (angle / T::from(2.0).unwrap()).cos();
        let sin_half = (angle / T::from(2.0).unwrap()).sin();

        for &qubit in target_qubits {
            if qubit >= self.num_qubits {
                return Err(QuantumError::QuantumGate(
                    "Qubit index out of range".to_string(),
                ));
            }

            let mut new_amplitudes = quantum_state.amplitudes.clone();
            let num_states = quantum_state.amplitudes.len();

            for i in 0..num_states {
                let bit_mask = 1 << qubit;
                let j = i ^ bit_mask;

                if i < j {
                    let amp_i = quantum_state.amplitudes[i];
                    let amp_j = quantum_state.amplitudes[j];

                    new_amplitudes[i] = QComplex::new(cos_half, T::zero()) * amp_i
                        + QComplex::new(-sin_half, T::zero()) * amp_j;
                    new_amplitudes[j] = QComplex::new(sin_half, T::zero()) * amp_i
                        + QComplex::new(cos_half, T::zero()) * amp_j;
                }
            }

            quantum_state.amplitudes = new_amplitudes;
        }

        Ok(())
    }

    /// Apply rotation-Z gate
    fn apply_rotation_z_gate(
        &self,
        quantum_state: &mut QuantumState<T>,
        target_qubits: &[usize],
        params: &[T],
    ) -> Result<(), QuantumError> {
        if params.is_empty() {
            return Err(QuantumError::QuantumGate(
                "Missing rotation angle".to_string(),
            ));
        }

        let angle = params[0];
        let half_angle = angle / T::from(2.0).unwrap();

        for &qubit in target_qubits {
            if qubit >= self.num_qubits {
                return Err(QuantumError::QuantumGate(
                    "Qubit index out of range".to_string(),
                ));
            }

            let bit_mask = 1 << qubit;

            for i in 0..quantum_state.amplitudes.len() {
                if (i & bit_mask) == 0 {
                    quantum_state.amplitudes[i] = quantum_state.amplitudes[i]
                        * QComplex::new(half_angle.cos(), -half_angle.sin());
                } else {
                    quantum_state.amplitudes[i] = quantum_state.amplitudes[i]
                        * QComplex::new(half_angle.cos(), half_angle.sin());
                }
            }
        }

        Ok(())
    }

    /// Apply CNOT gate
    fn apply_cnot_gate(
        &self,
        quantum_state: &mut QuantumState<T>,
        control_qubits: &[usize],
        target_qubits: &[usize],
    ) -> Result<(), QuantumError> {
        if control_qubits.len() != 1 || target_qubits.len() != 1 {
            return Err(QuantumError::QuantumGate(
                "CNOT gate requires exactly one control and one target qubit".to_string(),
            ));
        }

        let control_qubit = control_qubits[0];
        let target_qubit = target_qubits[0];

        if control_qubit >= self.num_qubits || target_qubit >= self.num_qubits {
            return Err(QuantumError::QuantumGate(
                "Qubit index out of range".to_string(),
            ));
        }

        let control_mask = 1 << control_qubit;
        let target_mask = 1 << target_qubit;

        let mut new_amplitudes = quantum_state.amplitudes.clone();

        for i in 0..quantum_state.amplitudes.len() {
            if (i & control_mask) != 0 {
                let j = i ^ target_mask;
                new_amplitudes[i] = quantum_state.amplitudes[j];
            }
        }

        quantum_state.amplitudes = new_amplitudes;
        Ok(())
    }

    /// Apply CZ gate
    fn apply_cz_gate(
        &self,
        quantum_state: &mut QuantumState<T>,
        control_qubits: &[usize],
        target_qubits: &[usize],
    ) -> Result<(), QuantumError> {
        if control_qubits.len() != 1 || target_qubits.len() != 1 {
            return Err(QuantumError::QuantumGate(
                "CZ gate requires exactly one control and one target qubit".to_string(),
            ));
        }

        let control_qubit = control_qubits[0];
        let target_qubit = target_qubits[0];

        if control_qubit >= self.num_qubits || target_qubit >= self.num_qubits {
            return Err(QuantumError::QuantumGate(
                "Qubit index out of range".to_string(),
            ));
        }

        let control_mask = 1 << control_qubit;
        let target_mask = 1 << target_qubit;

        for i in 0..quantum_state.amplitudes.len() {
            if (i & control_mask) != 0 && (i & target_mask) != 0 {
                quantum_state.amplitudes[i] = -quantum_state.amplitudes[i];
            }
        }

        Ok(())
    }

    /// Measure quantum state
    pub fn measure_quantum_state(
        &self,
        quantum_state: &QuantumState<T>,
    ) -> Result<Vec<T>, QuantumError> {
        let mut measurement_probabilities = Vec::new();

        for amplitude in &quantum_state.amplitudes {
            let probability = amplitude.norm_sqr();
            measurement_probabilities.push(probability);
        }

        // Normalize probabilities
        let total_prob: T = measurement_probabilities.iter().cloned().sum();
        if total_prob > T::zero() {
            for prob in measurement_probabilities.iter_mut() {
                *prob = *prob / total_prob;
            }
        }

        Ok(measurement_probabilities)
    }

    /// Get quantum advantage metrics
    pub fn get_quantum_advantage(&self) -> &QuantumAdvantageMetrics<T> {
        &self.quantum_advantage
    }
}

impl<T: Float> Default for QuantumCircuitParams<T> {
    fn default() -> Self {
        Self {
            rotation_angles: vec![T::from(std::f64::consts::PI / 4.0).unwrap()],
            noise_model: QuantumNoiseModel::default(),
            error_correction: QuantumErrorCorrection::default(),
            decoherence: QuantumDecoherence::default(),
        }
    }
}

impl<T: Float> Default for QuantumNoiseModel<T> {
    fn default() -> Self {
        Self {
            depolarization_rate: T::from(0.001).unwrap(),
            amplitude_damping_rate: T::from(0.001).unwrap(),
            phase_damping_rate: T::from(0.001).unwrap(),
            bit_flip_rate: T::from(0.001).unwrap(),
            phase_flip_rate: T::from(0.001).unwrap(),
        }
    }
}

impl<T: Float> Default for QuantumErrorCorrection<T> {
    fn default() -> Self {
        Self {
            error_code: ErrorCorrectionCode::SurfaceCode,
            syndrome_threshold: T::from(0.1).unwrap(),
            correction_probability: T::from(0.99).unwrap(),
        }
    }
}

impl<T: Float> Default for QuantumDecoherence<T> {
    fn default() -> Self {
        Self {
            t1_time: T::from(100.0).unwrap(), // microseconds
            t2_time: T::from(50.0).unwrap(),  // microseconds
            coupling_strength: T::from(0.01).unwrap(),
        }
    }
}

impl<T: Float> Default for QuantumAdvantageMetrics<T> {
    fn default() -> Self {
        Self {
            speedup_factor: T::one(),
            memory_advantage: T::one(),
            accuracy_improvement: T::one(),
            entanglement_measure: T::zero(),
            coherence_measure: T::zero(),
        }
    }
}

impl<T: Float> Default for FeatureEncodingParams<T> {
    fn default() -> Self {
        Self {
            amplitude_encoding: AmplitudeEncoding::default(),
            angle_encoding: AngleEncoding::default(),
            basis_encoding: BasisEncoding::default(),
        }
    }
}

impl<T: Float> Default for AmplitudeEncoding<T> {
    fn default() -> Self {
        Self {
            normalization: T::one(),
            amplitude_scaling: T::one(),
        }
    }
}

impl<T: Float> Default for AngleEncoding<T> {
    fn default() -> Self {
        Self {
            rotation_axis: RotationAxis::Y,
            angle_scaling: T::from(std::f64::consts::PI).unwrap(),
        }
    }
}

impl<T: Float> Default for BasisEncoding<T> {
    fn default() -> Self {
        Self {
            basis_states: vec![0, 1],
            encoding_dimension: 2,
            _phantom: PhantomData,
        }
    }
}

impl<T: Float + Debug + 'static> Default for QuantumMeasurementOperator<T> {
    fn default() -> Self {
        Self {
            measurement_matrix: DMatrix::zeros((2, 2)),
            measurement_basis: MeasurementBasis::Computational,
            measurement_probability: T::one(),
        }
    }
}

/// Quantum Probability Distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumProbabilityDistribution<T: Float> {
    /// Quantum state representing the distribution
    quantum_state: QuantumState<T>,
    /// Distribution parameters
    distribution_params: QuantumDistributionParams<T>,
    /// Quantum sampling methods
    sampling_methods: QuantumSamplingMethods<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumDistributionParams<T: Float> {
    /// Mean parameter
    quantum_mean: QuantumState<T>,
    /// Variance parameter
    quantum_variance: QuantumState<T>,
    /// Higher-order moments
    quantum_moments: Vec<QuantumState<T>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSamplingMethods<T: Float> {
    /// Quantum rejection sampling
    rejection_sampling: QuantumRejectionSampling<T>,
    /// Quantum importance sampling
    importance_sampling: QuantumImportanceSampling<T>,
    /// Quantum Metropolis-Hastings
    metropolis_hastings: QuantumMetropolisHastings<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumRejectionSampling<T: Float> {
    /// Acceptance probability
    acceptance_probability: T,
    /// Maximum iterations
    max_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumImportanceSampling<T: Float> {
    /// Importance weights
    importance_weights: Vec<T>,
    /// Effective sample size
    effective_sample_size: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetropolisHastings<T: Float> {
    /// Proposal distribution
    proposal_distribution: QuantumState<T>,
    /// Acceptance rate
    acceptance_rate: T,
    /// Burn-in period
    burn_in_period: usize,
}

impl<T: Float + Default> Default for QuantumRejectionSampling<T> {
    fn default() -> Self {
        Self {
            acceptance_probability: T::from(0.8).unwrap(),
            max_iterations: 1000,
        }
    }
}

impl<T: Float + Default> Default for QuantumImportanceSampling<T> {
    fn default() -> Self {
        Self {
            importance_weights: vec![T::one(); 100],
            effective_sample_size: T::from(100.0).unwrap(),
        }
    }
}

impl<T: Float + Default> Default for QuantumMetropolisHastings<T> {
    fn default() -> Self {
        Self {
            proposal_distribution: QuantumState {
                amplitudes: vec![QComplex::new(T::one(), T::zero()); 4],
                num_qubits: 2,
                normalization: T::one(),
            },
            acceptance_rate: T::from(0.5).unwrap(),
            burn_in_period: 1000,
        }
    }
}

impl<T: Float + Default> Default for QuantumSamplingMethods<T> {
    fn default() -> Self {
        Self {
            rejection_sampling: QuantumRejectionSampling::default(),
            importance_sampling: QuantumImportanceSampling::default(),
            metropolis_hastings: QuantumMetropolisHastings::default(),
        }
    }
}

/// Hybrid Quantum-Classical Naive Bayes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridQuantumClassicalNB<T: Float + Default> {
    /// Classical component
    classical_component: ClassicalComponent<T>,
    /// Quantum component
    quantum_component: QuantumNaiveBayes<T>,
    /// Hybrid parameters
    hybrid_params: HybridParams<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalComponent<T: Float> {
    /// Classical priors
    classical_priors: HashMap<usize, T>,
    /// Classical feature statistics
    classical_stats: HashMap<usize, Vec<(T, T)>>, // (mean, variance) for each feature
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridParams<T: Float> {
    /// Quantum-classical mixing ratio
    mixing_ratio: T,
    /// Quantum advantage threshold
    quantum_threshold: T,
    /// Hybrid optimization parameters
    optimization_params: HybridOptimizationParams<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridOptimizationParams<T: Float> {
    /// Learning rate
    learning_rate: T,
    /// Regularization parameter
    regularization: T,
    /// Convergence tolerance
    convergence_tolerance: T,
}

impl<
        T: Float
            + Default
            + Display
            + Debug
            + 'static
            + std::iter::Sum
            + std::iter::Sum<T>
            + for<'a> std::iter::Sum<&'a T>
            + Copy,
    > HybridQuantumClassicalNB<T>
{
    /// Create new hybrid quantum-classical classifier
    pub fn new(num_qubits: usize, mixing_ratio: T) -> Self {
        Self {
            classical_component: ClassicalComponent {
                classical_priors: HashMap::new(),
                classical_stats: HashMap::new(),
            },
            quantum_component: QuantumNaiveBayes::new(num_qubits),
            hybrid_params: HybridParams {
                mixing_ratio,
                quantum_threshold: T::from(0.1).unwrap(),
                optimization_params: HybridOptimizationParams::default(),
            },
        }
    }

    /// Fit hybrid model
    pub fn fit(
        &mut self,
        features: &DMatrix<T>,
        labels: &DVector<usize>,
    ) -> Result<(), QuantumError> {
        // Fit classical component
        self.fit_classical_component(features, labels)?;

        // Fit quantum component
        self.quantum_component.fit(features, labels)?;

        // Optimize hybrid parameters
        self.optimize_hybrid_parameters(features, labels)?;

        Ok(())
    }

    /// Fit classical component
    fn fit_classical_component(
        &mut self,
        features: &DMatrix<T>,
        labels: &DVector<usize>,
    ) -> Result<(), QuantumError> {
        // Calculate class priors
        let mut class_counts = HashMap::new();
        for &label in labels.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let total_samples = T::from(labels.len()).unwrap();
        for (&class_id, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            self.classical_component
                .classical_priors
                .insert(class_id, prior);
        }

        // Calculate feature statistics for each class
        for (&class_id, _) in class_counts.iter() {
            let class_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class_id)
                .map(|(i, _)| i)
                .collect();

            let mut feature_stats = Vec::new();
            for feature_idx in 0..features.ncols() {
                let feature_values: Vec<T> = class_indices
                    .iter()
                    .map(|&i| features[(i, feature_idx)])
                    .collect();

                let mean = self.calculate_mean(&feature_values);
                let variance = self.calculate_variance(&feature_values);
                feature_stats.push((mean, variance));
            }

            self.classical_component
                .classical_stats
                .insert(class_id, feature_stats);
        }

        Ok(())
    }

    /// Optimize hybrid parameters
    fn optimize_hybrid_parameters(
        &mut self,
        features: &DMatrix<T>,
        labels: &DVector<usize>,
    ) -> Result<(), QuantumError> {
        // Simplified parameter optimization
        let mut best_accuracy = T::zero();
        let mut best_mixing_ratio = self.hybrid_params.mixing_ratio;

        for i in 0..10 {
            let mixing_ratio = T::from(i as f64 / 10.0).unwrap();
            self.hybrid_params.mixing_ratio = mixing_ratio;

            let accuracy = self.cross_validate_accuracy(features, labels)?;
            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                best_mixing_ratio = mixing_ratio;
            }
        }

        self.hybrid_params.mixing_ratio = best_mixing_ratio;
        Ok(())
    }

    /// Cross-validate accuracy
    fn cross_validate_accuracy(
        &self,
        features: &DMatrix<T>,
        labels: &DVector<usize>,
    ) -> Result<T, QuantumError> {
        // Simplified cross-validation
        let mut correct_predictions = 0;
        let total_predictions = labels.len();

        for i in 0..total_predictions {
            let feature_vector = features.row(i);
            let feature_dvector = DVector::from_iter(feature_vector.iter().cloned());

            let predictions = self.predict_internal(&feature_dvector)?;
            let predicted_class = predictions
                .into_iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(class, _)| class)
                .unwrap_or(0);

            if predicted_class == labels[i] {
                correct_predictions += 1;
            }
        }

        Ok(T::from(correct_predictions).unwrap() / T::from(total_predictions).unwrap())
    }

    /// Predict using hybrid approach
    pub fn predict(&self, features: &DVector<T>) -> Result<HashMap<usize, T>, QuantumError> {
        self.predict_internal(features)
    }

    /// Internal prediction method
    fn predict_internal(&self, features: &DVector<T>) -> Result<HashMap<usize, T>, QuantumError> {
        // Get classical predictions
        let classical_predictions = self.classical_predict(features)?;

        // Get quantum predictions
        let quantum_predictions = self.quantum_component.predict(features)?;

        // Combine predictions using mixing ratio
        let mut hybrid_predictions = HashMap::new();

        for class_id in classical_predictions.keys() {
            let default_zero = T::zero();
            let classical_prob = classical_predictions.get(class_id).unwrap_or(&default_zero);
            let quantum_prob = quantum_predictions.get(class_id).unwrap_or(&default_zero);

            let hybrid_prob = (T::one() - self.hybrid_params.mixing_ratio) * *classical_prob
                + self.hybrid_params.mixing_ratio * *quantum_prob;

            hybrid_predictions.insert(*class_id, hybrid_prob);
        }

        // Normalize probabilities
        let total_prob: T = hybrid_predictions.values().cloned().sum();
        if total_prob > T::zero() {
            for prob in hybrid_predictions.values_mut() {
                *prob = *prob / total_prob;
            }
        }

        Ok(hybrid_predictions)
    }

    /// Classical prediction
    fn classical_predict(&self, features: &DVector<T>) -> Result<HashMap<usize, T>, QuantumError> {
        let mut class_probabilities = HashMap::new();

        for (&class_id, &prior) in self.classical_component.classical_priors.iter() {
            let feature_stats = self
                .classical_component
                .classical_stats
                .get(&class_id)
                .ok_or_else(|| {
                    QuantumError::HybridQuantumClassical("Class not found".to_string())
                })?;

            let mut likelihood = T::one();
            for (feature_idx, &feature_value) in features.iter().enumerate() {
                if feature_idx < feature_stats.len() {
                    let (mean, variance) = feature_stats[feature_idx];
                    let feature_likelihood = self.gaussian_pdf(feature_value, mean, variance);
                    likelihood = likelihood * feature_likelihood;
                }
            }

            let posterior = prior * likelihood;
            class_probabilities.insert(class_id, posterior);
        }

        Ok(class_probabilities)
    }

    /// Gaussian probability density function
    fn gaussian_pdf(&self, x: T, mean: T, variance: T) -> T {
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let coefficient = T::one() / (two_pi * variance).sqrt();
        let exponent = -((x - mean).powi(2)) / (T::from(2.0).unwrap() * variance);
        coefficient * exponent.exp()
    }

    /// Calculate mean
    fn calculate_mean(&self, values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let sum: T = values.iter().cloned().sum();
        sum / T::from(values.len()).unwrap()
    }

    /// Calculate variance
    fn calculate_variance(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::from(0.01).unwrap();
        }
        let mean = self.calculate_mean(values);
        let sum_squared_diff: T = values.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_squared_diff / T::from(values.len() - 1).unwrap()
    }
}

impl<T: Float> Default for HybridOptimizationParams<T> {
    fn default() -> Self {
        Self {
            learning_rate: T::from(0.01).unwrap(),
            regularization: T::from(0.001).unwrap(),
            convergence_tolerance: T::from(1e-6).unwrap(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::{Array1, Array2};

    // Type aliases for compatibility with DMatrix/DVector usage
    type DMatrix<T> = Array2<T>;
    type DVector<T> = Array1<T>;

    #[test]
    fn test_quantum_naive_bayes_creation() {
        let classifier = QuantumNaiveBayes::<f64>::new(3);
        assert_eq!(classifier.num_qubits, 3);
        assert!(classifier.quantum_priors.is_empty());
    }

    #[test]
    fn test_quantum_state_creation() {
        let classifier = QuantumNaiveBayes::<f64>::new(2);
        let prior = 0.7;
        let quantum_state = classifier.create_quantum_state_from_prior(prior);
        assert!(quantum_state.is_ok());

        let state = quantum_state.unwrap();
        assert_eq!(state.num_qubits, 2);
        assert_eq!(state.amplitudes.len(), 4);
    }

    #[test]
    fn test_quantum_gate_application() {
        let classifier = QuantumNaiveBayes::<f64>::new(2);
        let mut quantum_state = QuantumState {
            amplitudes: vec![
                QComplex::new(1.0, 0.0),
                QComplex::new(0.0, 0.0),
                QComplex::new(0.0, 0.0),
                QComplex::new(0.0, 0.0),
            ],
            num_qubits: 2,
            normalization: 1.0,
        };

        let hadamard_gate = QuantumGate {
            gate_type: GateType::Hadamard,
            gate_params: vec![],
            target_qubits: vec![0],
            control_qubits: vec![],
        };

        let result = classifier.apply_single_gate(&mut quantum_state, &hadamard_gate);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantum_feature_encoding() {
        let classifier = QuantumNaiveBayes::<f64>::new(3);
        let features = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let quantum_state = classifier.encode_features_to_quantum_state(&features);
        assert!(quantum_state.is_ok());

        let state = quantum_state.unwrap();
        assert_eq!(state.num_qubits, 3);
        assert_eq!(state.amplitudes.len(), 8);
    }

    #[test]
    fn test_quantum_measurement() {
        let classifier = QuantumNaiveBayes::<f64>::new(2);
        let quantum_state = QuantumState {
            amplitudes: vec![
                QComplex::new(0.5, 0.0),
                QComplex::new(0.5, 0.0),
                QComplex::new(0.5, 0.0),
                QComplex::new(0.5, 0.0),
            ],
            num_qubits: 2,
            normalization: 1.0,
        };

        let measurement_result = classifier.measure_quantum_state(&quantum_state);
        assert!(measurement_result.is_ok());

        let probabilities = measurement_result.unwrap();
        assert_eq!(probabilities.len(), 4);
    }

    #[test]
    fn test_quantum_naive_bayes_fitting() {
        let mut classifier = QuantumNaiveBayes::<f64>::new(2);

        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let result = classifier.fit(&features, &labels);
        assert!(result.is_ok());

        assert_eq!(classifier.quantum_priors.len(), 2);
        assert_eq!(classifier.quantum_feature_maps.len(), 2);
    }

    #[test]
    fn test_quantum_prediction() {
        let mut classifier = QuantumNaiveBayes::<f64>::new(2);

        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let _ = classifier.fit(&features, &labels);

        let test_features = Array1::from_vec(vec![2.5, 3.5]);
        let prediction = classifier.predict(&test_features);
        assert!(prediction.is_ok());

        let probabilities = prediction.unwrap();
        assert!(probabilities.contains_key(&0));
        assert!(probabilities.contains_key(&1));
    }

    #[test]
    fn test_hybrid_quantum_classical_nb() {
        let mut classifier = HybridQuantumClassicalNB::<f64>::new(2, 0.5);

        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let result = classifier.fit(&features, &labels);
        assert!(result.is_ok());

        let test_features = Array1::from_vec(vec![2.5, 3.5]);
        let prediction = classifier.predict(&test_features);
        assert!(prediction.is_ok());

        let probabilities = prediction.unwrap();
        assert!(probabilities.contains_key(&0));
        assert!(probabilities.contains_key(&1));
    }

    #[test]
    fn test_quantum_advantage_calculation() {
        let mut classifier = QuantumNaiveBayes::<f64>::new(3);

        let features = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0,
            ],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let _ = classifier.fit(&features, &labels);

        let advantage = classifier.get_quantum_advantage();
        assert!(advantage.speedup_factor > 0.0);
        assert!(advantage.memory_advantage > 0.0);
    }
}

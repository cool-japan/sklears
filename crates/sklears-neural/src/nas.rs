//! Neural Architecture Search (NAS) for automated neural network design.
//!
//! This module implements various neural architecture search methods including:
//! - Differentiable Architecture Search (DARTS)
//! - Efficient Neural Architecture Search (ENAS)
//! - Network Architecture Search space definition
//! - Cell-based search strategies
//! - Progressive architecture search
//!
//! NAS automatically discovers optimal neural network architectures by
//! exploring the space of possible architectures using optimization techniques.

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis, ScalarOperand};
use scirs2_core::random::{thread_rng, CoreRandom, Normal, Rng};
use sklears_core::{error::SklearsError, types::FloatBounds};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Safe conversion from f64 to generic float type
fn safe_from_f64<T: FloatBounds>(value: f64) -> NeuralResult<T> {
    T::from(value).ok_or_else(|| {
        SklearsError::NumericalError(format!("Failed to convert {} to target float type", value))
    })
}

/// Safe comparison for float types
fn safe_partial_cmp<T: FloatBounds>(a: T, b: T) -> NeuralResult<std::cmp::Ordering> {
    a.to_f64()
        .and_then(|a_f64| b.to_f64().map(|b_f64| (a_f64, b_f64)))
        .and_then(|(a_f64, b_f64)| a_f64.partial_cmp(&b_f64))
        .ok_or_else(|| SklearsError::NumericalError("Float comparison failed".to_string()))
}

/// Find maximum element with safe comparison
fn safe_max<'a, T: FloatBounds>(iter: impl Iterator<Item = &'a T>) -> NeuralResult<&'a T>
where
    T: 'a,
{
    iter.max_by(|a, b| {
        a.to_f64()
            .and_then(|a_f64| b.to_f64().map(|b_f64| a_f64.partial_cmp(&b_f64)))
            .flatten()
            .unwrap_or(std::cmp::Ordering::Equal)
    })
    .ok_or_else(|| SklearsError::InvalidInput("Empty iterator".to_string()))
}

/// Find maximum element with index and safe comparison
fn safe_max_by<T, F>(iter: impl Iterator<Item = T>, mut compare: F) -> NeuralResult<T>
where
    F: FnMut(&T, &T) -> NeuralResult<std::cmp::Ordering>,
{
    let mut max_item: Option<T> = None;

    for item in iter {
        max_item = Some(match max_item {
            None => item,
            Some(current_max) => {
                match compare(&current_max, &item) {
                    Ok(std::cmp::Ordering::Less) => item,
                    Ok(_) => current_max,
                    Err(_) => current_max, // On error, keep current max
                }
            }
        });
    }

    max_item.ok_or_else(|| SklearsError::InvalidInput("Empty iterator".to_string()))
}

/// Create normal distribution safely
fn safe_normal(mean: f64, std_dev: f64) -> NeuralResult<Normal<f64>> {
    Normal::new(mean, std_dev).map_err(|e| SklearsError::InvalidParameter {
        name: "distribution".to_string(),
        reason: format!("Failed to create Normal distribution: {}", e),
    })
}

/// Types of operations available in the search space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OperationType {
    /// No operation (skip connection)
    None,
    /// Max pooling 3x3
    MaxPool3x3,
    /// Average pooling 3x3
    AvgPool3x3,
    /// Skip connection (identity)
    Skip,
    /// Separable convolution 3x3
    SepConv3x3,
    /// Separable convolution 5x5
    SepConv5x5,
    /// Dilated convolution 3x3
    DilConv3x3,
    /// Dilated convolution 5x5
    DilConv5x5,
}

impl OperationType {
    /// Get all available operations
    pub fn all_operations() -> Vec<OperationType> {
        vec![
            OperationType::None,
            OperationType::MaxPool3x3,
            OperationType::AvgPool3x3,
            OperationType::Skip,
            OperationType::SepConv3x3,
            OperationType::SepConv5x5,
            OperationType::DilConv3x3,
            OperationType::DilConv5x5,
        ]
    }
}

/// Mixed operation: weighted sum of operations (for DARTS)
#[derive(Debug)]
pub struct MixedOperation<T: FloatBounds> {
    /// Architecture parameters (weights for each operation)
    alpha: Array1<T>,
    /// List of operation types
    operations: Vec<OperationType>,
    /// Operation weights (for neural network operations)
    op_weights: Vec<Option<Array2<T>>>,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum> MixedOperation<T> {
    /// Create a new mixed operation
    pub fn new(
        in_features: usize,
        out_features: usize,
        operations: Vec<OperationType>,
    ) -> NeuralResult<Self> {
        let n_ops = operations.len();
        let mut rng = thread_rng();
        let normal_dist = safe_normal(0.0, 1.0)?;

        // Initialize architecture parameters uniformly
        let init_value = safe_from_f64(1.0 / n_ops as f64)?;
        let alpha = Array1::from_elem(n_ops, init_value);

        // Initialize operation weights
        let mut op_weights = Vec::new();
        for _ in 0..n_ops {
            let std = (2.0 / in_features as f64).sqrt();
            let w = Array2::from_shape_fn((in_features, out_features), |_| {
                let sample = rng.sample::<f64, _>(normal_dist);
                safe_from_f64(sample * std).unwrap_or(T::zero())
            });
            op_weights.push(Some(w));
        }

        Ok(Self {
            alpha,
            operations,
            op_weights,
            in_features,
            out_features,
        })
    }

    /// Forward pass with weighted operations
    pub fn forward(&self, x: &Array2<T>) -> NeuralResult<Array2<T>> {
        // Apply softmax to architecture parameters
        let alpha_max = safe_max(self.alpha.iter())?;

        let exp_sum: T = self.alpha.iter().map(|&a| (a - *alpha_max).exp()).sum();

        let alpha_softmax: Array1<T> = self.alpha.mapv(|a| (a - *alpha_max).exp() / exp_sum);

        // Weighted sum of operations
        let mut output = Array2::zeros((x.nrows(), self.out_features));

        for (i, (&op_type, weight_opt)) in self
            .operations
            .iter()
            .zip(self.op_weights.iter())
            .enumerate()
        {
            let op_weight = alpha_softmax[i];

            let op_output = match op_type {
                OperationType::None => Array2::zeros((x.nrows(), self.out_features)),
                OperationType::Skip => {
                    if x.ncols() == self.out_features {
                        x.clone()
                    } else {
                        // Project to output dimension
                        if let Some(ref w) = weight_opt {
                            x.dot(w)
                        } else {
                            Array2::zeros((x.nrows(), self.out_features))
                        }
                    }
                }
                _ => {
                    // Apply learned transformation
                    if let Some(ref w) = weight_opt {
                        let transformed = x.dot(w);
                        // Apply ReLU
                        transformed.mapv(|val| if val > T::zero() { val } else { T::zero() })
                    } else {
                        Array2::zeros((x.nrows(), self.out_features))
                    }
                }
            };

            // Add weighted operation to output
            output = output + op_output.mapv(|val| val * op_weight);
        }

        Ok(output)
    }

    /// Get the most likely operation (discrete architecture)
    pub fn argmax_operation(&self) -> OperationType {
        let max_idx = self
            .alpha
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.to_f64()
                    .and_then(|a_f64| b.to_f64().map(|b_f64| a_f64.partial_cmp(&b_f64)))
                    .flatten()
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        self.operations[max_idx]
    }

    /// Get architecture parameters
    pub fn get_alpha(&self) -> &Array1<T> {
        &self.alpha
    }

    /// Update architecture parameters
    pub fn update_alpha(&mut self, gradient: &Array1<T>, learning_rate: T) {
        self.alpha = &self.alpha - &gradient.mapv(|g| g * learning_rate);
    }
}

/// DARTS Cell: basic building block for architecture search
#[derive(Debug)]
pub struct DARTSCell<T: FloatBounds> {
    /// Mixed operations connecting nodes
    mixed_ops: Vec<Vec<MixedOperation<T>>>,
    /// Number of intermediate nodes
    n_nodes: usize,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum> DARTSCell<T> {
    /// Create a new DARTS cell
    pub fn new(in_features: usize, out_features: usize, n_nodes: usize) -> NeuralResult<Self> {
        let operations = OperationType::all_operations();
        let mut mixed_ops = Vec::new();

        // Create mixed operations for each node
        for i in 0..n_nodes {
            let mut node_ops = Vec::new();

            // Each node can receive input from previous nodes
            let n_inputs = i + 2; // Including two input nodes

            for j in 0..n_inputs {
                // First node uses in_features, subsequent nodes use out_features
                let input_dim = if i == 0 || j < 2 {
                    in_features
                } else {
                    out_features
                };
                let op = MixedOperation::new(input_dim, out_features, operations.clone())?;
                node_ops.push(op);
            }

            mixed_ops.push(node_ops);
        }

        Ok(Self {
            mixed_ops,
            n_nodes,
            in_features,
            out_features,
        })
    }

    /// Forward pass through cell
    pub fn forward(&self, x: &Array2<T>) -> NeuralResult<Array2<T>> {
        // Store intermediate node outputs
        let mut node_outputs = vec![x.clone(), x.clone()]; // Two initial input nodes

        // Process each intermediate node
        for node_idx in 0..self.n_nodes {
            let mut node_output = Array2::zeros((x.nrows(), self.out_features));

            // Aggregate inputs from all previous nodes
            for (input_idx, mixed_op) in self.mixed_ops[node_idx].iter().enumerate() {
                let input = &node_outputs[input_idx];
                let op_output = mixed_op.forward(input)?;
                node_output = node_output + op_output;
            }

            // Apply ReLU
            node_output.mapv_inplace(|val| if val > T::zero() { val } else { T::zero() });

            node_outputs.push(node_output);
        }

        // Concatenate all intermediate node outputs
        // For simplicity, we average them here
        let n_intermediate = self.n_nodes;
        let mut output = Array2::zeros((x.nrows(), self.out_features));

        for node_output in node_outputs.iter().skip(2).take(n_intermediate) {
            output = output + node_output;
        }

        let divisor = safe_from_f64(n_intermediate as f64)?;
        output.mapv_inplace(|val| val / divisor);

        Ok(output)
    }

    /// Get discrete architecture after search
    pub fn get_discrete_architecture(&self) -> Vec<Vec<OperationType>> {
        self.mixed_ops
            .iter()
            .map(|node_ops| node_ops.iter().map(|op| op.argmax_operation()).collect())
            .collect()
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.mixed_ops
            .iter()
            .flatten()
            .map(|op| op.alpha.len())
            .sum()
    }
}

/// DARTS searcher for neural architecture search
pub struct DARTS<T: FloatBounds> {
    /// Normal cell for processing
    normal_cell: DARTSCell<T>,
    /// Reduction cell for downsampling
    reduction_cell: DARTSCell<T>,
    /// Number of cells to stack
    n_cells: usize,
    /// Learning rate for architecture parameters
    arch_learning_rate: T,
    /// Learning rate for network weights
    weight_learning_rate: T,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum> DARTS<T> {
    /// Create a new DARTS searcher
    pub fn new(
        in_features: usize,
        out_features: usize,
        n_nodes: usize,
        n_cells: usize,
        arch_learning_rate: T,
        weight_learning_rate: T,
    ) -> NeuralResult<Self> {
        // After first transformation, all cells work with out_features
        // So create cells that work with out_features for both dimensions
        let normal_cell = DARTSCell::new(out_features, out_features, n_nodes)?;
        let reduction_cell = DARTSCell::new(out_features, out_features, n_nodes)?;

        Ok(Self {
            normal_cell,
            reduction_cell,
            n_cells,
            arch_learning_rate,
            weight_learning_rate,
        })
    }

    /// Search step: forward pass through architecture
    pub fn forward(&self, x: &Array2<T>) -> NeuralResult<Array2<T>> {
        // First, we need to project input to out_features dimension if needed
        let mut h = if x.ncols() != self.normal_cell.out_features {
            // Simple projection using mean pooling/expansion
            let batch_size = x.nrows();
            let out_dim = self.normal_cell.out_features;
            let in_dim = x.ncols();

            if in_dim > out_dim {
                // Downsample
                Array2::from_shape_fn((batch_size, out_dim), |(i, j)| {
                    let start_idx = (j * in_dim) / out_dim;
                    let end_idx = ((j + 1) * in_dim) / out_dim;
                    let mut sum = T::zero();
                    for k in start_idx..end_idx {
                        sum = sum + x[[i, k]];
                    }
                    let divisor = safe_from_f64((end_idx - start_idx) as f64).unwrap_or(T::one());
                    sum / divisor
                })
            } else {
                // Upsample by repeating
                Array2::from_shape_fn((batch_size, out_dim), |(i, j)| x[[i, j % in_dim]])
            }
        } else {
            x.clone()
        };

        // Pass through stacked cells
        for i in 0..self.n_cells {
            h = if i % 3 == 2 {
                // Use reduction cell every 3rd cell
                self.reduction_cell.forward(&h)?
            } else {
                // Use normal cell
                self.normal_cell.forward(&h)?
            };
        }

        Ok(h)
    }

    /// Get final architecture after search
    pub fn get_architecture(&self) -> NeuralArchitecture {
        NeuralArchitecture {
            normal_cell: self.normal_cell.get_discrete_architecture(),
            reduction_cell: self.reduction_cell.get_discrete_architecture(),
            n_cells: self.n_cells,
        }
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.normal_cell.num_parameters() + self.reduction_cell.num_parameters()
    }
}

/// Discovered neural architecture
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuralArchitecture {
    /// Normal cell architecture
    pub normal_cell: Vec<Vec<OperationType>>,
    /// Reduction cell architecture
    pub reduction_cell: Vec<Vec<OperationType>>,
    /// Number of cells to stack
    pub n_cells: usize,
}

impl NeuralArchitecture {
    /// Create a new architecture
    pub fn new(
        normal_cell: Vec<Vec<OperationType>>,
        reduction_cell: Vec<Vec<OperationType>>,
        n_cells: usize,
    ) -> Self {
        Self {
            normal_cell,
            reduction_cell,
            n_cells,
        }
    }

    /// Get total number of operations
    pub fn num_operations(&self) -> usize {
        let normal_ops: usize = self.normal_cell.iter().map(|node| node.len()).sum();
        let reduction_ops: usize = self.reduction_cell.iter().map(|node| node.len()).sum();
        normal_ops + reduction_ops
    }

    /// Get architecture complexity score
    pub fn complexity_score(&self) -> f64 {
        let mut score = 0.0;

        // Compute complexity based on operation types
        for cell in &[&self.normal_cell, &self.reduction_cell] {
            for node_ops in *cell {
                for &op in node_ops {
                    score += match op {
                        OperationType::None => 0.0,
                        OperationType::Skip => 0.1,
                        OperationType::MaxPool3x3 | OperationType::AvgPool3x3 => 0.5,
                        OperationType::SepConv3x3 => 1.0,
                        OperationType::SepConv5x5 => 1.5,
                        OperationType::DilConv3x3 => 1.2,
                        OperationType::DilConv5x5 => 1.8,
                    };
                }
            }
        }

        score
    }
}

/// Progressive NAS: gradually grow architecture
#[derive(Debug)]
pub struct ProgressiveNAS<T: FloatBounds> {
    /// Current architecture
    architecture: Vec<OperationType>,
    /// Search space
    search_space: Vec<OperationType>,
    /// Maximum architecture length
    max_length: usize,
    /// Current position in progressive search
    current_position: usize,
    /// Weights for current operations
    weights: Vec<Array2<T>>,
    /// Input/output dimensions
    in_features: usize,
    out_features: usize,
}

impl<T: FloatBounds + ScalarOperand> ProgressiveNAS<T> {
    /// Create a new Progressive NAS searcher
    pub fn new(in_features: usize, out_features: usize, max_length: usize) -> Self {
        let search_space = OperationType::all_operations();

        Self {
            architecture: Vec::new(),
            search_space,
            max_length,
            current_position: 0,
            weights: Vec::new(),
            in_features,
            out_features,
        }
    }

    /// Add next operation to architecture
    pub fn add_operation(&mut self, operation: OperationType) -> NeuralResult<()> {
        if self.architecture.len() >= self.max_length {
            return Err(SklearsError::InvalidParameter {
                name: "architecture".to_string(),
                reason: "Architecture has reached maximum length".to_string(),
            });
        }

        self.architecture.push(operation);

        // Initialize weights for new operation
        let mut rng = thread_rng();
        let normal_dist = safe_normal(0.0, 1.0)?;
        let std = (2.0 / self.in_features as f64).sqrt();
        let w = Array2::from_shape_fn((self.in_features, self.out_features), |_| {
            let sample = rng.sample::<f64, _>(normal_dist);
            safe_from_f64(sample * std).unwrap_or(T::zero())
        });
        self.weights.push(w);

        self.current_position += 1;

        Ok(())
    }

    /// Forward pass through current architecture
    pub fn forward(&self, x: &Array2<T>) -> NeuralResult<Array2<T>> {
        let mut h = x.clone();

        for (op, weight) in self.architecture.iter().zip(self.weights.iter()) {
            h = match op {
                OperationType::None => Array2::zeros(h.dim()),
                OperationType::Skip => h,
                _ => {
                    let transformed = h.dot(weight);
                    transformed.mapv(|val| if val > T::zero() { val } else { T::zero() })
                }
            };
        }

        Ok(h)
    }

    /// Get current architecture
    pub fn get_architecture(&self) -> Vec<OperationType> {
        self.architecture.clone()
    }

    /// Get search space
    pub fn get_search_space(&self) -> &[OperationType] {
        &self.search_space
    }

    /// Get current progress
    pub fn get_progress(&self) -> (usize, usize) {
        (self.current_position, self.max_length)
    }
}

/// ENAS Controller: RNN-based controller for architecture search
#[derive(Debug)]
pub struct ENASController<T: FloatBounds> {
    /// Controller RNN hidden size
    hidden_size: usize,
    /// Number of layers to generate
    num_layers: usize,
    /// Number of operations in search space
    num_operations: usize,
    /// Controller weights for generating operations
    weights_hidden: Array2<T>,
    weights_output: Array2<T>,
    /// Controller hidden state
    hidden_state: Array1<T>,
    /// Learning rate for controller
    learning_rate: T,
    /// Entropy weight for exploration
    entropy_weight: T,
    /// Baseline for reward (moving average)
    baseline: T,
    /// Baseline decay rate
    baseline_decay: T,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum> ENASController<T> {
    /// Create a new ENAS controller
    pub fn new(
        hidden_size: usize,
        num_layers: usize,
        num_operations: usize,
        learning_rate: T,
    ) -> NeuralResult<Self> {
        let mut rng = thread_rng();
        let normal_dist = safe_normal(0.0, 1.0)?;

        // Initialize controller weights with Xavier initialization
        let std_hidden = (2.0 / hidden_size as f64).sqrt();
        let weights_hidden = Array2::from_shape_fn((hidden_size, hidden_size), |_| {
            let sample = rng.sample::<f64, _>(normal_dist);
            safe_from_f64(sample * std_hidden).unwrap_or(T::zero())
        });

        let std_output = (2.0 / (hidden_size + num_operations) as f64).sqrt();
        let weights_output = Array2::from_shape_fn((hidden_size, num_operations), |_| {
            let sample = rng.sample::<f64, _>(normal_dist);
            safe_from_f64(sample * std_output).unwrap_or(T::zero())
        });

        let hidden_state = Array1::zeros(hidden_size);

        Ok(Self {
            hidden_size,
            num_layers,
            num_operations,
            weights_hidden,
            weights_output,
            hidden_state,
            learning_rate,
            entropy_weight: safe_from_f64(0.01)?,
            baseline: T::zero(),
            baseline_decay: safe_from_f64(0.99)?,
        })
    }

    /// Sample an architecture from the controller
    pub fn sample_architecture(&mut self) -> NeuralResult<(Vec<OperationType>, Array2<T>)> {
        let mut architecture = Vec::new();
        let mut log_probs = Vec::new();
        let mut rng = thread_rng();

        // Reset hidden state
        self.hidden_state = Array1::zeros(self.hidden_size);

        for _ in 0..self.num_layers {
            // Update hidden state (simple RNN step)
            let new_hidden = self.hidden_state.dot(&self.weights_hidden);
            self.hidden_state = new_hidden.mapv(|x| x.tanh()); // tanh activation

            // Compute logits for operation selection
            let logits = self.hidden_state.dot(&self.weights_output);

            // Apply softmax to get probabilities
            let max_logit = safe_max(logits.iter())?;

            let exp_logits: Array1<T> = logits.mapv(|x| (x - *max_logit).exp());
            let sum_exp: T = exp_logits.iter().copied().sum();
            let probs = exp_logits.mapv(|x| x / sum_exp);

            // Sample operation based on probabilities
            let rand_val: f64 = rng.gen();
            let mut cumsum = 0.0;
            let mut selected_op_idx = 0;

            for (i, &p) in probs.iter().enumerate() {
                cumsum += p.to_f64().unwrap_or(0.0);
                if cumsum >= rand_val {
                    selected_op_idx = i;
                    break;
                }
            }

            // Store log probability
            let log_prob = probs[selected_op_idx].ln();
            log_probs.push(log_prob);

            // Convert index to operation type
            let operations = OperationType::all_operations();
            let selected_op = operations[selected_op_idx.min(operations.len() - 1)];
            architecture.push(selected_op);
        }

        // Convert log_probs to Array2 for compatibility
        let log_probs_array =
            Array2::from_shape_vec((1, log_probs.len()), log_probs).map_err(|e| {
                SklearsError::InvalidParameter {
                    name: "log_probs".to_string(),
                    reason: format!("Failed to create log_probs array: {}", e),
                }
            })?;

        Ok((architecture, log_probs_array))
    }

    /// Update controller using REINFORCE policy gradient
    pub fn update(&mut self, log_probs: &Array2<T>, reward: T) -> NeuralResult<()> {
        // Update baseline (exponential moving average)
        self.baseline =
            self.baseline * self.baseline_decay + reward * (T::one() - self.baseline_decay);

        // Compute advantage (reward - baseline)
        let advantage = reward - self.baseline;

        // Compute policy gradient
        // grad = -advantage * log_probs
        let policy_grad = log_probs.mapv(|lp| -advantage * lp);

        // Simple gradient descent update (simplified)
        // In full implementation, this would update the RNN weights
        // Here we just demonstrate the concept
        let grad_norm = policy_grad.iter().map(|&g| g * g).sum::<T>().sqrt();

        if grad_norm > T::zero() {
            // Normalize and apply learning rate
            let update_scale = self.learning_rate / grad_norm;

            // Update weights (simplified - in practice would backprop through RNN)
            let mut rng = thread_rng();
            let noise_std = update_scale.to_f64().unwrap_or(0.001) * 0.01;
            let noise_dist = safe_normal(0.0, 1.0)?;

            self.weights_output.mapv_inplace(|w| {
                let sample = rng.sample::<f64, _>(noise_dist);
                let noise = safe_from_f64(sample * noise_std).unwrap_or(T::zero());
                w + noise
            });
        }

        Ok(())
    }

    /// Get controller parameters count
    pub fn num_parameters(&self) -> usize {
        self.weights_hidden.len() + self.weights_output.len() + self.hidden_state.len()
    }
}

/// ENAS Searcher: Efficient Neural Architecture Search with parameter sharing
pub struct ENAS<T: FloatBounds> {
    /// Controller for generating architectures
    controller: ENASController<T>,
    /// Shared weights for child networks
    shared_weights: Vec<Array2<T>>,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
    /// Number of training steps
    num_steps: usize,
    /// Current step
    current_step: usize,
    /// Best architecture found
    best_architecture: Option<Vec<OperationType>>,
    /// Best reward achieved
    best_reward: Option<T>,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum> ENAS<T> {
    /// Create a new ENAS searcher
    pub fn new(
        in_features: usize,
        out_features: usize,
        hidden_size: usize,
        num_layers: usize,
        learning_rate: T,
        num_steps: usize,
    ) -> NeuralResult<Self> {
        let num_operations = OperationType::all_operations().len();
        let controller =
            ENASController::new(hidden_size, num_layers, num_operations, learning_rate)?;

        // Initialize shared weights for all operations
        let mut shared_weights = Vec::new();
        let mut rng = thread_rng();
        let normal_dist = safe_normal(0.0, 1.0)?;
        let std = (2.0 / in_features as f64).sqrt();

        for _ in 0..num_operations {
            let w = Array2::from_shape_fn((in_features, out_features), |_| {
                let sample = rng.sample::<f64, _>(normal_dist);
                safe_from_f64(sample * std).unwrap_or(T::zero())
            });
            shared_weights.push(w);
        }

        Ok(Self {
            controller,
            shared_weights,
            in_features,
            out_features,
            num_steps,
            current_step: 0,
            best_architecture: None,
            best_reward: None,
        })
    }

    /// Search step: sample architecture and train
    pub fn search_step(&mut self) -> NeuralResult<(Vec<OperationType>, T)> {
        // Sample architecture from controller
        let (architecture, log_probs) = self.controller.sample_architecture()?;

        // Evaluate architecture (simplified - in practice would train child network)
        let reward = self.evaluate_architecture(&architecture)?;

        // Update controller with reward
        self.controller.update(&log_probs, reward)?;

        // Update best architecture
        if self.best_reward.is_none() || reward > self.best_reward.unwrap() {
            self.best_architecture = Some(architecture.clone());
            self.best_reward = Some(reward);
        }

        self.current_step += 1;

        Ok((architecture, reward))
    }

    /// Evaluate an architecture (simplified)
    fn evaluate_architecture(&self, architecture: &[OperationType]) -> NeuralResult<T> {
        // Simplified evaluation: score based on operation complexity
        // In practice, this would involve training the child network
        let mut score = T::zero();
        let mut complexity_penalty = T::zero();

        for &op in architecture {
            score = score
                + match op {
                    OperationType::Skip => safe_from_f64(1.0).unwrap_or(T::one()),
                    OperationType::SepConv3x3 => safe_from_f64(0.9).unwrap_or(T::one()),
                    OperationType::SepConv5x5 => safe_from_f64(0.85).unwrap_or(T::one()),
                    OperationType::MaxPool3x3 | OperationType::AvgPool3x3 => {
                        safe_from_f64(0.8).unwrap_or(T::one())
                    }
                    OperationType::DilConv3x3 => safe_from_f64(0.88).unwrap_or(T::one()),
                    OperationType::DilConv5x5 => safe_from_f64(0.83).unwrap_or(T::one()),
                    OperationType::None => safe_from_f64(0.5).unwrap_or(T::zero()),
                };

            complexity_penalty = complexity_penalty
                + match op {
                    OperationType::None | OperationType::Skip => T::zero(),
                    OperationType::MaxPool3x3 | OperationType::AvgPool3x3 => {
                        safe_from_f64(0.01).unwrap_or(T::zero())
                    }
                    OperationType::SepConv3x3 => safe_from_f64(0.02).unwrap_or(T::zero()),
                    OperationType::DilConv3x3 => safe_from_f64(0.025).unwrap_or(T::zero()),
                    OperationType::SepConv5x5 => safe_from_f64(0.03).unwrap_or(T::zero()),
                    OperationType::DilConv5x5 => safe_from_f64(0.035).unwrap_or(T::zero()),
                };
        }

        // Balance performance and complexity
        let reward = score - complexity_penalty;

        Ok(reward)
    }

    /// Forward pass using a specific architecture
    pub fn forward(
        &self,
        x: &Array2<T>,
        architecture: &[OperationType],
    ) -> NeuralResult<Array2<T>> {
        let mut output = x.clone();
        let operations = OperationType::all_operations();

        for &op in architecture {
            // Find operation index
            let op_idx = operations.iter().position(|&o| o == op).unwrap_or(0);

            // Check if dimensions match, if not, skip this operation or adjust
            let current_dim = output.ncols();

            // Apply operation using shared weights
            output = match op {
                OperationType::None => Array2::zeros((output.nrows(), self.out_features)),
                OperationType::Skip => {
                    if current_dim == self.out_features {
                        // Keep output as-is
                        output
                    } else if current_dim == self.in_features {
                        // Project using shared weights
                        output.dot(&self.shared_weights[op_idx])
                    } else {
                        // For intermediate dimensions, use identity-like transformation
                        if current_dim > self.out_features {
                            // Downsample
                            Array2::from_shape_fn((output.nrows(), self.out_features), |(i, j)| {
                                let start = (j * current_dim) / self.out_features;
                                let end = ((j + 1) * current_dim) / self.out_features;
                                let mut sum = T::zero();
                                for k in start..end {
                                    sum = sum + output[[i, k]];
                                }
                                let divisor =
                                    safe_from_f64((end - start) as f64).unwrap_or(T::one());
                                sum / divisor
                            })
                        } else {
                            // Upsample by repeating
                            Array2::from_shape_fn((output.nrows(), self.out_features), |(i, j)| {
                                output[[i, j % current_dim]]
                            })
                        }
                    }
                }
                _ => {
                    // For other operations, only apply if dimensions match
                    if current_dim == self.in_features {
                        let transformed = output.dot(&self.shared_weights[op_idx]);
                        // ReLU activation
                        transformed.mapv(|val| if val > T::zero() { val } else { T::zero() })
                    } else if current_dim == self.out_features {
                        // Already at output dimension, apply identity-like transformation
                        output.mapv(|val| if val > T::zero() { val } else { T::zero() })
                    } else {
                        // Dimension mismatch - project to output dimension first
                        let projected = if current_dim > self.out_features {
                            Array2::from_shape_fn((output.nrows(), self.out_features), |(i, j)| {
                                let start = (j * current_dim) / self.out_features;
                                let end = ((j + 1) * current_dim) / self.out_features;
                                let mut sum = T::zero();
                                for k in start..end {
                                    sum = sum + output[[i, k]];
                                }
                                sum / T::from(end - start).unwrap()
                            })
                        } else {
                            Array2::from_shape_fn((output.nrows(), self.out_features), |(i, j)| {
                                output[[i, j % current_dim]]
                            })
                        };
                        projected.mapv(|val| if val > T::zero() { val } else { T::zero() })
                    }
                }
            };
        }

        Ok(output)
    }

    /// Get the best architecture found so far
    pub fn get_best_architecture(&self) -> Option<&Vec<OperationType>> {
        self.best_architecture.as_ref()
    }

    /// Get the best reward achieved
    pub fn get_best_reward(&self) -> Option<T> {
        self.best_reward
    }

    /// Get search progress
    pub fn get_progress(&self) -> (usize, usize) {
        (self.current_step, self.num_steps)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let controller_params = self.controller.num_parameters();
        let shared_params: usize = self.shared_weights.iter().map(|w| w.len()).sum();
        controller_params + shared_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_operation_types() {
        let ops = OperationType::all_operations();
        assert_eq!(ops.len(), 8);
        assert!(ops.contains(&OperationType::Skip));
        assert!(ops.contains(&OperationType::SepConv3x3));
    }

    #[test]
    fn test_mixed_operation_creation() {
        let ops = OperationType::all_operations();
        let mixed_op: MixedOperation<f64> = MixedOperation::new(10, 16, ops).unwrap();

        assert_eq!(mixed_op.in_features, 10);
        assert_eq!(mixed_op.out_features, 16);
        assert_eq!(mixed_op.alpha.len(), 8);
    }

    #[test]
    fn test_mixed_operation_forward() {
        let ops = vec![OperationType::Skip, OperationType::SepConv3x3];
        let mixed_op: MixedOperation<f64> = MixedOperation::new(8, 8, ops).unwrap();

        let x = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);
        let output = mixed_op.forward(&x).unwrap();

        assert_eq!(output.dim(), (4, 8));
    }

    #[test]
    fn test_mixed_operation_argmax() {
        let ops = OperationType::all_operations();
        let mut mixed_op: MixedOperation<f64> = MixedOperation::new(8, 8, ops).unwrap();

        // Set specific alpha values
        mixed_op.alpha[0] = 0.1;
        mixed_op.alpha[1] = 0.9;

        let best_op = mixed_op.argmax_operation();
        assert_eq!(best_op, OperationType::MaxPool3x3);
    }

    #[test]
    fn test_darts_cell_creation() {
        let cell: DARTSCell<f64> = DARTSCell::new(10, 16, 3).unwrap();

        assert_eq!(cell.n_nodes, 3);
        assert_eq!(cell.in_features, 10);
        assert_eq!(cell.out_features, 16);
        assert!(cell.num_parameters() > 0);
    }

    #[test]
    fn test_darts_cell_forward() {
        let cell: DARTSCell<f64> = DARTSCell::new(8, 12, 2).unwrap();
        let x = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);

        let output = cell.forward(&x).unwrap();
        assert_eq!(output.dim(), (4, 12));
    }

    #[test]
    fn test_darts_cell_discrete_architecture() {
        let cell: DARTSCell<f64> = DARTSCell::new(8, 8, 2).unwrap();
        let arch = cell.get_discrete_architecture();

        assert_eq!(arch.len(), 2); // 2 nodes
        assert_eq!(arch[0].len(), 2); // First node has 2 inputs
        assert_eq!(arch[1].len(), 3); // Second node has 3 inputs
    }

    #[test]
    fn test_darts_creation() {
        let darts: DARTS<f64> = DARTS::new(10, 16, 3, 6, 0.001, 0.01).unwrap();

        assert_eq!(darts.n_cells, 6);
        assert!(darts.num_parameters() > 0);
    }

    #[test]
    fn test_darts_forward() {
        let darts: DARTS<f64> = DARTS::new(8, 12, 2, 3, 0.001, 0.01).unwrap();
        let x = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);

        let output = darts.forward(&x).unwrap();
        assert_eq!(output.dim(), (4, 12));
    }

    #[test]
    fn test_darts_get_architecture() {
        let darts: DARTS<f64> = DARTS::new(8, 8, 2, 3, 0.001, 0.01).unwrap();
        let arch = darts.get_architecture();

        assert_eq!(arch.n_cells, 3);
        assert_eq!(arch.normal_cell.len(), 2);
        assert_eq!(arch.reduction_cell.len(), 2);
    }

    #[test]
    fn test_neural_architecture_creation() {
        let normal = vec![
            vec![OperationType::Skip, OperationType::SepConv3x3],
            vec![
                OperationType::SepConv3x3,
                OperationType::Skip,
                OperationType::MaxPool3x3,
            ],
        ];
        let reduction = vec![
            vec![OperationType::MaxPool3x3, OperationType::SepConv5x5],
            vec![
                OperationType::AvgPool3x3,
                OperationType::DilConv3x3,
                OperationType::Skip,
            ],
        ];

        let arch = NeuralArchitecture::new(normal, reduction, 5);

        assert_eq!(arch.n_cells, 5);
        assert_eq!(arch.num_operations(), 10);
    }

    #[test]
    fn test_architecture_complexity() {
        let normal = vec![vec![OperationType::Skip], vec![OperationType::SepConv3x3]];
        let reduction = vec![vec![OperationType::MaxPool3x3]];

        let arch = NeuralArchitecture::new(normal, reduction, 3);
        let complexity = arch.complexity_score();

        assert!(complexity > 0.0);
        assert!(complexity < 10.0); // Sanity check
    }

    #[test]
    fn test_progressive_nas_creation() {
        let pnas: ProgressiveNAS<f64> = ProgressiveNAS::new(10, 16, 5);

        assert_eq!(pnas.max_length, 5);
        assert_eq!(pnas.current_position, 0);
        assert_eq!(pnas.architecture.len(), 0);
    }

    #[test]
    fn test_progressive_nas_add_operation() {
        let mut pnas: ProgressiveNAS<f64> = ProgressiveNAS::new(8, 12, 3);

        pnas.add_operation(OperationType::Skip).unwrap();
        pnas.add_operation(OperationType::SepConv3x3).unwrap();

        assert_eq!(pnas.architecture.len(), 2);
        assert_eq!(pnas.current_position, 2);
    }

    #[test]
    fn test_progressive_nas_forward() {
        let mut pnas: ProgressiveNAS<f64> = ProgressiveNAS::new(8, 12, 3);

        pnas.add_operation(OperationType::SepConv3x3).unwrap();
        pnas.add_operation(OperationType::Skip).unwrap();

        let x = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);
        let output = pnas.forward(&x).unwrap();

        assert_eq!(output.dim(), (4, 12));
    }

    #[test]
    fn test_progressive_nas_max_length() {
        let mut pnas: ProgressiveNAS<f64> = ProgressiveNAS::new(8, 8, 2);

        pnas.add_operation(OperationType::Skip).unwrap();
        pnas.add_operation(OperationType::SepConv3x3).unwrap();

        // Should fail when exceeding max length
        let result = pnas.add_operation(OperationType::MaxPool3x3);
        assert!(result.is_err());
    }

    #[test]
    fn test_progressive_nas_progress() {
        let mut pnas: ProgressiveNAS<f64> = ProgressiveNAS::new(8, 8, 5);

        pnas.add_operation(OperationType::Skip).unwrap();
        pnas.add_operation(OperationType::SepConv3x3).unwrap();

        let (current, max) = pnas.get_progress();
        assert_eq!(current, 2);
        assert_eq!(max, 5);
    }

    #[test]
    fn test_enas_controller_creation() {
        let controller: ENASController<f64> = ENASController::new(16, 5, 8, 0.001).unwrap();

        assert_eq!(controller.hidden_size, 16);
        assert_eq!(controller.num_layers, 5);
        assert_eq!(controller.num_operations, 8);
        assert!(controller.num_parameters() > 0);
    }

    #[test]
    fn test_enas_controller_sample() {
        let mut controller: ENASController<f64> = ENASController::new(16, 3, 8, 0.001).unwrap();

        let (architecture, log_probs) = controller.sample_architecture().unwrap();

        assert_eq!(architecture.len(), 3);
        assert_eq!(log_probs.dim(), (1, 3));
    }

    #[test]
    fn test_enas_controller_update() {
        let mut controller: ENASController<f64> = ENASController::new(16, 3, 8, 0.001).unwrap();

        let (_, log_probs) = controller.sample_architecture().unwrap();
        let reward = 0.8;

        let result = controller.update(&log_probs, reward);
        assert!(result.is_ok());
    }

    #[test]
    fn test_enas_creation() {
        let enas: ENAS<f64> = ENAS::new(10, 16, 32, 5, 0.001, 100).unwrap();

        assert_eq!(enas.in_features, 10);
        assert_eq!(enas.out_features, 16);
        assert_eq!(enas.num_steps, 100);
        assert!(enas.num_parameters() > 0);
    }

    #[test]
    fn test_enas_search_step() {
        let mut enas: ENAS<f64> = ENAS::new(8, 12, 16, 3, 0.01, 50).unwrap();

        let (architecture, reward) = enas.search_step().unwrap();

        assert_eq!(architecture.len(), 3);
        assert!(reward > 0.0 || reward < 0.0 || reward == 0.0); // Just check it's a valid number
    }

    #[test]
    fn test_enas_forward() {
        let enas: ENAS<f64> = ENAS::new(8, 12, 16, 3, 0.01, 50).unwrap();
        let x = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);
        let architecture = vec![
            OperationType::Skip,
            OperationType::SepConv3x3,
            OperationType::MaxPool3x3,
        ];

        let output = enas.forward(&x, &architecture).unwrap();
        assert_eq!(output.dim(), (4, 12));
    }

    #[test]
    fn test_enas_best_architecture() {
        let mut enas: ENAS<f64> = ENAS::new(8, 8, 16, 3, 0.01, 10).unwrap();

        // Initially no best architecture
        assert!(enas.get_best_architecture().is_none());
        assert!(enas.get_best_reward().is_none());

        // After search step, should have a best architecture
        let _ = enas.search_step().unwrap();
        assert!(enas.get_best_architecture().is_some());
        assert!(enas.get_best_reward().is_some());
    }

    #[test]
    fn test_enas_progress() {
        let mut enas: ENAS<f64> = ENAS::new(8, 8, 16, 3, 0.01, 10).unwrap();

        let (current, total) = enas.get_progress();
        assert_eq!(current, 0);
        assert_eq!(total, 10);

        enas.search_step().unwrap();
        let (current, total) = enas.get_progress();
        assert_eq!(current, 1);
        assert_eq!(total, 10);
    }
}

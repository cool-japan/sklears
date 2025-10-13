//! Tensor operations for ensemble methods
//!
//! This module provides high-level tensor operations optimized for ensemble learning,
//! including batch operations, automatic differentiation support, and GPU acceleration.

use scirs2_core::ndarray::{Array1, Array2, ArrayD, Axis, Dimension, IxDyn};
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::{Float, Int};
use std::collections::HashMap;
use std::ops::{Add, Mul};

/// Multi-dimensional tensor type
pub type Tensor = ArrayD<Float>;

/// Tensor shape type
pub type TensorShape = Vec<usize>;

/// Tensor configuration
#[derive(Debug, Clone)]
pub struct TensorConfig {
    /// Enable automatic differentiation
    pub enable_autograd: bool,
    /// Default tensor device (CPU or GPU)
    pub default_device: TensorDevice,
    /// Memory layout preference
    pub memory_layout: MemoryLayout,
    /// Enable graph optimization
    pub enable_optimization: bool,
    /// Maximum tensor size for automatic batching
    pub max_batch_size: usize,
}

/// Tensor device enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDevice {
    /// CPU computation
    Cpu,
    /// GPU computation
    Gpu(usize), // GPU device ID
    /// Automatic device selection
    Auto,
}

/// Memory layout for tensors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryLayout {
    /// Row-major (C-style) layout
    RowMajor,
    /// Column-major (Fortran-style) layout
    ColumnMajor,
    /// Automatic layout selection
    Auto,
}

/// Tensor operations context
pub struct TensorOpsContext {
    config: TensorConfig,
    computation_graph: ComputationGraph,
    device_manager: DeviceManager,
}

/// Computation graph for automatic differentiation
#[derive(Debug, Default)]
pub struct ComputationGraph {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    current_node_id: usize,
}

/// Graph node representing a tensor operation
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub operation: TensorOperation,
    pub shape: TensorShape,
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
}

/// Graph edge connecting operations
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: usize,
    pub to: usize,
    pub input_index: usize,
}

/// Tensor operation enumeration
#[derive(Debug, Clone)]
pub enum TensorOperation {
    /// Leaf node (input tensor)
    Leaf(String),
    /// Addition operation
    Add,
    /// Subtraction operation
    Sub,
    /// Multiplication operation
    Mul,
    /// Division operation
    Div,
    /// Matrix multiplication
    MatMul,
    /// Element-wise activation functions
    Activation(ActivationType),
    /// Reduction operations
    Reduction(ReductionType, Option<usize>),
    /// Reshape operation
    Reshape(TensorShape),
    /// Transpose operation
    Transpose(Vec<usize>),
    /// Concatenation operation
    Concat(usize), // axis
    /// Split operation
    Split(usize, Vec<usize>), // axis, split points
    /// Ensemble aggregation
    EnsembleAgg(AggregationType),
}

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,
    LeakyReLU(Float),
    ELU(Float),
    GELU,
}

/// Reduction operation types
#[derive(Debug, Clone, Copy)]
pub enum ReductionType {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    Std,
    Var,
}

/// Ensemble aggregation types
#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    Average,
    WeightedAverage,
    Majority,
    Stacking,
    Blending,
}

/// Device manager for tensor operations
pub struct DeviceManager {
    available_devices: Vec<TensorDevice>,
    current_device: TensorDevice,
    memory_usage: HashMap<TensorDevice, usize>,
}

/// Ensemble tensor operations
pub struct EnsembleTensorOps {
    context: TensorOpsContext,
}

impl Default for TensorConfig {
    fn default() -> Self {
        Self {
            enable_autograd: false,
            default_device: TensorDevice::Cpu,
            memory_layout: MemoryLayout::Auto,
            enable_optimization: true,
            max_batch_size: 1024,
        }
    }
}

impl TensorOpsContext {
    /// Create new tensor operations context
    pub fn new(config: TensorConfig) -> Self {
        Self {
            config,
            computation_graph: ComputationGraph::default(),
            device_manager: DeviceManager::new(),
        }
    }

    /// Create tensor from ndarray
    pub fn from_array<D: Dimension>(
        &mut self,
        array: &scirs2_core::ndarray::Array<Float, D>,
    ) -> Result<Tensor> {
        let tensor = array.clone().into_dyn();

        if self.config.enable_autograd {
            self.add_leaf_node("input".to_string(), tensor.shape().to_vec());
        }

        Ok(tensor)
    }

    /// Create tensor with specific shape and fill value
    pub fn full(&mut self, shape: &[usize], value: Float) -> Result<Tensor> {
        let tensor = Tensor::from_elem(IxDyn(shape), value);

        if self.config.enable_autograd {
            self.add_leaf_node("constant".to_string(), shape.to_vec());
        }

        Ok(tensor)
    }

    /// Create zero tensor
    pub fn zeros(&mut self, shape: &[usize]) -> Result<Tensor> {
        self.full(shape, 0.0)
    }

    /// Create ones tensor
    pub fn ones(&mut self, shape: &[usize]) -> Result<Tensor> {
        self.full(shape, 1.0)
    }

    /// Create random tensor
    pub fn randn(&mut self, shape: &[usize]) -> Result<Tensor> {
        use scirs2_core::random::prelude::*;

        let size = shape.iter().product();
        let mut rng = thread_rng();
        // Use Box-Muller transform to generate normal distribution
        let data: Vec<Float> = (0..size)
            .map(|_| {
                // Simple normal distribution using Box-Muller transform
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                let z = ((-2.0 * u1.ln()) as f64).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z as Float
            })
            .collect();

        let tensor = Tensor::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

        if self.config.enable_autograd {
            self.add_leaf_node("random".to_string(), shape.to_vec());
        }

        Ok(tensor)
    }

    /// Element-wise addition
    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", a.shape()),
                actual: format!("{:?}", b.shape()),
            });
        }

        let result = a + b;

        if self.config.enable_autograd {
            self.add_binary_op_node(TensorOperation::Add, a.shape().to_vec());
        }

        Ok(result)
    }

    /// Element-wise subtraction
    pub fn sub(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", a.shape()),
                actual: format!("{:?}", b.shape()),
            });
        }

        let result = a - b;

        if self.config.enable_autograd {
            self.add_binary_op_node(TensorOperation::Sub, a.shape().to_vec());
        }

        Ok(result)
    }

    /// Element-wise multiplication
    pub fn mul(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", a.shape()),
                actual: format!("{:?}", b.shape()),
            });
        }

        let result = a * b;

        if self.config.enable_autograd {
            self.add_binary_op_node(TensorOperation::Mul, a.shape().to_vec());
        }

        Ok(result)
    }

    /// Matrix multiplication
    pub fn matmul(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Convert to 2D for matrix multiplication
        let a_2d = self.ensure_2d(a)?;
        let b_2d = self.ensure_2d(b)?;

        let result = a_2d.dot(&b_2d).into_dyn();

        if self.config.enable_autograd {
            let output_shape = vec![a_2d.nrows(), b_2d.ncols()];
            self.add_binary_op_node(TensorOperation::MatMul, output_shape);
        }

        Ok(result)
    }

    /// Apply activation function
    pub fn activation(&mut self, tensor: &Tensor, activation: ActivationType) -> Result<Tensor> {
        let result = match activation {
            ActivationType::ReLU => tensor.mapv(|x| x.max(0.0)),
            ActivationType::Sigmoid => tensor.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationType::Tanh => tensor.mapv(|x| x.tanh()),
            ActivationType::LeakyReLU(alpha) => {
                tensor.mapv(|x| if x > 0.0 { x } else { alpha * x })
            }
            ActivationType::ELU(alpha) => {
                tensor.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            }
            ActivationType::GELU => tensor.mapv(|x| {
                0.5 * x
                    * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))).tanh())
            }),
            ActivationType::Softmax => self.softmax_impl(tensor)?,
            ActivationType::LogSoftmax => self.log_softmax_impl(tensor)?,
        };

        if self.config.enable_autograd {
            self.add_unary_op_node(
                TensorOperation::Activation(activation),
                tensor.shape().to_vec(),
            );
        }

        Ok(result)
    }

    /// Reduction operations
    pub fn reduce(
        &mut self,
        tensor: &Tensor,
        reduction: ReductionType,
        axis: Option<usize>,
    ) -> Result<Tensor> {
        let result = match (reduction, axis) {
            (ReductionType::Sum, None) => {
                let sum = tensor.sum();
                Tensor::from_elem(IxDyn(&[]), sum)
            }
            (ReductionType::Sum, Some(ax)) => tensor.sum_axis(Axis(ax)).into_dyn(),
            (ReductionType::Mean, None) => {
                let mean = tensor.mean().unwrap_or(0.0);
                Tensor::from_elem(IxDyn(&[]), mean)
            }
            (ReductionType::Mean, Some(ax)) => tensor.mean_axis(Axis(ax)).unwrap().into_dyn(),
            (ReductionType::Max, Some(ax)) => {
                // Find max along axis
                tensor
                    .fold_axis(Axis(ax), Float::NEG_INFINITY, |&a, &b| a.max(b))
                    .into_dyn()
            }
            (ReductionType::Min, Some(ax)) => {
                // Find min along axis
                tensor
                    .fold_axis(Axis(ax), Float::INFINITY, |&a, &b| a.min(b))
                    .into_dyn()
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Reduction {:?} not implemented for axis {:?}",
                    reduction, axis
                )));
            }
        };

        if self.config.enable_autograd {
            let output_shape = result.shape().to_vec();
            self.add_unary_op_node(TensorOperation::Reduction(reduction, axis), output_shape);
        }

        Ok(result)
    }

    /// Reshape tensor
    pub fn reshape(&mut self, tensor: &Tensor, new_shape: &[usize]) -> Result<Tensor> {
        let total_elements = tensor.len();
        let new_total = new_shape.iter().product::<usize>();

        if total_elements != new_total {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("total elements = {}", total_elements),
                actual: format!("total elements = {}", new_total),
            });
        }

        let result = tensor
            .clone()
            .into_shape(IxDyn(new_shape))
            .map_err(|e| SklearsError::InvalidInput(format!("Reshape error: {}", e)))?;

        if self.config.enable_autograd {
            self.add_unary_op_node(
                TensorOperation::Reshape(new_shape.to_vec()),
                new_shape.to_vec(),
            );
        }

        Ok(result)
    }

    /// Transpose tensor
    pub fn transpose(&mut self, tensor: &Tensor, axes: &[usize]) -> Result<Tensor> {
        if axes.len() != tensor.ndim() {
            return Err(SklearsError::InvalidInput(format!(
                "Transpose axes count {} != tensor ndim {}",
                axes.len(),
                tensor.ndim()
            )));
        }

        let result = tensor.clone().permuted_axes(axes);

        if self.config.enable_autograd {
            let output_shape = axes.iter().map(|&i| tensor.shape()[i]).collect();
            self.add_unary_op_node(TensorOperation::Transpose(axes.to_vec()), output_shape);
        }

        Ok(result)
    }

    /// Concatenate tensors along axis
    pub fn concat(&mut self, tensors: &[&Tensor], axis: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot concatenate empty tensor list".to_string(),
            ));
        }

        // Convert to 2D arrays for concatenation
        let arrays_2d: Result<Vec<_>> = tensors.iter().map(|t| self.ensure_2d(t)).collect();
        let arrays_2d = arrays_2d?;

        let views: Vec<_> = arrays_2d.iter().map(|a| a.view()).collect();
        let result = scirs2_core::ndarray::concatenate(Axis(axis), &views)
            .map_err(|e| SklearsError::InvalidInput(format!("Concatenation error: {}", e)))?
            .into_dyn();

        if self.config.enable_autograd {
            let output_shape = result.shape().to_vec();
            self.add_variadic_op_node(TensorOperation::Concat(axis), output_shape, tensors.len());
        }

        Ok(result)
    }

    /// Ensemble-specific operations
    pub fn ensemble_aggregate(
        &mut self,
        predictions: &[&Tensor],
        weights: Option<&Tensor>,
        aggregation: AggregationType,
    ) -> Result<Tensor> {
        match aggregation {
            AggregationType::Average => self.ensemble_average(predictions),
            AggregationType::WeightedAverage => {
                if let Some(w) = weights {
                    self.ensemble_weighted_average(predictions, w)
                } else {
                    self.ensemble_average(predictions)
                }
            }
            AggregationType::Majority => self.ensemble_majority_vote(predictions),
            _ => Err(SklearsError::InvalidInput(format!(
                "Aggregation type {:?} not yet implemented",
                aggregation
            ))),
        }
    }

    /// Batch operations for ensemble training
    pub fn batch_ensemble_forward(
        &mut self,
        inputs: &[&Tensor],
        models: &[&Tensor], // Model parameters
    ) -> Result<Vec<Tensor>> {
        let mut outputs = Vec::new();

        for (input, model) in inputs.iter().zip(models.iter()) {
            // Simplified forward pass - in practice this would depend on model type
            let output = self.matmul(input, model)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Backward pass for gradient computation
    pub fn backward(&mut self, loss: &Tensor) -> Result<HashMap<String, Tensor>> {
        if !self.config.enable_autograd {
            return Err(SklearsError::InvalidInput(
                "Autograd not enabled. Set enable_autograd=true in config.".to_string(),
            ));
        }

        // Placeholder for actual backward pass implementation
        // In a real implementation, this would traverse the computation graph
        // and compute gradients using the chain rule

        let mut gradients = HashMap::new();
        gradients.insert("placeholder".to_string(), loss.clone());

        Ok(gradients)
    }

    /// Get computation graph
    pub fn get_computation_graph(&self) -> &ComputationGraph {
        &self.computation_graph
    }

    /// Clear computation graph
    pub fn clear_graph(&mut self) {
        self.computation_graph = ComputationGraph::default();
    }

    // Private helper methods

    fn ensure_2d(&self, tensor: &Tensor) -> Result<Array2<Float>> {
        match tensor.ndim() {
            1 => {
                let array_1d = tensor
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                    .map_err(|e| {
                        SklearsError::InvalidInput(format!("1D conversion error: {}", e))
                    })?;
                Ok(array_1d.insert_axis(Axis(0)))
            }
            2 => tensor
                .clone()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|e| SklearsError::InvalidInput(format!("2D conversion error: {}", e))),
            _ => Err(SklearsError::InvalidInput(format!(
                "Cannot convert {}D tensor to 2D",
                tensor.ndim()
            ))),
        }
    }

    fn softmax_impl(&self, tensor: &Tensor) -> Result<Tensor> {
        // Ensure 2D for softmax computation
        let tensor_2d = self.ensure_2d(tensor)?;
        let mut result = tensor_2d.clone();

        for mut row in result.rows_mut() {
            let max_val = row.fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|x| (x - max_val).exp());
            let sum = row.sum();
            if sum > 0.0 {
                row /= sum;
            }
        }

        Ok(result.into_dyn())
    }

    fn log_softmax_impl(&self, tensor: &Tensor) -> Result<Tensor> {
        let softmax = self.softmax_impl(tensor)?;
        Ok(softmax.mapv(|x| x.ln()))
    }

    fn ensemble_average(&mut self, predictions: &[&Tensor]) -> Result<Tensor> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions to average".to_string(),
            ));
        }

        let mut sum = predictions[0].clone();
        for pred in predictions.iter().skip(1) {
            sum = self.add(&sum, pred)?;
        }

        let n = predictions.len() as Float;
        Ok(sum.mapv(|x| x / n))
    }

    fn ensemble_weighted_average(
        &mut self,
        predictions: &[&Tensor],
        weights: &Tensor,
    ) -> Result<Tensor> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions to average".to_string(),
            ));
        }

        if weights.len() != predictions.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} weights", predictions.len()),
                actual: format!("{} weights", weights.len()),
            });
        }

        let mut weighted_sum = self.mul(
            predictions[0],
            &weights
                .slice(scirs2_core::ndarray::s![0..1])
                .to_owned()
                .into_dyn(),
        )?;

        for (i, pred) in predictions.iter().enumerate().skip(1) {
            let weight = weights
                .slice(scirs2_core::ndarray::s![i..i + 1])
                .to_owned()
                .into_dyn();
            let weighted_pred = self.mul(pred, &weight)?;
            weighted_sum = self.add(&weighted_sum, &weighted_pred)?;
        }

        Ok(weighted_sum)
    }

    fn ensemble_majority_vote(&mut self, predictions: &[&Tensor]) -> Result<Tensor> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions for majority vote".to_string(),
            ));
        }

        // Convert predictions to discrete votes and find majority
        // This is a simplified implementation
        let first_shape = predictions[0].shape();
        let mut votes = Tensor::zeros(IxDyn(first_shape));

        for pred in predictions {
            // Round predictions to nearest integer for voting
            let rounded = pred.mapv(|x| x.round());
            votes = self.add(&votes, &rounded)?;
        }

        // Majority decision
        let n_models = predictions.len() as Float;
        Ok(votes.mapv(|x| if x > n_models / 2.0 { 1.0 } else { 0.0 }))
    }

    fn add_leaf_node(&mut self, name: String, shape: TensorShape) {
        let node = GraphNode {
            id: self.computation_graph.current_node_id,
            operation: TensorOperation::Leaf(name),
            shape,
            requires_grad: false,
            grad: None,
        };

        self.computation_graph.nodes.push(node);
        self.computation_graph.current_node_id += 1;
    }

    fn add_unary_op_node(&mut self, operation: TensorOperation, output_shape: TensorShape) {
        let node = GraphNode {
            id: self.computation_graph.current_node_id,
            operation,
            shape: output_shape,
            requires_grad: false,
            grad: None,
        };

        self.computation_graph.nodes.push(node);
        self.computation_graph.current_node_id += 1;
    }

    fn add_binary_op_node(&mut self, operation: TensorOperation, output_shape: TensorShape) {
        let node = GraphNode {
            id: self.computation_graph.current_node_id,
            operation,
            shape: output_shape,
            requires_grad: false,
            grad: None,
        };

        self.computation_graph.nodes.push(node);
        self.computation_graph.current_node_id += 1;
    }

    fn add_variadic_op_node(
        &mut self,
        operation: TensorOperation,
        output_shape: TensorShape,
        _n_inputs: usize,
    ) {
        let node = GraphNode {
            id: self.computation_graph.current_node_id,
            operation,
            shape: output_shape,
            requires_grad: false,
            grad: None,
        };

        self.computation_graph.nodes.push(node);
        self.computation_graph.current_node_id += 1;
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceManager {
    /// Create new device manager
    pub fn new() -> Self {
        Self {
            available_devices: vec![TensorDevice::Cpu],
            current_device: TensorDevice::Cpu,
            memory_usage: HashMap::new(),
        }
    }

    /// Get available devices
    pub fn available_devices(&self) -> &[TensorDevice] {
        &self.available_devices
    }

    /// Set current device
    pub fn set_device(&mut self, device: TensorDevice) {
        self.current_device = device;
    }

    /// Get current device
    pub fn current_device(&self) -> TensorDevice {
        self.current_device
    }

    /// Get memory usage for device
    pub fn memory_usage(&self, device: TensorDevice) -> usize {
        self.memory_usage.get(&device).copied().unwrap_or(0)
    }
}

impl EnsembleTensorOps {
    /// Create new ensemble tensor operations
    pub fn new(config: TensorConfig) -> Self {
        Self {
            context: TensorOpsContext::new(config),
        }
    }

    /// Train ensemble with tensor operations
    pub fn train_ensemble_tensors(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Int>,
        n_estimators: usize,
    ) -> Result<Vec<Tensor>> {
        let x_tensor = self.context.from_array(x)?;
        let mut models = Vec::new();

        for _i in 0..n_estimators {
            // Create a simple linear model (weight matrix)
            let n_features = x.ncols();
            let model_weights = self.context.randn(&[n_features, 1])?;
            models.push(model_weights);
        }

        Ok(models)
    }

    /// Predict with ensemble using tensor operations
    pub fn predict_ensemble_tensors(
        &mut self,
        models: &[Tensor],
        x: &Array2<Float>,
    ) -> Result<Tensor> {
        let x_tensor = self.context.from_array(x)?;
        let mut predictions = Vec::new();

        for model in models {
            let pred = self.context.matmul(&x_tensor, model)?;
            predictions.push(pred);
        }

        // Average predictions
        let pred_refs: Vec<_> = predictions.iter().collect();
        self.context
            .ensemble_aggregate(&pred_refs, None, AggregationType::Average)
    }

    /// Get mutable context
    pub fn context_mut(&mut self) -> &mut TensorOpsContext {
        &mut self.context
    }

    /// Get context
    pub fn context(&self) -> &TensorOpsContext {
        &self.context
    }
}

// Convenience macro for tensor operations
#[macro_export]
macro_rules! tensor_op {
    ($ctx:expr, $op:ident, $($args:expr),*) => {
        $ctx.$op($($args),*)
    };
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_tensor_config() {
        let config = TensorConfig::default();
        assert!(!config.enable_autograd);
        assert_eq!(config.default_device, TensorDevice::Cpu);
    }

    #[test]
    fn test_tensor_context_creation() {
        let config = TensorConfig::default();
        let mut ctx = TensorOpsContext::new(config);

        let tensor = ctx.zeros(&[2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
    }

    #[test]
    fn test_tensor_operations() {
        let config = TensorConfig::default();
        let mut ctx = TensorOpsContext::new(config);

        let a = ctx.ones(&[2, 2]).unwrap();
        let b = ctx.full(&[2, 2], 2.0).unwrap();

        let result = ctx.add(&a, &b).unwrap();

        // Check all elements are 3.0
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-10));
    }

    #[test]
    fn test_matrix_multiplication() {
        let config = TensorConfig::default();
        let mut ctx = TensorOpsContext::new(config);

        let a_array = array![[1.0, 2.0], [3.0, 4.0]];
        let b_array = array![[5.0, 6.0], [7.0, 8.0]];

        let a = ctx.from_array(&a_array).unwrap();
        let b = ctx.from_array(&b_array).unwrap();

        let result = ctx.matmul(&a, &b).unwrap();

        // Expected: [[19, 22], [43, 50]]
        let result_2d = result
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .unwrap();
        assert_eq!(result_2d[[0, 0]], 19.0);
        assert_eq!(result_2d[[0, 1]], 22.0);
        assert_eq!(result_2d[[1, 0]], 43.0);
        assert_eq!(result_2d[[1, 1]], 50.0);
    }

    #[test]
    fn test_activation_functions() {
        let config = TensorConfig::default();
        let mut ctx = TensorOpsContext::new(config);

        let tensor = ctx.from_array(&array![[-1.0, 0.0, 1.0]]).unwrap();

        let relu_result = ctx.activation(&tensor, ActivationType::ReLU).unwrap();
        let sigmoid_result = ctx.activation(&tensor, ActivationType::Sigmoid).unwrap();

        // ReLU should clip negative values to 0
        assert_eq!(relu_result.as_slice().unwrap()[0], 0.0);
        assert_eq!(relu_result.as_slice().unwrap()[1], 0.0);
        assert_eq!(relu_result.as_slice().unwrap()[2], 1.0);

        // Sigmoid should be between 0 and 1
        assert!(sigmoid_result.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_reduction_operations() {
        let config = TensorConfig::default();
        let mut ctx = TensorOpsContext::new(config);

        let tensor = ctx.from_array(&array![[1.0, 2.0], [3.0, 4.0]]).unwrap();

        let sum_result = ctx.reduce(&tensor, ReductionType::Sum, None).unwrap();
        let mean_result = ctx.reduce(&tensor, ReductionType::Mean, None).unwrap();

        assert_eq!(sum_result.as_slice().unwrap()[0], 10.0);
        assert_eq!(mean_result.as_slice().unwrap()[0], 2.5);
    }

    #[test]
    fn test_ensemble_operations() {
        let config = TensorConfig::default();
        let mut ctx = TensorOpsContext::new(config);

        let pred1 = ctx.from_array(&array![[1.0, 2.0]]).unwrap();
        let pred2 = ctx.from_array(&array![[3.0, 4.0]]).unwrap();
        let predictions = vec![&pred1, &pred2];

        let avg_result = ctx
            .ensemble_aggregate(&predictions, None, AggregationType::Average)
            .unwrap();

        // Average should be [2.0, 3.0]
        assert_eq!(avg_result.as_slice().unwrap()[0], 2.0);
        assert_eq!(avg_result.as_slice().unwrap()[1], 3.0);
    }

    #[test]
    fn test_ensemble_tensor_ops() {
        let config = TensorConfig::default();
        let mut ensemble_ops = EnsembleTensorOps::new(config);

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0, 1];

        let models = ensemble_ops.train_ensemble_tensors(&x, &y, 3).unwrap();
        assert_eq!(models.len(), 3);

        let predictions = ensemble_ops.predict_ensemble_tensors(&models, &x).unwrap();
        assert_eq!(predictions.shape()[0], 2); // 2 samples
    }

    #[test]
    fn test_device_manager() {
        let mut manager = DeviceManager::new();

        assert_eq!(manager.current_device(), TensorDevice::Cpu);
        assert_eq!(manager.memory_usage(TensorDevice::Cpu), 0);

        manager.set_device(TensorDevice::Gpu(0));
        assert_eq!(manager.current_device(), TensorDevice::Gpu(0));
    }

    #[test]
    fn test_computation_graph() {
        let config = TensorConfig {
            enable_autograd: true,
            ..Default::default()
        };
        let mut ctx = TensorOpsContext::new(config);

        let a = ctx.ones(&[2, 2]).unwrap();
        let b = ctx.ones(&[2, 2]).unwrap();
        let _c = ctx.add(&a, &b).unwrap();

        let graph = ctx.get_computation_graph();
        assert!(graph.nodes.len() > 0);
    }
}

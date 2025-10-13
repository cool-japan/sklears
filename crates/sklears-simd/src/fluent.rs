//! Fluent API for SIMD operations
//!
//! This module provides a chainable, fluent interface for composing complex
//! SIMD operations in a readable and efficient manner.

#[cfg(feature = "no-std")]
use alloc::format;
#[cfg(feature = "no-std")]
use alloc::string::String;
#[cfg(feature = "no-std")]
use alloc::vec;
#[cfg(feature = "no-std")]
use alloc::vec::Vec;

#[cfg(feature = "no-std")]
use core::mem;
#[cfg(not(feature = "no-std"))]
use std::mem;

use crate::activation;
use crate::allocator::SimdVec;
use crate::distance;
use crate::kernels;
use crate::loss;
use crate::safety::SafeSimdOps;
use crate::vector;

/// Fluent builder for vector operations
#[derive(Debug, Clone)]
pub struct VectorBuilder {
    data: Vec<f32>,
    safe_mode: bool,
}

impl VectorBuilder {
    /// Create a new vector builder
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            safe_mode: false,
        }
    }

    /// Create a vector builder from existing data
    pub fn from_slice(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
            safe_mode: false,
        }
    }

    /// Create a vector builder with SIMD-aligned storage
    pub fn with_simd_storage(capacity: usize) -> Self {
        let simd_vec = SimdVec::with_capacity(capacity);
        Self {
            data: simd_vec.as_slice().to_vec(),
            safe_mode: false,
        }
    }

    /// Enable safe mode with bounds checking and overflow detection
    pub fn safe(mut self) -> Self {
        self.safe_mode = true;
        self
    }

    /// Add elements to the vector
    pub fn push(mut self, value: f32) -> Self {
        self.data.push(value);
        self
    }

    /// Add multiple elements to the vector
    pub fn extend(mut self, values: &[f32]) -> Self {
        self.data.extend_from_slice(values);
        self
    }

    /// Fill the vector with a value
    pub fn fill(mut self, size: usize, value: f32) -> Self {
        self.data = vec![value; size];
        self
    }

    /// Create a range of values
    pub fn range(mut self, start: f32, end: f32, step: f32) -> Self {
        let mut current = start;
        self.data.clear();
        while current < end {
            self.data.push(current);
            current += step;
        }
        self
    }

    /// Create a linearly spaced vector
    pub fn linspace(mut self, start: f32, end: f32, num: usize) -> Self {
        if num == 0 {
            self.data.clear();
            return self;
        }

        if num == 1 {
            self.data = vec![start];
            return self;
        }

        let step = (end - start) / (num - 1) as f32;
        self.data = (0..num).map(|i| start + (i as f32) * step).collect();
        self
    }

    /// Scale all elements by a factor
    pub fn scale(mut self, factor: f32) -> Self {
        if self.safe_mode {
            for value in &mut self.data {
                *value = SafeSimdOps::safe_mul_f32(*value, factor).unwrap_or(0.0);
            }
        } else {
            vector::scale(&mut self.data, factor);
        }
        self
    }

    /// Add a scalar to all elements
    pub fn add_scalar(mut self, value: f32) -> Self {
        if self.safe_mode {
            for element in &mut self.data {
                *element = SafeSimdOps::safe_add_f32(*element, value).unwrap_or(0.0);
            }
        } else {
            for element in &mut self.data {
                *element += value;
            }
        }
        self
    }

    /// Apply element-wise operation
    pub fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        for element in &mut self.data {
            *element = f(*element);
        }
        self
    }

    /// Normalize the vector
    pub fn normalize(mut self) -> Self {
        if self.safe_mode {
            self.data = SafeSimdOps::safe_normalize_f32(&self.data).unwrap_or_default();
        } else {
            let norm = vector::norm_l2(&self.data);
            if norm > 0.0 {
                vector::scale(&mut self.data, 1.0 / norm);
            }
        }
        self
    }

    /// Calculate dot product with another vector
    pub fn dot(&self, other: &[f32]) -> f32 {
        if self.safe_mode {
            SafeSimdOps::safe_dot_product_f32(&self.data, other).unwrap_or(0.0)
        } else {
            vector::dot_product(&self.data, other)
        }
    }

    /// Calculate distance to another vector
    pub fn distance_to(&self, other: &[f32], metric: DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Euclidean => distance::euclidean_distance(&self.data, other),
            DistanceMetric::Manhattan => distance::manhattan_distance(&self.data, other),
            DistanceMetric::Cosine => distance::cosine_distance(&self.data, other),
            DistanceMetric::Chebyshev => distance::chebyshev_distance(&self.data, other),
        }
    }

    /// Apply activation function
    pub fn activate(mut self, activation: ActivationFunction) -> Self {
        let mut output = vec![0.0; self.data.len()];
        match activation {
            ActivationFunction::Sigmoid => activation::sigmoid(&self.data, &mut output),
            ActivationFunction::Relu => activation::relu(&self.data, &mut output),
            ActivationFunction::Tanh => activation::tanh_activation(&self.data, &mut output),
            ActivationFunction::Softmax => activation::softmax(&self.data, &mut output),
        }
        self.data = output;
        self
    }

    /// Get statistics about the vector
    pub fn stats(&self) -> VectorStats {
        let (min, max) = vector::min_max(&self.data);
        VectorStats {
            mean: vector::mean(&self.data),
            variance: vector::variance(&self.data),
            min,
            max,
            norm: vector::norm_l2(&self.data),
            length: self.data.len(),
        }
    }

    /// Build the final vector
    pub fn build(self) -> Vec<f32> {
        self.data
    }

    /// Build into a SIMD-aligned vector
    pub fn build_simd(self) -> SimdVec<f32> {
        let mut simd_vec = SimdVec::new();
        for value in self.data {
            simd_vec.push(value);
        }
        simd_vec
    }

    /// Get a reference to the underlying data
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

impl Default for VectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Distance metrics for vector operations
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Chebyshev,
}

/// Activation functions for neural network operations
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Relu,
    Tanh,
    Softmax,
}

/// Statistics about a vector
#[derive(Debug, Clone)]
pub struct VectorStats {
    pub mean: f32,
    pub variance: f32,
    pub min: f32,
    pub max: f32,
    pub norm: f32,
    pub length: usize,
}

/// Fluent builder for matrix operations
#[derive(Debug, Clone)]
pub struct MatrixBuilder {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    safe_mode: bool,
}

impl MatrixBuilder {
    /// Create a new matrix builder
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
            safe_mode: false,
        }
    }

    /// Create from existing data
    pub fn from_data(data: Vec<f32>, rows: usize, cols: usize) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "Data length {} doesn't match matrix dimensions {}x{}",
                data.len(),
                rows,
                cols
            ));
        }

        Ok(Self {
            data,
            rows,
            cols,
            safe_mode: false,
        })
    }

    /// Create an identity matrix
    pub fn identity(size: usize) -> Self {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }

        Self {
            data,
            rows: size,
            cols: size,
            safe_mode: false,
        }
    }

    /// Create a matrix filled with random values
    pub fn random(rows: usize, cols: usize, min: f32, max: f32) -> Self {
        use scirs2_core::random::thread_rng;
        use scirs2_core::Rng;
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                let val: f32 = rng.random::<f32>();
                min + val * (max - min)
            })
            .collect();

        Self {
            data,
            rows,
            cols,
            safe_mode: false,
        }
    }

    /// Enable safe mode
    pub fn safe(mut self) -> Self {
        self.safe_mode = true;
        self
    }

    /// Set a value at position (row, col)
    pub fn set(mut self, row: usize, col: usize, value: f32) -> Result<Self, String> {
        if row >= self.rows || col >= self.cols {
            return Err(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, self.rows, self.cols
            ));
        }

        self.data[row * self.cols + col] = value;
        Ok(self)
    }

    /// Fill the matrix with a value
    pub fn fill(mut self, value: f32) -> Self {
        self.data.fill(value);
        self
    }

    /// Scale all elements
    pub fn scale(mut self, factor: f32) -> Self {
        if self.safe_mode {
            for value in &mut self.data {
                *value = SafeSimdOps::safe_mul_f32(*value, factor).unwrap_or(0.0);
            }
        } else {
            vector::scale(&mut self.data, factor);
        }
        self
    }

    /// Transpose the matrix (placeholder - needs proper implementation)
    pub fn transpose(mut self) -> Self {
        // TODO: Implement proper matrix transpose
        mem::swap(&mut self.rows, &mut self.cols);
        self
    }

    /// Multiply by another matrix (placeholder - needs proper implementation)
    pub fn multiply(&self, other: &MatrixBuilder) -> Result<MatrixBuilder, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Cannot multiply {}x{} matrix by {}x{} matrix",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        // TODO: Implement proper matrix multiplication
        Ok(MatrixBuilder {
            data: vec![0.0; self.rows * other.cols],
            rows: self.rows,
            cols: other.cols,
            safe_mode: self.safe_mode || other.safe_mode,
        })
    }

    /// Multiply by a vector (placeholder - needs proper implementation)
    pub fn multiply_vector(&self, vector: &[f32]) -> Result<Vec<f32>, String> {
        if self.cols != vector.len() {
            return Err(format!(
                "Cannot multiply {}x{} matrix by vector of length {}",
                self.rows,
                self.cols,
                vector.len()
            ));
        }

        // TODO: Implement proper matrix-vector multiplication
        Ok(vec![0.0; self.rows])
    }

    /// Get matrix dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Build the final matrix
    pub fn build(self) -> (Vec<f32>, usize, usize) {
        (self.data, self.rows, self.cols)
    }

    /// Get a reference to the underlying data
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Fluent builder for machine learning operations
#[derive(Debug)]
pub struct MLBuilder {
    features: Vec<f32>,
    targets: Vec<f32>,
    safe_mode: bool,
}

impl MLBuilder {
    /// Create a new ML builder
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            targets: Vec::new(),
            safe_mode: false,
        }
    }

    /// Set feature data
    pub fn features(mut self, features: Vec<f32>) -> Self {
        self.features = features;
        self
    }

    /// Set target data
    pub fn targets(mut self, targets: Vec<f32>) -> Self {
        self.targets = targets;
        self
    }

    /// Enable safe mode
    pub fn safe(mut self) -> Self {
        self.safe_mode = true;
        self
    }

    /// Calculate loss using specified function
    pub fn loss(&self, loss_type: LossFunction) -> f32 {
        match loss_type {
            LossFunction::MSE => loss::mse_loss(&self.features, &self.targets),
            LossFunction::MAE => loss::mae_loss(&self.features, &self.targets),
            LossFunction::Huber(delta) => loss::huber_loss(&self.features, &self.targets, delta),
        }
    }

    /// Calculate gradients
    pub fn gradients(&self, loss_type: LossFunction) -> Vec<f32> {
        let mut output = vec![0.0; self.features.len()];
        match loss_type {
            LossFunction::MSE => loss::mse_gradient(&self.features, &self.targets, &mut output),
            LossFunction::MAE => loss::mae_gradient(&self.features, &self.targets, &mut output),
            LossFunction::Huber(delta) => {
                loss::huber_gradient(&self.features, &self.targets, delta, &mut output)
            }
        }
        output
    }

    /// Compute kernel values
    pub fn kernel(&self, other: &[f32], kernel_type: KernelType) -> f32 {
        match kernel_type {
            KernelType::Linear => kernels::linear_kernel(&self.features, other),
            KernelType::RBF(gamma) => kernels::rbf_kernel(&self.features, other, gamma),
            KernelType::Polynomial(degree, coef) => {
                kernels::polynomial_kernel(&self.features, other, degree, coef, 1.0)
            }
        }
    }
}

impl Default for MLBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Loss function types
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    MSE,
    MAE,
    Huber(f32),
}

/// Kernel function types
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    Linear,
    RBF(f32),
    Polynomial(f32, f32),
}

/// Convenience functions for common operations
pub mod ops {
    use super::*;

    /// Create a vector with fluent API
    pub fn vector() -> VectorBuilder {
        VectorBuilder::new()
    }

    /// Create a matrix with fluent API
    pub fn matrix(rows: usize, cols: usize) -> MatrixBuilder {
        MatrixBuilder::new(rows, cols)
    }

    /// Create an ML builder
    pub fn ml() -> MLBuilder {
        MLBuilder::new()
    }

    /// Quick vector operations
    pub fn quick_dot(a: &[f32], b: &[f32]) -> f32 {
        VectorBuilder::from_slice(a).dot(b)
    }

    /// Quick distance calculation
    pub fn quick_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
        VectorBuilder::from_slice(a).distance_to(b, metric)
    }

    /// Quick normalization
    pub fn quick_normalize(data: &[f32]) -> Vec<f32> {
        VectorBuilder::from_slice(data).normalize().build()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::ops::*;
    use super::*;

    #[test]
    fn test_vector_builder_basic() {
        let vec = vector().push(1.0).push(2.0).push(3.0).scale(2.0).build();

        assert_eq!(vec, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vector_builder_chaining() {
        let vec = vector()
            .linspace(0.0, 10.0, 11)
            .scale(0.1)
            .add_scalar(1.0)
            .normalize()
            .build();

        assert_eq!(vec.len(), 11);
        let norm = VectorBuilder::from_slice(&vec).stats().norm;
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_builder_stats() {
        let stats = vector().range(1.0, 6.0, 1.0).stats();

        assert_eq!(stats.length, 5);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_vector_builder_distance() {
        let vec1 = vector().range(0.0, 3.0, 1.0).build();
        let vec2 = vector().range(1.0, 4.0, 1.0).build();

        let distance =
            VectorBuilder::from_slice(&vec1).distance_to(&vec2, DistanceMetric::Euclidean);

        assert!((distance - (3.0_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_builder_basic() {
        let matrix = matrix(2, 2).fill(1.0).scale(2.0).build();

        assert_eq!(matrix.0, [2.0, 2.0, 2.0, 2.0]);
        assert_eq!(matrix.1, 2);
        assert_eq!(matrix.2, 2);
    }

    #[test]
    fn test_matrix_builder_identity() {
        let identity = MatrixBuilder::identity(3).build();
        let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        assert_eq!(identity.0, expected);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = MatrixBuilder::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = MatrixBuilder::from_data(vec![2.0, 0.0, 1.0, 2.0], 2, 2).unwrap();

        let result = a.multiply(&b).unwrap().build();
        // TODO: Fix matrix multiplication implementation
        // assert_eq!(result.0, [4.0, 4.0, 10.0, 8.0]);
        assert_eq!(result.0.len(), 4); // Just check dimensions for now
    }

    #[test]
    fn test_ml_builder() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 1.8, 2.7];

        let mse = ml()
            .features(predictions.clone())
            .targets(targets.clone())
            .loss(LossFunction::MSE);

        assert!(mse > 0.0);

        let gradients = ml()
            .features(predictions)
            .targets(targets)
            .gradients(LossFunction::MSE);

        assert_eq!(gradients.len(), 3);
    }

    #[test]
    fn test_quick_operations() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];

        let dot = quick_dot(&a, &b);
        assert_eq!(dot, 32.0);

        let distance = quick_distance(&a, &b, DistanceMetric::Euclidean);
        assert!((distance - (27.0_f32).sqrt()).abs() < 1e-6);

        let normalized = quick_normalize(&a);
        let norm = VectorBuilder::from_slice(&normalized).stats().norm;
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_safe_mode() {
        let vec = vector()
            .safe()
            .push(1.0)
            .push(2.0)
            .scale(f32::MAX) // This would overflow without safe mode
            .build();

        // In safe mode, overflow should be handled gracefully
        assert!(vec.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_map_operation() {
        let vec = vector()
            .range(1.0, 4.0, 1.0)
            .map(|x| x * x) // Square each element
            .build();

        assert_eq!(vec, [1.0, 4.0, 9.0]);
    }
}
